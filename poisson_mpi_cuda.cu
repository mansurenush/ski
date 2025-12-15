// poisson_mpi_cuda.cu
#include <mpi.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

static const double A1 = -1.0, B1 =  1.0;
static const double A2 = -1.0, B2 =  1.0;

static const int    MAXITER = 100000;
static const double TOL     = 1e-6;

#define CUDA_CHECK(call) do {                                 \
  cudaError_t _e = (call);                                    \
  if (_e != cudaSuccess) {                                    \
    fprintf(stderr,"CUDA error %s:%d: %s\n",                  \
            __FILE__, __LINE__, cudaGetErrorString(_e));      \
    MPI_Abort(MPI_COMM_WORLD, 1);                             \
  }                                                           \
} while(0)

struct Grid {
  int M, N;
  double h1, h2, hmax, eps;
};

struct Domain {
  int px, py;
  int rankx, ranky;
  int istart, iend;
  int jstart, jend;
  int localM, localN;
  int left, right, top, bottom;
};

struct Timers {
  double t_init;
  double t_finalize;
  double t_total;
  double t_coeff;

  double t_comm_mpi;
  double t_comm_d2h;
  double t_comm_h2d;
  double t_comm_pack;
  double t_comm_unpack;

  double t_applyA;
  double t_Dinv;
  double t_combine_r;
  double t_rupdate;
  double t_axpy;
  double t_xpay;

  double t_dot_kernel;
  double t_dot_allreduce;
  double t_dot_d2h;
};

static Domain decompose2d(int M, int N, int size, int rank) {
  Domain d{};
  int bestpx = 1, bestpy = size;
  double bestratio = 1e100;

  for (int px = 1; px <= size; ++px) {
    if (size % px != 0) continue;
    int py = size / px;
    int mx = (M - 1) / px;
    int my = (N - 1) / py;
    if (mx < 1 || my < 1) continue;
    double ratio = std::max(double(mx)/my, double(my)/mx);
    if (ratio < bestratio) { bestratio = ratio; bestpx = px; bestpy = py; }
  }

  d.px = bestpx;
  d.py = bestpy;
  d.rankx = rank % d.px;
  d.ranky = rank / d.px;

  int pointsx = M - 1, pointsy = N - 1;
  int basemx = pointsx / d.px, extrax = pointsx % d.px;
  int basemy = pointsy / d.py, extray = pointsy % d.py;

  if (d.rankx < extrax) { d.istart = d.rankx*(basemx+1) + 1; d.iend = d.istart + (basemx+1) - 1; }
  else                  { d.istart = extrax*(basemx+1) + (d.rankx-extrax)*basemx + 1; d.iend = d.istart + basemx - 1; }

  if (d.ranky < extray) { d.jstart = d.ranky*(basemy+1) + 1; d.jend = d.jstart + (basemy+1) - 1; }
  else                  { d.jstart = extray*(basemy+1) + (d.ranky-extray)*basemy + 1; d.jend = d.jstart + basemy - 1; }

  d.localM = d.iend - d.istart + 1;
  d.localN = d.jend - d.jstart + 1;

  d.left   = (d.rankx == 0      ) ? MPI_PROC_NULL : (rank - 1);
  d.right  = (d.rankx == d.px-1 ) ? MPI_PROC_NULL : (rank + 1);
  d.bottom = (d.ranky == 0      ) ? MPI_PROC_NULL : (rank - d.px);
  d.top    = (d.ranky == d.py-1 ) ? MPI_PROC_NULL : (rank + d.px);
  return d;
}

// ---------------- CUDA kernels ----------------

__device__ __forceinline__ bool inDomain(double x, double y) {
  // Variant 8: y^2 < x < 1
  return (y*y < x) && (x < 1.0);
}

__global__ void kernel_init0(double* a, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) a[i] = 0.0;
}

__global__ void kernel_compute_coeff_F(
    int localM, int localN,
    int istart, int jstart,
    int M, int N,
    double h1, double h2,
    double eps,
    double* a, double* b, double* F)
{
  int i = blockIdx.y*blockDim.y + threadIdx.y; // 0..localM+1
  int j = blockIdx.x*blockDim.x + threadIdx.x; // 0..localN+1
  int pitch = localN + 2;

  if (i > localM+1 || j > localN+1) return;

  int gi = (istart - 1) + i;
  int gj = (jstart - 1) + j;

  double x = A1 + gi*h1;
  double y = A2 + gj*h2;

  double x_imh = x - 0.5*h1;
  double x_iph = x + 0.5*h1;
  double y_jmh = y - 0.5*h2;
  double y_jph = y + 0.5*h2;

  bool inA0 = inDomain(x_imh, y_jmh);
  bool inA1 = inDomain(x_imh, y_jph);
  double aval;
  if (inA0 && inA1) aval = 1.0;
  else if (!inA0 && !inA1) aval = 1.0/eps;
  else aval = 0.5 + 0.5/eps;

  bool inB0 = inDomain(x_imh, y_jmh);
  bool inB1 = inDomain(x_iph, y_jmh);
  double bval;
  if (inB0 && inB1) bval = 1.0;
  else if (!inB0 && !inB1) bval = 1.0/eps;
  else bval = 0.5 + 0.5/eps;

  int idx = i*pitch + j;
  a[idx] = aval;
  b[idx] = bval;

  double fval = 0.0;
  if (gi >= 1 && gi <= M-1 && gj >= 1 && gj <= N-1) fval = inDomain(x, y) ? 1.0 : 0.0;
  F[idx] = fval;
}

__global__ void kernel_applyA(
    int localM, int localN,
    double h1, double h2,
    const double* a, const double* b,
    const double* w,
    double* out)
{
  int i = 1 + blockIdx.y*blockDim.y + threadIdx.y;
  int j = 1 + blockIdx.x*blockDim.x + threadIdx.x;
  int pitch = localN + 2;
  if (i > localM || j > localN) return;

  double h1sq = h1*h1;
  double h2sq = h2*h2;

  int c  = i*pitch + j;
  int im = (i-1)*pitch + j;
  int ip = (i+1)*pitch + j;
  int jm = i*pitch + (j-1);
  int jp = i*pitch + (j+1);

  double val =
      -(a[ip] * (w[ip] - w[c]) - a[c] * (w[c] - w[im])) / h1sq
      -(b[jp] * (w[jp] - w[c]) - b[c] * (w[c] - w[jm])) / h2sq;

  out[c] = val;
}

__global__ void kernel_applyDinv(
    int localM, int localN,
    double h1, double h2,
    const double* a, const double* b,
    const double* r,
    double* z)
{
  int i = 1 + blockIdx.y*blockDim.y + threadIdx.y;
  int j = 1 + blockIdx.x*blockDim.x + threadIdx.x;
  int pitch = localN + 2;
  if (i > localM || j > localN) return;

  double h1sq = h1*h1;
  double h2sq = h2*h2;

  int c  = i*pitch + j;
  int ip = (i+1)*pitch + j;
  int jp = i*pitch + (j+1);

  double diag = (a[ip] + a[c]) / h1sq + (b[jp] + b[c]) / h2sq;
  z[c] = (fabs(diag) > 1e-30) ? (r[c] / diag) : 0.0;
}

__global__ void kernel_axpy(int n, double alpha, const double* x, double* y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] += alpha * x[i];
}

__global__ void kernel_xpay(int n, const double* x, double beta, double* y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + beta * y[i];
}

__global__ void kernel_copy(int n, const double* x, double* y) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}

__global__ void kernel_combine_r(int localM, int localN, const double* F, const double* Aw, double* r) {
  int i = 1 + blockIdx.y*blockDim.y + threadIdx.y;
  int j = 1 + blockIdx.x*blockDim.x + threadIdx.x;
  int pitch = localN + 2;
  if (i > localM || j > localN) return;
  int c = i*pitch + j;
  r[c] = F[c] - Aw[c];
}

__global__ void kernel_r_update(int localM, int localN, double alpha, const double* Ap, double* r) {
  int i = 1 + blockIdx.y*blockDim.y + threadIdx.y;
  int j = 1 + blockIdx.x*blockDim.x + threadIdx.x;
  int pitch = localN + 2;
  if (i > localM || j > localN) return;
  int c = i*pitch + j;
  r[c] -= alpha * Ap[c];
}

// ---- pack/unpack halo (bulk) ----

__global__ void kernel_pack_LR(int localM, int localN, const double* w, double* sendL, double* sendR) {
  // sendL[j] = w[1, j+1], sendR[j] = w[localM, j+1]
  int j = blockIdx.x * blockDim.x + threadIdx.x; // 0..localN-1
  if (j >= localN) return;
  int pitch = localN + 2;
  sendL[j] = w[1*pitch + (j+1)];
  sendR[j] = w[localM*pitch + (j+1)];
}

__global__ void kernel_unpack_LR(int localM, int localN, double* w, const double* recvL, const double* recvR,
                                 int hasLeft, int hasRight) {
  // w[0, j+1] and w[localM+1, j+1]
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= localN) return;
  int pitch = localN + 2;
  w[0*pitch + (j+1)]           = hasLeft  ? recvL[j] : 0.0;
  w[(localM+1)*pitch + (j+1)]  = hasRight ? recvR[j] : 0.0;
}

__global__ void kernel_pack_BT(int localM, int localN, const double* w, double* sendB, double* sendT) {
  // sendB[i] = w[i+1,1], sendT[i] = w[i+1, localN]
  int i = blockIdx.x * blockDim.x + threadIdx.x; // 0..localM-1
  if (i >= localM) return;
  int pitch = localN + 2;
  sendB[i] = w[(i+1)*pitch + 1];
  sendT[i] = w[(i+1)*pitch + localN];
}

__global__ void kernel_unpack_BT(int localM, int localN, double* w, const double* recvB, const double* recvT,
                                 int hasBottom, int hasTop) {
  // w[i+1,0] and w[i+1, localN+1]
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= localM) return;
  int pitch = localN + 2;
  w[(i+1)*pitch + 0]          = hasBottom ? recvB[i] : 0.0;
  w[(i+1)*pitch + (localN+1)] = hasTop    ? recvT[i] : 0.0;
}

// ---- dot partial without shared memory (p.10) ----
__global__ void kernel_dot_partial(int localM, int localN, const double* u, const double* v, double* partial) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  int pitch = localN + 2;
  int interior = localM * localN;

  double sum = 0.0;
  for (int k = tid; k < interior; k += stride) {
    int i = (k / localN) + 1;
    int j = (k % localN) + 1;
    int idx = i*pitch + j;
    sum += u[idx] * v[idx];
  }
  partial[tid] = sum;
}

static double gpu_dot_mpi(
    const Domain& d,
    const double* du, const double* dv,
    double* dPartials, double* hPartials,
    int blocks, int threads,
    cudaEvent_t evStart, cudaEvent_t evStop,
    MPI_Comm comm,
    Timers* timers)
{
  int nThreadsTotal = blocks * threads;

  cudaEventRecord(evStart);
  kernel_dot_partial<<<blocks, threads>>>(d.localM, d.localN, du, dv, dPartials);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, evStart, evStop));
  timers->t_dot_kernel += ms * 1e-3;

  double t0 = MPI_Wtime();
  CUDA_CHECK(cudaMemcpy(hPartials, dPartials, nThreadsTotal*sizeof(double), cudaMemcpyDeviceToHost));
  double t1 = MPI_Wtime();
  timers->t_dot_d2h += (t1 - t0);

  double local = 0.0;
  for (int i = 0; i < nThreadsTotal; ++i) local += hPartials[i];

  double t2 = MPI_Wtime();
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
  double t3 = MPI_Wtime();
  timers->t_dot_allreduce += (t3 - t2);

  return global;
}

// ---------------- halo exchange (bulk) with timing ----------------

struct HaloBuffers {
  // device
  double *d_sendL=nullptr, *d_sendR=nullptr, *d_recvL=nullptr, *d_recvR=nullptr; // size localN
  double *d_sendB=nullptr, *d_sendT=nullptr, *d_recvB=nullptr, *d_recvT=nullptr; // size localM
  // host pinned
  double *h_sendL=nullptr, *h_sendR=nullptr, *h_recvL=nullptr, *h_recvR=nullptr; // size localN
  double *h_sendB=nullptr, *h_sendT=nullptr, *h_recvB=nullptr, *h_recvT=nullptr; // size localM
};

static void halo_alloc(const Domain& d, HaloBuffers& hb) {
  // device
  CUDA_CHECK(cudaMalloc(&hb.d_sendL, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_sendR, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_recvL, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_recvR, d.localN*sizeof(double)));

  CUDA_CHECK(cudaMalloc(&hb.d_sendB, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_sendT, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_recvB, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&hb.d_recvT, d.localM*sizeof(double)));

  // host pinned
  CUDA_CHECK(cudaMallocHost(&hb.h_sendL, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_sendR, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_recvL, d.localN*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_recvR, d.localN*sizeof(double)));

  CUDA_CHECK(cudaMallocHost(&hb.h_sendB, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_sendT, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_recvB, d.localM*sizeof(double)));
  CUDA_CHECK(cudaMallocHost(&hb.h_recvT, d.localM*sizeof(double)));
}

static void halo_free(HaloBuffers& hb) {
  cudaFree(hb.d_sendL); cudaFree(hb.d_sendR); cudaFree(hb.d_recvL); cudaFree(hb.d_recvR);
  cudaFree(hb.d_sendB); cudaFree(hb.d_sendT); cudaFree(hb.d_recvB); cudaFree(hb.d_recvT);

  cudaFreeHost(hb.h_sendL); cudaFreeHost(hb.h_sendR); cudaFreeHost(hb.h_recvL); cudaFreeHost(hb.h_recvR);
  cudaFreeHost(hb.h_sendB); cudaFreeHost(hb.h_sendT); cudaFreeHost(hb.h_recvB); cudaFreeHost(hb.h_recvT);
}

static void exchange_boundaries_bulk(
    const Domain& d, double* dw,
    HaloBuffers& hb,
    cudaEvent_t evStart, cudaEvent_t evStop,
    MPI_Comm comm,
    Timers* timers)
{
  MPI_Status st;

  // pack on GPU
  int tpb = 256;

  cudaEventRecord(evStart);
  kernel_pack_LR<<<(d.localN + tpb - 1)/tpb, tpb>>>(d.localM, d.localN, dw, hb.d_sendL, hb.d_sendR);
  kernel_pack_BT<<<(d.localM + tpb - 1)/tpb, tpb>>>(d.localM, d.localN, dw, hb.d_sendB, hb.d_sendT);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float msPack = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msPack, evStart, evStop));
  timers->t_comm_pack += msPack * 1e-3;

  // D2H (bulk)
  double t0 = MPI_Wtime();
  CUDA_CHECK(cudaMemcpy(hb.h_sendL, hb.d_sendL, d.localN*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.h_sendR, hb.d_sendR, d.localN*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.h_sendB, hb.d_sendB, d.localM*sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hb.h_sendT, hb.d_sendT, d.localM*sizeof(double), cudaMemcpyDeviceToHost));
  double t1 = MPI_Wtime();
  timers->t_comm_d2h += (t1 - t0);

  // MPI обмен
  double t2 = MPI_Wtime();
  MPI_Sendrecv(hb.h_sendL, d.localN, MPI_DOUBLE, d.left,  10,
               hb.h_recvL, d.localN, MPI_DOUBLE, d.left,  11,
               comm, &st);

  MPI_Sendrecv(hb.h_sendR, d.localN, MPI_DOUBLE, d.right, 11,
               hb.h_recvR, d.localN, MPI_DOUBLE, d.right, 10,
               comm, &st);

  MPI_Sendrecv(hb.h_sendB, d.localM, MPI_DOUBLE, d.bottom, 20,
               hb.h_recvB, d.localM, MPI_DOUBLE, d.bottom, 21,
               comm, &st);

  MPI_Sendrecv(hb.h_sendT, d.localM, MPI_DOUBLE, d.top,    21,
               hb.h_recvT, d.localM, MPI_DOUBLE, d.top,    20,
               comm, &st);
  double t3 = MPI_Wtime();
  timers->t_comm_mpi += (t3 - t2);

  // H2D (bulk)
  double t4 = MPI_Wtime();
  CUDA_CHECK(cudaMemcpy(hb.d_recvL, hb.h_recvL, d.localN*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hb.d_recvR, hb.h_recvR, d.localN*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hb.d_recvB, hb.h_recvB, d.localM*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(hb.d_recvT, hb.h_recvT, d.localM*sizeof(double), cudaMemcpyHostToDevice));
  double t5 = MPI_Wtime();
  timers->t_comm_h2d += (t5 - t4);

  // unpack to halos on GPU (+ zeros on outer boundaries)
  int hasLeft   = (d.left   != MPI_PROC_NULL);
  int hasRight  = (d.right  != MPI_PROC_NULL);
  int hasBottom = (d.bottom != MPI_PROC_NULL);
  int hasTop    = (d.top    != MPI_PROC_NULL);

  cudaEventRecord(evStart);
  kernel_unpack_LR<<<(d.localN + tpb - 1)/tpb, tpb>>>(d.localM, d.localN, dw, hb.d_recvL, hb.d_recvR, hasLeft, hasRight);
  kernel_unpack_BT<<<(d.localM + tpb - 1)/tpb, tpb>>>(d.localM, d.localN, dw, hb.d_recvB, hb.d_recvT, hasBottom, hasTop);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float msUnpack = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msUnpack, evStart, evStop));
  timers->t_comm_unpack += msUnpack * 1e-3;
}

// ---------------- main ----------------

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank=0, size=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int M = 400, N = 600;
  if (argc >= 3) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); }

  Timers timers{};
  double t_init_start = MPI_Wtime();

  int ngpu = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ngpu));
  if (ngpu > 0) CUDA_CHECK(cudaSetDevice(rank % ngpu));

  Grid g{};
  g.M = M; g.N = N;
  g.h1 = (B1 - A1) / M;
  g.h2 = (B2 - A2) / N;
  g.hmax = std::max(g.h1, g.h2);
  g.eps  = g.hmax * g.hmax; // eps = h^2

  Domain d = decompose2d(M, N, size, rank);

  int pitch = d.localN + 2;
  int total = (d.localM + 2) * (d.localN + 2);

  double *da=nullptr, *db=nullptr, *dF=nullptr;
  double *dw=nullptr, *dr=nullptr, *dz=nullptr, *dp=nullptr, *dAp=nullptr, *dTmp=nullptr;

  CUDA_CHECK(cudaMalloc(&da,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&db,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dF,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dw,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dr,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dz,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dp,  total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dAp, total*sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dTmp,total*sizeof(double)));

  kernel_init0<<<(total+255)/256, 256>>>(dw, total);
  kernel_init0<<<(total+255)/256, 256>>>(da, total);
  kernel_init0<<<(total+255)/256, 256>>>(db, total);
  kernel_init0<<<(total+255)/256, 256>>>(dF, total);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t evStart, evStop;
  CUDA_CHECK(cudaEventCreate(&evStart));
  CUDA_CHECK(cudaEventCreate(&evStop));

  // coefficients
  dim3 blk2(16,16);
  dim3 grdCoeff((d.localN+2 + blk2.x-1)/blk2.x, (d.localM+2 + blk2.y-1)/blk2.y);

  cudaEventRecord(evStart);
  kernel_compute_coeff_F<<<grdCoeff, blk2>>>(
      d.localM, d.localN, d.istart, d.jstart,
      M, N, g.h1, g.h2, g.eps, da, db, dF);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float ms_coeff = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_coeff, evStart, evStop));
  timers.t_coeff = ms_coeff * 1e-3;

  // halo buffers (bulk, pinned host)
  HaloBuffers hb;
  halo_alloc(d, hb);

  // dot buffers
  int dotThreads = 256;
  int dotBlocks  = 256;
  int nThreadsTotal = dotBlocks * dotThreads;
  double* dPartials=nullptr;
  CUDA_CHECK(cudaMalloc(&dPartials, nThreadsTotal*sizeof(double)));
  std::vector<double> hPartials(nThreadsTotal);

  timers.t_init = MPI_Wtime() - t_init_start;

  // ---- Solve ----
  MPI_Barrier(MPI_COMM_WORLD);
  double t_total_start = MPI_Wtime();

  dim3 grdOp((d.localN + blk2.x-1)/blk2.x, (d.localM + blk2.y-1)/blk2.y);

  // initial halo for w
  exchange_boundaries_bulk(d, dw, hb, evStart, evStop, MPI_COMM_WORLD, &timers);

  // r0 = F - A w0
  cudaEventRecord(evStart);
  kernel_applyA<<<grdOp, blk2>>>(d.localM, d.localN, g.h1, g.h2, da, db, dw, dTmp);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float msA = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msA, evStart, evStop));
  timers.t_applyA += msA * 1e-3;

  cudaEventRecord(evStart);
  kernel_combine_r<<<grdOp, blk2>>>(d.localM, d.localN, dF, dTmp, dr);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float msCr = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msCr, evStart, evStop));
  timers.t_combine_r += msCr * 1e-3;

  cudaEventRecord(evStart);
  kernel_applyDinv<<<grdOp, blk2>>>(d.localM, d.localN, g.h1, g.h2, da, db, dr, dz);
  cudaEventRecord(evStop);
  CUDA_CHECK(cudaEventSynchronize(evStop));
  float msD = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msD, evStart, evStop));
  timers.t_Dinv += msD * 1e-3;

  kernel_copy<<<(total+255)/256, 256>>>(total, dz, dp);
  CUDA_CHECK(cudaDeviceSynchronize());

  double hscale = g.h1 * g.h2;
  double zr = gpu_dot_mpi(d, dz, dr, dPartials, hPartials.data(),
                          dotBlocks, dotThreads, evStart, evStop,
                          MPI_COMM_WORLD, &timers) * hscale;
  double rnorm = std::sqrt(gpu_dot_mpi(d, dr, dr, dPartials, hPartials.data(),
                                       dotBlocks, dotThreads, evStart, evStop,
                                       MPI_COMM_WORLD, &timers) * hscale);

  int it = 0;
  double finalResidual = rnorm;

  for (it = 0; it < MAXITER; ++it) {
    exchange_boundaries_bulk(d, dp, hb, evStart, evStop, MPI_COMM_WORLD, &timers);

    cudaEventRecord(evStart);
    kernel_applyA<<<grdOp, blk2>>>(d.localM, d.localN, g.h1, g.h2, da, db, dp, dAp);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float msAp = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msAp, evStart, evStop));
    timers.t_applyA += msAp * 1e-3;

    double pAp = gpu_dot_mpi(d, dAp, dp, dPartials, hPartials.data(),
                             dotBlocks, dotThreads, evStart, evStop,
                             MPI_COMM_WORLD, &timers) * hscale;
    if (fabs(pAp) < 1e-30) break;

    double alpha = zr / pAp;

    cudaEventRecord(evStart);
    kernel_axpy<<<(total+255)/256, 256>>>(total, alpha, dp, dw);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float msAx = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msAx, evStart, evStop));
    timers.t_axpy += msAx * 1e-3;

    cudaEventRecord(evStart);
    kernel_r_update<<<grdOp, blk2>>>(d.localM, d.localN, alpha, dAp, dr);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float msRu = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msRu, evStart, evStop));
    timers.t_rupdate += msRu * 1e-3;

    rnorm = std::sqrt(gpu_dot_mpi(d, dr, dr, dPartials, hPartials.data(),
                                  dotBlocks, dotThreads, evStart, evStop,
                                  MPI_COMM_WORLD, &timers) * hscale);
    finalResidual = rnorm;
    if (rnorm < TOL) break;

    cudaEventRecord(evStart);
    kernel_applyDinv<<<grdOp, blk2>>>(d.localM, d.localN, g.h1, g.h2, da, db, dr, dz);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float msDn = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msDn, evStart, evStop));
    timers.t_Dinv += msDn * 1e-3;

    double zrNew = gpu_dot_mpi(d, dz, dr, dPartials, hPartials.data(),
                               dotBlocks, dotThreads, evStart, evStop,
                               MPI_COMM_WORLD, &timers) * hscale;
    double beta = zrNew / zr;
    zr = zrNew;

    cudaEventRecord(evStart);
    kernel_xpay<<<(total+255)/256, 256>>>(total, dz, beta, dp);
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float msXp = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&msXp, evStart, evStop));
    timers.t_xpay += msXp * 1e-3;
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  timers.t_total = MPI_Wtime() - t_total_start;

  // finalize timing
  double t_fin0 = MPI_Wtime();

  cudaEventDestroy(evStart);
  cudaEventDestroy(evStop);

  cudaFree(dPartials);
  halo_free(hb);

  cudaFree(da); cudaFree(db); cudaFree(dF);
  cudaFree(dw); cudaFree(dr); cudaFree(dz); cudaFree(dp); cudaFree(dAp); cudaFree(dTmp);

  timers.t_finalize = MPI_Wtime() - t_fin0;

  // Reduce timings: max over ranks
  Timers gtimers{};
  MPI_Reduce(&timers, &gtimers, sizeof(Timers)/sizeof(double),
             MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    printf("MPI ranks: %d, Grid: %dx%d, Decomp: %dx%d\n", size, M, N, d.px, d.py);
    printf("Iterations: %d, residual: %.6e\n", it, finalResidual);

    printf("===== Timings (sec, max over ranks) =====\n");
    printf("Init:            %10.6f\n", gtimers.t_init);
    printf("Coeff (GPU):     %10.6f\n", gtimers.t_coeff);
    printf("Total solve:     %10.6f\n", gtimers.t_total);
    printf("Finalize:        %10.6f\n", gtimers.t_finalize);

    printf("\nComm MPI:        %10.6f\n", gtimers.t_comm_mpi);
    printf("Comm D2H:        %10.6f\n", gtimers.t_comm_d2h);
    printf("Comm H2D:        %10.6f\n", gtimers.t_comm_h2d);
    printf("Comm pack (GPU): %10.6f\n", gtimers.t_comm_pack);
    printf("Comm unpk (GPU): %10.6f\n", gtimers.t_comm_unpack);

    printf("\napplyA:          %10.6f\n", gtimers.t_applyA);
    printf("Dinv:            %10.6f\n", gtimers.t_Dinv);
    printf("combine_r:       %10.6f\n", gtimers.t_combine_r);
    printf("r_update:        %10.6f\n", gtimers.t_rupdate);
    printf("axpy:            %10.6f\n", gtimers.t_axpy);
    printf("xpay:            %10.6f\n", gtimers.t_xpay);

    printf("\ndot kernel:      %10.6f\n", gtimers.t_dot_kernel);
    printf("dot D2H:         %10.6f\n", gtimers.t_dot_d2h);
    printf("dot allreduce:   %10.6f\n", gtimers.t_dot_allreduce);
    printf("=========================================\n");
  }

  MPI_Finalize();
  return 0;
}

