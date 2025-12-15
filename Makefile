# Makefile (MPI + CUDA) per requirements: ARCH=sm_<N> (35/60), HOST_COMP=mpicc. [file:2]

ARCH      ?= sm_35
HOST_COMP ?= mpicc
NVCC      ?= nvcc

CXXSTD    ?= c++11
NVFLAGS   = -O3 -std=$(CXXSTD) -arch=$(ARCH) -Xcompiler "-O3"

TARGET    = poisson_mpi_cuda
SRC       = poisson_mpi_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -ccbin $(HOST_COMP) -o $@ $<

clean:
	rm -f $(TARGET)

