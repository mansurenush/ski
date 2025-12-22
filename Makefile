# Makefile (MPI + CUDA) per requirements: ARCH=sm_<N> (35/60), HOST_COMP=mpicc. [file:2]

ARCH      ?= sm_60
HOST_COMP ?= mpicxx
NVCC      ?= nvcc

CXXSTD    ?= c++11
NVFLAGS   = -arch=$(ARCH) -ccbin $(HOST_COMP) -O3 -Xcompiler -Wall,-fPIC -std=c++11

TARGET    = poisson_mpi_cuda
SRC       = poisson_mpi_cuda.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean