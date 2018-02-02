#CUDA_ARCH_FLAGS ?= -arch=sm_35
CC_FLAGS += -lcublas -lgflags -lpthread
# CC_FLAGS += -lcurand
# CC_FLAGS += -Xptxas
# CC_FLAGS += -v
CC_FLAGS += -O3
CC_FLAGS += --std=c++11 $(CUDA_ARCH_FLAGS)

CC_FLAGS2 =${CC_FLAGS}
CC_FLAGS2 += --default-stream per-thread

EXE = gemm multi-gemm test

all: $(EXE)

gemm : gemm.cu
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

multi-gemm : multi-gemm.cu
	nvcc  $< $(CC_FLAGS2) $(LIB_FLAGS) -o $@

test : test.cu
	nvcc  $< $(CC_FLAGS2) $(LIB_FLAGS) -o $@



clean:
	rm -f $(EXE)
