# Pin the requested CUDA 13 toolkit even when the parent shell exports an older
# CUDA_HOME.  A command-line assignment can still override this if needed.
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC := $(CUDA_HOME)/bin/nvcc
ARCH ?= sm_120
BUILD := build
TARGET := $(BUILD)/int4_gemm
SOURCES := src/main.cu src/quantize.cu src/repack.cu src/reference.cu \
  kernels/gemm_v00_reference.cu kernels/gemm_v01_pairwise.cu kernels/gemm_v02_repack_simt.cu \
  kernels/gemm_v03_tensorcore.cu kernels/gemm_v04_repack_tensorcore.cu kernels/gemm_final.cu
SOURCES += kernels/gemm_v09_async.cu
SOURCES += kernels/gemm_v15_tma.cu
SOURCES += kernels/gemm_v16_epilogue.cu
SOURCES += kernels/gemm_v16_register.cu
SOURCES += kernels/gemm_v16_tma.cu
SOURCES += kernels/gemm_v17_instruction.cu
OBJECTS := $(SOURCES:%.cu=$(BUILD)/%.o)
NVCCFLAGS := -std=c++17 -O3 -lineinfo --use_fast_math -arch=$(ARCH) -Iinclude -Xptxas=-v,-warn-spills
FP32_OUTPUT ?= 1
NVCCFLAGS += -DINT4_GEMM_ENABLE_FP32_OUTPUT=$(FP32_OUTPUT)

.PHONY: all clean ptx sass
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -arch=$(ARCH) -o $@ $^ -lcublas -lcuda

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc -o $@ $<

ptx: | results
	$(NVCC) $(NVCCFLAGS) -ptx kernels/gemm_v09_async.cu -o results/final.ptx
	$(NVCC) $(NVCCFLAGS) -ptx kernels/gemm_v15_tma.cu -o results/tma.ptx
	$(NVCC) $(NVCCFLAGS) -ptx kernels/gemm_v16_register.cu -o results/v16_register.ptx
	$(NVCC) $(NVCCFLAGS) -ptx kernels/gemm_v16_tma.cu -o results/v16_tma.ptx
	$(NVCC) $(NVCCFLAGS) -ptx kernels/gemm_v17_instruction.cu -o results/v17_instruction.ptx

sass: $(TARGET) | results
	$(CUDA_HOME)/bin/cuobjdump --dump-sass $(TARGET) > results/final.sass

results:
	mkdir -p results/profile

clean:
	rm -rf $(BUILD)
