# GPU 并行计算课程演示 Demo —— Makefile
# 目标架构 sm_86（RTX 3080 Laptop GPU, Ampere）
# 仅依赖 nvcc + CUDA Runtime；-lcublas 仅用于外部对照基线（不参与教学实现）

NVCC    = nvcc
ARCH    = -arch=sm_86
# 中文源码在中文 Windows 上默认被按 GBK(代码页936) 解析会触发 C4819 并报错，
# 因此强制主机编译器以 UTF-8 读取源码（cl.exe 的 /utf-8 标志）。
UTF8    = -Xcompiler /utf-8
INC     = -Iinclude
SRC     = src/main.cu \
          src/cpu_ref.cu \
          src/softmax.cu \
          src/gemm_naive.cu \
          src/gemm_tiled.cu \
          src/gemm_opt.cu \
          src/gemm_tc.cu \
          src/cublas_ref.cu \
          src/report.cu
OBJ     = $(SRC:.cu=.o)
BIN     = attention_demo
CXXFLAGS =

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) $(ARCH) $(UTF8) $(INC) -o $@ $(OBJ) -lcublas

%.o: %.cu
	$(NVCC) $(ARCH) $(UTF8) $(INC) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN) results.md
