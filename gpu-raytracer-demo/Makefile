# GPU 并行计算课程演示 Demo（光线追踪） —— Makefile
# 目标架构 sm_86（RTX 3080 Laptop GPU, Ampere）
# 仅依赖 nvcc + CUDA Runtime + OpenMP（编译器内置），无第三方性能库。

NVCC    = nvcc
ARCH    = -arch=sm_86
# 中文源码在中文 Windows 上默认被按 GBK(936) 解析会触发 C4819，强制 UTF-8 读取。
UTF8    = -Xcompiler /utf-8
INC     = -Iinclude
SRC     = src/main.cu \
          src/cpu_serial.cu \
          src/cpu_omp.cu \
          src/rt_naive.cu \
          src/rt_constant.cu \
          src/rt_opt.cu \
          src/bvh_build.cu \
          src/rt_bvh.cu \
          src/report.cu
OBJ     = $(SRC:.cu=.o)
BIN     = raytracer_demo

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) $(ARCH) $(UTF8) $(INC) -o $@ $(OBJ)

%.o: %.cu
	$(NVCC) $(ARCH) $(UTF8) $(INC) -c $< -o $@

# OpenMP 仅此翻译单元需要
src/cpu_omp.o: src/cpu_omp.cu
	$(NVCC) $(ARCH) $(UTF8) $(INC) -Xcompiler /openmp -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN) results.md ref.ppm out.ppm
