#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>

// 每一层实验的结果
struct LevelResult {
    const char* name;   // 层级名称
    float kernel_ms;    // 仅 GPU 计算耗时（CUDA events 测量）
    float e2e_ms;       // 端到端耗时（含分配 / H2D / 计算 / D2H，CPU 计时）
    float max_err;      // 与 CPU 参考的最大绝对误差（由 main 填充）
    bool ok;            // 是否在容差内（由 main 填充）
    float speedup;      // 相对 CPU 的加速比（由 main 填充）
};

// ---------- GEMM device kernels（行主序 C = A(M×K) · B(K×N)）----------
__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K);
__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K);
__global__ void gemm_opt_kernel(const float* A, const float* B, float* C,
                                int M, int N, int K);
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C,
                                 int M, int N, int K);

// 逐行 softmax（C = softmax(S * scale)），row = N 行
__global__ void softmax_kernel(float* S, int N, float scale);

// 精度转换辅助 kernel
__global__ void float2half_kernel(const float* src, half* dst, int n);
__global__ void half2float_kernel(const half* src, float* dst, int n);

// ---------- 各层 host 包装函数 ----------
// 输入 Q,K,V 为主机端浮点数组（行主序），输出 O 写回主机端浮点数组。
// 返回计时（max_err / ok / speedup 由 main 在拿到 O 后与 CPU 参考比对后填充）。
LevelResult attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int N, int d);
LevelResult attention_gpu_naive(const float* Q, const float* K, const float* V,
                                 float* O, int N, int d);
LevelResult attention_gpu_tiled(const float* Q, const float* K, const float* V,
                                 float* O, int N, int d);
LevelResult attention_gpu_opt(const float* Q, const float* K, const float* V,
                              float* O, int N, int d);
LevelResult attention_gpu_tc(const float* Q, const float* K, const float* V,
                             float* O, int N, int d);
LevelResult attention_cublas_fp32(const float* Q, const float* K,
                                  const float* V, float* O, int N, int d);
LevelResult attention_cublas_fp16(const float* Q, const float* K,
                                  const float* V, float* O, int N, int d);

// 生成 Markdown 报告
void write_report(const char* path, const std::string& md);

#endif  // ATTENTION_CUH
