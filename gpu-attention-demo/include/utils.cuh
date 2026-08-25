#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA 错误检查宏
#define CHECK(call)                                                          \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "[CUDA ERROR] %s:%d : %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                                 \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// cuBLAS 错误检查宏（返回 cublasStatus_t，而非 cudaError_t）
#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d : %s\n", __FILE__,         \
                    __LINE__, cublasGetStatusString(_s));                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// 用固定种子生成 [-0.5, 0.5) 的随机浮点数组（保证每次演示可复现）
static void init_random(float* a, int n, unsigned seed = 1234) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        a[i] = (float)(rand() % 10000) / 10000.0f - 0.5f;
    }
}

// 计算两个数组的最大绝对误差（用于正确性校验）
static float max_abs_error(const float* a, const float* b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

// 设备端：分配显存并从主机拷贝数据
static float* dev_dup(const float* h, int n) {
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// 设备端：仅分配显存（未初始化）
static float* dev_alloc(int n) {
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    return d;
}

#endif  // UTILS_CUH
