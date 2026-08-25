// L2：GPU 朴素并行
// 每个线程负责输出矩阵的一个元素，所有数据走全局内存。
// 教学点：「并行」本身带来的第一波加速（对比 L1 CPU 串行）。
#include "attention.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        float s = 0.0f;
        for (int k = 0; k < K; k++) {
            s += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = s;
    }
}

LevelResult attention_gpu_naive(const float* Q, const float* K, const float* V,
                                float* O, int N, int d) {
    auto t0 = std::chrono::high_resolution_clock::now();

    float *d_Q = dev_dup(Q, N * d);
    float *d_K = dev_dup(K, N * d);
    float *d_V = dev_dup(V, N * d);

    float* hKt = new float[d * N];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < N; j++)
            hKt[i * N + j] = K[j * d + i];
    float* d_Kt = dev_dup(hKt, d * N);

    float* d_S = dev_alloc(N * N);
    float* d_O = dev_alloc(N * d);

    float scale = 1.0f / sqrtf((float)d);
    dim3 blk(16, 16);
    dim3 g1((N + 15) / 16, (N + 15) / 16);
    dim3 g2((d + 15) / 16, (N + 15) / 16);

    CudaTimer timer;
    float kms = 0.0f;
    const int ITER = 6;
    for (int it = 0; it < ITER; it++) {
        timer.begin();
        gemm_naive_kernel<<<g1, blk>>>(d_Q, d_Kt, d_S, N, N, d);
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        gemm_naive_kernel<<<g2, blk>>>(d_S, d_V, d_O, N, d, N);
        timer.end();
        if (it > 0) kms += timer.ms();  // 跳过首次 warmup
    }
    kms /= (ITER - 1);

    CHECK(cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2e = std::chrono::duration<float, std::milli>(t1 - t0).count();

    CHECK(cudaFree(d_Q)); CHECK(cudaFree(d_K)); CHECK(cudaFree(d_V));
    CHECK(cudaFree(d_Kt)); CHECK(cudaFree(d_S));
    CHECK(cudaFree(d_O));
    delete[] hKt;

    LevelResult r;
    r.name = "L2 GPU Naive";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}
