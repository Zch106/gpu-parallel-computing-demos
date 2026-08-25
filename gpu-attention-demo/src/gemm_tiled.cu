// L3：共享内存分块（Shared Memory Tiling）
// 把 A、B 的小块搬运到 __shared__，再在共享内存内做乘加。
// 教学点：「并行之上的数据流优化」——合并访存(coalescing) + 数据复用，
//          大幅减少昂贵的全局内存流量。
#include "attention.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

__global__ void gemm_tiled_kernel(const float* A, const float* B, float* C,
                                  int M, int N, int K) {
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;

    float s = 0.0f;
    for (int t = 0; t < (K + 15) / 16; t++) {
        // 协作加载 tile，越界补 0
        if (row < M && t * 16 + tx < K)
            As[ty][tx] = A[row * K + t * 16 + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && t * 16 + ty < K)
            Bs[ty][tx] = B[(t * 16 + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; k++)
            s += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = s;
}

LevelResult attention_gpu_tiled(const float* Q, const float* K, const float* V,
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
        gemm_tiled_kernel<<<g1, blk>>>(d_Q, d_Kt, d_S, N, N, d);
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        gemm_tiled_kernel<<<g2, blk>>>(d_S, d_V, d_O, N, d, N);
        timer.end();
        if (it > 0) kms += timer.ms();
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
    r.name = "L3 GPU Tiled (shared mem)";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}
