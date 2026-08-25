// L4：分块 + 深度优化
// 相对 L3 的进一步压榨：
//   1) float4 向量化加载 —— 每个线程一次搬运 128 bit，提升全局内存带宽利用率；
//   2) 共享内存第 2 维 +1 padding —— 消除 4-byte bank conflict；
//   3) 循环展开 (#pragma unroll) —— 减少分支 / 提升指令级并行。
// 每个线程负责 2×2 个输出元素（8×8 线程块 → 16×16 tile）。
#include "attention.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

__global__ void gemm_opt_kernel(const float* A, const float* B, float* C,
                                 int M, int N, int K) {
    __shared__ float As[16][17];   // +1 消除 bank conflict
    __shared__ float Bs[16][17];

    int tx = threadIdx.x;   // 0..7
    int ty = threadIdx.y;   // 0..7
    int gid = ty * 8 + tx;

    int arow_tile = gid / 4;          // 0..15
    int acol_tile = (gid % 4) * 4;    // 0,4,8,12
    int brow_tile = gid / 4;
    int bcol_tile = (gid % 4) * 4;

    int out_row0 = blockIdx.y * 16 + ty * 2;
    int out_col0 = blockIdx.x * 16 + tx * 2;

    float c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    for (int t = 0; t < (K + 15) / 16; t++) {
        int arow = blockIdx.y * 16 + arow_tile;
        int acol = t * 16 + acol_tile;
        int brow = t * 16 + brow_tile;
        int bcol = blockIdx.x * 16 + bcol_tile;

        // float4 向量化加载（同一行内连续 4 个元素 → 合并访问）
        if (arow < M) {
            float4 a4 = *reinterpret_cast<const float4*>(&A[arow * K + acol]);
            As[arow_tile][acol_tile]     = a4.x;
            As[arow_tile][acol_tile + 1] = a4.y;
            As[arow_tile][acol_tile + 2] = a4.z;
            As[arow_tile][acol_tile + 3] = a4.w;
        } else {
            As[arow_tile][acol_tile] = As[arow_tile][acol_tile + 1] =
            As[arow_tile][acol_tile + 2] = As[arow_tile][acol_tile + 3] = 0.0f;
        }

        if (bcol < N) {
            float4 b4 = *reinterpret_cast<const float4*>(&B[brow * N + bcol]);
            Bs[brow_tile][bcol_tile]     = b4.x;
            Bs[brow_tile][bcol_tile + 1] = b4.y;
            Bs[brow_tile][bcol_tile + 2] = b4.z;
            Bs[brow_tile][bcol_tile + 3] = b4.w;
        } else {
            Bs[brow_tile][bcol_tile] = Bs[brow_tile][bcol_tile + 1] =
            Bs[brow_tile][bcol_tile + 2] = Bs[brow_tile][bcol_tile + 3] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 16; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][tx * 2];
            float b1 = Bs[k][tx * 2 + 1];
            c00 += a0 * b0; c01 += a0 * b1;
            c10 += a1 * b0; c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (out_row0 < M && out_col0 < N)         C[out_row0 * N + out_col0]         = c00;
    if (out_row0 < M && out_col0 + 1 < N)     C[out_row0 * N + out_col0 + 1]     = c01;
    if (out_row0 + 1 < M && out_col0 < N)     C[(out_row0 + 1) * N + out_col0]   = c10;
    if (out_row0 + 1 < M && out_col0 + 1 < N) C[(out_row0 + 1) * N + out_col0 + 1] = c11;
}

LevelResult attention_gpu_opt(const float* Q, const float* K, const float* V,
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
    dim3 blk(8, 8);
    dim3 g1((N + 15) / 16, (N + 15) / 16);
    dim3 g2((d + 15) / 16, (N + 15) / 16);

    CudaTimer timer;
    float kms = 0.0f;
    const int ITER = 6;
    for (int it = 0; it < ITER; it++) {
        timer.begin();
        gemm_opt_kernel<<<g1, blk>>>(d_Q, d_Kt, d_S, N, N, d);
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        gemm_opt_kernel<<<g2, blk>>>(d_S, d_V, d_O, N, d, N);
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
    r.name = "L4 GPU Opt (float4+pad)";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}
