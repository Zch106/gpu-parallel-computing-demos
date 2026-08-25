// L5：Tensor Core（WMMA, FP16→FP32）
// 用 Ampere 的 Tensor Core 做矩阵乘：输入 FP16，累加 FP32，输出 FP32。
// 教学点：「现代 AI 硬件」再一个量级的跃迁 —— 这正是大模型推理的底层算力来源。
#include "attention.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>
#include <mma.h>

using namespace nvcuda;

// FP16 <-> FP32 设备转换
__global__ void float2half_kernel(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}
__global__ void half2float_kernel(const half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

// WMMA GEMM：A(M×K) FP16, B(K×N) FP16 -> C(M×N) FP32 累加
// 一个 block = 一个 warp(32线程)，负责 16×16 输出 tile。
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
    int warpM = blockIdx.y;  // tile 行
    int warpN = blockIdx.x;  // tile 列

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16, aCol = k;
        int bRow = k, bCol = warpN * 16;
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int cRow = warpM * 16, cCol = warpN * 16;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}

LevelResult attention_gpu_tc(const float* Q, const float* K, const float* V,
                             float* O, int N, int d) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 主机端转 FP16
    int nd = N * d, nn = N * N;
    half *hQ = new half[nd], *hK = new half[nd], *hV = new half[nd];
    half *hKt = new half[d * N];
    for (int i = 0; i < nd; i++) { hQ[i] = __float2half(Q[i]); hK[i] = __float2half(K[i]); hV[i] = __float2half(V[i]); }
    for (int i = 0; i < d; i++)
        for (int j = 0; j < N; j++)
            hKt[i * N + j] = hK[j * d + i];

    half *d_Q, *d_K, *d_V, *d_Kt, *d_Ph;
    float *d_S, *d_P, *d_O;
    CHECK(cudaMalloc(&d_Q, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_K, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_V, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_Kt, d * N * sizeof(half)));
    CHECK(cudaMalloc(&d_Ph, nn * sizeof(half)));
    CHECK(cudaMalloc(&d_S, nn * sizeof(float)));
    CHECK(cudaMalloc(&d_P, nn * sizeof(float)));
    CHECK(cudaMalloc(&d_O, nd * sizeof(float)));

    CHECK(cudaMemcpy(d_Q, hQ, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, hK, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, hV, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Kt, hKt, d * N * sizeof(half), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)d);

    dim3 warp(32, 1);
    dim3 g1(N / 16, N / 16);
    dim3 g2(d / 16, N / 16);
    int cv = (nn + 255) / 256;

    CudaTimer timer;
    float kms = 0.0f;
    const int ITER = 6;
    for (int it = 0; it < ITER; it++) {
        timer.begin();
        // S = Q · Kᵀ   (FP16)
        wmma_gemm_kernel<<<g1, warp>>>(d_Q, d_Kt, d_S, N, N, d);
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        // P(float) -> Ph(FP16)
        float2half_kernel<<<cv, 256>>>(d_S, d_Ph, nn);
        // O = P · V    (FP16)
        wmma_gemm_kernel<<<g2, warp>>>(d_Ph, d_V, d_O, N, d, N);
        timer.end();
        if (it > 0) kms += timer.ms();
    }
    kms /= (ITER - 1);

    CHECK(cudaMemcpy(O, d_O, nd * sizeof(float), cudaMemcpyDeviceToHost));
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2e = std::chrono::duration<float, std::milli>(t1 - t0).count();

    CHECK(cudaFree(d_Q)); CHECK(cudaFree(d_K)); CHECK(cudaFree(d_V));
    CHECK(cudaFree(d_Kt)); CHECK(cudaFree(d_Ph));
    CHECK(cudaFree(d_S)); CHECK(cudaFree(d_P)); CHECK(cudaFree(d_O));
    delete[] hQ; delete[] hK; delete[] hV; delete[] hKt;

    LevelResult r;
    r.name = "L5 GPU Tensor Core (WMMA FP16)";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}
