// L1：CPU 串行基线（单线程朴素实现）
// 同时作为「正确性参考」：后续所有 GPU 层级都与它比对误差。
#include "attention.cuh"
#include "utils.cuh"
#include <chrono>

// 朴素三重循环 GEMM（行主序 C = A(M×K) · B(K×N)）
static void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) {
                s += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}

// 逐行 softmax（在已经乘过 scale 的 S 上做）
static void softmax_cpu(float* S, int N) {
    for (int i = 0; i < N; i++) {
        float* row = S + i * N;
        float mx = -1e30f;
        for (int j = 0; j < N; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        for (int j = 0; j < N; j++) row[j] /= sum;
    }
}

LevelResult attention_cpu(const float* Q, const float* K, const float* V,
                          float* O, int N, int d) {
    float* S  = new float[N * N];
    float* Kt = new float[d * N];  // Kᵀ：Kt[i*N + j] = K[j*d + i]

    for (int i = 0; i < d; i++)
        for (int j = 0; j < N; j++)
            Kt[i * N + j] = K[j * d + i];

    float scale = 1.0f / sqrtf((float)d);

    auto t0 = std::chrono::high_resolution_clock::now();

    // S = Q · Kᵀ   (M=N, N=N, K=d)
    gemm_cpu(Q, Kt, S, N, N, d);
    for (int i = 0; i < N * N; i++) S[i] *= scale;
    softmax_cpu(S, N);
    // O = S · V    (M=N, N=d, K=N)
    gemm_cpu(S, V, O, N, d, N);

    auto t1 = std::chrono::high_resolution_clock::now();

    delete[] S;
    delete[] Kt;

    LevelResult r;
    r.name     = "L1 CPU Serial";
    r.kernel_ms = 0.0f;
    r.e2e_ms   = std::chrono::duration<float, std::milli>(t1 - t0).count();
    r.max_err  = 0.0f;
    r.ok       = true;
    r.speedup  = 1.0f;
    return r;
}
