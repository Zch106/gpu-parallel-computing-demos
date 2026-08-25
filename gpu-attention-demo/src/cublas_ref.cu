// ★ 外部对照基线：cuBLAS（工业级性能库）
// 仅用于「对比参考」，不参与教学阶梯。
//   - attention_cublas_fp32：cublasSgemm (FP32)，对照 L2–L4
//   - attention_cublas_fp16：cublasGemmEx (FP16 + Tensor Op)，对照 L5
//
// 行主序 C = A*B 的 cuBLAS 调用要点：
//   把「第二个操作数」作为 cuBLAS 的 A、把「第一个操作数」作为 cuBLAS 的 B，
//   并令 m = C 的列数、n = C 的行数、k = 内维，lda/ldb/ldc 取各自行主序行跨度。
#include "attention.cuh"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>
#include <cublas_v2.h>

// ---------- FP32 对照 ----------
LevelResult attention_cublas_fp32(const float* Q, const float* K, const float* V,
                                  float* O, int N, int d) {
    auto t0 = std::chrono::high_resolution_clock::now();

    float* hKt = new float[d * N];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < N; j++)
            hKt[i * N + j] = K[j * d + i];

    float *d_Q = dev_dup(Q, N * d);
    float *d_K = dev_dup(K, N * d);
    float *d_V = dev_dup(V, N * d);
    float *d_Kt = dev_dup(hKt, d * N);
    float *d_S = dev_alloc(N * N);
    float *d_O = dev_alloc(N * d);

    float scale = 1.0f / sqrtf((float)d);
    const float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CudaTimer timer;
    float kms = 0.0f;
    const int ITER = 6;
    for (int it = 0; it < ITER; it++) {
        timer.begin();
        // S = Q · Kᵀ   (cuBLAS: A_arg=Kt, B_arg=Q, m=N, n=N, k=d)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, d,
                          &alpha, d_Kt, N, d_Q, d, &beta, d_S, N));
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        // O = P · V     (cuBLAS: A_arg=V, B_arg=P, m=d, n=N, k=N)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
                          &alpha, d_V, d, d_S, N, &beta, d_O, d));
        CHECK(cudaDeviceSynchronize());
        timer.end();
        if (it > 0) kms += timer.ms();
    }
    kms /= (ITER - 1);

    CUBLAS_CHECK(cublasDestroy(handle));
    CHECK(cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost));
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2e = std::chrono::duration<float, std::milli>(t1 - t0).count();

    CHECK(cudaFree(d_Q)); CHECK(cudaFree(d_K)); CHECK(cudaFree(d_V));
    CHECK(cudaFree(d_Kt)); CHECK(cudaFree(d_S));
    CHECK(cudaFree(d_O));
    delete[] hKt;

    LevelResult r;
    r.name = "cuBLAS FP32 (baseline)";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}

// ---------- FP16 (Tensor Op) 对照 ----------
LevelResult attention_cublas_fp16(const float* Q, const float* K, const float* V,
                                  float* O, int N, int d) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int nd = N * d, nn = N * N;
    half *hQ = new half[nd], *hK = new half[nd], *hV = new half[nd];
    half *hKt = new half[d * N];
    for (int i = 0; i < nd; i++) { hQ[i] = __float2half(Q[i]); hK[i] = __float2half(K[i]); hV[i] = __float2half(V[i]); }
    for (int i = 0; i < d; i++)
        for (int j = 0; j < N; j++)
            hKt[i * N + j] = hK[j * d + i];

    half *d_Q, *d_K, *d_V, *d_Kt, *d_Ph, *d_Sh, *d_Oh;
    float *d_S, *d_P, *d_Of;
    CHECK(cudaMalloc(&d_Q, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_K, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_V, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_Kt, d * N * sizeof(half)));
    CHECK(cudaMalloc(&d_Ph, nn * sizeof(half)));
    CHECK(cudaMalloc(&d_Sh, nn * sizeof(half)));
    CHECK(cudaMalloc(&d_Oh, nd * sizeof(half)));
    CHECK(cudaMalloc(&d_S, nn * sizeof(float)));
    CHECK(cudaMalloc(&d_P, nn * sizeof(float)));
    CHECK(cudaMalloc(&d_Of, nd * sizeof(float)));

    CHECK(cudaMemcpy(d_Q, hQ, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_K, hK, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_V, hV, nd * sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Kt, hKt, d * N * sizeof(half), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)d);
    const float alpha = 1.0f, beta = 0.0f;
    int cv = (nn + 255) / 256;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CudaTimer timer;
    float kms = 0.0f;
    const int ITER = 6;
    for (int it = 0; it < ITER; it++) {
        timer.begin();
        // S = Q · Kᵀ  (FP16)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, d,
                           &alpha, d_Kt, CUDA_R_16F, N, d_Q, CUDA_R_16F, d,
                           &beta, d_Sh, CUDA_R_16F, N,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // S_half -> S_float, 再 softmax
        half2float_kernel<<<cv, 256>>>(d_Sh, d_S, nn);
        softmax_kernel<<<N, 256>>>(d_S, N, scale);
        float2half_kernel<<<cv, 256>>>(d_S, d_Ph, nn);
        // O = P · V  (FP16)
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, N, N,
                           &alpha, d_V, CUDA_R_16F, d, d_Ph, CUDA_R_16F, N,
                           &beta, d_Oh, CUDA_R_16F, d,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        half2float_kernel<<<cv, 256>>>(d_Oh, d_Of, nd);
        CHECK(cudaDeviceSynchronize());
        timer.end();
        if (it > 0) kms += timer.ms();
    }
    kms /= (ITER - 1);

    CUBLAS_CHECK(cublasDestroy(handle));
    CHECK(cudaMemcpy(O, d_Of, nd * sizeof(float), cudaMemcpyDeviceToHost));
    auto t1 = std::chrono::high_resolution_clock::now();
    float e2e = std::chrono::duration<float, std::milli>(t1 - t0).count();

    CHECK(cudaFree(d_Q)); CHECK(cudaFree(d_K)); CHECK(cudaFree(d_V));
    CHECK(cudaFree(d_Kt)); CHECK(cudaFree(d_Ph)); CHECK(cudaFree(d_Sh));
    CHECK(cudaFree(d_Oh)); CHECK(cudaFree(d_S)); CHECK(cudaFree(d_P));
    CHECK(cudaFree(d_Of));
    delete[] hQ; delete[] hK; delete[] hV; delete[] hKt;

    LevelResult r;
    r.name = "cuBLAS FP16 (Tensor Op, baseline)";
    r.kernel_ms = kms;
    r.e2e_ms = e2e;
    r.max_err = 0; r.ok = true; r.speedup = 0;
    return r;
}
