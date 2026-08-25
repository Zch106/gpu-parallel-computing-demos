#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>

// 基于 CUDA Event 的计时器：只包住 GPU kernel 计算，排除 H2D/D2H 拷贝开销。
struct CudaTimer {
    cudaEvent_t start_, stop_;
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void begin() { cudaEventRecord(start_); }
    void end() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }
    float ms() {
        float t = 0.0f;
        cudaEventElapsedTime(&t, start_, stop_);
        return t;
    }
};

#endif  // TIMER_CUH
