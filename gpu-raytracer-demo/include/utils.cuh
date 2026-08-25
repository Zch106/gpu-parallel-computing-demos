#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include "vec3.h"

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

// 正确性校验：一次性算出 max / mean / 命中率三项指标
// 光线追踪中，百万像素里只要有几条擦边光线在 CPU/GPU 走了不同路径
// （命中球 vs 命中背景，色差可达 0.5+），就会把 max 拉高。
// 因此用 mean + 命中率 作为通过判据更合理。
struct ErrStats {
    float max_err = 0;
    float mean_err = 0;
    float pass_rate = 0;  // 逐通道误差 < tol 的像素占比 [0,1]
};

static ErrStats error_stats(const vec3* a, const vec3* b, int n, float tol = 1e-2f) {
    ErrStats s;
    double sum = 0.0;
    int pass = 0;
    for (int i = 0; i < n; i++) {
        float ex = fabsf(a[i].x - b[i].x);
        float ey = fabsf(a[i].y - b[i].y);
        float ez = fabsf(a[i].z - b[i].z);
        float em = fmaxf(ex, fmaxf(ey, ez));
        if (em > s.max_err) s.max_err = em;
        sum += ex + ey + ez;
        if (em < tol) pass++;
    }
    s.mean_err = (float)(sum / (3.0 * n));
    s.pass_rate = (float)((double)pass / n);
    return s;
}

// 设备端：分配显存并从主机拷贝
static float* dev_dup(const float* h, int n) {
    float* d = nullptr;
    CHECK(cudaMalloc(&d, n * sizeof(float)));
    CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// 设备端：仅分配显存（未初始化）
static void* dev_alloc(size_t bytes) {
    void* d = nullptr;
    CHECK(cudaMalloc(&d, bytes));
    return d;
}

// 写 PPM (P6) 图像，零依赖
static void write_ppm(const char* path, const vec3* fb, int W, int H) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W * H; i++) {
        const vec3& c = fb[i];
        float mx = fmaxf(1.f, fmaxf(c.x, fmaxf(c.y, c.z)));  // 简单 tonemap，避免过曝
        unsigned char r = (unsigned char)(255 * c.x / mx);
        unsigned char g = (unsigned char)(255 * c.y / mx);
        unsigned char b = (unsigned char)(255 * c.z / mx);
        ofs << r << g << b;
    }
}

#endif  // UTILS_CUH
