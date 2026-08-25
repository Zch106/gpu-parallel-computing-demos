#ifndef RT_API_H
#define RT_API_H

#include "scene.h"
#include "vec3.h"
#include <vector>
#include <string>

// 各层渲染函数的统一返回/签名，供 main.cu 调用
// fb 由函数内部 new 出来（长度 = W*H），调用方负责 delete[]。
// kernel_ms：仅 GPU kernel 计算时间（不含 H2D/D2H）；CPU 层置 0。
// e2e_ms：整段耗时（含拷贝 / 分配），用于加速比计算。

void rt_cpu_serial(const Scene& s, vec3* fb, int max_depth);
void rt_cpu_omp(const Scene& s, vec3* fb, int max_depth);

void rt_gpu_naive(const Scene& s, vec3* fb, int max_depth,
                  float& kernel_ms, float& e2e_ms);
void rt_gpu_constant(const Scene& s, vec3* fb, int max_depth,
                     float& kernel_ms, float& e2e_ms);
void rt_gpu_opt(const Scene& s, vec3* fb, int max_depth,
                float& kernel_ms, float& e2e_ms);
void rt_gpu_bvh(const Scene& s, vec3* fb, int max_depth,
                float& kernel_ms, float& e2e_ms);

// 报告用的一行结果
struct Row {
    std::string name;
    float kernel_ms = 0;
    float e2e_ms = 0;
    float speedup = 1;
    float max_err = 0;
    float mean_err = 0;
    float pass_rate = 0;  // 逐通道误差 < tol 的像素占比
    bool ok = false;
};

void write_report(const char* path, const std::vector<Row>& rows,
                  int W, int H, int nsph, int depth);

#endif  // RT_API_H
