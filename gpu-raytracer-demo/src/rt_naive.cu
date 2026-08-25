// L3：GPU 朴素并行 —— 每像素一个线程，场景（球/光）放在全局内存
// 这是"并行本身"的第一波加速；后续 L4/L5/L6 在此之上做优化。
#include "rt_common.cuh"
#include "rt_api.h"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

__global__ void rt_naive_kernel(const Sphere* __restrict__ spheres, int nsph,
                                const Light* __restrict__ lights, int nlight,
                                vec3* fb, int W, int H, float fov, int max_depth) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (pix >= total) return;
    int i = pix % W, j = pix / W;
    float dx = (i + 0.5f) - W / 2.f;
    float dy = -(j + 0.5f) + H / 2.f;
    float dz = -H / (2.f * tanf(fov / 2.f));
    vec3 dir = vec3(dx, dy, dz).normalized();
    GlobalInter inter;
    inter.spheres = spheres;
    inter.nsph = nsph;
    inter.lights = lights;
    inter.nlight = nlight;
    fb[pix] = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
}

void rt_gpu_naive(const Scene& s, vec3* fb, int max_depth,
                  float& kernel_ms, float& e2e_ms) {
    int W = s.cam.width, H = s.cam.height, total = W * H;
    size_t sz = total * sizeof(vec3);

    Sphere* d_spheres = (Sphere*)dev_alloc(s.nsph * sizeof(Sphere));
    Light* d_lights = (Light*)dev_alloc(s.nlight * sizeof(Light));
    vec3* d_fb = (vec3*)dev_alloc(sz);
    CHECK(cudaMemcpy(d_spheres, s.spheres, s.nsph * sizeof(Sphere), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lights, s.lights, s.nlight * sizeof(Light), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // warmup（吸收 CUDA context 初始化开销，避免污染计时）
    rt_naive_kernel<<<blocks, threads>>>(d_spheres, s.nsph, d_lights, s.nlight,
                                         d_fb, W, H, s.cam.fov, max_depth);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    CudaTimer timer;
    timer.begin();
    rt_naive_kernel<<<blocks, threads>>>(d_spheres, s.nsph, d_lights, s.nlight,
                                         d_fb, W, H, s.cam.fov, max_depth);
    timer.end();
    auto t1 = std::chrono::high_resolution_clock::now();

    kernel_ms = timer.ms();
    e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    CHECK(cudaMemcpy(fb, d_fb, sz, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_spheres));
    CHECK(cudaFree(d_lights));
    CHECK(cudaFree(d_fb));
}
