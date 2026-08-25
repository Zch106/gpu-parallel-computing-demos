// L6：GPU BVH 加速 —— 在 L3 的基础上，把"遍历全部球"换成"栈式 AABB 遍历到叶子再测少量球"。
// 球数越多，加速越明显；结果应与 L3 在浮点容差内逐像素一致（BVH 只加速、不改结果）。
#include "rt_common.cuh"
#include "rt_api.h"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

void build_bvh(const Sphere* h_sph, int n, BVHNode*& d_nodes, int*& d_leaf, int& nnodes);

__global__ void rt_bvh_kernel(const Sphere* __restrict__ spheres, int nsph,
                             const Light* __restrict__ lights, int nlight,
                             const BVHNode* __restrict__ nodes, const int* __restrict__ leaf_idx,
                             vec3* fb, int W, int H, float fov, int max_depth) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (pix >= total) return;
    int i = pix % W, j = pix / W;
    float dx = (i + 0.5f) - W / 2.f;
    float dy = -(j + 0.5f) + H / 2.f;
    float dz = -H / (2.f * tanf(fov / 2.f));
    vec3 dir = vec3(dx, dy, dz).normalized();
    BVHInter inter;
    inter.spheres = spheres;
    inter.nsph = nsph;
    inter.lights = lights;
    inter.nlight = nlight;
    inter.nodes = nodes;
    inter.leaf_idx = leaf_idx;
    fb[pix] = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
}

void rt_gpu_bvh(const Scene& s, vec3* fb, int max_depth,
                float& kernel_ms, float& e2e_ms) {
    int W = s.cam.width, H = s.cam.height, total = W * H;
    size_t sz = total * sizeof(vec3);

    Sphere* d_spheres = (Sphere*)dev_alloc(s.nsph * sizeof(Sphere));
    Light* d_lights = (Light*)dev_alloc(s.nlight * sizeof(Light));
    vec3* d_fb = (vec3*)dev_alloc(sz);
    CHECK(cudaMemcpy(d_spheres, s.spheres, s.nsph * sizeof(Sphere), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lights, s.lights, s.nlight * sizeof(Light), cudaMemcpyHostToDevice));

    BVHNode* d_nodes = nullptr;
    int* d_leaf = nullptr;
    int nnodes = 0;
    build_bvh(s.spheres, s.nsph, d_nodes, d_leaf, nnodes);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // BVH 遍历 + 迭代式 trace_ray 的局部变量较多，适当增大 GPU 线程调用栈
    CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    rt_bvh_kernel<<<blocks, threads>>>(d_spheres, s.nsph, d_lights, s.nlight,
                                       d_nodes, d_leaf, d_fb, W, H, s.cam.fov, max_depth);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    CudaTimer timer;
    timer.begin();
    rt_bvh_kernel<<<blocks, threads>>>(d_spheres, s.nsph, d_lights, s.nlight,
                                       d_nodes, d_leaf, d_fb, W, H, s.cam.fov, max_depth);
    timer.end();
    auto t1 = std::chrono::high_resolution_clock::now();

    kernel_ms = timer.ms();
    e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    CHECK(cudaMemcpy(fb, d_fb, sz, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_spheres));
    CHECK(cudaFree(d_lights));
    CHECK(cudaFree(d_fb));
    CHECK(cudaFree(d_nodes));
    CHECK(cudaFree(d_leaf));
}
