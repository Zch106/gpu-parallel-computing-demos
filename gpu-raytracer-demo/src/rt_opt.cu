// L5：GPU 深度优化 —— 在"已并行"的基础上再榨带宽/延迟：
//   1) 球数据按 float4(center.xyz, radius) 打包，用 __ldg 做向量化(128-bit)只读缓存加载；
//   2) 球遍历循环展开（#pragma unroll）；
//   3) 帧缓冲以 float4 合并写回全局内存（16 字节对齐，最大化访存合并）。
// 注意：光线追踪的瓶颈主要在算力而非带宽，故 L4->L5 提升有限；
//       真正"换算法"的大幅加速在 L6（BVH）。
#include "rt_common.cuh"
#include "rt_api.h"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

struct OptInter {
    const float4* cr = nullptr;  // 打包: (center.x, center.y, center.z, radius)
    const Sphere* sph = nullptr;  // 仅取材质
    int nsph = 0;
    const Light* lights = nullptr;
    int nlight = 0;

    __host__ __device__ bool intersect(const vec3& o, const vec3& d, Hit& h) const {
        h.hit = false;
        h.t = 1e10f;
        h.mat_idx = -2;
        // 棋盘格地面
        if (fabsf(d.y) > 1e-3f) {
            float t = -(o.y + 4.f) / d.y;
            if (t > 1e-3f && t < h.t) {
                vec3 p = o + d * t;
                if (fabsf(p.x) < 10.f && p.z < -10.f && p.z > -30.f) {
                    h.hit = true;
                    h.t = t;
                    h.point = p;
                    h.N = vec3(0, 1, 0);
                    h.mat_idx = -1;
                }
            }
        }
        // 热循环只用到 center + radius，打包成 float4 向量化加载
#pragma unroll 4
        for (int i = 0; i < nsph; i++) {
            float4 c;
#ifdef __CUDA_ARCH__
            c = __ldg(&cr[i]);
#else
            c = cr[i];
#endif
            vec3 center(c.x, c.y, c.z);
            float radius = c.w;
            vec3 L = center - o;
            float tca = L * d;
            float d2 = L * L - tca * tca;
            if (d2 > radius * radius) continue;
            float thc = sqrtf(radius * radius - d2);
            float t0 = tca - thc, t1 = tca + thc;
            float tt;
            if (t0 > 1e-3f) tt = t0;
            else if (t1 > 1e-3f) tt = t1;
            else continue;
            if (tt < h.t) {
                h.hit = true;
                h.t = tt;
                h.point = o + d * tt;
                h.N = (o + d * tt - center).normalized();
                h.mat_idx = i;
            }
        }
        return h.hit;
    }
    __host__ __device__ Material mat(int idx) const { return sph[idx].material; }
    __host__ __device__ vec3 light_pos(int i) const { return lights[i].position; }
};

__global__ void rt_opt_kernel(const float4* __restrict__ cr, const Sphere* __restrict__ sph,
                             int nsph, const Light* __restrict__ lights, int nlight,
                             float4* fb4, int W, int H, float fov, int max_depth) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (pix >= total) return;
    int i = pix % W, j = pix / W;
    float dx = (i + 0.5f) - W / 2.f;
    float dy = -(j + 0.5f) + H / 2.f;
    float dz = -H / (2.f * tanf(fov / 2.f));
    vec3 dir = vec3(dx, dy, dz).normalized();
    OptInter inter;
    inter.cr = cr;
    inter.sph = sph;
    inter.nsph = nsph;
    inter.lights = lights;
    inter.nlight = nlight;
    vec3 c = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
    fb4[pix] = make_float4(c.x, c.y, c.z, 0.f);  // 16 字节合并写
}

void rt_gpu_opt(const Scene& s, vec3* fb, int max_depth,
                float& kernel_ms, float& e2e_ms) {
    int W = s.cam.width, H = s.cam.height, total = W * H;

    // 打包 center+radius 为 float4
    float4* h_cr = new float4[s.nsph];
    for (int i = 0; i < s.nsph; i++) {
        h_cr[i] = make_float4(s.spheres[i].center.x, s.spheres[i].center.y,
                             s.spheres[i].center.z, s.spheres[i].radius);
    }
    float4* d_cr = (float4*)dev_alloc(s.nsph * sizeof(float4));
    Sphere* d_sph = (Sphere*)dev_alloc(s.nsph * sizeof(Sphere));
    Light* d_lights = (Light*)dev_alloc(s.nlight * sizeof(Light));
    float4* d_fb4 = (float4*)dev_alloc(total * sizeof(float4));
    CHECK(cudaMemcpy(d_cr, h_cr, s.nsph * sizeof(float4), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_sph, s.spheres, s.nsph * sizeof(Sphere), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_lights, s.lights, s.nlight * sizeof(Light), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rt_opt_kernel<<<blocks, threads>>>(d_cr, d_sph, s.nsph, d_lights, s.nlight,
                                       d_fb4, W, H, s.cam.fov, max_depth);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    CudaTimer timer;
    timer.begin();
    rt_opt_kernel<<<blocks, threads>>>(d_cr, d_sph, s.nsph, d_lights, s.nlight,
                                       d_fb4, W, H, s.cam.fov, max_depth);
    timer.end();
    auto t1 = std::chrono::high_resolution_clock::now();

    kernel_ms = timer.ms();
    e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    float4* h_fb4 = new float4[total];
    CHECK(cudaMemcpy(h_fb4, d_fb4, total * sizeof(float4), cudaMemcpyDeviceToHost));
    for (int i = 0; i < total; i++) fb[i] = vec3(h_fb4[i].x, h_fb4[i].y, h_fb4[i].z);

    delete[] h_cr;
    delete[] h_fb4;
    CHECK(cudaFree(d_cr));
    CHECK(cudaFree(d_sph));
    CHECK(cudaFree(d_lights));
    CHECK(cudaFree(d_fb4));
}
