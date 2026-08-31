// L4：GPU __constant__ 场景 —— 把整个场景拷入常量内存（广播 + cache，削减全局流量）
// 这是"并行之上的数据流优化"：所有线程只读同一份场景，命中 constant cache 而非反复访问全局内存。
//
// 注意：常量内存上限 64KB。Sphere 约 52 字节 -> 本 demo 上限约 1024 个球。
//       若要更多球，请改用 L5/L6（全局/BVH）。
#include "rt_common.cuh"
#include "rt_api.h"
#include "utils.cuh"
#include "timer.cuh"
#include <chrono>

#define MAX_CONST_SPH 1024
#define MAX_CONST_LGT 16

__constant__ Sphere c_sph[MAX_CONST_SPH];
__constant__ Light c_lgt[MAX_CONST_LGT];
__constant__ int c_nsph = 0;
__constant__ int c_nlight = 0;

// 场景相交（直接访问 __constant__ 缓冲，真正命中 constant cache）
__device__ void scene_intersect_const(const vec3& o, const vec3& d, Hit& h) {
    h.hit = false;
    h.t = 1e10f;
    h.mat_idx = -2;
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
    for (int i = 0; i < c_nsph; i++) {
        const Sphere& s = c_sph[i];
        vec3 L = s.center - o;
        float tca = L * d;
        float d2 = L * L - tca * tca;
        if (d2 > s.radius * s.radius) continue;
        float thc = sqrtf(s.radius * s.radius - d2);
        float t0 = tca - thc, t1 = tca + thc;
        float tt;
        if (t0 > 1e-3f) tt = t0;
        else if (t1 > 1e-3f) tt = t1;
        else continue;
        if (tt < h.t) {
            h.hit = true;
            h.t = tt;
            h.point = o + d * tt;
            h.N = (o + d * tt - s.center).normalized();
            h.mat_idx = i;
        }
    }
}

// 与 trace_ray 完全一致的迭代栈逻辑，但场景直接来自 __constant__
__device__ vec3 trace_const(vec3 orig, vec3 dir, int max_depth) {
    vec3 color(0, 0, 0);
    struct SE {
        vec3 o, d;
        float w;
        int depth;
    };
    SE stack[24];
    int sp = 0;
    stack[sp++] = {orig, dir, 1.f, 0};
    while (sp > 0) {
        SE e = stack[--sp];
        Hit h;
        scene_intersect_const(e.o, e.d, h);
        if (!h.hit || e.depth >= max_depth) {
            color = color + vec3(0.2f, 0.7f, 0.8f) * e.w;
            continue;
        }
        vec3 point = h.point;
        vec3 N = h.N;
        Material m = (h.mat_idx >= 0) ? c_sph[h.mat_idx].material : checker_material(point);

        float diffuse_int = 0.f, spec_int = 0.f;
        for (int l = 0; l < c_nlight; l++) {
            vec3 lp = c_lgt[l].position;
            vec3 ld = (lp - point).normalized();
            vec3 so = (ld * N < 0) ? point - N * 1e-3f : point + N * 1e-3f;
            Hit sh;
            scene_intersect_const(so, ld, sh);
            if (sh.hit && (sh.point - point).norm() < (lp - point).norm()) continue;
            diffuse_int += fmaxf(0.f, ld * N);
            spec_int += powf(fmaxf(0.f, -reflect(-ld, N) * e.d), m.specular_exponent);
        }
        vec3 local = m.diffuse_color * diffuse_int * m.albedo[0] +
                     vec3(1, 1, 1) * spec_int * m.albedo[1];
        color = color + local * e.w;
        if (e.depth < max_depth) {
            vec3 rdir = reflect(e.d, N).normalized();
            vec3 rrefr = refract(e.d, N, m.refractive_index).normalized();
            vec3 ro = (rdir * N < 0) ? point - N * 1e-3f : point + N * 1e-3f;
            vec3 foorig = (rrefr * N < 0) ? point - N * 1e-3f : point + N * 1e-3f;
            stack[sp++] = {ro, rdir, e.w * m.albedo[2], e.depth + 1};
            stack[sp++] = {foorig, rrefr, e.w * m.albedo[3], e.depth + 1};
        }
    }
    return color;
}

__global__ void rt_const_kernel(vec3* fb, int W, int H, float fov, int max_depth) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (pix >= total) return;
    int i = pix % W, j = pix / W;
    float dx = (i + 0.5f) - W / 2.f;
    float dy = -(j + 0.5f) + H / 2.f;
    float dz = -H / (2.f * tanf(fov / 2.f));
    vec3 dir = vec3(dx, dy, dz).normalized();
    fb[pix] = trace_const(vec3(0, 0, 0), dir, max_depth);
}

// 退回路径：球数/光源数超出常量内存容量时改用全局内存 + GlobalInter。
// 数值结果与 L3 完全一致，保证 L4 在任何球数下都能正确运行而非报错中止。
__global__ void rt_const_fallback_kernel(const Sphere* sph, int nsph,
                                         const Light* lgt, int nlight,
                                         vec3* fb, int W, int H, float fov, int max_depth) {
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (pix >= total) return;
    int i = pix % W, j = pix / W;
    float dx = (i + 0.5f) - W / 2.f;
    float dy = -(j + 0.5f) + H / 2.f;
    float dz = -H / (2.f * tanf(fov / 2.f));
    vec3 dir = vec3(dx, dy, dz).normalized();
    GlobalInter inter{sph, nsph, lgt, nlight};
    fb[pix] = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
}

void rt_gpu_constant(const Scene& s, vec3* fb, int max_depth,
                    float& kernel_ms, float& e2e_ms) {
    int W = s.cam.width, H = s.cam.height, total = W * H;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // 超出常量内存容量：退回全局内存（朴素）路径，结果仍正确
    if (s.nsph > MAX_CONST_SPH || s.nlight > MAX_CONST_LGT) {
        fprintf(stderr, "[WARN] 场景规模(球=%d/光源=%d)超过常量内存上限(球=%d/光源=%d)，L4 已退回全局内存朴素路径(数值结果同 L3)\n",
                s.nsph, s.nlight, MAX_CONST_SPH, MAX_CONST_LGT);

        Sphere* d_sph = nullptr;
        Light* d_lgt = nullptr;
        CHECK(cudaMalloc(&d_sph, s.nsph * sizeof(Sphere)));
        CHECK(cudaMemcpy(d_sph, s.spheres, s.nsph * sizeof(Sphere), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&d_lgt, s.nlight * sizeof(Light)));
        CHECK(cudaMemcpy(d_lgt, s.lights, s.nlight * sizeof(Light), cudaMemcpyHostToDevice));
        vec3* d_fb = (vec3*)dev_alloc(total * sizeof(vec3));

        rt_const_fallback_kernel<<<blocks, threads>>>(d_sph, s.nsph, d_lgt, s.nlight,
                                                     d_fb, W, H, s.cam.fov, max_depth);
        CHECK(cudaDeviceSynchronize());

        auto t0 = std::chrono::high_resolution_clock::now();
        CudaTimer timer;
        timer.begin();
        rt_const_fallback_kernel<<<blocks, threads>>>(d_sph, s.nsph, d_lgt, s.nlight,
                                                     d_fb, W, H, s.cam.fov, max_depth);
        timer.end();
        auto t1 = std::chrono::high_resolution_clock::now();

        kernel_ms = timer.ms();
        e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        CHECK(cudaMemcpy(fb, d_fb, total * sizeof(vec3), cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_fb));
        CHECK(cudaFree(d_sph));
        CHECK(cudaFree(d_lgt));
        return;
    }

    CHECK(cudaMemcpyToSymbol(c_sph, s.spheres, s.nsph * sizeof(Sphere)));
    CHECK(cudaMemcpyToSymbol(c_lgt, s.lights, s.nlight * sizeof(Light)));
    CHECK(cudaMemcpyToSymbol(c_nsph, &s.nsph, sizeof(int)));
    CHECK(cudaMemcpyToSymbol(c_nlight, &s.nlight, sizeof(int)));

    vec3* d_fb = (vec3*)dev_alloc(total * sizeof(vec3));

    rt_const_kernel<<<blocks, threads>>>(d_fb, W, H, s.cam.fov, max_depth);
    CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    CudaTimer timer;
    timer.begin();
    rt_const_kernel<<<blocks, threads>>>(d_fb, W, H, s.cam.fov, max_depth);
    timer.end();
    auto t1 = std::chrono::high_resolution_clock::now();

    kernel_ms = timer.ms();
    e2e_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    CHECK(cudaMemcpy(fb, d_fb, total * sizeof(vec3), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_fb));
}
