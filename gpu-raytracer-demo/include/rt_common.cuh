#ifndef RT_COMMON_CUH
#define RT_COMMON_CUH

#include "vec3.h"
#include "scene.h"

// 一次相交查询的结果
struct Hit {
    bool hit = false;
    float t = 1e10f;
    vec3 point = {0, 0, 0};
    vec3 N = {0, 0, 0};
    int mat_idx = -2;  // >=0: 球体索引; -1: 棋盘格地面; -2: 未命中
};

// ---------- 全局内存版 Intersector（CPU 基线 / GPU 朴素 / GPU 优化 / BVH 共用）----------
struct GlobalInter {
    const Sphere* spheres = nullptr;
    int nsph = 0;
    const Light* lights = nullptr;
    int nlight = 0;

    __host__ __device__ bool intersect(const vec3& o, const vec3& d, Hit& h) const {
        h.hit = false;
        h.t = 1e10f;
        h.mat_idx = -2;

        // 棋盘格地面（y = -4，限定区域），与原版一致
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

        for (int i = 0; i < nsph; i++) {
            const Sphere& s = spheres[i];
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
        return h.hit;
    }
    __host__ __device__ Material mat(int idx) const { return spheres[idx].material; }
    __host__ __device__ vec3 light_pos(int i) const { return lights[i].position; }
};

// ---------- BVH 节点与 BVH 版 Intersector ----------
struct BVHNode {
    vec3 bmin, bmax;
    int left = -1;   // 子节点索引；<0 表示叶子
    int right = -1;
    int start = 0;   // 叶子：sphere 索引区间 [start, start+count)
    int count = 0;
};

__host__ __device__ inline bool aabb_hit(const vec3& o, const vec3& d,
                                         const vec3& bmin, const vec3& bmax) {
    float tmin = -1e30f, tmax = 1e30f;
    for (int a = 0; a < 3; a++) {
        float inv = 1.f / d[a];
        float t0 = (bmin[a] - o[a]) * inv;
        float t1 = (bmax[a] - o[a]) * inv;
        if (inv < 0.f) {
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        tmin = fmaxf(tmin, t0);
        tmax = fminf(tmax, t1);
    }
    return tmax >= fmaxf(tmin, 0.f);
}

struct BVHInter {
    const Sphere* spheres = nullptr;
    int nsph = 0;
    const Light* lights = nullptr;
    int nlight = 0;
    const BVHNode* nodes = nullptr;
    const int* leaf_idx = nullptr;  // 叶子节点引用的 sphere 索引

    __host__ __device__ bool intersect(const vec3& o, const vec3& d, Hit& h) const {
        h.hit = false;
        h.t = 1e10f;
        h.mat_idx = -2;
        // 棋盘格地面（与全局版一致，否则地面会消失）
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
        int nstack[32];
        int ns = 0;
        nstack[ns++] = 0;  // 根节点
        while (ns > 0) {
            int ni = nstack[--ns];
            const BVHNode& nd = nodes[ni];
            if (!aabb_hit(o, d, nd.bmin, nd.bmax)) continue;
            if (nd.left < 0) {  // 叶子
                for (int k = nd.start; k < nd.start + nd.count; k++) {
                    int si = leaf_idx[k];
                    const Sphere& s = spheres[si];
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
                        h.mat_idx = si;
                    }
                }
            } else {
                nstack[ns++] = nd.left;
                nstack[ns++] = nd.right;
            }
        }
        return h.hit;
    }
    __host__ __device__ Material mat(int idx) const { return spheres[idx].material; }
    __host__ __device__ vec3 light_pos(int i) const { return lights[i].position; }
};

// ---------- 统一的光线追踪核心（迭代栈实现递归反射/折射）----------
// 注意：CUDA 不支持常规递归，这里用定长栈把 cast_ray 的递归展开成迭代。
// 所有层级共用此函数，保证结果可比；L4（__constant__ 场景）单独写一份以真正命中 constant cache。
template <typename Inter>
__host__ __device__ vec3 trace_ray(vec3 orig, vec3 dir, int max_depth, const Inter& inter) {
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
        bool ok = inter.intersect(e.o, e.d, h);
        if (!ok || e.depth >= max_depth) {
            color = color + vec3(0.2f, 0.7f, 0.8f) * e.w;  // 背景色
            continue;
        }
        vec3 point = h.point;
        vec3 N = h.N;
        Material m = (h.mat_idx >= 0) ? inter.mat(h.mat_idx) : checker_material(point);

        float diffuse_int = 0.f, spec_int = 0.f;
        for (int l = 0; l < inter.nlight; l++) {
            vec3 lp = inter.light_pos(l);
            vec3 ld = (lp - point).normalized();
            // 阴影射线（微移交点避免自交）
            vec3 so = (ld * N < 0) ? point - N * 1e-3f : point + N * 1e-3f;
            Hit sh;
            if (inter.intersect(so, ld, sh)) {
                if ((sh.point - point).norm() < (lp - point).norm()) continue;
            }
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

#endif  // RT_COMMON_CUH
