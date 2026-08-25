// L2：CPU 多核并行基线（OpenMP）
// 仅把像素循环并行化，算法与 L1 完全一致 -> 结果应与 L1 逐像素相同。
#include "rt_common.cuh"
#include "rt_api.h"
#include <omp.h>

void rt_cpu_omp(const Scene& s, vec3* fb, int max_depth) {
    GlobalInter inter;
    inter.spheres = s.spheres;
    inter.nsph = s.nsph;
    inter.lights = s.lights;
    inter.nlight = s.nlight;

    int W = s.cam.width, H = s.cam.height;
    float fov = s.cam.fov;
    float inv_f = 1.f / (2.f * tanf(fov / 2.f));
    int total = W * H;

#pragma omp parallel for schedule(dynamic)
    for (int pix = 0; pix < total; pix++) {
        int i = pix % W;
        int j = pix / W;
        float dx = (i + 0.5f) - W / 2.f;
        float dy = -(j + 0.5f) + H / 2.f;
        float dz = -H * inv_f;
        vec3 dir = vec3(dx, dy, dz).normalized();
        fb[pix] = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
    }
}
