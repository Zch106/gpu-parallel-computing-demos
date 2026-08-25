// L1：CPU 串行基线（单线程逐像素）
// 使用与 GPU 层完全相同的 trace_ray 模板，保证结果可比，作为正确参考。
#include "rt_common.cuh"
#include "rt_api.h"

void rt_cpu_serial(const Scene& s, vec3* fb, int max_depth) {
    GlobalInter inter;
    inter.spheres = s.spheres;
    inter.nsph = s.nsph;
    inter.lights = s.lights;
    inter.nlight = s.nlight;

    int W = s.cam.width, H = s.cam.height;
    float fov = s.cam.fov;
    float inv_f = 1.f / (2.f * tanf(fov / 2.f));

    for (int pix = 0; pix < W * H; pix++) {
        int i = pix % W;
        int j = pix / W;
        float dx = (i + 0.5f) - W / 2.f;
        float dy = -(j + 0.5f) + H / 2.f;
        float dz = -H * inv_f;
        vec3 dir = vec3(dx, dy, dz).normalized();
        fb[pix] = trace_ray(vec3(0, 0, 0), dir, max_depth, inter);
    }
}
