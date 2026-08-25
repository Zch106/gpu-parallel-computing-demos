// 主程序：编排 L1(CPU串行) -> L2(CPU OpenMP) -> L3~L6(GPU 各优化层)
// 逐项计时 + 与 L1 做数值校验 + 写 PPM 图像 + 生成 Markdown 报告。
#include "rt_common.cuh"
#include "rt_api.h"
#include "utils.cuh"
#include "vec3.h"
#include "scene.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>

static void make_material(Material& m, vec3 diff, float a0, float a1, float a2, float a3,
                         float spec, float refr) {
    m.diffuse_color = diff;
    m.albedo[0] = a0; m.albedo[1] = a1; m.albedo[2] = a2; m.albedo[3] = a3;
    m.specular_exponent = spec;
    m.refractive_index = refr;
}

// 构建场景：4 个参考球 + 3 个光源 + nrandom 个随机球（重负载来源）
static void build_scene(Scene& s, int nrandom) {
    std::vector<Sphere> sph;
    std::vector<Light> lgt;

    Material ivory, glass, rubber, mirror;
    make_material(ivory,  {0.4f, 0.4f, 0.3f}, 0.9f, 0.5f, 0.1f, 0.0f, 50.f, 1.0f);
    make_material(glass,  {0.6f, 0.7f, 0.8f}, 0.0f, 0.9f, 0.1f, 0.8f, 125.f, 1.5f);
    make_material(rubber, {0.3f, 0.1f, 0.1f}, 1.4f, 0.3f, 0.0f, 0.0f, 10.f, 1.0f);
    make_material(mirror, {1.0f, 1.0f, 1.0f}, 0.0f, 16.0f, 0.8f, 0.0f, 1425.f, 1.0f);

    sph.push_back({{-3, 0, -16}, 2, ivory});
    sph.push_back({{-1.0f, -1.5f, -12}, 2, glass});
    sph.push_back({{1.5f, -0.5f, -18}, 3, rubber});
    sph.push_back({{7, 5, -18}, 4, mirror});

    // 随机球
    srand(20240824);
    for (int k = 0; k < nrandom; k++) {
        float cx = -25.f + (rand() % 1000) / 1000.f * 50.f;
        float cy = -6.f + (rand() % 1000) / 1000.f * 12.f;
        float cz = -50.f + (rand() % 1000) / 1000.f * 42.f;
        float r = 0.6f + (rand() % 1000) / 1000.f * 2.4f;
        Material m;
        int pick = rand() % 4;
        if (pick == 0) m = ivory;
        else if (pick == 1) m = glass;
        else if (pick == 2) m = rubber;
        else m = mirror;
        // 轻微随机化颜色，增加多样性
        m.diffuse_color = m.diffuse_color * (0.7f + (rand() % 1000) / 1000.f * 0.6f);
        sph.push_back({{cx, cy, cz}, r, m});
    }

    lgt.push_back({{-20, 20, 20}, 1.f});
    lgt.push_back({{30, 50, -25}, 1.f});
    lgt.push_back({{30, 20, 30}, 1.f});

    s.nsph = (int)sph.size();
    s.nlight = (int)lgt.size();
    s.spheres = new Sphere[s.nsph];
    s.lights = new Light[s.nlight];
    for (int i = 0; i < s.nsph; i++) s.spheres[i] = sph[i];
    for (int i = 0; i < s.nlight; i++) s.lights[i] = lgt[i];
}

int main(int argc, char** argv) {
    int width = 1024, height = 1024, nrandom = 256, max_depth = 5;
    bool save_all = false;

    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "--width" && a + 1 < argc) width = atoi(argv[++a]);
        else if (arg == "--height" && a + 1 < argc) height = atoi(argv[++a]);
        else if ((arg == "--spheres" || arg == "-s") && a + 1 < argc) nrandom = atoi(argv[++a]);
        else if ((arg == "--depth" || arg == "-d") && a + 1 < argc) max_depth = atoi(argv[++a]);
        else if (arg == "--save-all") save_all = true;
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: raytracer_demo [--width W] [--height H] [--spheres N] [--depth D] [--save-all]\n");
            return 0;
        }
    }
    if (max_depth > 10) max_depth = 10;  // 栈容量保护

    Scene s;
    s.cam.width = width; s.cam.height = height; s.cam.fov = 1.05f;
    build_scene(s, nrandom);
    int total = width * height;

    printf("=== GPU 光线追踪并行计算 Demo ===\n");
    printf("分辨率 %dx%d, 球数 %d, 反射/折射深度 %d\n\n", width, height, s.nsph, max_depth);

    std::vector<Row> rows;
    vec3* ref = new vec3[total];
    vec3* fb = nullptr;

    // ---- L1: CPU 串行（正确参考）----
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        rt_cpu_serial(s, ref, max_depth);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        Row r; r.name = "L1 CPU Serial"; r.kernel_ms = 0; r.e2e_ms = (float)ms;
        r.speedup = 1.f; r.max_err = 0; r.mean_err = 0; r.pass_rate = 1.f; r.ok = true;
        rows.push_back(r);
        printf("[L1] CPU serial: %.1f ms\n", ms);
        write_ppm("ref.ppm", ref, width, height);
    }

    // ---- L2: CPU OpenMP ----
    {
        fb = new vec3[total];
        auto t0 = std::chrono::high_resolution_clock::now();
        rt_cpu_omp(s, fb, max_depth);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ErrStats es = error_stats(fb, ref, total);
        Row r; r.name = "L2 CPU OpenMP"; r.kernel_ms = 0; r.e2e_ms = (float)ms;
        r.speedup = rows[0].e2e_ms / (float)ms; r.max_err = es.max_err;
        r.mean_err = es.mean_err; r.pass_rate = es.pass_rate;
        r.ok = es.mean_err < 1e-2f && es.pass_rate > 0.99f;
        rows.push_back(r);
        printf("[L2] CPU OpenMP: %.1f ms  speedup=%.1fx  max=%.2e mean=%.2e pass=%.2f%% %s\n",
               ms, r.speedup, es.max_err, es.mean_err, es.pass_rate * 100, r.ok ? "OK" : "NO");
        if (save_all) write_ppm("out_L2.ppm", fb, width, height);
        delete[] fb; fb = nullptr;
    }

    // ---- GPU 各层 ----
    auto run_gpu = [&](const char* name, void (*fn)(const Scene&, vec3*, int, float&, float&)) {
        fb = new vec3[total];
        float k = 0, e = 0;
        fn(s, fb, max_depth, k, e);
        ErrStats es = error_stats(fb, ref, total);
        Row r; r.name = name; r.kernel_ms = k; r.e2e_ms = e;
        r.speedup = rows[0].e2e_ms / e; r.max_err = es.max_err;
        r.mean_err = es.mean_err; r.pass_rate = es.pass_rate;
        r.ok = es.mean_err < 1e-2f && es.pass_rate > 0.99f;
        rows.push_back(r);
        printf("[%s] kernel=%.2f ms  e2e=%.2f ms  speedup=%.1fx  max=%.2e mean=%.2e pass=%.2f%% %s\n",
               name, k, e, r.speedup, es.max_err, es.mean_err, es.pass_rate * 100, r.ok ? "OK" : "NO");
        if (save_all) {
            std::string fn2 = std::string("out_") + name + ".ppm";
            write_ppm(fn2.c_str(), fb, width, height);
        }
        return fb;  // 调用方负责 delete
    };

    fb = run_gpu("L3 GPU Naive", rt_gpu_naive);
    delete[] fb; fb = nullptr;
    fb = run_gpu("L4 GPU Constant", rt_gpu_constant);
    delete[] fb; fb = nullptr;
    fb = run_gpu("L5 GPU Opt", rt_gpu_opt);
    delete[] fb; fb = nullptr;
    fb = run_gpu("L6 GPU BVH", rt_gpu_bvh);
    // 用 L6 结果作为最终输出图
    write_ppm("out.ppm", fb, width, height);
    delete[] fb; fb = nullptr;

    // ---- 报告 ----
    write_report("results.md", rows, width, height, s.nsph, max_depth);
    printf("\n图像已写出：ref.ppm (L1参考) / out.ppm (L6结果)\n");
    printf("报告已生成：results.md\n");

    delete[] ref;
    delete[] s.spheres;
    delete[] s.lights;
    return 0;
}
