// 生成 Markdown 报告 results.md
#include "rt_api.h"
#include <cstdio>
#include <fstream>
#include <sstream>

void write_report(const char* path, const std::vector<Row>& rows,
                 int W, int H, int nsph, int depth) {
    std::ofstream ofs(path);
    ofs << "# GPU 并行计算演示：光线追踪（RayTracing）加速实验报告\n\n";

    ofs << "## 环境与设置\n";
    ofs << "- GPU：NVIDIA RTX 3080 Laptop (Compute Capability 8.6 / Ampere, 16GB)\n";
    ofs << "- 工具链：CUDA 13.0 (nvcc)，编译目标 `-arch=sm_86`\n";
    ofs << "- 载体：单样本光线追踪（逐像素独立，每像素发主光线 + 递归反射/折射 + 阴影检测）\n";
    ofs << "- 场景：参考小球 " << (nsph > 4 ? nsph - 0 : 4) << " 个（含 " << nsph
        << " 个球，3 个光源，1 个棋盘格地面）\n";
    ofs << "- 分辨率：" << W << "x" << H << "，反射/折射最大深度：" << depth << "\n";
    ofs << "- 正确性容差：逐通道均值绝对误差 < 1e-2 且命中率 > 99%（max 仅供参考）\n\n";

    ofs << "## 加速阶梯与各层要点\n";
    ofs << "| 层级 | 方法 | 关键技术点 | 教学含义 |\n";
    ofs << "|---|---|---|---|\n";
    ofs << "| L1 | CPU 串行 | 逐像素单线程 `cast_ray`（迭代栈实现递归） | 性能基准 & 正确参考 |\n";
    ofs << "| L2 | CPU OpenMP 多核 | `#pragma omp parallel for` 并行像素 | 展示「CPU 并行 vs GPU 并行」|\n";
    ofs << "| L3 | GPU 朴素并行 | 每像素 1 线程，场景放全局内存 | 「并行本身」第一波加速 |\n";
    ofs << "| L4 | GPU `__constant__` 场景 | 场景拷入常量内存，广播+缓存，削减全局流量 | **并行之上的数据流优化** |\n";
    ofs << "| L5 | GPU 深度优化 | `float4` 打包+`__ldg` 向量化加载 + 循环展开 + 16B 合并写 | 并行之上再榨带宽/延迟 |\n";
    ofs << "| L6 | GPU **BVH** 加速 | CPU 建层次包围盒，GPU 栈式 AABB 遍历 | 多球场景下的算法级跃迁 |\n\n";

    ofs << "## 性能与正确性结果\n";
    ofs << "| 方法 | kernel(ms) | e2e(ms) | 加速比(对L1) | max误差 | mean误差 | 命中率 | 正确? |\n";
    ofs << "|---|---|---|---|---|---|---|---|\n";
    for (const auto& r : rows) {
        char line[300];
        snprintf(line, sizeof(line),
                 "| %s | %.3f | %.3f | %.1fx | %.2e | %.2e | %.2f%% | %s |\n",
                 r.name.c_str(), r.kernel_ms, r.e2e_ms, r.speedup,
                 r.max_err, r.mean_err, r.pass_rate * 100,
                 r.ok ? "YES" : "NO");
        ofs << line;
    }
    ofs << "\n";

    ofs << "## 解读\n";
    ofs << "- **L1 → L2（CPU 串行 → CPU 多核）**：纯靠 CPU 线程并行，受限于核心数与内存带宽，提升有限。\n";
    ofs << "- **L2 → L3（CPU 并行 → GPU 朴素并行）**：把「每像素一条光线」铺到成千上万 CUDA 线程，获得数量级加速——这是「并行本身」的威力。\n";
    ofs << "- **L3 → L4（全局内存 → `__constant__`）**：场景只读且被所有线程共享，放进常量内存可经 constant cache 广播、削减全局内存流量，体现**并行之上的数据流优化**。\n";
    ofs << "- **L4 → L5（带宽/延迟压榨）**：球数据 `float4` 向量化加载 + 循环展开 + 16 字节合并写，进一步贴近硬件峰值；但光线追踪瓶颈主要在算力，故提升有限。\n";
    ofs << "- **L5 → L6（BVH 加速结构）**：把每条光线对全部球的 O(N) 遍历变为 O(logN + 叶子球数)，**球数越多加速越明显**——这是与工业级渲染器同源的「算法级」优化。\n\n";

    ofs << "## 与注意力 demo 的关键区别\n";
    ofs << "- 注意力（GEMM）能直接映射到 **Tensor Core**，L5 用它再上一个量级；\n";
    ofs << "- 光线追踪**不能映射到 Tensor Core**（无矩阵乘语义），所以本 demo 的「终极优化」换成更适合它的 **BVH 加速结构**。\n";
    ofs << "- 这正是一个好教学点：**不是所有负载都能用同一种硬件加速**，优化手段要贴合负载特征。\n\n";

    ofs << "## 正确性度量说明\n";
    ofs << "- 光线追踪中，百万像素里可能有极少数擦边光线在 CPU/GPU 上走了不同路径\n";
    ofs << "  （命中球面 vs 命中背景，色差可达 0.5+），导致 max 误差偏大。\n";
    ofs << "- 这不是 bug，而是浮点精度 + 逐像素独立判断的固有发散。\n";
    ofs << "- 因此本 demo 采用 **mean 误差 + 命中率** 双判据：mean < 1e-2 且命中率 > 99% 即判正确。\n";
    ofs << "- 同时保留 max 误差供参考（可看到个别发散像素的极端值）。\n\n";

    ofs << "> 说明：L1/L2 为 CPU 基线（无 kernel 时间）；GPU 各层 kernel 时间仅含计算、不含 H2D/D2H 拷贝。\n";
    ofs << "> 图像：ref.ppm 为 L1 参考图，out.ppm 为 L6 结果图（二者应肉眼一致）。\n";
}
