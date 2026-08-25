# GPU 并行计算演示：光线追踪（RayTracing）加速实验

> 课程助教用 demo：用一个**计算量更大**的程序展示 GPU 并行计算的加速能力。
> 参考实现：[ssloy/tinyraytracer](https://github.com/ssloy/tinyraytracer)
> （逐像素独立：每条像素发一条主光线 → 递归反射/折射 → 对每个光源做阴影检测）

## 与「注意力 demo」的关系
- 同样的**教学范式**：CPU 基线 → GPU 朴素并行 → 并行之上的数据流/算法优化 → 加速比报告。
- 关键区别：**光线追踪不能映射到 Tensor Core**（没有矩阵乘语义），所以本 demo 的「终极优化」换成更适合它的 **BVH 加速结构**。这是很好的教学点——*优化手段要贴合负载特征*。
- 输出是**可渲染的 PPM 图像**（比注意力矩阵更直观，适合课堂展示）。

## 优化阶梯
| 层级 | 方法 | 关键技术点 | 教学含义 |
|---|---|---|---|
| L1 | CPU 串行 | 逐像素单线程 `cast_ray`（迭代栈实现递归） | 性能基准 & 正确参考 |
| L2 | CPU OpenMP 多核 | `#pragma omp parallel for` | 展示「CPU 并行 vs GPU 并行」 |
| L3 | GPU 朴素并行 | 每像素 1 线程，场景全局内存 | 「并行本身」第一波加速 |
| L4 | GPU `__constant__` 场景 | 常量内存广播+缓存，削减全局流量 | **并行之上的数据流优化** |
| L5 | GPU 深度优化 | `float4` 向量化加载 + 循环展开 + 16B 合并写 | 并行之上再榨带宽/延迟 |
| L6 | GPU **BVH** | CPU 建层次包围盒，GPU 栈式 AABB 遍历 | 多球场景下的算法级跃迁 |

## 构建与运行
```bat
build.bat                 :: Windows 一键编译（自动定位 VS 环境）
```
或（WSL / 装有 make 的环境）：
```bash
make                      :: 生成 raytracer_demo
```

```bash
raytracer_demo                                  :: 默认重负载：1024x1024, ~260 球, 深度 5
raytracer_demo --width 512 --height 512 --spheres 64 --depth 4   :: 轻量（快速预览 / 验证）
raytracer_demo --spheres 512 --depth 6            :: 更重（放大 GPU 与 BVH 增益）
raytracer_demo --save-all                        :: 把每一层的渲染图都存成 ppm
```

## 输出
- 终端打印各层耗时 / 加速比 / 与 L1 的误差表（max / mean / 命中率三项）。
- `ref.ppm`（L1 参考图）、`out.ppm`（L6 结果图），`--save-all` 时各层都存。
- `results.md`：自动生成的 Markdown 报告（含加速阶梯说明与解读）。
- 正确性判据：**mean 误差 < 1e-2 且命中率 > 99%**。max 误差仅供参考——百万像素中极少数擦边光线可能在 CPU/GPU 上走不同路径（命中球面 vs 命中背景），导致 max 偏大，这不是 bug。

## 工程说明
- 纯 C++/CUDA + OpenMP（编译器内置），**无 Python、无第三方性能库**。
- `include/rt_common.cuh` 中的 `trace_ray` 模板是被 L1–L3/L5/L6 共用的同一套光线追踪核心（迭代栈版），保证结果可比；L4 单独写一份以真正命中 constant cache。
- 编译已固化 `-Xcompiler /utf-8`（中文系统防 C4819）与 `-ccbin` 指向 `cl.exe`。

## 注意事项
- 常量内存上限 64KB，L4 最多约 1024 个球（超出会自动提示并退回朴素行为）。
- 反射/折射最大深度默认 5（栈容量保护上限 10），调大会显著增加每像素计算量。
- 光线追踪瓶颈主要在算力，故 L4→L5 提升有限；**球体越多，L6(BVH) 的加速越明显**。
- `refract` 函数已改为非递归实现（原版递归会溢出 GPU 线程调用栈，尤其配合 BVH 遍历时）。
- 进阶扩展：若想和工业级库对照，可引入 NVIDIA **OptiX**（需单独安装 SDK，超出本 demo 轻量化范围，故未内置）。
