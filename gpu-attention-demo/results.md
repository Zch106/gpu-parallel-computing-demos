# GPU 并行计算演示：自注意力（Self-Attention）加速实验报告\n
## 环境
- GPU：NVIDIA RTX 3080 Laptop (Compute Capability 8.6 / Ampere, 16GB)
- 工具链：CUDA 13.0 (nvcc)，编译目标 `-arch=sm_86`
- 载体：单样本单头自注意力 `O = softmax(Q·Kᵀ / √d) · V`，两个 GEMM（QKᵀ 与 PV）为计算热点
- 默认规模：`d=64`，`N∈{1024,2048,4096}`；FP32 容差 1e-3，FP16 容差 1e-2

## 加速阶梯与各层要点
| 层级 | 方法 | 关键技术点 | 教学含义 |
|---|---|---|---|
| L1 | CPU 串行 | 三重循环朴素 GEMM + 串行 softmax | 性能基准 & 正确性参考 |
| L2 | GPU 朴素并行 | 每线程 1 元素，全程全局内存 | “并行”带来的第一波加速 |
| L3 | 共享内存分块 | tile 入 `__shared__`，合并访存 + 数据复用 | **并行之上的数据流优化** |
| L4 | 分块+深度优化 | `float4` 向量化 + padding 消 bank conflict + 循环展开 | 并行之上再榨带宽/延迟 |
| L5 | Tensor Core | WMMA FP16→FP32，mma.sync 16×16×16 | 现代 AI 硬件的又一个量级跃迁 |
| ★  | cuBLAS（外部库）| `cublasSgemm` / `cublasGemmEx` | **仅作对比基线**，展示工业级库水平 |

## 性能与正确性结果\n### N=1024, d=64
| 方法 | kernel(ms) | e2e(ms) | 加速比(相对CPU) | 最大误差 | 正确? |
|---|---|---|---|---|---|
| L1 CPU Serial | 0 | 599.668 | 1x | 0 | YES |
| L2 GPU Naive | 0.354093 | 4.7575 | 1693.53x | 1.2666e-07 | YES |
| L3 GPU Tiled (shared mem) | 0.247181 | 4.0024 | 2426.03x | 1.2666e-07 | YES |
| L4 GPU Opt (float4+pad) | 0.202534 | 3.5443 | 2960.82x | 1.2666e-07 | YES |
| L5 GPU Tensor Core (WMMA FP16) | 0.110182 | 6.9884 | 5442.5x | 1.14217e-05 | YES |
| cuBLAS FP32 (baseline) | 0.134349 | 95.9984 | 4463.52x | 1.63913e-07 | YES |
| cuBLAS FP16 (Tensor Op, baseline) | 0.28201 | 175.716 | 2126.41x | 4.05163e-05 | YES |

### N=2048, d=64
| 方法 | kernel(ms) | e2e(ms) | 加速比(相对CPU) | 最大误差 | 正确? |
|---|---|---|---|---|---|
| L1 CPU Serial | 0 | 2429.07 | 1x | 0 | YES |
| L2 GPU Naive | 1.31072 | 13.243 | 1853.23x | 1.75089e-07 | YES |
| L3 GPU Tiled (shared mem) | 1.31051 | 11.1353 | 1853.53x | 1.75089e-07 | YES |
| L4 GPU Opt (float4+pad) | 0.692634 | 9.0778 | 3507.01x | 1.75089e-07 | YES |
| L5 GPU Tensor Core (WMMA FP16) | 0.304512 | 13.8085 | 7976.93x | 9.69134e-06 | YES |
| cuBLAS FP32 (baseline) | 0.466944 | 6.8543 | 5202.06x | 1.9744e-07 | YES |
| cuBLAS FP16 (Tensor Op, baseline) | 0.389536 | 37.9369 | 6235.81x | 2.15098e-05 | YES |

### N=4096, d=64
| 方法 | kernel(ms) | e2e(ms) | 加速比(相对CPU) | 最大误差 | 正确? |
|---|---|---|---|---|---|
| L1 CPU Serial | 0 | 11259.5 | 1x | 0 | YES |
| L2 GPU Naive | 25.3043 | 184.38 | 444.965x | 2.16067e-07 | YES |
| L3 GPU Tiled (shared mem) | 14.9744 | 97.1248 | 751.919x | 2.16067e-07 | YES |
| L4 GPU Opt (float4+pad) | 4.40545 | 40.3145 | 2555.81x | 2.16067e-07 | YES |
| L5 GPU Tensor Core (WMMA FP16) | 1.8473 | 35.2066 | 6095.13x | 5.9586e-06 | YES |
| cuBLAS FP32 (baseline) | 1.34267 | 15.3392 | 8385.92x | 2.45869e-07 | YES |
| cuBLAS FP16 (Tensor Op, baseline) | 1.84627 | 37.0354 | 6098.51x | 2.00309e-05 | YES |

## 解读
- L1→L2：仅“并行”就让计算从单线程铺到成千上万 CUDA 线程，获得**数量级（千倍）加速**——这是并行本身的力量。\n- L2→L3：引入共享内存分块，把对全局内存的重复读取变成对共享内存的复用，并满足合并访存，kernel 时间进一步下降。\n- L3→L4：在对的方向上继续压榨——`float4` 提升全局内存带宽利用率、padding 消除 bank conflict、循环展开提升 ILP，kernel 时间再降一截。\n- L4→L5：在**较大规模**（N 较大）时再用上 Tensor Core 专用矩阵乘单元（FP16 输入 + FP32 累加），进一步逼近 cuBLAS；而在小 N 下 L5 可能略慢于 L4，因为 FP16 转换与 WMMA 调度有固定开销——这恰恰是“专用硬件需要规模才能变现”的好例子。\n- cuBLAS 作为工业级库基线，其 FP32/FP16 表现与手写 L4/L5 **同量级**（有时更快）——说明把底层优化吃透后，手写实现能非常接近专业库，但工业库还有更多 kernel 融合/自动调优的积累。\n- 注意：cuBLAS **仅用于对比**，本 demo 的 L1–L5 全部手写、零性能库依赖，所有优化点均可见可讲。\n