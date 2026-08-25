# GPU 并行计算课程演示 Demo：自注意力加速实验

面向《GPU 并行计算》课程的课堂演示。用一个**贴近大模型推理真实热点**的算子——Transformer
自注意力（Self-Attention）前向——讲清楚「并行计算的加速能力」：

1. **并行本身**就能大幅加速（对比 CPU 串行，可达千倍级）；
2. 在「已经并行」之后，还能通过 **CUDA 底层的数据流 / 访存优化** 让 kernel 时间逐层下降
   （共享内存分块、向量化、Tensor Core）；
3. 手写优化到底离 **工业级性能库（cuBLAS）** 有多近。

## 演示载体

单样本、单头的自注意力（序列长度 `N`，头维度 `d`）：

```
Q, K, V ∈ ℝ^{N×d}
S = Q · Kᵀ            → ℝ^{N×N}   (GEMM，内层维度 d)
P = softmax(S / √d)   → ℝ^{N×N}   (逐行 softmax)
O = P · V             → ℝ^{N×d}   (GEMM，内层维度 N)
```

两个 GEMM（QKᵀ 与 PV）是计算热点，也是各层优化的唯一变量；softmax 用**一个共享的
GPU kernel**，保持「变量隔离」、教学聚焦。

## 五层优化阶梯 + 外部对照

| 层级 | 方法 | 关键技术点 | 教学含义 |
|---|---|---|---|
| L1 | CPU 串行基线 | 三重循环朴素 GEMM + 串行 softmax | 性能基准 & 正确性参考 |
| L2 | GPU 朴素并行 | 每线程 1 元素，全程全局内存 | “并行”带来的第一波加速 |
| L3 | 共享内存分块 | tile 入 `__shared__`，合并访存 + 数据复用 | **并行之上的数据流优化** |
| L4 | 分块+深度优化 | `float4` 向量化 + padding 消 bank conflict + 循环展开 | 并行之上再榨带宽/延迟 |
| L5 | Tensor Core | WMMA FP16→FP32，mma.sync 16×16×16 | 现代 AI 硬件：专用矩阵乘单元 |
| ★  | cuBLAS（外部库）| `cublasSgemm` / `cublasGemmEx` | **仅作对比基线**，不参与教学阶梯 |

> **库的使用边界**：cuBLAS 只用于对比。L1–L5 全部手写、零性能库依赖，保证所有优化点可见可讲。

## 工程结构（轻量：仅 nvcc + CUDA Runtime + 链接 cuBLAS 仅作对照）

```
gpu-attention-demo/
├── Makefile
├── README.md
├── include/
│   ├── timer.cuh          # 基于 cudaEvent 的计时器
│   ├── utils.cuh          # 随机初始化、误差校验、设备分配辅助
│   └── attention.cuh      # 各层 kernel / 包装函数声明与结果结构体
├── src/
│   ├── main.cu            # 编排：跑 L1–L5 + cuBLAS、计时、校验、打印
│   ├── cpu_ref.cu         # L1 CPU 串行注意力（正确性与性能基准）
│   ├── softmax.cu         # 共享的 GPU 逐行 softmax kernel
│   ├── gemm_naive.cu      # L2
│   ├── gemm_tiled.cu      # L3
│   ├── gemm_opt.cu        # L4（float4 + padding + unroll）
│   ├── gemm_tc.cu         # L5（WMMA FP16→FP32）
│   ├── cublas_ref.cu      # ★ cuBLAS FP32 / FP16 对照
│   └── report.cu          # 生成 results.md
└── results.md             # 运行自动生成
```

## 构建与运行

```bash
cd gpu-attention-demo
make                # 编译（-arch=sm_86，链接 -lcublas 仅用于对照）

./attention_demo                 # 默认 d=64, N∈{1024,2048,4096}
./attention_demo --N 2048 --d 64 # 指定规模
```

运行后：
- 终端实时打印每个规模的六行（L1–L5 + cuBLAS）耗时 / 加速比 / 最大误差 / 正确性；
- 同时生成 `results.md`（含汇总表 + 各层要点与解读），可直接发给学生。

## 计时与校验说明

- **kernel(ms)**：用 CUDA Events 只包住 GPU 计算 kernel（排除 H2D/D2H 拷贝）；
- **e2e(ms)**：端到端（含分配 / 拷贝 / 计算），CPU 计时；
- **加速比**：相对 CPU（L1）计算耗时；
- **正确性**：每层输出与 CPU 参考做最大绝对误差比对，FP32 容差 1e-3、FP16 容差 1e-2，超限标 `NO`。

## 教学提示

- 小尺寸下 kernel 启动 / 传输开销占比高，加速比不直观；默认尺寸已选在「GPU 优势区」。
- 3080 Laptop GPU 带宽 / 功耗低于桌面版，但仍是 Ampere sm_86，支持 WMMA FP16，L5 与
  cuBLAS Tensor Op 均可运行。
- 建议讲解顺序：先跑 L1 看 CPU 多慢 → L2 看「并行」的暴力加速 → L3 讲「访存才是瓶颈」→
  L4 讲「继续榨干带宽」→ L5 讲「专用硬件」→ 最后用 cuBLAS 收尾，给学生建立量级感。
