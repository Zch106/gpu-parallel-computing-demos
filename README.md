# GPU 并行计算课程演示 Demo

面向《GPU 并行计算》课程的课堂演示项目，用两个**计算量大、贴近真实场景**的程序讲清楚
「GPU 并行计算的加速能力」与「不同负载需要不同优化手段」。

## 两个 Demo

| Demo | 载体 | 优化阶梯 | 终极优化 | 教学核心 |
|------|------|---------|---------|---------|
| [gpu-attention-demo](gpu-attention-demo/) | Transformer 自注意力（GEMM 为主） | L1→L5 + cuBLAS 对照 | **Tensor Core** (WMMA) | 矩阵类负载 → 专用硬件单元 |
| [gpu-raytracer-demo](gpu-raytracer-demo/) | 光线追踪（逐像素独立） | L1→L6 | **BVH 加速结构** | 非矩阵负载 → 算法级优化 |

### 为什么选这两个？

它们形成完美的教学对照：

- **注意力**（GEMM）能直接映射到 Tensor Core，终极优化用它再上一个量级；
- **光线追踪**没有矩阵乘语义，**不能**映射到 Tensor Core，终极优化换成更适合它的 BVH 加速结构；
- 这是一个好教学点：**不是所有负载都能用同一种硬件加速，优化手段要贴合负载特征。**

## 环境要求

- NVIDIA GPU (Compute Capability ≥ 7.0，L5 需 ≥ 7.5 for WMMA)
- CUDA Toolkit 12.0+ (本项目在 CUDA 13.0 上验证)
- C++ 编译器 (MSVC / GCC / Clang)
- Windows: VS Build Tools (含 C++ 桌面开发负载) + `build.bat`
- Linux/WSL: `make`

## 快速开始

```bash
# Demo 1: 自注意力
cd gpu-attention-demo
make            # 或 Windows: build.bat
./attention_demo

# Demo 2: 光线追踪
cd gpu-raytracer-demo
make            # 或 Windows: build.bat
./raytracer_demo                              # 默认 1024x1024, 260球, 深度5
./raytracer_demo --width 512 --spheres 64 --depth 4   # 轻量预览
```

## 共同的教学范式

1. **CPU 基线** → 让学生感受「不并行有多慢」
2. **GPU 朴素并行** → 展示「并行本身」的暴力加速（千倍级）
3. **数据流/访存优化** → 在并行之上再榨带宽、削延迟
4. **算法级/硬件级终极优化** → 负载特征决定方向
5. **加速比报告** → 自动生成 Markdown，可直接发给学生

## 验证环境

- GPU: NVIDIA RTX 3080 Laptop (Ampere, CC 8.6, 16GB)
- CUDA 13.0, MSVC 14.44, Windows SDK 10.0.26100.0
- 编译: `-arch=sm_86 -Xcompiler "/utf-8" -Xcompiler "/openmp"`

## License

MIT
