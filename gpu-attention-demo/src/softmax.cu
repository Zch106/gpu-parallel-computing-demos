// 共享的 GPU 逐行 softmax kernel（L2–L5 复用，保持「变量隔离」）
// 每个 block 处理 S 的一行（N 个元素），先求行内最大值，再 exp / 求和 / 归一化。
#include "attention.cuh"
#include "utils.cuh"

__global__ void softmax_kernel(float* S, int N, float scale) {
    int row = blockIdx.x;
    float* s = S + row * N;
    __shared__ float buf[256];

    // 1) 行内最大值
    float mx = -1e30f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = s[i] * scale;
        if (v > mx) mx = v;
    }
    buf[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] = max(buf[threadIdx.x], buf[threadIdx.x + s]);
        __syncthreads();
    }
    mx = buf[0];

    // 2) 行内 exp 求和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = s[i] * scale;
        sum += expf(v - mx);
    }
    buf[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    sum = buf[0];

    // 3) 归一化写回
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float v = s[i] * scale;
        s[i] = expf(v - mx) / sum;
    }
}
