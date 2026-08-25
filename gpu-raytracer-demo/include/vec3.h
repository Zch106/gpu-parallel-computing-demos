#ifndef VEC3_H
#define VEC3_H

#include <cmath>

// 极简 vec3（host/device 通用），运算符与原版 tinyraytracer 一致：
//   vec3 * float  -> 标量缩放
//   vec3 * vec3   -> 点积（dot）
struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() = default;
    __host__ __device__ vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    __host__ __device__ vec3 operator*(float v) const { return vec3(x * v, y * v, z * v); }
    __host__ __device__ float operator*(const vec3& v) const { return x * v.x + y * v.y + z * v.z; }  // dot
    __host__ __device__ vec3 operator+(const vec3& v) const { return vec3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ vec3 operator-(const vec3& v) const { return vec3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }

    __host__ __device__ float norm() const { return sqrtf(x * x + y * y + z * z); }
    __host__ __device__ vec3 normalized() const { return (*this) * (1.f / norm()); }
};

__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline vec3 reflect(const vec3& I, const vec3& N) {
    return I - N * 2.f * (I * N);
}

// 折射（Snell 定律）；与 tinyraytracer 完全一致的处理（k<0 视为全反射，返回红色占位）
// 注意：原版用递归处理"光线从内部射出"的情况，但 CUDA 的递归会吃掉有限的硬件调用栈
// （尤其配合 BVH 遍历栈时直接栈溢出）。这里改成等价的非递归写法，语义完全一致。
__host__ __device__ inline vec3 refract(const vec3& I, const vec3& N, float eta_t, float eta_i = 1.f) {
    vec3 n = N;
    float cosi = -fminf(1.f, fmaxf(-1.f, I * n));
    if (cosi < 0) {  // 光线从物体内部射出：翻转法线 + 交换介质
        n = -n;
        float tmp = eta_t; eta_t = eta_i; eta_i = tmp;
        cosi = -fminf(1.f, fmaxf(-1.f, I * n));
    }
    float eta = eta_i / eta_t;
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? vec3(1, 0, 0) : I * eta + n * (eta * cosi - sqrtf(k));
}

#endif  // VEC3_H
