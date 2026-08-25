#ifndef SCENE_H
#define SCENE_H

#include "vec3.h"

// ---------- 场景数据结构（POD，可直接 cudaMemcpy 到设备）----------
struct Material {
    vec3 diffuse_color;
    float albedo[4];
    float specular_exponent;
    float refractive_index;
};

struct Sphere {
    vec3 center;
    float radius;
    Material material;
};

struct Light {
    vec3 position;
    float intensity;
};

struct Camera {
    int width = 1024;
    int height = 1024;
    float fov = 1.05f;  // tan(fov/2) 用于投影
};

// 棋盘格地面材质（y = -4 平面），与原版一致
__host__ __device__ inline Material checker_material(const vec3& p) {
    Material m;
    m.diffuse_color = ((int(0.5f * p.x + 1000) + int(0.5f * p.z)) & 1)
                          ? vec3(0.3f, 0.3f, 0.3f)
                          : vec3(0.3f, 0.2f, 0.1f);
    m.albedo[0] = 2;  // 纯漫反射，亮度系数 2
    m.albedo[1] = 0; m.albedo[2] = 0; m.albedo[3] = 0;
    m.specular_exponent = 0;
    m.refractive_index = 1;
    return m;
}

// 场景容器（host 端构建，构建后拷到设备）
struct Scene {
    Sphere* spheres = nullptr;
    int nsph = 0;
    Light* lights = nullptr;
    int nlight = 0;
    Camera cam;
};

#endif  // SCENE_H
