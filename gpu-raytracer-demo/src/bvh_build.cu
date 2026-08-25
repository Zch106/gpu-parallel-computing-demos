// L6 配套：在 CPU 端构建 BVH（层次包围盒），并把节点/叶子索引拷到设备。
// 这是"算法级"优化：把每条光线对全部球的 O(N) 遍历，变成 O(logN + 叶子球数)。
#include "rt_common.cuh"
#include "utils.cuh"
#include <vector>
#include <algorithm>
#include <functional>

void build_bvh(const Sphere* h_sph, int n, BVHNode*& d_nodes, int*& d_leaf, int& nnodes) {
    std::vector<BVHNode> nodes;
    std::vector<int> leaf_idx;
    const int LEAF = 4;  // 叶子最多 4 个球

    nodes.push_back(BVHNode());  // 根节点 index 0

    std::function<void(const std::vector<int>&, int)> build = [&](const std::vector<int>& cur, int ni) {
        // 计算整体 AABB 与质心范围
        vec3 bmin(1e30f, 1e30f, 1e30f), bmax(-1e30f, -1e30f, -1e30f);
        vec3 cmin(1e30f, 1e30f, 1e30f), cmax(-1e30f, -1e30f, -1e30f);
        for (int id : cur) {
            const Sphere& s = h_sph[id];
            vec3 lo = s.center - vec3(s.radius, s.radius, s.radius);
            vec3 hi = s.center + vec3(s.radius, s.radius, s.radius);
            bmin = vec3(fminf(bmin.x, lo.x), fminf(bmin.y, lo.y), fminf(bmin.z, lo.z));
            bmax = vec3(fmaxf(bmax.x, hi.x), fmaxf(bmax.y, hi.y), fmaxf(bmax.z, hi.z));
            cmin = vec3(fminf(cmin.x, s.center.x), fminf(cmin.y, s.center.y), fminf(cmin.z, s.center.z));
            cmax = vec3(fmaxf(cmax.x, s.center.x), fmaxf(cmax.y, s.center.y), fmaxf(cmax.z, s.center.z));
        }
        nodes[ni].bmin = bmin;
        nodes[ni].bmax = bmax;

        if ((int)cur.size() <= LEAF) {
            nodes[ni].left = -1;
            nodes[ni].right = -1;
            nodes[ni].start = (int)leaf_idx.size();
            nodes[ni].count = (int)cur.size();
            for (int id : cur) leaf_idx.push_back(id);
            return;
        }
        // 沿最长轴对质心排序后二分
        vec3 ext = cmax - cmin;
        int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
        std::vector<int> sorted = cur;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return h_sph[a].center[axis] < h_sph[b].center[axis];
        });
        int mid = (int)sorted.size() / 2;
        std::vector<int> left(sorted.begin(), sorted.begin() + mid);
        std::vector<int> right(sorted.begin() + mid, sorted.end());
        int lnode = (int)nodes.size(); nodes.push_back(BVHNode());
        int rnode = (int)nodes.size(); nodes.push_back(BVHNode());
        nodes[ni].left = lnode;
        nodes[ni].right = rnode;
        build(left, lnode);
        build(right, rnode);
    };

    std::vector<int> ids(n);
    for (int i = 0; i < n; i++) ids[i] = i;
    build(ids, 0);

    nnodes = (int)nodes.size();
    CHECK(cudaMalloc(&d_nodes, nnodes * sizeof(BVHNode)));
    CHECK(cudaMalloc(&d_leaf, leaf_idx.size() * sizeof(int)));
    CHECK(cudaMemcpy(d_nodes, nodes.data(), nnodes * sizeof(BVHNode), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_leaf, leaf_idx.data(), leaf_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
}
