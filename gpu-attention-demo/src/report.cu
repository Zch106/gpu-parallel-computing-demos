// 将生成的 Markdown 报告写入文件
#include "attention.cuh"
#include <fstream>

void write_report(const char* path, const std::string& md) {
    std::ofstream f(path);
    if (!f) return;
    f << md;
    f.close();
}
