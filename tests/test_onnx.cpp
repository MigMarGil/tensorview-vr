// Host-side test for the ONNX parser + layout:  see README ("Test the parser on your PC").
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>
#include "onnx_model.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.onnx\n", argv[0]); return 2; }
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    tv::Model m; std::string err;
    if (!tv::parseOnnx(buf.data(), buf.size(), argv[1], m, err)) { std::fprintf(stderr, "PARSE ERROR: %s\n", err.c_str()); return 1; }
    std::printf("nodes=%zu params=%llu flops=%llu mem_kb=%llu size=%.2fx%.2f m\n", m.nodes.size(),
                (unsigned long long)m.total_params, (unsigned long long)m.total_flops,
                (unsigned long long)m.total_mem_kb, m.width, m.height);
    for (const auto& n : m.nodes)
        std::printf("%2d L%-2d %-22s %-14s %-18s -> %-18s p=%-8llu fl=%-11llu  (%.2f,%.2f,%.2f) out=%zu\n", n.id, n.layer,
                    n.name.c_str(), n.op.c_str(), n.in_shape.c_str(), n.out_shape.c_str(),
                    (unsigned long long)n.params, (unsigned long long)n.flops, n.x, n.y, n.z, n.outputs.size());
    return 0;
}
