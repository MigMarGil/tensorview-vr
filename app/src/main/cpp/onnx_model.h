#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tv {

struct GraphNode {
    int id = 0;
    std::string name, op, in_shape, out_shape;
    uint64_t params = 0, flops = 0, mem_kb = 0;
    // Layout (meters, relative to graph center)
    float x = 0, y = 0, z = 0;
    float hx = 0.03f, hy = 0.03f, hz = 0.03f;  // half extents
    float r = 0.5f, g = 0.5f, b = 0.5f;
    int layer = 0;
    std::vector<int> outputs;  // ids of consumer nodes
};

struct Model {
    std::string name;
    std::vector<GraphNode> nodes;
    uint64_t total_params = 0, total_flops = 0, total_mem_kb = 0;
    float width = 1.0f, height = 0.5f;
};

// Parses an ONNX ModelProto (protobuf wire format) from memory. No external dependencies.
bool parseOnnx(const uint8_t* data, size_t size, const std::string& name, Model& out, std::string& err);

// Assigns layered 3D positions, sizes and colors to every node.
void layoutModel(Model& m);

}  // namespace tv
