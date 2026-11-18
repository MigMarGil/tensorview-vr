#include "onnx_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>

namespace tv {
namespace {

// ---------------------------------------------------------------- protobuf reader
struct Reader {
    const uint8_t* p = nullptr;
    const uint8_t* e = nullptr;

    bool varint(uint64_t& v) {
        v = 0;
        int shift = 0;
        while (p < e) {
            uint8_t b = *p++;
            v |= static_cast<uint64_t>(b & 0x7F) << shift;
            if (!(b & 0x80)) return true;
            shift += 7;
            if (shift > 63) return false;
        }
        return false;
    }
    bool tag(uint32_t& field, uint32_t& wire) {
        uint64_t t;
        if (!varint(t)) return false;
        field = static_cast<uint32_t>(t >> 3);
        wire = static_cast<uint32_t>(t & 7);
        return true;
    }
    bool sub(Reader& r) {
        uint64_t n;
        if (!varint(n) || n > static_cast<uint64_t>(e - p)) return false;
        r.p = p;
        r.e = p + n;
        p += n;
        return true;
    }
    bool str(std::string& s) {
        Reader r;
        if (!sub(r)) return false;
        s.assign(reinterpret_cast<const char*>(r.p), static_cast<size_t>(r.e - r.p));
        return true;
    }
    bool skip(uint32_t wire) {
        switch (wire) {
            case 0: { uint64_t v; return varint(v); }
            case 1: if (e - p < 8) return false; p += 8; return true;
            case 2: { Reader s; return sub(s); }
            case 5: if (e - p < 4) return false; p += 4; return true;
            default: return false;
        }
    }
};

bool readInt64s(Reader& r, uint32_t wire, std::vector<int64_t>& out) {
    if (wire == 0) {
        uint64_t v;
        if (!r.varint(v)) return false;
        out.push_back(static_cast<int64_t>(v));
        return true;
    }
    if (wire == 2) {  // packed
        Reader s;
        if (!r.sub(s)) return false;
        while (s.p < s.e) {
            uint64_t v;
            if (!s.varint(v)) return false;
            out.push_back(static_cast<int64_t>(v));
        }
        return true;
    }
    return r.skip(wire);
}

// TensorProto: dims = 1, name = 8
bool parseTensor(Reader r, std::string& name, std::vector<int64_t>& dims) {
    uint32_t f, w;
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        if (f == 1) { if (!readInt64s(r, w, dims)) return false; }
        else if (f == 8 && w == 2) { if (!r.str(name)) return false; }
        else if (!r.skip(w)) return false;
    }
    return true;
}

// TensorShapeProto.Dimension: dim_value = 1, dim_param = 2
bool parseDim(Reader r, int64_t& value) {
    uint32_t f, w;
    value = 1;  // symbolic / unknown dimension -> assume 1 (batch)
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        if (f == 1 && w == 0) {
            uint64_t v;
            if (!r.varint(v)) return false;
            value = v > 0 ? static_cast<int64_t>(v) : 1;
        } else if (!r.skip(w)) return false;
    }
    return true;
}

bool parseShape(Reader r, std::vector<int64_t>& dims) {
    uint32_t f, w;
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        if (f == 1 && w == 2) {
            Reader d; int64_t v;
            if (!r.sub(d) || !parseDim(d, v)) return false;
            dims.push_back(v);
        } else if (!r.skip(w)) return false;
    }
    return true;
}

// TypeProto.Tensor: elem_type = 1, shape = 2
bool parseTensorType(Reader r, std::vector<int64_t>& dims, bool& has) {
    uint32_t f, w;
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        if (f == 2 && w == 2) {
            Reader s;
            if (!r.sub(s) || !parseShape(s, dims)) return false;
            has = true;
        } else if (!r.skip(w)) return false;
    }
    return true;
}

// ValueInfoProto: name = 1, type = 2 (TypeProto.tensor_type = 1)
bool parseValueInfo(Reader r, std::string& name, std::vector<int64_t>& dims, bool& has) {
    uint32_t f, w;
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        if (f == 1 && w == 2) { if (!r.str(name)) return false; }
        else if (f == 2 && w == 2) {
            Reader t;
            if (!r.sub(t)) return false;
            uint32_t tf, tw;
            while (t.p < t.e) {
                if (!t.tag(tf, tw)) return false;
                if (tf == 1 && tw == 2) {
                    Reader tt;
                    if (!t.sub(tt) || !parseTensorType(tt, dims, has)) return false;
                } else if (!t.skip(tw)) return false;
            }
        } else if (!r.skip(w)) return false;
    }
    return true;
}

struct RawNode {
    std::string name, op;
    std::vector<std::string> in, out;
    int64_t group = 1;
};

// NodeProto: input = 1, output = 2, name = 3, op_type = 4, attribute = 5
bool parseNode(Reader r, RawNode& n) {
    uint32_t f, w;
    while (r.p < r.e) {
        if (!r.tag(f, w)) return false;
        std::string s;
        if (f == 1 && w == 2) { if (!r.str(s)) return false; n.in.push_back(s); }
        else if (f == 2 && w == 2) { if (!r.str(s)) return false; n.out.push_back(s); }
        else if (f == 3 && w == 2) { if (!r.str(n.name)) return false; }
        else if (f == 4 && w == 2) { if (!r.str(n.op)) return false; }
        else if (f == 5 && w == 2) {  // AttributeProto: name = 1, i = 3
            Reader a;
            if (!r.sub(a)) return false;
            std::string an; int64_t iv = 0; uint32_t af, aw;
            while (a.p < a.e) {
                if (!a.tag(af, aw)) return false;
                if (af == 1 && aw == 2) { if (!a.str(an)) return false; }
                else if (af == 3 && aw == 0) { uint64_t v; if (!a.varint(v)) return false; iv = static_cast<int64_t>(v); }
                else if (!a.skip(aw)) return false;
            }
            if (an == "group" && iv > 0) n.group = iv;
        } else if (!r.skip(w)) return false;
    }
    return true;
}

// ---------------------------------------------------------------- helpers
uint64_t numel(const std::vector<int64_t>& d) {
    if (d.empty()) return 0;
    uint64_t n = 1;
    for (int64_t v : d) n *= static_cast<uint64_t>(v > 0 ? v : 1);
    return n;
}

std::string shapeStr(const std::vector<int64_t>* d) {
    if (!d || d->empty()) return "?";
    std::string s = "[";
    for (size_t i = 0; i < d->size(); ++i) {
        if (i) s += ",";
        s += std::to_string((*d)[i]);
    }
    return s + "]";
}

bool isShapePreserving(const std::string& op) {
    static const std::unordered_set<std::string> k = {
        "Relu", "LeakyRelu", "Sigmoid", "Tanh", "Clip", "BatchNormalization", "Add", "Sub", "Mul", "Div",
        "Softmax", "Dropout", "Identity", "HardSwish", "HardSigmoid", "Gelu", "Erf", "Sqrt", "Neg"};
    return k.count(op) != 0;
}

void colorFor(const std::string& op, float& r, float& g, float& b) {
    auto has = [&](const char* s) { return op.find(s) != std::string::npos; };
    if (op == "Input") { r = .20f; g = .65f; b = 1.0f; }
    else if (has("Conv")) { r = 1.0f; g = .50f; b = .15f; }
    else if (has("Relu") || has("Sigmoid") || has("Tanh") || has("Gelu") || has("Clip") || has("Swish")) { r = .18f; g = .85f; b = .45f; }
    else if (op == "Add" || op == "Sum" || op == "Concat" || op == "Mul") { r = .95f; g = .80f; b = .20f; }
    else if (has("Pool")) { r = .35f; g = .55f; b = 1.0f; }
    else if (op == "Gemm" || op == "MatMul") { r = .15f; g = .90f; b = .85f; }
    else if (has("Softmax")) { r = .95f; g = .35f; b = .85f; }
    else if (has("Norm")) { r = .65f; g = .55f; b = .95f; }
    else { r = .60f; g = .62f; b = .68f; }
}

}  // namespace

// ---------------------------------------------------------------- parser
bool parseOnnx(const uint8_t* data, size_t size, const std::string& name, Model& out, std::string& err) {
    out = Model();
    out.name = name;
    Reader top{data, data + size};
    Reader graph;
    bool found = false;
    uint32_t f, w;
    while (top.p < top.e) {
        if (!top.tag(f, w)) { err = "corrupt ONNX header"; return false; }
        if (f == 7 && w == 2) {  // ModelProto.graph
            if (!top.sub(graph)) { err = "corrupt graph"; return false; }
            found = true;
        } else if (!top.skip(w)) { err = "corrupt ONNX file"; return false; }
    }
    if (!found) { err = "no graph found (not an ONNX file?)"; return false; }

    std::vector<RawNode> raw;
    std::unordered_map<std::string, std::vector<int64_t>> init, shape;
    std::vector<std::string> dataInputs;
    std::vector<std::pair<std::string, std::vector<int64_t>>> graphInputs;

    while (graph.p < graph.e) {
        if (!graph.tag(f, w)) { err = "corrupt graph field"; return false; }
        if (w != 2) { if (!graph.skip(w)) { err = "corrupt graph"; return false; } continue; }
        Reader s;
        if (!graph.sub(s)) { err = "corrupt graph"; return false; }
        if (f == 1) {  // node
            RawNode n;
            if (!parseNode(s, n)) { err = "corrupt node"; return false; }
            raw.push_back(std::move(n));
        } else if (f == 5) {  // initializer
            std::string nm; std::vector<int64_t> d;
            if (!parseTensor(s, nm, d)) { err = "corrupt initializer"; return false; }
            init[nm] = d;
        } else if (f == 11 || f == 12 || f == 13) {  // input / output / value_info
            std::string nm; std::vector<int64_t> d; bool has = false;
            if (!parseValueInfo(s, nm, d, has)) { err = "corrupt value info"; return false; }
            if (has && !d.empty()) shape[nm] = d;
            if (f == 11) graphInputs.push_back({nm, d});
        }
    }
    if (raw.empty()) { err = "graph has no nodes"; return false; }

    std::unordered_map<std::string, int> producer;
    auto addNode = [&](GraphNode&& g) -> int {
        g.id = static_cast<int>(out.nodes.size());
        out.nodes.push_back(std::move(g));
        return out.nodes.back().id;
    };

    for (auto& gi : graphInputs) {
        if (init.count(gi.first)) continue;  // old IR: weights also listed as inputs
        GraphNode g;
        g.name = gi.first; g.op = "Input";
        g.out_shape = shapeStr(shape.count(gi.first) ? &shape[gi.first] : nullptr);
        g.in_shape = g.out_shape;
        g.mem_kb = (numel(shape[gi.first]) * 4 + 1023) / 1024;
        producer[gi.first] = addNode(std::move(g));
    }

    for (auto& rn : raw) {
        // propagate shapes through shape-preserving ops when value_info is missing
        if (!rn.out.empty() && !shape.count(rn.out[0]) && isShapePreserving(rn.op)) {
            for (auto& in : rn.in)
                if (!init.count(in) && shape.count(in)) { shape[rn.out[0]] = shape[in]; break; }
        }
        GraphNode g;
        g.op = rn.op;
        g.name = rn.name.empty() ? (rn.op + "_" + std::to_string(out.nodes.size())) : rn.name;

        const std::vector<int64_t>* inShape = nullptr;
        for (auto& in : rn.in)
            if (!init.count(in) && shape.count(in)) { inShape = &shape[in]; break; }
        const std::vector<int64_t>* outShape = (!rn.out.empty() && shape.count(rn.out[0])) ? &shape[rn.out[0]] : nullptr;
        g.in_shape = shapeStr(inShape);
        g.out_shape = shapeStr(outShape);

        for (auto& in : rn.in)
            if (init.count(in)) g.params += numel(init[in]);

        const uint64_t outN = outShape ? numel(*outShape) : 0;
        g.mem_kb = (outN * 4 + 1023) / 1024;

        if (rn.op == "Conv" && rn.in.size() > 1 && init.count(rn.in[1]) && init[rn.in[1]].size() >= 3) {
            const auto& wd = init[rn.in[1]];  // [Cout, Cin/group, kh, kw ...]
            uint64_t kmac = 1;
            for (size_t i = 1; i < wd.size(); ++i) kmac *= static_cast<uint64_t>(wd[i] > 0 ? wd[i] : 1);
            g.flops = 2 * outN * kmac + (rn.in.size() > 2 ? outN : 0);
            if (rn.group > 1) g.op = (static_cast<int64_t>(wd[1]) == 1) ? "DepthwiseConv" : "GroupConv";
        } else if ((rn.op == "Gemm" || rn.op == "MatMul") && rn.in.size() > 1) {
            uint64_t k = 0;
            if (init.count(rn.in[1]) && outShape && !outShape->empty()) {
                uint64_t wn = numel(init[rn.in[1]]), lastDim = static_cast<uint64_t>(outShape->back());
                k = lastDim ? wn / lastDim : 0;
            } else if (inShape && !inShape->empty()) k = static_cast<uint64_t>(inShape->back());
            g.flops = 2 * outN * k;
        } else if (rn.op == "Softmax") {
            g.flops = 5 * outN;
        } else {
            g.flops = outN;  // elementwise estimate
        }

        const int id = addNode(std::move(g));
        for (auto& in : rn.in) {
            auto it = producer.find(in);
            if (it == producer.end() || it->second == id) continue;
            auto& outs = out.nodes[it->second].outputs;
            if (std::find(outs.begin(), outs.end(), id) == outs.end()) outs.push_back(id);
        }
        for (auto& o : rn.out) producer[o] = id;
    }

    for (auto& n : out.nodes) {
        out.total_params += n.params;
        out.total_flops += n.flops;
        out.total_mem_kb += n.mem_kb;
    }
    layoutModel(out);
    return true;
}

// ---------------------------------------------------------------- layout
// Layered DAG layout folded into "snake" rows so deep networks stay readable at arm's length.
void layoutModel(Model& m) {
    const int n = static_cast<int>(m.nodes.size());
    if (n == 0) return;
    std::vector<int> layer(n, 0);
    for (int i = 0; i < n; ++i)
        for (int c : m.nodes[i].outputs)
            if (c > i) layer[c] = std::max(layer[c], layer[i] + 1);  // nodes are topologically ordered
    int L = 1;
    for (int v : layer) L = std::max(L, v + 1);

    const int rows = (L + 13) / 14;                 // at most ~14 layers per row
    const int cols = (L + rows - 1) / rows;
    const float dx = std::min(0.22f, 2.0f / static_cast<float>(std::max(1, cols - 1)));
    const float dy = 0.13f, rowH = 0.5f;

    std::vector<int> count(L, 0), seen(L, 0);
    for (int v : layer) count[v]++;
    int maxCount = 1;
    for (int c : count) maxCount = std::max(maxCount, c);

    for (int i = 0; i < n; ++i) {
        GraphNode& g = m.nodes[i];
        const int l = layer[i];
        const int idx = seen[l]++;
        const int row = l / cols;
        int col = l % cols;
        if (row % 2 == 1) col = cols - 1 - col;      // snake: odd rows run right-to-left
        const float branch = static_cast<float>(idx) - 0.5f * static_cast<float>(count[l] - 1);
        g.layer = l;
        g.x = (static_cast<float>(col) - 0.5f * static_cast<float>(cols - 1)) * dx;
        g.y = (0.5f * static_cast<float>(rows - 1) - static_cast<float>(row)) * rowH + branch * dy;
        g.z = branch * 0.06f;
        const float s = std::min(1.0f, std::max(0.0f, (std::log10(static_cast<float>(std::max<uint64_t>(g.flops, 10))) - 1.0f) / 9.0f));
        g.hy = 0.015f + 0.030f * s;
        g.hx = std::min(dx * 0.34f, 0.018f + 0.030f * s);
        g.hz = g.hx;
        colorFor(g.op, g.r, g.g, g.b);
    }
    m.width = static_cast<float>(cols - 1) * dx + 0.12f;
    m.height = static_cast<float>(rows - 1) * rowH + static_cast<float>(maxCount - 1) * dy + 0.12f;
}

}  // namespace tv
