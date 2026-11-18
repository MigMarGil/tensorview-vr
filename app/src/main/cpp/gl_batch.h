#pragma once
#include <string>
#include <vector>
#include "tv_math.h"

namespace tv {

struct Color { float r, g, b, a; };
struct Vertex { float x, y, z, u, v, r, g, b, a; };

// CPU-side triangle batch: solid boxes, beams, quads and bitmap text.
class Batch {
public:
    std::vector<Vertex> v;
    void clear() { v.clear(); }
    void quad(V3 a, V3 b, V3 c, V3 d, Color col);                 // solid quad (a,b,c,d in order)
    void box(V3 center, V3 half, Color col);                      // shaded solid box
    void beam(V3 a, V3 b, float width, Color col);                // square-section segment
    void boxFrame(V3 center, V3 half, float width, Color col);    // 12 edges of a box
    void panel(V3 center, float w, float h, Color col);           // quad facing +Z
    void text(const std::string& s, V3 topLeft, float pixel, Color col);  // faces +Z, 6x8 px per char
};

// OpenGL ES 3 renderer for Batch (one shader + one font/white texture atlas).
class GlRenderer {
public:
    bool init(bool srgbTarget);
    void draw(const Batch& b, const Mat4& viewProj, bool depthWrite);
    void shutdown();

private:
    unsigned int prog_ = 0, vao_ = 0, vbo_ = 0, tex_ = 0;
    int locVP_ = -1, locTex_ = -1, locSrgb_ = -1;
    bool srgb_ = false;
};

}  // namespace tv
