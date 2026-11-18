#include "gl_batch.h"

#include <GLES3/gl3.h>
#include <android/log.h>
#include <cstddef>

#include "font5x7.h"

#define GLOGE(...) __android_log_print(ANDROID_LOG_ERROR, "TensorViewVR", __VA_ARGS__)

namespace tv {
namespace {
const int kAtlasW = 128, kAtlasH = 48;  // 16 x 6 cells of 8x8 px; cell 95 is solid white
const float kWhiteU = (15 * 8 + 4) / static_cast<float>(kAtlasW);
const float kWhiteV = (5 * 8 + 4) / static_cast<float>(kAtlasH);

inline Color scaled(Color c, float s) { return {c.r * s, c.g * s, c.b * s, c.a}; }

const char* kVert = R"(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aCol;
uniform mat4 uVP;
out vec2 vUV;
out vec4 vCol;
void main() {
    vUV = aUV;
    vCol = aCol;
    gl_Position = uVP * vec4(aPos, 1.0);
})";

const char* kFrag = R"(#version 300 es
precision mediump float;
in vec2 vUV;
in vec4 vCol;
uniform sampler2D uTex;
uniform float uSrgb;
out vec4 oColor;
void main() {
    vec4 c = vCol * texture(uTex, vUV);
    if (uSrgb > 0.5) c.rgb = pow(c.rgb, vec3(2.2));
    oColor = c;
})";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        GLOGE("shader compile error: %s", log);
        return 0;
    }
    return s;
}
}  // namespace

// ------------------------------------------------------------------ Batch
void Batch::quad(V3 a, V3 b, V3 c, V3 d, Color col) {
    auto push = [&](V3 p) { v.push_back({p.x, p.y, p.z, kWhiteU, kWhiteV, col.r, col.g, col.b, col.a}); };
    push(a); push(b); push(c);
    push(a); push(c); push(d);
}

void Batch::box(V3 c, V3 h, Color col) {
    const V3 p[8] = {{c.x - h.x, c.y - h.y, c.z - h.z}, {c.x + h.x, c.y - h.y, c.z - h.z},
                     {c.x + h.x, c.y + h.y, c.z - h.z}, {c.x - h.x, c.y + h.y, c.z - h.z},
                     {c.x - h.x, c.y - h.y, c.z + h.z}, {c.x + h.x, c.y - h.y, c.z + h.z},
                     {c.x + h.x, c.y + h.y, c.z + h.z}, {c.x - h.x, c.y + h.y, c.z + h.z}};
    quad(p[4], p[5], p[6], p[7], scaled(col, 0.90f));  // front (+z)
    quad(p[1], p[0], p[3], p[2], scaled(col, 0.55f));  // back
    quad(p[7], p[6], p[2], p[3], scaled(col, 1.00f));  // top
    quad(p[0], p[1], p[5], p[4], scaled(col, 0.45f));  // bottom
    quad(p[5], p[1], p[2], p[6], scaled(col, 0.72f));  // right
    quad(p[0], p[4], p[7], p[3], scaled(col, 0.72f));  // left
}

void Batch::beam(V3 a, V3 b, float w, Color col) {
    V3 d = b - a;
    if (len(d) < 1e-5f) return;
    V3 ref = std::fabs(norm(d).y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0};
    V3 s = norm(cross(d, ref)) * (w * 0.5f);
    V3 t = norm(cross(d, s)) * (w * 0.5f);
    const V3 off[4] = {s + t, t - s, (s + t) * -1.0f, s - t};
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        quad(a + off[i], a + off[j], b + off[j], b + off[i], scaled(col, i % 2 ? 0.75f : 1.0f));
    }
}

void Batch::boxFrame(V3 c, V3 h, float w, Color col) {
    const float xs[2] = {c.x - h.x, c.x + h.x}, ys[2] = {c.y - h.y, c.y + h.y}, zs[2] = {c.z - h.z, c.z + h.z};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            beam({xs[0], ys[i], zs[j]}, {xs[1], ys[i], zs[j]}, w, col);
            beam({xs[i], ys[0], zs[j]}, {xs[i], ys[1], zs[j]}, w, col);
            beam({xs[i], ys[j], zs[0]}, {xs[i], ys[j], zs[1]}, w, col);
        }
}

void Batch::panel(V3 c, float w, float h, Color col) {
    quad({c.x - w / 2, c.y - h / 2, c.z}, {c.x + w / 2, c.y - h / 2, c.z},
         {c.x + w / 2, c.y + h / 2, c.z}, {c.x - w / 2, c.y + h / 2, c.z}, col);
}

void Batch::text(const std::string& s, V3 o, float px, Color col) {
    float x = o.x;
    for (char ch : s) {
        int code = static_cast<unsigned char>(ch);
        if (code < 32 || code > 126) code = '?';
        const int idx = code - 32, cx = idx % 16, cy = idx / 16;
        const float u0 = (cx * 8 + 0) / static_cast<float>(kAtlasW), u1 = (cx * 8 + 6) / static_cast<float>(kAtlasW);
        const float v0 = (cy * 8 + 0) / static_cast<float>(kAtlasH), v1 = (cy * 8 + 8) / static_cast<float>(kAtlasH);
        const float x0 = x, x1 = x + 6 * px, y0 = o.y, y1 = o.y - 8 * px;
        auto push = [&](float X, float Y, float U, float V) { v.push_back({X, Y, o.z, U, V, col.r, col.g, col.b, col.a}); };
        push(x0, y0, u0, v0); push(x1, y0, u1, v0); push(x1, y1, u1, v1);
        push(x0, y0, u0, v0); push(x1, y1, u1, v1); push(x0, y1, u0, v1);
        x += 6 * px;
    }
}

// ------------------------------------------------------------------ GlRenderer
bool GlRenderer::init(bool srgbTarget) {
    srgb_ = srgbTarget;
    GLuint vs = compile(GL_VERTEX_SHADER, kVert), fs = compile(GL_FRAGMENT_SHADER, kFrag);
    if (!vs || !fs) return false;
    prog_ = glCreateProgram();
    glAttachShader(prog_, vs);
    glAttachShader(prog_, fs);
    glLinkProgram(prog_);
    GLint ok = 0;
    glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
    if (!ok) { GLOGE("program link failed"); return false; }
    glDeleteShader(vs);
    glDeleteShader(fs);
    locVP_ = glGetUniformLocation(prog_, "uVP");
    locTex_ = glGetUniformLocation(prog_, "uTex");
    locSrgb_ = glGetUniformLocation(prog_, "uSrgb");

    // Font + white texel atlas
    std::vector<uint8_t> px(static_cast<size_t>(kAtlasW) * kAtlasH * 4, 0);
    auto set = [&](int x, int y) {
        uint8_t* p = &px[(static_cast<size_t>(y) * kAtlasW + x) * 4];
        p[0] = p[1] = p[2] = p[3] = 255;
    };
    for (int i = 0; i < 95; ++i)
        for (int c = 0; c < 5; ++c)
            for (int r = 0; r < 7; ++r)
                if ((kFont5x7[i][c] >> r) & 1) set((i % 16) * 8 + 1 + c, (i / 16) * 8 + 1 + r);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) set(15 * 8 + x, 5 * 8 + y);
    glGenTextures(1, &tex_);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, kAtlasW, kAtlasH, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, u)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, r)));
    glBindVertexArray(0);
    return true;
}

void GlRenderer::draw(const Batch& b, const Mat4& vp, bool depthWrite) {
    if (b.v.empty() || !prog_) return;
    glUseProgram(prog_);
    glUniformMatrix4fv(locVP_, 1, GL_FALSE, vp.m);
    glUniform1i(locTex_, 0);
    glUniform1f(locSrgb_, srgb_ ? 1.0f : 0.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(b.v.size() * sizeof(Vertex)), b.v.data(), GL_STREAM_DRAW);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(depthWrite ? GL_TRUE : GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(b.v.size()));
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
}

void GlRenderer::shutdown() {
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (tex_) glDeleteTextures(1, &tex_);
    if (prog_) glDeleteProgram(prog_);
    vbo_ = vao_ = tex_ = prog_ = 0;
}

}  // namespace tv
