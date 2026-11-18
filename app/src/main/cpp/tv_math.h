#pragma once
#include <cmath>

namespace tv {

struct V3 { float x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a) { float l = len(a); return l > 1e-6f ? a * (1.0f / l) : V3{0, 0, -1}; }

struct Quat { float x = 0, y = 0, z = 0, w = 1; };

// Rotates v by unit quaternion q.
inline V3 rotate(const Quat& q, V3 v) {
    V3 u{q.x, q.y, q.z};
    V3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}

// Column-major 4x4 matrix: m[col * 4 + row].
struct Mat4 { float m[16]; };

inline Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int rr = 0; rr < 4; ++rr) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + rr] * b.m[c * 4 + k];
            r.m[c * 4 + rr] = s;
        }
    return r;
}

// Asymmetric OpenGL projection from OpenXR field-of-view angles (radians).
inline Mat4 projectionFromFov(float angleLeft, float angleRight, float angleUp, float angleDown,
                              float nearZ, float farZ) {
    const float tl = std::tan(angleLeft), tr = std::tan(angleRight);
    const float tu = std::tan(angleUp), td = std::tan(angleDown);
    const float w = tr - tl, h = tu - td;
    Mat4 r;
    for (float& v : r.m) v = 0;
    r.m[0] = 2.0f / w;
    r.m[5] = 2.0f / h;
    r.m[8] = (tr + tl) / w;
    r.m[9] = (tu + td) / h;
    r.m[10] = -(farZ + nearZ) / (farZ - nearZ);
    r.m[11] = -1.0f;
    r.m[14] = -2.0f * farZ * nearZ / (farZ - nearZ);
    return r;
}

// View matrix = inverse of the rigid pose (position p, orientation q).
inline Mat4 viewFromPose(const V3& p, const Quat& q) {
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    // R[row][col] (world-from-eye rotation)
    const float R[3][3] = {{1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)},
                           {2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)},
                           {2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)}};
    const float pp[3] = {p.x, p.y, p.z};
    Mat4 r;
    for (float& v : r.m) v = 0;
    for (int row = 0; row < 3; ++row) {
        float t = 0;
        for (int col = 0; col < 3; ++col) {
            r.m[col * 4 + row] = R[col][row];  // transpose
            t += R[col][row] * pp[col];
        }
        r.m[12 + row] = -t;
    }
    r.m[15] = 1.0f;
    return r;
}

}  // namespace tv
