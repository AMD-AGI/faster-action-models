#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include <cmath>

namespace hipgl {
namespace math {

// ── Scalar helpers ──────────────────────────────────────────────────────────

HIPGL_HOST_DEVICE inline float clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

HIPGL_HOST_DEVICE inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

HIPGL_HOST_DEVICE inline float min2(float a, float b) { return a < b ? a : b; }
HIPGL_HOST_DEVICE inline float max2(float a, float b) { return a > b ? a : b; }
HIPGL_HOST_DEVICE inline float min3(float a, float b, float c) { return min2(a, min2(b, c)); }
HIPGL_HOST_DEVICE inline float max3(float a, float b, float c) { return max2(a, max2(b, c)); }

HIPGL_HOST_DEVICE inline int min2i(int a, int b) { return a < b ? a : b; }
HIPGL_HOST_DEVICE inline int max2i(int a, int b) { return a > b ? a : b; }
HIPGL_HOST_DEVICE inline int clampi(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// ── Vec2 ────────────────────────────────────────────────────────────────────

struct Vec2 {
    float x, y;

    HIPGL_HOST_DEVICE Vec2() : x(0), y(0) {}
    HIPGL_HOST_DEVICE Vec2(float x_, float y_) : x(x_), y(y_) {}

    HIPGL_HOST_DEVICE Vec2 operator+(Vec2 b) const { return {x + b.x, y + b.y}; }
    HIPGL_HOST_DEVICE Vec2 operator-(Vec2 b) const { return {x - b.x, y - b.y}; }
    HIPGL_HOST_DEVICE Vec2 operator*(float s) const { return {x * s, y * s}; }
};

// ── Vec3 ────────────────────────────────────────────────────────────────────

struct Vec3 {
    float x, y, z;

    HIPGL_HOST_DEVICE Vec3() : x(0), y(0), z(0) {}
    HIPGL_HOST_DEVICE Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    HIPGL_HOST_DEVICE explicit Vec3(float s) : x(s), y(s), z(s) {}
    HIPGL_HOST_DEVICE Vec3(const float* p) : x(p[0]), y(p[1]), z(p[2]) {}

    HIPGL_HOST_DEVICE Vec3  operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    HIPGL_HOST_DEVICE Vec3  operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    HIPGL_HOST_DEVICE Vec3  operator*(float s) const { return {x * s, y * s, z * s}; }
    HIPGL_HOST_DEVICE Vec3  operator*(Vec3 b) const { return {x * b.x, y * b.y, z * b.z}; }
    HIPGL_HOST_DEVICE Vec3  operator-() const { return {-x, -y, -z}; }
    HIPGL_HOST_DEVICE Vec3& operator+=(Vec3 b) { x += b.x; y += b.y; z += b.z; return *this; }

    HIPGL_HOST_DEVICE float dot(Vec3 b) const { return x * b.x + y * b.y + z * b.z; }

    HIPGL_HOST_DEVICE Vec3 cross(Vec3 b) const {
        return {y * b.z - z * b.y,
                z * b.x - x * b.z,
                x * b.y - y * b.x};
    }

    HIPGL_HOST_DEVICE float length_sq() const { return dot(*this); }
    HIPGL_HOST_DEVICE float length() const { return sqrtf(length_sq()); }

    HIPGL_HOST_DEVICE Vec3 normalized() const {
        float len = length();
        if (len < 1e-8f) return {0, 0, 0};
        float inv = 1.0f / len;
        return {x * inv, y * inv, z * inv};
    }

    HIPGL_HOST_DEVICE void store(float* p) const { p[0] = x; p[1] = y; p[2] = z; }
};

HIPGL_HOST_DEVICE inline Vec3 operator*(float s, Vec3 v) { return v * s; }

// ── Vec4 ────────────────────────────────────────────────────────────────────

struct Vec4 {
    float x, y, z, w;

    HIPGL_HOST_DEVICE Vec4() : x(0), y(0), z(0), w(0) {}
    HIPGL_HOST_DEVICE Vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    HIPGL_HOST_DEVICE Vec4(Vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
    HIPGL_HOST_DEVICE explicit Vec4(float s) : x(s), y(s), z(s), w(s) {}
    HIPGL_HOST_DEVICE Vec4(const float* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}

    HIPGL_HOST_DEVICE Vec4  operator+(Vec4 b) const { return {x + b.x, y + b.y, z + b.z, w + b.w}; }
    HIPGL_HOST_DEVICE Vec4  operator-(Vec4 b) const { return {x - b.x, y - b.y, z - b.z, w - b.w}; }
    HIPGL_HOST_DEVICE Vec4  operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
    HIPGL_HOST_DEVICE Vec4& operator+=(Vec4 b) { x += b.x; y += b.y; z += b.z; w += b.w; return *this; }

    HIPGL_HOST_DEVICE float  dot(Vec4 b) const { return x * b.x + y * b.y + z * b.z + w * b.w; }
    HIPGL_HOST_DEVICE Vec3   xyz() const { return {x, y, z}; }

    HIPGL_HOST_DEVICE void store(float* p) const { p[0] = x; p[1] = y; p[2] = z; p[3] = w; }
};

HIPGL_HOST_DEVICE inline Vec4 operator*(float s, Vec4 v) { return v * s; }

// ── Mat4 (column-major, OpenGL convention) ──────────────────────────────────
//
//  Storage: m[col][row]  →  memory layout matches OpenGL's column-major order.
//  Element access: operator()(row, col)
//

struct Mat4 {
    float m[4][4]; // m[col][row]

    HIPGL_HOST_DEVICE Mat4() {
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                m[c][r] = 0.0f;
    }

    HIPGL_HOST_DEVICE float& operator()(int row, int col)       { return m[col][row]; }
    HIPGL_HOST_DEVICE float  operator()(int row, int col) const { return m[col][row]; }

    static HIPGL_HOST_DEVICE Mat4 identity() {
        Mat4 I;
        I.m[0][0] = I.m[1][1] = I.m[2][2] = I.m[3][3] = 1.0f;
        return I;
    }

    // Matrix-vector multiply: M * v
    HIPGL_HOST_DEVICE Vec4 operator*(Vec4 v) const {
        Vec4 r;
        r.x = m[0][0]*v.x + m[1][0]*v.y + m[2][0]*v.z + m[3][0]*v.w;
        r.y = m[0][1]*v.x + m[1][1]*v.y + m[2][1]*v.z + m[3][1]*v.w;
        r.z = m[0][2]*v.x + m[1][2]*v.y + m[2][2]*v.z + m[3][2]*v.w;
        r.w = m[0][3]*v.x + m[1][3]*v.y + m[2][3]*v.z + m[3][3]*v.w;
        return r;
    }

    // Matrix-matrix multiply: A * B
    HIPGL_HOST_DEVICE Mat4 operator*(Mat4 B) const {
        Mat4 R;
        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 4; r++) {
                R.m[c][r] = m[0][r]*B.m[c][0]
                           + m[1][r]*B.m[c][1]
                           + m[2][r]*B.m[c][2]
                           + m[3][r]*B.m[c][3];
            }
        }
        return R;
    }

    // Transform a Vec3 as a point (w=1), return Vec3 after perspective divide
    HIPGL_HOST_DEVICE Vec3 transform_point(Vec3 p) const {
        Vec4 h = (*this) * Vec4(p, 1.0f);
        if (fabsf(h.w) > 1e-8f) {
            float inv = 1.0f / h.w;
            return {h.x * inv, h.y * inv, h.z * inv};
        }
        return h.xyz();
    }

    // Transform a Vec3 as a direction (w=0)
    HIPGL_HOST_DEVICE Vec3 transform_dir(Vec3 d) const {
        Vec4 r = (*this) * Vec4(d, 0.0f);
        return r.xyz();
    }

    // Load from flat column-major float[16] array (OpenGL format)
    static HIPGL_HOST_DEVICE Mat4 from_column_major(const float* p) {
        Mat4 M;
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                M.m[c][r] = p[c * 4 + r];
        return M;
    }

    // Store to flat column-major float[16] array
    HIPGL_HOST_DEVICE void to_column_major(float* p) const {
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                p[c * 4 + r] = m[c][r];
    }

    // Transpose
    HIPGL_HOST_DEVICE Mat4 transposed() const {
        Mat4 T;
        for (int c = 0; c < 4; c++)
            for (int r = 0; r < 4; r++)
                T.m[r][c] = m[c][r];
        return T;
    }

    // Upper-left 3x3 → flat 9-element array (column-major)
    HIPGL_HOST_DEVICE void upper3x3(float out[9]) const {
        out[0] = m[0][0]; out[1] = m[0][1]; out[2] = m[0][2];
        out[3] = m[1][0]; out[4] = m[1][1]; out[5] = m[1][2];
        out[6] = m[2][0]; out[7] = m[2][1]; out[8] = m[2][2];
    }

    // Compute inverse (general 4x4, Cramer's rule) - host only for simplicity
    HIPGL_HOST Mat4 inverse() const;
};

// ── Common transform constructors ───────────────────────────────────────────

inline Mat4 perspective(float fovy_rad, float aspect, float near, float far) {
    Mat4 M;
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    M(0,0) = f / aspect;
    M(1,1) = f;
    M(2,2) = (far + near) / (near - far);
    M(3,2) = -1.0f;
    M(2,3) = (2.0f * far * near) / (near - far);
    return M;
}

inline Mat4 ortho(float left, float right, float bottom, float top, float near, float far) {
    Mat4 M;
    M(0,0) =  2.0f / (right - left);
    M(1,1) =  2.0f / (top - bottom);
    M(2,2) = -2.0f / (far - near);
    M(0,3) = -(right + left) / (right - left);
    M(1,3) = -(top + bottom) / (top - bottom);
    M(2,3) = -(far + near) / (far - near);
    M(3,3) =  1.0f;
    return M;
}

inline Mat4 look_at(Vec3 eye, Vec3 center, Vec3 up) {
    Vec3 f = (center - eye).normalized();
    Vec3 s = f.cross(up).normalized();
    Vec3 u = s.cross(f);
    Mat4 M = Mat4::identity();
    M(0,0) =  s.x;  M(0,1) =  s.y;  M(0,2) =  s.z;
    M(1,0) =  u.x;  M(1,1) =  u.y;  M(1,2) =  u.z;
    M(2,0) = -f.x;  M(2,1) = -f.y;  M(2,2) = -f.z;
    M(0,3) = -(s.dot(eye));
    M(1,3) = -(u.dot(eye));
    M(2,3) =  f.dot(eye);
    return M;
}

inline Mat4 translate(Vec3 t) {
    Mat4 M = Mat4::identity();
    M(0,3) = t.x;
    M(1,3) = t.y;
    M(2,3) = t.z;
    return M;
}

inline Mat4 scale(Vec3 s) {
    Mat4 M;
    M(0,0) = s.x;
    M(1,1) = s.y;
    M(2,2) = s.z;
    M(3,3) = 1.0f;
    return M;
}

inline Mat4 rotate(float angle_rad, Vec3 axis) {
    Vec3 a = axis.normalized();
    float c = cosf(angle_rad);
    float s = sinf(angle_rad);
    float t = 1.0f - c;

    Mat4 M;
    M(0,0) = t*a.x*a.x + c;
    M(1,0) = t*a.x*a.y + s*a.z;
    M(2,0) = t*a.x*a.z - s*a.y;

    M(0,1) = t*a.x*a.y - s*a.z;
    M(1,1) = t*a.y*a.y + c;
    M(2,1) = t*a.y*a.z + s*a.x;

    M(0,2) = t*a.x*a.z + s*a.y;
    M(1,2) = t*a.y*a.z - s*a.x;
    M(2,2) = t*a.z*a.z + c;

    M(3,3) = 1.0f;
    return M;
}

} // namespace math
} // namespace hipgl
