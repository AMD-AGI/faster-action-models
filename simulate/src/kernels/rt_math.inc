#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include "hipgl/physics/rt_types.h"   // RT_HD, RT_MINVAL

#ifdef RT_PARITY_FP
#define RT_FP_PARITY _Pragma("clang fp contract(off)")
#else
#define RT_FP_PARITY
#endif

namespace hipgl {
namespace physics {

RT_HD inline float3 rt_f3(float x, float y, float z) { return make_float3(x, y, z); }

RT_HD inline float3 rt_add(float3 a, float3 b) { RT_FP_PARITY return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
RT_HD inline float3 rt_sub(float3 a, float3 b) { RT_FP_PARITY return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
RT_HD inline float3 rt_mul(float3 a, float s)  { RT_FP_PARITY return make_float3(a.x * s, a.y * s, a.z * s); }
RT_HD inline float  rt_dot(float3 a, float3 b) { RT_FP_PARITY return a.x * b.x + a.y * b.y + a.z * b.z; }
RT_HD inline float3 rt_cross(float3 a, float3 b) {
    RT_FP_PARITY
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
RT_HD inline float  rt_len(float3 a) { RT_FP_PARITY return sqrtf(rt_dot(a, a)); }

// Safe normalize: vectors shorter than 1e-15 are returned unchanged (so a zero
// vector stays exactly zero, like mjw tmath.normalize3).
RT_HD inline float3 rt_norm(float3 a) {
    RT_FP_PARITY
    const float n = rt_len(a);
    if (n < RT_MINVAL) return a;
    const float inv = 1.0f / n;
    return rt_mul(a, inv);
}

RT_HD inline float3 rt_min3(float3 a, float3 b) { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
RT_HD inline float3 rt_max3(float3 a, float3 b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
RT_HD inline float3 rt_absf3(float3 a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }

// R row-major 3x3: R v and R^T v. Geom frames map local->world with R, so the
// mjw _ray_map (local = R^T (o - p), R^T d) is rt_mat_tvec.
RT_HD inline float3 rt_mat_vec(const float R[9], float3 v) {
    RT_FP_PARITY
    return make_float3(R[0] * v.x + R[1] * v.y + R[2] * v.z,
                       R[3] * v.x + R[4] * v.y + R[5] * v.z,
                       R[6] * v.x + R[7] * v.y + R[8] * v.z);
}
RT_HD inline float3 rt_mat_tvec(const float R[9], float3 v) {
    RT_FP_PARITY
    return make_float3(R[0] * v.x + R[3] * v.y + R[6] * v.z,
                       R[1] * v.x + R[4] * v.y + R[7] * v.z,
                       R[2] * v.x + R[5] * v.y + R[8] * v.z);
}

// Unit quaternion (w, x, y, z) -> row-major rotation (MuJoCo mju_quat2Mat).
RT_HD inline void rt_quat_to_mat(const float q[4], float R[9]) {
    RT_FP_PARITY
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    R[0] = 1.f - 2.f * (yy + zz); R[1] = 2.f * (xy - wz);       R[2] = 2.f * (xz + wy);
    R[3] = 2.f * (xy + wz);       R[4] = 1.f - 2.f * (xx + zz); R[5] = 2.f * (yz - wx);
    R[6] = 2.f * (xz - wy);       R[7] = 2.f * (yz + wx);       R[8] = 1.f - 2.f * (xx + yy);
}

// Per-component 1/d for slab tests. |d| <= 1e-20 maps to +-1e20 so
// (lo - o) * inv stays finite (no 0 * inf NaN). The sign follows mjw
// (render.py _ray_mesh_bvh: where(d >= 0, big, -big)), NOT copysign: -0.0
// compares >= 0 and so maps to +1e20. This matters only for a ray whose
// origin lies exactly on a box face with that direction component == -0.0.
RT_HD inline float rt_safe_inv1(float d) {
    return fabsf(d) <= 1e-20f ? (d >= 0.0f ? 1e20f : -1e20f) : 1.0f / d;
}
RT_HD inline float3 rt_safe_inv(float3 d) {
    return make_float3(rt_safe_inv1(d.x), rt_safe_inv1(d.y), rt_safe_inv1(d.z));
}

}  // namespace physics
}  // namespace hipgl
