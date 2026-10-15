// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions of this file are ported from MuJoCo Warp (mujoco_warp/_src/ray.py and render.py ray/primitive intersection),
// Copyright 2025 The Newton Developers, licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Modified by Advanced Micro Devices, Inc., 2026:
// ported to HIP / C++. See THIRD_PARTY_NOTICES.md.
// rt_intersect.inc — ray/primitive and ray/triangle intersectors (host + device).
//
// Faithful fp32 ports of mujoco_warp ray.py (_ray_quad, ray_plane, ray_sphere,
// ray_capsule, ray_cylinder, ray_box) and render.py (_ray_triangle_uv and the
// orthogonal basis in _ray_mesh_uvf).
//
// Conventions shared by every primitive test:
//   * Inputs are in the geom LOCAL frame: lp = R^T (o - p), lv = R^T d. d need
//     not be unit; the returned t is in units of d, so it equals the world t.
//   * Return t >= tmin, or -1 on miss. mjw is tmin = 0; tmin must be >= 0
//     (the -1 miss sentinel of _ray_quad relies on it, exactly as in mjw).
//   * ln = LOCAL unit outward normal (caller rotates by R). mjw returns the
//     world normal R * ln; doing the rotation once in the caller is equivalent.
//   * Every mjw "x >= 0" acceptance becomes "x >= tmin"; all other epsilons
//     (1e-15) and comparison directions are copied verbatim.
//   * Capsule/cylinder/box keep mjw's bounding-sphere early-out. mjw evaluates
//     it in world space; here it is evaluated in the local frame (identical up
//     to rounding since the frame is rigid).
#pragma once

#include "rt_math.inc"

namespace hipgl {
namespace physics {

// Solutions of a x^2 + 2 b x + c = 0 (mjw _ray_quad, ray.py:86). Returns the
// smaller root if >= tmin, else the larger if >= tmin, else -1. x0/x1 are the
// raw roots (x0 <= x1 for a > 0), both -1 when the discriminant is < 1e-15.
RT_HD inline float rt_ray_quad(float a, float b, float c, float tmin, float& x0, float& x1) {
    RT_FP_PARITY
    float det = b * b - a * c;
    if (det < RT_MINVAL) { x0 = -1.f; x1 = -1.f; return -1.f; }
    det = sqrtf(det);
    const float den = 1.0f / (a != 0.f ? a : RT_MINVAL);   // tmath.safe_div(1, a)
    x0 = (-b - det) * den;
    x1 = (-b + det) * den;
    if (x0 >= tmin) return x0;
    if (x1 >= tmin) return x1;
    return -1.f;
}

// Bounding-sphere early-out (radius^2 = r2, centred at the geom origin).
RT_HD inline bool rt_bsphere(float3 lp, float3 lv, float r2, float tmin) {
    RT_FP_PARITY
    float x0, x1;
    return rt_ray_quad(rt_dot(lv, lv), rt_dot(lv, lp), rt_dot(lp, lp) - r2, tmin, x0, x1) >= 0.f;
}

// Ray/AABB slab test. iv = rt_safe_inv(dir). tn = entry distance clamped to
// tmin. True if the clamped interval [tn, min(tf, tmax)] is non-empty.
RT_HD inline bool rt_slab(float3 o, float3 iv, float3 lo, float3 hi, float tmin, float tmax, float& tn) {
    const float ax = (lo.x - o.x) * iv.x, bx = (hi.x - o.x) * iv.x;
    const float ay = (lo.y - o.y) * iv.y, by = (hi.y - o.y) * iv.y;
    const float az = (lo.z - o.z) * iv.z, bz = (hi.z - o.z) * iv.z;
    const float t0 = fmaxf(fmaxf(fminf(ax, bx), fminf(ay, by)), fmaxf(fminf(az, bz), tmin));
    const float t1 = fminf(fminf(fmaxf(ax, bx), fmaxf(ay, by)), fminf(fmaxf(az, bz), tmax));
    tn = t0;
    return t0 <= t1;
}

// Plane (ray.py ray_plane): one-sided, only rays travelling against +z hit.
// size[0]/size[1] <= 0 means unbounded in that axis.
RT_HD inline float rt_hit_plane(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    if (lv.z > -RT_MINVAL) return -1.f;
    const float x = -lp.z / lv.z;
    if (x < tmin) return -1.f;
    const float p0 = lp.x + x * lv.x;
    const float p1 = lp.y + x * lv.y;
    if ((size[0] <= 0.f || fabsf(p0) <= size[0]) && (size[1] <= 0.f || fabsf(p1) <= size[1])) {
        ln = make_float3(0.f, 0.f, 1.f);
        return x;
    }
    return -1.f;
}

// Sphere (ray.py ray_sphere), radius size[0]. Starting inside returns the far root.
RT_HD inline float rt_hit_sphere(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    const float a = rt_dot(lv, lv);
    const float b = rt_dot(lv, lp);
    const float c = rt_dot(lp, lp) - size[0] * size[0];
    float x0, x1;
    const float sol = rt_ray_quad(a, b, c, tmin, x0, x1);
    if (sol >= 0.f) ln = rt_norm(rt_add(lp, rt_mul(lv, sol)));
    return sol;
}

// Ellipsoid with semi-axes size[0..2]: the unit sphere in the scaled frame
// (same ray parameter t); normal = gradient l / size^2.
RT_HD inline float rt_hit_ellipsoid(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    const float3 is = make_float3(1.0f / size[0], 1.0f / size[1], 1.0f / size[2]);
    const float3 p = make_float3(lp.x * is.x, lp.y * is.y, lp.z * is.z);
    const float3 v = make_float3(lv.x * is.x, lv.y * is.y, lv.z * is.z);
    float x0, x1;
    const float sol = rt_ray_quad(rt_dot(v, v), rt_dot(v, p), rt_dot(p, p) - 1.0f, tmin, x0, x1);
    if (sol >= 0.f) {
        const float3 q = rt_add(p, rt_mul(v, sol));
        ln = rt_norm(make_float3(q.x * is.x, q.y * is.y, q.z * is.z));
    }
    return sol;
}

// Capsule (ray.py ray_capsule): radius size[0], half-length size[1] along z.
// Body = infinite cylinder clipped to |z| <= h; caps = spheres at +-h accepted
// only on their outer hemispheres. Smallest valid root wins (strict <).
RT_HD inline float rt_hit_capsule(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    const float r = size[0], h = size[1];
    const float rs = r + h;
    if (!rt_bsphere(lp, lv, rs * rs, tmin)) return -1.f;

    float x = -1.f;
    int part = 0;   // -1 bottom cap, 0 body, 1 top cap
    const float sq0 = r * r;
    float a = lv.x * lv.x + lv.y * lv.y;
    float b = lv.x * lp.x + lv.y * lp.y;
    float c = lp.x * lp.x + lp.y * lp.y - sq0;
    float x0, x1;
    const float sol = rt_ray_quad(a, b, c, tmin, x0, x1);
    if (sol >= 0.f && fabsf(lp.z + sol * lv.z) <= h) {
        if (x < 0.f || sol < x) x = sol;
    }

    // top cap
    float3 ld = make_float3(lp.x, lp.y, lp.z - h);
    a += lv.z * lv.z;
    b = rt_dot(lv, ld);
    c = rt_dot(ld, ld) - sq0;
    rt_ray_quad(a, b, c, tmin, x0, x1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float xi = i == 0 ? x0 : x1;
        if (xi >= tmin && lp.z + xi * lv.z >= h) {
            if (x < 0.f || xi < x) { x = xi; part = 1; }
        }
    }

    // bottom cap
    ld.z = lp.z + h;
    b = rt_dot(lv, ld);
    c = rt_dot(ld, ld) - sq0;
    rt_ray_quad(a, b, c, tmin, x0, x1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const float xi = i == 0 ? x0 : x1;
        if (xi >= tmin && lp.z + xi * lv.z <= -h) {
            if (x < 0.f || xi < x) { x = xi; part = -1; }
        }
    }

    if (x >= 0.f) {
        const float nz = part == 0 ? 0.f : lp.z + lv.z * x - h * (float)part;
        ln = rt_norm(make_float3(lp.x + lv.x * x, lp.y + lv.y * x, nz));
    }
    return x;
}

// Cylinder (ray.py ray_cylinder): radius size[0], half-length size[1] along z.
// Flat caps first (bottom then top), then the round side; strict < keeps the
// earlier candidate on exact ties, as in mjw.
RT_HD inline float rt_hit_cylinder(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    const float r = size[0], h = size[1];
    if (!rt_bsphere(lp, lv, r * r + h * h, tmin)) return -1.f;

    float x = -1.f;
    int part = 0;
    if (fabsf(lv.z) > RT_MINVAL) {
#pragma unroll
        for (int side = -1; side < 2; side += 2) {
            const float sol = ((float)side * h - lp.z) / lv.z;
            if (sol >= tmin) {
                const float p0 = lp.x + sol * lv.x;
                const float p1 = lp.y + sol * lv.y;
                if (p0 * p0 + p1 * p1 <= r * r) {
                    if (x < 0.f || sol < x) { x = sol; part = side; }
                }
            }
        }
    }

    const float a = lv.x * lv.x + lv.y * lv.y;
    const float b = lv.x * lp.x + lv.y * lp.y;
    const float c = lp.x * lp.x + lp.y * lp.y - r * r;
    float x0, x1;
    const float sol = rt_ray_quad(a, b, c, tmin, x0, x1);
    if (sol >= 0.f && fabsf(lp.z + sol * lv.z) <= h) {
        if (x < 0.f || sol < x) { x = sol; part = 0; }
    }

    if (x >= 0.f) {
        ln = part == 0 ? rt_norm(make_float3(lp.x + lv.x * x, lp.y + lv.y * x, 0.f))
                       : make_float3(0.f, 0.f, (float)part);
    }
    return x;
}

// Box (ray.py ray_box): half extents size[0..2]. Each face with |lv_i| > 1e-15
// is tested with inclusive bounds on the other two axes; the smallest t >= tmin
// wins (strict <, faces visited x-,x+,y-,y+,z-,z+). Normal = side * e_axis.
RT_HD inline float rt_hit_box(float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    RT_FP_PARITY
    const float sx = size[0], sy = size[1], sz = size[2];
    if (!rt_bsphere(lp, lv, sx * sx + sy * sy + sz * sz, tmin)) return -1.f;

    const float P[3] = {lp.x, lp.y, lp.z};
    const float V[3] = {lv.x, lv.y, lv.z};
    float x = -1.f;
    int face_axis = -1, face_side = -1;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int j0 = i == 0 ? 1 : 0;      // _IFACE: 0->(1,2), 1->(0,2), 2->(0,1)
        const int j1 = i == 2 ? 1 : 2;
        if (fabsf(V[i]) > RT_MINVAL) {
#pragma unroll
            for (int side = -1; side < 2; side += 2) {
                const float sol = ((float)side * size[i] - P[i]) / V[i];
                if (sol >= tmin) {
                    const float p0 = P[j0] + sol * V[j0];
                    const float p1 = P[j1] + sol * V[j1];
                    if (fabsf(p0) <= size[j0] && fabsf(p1) <= size[j1]) {
                        if (x < 0.f || sol < x) { x = sol; face_axis = i; face_side = side; }
                    }
                }
            }
        }
    }
    if (x >= 0.f) {
        const float fs = (float)face_side;
        ln = make_float3(face_axis == 0 ? fs : 0.f, face_axis == 1 ? fs : 0.f, face_axis == 2 ? fs : 0.f);
    }
    return x;
}

// Dispatch by GeomType. Meshes are handled by the BLAS walk, so return a miss.
RT_HD inline float rt_hit_prim(int type, float3 lp, float3 lv, const float* size, float tmin, float3& ln) {
    switch (type) {
        case GEOM_PLANE:    return rt_hit_plane(lp, lv, size, tmin, ln);
        case GEOM_SPHERE:   return rt_hit_sphere(lp, lv, size, tmin, ln);
        case GEOM_CAPSULE:  return rt_hit_capsule(lp, lv, size, tmin, ln);
        case GEOM_CYLINDER: return rt_hit_cylinder(lp, lv, size, tmin, ln);
        case GEOM_BOX:      return rt_hit_box(lp, lv, size, tmin, ln);
        case RT_GEOM_ELLIPSOID: return rt_hit_ellipsoid(lp, lv, size, tmin, ln);
        default:            return -1.f;
    }
}

// Orthonormal basis of the plane perpendicular to lv (render.py _ray_mesh_uvf,
// the b0/b1 block). Computed once per (ray, mesh instance).
RT_HD inline void rt_tri_basis(float3 lv, float3& b0, float3& b1) {
    RT_FP_PARITY
    const float ax = fabsf(lv.x), ay = fabsf(lv.y), az = fabsf(lv.z);
    if (ax < ay) {
        b0 = ax < az ? make_float3(0.f, lv.z, -lv.y) : make_float3(lv.y, -lv.x, 0.f);
    } else {
        b0 = ay < az ? make_float3(-lv.z, 0.f, lv.x) : make_float3(lv.y, -lv.x, 0.f);
    }
    b0 = rt_norm(b0);
    b1 = rt_norm(rt_cross(lv, b0));
}

// mjw triangle test (render.py _ray_triangle_uv). Two-sided. Projects the
// corners onto (b0, b1), rejects when all three lie strictly on one side of
// either axis, solves the 2x2 barycentric system, then the plane distance.
// u = weight of v0, v = weight of v1. ng = (v0 - v2) x (v1 - v2), unnormalized.
RT_HD inline bool rt_tri_mjw(float3 lp, float3 lv, float3 b0, float3 b1,
                             float3 v0, float3 v1, float3 v2, float tmin,
                             float& t, float& u, float& v, float3& ng) {
    RT_FP_PARITY
    const float3 d0 = rt_sub(v0, lp), d1 = rt_sub(v1, lp), d2 = rt_sub(v2, lp);
    const float p00 = rt_dot(d0, b0), p01 = rt_dot(d0, b1);
    const float p10 = rt_dot(d1, b0), p11 = rt_dot(d1, b1);
    const float p20 = rt_dot(d2, b0), p21 = rt_dot(d2, b1);
    if ((p00 > 0.f && p10 > 0.f && p20 > 0.f) || (p00 < 0.f && p10 < 0.f && p20 < 0.f) ||
        (p01 > 0.f && p11 > 0.f && p21 > 0.f) || (p01 < 0.f && p11 < 0.f && p21 < 0.f))
        return false;

    const float A00 = p00 - p20, A10 = p10 - p20;
    const float A01 = p01 - p21, A11 = p11 - p21;
    const float bb0 = -p20, bb1 = -p21;
    const float det = A00 * A11 - A10 * A01;
    if (fabsf(det) < RT_MINVAL) return false;
    const float t0 = (A11 * bb0 - A10 * bb1) / det;
    const float t1 = (-A01 * bb0 + A00 * bb1) / det;
    if (t0 < 0.f || t1 < 0.f || t0 + t1 > 1.f) return false;

    const float3 e0 = rt_sub(v0, v2), e1 = rt_sub(v1, v2), e2 = rt_sub(lp, v2);
    const float3 n = rt_cross(e0, e1);
    const float denom = rt_dot(lv, n);
    if (fabsf(denom) < RT_MINVAL) return false;
    const float dist = -rt_dot(e2, n) / denom;
    if (!(dist >= tmin)) return false;
    t = dist; u = t0; v = t1; ng = n;
    return true;
}

// Moller-Trumbore, edge-inclusive, written around v2 so u/v/ng match
// rt_tri_mjw: u = weight of v0, v = weight of v1, ng = (v0-v2) x (v1-v2).
RT_HD inline bool rt_tri_mt(float3 lp, float3 lv, float3 v0, float3 v1, float3 v2, float tmin,
                            float& t, float& u, float& v, float3& ng) {
    RT_FP_PARITY
    const float3 e0 = rt_sub(v0, v2), e1 = rt_sub(v1, v2);
    const float3 P = rt_cross(lv, e1);
    const float det = rt_dot(e0, P);
    if (fabsf(det) < RT_MINVAL) return false;
    const float inv = 1.0f / det;
    const float3 T = rt_sub(lp, v2);
    const float uu = rt_dot(T, P) * inv;
    if (uu < 0.f || uu > 1.f) return false;
    const float3 Q = rt_cross(T, e0);
    const float vv = rt_dot(lv, Q) * inv;
    if (vv < 0.f || uu + vv > 1.f) return false;
    const float tt = rt_dot(e1, Q) * inv;
    if (!(tt >= tmin)) return false;
    t = tt; u = uu; v = vv; ng = rt_cross(e0, e1);
    return true;
}

}  // namespace physics
}  // namespace hipgl
