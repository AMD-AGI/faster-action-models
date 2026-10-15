// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions of this file are transcribed from MuJoCo 3.9.0 (engine_collision_box.c box-box collision, engine_util_misc.c mju_outsideBox),
// Copyright 2021 DeepMind Technologies Limited, licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Modified by Advanced Micro Devices, Inc., 2026:
// ported to HIP device code. See THIRD_PARTY_NOTICES.md.
// Collision detection kernels (v2: multi-contact + full SAT)
//
// Broadphase:  O(N^2) AABB overlap test. One thread per geom pair per env.
// Narrowphase: Per overlapping pair, compute contact points/normals/depths.
//              Multi-contact for box-plane (up to 4), capsule-plane (up to 2),
//              and box-box (up to 4, full 15-axis SAT + face clipping).

#ifdef MJC_WANT_PROF
#define MJC_FAN_PROF 97
#endif
#include <hip/hip_runtime.h>
#include <cstdlib>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/world_mask.h"
#include "hipgl/physics/physics_mesh_box.h"
#include "hipgl/physics/physics_mjccd_worker.h"   // MuJoCo 3.9 mjc_Convex port (namespace mjccd)

namespace hipgl {
namespace physics {
namespace kernels {

// world skip: per-world freeze mask (see world_mask.h).
HIPGL_WORLD_MASK_DEFINE(set_world_mask_collision)

constexpr int MAX_PAIR_CONTACTS = 8;
constexpr int MAX_CLIP_VERTS    = 8;
constexpr int MAX_CANDIDATE_CONTACTS = 32;  // max candidates before manifold selection

// ── Helpers ─────────────────────────────────────────────────────────────────

HIPGL_DEVICE inline float dev_min(float a, float b) { return a < b ? a : b; }
HIPGL_DEVICE inline float dev_max(float a, float b) { return a > b ? a : b; }
HIPGL_DEVICE inline float dev_abs(float a) { return a < 0 ? -a : a; }
HIPGL_DEVICE inline float dev_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

HIPGL_DEVICE inline void quat_rotate_vec(const float* q, const float* v, float* out) {
    float tx = 2.0f * (q[2]*v[2] - q[3]*v[1]);
    float ty = 2.0f * (q[3]*v[0] - q[1]*v[2]);
    float tz = 2.0f * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}

HIPGL_DEVICE inline void quat_inv_rotate_vec(const float* q, const float* v, float* out) {
    float cq[4] = {q[0], -q[1], -q[2], -q[3]};
    quat_rotate_vec(cq, v, out);
}

HIPGL_DEVICE inline float vec3_dot(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

HIPGL_DEVICE inline float vec3_dist(const float* a, const float* b) {
    float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

HIPGL_DEVICE inline void vec3_sub(const float* a, const float* b, float* out) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}

HIPGL_DEVICE inline void vec3_normalize(float* v) {
    float len = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-10f) { float inv = 1.0f/len; v[0]*=inv; v[1]*=inv; v[2]*=inv; }
}

HIPGL_DEVICE inline void vec3_cross(const float* a, const float* b, float* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

HIPGL_DEVICE inline float vec3_len(const float* v) {
    return sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// ── Double-precision overloads (for float64 collision pipeline) ─────────────

HIPGL_DEVICE inline double dev_min_d(double a, double b) { return a < b ? a : b; }
HIPGL_DEVICE inline double dev_max_d(double a, double b) { return a > b ? a : b; }
HIPGL_DEVICE inline double dev_abs_d(double a) { return a < 0 ? -a : a; }
HIPGL_DEVICE inline double dev_clamp_d(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

HIPGL_DEVICE inline void quat_rotate_vec_d(const double* q, const double* v, double* out) {
    double tx = 2.0 * (q[2]*v[2] - q[3]*v[1]);
    double ty = 2.0 * (q[3]*v[0] - q[1]*v[2]);
    double tz = 2.0 * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}

HIPGL_DEVICE inline void quat_inv_rotate_vec_d(const double* q, const double* v, double* out) {
    double cq[4] = {q[0], -q[1], -q[2], -q[3]};
    quat_rotate_vec_d(cq, v, out);
}

HIPGL_DEVICE inline double vec3_dot_d(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

HIPGL_DEVICE inline double vec3_dist_d(const double* a, const double* b) {
    double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

HIPGL_DEVICE inline void vec3_sub_d(const double* a, const double* b, double* out) {
    out[0] = a[0]-b[0]; out[1] = a[1]-b[1]; out[2] = a[2]-b[2];
}

HIPGL_DEVICE inline void vec3_normalize_d(double* v) {
    double len = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len > 1e-15) { double inv = 1.0/len; v[0]*=inv; v[1]*=inv; v[2]*=inv; }
}

HIPGL_DEVICE inline void vec3_cross_d(const double* a, const double* b, double* out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

HIPGL_DEVICE inline double vec3_len_d(const double* v) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Helper: extract 3x3 rotation matrix from quaternion (double)
HIPGL_DEVICE inline void quat_to_mat3_d(const double* q, double R[3][3]) {
    double w=q[0], x=q[1], y=q[2], z=q[3];
    double xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    R[0][0]=1-2*(yy+zz); R[0][1]=2*(xy-wz);   R[0][2]=2*(xz+wy);
    R[1][0]=2*(xy+wz);   R[1][1]=1-2*(xx+zz); R[1][2]=2*(yz-wx);
    R[2][0]=2*(xz-wy);   R[2][1]=2*(yz+wx);   R[2][2]=1-2*(xx+yy);
}

// Helper: get column k of rotation matrix from quaternion (double)
HIPGL_DEVICE inline void mat_col_d(const double* q, int k, double* out) {
    double R[3][3];
    quat_to_mat3_d(q, R);
    out[0] = R[0][k]; out[1] = R[1][k]; out[2] = R[2][k];
}

// ── AABB overlap test ───────────────────────────────────────────────────────

HIPGL_DEVICE inline bool aabb_overlap(const AABB& a, const AABB& b) {
    return a.min_x <= b.max_x && a.max_x >= b.min_x &&
           a.min_y <= b.max_y && a.max_y >= b.min_y &&
           a.min_z <= b.max_z && a.max_z >= b.min_z;
}

// ── MuJoCo-style 4-point manifold selection ─────────────────────────────────
// Selects 4 maximally-spaced points from N candidates for stable contact support.
// Algorithm: A=first, B=furthest from A, C=furthest from line AB, D=furthest from triangle ABC.

HIPGL_DEVICE int manifold_select_4(
    float cand_pos[][3], float cand_norm[][3], float cand_depth[],
    int ncand,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    const float DEDUP_EPS2 = 1e-12f;   // (1 micron)^2
    if (ncand <= MAX_PAIR_CONTACTS) {
        int nout = 0;
        for (int i = 0; i < ncand; i++) {
            bool dup = false;
            for (int j = 0; j < nout; j++) {
                float dx = cand_pos[i][0] - out_pos[j][0];
                float dy = cand_pos[i][1] - out_pos[j][1];
                float dz = cand_pos[i][2] - out_pos[j][2];
                if (dx*dx + dy*dy + dz*dz < DEDUP_EPS2) {
                    // Keep the deeper of the two coincident points.
                    if (cand_depth[i] > out_depth[j]) out_depth[j] = cand_depth[i];
                    dup = true;
                    break;
                }
            }
            if (dup) continue;
            out_pos[nout][0] = cand_pos[i][0]; out_pos[nout][1] = cand_pos[i][1]; out_pos[nout][2] = cand_pos[i][2];
            out_norm[nout][0] = cand_norm[i][0]; out_norm[nout][1] = cand_norm[i][1]; out_norm[nout][2] = cand_norm[i][2];
            out_depth[nout] = cand_depth[i];
            nout++;
        }
        return nout;
    }

    int sel[4] = {0, 0, 0, 0};

    // A: first candidate
    sel[0] = 0;

    // B: furthest from A
    float best_d2 = -1.0f;
    for (int i = 1; i < ncand; i++) {
        float dx = cand_pos[i][0] - cand_pos[sel[0]][0];
        float dy = cand_pos[i][1] - cand_pos[sel[0]][1];
        float dz = cand_pos[i][2] - cand_pos[sel[0]][2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 > best_d2) { best_d2 = d2; sel[1] = i; }
    }

    // C: furthest from line A-B (perpendicular distance)
    float ab[3] = {cand_pos[sel[1]][0] - cand_pos[sel[0]][0],
                   cand_pos[sel[1]][1] - cand_pos[sel[0]][1],
                   cand_pos[sel[1]][2] - cand_pos[sel[0]][2]};
    float ab_len = sqrtf(ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]);
    if (ab_len > 1e-10f) { ab[0] /= ab_len; ab[1] /= ab_len; ab[2] /= ab_len; }

    best_d2 = -1.0f;
    for (int i = 0; i < ncand; i++) {
        if (i == sel[0] || i == sel[1]) continue;
        float ap[3] = {cand_pos[i][0] - cand_pos[sel[0]][0],
                       cand_pos[i][1] - cand_pos[sel[0]][1],
                       cand_pos[i][2] - cand_pos[sel[0]][2]};
        float proj = ap[0]*ab[0] + ap[1]*ab[1] + ap[2]*ab[2];
        float perp[3] = {ap[0] - proj*ab[0], ap[1] - proj*ab[1], ap[2] - proj*ab[2]};
        float d2 = perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2];
        if (d2 > best_d2) { best_d2 = d2; sel[2] = i; }
    }

    // D: furthest from plane of triangle A-B-C
    float ac[3] = {cand_pos[sel[2]][0] - cand_pos[sel[0]][0],
                   cand_pos[sel[2]][1] - cand_pos[sel[0]][1],
                   cand_pos[sel[2]][2] - cand_pos[sel[0]][2]};
    float tri_n[3];
    tri_n[0] = ab[1]*ac[2] - ab[2]*ac[1];
    tri_n[1] = ab[2]*ac[0] - ab[0]*ac[2];
    tri_n[2] = ab[0]*ac[1] - ab[1]*ac[0];
    float tri_n_len = sqrtf(tri_n[0]*tri_n[0] + tri_n[1]*tri_n[1] + tri_n[2]*tri_n[2]);
    if (tri_n_len > 1e-10f) { tri_n[0] /= tri_n_len; tri_n[1] /= tri_n_len; tri_n[2] /= tri_n_len; }

    float best_dist = -1.0f;
    for (int i = 0; i < ncand; i++) {
        if (i == sel[0] || i == sel[1] || i == sel[2]) continue;
        float ap[3] = {cand_pos[i][0] - cand_pos[sel[0]][0],
                       cand_pos[i][1] - cand_pos[sel[0]][1],
                       cand_pos[i][2] - cand_pos[sel[0]][2]};
        float d = fabsf(ap[0]*tri_n[0] + ap[1]*tri_n[1] + ap[2]*tri_n[2]);
        if (d > best_dist) { best_dist = d; sel[3] = i; }
    }

    // If all points are coplanar, pick the one furthest from any selected point
    if (best_dist < 1e-8f) {
        float best_min_d2 = -1.0f;
        for (int i = 0; i < ncand; i++) {
            if (i == sel[0] || i == sel[1] || i == sel[2]) continue;
            float min_d2 = 1e20f;
            for (int s = 0; s < 3; s++) {
                float dx = cand_pos[i][0] - cand_pos[sel[s]][0];
                float dy = cand_pos[i][1] - cand_pos[sel[s]][1];
                float dz = cand_pos[i][2] - cand_pos[sel[s]][2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < min_d2) min_d2 = d2;
            }
            if (min_d2 > best_min_d2) { best_min_d2 = min_d2; sel[3] = i; }
        }
    }

    for (int k = 0; k < 4; k++) {
        int s = sel[k];
        out_pos[k][0] = cand_pos[s][0]; out_pos[k][1] = cand_pos[s][1]; out_pos[k][2] = cand_pos[s][2];
        out_norm[k][0] = cand_norm[s][0]; out_norm[k][1] = cand_norm[s][1]; out_norm[k][2] = cand_norm[s][2];
        out_depth[k] = cand_depth[s];
    }
    return 4;
}

// ── Sphere-sphere ───────────────────────────────────────────────────────────

HIPGL_DEVICE bool collide_sphere_sphere(
    const float* p1, float r1, const float* p2, float r2,
    float* contact_pos, float* contact_normal, float* depth)
{
    float d[3]; vec3_sub(p2, p1, d);
    float dist = sqrtf(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]);
    float total_r = r1 + r2;
    if (dist > total_r) return false;

    *depth = total_r - dist;
    if (dist > 1e-8f) {
        float inv = 1.0f / dist;
        contact_normal[0] = d[0]*inv; contact_normal[1] = d[1]*inv; contact_normal[2] = d[2]*inv;
    } else {
        contact_normal[0] = 0; contact_normal[1] = 0; contact_normal[2] = 1;
    }
    contact_pos[0] = p1[0] + contact_normal[0] * (r1 - *depth * 0.5f);
    contact_pos[1] = p1[1] + contact_normal[1] * (r1 - *depth * 0.5f);
    contact_pos[2] = p1[2] + contact_normal[2] * (r1 - *depth * 0.5f);
    return true;
}

// ── Sphere-box ──────────────────────────────────────────────────────────────

HIPGL_DEVICE bool collide_sphere_box(
    const float* sp, float sr,
    const float* bp, const float* bq, const float* bsize,
    float* contact_pos, float* contact_normal, float* depth)
{
    float rel[3]; vec3_sub(sp, bp, rel);
    float local[3];
    quat_inv_rotate_vec(bq, rel, local);

    float closest[3];
    closest[0] = dev_clamp(local[0], -bsize[0], bsize[0]);
    closest[1] = dev_clamp(local[1], -bsize[1], bsize[1]);
    closest[2] = dev_clamp(local[2], -bsize[2], bsize[2]);

    float diff[3] = {local[0]-closest[0], local[1]-closest[1], local[2]-closest[2]};
    float dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2];

    if (dist_sq > sr * sr && dist_sq > 1e-12f) return false;

    float dist = sqrtf(dist_sq);
    float n_local[3];
    if (dist > 1e-8f) {
        float inv = 1.0f / dist;
        n_local[0] = diff[0]*inv; n_local[1] = diff[1]*inv; n_local[2] = diff[2]*inv;
        *depth = sr - dist;
    } else {
        float face_dist[6] = {
            bsize[0] - local[0], bsize[0] + local[0],
            bsize[1] - local[1], bsize[1] + local[1],
            bsize[2] - local[2], bsize[2] + local[2]
        };
        int min_face = 0;
        float min_d = face_dist[0];
        for (int i = 1; i < 6; i++) {
            if (face_dist[i] < min_d) { min_d = face_dist[i]; min_face = i; }
        }
        n_local[0] = n_local[1] = n_local[2] = 0;
        int axis = min_face / 2;
        n_local[axis] = (min_face % 2 == 0) ? 1.0f : -1.0f;
        *depth = sr + min_d;
    }

    quat_rotate_vec(bq, n_local, contact_normal);
    float cp_local[3] = {closest[0], closest[1], closest[2]};
    float cp_world[3];
    quat_rotate_vec(bq, cp_local, cp_world);
    // MuJoCo convention: contact position at midpoint between the two surface points.
    // box_surface = bp + cp_world; sphere_surface = sphere_center - sr * normal.
    // midpoint = box_surface + 0.5*(sphere_surface - box_surface)
    //          = box_surface - 0.5*depth * normal  (since depth = sr - dist)
    float half_d = *depth * 0.5f;
    contact_pos[0] = bp[0] + cp_world[0] - contact_normal[0] * half_d;
    contact_pos[1] = bp[1] + cp_world[1] - contact_normal[1] * half_d;
    contact_pos[2] = bp[2] + cp_world[2] - contact_normal[2] * half_d;
    return true;
}

// ── Closest point on line segment ───────────────────────────────────────────

HIPGL_DEVICE void closest_point_on_segment(
    const float* p, const float* a, const float* b, float* result)
{
    float ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    float ap[3] = {p[0]-a[0], p[1]-a[1], p[2]-a[2]};
    float ab_dot = vec3_dot(ab, ab);
    if (ab_dot < 1e-12f) { result[0]=a[0]; result[1]=a[1]; result[2]=a[2]; return; }
    float t = dev_clamp(vec3_dot(ap, ab) / ab_dot, 0.0f, 1.0f);
    result[0] = a[0] + t*ab[0]; result[1] = a[1] + t*ab[1]; result[2] = a[2] + t*ab[2];
}

// ── Sphere-capsule ──────────────────────────────────────────────────────────

HIPGL_DEVICE bool collide_sphere_capsule(
    const float* sp, float sr,
    const float* cp, const float* cq, float cr, float chalf,
    float* contact_pos, float* contact_normal, float* depth)
{
    float local_axis[3] = {0, 0, 1};
    float axis[3];
    quat_rotate_vec(cq, local_axis, axis);
    float a[3] = {cp[0]-axis[0]*chalf, cp[1]-axis[1]*chalf, cp[2]-axis[2]*chalf};
    float b[3] = {cp[0]+axis[0]*chalf, cp[1]+axis[1]*chalf, cp[2]+axis[2]*chalf};
    float closest[3];
    closest_point_on_segment(sp, a, b, closest);
    return collide_sphere_sphere(sp, sr, closest, cr, contact_pos, contact_normal, depth);
}

// ── Capsule-capsule ─────────────────────────────────────────────────────────

HIPGL_DEVICE bool collide_capsule_capsule(
    const float* p1, const float* q1, float r1, float h1,
    const float* p2, const float* q2, float r2, float h2,
    float* contact_pos, float* contact_normal, float* depth)
{
    float local_axis[3] = {0, 0, 1};
    float ax1[3], ax2[3];
    quat_rotate_vec(q1, local_axis, ax1);
    quat_rotate_vec(q2, local_axis, ax2);

    float a1[3] = {p1[0]-ax1[0]*h1, p1[1]-ax1[1]*h1, p1[2]-ax1[2]*h1};
    float b1[3] = {p1[0]+ax1[0]*h1, p1[1]+ax1[1]*h1, p1[2]+ax1[2]*h1};
    float a2[3] = {p2[0]-ax2[0]*h2, p2[1]-ax2[1]*h2, p2[2]-ax2[2]*h2};
    float b2[3] = {p2[0]+ax2[0]*h2, p2[1]+ax2[1]*h2, p2[2]+ax2[2]*h2};

    float d1[3] = {b1[0]-a1[0], b1[1]-a1[1], b1[2]-a1[2]};
    float d2[3] = {b2[0]-a2[0], b2[1]-a2[1], b2[2]-a2[2]};
    float r_vec[3] = {a1[0]-a2[0], a1[1]-a2[1], a1[2]-a2[2]};

    float a = vec3_dot(d1, d1);
    float e_val = vec3_dot(d2, d2);
    float f = vec3_dot(d2, r_vec);

    float s_param, t_param;
    if (a < 1e-10f && e_val < 1e-10f) {
        s_param = t_param = 0.0f;
    } else if (a < 1e-10f) {
        s_param = 0.0f;
        t_param = dev_clamp(f / e_val, 0.0f, 1.0f);
    } else {
        float c_val = vec3_dot(d1, r_vec);
        if (e_val < 1e-10f) {
            t_param = 0.0f;
            s_param = dev_clamp(-c_val / a, 0.0f, 1.0f);
        } else {
            float b_val = vec3_dot(d1, d2);
            float denom = a * e_val - b_val * b_val;
            s_param = (denom > 1e-10f) ? dev_clamp((b_val*f - c_val*e_val) / denom, 0.0f, 1.0f) : 0.0f;
            t_param = dev_clamp((b_val*s_param + f) / e_val, 0.0f, 1.0f);
            s_param = dev_clamp((b_val*t_param - c_val) / a, 0.0f, 1.0f);
        }
    }

    float c1[3] = {a1[0]+s_param*d1[0], a1[1]+s_param*d1[1], a1[2]+s_param*d1[2]};
    float c2[3] = {a2[0]+t_param*d2[0], a2[1]+t_param*d2[1], a2[2]+t_param*d2[2]};
    return collide_sphere_sphere(c1, r1, c2, r2, contact_pos, contact_normal, depth);
}

// ── Box-capsule (multi-contact) ─────────────────────────────────────────────
// Test both capsule endpoints as spheres against box, plus the closest point
// on the capsule segment. Returns up to MAX_PAIR_CONTACTS contacts.

HIPGL_DEVICE int collide_box_capsule_multi(
    const float* bp, const float* bq, const float* bs,
    const float* cp, const float* cq, float cr, float chalf,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    float local_axis[3] = {0, 0, 1};
    float axis[3];
    quat_rotate_vec(cq, local_axis, axis);

    float endpoints[2][3];
    endpoints[0][0] = cp[0]-axis[0]*chalf; endpoints[0][1] = cp[1]-axis[1]*chalf; endpoints[0][2] = cp[2]-axis[2]*chalf;
    endpoints[1][0] = cp[0]+axis[0]*chalf; endpoints[1][1] = cp[1]+axis[1]*chalf; endpoints[1][2] = cp[2]+axis[2]*chalf;

    int n_hits = 0;
    for (int ep = 0; ep < 2 && n_hits < MAX_PAIR_CONTACTS; ep++) {
        float cpos[3], cnorm[3], cdepth;
        if (collide_sphere_box(endpoints[ep], cr, bp, bq, bs, cpos, cnorm, &cdepth)) {
            // cnorm points from box toward sphere (capsule endpoint).
            // This matches the convention: normal from lower-type geom (box) toward
            // higher-type geom (capsule). The broadphase swap flip handles body ordering.
            out_norm[n_hits][0] = cnorm[0];
            out_norm[n_hits][1] = cnorm[1];
            out_norm[n_hits][2] = cnorm[2];
            out_pos[n_hits][0] = cpos[0];
            out_pos[n_hits][1] = cpos[1];
            out_pos[n_hits][2] = cpos[2];
            out_depth[n_hits] = cdepth;
            n_hits++;
        }
    }
    return n_hits;
}

// ── Sutherland-Hodgman polygon clip against a half-space ────────────────────
// Keeps vertices where dot(v - plane_pt, plane_n) <= 0

HIPGL_DEVICE int clip_poly(
    const float vin[][3], int nin,
    const float* plane_pt, const float* plane_n, float plane_d,
    float vout[][3])
{
    int nout = 0;
    for (int i = 0; i < nin && nout < MAX_CLIP_VERTS; i++) {
        int j = (i + 1) % nin;
        float di_v[3] = {vin[i][0]-plane_pt[0], vin[i][1]-plane_pt[1], vin[i][2]-plane_pt[2]};
        float dj_v[3] = {vin[j][0]-plane_pt[0], vin[j][1]-plane_pt[1], vin[j][2]-plane_pt[2]};
        float di = vec3_dot(di_v, plane_n) - plane_d;
        float dj = vec3_dot(dj_v, plane_n) - plane_d;
        if (di <= 0) {
            vout[nout][0] = vin[i][0]; vout[nout][1] = vin[i][1]; vout[nout][2] = vin[i][2];
            nout++;
        }
        if ((di <= 0) != (dj <= 0) && nout < MAX_CLIP_VERTS) {
            float t = di / (di - dj);
            vout[nout][0] = vin[i][0] + t*(vin[j][0]-vin[i][0]);
            vout[nout][1] = vin[i][1] + t*(vin[j][1]-vin[i][1]);
            vout[nout][2] = vin[i][2] + t*(vin[j][2]-vin[i][2]);
            nout++;
        }
    }
    return nout;
}

// Forward decl: defined with other double-precision narrowphase helpers below.
HIPGL_DEVICE int manifold_select_4_d(
    double cand_pos[][3], double cand_norm[][3], double cand_depth[],
    int ncand,
    double out_pos[][3], double out_norm[][3], double out_depth[]);

// ── Box-box: double-precision clipping ─────────────────────────────────────

HIPGL_DEVICE int clip_poly_d(
    const double vin[][3], int nin,
    const double* plane_pt, const double* plane_n, double plane_d,
    double vout[][3])
{
    int nout = 0;
    for (int i = 0; i < nin && nout < MAX_CLIP_VERTS; i++) {
        int j = (i + 1) % nin;
        double di_v[3], dj_v[3];
        vec3_sub_d(vin[i], plane_pt, di_v);
        vec3_sub_d(vin[j], plane_pt, dj_v);
        double di = vec3_dot_d(di_v, plane_n) - plane_d;
        double dj = vec3_dot_d(dj_v, plane_n) - plane_d;
        if (di <= 0) {
            vout[nout][0] = vin[i][0]; vout[nout][1] = vin[i][1]; vout[nout][2] = vin[i][2];
            nout++;
        }
        if ((di <= 0) != (dj <= 0) && nout < MAX_CLIP_VERTS) {
            double t = di / (di - dj);
            vout[nout][0] = vin[i][0] + t * (vin[j][0] - vin[i][0]);
            vout[nout][1] = vin[i][1] + t * (vin[j][1] - vin[i][1]);
            vout[nout][2] = vin[i][2] + t * (vin[j][2] - vin[i][2]);
            nout++;
        }
    }
    return nout;
}

#ifndef HP_BOXBOX_DIST_SCALE
#define HP_BOXBOX_DIST_SCALE 2
#endif
namespace mjbb {
constexpr double kMinVal = 1e-15;   // mjMINVAL
constexpr int kMaxPt = 24;          // box-box never exceeds 16+4+4 candidates
HIPGL_DEVICE inline void copy3(double* r, const double* a) { r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; }
HIPGL_DEVICE inline void zero3(double* r) { r[0] = r[1] = r[2] = 0.0; }
HIPGL_DEVICE inline void sub3(double* r, const double* a, const double* b) { r[0] = a[0]-b[0]; r[1] = a[1]-b[1]; r[2] = a[2]-b[2]; }
HIPGL_DEVICE inline void add3(double* r, const double* a, const double* b) { r[0] = a[0]+b[0]; r[1] = a[1]+b[1]; r[2] = a[2]+b[2]; }
HIPGL_DEVICE inline void addTo3(double* r, const double* a) { r[0] += a[0]; r[1] += a[1]; r[2] += a[2]; }
HIPGL_DEVICE inline void addToScl3(double* r, const double* a, double s) { r[0] += a[0]*s; r[1] += a[1]*s; r[2] += a[2]*s; }
HIPGL_DEVICE inline void scl3(double* r, const double* a, double s) { r[0] = a[0]*s; r[1] = a[1]*s; r[2] = a[2]*s; }
HIPGL_DEVICE inline double dot3(const double* a, const double* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
HIPGL_DEVICE inline void mulMatVec3(double* r, const double* M, const double* v) {
    double t[3] = {M[0]*v[0] + M[1]*v[1] + M[2]*v[2], M[3]*v[0] + M[4]*v[1] + M[5]*v[2], M[6]*v[0] + M[7]*v[1] + M[8]*v[2]};
    copy3(r, t);
}
HIPGL_DEVICE inline void mulMatTVec3(double* r, const double* M, const double* v) {
    double t[3] = {M[0]*v[0] + M[3]*v[1] + M[6]*v[2], M[1]*v[0] + M[4]*v[1] + M[7]*v[2], M[2]*v[0] + M[5]*v[1] + M[8]*v[2]};
    copy3(r, t);
}
HIPGL_DEVICE inline void mulMatTMat3(double* R, const double* A, const double* B) {   // R = A' * B
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        R[3*i+j] = A[i]*B[j] + A[3+i]*B[3+j] + A[6+i]*B[6+j];
}
HIPGL_DEVICE inline void mulMatMatT3(double* R, const double* A, const double* B) {   // R = A * B'
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++)
        R[3*i+j] = A[3*i]*B[3*j] + A[3*i+1]*B[3*j+1] + A[3*i+2]*B[3*j+2];
}
HIPGL_DEVICE inline void transpose3(double* R, const double* A) {
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) R[3*i+j] = A[3*j+i];
}
HIPGL_DEVICE inline double normalize3(double* v) {                                    // mju_normalize3
    double n = sqrt(dot3(v, v));
    if (n < kMinVal) { v[0] = 1; v[1] = 0; v[2] = 0; }
    else { double s = 1.0 / n; v[0] *= s; v[1] *= s; v[2] *= s; }
    return n;
}
// mju_outsideBox: 1 outside the inflated box, -1 inside the deflated box, 0 in between
HIPGL_DEVICE inline int outsideBox(const double* point, const double* pos, const double* mat,
                                   const double* size, double inflate) {
    double vec[3] = {point[0]-pos[0], point[1]-pos[1], point[2]-pos[2]};
    mulMatTVec3(vec, mat, vec);
    double big[3] = {size[0]*inflate, size[1]*inflate, size[2]*inflate};
    if (vec[0] > big[0] || vec[0] < -big[0] || vec[1] > big[1] || vec[1] < -big[1] ||
        vec[2] > big[2] || vec[2] < -big[2]) return 1;
    double sm[3] = {size[0]/inflate, size[1]/inflate, size[2]/inflate};
    if (vec[0] < sm[0] && vec[0] > -sm[0] && vec[1] < sm[1] && vec[1] > -sm[1] &&
        vec[2] < sm[2] && vec[2] > -sm[2]) return -1;
    return 0;
}
// rotmore / (i0,i1,i2,f0,f1,f2) for a face code 0..5 (MuJoCo repeats this block twice)
HIPGL_DEVICE inline void faceFrame(int q, double* rotmore, int& i0, int& i1, int& i2,
                                   double& f0, double& f1, double& f2) {
    for (int k = 0; k < 9; k++) rotmore[k] = 0;
    if (q == 0)      rotmore[2] = -1, rotmore[4] = +1, rotmore[6] = +1;
    else if (q == 1) rotmore[0] = +1, rotmore[5] = -1, rotmore[7] = +1;
    else if (q == 2) rotmore[0] = +1, rotmore[4] = +1, rotmore[8] = +1;
    else if (q == 3) rotmore[2] = +1, rotmore[4] = +1, rotmore[6] = -1;
    else if (q == 4) rotmore[0] = +1, rotmore[5] = +1, rotmore[7] = -1;
    else if (q == 5) rotmore[0] = -1, rotmore[4] = +1, rotmore[8] = -1;
    i0 = 0; i1 = 1; i2 = 2; f0 = f1 = f2 = 1;
    if (q == 0)      { i0 = 2; f0 = -1; i2 = 0; }
    else if (q == 1) { i1 = 2; f1 = -1; i2 = 1; }
    else if (q == 3) { i0 = 2; i2 = 0; f2 = -1; }
    else if (q == 4) { i1 = 2; i2 = 1; f2 = -1; }
    else if (q == 5) { f0 = -1; f2 = -1; }
}

// _boxbox. mat = row-major world-from-local rotation (columns are the box axes).
// Outputs MuJoCo contacts: pos, normal (geom1 -> geom2), dist (negative = penetration).
HIPGL_DEVICE int boxbox_raw(const double* pos1, const double* mat1, const double* size1,
                            const double* pos2, const double* mat2, const double* size2,
                            double margin, double cpos[][3], double cnorm[][3], double cdist[]) {
    double pos12[3], pos21[3], rot[9], rott[9], rotabs[9], rottabs[9], tmp1[3], tmp2[3], plen1[3], plen2[3];
    double rotmore[9], p[3], r[9], s[3], ss[3], lp[3], rt[9], points[kMaxPt][3], depth[kMaxPt],
           pts[6][3], ppts2[4][2], pu[4][3], axi[3][3];
    double linesu[4][6], lines[4][6], clnorm[3] = {0, 0, 0}, rnorm[3];
    double penetration, c1, c2, c3, a, b, c, d, lx, ly, hz, l, x, y, u, v, llx, lly, innorm, margin2;
    int i0, i1, i2; double f0, f1, f2;
    int i, j, q, code, q1, q2, clcorner, n, m, k;
    int cle1 = 0, cle2 = 0, in = 0, ax1, ax2, pax1, pax2, clface, nl, nf;
    for (i = 0; i < 6; i++) zero3(pts[i]);

    n = 0; code = -1; margin2 = margin * margin;
    sub3(tmp1, pos2, pos1); mulMatTVec3(pos21, mat1, tmp1);
    sub3(tmp1, pos1, pos2); mulMatTVec3(pos12, mat2, tmp1);
    mulMatTMat3(rot, mat1, mat2);
    transpose3(rott, rot);
    for (i = 0; i < 9; i++) rotabs[i] = fabs(rot[i]);
    for (i = 0; i < 9; i++) rottabs[i] = fabs(rott[i]);
    mulMatVec3(plen2, rotabs, size2);
    mulMatTVec3(plen1, rotabs, size1);

    for (i = 0, penetration = margin; i < 3; i++) penetration += size1[i] * 3 + size2[i] * 3;
    for (i = 0; i < 3; i++) {
        c1 = -fabs(pos21[i]) + size1[i] + plen2[i];
        c2 = -fabs(pos12[i]) + size2[i] + plen1[i];
        if (c1 < -margin || c2 < -margin) return 0;
        if (c1 < penetration) { penetration = c1; code = i + 3 * (pos21[i] < 0) + 0; }
        if (c2 < penetration) { penetration = c2; code = i + 3 * (pos12[i] < 0) + 6; }
    }
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            zero3(tmp2);
            if (i == 0)      { tmp2[1] = -rott[3*j+2]; tmp2[2] = +rott[3*j+1]; }
            else if (i == 1) { tmp2[0] = +rott[3*j+2]; tmp2[2] = -rott[3*j+0]; }
            else if (i == 2) { tmp2[0] = -rott[3*j+1]; tmp2[1] = +rott[3*j+0]; }
            c1 = normalize3(tmp2);
            if (c1 < kMinVal) continue;
            c2 = dot3(pos21, tmp2);
            c3 = 0;
            for (k = 0; k < 3; k++) if (k != i) c3 += size1[k] * fabs(tmp2[k]);
            for (k = 0; k < 3; k++) if (k != j) c3 += size2[k] * rotabs[3*i + 3 - k - j] / c1;
            c3 -= fabs(c2);
            if (c3 < -margin) return 0;
            if (c3 < penetration * (1 - 1e-12)) {
                penetration = c3;
                for (k = cle1 = 0; k < 3; k++)
                    if (k != i) if ((tmp2[k] > 0) ^ (c2 < 0)) cle1 += 1 << k;
                for (k = cle2 = 0; k < 3; k++)
                    if (k != j) if ((rot[3*i + 3 - k - j] > 0) ^ (c2 < 0) ^ ((k - j + 3) % 3 == 1)) cle2 += 1 << k;
                code = 12 + i * 3 + j;
                copy3(clnorm, tmp2);
                in = c2 < 0;
            }
        }
    }
    if (code == -1) return 0;

    if (code < 12) {
        // ---------------------------------------------------------- face path
        q1 = code % 6; q2 = code / 6;
        faceFrame(q1, rotmore, i0, i1, i2, f0, f1, f2);
        auto rotaxis = [&](double* vr, const double* vi) { vr[0] = vi[i0]*f0; vr[1] = vi[i1]*f1; vr[2] = vi[i2]*f2; };
        auto rotmatx = [&](double* mr, const double* mi) { scl3(mr+0, mi+i0*3, f0); scl3(mr+3, mi+i1*3, f1); scl3(mr+6, mi+i2*3, f2); };
        if (q2) { mulMatMatT3(r, rotmore, rot); rotaxis(p, pos12); rotaxis(tmp1, size2); copy3(s, size1); }
        else    { rotmatx(r, rot);              rotaxis(p, pos21); rotaxis(tmp1, size1); copy3(s, size2); }
        transpose3(rt, r);
        for (i = 0; i < 3; i++) ss[i] = fabs(tmp1[i]);
        lx = ss[0]; ly = ss[1]; hz = ss[2];
        p[2] -= hz;
        copy3(lp, p);
        for (clcorner = 0, i = 0; i < 3; i++) if (r[6+i] < 0) clcorner += 1 << i;
        addToScl3(lp, rt + 0, s[0] * ((clcorner & 1) ? 1 : -1));
        addToScl3(lp, rt + 3, s[1] * ((clcorner & 2) ? 1 : -1));
        addToScl3(lp, rt + 6, s[2] * ((clcorner & 4) ? 1 : -1));
        m = k = 0;
        copy3(pts[m++], lp);
        for (i = 0; i < 3; i++)
            if (fabs(r[6+i]) < 0.5) scl3(pts[m++], rt + 3*i, s[i] * ((clcorner & (1 << i)) ? -2 : 2));
        add3(pts[3], pts[0], pts[1]);
        add3(pts[4], pts[0], pts[2]);
        add3(pts[5], pts[3], pts[2]);
        if (m > 1) { copy3(lines[k] + 0, pts[0]); copy3(lines[k++] + 3, pts[1]); }
        if (m > 2) {
            copy3(lines[k] + 0, pts[0]); copy3(lines[k++] + 3, pts[2]);
            copy3(lines[k] + 0, pts[3]); copy3(lines[k++] + 3, pts[2]);
            copy3(lines[k] + 0, pts[4]); copy3(lines[k++] + 3, pts[1]);
        }
        for (i = 0; i < k; i++) {
            for (q = 0; q < 2; q++) {
                a = lines[i][0+q]; b = lines[i][3+q]; c = lines[i][1-q]; d = lines[i][4-q];
                if (fabs(b) > kMinVal) {
                    for (j = -1; j <= 1; j += 2) {
                        l = ss[q] * j;
                        c1 = (l - a) * (1 / b);
                        if (c1 < 0 || c1 > 1) continue;
                        c2 = c + d * c1;
                        if (fabs(c2) > ss[1-q]) continue;
                        if (n < kMaxPt) { copy3(points[n], lines[i]); addToScl3(points[n++], lines[i] + 3, c1); }
                    }
                }
            }
        }
        a = pts[1][0]; b = pts[2][0]; c = pts[1][1]; d = pts[2][1];
        c1 = a * d - b * c;
        if (m > 2) {
            for (i = 0; i < 4; i++) {
                llx = i / 2 ? lx : -lx;
                lly = i % 2 ? ly : -ly;
                x = llx - pts[0][0]; y = lly - pts[0][1];
                u = (x * d - y * b) * (1 / c1);
                v = (y * a - x * c) * (1 / c1);
                if (u <= 0 || v <= 0 || u >= 1 || v >= 1) continue;
                if (n < kMaxPt) {
                    points[n][0] = llx; points[n][1] = lly;
                    points[n][2] = (pts[0][2] + u * pts[1][2] + v * pts[2][2]);
                    n++;
                }
            }
        }
        for (i = 0; i < (1 << (m - 1)); i++) {
            copy3(tmp1, pts[i == 0 ? 0 : i + 2]);
            if (i) if (tmp1[0] <= -lx || tmp1[0] >= lx) continue;
            if (i) if (tmp1[1] <= -ly || tmp1[1] >= ly) continue;
            if (n < kMaxPt) copy3(points[n++], tmp1);
        }
        m = n; n = 0;
        for (i = 0; i < m; i++) {
            if (points[i][2] > margin) continue;
            if (n != i) copy3(points[n], points[i]);
            depth[n] = points[n][2];
            points[n][2] *= 0.5;
            n++;
        }
        mulMatMatT3(r, q2 ? mat2 : mat1, rotmore);
        copy3(p, q2 ? pos2 : pos1);
        tmp2[0] = (q2 ? -1 : 1) * r[2];
        tmp2[1] = (q2 ? -1 : 1) * r[5];
        tmp2[2] = (q2 ? -1 : 1) * r[8];
        for (i = 0; i < n; i++) {
            cdist[i] = HP_BOXBOX_DIST_SCALE * points[i][2];  // MuJoCo >= 3.4: 2; <= 3.3.x: 1
            points[i][2] += hz;
            double w[3]; mulMatVec3(w, r, points[i]);
            add3(cpos[i], w, p);
            copy3(cnorm[i], tmp2);
        }
        return n;
    }

    // ------------------------------------------------------------- edge-edge
    code -= 12;
    q1 = code / 3; q2 = code % 3;
    ax1 = ax2 = pax1 = pax2 = 0;
    if (q2 == 0) ax1 = 1, ax2 = 2;
    if (q2 == 1) ax1 = 0, ax2 = 2;
    if (q2 == 2) ax1 = 1, ax2 = 0;
    if (q1 == 0) pax1 = 1, pax2 = 2;
    if (q1 == 1) pax1 = 0, pax2 = 2;
    if (q1 == 2) pax1 = 1, pax2 = 0;
    if (rotabs[3*q1 + ax1] < rotabs[3*q1 + ax2]) { ax1 = ax2; ax2 = 3 - q2 - ax1; }
    if (rottabs[3*q2 + pax1] < rottabs[3*q2 + pax2]) { pax1 = pax2; pax2 = 3 - q1 - pax1; }
    if (cle1 & (1 << pax2)) clface = pax2; else clface = pax2 + 3;

    faceFrame(clface, rotmore, i0, i1, i2, f0, f1, f2);
    auto rotaxis = [&](double* vr, const double* vi) { vr[0] = vi[i0]*f0; vr[1] = vi[i1]*f1; vr[2] = vi[i2]*f2; };
    auto rotmatx = [&](double* mr, const double* mi) { scl3(mr+0, mi+i0*3, f0); scl3(mr+3, mi+i1*3, f1); scl3(mr+6, mi+i2*3, f2); };
    rotaxis(p, pos21);
    rotaxis(rnorm, clnorm);
    rotmatx(r, rot);
    mulMatTVec3(tmp1, rotmore, size1);
    for (i = 0; i < 3; i++) s[i] = fabs(tmp1[i]);
    transpose3(rt, r);
    lx = s[0]; ly = s[1]; hz = s[2];
    p[2] -= hz;

    n = 0;
    copy3(points[n], p);
    addToScl3(points[n], rt + 3*ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? 1 : -1));
    addToScl3(points[n], rt + 3*ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
    copy3(points[n+1], points[n]);
    addToScl3(points[n], rt + 3*q2, size2[q2]);
    n = 1;
    addToScl3(points[n], rt + 3*q2, -size2[q2]);
    n = 2;
    copy3(points[n], p);
    addToScl3(points[n], rt + 3*ax1, size2[ax1] * ((cle2 & (1 << ax1)) ? -1 : 1));
    addToScl3(points[n], rt + 3*ax2, size2[ax2] * ((cle2 & (1 << ax2)) ? 1 : -1));
    copy3(points[n+1], points[n]);
    addToScl3(points[n], rt + 3*q2, size2[q2]);
    n = 3;
    addToScl3(points[n], rt + 3*q2, -size2[q2]);
    n = 4;

    copy3(axi[0], points[0]);
    sub3(axi[1], points[1], points[0]);
    sub3(axi[2], points[2], points[0]);
    if (fabs(rnorm[2]) < kMinVal) return 0;
    innorm = (1 / rnorm[2]) * (in ? -1 : 1);
    for (i = 0; i < 4; i++) {
        c1 = -points[i][2] * (1 / rnorm[2]);
        copy3(pu[i], points[i]);
        addToScl3(points[i], rnorm, c1);
        ppts2[i][0] = points[i][0];
        ppts2[i][1] = points[i][1];
    }
    copy3(pts[0], points[0]);
    sub3(pts[1], points[1], points[0]);
    sub3(pts[2], points[2], points[0]);

    m = 3; k = 0; n = 0;
    if (m > 1) {
        copy3(lines[k] + 0, pts[0]);  copy3(lines[k] + 3, pts[1]);
        copy3(linesu[k] + 0, axi[0]); copy3(linesu[k++] + 3, axi[1]);
    }
    if (m > 2) {
        copy3(lines[k] + 0, pts[0]);  copy3(lines[k] + 3, pts[2]);
        copy3(linesu[k] + 0, axi[0]); copy3(linesu[k++] + 3, axi[2]);
        add3(lines[k] + 0, pts[0], pts[1]);  copy3(lines[k] + 3, pts[2]);
        add3(linesu[k] + 0, axi[0], axi[1]); copy3(linesu[k++] + 3, axi[2]);
        add3(lines[k] + 0, pts[0], pts[2]);  copy3(lines[k] + 3, pts[1]);
        add3(linesu[k] + 0, axi[0], axi[2]); copy3(linesu[k++] + 3, axi[1]);
    }
    for (i = 0; i < k; i++) {
        for (q = 0; q < 2; q++) {
            a = lines[i][0+q]; b = lines[i][3+q]; c = lines[i][1-q]; d = lines[i][4-q];
            if (fabs(b) > kMinVal) {
                for (j = -1; j <= 1; j += 2) {
                    if (n < kMaxPt) {
                        l = s[q] * j;
                        c1 = (l - a) * (1 / b);
                        if (c1 < 0 || c1 > 1) continue;
                        c2 = c + d * c1;
                        if (fabs(c2) > s[1-q]) continue;
                        if ((linesu[i][2] + linesu[i][5] * c1) * innorm > margin) continue;
                        scl3(points[n], linesu[i], 0.5);
                        addToScl3(points[n], linesu[i] + 3, 0.5 * c1);
                        points[n][0+q] += 0.5 * l;
                        points[n][1-q] += 0.5 * c2;
                        depth[n] = points[n][2] * innorm * 2;
                        n++;
                    }
                }
            }
        }
    }
    nl = n;
    a = pts[1][0]; b = pts[2][0]; c = pts[1][1]; d = pts[2][1];
    c1 = a * d - b * c;
    for (i = 0; i < 4; i++) {
        if (n < kMaxPt) {
            llx = i / 2 ? lx : -lx;
            lly = i % 2 ? ly : -ly;
            x = llx - pts[0][0]; y = lly - pts[0][1];
            u = (x * d - y * b) * (1 / c1);
            v = (y * a - x * c) * (1 / c1);
            if (nl == 0) { if ((u < 0 || u > 1) && (v < 0 || v > 1)) continue; }
            else         { if ((u < 0 || u > 1 || v < 0 || v > 1)) continue; }
            if (u < 0) u = 0;
            if (u > 1) u = 1;
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            scl3(tmp1, pu[0], 1 - u - v);
            addToScl3(tmp1, pu[1], u);
            addToScl3(tmp1, pu[2], v);
            points[n][0] = llx; points[n][1] = lly; points[n][2] = 0;
            sub3(tmp2, points[n], tmp1);
            c1 = dot3(tmp2, tmp2);
            if (tmp1[2] > 0) if (c1 > margin2) continue;
            addTo3(points[n], tmp1);
            scl3(points[n], points[n], 0.5);
            depth[n] = sqrt(c1) * (tmp1[2] < 0 ? -1 : 1);
            n++;
        }
    }
    nf = n;
    for (i = 0; i < 4; i++) {
        if (n < kMaxPt) {
            x = ppts2[i][0]; y = ppts2[i][1];
            if (nl == 0) {
                if (nf == 0) {
                } else {
                    if (x < -lx || x > lx) if (y < -ly || y > ly) continue;
                }
            } else {
                if (x < -lx || x > lx || y < -ly || y > ly) continue;
            }
            for (c1 = 0, j = 0; j < 2; j++)
                if (ppts2[i][j] < -s[j])      c1 += (ppts2[i][j] + s[j]) * (ppts2[i][j] + s[j]);
                else if (ppts2[i][j] > s[j])  c1 += (ppts2[i][j] - s[j]) * (ppts2[i][j] - s[j]);
            c1 += pu[i][2] * innorm * pu[i][2] * innorm;
            if (pu[i][2] > 0) if (c1 > margin2) continue;
            tmp1[0] = ppts2[i][0] * 0.5; tmp1[1] = ppts2[i][1] * 0.5; tmp1[2] = 0;
            for (j = 0; j < 2; j++) {
                if (ppts2[i][j] < -s[j])     tmp1[j] = -s[j] * 0.5;
                else if (ppts2[i][j] > s[j]) tmp1[j] = +s[j] * 0.5;
            }
            addToScl3(tmp1, pu[i], 0.5);
            copy3(points[n], tmp1);
            depth[n] = sqrt(c1) * (pu[i][2] < 0 ? -1 : 1);
            n++;
        }
    }
    mulMatMatT3(r, mat1, rotmore);
    mulMatVec3(tmp1, r, rnorm);
    double nrm[3]; scl3(nrm, tmp1, in ? -1 : 1);
    for (i = 0; i < n; i++) {
        cdist[i] = depth[i];
        points[i][2] += hz;
        mulMatVec3(tmp2, r, points[i]);
        add3(cpos[i], tmp2, pos1);
        copy3(cnorm[i], nrm);
    }
    return n;
}
}  // namespace mjbb

// mjc_BoxBox wrapper (outside-box filter + exact-duplicate removal), in
// The GPU simulator's calling convention: quaternions in, depth = -dist out, normal
// box1 -> box2. MuJoCo keeps every contact; if a pair ever exceeds
// MAX_PAIR_CONTACTS we keep the deepest ones.
HIPGL_DEVICE int collide_box_box_full_d(
    const double* p1, const double* q1, const double* s1,
    const double* p2, const double* q2, const double* s2,
    double out_pos[][3], double out_norm[][3], double out_depth[],
    double margin = 0.0)
{
    double mat1[9], mat2[9];
    for (int c = 0; c < 3; c++) {
        double local[3] = {0, 0, 0}, ax[3];
        local[c] = 1.0;
        quat_rotate_vec_d(q1, local, ax);
        mat1[c] = ax[0]; mat1[3 + c] = ax[1]; mat1[6 + c] = ax[2];
        quat_rotate_vec_d(q2, local, ax);
        mat2[c] = ax[0]; mat2[3 + c] = ax[1]; mat2[6 + c] = ax[2];
    }
    double cpos[mjbb::kMaxPt][3], cnorm[mjbb::kMaxPt][3], cdist[mjbb::kMaxPt];
    int num = mjbb::boxbox_raw(p1, mat1, s1, p2, mat2, s2, margin, cpos, cnorm, cdist);

    int dupe[mjbb::kMaxPt];
    double sz1[3] = {s1[0] + margin, s1[1] + margin, s1[2] + margin};
    double sz2[3] = {s2[0] + margin, s2[1] + margin, s2[2] + margin};
    for (int i = 0; i < num; i++) {
        dupe[i] = 0;
        int out1 = mjbb::outsideBox(cpos[i], p1, mat1, sz1, 1.01);
        int out2 = mjbb::outsideBox(cpos[i], p2, mat2, sz2, 1.01);
        if ((out1 == 1 && out2 != -1) || (out2 == 1 && out1 != -1)) dupe[i] = -1;
    }
    for (int i = 0; i < num - 1; i++) {
        if (dupe[i] == -1) continue;
        for (int j = i + 1; j < num; j++) {
            if (dupe[j] == -1) continue;
            if (cpos[i][0] == cpos[j][0] && cpos[i][1] == cpos[j][1] && cpos[i][2] == cpos[j][2]) {
                dupe[i] = -1;
                break;
            }
        }
    }
    int nout = 0;
    for (int j = 0; j < num; j++) {
        if (dupe[j] != 0) continue;
        if (nout < MAX_PAIR_CONTACTS) {
            mjbb::copy3(out_pos[nout], cpos[j]); mjbb::copy3(out_norm[nout], cnorm[j]);
            out_depth[nout] = -cdist[j];
            nout++;
        } else {
            int w = 0;   // over capacity: replace the shallowest kept contact if this one is deeper
            for (int t = 1; t < MAX_PAIR_CONTACTS; t++) if (out_depth[t] < out_depth[w]) w = t;
            if (-cdist[j] > out_depth[w]) {
                mjbb::copy3(out_pos[w], cpos[j]); mjbb::copy3(out_norm[w], cnorm[j]);
                out_depth[w] = -cdist[j];
            }
        }
    }
    return nout;
}

// ── Plane collision functions (multi-contact) ───────────────────────────────

HIPGL_DEVICE bool collide_sphere_plane(
    const float* sphere_pos, float radius,
    const float* plane_pos, const float* plane_quat,
    float* contact_pos, float* contact_normal, float* depth)
{
    float local_up[3] = {0, 0, 1};
    float normal[3];
    quat_rotate_vec(plane_quat, local_up, normal);

    float diff[3];
    vec3_sub(sphere_pos, plane_pos, diff);
    float dist = vec3_dot(diff, normal) - radius;
    if (dist > 0) return false;

    *depth = -dist;
    contact_normal[0] = -normal[0];
    contact_normal[1] = -normal[1];
    contact_normal[2] = -normal[2];
    // MuJoCo convention: contact at midpoint between surfaces
    float half_depth = *depth * 0.5f;
    contact_pos[0] = sphere_pos[0] - normal[0] * (radius - half_depth);
    contact_pos[1] = sphere_pos[1] - normal[1] * (radius - half_depth);
    contact_pos[2] = sphere_pos[2] - normal[2] * (radius - half_depth);
    return true;
}

HIPGL_DEVICE int collide_box_plane_multi(
    const float* box_pos, const float* box_quat, const float* box_size,
    const float* plane_pos, const float* plane_quat,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    float local_up[3] = {0, 0, 1};
    float normal[3];
    quat_rotate_vec(plane_quat, local_up, normal);

    float cn[3] = {-normal[0], -normal[1], -normal[2]};

    float corner_signs[8][3] = {
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
        {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
    };

    float corner_depths[8];
    float corner_pos[8][3];
    int nbelow = 0;

    for (int c = 0; c < 8; c++) {
        float local[3] = {
            corner_signs[c][0]*box_size[0],
            corner_signs[c][1]*box_size[1],
            corner_signs[c][2]*box_size[2]
        };
        float wc[3];
        quat_rotate_vec(box_quat, local, wc);
        corner_pos[c][0] = box_pos[0] + wc[0];
        corner_pos[c][1] = box_pos[1] + wc[1];
        corner_pos[c][2] = box_pos[2] + wc[2];

        float diff[3];
        vec3_sub(corner_pos[c], plane_pos, diff);
        float d = vec3_dot(diff, normal);
        corner_depths[c] = -d;
        if (d <= 0) nbelow++;
    }

    if (nbelow == 0) return 0;

    // Collect penetrating corners as candidates, then select 4 well-spaced
    float cand_pos[8][3], cand_norm[8][3], cand_depth[8];
    int ncand = 0;
    for (int c = 0; c < 8; c++) {
        if (corner_depths[c] < 0) continue;
        float half_d = corner_depths[c] * 0.5f;
        cand_pos[ncand][0] = corner_pos[c][0] + normal[0] * half_d;
        cand_pos[ncand][1] = corner_pos[c][1] + normal[1] * half_d;
        cand_pos[ncand][2] = corner_pos[c][2] + normal[2] * half_d;
        cand_norm[ncand][0] = cn[0]; cand_norm[ncand][1] = cn[1]; cand_norm[ncand][2] = cn[2];
        cand_depth[ncand] = corner_depths[c];
        ncand++;
    }

    return manifold_select_4(cand_pos, cand_norm, cand_depth, ncand, out_pos, out_norm, out_depth);
}

HIPGL_DEVICE int collide_capsule_plane_multi(
    const float* cap_pos, const float* cap_quat, float radius, float half_len,
    const float* plane_pos, const float* plane_quat,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    float local_up[3] = {0, 0, 1};
    float normal[3];
    quat_rotate_vec(plane_quat, local_up, normal);

    float cn[3] = {-normal[0], -normal[1], -normal[2]};

    float cap_axis_local[3] = {0, 0, 1};
    float cap_axis[3];
    quat_rotate_vec(cap_quat, cap_axis_local, cap_axis);

    float ep[2][3];
    for (int k = 0; k < 3; k++) {
        ep[0][k] = cap_pos[k] + cap_axis[k] * half_len;
        ep[1][k] = cap_pos[k] - cap_axis[k] * half_len;
    }

    int ncontacts = 0;
    for (int i = 0; i < 2 && ncontacts < MAX_PAIR_CONTACTS; i++) {
        float diff[3];
        vec3_sub(ep[i], plane_pos, diff);
        float d = vec3_dot(diff, normal) - radius;
        if (d <= 0) {
            float dep = -d;
            out_depth[ncontacts] = dep;
            // MuJoCo convention: contact at midpoint between surfaces
            float half_dep = dep * 0.5f;
            out_pos[ncontacts][0] = ep[i][0] - normal[0] * (radius - half_dep);
            out_pos[ncontacts][1] = ep[i][1] - normal[1] * (radius - half_dep);
            out_pos[ncontacts][2] = ep[i][2] - normal[2] * (radius - half_dep);
            out_norm[ncontacts][0] = cn[0];
            out_norm[ncontacts][1] = cn[1];
            out_norm[ncontacts][2] = cn[2];
            ncontacts++;
        }
    }
    return ncontacts;
}

// ── Mesh-plane collision (hull vertices vs infinite plane) ───────────────────

HIPGL_DEVICE int collide_mesh_plane(
    const float* mesh_xpos, const float* mesh_xquat,
    const float* plane_pos, const float* plane_quat,
    int hull_vert_adr, int hull_nvert,
    const float* hull_verts,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    float local_up[3] = {0, 0, 1};
    float normal[3];
    quat_rotate_vec(plane_quat, local_up, normal);

    float cand_pos[MAX_CANDIDATE_CONTACTS][3];
    float cand_norm[MAX_CANDIDATE_CONTACTS][3];
    float cand_depth[MAX_CANDIDATE_CONTACTS];
    int ncand = 0;

    for (int vi = 0; vi < hull_nvert && vi < MAX_CONVEX_HULL_VERTS; vi++) {
        float lv[3] = {hull_verts[(hull_vert_adr + vi) * 3],
                        hull_verts[(hull_vert_adr + vi) * 3 + 1],
                        hull_verts[(hull_vert_adr + vi) * 3 + 2]};
        float wv[3];
        quat_rotate_vec(mesh_xquat, lv, wv);
        wv[0] += mesh_xpos[0]; wv[1] += mesh_xpos[1]; wv[2] += mesh_xpos[2];

        float diff[3];
        vec3_sub(wv, plane_pos, diff);
        float dist = vec3_dot(diff, normal);

        if (dist < 0.0f && ncand < MAX_CANDIDATE_CONTACTS) {
            float depth = -dist;
            float cp[3] = {wv[0] + normal[0] * depth * 0.5f,
                           wv[1] + normal[1] * depth * 0.5f,
                           wv[2] + normal[2] * depth * 0.5f};
            cand_pos[ncand][0] = cp[0]; cand_pos[ncand][1] = cp[1]; cand_pos[ncand][2] = cp[2];
            cand_norm[ncand][0] = -normal[0]; cand_norm[ncand][1] = -normal[1]; cand_norm[ncand][2] = -normal[2];
            cand_depth[ncand] = depth;
            ncand++;
        }
    }

    return manifold_select_4(cand_pos, cand_norm, cand_depth, ncand, out_pos, out_norm, out_depth);
}

// ── Mesh-box collision (complete convex SAT and extremal-feature contact) ──
// Includes crossing edges and directional containment, even when neither
// solid contains a vertex of the other. Nonunique support patches use an
// area centroid or segment midpoint, independent of hull triangulation.
// The legacy vertex tests remain only for separated positive-margin pairs
// that pass the complete SAT proximity filter.

HIPGL_DEVICE int collide_mesh_box(
    const float* mesh_xpos, const float* mesh_xquat,
    const float* box_pos, const float* box_quat, const float* box_size,
    int hull_vert_adr, int hull_nvert,
    const float* hull_verts,
    int hull_face_adr, int hull_nface,
    const float* hull_face_normals,
    const int* hull_faces,
    float out_pos[][3], float out_norm[][3], float out_depth[],
    float margin = 0.0f)
{
    int complete = ::hipgl::physics::mesh_box_detail::contact(
        mesh_xpos, mesh_xquat, box_pos, box_quat, box_size,
        hull_vert_adr, hull_nvert, hull_verts, hull_face_adr, hull_nface,
        hull_face_normals, hull_faces, out_pos, out_norm, out_depth, margin);
    if (complete >= 0) return complete;

    // Legacy positive-margin proximity behavior only. At zero margin, the
    // complete SAT/support-feature path above handles every contact. Preserve
    // the existing per-pair normal correction in this compatibility fallback.
    float ex[3]={1,0,0}, ey[3]={0,1,0}, ez[3]={0,0,1};
    float ax[3][3];
    quat_rotate_vec(box_quat, ex, ax[0]);
    quat_rotate_vec(box_quat, ey, ax[1]);
    quat_rotate_vec(box_quat, ez, ax[2]);

    float cand_pos[MAX_CANDIDATE_CONTACTS][3];
    float cand_norm[MAX_CANDIDATE_CONTACTS][3];
    float cand_depth[MAX_CANDIDATE_CONTACTS];
    int ncand = 0;

    // Direction 1: mesh hull vertices penetrating box
    for (int vi = 0; vi < hull_nvert && vi < MAX_CONVEX_HULL_VERTS; vi++) {
        float lv[3] = {hull_verts[(hull_vert_adr + vi)*3],
                        hull_verts[(hull_vert_adr + vi)*3+1],
                        hull_verts[(hull_vert_adr + vi)*3+2]};
        float wv[3];
        quat_rotate_vec(mesh_xquat, lv, wv);
        wv[0]+=mesh_xpos[0]; wv[1]+=mesh_xpos[1]; wv[2]+=mesh_xpos[2];

        float diff[3]; vec3_sub(wv, box_pos, diff);
        float loc[3]={vec3_dot(diff,ax[0]), vec3_dot(diff,ax[1]), vec3_dot(diff,ax[2])};

        float pen[6]={box_size[0]-loc[0]+margin, box_size[0]+loc[0]+margin,
                      box_size[1]-loc[1]+margin, box_size[1]+loc[1]+margin,
                      box_size[2]-loc[2]+margin, box_size[2]+loc[2]+margin};
        float min_pen=pen[0]; int min_f=0;
        for(int f=1;f<6;f++){if(pen[f]<min_pen){min_pen=pen[f];min_f=f;}}
        if(min_pen<=0.0f) continue;
        if(ncand>=MAX_CANDIDATE_CONTACTS) continue;

        int axis_idx=min_f/2;
        float sign=(min_f%2==0)?1.0f:-1.0f;
        float nrm[3]={sign*ax[axis_idx][0], sign*ax[axis_idx][1], sign*ax[axis_idx][2]};
        float half=min_pen*0.5f;

        cand_pos[ncand][0]=wv[0]+nrm[0]*half;
        cand_pos[ncand][1]=wv[1]+nrm[1]*half;
        cand_pos[ncand][2]=wv[2]+nrm[2]*half;
        // Negated for the same reason as collide_mesh_plane: `nrm` is the direction
        // the mesh vertex must travel to escape the box, and the solver expects the
        // opposite. Un-negated, mesh-box contacts were attractive and a mesh dropped
        // on a box sank straight through it.
        cand_norm[ncand][0]=-nrm[0]; cand_norm[ncand][1]=-nrm[1]; cand_norm[ncand][2]=-nrm[2];
        cand_depth[ncand]=min_pen;
        ncand++;
    }

    if (hull_nface > 0 && hull_faces && hull_face_normals) {
        // Inverse mesh transform for world→mesh-local conversion
        float mesh_iquat[4] = {mesh_xquat[0], -mesh_xquat[1], -mesh_xquat[2], -mesh_xquat[3]};

        // Separating face for the PAIR, resolved lazily on the first penetrating
        // corner and then shared by all of them (see the SAT note below).
        int pair_face = -1;
        if (hull_nface <= MAX_CONVEX_HULL_FACES) {
            float best_pair_pen = 1e20f;
            for (int fi = 0; fi < hull_nface; fi++) {
                float fn[3] = {hull_face_normals[(hull_face_adr + fi)*3],
                               hull_face_normals[(hull_face_adr + fi)*3+1],
                               hull_face_normals[(hull_face_adr + fi)*3+2]};
                int v0i = hull_faces[(hull_face_adr + fi)*3];
                float fv[3] = {hull_verts[(hull_vert_adr + v0i)*3],
                               hull_verts[(hull_vert_adr + v0i)*3+1],
                               hull_verts[(hull_vert_adr + v0i)*3+2]};
                float deepest = -1e20f;
                for (int cj = 0; cj < 8; cj++) {
                    float jx = (cj & 1) ? box_size[0] : -box_size[0];
                    float jy = (cj & 2) ? box_size[1] : -box_size[1];
                    float jz = (cj & 4) ? box_size[2] : -box_size[2];
                    float cw[3] = {
                        box_pos[0] + jx*ax[0][0] + jy*ax[1][0] + jz*ax[2][0],
                        box_pos[1] + jx*ax[0][1] + jy*ax[1][1] + jz*ax[2][1],
                        box_pos[2] + jx*ax[0][2] + jy*ax[1][2] + jz*ax[2][2]};
                    float rl[3] = {cw[0]-mesh_xpos[0], cw[1]-mesh_xpos[1],
                                   cw[2]-mesh_xpos[2]};
                    float cl[3];
                    quat_rotate_vec(mesh_iquat, rl, cl);
                    float df[3] = {cl[0]-fv[0], cl[1]-fv[1], cl[2]-fv[2]};
                    float p = margin - vec3_dot(df, fn);
                    if (p > deepest) deepest = p;
                }
                if (deepest > 0.0f && deepest < best_pair_pen) {
                    best_pair_pen = deepest; pair_face = fi;
                }
            }
        }

        for (int ci = 0; ci < 8; ci++) {
            // Generate box corner in world space
            float sx = (ci & 1) ? box_size[0] : -box_size[0];
            float sy = (ci & 2) ? box_size[1] : -box_size[1];
            float sz = (ci & 4) ? box_size[2] : -box_size[2];
            float corner_w[3] = {
                box_pos[0] + sx*ax[0][0] + sy*ax[1][0] + sz*ax[2][0],
                box_pos[1] + sx*ax[0][1] + sy*ax[1][1] + sz*ax[2][1],
                box_pos[2] + sx*ax[0][2] + sy*ax[1][2] + sz*ax[2][2]
            };

            // Transform corner into mesh local frame
            float rel[3] = {corner_w[0]-mesh_xpos[0], corner_w[1]-mesh_xpos[1], corner_w[2]-mesh_xpos[2]};
            float corner_local[3];
            quat_rotate_vec(mesh_iquat, rel, corner_local);

            // Test against all hull face planes.
            //
            // A point is inside a convex hull only if it is behind EVERY face plane, so
            // this test is sound only when every plane is checked. Truncating the loop
            // at MAX_CONVEX_HULL_FACES silently inverts its meaning: with 512 of
            // robot0_link1's 12516 hull faces tested (4%), a point 0.73 m away passed as
            // "inside" and produced a phantom contact at 0.298 m depth against the table,
            // which slammed 4.16 rad/s into the arm on the first step. MuJoCo reports
            // zero contacts for that pair. Bail out rather than answer from a subset.
            float max_signed_dist = -1e20f;
            int best_face = -1;
            bool inside = true;
            if (hull_nface > MAX_CONVEX_HULL_FACES) continue;
            for (int fi = 0; fi < hull_nface; fi++) {
                // Face normal (in mesh local frame)
                float fn[3] = {hull_face_normals[(hull_face_adr + fi)*3],
                               hull_face_normals[(hull_face_adr + fi)*3+1],
                               hull_face_normals[(hull_face_adr + fi)*3+2]};
                // Reference vertex on the face (first vertex of the triangle)
                int v0_idx = hull_faces[(hull_face_adr + fi)*3];
                float fv[3] = {hull_verts[(hull_vert_adr + v0_idx)*3],
                               hull_verts[(hull_vert_adr + v0_idx)*3+1],
                               hull_verts[(hull_vert_adr + v0_idx)*3+2]};
                // Signed distance: positive = outside hull
                float diff[3] = {corner_local[0]-fv[0], corner_local[1]-fv[1], corner_local[2]-fv[2]};
                float sd = vec3_dot(diff, fn);
                if (sd > margin) { inside = false; break; }
                if (sd > max_signed_dist) { max_signed_dist = sd; best_face = fi; }
            }

            if (!inside || best_face < 0) continue;
            float pen_depth;
            if (pair_face >= 0) {
                best_face = pair_face;
                float fn[3] = {hull_face_normals[(hull_face_adr + pair_face)*3],
                               hull_face_normals[(hull_face_adr + pair_face)*3+1],
                               hull_face_normals[(hull_face_adr + pair_face)*3+2]};
                int v0i = hull_faces[(hull_face_adr + pair_face)*3];
                float fv[3] = {hull_verts[(hull_vert_adr + v0i)*3],
                               hull_verts[(hull_vert_adr + v0i)*3+1],
                               hull_verts[(hull_vert_adr + v0i)*3+2]};
                float df[3] = {corner_local[0]-fv[0], corner_local[1]-fv[1],
                               corner_local[2]-fv[2]};
                pen_depth = margin - vec3_dot(df, fn);
            } else {
                pen_depth = margin - max_signed_dist;
            }
            if (pen_depth <= 0.0f) continue;
            if (ncand >= MAX_CANDIDATE_CONTACTS) continue;

            // Contact normal = outward face normal of the mesh hull (world frame)
            // Convention: same as direction 1 — normal points outward from the
            // penetrated surface.  Direction 1: box outward normal pushes mesh
            // vertex out of box.  Direction 2: mesh hull outward normal pushes
            // box corner out of mesh.  Both point outward from the shape that
            // was penetrated, matching collide_mesh_plane convention.
            float ln[3] = {hull_face_normals[(hull_face_adr + best_face)*3],
                           hull_face_normals[(hull_face_adr + best_face)*3+1],
                           hull_face_normals[(hull_face_adr + best_face)*3+2]};
            float wn[3];
            quat_rotate_vec(mesh_xquat, ln, wn);

            float half = pen_depth * 0.5f;
            cand_pos[ncand][0] = corner_w[0] + wn[0]*half;
            cand_pos[ncand][1] = corner_w[1] + wn[1]*half;
            cand_pos[ncand][2] = corner_w[2] + wn[2]*half;
            cand_norm[ncand][0] = wn[0]; cand_norm[ncand][1] = wn[1]; cand_norm[ncand][2] = wn[2];
            cand_depth[ncand] = pen_depth;
            ncand++;
        }
    }

    return manifold_select_4(cand_pos, cand_norm, cand_depth, ncand, out_pos, out_norm, out_depth);
}

// ── Convex-convex SAT + Sutherland-Hodgman face clipping ─────────────────────
// Generalized from the (former) float box-box collider to arbitrary convex hulls.
// Returns up to MAX_PAIR_CONTACTS contact points.

HIPGL_DEVICE float support_hull(
    const float* verts, int adr, int nvert,
    const float* xpos, const float* xquat,
    const float* axis, float* support_point)
{
    float best_dot = -1e20f;
    int best_idx = 0;
    for (int vi = 0; vi < nvert; vi++) {
        float lv[3] = {verts[(adr + vi) * 3],
                        verts[(adr + vi) * 3 + 1],
                        verts[(adr + vi) * 3 + 2]};
        float wv[3];
        quat_rotate_vec(xquat, lv, wv);
        wv[0] += xpos[0]; wv[1] += xpos[1]; wv[2] += xpos[2];
        float d = vec3_dot(wv, axis);
        if (d > best_dot) { best_dot = d; best_idx = vi; support_point[0] = wv[0]; support_point[1] = wv[1]; support_point[2] = wv[2]; }
    }
    return best_dot;
}

HIPGL_DEVICE float sat_axis_test(
    const float* verts1, int adr1, int nv1, const float* xpos1, const float* xquat1,
    const float* verts2, int adr2, int nv2, const float* xpos2, const float* xquat2,
    const float* axis)
{
    float sp1[3], sp2[3];
    float max1 = support_hull(verts1, adr1, nv1, xpos1, xquat1, axis, sp1);
    float neg_axis[3] = {-axis[0], -axis[1], -axis[2]};
    float min2_neg = support_hull(verts2, adr2, nv2, xpos2, xquat2, neg_axis, sp2);
    return max1 - (-min2_neg);
}

HIPGL_DEVICE int collide_convex_convex(
    const float* xpos1, const float* xquat1,
    int vert_adr1, int nvert1, int face_adr1, int nface1, int edge_adr1, int nedge1,
    const float* xpos2, const float* xquat2,
    int vert_adr2, int nvert2, int face_adr2, int nface2, int edge_adr2, int nedge2,
    const float* hull_verts, const float* hull_face_normals,
    const int* hull_faces, const int* hull_edges,
    float out_pos[][3], float out_norm[][3], float out_depth[])
{
    float best_sep = -1e20f;
    float best_axis[3] = {0, 0, 1};
    int best_type = 0; // 0=face1, 1=face2, 2=edge-edge

    // Test face normals of hull 1
    for (int fi = 0; fi < nface1 && fi < MAX_CONVEX_HULL_FACES; fi++) {
        float ln[3] = {hull_face_normals[(face_adr1 + fi) * 3],
                        hull_face_normals[(face_adr1 + fi) * 3 + 1],
                        hull_face_normals[(face_adr1 + fi) * 3 + 2]};
        float wn[3];
        quat_rotate_vec(xquat1, ln, wn);
        float len = vec3_len(wn);
        if (len < 1e-10f) continue;
        wn[0] /= len; wn[1] /= len; wn[2] /= len;

        float sep = sat_axis_test(hull_verts, vert_adr1, nvert1, xpos1, xquat1,
                                  hull_verts, vert_adr2, nvert2, xpos2, xquat2, wn);
        if (sep > 0.0f) return 0;
        if (sep > best_sep) { best_sep = sep; best_axis[0] = wn[0]; best_axis[1] = wn[1]; best_axis[2] = wn[2]; best_type = 0; }
    }

    // Test face normals of hull 2
    for (int fi = 0; fi < nface2 && fi < MAX_CONVEX_HULL_FACES; fi++) {
        float ln[3] = {hull_face_normals[(face_adr2 + fi) * 3],
                        hull_face_normals[(face_adr2 + fi) * 3 + 1],
                        hull_face_normals[(face_adr2 + fi) * 3 + 2]};
        float wn[3];
        quat_rotate_vec(xquat2, ln, wn);
        float len = vec3_len(wn);
        if (len < 1e-10f) continue;
        wn[0] /= len; wn[1] /= len; wn[2] /= len;

        float sep = sat_axis_test(hull_verts, vert_adr1, nvert1, xpos1, xquat1,
                                  hull_verts, vert_adr2, nvert2, xpos2, xquat2, wn);
        if (sep > 0.0f) return 0;
        if (sep > best_sep) { best_sep = sep; best_axis[0] = wn[0]; best_axis[1] = wn[1]; best_axis[2] = wn[2]; best_type = 1; }
    }

    // Test edge-edge cross products (limited to avoid excessive computation)
    int max_edge_tests = (nedge1 < 32 ? nedge1 : 32) * (nedge2 < 32 ? nedge2 : 32);
    int edge_count = 0;
    for (int ei = 0; ei < nedge1 && ei < 32; ei++) {
        int e1a = hull_edges[(edge_adr1 + ei) * 2];
        int e1b = hull_edges[(edge_adr1 + ei) * 2 + 1];
        float le1a[3] = {hull_verts[(vert_adr1 + e1a) * 3], hull_verts[(vert_adr1 + e1a) * 3 + 1], hull_verts[(vert_adr1 + e1a) * 3 + 2]};
        float le1b[3] = {hull_verts[(vert_adr1 + e1b) * 3], hull_verts[(vert_adr1 + e1b) * 3 + 1], hull_verts[(vert_adr1 + e1b) * 3 + 2]};
        float we1a[3], we1b[3];
        quat_rotate_vec(xquat1, le1a, we1a); we1a[0] += xpos1[0]; we1a[1] += xpos1[1]; we1a[2] += xpos1[2];
        quat_rotate_vec(xquat1, le1b, we1b); we1b[0] += xpos1[0]; we1b[1] += xpos1[1]; we1b[2] += xpos1[2];
        float dir1[3] = {we1b[0]-we1a[0], we1b[1]-we1a[1], we1b[2]-we1a[2]};

        for (int ej = 0; ej < nedge2 && ej < 32; ej++) {
            int e2a = hull_edges[(edge_adr2 + ej) * 2];
            int e2b = hull_edges[(edge_adr2 + ej) * 2 + 1];
            float le2a[3] = {hull_verts[(vert_adr2 + e2a) * 3], hull_verts[(vert_adr2 + e2a) * 3 + 1], hull_verts[(vert_adr2 + e2a) * 3 + 2]};
            float le2b[3] = {hull_verts[(vert_adr2 + e2b) * 3], hull_verts[(vert_adr2 + e2b) * 3 + 1], hull_verts[(vert_adr2 + e2b) * 3 + 2]};
            float we2a[3], we2b[3];
            quat_rotate_vec(xquat2, le2a, we2a); we2a[0] += xpos2[0]; we2a[1] += xpos2[1]; we2a[2] += xpos2[2];
            quat_rotate_vec(xquat2, le2b, we2b); we2b[0] += xpos2[0]; we2b[1] += xpos2[1]; we2b[2] += xpos2[2];
            float dir2[3] = {we2b[0]-we2a[0], we2b[1]-we2a[1], we2b[2]-we2a[2]};

            float cross[3];
            vec3_cross(dir1, dir2, cross);
            float len = vec3_len(cross);
            if (len < 1e-8f) continue;
            cross[0] /= len; cross[1] /= len; cross[2] /= len;

            // Orient axis from hull1 to hull2
            float center_diff[3] = {xpos2[0]-xpos1[0], xpos2[1]-xpos1[1], xpos2[2]-xpos1[2]};
            if (vec3_dot(cross, center_diff) < 0) { cross[0] = -cross[0]; cross[1] = -cross[1]; cross[2] = -cross[2]; }

            float sep = sat_axis_test(hull_verts, vert_adr1, nvert1, xpos1, xquat1,
                                      hull_verts, vert_adr2, nvert2, xpos2, xquat2, cross);
            if (sep > 0.0f) return 0;
            if (sep > best_sep) { best_sep = sep; best_axis[0] = cross[0]; best_axis[1] = cross[1]; best_axis[2] = cross[2]; best_type = 2; }
        }
    }

    if (best_sep >= 0.0f) return 0;
    float pen_depth = -best_sep;

    // Orient normal from geom1 toward geom2
    float cd[3] = {xpos2[0]-xpos1[0], xpos2[1]-xpos1[1], xpos2[2]-xpos1[2]};
    if (vec3_dot(best_axis, cd) < 0) { best_axis[0] = -best_axis[0]; best_axis[1] = -best_axis[1]; best_axis[2] = -best_axis[2]; }

    // Find reference and incident faces for clipping
    // Reference face: face on hull whose normal is most aligned with best_axis
    int ref_hull = 0; // 0=hull1, 1=hull2
    int ref_face_idx = 0;
    float best_align = -1e20f;

    for (int fi = 0; fi < nface1 && fi < MAX_CONVEX_HULL_FACES; fi++) {
        float ln[3] = {hull_face_normals[(face_adr1+fi)*3], hull_face_normals[(face_adr1+fi)*3+1], hull_face_normals[(face_adr1+fi)*3+2]};
        float wn[3]; quat_rotate_vec(xquat1, ln, wn);
        float d = vec3_dot(wn, best_axis);
        if (d > best_align) { best_align = d; ref_face_idx = fi; ref_hull = 0; }
    }
    for (int fi = 0; fi < nface2 && fi < MAX_CONVEX_HULL_FACES; fi++) {
        float ln[3] = {hull_face_normals[(face_adr2+fi)*3], hull_face_normals[(face_adr2+fi)*3+1], hull_face_normals[(face_adr2+fi)*3+2]};
        float wn[3]; quat_rotate_vec(xquat2, ln, wn);
        float d = -vec3_dot(wn, best_axis);
        if (d > best_align) { best_align = d; ref_face_idx = fi; ref_hull = 1; }
    }

    // Get reference face vertices in world space
    int r_face_adr = (ref_hull == 0) ? face_adr1 : face_adr2;
    int r_vert_adr = (ref_hull == 0) ? vert_adr1 : vert_adr2;
    const float* r_xpos = (ref_hull == 0) ? xpos1 : xpos2;
    const float* r_xquat = (ref_hull == 0) ? xquat1 : xquat2;
    int rv0 = hull_faces[(r_face_adr + ref_face_idx) * 3];
    int rv1 = hull_faces[(r_face_adr + ref_face_idx) * 3 + 1];
    int rv2 = hull_faces[(r_face_adr + ref_face_idx) * 3 + 2];

    float ref_v[3][3];
    for (int k = 0; k < 3; k++) {
        int rv = (k == 0) ? rv0 : (k == 1) ? rv1 : rv2;
        float lv[3] = {hull_verts[(r_vert_adr + rv)*3], hull_verts[(r_vert_adr + rv)*3+1], hull_verts[(r_vert_adr + rv)*3+2]};
        quat_rotate_vec(r_xquat, lv, ref_v[k]);
        ref_v[k][0] += r_xpos[0]; ref_v[k][1] += r_xpos[1]; ref_v[k][2] += r_xpos[2];
    }

    float ref_normal[3];
    {
        float e1[3] = {ref_v[1][0]-ref_v[0][0], ref_v[1][1]-ref_v[0][1], ref_v[1][2]-ref_v[0][2]};
        float e2[3] = {ref_v[2][0]-ref_v[0][0], ref_v[2][1]-ref_v[0][1], ref_v[2][2]-ref_v[0][2]};
        vec3_cross(e1, e2, ref_normal);
        vec3_normalize(ref_normal);
        if (vec3_dot(ref_normal, best_axis) < 0) { ref_normal[0] = -ref_normal[0]; ref_normal[1] = -ref_normal[1]; ref_normal[2] = -ref_normal[2]; }
    }
    float ref_d = vec3_dot(ref_normal, ref_v[0]);

    // Get incident face: face on the other hull most anti-aligned with ref_normal
    int inc_face_adr = (ref_hull == 0) ? face_adr2 : face_adr1;
    int inc_vert_adr = (ref_hull == 0) ? vert_adr2 : vert_adr1;
    int inc_nface = (ref_hull == 0) ? nface2 : nface1;
    const float* inc_xpos = (ref_hull == 0) ? xpos2 : xpos1;
    const float* inc_xquat = (ref_hull == 0) ? xquat2 : xquat1;

    int inc_face_idx = 0;
    float worst_align = 1e20f;
    for (int fi = 0; fi < inc_nface && fi < MAX_CONVEX_HULL_FACES; fi++) {
        float ln[3] = {hull_face_normals[(inc_face_adr+fi)*3], hull_face_normals[(inc_face_adr+fi)*3+1], hull_face_normals[(inc_face_adr+fi)*3+2]};
        float wn[3]; quat_rotate_vec(inc_xquat, ln, wn);
        float d = vec3_dot(wn, ref_normal);
        if (d < worst_align) { worst_align = d; inc_face_idx = fi; }
    }

    // Get incident face vertices (triangle)
    float clip_poly[MAX_CLIP_VERTS][3];
    int clip_n = 3;
    for (int k = 0; k < 3; k++) {
        int iv = hull_faces[(inc_face_adr + inc_face_idx) * 3 + k];
        float lv[3] = {hull_verts[(inc_vert_adr + iv)*3], hull_verts[(inc_vert_adr + iv)*3+1], hull_verts[(inc_vert_adr + iv)*3+2]};
        quat_rotate_vec(inc_xquat, lv, clip_poly[k]);
        clip_poly[k][0] += inc_xpos[0]; clip_poly[k][1] += inc_xpos[1]; clip_poly[k][2] += inc_xpos[2];
    }

    // Sutherland-Hodgman clip incident polygon against reference face side planes
    for (int edge = 0; edge < 3 && clip_n > 0; edge++) {
        int ea = edge, eb = (edge + 1) % 3;
        float side_edge[3] = {ref_v[eb][0]-ref_v[ea][0], ref_v[eb][1]-ref_v[ea][1], ref_v[eb][2]-ref_v[ea][2]};
        float side_normal[3];
        vec3_cross(ref_normal, side_edge, side_normal);
        vec3_normalize(side_normal);
        float side_d = vec3_dot(side_normal, ref_v[ea]);

        float new_poly[MAX_CLIP_VERTS][3];
        int new_n = 0;
        for (int ci = 0; ci < clip_n && new_n < MAX_CLIP_VERTS - 1; ci++) {
            int cj = (ci + 1) % clip_n;
            float di = vec3_dot(side_normal, clip_poly[ci]) - side_d;
            float dj = vec3_dot(side_normal, clip_poly[cj]) - side_d;
            if (di >= 0) {
                new_poly[new_n][0] = clip_poly[ci][0]; new_poly[new_n][1] = clip_poly[ci][1]; new_poly[new_n][2] = clip_poly[ci][2]; new_n++;
            }
            if ((di >= 0) != (dj >= 0) && new_n < MAX_CLIP_VERTS) {
                float t = di / (di - dj);
                new_poly[new_n][0] = clip_poly[ci][0] + t * (clip_poly[cj][0] - clip_poly[ci][0]);
                new_poly[new_n][1] = clip_poly[ci][1] + t * (clip_poly[cj][1] - clip_poly[ci][1]);
                new_poly[new_n][2] = clip_poly[ci][2] + t * (clip_poly[cj][2] - clip_poly[ci][2]);
                new_n++;
            }
        }
        clip_n = new_n;
        for (int ci = 0; ci < clip_n; ci++) { clip_poly[ci][0] = new_poly[ci][0]; clip_poly[ci][1] = new_poly[ci][1]; clip_poly[ci][2] = new_poly[ci][2]; }
    }

    // Collect all penetrating clipped points as candidates, then select 4 well-spaced
    float cand_pos[MAX_CLIP_VERTS][3];
    float cand_norm[MAX_CLIP_VERTS][3];
    float cand_depth[MAX_CLIP_VERTS];
    int ncand = 0;
    for (int ci = 0; ci < clip_n && ncand < MAX_CLIP_VERTS; ci++) {
        float d = vec3_dot(ref_normal, clip_poly[ci]) - ref_d;
        if (d <= 0.0f) {
            float depth = -d;
            cand_pos[ncand][0] = clip_poly[ci][0] - ref_normal[0] * d * 0.5f;
            cand_pos[ncand][1] = clip_poly[ci][1] - ref_normal[1] * d * 0.5f;
            cand_pos[ncand][2] = clip_poly[ci][2] - ref_normal[2] * d * 0.5f;
            cand_norm[ncand][0] = ref_normal[0]; cand_norm[ncand][1] = ref_normal[1]; cand_norm[ncand][2] = ref_normal[2];
            cand_depth[ncand] = depth;
            ncand++;
        }
    }

    return manifold_select_4(cand_pos, cand_norm, cand_depth, ncand, out_pos, out_norm, out_depth);
}

// ══════════════════════════════════════════════════════════════════════════════
// Double-precision narrowphase functions
// These match the float versions exactly but use double throughout for
// MuJoCo float64 parity. The broadphase calls these when double geom data
// is available.
// ══════════════════════════════════════════════════════════════════════════════

HIPGL_DEVICE int manifold_select_4_d(
    double cand_pos[][3], double cand_norm[][3], double cand_depth[],
    int ncand,
    double out_pos[][3], double out_norm[][3], double out_depth[])
{
    if (ncand <= MAX_PAIR_CONTACTS) {
        for (int i = 0; i < ncand; i++) {
            for (int k = 0; k < 3; k++) { out_pos[i][k] = cand_pos[i][k]; out_norm[i][k] = cand_norm[i][k]; }
            out_depth[i] = cand_depth[i];
        }
        return ncand;
    }
    int sel[4] = {0, 0, 0, 0};
    double best_d2 = -1.0;
    for (int i = 1; i < ncand; i++) {
        double dx = cand_pos[i][0]-cand_pos[0][0], dy = cand_pos[i][1]-cand_pos[0][1], dz = cand_pos[i][2]-cand_pos[0][2];
        double d2 = dx*dx+dy*dy+dz*dz;
        if (d2 > best_d2) { best_d2 = d2; sel[1] = i; }
    }
    double ab[3] = {cand_pos[sel[1]][0]-cand_pos[0][0], cand_pos[sel[1]][1]-cand_pos[0][1], cand_pos[sel[1]][2]-cand_pos[0][2]};
    double ab_len = sqrt(ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]);
    if (ab_len > 1e-15) { ab[0]/=ab_len; ab[1]/=ab_len; ab[2]/=ab_len; }
    best_d2 = -1.0;
    for (int i = 0; i < ncand; i++) {
        if (i == sel[0] || i == sel[1]) continue;
        double ap[3] = {cand_pos[i][0]-cand_pos[0][0], cand_pos[i][1]-cand_pos[0][1], cand_pos[i][2]-cand_pos[0][2]};
        double proj = ap[0]*ab[0]+ap[1]*ab[1]+ap[2]*ab[2];
        double perp[3] = {ap[0]-proj*ab[0], ap[1]-proj*ab[1], ap[2]-proj*ab[2]};
        double d2 = perp[0]*perp[0]+perp[1]*perp[1]+perp[2]*perp[2];
        if (d2 > best_d2) { best_d2 = d2; sel[2] = i; }
    }
    double ac[3] = {cand_pos[sel[2]][0]-cand_pos[0][0], cand_pos[sel[2]][1]-cand_pos[0][1], cand_pos[sel[2]][2]-cand_pos[0][2]};
    double tri_n[3]; vec3_cross_d(ab, ac, tri_n);
    double tri_n_len = vec3_len_d(tri_n);
    if (tri_n_len > 1e-15) { tri_n[0]/=tri_n_len; tri_n[1]/=tri_n_len; tri_n[2]/=tri_n_len; }
    double best_dist = -1.0;
    for (int i = 0; i < ncand; i++) {
        if (i == sel[0] || i == sel[1] || i == sel[2]) continue;
        double ap[3] = {cand_pos[i][0]-cand_pos[0][0], cand_pos[i][1]-cand_pos[0][1], cand_pos[i][2]-cand_pos[0][2]};
        double d = fabs(vec3_dot_d(ap, tri_n));
        if (d > best_dist) { best_dist = d; sel[3] = i; }
    }
    if (best_dist < 1e-12) {
        double best_min_d2 = -1.0;
        for (int i = 0; i < ncand; i++) {
            if (i == sel[0] || i == sel[1] || i == sel[2]) continue;
            double min_d2 = 1e30;
            for (int s = 0; s < 3; s++) {
                double dx=cand_pos[i][0]-cand_pos[sel[s]][0], dy=cand_pos[i][1]-cand_pos[sel[s]][1], dz=cand_pos[i][2]-cand_pos[sel[s]][2];
                double d2=dx*dx+dy*dy+dz*dz;
                if (d2 < min_d2) min_d2 = d2;
            }
            if (min_d2 > best_min_d2) { best_min_d2 = min_d2; sel[3] = i; }
        }
    }
    for (int k = 0; k < 4; k++) {
        int s = sel[k];
        for (int j = 0; j < 3; j++) { out_pos[k][j] = cand_pos[s][j]; out_norm[k][j] = cand_norm[s][j]; }
        out_depth[k] = cand_depth[s];
    }
    return 4;
}

HIPGL_DEVICE bool collide_sphere_sphere_d(
    const double* p1, double r1, const double* p2, double r2,
    double* contact_pos, double* contact_normal, double* depth,
    double margin = 0.0)
{
    double d[3]; vec3_sub_d(p2, p1, d);
    double dist = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
    if (dist > r1 + r2 + margin) return false;
    *depth = r1 + r2 + margin - dist;
    if (dist > 1e-15) {
        double inv = 1.0/dist;
        contact_normal[0]=d[0]*inv; contact_normal[1]=d[1]*inv; contact_normal[2]=d[2]*inv;
    } else {
        contact_normal[0]=0; contact_normal[1]=0; contact_normal[2]=1;
    }
    contact_pos[0] = p1[0] + contact_normal[0]*(r1 - *depth*0.5);
    contact_pos[1] = p1[1] + contact_normal[1]*(r1 - *depth*0.5);
    contact_pos[2] = p1[2] + contact_normal[2]*(r1 - *depth*0.5);
    return true;
}

HIPGL_DEVICE bool collide_sphere_plane_d(
    const double* sphere_pos, double radius,
    const double* plane_pos, const double* plane_quat,
    double* contact_pos, double* contact_normal, double* depth,
    double margin = 0.0)
{
    double local_up[3] = {0, 0, 1};
    double normal[3];
    quat_rotate_vec_d(plane_quat, local_up, normal);
    double diff[3]; vec3_sub_d(sphere_pos, plane_pos, diff);
    double dist = vec3_dot_d(diff, normal) - radius;
    if (dist > margin) return false;
    *depth = margin - dist;
    contact_normal[0]=-normal[0]; contact_normal[1]=-normal[1]; contact_normal[2]=-normal[2];
    double half_depth = *depth * 0.5;
    contact_pos[0] = sphere_pos[0] - normal[0]*(radius - half_depth);
    contact_pos[1] = sphere_pos[1] - normal[1]*(radius - half_depth);
    contact_pos[2] = sphere_pos[2] - normal[2]*(radius - half_depth);
    return true;
}

HIPGL_DEVICE int collide_box_plane_multi_d(
    const double* box_pos, const double* box_quat, const double* box_size,
    const double* plane_pos, const double* plane_quat,
    double out_pos[][3], double out_norm[][3], double out_depth[],
    double margin = 0.0)
{
    double local_up[3] = {0, 0, 1};
    double normal[3];
    quat_rotate_vec_d(plane_quat, local_up, normal);
    double cn[3] = {-normal[0], -normal[1], -normal[2]};
    double corner_signs[8][3] = {
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}
    };
    double corner_depths[8], corner_pos[8][3];
    int nbelow = 0;
    for (int c = 0; c < 8; c++) {
        double local[3] = {corner_signs[c][0]*box_size[0], corner_signs[c][1]*box_size[1], corner_signs[c][2]*box_size[2]};
        double wc[3];
        quat_rotate_vec_d(box_quat, local, wc);
        corner_pos[c][0]=box_pos[0]+wc[0]; corner_pos[c][1]=box_pos[1]+wc[1]; corner_pos[c][2]=box_pos[2]+wc[2];
        double diff[3]; vec3_sub_d(corner_pos[c], plane_pos, diff);
        double d = vec3_dot_d(diff, normal);
        corner_depths[c] = margin - d;
        if (d <= margin) nbelow++;
    }
    if (nbelow == 0) return 0;
    double cand_pos[8][3], cand_norm[8][3], cand_depth[8];
    int ncand = 0;
    for (int c = 0; c < 8; c++) {
        if (corner_depths[c] < 0) continue;
        double half_d = corner_depths[c] * 0.5;
        cand_pos[ncand][0]=corner_pos[c][0]+normal[0]*half_d;
        cand_pos[ncand][1]=corner_pos[c][1]+normal[1]*half_d;
        cand_pos[ncand][2]=corner_pos[c][2]+normal[2]*half_d;
        cand_norm[ncand][0]=cn[0]; cand_norm[ncand][1]=cn[1]; cand_norm[ncand][2]=cn[2];
        cand_depth[ncand] = corner_depths[c];
        ncand++;
    }
    return manifold_select_4_d(cand_pos, cand_norm, cand_depth, ncand, out_pos, out_norm, out_depth);
}

HIPGL_DEVICE int collide_capsule_plane_multi_d(
    const double* cap_pos, const double* cap_quat, double radius, double half_len,
    const double* plane_pos, const double* plane_quat,
    double out_pos[][3], double out_norm[][3], double out_depth[],
    double margin = 0.0)
{
    double local_up[3] = {0, 0, 1};
    double normal[3];
    quat_rotate_vec_d(plane_quat, local_up, normal);
    double cn[3] = {-normal[0], -normal[1], -normal[2]};
    double cap_axis_local[3] = {0, 0, 1};
    double cap_axis[3];
    quat_rotate_vec_d(cap_quat, cap_axis_local, cap_axis);
    double ep[2][3];
    for (int k = 0; k < 3; k++) {
        ep[0][k] = cap_pos[k] + cap_axis[k]*half_len;
        ep[1][k] = cap_pos[k] - cap_axis[k]*half_len;
    }
    int ncontacts = 0;
    for (int i = 0; i < 2 && ncontacts < MAX_PAIR_CONTACTS; i++) {
        double diff[3]; vec3_sub_d(ep[i], plane_pos, diff);
        double d = vec3_dot_d(diff, normal) - radius;
        if (d <= margin) {
            double dep = margin - d;
            out_depth[ncontacts] = dep;
            double half_dep = dep * 0.5;
            out_pos[ncontacts][0] = ep[i][0] - normal[0]*(radius - half_dep);
            out_pos[ncontacts][1] = ep[i][1] - normal[1]*(radius - half_dep);
            out_pos[ncontacts][2] = ep[i][2] - normal[2]*(radius - half_dep);
            out_norm[ncontacts][0]=cn[0]; out_norm[ncontacts][1]=cn[1]; out_norm[ncontacts][2]=cn[2];
            ncontacts++;
        }
    }
    return ncontacts;
}

HIPGL_DEVICE void closest_point_on_segment_d(
    const double* p, const double* a, const double* b, double* result)
{
    double ab[3] = {b[0]-a[0], b[1]-a[1], b[2]-a[2]};
    double ap[3] = {p[0]-a[0], p[1]-a[1], p[2]-a[2]};
    double ab_dot = vec3_dot_d(ab, ab);
    if (ab_dot < 1e-20) { result[0]=a[0]; result[1]=a[1]; result[2]=a[2]; return; }
    double t = dev_clamp_d(vec3_dot_d(ap, ab) / ab_dot, 0.0, 1.0);
    result[0]=a[0]+t*ab[0]; result[1]=a[1]+t*ab[1]; result[2]=a[2]+t*ab[2];
}

HIPGL_DEVICE bool collide_sphere_box_d(
    const double* sp, double sr,
    const double* bp, const double* bq, const double* bsize,
    double* contact_pos, double* contact_normal, double* depth,
    double margin = 0.0)
{
    double rel[3]; vec3_sub_d(sp, bp, rel);
    double local[3];
    quat_inv_rotate_vec_d(bq, rel, local);
    double closest[3];
    closest[0]=dev_clamp_d(local[0],-bsize[0],bsize[0]);
    closest[1]=dev_clamp_d(local[1],-bsize[1],bsize[1]);
    closest[2]=dev_clamp_d(local[2],-bsize[2],bsize[2]);
    double diff[3]={local[0]-closest[0], local[1]-closest[1], local[2]-closest[2]};
    double dist_sq=diff[0]*diff[0]+diff[1]*diff[1]+diff[2]*diff[2];
    double threshold = sr + margin;
    if (dist_sq > threshold*threshold && dist_sq > 1e-20) return false;
    double dist = sqrt(dist_sq);
    double n_local[3];
    if (dist > 1e-15) {
        double inv=1.0/dist;
        n_local[0]=diff[0]*inv; n_local[1]=diff[1]*inv; n_local[2]=diff[2]*inv;
        *depth = sr + margin - dist;
    } else {
        double face_dist[6]={bsize[0]-local[0],bsize[0]+local[0],bsize[1]-local[1],bsize[1]+local[1],bsize[2]-local[2],bsize[2]+local[2]};
        int min_face=0; double min_d=face_dist[0];
        for (int i=1;i<6;i++) { if (face_dist[i]<min_d){min_d=face_dist[i];min_face=i;} }
        n_local[0]=n_local[1]=n_local[2]=0;
        n_local[min_face/2]=(min_face%2==0)?1.0:-1.0;
        *depth=sr+margin+min_d;
    }
    quat_rotate_vec_d(bq, n_local, contact_normal);
    double cp_world[3];
    quat_rotate_vec_d(bq, closest, cp_world);
    double half_d=*depth*0.5;
    contact_pos[0]=bp[0]+cp_world[0]-contact_normal[0]*half_d;
    contact_pos[1]=bp[1]+cp_world[1]-contact_normal[1]*half_d;
    contact_pos[2]=bp[2]+cp_world[2]-contact_normal[2]*half_d;
    return true;
}

HIPGL_DEVICE bool collide_sphere_capsule_d(
    const double* sp, double sr,
    const double* cp, const double* cq, double cr, double chalf,
    double* contact_pos, double* contact_normal, double* depth,
    double margin = 0.0)
{
    double local_axis[3]={0,0,1}, axis[3];
    quat_rotate_vec_d(cq, local_axis, axis);
    double a[3]={cp[0]-axis[0]*chalf, cp[1]-axis[1]*chalf, cp[2]-axis[2]*chalf};
    double b[3]={cp[0]+axis[0]*chalf, cp[1]+axis[1]*chalf, cp[2]+axis[2]*chalf};
    double closest[3];
    closest_point_on_segment_d(sp, a, b, closest);
    return collide_sphere_sphere_d(sp, sr, closest, cr, contact_pos, contact_normal, depth, margin);
}

HIPGL_DEVICE bool collide_capsule_capsule_d(
    const double* p1, const double* q1, double r1, double h1,
    const double* p2, const double* q2, double r2, double h2,
    double* contact_pos, double* contact_normal, double* depth,
    double margin = 0.0)
{
    double la[3]={0,0,1}, ax1[3], ax2[3];
    quat_rotate_vec_d(q1, la, ax1); quat_rotate_vec_d(q2, la, ax2);
    double a1[3]={p1[0]-ax1[0]*h1,p1[1]-ax1[1]*h1,p1[2]-ax1[2]*h1};
    double b1[3]={p1[0]+ax1[0]*h1,p1[1]+ax1[1]*h1,p1[2]+ax1[2]*h1};
    double a2[3]={p2[0]-ax2[0]*h2,p2[1]-ax2[1]*h2,p2[2]-ax2[2]*h2};
    double b2[3]={p2[0]+ax2[0]*h2,p2[1]+ax2[1]*h2,p2[2]+ax2[2]*h2};
    double d1[3]={b1[0]-a1[0],b1[1]-a1[1],b1[2]-a1[2]};
    double d2[3]={b2[0]-a2[0],b2[1]-a2[1],b2[2]-a2[2]};
    double rv[3]={a1[0]-a2[0],a1[1]-a2[1],a1[2]-a2[2]};
    double a=vec3_dot_d(d1,d1), e_val=vec3_dot_d(d2,d2), f=vec3_dot_d(d2,rv);
    double s_param, t_param;
    if (a<1e-20 && e_val<1e-20) { s_param=t_param=0.0; }
    else if (a<1e-20) { s_param=0.0; t_param=dev_clamp_d(f/e_val,0.0,1.0); }
    else {
        double c_val=vec3_dot_d(d1,rv);
        if (e_val<1e-20) { t_param=0.0; s_param=dev_clamp_d(-c_val/a,0.0,1.0); }
        else {
            double b_val=vec3_dot_d(d1,d2), denom=a*e_val-b_val*b_val;
            s_param=(denom>1e-20)?dev_clamp_d((b_val*f-c_val*e_val)/denom,0.0,1.0):0.0;
            t_param=dev_clamp_d((b_val*s_param+f)/e_val,0.0,1.0);
            s_param=dev_clamp_d((b_val*t_param-c_val)/a,0.0,1.0);
        }
    }
    double c1[3]={a1[0]+s_param*d1[0],a1[1]+s_param*d1[1],a1[2]+s_param*d1[2]};
    double c2[3]={a2[0]+t_param*d2[0],a2[1]+t_param*d2[1],a2[2]+t_param*d2[2]};
    return collide_sphere_sphere_d(c1, r1, c2, r2, contact_pos, contact_normal, depth, margin);
}

HIPGL_DEVICE int collide_box_capsule_multi_d(
    const double* bp, const double* bq, const double* bs,
    const double* cp, const double* cq, double cr, double chalf,
    double out_pos[][3], double out_norm[][3], double out_depth[],
    double margin = 0.0)
{
    double la[3]={0,0,1}, axis[3];
    quat_rotate_vec_d(cq, la, axis);
    double endpoints[2][3];
    for (int k=0;k<3;k++){endpoints[0][k]=cp[k]-axis[k]*chalf; endpoints[1][k]=cp[k]+axis[k]*chalf;}
    int n=0;
    for (int i=0; i<2 && n<MAX_PAIR_CONTACTS; i++) {
        double cpos[3], cnorm[3], cdepth;
        if (collide_sphere_box_d(endpoints[i], cr, bp, bq, bs, cpos, cnorm, &cdepth, margin)) {
            for (int k=0;k<3;k++){out_pos[n][k]=cpos[k]; out_norm[n][k]=cnorm[k];}
            out_depth[n]=cdepth; n++;
        }
    }

    if (n == 0) {
        // Box-local frame.
        double biq[4] = {bq[0], -bq[1], -bq[2], -bq[3]};
        double rel[3] = {cp[0]-bp[0], cp[1]-bp[1], cp[2]-bp[2]};
        double c_loc[3], ax_loc[3];
        quat_rotate_vec_d(biq, rel, c_loc);
        quat_rotate_vec_d(biq, axis, ax_loc);

        // Walk the segment for the parameter minimising distance to the box. The
        // clamped-per-axis target is not separable for an OBB, so take the best of a
        // coarse sampling and refine -- cheap, and robust for any orientation.
        double best_t = 0.0, best_d2 = 1e30;
        const int NS = 32;
        for (int i = 0; i <= NS; i++) {
            double t = -chalf + (2.0*chalf) * ((double)i / (double)NS);
            double p[3] = {c_loc[0]+ax_loc[0]*t, c_loc[1]+ax_loc[1]*t, c_loc[2]+ax_loc[2]*t};
            double d2 = 0.0;
            for (int k=0;k<3;k++){
                double q = p[k];
                if (q >  bs[k]) { double e = q - bs[k]; d2 += e*e; }
                else if (q < -bs[k]) { double e = -bs[k] - q; d2 += e*e; }
            }
            if (d2 < best_d2) { best_d2 = d2; best_t = t; }
        }
        double step = (2.0*chalf) / (double)NS;
        for (int iter = 0; iter < 24; iter++) {
            step *= 0.5;
            for (int s = -1; s <= 1; s += 2) {
                double t = best_t + s*step;
                if (t < -chalf) t = -chalf;
                if (t >  chalf) t =  chalf;
                double p[3] = {c_loc[0]+ax_loc[0]*t, c_loc[1]+ax_loc[1]*t, c_loc[2]+ax_loc[2]*t};
                double d2 = 0.0;
                for (int k=0;k<3;k++){
                    double q = p[k];
                    if (q >  bs[k]) { double e = q - bs[k]; d2 += e*e; }
                    else if (q < -bs[k]) { double e = -bs[k] - q; d2 += e*e; }
                }
                if (d2 < best_d2) { best_d2 = d2; best_t = t; }
            }
        }

        double mid_w[3] = {cp[0]+axis[0]*best_t, cp[1]+axis[1]*best_t, cp[2]+axis[2]*best_t};
        double cpos[3], cnorm[3], cdepth;
        if (collide_sphere_box_d(mid_w, cr, bp, bq, bs, cpos, cnorm, &cdepth, margin)) {
            for (int k=0;k<3;k++){out_pos[n][k]=cpos[k]; out_norm[n][k]=cnorm[k];}
            out_depth[n]=cdepth; n++;
        }
    }
    return n;
}

// ── Broadphase + narrowphase kernel ─────────────────────────────────────────

// ── MuJoCo convex collision (box-mesh / mesh-mesh) ──────────────────────────
// physics_mjccd.h is a bit-exact (CPU-verified vs libmujoco 3.9.0) transcription of
// mjc_Convex's native-CCD path. Mesh pairs are queued by kernel_broadphase and processed
// by kernel_convex_mj so the ~0.2 MB/worker EPA/multicontact scratch lives in global memory.
namespace mjpose {
// MuJoCo engine_util_spatial.c:66 mju_mulQuat
HIPGL_DEVICE inline void mulQuat(double res[4], const double qa[4], const double qb[4]) {
    double tmp[4] = {
        qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3],
        qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2],
        qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1],
        qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
    };
    res[0] = tmp[0]; res[1] = tmp[1]; res[2] = tmp[2]; res[3] = tmp[3];
}
// MuJoCo engine_util_spatial.c:145 mju_quat2Mat
HIPGL_DEVICE inline void quat2Mat(double res[9], const double quat[4]) {
    if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
        res[0] = 1; res[1] = 0; res[2] = 0; res[3] = 0; res[4] = 1; res[5] = 0;
        res[6] = 0; res[7] = 0; res[8] = 1;
        return;
    }
    double q00 = quat[0]*quat[0], q01 = quat[0]*quat[1], q02 = quat[0]*quat[2], q03 = quat[0]*quat[3];
    double q11 = quat[1]*quat[1], q12 = quat[1]*quat[2], q13 = quat[1]*quat[3];
    double q22 = quat[2]*quat[2], q23 = quat[2]*quat[3], q33 = quat[3]*quat[3];
    res[0] = q00 + q11 - q22 - q33;
    res[4] = q00 - q11 + q22 - q33;
    res[8] = q00 - q11 - q22 + q33;
    res[1] = 2*(q12 - q03);
    res[2] = 2*(q13 + q02);
    res[3] = 2*(q12 + q03);
    res[5] = 2*(q23 - q01);
    res[6] = 2*(q13 - q02);
    res[7] = 2*(q23 + q01);
}
// MuJoCo engine_core_util.c:944 mj_local2Global, mjSAMEFRAME_NONE branch (the BODY/INERTIA
// shortcuts compute the same values: MuJoCo builds xipos/ximat with this formula)
HIPGL_DEVICE inline void local2Global(double xpos[3], double xmat[9], const double bxpos[3],
                                      const double bxquat[4], const double pos[3], const double quat[4]) {
    double bmat[9];
    quat2Mat(bmat, bxquat);
    xpos[0] = bmat[0]*pos[0] + bmat[1]*pos[1] + bmat[2]*pos[2];
    xpos[1] = bmat[3]*pos[0] + bmat[4]*pos[1] + bmat[5]*pos[2];
    xpos[2] = bmat[6]*pos[0] + bmat[7]*pos[1] + bmat[8]*pos[2];
    xpos[0] += bxpos[0]; xpos[1] += bxpos[1]; xpos[2] += bxpos[2];
    double tmp[4];
    mulQuat(tmp, bxquat, quat);
    quat2Mat(xmat, tmp);
}
}  // namespace mjpose

// Per-pair static filter mask for kernel_broadphase<*, true>: 1 iff pair
// `pair_idx` survives the env-independent filters (same body, weld, parent
// weld, contype/conaffinity, explicit pair exclusions).  The index inversion
// and the filter code are verbatim copies of kernel_broadphase's, so a lane
// of the masked kernel returns early exactly when the unmasked lane would.
HIPGL_GLOBAL void kernel_broadphase_pair_mask(
    int ngeom,
    const int*    __restrict__ geom_contype,
    const int*    __restrict__ geom_conaffinity,
    const int*    __restrict__ geom_body,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_weldid,
    const int*   __restrict__ pair_filter,
    int npair_filter,
    unsigned char* __restrict__ pair_mask)
{
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = ngeom * (ngeom - 1) / 2;
    if (pair_idx >= total_pairs) return;
    pair_mask[pair_idx] = 0;

    // Invert the strictly-upper-triangular pair index in O(1).
    // Row i holds (ngeom-1-i) pairs, so the number of pairs before row i is
    //     S(i) = i*(2*ngeom-1-i)/2
    // and we want the largest i with S(i) <= pair_idx.  Solving the quadratic
    // gives the exact answer up to one ulp; the two corrective steps make it
    // exact for every input.  This replaces a scan that averaged ~ngeom/3
    // dependent iterations PER THREAD (ngeom = 270 here) and dominated the
    // kernel — the pair enumeration itself is unchanged, so the (i, j) each
    // thread handles is bit-identical to before.
    int i = 0, j = 0;
    {
        const double bq = 2.0 * (double)ngeom - 1.0;
        double disc = bq * bq - 8.0 * (double)pair_idx;
        if (disc < 0.0) disc = 0.0;
        i = (int)((bq - sqrt(disc)) * 0.5);
        if (i < 0) i = 0;
        if (i > ngeom - 2) i = ngeom - 2;
        long long Si = (long long)i * (2LL * ngeom - 1 - i) / 2;
        while (i > 0 && Si > (long long)pair_idx) {
            i--;
            Si = (long long)i * (2LL * ngeom - 1 - i) / 2;
        }
        long long Si1 = (long long)(i + 1) * (2LL * ngeom - 2 - i) / 2;
        while (i < ngeom - 2 && Si1 <= (long long)pair_idx) {
            i++;
            Si = Si1;
            Si1 = (long long)(i + 1) * (2LL * ngeom - 2 - i) / 2;
        }
        j = i + 1 + (int)((long long)pair_idx - Si);
    }

    int body_i = geom_body[i];
    int body_j = geom_body[j];
    if (body_i == body_j) return;

    // MuJoCo weld-based filtering: bodies welded together act as one rigid body
    if (body_weldid) {
        int weld_i = body_weldid[body_i];
        int weld_j = body_weldid[body_j];
        if (weld_i == weld_j) return;

        // filterparent: skip parent-child weld body pairs (MuJoCo convention).
        // Exempt world body (body 0): MuJoCo never filters world-body contacts
        // via filterparent because world is always a valid collision partner.
        if (body_parent && weld_i > 0 && weld_j > 0) {
            int pw_i = body_weldid[body_parent[weld_i]];
            int pw_j = body_weldid[body_parent[weld_j]];
            if (pw_i == weld_j || pw_j == weld_i) return;
        }
    }

    if ((geom_contype[i] & geom_conaffinity[j]) == 0 &&
        (geom_contype[j] & geom_conaffinity[i]) == 0) return;

    if (geom_contype[i] == 0 && geom_conaffinity[i] == 0) return;
    if (geom_contype[j] == 0 && geom_conaffinity[j] == 0) return;

    if (npair_filter > 0 && pair_filter) {
        int b_lo = (body_i < body_j) ? body_i : body_j;
        int b_hi = (body_i < body_j) ? body_j : body_i;
        int key = (b_lo << 16) | b_hi;
        int lo = 0, hi = npair_filter - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int val = pair_filter[mid];
            if (val == key) return;
            if (val < key) lo = mid + 1; else hi = mid - 1;
        }
    }

    pair_mask[pair_idx] = 1;
}

// HULLS=false is the instantiation for models with no convex-hull data
// (mesh_convex_vert == nullptr).  Both GEOM_MESH branches below are already
// guarded on `mesh_convex_vert`, so for such models they are dead code -- but
// their float candidate/clip arrays set the kernel's private-segment size
// (4272 B/lane -> 1584 B/lane without them).  Dropping them at compile time
// leaves every lane's work and its atomicAdd issue sequence unchanged.
template <bool HULLS, bool MASK, bool COMPACT = false>
HIPGL_GLOBAL void __launch_bounds__(256) kernel_broadphase(
    int ngeom, int num_envs, int nbody,
    const int*    __restrict__ geom_contype,
    const int*    __restrict__ geom_conaffinity,
    const int*    __restrict__ geom_body,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_weldid,
    const AABB*   __restrict__ geom_aabb,
    const int*    __restrict__ geom_type,
    const double* __restrict__ geom_size_d,
    const double* __restrict__ geom_friction_d,
    int friction_combine,
    const double* __restrict__ geom_xpos_d,
    const double* __restrict__ geom_xquat_d,
    float margin,
    const float*  __restrict__ geom_margin,
    const int*   __restrict__ pair_filter,
    int npair_filter,
    int max_contacts_per_body_pair,
    ContactPair* __restrict__ contacts,
    int*         __restrict__ ncon,
    int*         __restrict__ body_pair_ncon,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert,
    const int*   __restrict__ mesh_convex_face_adr,
    const int*   __restrict__ mesh_convex_face_num,
    const int*   __restrict__ mesh_convex_face,
    const float* __restrict__ mesh_convex_face_normal,
    const int*   __restrict__ mesh_convex_edge_adr,
    const int*   __restrict__ mesh_convex_edge_num,
    const int*   __restrict__ mesh_convex_edge,
    const int*   __restrict__ geom_mj_mesh,
    int*         __restrict__ mjc_queue,
    int*         __restrict__ mjc_queue_count,
    int mjc_queue_cap,
    const unsigned char* __restrict__ pair_mask,
    const int* __restrict__ chunk_base = nullptr)
{
    int pair_idx;
    if constexpr (COMPACT) {
        // Chunk-compacted launch: chunk_base[blockIdx.x] is the first pair
        // index of a blockDim.x-aligned chunk of the pair triangle that holds
        // at least one pair_mask==1 pair.  Each workgroup covers exactly one
        // such chunk with the SAME lane -> pair mapping as the full-triangle
        // launch (block == HIPGL_BP_CHUNK == launch_broadphase_masked's
        // block), so every wave that can reach the narrowphase runs with the
        // identical lane set, (i, j) per lane and intra-wave atomic order.
        // Only workgroups whose pairs ALL fail the static mask are skipped.
        pair_idx = chunk_base[blockIdx.x] + (int)threadIdx.x;
    } else {
        pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    }
    int env = blockIdx.y;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;

    int total_pairs = ngeom * (ngeom - 1) / 2;
    if (pair_idx >= total_pairs) return;
    if constexpr (MASK) {
        if (!pair_mask[pair_idx]) return;
    }

    // Invert the strictly-upper-triangular pair index in O(1).
    // Row i holds (ngeom-1-i) pairs, so the number of pairs before row i is
    //     S(i) = i*(2*ngeom-1-i)/2
    // and we want the largest i with S(i) <= pair_idx.  Solving the quadratic
    // gives the exact answer up to one ulp; the two corrective steps make it
    // exact for every input.  This replaces a scan that averaged ~ngeom/3
    // dependent iterations PER THREAD (ngeom = 270 here) and dominated the
    // kernel — the pair enumeration itself is unchanged, so the (i, j) each
    // thread handles is bit-identical to before.
    int i = 0, j = 0;
    {
        const double bq = 2.0 * (double)ngeom - 1.0;
        double disc = bq * bq - 8.0 * (double)pair_idx;
        if (disc < 0.0) disc = 0.0;
        i = (int)((bq - sqrt(disc)) * 0.5);
        if (i < 0) i = 0;
        if (i > ngeom - 2) i = ngeom - 2;
        long long Si = (long long)i * (2LL * ngeom - 1 - i) / 2;
        while (i > 0 && Si > (long long)pair_idx) {
            i--;
            Si = (long long)i * (2LL * ngeom - 1 - i) / 2;
        }
        long long Si1 = (long long)(i + 1) * (2LL * ngeom - 2 - i) / 2;
        while (i < ngeom - 2 && Si1 <= (long long)pair_idx) {
            i++;
            Si = Si1;
            Si1 = (long long)(i + 1) * (2LL * ngeom - 2 - i) / 2;
        }
        j = i + 1 + (int)((long long)pair_idx - Si);
    }

    int body_i = geom_body[i];
    int body_j = geom_body[j];
    // Static (env-independent) pair filters.  With MASK they were evaluated
    // once, by this exact code in kernel_broadphase_pair_mask, and a lane
    // whose pair fails them already returned above.
    if constexpr (!MASK) {
    if (body_i == body_j) return;

    // MuJoCo weld-based filtering: bodies welded together act as one rigid body
    if (body_weldid) {
        int weld_i = body_weldid[body_i];
        int weld_j = body_weldid[body_j];
        if (weld_i == weld_j) return;

        // filterparent: skip parent-child weld body pairs (MuJoCo convention).
        // Exempt world body (body 0): MuJoCo never filters world-body contacts
        // via filterparent because world is always a valid collision partner.
        if (body_parent && weld_i > 0 && weld_j > 0) {
            int pw_i = body_weldid[body_parent[weld_i]];
            int pw_j = body_weldid[body_parent[weld_j]];
            if (pw_i == weld_j || pw_j == weld_i) return;
        }
    }

    if ((geom_contype[i] & geom_conaffinity[j]) == 0 &&
        (geom_contype[j] & geom_conaffinity[i]) == 0) return;

    if (geom_contype[i] == 0 && geom_conaffinity[i] == 0) return;
    if (geom_contype[j] == 0 && geom_conaffinity[j] == 0) return;

    if (npair_filter > 0 && pair_filter) {
        int b_lo = (body_i < body_j) ? body_i : body_j;
        int b_hi = (body_i < body_j) ? body_j : body_i;
        int key = (b_lo << 16) | b_hi;
        int lo = 0, hi = npair_filter - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int val = pair_filter[mid];
            if (val == key) return;
            if (val < key) lo = mid + 1; else hi = mid - 1;
        }
    }

    }

    // AABB overlap (expanded by global margin + per-geom margin)
    int aabb_i = env * ngeom + i;
    int aabb_j = env * ngeom + j;
    AABB ai = geom_aabb[aabb_i];
    AABB aj = geom_aabb[aabb_j];
    float margin_i = margin + (geom_margin ? geom_margin[i] : 0.0f);
    float margin_j = margin + (geom_margin ? geom_margin[j] : 0.0f);
    ai.min_x -= margin_i; ai.min_y -= margin_i; ai.min_z -= margin_i;
    ai.max_x += margin_i; ai.max_y += margin_i; ai.max_z += margin_i;
    aj.min_x -= margin_j; aj.min_y -= margin_j; aj.min_z -= margin_j;
    aj.max_x += margin_j; aj.max_y += margin_j; aj.max_z += margin_j;
    if (!aabb_overlap(ai, aj)) return;

    int pi3 = env * ngeom * 3 + i * 3;
    int pj3 = env * ngeom * 3 + j * 3;
    int qi4 = env * ngeom * 4 + i * 4;
    int qj4 = env * ngeom * 4 + j * 4;

    double pi_d[3] = {geom_xpos_d[pi3], geom_xpos_d[pi3+1], geom_xpos_d[pi3+2]};
    double pj_d[3] = {geom_xpos_d[pj3], geom_xpos_d[pj3+1], geom_xpos_d[pj3+2]};
    double qi_d[4] = {geom_xquat_d[qi4], geom_xquat_d[qi4+1], geom_xquat_d[qi4+2], geom_xquat_d[qi4+3]};
    double qj_d[4] = {geom_xquat_d[qj4], geom_xquat_d[qj4+1], geom_xquat_d[qj4+2], geom_xquat_d[qj4+3]};

    int ti = geom_type[i], tj = geom_type[j];
    double si_d[3] = {geom_size_d[i*3], geom_size_d[i*3+1], geom_size_d[i*3+2]};
    double sj_d[3] = {geom_size_d[j*3], geom_size_d[j*3+1], geom_size_d[j*3+2]};

    int type_lo = (ti <= tj) ? ti : tj;
    int type_hi = (ti <= tj) ? tj : ti;
    bool swapped = (ti > tj);

    // Value selects (not pointer selects of private arrays), so the pair data
    // stays in registers instead of being forced to scratch.
    double plo[3], phi_p[3], qlo[4], qhi[4], slo[3], shi[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        plo[k] = swapped ? pj_d[k] : pi_d[k];  phi_p[k] = swapped ? pi_d[k] : pj_d[k];
        slo[k] = swapped ? sj_d[k] : si_d[k];  shi[k]   = swapped ? si_d[k] : sj_d[k];
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
        qlo[k] = swapped ? qj_d[k] : qi_d[k];  qhi[k]   = swapped ? qi_d[k] : qj_d[k];
    }

    double fri_i = geom_friction_d ? geom_friction_d[i * 3] : 1.0;
    double fri_j = geom_friction_d ? geom_friction_d[j * 3] : 1.0;
    double combined_friction = friction_combine ? 0.5 * (fri_i + fri_j) : fmax(fri_i, fri_j);

    double pair_margin = 0.0;
    if (geom_margin) {
        // MuJoCo 3.2.6 / 3.3.2 mj_collideGeoms: margin = max(geom_margin[g1], geom_margin[g2])
        // (gap not modelled).
        pair_margin = fmax((double)geom_margin[i], (double)geom_margin[j]);
    }

    // MuJoCo convex path: defer box-mesh / mesh-mesh pairs to kernel_convex_mj
    // (mjc_Convex port) when the MuJoCo mesh data was provided. Queue overflow falls
    // through to the legacy narrowphase below.
    // Cylinder pairs with box / mesh / cylinder are mjc_Convex in MuJoCo 3.3.2 as well (mjCOLLISIONFUNC);
    // the legacy narrowphase has no cylinder collider, so without this they produce no contacts at all
    // (e.g. RoboCasa door / drawer handles vs the hand).
    const bool cyl_pair = (ti == GEOM_CYLINDER || tj == GEOM_CYLINDER) &&
        (ti == GEOM_CYLINDER || ti == GEOM_BOX || ti == GEOM_MESH) &&
        (tj == GEOM_CYLINDER || tj == GEOM_BOX || tj == GEOM_MESH);
    if (mjc_queue && geom_mj_mesh &&
        ((type_lo == GEOM_BOX && type_hi == GEOM_MESH) || (type_lo == GEOM_MESH && type_hi == GEOM_MESH) || cyl_pair) &&
        (ti != GEOM_MESH || geom_mj_mesh[i] >= 0) && (tj != GEOM_MESH || geom_mj_mesh[j] >= 0)) {
        int q = atomicAdd(mjc_queue_count, 1);
        if (q < mjc_queue_cap) {
            mjc_queue[q * 3 + 0] = env;
            mjc_queue[q * 3 + 1] = i;
            mjc_queue[q * 3 + 2] = j;
            return;
        }
    }

    // Multi-contact output buffers (double precision)
    double mc_pos[MAX_PAIR_CONTACTS][3];
    double mc_norm[MAX_PAIR_CONTACTS][3];
    double mc_depth[MAX_PAIR_CONTACTS];
    int n_hits = 0;

    // ── Dispatch narrowphase (double precision) ─────────────────────────

    if (type_hi == GEOM_PLANE) {
        if (type_lo == GEOM_SPHERE) {
            double cpos[3], cnorm[3], cdepth;
            if (collide_sphere_plane_d(plo, slo[0], phi_p, qhi, cpos, cnorm, &cdepth, pair_margin)) {
                for (int k = 0; k < 3; k++) { mc_pos[0][k] = cpos[k]; mc_norm[0][k] = cnorm[k]; }
                mc_depth[0] = cdepth;
                n_hits = 1;
            }
        }
        else if (type_lo == GEOM_BOX) {
            // Register-resident box-plane path including the contact
            // emission tail.  Same operations and order as
            // collide_box_plane_multi_d (+ its identity manifold_select_4_d,
            // ncand <= 8 == MAX_PAIR_CONTACTS) routed through mc_* and the
            // shared tail below; corner c is emitted iff !(depth_c < 0) and
            // emitted contacts keep corner order.
            double local_up[3] = {0, 0, 1};
            double normal[3];
            quat_rotate_vec_d(qhi, local_up, normal);
            double cn[3] = {-normal[0], -normal[1], -normal[2]};
            const double corner_signs[8][3] = {
                {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}
            };
            double corner_depths[8], corner_pos[8][3];
            int nbelow = 0;
#pragma unroll
            for (int c = 0; c < 8; c++) {
                double local[3] = {corner_signs[c][0]*slo[0], corner_signs[c][1]*slo[1], corner_signs[c][2]*slo[2]};
                double wc[3];
                quat_rotate_vec_d(qlo, local, wc);
                corner_pos[c][0]=plo[0]+wc[0]; corner_pos[c][1]=plo[1]+wc[1]; corner_pos[c][2]=plo[2]+wc[2];
                double diff[3]; vec3_sub_d(corner_pos[c], phi_p, diff);
                double d = vec3_dot_d(diff, normal);
                corner_depths[c] = pair_margin - d;
                if (d <= pair_margin) nbelow++;
            }
            if (nbelow == 0) return;
            bool emit[8];
            int nh = 0;
            double first_depth = 0.0;
#pragma unroll
            for (int c = 7; c >= 0; c--) {
                emit[c] = !(corner_depths[c] < 0);
                if (emit[c]) { nh++; first_depth = corner_depths[c]; }
            }
            if (nh <= 0) return;
#pragma unroll
            for (int c = 0; c < 8; c++) {
                double half_d = corner_depths[c] * 0.5;
                corner_pos[c][0] = corner_pos[c][0] + normal[0] * half_d;
                corner_pos[c][1] = corner_pos[c][1] + normal[1] * half_d;
                corner_pos[c][2] = corner_pos[c][2] + normal[2] * half_d;
            }
            double sn[3];
#pragma unroll
            for (int k = 0; k < 3; k++) sn[k] = swapped ? -cn[k] : cn[k];
            bool merged = false;
            double mpos[3] = {0.0, 0.0, 0.0};
            if (nh > 1 && body_parent) {
                int par_i = body_parent[body_i];
                int par_j = body_parent[body_j];
                if (par_i == par_j && par_i > 0) {
                    bool all_same_depth = true;
#pragma unroll
                    for (int c = 0; c < 8; c++)
                        if (emit[c] && dev_abs_d(corner_depths[c] - first_depth) > 1e-5) all_same_depth = false;
                    if (all_same_depth) {
                        double cx = 0, cy = 0, cz = 0;
#pragma unroll
                        for (int c = 0; c < 8; c++)
                            if (emit[c]) { cx += corner_pos[c][0]; cy += corner_pos[c][1]; cz += corner_pos[c][2]; }
                        double inv = 1.0 / nh;
                        mpos[0] = cx * inv; mpos[1] = cy * inv; mpos[2] = cz * inv;
                        merged = true;
                        nh = 1;
                    }
                }
            }
            if (max_contacts_per_body_pair > 0 && body_pair_ncon) {
                int b_lo = (body_i < body_j) ? body_i : body_j;
                int b_hi = (body_i < body_j) ? body_j : body_i;
                int bp_idx = env * nbody * nbody + b_lo * nbody + b_hi;
                int prev = atomicAdd(&body_pair_ncon[bp_idx], nh);
                if (prev >= max_contacts_per_body_pair) {
                    atomicAdd(&body_pair_ncon[bp_idx], -nh);
                    return;
                }
                if (prev + nh > max_contacts_per_body_pair) {
                    nh = max_contacts_per_body_pair - prev;
                }
            }
            int slot = atomicAdd(&ncon[env], nh);
            if (slot + nh > MAX_CONTACTS) {
                atomicAdd(&ncon[env], -nh);
                return;
            }
            ContactPair* cbase = contacts + env * MAX_CONTACTS + slot;
            if (merged) {
                ContactPair& cp = cbase[0];
                cp.geom1 = i; cp.geom2 = j; cp.body1 = body_i; cp.body2 = body_j;
                cp.pos[0] = mpos[0]; cp.pos[1] = mpos[1]; cp.pos[2] = mpos[2];
                cp.normal[0] = sn[0]; cp.normal[1] = sn[1]; cp.normal[2] = sn[2];
                cp.depth = first_depth;
                cp.friction = combined_friction;
                return;
            }
            int k = 0;
#pragma unroll
            for (int c = 0; c < 8; c++) {
                if (emit[c] && k < nh) {
                    ContactPair& cp = cbase[k];
                    cp.geom1 = i; cp.geom2 = j; cp.body1 = body_i; cp.body2 = body_j;
                    cp.pos[0] = corner_pos[c][0]; cp.pos[1] = corner_pos[c][1]; cp.pos[2] = corner_pos[c][2];
                    cp.normal[0] = sn[0]; cp.normal[1] = sn[1]; cp.normal[2] = sn[2];
                    cp.depth = corner_depths[c];
                    cp.friction = combined_friction;
                    k++;
                }
            }
            return;
        }
        else if (type_lo == GEOM_CAPSULE) {
            n_hits = collide_capsule_plane_multi_d(plo, qlo, slo[0], slo[1], phi_p, qhi,
                                                    mc_pos, mc_norm, mc_depth, pair_margin);
        }
        else if (HULLS && type_lo == GEOM_MESH && geom_meshid && mesh_convex_vert) {
            // Mesh collision stays float for now (convert hull verts is complex)
            float mc_pos_f[MAX_PAIR_CONTACTS][3], mc_norm_f[MAX_PAIR_CONTACTS][3], mc_depth_f[MAX_PAIR_CONTACTS];
            float plo_f[3]={(float)plo[0],(float)plo[1],(float)plo[2]};
            float qlo_f[4]={(float)qlo[0],(float)qlo[1],(float)qlo[2],(float)qlo[3]};
            float phi_f[3]={(float)phi_p[0],(float)phi_p[1],(float)phi_p[2]};
            float qhi_f[4]={(float)qhi[0],(float)qhi[1],(float)qhi[2],(float)qhi[3]};
            int lo_geom = swapped ? j : i;
            int mid = geom_meshid[lo_geom];
            if (mid >= 0 && mesh_convex_vert_num && mesh_convex_vert_num[mid] > 0) {
                n_hits = collide_mesh_plane(plo_f, qlo_f, phi_f, qhi_f,
                    mesh_convex_vert_adr[mid], mesh_convex_vert_num[mid],
                    mesh_convex_vert, mc_pos_f, mc_norm_f, mc_depth_f);
                for (int c=0;c<n_hits;c++) {
                    for (int k=0;k<3;k++){mc_pos[c][k]=(double)mc_pos_f[c][k]; mc_norm[c][k]=(double)mc_norm_f[c][k];}
                    mc_depth[c]=(double)mc_depth_f[c];
                }
            }
        }
    }
    else if (type_lo == GEOM_SPHERE && type_hi == GEOM_SPHERE) {
        double cpos[3], cnorm[3], cdepth;
        if (collide_sphere_sphere_d(plo, slo[0], phi_p, shi[0], cpos, cnorm, &cdepth, pair_margin)) {
            for (int k = 0; k < 3; k++) { mc_pos[0][k] = cpos[k]; mc_norm[0][k] = cnorm[k]; }
            mc_depth[0] = cdepth; n_hits = 1;
        }
    }
    else if (type_lo == GEOM_SPHERE && type_hi == GEOM_BOX) {
        double cpos[3], cnorm[3], cdepth;
        if (collide_sphere_box_d(plo, slo[0], phi_p, qhi, shi, cpos, cnorm, &cdepth, pair_margin)) {
            for (int k = 0; k < 3; k++) { mc_pos[0][k] = cpos[k]; mc_norm[0][k] = cnorm[k]; }
            mc_depth[0] = cdepth; n_hits = 1;
        }
    }
    else if (type_lo == GEOM_SPHERE && type_hi == GEOM_CAPSULE) {
        double cpos[3], cnorm[3], cdepth;
        if (collide_sphere_capsule_d(plo, slo[0], phi_p, qhi, shi[0], shi[1], cpos, cnorm, &cdepth, pair_margin)) {
            for (int k = 0; k < 3; k++) { mc_pos[0][k] = cpos[k]; mc_norm[0][k] = cnorm[k]; }
            mc_depth[0] = cdepth; n_hits = 1;
        }
    }
    else if (type_lo == GEOM_BOX && type_hi == GEOM_BOX) {
        n_hits = collide_box_box_full_d(plo, qlo, slo, phi_p, qhi, shi,
                                         mc_pos, mc_norm, mc_depth, pair_margin);
    }
    else if (type_lo == GEOM_BOX && type_hi == GEOM_CAPSULE) {
        n_hits = collide_box_capsule_multi_d(plo, qlo, slo, phi_p, qhi, shi[0], shi[1],
                                              mc_pos, mc_norm, mc_depth, pair_margin);
    }
    else if (type_lo == GEOM_CAPSULE && type_hi == GEOM_CAPSULE) {
        double la[3]={0,0,1}, ax_lo[3], ax_hi[3];
        quat_rotate_vec_d(qlo, la, ax_lo);
        quat_rotate_vec_d(qhi, la, ax_hi);
        double ep_lo[2][3], ep_hi[2][3];
        for (int k=0;k<3;k++){
            ep_lo[0][k]=plo[k]-ax_lo[k]*slo[1]; ep_lo[1][k]=plo[k]+ax_lo[k]*slo[1];
            ep_hi[0][k]=phi_p[k]-ax_hi[k]*shi[1]; ep_hi[1][k]=phi_p[k]+ax_hi[k]*shi[1];
        }
        double cand_pos_d[MAX_PAIR_CONTACTS][3], cand_norm_d[MAX_PAIR_CONTACTS][3], cand_depth_d[MAX_PAIR_CONTACTS];
        int ncand=0;
        for (int e=0; e<2 && ncand<MAX_PAIR_CONTACTS; e++){
            double closest[3];
            closest_point_on_segment_d(ep_lo[e], ep_hi[0], ep_hi[1], closest);
            double cpos[3], cnorm[3], cdepth;
            if (collide_sphere_sphere_d(ep_lo[e], slo[0], closest, shi[0], cpos, cnorm, &cdepth, pair_margin)){
                for(int k=0;k<3;k++){cand_pos_d[ncand][k]=cpos[k]; cand_norm_d[ncand][k]=cnorm[k];}
                cand_depth_d[ncand]=cdepth; ncand++;
            }
        }
        for (int e=0; e<2 && ncand<MAX_PAIR_CONTACTS; e++){
            double closest[3];
            closest_point_on_segment_d(ep_hi[e], ep_lo[0], ep_lo[1], closest);
            double cpos[3], cnorm[3], cdepth;
            if (collide_sphere_sphere_d(closest, slo[0], ep_hi[e], shi[0], cpos, cnorm, &cdepth, pair_margin)){
                bool dup=false;
                for(int c=0;c<ncand;c++){if(vec3_dist_d(cpos,cand_pos_d[c])<1e-10){dup=true;break;}}
                if (dup) continue;
                for(int k=0;k<3;k++){cand_pos_d[ncand][k]=cpos[k]; cand_norm_d[ncand][k]=cnorm[k];}
                cand_depth_d[ncand]=cdepth; ncand++;
            }
        }
        if (ncand == 0) {
            double cpos[3], cnorm[3], cdepth;
            if (collide_capsule_capsule_d(plo, qlo, slo[0], slo[1],
                                          phi_p, qhi, shi[0], shi[1],
                                          cpos, cnorm, &cdepth, pair_margin)) {
                for(int k=0;k<3;k++){cand_pos_d[0][k]=cpos[k]; cand_norm_d[0][k]=cnorm[k];}
                cand_depth_d[0]=cdepth; ncand=1;
            }
        }
        n_hits=(ncand<MAX_PAIR_CONTACTS)?ncand:MAX_PAIR_CONTACTS;
        for(int c=0;c<n_hits;c++){
            for(int k=0;k<3;k++){mc_pos[c][k]=cand_pos_d[c][k]; mc_norm[c][k]=cand_norm_d[c][k];}
            mc_depth[c]=cand_depth_d[c];
        }
    }

    // ── GEOM_MESH dispatch (float fallback — hull data is float) ───────
    if (HULLS && n_hits == 0 && (ti == GEOM_MESH || tj == GEOM_MESH) && geom_meshid && mesh_convex_vert) {
        int mesh_geom = (ti == GEOM_MESH) ? i : j;
        int other_geom = (ti == GEOM_MESH) ? j : i;
        int other_type = (ti == GEOM_MESH) ? tj : ti;
        float mp[3], mq[4], op[3], oq[4], os[3];
        if (ti == GEOM_MESH) {
            for(int k=0;k<3;k++){mp[k]=(float)pi_d[k]; op[k]=(float)pj_d[k];}
            for(int k=0;k<4;k++){mq[k]=(float)qi_d[k]; oq[k]=(float)qj_d[k];}
            for(int k=0;k<3;k++) os[k]=(float)sj_d[k];
        } else {
            for(int k=0;k<3;k++){mp[k]=(float)pj_d[k]; op[k]=(float)pi_d[k];}
            for(int k=0;k<4;k++){mq[k]=(float)qj_d[k]; oq[k]=(float)qi_d[k];}
            for(int k=0;k<3;k++) os[k]=(float)si_d[k];
        }

        int mid = geom_meshid[mesh_geom];
        if (mid >= 0 && mesh_convex_vert_num[mid] > 0) {
            int va = mesh_convex_vert_adr[mid], vn = mesh_convex_vert_num[mid];
            int fa = mesh_convex_face_adr[mid], fn = mesh_convex_face_num[mid];
            int ea = mesh_convex_edge_adr[mid], en = mesh_convex_edge_num[mid];

            if (other_type == GEOM_PLANE) {
                float mc_pf[MAX_PAIR_CONTACTS][3], mc_nf[MAX_PAIR_CONTACTS][3], mc_df[MAX_PAIR_CONTACTS];
                n_hits = collide_mesh_plane(mp, mq, op, oq, va, vn,
                                            mesh_convex_vert, mc_pf, mc_nf, mc_df);
                for(int c=0;c<n_hits;c++){
                    for(int k=0;k<3;k++){mc_pos[c][k]=(double)mc_pf[c][k]; mc_norm[c][k]=(double)mc_nf[c][k];}
                    mc_depth[c]=(double)mc_df[c];
                }
                if (ti != GEOM_MESH) {
                    for (int c = 0; c < n_hits; c++) {
                        mc_norm[c][0] = -mc_norm[c][0]; mc_norm[c][1] = -mc_norm[c][1]; mc_norm[c][2] = -mc_norm[c][2];
                    }
                }
                swapped = false;
            }
            else if (other_type == GEOM_MESH) {
                int mid2 = geom_meshid[other_geom];
                if (mid2 >= 0 && mesh_convex_vert_num[mid2] > 0) {
                    int va2 = mesh_convex_vert_adr[mid2], vn2 = mesh_convex_vert_num[mid2];
                    int fa2 = mesh_convex_face_adr[mid2], fn2 = mesh_convex_face_num[mid2];
                    int ea2 = mesh_convex_edge_adr[mid2], en2 = mesh_convex_edge_num[mid2];
                    float pi_f[3]={(float)pi_d[0],(float)pi_d[1],(float)pi_d[2]};
                    float pj_f[3]={(float)pj_d[0],(float)pj_d[1],(float)pj_d[2]};
                    float qi_f[4]={(float)qi_d[0],(float)qi_d[1],(float)qi_d[2],(float)qi_d[3]};
                    float qj_f[4]={(float)qj_d[0],(float)qj_d[1],(float)qj_d[2],(float)qj_d[3]};
                    float mc_pos_f[MAX_PAIR_CONTACTS][3], mc_norm_f[MAX_PAIR_CONTACTS][3], mc_depth_f[MAX_PAIR_CONTACTS];
                    n_hits = collide_convex_convex(
                        pi_f, qi_f, mesh_convex_vert_adr[geom_meshid[i]], mesh_convex_vert_num[geom_meshid[i]],
                        mesh_convex_face_adr[geom_meshid[i]], mesh_convex_face_num[geom_meshid[i]],
                        mesh_convex_edge_adr[geom_meshid[i]], mesh_convex_edge_num[geom_meshid[i]],
                        pj_f, qj_f, mesh_convex_vert_adr[geom_meshid[j]], mesh_convex_vert_num[geom_meshid[j]],
                        mesh_convex_face_adr[geom_meshid[j]], mesh_convex_face_num[geom_meshid[j]],
                        mesh_convex_edge_adr[geom_meshid[j]], mesh_convex_edge_num[geom_meshid[j]],
                        mesh_convex_vert, mesh_convex_face_normal,
                        mesh_convex_face, mesh_convex_edge,
                        mc_pos_f, mc_norm_f, mc_depth_f);
                    for(int c=0;c<n_hits;c++){
                        for(int k=0;k<3;k++){mc_pos[c][k]=(double)mc_pos_f[c][k]; mc_norm[c][k]=(double)mc_norm_f[c][k];}
                        mc_depth[c]=(double)mc_depth_f[c];
                    }
                    swapped = false;
                }
            }
            else if (other_type == GEOM_BOX) {
                float mc_bf[MAX_PAIR_CONTACTS][3], mc_bnf[MAX_PAIR_CONTACTS][3], mc_bdf[MAX_PAIR_CONTACTS];
                n_hits = collide_mesh_box(mp, mq, op, oq, os,
                    va, vn, mesh_convex_vert,
                    fa, fn, mesh_convex_face_normal, mesh_convex_face,
                    mc_bf, mc_bnf, mc_bdf, (float)pair_margin);
                for(int c=0;c<n_hits;c++){
                    for(int k=0;k<3;k++){mc_pos[c][k]=(double)mc_bf[c][k]; mc_norm[c][k]=(double)mc_bnf[c][k];}
                    mc_depth[c]=(double)mc_bdf[c];
                }
                if (ti != GEOM_MESH) {
                    for(int c=0;c<n_hits;c++){
                        mc_norm[c][0]=-mc_norm[c][0]; mc_norm[c][1]=-mc_norm[c][1]; mc_norm[c][2]=-mc_norm[c][2];
                    }
                }
                swapped = false;
            }
            else if (other_type == GEOM_SPHERE) {
                float neg_dir[3] = {mp[0]-op[0], mp[1]-op[1], mp[2]-op[2]};
                vec3_normalize(neg_dir);
                float closest[3];
                support_hull(mesh_convex_vert, va, vn, mp, mq, neg_dir, closest);
                float diff[3]; vec3_sub(op, closest, diff);
                float dist = vec3_len(diff);
                if (dist < os[0]) {
                    double dep = (double)os[0] - (double)dist;
                    mc_depth[0] = dep;
                    if (dist > 1e-10f) {
                        mc_norm[0][0]=(double)diff[0]/(double)dist; mc_norm[0][1]=(double)diff[1]/(double)dist; mc_norm[0][2]=(double)diff[2]/(double)dist;
                    } else {
                        mc_norm[0][0]=0; mc_norm[0][1]=0; mc_norm[0][2]=1;
                    }
                    mc_pos[0][0]=(double)closest[0]+mc_norm[0][0]*dep*0.5;
                    mc_pos[0][1]=(double)closest[1]+mc_norm[0][1]*dep*0.5;
                    mc_pos[0][2]=(double)closest[2]+mc_norm[0][2]*dep*0.5;
                    n_hits = 1;
                    if (ti != GEOM_MESH) {
                        mc_norm[0][0]=-mc_norm[0][0]; mc_norm[0][1]=-mc_norm[0][1]; mc_norm[0][2]=-mc_norm[0][2];
                    }
                    swapped = false;
                }
            }
        }
    }

    if (n_hits <= 0) return;

    // Flip normals for swapped geom pairs (make normal point from geom i to geom j)
    if (swapped) {
        for (int c = 0; c < n_hits; c++) {
            mc_norm[c][0] = -mc_norm[c][0];
            mc_norm[c][1] = -mc_norm[c][1];
            mc_norm[c][2] = -mc_norm[c][2];
        }
    }

    // Merge redundant box-box contacts for sibling bodies in the same mechanism.
    // When two boxes share the same non-world parent body and produce multiple
    // contacts at the same depth (degenerate face-face overlap), collapse to a
    // single centroid contact. This matches MuJoCo's behavior for e.g. cabinet
    // drawer-drawer contacts while preserving multi-contact for free bodies.
    if (n_hits > 1 && body_parent) {
        int par_i = body_parent[body_i];
        int par_j = body_parent[body_j];
        if (par_i == par_j && par_i > 0) {
            bool all_same_depth = true;
            for (int c = 1; c < n_hits; c++) {
                if (dev_abs_d(mc_depth[c] - mc_depth[0]) > 1e-5) {
                    all_same_depth = false;
                    break;
                }
            }
            if (all_same_depth) {
                double cx = 0, cy = 0, cz = 0;
                for (int c = 0; c < n_hits; c++) {
                    cx += mc_pos[c][0]; cy += mc_pos[c][1]; cz += mc_pos[c][2];
                }
                double inv = 1.0 / n_hits;
                mc_pos[0][0] = cx * inv; mc_pos[0][1] = cy * inv; mc_pos[0][2] = cz * inv;
                n_hits = 1;
            }
        }
    }

    // Per-body-pair contact limit: prevents convex decomposition geoms from
    // generating 100+ contacts per object (MuJoCo uses mesh collision → ~4 contacts)
    if (max_contacts_per_body_pair > 0 && body_pair_ncon) {
        int b_lo = (body_i < body_j) ? body_i : body_j;
        int b_hi = (body_i < body_j) ? body_j : body_i;
        int bp_idx = env * nbody * nbody + b_lo * nbody + b_hi;
        int prev = atomicAdd(&body_pair_ncon[bp_idx], n_hits);
        if (prev >= max_contacts_per_body_pair) {
            atomicAdd(&body_pair_ncon[bp_idx], -n_hits);
            return;
        }
        if (prev + n_hits > max_contacts_per_body_pair) {
            n_hits = max_contacts_per_body_pair - prev;
        }
    }

    // Store contacts
    int slot = atomicAdd(&ncon[env], n_hits);
    if (slot + n_hits > MAX_CONTACTS) {
        atomicAdd(&ncon[env], -n_hits);
        return;
    }

    for (int c = 0; c < n_hits; c++) {
        ContactPair& cp = contacts[env * MAX_CONTACTS + slot + c];
        cp.geom1 = i;
        cp.geom2 = j;
        cp.body1 = body_i;
        cp.body2 = body_j;
        cp.pos[0] = mc_pos[c][0]; cp.pos[1] = mc_pos[c][1]; cp.pos[2] = mc_pos[c][2];
        cp.normal[0] = mc_norm[c][0]; cp.normal[1] = mc_norm[c][1]; cp.normal[2] = mc_norm[c][2];
        cp.depth = mc_depth[c];
        cp.friction = combined_friction;
    }
}

// ── MuJoCo convex queue kernel ──────────────────────────────────────────────
// One worker per thread; each worker owns one MjcWorkerScratch and loops over the queue.
// Geom order is MuJoCo's: for box-mesh the box is g1 (MuJoCo's type order BOX < MESH, same
// as the GPU simulator's GEOM_BOX < GEOM_MESH); for mesh-mesh the lower geom index is g1.
// Launch mapping: each wave runs `ppw` workers on its lanes [0, ppw) (ppw = 1 -> one queued
// pair per wave, so a pair never executes the union of other pairs' divergent GJK/EPA/multiCCD
// paths), BLOCK/64 waves per block. Each pair's contacts are staged in its queue slot and
// emitted by kernel_convex_mj_emit in queue-index order (the order the single-wave atomics of
// the original one-thread-per-pair launch produced), so the contact order is deterministic.
// MuJoCo mj_filterSphere for two non-plane geoms (mj_collideGeoms runs it before every narrowphase):
// skip when the geom centres (MuJoCo's geom frames) are farther apart than rbound1 + rbound2 + margin.
// MuJoCo mjtGeom value of a GPU geom type (mjc_Convex orders g1 / g2 by MuJoCo type)
HIPGL_DEVICE inline int mj_geom_type(int t) {
    return t == GEOM_CYLINDER ? 5 : t == GEOM_BOX ? 6 : t == GEOM_MESH ? 7 : t == GEOM_SPHERE ? 2 : t == GEOM_CAPSULE ? 3 : 0;
}

HIPGL_DEVICE inline bool mjc_sphere_far(const double* p1, const double* p2, double rb1, double rb2, double margin)
{
    if (!(rb1 > 0.0 && rb2 > 0.0)) return false;
    const double dx = p1[0] - p2[0], dy = p1[1] - p2[1], dz = p1[2] - p2[2];
    const double bound = rb1 + rb2 + margin;
    return dx * dx + dy * dy + dz * dz > bound * bound;
}

template <int BLOCK>
HIPGL_GLOBAL void __launch_bounds__(BLOCK) kernel_convex_mj(
    int ngeom, int nbody,
    const int*    __restrict__ geom_type,
    const int*    __restrict__ geom_body,
    const double* __restrict__ geom_friction_d,
    int friction_combine,
    const float*  __restrict__ geom_margin,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const int*    __restrict__ geom_mj_mesh,
    const double* __restrict__ geom_mj_pos,
    const double* __restrict__ geom_mj_quat,
    const double* __restrict__ geom_mj_size,
    const double* __restrict__ geom_mj_rbound,
    mjccd::MeshModel mm,
    mjccd::Config cfg,
    const int*    __restrict__ mjc_queue,
    const int*    __restrict__ mjc_queue_count,
    int mjc_queue_cap,
    MjcWorkerScratch* __restrict__ scratch,
    int workers,
    int ppw,
    int*          __restrict__ stage_n,
    ContactPair*  __restrict__ stage_c,
    int*          __restrict__ qhist,
    int*          __restrict__ fallback_count,
    const int*    __restrict__ geom_onesided)
{
    const int lane = (int)threadIdx.x & 63;
    // ppw == 1 (64-lane blocks): the whole wave stays alive and runs the pair together (lane 0 owns the result
    // and every write; lanes 1..63 help in the wave-cooperative parts of mjccd::convex)
    const bool coop = (ppw == 1 && BLOCK == 64);
    if (!coop && lane >= ppw) return;
    const bool owner = !coop || lane == 0;
    const int wave = (int)(blockIdx.x * (BLOCK / 64) + (threadIdx.x >> 6));
    int w = coop ? wave : wave * ppw + lane;
    if (w >= workers) return;
    int nq = *mjc_queue_count;
    if (qhist && w == 0 && owner) atomicAdd(&qhist[nq < MJC_QHIST_BINS - 1 ? nq : MJC_QHIST_BINS - 1], 1);
    if (nq > mjc_queue_cap) nq = mjc_queue_cap;
    MjcWorkerScratch* ws = scratch + w;

    for (int q = w; q < nq; q += workers) {
        int env = mjc_queue[q * 3 + 0];
        int i   = mjc_queue[q * 3 + 1];
        int j   = mjc_queue[q * 3 + 2];
        int ti = geom_type[i], tj = geom_type[j];
        bool swapped = (mj_geom_type(ti) > mj_geom_type(tj));   // g1 = lower MuJoCo type (cylinder, box, mesh)
        int g1 = swapped ? j : i, g2 = swapped ? i : j;

        mjccd::Obj o[2];
        int gg[2] = {g1, g2};
        for (int k = 0; k < 2; k++) {
            int g = gg[k];
            int b = geom_body[g];
            double xpos[3], xmat[9];
            mjpose::local2Global(xpos, xmat, xpos_d + (env * nbody + b) * 3, xquat_d + (env * nbody + b) * 4,
                                 geom_mj_pos + g * 3, geom_mj_quat + g * 4);
            if (geom_type[g] == GEOM_BOX) {
                mjccd::initBoxObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3, 0.0);
            } else if (geom_type[g] == GEOM_CYLINDER) {
                mjccd::initCylinderObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3, 0.0);
            } else {
                int id = geom_mj_mesh[g];
                int vadr = mm.mesh_vertadr[id], padr = mm.mesh_polyadr[id];
                mjccd::initMeshObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3,
                                   mm.mesh_vertnum[id], mm.mesh_vert + 3 * vadr,
                                   mjccd::meshGraph(mm.mesh_graph, mm.mesh_graphadr[id]),
                                   mm.mesh_polynum[id],
                                   mm.mesh_polymapadr + vadr, mm.mesh_polymapnum + vadr, mm.mesh_polymap,
                                   mm.mesh_polynormal + 3 * padr,
                                   mm.mesh_polyvertadr + padr, mm.mesh_polyvertnum + padr, mm.mesh_polyvert, 0.0);
            }
        }
        // One-sided static box (instinct_sim_onesided_geoms, PhysX triangle-mesh semantics): a mesh
        // hull whose center (MuJoCo's recentered geom frame) is below the box's top face gets no
        // contact, so an object spawned with a hull deep in the tabletop sinks onto its other
        // hulls instead of being pushed out.
        if (geom_onesided && geom_type[g1] == GEOM_BOX && geom_onesided[g1]) {
            const double* R = o[0].mat;
            double d0 = o[1].pos[0] - o[0].pos[0], d1 = o[1].pos[1] - o[0].pos[1], d2 = o[1].pos[2] - o[0].pos[2];
            if (R[2] * d0 + R[5] * d1 + R[8] * d2 < geom_mj_size[g1 * 3 + 2]) { if (owner) stage_n[q] = 0; continue; }
        }
        // MuJoCo 3.3.2 mj_collideGeoms: margin = max of the two geom margins (gap not modelled)
        double margin = geom_margin ? fmax((double)geom_margin[i], (double)geom_margin[j]) : 0.0;
        if (mjc_sphere_far(o[0].pos, o[1].pos, geom_mj_rbound[g1], geom_mj_rbound[g2], margin)) {
            if (owner) stage_n[q] = 0;
            continue;
        }

        double cpos[MJCCD_STATUS_NX][3], cnorm[MJCCD_STATUS_NX][3], cdist[MJCCD_STATUS_NX];
        int n = mjccd::convex(o[0], o[1], margin, cfg, &ws->scratch, cpos, cnorm, cdist, &ws->status,
                              geom_mj_rbound[g1], geom_mj_rbound[g2], coop);
        if (!owner) continue;
        if (n < 0) { atomicAdd(fallback_count, 1); stage_n[q] = 0; continue; }
        stage_n[q] = n;
        if (n == 0) continue;

        int body_i = geom_body[i], body_j = geom_body[j];
        double fri_i = geom_friction_d ? geom_friction_d[i * 3] : 1.0;
        double fri_j = geom_friction_d ? geom_friction_d[j * 3] : 1.0;
        double combined_friction = friction_combine ? 0.5 * (fri_i + fri_j) : fmax(fri_i, fri_j);
        // MuJoCo normal points g1 -> g2; the GPU simulator stores it from geom i to geom j.
        double sgn = swapped ? -1.0 : 1.0;
        ContactPair* out = stage_c + (size_t)q * MJCCD_STATUS_NX;
        for (int c = 0; c < n; c++) {
            ContactPair& cp = out[c];
            cp.geom1 = i; cp.geom2 = j; cp.body1 = body_i; cp.body2 = body_j;
            cp.pos[0] = cpos[c][0]; cp.pos[1] = cpos[c][1]; cp.pos[2] = cpos[c][2];
            cp.normal[0] = sgn * cnorm[c][0]; cp.normal[1] = sgn * cnorm[c][1]; cp.normal[2] = sgn * cnorm[c][2];
            // cdist is MuJoCo's con->dist (geometric); hip depth is penetration into the margin
            // band, margin - dist, as the primitive colliders emit and the constraint rows expect.
            cp.depth = margin - cdist[c];
            cp.friction = combined_friction;
        }
    }
}

// ── MuJoCo convex queue kernel, multiCCD fan-out ────────────────────────────
// One queued pair per workgroup of MJC_FAN_WAVES waves (lane 0 of each wave active).  Wave 0
// runs mjc_Convex's initial penetration (mjccd::convexBegin); when the margin > 0 perturbation
// loop applies, wave k runs perturbed rerun k concurrently (mjccd::multiccdRerun, own global
// scratch/status and own LDS polytope slot), and wave 0 then applies the loop's sequential
// acceptance (mjccd::multiccdMerge, bit-identical to mjccd::convex).  Contacts are staged per
// queue slot and emitted by kernel_convex_mj_emit exactly as for kernel_convex_mj.
#ifndef MJC_FAN_COOP
#define MJC_FAN_COOP 1      // whole waves run the fan-out ccds cooperatively
#endif
#ifndef MJC_FAN_LDS_STATUS
#define MJC_FAN_LDS_STATUS 1  // per-wave mjccd::Status in LDS (coop only)
#endif
struct MjcFanShared {
    mjccd::Obj obj1, obj2;
    double pos0[3], normal0[3];
    double margin;
    int pending;
    mjccd::MulticcdRerunOut r[MJC_FAN_WAVES];
};

HIPGL_DEVICE inline void mjc_stage_contacts(int q, int n, int i, int j, bool swapped,
    const int* __restrict__ geom_body, const double* __restrict__ geom_friction_d, int friction_combine,
    const double (*cpos)[3], const double (*cnorm)[3], const double* cdist, double margin,
    int* __restrict__ stage_n, ContactPair* __restrict__ stage_c)
{
    stage_n[q] = n;
    if (n == 0) return;
    int body_i = geom_body[i], body_j = geom_body[j];
    double fri_i = geom_friction_d ? geom_friction_d[i * 3] : 1.0;
    double fri_j = geom_friction_d ? geom_friction_d[j * 3] : 1.0;
    double combined_friction = friction_combine ? 0.5 * (fri_i + fri_j) : fmax(fri_i, fri_j);
    double sgn = swapped ? -1.0 : 1.0;
    ContactPair* out = stage_c + (size_t)q * MJCCD_STATUS_NX;
    for (int c = 0; c < n; c++) {
        ContactPair& cp = out[c];
        cp.geom1 = i; cp.geom2 = j; cp.body1 = body_i; cp.body2 = body_j;
        cp.pos[0] = cpos[c][0]; cp.pos[1] = cpos[c][1]; cp.pos[2] = cpos[c][2];
        cp.normal[0] = sgn * cnorm[c][0]; cp.normal[1] = sgn * cnorm[c][1]; cp.normal[2] = sgn * cnorm[c][2];
        cp.depth = margin - cdist[c];   // see kernel_convex_mj
        cp.friction = combined_friction;
    }
}

HIPGL_GLOBAL void __launch_bounds__(64 * MJC_FAN_WAVES) kernel_convex_mj_fan(
    int ngeom, int nbody,
    const int*    __restrict__ geom_type,
    const int*    __restrict__ geom_body,
    const double* __restrict__ geom_friction_d,
    int friction_combine,
    const float*  __restrict__ geom_margin,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const int*    __restrict__ geom_mj_mesh,
    const double* __restrict__ geom_mj_pos,
    const double* __restrict__ geom_mj_quat,
    const double* __restrict__ geom_mj_size,
    const double* __restrict__ geom_mj_rbound,
    mjccd::MeshModel mm,
    mjccd::Config cfg,
    const int*    __restrict__ mjc_queue,
    const int*    __restrict__ mjc_queue_count,
    int mjc_queue_cap,
    MjcWorkerScratch* __restrict__ scratch,
    int workers,
    int*          __restrict__ stage_n,
    ContactPair*  __restrict__ stage_c,
    int*          __restrict__ qhist,
    int*          __restrict__ fallback_count,
    const int*    __restrict__ geom_onesided,
    int*          __restrict__ recompute_count)
{
    __shared__ MjcFanShared sh;
    const int lane = (int)threadIdx.x & 63;
    const int wv = (int)threadIdx.x >> 6;
    const int w = (int)blockIdx.x;
    if (w >= workers) return;                       // uniform per workgroup
    int nq = *mjc_queue_count;
    if (qhist && w == 0 && threadIdx.x == 0) atomicAdd(&qhist[nq < MJC_QHIST_BINS - 1 ? nq : MJC_QHIST_BINS - 1], 1);
    if (nq > mjc_queue_cap) nq = mjc_queue_cap;
    MjcWorkerScratch* ws = scratch + (size_t)w * MJC_FAN_WAVES + wv;
    // MJC_FAN_COOP: every lane of a wave stays alive and runs that wave's ccd together (identical
    // inputs, lanes cooperate in the support / EPA scans); lane 0 of the wave owns every write
    const int coop = MJC_FAN_COOP;
    const bool active = coop || lane == 0;           // lanes running a ccd
    const bool leader = (wv == 0) && active;         // wave 0 runs begin / merge
    const bool owner0 = (threadIdx.x == 0);          // the single writer of wave-0 results
    mjccd::Status* st = &ws->status;
#if MJC_FAN_LDS_STATUS
    __shared__ mjccd::Status sh_st[MJC_FAN_WAVES];
    if (coop) st = &sh_st[wv];
#endif

    for (int q = w; q < nq; q += workers) {
        int i = 0, j = 0, g1 = 0, g2 = 0;
        bool swapped = false;
        double cpos[MJCCD_STATUS_NX][3], cnorm[MJCCD_STATUS_NX][3], cdist[MJCCD_STATUS_NX];
        double tolerance = 0;
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
        if (lane == 0) for (int kk = 0; kk < 24; kk++) mjccd::mjp_row(wv)[kk] = 0;
        long long _p_t0 = MJP_T(), _p_tb = 0, _p_tm = 0;
        long long _p_b[24] = {0};
        int _p_ty1 = 0, _p_ty2 = 0, _p_n = 0, _p_nrec = 0;
#endif
        if (owner0) sh.pending = 0;
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
        long long _p_ts0 = 0, _p_ts1 = 0;
#endif
        {
            // every thread builds the pair's objects (identical values) so that the whole
            // workgroup can stage the meshes into LDS; wave 0 then runs convexBegin on them
            int env = mjc_queue[q * 3 + 0];
            i   = mjc_queue[q * 3 + 1];
            j   = mjc_queue[q * 3 + 2];
            int ti = geom_type[i], tj = geom_type[j];
            swapped = (mj_geom_type(ti) > mj_geom_type(tj));
            g1 = swapped ? j : i; g2 = swapped ? i : j;
            mjccd::Obj o[2];
            int gg[2] = {g1, g2};
            for (int k = 0; k < 2; k++) {
                int g = gg[k];
                int b = geom_body[g];
                double xpos[3], xmat[9];
                mjpose::local2Global(xpos, xmat, xpos_d + (env * nbody + b) * 3, xquat_d + (env * nbody + b) * 4,
                                     geom_mj_pos + g * 3, geom_mj_quat + g * 4);
                if (geom_type[g] == GEOM_BOX) {
                    mjccd::initBoxObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3, 0.0);
                } else if (geom_type[g] == GEOM_CYLINDER) {
                    mjccd::initCylinderObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3, 0.0);
                } else {
                    int id = geom_mj_mesh[g];
                    int vadr = mm.mesh_vertadr[id], padr = mm.mesh_polyadr[id];
                    mjccd::initMeshObj(&o[k], g, xpos, xmat, geom_mj_size + g * 3,
                                       mm.mesh_vertnum[id], mm.mesh_vert + 3 * vadr,
                                       mjccd::meshGraph(mm.mesh_graph, mm.mesh_graphadr[id]),
                                       mm.mesh_polynum[id],
                                       mm.mesh_polymapadr + vadr, mm.mesh_polymapnum + vadr, mm.mesh_polymap,
                                       mm.mesh_polynormal + 3 * padr,
                                       mm.mesh_polyvertadr + padr, mm.mesh_polyvertnum + padr, mm.mesh_polyvert, 0.0);
                }
            }
            bool skip = false;
            if (geom_onesided && geom_type[g1] == GEOM_BOX && geom_onesided[g1]) {
                const double* R = o[0].mat;
                double d0 = o[1].pos[0] - o[0].pos[0], d1 = o[1].pos[1] - o[0].pos[1], d2 = o[1].pos[2] - o[0].pos[2];
                if (R[2] * d0 + R[5] * d1 + R[8] * d2 < geom_mj_size[g1 * 3 + 2]) { if (owner0) stage_n[q] = 0; skip = true; }
            }
            if (!skip && mjc_sphere_far(o[0].pos, o[1].pos, geom_mj_rbound[g1], geom_mj_rbound[g2],
                                        geom_margin ? fmax((double)geom_margin[i], (double)geom_margin[j]) : 0.0)) {
                if (owner0) stage_n[q] = 0;
                skip = true;
            }
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_MESH_WORDS > 0
            const bool staged = coop && !cfg.disable_nativeccd;   // uniform over the workgroup
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
            _p_ts0 = MJP_T() - _p_t0;
#endif
            if (!skip && staged) mjccd::stageMeshesLDSBlock(&o[0], &o[1]);
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
            _p_ts1 = MJP_T() - _p_t0;
#endif
#else
            const bool staged = false;
#endif
            if (!skip && leader) {
                double margin = geom_margin ? fmax((double)geom_margin[i], (double)geom_margin[j]) : 0.0;
                mjccd::Obj b1, b2;
                int n = mjccd::convexBegin(o[0], o[1], margin, cfg, &ws->scratch, cpos, cnorm, cdist, st,
                                           geom_mj_rbound[g1], geom_mj_rbound[g2], &b1, &b2,
                                           &tolerance, true, coop, staged);
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
                _p_tb = MJP_T() - _p_t0;
                for (int kk = 0; kk < 24; kk++) { _p_b[kk] = mjccd::mjp_row(wv)[kk]; }
                if (lane == 0) for (int kk = 0; kk < 24; kk++) mjccd::mjp_row(wv)[kk] = 0;
                _p_ty1 = o[0].geom_type; _p_ty2 = o[1].geom_type; _p_n = n;
#endif
                if (!owner0) {
                } else if (n == MJCCD_MULTI_PENDING) {
                    sh.obj1 = b1; sh.obj2 = b2;
                    sh.pending = 1;
                    sh.margin = margin;
                    for (int k = 0; k < 3; k++) { sh.pos0[k] = cpos[0][k]; sh.normal0[k] = cnorm[0][k]; }
                } else if (n < 0) {
                    atomicAdd(fallback_count, 1); stage_n[q] = 0;
                } else {
                    mjc_stage_contacts(q, n, i, j, swapped, geom_body, geom_friction_d, friction_combine,
                                       cpos, cnorm, cdist, margin, stage_n, stage_c);
                }
            }
        }
        __syncthreads();
        if (sh.pending) {
            if (active) {
                const mjccd::Obj& b1 = sh.obj1; const mjccd::Obj& b2 = sh.obj2;
                double pos0[3] = {sh.pos0[0], sh.pos0[1], sh.pos0[2]};
                double nrm0[3] = {sh.normal0[0], sh.normal0[1], sh.normal0[2]};
                double rp[1][3], rn[1][3], rd[1];
                int ws_end[4];
                int n = mjccd::multiccdRerun(cfg, b1, b2, sh.margin, pos0, nrm0, wv, &ws->scratch, st,
                                             rp, rn, rd, ws_end);
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
                if (lane == 0) mjccd::mjp_row(wv)[11] = MJP_T() - _p_t0;
#endif
                if (lane == 0) {
                mjccd::MulticcdRerunOut& o = sh.r[wv];
                o.n = n;
                if (n > 0) {
                    for (int k = 0; k < 3; k++) { o.pos[k] = rp[0][k]; o.normal[k] = rn[0][k]; }
                    o.dist = rd[0];
                }
                for (int k = 0; k < 4; k++) o.ws_end[k] = ws_end[k];
                }
                // the merge's warm-start test for rerun wv+1, off the merge's serial path
                int ssn = -1;
                if (wv + 1 < MJC_FAN_WAVES && n >= 0)
                    ssn = mjccd::multiccdSameStartNext(cfg, b1, b2, pos0, nrm0, wv, ws_end);
                if (lane == 0) sh.r[wv].ss_next = ssn;
            }
            __syncthreads();
            if (leader) {
                const mjccd::Obj& b1 = sh.obj1; const mjccd::Obj& b2 = sh.obj2;
                int nrec = 0;
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
                long long _p_tm0 = MJP_T();
#endif
                int n = mjccd::multiccdMerge(cfg, b1, b2, sh.margin, tolerance, sh.r, &ws->scratch, st,
                                             cpos, cnorm, cdist, &nrec);
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
                _p_tm = MJP_T() - _p_tm0; _p_nrec = nrec;
#endif
                if (!owner0) {
                } else {
                if (recompute_count && nrec) atomicAdd(recompute_count, nrec);
                if (n < 0) { atomicAdd(fallback_count, 1); stage_n[q] = 0; }
                else mjc_stage_contacts(q, n, i, j, swapped, geom_body, geom_friction_d, friction_combine,
                                        cpos, cnorm, cdist, sh.margin, stage_n, stage_c);
                }
            }
        }
        __syncthreads();                            // sh is rewritten by the next pair
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
        if (owner0) {
            static __device__ unsigned int _p_ctr = 0;
            unsigned c = atomicAdd(&_p_ctr, 1u);
            if (MJC_FAN_PROF == 1 ? (c >= 40000u && c < 46000u) : (c % MJC_FAN_PROF == 0 && c < 4000u * MJC_FAN_PROF)) {
                long long T = MJP_T() - _p_t0;
                const long long* r0 = mjccd::mjp_row(0);
                const long long* r1 = mjccd::mjp_row(1);
                const long long* r2 = mjccd::mjp_row(2);
                const long long* r3 = mjccd::mjp_row(3);
                printf("FP q %d ty %d %d n %d pend %d T %lld B %lld M %lld nrec %d | b sup %lld %lld gjk %lld gi %lld poly %lld epa %lld nccd %lld"
                       " | r %lld %lld %lld %lld | rsup %lld %lld %lld %lld rn %lld %lld %lld %lld | w0 sup %lld %lld gjk %lld gi %lld poly %lld epa %lld nccd %lld ss %lld %lld s1 %lld s2 %lld hcst %lld pre %lld post %lld\n",
                       q, _p_ty1, _p_ty2, _p_n, (int)sh.pending, T, _p_tb, _p_tm, _p_nrec,
                       _p_b[0], _p_b[1], _p_b[2], _p_b[3], _p_b[4], _p_b[5], _p_b[6],
                       r0[11], r1[11], r2[11], r3[11],
                       r0[0], r1[0], r2[0], r3[0], r0[1], r1[1], r2[1], r3[1],
                       r0[0], r0[1], r0[2], r0[3], r0[4], r0[5], r0[6], r0[7], r0[8], r0[9], r0[10], r0[12], r0[14], r0[13]);
                printf("FE q %d scan %lld hzn %lld att %lld nit %lld | w0 scan %lld hzn %lld att %lld nit %lld\n", q, _p_b[9], _p_b[10], _p_b[14], _p_b[15], r0[9], r0[10], r0[14], r0[15]);
                printf("FB q %d init %lld stg %lld wstg %lld pen %lld post %lld pre %lld B %lld gjk %lld poly %lld epa %lld T %lld\n", q, _p_ts0, _p_ts1 - _p_ts0, _p_b[16], _p_b[17], _p_b[18], _p_b[19], _p_tb, _p_b[2], _p_b[4], _p_b[5], T);
            }
        }
        __syncthreads();
#endif
    }
}

// Ordered contact write-out for kernel_convex_mj: one wave per env walks the queue in index
// order and applies exactly the sequential semantics of the original atomics (per-body-pair
// cap, then ncon slot / MAX_CONTACTS overflow) pair by pair, so contacts land in queue order.
HIPGL_GLOBAL void __launch_bounds__(64) kernel_convex_mj_emit(
    int num_envs, int nbody,
    const int*    __restrict__ mjc_queue,
    const int*    __restrict__ mjc_queue_count,
    int mjc_queue_cap,
    const int*    __restrict__ stage_n,
    const ContactPair* __restrict__ stage_c,
    int max_contacts_per_body_pair,
    int*          __restrict__ body_pair_ncon,
    ContactPair*  __restrict__ contacts,
    int*          __restrict__ ncon)
{
    const int env = blockIdx.x;
    if (env >= num_envs) return;
    const int lane = threadIdx.x;
    int nq = *mjc_queue_count;
    if (nq > mjc_queue_cap) nq = mjc_queue_cap;
    int cnt = ncon[env];                       // broadphase contacts already written
    for (int base = 0; base < nq; base += 64) {
        int q = base + lane;
        int nn = 0;
        if (q < nq && mjc_queue[q * 3 + 0] == env) nn = stage_n[q];
        unsigned long long m = __ballot(nn > 0);
        while (m) {
            int src = __ffsll((long long)m) - 1;
            m &= m - 1;
            int qq = base + src;
            int n = __shfl(nn, src);
            const ContactPair* in = stage_c + (size_t)qq * MJCCD_STATUS_NX;
            if (max_contacts_per_body_pair > 0 && body_pair_ncon) {
                int b1 = in[0].body1, b2 = in[0].body2;
                int b_lo = (b1 < b2) ? b1 : b2;
                int b_hi = (b1 < b2) ? b2 : b1;
                int bp_idx = env * nbody * nbody + b_lo * nbody + b_hi;
                int prev = body_pair_ncon[bp_idx];
                __syncthreads();
                if (prev >= max_contacts_per_body_pair) continue;
                if (lane == 0) body_pair_ncon[bp_idx] = prev + n;
                __syncthreads();
                if (prev + n > max_contacts_per_body_pair) n = max_contacts_per_body_pair - prev;
            }
            int slot = cnt;
            if (slot + n > MAX_CONTACTS) continue;
            cnt += n;
            for (int c = lane; c < n; c += 64) contacts[env * MAX_CONTACTS + slot + c] = in[c];
        }
    }
    if (lane == 0) ncon[env] = cnt;
}

// ── Host launch wrappers ────────────────────────────────────────────────────

#ifndef HIPGL_BP_CHUNK
#define HIPGL_BP_CHUNK 256   // must equal launch_broadphase_masked's block size
#endif

template <bool HULLS, bool MASK, bool COMPACT = false>
static void launch_broadphase_t(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream, int ng, int ne, dim3 grid, int block,
    const unsigned char* pair_mask, const int* chunk_base = nullptr)
{
    const bool use_mjc = model.mjc_enabled && data.mjc_queue && data.mjc_scratch;
    if (use_mjc) hipMemsetAsync(data.mjc_queue_count, 0, sizeof(int), stream);

    hipLaunchKernelGGL((kernel_broadphase<HULLS, MASK, COMPACT>), grid, block, 0, stream,
        ng, ne, model.nbody,
        model.geom_contype, model.geom_conaffinity, model.geom_body,
        model.body_parent, model.body_weldid,
        data.geom_aabb,
        model.geom_type, model.geom_size_d,
        model.geom_friction_d, model.friction_combine,
        data.geom_xpos_d, data.geom_xquat_d,
        opts.contact_margin,
        model.geom_margin,
        model.pair_filter, model.npair_filter,
        opts.max_contacts_per_body_pair,
        data.contacts, data.ncon, data.body_pair_ncon,
        model.geom_meshid,
        model.mesh_convex_vert_adr, model.mesh_convex_vert_num,
        model.mesh_convex_vert,
        model.mesh_convex_face_adr, model.mesh_convex_face_num,
        model.mesh_convex_face, model.mesh_convex_face_normal,
        model.mesh_convex_edge_adr, model.mesh_convex_edge_num,
        model.mesh_convex_edge,
        use_mjc ? model.geom_mj_mesh : nullptr,
        use_mjc ? data.mjc_queue : nullptr,
        use_mjc ? data.mjc_queue_count : nullptr,
        use_mjc ? data.mjc_queue_cap : 0,
        pair_mask, chunk_base);
}

// Fill `pair_mask` (ngeom*(ngeom-1)/2 bytes) for launch_broadphase.  The mask
// depends only on static model arrays, so it is built once and reused.
void launch_broadphase_pair_mask(
    const DevicePhysicsModel& model,
    unsigned char* pair_mask,
    hipStream_t stream)
{
    int ng = model.ngeom;
    if (ng < 2 || !pair_mask) return;
    int total_pairs = ng * (ng - 1) / 2;
    const int block = 256;
    int grid = (total_pairs + block - 1) / block;
    hipLaunchKernelGGL(kernel_broadphase_pair_mask, dim3(grid), dim3(block), 0, stream,
        ng, model.geom_contype, model.geom_conaffinity, model.geom_body,
        model.body_parent, model.body_weldid,
        model.pair_filter, model.npair_filter, pair_mask);
}

void launch_broadphase_masked(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    const unsigned char* pair_mask)
{
    int ng = model.ngeom;
    int ne = data.num_envs;
    if (ng < 2 || ne <= 0) return;

    int total_pairs = ng * (ng - 1) / 2;
    const int block = 256;
    int grid_x = (total_pairs + block - 1) / block;
    dim3 grid(grid_x, ne);

    if (model.mesh_convex_vert) {
        if (pair_mask) launch_broadphase_t<true, true>(model, data, opts, stream, ng, ne, grid, block, pair_mask);
        else           launch_broadphase_t<true, false>(model, data, opts, stream, ng, ne, grid, block, nullptr);
    } else {
        if (pair_mask) launch_broadphase_t<false, true>(model, data, opts, stream, ng, ne, grid, block, pair_mask);
        else           launch_broadphase_t<false, false>(model, data, opts, stream, ng, ne, grid, block, nullptr);
    }
}

int broadphase_chunk_size() { return HIPGL_BP_CHUNK; }

// Masked broadphase over the non-empty chunks only.  `chunk_base` holds the
// first pair index of every HIPGL_BP_CHUNK-aligned chunk of the pair triangle
// that has at least one pair_mask==1 pair (ascending order), built once per
// mask by the engine.  One workgroup of HIPGL_BP_CHUNK lanes per chunk; the
// work of every launched lane is identical to launch_broadphase_masked's.
void launch_broadphase_compact(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    const unsigned char* pair_mask,
    const int* chunk_base, int nchunk)
{
    int ng = model.ngeom;
    int ne = data.num_envs;
    if (ng < 2 || ne <= 0 || nchunk <= 0) return;
    if (!pair_mask || !chunk_base) {
        launch_broadphase_masked(model, data, opts, stream, pair_mask);
        return;
    }
    const int block = HIPGL_BP_CHUNK;
    dim3 grid(nchunk, ne);
    if (model.mesh_convex_vert)
        launch_broadphase_t<true, true, true>(model, data, opts, stream, ng, ne, grid, block, pair_mask, chunk_base);
    else
        launch_broadphase_t<false, true, true>(model, data, opts, stream, ng, ne, grid, block, pair_mask, chunk_base);
}

void launch_broadphase(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    launch_broadphase_masked(model, data, opts, stream, nullptr);
}

template <int BLOCK>
static void launch_convex_mj_t(const DevicePhysicsModel& model, DevicePhysicsData& data, hipStream_t stream,
                               const mjccd::MeshModel& mm, const mjccd::Config& cfg, int grid, int ppw,
                               int* stage_n, ContactPair* stage_c, int* qhist)
{
    hipLaunchKernelGGL(kernel_convex_mj<BLOCK>, dim3(grid), dim3(BLOCK), 0, stream,
        model.ngeom, model.nbody, model.geom_type, model.geom_body,
        model.geom_friction_d, model.friction_combine, model.geom_margin,
        data.xpos_d, data.xquat_d,
        model.geom_mj_mesh, model.geom_mj_pos, model.geom_mj_quat, model.geom_mj_size,
        model.geom_mj_rbound,
        mm, cfg,
        data.mjc_queue, data.mjc_queue_count, data.mjc_queue_cap,
        static_cast<MjcWorkerScratch*>(data.mjc_scratch), data.mjc_workers, ppw,
        stage_n, stage_c, qhist,
        data.mjc_fallback_count, model.geom_onesided);
}


void launch_narrowphase(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    // Narrowphase is fused with broadphase in kernel_broadphase, except box-mesh and
    // mesh-mesh pairs, which kernel_broadphase queues for the MuJoCo convex port.
    if (!(model.mjc_enabled && data.mjc_queue && data.mjc_scratch)) return;
    mjccd::MeshModel mm;
    mm.mesh_vertadr = model.mj_mesh_vertadr;   mm.mesh_vertnum = model.mj_mesh_vertnum;
    mm.mesh_vert = model.mj_mesh_vert;
    mm.mesh_graphadr = model.mj_mesh_graphadr; mm.mesh_graph = model.mj_mesh_graph;
    mm.mesh_polynum = model.mj_mesh_polynum;   mm.mesh_polyadr = model.mj_mesh_polyadr;
    mm.mesh_polynormal = model.mj_mesh_polynormal;
    mm.mesh_polyvertadr = model.mj_mesh_polyvertadr; mm.mesh_polyvertnum = model.mj_mesh_polyvertnum;
    mm.mesh_polyvert = model.mj_mesh_polyvert;
    mm.mesh_polymapadr = model.mj_mesh_polymapadr;   mm.mesh_polymapnum = model.mj_mesh_polymapnum;
    mm.mesh_polymap = model.mj_mesh_polymap;
    mjccd::Config cfg;
    cfg.ccd_iterations = model.mjc_ccd_iterations;
    cfg.ccd_tolerance = model.mjc_ccd_tolerance;
    cfg.disable_nativeccd = model.mjc_disable_nativeccd;
    cfg.disable_multiccd = model.mjc_disable_multiccd;
    static const int ppw = [] { const char* e = std::getenv("HP_MJC_PPW"); int v = e ? std::atoi(e) : 1;
                                return (v < 1) ? 1 : (v > 64 ? 64 : v); }();
    static const int wpb = [] { const char* e = std::getenv("HP_MJC_WPB"); int v = e ? std::atoi(e) : 1;
                                return (v == 4) ? 4 : 1; }();
    static const bool qhist_on = [] { const char* e = std::getenv("HP_MJC_QHIST"); return e && std::atoi(e) != 0; }();
    const MjcAuxLayout aux = mjc_aux_layout(data.mjc_workers, data.mjc_queue_cap);
    char* auxb = static_cast<char*>(data.mjc_scratch);
    int* stage_n = reinterpret_cast<int*>(auxb + aux.n_off);
    ContactPair* stage_c = reinterpret_cast<ContactPair*>(auxb + aux.c_off);
    int* qhist = qhist_on ? reinterpret_cast<int*>(auxb + aux.h_off) : nullptr;
    // HP_MJC_FAN=0: one wave per pair runs the 4 multiCCD reruns serially (kernel_convex_mj)
    static const bool fan = [] { const char* e = std::getenv("HP_MJC_FAN"); return !(e && std::atoi(e) == 0); }();
    if (fan) {
        int* recomp = reinterpret_cast<int*>(auxb + aux.r_off);
        hipLaunchKernelGGL(kernel_convex_mj_fan, dim3(data.mjc_workers), dim3(64 * MJC_FAN_WAVES), 0, stream,
            model.ngeom, model.nbody, model.geom_type, model.geom_body,
            model.geom_friction_d, model.friction_combine, model.geom_margin,
            data.xpos_d, data.xquat_d,
            model.geom_mj_mesh, model.geom_mj_pos, model.geom_mj_quat, model.geom_mj_size,
            model.geom_mj_rbound,
            mm, cfg,
            data.mjc_queue, data.mjc_queue_count, data.mjc_queue_cap,
            static_cast<MjcWorkerScratch*>(data.mjc_scratch), data.mjc_workers,
            stage_n, stage_c, qhist,
            data.mjc_fallback_count, model.geom_onesided, recomp);
    } else {
        const int waves = (data.mjc_workers + ppw - 1) / ppw;
        const int grid = (waves + wpb - 1) / wpb;
        if (wpb == 4) launch_convex_mj_t<256>(model, data, stream, mm, cfg, grid, ppw, stage_n, stage_c, qhist);
        else          launch_convex_mj_t<64>(model, data, stream, mm, cfg, grid, ppw, stage_n, stage_c, qhist);
    }
    hipLaunchKernelGGL(kernel_convex_mj_emit, dim3(data.num_envs), dim3(64), 0, stream,
        data.num_envs, model.nbody,
        data.mjc_queue, data.mjc_queue_count, data.mjc_queue_cap,
        stage_n, stage_c,
        opts.max_contacts_per_body_pair, data.body_pair_ncon,
        data.contacts, data.ncon);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
