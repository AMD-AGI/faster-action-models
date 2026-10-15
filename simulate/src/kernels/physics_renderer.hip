// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Shared-memory rendering kernel
//
// Generates triangle mesh vertices for physics geoms directly on GPU.
// Reads geom world transforms (xpos, xquat) and sizes from device memory.
// Writes vertex positions, normals, and colors into a flat vertex buffer
// that can be passed directly to hipGL Pipeline::draw_arrays().
//
// Geom types:
//   Box:     24 vertices (6 faces x 4 verts, expanded to triangles = 36 verts)
//   Sphere:  Icosphere approximation (42 verts for level-1 subdivision)
//   Capsule: Cylinder body + hemisphere caps (~72 verts)

#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_types.h"

namespace hipgl {
namespace physics {
namespace kernels {

// Legacy opt-in reflectance emission approximation. CameraRenderer now renders
// actual planar reflections in planar_reflection.cpp; keep this gain at zero
// with that pass. The override remains available for old diagnostic builds.
#ifndef HIPGL_REFLECT_GAIN
#define HIPGL_REFLECT_GAIN 0.0f
#endif


// ── Quaternion rotate helper ────────────────────────────────────────────────

HIPGL_DEVICE inline void qrot(const float* q, const float* v, float* out) {
    float tx = 2.0f * (q[2]*v[2] - q[3]*v[1]);
    float ty = 2.0f * (q[3]*v[0] - q[1]*v[2]);
    float tz = 2.0f * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}


HIPGL_DEVICE inline void write_tc2(float* tc_out, int vert, float u, float v) {
    if (!tc_out) return;
    float* p = tc_out + (size_t)vert * TEXCOORD_COMPONENTS;
    p[0] = u;
    p[1] = v;
#pragma unroll
    for (int k = 2; k < TEXCOORD_COMPONENTS; k++) p[k] = 0.0f;
}

// Templated on the stride so the guard binds at INSTANTIATION: a plain
// function body's static_assert fires even when the function is never called,
// which would make it impossible to compile this helper while the stride is
// still 2 -- and compiling at width 2 with the new code shape in place is
// exactly how the refactor is shown to be bit-neutral before the width moves.
template <int TC = TEXCOORD_COMPONENTS>
HIPGL_DEVICE inline void write_tc3(float* tc_out, int vert, float x, float y, float z) {
    // A 3-component direction cannot be represented while the stride is 2.
    // Silently dropping z would make a cube-mapped geom sample a meaningless
    // 2D coordinate, so refuse to compile a CALLER in that configuration.
    static_assert(TC >= 3, "write_tc3 needs TEXCOORD_COMPONENTS >= 3");
    if (!tc_out) return;
    float* p = tc_out + (size_t)vert * TC;
    p[0] = x;
    p[1] = y;
    p[2] = z;
#pragma unroll
    for (int k = 3; k < TC; k++) p[k] = 0.0f;
}

// ── Write a triangle (3 verts) from positions and normal ────────────────────

HIPGL_DEVICE inline void emit_tri(
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base,
    const float* p0, const float* p1, const float* p2,
    const float* n, const float* color)
{
    for (int k = 0; k < 3; k++) {
        pos_out[(vert_base+0)*3+k] = p0[k];
        pos_out[(vert_base+1)*3+k] = p1[k];
        pos_out[(vert_base+2)*3+k] = p2[k];
        nrm_out[(vert_base+0)*3+k] = n[k];
        nrm_out[(vert_base+1)*3+k] = n[k];
        nrm_out[(vert_base+2)*3+k] = n[k];
    }
    if (col_out) {
        for (int v = 0; v < 3; v++) {
            write_tc2(texcoord_out, vert_base+v, 0.0f, 0.0f);
            texid_out[vert_base+v] = -1;
            col_out[(vert_base+v)*4+0] = color[0];
            col_out[(vert_base+v)*4+1] = color[1];
            col_out[(vert_base+v)*4+2] = color[2];
            col_out[(vert_base+v)*4+3] = color[3];
        }
    }
}

HIPGL_DEVICE inline void emit_tri_textured(
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base,
    const float* p0, const float* p1, const float* p2,
    const float* n,
    float u0, float v0, float u1, float v1, float u2, float v2,
    int texid,
    const float* fallback_color)
{
    for (int k = 0; k < 3; k++) {
        pos_out[(vert_base+0)*3+k] = p0[k];
        pos_out[(vert_base+1)*3+k] = p1[k];
        pos_out[(vert_base+2)*3+k] = p2[k];
        nrm_out[(vert_base+0)*3+k] = n[k];
        nrm_out[(vert_base+1)*3+k] = n[k];
        nrm_out[(vert_base+2)*3+k] = n[k];
    }
    if (col_out) {
        float uvs[3][2] = {{u0,v0},{u1,v1},{u2,v2}};
        for (int vi = 0; vi < 3; vi++) {
            write_tc2(texcoord_out, vert_base+vi, uvs[vi][0], uvs[vi][1]);
            texid_out[vert_base+vi] = texid;
            col_out[(vert_base+vi)*4+0] = fallback_color[0];
            col_out[(vert_base+vi)*4+1] = fallback_color[1];
            col_out[(vert_base+vi)*4+2] = fallback_color[2];
            col_out[(vert_base+vi)*4+3] = fallback_color[3];
        }
    }
}

#ifndef HIPGL_BOX_NUMQUADS
#define HIPGL_BOX_NUMQUADS 4
#endif
constexpr int BOX_NUMQUADS = HIPGL_BOX_NUMQUADS;
constexpr int BOX_TRIS = 6 * BOX_NUMQUADS * BOX_NUMQUADS * 2;
constexpr int BOX_VERTS = BOX_TRIS * 3;
constexpr int SPHERE_SLICES = 28;
constexpr int SPHERE_STACKS = 16;
constexpr int SPHERE_VERTS = SPHERE_SLICES * SPHERE_STACKS * 6;
constexpr int CAPSULE_NSEG = 16;
constexpr int CAPSULE_HEMI_RINGS = 8;
constexpr int CAPSULE_CYLINDER_VERTS = CAPSULE_NSEG * 6;
constexpr int CAPSULE_HEMI_VERTS = CAPSULE_NSEG * CAPSULE_HEMI_RINGS * 6;
constexpr int CAPSULE_VERTS = CAPSULE_CYLINDER_VERTS + 2 * CAPSULE_HEMI_VERTS;
constexpr int CYLINDER_NSEG = 16;
constexpr int CYLINDER_VERTS = CYLINDER_NSEG * 6 + CYLINDER_NSEG * 3 * 2;
#ifndef HIPGL_PLANE_SUBDIV
#define HIPGL_PLANE_SUBDIV 10
#endif
constexpr int PLANE_SUBDIV = HIPGL_PLANE_SUBDIV;
constexpr int PLANE_VERTS = PLANE_SUBDIV * PLANE_SUBDIV * 6;
constexpr int MAX_VERTS_PER_GEOM = (SPHERE_VERTS > CAPSULE_VERTS ?
    (SPHERE_VERTS > PLANE_VERTS ? SPHERE_VERTS : PLANE_VERTS) :
    (CAPSULE_VERTS > PLANE_VERTS ? CAPSULE_VERTS : PLANE_VERTS));

// ── Plane geometry ──────────────────────────────────────────────────────────

HIPGL_DEVICE void generate_plane(
    const float* gpos, const float* gquat, const float* gsize, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base,
    int texid,
    float tex_rep_s = 1.0f, float tex_rep_t = 1.0f)
{
    float hx = gsize[0] > 0.01f ? gsize[0] : 5.0f;
    float hy = gsize[1] > 0.01f ? gsize[1] : 5.0f;

    float local_n[3] = {0, 0, 1};
    float wn[3];
    qrot(gquat, local_n, wn);

    int vi = vert_base;
    constexpr int N = PLANE_SUBDIV;
    for (int iy = 0; iy < N; iy++) {
        for (int ix = 0; ix < N; ix++) {
            float fx0 = (float)ix / N, fx1 = (float)(ix+1) / N;
            float fy0 = (float)iy / N, fy1 = (float)(iy+1) / N;

            float lp[4][3] = {
                {-hx + fx0*2*hx, -hy + fy0*2*hy, 0},
                {-hx + fx1*2*hx, -hy + fy0*2*hy, 0},
                {-hx + fx1*2*hx, -hy + fy1*2*hy, 0},
                {-hx + fx0*2*hx, -hy + fy1*2*hy, 0}
            };
            float wc[4][3];
            for (int c = 0; c < 4; c++) {
                float rotated[3];
                qrot(gquat, lp[c], rotated);
                wc[c][0] = gpos[0] + rotated[0];
                wc[c][1] = gpos[1] + rotated[1];
                wc[c][2] = gpos[2] + rotated[2];
            }

            if (texid >= 0) {
                float v0 = fy0 * tex_rep_t, v1 = fy1 * tex_rep_t;
#ifndef HIPGL_BOX_V_SIGN
#define HIPGL_BOX_V_SIGN (-1.0f)
#endif
#ifndef HIPGL_PLANE_V_FLIP
#define HIPGL_PLANE_V_FLIP 1
#endif
#if HIPGL_PLANE_V_FLIP
                v0 = -v0;
                v1 = -v1;
#endif
                float u0 = fx0 * tex_rep_s, u1 = fx1 * tex_rep_s;
                emit_tri_textured(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                    wc[0], wc[1], wc[2], wn,
                    u0, v0, u1, v0, u1, v1,
                    texid, color);
                vi += 3;
                emit_tri_textured(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                    wc[0], wc[2], wc[3], wn,
                    u0, v0, u1, v1, u0, v1,
                    texid, color);
                vi += 3;
            } else {
                emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi, wc[0], wc[1], wc[2], wn, color);
                vi += 3;
                emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi, wc[0], wc[2], wc[3], wn, color);
                vi += 3;
            }
        }
    }
}

HIPGL_DEVICE inline int cube_face_of(float x, float y, float z)
{
    float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    if (ax < 1e-20f && ay < 1e-20f && az < 1e-20f) return 0;
    if (ax >= ay && ax >= az) return x > 0.0f ? 0 : 1;
    if (ay >= az)             return y > 0.0f ? 2 : 3;
    return z > 0.0f ? 4 : 5;
}

// (s,t) for an EXPLICIT face. Evaluating a corner against a face chosen
// elsewhere lets the mapping extrapolate smoothly past a face boundary instead
// of jumping to a different face's mapping mid-quad.
HIPGL_DEVICE inline void cube_st_on_face(int face, float x, float y, float z,
                                         float& s, float& t)
{
    float sc, tc, ma;
    switch (face) {
        case 0:  sc = -z; tc = -y; ma = fabsf(x); break;
        case 1:  sc =  z; tc = -y; ma = fabsf(x); break;
        case 2:  sc =  x; tc =  z; ma = fabsf(y); break;
        case 3:  sc =  x; tc = -z; ma = fabsf(y); break;
        case 4:  sc =  x; tc = -y; ma = fabsf(z); break;
        default: sc = -x; tc = -y; ma = fabsf(z); break;
    }
    if (ma < 1e-20f) { s = 0.5f; t = 0.5f; return; }
    s = 0.5f * (sc / ma + 1.0f);
    t = 0.5f * (tc / ma + 1.0f);
}

HIPGL_DEVICE inline void cube_face_st(float x, float y, float z,
                                      int& face, float& s, float& t)
{
    face = cube_face_of(x, y, z);
    if (fabsf(x) < 1e-20f && fabsf(y) < 1e-20f && fabsf(z) < 1e-20f) {
        face = 0; s = 0.5f; t = 0.5f; return;
    }
    cube_st_on_face(face, x, y, z, s, t);
}

// ── Box geometry ────────────────────────────────────────────────────────────

HIPGL_DEVICE void generate_box(
    const float* gpos, const float* gquat, const float* gsize, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base,
    int texid,
    float tex_rep_s = 1.0f, float tex_rep_t = 1.0f,
    int is_cube = 0)
{
    float hx = gsize[0], hy = gsize[1], hz = gsize[2];
    constexpr int NQ = BOX_NUMQUADS;
    float d = 2.0f / NQ;

    // Face definitions: axis, sign, and half-sizes for each of 6 faces
    // MuJoCo box subdivision: NQ x NQ quads per face, coordinates in [-1,1]^2
    // mapped to the face's local coordinate system
    struct FaceDef {
        float normal[3];
        float origin[3];  // corner at u=0, v=0
        float u_axis[3];  // direction for u
        float v_axis[3];  // direction for v
    };

    FaceDef face_defs[6] = {
        // +Z top:    u×v = (0,0,+) ✓
        {{0,0,1},  {-hx,-hy, hz}, {2*hx,0,0},  {0,2*hy,0}},
        // -Z bottom: u×v must give (0,0,-1), so swap axes
        {{0,0,-1}, {-hx,-hy,-hz}, {0,2*hy,0},  {2*hx,0,0}},
        // -Y front:  u×v = (0,-,0) ✓
        {{0,-1,0}, {-hx,-hy,-hz}, {2*hx,0,0},  {0,0,2*hz}},
        // +Y back:   u×v must give (0,+1,0), so swap axes
        {{0,1,0},  {-hx, hy,-hz}, {0,0,2*hz},  {2*hx,0,0}},
        // -X left:   u×v must give (-1,0,0), so swap axes
        {{-1,0,0}, {-hx,-hy,-hz}, {0,0,2*hz},  {0,2*hy,0}},
        // +X right:  u×v = (+,0,0) ✓
        {{1,0,0},  { hx,-hy,-hz}, {0,2*hy,0},  {0,0,2*hz}},
    };

    int vi = vert_base;
    for (int f = 0; f < 6; f++) {
        float wn[3];
        qrot(gquat, face_defs[f].normal, wn);

        for (int qx = 0; qx < NQ; qx++) {
            for (int qy = 0; qy < NQ; qy++) {
                float su0 = (float)qx / NQ, su1 = (float)(qx+1) / NQ;
                float sv0 = (float)qy / NQ, sv1 = (float)(qy+1) / NQ;

                float local[4][3];
                float lpos[4][3]; // local position before rotation (for GL_OBJECT_LINEAR UV)
                float suv[4][2] = {{su0,sv0},{su1,sv0},{su1,sv1},{su0,sv1}};
                for (int c = 0; c < 4; c++) {
                    float su = suv[c][0], sv = suv[c][1];
                    float lp[3] = {
                        face_defs[f].origin[0] + su * face_defs[f].u_axis[0] + sv * face_defs[f].v_axis[0],
                        face_defs[f].origin[1] + su * face_defs[f].u_axis[1] + sv * face_defs[f].v_axis[1],
                        face_defs[f].origin[2] + su * face_defs[f].u_axis[2] + sv * face_defs[f].v_axis[2]
                    };
                    lpos[c][0] = lp[0]; lpos[c][1] = lp[1]; lpos[c][2] = lp[2];
                    float rp[3];
                    qrot(gquat, lp, rp);
                    local[c][0] = gpos[0] + rp[0];
                    local[c][1] = gpos[1] + rp[1];
                    local[c][2] = gpos[2] + rp[2];
                }

                if (texid >= 0) {
                    float inv_hx = (hx > 1e-6f) ? 1.0f / hx : 0.0f;
                    float inv_hy = (hy > 1e-6f) ? 1.0f / hy : 0.0f;
                    float inv_hz = (hz > 1e-6f) ? 1.0f / hz : 0.0f;
                    float uv[4][2];
                    float dir3[4][3];
                    if (is_cube) {
                        for (int c = 0; c < 4; c++) {
                            dir3[c][0] = lpos[c][0] * inv_hx * tex_rep_s;
                            dir3[c][1] = lpos[c][1] * inv_hy * tex_rep_t;
                            dir3[c][2] = lpos[c][2] * inv_hz;
                        }
                    } else {
                        // GL_OBJECT_LINEAR 2D: S = 0.5*(x/hx + 1)*rep_s,
                        //                      T = 0.5*(y/hy + 1)*rep_t
                        for (int c = 0; c < 4; c++) {
                            uv[c][0] = 0.5f * (lpos[c][0] * inv_hx + 1.0f) * tex_rep_s;
                            // FIDELITY: MuJoCo negates v in 2-D texgen for EVERY
                            // non-mesh-texcoord geom, not just planes -- mjwarp
                            // render.py:366 has a single `else:` branch with no
                            // per-type v sign, and :412 is v = -0.5*scl1*ly - 0.5.
                            // The plane fix (ca3174b) only touched generate_plane,
                            // so textured BOXES were left v-mirrored.
                            // Scope is narrow: only geoms on the 2-D path, i.e.
                            // flat_stove_1_base_vis and the wall_*_visual boxes.
                            // Cube-textured boxes (the tables) take write_tc3 and
                            // are unaffected.
                            uv[c][1] = HIPGL_BOX_V_SIGN *
                                       0.5f * (lpos[c][1] * inv_hy + 1.0f) * tex_rep_t;
                        }
                    }
                    // Corner order of the two triangles: (0,1,2) and (0,2,3).
                    const int tri_c[2][3] = {{0,1,2},{0,2,3}};
                    for (int t = 0; t < 2; t++) {
                        const int* cc = tri_c[t];
                        if (is_cube) {
                            emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                                     local[cc[0]], local[cc[1]], local[cc[2]], wn, color);
                            // emit_tri zeroes texcoord and sets texid -1; the
                            // cube direction and texid are applied here.
                            for (int v = 0; v < 3; v++) {
                                write_tc3(texcoord_out, vi+v,
                                          dir3[cc[v]][0], dir3[cc[v]][1], dir3[cc[v]][2]);
                                texid_out[vi+v] = texid;
                            }
                        } else {
                            emit_tri_textured(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                                local[cc[0]], local[cc[1]], local[cc[2]], wn,
                                uv[cc[0]][0], uv[cc[0]][1], uv[cc[1]][0], uv[cc[1]][1],
                                uv[cc[2]][0], uv[cc[2]][1],
                                texid, color);
                        }
                        vi += 3;
                    }
                } else {
                    emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi, local[0], local[1], local[2], wn, color);
                    vi += 3;
                    emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi, local[0], local[2], local[3], wn, color);
                    vi += 3;
                }
            }
        }
    }
}

// ── Sphere geometry (lat/lon tessellation matching MuJoCo) ──────────────────

HIPGL_DEVICE void generate_sphere(
    const float* gpos, float radius, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out, int vert_base)
{
    constexpr float PI = 3.14159265f;
    constexpr int slices = SPHERE_SLICES;
    constexpr int stacks = SPHERE_STACKS;

    int vi = vert_base;
    for (int j = 0; j < stacks; j++) {
        float theta0 = PI * j / stacks;
        float theta1 = PI * (j + 1) / stacks;
        float st0 = sinf(theta0), ct0 = cosf(theta0);
        float st1 = sinf(theta1), ct1 = cosf(theta1);

        for (int i = 0; i < slices; i++) {
            float phi0 = 2.0f * PI * i / slices;
            float phi1 = 2.0f * PI * (i + 1) / slices;
            float sp0 = sinf(phi0), cp0 = cosf(phi0);
            float sp1 = sinf(phi1), cp1 = cosf(phi1);

            // Four corners of the quad
            float n00[3] = {st0*cp0, st0*sp0, ct0};
            float n10[3] = {st0*cp1, st0*sp1, ct0};
            float n01[3] = {st1*cp0, st1*sp0, ct1};
            float n11[3] = {st1*cp1, st1*sp1, ct1};

            float p00[3] = {gpos[0]+n00[0]*radius, gpos[1]+n00[1]*radius, gpos[2]+n00[2]*radius};
            float p10[3] = {gpos[0]+n10[0]*radius, gpos[1]+n10[1]*radius, gpos[2]+n10[2]*radius};
            float p01[3] = {gpos[0]+n01[0]*radius, gpos[1]+n01[1]*radius, gpos[2]+n01[2]*radius};
            float p11[3] = {gpos[0]+n11[0]*radius, gpos[1]+n11[1]*radius, gpos[2]+n11[2]*radius};

            // Emit smooth-shaded triangle pair with per-vertex normals
            // Tri 1: p00, p10, p11
            for (int k = 0; k < 3; k++) {
                pos_out[(vi+0)*3+k] = p00[k];
                pos_out[(vi+1)*3+k] = p10[k];
                pos_out[(vi+2)*3+k] = p11[k];
                nrm_out[(vi+0)*3+k] = n00[k];
                nrm_out[(vi+1)*3+k] = n10[k];
                nrm_out[(vi+2)*3+k] = n11[k];
            }
            for (int v = 0; v < 3; v++) {
                write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);
                texid_out[vi+v] = -1;
                col_out[(vi+v)*4+0] = color[0]; col_out[(vi+v)*4+1] = color[1];
                col_out[(vi+v)*4+2] = color[2]; col_out[(vi+v)*4+3] = color[3];
            }
            vi += 3;

            // Tri 2: p00, p11, p01
            for (int k = 0; k < 3; k++) {
                pos_out[(vi+0)*3+k] = p00[k];
                pos_out[(vi+1)*3+k] = p11[k];
                pos_out[(vi+2)*3+k] = p01[k];
                nrm_out[(vi+0)*3+k] = n00[k];
                nrm_out[(vi+1)*3+k] = n11[k];
                nrm_out[(vi+2)*3+k] = n01[k];
            }
            for (int v = 0; v < 3; v++) {
                write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);
                texid_out[vi+v] = -1;
                col_out[(vi+v)*4+0] = color[0]; col_out[(vi+v)*4+1] = color[1];
                col_out[(vi+v)*4+2] = color[2]; col_out[(vi+v)*4+3] = color[3];
            }
            vi += 3;
        }
    }
}

// ── Capsule geometry (cylinder + hemispherical caps) ────────────────────────

HIPGL_DEVICE void generate_cylinder(
    const float* gpos, const float* gquat, float radius, float half_len, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out, int vert_base,
    int texid = -1, int is_cube = 0,
    float tex_rep_s = 1.0f, float tex_rep_t = 1.0f)
{
    constexpr int NSEG = CYLINDER_NSEG;
    constexpr float TWO_PI = 6.2831853f;

    float local_axis[3] = {0, 0, 1};
    float axis[3];
    qrot(gquat, local_axis, axis);

    float top[3] = {gpos[0]+axis[0]*half_len, gpos[1]+axis[1]*half_len, gpos[2]+axis[2]*half_len};
    float bot[3] = {gpos[0]-axis[0]*half_len, gpos[1]-axis[1]*half_len, gpos[2]-axis[2]*half_len};

    float up_ref[3] = {0, 0, 1};
    if (fabsf(axis[2]) > 0.9f) { up_ref[0] = 1; up_ref[1] = 0; up_ref[2] = 0; }
    float right[3] = {
        axis[1]*up_ref[2] - axis[2]*up_ref[1],
        axis[2]*up_ref[0] - axis[0]*up_ref[2],
        axis[0]*up_ref[1] - axis[1]*up_ref[0]
    };
    float rlen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    if (rlen > 1e-8f) { right[0]/=rlen; right[1]/=rlen; right[2]/=rlen; }
    float fwd[3] = {
        right[1]*axis[2] - right[2]*axis[1],
        right[2]*axis[0] - right[0]*axis[2],
        right[0]*axis[1] - right[1]*axis[0]
    };

    float inv_q[4] = {gquat[0], -gquat[1], -gquat[2], -gquat[3]};
    auto write_cyl_tc = [&](int vert, float cx, float sy, float zh) {
        float wdir[3];
        for (int k = 0; k < 3; k++) wdir[k] = right[k]*cx + fwd[k]*sy + axis[k]*zh;
        float odir[3];
        qrot(inv_q, wdir, odir);            // world -> object
        if (is_cube) {
            // 3D direction; the fragment stage selects the cube face. A barrel
            // quad spans more than one face, so this cannot be resolved here.
            write_tc3(texcoord_out, vert, odir[0], odir[1], odir[2]);
        } else {
            write_tc2(texcoord_out, vert,
                      0.5f*tex_rep_s*odir[0] - 0.5f,
                      0.5f*tex_rep_t*odir[1] - 0.5f);
        }
    };

    int vi = vert_base;

    for (int i = 0; i < NSEG; i++) {
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        float n0[3], n1[3];
        for (int k = 0; k < 3; k++) { n0[k] = right[k]*c0 + fwd[k]*s0; n1[k] = right[k]*c1 + fwd[k]*s1; }
        float t0[3], t1[3], b0[3], b1[3];
        for (int k = 0; k < 3; k++) {
            t0[k] = top[k]+n0[k]*radius; t1[k] = top[k]+n1[k]*radius;
            b0[k] = bot[k]+n0[k]*radius; b1[k] = bot[k]+n1[k]*radius;
        }
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=t0[k];pos_out[(vi+1)*3+k]=t1[k];pos_out[(vi+2)*3+k]=b0[k];
        nrm_out[(vi+0)*3+k]=n0[k];nrm_out[(vi+1)*3+k]=n1[k];nrm_out[(vi+2)*3+k]=n0[k];}
        write_cyl_tc(vi+0, c0, s0, 1.0f);
        write_cyl_tc(vi+1, c1, s1, 1.0f);
        write_cyl_tc(vi+2, c0, s0, -1.0f);
        for (int v = 0; v < 3; v++) {
            texid_out[vi+v] = texid;
            col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
            col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
        }
        vi+=3;
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=t1[k];pos_out[(vi+1)*3+k]=b1[k];pos_out[(vi+2)*3+k]=b0[k];
        nrm_out[(vi+0)*3+k]=n1[k];nrm_out[(vi+1)*3+k]=n1[k];nrm_out[(vi+2)*3+k]=n0[k];}
        write_cyl_tc(vi+0, c1, s1, 1.0f);
        write_cyl_tc(vi+1, c1, s1, -1.0f);
        write_cyl_tc(vi+2, c0, s0, -1.0f);
        for (int v = 0; v < 3; v++) {
            texid_out[vi+v] = texid;
            col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
            col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
        }
        vi+=3;
    }

    // Top cap (disk)
    for (int i = 0; i < NSEG; i++) {
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        float e0[3], e1[3];
        for (int k = 0; k < 3; k++) {
            e0[k] = top[k] + (right[k]*c0 + fwd[k]*s0)*radius;
            e1[k] = top[k] + (right[k]*c1 + fwd[k]*s1)*radius;
        }
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=top[k];pos_out[(vi+1)*3+k]=e0[k];pos_out[(vi+2)*3+k]=e1[k];
        nrm_out[(vi+0)*3+k]=axis[k];nrm_out[(vi+1)*3+k]=axis[k];nrm_out[(vi+2)*3+k]=axis[k];}
        write_cyl_tc(vi+0, 0.0f, 0.0f, 1.0f);
        write_cyl_tc(vi+1, c0, s0, 1.0f);
        write_cyl_tc(vi+2, c1, s1, 1.0f);
        for (int v = 0; v < 3; v++) {
            texid_out[vi+v] = texid;
            col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
            col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
        }
        vi+=3;
    }

    // Bottom cap (disk, reversed winding)
    float neg_axis[3] = {-axis[0], -axis[1], -axis[2]};
    for (int i = 0; i < NSEG; i++) {
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        float e0[3], e1[3];
        for (int k = 0; k < 3; k++) {
            e0[k] = bot[k] + (right[k]*c0 + fwd[k]*s0)*radius;
            e1[k] = bot[k] + (right[k]*c1 + fwd[k]*s1)*radius;
        }
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=bot[k];pos_out[(vi+1)*3+k]=e1[k];pos_out[(vi+2)*3+k]=e0[k];
        nrm_out[(vi+0)*3+k]=neg_axis[k];nrm_out[(vi+1)*3+k]=neg_axis[k];nrm_out[(vi+2)*3+k]=neg_axis[k];}
        write_cyl_tc(vi+0, 0.0f, 0.0f, -1.0f);
        write_cyl_tc(vi+1, c1, s1, -1.0f);
        write_cyl_tc(vi+2, c0, s0, -1.0f);
        for (int v = 0; v < 3; v++) {
            texid_out[vi+v] = texid;
            col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
            col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
        }
        vi+=3;
    }
}

HIPGL_DEVICE void generate_capsule(
    const float* gpos, const float* gquat, float radius, float half_len, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out, int vert_base)
{
    constexpr int NSEG = CAPSULE_NSEG;
    constexpr int NRINGS = CAPSULE_HEMI_RINGS;
    constexpr float PI = 3.14159265f;
    constexpr float TWO_PI = 6.2831853f;

    float local_axis[3] = {0, 0, 1};
    float axis[3];
    qrot(gquat, local_axis, axis);

    float top[3] = {gpos[0]+axis[0]*half_len, gpos[1]+axis[1]*half_len, gpos[2]+axis[2]*half_len};
    float bot[3] = {gpos[0]-axis[0]*half_len, gpos[1]-axis[1]*half_len, gpos[2]-axis[2]*half_len};

    float up_ref[3] = {0, 0, 1};
    if (fabsf(axis[2]) > 0.9f) { up_ref[0] = 1; up_ref[1] = 0; up_ref[2] = 0; }
    float right[3] = {
        axis[1]*up_ref[2] - axis[2]*up_ref[1],
        axis[2]*up_ref[0] - axis[0]*up_ref[2],
        axis[0]*up_ref[1] - axis[1]*up_ref[0]
    };
    float rlen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    if (rlen > 1e-8f) { right[0]/=rlen; right[1]/=rlen; right[2]/=rlen; }
    float fwd[3] = {
        right[1]*axis[2] - right[2]*axis[1],
        right[2]*axis[0] - right[0]*axis[2],
        right[0]*axis[1] - right[1]*axis[0]
    };

    int vi = vert_base;

    // Helper: get normal direction at (phi, theta_elev) in local frame
    // theta_elev: 0=equator, PI/2=pole
    #define NORM_DIR(n_out, cphi, sphi, cthe, sthe) do { \
        for (int k = 0; k < 3; k++) \
            n_out[k] = (right[k]*cphi + fwd[k]*sphi)*cthe + axis[k]*sthe; \
    } while(0)

    // Cylinder body: NSEG segments, each 2 triangles with per-vertex normals
    for (int i = 0; i < NSEG; i++) {
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);

        float n0[3], n1[3];
        for (int k = 0; k < 3; k++) { n0[k] = right[k]*c0 + fwd[k]*s0; n1[k] = right[k]*c1 + fwd[k]*s1; }
        float t0[3], t1[3], b0[3], b1[3];
        for (int k = 0; k < 3; k++) {
            t0[k] = top[k]+n0[k]*radius; t1[k] = top[k]+n1[k]*radius;
            b0[k] = bot[k]+n0[k]*radius; b1[k] = bot[k]+n1[k]*radius;
        }
        // Tri 1: t0, t1, b0 (per-vertex normals)
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=t0[k];pos_out[(vi+1)*3+k]=t1[k];pos_out[(vi+2)*3+k]=b0[k];
        nrm_out[(vi+0)*3+k]=n0[k];nrm_out[(vi+1)*3+k]=n1[k];nrm_out[(vi+2)*3+k]=n0[k];}
        for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
        vi+=3;
        // Tri 2: t1, b1, b0
        for (int k=0;k<3;k++){pos_out[(vi+0)*3+k]=t1[k];pos_out[(vi+1)*3+k]=b1[k];pos_out[(vi+2)*3+k]=b0[k];
        nrm_out[(vi+0)*3+k]=n1[k];nrm_out[(vi+1)*3+k]=n1[k];nrm_out[(vi+2)*3+k]=n0[k];}
        for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
        vi+=3;
    }

    // Top hemisphere: rings from equator (theta=0) to pole (theta=PI/2)
    for (int j = 0; j < NRINGS; j++) {
        float t0 = (PI*0.5f)*j/NRINGS, t1 = (PI*0.5f)*(j+1)/NRINGS;
        float ct0 = cosf(t0), st0 = sinf(t0), ct1 = cosf(t1), st1 = sinf(t1);
        for (int i = 0; i < NSEG; i++) {
            float a0 = TWO_PI*i/NSEG, a1 = TWO_PI*(i+1)/NSEG;
            float ca0=cosf(a0),sa0=sinf(a0),ca1=cosf(a1),sa1=sinf(a1);
            float n00[3],n10[3],n01[3],n11[3];
            NORM_DIR(n00,ca0,sa0,ct0,st0); NORM_DIR(n10,ca1,sa1,ct0,st0);
            NORM_DIR(n01,ca0,sa0,ct1,st1); NORM_DIR(n11,ca1,sa1,ct1,st1);
            float p00[3],p10[3],p01[3],p11[3];
            for(int k=0;k<3;k++){p00[k]=top[k]+n00[k]*radius;p10[k]=top[k]+n10[k]*radius;
            p01[k]=top[k]+n01[k]*radius;p11[k]=top[k]+n11[k]*radius;}
            for(int k=0;k<3;k++){pos_out[(vi+0)*3+k]=p00[k];pos_out[(vi+1)*3+k]=p10[k];pos_out[(vi+2)*3+k]=p11[k];
            nrm_out[(vi+0)*3+k]=n00[k];nrm_out[(vi+1)*3+k]=n10[k];nrm_out[(vi+2)*3+k]=n11[k];}
            for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
            vi+=3;
            for(int k=0;k<3;k++){pos_out[(vi+0)*3+k]=p00[k];pos_out[(vi+1)*3+k]=p11[k];pos_out[(vi+2)*3+k]=p01[k];
            nrm_out[(vi+0)*3+k]=n00[k];nrm_out[(vi+1)*3+k]=n11[k];nrm_out[(vi+2)*3+k]=n01[k];}
            for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
            vi+=3;
        }
    }

    // Bottom hemisphere: rings from equator to negative pole
    for (int j = 0; j < NRINGS; j++) {
        float t0 = (PI*0.5f)*j/NRINGS, t1 = (PI*0.5f)*(j+1)/NRINGS;
        float ct0 = cosf(t0), st0 = -sinf(t0), ct1 = cosf(t1), st1 = -sinf(t1);
        for (int i = 0; i < NSEG; i++) {
            float a0 = TWO_PI*i/NSEG, a1 = TWO_PI*(i+1)/NSEG;
            float ca0=cosf(a0),sa0=sinf(a0),ca1=cosf(a1),sa1=sinf(a1);
            float n00[3],n10[3],n01[3],n11[3];
            NORM_DIR(n00,ca0,sa0,ct0,st0); NORM_DIR(n10,ca1,sa1,ct0,st0);
            NORM_DIR(n01,ca0,sa0,ct1,st1); NORM_DIR(n11,ca1,sa1,ct1,st1);
            float p00[3],p10[3],p01[3],p11[3];
            for(int k=0;k<3;k++){p00[k]=bot[k]+n00[k]*radius;p10[k]=bot[k]+n10[k]*radius;
            p01[k]=bot[k]+n01[k]*radius;p11[k]=bot[k]+n11[k]*radius;}
            for(int k=0;k<3;k++){pos_out[(vi+0)*3+k]=p00[k];pos_out[(vi+1)*3+k]=p11[k];pos_out[(vi+2)*3+k]=p10[k];
            nrm_out[(vi+0)*3+k]=n00[k];nrm_out[(vi+1)*3+k]=n11[k];nrm_out[(vi+2)*3+k]=n10[k];}
            for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
            vi+=3;
            for(int k=0;k<3;k++){pos_out[(vi+0)*3+k]=p00[k];pos_out[(vi+1)*3+k]=p01[k];pos_out[(vi+2)*3+k]=p11[k];
            nrm_out[(vi+0)*3+k]=n00[k];nrm_out[(vi+1)*3+k]=n01[k];nrm_out[(vi+2)*3+k]=n11[k];}
            for(int v=0;v<3;v++){write_tc2(texcoord_out, vi+v, 0.0f, 0.0f);texid_out[vi+v]=-1;col_out[(vi+v)*4+0]=color[0];col_out[(vi+v)*4+1]=color[1];col_out[(vi+v)*4+2]=color[2];col_out[(vi+v)*4+3]=color[3];}
            vi+=3;
        }
    }
    #undef NORM_DIR
}


constexpr int SPHERE_TRIS   = SPHERE_SLICES * SPHERE_STACKS * 2;
constexpr int CAPSULE_TRIS  = CAPSULE_VERTS / 3;
constexpr int CYLINDER_TRIS = CYLINDER_VERTS / 3;
constexpr int PLANE_TRIS    = PLANE_VERTS / 3;
constexpr int MAX_TRIS_A = (BOX_TRIS > SPHERE_TRIS ? BOX_TRIS : SPHERE_TRIS);
constexpr int MAX_TRIS_B = (CAPSULE_TRIS > CYLINDER_TRIS ? CAPSULE_TRIS : CYLINDER_TRIS);
constexpr int MAX_TRIS_C = (MAX_TRIS_B > PLANE_TRIS ? MAX_TRIS_B : PLANE_TRIS);
constexpr int MAX_TRIS_PER_GEOM = (MAX_TRIS_A > MAX_TRIS_C ? MAX_TRIS_A : MAX_TRIS_C);

HIPGL_DEVICE inline int tris_for_gtype(int gtype) {
    if (gtype == GEOM_BOX)      return BOX_TRIS;
    if (gtype == GEOM_SPHERE)   return SPHERE_TRIS;
    if (gtype == GEOM_CAPSULE)  return CAPSULE_TRIS;
    if (gtype == GEOM_CYLINDER) return CYLINDER_TRIS;
    if (gtype == GEOM_PLANE)    return PLANE_TRIS;
    return 0;   // GEOM_MESH and anything unrecognised: the serial kernel wrote
                // nothing for these either.
}

// Triangle with PER-VERTEX normals, flat texcoord (0,0) and a constant texid.
HIPGL_DEVICE inline void emit_tri_pn(
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vb,
    const float* p0, const float* p1, const float* p2,
    const float* n0, const float* n1, const float* n2,
    const float* color, int texid_val)
{
    for (int k = 0; k < 3; k++) {
        pos_out[(vb+0)*3+k] = p0[k];
        pos_out[(vb+1)*3+k] = p1[k];
        pos_out[(vb+2)*3+k] = p2[k];
        nrm_out[(vb+0)*3+k] = n0[k];
        nrm_out[(vb+1)*3+k] = n1[k];
        nrm_out[(vb+2)*3+k] = n2[k];
    }
    if (col_out) {
        for (int v = 0; v < 3; v++) {
            write_tc2(texcoord_out, vb+v, 0.0f, 0.0f);
            texid_out[vb+v] = texid_val;
            col_out[(vb+v)*4+0] = color[0];
            col_out[(vb+v)*4+1] = color[1];
            col_out[(vb+v)*4+2] = color[2];
            col_out[(vb+v)*4+3] = color[3];
        }
    }
}

// ── Plane, one triangle ─────────────────────────────────────────────────────
HIPGL_DEVICE void generate_plane_tri(
    const float* gpos, const float* gquat, const float* gsize, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base, int texid, float tex_rep_s, float tex_rep_t, int t)
{
    float hx = gsize[0] > 0.01f ? gsize[0] : 5.0f;
    float hy = gsize[1] > 0.01f ? gsize[1] : 5.0f;

    float local_n[3] = {0, 0, 1};
    float wn[3];
    qrot(gquat, local_n, wn);

    constexpr int N = PLANE_SUBDIV;
    int q = t >> 1, sub = t & 1;
    int iy = q / N, ix = q - iy * N;
    float fx0 = (float)ix / N, fx1 = (float)(ix+1) / N;
    float fy0 = (float)iy / N, fy1 = (float)(iy+1) / N;

    // Quad corners are (fx0,fy0),(fx1,fy0),(fx1,fy1),(fx0,fy1); triangle 0 is
    // corners (0,1,2) and triangle 1 is (0,2,3).
    float cu[3] = {fx0, fx1, sub ? fx0 : fx1};
    float cv[3] = {fy0, sub ? fy1 : fy0, fy1};

    float wc[3][3];
    for (int c = 0; c < 3; c++) {
        float lp[3] = {-hx + cu[c]*2*hx, -hy + cv[c]*2*hy, 0};
        float rotated[3];
        qrot(gquat, lp, rotated);
        wc[c][0] = gpos[0] + rotated[0];
        wc[c][1] = gpos[1] + rotated[1];
        wc[c][2] = gpos[2] + rotated[2];
    }

    int vi = vert_base + t*3;
    if (texid >= 0) {
        float uu[3], vv[3];
        for (int c = 0; c < 3; c++) {
            uu[c] = cu[c] * tex_rep_s;
            float vt = cv[c] * tex_rep_t;
#if HIPGL_PLANE_V_FLIP
            vt = -vt;
#endif
            vv[c] = vt;
        }
        emit_tri_textured(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
            wc[0], wc[1], wc[2], wn,
            uu[0], vv[0], uu[1], vv[1], uu[2], vv[2],
            texid, color);
    } else {
        emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                 wc[0], wc[1], wc[2], wn, color);
    }
}

// ── Box, one triangle ───────────────────────────────────────────────────────
// The quad body below is a VERBATIM copy of generate_box's inner block --
// including building all four corners and the suv/local/lpos arrays -- because
// re-expressing it (e.g. transforming only the three corners this triangle
// needs) changes how the compiler contracts `origin + su*u_axis + sv*v_axis`
// into FMAs and moved a handful of pixels by 1 LSB. Keeping the expression
// shape identical is what makes the output bit-identical; the only change is
// that ONE triangle of ONE quad is emitted per thread.
HIPGL_DEVICE void generate_box_tri(
    const float* gpos, const float* gquat, const float* gsize, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base, int texid, float tex_rep_s, float tex_rep_t, int is_cube, int t)
{
    float hx = gsize[0], hy = gsize[1], hz = gsize[2];
    constexpr int NQ = BOX_NUMQUADS;

    int qq = t >> 1, sub = t & 1;
    int f  = qq / (NQ*NQ);
    int r  = qq - f*(NQ*NQ);
    int qx = r / NQ;
    int qy = r - qx*NQ;

    struct FaceDef {
        float normal[3];
        float origin[3];
        float u_axis[3];
        float v_axis[3];
    };
    FaceDef fd;
    switch (f) {
        case 0:  fd = FaceDef{{0,0,1},  {-hx,-hy, hz}, {2*hx,0,0},  {0,2*hy,0}};  break;
        case 1:  fd = FaceDef{{0,0,-1}, {-hx,-hy,-hz}, {0,2*hy,0},  {2*hx,0,0}};  break;
        case 2:  fd = FaceDef{{0,-1,0}, {-hx,-hy,-hz}, {2*hx,0,0},  {0,0,2*hz}};  break;
        case 3:  fd = FaceDef{{0,1,0},  {-hx, hy,-hz}, {0,0,2*hz},  {2*hx,0,0}};  break;
        case 4:  fd = FaceDef{{-1,0,0}, {-hx,-hy,-hz}, {0,0,2*hz},  {0,2*hy,0}};  break;
        default: fd = FaceDef{{1,0,0},  { hx,-hy,-hz}, {0,2*hy,0},  {0,0,2*hz}};  break;
    }

    float wn[3];
    qrot(gquat, fd.normal, wn);

    float su0 = (float)qx / NQ, su1 = (float)(qx+1) / NQ;
    float sv0 = (float)qy / NQ, sv1 = (float)(qy+1) / NQ;

    float local[4][3];
    float lpos[4][3];
    float suv[4][2] = {{su0,sv0},{su1,sv0},{su1,sv1},{su0,sv1}};
    for (int c = 0; c < 4; c++) {
        float su = suv[c][0], sv = suv[c][1];
        float lp[3] = {
            fd.origin[0] + su * fd.u_axis[0] + sv * fd.v_axis[0],
            fd.origin[1] + su * fd.u_axis[1] + sv * fd.v_axis[1],
            fd.origin[2] + su * fd.u_axis[2] + sv * fd.v_axis[2]
        };
        lpos[c][0] = lp[0]; lpos[c][1] = lp[1]; lpos[c][2] = lp[2];
        float rp[3];
        qrot(gquat, lp, rp);
        local[c][0] = gpos[0] + rp[0];
        local[c][1] = gpos[1] + rp[1];
        local[c][2] = gpos[2] + rp[2];
    }

    int vi = vert_base + t*3;
    const int tri_c[2][3] = {{0,1,2},{0,2,3}};
    const int* cc = tri_c[sub];

    if (texid >= 0) {
        float inv_hx = (hx > 1e-6f) ? 1.0f / hx : 0.0f;
        float inv_hy = (hy > 1e-6f) ? 1.0f / hy : 0.0f;
        float inv_hz = (hz > 1e-6f) ? 1.0f / hz : 0.0f;
        float uv[4][2];
        float dir3[4][3];
        if (is_cube) {
            for (int c = 0; c < 4; c++) {
                dir3[c][0] = lpos[c][0] * inv_hx * tex_rep_s;
                dir3[c][1] = lpos[c][1] * inv_hy * tex_rep_t;
                dir3[c][2] = lpos[c][2] * inv_hz;
            }
        } else {
            for (int c = 0; c < 4; c++) {
                uv[c][0] = 0.5f * (lpos[c][0] * inv_hx + 1.0f) * tex_rep_s;
                uv[c][1] = HIPGL_BOX_V_SIGN *
                           0.5f * (lpos[c][1] * inv_hy + 1.0f) * tex_rep_t;
            }
        }
        if (is_cube) {
            emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                     local[cc[0]], local[cc[1]], local[cc[2]], wn, color);
            if (texid_out) {
                for (int v = 0; v < 3; v++) {
                    write_tc3(texcoord_out, vi+v,
                              dir3[cc[v]][0], dir3[cc[v]][1], dir3[cc[v]][2]);
                    texid_out[vi+v] = texid;
                }
            }
        } else {
            emit_tri_textured(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                local[cc[0]], local[cc[1]], local[cc[2]], wn,
                uv[cc[0]][0], uv[cc[0]][1], uv[cc[1]][0], uv[cc[1]][1],
                uv[cc[2]][0], uv[cc[2]][1],
                texid, color);
        }
    } else {
        emit_tri(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                 local[cc[0]], local[cc[1]], local[cc[2]], wn, color);
    }
}

// ── Sphere, one triangle ────────────────────────────────────────────────────
HIPGL_DEVICE void generate_sphere_tri(
    const float* gpos, float radius, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base, int t)
{
    constexpr float PI = 3.14159265f;
    constexpr int slices = SPHERE_SLICES;
    constexpr int stacks = SPHERE_STACKS;

    int q = t >> 1, sub = t & 1;
    int j = q / slices, i = q - j * slices;

    float theta0 = PI * j / stacks;
    float theta1 = PI * (j + 1) / stacks;
    float st0 = sinf(theta0), ct0 = cosf(theta0);
    float st1 = sinf(theta1), ct1 = cosf(theta1);

    float phi0 = 2.0f * PI * i / slices;
    float phi1 = 2.0f * PI * (i + 1) / slices;
    float sp0 = sinf(phi0), cp0 = cosf(phi0);
    float sp1 = sinf(phi1), cp1 = cosf(phi1);

    float n00[3] = {st0*cp0, st0*sp0, ct0};
    float n10[3] = {st0*cp1, st0*sp1, ct0};
    float n01[3] = {st1*cp0, st1*sp0, ct1};
    float n11[3] = {st1*cp1, st1*sp1, ct1};

    // Tri 0: n00, n10, n11.  Tri 1: n00, n11, n01.
    float nb[3], nc[3];
    for (int k = 0; k < 3; k++) {
        nb[k] = sub ? n11[k] : n10[k];
        nc[k] = sub ? n01[k] : n11[k];
    }
    float p0[3], p1[3], p2[3];
    for (int k = 0; k < 3; k++) {
        p0[k] = gpos[k] + n00[k]*radius;
        p1[k] = gpos[k] + nb[k]*radius;
        p2[k] = gpos[k] + nc[k]*radius;
    }
    emit_tri_pn(pos_out, nrm_out, texcoord_out, texid_out, col_out, vert_base + t*3,
                p0, p1, p2, n00, nb, nc, color, -1);
}

// ── Capsule, one triangle ───────────────────────────────────────────────────
HIPGL_DEVICE void generate_capsule_tri(
    const float* gpos, const float* gquat, float radius, float half_len, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base, int t)
{
    constexpr int NSEG = CAPSULE_NSEG;
    constexpr int NRINGS = CAPSULE_HEMI_RINGS;
    constexpr float PI = 3.14159265f;
    constexpr float TWO_PI = 6.2831853f;

    float local_axis[3] = {0, 0, 1};
    float axis[3];
    qrot(gquat, local_axis, axis);

    float top[3] = {gpos[0]+axis[0]*half_len, gpos[1]+axis[1]*half_len, gpos[2]+axis[2]*half_len};
    float bot[3] = {gpos[0]-axis[0]*half_len, gpos[1]-axis[1]*half_len, gpos[2]-axis[2]*half_len};

    float up_ref[3] = {0, 0, 1};
    if (fabsf(axis[2]) > 0.9f) { up_ref[0] = 1; up_ref[1] = 0; up_ref[2] = 0; }
    float right[3] = {
        axis[1]*up_ref[2] - axis[2]*up_ref[1],
        axis[2]*up_ref[0] - axis[0]*up_ref[2],
        axis[0]*up_ref[1] - axis[1]*up_ref[0]
    };
    float rlen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    if (rlen > 1e-8f) { right[0]/=rlen; right[1]/=rlen; right[2]/=rlen; }
    float fwd[3] = {
        right[1]*axis[2] - right[2]*axis[1],
        right[2]*axis[0] - right[0]*axis[2],
        right[0]*axis[1] - right[1]*axis[0]
    };

    int vi = vert_base + t*3;

    if (t < NSEG*2) {
        int i = t >> 1, sub = t & 1;
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        float n0[3], n1[3];
        for (int k = 0; k < 3; k++) { n0[k] = right[k]*c0 + fwd[k]*s0; n1[k] = right[k]*c1 + fwd[k]*s1; }
        float t0[3], t1[3], b0[3], b1[3];
        for (int k = 0; k < 3; k++) {
            t0[k] = top[k]+n0[k]*radius; t1[k] = top[k]+n1[k]*radius;
            b0[k] = bot[k]+n0[k]*radius; b1[k] = bot[k]+n1[k]*radius;
        }
        // Tri 0: (t0,t1,b0) / (n0,n1,n0).  Tri 1: (t1,b1,b0) / (n1,n1,n0).
        float p0[3], p1[3], q0[3], q1[3];
        for (int k = 0; k < 3; k++) {
            p0[k] = sub ? t1[k] : t0[k];
            q0[k] = sub ? n1[k] : n0[k];
            p1[k] = sub ? b1[k] : t1[k];
            q1[k] = n1[k];
        }
        emit_tri_pn(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                    p0, p1, b0, q0, q1, n0, color, -1);
        return;
    }

    int t2 = t - NSEG*2;
    int bottom = (t2 >= NRINGS*NSEG*2) ? 1 : 0;
    if (bottom) t2 -= NRINGS*NSEG*2;
    int q = t2 >> 1, sub = t2 & 1;
    int j = q / NSEG, i = q - j*NSEG;

    float th0 = (PI*0.5f)*j/NRINGS, th1 = (PI*0.5f)*(j+1)/NRINGS;
    float ct0 = cosf(th0), st0 = sinf(th0), ct1 = cosf(th1), st1 = sinf(th1);
    if (bottom) { st0 = -st0; st1 = -st1; }

    float a0 = TWO_PI*i/NSEG, a1 = TWO_PI*(i+1)/NSEG;
    float ca0=cosf(a0),sa0=sinf(a0),ca1=cosf(a1),sa1=sinf(a1);

    float n00[3],n10[3],n01[3],n11[3];
    for (int k = 0; k < 3; k++) {
        n00[k] = (right[k]*ca0 + fwd[k]*sa0)*ct0 + axis[k]*st0;
        n10[k] = (right[k]*ca1 + fwd[k]*sa1)*ct0 + axis[k]*st0;
        n01[k] = (right[k]*ca0 + fwd[k]*sa0)*ct1 + axis[k]*st1;
        n11[k] = (right[k]*ca1 + fwd[k]*sa1)*ct1 + axis[k]*st1;
    }
    // top:    tri0 (n00,n10,n11)  tri1 (n00,n11,n01)
    // bottom: tri0 (n00,n11,n10)  tri1 (n00,n01,n11)
    float nb[3], nc[3];
    for (int k = 0; k < 3; k++) {
        float b_top = sub ? n11[k] : n10[k];
        float b_bot = sub ? n01[k] : n11[k];
        nb[k] = bottom ? b_bot : b_top;
        float c_top = sub ? n01[k] : n11[k];
        float c_bot = sub ? n11[k] : n10[k];
        nc[k] = bottom ? c_bot : c_top;
    }
    float p0[3], p1[3], p2[3];
    for (int k = 0; k < 3; k++) {
        float ctr = bottom ? bot[k] : top[k];
        p0[k] = ctr + n00[k]*radius;
        p1[k] = ctr + nb[k]*radius;
        p2[k] = ctr + nc[k]*radius;
    }
    emit_tri_pn(pos_out, nrm_out, texcoord_out, texid_out, col_out, vi,
                p0, p1, p2, n00, nb, nc, color, -1);
}

// ── Cylinder, one triangle ──────────────────────────────────────────────────
HIPGL_DEVICE void generate_cylinder_tri(
    const float* gpos, const float* gquat, float radius, float half_len, const float* color,
    float* pos_out, float* nrm_out, float* texcoord_out, int* texid_out, float* col_out,
    int vert_base, int texid, int is_cube, float tex_rep_s, float tex_rep_t, int t)
{
    constexpr int NSEG = CYLINDER_NSEG;
    constexpr float TWO_PI = 6.2831853f;

    float local_axis[3] = {0, 0, 1};
    float axis[3];
    qrot(gquat, local_axis, axis);

    float top[3] = {gpos[0]+axis[0]*half_len, gpos[1]+axis[1]*half_len, gpos[2]+axis[2]*half_len};
    float bot[3] = {gpos[0]-axis[0]*half_len, gpos[1]-axis[1]*half_len, gpos[2]-axis[2]*half_len};

    float up_ref[3] = {0, 0, 1};
    if (fabsf(axis[2]) > 0.9f) { up_ref[0] = 1; up_ref[1] = 0; up_ref[2] = 0; }
    float right[3] = {
        axis[1]*up_ref[2] - axis[2]*up_ref[1],
        axis[2]*up_ref[0] - axis[0]*up_ref[2],
        axis[0]*up_ref[1] - axis[1]*up_ref[0]
    };
    float rlen = sqrtf(right[0]*right[0]+right[1]*right[1]+right[2]*right[2]);
    if (rlen > 1e-8f) { right[0]/=rlen; right[1]/=rlen; right[2]/=rlen; }
    float fwd[3] = {
        right[1]*axis[2] - right[2]*axis[1],
        right[2]*axis[0] - right[0]*axis[2],
        right[0]*axis[1] - right[1]*axis[0]
    };

    float inv_q[4] = {gquat[0], -gquat[1], -gquat[2], -gquat[3]};
    auto write_cyl_tc = [&](int vert, float cx, float sy, float zh) {
        float wdir[3];
        for (int k = 0; k < 3; k++) wdir[k] = right[k]*cx + fwd[k]*sy + axis[k]*zh;
        float odir[3];
        qrot(inv_q, wdir, odir);
        if (is_cube) {
            write_tc3(texcoord_out, vert, odir[0], odir[1], odir[2]);
        } else {
            write_tc2(texcoord_out, vert,
                      0.5f*tex_rep_s*odir[0] - 0.5f,
                      0.5f*tex_rep_t*odir[1] - 0.5f);
        }
    };

    int vi = vert_base + t*3;

    if (t < NSEG*2) {
        int i = t >> 1, sub = t & 1;
        float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        float n0[3], n1[3];
        for (int k = 0; k < 3; k++) { n0[k] = right[k]*c0 + fwd[k]*s0; n1[k] = right[k]*c1 + fwd[k]*s1; }
        float t0[3], t1[3], b0[3], b1[3];
        for (int k = 0; k < 3; k++) {
            t0[k] = top[k]+n0[k]*radius; t1[k] = top[k]+n1[k]*radius;
            b0[k] = bot[k]+n0[k]*radius; b1[k] = bot[k]+n1[k]*radius;
        }
        for (int k = 0; k < 3; k++) {
            pos_out[(vi+0)*3+k] = sub ? t1[k] : t0[k];
            pos_out[(vi+1)*3+k] = sub ? b1[k] : t1[k];
            pos_out[(vi+2)*3+k] = b0[k];
            nrm_out[(vi+0)*3+k] = sub ? n1[k] : n0[k];
            nrm_out[(vi+1)*3+k] = n1[k];
            nrm_out[(vi+2)*3+k] = n0[k];
        }
        // tri0 tc: (c0,s0,1)(c1,s1,1)(c0,s0,-1); tri1: (c1,s1,1)(c1,s1,-1)(c0,s0,-1)
        if (col_out) {
            write_cyl_tc(vi+0, sub ? c1 : c0, sub ? s1 : s0, 1.0f);
            write_cyl_tc(vi+1, c1, s1, sub ? -1.0f : 1.0f);
            write_cyl_tc(vi+2, c0, s0, -1.0f);
            for (int v = 0; v < 3; v++) {
                texid_out[vi+v] = texid;
                col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
                col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
            }
        }
        return;
    }

    int t2 = t - NSEG*2;
    int i, is_bot;
    if (t2 < NSEG) { i = t2; is_bot = 0; } else { i = t2 - NSEG; is_bot = 1; }

    float a0 = TWO_PI * i / NSEG, a1 = TWO_PI * (i+1) / NSEG;
    float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
    float e0[3], e1[3];
    for (int k = 0; k < 3; k++) {
        float ctr = is_bot ? bot[k] : top[k];
        e0[k] = ctr + (right[k]*c0 + fwd[k]*s0)*radius;
        e1[k] = ctr + (right[k]*c1 + fwd[k]*s1)*radius;
    }
    for (int k = 0; k < 3; k++) {
        float ctr = is_bot ? bot[k] : top[k];
        float nn  = is_bot ? -axis[k] : axis[k];
        pos_out[(vi+0)*3+k] = ctr;
        pos_out[(vi+1)*3+k] = is_bot ? e1[k] : e0[k];
        pos_out[(vi+2)*3+k] = is_bot ? e0[k] : e1[k];
        nrm_out[(vi+0)*3+k] = nn;
        nrm_out[(vi+1)*3+k] = nn;
        nrm_out[(vi+2)*3+k] = nn;
    }
    if (col_out) {
        float zh = is_bot ? -1.0f : 1.0f;
        write_cyl_tc(vi+0, 0.0f, 0.0f, zh);
        write_cyl_tc(vi+1, is_bot ? c1 : c0, is_bot ? s1 : s0, zh);
        write_cyl_tc(vi+2, is_bot ? c0 : c1, is_bot ? s0 : s1, zh);
        for (int v = 0; v < 3; v++) {
            texid_out[vi+v] = texid;
            col_out[(vi+v)*4+0]=color[0]; col_out[(vi+v)*4+1]=color[1];
            col_out[(vi+v)*4+2]=color[2]; col_out[(vi+v)*4+3]=color[3];
        }
    }
}



#ifndef HIPGL_MESH_SEG_BLOCK
#define HIPGL_MESH_SEG_BLOCK 256
#endif

template <bool WRITE_ATTR, bool WRITE_NRM = true>
HIPGL_DEVICE inline void body_transform_mesh_seg(
    int gid, int lbase, int cnt,
    int ngeom, int env_idx,
    const float* __restrict__ geom_rgba,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_vert_adr,
    const float* __restrict__ mesh_vert,
    const float* __restrict__ mesh_normal,
    const float* __restrict__ mesh_texcoord,
    const int*   __restrict__ geom_texid_arr,
    const float* __restrict__ geom_texrepeat_arr,
    const unsigned char* __restrict__ tex_data,
    int          ntexture,
    const float* __restrict__ geom_specular_arr,
    const float* __restrict__ geom_shininess_arr,
    const float* __restrict__ geom_emission_arr,
    const float* __restrict__ geom_reflectance_arr,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors,
    float*       __restrict__ out_specular,
    float*       __restrict__ out_shininess,
    float*       __restrict__ out_emission)
{
    int lane = threadIdx.x;
    if (lane >= cnt) return;
    const int local_i = lbase + lane;

    const int mid = geom_meshid[gid];
    const int adr = mesh_vert_adr[mid];
    int src = (adr + local_i) * 3;
    float lp[3] = {mesh_vert[src], mesh_vert[src+1], mesh_vert[src+2]};
    float ln[3] = {mesh_normal[src], mesh_normal[src+1], mesh_normal[src+2]};

    const int pidx3 = env_idx * ngeom * 3 + gid * 3;
    const int qidx4 = env_idx * ngeom * 4 + gid * 4;
    float gpos[3] = {geom_xpos[pidx3], geom_xpos[pidx3+1], geom_xpos[pidx3+2]};
    float gquat[4] = {geom_xquat[qidx4], geom_xquat[qidx4+1], geom_xquat[qidx4+2], geom_xquat[qidx4+3]};

    float wp[3], wn[3];
    qrot(gquat, lp, wp);
    qrot(gquat, ln, wn);

    const int voffset = geom_vert_offset[gid];
    const int dst = voffset + local_i;

    positions[dst*3+0] = gpos[0] + wp[0];
    positions[dst*3+1] = gpos[1] + wp[1];
    positions[dst*3+2] = gpos[2] + wp[2];
    if constexpr (WRITE_NRM) {
    normals[dst*3+0] = wn[0];
    normals[dst*3+1] = wn[1];
    normals[dst*3+2] = wn[2];
    }

    if (!WRITE_ATTR) return;
    write_tc2(texcoords, dst, 0.0f, 0.0f);
    texids[dst] = -1;

    float cr = geom_rgba[gid*4+0];
    float cg = geom_rgba[gid*4+1];
    float cb = geom_rgba[gid*4+2];
    float ca = geom_rgba[gid*4+3];

    int texid = geom_texid_arr ? geom_texid_arr[gid] : -1;
    if (texid >= 0 && texid < ntexture && tex_data && mesh_texcoord) {
        int uv_src = (adr + local_i) * MESH_UV_COMPONENTS;
        float u = mesh_texcoord[uv_src];
        float v = mesh_texcoord[uv_src + 1];
        if (geom_texrepeat_arr) {
            u *= geom_texrepeat_arr[gid * 2 + 0];
            v *= geom_texrepeat_arr[gid * 2 + 1];
        }
        write_tc2(texcoords, dst, u, v);
        texids[dst] = texid;
        cr = 1.0f; cg = 1.0f; cb = 1.0f; ca = 1.0f;
    }

    colors[dst*4+0] = cr;
    colors[dst*4+1] = cg;
    colors[dst*4+2] = cb;
    colors[dst*4+3] = ca;

    if (out_specular)  out_specular[dst]  = geom_specular_arr ? geom_specular_arr[gid] : 0.5f;
    if (out_shininess) out_shininess[dst] = (geom_shininess_arr ? geom_shininess_arr[gid] : 0.5f) * 128.0f;
    if (out_emission) {
        float e = geom_emission_arr ? geom_emission_arr[gid] : 0.0f;
        if (geom_reflectance_arr) e += HIPGL_REFLECT_GAIN * geom_reflectance_arr[gid];
        out_emission[dst] = e;
    }
}

template <bool WRITE_ATTR>
HIPGL_GLOBAL void kernel_transform_mesh_verts(
    int ngeom, int env_idx,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_rgba,       // [ngeom*4]
    const float* __restrict__ geom_xpos,       // [ne*ng*3]
    const float* __restrict__ geom_xquat,      // [ne*ng*4]
    const int*   __restrict__ geom_vert_offset, // [ngeom]
    const int*   __restrict__ geom_meshid,     // [ngeom]
    const int*   __restrict__ mesh_vert_adr,   // [nmesh]
    const int*   __restrict__ mesh_vert_num,   // [nmesh]
    const float* __restrict__ mesh_vert,       // packed mesh verts
    const float* __restrict__ mesh_normal,     // packed mesh normals
    const float* __restrict__ mesh_texcoord,   // packed mesh UVs (may be null)
    const int*   __restrict__ geom_texid_arr,  // [ngeom] texture id per geom
    const float* __restrict__ geom_texrepeat_arr, // [ngeom * 2] texture repeat
    const unsigned char* __restrict__ tex_data, // packed RGB texture data
    int          ntexture,
    const float* __restrict__ geom_specular_arr,  // [ngeom]
    const float* __restrict__ geom_shininess_arr, // [ngeom]
    const float* __restrict__ geom_emission_arr,
    const float* __restrict__ geom_reflectance_arr,  // [ngeom]
    int          total_mesh_verts,             // total vertices to process
    const int*   __restrict__ vert_to_geom,    // [total_mesh_verts]
    const int*   __restrict__ vert_local_idx,  // [total_mesh_verts]
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors,
    float*       __restrict__ out_specular,
    float*       __restrict__ out_shininess,
    float*       __restrict__ out_emission)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= total_mesh_verts) return;

    int gid = vert_to_geom[vid];
    int local_i = vert_local_idx[vid];
    int mid = geom_meshid[gid];

    int src = (mesh_vert_adr[mid] + local_i) * 3;
    float lp[3] = {mesh_vert[src], mesh_vert[src+1], mesh_vert[src+2]};
    float ln[3] = {mesh_normal[src], mesh_normal[src+1], mesh_normal[src+2]};

    int pidx3 = env_idx * ngeom * 3 + gid * 3;
    int qidx4 = env_idx * ngeom * 4 + gid * 4;
    float gpos[3] = {geom_xpos[pidx3], geom_xpos[pidx3+1], geom_xpos[pidx3+2]};
    float gquat[4] = {geom_xquat[qidx4], geom_xquat[qidx4+1], geom_xquat[qidx4+2], geom_xquat[qidx4+3]};

    float wp[3], wn[3];
    qrot(gquat, lp, wp);
    qrot(gquat, ln, wn);

    int voffset = geom_vert_offset[gid];
    int dst = voffset + local_i;

    positions[dst*3+0] = gpos[0] + wp[0];
    positions[dst*3+1] = gpos[1] + wp[1];
    positions[dst*3+2] = gpos[2] + wp[2];
    normals[dst*3+0] = wn[0];
    normals[dst*3+1] = wn[1];
    normals[dst*3+2] = wn[2];

    if (!WRITE_ATTR) return;
    write_tc2(texcoords, dst, 0.0f, 0.0f);
    texids[dst] = -1;

    float cr = geom_rgba[gid*4+0];
    float cg = geom_rgba[gid*4+1];
    float cb = geom_rgba[gid*4+2];
    float ca = geom_rgba[gid*4+3];

    int texid = geom_texid_arr ? geom_texid_arr[gid] : -1;
    if (texid >= 0 && texid < ntexture && tex_data && mesh_texcoord) {
        // MESH_UV_COMPONENTS, not TEXCOORD_COMPONENTS: mesh UVs are genuinely
        // 2D in mjModel and in every OBJ/MSH on disk, and stay 2D in the model
        // descriptor. write_tc2 EXPANDS them to (u, v, 0) as they enter the
        // vertex buffer. Reinterpreting the source at the vertex stride instead
        // would shear the data -- that is the failure mode this constant names.
        int uv_src = (mesh_vert_adr[mid] + local_i) * MESH_UV_COMPONENTS;
        float u = mesh_texcoord[uv_src];
        float v = mesh_texcoord[uv_src + 1];
        if (geom_texrepeat_arr) {
            u *= geom_texrepeat_arr[gid * 2 + 0];
            v *= geom_texrepeat_arr[gid * 2 + 1];
        }
        write_tc2(texcoords, dst, u, v);
        texids[dst] = texid;
        cr = 1.0f;
        cg = 1.0f;
        cb = 1.0f;
        // Textures modulate RGB; retain intrinsic material alpha for blending.
    }

    colors[dst*4+0] = cr;
    colors[dst*4+1] = cg;
    colors[dst*4+2] = cb;
    colors[dst*4+3] = ca;

    if (out_specular)  out_specular[dst]  = geom_specular_arr ? geom_specular_arr[gid] : 0.5f;
    if (out_shininess) out_shininess[dst] = (geom_shininess_arr ? geom_shininess_arr[gid] : 0.5f) * 128.0f;
    if (out_emission) {
        float e = geom_emission_arr ? geom_emission_arr[gid] : 0.0f;
        // see the reflectance note above kernel_build_render_geometry
        if (geom_reflectance_arr) e += HIPGL_REFLECT_GAIN * geom_reflectance_arr[gid];
        out_emission[dst] = e;
    }
}


// ── Triangle-parallel primitive geometry kernel ─────────────────────────────
// blockIdx.y selects the geom, (blockIdx.x, threadIdx.x) the triangle within
// it. Replaces kernel_build_render_geometry's one-thread-per-geom mapping;
// the per-geom scalar setup is identical, only the emit is parallel.
#ifndef HIPGL_GEOM_TRI_BLOCK
#define HIPGL_GEOM_TRI_BLOCK 256
#endif

#ifndef HIPGL_GEOM_TRI_SUBS
#define HIPGL_GEOM_TRI_SUBS 16
#endif

template <bool WRITE_ATTR>
HIPGL_DEVICE inline void body_build_render_geometry_tri(
    int gid, int sub,
    int ngeom, int env_idx,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ geom_rgba,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_vert_count,
    const int*   __restrict__ geom_active,
    const int*   __restrict__ geom_texid_arr,
    const unsigned char* __restrict__ tex_data,
    const int*   __restrict__ tex_iscube,
    int          ntexture,
    const float* __restrict__ geom_texrepeat,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors)
{
    if (gid >= ngeom) return;

    if (geom_active && !geom_active[gid]) return;
    if (geom_vert_count && geom_vert_count[gid] <= 0) return;
    int gtype = geom_type[gid];
    int ntri = tris_for_gtype(gtype);
    int t = sub * HIPGL_GEOM_TRI_BLOCK + threadIdx.x;
    if (t >= ntri) return;

    if (!WRITE_ATTR) { texcoords = nullptr; texids = nullptr; colors = nullptr; }

    int pidx3 = env_idx * ngeom * 3 + gid * 3;
    int qidx4 = env_idx * ngeom * 4 + gid * 4;

    float gpos[3] = {geom_xpos[pidx3], geom_xpos[pidx3+1], geom_xpos[pidx3+2]};
    float gquat[4] = {geom_xquat[qidx4], geom_xquat[qidx4+1], geom_xquat[qidx4+2], geom_xquat[qidx4+3]};

    float gs[3] = {geom_size[gid*3], geom_size[gid*3+1], geom_size[gid*3+2]};
    float gc[4] = {geom_rgba[gid*4], geom_rgba[gid*4+1], geom_rgba[gid*4+2], geom_rgba[gid*4+3]};
    int texid = geom_texid_arr ? geom_texid_arr[gid] : -1;
    float tex_rep_s = 1.0f, tex_rep_t = 1.0f;
    if (geom_texrepeat && texid >= 0) {
        tex_rep_s = geom_texrepeat[gid * 2 + 0];
        tex_rep_t = geom_texrepeat[gid * 2 + 1];
    }

    bool primitive_textured =
        (gtype == GEOM_BOX || gtype == GEOM_PLANE || gtype == GEOM_CYLINDER) &&
        texid >= 0 && texid < ntexture && tex_data;
    int tex_is_cube = (tex_iscube && texid >= 0 && texid < ntexture)
                      ? tex_iscube[texid] : 0;
    if (primitive_textured) {
        gc[0] = 1.0f; gc[1] = 1.0f; gc[2] = 1.0f;
    }

    int voffset = geom_vert_offset[gid];

    if (gtype == GEOM_BOX) {
        generate_box_tri(gpos, gquat, gs, gc, positions, normals, texcoords, texids, colors,
                         voffset, primitive_textured ? texid : -1,
                         tex_rep_s, tex_rep_t, tex_is_cube, t);
    } else if (gtype == GEOM_SPHERE) {
        generate_sphere_tri(gpos, gs[0], gc, positions, normals, texcoords, texids, colors,
                            voffset, t);
    } else if (gtype == GEOM_CAPSULE) {
        generate_capsule_tri(gpos, gquat, gs[0], gs[1], gc, positions, normals, texcoords,
                             texids, colors, voffset, t);
    } else if (gtype == GEOM_CYLINDER) {
        generate_cylinder_tri(gpos, gquat, gs[0], gs[1], gc, positions, normals, texcoords,
                              texids, colors, voffset,
                              primitive_textured ? texid : -1,
                              tex_is_cube, tex_rep_s, tex_rep_t, t);
    } else if (gtype == GEOM_PLANE) {
        generate_plane_tri(gpos, gquat, gs, gc, positions, normals, texcoords, texids, colors,
                           voffset, primitive_textured ? texid : -1,
                           tex_rep_s, tex_rep_t, t);
    }
}


template <bool WRITE_ATTR, bool WRITE_NRM = true>
HIPGL_GLOBAL void __launch_bounds__(HIPGL_GEOM_TRI_BLOCK)
kernel_build_geometry_fused(
    int nblk, const int4* __restrict__ desc,
    int ngeom, int env_idx,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ geom_rgba,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_vert_count,
    const int*   __restrict__ geom_active,
    const int*   __restrict__ geom_texid_arr,
    const unsigned char* __restrict__ tex_data,
    const int*   __restrict__ tex_iscube,
    int          ntexture,
    const float* __restrict__ geom_texrepeat,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_vert_adr,
    const float* __restrict__ mesh_vert,
    const float* __restrict__ mesh_normal,
    const float* __restrict__ mesh_texcoord,
    const float* __restrict__ geom_specular_arr,
    const float* __restrict__ geom_shininess_arr,
    const float* __restrict__ geom_emission_arr,
    const float* __restrict__ geom_reflectance_arr,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors,
    float*       __restrict__ out_specular,
    float*       __restrict__ out_shininess,
    float*       __restrict__ out_emission,
    int          env_stride,
    long long    slice_stride)
{
    int c = blockIdx.x;
    if (c >= nblk) return;
    const int wsl = blockIdx.y;
    if (wsl) {
        env_idx  += wsl * env_stride;
        positions += (size_t)wsl * (size_t)slice_stride;
        normals   += (size_t)wsl * (size_t)slice_stride;
    }
    int4 d = desc[c];
    if (d.w == 0) {
        body_transform_mesh_seg<WRITE_ATTR, WRITE_NRM>(
            d.x, d.y, d.z, ngeom, env_idx,
            geom_rgba, geom_xpos, geom_xquat, geom_vert_offset,
            geom_meshid, mesh_vert_adr, mesh_vert, mesh_normal, mesh_texcoord,
            geom_texid_arr, geom_texrepeat, tex_data, ntexture,
            geom_specular_arr, geom_shininess_arr, geom_emission_arr,
            geom_reflectance_arr,
            positions, normals, texcoords, texids, colors,
            out_specular, out_shininess, out_emission);
    } else {
        body_build_render_geometry_tri<WRITE_ATTR>(
            d.x, d.y, ngeom, env_idx,
            geom_type, geom_size, geom_rgba, geom_xpos, geom_xquat,
            geom_vert_offset, geom_vert_count, geom_active,
            geom_texid_arr, tex_data, tex_iscube, ntexture, geom_texrepeat,
            positions, normals, texcoords, texids, colors);
    }
}


HIPGL_DEVICE inline void model_emit_box_tri(
    const float* gsize, float* pos_out, float* nrm_out, int vert_base, int t)
{
    float hx = gsize[0], hy = gsize[1], hz = gsize[2];
    constexpr int NQ = BOX_NUMQUADS;

    int qq = t >> 1, sub = t & 1;
    int f  = qq / (NQ*NQ);
    int r  = qq - f*(NQ*NQ);
    int qx = r / NQ;
    int qy = r - qx*NQ;

    struct FaceDef {
        float normal[3];
        float origin[3];
        float u_axis[3];
        float v_axis[3];
    };
    FaceDef fd;
    switch (f) {
        case 0:  fd = FaceDef{{0,0,1},  {-hx,-hy, hz}, {2*hx,0,0},  {0,2*hy,0}};  break;
        case 1:  fd = FaceDef{{0,0,-1}, {-hx,-hy,-hz}, {0,2*hy,0},  {2*hx,0,0}};  break;
        case 2:  fd = FaceDef{{0,-1,0}, {-hx,-hy,-hz}, {2*hx,0,0},  {0,0,2*hz}};  break;
        case 3:  fd = FaceDef{{0,1,0},  {-hx, hy,-hz}, {0,0,2*hz},  {2*hx,0,0}};  break;
        case 4:  fd = FaceDef{{-1,0,0}, {-hx,-hy,-hz}, {0,0,2*hz},  {0,2*hy,0}};  break;
        default: fd = FaceDef{{1,0,0},  { hx,-hy,-hz}, {0,2*hy,0},  {0,0,2*hz}};  break;
    }

    float su0 = (float)qx / NQ, su1 = (float)(qx+1) / NQ;
    float sv0 = (float)qy / NQ, sv1 = (float)(qy+1) / NQ;

    float lpos[4][3];
    float suv[4][2] = {{su0,sv0},{su1,sv0},{su1,sv1},{su0,sv1}};
    for (int c = 0; c < 4; c++) {
        float su = suv[c][0], sv = suv[c][1];
        float lp[3] = {
            fd.origin[0] + su * fd.u_axis[0] + sv * fd.v_axis[0],
            fd.origin[1] + su * fd.u_axis[1] + sv * fd.v_axis[1],
            fd.origin[2] + su * fd.u_axis[2] + sv * fd.v_axis[2]
        };
        lpos[c][0] = lp[0]; lpos[c][1] = lp[1]; lpos[c][2] = lp[2];
    }

    int vi = vert_base + t*3;
    const int tri_c[2][3] = {{0,1,2},{0,2,3}};
    const int* cc = tri_c[sub];
    for (int v = 0; v < 3; v++) {
        for (int k = 0; k < 3; k++) {
            pos_out[(vi+v)*3+k] = lpos[cc[v]][k];
            nrm_out[(vi+v)*3+k] = fd.normal[k];
        }
    }
}

HIPGL_DEVICE inline void model_emit_plane_tri(
    const float* gsize, float* pos_out, float* nrm_out, int vert_base, int t)
{
    float hx = gsize[0] > 0.01f ? gsize[0] : 5.0f;
    float hy = gsize[1] > 0.01f ? gsize[1] : 5.0f;
    const float local_n[3] = {0, 0, 1};

    constexpr int N = PLANE_SUBDIV;
    int q = t >> 1, sub = t & 1;
    int iy = q / N, ix = q - iy * N;
    float fx0 = (float)ix / N, fx1 = (float)(ix+1) / N;
    float fy0 = (float)iy / N, fy1 = (float)(iy+1) / N;

    float cu[3] = {fx0, fx1, sub ? fx0 : fx1};
    float cv[3] = {fy0, sub ? fy1 : fy0, fy1};

    int vi = vert_base + t*3;
    for (int c = 0; c < 3; c++) {
        float lp[3] = {-hx + cu[c]*2*hx, -hy + cv[c]*2*hy, 0};
        for (int k = 0; k < 3; k++) {
            pos_out[(vi+c)*3+k] = lp[k];
            nrm_out[(vi+c)*3+k] = local_n[k];
        }
    }
}

// Same descriptor stream the fused per-world builder consumes (tag 0 = mesh
// segment, tag 1 = primitive tile); the host hands it only the FOLDABLE
// descriptors.
HIPGL_GLOBAL void __launch_bounds__(HIPGL_GEOM_TRI_BLOCK)
kernel_build_geometry_model(
    int nblk, const int4* __restrict__ desc,
    int ngeom,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_vert_adr,
    const float* __restrict__ mesh_vert,
    const float* __restrict__ mesh_normal,
    float*       __restrict__ positions,
    float*       __restrict__ normals)
{
    int c = blockIdx.x;
    if (c >= nblk) return;
    int4 d = desc[c];
    (void)geom_type; (void)geom_size; (void)ngeom;
    if (d.w != 0) return;                 // mesh segments only
    const int lane = threadIdx.x;
    if (lane >= d.z) return;
    const int gid = d.x;
    const int local_i = d.y + lane;
    const int mid = geom_meshid[gid];
    const int src = (mesh_vert_adr[mid] + local_i) * 3;
    const int dst = geom_vert_offset[gid] + local_i;
    normals[dst*3+0] = mesh_normal[src];
    normals[dst*3+1] = mesh_normal[src+1];
    normals[dst*3+2] = mesh_normal[src+2];
    positions[dst*3+0] = mesh_vert[src];
    positions[dst*3+1] = mesh_vert[src+1];
    positions[dst*3+2] = mesh_vert[src+2];
}

template <bool WRITE_ATTR>
HIPGL_GLOBAL void __launch_bounds__(HIPGL_GEOM_TRI_BLOCK)
kernel_build_render_geometry_tri(
    int nwork, const int* __restrict__ work_list,
    int ngeom, int env_idx,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ geom_rgba,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_vert_count,
    const int*   __restrict__ geom_active,
    const int*   __restrict__ geom_texid_arr,
    const unsigned char* __restrict__ tex_data,
    const int*   __restrict__ tex_iscube,
    int          ntexture,
    const float* __restrict__ geom_texrepeat,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors)
{
    int w = blockIdx.x;
    if (w >= nwork) return;
    int packed = work_list[w];
    int gid = packed / HIPGL_GEOM_TRI_SUBS;
    int sub = packed - gid * HIPGL_GEOM_TRI_SUBS;
    body_build_render_geometry_tri<WRITE_ATTR>(
        gid, sub, ngeom, env_idx,
        geom_type, geom_size, geom_rgba, geom_xpos, geom_xquat,
        geom_vert_offset, geom_vert_count, geom_active,
        geom_texid_arr, tex_data, tex_iscube, ntexture, geom_texrepeat,
        positions, normals, texcoords, texids, colors);
}

template <bool WRITE_ATTR>
HIPGL_GLOBAL void __launch_bounds__(HIPGL_MESH_SEG_BLOCK)
kernel_transform_mesh_seg(
    int nseg, const int4* __restrict__ seg,
    int ngeom, int env_idx,
    const float* __restrict__ geom_rgba,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    const int*   __restrict__ geom_vert_offset,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_vert_adr,
    const float* __restrict__ mesh_vert,
    const float* __restrict__ mesh_normal,
    const float* __restrict__ mesh_texcoord,
    const int*   __restrict__ geom_texid_arr,
    const float* __restrict__ geom_texrepeat_arr,
    const unsigned char* __restrict__ tex_data,
    int          ntexture,
    const float* __restrict__ geom_specular_arr,
    const float* __restrict__ geom_shininess_arr,
    const float* __restrict__ geom_emission_arr,
    const float* __restrict__ geom_reflectance_arr,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors,
    float*       __restrict__ out_specular,
    float*       __restrict__ out_shininess,
    float*       __restrict__ out_emission)
{
    int c = blockIdx.x;
    if (c >= nseg) return;
    int4 d = seg[c];
    body_transform_mesh_seg<WRITE_ATTR>(
        d.x, d.y, d.z, ngeom, env_idx,
        geom_rgba, geom_xpos, geom_xquat, geom_vert_offset,
        geom_meshid, mesh_vert_adr, mesh_vert, mesh_normal, mesh_texcoord,
        geom_texid_arr, geom_texrepeat_arr, tex_data, ntexture,
        geom_specular_arr, geom_shininess_arr, geom_emission_arr,
        geom_reflectance_arr,
        positions, normals, texcoords, texids, colors,
        out_specular, out_shininess, out_emission);
}

// ── Main geometry builder kernel (primitives only) ──────────────────────────
// One thread per geom. Mesh geoms are handled by kernel_transform_mesh_verts.

HIPGL_GLOBAL void kernel_build_render_geometry(
    int ngeom, int env_idx,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,       // [ngeom*3]
    const float* __restrict__ geom_rgba,       // [ngeom*4]
    const float* __restrict__ geom_xpos,       // [ne*ng*3]
    const float* __restrict__ geom_xquat,      // [ne*ng*4]
    const int*   __restrict__ geom_vert_offset, // [ngeom]
    const int*   __restrict__ geom_meshid,     // [ngeom] mesh index, -1 for primitives
    const int*   __restrict__ mesh_vert_adr,   // [nmesh] or null
    const int*   __restrict__ mesh_vert_num,   // [nmesh] or null
    const float* __restrict__ mesh_vert,       // packed mesh verts or null
    const float* __restrict__ mesh_normal,     // packed mesh normals or null
    const int*   __restrict__ geom_texid_arr,  // [ngeom] texture id, -1 if none
    const unsigned char* __restrict__ tex_data,
    const int*   __restrict__ tex_width,
    const int*   __restrict__ tex_height,
    const int*   __restrict__ tex_offset,
    const int*   __restrict__ tex_iscube,      // [ntexture] 1 = cube map, may be null
    int          ntexture,
    const float* __restrict__ geom_texrepeat,  // [ngeom*2] texture repeat (S, T)
    const float* __restrict__ geom_specular_arr, // [ngeom] MuJoCo specular per geom
    const float* __restrict__ geom_shininess_arr,// [ngeom] MuJoCo shininess per geom
    const float* __restrict__ geom_emission_arr, // [ngeom] MuJoCo emission per geom
    const float* __restrict__ geom_reflectance_arr, // [ngeom] see reflectance note at top
    float*       __restrict__ positions,       // output
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors,
    float*       __restrict__ out_specular,    // per-vertex specular
    float*       __restrict__ out_shininess,   // per-vertex shininess
    float*       __restrict__ out_emission,    // per-vertex emission
    int*         __restrict__ total_verts_out)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ngeom) return;

    int pidx3 = env_idx * ngeom * 3 + gid * 3;
    int qidx4 = env_idx * ngeom * 4 + gid * 4;

    float gpos[3] = {geom_xpos[pidx3], geom_xpos[pidx3+1], geom_xpos[pidx3+2]};
    float gquat[4] = {geom_xquat[qidx4], geom_xquat[qidx4+1], geom_xquat[qidx4+2], geom_xquat[qidx4+3]};

    int gtype = geom_type[gid];
    float gs[3] = {geom_size[gid*3], geom_size[gid*3+1], geom_size[gid*3+2]};
    float gc[4] = {geom_rgba[gid*4], geom_rgba[gid*4+1], geom_rgba[gid*4+2], geom_rgba[gid*4+3]};
    int texid = geom_texid_arr ? geom_texid_arr[gid] : -1;
    float tex_rep_s = 1.0f, tex_rep_t = 1.0f;
    if (geom_texrepeat && texid >= 0) {
        tex_rep_s = geom_texrepeat[gid * 2 + 0];
        tex_rep_t = geom_texrepeat[gid * 2 + 1];
    }

    // For textured box/plane geoms, use white base color so fragment texturing
    // provides the full base color before Gouraud lighting is applied.
    // (matching MuJoCo's GL_COLOR_MATERIAL + GL_MODULATE where mat_rgba=(1,1,1))
    bool primitive_textured =
        (gtype == GEOM_BOX || gtype == GEOM_PLANE || gtype == GEOM_CYLINDER) &&
        texid >= 0 && texid < ntexture && tex_data;
    int tex_is_cube = (tex_iscube && texid >= 0 && texid < ntexture)
                      ? tex_iscube[texid] : 0;
    if (primitive_textured) {
        gc[0] = 1.0f; gc[1] = 1.0f; gc[2] = 1.0f;
    }

    int voffset = geom_vert_offset[gid];
    int nverts = 0;

    if (gtype == GEOM_MESH) {
        return;
    } else if (gtype == GEOM_BOX) {
        generate_box(gpos, gquat, gs, gc, positions, normals, texcoords, texids, colors, voffset,
                     primitive_textured ? texid : -1,
                     tex_rep_s, tex_rep_t, tex_is_cube);
        nverts = BOX_VERTS;
    } else if (gtype == GEOM_SPHERE) {
        generate_sphere(gpos, gs[0], gc, positions, normals, texcoords, texids, colors, voffset);
        nverts = SPHERE_VERTS;
    } else if (gtype == GEOM_CAPSULE) {
        generate_capsule(gpos, gquat, gs[0], gs[1], gc, positions, normals, texcoords, texids, colors, voffset);
        nverts = CAPSULE_VERTS;
    } else if (gtype == GEOM_CYLINDER) {
        generate_cylinder(gpos, gquat, gs[0], gs[1], gc, positions, normals, texcoords, texids, colors, voffset,
                          primitive_textured ? texid : -1,
                          tex_is_cube, tex_rep_s, tex_rep_t);
        nverts = CYLINDER_VERTS;
    } else if (gtype == GEOM_PLANE) {
        generate_plane(gpos, gquat, gs, gc, positions, normals, texcoords, texids, colors, voffset,
                       primitive_textured ? texid : -1,
                       tex_rep_s, tex_rep_t);
        nverts = PLANE_VERTS;
    }

    (void)nverts; (void)voffset;
}

HIPGL_GLOBAL void kernel_fill_primitive_materials(
    int n_verts,
    const int*   __restrict__ prim_vert_to_geom,
    const float* __restrict__ geom_specular_arr,
    const float* __restrict__ geom_shininess_arr,
    const float* __restrict__ geom_emission_arr,
    const float* __restrict__ geom_reflectance_arr,
    float*       __restrict__ out_specular,
    float*       __restrict__ out_shininess,
    float*       __restrict__ out_emission)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_verts) return;
    int gid = prim_vert_to_geom[v];
    if (gid < 0) return;  // not a primitive vertex -- leave untouched

    if (out_specular && out_shininess) {
        out_specular[v]  = geom_specular_arr ? geom_specular_arr[gid] : 0.5f;
        out_shininess[v] = (geom_shininess_arr ? geom_shininess_arr[gid] : 0.5f) * 128.0f;
    }
    if (out_emission) {
        float emis = geom_emission_arr ? geom_emission_arr[gid] : 0.0f;
        // see the reflectance note at the top of this file
        if (geom_reflectance_arr) emis += HIPGL_REFLECT_GAIN * geom_reflectance_arr[gid];
        out_emission[v] = emis;
    }
}

HIPGL_GLOBAL void kernel_build_render_sites(
    int nsite, int env_idx,
    const int*   __restrict__ site_type,
    const int*   __restrict__ site_group,
    const float* __restrict__ site_size,
    const float* __restrict__ site_rgba,
    const float* __restrict__ site_xpos,
    const float* __restrict__ site_xquat,
    const int*   __restrict__ site_vert_offset,
    float*       __restrict__ positions,
    float*       __restrict__ normals,
    float*       __restrict__ texcoords,
    int*         __restrict__ texids,
    float*       __restrict__ colors)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= nsite) return;

    float gc[4] = {
        site_rgba[sid*4+0], site_rgba[sid*4+1],
        site_rgba[sid*4+2], site_rgba[sid*4+3]
    };
    if (gc[3] < 0.01f || site_group[sid] >= 3) return;

    int pidx3 = env_idx * nsite * 3 + sid * 3;
    int qidx4 = env_idx * nsite * 4 + sid * 4;
    float gpos[3] = {site_xpos[pidx3], site_xpos[pidx3+1], site_xpos[pidx3+2]};
    float gquat[4] = {site_xquat[qidx4], site_xquat[qidx4+1], site_xquat[qidx4+2], site_xquat[qidx4+3]};
    float gs[3] = {site_size[sid*3], site_size[sid*3+1], site_size[sid*3+2]};
    int stype = site_type[sid];
    int voffset = site_vert_offset[sid];

    if (stype == SITE_TYPE_SPHERE) {
        generate_sphere(gpos, gs[0], gc, positions, normals, texcoords, texids, colors, voffset);
    } else if (stype == SITE_TYPE_CAPSULE || stype == SITE_TYPE_CYLINDER) {
        generate_capsule(gpos, gquat, gs[0], gs[1], gc, positions, normals, texcoords, texids, colors, voffset);
    } else if (stype == SITE_TYPE_BOX) {
        generate_box(gpos, gquat, gs, gc, positions, normals, texcoords, texids, colors, voffset, -1);
    }
}

// ── Host wrapper ────────────────────────────────────────────────────────────

int geom_tri_subblocks(int gtype) {
    int ntri = 0;
    if      (gtype == GEOM_BOX)      ntri = BOX_TRIS;
    else if (gtype == GEOM_SPHERE)   ntri = SPHERE_TRIS;
    else if (gtype == GEOM_CAPSULE)  ntri = CAPSULE_TRIS;
    else if (gtype == GEOM_CYLINDER) ntri = CYLINDER_TRIS;
    else if (gtype == GEOM_PLANE)    ntri = PLANE_TRIS;
    if (ntri <= 0) return 0;
    int n = (ntri + HIPGL_GEOM_TRI_BLOCK - 1) / HIPGL_GEOM_TRI_BLOCK;
    return n < HIPGL_GEOM_TRI_SUBS ? n : HIPGL_GEOM_TRI_SUBS;
}
int geom_tri_pack(int gid, int sub) { return gid * HIPGL_GEOM_TRI_SUBS + sub; }

void launch_build_render_geometry(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    const int* d_geom_vert_count,
    const int* d_geom_active,
    const int* d_geom_tri_work, int n_tri_work,
    bool write_attribs,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    int* d_total_verts,
    hipStream_t stream)
{
    if (ngeom <= 0 || n_tri_work <= 0 || !d_geom_tri_work) return;
    const int block = HIPGL_GEOM_TRI_BLOCK;
    dim3 grid((unsigned)n_tri_work, 1u, 1u);
    auto kfn = write_attribs ? kernel_build_render_geometry_tri<true>
                             : kernel_build_render_geometry_tri<false>;
    hipLaunchKernelGGL(kfn, grid, dim3(block,1,1), 0, stream,
        n_tri_work, d_geom_tri_work,
        ngeom, env_idx,
        model.geom_type, model.geom_size, model.geom_rgba,
        data.geom_xpos, data.geom_xquat,
        d_geom_vert_offset,
        d_geom_vert_count,
        d_geom_active,
        model.geom_texid,
        model.tex_data,
        model.tex_iscube,
        model.ntexture,
        model.geom_texrepeat,
        d_positions, d_normals, d_texcoords, d_texids, d_colors);
    (void)d_specular; (void)d_shininess; (void)d_emission; (void)d_total_verts;
}

void launch_fill_primitive_materials(
    int n_verts,
    const int* d_prim_vert_to_geom,
    const DevicePhysicsModel& model,
    float* d_specular, float* d_shininess, float* d_emission,
    hipStream_t stream)
{
    if (n_verts <= 0 || !d_prim_vert_to_geom) return;
    const int block = 256;  // pure elementwise store; wide blocks are fine here
    int grid = (n_verts + block - 1) / block;
    hipLaunchKernelGGL(kernel_fill_primitive_materials, grid, block, 0, stream,
        n_verts, d_prim_vert_to_geom,
        model.geom_specular, model.geom_shininess, model.geom_emission,
        model.geom_reflectance,
        d_specular, d_shininess, d_emission);
}

void launch_build_render_sites(
    int nsite, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_site_vert_offset,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    hipStream_t stream)
{
    if (nsite <= 0) return;
    const int block = 256;
    int grid = (nsite + block - 1) / block;
    hipLaunchKernelGGL(kernel_build_render_sites, grid, block, 0, stream,
        nsite, env_idx,
        model.site_type, model.site_group,
        model.site_size, model.site_rgba,
        data.site_xpos, data.site_xquat,
        d_site_vert_offset,
        d_positions, d_normals, d_texcoords, d_texids, d_colors);
}

int mesh_seg_block() { return HIPGL_MESH_SEG_BLOCK; }

void launch_build_geometry_fused(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    const int* d_geom_vert_count,
    const int* d_geom_active,
    int nblk,
    const void* d_desc,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs, bool write_normals,
    hipStream_t stream,
    int nworlds, int env_stride, long long slice_stride)
{
    if (nblk <= 0 || !d_desc) return;
    if (nworlds < 1) nworlds = 1;
    auto kfn = write_attribs ? kernel_build_geometry_fused<true>
                             : (write_normals ? kernel_build_geometry_fused<false, true>
                                              : kernel_build_geometry_fused<false, false>);
    hipLaunchKernelGGL(kfn, dim3((unsigned)nblk,(unsigned)nworlds,1), dim3(HIPGL_GEOM_TRI_BLOCK,1,1), 0, stream,
        nblk, (const int4*)d_desc,
        ngeom, env_idx,
        model.geom_type, model.geom_size, model.geom_rgba,
        data.geom_xpos, data.geom_xquat,
        d_geom_vert_offset, d_geom_vert_count, d_geom_active,
        model.geom_texid, model.tex_data, model.tex_iscube, model.ntexture,
        model.geom_texrepeat,
        model.geom_meshid, model.mesh_vert_adr,
        model.mesh_vert, model.mesh_normal, model.mesh_texcoord,
        model.geom_specular, model.geom_shininess, model.geom_emission,
        model.geom_reflectance,
        d_positions, d_normals, d_texcoords, d_texids, d_colors,
        d_specular, d_shininess, d_emission,
        env_stride, slice_stride);
}

void launch_build_geometry_model(
    int ngeom,
    const DevicePhysicsModel& model,
    const int* d_geom_vert_offset,
    int nblk,
    const void* d_desc,
    float* d_positions, float* d_normals,
    hipStream_t stream)
{
    if (nblk <= 0 || !d_desc) return;
    hipLaunchKernelGGL(kernel_build_geometry_model,
        dim3((unsigned)nblk,1,1), dim3(HIPGL_GEOM_TRI_BLOCK,1,1), 0, stream,
        nblk, (const int4*)d_desc, ngeom,
        model.geom_type, model.geom_size,
        d_geom_vert_offset, model.geom_meshid, model.mesh_vert_adr,
        model.mesh_vert, model.mesh_normal,
        d_positions, d_normals);
}

void launch_transform_mesh_seg(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    int nseg,
    const void* d_seg,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs,
    hipStream_t stream)
{
    if (nseg <= 0 || !d_seg) return;
    auto kfn = write_attribs ? kernel_transform_mesh_seg<true>
                             : kernel_transform_mesh_seg<false>;
    hipLaunchKernelGGL(kfn, dim3((unsigned)nseg,1,1), dim3(HIPGL_MESH_SEG_BLOCK,1,1), 0, stream,
        nseg, (const int4*)d_seg,
        ngeom, env_idx,
        model.geom_rgba,
        data.geom_xpos, data.geom_xquat,
        d_geom_vert_offset,
        model.geom_meshid,
        model.mesh_vert_adr,
        model.mesh_vert, model.mesh_normal,
        model.mesh_texcoord,
        model.geom_texid,
        model.geom_texrepeat,
        model.tex_data,
        model.ntexture,
        model.geom_specular, model.geom_shininess, model.geom_emission,
        model.geom_reflectance,
        d_positions, d_normals, d_texcoords, d_texids, d_colors,
        d_specular, d_shininess, d_emission);
}

void launch_transform_mesh_verts(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    int total_mesh_verts,
    const int* d_vert_to_geom,
    const int* d_vert_local_idx,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs,
    hipStream_t stream)
{
    if (total_mesh_verts <= 0) return;
    const int block = 256;
    int grid = (total_mesh_verts + block - 1) / block;
    // Hipify safety: no <<<>>> inside an if/else -- pick the instantiation via
    // a function pointer to the two template specialisations.
    auto kfn = write_attribs ? kernel_transform_mesh_verts<true>
                             : kernel_transform_mesh_verts<false>;
    hipLaunchKernelGGL(kfn, grid, block, 0, stream,
        ngeom, env_idx,
        model.geom_type, model.geom_rgba,
        data.geom_xpos, data.geom_xquat,
        d_geom_vert_offset,
        model.geom_meshid,
        model.mesh_vert_adr, model.mesh_vert_num,
        model.mesh_vert, model.mesh_normal,
        model.mesh_texcoord,
        model.geom_texid,
        model.geom_texrepeat,
        model.tex_data,
        model.ntexture,
        model.geom_specular, model.geom_shininess, model.geom_emission,
        model.geom_reflectance,
        total_mesh_verts,
        d_vert_to_geom, d_vert_local_idx,
        d_positions, d_normals, d_texcoords, d_texids, d_colors,
        d_specular, d_shininess, d_emission);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
