// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Fused vertex transform + triangle setup kernel
// One thread per triangle: transforms 3 vertices, builds edge equations and bbox.
// Eliminates the intermediate TransformedVertex buffer write/read between separate kernels.

#include <hip/hip_runtime.h>
#include "hipgl/types.h"
#include "hipgl/math_types.h"
#include "hipgl/kernels.h"
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include <cstddef>

namespace hipgl {
namespace kernels {

// Sub-pixel area cull threshold, in device constant memory so it can be
// ablated at runtime (HIPGL_MIN_TRI_AREA) without changing five launch
// signatures. Defaults to DEFAULT_MIN_TRI_AREA = the historical 0.25f, so a
// build that never calls set_min_tri_area() behaves exactly as before.
__device__ __constant__ float d_min_tri_area = DEFAULT_MIN_TRI_AREA;

__device__ __forceinline__ uint32_t hipgl_f2ord_setup(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int hp_subdraw_of_setup(int idx, const BinSubdraws& sd)
{
    int r = 0;
#pragma unroll
    for (int k = 0; k < 7; ++k) r += (k < sd.nsub - 1 && idx >= sd.split[k]) ? 1 : 0;
    if (sd.nsub > 8) {
#pragma unroll
        for (int k = 7; k < HP_MAX_SUBDRAWS - 1; ++k)
            r += (k < sd.nsub - 1 && idx >= sd.split[k]) ? 1 : 0;
    }
    return hp_subdraw_perm(r, sd);
}

__device__ __forceinline__ void hp_emit_binrec(const BinEmit& e, int idx, bool packed_ids,
                                               bool live, int bb_min_x, int bb_min_y,
                                               int bb_max_x, int bb_max_y, uint32_t z_in)
{
    int lo_x = 0, lo_y = 0, nx = 1, ny = 0;
    uint32_t z_ord = 0x80000000u;           // == hipgl_f2ord(0.0f), the empty-record value
    if (live) {
        const int tmin_x = math::max2i(bb_min_x, 0);
        const int tmin_y = math::max2i(bb_min_y, 0);
        const int tmax_x = math::min2i(bb_max_x, e.fb_w - 1);
        const int tmax_y = math::min2i(bb_max_y, e.fb_h - 1);
        if (tmin_x <= tmax_x && tmin_y <= tmax_y) {
            lo_x = math::max2i(tmin_x / TILE_SIZE, 0);
            const int hi_x = math::min2i(tmax_x / TILE_SIZE, e.tiles_x - 1);
            lo_y = math::max2i(tmin_y / TILE_SIZE, 0);
            const int hi_y = math::min2i(tmax_y / TILE_SIZE, e.tiles_y - 1);
            if (lo_x <= hi_x && lo_y <= hi_y) {
                nx = hi_x - lo_x + 1;
                ny = hi_y - lo_y + 1;
                z_ord = z_in;
            }
        }
    }
    if (e.subs.nsub > 1) {
        const int sb = hp_subdraw_of_setup(idx, e.subs);
        if ((e.subs.dead_mask >> sb) & 1) {
            lo_x = 0; nx = 1; ny = 0; z_ord = 0x80000000u;
        }
        lo_y += sb * e.tiles_y;
    }
    BinRec r;
    r.xy  = (unsigned int)lo_x | ((unsigned int)lo_y << 16);
    r.nxy = (unsigned int)nx   | ((unsigned int)ny   << 16);
    r.key = ((unsigned long long)z_ord << 32)
          | (unsigned long long)(uint32_t)(packed_ids ? 2 * idx : idx);
    e.rec[idx] = r;
}

static inline int hp_nc_fold_flag() {
    static const int v = [] {
        const char* e = getenv("HP_NC_FOLD");
        return e ? atoi(e) : 1;
    }();
    return v;
}
bool fused_vertex_binrec_supported() { return hp_nc_fold_flag() != 0; }

HIPGL_DEVICE __forceinline__ void hp_qrot(const float* q, const float* v, float* out) {
    float tx = 2.0f * (q[2]*v[2] - q[3]*v[1]);
    float ty = 2.0f * (q[3]*v[0] - q[1]*v[2]);
    float tz = 2.0f * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}

HIPGL_DEVICE __forceinline__ void hp_pose_apply3(
    float (&P)[3][3],
    float q0, float q1, float q2, float q3,
    float gx, float gy, float gz)
{
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float vx = P[k][0], vy = P[k][1], vz = P[k][2];
        const float tx = 2.0f * (q2*vz - q3*vy);
        const float ty = 2.0f * (q3*vx - q1*vz);
        const float tz = 2.0f * (q1*vy - q2*vx);
        P[k][0] = gx + (vx + q0*tx + (q2*tz - q3*ty));
        P[k][1] = gy + (vy + q0*ty + (q3*tx - q1*tz));
        P[k][2] = gz + (vz + q0*tz + (q1*ty - q2*tx));
    }
}

// Thread-local latch: the worker thread that enqueued build_geometry(env) for
// this world is the same thread that issues its draws, in order, on its own
// stream.
static thread_local VtxPoseSrc t_pose_src{};
static thread_local bool       t_pose_valid = false;

void set_vertex_pose_source(const VtxPoseSrc& src) { t_pose_src = src; t_pose_valid = true; }
void clear_vertex_pose_source() { t_pose_valid = false; }

static inline int hp_vtx_pose_on() {
    static const int v = [] {
        const char* e = getenv("HP_VTX_POSE");
        return e ? atoi(e) : 1;
    }();
    return v;
}

// Resolve the draw's pose descriptor. A draw whose position pointer is not
// inside the latched buffer (site overlays, host-uploaded arrays) gets an
// EMPTY descriptor and runs the original, unfolded kernel.
static inline VtxPose hp_pose_for_draw(const float* positions) {
    VtxPose vp{};
    if (!hp_vtx_pose_on() || !t_pose_valid) return vp;
    const VtxPoseSrc& s = t_pose_src;
    if (!s.vert_geom || !s.xpos || !s.xquat || !s.pos_base || s.nvert <= 0) return vp;
    if (positions < s.pos_base) return vp;
    const long long d = (long long)(positions - s.pos_base);
    if (d % 9 != 0) return vp;            // vertex base must be a whole triangle
    const long long v0 = d / 3;
    if (v0 >= s.nvert) return vp;
    vp.vert_geom = s.vert_geom;
    vp.xpos = s.xpos;
    vp.xquat = s.xquat;
    vp.tbase = (int)(v0 / 3);
    vp.ngeom = s.ngeom;
    vp.env = s.env;
    return vp;
}

void set_min_tri_area(float v) {
    hipMemcpyToSymbol(HIP_SYMBOL(d_min_tri_area), &v, sizeof(float));
}

__device__ __constant__ int d_geom_slim = 1;

static int hipgl_geom_slim_env() {
    const char* e = getenv("HP_GEOM_SLIM");
    return e ? atoi(e) : 1;
}

// Copies ONLY the HOT-64 prefix (screen_z, inv_w, color[4], lit_color[4],
// texcoord[3], texid, mat_specular, mat_shininess) -- 16 contiguous dwords on a
// 64-B boundary.  No arithmetic: the values are the identical registers the
// full-struct assignment would have stored.
HIPGL_DEVICE inline void store_vertex_hot64(TransformedVertex& __restrict__ dst,
                                            const TransformedVertex& src)
{
    dst.screen_z = src.screen_z;
    dst.inv_w    = src.inv_w;
#pragma unroll
    for (int i = 0; i < 4; i++) dst.color[i] = src.color[i];
#pragma unroll
    for (int i = 0; i < 4; i++) dst.lit_color[i] = src.lit_color[i];
#pragma unroll
    for (int k = 0; k < TEXCOORD_COMPONENTS; k++) dst.texcoord[k] = src.texcoord[k];
    dst.texid         = src.texid;
    dst.mat_specular  = src.mat_specular;
    dst.mat_shininess = src.mat_shininess;
}

HIPGL_DEVICE inline void fused_transform_vertex(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int vertex_idx,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex& v,
    const float* __restrict__ clip_in = nullptr,
    const float* __restrict__ pos_in = nullptr,
    const float* __restrict__ n_in = nullptr);

HIPGL_DEVICE inline void fused_transform_vertex_v(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int vertex_idx,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex& v,
    const float* __restrict__ clip_in,
    const float* __restrict__ pos_in,
    bool have_n, float n_in0, float n_in1, float n_in2)
{
    const float* pos = pos_in ? pos_in : (positions + vertex_idx * 3);

    const float x = pos[0], y = pos[1], z = pos[2];

    float nx_in = 0.0f, ny_in = 1.0f, nz_in = 0.0f;
    if (normals) {
        if (have_n) {
            nx_in = n_in0; ny_in = n_in1; nz_in = n_in2;
        } else {
            const float* n = normals + vertex_idx * 3;
            nx_in = n[0]; ny_in = n[1]; nz_in = n[2];
        }
    }
    float tc_in[TEXCOORD_COMPONENTS];
    if (texcoords) {
#pragma unroll
        for (int k = 0; k < TEXCOORD_COMPONENTS; k++)
            tc_in[k] = texcoords[vertex_idx * TEXCOORD_COMPONENTS + k];
    } else {
#pragma unroll
        for (int k = 0; k < TEXCOORD_COMPONENTS; k++) tc_in[k] = 0.0f;
    }
    const int   texid_in = texids ? texids[vertex_idx] : -1;
    float c_in[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    if (colors) {
        c_in[0] = colors[vertex_idx * 4];
        c_in[1] = colors[vertex_idx * 4 + 1];
        c_in[2] = colors[vertex_idx * 4 + 2];
        c_in[3] = colors[vertex_idx * 4 + 3];
    }
    float sc_in[3] = { 0.0f, 0.0f, 0.0f };
    if (spec_colors) {
        sc_in[0] = spec_colors[vertex_idx * 3];
        sc_in[1] = spec_colors[vertex_idx * 3 + 1];
        sc_in[2] = spec_colors[vertex_idx * 3 + 2];
    }
    const float spec_in = per_vert_specular  ? per_vert_specular[vertex_idx]  : 0.5f;
    // physics_renderer.hip already scales MuJoCo [0,1] shininess to OpenGL [0,128]
    const float shin_in = per_vert_shininess ? per_vert_shininess[vertex_idx] : 64.0f;
    const float emis_in = per_vert_emission  ? per_vert_emission[vertex_idx]  : 0.0f;

    float clip[4];
    if (clip_in) {
        clip[0] = clip_in[0]; clip[1] = clip_in[1];
        clip[2] = clip_in[2]; clip[3] = clip_in[3];
    } else {
    clip[0] = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
    clip[1] = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
    clip[2] = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
    clip[3] = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
    }

    float inv_w = 1.0f;
    float ndc[3];
    if (fabsf(clip[3]) > 1e-8f) {
        inv_w = 1.0f / clip[3];
        ndc[0] = clip[0] * inv_w;
        ndc[1] = clip[1] * inv_w;
        ndc[2] = clip[2] * inv_w;
    } else {
        ndc[0] = clip[0]; ndc[1] = clip[1]; ndc[2] = clip[2];
    }

    v.screen_z = ndc[2] * 0.5f + 0.5f;
    v.inv_w = inv_w;

    v.world_pos[0] = model_view[0]*x + model_view[4]*y + model_view[8]*z + model_view[12];
    v.world_pos[1] = model_view[1]*x + model_view[5]*y + model_view[9]*z + model_view[13];
    v.world_pos[2] = model_view[2]*x + model_view[6]*y + model_view[10]*z + model_view[14];

    if (normals) {
        v.world_normal[0] = normal_mat[0]*nx_in + normal_mat[3]*ny_in + normal_mat[6]*nz_in;
        v.world_normal[1] = normal_mat[1]*nx_in + normal_mat[4]*ny_in + normal_mat[7]*nz_in;
        v.world_normal[2] = normal_mat[2]*nx_in + normal_mat[5]*ny_in + normal_mat[8]*nz_in;
    } else {
        v.world_normal[0] = 0.0f; v.world_normal[1] = 1.0f; v.world_normal[2] = 0.0f;
    }

    // Loop over TEXCOORD_COMPONENTS rather than naming 0 and 1: with a
    // 3-component vertex texcoord the third carries the cube-map DIRECTION,
    // and a read that stops at component 1 silently drops it, leaving the
    // fragment stage to select a face from an uninitialised value.
#pragma unroll
    for (int k = 0; k < TEXCOORD_COMPONENTS; k++) v.texcoord[k] = tc_in[k];

    v.texid = texid_in;

    v.color[0] = c_in[0];
    v.color[1] = c_in[1];
    v.color[2] = c_in[2];
    v.color[3] = c_in[3];

    v.spec_color[0] = sc_in[0];
    v.spec_color[1] = sc_in[1];
    v.spec_color[2] = sc_in[2];

    v.mat_specular = spec_in;
    v.mat_shininess = shin_in;
    v.mat_emission = emis_in;

    // Default: unlit
    v.lit_color[0] = v.color[0];
    v.lit_color[1] = v.color[1];
    v.lit_color[2] = v.color[2];
    v.lit_color[3] = v.color[3];

    // Setup-only tail, written last to match declaration order so the whole
    // struct still goes out as wide ascending stores.
    v.clip_zw[0] = clip[2]; v.clip_zw[1] = clip[3];
    v.screen_x = vp_x + (ndc[0] * 0.5f + 0.5f) * static_cast<float>(vp_w);
    v.screen_y = vp_y + (1.0f - (ndc[1] * 0.5f + 0.5f)) * static_cast<float>(vp_h);
}

HIPGL_DEVICE inline void fused_transform_vertex(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int vertex_idx,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex& v,
    const float* __restrict__ clip_in,
    const float* __restrict__ pos_in,
    const float* __restrict__ n_in)
{
    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f;
    bool have_n = false;
    if (n_in) { n0 = n_in[0]; n1 = n_in[1]; n2 = n_in[2]; have_n = true; }
    fused_transform_vertex_v(positions, normals, texcoords, texids, colors,
        spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
        vertex_idx, mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, v,
        clip_in, pos_in, have_n, n0, n1, n2);
}

// Per-vertex Gouraud shading: compute lighting at vertex using eye-space
// position, normal, and material properties. Matches MuJoCo's GL_SMOOTH
// Gouraud interpolation where lighting is evaluated per-vertex and the
// resulting color is linearly interpolated across the triangle.
HIPGL_DEVICE inline void compute_gouraud_lighting(
    TransformedVertex& v,
    const RenderState& state)
{
    // MuJoCo's renderGeomReflection scales material RGB before lighting.
    // Ambient, diffuse and emission change; the specular material stays fixed.
    if (state.reflect_dim > 0.0f) {
        v.color[0] *= state.reflect_dim;
        v.color[1] *= state.reflect_dim;
        v.color[2] *= state.reflect_dim;
    }
    float r = v.color[0], g = v.color[1], b = v.color[2];
    float ex = v.world_pos[0], ey = v.world_pos[1], ez = v.world_pos[2];
    float nx = v.world_normal[0], ny = v.world_normal[1], nz = v.world_normal[2];

    float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nlen > 1e-8f) { nx /= nlen; ny /= nlen; nz /= nlen; }
    if (state.two_sided_lighting && nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }


    float frag_mat_spec = v.mat_specular;
    float frag_mat_shin = v.mat_shininess;
    float frag_mat_emis = v.mat_emission;

    float lit_r = frag_mat_emis * r;
    float lit_g = frag_mat_emis * g;
    float lit_b = frag_mat_emis * b;
    float spec_r = 0.0f, spec_g = 0.0f, spec_b = 0.0f;

    // #pragma unroll 1: MAX_LIGHTS is 8 and the body carries two powf's, a
    // cosf and three normalises. Fully unrolling it (the default) emitted
    // ~4100 instructions for this kernel at VGPR 128 with 7 scratch_store /
    // 9 scratch_load of REAL spill traffic, and it is inlined THREE times
    // (once per vertex). The light set is wave-uniform (RenderState arrives in
    // the kernel arg segment), so a rolled loop costs one scalar branch per
    // disabled light and nothing else.
#pragma unroll 1
    for (int li = 0; li < MAX_LIGHTS; li++) {
        if (!state.lights[li].enabled) continue;

        float lx, ly, lz;
        float atten = 1.0f;

        if (state.lights[li].position[3] == 0.0f) {
            lx = state.lights[li].position[0];
            ly = state.lights[li].position[1];
            lz = state.lights[li].position[2];
            float ll = sqrtf(lx*lx + ly*ly + lz*lz);
            if (ll > 1e-8f) { lx /= ll; ly /= ll; lz /= ll; }
        } else {
            lx = state.lights[li].position[0] - ex;
            ly = state.lights[li].position[1] - ey;
            lz = state.lights[li].position[2] - ez;
            float dist = sqrtf(lx*lx + ly*ly + lz*lz);
            if (dist > 1e-8f) { lx /= dist; ly /= dist; lz /= dist; }
            atten = 1.0f / (state.lights[li].constant_atten
                          + state.lights[li].linear_atten * dist
                          + state.lights[li].quadratic_atten * dist * dist);
        }

        float spot_atten = 1.0f;
        if (state.lights[li].spot_cutoff < 180.0f && state.lights[li].position[3] != 0.0f) {
            float sdx = state.lights[li].direction[0];
            float sdy = state.lights[li].direction[1];
            float sdz = state.lights[li].direction[2];
            float sdlen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
            if (sdlen > 1e-8f) { sdx /= sdlen; sdy /= sdlen; sdz /= sdlen; }
            float cos_angle = -(lx*sdx + ly*sdy + lz*sdz);
            float cos_cutoff = cosf(state.lights[li].spot_cutoff * 3.14159265f / 180.0f);
            if (cos_angle < cos_cutoff) {
                spot_atten = 0.0f;
            } else {
                spot_atten = powf(cos_angle, state.lights[li].spot_exponent);
            }
        }
        atten *= spot_atten;

        float NdotL = nx*lx + ny*ly + nz*lz;
        if (NdotL < 0.0f) NdotL = 0.0f;

        lit_r += atten * (state.lights[li].ambient[0] * r
                        + state.lights[li].diffuse[0] * r * NdotL);
        lit_g += atten * (state.lights[li].ambient[1] * g
                        + state.lights[li].diffuse[1] * g * NdotL);
        lit_b += atten * (state.lights[li].ambient[2] * b
                        + state.lights[li].diffuse[2] * b * NdotL);

        if (NdotL > 0.0f && frag_mat_spec > 0.0f) {
            float vdx = -ex, vdy = -ey, vdz = -ez;
            float vlen = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
            if (vlen > 1e-8f) { vdx /= vlen; vdy /= vlen; vdz /= vlen; }
            float hx = lx + vdx, hy = ly + vdy, hz = lz + vdz;
            float hlen = sqrtf(hx*hx + hy*hy + hz*hz);
            if (hlen > 1e-8f) { hx /= hlen; hy /= hlen; hz /= hlen; }

            float NdotH = nx*hx + ny*hy + nz*hz;
            if (NdotH < 0.0f) NdotH = 0.0f;
            float spec_power = (frag_mat_shin > 0.0f) ? powf(NdotH, frag_mat_shin) : 1.0f;

            spec_r += atten * state.lights[li].specular[0] * frag_mat_spec * spec_power;
            spec_g += atten * state.lights[li].specular[1] * frag_mat_spec * spec_power;
            spec_b += atten * state.lights[li].specular[2] * frag_mat_spec * spec_power;
        }
    }

    // GL_SINGLE_COLOR: specular added before texture, so include in lit_color
    auto clamp01 = [](float x) -> float { return x > 1.0f ? 1.0f : (x < 0.0f ? 0.0f : x); };
    v.lit_color[0] = clamp01(lit_r + clamp01(spec_r));
    v.lit_color[1] = clamp01(lit_g + clamp01(spec_g));
    v.lit_color[2] = clamp01(lit_b + clamp01(spec_b));
    v.lit_color[3] = v.color[3];
}


HIPGL_DEVICE inline void compute_gouraud_lighting3(
    TransformedVertex& v0, TransformedVertex& v1, TransformedVertex& v2,
    const RenderState& state)
{
    if (state.reflect_dim > 0.0f) {
        v0.color[0] *= state.reflect_dim; v0.color[1] *= state.reflect_dim; v0.color[2] *= state.reflect_dim;
        v1.color[0] *= state.reflect_dim; v1.color[1] *= state.reflect_dim; v1.color[2] *= state.reflect_dim;
        v2.color[0] *= state.reflect_dim; v2.color[1] *= state.reflect_dim; v2.color[2] *= state.reflect_dim;
    }
    float r[3]  = { v0.color[0], v1.color[0], v2.color[0] };
    float g[3]  = { v0.color[1], v1.color[1], v2.color[1] };
    float b[3]  = { v0.color[2], v1.color[2], v2.color[2] };
    float ex[3] = { v0.world_pos[0], v1.world_pos[0], v2.world_pos[0] };
    float ey[3] = { v0.world_pos[1], v1.world_pos[1], v2.world_pos[1] };
    float ez[3] = { v0.world_pos[2], v1.world_pos[2], v2.world_pos[2] };
    float nx[3] = { v0.world_normal[0], v1.world_normal[0], v2.world_normal[0] };
    float ny[3] = { v0.world_normal[1], v1.world_normal[1], v2.world_normal[1] };
    float nz[3] = { v0.world_normal[2], v1.world_normal[2], v2.world_normal[2] };
    float fms[3] = { v0.mat_specular,  v1.mat_specular,  v2.mat_specular  };
    float fmh[3] = { v0.mat_shininess, v1.mat_shininess, v2.mat_shininess };
    float fme[3] = { v0.mat_emission,  v1.mat_emission,  v2.mat_emission  };
    float lit_r[3], lit_g[3], lit_b[3], spec_r[3], spec_g[3], spec_b[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        float nlen = sqrtf(nx[j]*nx[j] + ny[j]*ny[j] + nz[j]*nz[j]);
        if (nlen > 1e-8f) { nx[j] /= nlen; ny[j] /= nlen; nz[j] /= nlen; }
        if (state.two_sided_lighting && nz[j] < 0.0f) { nx[j] = -nx[j]; ny[j] = -ny[j]; nz[j] = -nz[j]; }
        lit_r[j] = fme[j] * r[j];
        lit_g[j] = fme[j] * g[j];
        lit_b[j] = fme[j] * b[j];
        spec_r[j] = 0.0f; spec_g[j] = 0.0f; spec_b[j] = 0.0f;
    }

#pragma unroll 1
    for (int li = 0; li < MAX_LIGHTS; li++) {
        if (!state.lights[li].enabled) continue;

        float lx[3], ly[3], lz[3];
        float atten[3] = { 1.0f, 1.0f, 1.0f };

        if (state.lights[li].position[3] == 0.0f) {
            float dx = state.lights[li].position[0];
            float dy = state.lights[li].position[1];
            float dz = state.lights[li].position[2];
            float ll = sqrtf(dx*dx + dy*dy + dz*dz);
            if (ll > 1e-8f) { dx /= ll; dy /= ll; dz /= ll; }
#pragma unroll
            for (int j = 0; j < 3; ++j) { lx[j] = dx; ly[j] = dy; lz[j] = dz; }
        } else {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                lx[j] = state.lights[li].position[0] - ex[j];
                ly[j] = state.lights[li].position[1] - ey[j];
                lz[j] = state.lights[li].position[2] - ez[j];
                float dist = sqrtf(lx[j]*lx[j] + ly[j]*ly[j] + lz[j]*lz[j]);
                if (dist > 1e-8f) { lx[j] /= dist; ly[j] /= dist; lz[j] /= dist; }
                atten[j] = 1.0f / (state.lights[li].constant_atten
                                 + state.lights[li].linear_atten * dist
                                 + state.lights[li].quadratic_atten * dist * dist);
            }
        }

        if (state.lights[li].spot_cutoff < 180.0f && state.lights[li].position[3] != 0.0f) {
            float sdx = state.lights[li].direction[0];
            float sdy = state.lights[li].direction[1];
            float sdz = state.lights[li].direction[2];
            float sdlen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
            if (sdlen > 1e-8f) { sdx /= sdlen; sdy /= sdlen; sdz /= sdlen; }
            float cos_cutoff = cosf(state.lights[li].spot_cutoff * 3.14159265f / 180.0f);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                float cos_angle = -(lx[j]*sdx + ly[j]*sdy + lz[j]*sdz);
                float spot_atten;
                if (cos_angle < cos_cutoff) spot_atten = 0.0f;
                else spot_atten = powf(cos_angle, state.lights[li].spot_exponent);
                atten[j] *= spot_atten;
            }
        }

#pragma unroll
        for (int j = 0; j < 3; ++j) {
            float NdotL = nx[j]*lx[j] + ny[j]*ly[j] + nz[j]*lz[j];
            if (NdotL < 0.0f) NdotL = 0.0f;

            lit_r[j] += atten[j] * (state.lights[li].ambient[0] * r[j]
                                  + state.lights[li].diffuse[0] * r[j] * NdotL);
            lit_g[j] += atten[j] * (state.lights[li].ambient[1] * g[j]
                                  + state.lights[li].diffuse[1] * g[j] * NdotL);
            lit_b[j] += atten[j] * (state.lights[li].ambient[2] * b[j]
                                  + state.lights[li].diffuse[2] * b[j] * NdotL);

            if (NdotL > 0.0f && fms[j] > 0.0f) {
                float vdx = -ex[j], vdy = -ey[j], vdz = -ez[j];
                float vlen = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
                if (vlen > 1e-8f) { vdx /= vlen; vdy /= vlen; vdz /= vlen; }
                float hx = lx[j] + vdx, hy = ly[j] + vdy, hz = lz[j] + vdz;
                float hlen = sqrtf(hx*hx + hy*hy + hz*hz);
                if (hlen > 1e-8f) { hx /= hlen; hy /= hlen; hz /= hlen; }

                float NdotH = nx[j]*hx + ny[j]*hy + nz[j]*hz;
                if (NdotH < 0.0f) NdotH = 0.0f;
                float spec_power = (fmh[j] > 0.0f) ? powf(NdotH, fmh[j]) : 1.0f;

                spec_r[j] += atten[j] * state.lights[li].specular[0] * fms[j] * spec_power;
                spec_g[j] += atten[j] * state.lights[li].specular[1] * fms[j] * spec_power;
                spec_b[j] += atten[j] * state.lights[li].specular[2] * fms[j] * spec_power;
            }
        }
    }

    auto clamp01 = [](float x) -> float { return x > 1.0f ? 1.0f : (x < 0.0f ? 0.0f : x); };
    v0.lit_color[0] = clamp01(lit_r[0] + clamp01(spec_r[0]));
    v0.lit_color[1] = clamp01(lit_g[0] + clamp01(spec_g[0]));
    v0.lit_color[2] = clamp01(lit_b[0] + clamp01(spec_b[0]));
    v0.lit_color[3] = v0.color[3];
    v1.lit_color[0] = clamp01(lit_r[1] + clamp01(spec_r[1]));
    v1.lit_color[1] = clamp01(lit_g[1] + clamp01(spec_g[1]));
    v1.lit_color[2] = clamp01(lit_b[1] + clamp01(spec_b[1]));
    v1.lit_color[3] = v1.color[3];
    v2.lit_color[0] = clamp01(lit_r[2] + clamp01(spec_r[2]));
    v2.lit_color[1] = clamp01(lit_g[2] + clamp01(spec_g[2]));
    v2.lit_color[2] = clamp01(lit_b[2] + clamp01(spec_b[2]));
    v2.lit_color[3] = v2.color[3];
}


// ── CULL-ONLY PREPASS ───────────────────────────────────────────────────────
// Everything the near-plane test and the sub-pixel/backface area test need is
// four numbers per vertex: screen_x, screen_y and clip.z + clip.w. Computing
// ONLY those first costs 12 VGPRs for the whole triangle, versus the 87 VGPRs
// that keeping three fully-populated TransformedVertex structs alive across the
// setup arithmetic costs -- which is what pinned this kernel at VGPR 128 with
// 60 B/lane of REAL scratch traffic (7 scratch_store / 9 scratch_load in the
// gfx950 ISA, unlike the fine rasterizer's dead reservation) and 4 waves/SIMD.
//
// The clip-space product is recomputed in fused_transform_vertex afterwards for
// the survivors. That is 16 FMAs per vertex against 87 registers of pressure,
// and it is the SAME source expression, so screen_x/screen_y are bit-identical
// to the value the store path writes.
HIPGL_DEVICE inline void fused_vertex_cull_prepass_p(
    float x, float y, float z,
    const float* __restrict__ mvp,
    int vp_x, int vp_y, int vp_w, int vp_h,
    float* __restrict__ clip,
    float& out_sx, float& out_sy, float& out_zpw, float& out_sz)
{
    clip[0] = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
    clip[1] = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
    clip[2] = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
    clip[3] = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];

    float ndc0, ndc1, ndc2;
    if (fabsf(clip[3]) > 1e-8f) {
        float inv_w = 1.0f / clip[3];
        ndc0 = clip[0] * inv_w;
        ndc1 = clip[1] * inv_w;
        ndc2 = clip[2] * inv_w;
    } else {
        ndc0 = clip[0]; ndc1 = clip[1]; ndc2 = clip[2];
    }
    out_sz = ndc2 * 0.5f + 0.5f;

    out_zpw = clip[2] + clip[3];
    out_sx = vp_x + (ndc0 * 0.5f + 0.5f) * static_cast<float>(vp_w);
    out_sy = vp_y + (1.0f - (ndc1 * 0.5f + 0.5f)) * static_cast<float>(vp_h);
}

// Loading wrapper, kept so any caller without the positions already in
// registers is unchanged.
HIPGL_DEVICE inline void fused_vertex_cull_prepass(
    const float* __restrict__ positions,
    int vertex_idx,
    const float* __restrict__ mvp,
    int vp_x, int vp_y, int vp_w, int vp_h,
    float* __restrict__ clip,
    float& out_sx, float& out_sy, float& out_zpw, float& out_sz)
{
    const float* pos = positions + vertex_idx * 3;
    fused_vertex_cull_prepass_p(pos[0], pos[1], pos[2], mvp,
                                vp_x, vp_y, vp_w, vp_h, clip,
                                out_sx, out_sy, out_zpw, out_sz);
}

HIPGL_DEVICE inline bool reflection_triangle_outside_p(
    const float (&P)[3][3],
    const float* __restrict__ mvp, const float* __restrict__ model_view,
    const float* __restrict__ plane)
{
    const float p0 = plane[0], p1 = plane[1], p2 = plane[2], p3 = plane[3];
    // The eye-space plane, pulled back through model_view into object space.
    const float q0 = p0*model_view[0] + p1*model_view[1] + p2*model_view[2];
    const float q1 = p0*model_view[4] + p1*model_view[5] + p2*model_view[6];
    const float q2 = p0*model_view[8] + p1*model_view[9] + p2*model_view[10];
    const float q3 = p0*model_view[12] + p1*model_view[13] + p2*model_view[14] - p3;
    // The same pull-back of the UNSUMMED-TERM margin. Every factor is an
    // absolute value, so the composition is term-for-term the original sum.
    const float a0 = fabsf(p0), a1 = fabsf(p1), a2 = fabsf(p2);
    const float r0 = a0*fabsf(model_view[0]) + a1*fabsf(model_view[1]) + a2*fabsf(model_view[2]);
    const float r1 = a0*fabsf(model_view[4]) + a1*fabsf(model_view[5]) + a2*fabsf(model_view[6]);
    const float r2 = a0*fabsf(model_view[8]) + a1*fabsf(model_view[9]) + a2*fabsf(model_view[10]);
    const float r3 = a0*fabsf(model_view[12]) + a1*fabsf(model_view[13])
                   + a2*fabsf(model_view[14]) + fabsf(p3);

    float max_distance = -FLT_MAX;
    float max_scale = 1.0f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float x = P[k][0], y = P[k][1], z = P[k][2];
        const float w = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
        if (!(w > 1e-8f) || !isfinite(w)) return false;

        const float distance = q0*x + q1*y + q2*z + q3;
        if (!(distance < 0.0f) || !isfinite(distance)) return false;

        // Include the unsummed transform terms so cancellation in translated
        // scenes cannot make the error allowance artificially small. The
        // margin also covers the subsequent positive-weight interpolation and
        // plane dot product. Vertices close to the reflection plane are
        // deliberately not culled.
        const float scale = r0*fabsf(x) + r1*fabsf(y) + r2*fabsf(z) + r3;
        if (!isfinite(scale)) return false;
        max_distance = fmaxf(max_distance, distance);
        max_scale = fmaxf(max_scale, scale);
    }
    constexpr float CLIP_GUARD = 64.0f * FLT_EPSILON;
    return max_distance < -CLIP_GUARD * max_scale;
}

HIPGL_DEVICE inline bool reflection_triangle_outside(
    const float* __restrict__ positions, int first_vertex,
    const float* __restrict__ mvp, const float* __restrict__ model_view,
    const float* __restrict__ plane)
{
    float P[3][3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float* p = positions + (first_vertex + k) * 3;
        P[k][0] = p[0]; P[k][1] = p[1]; P[k][2] = p[2];
    }
    return reflection_triangle_outside_p(P, mvp, model_view, plane);
}

// The common kernel only queues straddlers. Reconstruction, clipping and
// interpolation live in a separate kernel so the normal path does not carry
// the clipped polygon's vertex registers or local-memory footprint.
#include "near_clip_setup.inc"

// Match the launcher's 128-thread block. The default 1024-thread bound forces
// the fixed clipping temporaries to spill and makes the runtime reclaim GPU
// scratch between otherwise unrelated draws.
HIPGL_GLOBAL __launch_bounds__(128) void kernel_fused_near_clip(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors, const float* per_vert_specular,
    const float* per_vert_shininess, const float* per_vert_emission,
    int source_count, const float* mvp, const float* model_view,
    const float* normal_mat, int vp_x, int vp_y, int vp_w, int vp_h,
    RenderState state, TriangleSetup* out, int32_t* tri_stats,
    NearClipQueue near_clip)
{
    const int count = math::min2i(*near_clip.count, source_count);
    const int stride = blockDim.x * gridDim.x;
    for (int item = blockIdx.x * blockDim.x + threadIdx.x;
         item < count; item += stride) {
        const int source = near_clip.ids[item];
        if (source < 0 || source >= source_count) continue;
        near_clip_emit_source_triangle(positions, normals, texcoords, texids, colors,
            spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
            mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, state,
            source, source, source_count + source, out, tri_stats);
    }
}

static void launch_fused_near_clip(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors, const float* per_vert_specular,
    const float* per_vert_shininess, const float* per_vert_emission,
    int source_count, const float* mvp, const float* model_view,
    const float* normal_mat, int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state, TriangleSetup* out, int32_t* tri_stats,
    hipStream_t stream, NearClipQueue near_clip)
{
    if (!near_clip.ids || !near_clip.count) return;
    // A bounded grid also handles arbitrarily many queued triangles through
    // its device-count loop. Empty queues need no host readback or sync.
    static const int nc_block_env = [] {
        const char* e = getenv("HP_NC_BLOCK");
        return e ? atoi(e) : 128;
    }();
    const int block = nc_block_env;
    const int grid = math::min2i((source_count + block - 1) / block, 64);
    hipLaunchKernelGGL(kernel_fused_near_clip, grid, block, 0, stream,
        positions, normals, texcoords, texids, colors, spec_colors,
        per_vert_specular, per_vert_shininess, per_vert_emission, source_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        state, out, tri_stats, near_clip);
}

#ifndef HP_PLAIN_WPEU
#define HP_PLAIN_WPEU 0
#endif
#if HP_PLAIN_WPEU == 0
#define HP_PLAIN_LBQ(L) __launch_bounds__(L)
#else
#define HP_PLAIN_LBQ(L) __launch_bounds__(L, HP_PLAIN_WPEU)
#endif
template <bool NC_FOLD = false, int LB = 256>
HIPGL_GLOBAL HP_PLAIN_LBQ(LB) void kernel_fused_vertex_tri(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int tri_count,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* __restrict__ out,
    int32_t* __restrict__ tri_stats, NearClipQueue near_clip,
    RenderState nc_state)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tri_count) return;
    const bool clipping = near_clip.ids && near_clip.count;
    const int output_index = idx;
    if (tri_stats) tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_INPUT] = 1;

    TriangleSetup& tri = out[output_index];
    if (clipping && tri_stats) {
        // Diagnostic readers may enumerate every reserved slot. Production
        // binning visits suffix records only through the straddler queue.
        hipgl_mark_degenerate(out[tri_count + idx]);
        out[tri_count + idx].first_vertex = idx * 3;
    }

    // DRAW-LOCAL first vertex. The draw already offsets its base pointers by
    // the group offset, so this is relative to that base; the host adds the
    // offset back at readback, where the per-geom vertex spans live.
    tri.first_vertex = idx * 3;

    int i0 = idx * 3;
    int i1 = idx * 3 + 1;
    int i2 = idx * 3 + 2;

    fused_transform_vertex(positions, normals, texcoords, texids, colors, spec_colors,
                           per_vert_specular, per_vert_shininess, per_vert_emission, i0,
                           mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tri.v[0]);
    fused_transform_vertex(positions, normals, texcoords, texids, colors, spec_colors,
                           per_vert_specular, per_vert_shininess, per_vert_emission, i1,
                           mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tri.v[1]);
    fused_transform_vertex(positions, normals, texcoords, texids, colors, spec_colors,
                           per_vert_specular, per_vert_shininess, per_vert_emission, i2,
                           mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tri.v[2]);

    // Fully hidden triangles can be discarded. Mixed triangles are queued for
    // proper clipping before perspective division when scratch is supplied.
    // In clip space, a vertex is behind near plane when clip.z + clip.w < 0.
    bool behind0 = (tri.v[0].clip_zw[0] + tri.v[0].clip_zw[1]) < 0.0f;
    bool behind1 = (tri.v[1].clip_zw[0] + tri.v[1].clip_zw[1]) < 0.0f;
    bool behind2 = (tri.v[2].clip_zw[0] + tri.v[2].clip_zw[1]) < 0.0f;
    if (behind0 || behind1 || behind2) {
        hipgl_mark_degenerate(tri);
        // NOTE: near-plane clipping and the sub-pixel area cull below BOTH set
        // tri.degenerate, so the flag alone cannot say which fired. Counted
        // separately here.
        if (tri_stats) tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_CLIPPED] = 1;
        if (clipping && !(behind0 && behind1 && behind2)) {
            const int slot = atomicAdd(near_clip.count, 1);
            near_clip.ids[slot] = idx;
            if constexpr (NC_FOLD) {
                near_clip_emit_source_triangle(positions, normals, texcoords, texids,
                    colors, spec_colors, per_vert_specular, per_vert_shininess,
                    per_vert_emission, mvp, model_view, normal_mat,
                    vp_x, vp_y, vp_w, vp_h, nc_state,
                    idx, idx, tri_count + idx, out, tri_stats);
            }
        }
        return;
    }

    // Snap to 1/256 sub-pixel grid so that triangles sharing an edge
    // produce bit-identical edge equations for the shared edge.
    constexpr float SUBPX = 256.0f;
    constexpr float INV_SUBPX = 1.0f / 256.0f;
    float x0 = roundf(tri.v[0].screen_x * SUBPX) * INV_SUBPX;
    float y0 = roundf(tri.v[0].screen_y * SUBPX) * INV_SUBPX;
    float x1 = roundf(tri.v[1].screen_x * SUBPX) * INV_SUBPX;
    float y1 = roundf(tri.v[1].screen_y * SUBPX) * INV_SUBPX;
    float x2 = roundf(tri.v[2].screen_x * SUBPX) * INV_SUBPX;
    float y2 = roundf(tri.v[2].screen_y * SUBPX) * INV_SUBPX;

    float e0x = x1 - x0, e0y = y1 - y0;
    float e1x = x2 - x1, e1y = y2 - y1;
    float e2x = x0 - x2, e2y = y0 - y2;

    float area = e0x * e2y - e0y * e2x;
    hipgl_set_flags(tri, fabsf(area) < d_min_tri_area, area < 0.0f);
    if (tri_stats) {
        if (tri.degenerate)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_DEGENERATE] = 1;
        if (tri.backfacing)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_BACKFACE] = 1;
    }

    float sign = (area > 0.0f) ? -1.0f : 1.0f;

    float A0s = sign * (-e0y), B0s = sign * e0x;
    float A1s = sign * (-e1y), B1s = sign * e1x;
    float A2s = sign * (-e2y), B2s = sign * e2x;

    tri.edge_eq[0][0] = A0s;
    tri.edge_eq[0][1] = B0s;
    tri.edge_eq[0][2] = (float)(-((double)A0s * (double)x0 + (double)B0s * (double)y0));

    tri.edge_eq[1][0] = A1s;
    tri.edge_eq[1][1] = B1s;
    tri.edge_eq[1][2] = (float)(-((double)A1s * (double)x1 + (double)B1s * (double)y1));

    tri.edge_eq[2][0] = A2s;
    tri.edge_eq[2][1] = B2s;
    tri.edge_eq[2][2] = (float)(-((double)A2s * (double)x2 + (double)B2s * (double)y2));

    // Coordinates have 8 fractional bits; the edge-function quantum
    // is their product, 1/65536. A 1/256 bias erodes small triangles.
    constexpr float FILL_BIAS = -1.0f / 65536.0f;
    tri.edge_bias[0] = (A0s > 0.0f || (A0s == 0.0f && B0s > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[1] = (A1s > 0.0f || (A1s == 0.0f && B1s > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[2] = (A2s > 0.0f || (A2s == 0.0f && B2s > 0.0f)) ? 0.0f : FILL_BIAS;

    float min_x = math::min3(x0, x1, x2);
    float min_y = math::min3(y0, y1, y2);
    float max_x = math::max3(x0, x1, x2);
    float max_y = math::max3(y0, y1, y2);

    hipgl_set_bbox(tri, static_cast<int>(floorf(min_x)),
                        static_cast<int>(floorf(min_y)),
                        static_cast<int>(ceilf(max_x)),
                        static_cast<int>(ceilf(max_y)));

    tri.z_ord = hipgl_f2ord_setup(fminf(tri.v[0].screen_z,
                                  fminf(tri.v[1].screen_z, tri.v[2].screen_z)));
}

template <bool REFLECTION_EARLY_CULL, bool REFL_RECT, bool NC_FOLD, int MODE = 0,
          bool POSE_FOLD = false, int POSE_MODE = 1, bool NC_GSTAGE = false>
HIPGL_DEVICE __forceinline__ void hp_fused_vtx_body(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int tri_count,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state,
    TriangleSetup* __restrict__ out, NearClipQueue near_clip,
    int bf_skip_mask, BinEmit emit, bool refl_active, VtxPose pose = VtxPose{},
    unsigned hp_bx = 0u, bool hp_bx_given = false,
    bool* hp_nc_queued = nullptr)
{
    const unsigned hp_blk = hp_bx_given ? hp_bx : blockIdx.x;
    int idx, vk;
    if constexpr (MODE == 2) {
        const int gid = (int)hp_blk * blockDim.x + threadIdx.x;
        idx = gid / 3;
        vk  = gid - idx * 3;
    } else {
        idx = (int)hp_blk * blockDim.x + threadIdx.x;
        vk  = 0;
    }
    // The one lane of the triangle that owns every SIDE EFFECT that is not
    // idempotent (the near-clip queue atomic and the straddler fold) and every
    // store that three lanes would otherwise write three times.
    const bool lead = (MODE != 2) || (vk == 0);
    if (idx >= tri_count) return;
    const bool clipping = near_clip.ids && near_clip.count;
    const bool emit_rec = (emit.rec != nullptr);
    if (emit_rec && emit.skip_dead_core && emit.subs.nsub > 1 && emit.subs.dead_mask) {
        const int sb = hp_subdraw_of_setup(idx, emit.subs);
        if ((emit.subs.dead_mask >> sb) & 1) {
            if (lead) {
                BinRec r;
                r.xy  = (unsigned int)((unsigned int)(sb * emit.tiles_y) << 16);
                r.nxy = 1u;
                r.key = ((unsigned long long)0x80000000u << 32)
                      | (unsigned long long)(uint32_t)(clipping ? 2 * idx : idx);
                emit.rec[idx] = r;
            }
            return;
        }
    }
    const int output_index = idx;
    int32_t* tri_stats = state.tri_stats;
    if (tri_stats) tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_INPUT] = 1;

    TriangleSetup& tri = out[output_index];
    if (clipping && tri_stats && lead) {
        hipgl_mark_degenerate(out[tri_count + idx]);
        out[tri_count + idx].first_vertex = idx * 3;
    }

    float P[3][3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float* p = positions + (idx * 3 + k) * 3;
        P[k][0] = p[0]; P[k][1] = p[1]; P[k][2] = p[2];
    }

    int pose_gid = -1;
    if constexpr (POSE_FOLD) {
        // PER-TRIANGLE map: one CONTIGUOUS dword per lane. (A per-vertex map
        // indexed at idx*3 costs 12 cache lines per wave instead of one.)
        const int gid = pose.vert_geom[pose.tbase + idx];
        pose_gid = gid;
        if constexpr (POSE_MODE == 2) {
            const int sg = __builtin_amdgcn_readfirstlane(gid);
            if (__all(gid == sg)) {
                if (sg >= 0) {
                    const int p3 = pose.env * pose.ngeom * 3 + sg * 3;
                    const int q4 = pose.env * pose.ngeom * 4 + sg * 4;
                    hp_pose_apply3(P, pose.xquat[q4], pose.xquat[q4+1],
                                   pose.xquat[q4+2], pose.xquat[q4+3],
                                   pose.xpos[p3], pose.xpos[p3+1], pose.xpos[p3+2]);
                }
            } else if (gid >= 0) {
                const int p3 = pose.env * pose.ngeom * 3 + gid * 3;
                const int q4 = pose.env * pose.ngeom * 4 + gid * 4;
                hp_pose_apply3(P, pose.xquat[q4], pose.xquat[q4+1],
                               pose.xquat[q4+2], pose.xquat[q4+3],
                               pose.xpos[p3], pose.xpos[p3+1], pose.xpos[p3+2]);
            }
        } else if (gid >= 0) {
            const int p3 = pose.env * pose.ngeom * 3 + gid * 3;
            const int q4 = pose.env * pose.ngeom * 4 + gid * 4;
            hp_pose_apply3(P, pose.xquat[q4], pose.xquat[q4+1],
                           pose.xquat[q4+2], pose.xquat[q4+3],
                           pose.xpos[p3], pose.xpos[p3+1], pose.xpos[p3+2]);
        }
    }

    // Cull before the screen-coordinate prepass so its live clip/edge values
    // do not overlap the reflection-plane arithmetic. The host dispatches the
    // original specialization for ordinary draws and for the disabled ablation,
    // preserving their register allocation and floating-point code generation.
    if constexpr (REFLECTION_EARLY_CULL) {
        if (refl_active && reflection_triangle_outside_p(P, mvp, model_view, state.clip_plane)) {
            if (!emit.skip_dead_core) hipgl_mark_degenerate(tri);
            if (tri_stats) {
                tri.first_vertex = idx * 3;
                tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_CLIPPED] = 1;
            }
            if (emit_rec && lead) hp_emit_binrec(emit, idx, clipping, false, 0, 0, 0, 0, 0u);
            return;
        }
    }

    // Cull-only prepass: 12 VGPRs instead of 3 staged TransformedVertex (87).
    float sx[3], sy[3], zpw[3], sz[3], clip3[3][4];
#pragma unroll
    for (int k = 0; k < 3; k++)
        fused_vertex_cull_prepass_p(P[k][0], P[k][1], P[k][2], mvp,
                                    vp_x, vp_y, vp_w, vp_h, clip3[k],
                                    sx[k], sy[k], zpw[k], sz[k]);

    // first_vertex is DIAGNOSTIC ONLY: its single reader is the fb.attr_buffer
    // capture (bin_rasterize.hip:605/1093), which is null in production and is
    // covered by the same tri_stats gate as the slim payload below.
    if (tri_stats) tri.first_vertex = idx * 3;

    // CULL FIRST, SHADE AND STORE LAST.
    //
    // Both consumers of TriangleSetup::v[] -- the binner's z_min and the fine
    // rasterizer's attribute interpolation -- are reached only through a
    // `!tri.degenerate` guard (bin_rasterize.hip:26/503/1504/1967/2048/2150),
    // so a triangle that is near-plane clipped or sub-pixel culled has its 360 B
    // of vertex payload written and NEVER read, and its three inlined
    // compute_gouraud_lighting() calls thrown away. Making both conditional is
    // also what finally makes this kernel resolution-DEPENDENT: the cull rate
    // rises as the SSAA framebuffer shrinks.

    // Near-plane clipping
    if (zpw[0] < 0.0f || zpw[1] < 0.0f || zpw[2] < 0.0f) {
        if (!emit.skip_dead_core) hipgl_mark_degenerate(tri);
        // NOTE: near-plane clipping and the sub-pixel area cull below BOTH set
        // tri.degenerate, so the flag alone cannot say which fired. Counted
        // separately here.
        if (tri_stats) tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_CLIPPED] = 1;
        bool folded = false;
        if (clipping && lead && !(zpw[0] < 0.f && zpw[1] < 0.f && zpw[2] < 0.f)) {
            const int slot = atomicAdd(near_clip.count, 1);
            near_clip.ids[slot] = idx;
            if (hp_nc_queued) *hp_nc_queued = true;
            if constexpr (NC_FOLD) {
                float nq0 = 1.0f, nq1 = 0.0f, nq2 = 0.0f, nq3 = 0.0f;
                bool hp_have_gq = false;
                if constexpr (POSE_FOLD) {
                    if (pose_gid >= 0) {
                        const int q4 = pose.env * pose.ngeom * 4 + pose_gid * 4;
                        nq0 = pose.xquat[q4];
                        nq1 = pose.xquat[q4+1];
                        nq2 = pose.xquat[q4+2];
                        nq3 = pose.xquat[q4+3];
                        hp_have_gq = true;
                    }
                }
                if constexpr (NC_GSTAGE) {
                    near_clip_emit_source_triangle_g(positions, normals, texcoords,
                        texids, colors, spec_colors, per_vert_specular,
                        per_vert_shininess, per_vert_emission, mvp, model_view,
                        normal_mat, vp_x, vp_y, vp_w, vp_h, state,
                        idx, idx, tri_count + idx, out, tri_stats,
                        true,
                        P[0][0], P[0][1], P[0][2],
                        P[1][0], P[1][1], P[1][2],
                        P[2][0], P[2][1], P[2][2],
                        hp_have_gq, nq0, nq1, nq2, nq3);
                } else {
                    near_clip_emit_source_triangle(positions, normals, texcoords, texids,
                        colors, spec_colors, per_vert_specular, per_vert_shininess,
                        per_vert_emission, mvp, model_view, normal_mat,
                        vp_x, vp_y, vp_w, vp_h, state,
                        idx, idx, tri_count + idx, out, tri_stats,
                        true,
                        P[0][0], P[0][1], P[0][2],
                        P[1][0], P[1][1], P[1][2],
                        P[2][0], P[2][1], P[2][2],
                        hp_have_gq, nq0, nq1, nq2, nq3);
                }
                folded = true;
            }
        }
        if (emit_rec && lead) {
            if (folded) {
                const TriangleSetup& ct = out[idx];
                const bool live_c = (ct.degenerate == 0) &&
                    (((emit.cull_mask >> ((ct.backfacing != 0) ? 1 : 0)) & 1) == 0);
                hp_emit_binrec(emit, idx, clipping, live_c,
                               ct.bbox_min_x, ct.bbox_min_y,
                               ct.bbox_max_x, ct.bbox_max_y, ct.z_ord);
            } else {
                // tri.degenerate == 1 and nothing rewrote it, so the count pass
                // would have rejected on `degenerate` before reading bbox/z_ord.
                hp_emit_binrec(emit, idx, clipping, false, 0, 0, 0, 0, 0u);
            }
        }
        return;
    }

    constexpr float SUBPX = 256.0f;
    constexpr float INV_SUBPX = 1.0f / 256.0f;
    float x0 = roundf(sx[0] * SUBPX) * INV_SUBPX;
    float y0 = roundf(sy[0] * SUBPX) * INV_SUBPX;
    float x1 = roundf(sx[1] * SUBPX) * INV_SUBPX;
    float y1 = roundf(sy[1] * SUBPX) * INV_SUBPX;
    float x2 = roundf(sx[2] * SUBPX) * INV_SUBPX;
    float y2 = roundf(sy[2] * SUBPX) * INV_SUBPX;

    float e0x = x1 - x0, e0y = y1 - y0;
    float e1x = x2 - x1, e1y = y2 - y1;
    float e2x = x0 - x2, e2y = y0 - y2;

    float area = e0x * e2y - e0y * e2x;
    const bool is_degenerate = (fabsf(area) < d_min_tri_area);
    const bool bf_v = (area < 0.0f);

    float min_x = math::min3(x0, x1, x2);
    float min_y = math::min3(y0, y1, y2);
    float max_x = math::max3(x0, x1, x2);
    float max_y = math::max3(y0, y1, y2);
    const int bb_min_x = static_cast<int>(floorf(min_x));
    const int bb_min_y = static_cast<int>(floorf(min_y));
    const int bb_max_x = static_cast<int>(ceilf(max_x));
    const int bb_max_y = static_cast<int>(ceilf(max_y));
    const uint32_t z_ord_v = hipgl_f2ord_setup(fminf(sz[0], fminf(sz[1], sz[2])));

    if constexpr (REFL_RECT) {
        if (state.reflect_mask != nullptr && state.reflect_rect[2] >= 0) {
            if (bb_max_x < state.reflect_rect[0] || bb_min_x > state.reflect_rect[2] ||
                bb_max_y < state.reflect_rect[1] || bb_min_y > state.reflect_rect[3]) {
                if (!emit.skip_dead_core) hipgl_mark_degenerate(tri);
                if (tri_stats) {
                    tri.first_vertex = idx * 3;
                    tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_CLIPPED] = 1;
                }
                if (emit_rec && lead) hp_emit_binrec(emit, idx, clipping, false, 0, 0, 0, 0, 0u);
                return;
            }
        }
    }

    const bool fb_empty =
        (math::max2i(bb_min_x, 0) > math::min2i(bb_max_x, vp_w - 1) ||
         math::max2i(bb_min_y, 0) > math::min2i(bb_max_y, vp_h - 1));
    constexpr float SP_GUARD = 1.0f / 32.0f;
    const float kx0 = ceilf (min_x - (SP_GUARD + 0.5f));
    const float kx1 = floorf(max_x + (SP_GUARD - 0.5f));
    const float ky0 = ceilf (min_y - (SP_GUARD + 0.5f));
    const float ky1 = floorf(max_y + (SP_GUARD - 0.5f));
    bool no_sample = emit.center_samples_only && ((kx1 < kx0) || (ky1 < ky0));
    if (emit.center_samples_only && !no_sample && !fb_empty) {
        const float sgn = (area > 0.0f) ? -1.0f : 1.0f;
        const float bx0 = kx0 + 0.5f, by0 = ky0 + 0.5f;
        const float wdx = kx1 - kx0, wdy = ky1 - ky0;
        const float ea[3] = { sgn * (-e0y), sgn * (-e1y), sgn * (-e2y) };
        const float eb[3] = { sgn *   e0x,  sgn *   e1x,  sgn *   e2x  };
        const float px[3] = { x0, x1, x2 };
        const float py[3] = { y0, y1, y2 };
#pragma unroll
        for (int e = 0; e < 3; ++e) {
            const float base = ea[e] * (bx0 - px[e]) + eb[e] * (by0 - py[e]);
            const float hi   = base + fmaxf(ea[e], 0.0f) * wdx
                                    + fmaxf(eb[e], 0.0f) * wdy;
            const float mag  = fabsf(base) + fabsf(ea[e]) * wdx
                                           + fabsf(eb[e]) * wdy + 1.0f;
            if (hi < -(1e-5f * mag)) no_sample = true;
        }
    }

    if (emit_rec && lead) {
        const bool live = !is_degenerate && !no_sample &&
                          (((emit.cull_mask >> (bf_v ? 1 : 0)) & 1) == 0);
        hp_emit_binrec(emit, idx, clipping, live,
                       bb_min_x, bb_min_y, bb_max_x, bb_max_y, z_ord_v);
    }

    const bool dead_core =
        is_degenerate || no_sample ||
        (((bf_skip_mask >> (bf_v ? 1 : 0)) & 1) != 0) ||
        ((bf_skip_mask & 4) != 0 && fb_empty);
    if (dead_core && emit.skip_dead_core) return;

    if (lead) {
    hipgl_set_flags(tri, is_degenerate, bf_v);
    if (tri_stats) {
        if (is_degenerate)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_DEGENERATE] = 1;
        if (area < 0.0f)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_BACKFACE] = 1;
    }

    float sign = (area > 0.0f) ? -1.0f : 1.0f;

    float A0s = sign * (-e0y), B0s = sign * e0x;
    float A1s = sign * (-e1y), B1s = sign * e1x;
    float A2s = sign * (-e2y), B2s = sign * e2x;

    tri.edge_eq[0][0] = A0s;
    tri.edge_eq[0][1] = B0s;
    tri.edge_eq[0][2] = (float)(-((double)A0s * (double)x0 + (double)B0s * (double)y0));
    tri.edge_eq[1][0] = A1s;
    tri.edge_eq[1][1] = B1s;
    tri.edge_eq[1][2] = (float)(-((double)A1s * (double)x1 + (double)B1s * (double)y1));
    tri.edge_eq[2][0] = A2s;
    tri.edge_eq[2][1] = B2s;
    tri.edge_eq[2][2] = (float)(-((double)A2s * (double)x2 + (double)B2s * (double)y2));

    // Coordinates have 8 fractional bits; the edge-function quantum
    // is their product, 1/65536. A 1/256 bias erodes small triangles.
    constexpr float FILL_BIAS = -1.0f / 65536.0f;
    tri.edge_bias[0] = (A0s > 0.0f || (A0s == 0.0f && B0s > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[1] = (A1s > 0.0f || (A1s == 0.0f && B1s > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[2] = (A2s > 0.0f || (A2s == 0.0f && B2s > 0.0f)) ? 0.0f : FILL_BIAS;

    hipgl_set_bbox(tri, bb_min_x, bb_min_y, bb_max_x, bb_max_y);
    tri.z_ord = z_ord_v;
    }   // lead

    // Sub-pixel-culled triangles are dropped by every reader of v[]; nothing
    // below is observable for them.
    if (is_degenerate) return;

    if (((bf_skip_mask >> (bf_v ? 1 : 0)) & 1) != 0) return;

    // Bit 2: the SECOND reject the count pass applies -- bbox clamped to the
    // framebuffer is empty. Same conditions, same free ride: the four bbox
    // ints are already in registers. fb_width/fb_height at the count pass are
    // the same vw/vh this kernel receives as vp_w/vp_h.
    if ((bf_skip_mask & 4) != 0 && fb_empty) return;

    // ONE vertex live at a time. Staging is still required -- handing
    // fused_transform_vertex a reference INTO the 436 B-strided global buffer
    // makes each of its ~30 field writes a standalone narrow global_store in
    // non-address order -- but staging ONE TransformedVertex costs 29 VGPRs,
    // not 87, and the bulk copy still goes out as wide in-order stores.
    const bool do_light = (state.gouraud_shading && state.lighting_enabled);
    // See the HOT-64 note at the top of this file. `slim` is wave-uniform.
    const bool slim = d_geom_slim && do_light && !state.clip_plane_enabled
                      && (tri_stats == nullptr);
    if constexpr (MODE == 2) {
        // Lane vk owns vertex vk. The prepass above already produced all three
        // vertices' clip products in registers; SELECT this lane's four floats
        // (and its three object-space coordinates) with v_cndmask rather than
        // indexing the private arrays, which would demote them to scratch.
        float cin[4], pin[3];
#pragma unroll
        for (int c = 0; c < 4; ++c)
            cin[c] = (vk == 0) ? clip3[0][c] : ((vk == 1) ? clip3[1][c] : clip3[2][c]);
#pragma unroll
        for (int c = 0; c < 3; ++c)
            pin[c] = (vk == 0) ? P[0][c] : ((vk == 1) ? P[1][c] : P[2][c]);
        TransformedVertex tvk;
        float nwb2[3] = {0.0f, 0.0f, 0.0f};
        bool have_n2 = false;
        if constexpr (POSE_FOLD) {
            if (pose_gid >= 0 && normals) {
                const int q4 = pose.env * pose.ngeom * 4 + pose_gid * 4;
                const float gqn[4] = { pose.xquat[q4], pose.xquat[q4+1],
                                       pose.xquat[q4+2], pose.xquat[q4+3] };
                const float* nm = normals + (idx * 3 + vk) * 3;
                const float ln[3] = { nm[0], nm[1], nm[2] };
                hp_qrot(gqn, ln, nwb2);
                have_n2 = true;
            }
        }
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors, spec_colors,
                               per_vert_specular, per_vert_shininess, per_vert_emission,
                               idx * 3 + vk,
                               mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tvk,
                               cin, pin, have_n2, nwb2[0], nwb2[1], nwb2[2]);
        if (do_light) compute_gouraud_lighting(tvk, state);
        TransformedVertex* __restrict__ dst = &tri.v[vk];
        if (slim) store_vertex_hot64(*dst, tvk);
        else      *dst = tvk;
        return;
    }
    if constexpr (MODE == 1) {
        TransformedVertex tv0, tv1, tv2;
    float nwbuf[3] = {0.0f, 0.0f, 0.0f};
    auto hp_fold_n = [&](int k) -> bool {
        if constexpr (POSE_FOLD) {
            if (pose_gid >= 0 && normals) {
                const int q4 = pose.env * pose.ngeom * 4 + pose_gid * 4;
                const float gqn[4] = { pose.xquat[q4], pose.xquat[q4+1],
                                       pose.xquat[q4+2], pose.xquat[q4+3] };
                const float* nm = normals + (idx * 3 + k) * 3;
                const float ln[3] = { nm[0], nm[1], nm[2] };
                hp_qrot(gqn, ln, nwbuf);
                return true;
            }
        }
        return false;
    };
        const bool hn0 = hp_fold_n(0);
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors, spec_colors,
                               per_vert_specular, per_vert_shininess, per_vert_emission,
                               idx * 3 + 0,
                               mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tv0,
                               clip3[0], P[0], hn0, nwbuf[0], nwbuf[1], nwbuf[2]);
        const bool hn1 = hp_fold_n(1);
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors, spec_colors,
                               per_vert_specular, per_vert_shininess, per_vert_emission,
                               idx * 3 + 1,
                               mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tv1,
                               clip3[1], P[1], hn1, nwbuf[0], nwbuf[1], nwbuf[2]);
        const bool hn2m = hp_fold_n(2);
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors, spec_colors,
                               per_vert_specular, per_vert_shininess, per_vert_emission,
                               idx * 3 + 2,
                               mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tv2,
                               clip3[2], P[2], hn2m, nwbuf[0], nwbuf[1], nwbuf[2]);
        if (do_light) compute_gouraud_lighting3(tv0, tv1, tv2, state);
        if (slim) {
            store_vertex_hot64(tri.v[0], tv0);
            store_vertex_hot64(tri.v[1], tv1);
            store_vertex_hot64(tri.v[2], tv2);
        } else {
            tri.v[0] = tv0;
            tri.v[1] = tv1;
            tri.v[2] = tv2;
        }
        return;
    }
#pragma unroll
    for (int k = 0; k < 3; k++) {
        TransformedVertex tvk;
        float nwb[3] = {0.0f, 0.0f, 0.0f};
        bool have_n = false;
        if constexpr (POSE_FOLD) {
            if (pose_gid >= 0 && normals) {
                const int q4 = pose.env * pose.ngeom * 4 + pose_gid * 4;
                const float gqn[4] = { pose.xquat[q4], pose.xquat[q4+1],
                                       pose.xquat[q4+2], pose.xquat[q4+3] };
                const float* nm = normals + (idx * 3 + k) * 3;
                const float ln[3] = { nm[0], nm[1], nm[2] };
                hp_qrot(gqn, ln, nwb);
                have_n = true;
            }
        }
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors, spec_colors,
                               per_vert_specular, per_vert_shininess, per_vert_emission,
                               idx * 3 + k,
                               mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, tvk,
                               clip3[k], P[k], have_n, nwb[0], nwb[1], nwb[2]);
        if (do_light) compute_gouraud_lighting(tvk, state);
        if (slim) store_vertex_hot64(tri.v[k], tvk);
        else      tri.v[k] = tvk;
    }
}

#ifndef HP_SAT_WPEU
#define HP_SAT_WPEU 4
#endif
// HP_MICRO_WPEU does the same for the MICRO instantiations (LB == 64): the
// 3-lanes-per-triangle mirror passes run NINE waves total, so they cannot use
// occupancy at all and a relaxed budget is pure MLP. 0/0 = the untouched
// allocation both arms shipped with.
#ifndef HP_MICRO_WPEU
#define HP_MICRO_WPEU 0
#endif
#if HP_SAT_WPEU == 0 && HP_MICRO_WPEU == 0
#define HP_SAT_LBQ(L) __launch_bounds__(L)
#else
#define HP_SAT_LBQ(L) __launch_bounds__(L, \
    ((L) == 256 ? (HP_SAT_WPEU ? HP_SAT_WPEU : 8) \
                : (HP_MICRO_WPEU ? HP_MICRO_WPEU : 6)))
#endif
template <bool REFLECTION_EARLY_CULL, bool NC_FOLD, int LB = 256, int MODE = 0,
          bool POSE_FOLD = false>
HIPGL_GLOBAL HP_SAT_LBQ(LB) void kernel_fused_vertex_tri_gouraud(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int tri_count,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    RenderState state,
    TriangleSetup* __restrict__ out, NearClipQueue near_clip,
    int bf_skip_mask, BinEmit emit, VtxPose pose)
{
    hp_fused_vtx_body<REFLECTION_EARLY_CULL, REFLECTION_EARLY_CULL, NC_FOLD, MODE, POSE_FOLD,
                      1, NC_FOLD>(
        positions, normals, texcoords, texids, colors, spec_colors,
        per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        state, out, near_clip, bf_skip_mask, emit, true, pose);
}

#ifndef HP_Z2_WPEU
#define HP_Z2_WPEU 0
#endif
#if HP_Z2_WPEU == 0
#define HP_Z2_LBQ(L) __launch_bounds__(L)
#else
#define HP_Z2_LBQ(L) __launch_bounds__(L, HP_Z2_WPEU)
#endif
template <bool NC_FOLD, bool POSE_FOLD = false, int POSE_MODE = 1, int LB = 256>
HIPGL_GLOBAL HP_Z2_LBQ(LB) void kernel_fused_vertex_tri_gouraud_z2(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int tri_count,
    int vp_x, int vp_y, int vp_w, int vp_h,
    VtxZSlot s0, VtxZSlot s1, VtxPose pose)
{
#ifndef HP_Z2_SLOT1_ROT
#define HP_Z2_SLOT1_ROT 8
#endif
    const unsigned hp_slot = blockIdx.z;
#if HP_Z2_SLOT1_ROT == 0
    const unsigned hp_blk  = blockIdx.x;
#else
    const unsigned hp_rot_ = blockIdx.x + (unsigned)HP_Z2_SLOT1_ROT;
    const unsigned hp_blk  = hp_slot ? (hp_rot_ >= gridDim.x ? hp_rot_ - gridDim.x
                                                             : hp_rot_)
                                     : blockIdx.x;
#endif
    if (hp_slot == 0) {
        hp_fused_vtx_body<true, true, NC_FOLD, 0, POSE_FOLD, POSE_MODE, true>(
            positions, normals, texcoords, texids, colors, spec_colors,
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
            s0.mvp, s0.model_view, s0.normal_mat, vp_x, vp_y, vp_w, vp_h,
            s0.state, s0.out, s0.near_clip, s0.bf_skip_mask, s0.emit, true, pose,
            hp_blk, true);
    } else {
        hp_fused_vtx_body<false, false, NC_FOLD, 0, POSE_FOLD, POSE_MODE, true>(
            positions, normals, texcoords, texids, colors, spec_colors,
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
            s1.mvp, s1.model_view, s1.normal_mat, vp_x, vp_y, vp_w, vp_h,
            s1.state, s1.out, s1.near_clip, s1.bf_skip_mask, s1.emit, true, pose,
            hp_blk, true);
    }
}

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats, NearClipQueue near_clip)
{
    launch_fused_vertex_tri(positions, normals, texcoords, texids, colors,
        nullptr, nullptr, nullptr, nullptr, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        out, stream, stats, near_clip);
}

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats, NearClipQueue near_clip)
{
    launch_fused_vertex_tri(positions, normals, texcoords, texids, colors,
        spec_colors, nullptr, nullptr, nullptr, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        out, stream, stats, near_clip);
}

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats, NearClipQueue near_clip)
{
    launch_fused_vertex_tri(positions, normals, texcoords, texids, colors,
        spec_colors, per_vert_specular, per_vert_shininess, nullptr, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        out, stream, stats, near_clip);
}

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats, NearClipQueue near_clip)
{
    if (!positions || !out || tri_count <= 0) return;
    static const int plain_small_block = [] {
        const char* e = getenv("HP_VTX_PLAIN_BLOCK");
        return e ? atoi(e) : 64;
    }();
    static const int plain_small_tris = [] {
        const char* e = getenv("HP_VTX_PLAIN_TRIS");
        return e ? atoi(e) : 4096;
    }();
    int block = 256;
    if (plain_small_block >= 16 && plain_small_block < block &&
        tri_count <= plain_small_tris) block = plain_small_block;
    const int grid = (tri_count + block - 1) / block;
    static const int plain_nc_fold = [] {
        const char* e = getenv("HP_PLAIN_NC_FOLD");
        return e ? atoi(e) : 1;
    }();
    const RenderState nc_state{};
    const bool do_fold = plain_nc_fold && near_clip.ids && near_clip.count;
#define HP_PLAIN_ARGS positions, normals, texcoords, texids, colors, spec_colors, \
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count, \
            mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, out, stats, \
            near_clip, nc_state
    if (do_fold) {
        if (block == 64)
            hipLaunchKernelGGL((kernel_fused_vertex_tri<true, 64>), grid, block, 0, stream, HP_PLAIN_ARGS);
        else if (block == 128)
            hipLaunchKernelGGL((kernel_fused_vertex_tri<true, 128>), grid, block, 0, stream, HP_PLAIN_ARGS);
        else
            hipLaunchKernelGGL((kernel_fused_vertex_tri<true, 256>), grid, block, 0, stream, HP_PLAIN_ARGS);
        return;
    }
    hipLaunchKernelGGL((kernel_fused_vertex_tri<false, 256>), grid, block, 0, stream,
        positions, normals, texcoords, texids, colors, spec_colors,
        per_vert_specular, per_vert_shininess, per_vert_emission,
        tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, out, stats,
        near_clip, nc_state);
    launch_fused_near_clip(positions, normals, texcoords, texids, colors,
        spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
        tri_count, mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        RenderState{}, out, stats, stream, near_clip);
}

template <int NT>
HIPGL_DEVICE __forceinline__ void hpvc_block_scan(int* s_off, int tid, int n)
{
    if (NT <= 64) {
        if (tid < 64) {
            int v = (tid < NT) ? n : 0;
#pragma unroll
            for (int d = 1; d < NT; d <<= 1) {
                int t = __shfl_up(v, d, 64);
                if (tid >= d) v += t;
            }
            if (tid == 0) s_off[0] = 0;
            if (tid < NT) s_off[tid + 1] = v;
        }
        __syncthreads();
    } else {
        constexpr int NW = (NT > 64) ? (NT / 64) : 1;
        __shared__ int s_wtot[NW];
        const int lane = tid & 63;
        const int wid  = tid >> 6;
        int v = (tid < NT) ? n : 0;
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            const int y = __shfl_up(v, off, 64);
            if (lane >= off) v += y;
        }
        if (lane == 63 && wid < NW) s_wtot[wid] = v;
        __syncthreads();
        int carry = 0;
#pragma unroll
        for (int w = 0; w < NW; ++w) carry += (w < wid) ? s_wtot[w] : 0;
        if (tid == 0) s_off[0] = 0;
        if (tid < NT) s_off[tid + 1] = carry + v;
        __syncthreads();
    }
}

HIPGL_DEVICE __forceinline__ void hpvc_agg_count(int* __restrict__ counts, int t, bool ok)
{
    unsigned long long active = __ballot(ok ? 1 : 0);
    const int lane = (int)(threadIdx.x & 63u);
    while (active) {
        const int leader = __builtin_ctzll(active);
        const int lt = __shfl(t, leader, 64);
        const unsigned long long same = active & __ballot(ok && t == lt);
        if (lane == leader) atomicAdd(&counts[lt], (int)__popcll(same));
        active &= ~same;
    }
}

template <int NT>
HIPGL_DEVICE __forceinline__ int hpvc_find_owner(const int* s_off, int p)
{
    int base = 0;
#pragma unroll
    for (int n = NT; n > 1; n >>= 1) {
        const int half = n >> 1;
        base = (s_off[base + half] <= p) ? (base + half) : base;
    }
    return base;
}

HIPGL_DEVICE __forceinline__ void hpvc_rect_clip_tiles(
    int& lo_x, int& lo_y, int& nx, int& ny, int tiles_y, const TileRect& rect)
{
    if (ny <= 0) return;
    int sub = 0;
    if (rect.nsub > 1) {
        if (rect.nsub <= 8) {
#pragma unroll
            for (int k = 1; k < 8; ++k) sub += (lo_y >= k * tiles_y) ? tiles_y : 0;
        } else {
#pragma unroll
            for (int k = 1; k < HP_MAX_BLOCKS; ++k)
                sub += (lo_y >= k * tiles_y) ? tiles_y : 0;
        }
    }
    const int ly  = lo_y - sub;
    const int nlx = max(lo_x, rect.x0);
    const int nhx = min(lo_x + nx - 1, rect.x1);
    const int nly = max(ly, rect.y0);
    const int nhy = min(ly + ny - 1, rect.y1);
    if (nlx > nhx || nly > nhy) { lo_x = 0; lo_y = sub; nx = 1; ny = 0; return; }
    lo_x = nlx; nx = nhx - nlx + 1;
    lo_y = nly + sub; ny = nhy - nly + 1;
}

HIPGL_DEVICE __forceinline__ BinRec hpvc_near_clip_binrec(const TriangleSetup& tri,
    int order_id, int tiles_x, int tiles_y, int fb_width, int fb_height)
{
    int lo_x = 0, lo_y = 0, nx = 1, ny = 0;
    uint32_t z_ord = 0x80000000u;
    if (!tri.degenerate) {
        const int tmin_x = max(tri.bbox_min_x, 0);
        const int tmin_y = max(tri.bbox_min_y, 0);
        const int tmax_x = min(tri.bbox_max_x, fb_width - 1);
        const int tmax_y = min(tri.bbox_max_y, fb_height - 1);
        if (tmin_x <= tmax_x && tmin_y <= tmax_y) {
            lo_x = max(tmin_x / TILE_SIZE, 0);
            lo_y = max(tmin_y / TILE_SIZE, 0);
            const int hi_x = min(tmax_x / TILE_SIZE, tiles_x - 1);
            const int hi_y = min(tmax_y / TILE_SIZE, tiles_y - 1);
            if (lo_x <= hi_x && lo_y <= hi_y) {
                nx = hi_x - lo_x + 1;
                ny = hi_y - lo_y + 1;
                z_ord = tri.z_ord;
            }
        }
    }
    BinRec r;
    r.xy = (unsigned)lo_x | ((unsigned)lo_y << 16);
    r.nxy = (unsigned)nx | ((unsigned)ny << 16);
    r.key = ((unsigned long long)z_ord << 32) | (uint32_t)order_id;
    return r;
}

#define HPVC_ILP 4
// NT == blockDim.x == the vertex CTA; lane tid owns triangle blockIdx.x*NT+tid,
// exactly the triangle hp_fused_vtx_body (MODE 0) gave it.
template <int NT>
HIPGL_DEVICE __forceinline__ void hpvc_count_tail(
    const BinEmit& emit, int tri_count, const TriangleSetup* __restrict__ tris,
    int* __restrict__ tile_counts, const TileRect& rect, bool queued,
    int hp_bx_ov = -1)
{
    __shared__ int   s_off[NT + 1];
    __shared__ int   s_lox[NT];
    __shared__ int   s_loy[NT];
    __shared__ int   s_nx [NT];
    __shared__ float s_inv[NT];
    __shared__ int   s_qid[NT];
    __shared__ int   s_nq;
    __shared__ BinRec s_rec;
    constexpr bool AGG = (NT >= 64);
    const int tid = threadIdx.x;
    const int idx = (hp_bx_ov >= 0 ? hp_bx_ov : (int)blockIdx.x) * NT + tid;
    const int tiles_x = emit.tiles_x, tiles_y = emit.tiles_y;
    if (tid == 0) s_nq = 0;
    int n = 0, lo_x = 0, lo_y = 0, nx = 1, ny = 0;
    if (idx < tri_count) {
        const BinRec r = emit.rec[idx];
        lo_x = (int)(r.xy & 0xFFFFu);
        lo_y = (int)(r.xy >> 16);
        nx   = (int)(r.nxy & 0xFFFFu);
        ny   = (int)(r.nxy >> 16);
        if (rect.x1 >= 0) hpvc_rect_clip_tiles(lo_x, lo_y, nx, ny, tiles_y, rect);
        n    = nx * ny;
    }
    s_lox[tid] = lo_x;
    s_loy[tid] = lo_y;
    s_nx [tid] = nx;
    s_inv[tid] = 1.0f / (float)nx;
    hpvc_block_scan<NT>(s_off, tid, n);     // barrier: s_nq = 0 now visible
    if (queued) {
        const int q = atomicAdd(&s_nq, 1);
        s_qid[q] = idx;
    }
    const int total = s_off[NT];
    const int CSTRIDE = NT * HPVC_ILP;
    int p_base = tid;
    for (; p_base + CSTRIDE - NT < total; p_base += CSTRIDE) {
        int t_[HPVC_ILP];
#pragma unroll
        for (int u = 0; u < HPVC_ILP; ++u) {
            const int p = p_base + u * NT;
            int j = hpvc_find_owner<NT>(s_off, p);
            int local = p - s_off[j];
            int jnx = s_nx[j];
            int q = (int)((float)local * s_inv[j]);
            int r = local - q * jnx;
            if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
            t_[u] = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        }
#pragma unroll
        for (int u = 0; u < HPVC_ILP; ++u) {
            if constexpr (AGG) hpvc_agg_count(tile_counts, t_[u], true);
            else               atomicAdd(&tile_counts[t_[u]], 1);
        }
    }
    for (int p = p_base; p < total; p += NT) {
        int j = hpvc_find_owner<NT>(s_off, p);
        int local = p - s_off[j];
        int jnx = s_nx[j];
        int q = (int)((float)local * s_inv[j]);
        int r = local - q * jnx;
        if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
        int t = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        if constexpr (AGG) hpvc_agg_count(tile_counts, t, true);
        else               atomicAdd(&tile_counts[t], 1);
    }
    // Near-plane extras queued by THIS block (the count epilogue's body).
    __syncthreads();
    const int nq = s_nq;
    const BinSubdraws& subs = emit.subs;
    for (int qi = 0; qi < nq; ++qi) {
        const int source = s_qid[qi];
        __syncthreads();
        if (tid == 0) {
            s_rec = hpvc_near_clip_binrec(tris[tri_count + source],
                2 * source + 1, tiles_x, tiles_y, emit.fb_w, emit.fb_h);
            if (subs.nsub > 1) {
                const int nsb = hp_subdraw_of_setup(source, subs);
                if ((subs.dead_mask >> nsb) & 1) s_rec.nxy = 1u;
                s_rec.xy += (unsigned int)(nsb * tiles_y) << 16;
            }
            if (rect.x1 >= 0) {
                int rlx = (int)(s_rec.xy & 0xFFFFu), rly = (int)(s_rec.xy >> 16);
                int rnx = (int)(s_rec.nxy & 0xFFFFu), rny = (int)(s_rec.nxy >> 16);
                hpvc_rect_clip_tiles(rlx, rly, rnx, rny, tiles_y, rect);
                s_rec.xy  = (unsigned int)rlx | ((unsigned int)rly << 16);
                s_rec.nxy = (unsigned int)rnx | ((unsigned int)rny << 16);
            }
            emit.rec[tri_count + source] = s_rec;
        }
        __syncthreads();
        const int qlx = s_rec.xy & 0xFFFFu;
        const int qly = s_rec.xy >> 16;
        const int qnx = s_rec.nxy & 0xFFFFu;
        const int qny = s_rec.nxy >> 16;
        for (int p = tid; p < qnx * qny; p += NT)
            atomicAdd(&tile_counts[(qly + p / qnx) * tiles_x + qlx + p % qnx], 1);
    }
}

template <bool REFLECTION_EARLY_CULL, int LB = 256, bool POSE_FOLD = false>
HIPGL_GLOBAL HP_SAT_LBQ(LB) void kernel_fused_vertex_tri_gouraud_cnt(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const int* __restrict__ texids,
    const float* __restrict__ colors,
    const float* __restrict__ spec_colors,
    const float* __restrict__ per_vert_specular,
    const float* __restrict__ per_vert_shininess,
    const float* __restrict__ per_vert_emission,
    int tri_count,
    const float* __restrict__ mvp,
    const float* __restrict__ model_view,
    const float* __restrict__ normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    RenderState state,
    TriangleSetup* __restrict__ out, NearClipQueue near_clip,
    int bf_skip_mask, BinEmit emit, VtxPose pose,
    int* __restrict__ tile_counts, TileRect rect)
{
    bool queued = false;
    // Same instantiation of the body as kernel_fused_vertex_tri_gouraud
    // <REFLECTION_EARLY_CULL, true, LB, 0, POSE_FOLD>.
    hp_fused_vtx_body<REFLECTION_EARLY_CULL, REFLECTION_EARLY_CULL, true, 0, POSE_FOLD,
                      1, true>(
        positions, normals, texcoords, texids, colors, spec_colors,
        per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        state, out, near_clip, bf_skip_mask, emit, true, pose, 0u, false, &queued);
    hpvc_count_tail<LB>(emit, tri_count, out, tile_counts, rect, queued);
}

template <bool REFLECTION_EARLY_CULL, int LB = 256, bool POSE_FOLD = false>
HIPGL_GLOBAL HP_SAT_LBQ(LB) void kernel_fused_vertex_tri_gouraud_cnt_batch(
    CqVtxBatch b, RenderState state)
{
    int c = 0;
    for (int i = 1; i < b.n; ++i) c = ((int)blockIdx.x >= b.d[i].bstart) ? i : c;
    const CqVtxDesc& d = b.d[c];
    const int lbx = (int)blockIdx.x - d.bstart;
    bool queued = false;
    hp_fused_vtx_body<REFLECTION_EARLY_CULL, REFLECTION_EARLY_CULL, true, 0, POSE_FOLD,
                      1, true>(
        d.positions, d.normals, d.texcoords, d.texids, d.colors, d.spec_colors,
        d.per_vert_specular, d.per_vert_shininess, d.per_vert_emission, d.tri_count,
        d.mvp, d.model_view, d.normal_mat, d.vp_x, d.vp_y, d.vp_w, d.vp_h,
        state, d.out, d.near_clip, d.bf_skip_mask, d.emit, true, d.pose,
        (unsigned)lbx, true, &queued);
    hpvc_count_tail<LB>(d.emit, d.tri_count, d.out, d.tile_counts, d.rect, queued, lbx);
}

CqChainRec*& cq_rec_slot()
{
    static thread_local CqChainRec* p = nullptr;
    return p;
}

template <bool R, int L, bool P>
static void hp_cq_vtx_go(const CqVtxRec* const* r, int n, hipStream_t stream)
{
    if (n == 1) {
        const CqVtxDesc& d = r[0]->d;
        hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_cnt<R, L, P>), d.grid, L, 0, stream,
            d.positions, d.normals, d.texcoords, d.texids, d.colors, d.spec_colors,
            d.per_vert_specular, d.per_vert_shininess, d.per_vert_emission, d.tri_count,
            d.mvp, d.model_view, d.normal_mat, d.vp_x, d.vp_y, d.vp_w, d.vp_h,
            r[0]->state, d.out, d.near_clip, d.bf_skip_mask, d.emit, d.pose,
            d.tile_counts, d.rect);
        return;
    }
    CqVtxBatch b;
    memset(&b, 0, sizeof(b));
    int tot = 0;
    for (int i = 0; i < n; ++i) {
        b.d[i] = r[i]->d;
        b.d[i].bstart = tot;
        tot += b.d[i].grid;
    }
    b.n = n;
    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_cnt_batch<R, L, P>), tot, L, 0, stream,
                       b, r[0]->state);
}

static void hp_cq_vtx_dispatch(const CqVtxRec* const* r, int n, hipStream_t stream)
{
    const int refl = r[0]->refl, lb = r[0]->lb, pf = r[0]->pf;
#define HP_CQV(R, L, P) if (refl == (int)R && lb == L && pf == (int)P) { hp_cq_vtx_go<R, L, P>(r, n, stream); return; }
    HP_CQV(true, 64, true)  HP_CQV(true, 128, true)  HP_CQV(true, 256, true)  HP_CQV(true, 512, true)
    HP_CQV(false, 64, true) HP_CQV(false, 128, true) HP_CQV(false, 256, true) HP_CQV(false, 512, true)
    HP_CQV(true, 256, false) HP_CQV(false, 256, false)
#undef HP_CQV
}

static size_t hp_rs_cmp_bytes()
{
    return offsetof(RenderState, reflect_rect) + sizeof(int32_t) * 4;
}

void launch_cq_vtx(const CqVtxRec* const* r, int n, hipStream_t stream)
{
    // Group by (instantiation, RenderState modulo the per-draw source count).
    bool done[HP_CQ_MAX] = {};
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        const CqVtxRec* grp[HP_CQ_MAX];
        int g = 0;
        RenderState si = r[i]->state;
        si.near_clip_source_count = 0; si.tri_stats = nullptr;
        for (int j = i; j < n; ++j) {
            if (done[j]) continue;
            const CqVtxRec* q = r[j];
            if (q->refl != r[i]->refl || q->lb != r[i]->lb || q->pf != r[i]->pf) continue;
            RenderState sj = q->state;
            sj.near_clip_source_count = 0; sj.tri_stats = nullptr;
            if (memcmp(&si, &sj, hp_rs_cmp_bytes()) != 0) continue;
            done[j] = true;
            grp[g++] = q;
        }
        hp_cq_vtx_dispatch(grp, g, stream);
    }
}

bool launch_fused_vertex_tri_gouraud_ex(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state,
    TriangleSetup* out, hipStream_t stream, NearClipQueue near_clip, BinEmit emit_in,
    const VtxCntFuse* cf)
{
    // No stats parameter: this variant reads state.tri_stats, so the pointer
    // cannot get out of sync with the RenderState the kernel actually sees.
    if (!positions || !out || tri_count <= 0) return false;
    BinEmit emit = emit_in;
    static const int slim_env = [] {
        int v = hipgl_geom_slim_env();
        hipMemcpyToSymbol(HIP_SYMBOL(d_geom_slim), &v, sizeof(int));
        return v;
    }();
    (void)slim_env;
    // Host dispatch instead of a device constant: each device/worker sees the
    // same immutable ablation choice without another symbol upload. Only a
    // reflection draw compiles/executes the additional plane-cull arithmetic.
    static const bool reflection_early_cull = [] {
        const char* env = getenv("HIPGL_REFLECTION_EARLY_CULL");
        return !env || atoi(env) != 0;
    }();
    static const bool det_bin = [] {
        const char* e = getenv("HIPGL_DET_BIN");
        return e ? (e[0] == '1') : true;
    }();
    static const int bf_skip_env = [] {
        const char* e = getenv("HP_BF_SKIP");
        return e ? atoi(e) : 1;
    }();
    const int bf_skip_mask = (bf_skip_env && det_bin && state.tri_stats == nullptr)
                           ? (backface_cull_mask(state) | 4) : 0;
    // The core-line deletion reuses bf_skip_mask's own preconditions (the
    // deterministic binning chain and tri_stats == nullptr), so it can never be
    // on where that reject is off.
    if (bf_skip_mask == 0 || emit.rec == nullptr) emit.skip_dead_core = false;
    static const int small_block = [] {
        const char* e = getenv("HP_VTX_SMALL_BLOCK");
        return e ? atoi(e) : 64;
    }();
    static const int small_tris = [] {
        const char* e = getenv("HP_VTX_SMALL_TRIS");
        return e ? atoi(e) : 4096;
    }();
    static const int sat_block_env = [] {
        const char* e = getenv("HP_VTX_SAT_BLOCK");
        int v = e ? atoi(e) : 256;
        return (v == 64 || v == 128 || v == 256 || v == 512) ? v : 256;
    }();
    int block = sat_block_env;
    if (small_block >= 16 && small_block < block && tri_count <= small_tris)
        block = small_block;
    const int grid = (tri_count + block - 1) / block;
    const int nc_fold = hp_nc_fold_flag();
    const bool refl = reflection_early_cull && state.clip_plane_enabled;
    static const int coop_tris = [] {
        const char* e = getenv("HP_VTX_COOP_TRIS");
        return e ? atoi(e) : 1024;
    }();
    static const int coop_mode = [] {
        const char* e = getenv("HP_VTX_COOP");
        return e ? atoi(e) : 2;
    }();
    const VtxPose pose = hp_pose_for_draw(positions);
    const bool pf = (pose.vert_geom != nullptr);
#define HP_VTX_ARGS positions, normals, texcoords, texids, colors, spec_colors, \
        per_vert_specular, per_vert_shininess, per_vert_emission, tri_count, \
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, state, out, near_clip, \
        bf_skip_mask, emit, pose
    const bool micro = (block <= 64) && (tri_count <= coop_tris) && !refl;
    if (nc_fold) {
        if (micro && coop_mode == 2) {
            // Three lanes per triangle: 3x the threads, same 64-thread CTA, so
            // 3x the resident waves spread over 3x the CUs.
            const int cgrid = (3 * tri_count + block - 1) / block;
            if (pf)
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 64, 2, true>),
                                   cgrid, block, 0, stream, HP_VTX_ARGS);
            else
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 64, 2>),
                                   cgrid, block, 0, stream, HP_VTX_ARGS);
            return false;
        }
        if (micro && coop_mode == 1) {
            if (pf)
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 64, 1, true>),
                                   grid, block, 0, stream, HP_VTX_ARGS);
            else
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 64, 1>),
                                   grid, block, 0, stream, HP_VTX_ARGS);
            return false;
        }
        if (cf && cf->tile_counts && emit.rec &&
            (pf ? (block == 64 || block == 128 || block == 256 || block == 512)
                : (block == 256))) {
            if (CqChainRec* cq = cq_rec_slot()) {
                if (cq->want & 1) {
                    CqVtxDesc& d = cq->v.d;
                    memset(&d, 0, sizeof(d));
                    d.positions = positions; d.normals = normals; d.texcoords = texcoords;
                    d.texids = texids; d.colors = colors; d.spec_colors = spec_colors;
                    d.per_vert_specular = per_vert_specular;
                    d.per_vert_shininess = per_vert_shininess;
                    d.per_vert_emission = per_vert_emission;
                    d.mvp = mvp; d.model_view = model_view; d.normal_mat = normal_mat;
                    d.out = out; d.near_clip = near_clip; d.tile_counts = cf->tile_counts;
                    d.emit = emit; d.pose = pose; d.rect = cf->rect;
                    d.tri_count = tri_count; d.bf_skip_mask = bf_skip_mask;
                    d.vp_x = vp_x; d.vp_y = vp_y; d.vp_w = vp_w; d.vp_h = vp_h;
                    d.grid = grid; d.bstart = 0;
                    cq->v.state = state;
                    cq->v.refl = refl ? 1 : 0;
                    cq->v.lb = pf ? block : 256;
                    cq->v.pf = pf ? 1 : 0;
                    cq->got |= 1;
                    return true;
                }
            }
#define HP_VC_ARGS HP_VTX_ARGS, cf->tile_counts, cf->rect
#define HP_VC_GO(R, L, P) hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_cnt<R, L, P>), \
                                             grid, block, 0, stream, HP_VC_ARGS)
            if (refl) {
                if (pf) {
                    if (block == 128)      HP_VC_GO(true, 128, true);
                    else if (block == 512) HP_VC_GO(true, 512, true);
                    else if (block == 64)  HP_VC_GO(true, 64, true);
                    else                   HP_VC_GO(true, 256, true);
                } else {
                    HP_VC_GO(true, 256, false);
                }
            } else {
                if (pf) {
                    if (block == 128)      HP_VC_GO(false, 128, true);
                    else if (block == 512) HP_VC_GO(false, 512, true);
                    else if (block == 64)  HP_VC_GO(false, 64, true);
                    else                   HP_VC_GO(false, 256, true);
                } else {
                    HP_VC_GO(false, 256, false);
                }
            }
#undef HP_VC_GO
#undef HP_VC_ARGS
            return true;
        }
        if (refl) {
            if (pf) {
                if (block == 128)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, true, 128, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else if (block == 512)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, true, 512, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else if (block == 64)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, true, 64, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, true, 256, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
            } else
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, true>), grid, block, 0, stream, HP_VTX_ARGS);
        } else {
            if (pf) {
                if (block == 128)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 128, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else if (block == 512)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 512, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else if (block == 64)
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 64, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
                else
                    hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true, 256, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
            } else
                hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, true>), grid, block, 0, stream, HP_VTX_ARGS);
        }
        return false;
    }
    if (refl) {
        if (pf)
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, false, 256, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
        else
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<true, false>), grid, block, 0, stream, HP_VTX_ARGS);
    } else {
        if (pf)
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, false, 256, 0, true>), grid, block, 0, stream, HP_VTX_ARGS);
        else
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud<false, false>), grid, block, 0, stream, HP_VTX_ARGS);
    }
    launch_fused_near_clip(positions, normals, texcoords, texids, colors,
        spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
        tri_count, mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h,
        state, out, state.tri_stats, stream, near_clip);
    return false;
}

void launch_fused_vertex_tri_gouraud(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state,
    TriangleSetup* out, hipStream_t stream, NearClipQueue near_clip, BinEmit emit_in)
{
    (void)launch_fused_vertex_tri_gouraud_ex(positions, normals, texcoords, texids, colors,
        spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, state, out, stream,
        near_clip, emit_in, nullptr);
}


bool launch_fused_vertex_tri_gouraud_z2(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    int vp_x, int vp_y, int vp_w, int vp_h,
    VtxZSlot s0, VtxZSlot s1, hipStream_t stream)
{
    if (!positions || tri_count <= 0) return false;
    if (!s0.out || !s1.out) return false;
    static const int slim_env = [] {
        int v = hipgl_geom_slim_env();
        hipMemcpyToSymbol(HIP_SYMBOL(d_geom_slim), &v, sizeof(int));
        return v;
    }();
    (void)slim_env;
    static const bool reflection_early_cull = [] {
        const char* env = getenv("HIPGL_REFLECTION_EARLY_CULL");
        return !env || atoi(env) != 0;
    }();
    static const bool det_bin = [] {
        const char* e = getenv("HIPGL_DET_BIN");
        return e ? (e[0] == '1') : true;
    }();
    static const int bf_skip_env = [] {
        const char* e = getenv("HP_BF_SKIP");
        return e ? atoi(e) : 1;
    }();
    static const int small_tris = [] {
        const char* e = getenv("HP_VTX_SMALL_TRIS");
        return e ? atoi(e) : 4096;
    }();
    // Only the SATURATED shape. A draw small enough for the 64-thread block or
    // the cooperative micro path is not what this exists for, and keeping it
    // out means the launch shape here is always the 256-thread one.
    if (tri_count <= small_tris) return false;
    // The reflected slot needs the plane cull the <true,...> instantiation
    // carries; with the ablation on, the single-slot path would compile a
    // different kernel, so leave it alone.
    if (!reflection_early_cull) return false;
    if (!hp_nc_fold_flag()) return false;
    VtxZSlot* sl[2] = {&s0, &s1};
    for (int i = 0; i < 2; ++i) {
        VtxZSlot& s = *sl[i];
        s.bf_skip_mask = (bf_skip_env && det_bin && s.state.tri_stats == nullptr)
                       ? (backface_cull_mask(s.state) | 4) : 0;
        if (s.bf_skip_mask == 0 || s.emit.rec == nullptr) s.emit.skip_dead_core = false;
    }
    static const int z2_block_env = [] {
        const char* e = getenv("HP_Z2_BLOCK");
        int v = e ? atoi(e) : 256;
        return (v == 64 || v == 128 || v == 256 || v == 512) ? v : 256;
    }();
    const int bsz = z2_block_env;
    dim3 block((unsigned)bsz, 1, 1);
    dim3 grid((unsigned)((tri_count + bsz - 1) / bsz), 1u, 2u);
    const VtxPose pose = hp_pose_for_draw(positions);
    static const int pose_scalar = [] {
        const char* e = getenv("HP_VTX_POSE_SCALAR");
        return e ? atoi(e) : 2;
    }();
#define HP_Z2_LAUNCH_ARGS positions, normals, texcoords, texids, colors, spec_colors, \
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count, \
            vp_x, vp_y, vp_w, vp_h, s0, s1, pose
    if (pose.vert_geom && pose_scalar == 2) {
        if (bsz == 64)
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true, true, 2, 64>), grid, block, 0, stream, HP_Z2_LAUNCH_ARGS);
        else if (bsz == 128)
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true, true, 2, 128>), grid, block, 0, stream, HP_Z2_LAUNCH_ARGS);
        else if (bsz == 512)
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true, true, 2, 512>), grid, block, 0, stream, HP_Z2_LAUNCH_ARGS);
        else
            hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true, true, 2, 256>), grid, block, 0, stream, HP_Z2_LAUNCH_ARGS);
    }
    else if (pose.vert_geom)
        hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true, true>), grid, block, 0, stream,
            positions, normals, texcoords, texids, colors, spec_colors,
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
            vp_x, vp_y, vp_w, vp_h, s0, s1, pose);
    else
        hipLaunchKernelGGL((kernel_fused_vertex_tri_gouraud_z2<true>), grid, block, 0, stream,
            positions, normals, texcoords, texids, colors, spec_colors,
            per_vert_specular, per_vert_shininess, per_vert_emission, tri_count,
            vp_x, vp_y, vp_w, vp_h, s0, s1, pose);
    return true;
}

} // namespace kernels
} // namespace hipgl
