#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include "rt_math.inc"
#include "rt_texture.inc"

namespace hipgl {
namespace physics {

__device__ __forceinline__ float rt_sat(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ __forceinline__ float3 rt_sat3(float3 c) { return make_float3(rt_sat(c.x), rt_sat(c.y), rt_sat(c.z)); }
__device__ __forceinline__ float3 rt_mul3(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __forceinline__ float3 rt_ld3(const float* p) { return make_float3(p[0], p[1], p[2]); }
__device__ __forceinline__ float3 rt_cam_axis(const RtCam& C, int k) {
    return make_float3(C.Rc[k], C.Rc[3 + k], C.Rc[6 + k]);
}

// Wave-uniform read-only load through the constant address space (s_load:
// SGPRs instead of a vector load replicated into every lane's VGPRs).
template <typename T>
__device__ __forceinline__ T rt_sh_sload(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return *(const __attribute__((address_space(4))) T*)p;
#else
    return *p;
#endif
}

// Material of one shading evaluation.
struct RtMat { float3 C; float spec, shin, emis; };

// ── compile-time shading preset ─────────────────────────────────────────────
// Every shading function takes a preset class Sp whose accessors return the
// per-call-uniform frame fields that select code paths. RtSpecDyn reads them
// from the frame (any configuration); RtSpecFix<..> returns constants the host
// verified to equal the frame's (rt_pick_spec), so the paths a preset never
// takes are deleted from that instantiation (fewer VGPRs, higher occupancy).
// The arithmetic of the remaining paths is unchanged.
struct RtSpecDyn {
    static constexpr bool kFixed = false;
    static constexpr int kShade = 0;
    __device__ __forceinline__ static int  shade(const RtFrame& F) { return F.shade; }
    __device__ __forceinline__ static int  mode(const RtFrame& F) { return F.mode; }
    __device__ __forceinline__ static bool linear(const RtFrame& F) { return F.linear != 0; }
    __device__ __forceinline__ static bool lay(const RtFrame& F) { return F.ntransp > 0; }
    __device__ __forceinline__ static bool fog(const RtFrame& F) { return F.fog_on != 0; }
};
template <int SH, bool LIN, bool LAY, bool FOG>
struct RtSpecFix {
    static constexpr int kShade = SH;
    static constexpr bool kFixed = true;
    static constexpr bool kLinear = LIN, kLay = LAY, kFog = FOG;
    __device__ __forceinline__ static constexpr int  shade(const RtFrame&) { return SH; }
    __device__ __forceinline__ static constexpr int  mode(const RtFrame&) { return RT_MODE_HIPGL; }
    __device__ __forceinline__ static constexpr bool linear(const RtFrame&) { return LIN; }
    __device__ __forceinline__ static constexpr bool lay(const RtFrame&) { return LAY; }
    __device__ __forceinline__ static constexpr bool fog(const RtFrame&) { return FOG; }
};

// ── lighting ────────────────────────────────────────────────────────────────
// HIPGL: GL 1.x fixed function, single colour, local viewer, no global
// ambient (MuJoCo initLights with nlight > 0), ambient attenuated and spotted
// (GL semantics). Light 0 is the headlight; its direction (toward the light,
// world) is `hl`. Returns clamp(lit + clamp(spec)).
template <class Sp = RtSpecDyn>
__device__ inline float3 rt_light_gl(const RtFrame& F, float3 hl, float3 P, float3 N, float3 V,
                                     const RtMat& m, unsigned vis = ~0u) {
    const bool mesa = (Sp::shade(F) & RT_SH_MESA_FF) != 0;
    float3 lit = rt_mul(m.C, m.emis);   // global ambient x colour
    float3 sp = make_float3(0.f, 0.f, 0.f);
#pragma unroll 1
    for (int i = 0; i < F.nlight; ++i) {
        const RtLight L = rt_sh_sload(F.lights + i);
        if (!L.active) continue;
        float3 Ld;
        float att = 1.0f;
        if (L.type != 1) {
            // headlight / point (GL default position (0,0,1,0), eye space)
            Ld = (i == 0 || L.type == 2) ? hl : make_float3(-L.dir[0], -L.dir[1], -L.dir[2]);
        } else {
            Ld = rt_sub(rt_ld3(L.pos), P);
            const float dist = rt_len(Ld);
            if (dist > 1e-8f) Ld = rt_mul(Ld, 1.0f / dist);
            if (!L.att_default) att = 1.0f / (L.att[0] + L.att[1] * dist + L.att[2] * dist * dist);
            const float cosA = -rt_dot(Ld, rt_ld3(L.dir));
            att *= (cosA < L.cos_cut) ? 0.0f : powf(cosA, L.expo);
        }
        const float NdLr = rt_dot(N, Ld);
        // RT_SH_SHADOW: an occluded light (vis bit clear) keeps only its ambient term
        const float NdL = ((vis >> i) & 1u) ? fmaxf(NdLr, 0.0f) : 0.0f;
        lit.x += att * (L.amb[0] * m.C.x + L.dif[0] * m.C.x * NdL);
        lit.y += att * (L.amb[1] * m.C.y + L.dif[1] * m.C.y * NdL);
        lit.z += att * (L.amb[2] * m.C.z + L.dif[2] * m.C.z * NdL);
        if (NdL > 0.0f && m.spec > 0.0f) {
            // One powf call site for both specular forms (the same values
            // reach the same function; fewer inlined pow expansions).
            float pb, pe;
            bool usep = true;
            if (m.shin > 0.0f) {
                const float3 H = rt_norm(rt_add(Ld, V));
                pb = fmaxf(rt_dot(N, H), 0.0f); pe = m.shin;
            } else if (mesa && L.type == 1) {
                // ffvertex_prog.c with material_shininess_is_zero replicates
                // N.L into every lit() input, so an attenuated (spot) light
                // gets pow(N.L, clamp(N.L)) where GL says 1.
                pb = NdL; pe = fminf(NdLr, 128.0f);
            } else {
                pb = 1.0f; pe = 1.0f; usep = false;
            }
            float f = 1.0f;
            if (usep) f = powf(pb, pe);
            const float s = att * m.spec * f;
            sp.x += s * L.spc[0]; sp.y += s * L.spc[1]; sp.z += s * L.spc[2];
        }
    }
    if (F.gl_amb) lit = rt_add(lit, rt_mul(m.C, 0.3f));   // no active light: GL global ambient x colour
    return rt_sat3(rt_add(lit, rt_sat3(sp)));
}

// MJW (mujoco_warp render.py:1750-2002): emission + headlight ambient + per
// light ambient (weighted by attenuation/spot) + diffuse + specular (not x C),
// clamped; the texel multiplies afterwards. Lights are the same table.
__device__ inline float3 rt_light_mjw(const RtFrame& F, float3 hl, float3 P, float3 N, float3 V,
                                      const RtMat& m) {
    float3 col = rt_mul(m.C, m.emis);
    const bool head = F.nlight > 0 && F.lights[0].active;
    if (head) col = rt_add(col, rt_mul3(m.C, rt_ld3(F.lights[0].amb)));
    else if (F.nlight <= 1) col = rt_add(col, rt_mul(m.C, 0.3f));
#pragma unroll 1
    for (int i = 0; i < F.nlight; ++i) {
        const RtLight L = rt_sh_sload(F.lights + i);
        if (!L.active) continue;
        float3 Ld;
        float att = 1.0f;
        if (L.type == 0) {
            Ld = (i == 0) ? hl : make_float3(-L.dir[0], -L.dir[1], -L.dir[2]);
        } else {
            Ld = rt_sub(rt_ld3(L.pos), P);
            const float dist = rt_len(Ld);
            if (dist > 1e-8f) Ld = rt_mul(Ld, 1.0f / dist);
            if (!L.att_default) att = 1.0f / (L.att[0] + L.att[1] * dist + L.att[2] * dist * dist);
            const float cosA = -rt_dot(Ld, rt_ld3(L.dir));
            att *= (cosA < L.cos_cut) ? 0.0f : powf(cosA, L.expo);
        }
        if (i > 0) col = rt_add(col, rt_mul(rt_mul3(m.C, rt_ld3(L.amb)), att));
        const float NdL = fmaxf(rt_dot(N, Ld), 0.0f);
        if (NdL <= 0.0f) continue;
        col = rt_add(col, rt_mul(rt_mul3(m.C, rt_ld3(L.dif)), NdL * att));
        if (m.spec > 0.0f) {
            const float3 H = rt_norm(rt_add(Ld, V));
            const float NdH = fmaxf(rt_dot(N, H), 0.0f);
            const float s = att * m.spec * ((m.shin > 0.0f) ? powf(NdH, m.shin) : 1.0f);
            col = rt_add(col, rt_mul(rt_ld3(L.spc), s));
        }
    }
    return rt_sat3(col);
}

template <class Sp = RtSpecDyn>
__device__ __forceinline__ float3 rt_light(const RtFrame& F, float3 hl, float3 P, float3 N, float3 V,
                                           const RtMat& m, unsigned vis = ~0u) {
    return Sp::mode(F) == RT_MODE_MJW ? rt_light_mjw(F, hl, P, N, V, m) : rt_light_gl<Sp>(F, hl, P, N, V, m, vis);
}

// ── geometry helpers ────────────────────────────────────────────────────────
// Solve P - v2 = a (v0 - v2) + b (v1 - v2) in the least-squares sense (exact
// for P in the triangle's plane): the (corner0, corner1) barycentrics of P.
__device__ inline void rt_bary2(float3 e0, float3 e1, float3 d, float& a, float& b) {
    const float m00 = rt_dot(e0, e0), m01 = rt_dot(e0, e1), m11 = rt_dot(e1, e1);
    const float r0 = rt_dot(e0, d), r1 = rt_dot(e1, d);
    const float det = m00 * m11 - m01 * m01;
    const float inv = fabsf(det) > 1e-30f ? 1.0f / det : 0.0f;
    a = (r0 * m11 - r1 * m01) * inv;
    b = (r1 * m00 - r0 * m01) * inv;
}

// One sample's surface: everything shading needs, in world and local frames.
struct RtSurf {
    float3 P, l;        // hit, world and geom-local
    float3 Ng;          // geometric (world, unit) normal: triangle / analytic
    float3 nl;          // local shading normal of the hit (prims: analytic)
    float3 dPx, dPy;    // world position derivatives per LOD-grid step
    int    li, tri;
    float  u, v;        // mesh barycentrics (corner 0, corner 1)
};

// Ray differential (Igehy) of a hit at t along unit rd onto the plane (P, Ng).
// D = rd / cz is the unnormalised camera direction (camera z = -1), whose
// screen derivatives are dDx, dDy; P = eye + t rd = eye + (t cz) D.
__device__ inline void rt_ray_diff(float3 rd, float t, float cz, float3 Ng, float3 dDx, float3 dDy,
                                   float3& dPx, float3& dPy) {
    float rn = rt_dot(rd, Ng);
    if (fabsf(rn) < 1e-6f) rn = rn < 0.0f ? -1e-6f : 1e-6f;
    const float s = t * cz;
    dPx = rt_mul(rt_sub(dDx, rt_mul(rd, rt_dot(dDx, Ng) / rn)), s);
    dPy = rt_mul(rt_sub(dDy, rt_mul(rd, rt_dot(dDy, Ng) / rn)), s);
}

template <class Sp = RtSpecDyn>
__device__ inline bool rt_texgen_prim(const RtFrame& F, const RtGeom& g, bool cube, float a[3], float b[3]) {
    const bool mj = (Sp::shade(F) & RT_SH_TEXGEN_MJ) != 0 || Sp::mode(F) == RT_MODE_MJW;
    const float rs = g.texrep[0], rt = g.texrep[1];
    const float hx = g.size[0], hy = g.size[1], hz = g.size[2];
    a[0] = a[1] = a[2] = 0.0f; b[0] = b[1] = b[2] = 0.0f;
    switch (g.type) {
    case GEOM_BOX: {
        const float ix = hx > 1e-6f ? 1.0f / hx : 0.0f;
        const float iy = hy > 1e-6f ? 1.0f / hy : 0.0f;
        const float iz = hz > 1e-6f ? 1.0f / hz : 0.0f;
        if (cube) {
            a[0] = mj ? ix : ix * rs; a[1] = mj ? iy : iy * rt; a[2] = iz;
        } else {
            a[0] = 0.5f * rs * ix; a[1] = -0.5f * rt * iy;
            b[0] = mj ? -0.5f : 0.5f * rs; b[1] = mj ? -0.5f : -0.5f * rt;
        }
        return true;
    }
    case GEOM_PLANE: {
        if (mj) {
            // plane-specific display list: object coords are metric; scl =
            // texrep / size (size > 0), hp folds texuniform into texrep.
            a[0] = 0.5f * rs * (hx > 0.0f ? 1.0f / hx : 1.0f);
            a[1] = -0.5f * rt * (hy > 0.0f ? 1.0f / hy : 1.0f);
            b[0] = -0.5f; b[1] = -0.5f;
        } else {
            const float px = hx > 0.01f ? hx : 5.0f, py = hy > 0.01f ? hy : 5.0f;
            a[0] = rs / (2.0f * px); b[0] = 0.5f * rs;
            a[1] = -rt / (2.0f * py); b[1] = -0.5f * rt;
        }
        return true;
    }
    case GEOM_CYLINDER: {
        const float ir = hx > 1e-9f ? 1.0f / hx : 0.0f;
        const float ih = hy > 1e-9f ? 1.0f / hy : 0.0f;
        if (cube) { a[0] = ir; a[1] = ir; a[2] = ih; }
        else {
            a[0] = 0.5f * rs * ir; a[1] = (mj ? -0.5f : 0.5f) * rt * ir;
            b[0] = -0.5f; b[1] = -0.5f;
        }
        return true;
    }
    case GEOM_SPHERE:
    case GEOM_CAPSULE: {
        if (!mj) return false;     // the raster never textures spheres/capsules
        const float ir = hx > 1e-9f ? 1.0f / hx : 0.0f;
        const float iz = g.type == GEOM_SPHERE ? ir : (hy + hx > 1e-9f ? 1.0f / (hy + hx) : 0.0f);
        if (cube) { a[0] = ir; a[1] = ir; a[2] = iz; }
        else { a[0] = 0.5f * rs * ir; a[1] = -0.5f * rt * ir; b[0] = -0.5f; b[1] = -0.5f; }
        return true;
    }
    default:
        return false;
    }
}

// Texel of a surface (1,1,1 if untextured). R = geom rotation (local->world).
template <class Sp = RtSpecDyn>
__device__ inline float3 rt_texel(const RtFrame& F, const RtGeom& g, const float R[9], const RtSurf& S) {
    if (!(g.flags & RT_F_TEXTURED)) return make_float3(1.f, 1.f, 1.f);
    const bool cube = (g.flags & RT_F_CUBE) != 0;
    const bool mj = (Sp::shade(F) & RT_SH_TEXGEN_MJ) != 0 || Sp::mode(F) == RT_MODE_MJW;
    const float3 dlx = rt_mat_tvec(R, S.dPx), dly = rt_mat_tvec(R, S.dPy);
    float tc[3], tx[3], ty[3];
    if ((g.flags & RT_F_TEXGEN) && F.texgen && mj && g.gid >= 0) {
        // explicit object-linear planes (MuJoCo settexture in hip's geom frame)
        for (int k = 0; k < 3; ++k) {
            const float4 P = F.texgen[g.gid * 3 + k];
            tc[k] = P.x * S.l.x + P.y * S.l.y + P.z * S.l.z + P.w;
            tx[k] = P.x * dlx.x + P.y * dlx.y + P.z * dlx.z;
            ty[k] = P.x * dly.x + P.y * dly.y + P.z * dly.z;
        }
    } else if (g.type == GEOM_MESH) {
        if (cube) {
            tc[0] = S.l.x; tc[1] = S.l.y; tc[2] = S.l.z;
            tx[0] = dlx.x; tx[1] = dlx.y; tx[2] = dlx.z;
            ty[0] = dly.x; ty[1] = dly.y; ty[2] = dly.z;
        } else {
            if (!F.mesh_texcoord || !F.mesh_vert || S.tri < 0) return make_float3(1.f, 1.f, 1.f);
            const int c0 = g.vert_adr + 3 * S.tri;
            const float* T = F.mesh_texcoord + (size_t)c0 * 2;
            const float* Vv = F.mesh_vert + (size_t)c0 * 3;
            const float w2 = 1.0f - S.u - S.v;
            const float rs = mj ? 1.0f : g.texrep[0], rt = mj ? 1.0f : g.texrep[1];
            tc[0] = (S.u * T[0] + S.v * T[2] + w2 * T[4]) * rs;
            tc[1] = (S.u * T[1] + S.v * T[3] + w2 * T[5]) * rt;
            tc[2] = 0.0f;
            const float3 v2 = rt_ld3(Vv + 6);
            const float3 e0 = rt_sub(rt_ld3(Vv), v2), e1 = rt_sub(rt_ld3(Vv + 3), v2);
            float ax, bx, ay, by;
            rt_bary2(e0, e1, dlx, ax, bx);
            rt_bary2(e0, e1, dly, ay, by);
            const float du0 = T[0] - T[4], du1 = T[2] - T[4];
            const float dv0 = T[1] - T[5], dv1 = T[3] - T[5];
            tx[0] = (ax * du0 + bx * du1) * rs; tx[1] = (ax * dv0 + bx * dv1) * rt; tx[2] = 0.0f;
            ty[0] = (ay * du0 + by * du1) * rs; ty[1] = (ay * dv0 + by * dv1) * rt; ty[2] = 0.0f;
        }
    } else {
        float a[3], b[3];
        if (!rt_texgen_prim<Sp>(F, g, cube, a, b)) return make_float3(1.f, 1.f, 1.f);
        tc[0] = a[0] * S.l.x + b[0]; tc[1] = a[1] * S.l.y + b[1]; tc[2] = a[2] * S.l.z + b[2];
        tx[0] = a[0] * dlx.x; tx[1] = a[1] * dlx.y; tx[2] = a[2] * dlx.z;
        ty[0] = a[0] * dly.x; ty[1] = a[1] * dly.y; ty[2] = a[2] * dly.z;
    }
    float r, gg, bb;
    const int tflags = (F.tex_flags && g.texid < F.ntex_flags) ? F.tex_flags[g.texid] : 0;
    // Mesa/llvmpipe sampler emulation is a property of MuJoCo's GL frames, not of mjw's
    const bool mesa = (Sp::shade(F) & RT_SH_MESA_TEX) && Sp::mode(F) != RT_MODE_MJW;
    const bool srgb = (tflags & RT_TEX_SRGB) != 0;
    const bool hw = !mesa && (Sp::shade(F) & RT_SH_HW_TEX) && Sp::mode(F) != RT_MODE_MJW && F.tex.tex_data &&
                    g.texid < F.tex.ntexture;
    const TextureMipLevel* hw_mip = (F.hw_mip && g.texid < F.hw_ntex) ? F.hw_mip : nullptr;
    if (hw) {
        if (cube) rt_tex_sample_hw<true>(F.tex, F.hw_data, hw_mip, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
        else      rt_tex_sample_hw<false>(F.tex, F.hw_data, hw_mip, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
    } else if (cube) rt_tex_sample_grad<true>(F.tex, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
    else if (mesa && (tflags & RT_TEX_SRGB_DATA) && F.srgb_data)
        rt_tex_sample_srgb(F.tex, F.srgb_data, F.srgb_mip, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
    else if (mesa && !srgb && F.tex.tex_data && g.texid < F.tex.ntexture)
        rt_tex_sample_aos(F.tex, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
    else      rt_tex_sample_grad<false>(F.tex, g.texid, tc, tx, ty, F.lod_bias, r, gg, bb);
    if (Sp::linear(F)) {   // linear_output: decode sRGB texels (bin_rasterize.hip decode_texel)
        auto dec = [](float c) {
            return c <= 0.04045f ? c * (1.0f / 12.92f) : __powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
        };
        r = dec(r); gg = dec(gg); bb = dec(bb);
    }
    return make_float3(r, gg, bb);
}

// ── virtual tessellation (RT_SH_VTESS) ──────────────────────────────────────
// MuJoCo draws primitives as GL_QUADS from display lists (render_context.c
// makeBuiltin / makePlane) and GL lights their corners (GL_SMOOTH). For a hit
// at (fa, fb) in a quad whose corners c0..c3 sit at (0,0) (1,0) (1,1) (0,1)
// of the unit square ("P" order; callers mirror fb for the reversed faces),
// pick the GL triangle and its corner weights. Mesa splits a quad into
// (c0,c1,c3),(c1,c2,c3); RT_SH_QUAD_ALT selects (c0,c1,c2),(c0,c2,c3).
// Outputs corner indices k[3] (0..3) and weights w[3].
__device__ inline void rt_quad_tri(float fa, float fb, bool alt, int& k0, int& k1, int& k2,
                                   float& w0, float& w1, float& w2) {
    if (!alt) {
        if (fa + fb <= 1.0f) { k0 = 0; k1 = 1; k2 = 3; w0 = 1.0f - fa - fb; w1 = fa; w2 = fb; }
        else                 { k0 = 1; k1 = 2; k2 = 3; w0 = 1.0f - fb; w1 = fa + fb - 1.0f; w2 = 1.0f - fa; }
    } else {
        if (fa >= fb) { k0 = 0; k1 = 1; k2 = 2; w0 = 1.0f - fa; w1 = fa - fb; w2 = fb; }
        else          { k0 = 0; k1 = 2; k2 = 3; w0 = 1.0f - fb; w1 = fa; w2 = fb - fa; }
    }
}

// Cell of MuJoCo's finite-plane grid along one axis: [-sz, -sz+pad], nn cells
// of 2h, [.., sz]. Returns the cell's [lo, hi], and a code naming the cell
// (equal codes on one geom axis give bit-identical lo / hi).
__device__ inline int rt_plane_cell(float x, float sz, float h, float pad, float& lo, float& hi) {
    const float x0 = -sz + pad;
    const int nn = (int)rintf((sz - pad) / h);
    if (pad > 0.0f && x < x0) { lo = -sz; hi = x0; return -1; }
    int k = (int)floorf((x - x0) / (2.0f * h));
    if (k >= nn) {
        if (pad > 0.0f) { lo = x0 + (float)nn * 2.0f * h; hi = sz; return nn; }
        k = nn - 1;
    }
    if (k < 0) k = 0;
    lo = x0 + (float)k * 2.0f * h;
    hi = lo + 2.0f * h;
    return k;
}

// The 4 corners of a virtual quad as scalars (an array indexed by a runtime
// corner id would be placed in scratch).
struct RtQuad {
    float3 q0, q1, q2, q3, n0, n1, n2, n3;
    __device__ __forceinline__ float3 q(int i) const { return i == 0 ? q0 : i == 1 ? q1 : i == 2 ? q2 : q3; }
    __device__ __forceinline__ float3 n(int i) const { return i == 0 ? n0 : i == 1 ? n1 : i == 2 ? n2 : n3; }
    __device__ __forceinline__ void setq(int i, float3 v) { if (i == 0) q0 = v; else if (i == 1) q1 = v; else if (i == 2) q2 = v; else q3 = v; }
    __device__ __forceinline__ void setn(float3 v) { n0 = n1 = n2 = n3 = v; }
};

// Corners (local positions Q, local normals Nq) of the virtual quad that holds
// local hit l on primitive g, and the hit's position (fa, fb) in P order.
// Returns false when there is no tessellation to emulate (per-hit shading).
// (qa, qb) name the quad within its geom: equal ids give bit-identical Q
// (the wave-level corner sharing in rt_shade_surface relies on it).
// All array indices are compile-time constants after unrolling (a dynamically
// indexed private array would be placed in scratch).
__device__ inline bool rt_vtess_quad(const RtGeom& g, float3 l, float3 nl, RtQuad& Q,
                                     float& fa, float& fb, int& qa, int& qb) {
    constexpr int   NQ = 4;            // m->vis.quality.numquads (MuJoCo default)
    constexpr int   NSLICE = 28;       // numslices
    constexpr int   NSTACK = 16;       // numstacks
    constexpr float TWO_PI = 6.28318530717958647692f;
    if (g.type == GEOM_BOX) {
        // Face from the analytic normal; (a, b) unit-cube params per
        // makeBuiltin's box list. 'rev' faces emit (a0,b1),(a1,b1),(a1,b0),(a0,b0).
        const float sx = g.size[0], sy = g.size[1], sz = g.size[2];
        const float ux = sx > 0.f ? l.x / sx : 0.f, uy = sy > 0.f ? l.y / sy : 0.f, uz = sz > 0.f ? l.z / sz : 0.f;
        const float anx = fabsf(nl.x), any = fabsf(nl.y), anz = fabsf(nl.z);
        // ax: 0 x, 1 y, 2 z ; params (a, b) = (x,y) for z faces, (y,z) for x faces, (x,z) for y faces
        int ax; float sg; bool rev;
        if (anz >= anx && anz >= any) { ax = 2; sg = nl.z > 0.f ? 1.f : -1.f; rev = sg < 0.f; }
        else if (anx >= any)          { ax = 0; sg = nl.x > 0.f ? 1.f : -1.f; rev = sg < 0.f; }
        else                          { ax = 1; sg = nl.y > 0.f ? 1.f : -1.f; rev = sg > 0.f; }
        const float a = ax == 0 ? uy : ux;
        const float b = ax == 2 ? uy : uz;
        const float d = 2.0f / NQ;
        const int ca = min(NQ - 1, max(0, (int)floorf((a + 1.0f) / d)));
        const int cb = min(NQ - 1, max(0, (int)floorf((b + 1.0f) / d)));
        const float a0 = d * ca - 1.0f, b0 = d * cb - 1.0f;
        qa = ((ax * 2 + (sg > 0.f ? 1 : 0)) * NQ + ca) * NQ + cb; qb = 0;
        fa = (a - a0) / d;
        fb = (b - b0) / d;
        const float3 nrm = make_float3(ax == 0 ? sg : 0.f, ax == 1 ? sg : 0.f, ax == 2 ? sg : 0.f);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int ci = (k == 1 || k == 2) ? 1 : 0, cj = (k >= 2) ? 1 : 0;
            const float ua = a0 + d * ci;
            const float ub = b0 + d * (rev ? 1 - cj : cj);
            // unit (x, y, z) of the corner
            const float cx = ax == 0 ? sg : ua;
            const float cy = ax == 1 ? sg : (ax == 0 ? ua : ub);
            const float cz2 = ax == 2 ? sg : ub;
            Q.setq(k, make_float3(cx * sx, cy * sy, cz2 * sz));
        }
        Q.setn(nrm);
        if (rev) fb = 1.0f - fb;
        return true;
    }
    if (g.type == GEOM_PLANE) {
        if (g.vt0 == 0.0f || g.vt2 == 0.0f) return false;     // no grid recorded: per hit
        float x0, x1, y0, y1;
        if (g.vt0 < 0.0f) { const float d = -g.vt0; const float cx = floorf(l.x / d); qa = __float_as_int(cx); x0 = cx * d; x1 = x0 + d; }
        else qa = rt_plane_cell(l.x, g.size[0], g.vt0, g.vt1, x0, x1);
        if (g.vt2 < 0.0f) { const float d = -g.vt2; const float cy = floorf(l.y / d); qb = __float_as_int(cy); y0 = cy * d; y1 = y0 + d; }
        else qb = rt_plane_cell(l.y, g.size[1], g.vt2, g.vt3, y0, y1);
        fa = x1 > x0 ? (l.x - x0) / (x1 - x0) : 0.0f;
        fb = y1 > y0 ? (l.y - y0) / (y1 - y0) : 0.0f;
        Q.q0 = make_float3(x0, y0, 0.f); Q.q1 = make_float3(x1, y0, 0.f);
        Q.q2 = make_float3(x1, y1, 0.f); Q.q3 = make_float3(x0, y1, 0.f);
        Q.setn(make_float3(0.f, 0.f, 1.f));
        return true;
    }
    if (g.type == GEOM_CYLINDER) {
        const float r = g.size[0], hh = g.size[1];
        float az = atan2f(l.y, l.x);
        if (az < 0.0f) az += TWO_PI;
        const float daz = TWO_PI / NSLICE;
        const int j = min(NSLICE - 1, max(0, (int)floorf(az / daz)));
        const float az1 = daz * j, az2 = daz * (j + 1);
        const float c1 = cosf(az1), s1 = sinf(az1), c2 = cosf(az2), s2 = sinf(az2);
        if (fabsf(nl.z) < 0.5f) {
            // barrel: quads (az1,h1) (az2,h1) (az2,h2) (az1,h2), P order;
            // fa along the chord (the facet is planar), fb along z.
            const float zu = hh > 0.f ? l.z / hh : 0.f;
            const float dh = 2.0f / NSTACK;
            const int i = min(NSTACK - 1, max(0, (int)floorf((zu + 1.0f) / dh)));
            const float h1 = dh * i - 1.0f, h2 = h1 + dh;
            qa = j; qb = 0x10000 | i;
            const float ex = (c2 - c1) * r, ey = (s2 - s1) * r;
            const float el2 = ex * ex + ey * ey;
            fa = el2 > 0.f ? ((l.x - c1 * r) * ex + (l.y - s1 * r) * ey) / el2 : 0.f;
            fb = (zu - h1) / dh;
            Q.q0 = make_float3(c1 * r, s1 * r, h1 * hh); Q.q1 = make_float3(c2 * r, s2 * r, h1 * hh);
            Q.q2 = make_float3(c2 * r, s2 * r, h2 * hh); Q.q3 = make_float3(c1 * r, s1 * r, h2 * hh);
            Q.n0 = Q.n3 = make_float3(c1, s1, 0.f);
            Q.n1 = Q.n2 = make_float3(c2, s2, 0.f);
            return true;
        }
        // caps (disk): ring i spans radii ((i+1)/n, (i+2)/n), n = NSTACK/2;
        // the innermost ring is a fan of triangles around the centre.
        const int n = NSTACK / 2;
        const float sg = nl.z > 0.f ? 1.f : -1.f;
        const float ru = r > 0.f ? sqrtf(l.x * l.x + l.y * l.y) / r : 0.f;
        const float3 nz = make_float3(0.f, 0.f, sg);
        Q.setn(nz);
        int ring = (int)floorf(ru * n) - 1;      // quad ring index, -1 = fan
        if (ring > n - 2) ring = n - 2;
        qa = j; qb = (ring < 0 ? 0x30000 : (0x20000 | ring)) | (sg > 0.f ? 0x100 : 0);
        if (ring < 0) {
            // fan triangle (rim1, rim2, centre) as a degenerate quad c2 == c3
            const float rr = r / n;
            Q.q0 = make_float3(c1 * rr, s1 * rr, sg * hh); Q.q1 = make_float3(c2 * rr, s2 * rr, sg * hh);
            Q.q2 = Q.q3 = make_float3(0.f, 0.f, sg * hh);
            fa = -1.0f; fb = -1.0f;    // flag: caller solves the triangle directly
            return true;
        }
        const float r1 = (ring + 1) / (float)n * r, r2 = (ring + 2) / (float)n * r;
        // sign > 0: (az1,r2) (az2,r2) (az2,r1) (az1,r1) ; sign < 0: reversed.
        const float3 P0 = make_float3(c1 * r2, s1 * r2, sg * hh), P1 = make_float3(c2 * r2, s2 * r2, sg * hh);
        const float3 P2 = make_float3(c2 * r1, s2 * r1, sg * hh), P3 = make_float3(c1 * r1, s1 * r1, sg * hh);
        const bool up = sg > 0.f;
        Q.q0 = up ? P0 : P3; Q.q1 = up ? P1 : P2; Q.q2 = up ? P2 : P1; Q.q3 = up ? P3 : P0;
        fa = -2.0f; fb = -2.0f;        // flag: general planar quad, caller solves
        return true;
    }
    return false;
}

// ── wave-shared lighting (rt_shade_surface) ─────────────────────────────────
// LDS staging of one wave's distinct lighting evaluations (workgroup = one
// wave). Item: A = local corner / world point, B = local normal / world unit
// normal, A.w = owner lane (| 64: world point). Owner (group root) lane:
// viewer (corner) or V (point), instance, material, vis. The result of item s
// overwrites A[s].xyz.
constexpr int RT_LT_ITEMS = 3 * 64;
struct RtLtLds {
    float4 a[RT_LT_ITEMS];
    float4 b[RT_LT_ITEMS];
    float4 rv[64];          // viewer / V, w: material spec
    float4 rc[64];          // material colour, w: shin
    float4 rx[64];          // x: emis, y: vis, zw: instance pointer
};

// Wave sharing pays off where many lanes light the same corners: MuJoCo's
// virtual quads (RT_SH_VTESS); the other presets keep the per-lane paths.
template <class Sp>
__device__ __forceinline__ constexpr bool rt_lt_share() {
    if constexpr (Sp::kFixed) return (Sp::kShade & RT_SH_VTESS) != 0;
    else return false;
}

template <class Sp = RtSpecDyn>
__device__ __attribute__((always_inline)) inline float3 rt_light_shared(
        const RtFrame& F, float3 hl, const RtInst& I, const RtMat& m, float3 eye, unsigned vis, bool share,
        int nit, float3 A0, float3 A1, float3 A2, float3 B0, float3 B1, float3 B2, float3 Vw,
        float w0, float w1, float w2, int k0, int k1, int k2) {
    __shared__ RtLtLds s_lt;
    const int lane = (int)__lane_id();
    const unsigned long long lt = (1ull << lane) - 1ull;
    // 1. group lanes with equal keys: parent = lowest equal-key neighbour
    //    (left / up-left / up / up-right in the 8-wide lane grid), then
    //    pointer jumping to the group root.
    if (!share || nit != 3) k0 = -1 - lane;
    int par = lane;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const int d = q == 0 ? 9 : q == 1 ? 8 : q == 2 ? 7 : 1;
        const int nb = lane >= d ? lane - d : lane;
        const int a = __shfl(k0, nb), b = __shfl(k1, nb), c = __shfl(k2, nb);
        if (k0 >= 0 && nb < par && a == k0 && b == k1 && c == k2) par = nb;
    }
    for (;;) {
        const int pp = __shfl(par, par);
        const bool ch = pp != par;
        par = pp;
        if (__ballot(ch) == 0ull) break;
    }
    const bool root = par == lane && nit > 0;
    const unsigned long long m3 = __ballot(root && nit == 3), m1 = __ballot(root && nit == 1);
    const int base = 3 * __popcll(m3 & lt) + __popcll(m1 & lt);
    const int nall = 3 * __popcll(m3) + __popcll(m1);
    const int rb = __shfl(base, par);
    __syncthreads();
    if (root) {
        const unsigned long long ip = (unsigned long long)(uintptr_t)&I;
        const float3 v = nit == 3 ? eye : Vw;
        s_lt.rv[lane] = make_float4(v.x, v.y, v.z, m.spec);
        s_lt.rc[lane] = make_float4(m.C.x, m.C.y, m.C.z, m.shin);
        s_lt.rx[lane] = make_float4(m.emis, __uint_as_float(vis), __uint_as_float((unsigned)ip),
                                    __uint_as_float((unsigned)(ip >> 32)));
        const float tag = __int_as_float(nit == 3 ? lane : lane | 64);
        s_lt.a[base] = make_float4(A0.x, A0.y, A0.z, tag);
        s_lt.b[base] = make_float4(B0.x, B0.y, B0.z, 0.f);
        if (nit == 3) {
            s_lt.a[base + 1] = make_float4(A1.x, A1.y, A1.z, tag);
            s_lt.b[base + 1] = make_float4(B1.x, B1.y, B1.z, 0.f);
            s_lt.a[base + 2] = make_float4(A2.x, A2.y, A2.z, tag);
            s_lt.b[base + 2] = make_float4(B2.x, B2.y, B2.z, 0.f);
        }
    }
    __syncthreads();
    // 2. evaluate the distinct items, one per lane per pass
#pragma unroll 1
    for (int s0 = 0; s0 < nall; s0 += 64) {
        const int s = s0 + lane;
        if (s < nall) {
            const float4 a = s_lt.a[s], b = s_lt.b[s];
            const int tag = __float_as_int(a.w);
            const int r = tag & 63;
            const float4 rv = s_lt.rv[r], rc = s_lt.rc[r], rx = s_lt.rx[r];
            RtMat mm;
            mm.C = make_float3(rc.x, rc.y, rc.z); mm.spec = rv.w; mm.shin = rc.w; mm.emis = rx.x;
            const float3 av = make_float3(a.x, a.y, a.z), bv = make_float3(b.x, b.y, b.z);
            const float3 vv = make_float3(rv.x, rv.y, rv.z);
            float3 P, N, V;
            if (tag & 64) {
                P = av; N = bv; V = vv;
            } else {
                const RtInst* Ip = (const RtInst*)(uintptr_t)(((unsigned long long)__float_as_uint(rx.w) << 32) |
                                                             (unsigned long long)__float_as_uint(rx.z));
                P = rt_add(rt_ld3(Ip->p), rt_mat_vec(Ip->R, av));
                N = rt_norm(rt_mat_vec(Ip->R, bv));
                V = rt_norm(rt_sub(vv, P));
            }
            const float3 f = rt_light<Sp>(F, hl, P, N, V, mm, __float_as_uint(rx.y));
            s_lt.a[s] = make_float4(f.x, f.y, f.z, a.w);
        }
    }
    __syncthreads();
    // 3. gather: corners summed in order 0, 1, 2 (as the per-lane loop did)
    float3 col = make_float3(0.f, 0.f, 0.f);
    if (nit == 1) {
        const float4 f = s_lt.a[rb];
        col = make_float3(f.x, f.y, f.z);
    } else if (nit == 3) {
        const float4 f0 = s_lt.a[rb], f1 = s_lt.a[rb + 1], f2 = s_lt.a[rb + 2];
        col = rt_add(col, rt_mul(make_float3(f0.x, f0.y, f0.z), w0));
        col = rt_add(col, rt_mul(make_float3(f1.x, f1.y, f1.z), w1));
        col = rt_add(col, rt_mul(make_float3(f2.x, f2.y, f2.z), w2));
    }
    return col;
}

// ── surface shading (no mirror) ─────────────────────────────────────────────
// Colour of surface S of geom g (instance I) seen along unit rd from `eye`
// (the camera, or its mirror image for a reflected hit: Gouraud corners need
// the viewer position, GL_LIGHT_MODEL_LOCAL_VIEWER). refl_dim > 0 scales the
// material colour (MuJoCo renderGeomReflection). Clamped to [0,1], before fog
// / output encoding.
template <class Sp = RtSpecDyn>
__device__ inline float3 rt_shade_surface(const RtFrame& F, const RtCam& C, const RtInst& I,
                                          const RtGeom& g, const RtSurf& S, float3 eye, float3 rd,
                                          float refl_dim, unsigned vis = ~0u, bool lit = true,
                                          bool share = true) {
    const float3 hl = rt_cam_axis(C, 2);           // headlight: toward the viewer (camera +Z)
    if (Sp::mode(F) == RT_MODE_GREY) {
        const float3 N = rt_norm(rt_mat_vec(I.R, rt_norm(S.nl)));
        const float gv = fabsf(rt_dot(N, rt_mul(rd, -1.0f)));
        return make_float3(gv, gv, gv);
    }
    const int shade = Sp::shade(F);
    const bool mjw = Sp::mode(F) == RT_MODE_MJW;
    const bool textured = (g.flags & RT_F_TEXTURED) != 0 &&
        (g.type == GEOM_MESH || g.type == GEOM_BOX || g.type == GEOM_PLANE || g.type == GEOM_CYLINDER ||
         (shade & RT_SH_TEXGEN_MJ) != 0 || mjw);
    RtMat m;
    m.C = make_float3(g.rgba[0], g.rgba[1], g.rgba[2]);
    if ((g.flags & RT_F_SITE) && F.site_rgba && I.pad < F.site_rgba_nw) {
        const float* c = F.site_rgba + ((size_t)I.pad * F.nsite + g.pad0) * 4;
        m.C = make_float3(c[0], c[1], c[2]);
    }
    if (textured && !mjw && !(shade & RT_SH_TEX_RGBA)) m.C = make_float3(1.f, 1.f, 1.f);
    if (refl_dim > 0.0f) m.C = rt_mul(m.C, refl_dim);
    m.spec = g.spec; m.shin = g.shin; m.emis = g.emis;

    // The texel does not depend on the lighting: sample it first, so only its
    // colour (not the surface's texture coordinates and footprint) stays live
    // across the lighting loops. Same product as texel-after-lighting.
    const float3 tex = textured ? rt_texel<Sp>(F, g, I.R, S) : make_float3(1.f, 1.f, 1.f);
    float3 col;
    if constexpr (rt_lt_share<Sp>()) {
        // ── lighting, shared across the wave ───────────────────────────────────
        // Every lane needs either 3 lit corners (Gouraud mesh triangle / virtual
        // quad triangle, blended by its weights) or 1 lit point (per hit). Lanes on
        // the same triangle (same geom, mesh triangle or quad + corner triple) have
        // bit-identical corner inputs (local corner, local normal, instance,
        // material, and the viewer when it is wave-uniform), so one evaluation per
        // distinct corner serves them all. The wave groups neighbouring lanes with
        // equal keys, compacts the distinct evaluations into LDS (one per lane per
        // pass, ceil(n / 64) passes instead of one pass per corner per path), runs
        // ONE rt_light call site and gathers. Each evaluation is the original
        // expression on the original inputs, and every lane sums its corners in
        // the original order: the colour is bit-identical.
        int nit = 0;                 // 0 none, 1 world point, 3 corners
        float3 A0, A1, A2, B0, B1, B2;   // corners: local positions / local normals; point: P, N
        float3 Vw = make_float3(0.f, 0.f, 0.f);   // point: V
        float w0 = 0.f, w1 = 0.f, w2 = 0.f;
        int k0 = -1, k1 = 0, k2 = 0;  // sharing key (k0 < 0: not shared)
        bool done = false;
        const bool gouraud = !mjw && (shade & RT_SH_GOURAUD) != 0;
        if (lit && g.type == GEOM_MESH && S.tri >= 0 && F.mesh_vert && F.mesh_normal) {
            const int c0 = g.vert_adr + 3 * S.tri;
            const float* Vv = F.mesh_vert + (size_t)c0 * 3;
            const float* Nn = F.mesh_normal + (size_t)c0 * 3;
            // mjr_uploadMesh rule: a corner normal off the face normal fn by more
            // than acos(0.8) is replaced by fn.
            const bool mnrm = !mjw && (shade & RT_SH_MJ_MESHNRM) && (g.flags & RT_F_SRC_NO_UV);
            float3 fn = make_float3(0.f, 0.f, 0.f);
            if (mnrm) {
                const float3 v0 = rt_ld3(Vv), v1 = rt_ld3(Vv + 3), v2 = rt_ld3(Vv + 6);
                fn = rt_norm(rt_cross(rt_sub(v1, v0), rt_sub(v2, v0)));
            }
            const float wb = 1.0f - S.u - S.v;
            float3 n0 = rt_ld3(Nn), n1 = rt_ld3(Nn + 3), n2 = rt_ld3(Nn + 6);
            if (mnrm) {
                if (rt_dot(n0, fn) < 0.8f) n0 = fn;
                if (rt_dot(n1, fn) < 0.8f) n1 = fn;
                if (rt_dot(n2, fn) < 0.8f) n2 = fn;
            }
            if (gouraud) {
                nit = 3;
                A0 = rt_ld3(Vv); A1 = rt_ld3(Vv + 3); A2 = rt_ld3(Vv + 6);
                B0 = n0; B1 = n1; B2 = n2;
                w0 = S.u; w1 = S.v; w2 = wb;
                k0 = S.li * 128; k1 = S.tri; k2 = 0;
            } else {
                const float3 nb = rt_add(rt_add(rt_mul(n0, S.u), rt_mul(n1, S.v)), rt_mul(n2, wb));
                nit = 1;
                A0 = S.P; B0 = rt_norm(rt_mat_vec(I.R, nb)); Vw = rt_mul(rd, -1.0f);
            }
            done = true;
        } else if (lit && !mjw && (shade & RT_SH_VTESS)) {
            RtQuad Q;
            float fa, fb;
            int qa = 0, qb = 0;
            if (rt_vtess_quad(g, S.l, S.nl, Q, fa, fb, qa, qb)) {
                int c0, c1, c2;
                if (fa >= -0.5f) {
                    rt_quad_tri(fa, fb, (shade & RT_SH_QUAD_ALT) != 0, c0, c1, c2, w0, w1, w2);
                } else {
                    // planar quad / fan triangle given only by its corners: split
                    // as rt_quad_tri would and solve the barycentrics in 3D.
                    const bool fan = fa > -1.5f;
                    const bool alt = (shade & RT_SH_QUAD_ALT) != 0;
                    const int a0i = 0, a1i = 1, a2i = (fan || alt) ? 2 : 3;
                    const int b0i = fan ? 0 : (alt ? 0 : 1), b1i = fan ? 1 : 2, b2i = fan ? 2 : 3;
                    float a0, a1;
                    const float3 qa2 = Q.q(a2i);
                    rt_bary2(rt_sub(Q.q(a0i), qa2), rt_sub(Q.q(a1i), qa2), rt_sub(S.l, qa2), a0, a1);
                    const bool inA = fan || (a0 >= -1e-6f && a1 >= -1e-6f && a0 + a1 <= 1.0f + 1e-6f);
                    if (inA) { c0 = a0i; c1 = a1i; c2 = a2i; }
                    else {
                        const float3 qb2 = Q.q(b2i);
                        rt_bary2(rt_sub(Q.q(b0i), qb2), rt_sub(Q.q(b1i), qb2), rt_sub(S.l, qb2), a0, a1);
                        c0 = b0i; c1 = b1i; c2 = b2i;
                    }
                    w0 = a0; w1 = a1; w2 = 1.0f - a0 - a1;
                }
                nit = 3;
                A0 = Q.q(c0); A1 = Q.q(c1); A2 = Q.q(c2);
                B0 = Q.n(c0); B1 = Q.n(c1); B2 = Q.n(c2);
                // the key names the quad AND the corner triple taken from it
                // (a hit far outside its cell can take the planar branch above)
                k0 = (S.li * 2 + 1) * 64 + (c0 | (c1 << 2) | (c2 << 4)); k1 = qa; k2 = qb;
                done = true;
            }
        }
        if (lit && !done) {
            nit = 1;
            A0 = S.P; B0 = rt_norm(rt_mat_vec(I.R, rt_norm(S.nl))); Vw = rt_mul(rd, -1.0f);
        }
        col = rt_light_shared<Sp>(F, hl, I, m, eye, vis, share && vis == ~0u, nit,
                                         A0, A1, A2, B0, B1, B2, Vw, w0, w1, w2, k0, k1, k2);
    } else {
        bool done = false;
        const bool gouraud = !mjw && (shade & RT_SH_GOURAUD) != 0;
        if (g.type == GEOM_MESH && S.tri >= 0 && F.mesh_vert && F.mesh_normal) {
            const int c0 = g.vert_adr + 3 * S.tri;
            const float* Vv = F.mesh_vert + (size_t)c0 * 3;
            const float* Nn = F.mesh_normal + (size_t)c0 * 3;
            // mjr_uploadMesh rule: a corner normal off the face normal fn by more
            // than acos(0.8) is replaced by fn.
            const bool mnrm = !mjw && (shade & RT_SH_MJ_MESHNRM) && (g.flags & RT_F_SRC_NO_UV);
            float3 fn = make_float3(0.f, 0.f, 0.f);
            if (mnrm) {
                const float3 v0 = rt_ld3(Vv), v1 = rt_ld3(Vv + 3), v2 = rt_ld3(Vv + 6);
                fn = rt_norm(rt_cross(rt_sub(v1, v0), rt_sub(v2, v0)));
            }
            const float w2 = 1.0f - S.u - S.v;
            if (gouraud) {
                const float3 p = rt_ld3(I.p);
                col = make_float3(0.f, 0.f, 0.f);
                // corners re-read per iteration (not 18 floats live across the loop)
    #pragma unroll 1
                for (int c = 0; c < 3; ++c) {
                    const float3 vc = rt_ld3(Vv + 3 * c);
                    float3 nc = rt_ld3(Nn + 3 * c);
                    if (mnrm && rt_dot(nc, fn) < 0.8f) nc = fn;
                    const float wc = c == 0 ? S.u : (c == 1 ? S.v : w2);
                    const float3 Pc = rt_add(p, rt_mat_vec(I.R, vc));
                    const float3 fc = rt_light<Sp>(F, hl, Pc, rt_norm(rt_mat_vec(I.R, nc)), rt_norm(rt_sub(eye, Pc)), m, vis);
                    col = rt_add(col, rt_mul(fc, wc));
                }
            } else {
                float3 n0 = rt_ld3(Nn), n1 = rt_ld3(Nn + 3), n2 = rt_ld3(Nn + 6);
                if (mnrm) {
                    if (rt_dot(n0, fn) < 0.8f) n0 = fn;
                    if (rt_dot(n1, fn) < 0.8f) n1 = fn;
                    if (rt_dot(n2, fn) < 0.8f) n2 = fn;
                }
                const float3 nb = rt_add(rt_add(rt_mul(n0, S.u), rt_mul(n1, S.v)), rt_mul(n2, w2));
                const float3 N = rt_norm(rt_mat_vec(I.R, nb));
                col = rt_light<Sp>(F, hl, S.P, N, rt_mul(rd, -1.0f), m, vis);
            }
            done = true;
        } else if (!mjw && (shade & RT_SH_VTESS)) {
            RtQuad Q;
            float fa, fb;
            int qa_, qb_;
            if (rt_vtess_quad(g, S.l, S.nl, Q, fa, fb, qa_, qb_)) {
                int k0, k1, k2; float w0, w1, w2;
                if (fa >= -0.5f) {
                    rt_quad_tri(fa, fb, (shade & RT_SH_QUAD_ALT) != 0, k0, k1, k2, w0, w1, w2);
                } else {
                    // planar quad / fan triangle given only by its corners: split
                    // as rt_quad_tri would and solve the barycentrics in 3D.
                    const bool fan = fa > -1.5f;
                    const bool alt = (shade & RT_SH_QUAD_ALT) != 0;
                    const int a0i = 0, a1i = 1, a2i = (fan || alt) ? 2 : 3;
                    const int b0i = fan ? 0 : (alt ? 0 : 1), b1i = fan ? 1 : 2, b2i = fan ? 2 : 3;
                    float a0, a1;
                    const float3 qa2 = Q.q(a2i);
                    rt_bary2(rt_sub(Q.q(a0i), qa2), rt_sub(Q.q(a1i), qa2), rt_sub(S.l, qa2), a0, a1);
                    const bool inA = fan || (a0 >= -1e-6f && a1 >= -1e-6f && a0 + a1 <= 1.0f + 1e-6f);
                    if (inA) { k0 = a0i; k1 = a1i; k2 = a2i; }
                    else {
                        const float3 qb2 = Q.q(b2i);
                        rt_bary2(rt_sub(Q.q(b0i), qb2), rt_sub(Q.q(b1i), qb2), rt_sub(S.l, qb2), a0, a1);
                        k0 = b0i; k1 = b1i; k2 = b2i;
                    }
                    w0 = a0; w1 = a1; w2 = 1.0f - a0 - a1;
                }
                const float3 p = rt_ld3(I.p);
                const float3 Q0 = Q.q(k0), Q1 = Q.q(k1), Q2 = Q.q(k2);
                const float3 N0 = Q.n(k0), N1 = Q.n(k1), N2 = Q.n(k2);
                col = make_float3(0.f, 0.f, 0.f);
    #pragma unroll 1
                for (int c = 0; c < 3; ++c) {
                    const float3 Qc = c == 0 ? Q0 : (c == 1 ? Q1 : Q2);
                    const float3 Nc = c == 0 ? N0 : (c == 1 ? N1 : N2);
                    const float wc = c == 0 ? w0 : (c == 1 ? w1 : w2);
                    const float3 Pc = rt_add(p, rt_mat_vec(I.R, Qc));
                    const float3 fc = rt_light<Sp>(F, hl, Pc, rt_norm(rt_mat_vec(I.R, Nc)), rt_norm(rt_sub(eye, Pc)), m, vis);
                    col = rt_add(col, rt_mul(fc, wc));
                }
                done = true;
            }
        }
        if (!done) {
            const float3 N = rt_norm(rt_mat_vec(I.R, rt_norm(S.nl)));
            col = rt_light<Sp>(F, hl, S.P, N, rt_mul(rd, -1.0f), m, vis);
        }
    }
    if (textured) col = rt_mul3(col, tex);
    return rt_sat3(col);
}

__device__ __forceinline__ bool rt_q8_pow_ok(float g) { return g >= 0.25f && g <= 1.0f; }
__device__ __forceinline__ float rt_q8_of(float y) { return __builtin_fmaf(rt_sat(y), 255.0f, 0.5f); }
__device__ __forceinline__ float3 rt_lin_q8(float3 col, float e, float g) {
    const float a0 = rt_sat(col.x * e), a1 = rt_sat(col.y * e), a2 = rt_sat(col.z * e);
    auto approx = [&](float a, float& z) -> bool {   // true: z decides Q
        const float ya = __builtin_amdgcn_exp2f(g * __builtin_amdgcn_logf(a));
        z = rt_q8_of(ya);
        const float fr = z - floorf(z);
        const float tol = ya * (255.0f * 8e-6f) + (1.0f / 16384.0f);
        return a >= 1.17549435e-38f && fr >= tol && fr <= 1.0f - tol;
    };
    float z0, z1, z2;
    const bool ok0 = approx(a0, z0), ok1 = approx(a1, z1), ok2 = approx(a2, z2);
    if (!(ok0 && ok1 && ok2)) {
        // rare: exact powf for the undecided channels (one call site)
#pragma unroll 1
        for (int c = 0; c < 3; ++c) {
            const bool need = c == 0 ? !ok0 : (c == 1 ? !ok1 : !ok2);
            if (need) {
                const float a = c == 0 ? a0 : (c == 1 ? a1 : a2);
                const float z = rt_q8_of(__powf(a, g));
                if (c == 0) z0 = z; else if (c == 1) z1 = z; else z2 = z;
            }
        }
    }
    constexpr float r = 1.0f / 255.0f;
    return make_float3((float)(uint32_t)z0 * r, (float)(uint32_t)z1 * r, (float)(uint32_t)z2 * r);
}

template <class Sp>
__device__ __forceinline__ constexpr bool rt_lin_nofog() {
    if constexpr (Sp::kFixed) return Sp::kLinear && !Sp::kFog;
    else return false;
}

// q8: the caller keeps only Q of the result (see rt_lin_q8).
template <class Sp = RtSpecDyn>
__device__ inline float3 rt_post(const RtFrame& F, const RtCam& C, float3 P, float3 col, bool q8 = false) {
    if constexpr (rt_lin_nofog<Sp>()) {
        // compile-time linear preset without fog (raster): only the encode
        if (q8 && rt_q8_pow_ok(F.inv_gamma)) return rt_lin_q8(col, F.exposure, F.inv_gamma);
    }
    if (Sp::linear(F)) {
        col = make_float3(__powf(rt_sat(col.x * F.exposure), F.inv_gamma),
                          __powf(rt_sat(col.y * F.exposure), F.inv_gamma),
                          __powf(rt_sat(col.z * F.exposure), F.inv_gamma));
    }
    if (Sp::fog(F) && F.fog_e > F.fog_s) {
        const float3 fwd = rt_mul(rt_cam_axis(C, 2), -1.0f);
        const float dist = fmaxf(0.0f, rt_dot(rt_sub(P, rt_ld3(C.eye)), fwd));
        const float f = rt_sat((F.fog_e - dist) / (F.fog_e - F.fog_s));
        col = make_float3(f * col.x + (1.0f - f) * F.fog_rgb[0],
                          f * col.y + (1.0f - f) * F.fog_rgb[1],
                          f * col.z + (1.0f - f) * F.fog_rgb[2]);
    }
    return rt_sat3(col);
}

// Background of a miss on sample row sy of an Hn-row sample grid.
__device__ inline float3 rt_background(const RtFrame& F, int sy, int Hn) {
    if (F.grad) {
        const float t = Hn > 1 ? (float)sy / (float)(Hn - 1) : 0.0f;
        return make_float3(F.gbot[0] + t * (F.gtop[0] - F.gbot[0]),
                           F.gbot[1] + t * (F.gtop[1] - F.gbot[1]),
                           F.gbot[2] + t * (F.gtop[2] - F.gbot[2]));
    }
    return make_float3(F.bg[0], F.bg[1], F.bg[2]);
}

// ── compact hits ─────────────────────────────────────────────────────────
// The shading code keeps only (t, li, tri) of a hit and rebuilds the rest
// (barycentrics, normals) from the hit point: this keeps few registers live
// across the secondary traces, and makes the hook independent of what the
// traversal returns beyond t / li / tri.
struct RtCHit { float t; int li, tri; };

template <class Hit>
__device__ __forceinline__ RtCHit rt_compact(const Hit& h) { return RtCHit{h.t, h.li, h.tri}; }

// Analytic local normal of primitive g at (on-surface or nearby) point l:
// the face / patch whose surface is nearest.
__device__ inline float3 rt_prim_normal(const RtGeom& g, float3 l) {
    switch (g.type) {
    case GEOM_PLANE: return make_float3(0.f, 0.f, 1.f);
    case GEOM_SPHERE: return l;
    case RT_GEOM_ELLIPSOID:
        return make_float3(l.x / (g.size[0] * g.size[0]), l.y / (g.size[1] * g.size[1]), l.z / (g.size[2] * g.size[2]));
    case GEOM_BOX: {
        const float dx = fabsf(fabsf(l.x) - g.size[0]);
        const float dy = fabsf(fabsf(l.y) - g.size[1]);
        const float dz = fabsf(fabsf(l.z) - g.size[2]);
        if (dz <= dx && dz <= dy) return make_float3(0.f, 0.f, l.z >= 0.f ? 1.f : -1.f);
        if (dx <= dy) return make_float3(l.x >= 0.f ? 1.f : -1.f, 0.f, 0.f);
        return make_float3(0.f, l.y >= 0.f ? 1.f : -1.f, 0.f);
    }
    case GEOM_CYLINDER: {
        const float side = fabsf(sqrtf(l.x * l.x + l.y * l.y) - g.size[0]);
        const float cap = fabsf(fabsf(l.z) - g.size[1]);
        if (cap < side) return make_float3(0.f, 0.f, l.z >= 0.f ? 1.f : -1.f);
        return make_float3(l.x, l.y, 0.f);
    }
    case GEOM_CAPSULE: {
        const float h = g.size[1];
        if (fabsf(l.z) <= h) return make_float3(l.x, l.y, 0.f);
        return make_float3(l.x, l.y, l.z - (l.z > 0.f ? h : -h));
    }
    default: return make_float3(0.f, 0.f, 1.f);
    }
}

// Small integer naming the flat patch a primitive hit lies on (MSAA grouping:
// GL shades one fragment per primitive covering the pixel).
__device__ inline int rt_face_code(const RtGeom& g, float3 nl) {
    if (g.type == GEOM_BOX) {
        const float ax = fabsf(nl.x), ay = fabsf(nl.y), az = fabsf(nl.z);
        if (az >= ax && az >= ay) return nl.z > 0.f ? 4 : 5;
        if (ax >= ay) return nl.x > 0.f ? 0 : 1;
        return nl.y > 0.f ? 2 : 3;
    }
    if (g.type == GEOM_CYLINDER) return nl.z > 0.5f ? 1 : (nl.z < -0.5f ? 2 : 0);
    return 0;
}

// Rebuild the surface of compact hit h (h.li >= 0) at world point P.
__device__ inline void rt_surf_at(const RtFrame& F, const RtInst& I, const RtGeom& g, int li, int tri,
                                  float3 P, RtSurf& S) {
    S.li = li; S.tri = tri; S.u = S.v = 0.0f;
    S.P = P;
    S.l = rt_mat_tvec(I.R, rt_sub(P, rt_ld3(I.p)));
    if (g.type == GEOM_MESH && tri >= 0 && F.mesh_vert) {
        const float* Vv = F.mesh_vert + (size_t)(g.vert_adr + 3 * tri) * 3;
        const float3 v0 = rt_ld3(Vv), v1 = rt_ld3(Vv + 3), v2 = rt_ld3(Vv + 6);
        rt_bary2(rt_sub(v0, v2), rt_sub(v1, v2), rt_sub(S.l, v2), S.u, S.v);
        S.nl = rt_cross(rt_sub(v1, v0), rt_sub(v2, v0));
    } else {
        S.nl = rt_prim_normal(g, S.l);
    }
    S.Ng = rt_norm(rt_mat_vec(I.R, rt_norm(S.nl)));
}

// Mirror ray of a primary hit on the +Z face of the reflective geom.
__device__ inline bool rt_mirror_ray(const RtFrame& F, const RtSurf& S, float3 rd, float3& o, float3& r) {
    if (S.li != F.mirror_li || !(S.nl.z > 0.5f)) return false;
    const float3 N = S.Ng;
    o = rt_add(S.P, rt_mul(N, 1e-4f));
    r = rt_sub(rd, rt_mul(N, 2.0f * rt_dot(rd, N)));
    return true;
}

// Colour of the reflected hit h2 (MuJoCo renderGeomReflection): rgb *
// reflectance before lighting, specular unscaled, lights in world space,
// black where the reflected ray misses (the stencil region keeps the clear
// colour). The texture footprint continues the primary differential through
// the planar mirror: from the virtual eye the cone grows by (t1 + t2) / t1.
__device__ inline float3 rt_mirror_color(const RtFrame& F, const RtCam& C, const RtInst* __restrict__ inst,
                                         const RtSurf& S, float3 o, float3 r, RtCHit h2) {
    const RtInst& I2 = inst[h2.li];
    const RtGeom& g2 = F.geom[h2.li];
    RtSurf S2;
    rt_surf_at(F, I2, g2, h2.li, h2.tri, rt_add(o, rt_mul(r, h2.t)), S2);
    const float3 N = S.Ng;
    const float3 rx = rt_sub(S.dPx, rt_mul(N, 2.0f * rt_dot(S.dPx, N)));
    const float3 ry = rt_sub(S.dPy, rt_mul(N, 2.0f * rt_dot(S.dPy, N)));
    float rn = rt_dot(r, S2.Ng);
    if (fabsf(rn) < 1e-6f) rn = rn < 0.f ? -1e-6f : 1e-6f;
    const float3 cam = rt_ld3(C.eye);
    const float tr = h2.t / fmaxf(rt_len(rt_sub(S.P, cam)), 1e-6f);
    const float3 gx = rt_add(rx, rt_mul(rx, tr)), gy = rt_add(ry, rt_mul(ry, tr));
    S2.dPx = rt_sub(gx, rt_mul(r, rt_dot(gx, S2.Ng) / rn));
    S2.dPy = rt_sub(gy, rt_mul(r, rt_dot(gy, S2.Ng) / rn));
    // virtual viewer: the camera mirrored in the mirror plane (through S.P)
    const float3 veye = rt_sub(cam, rt_mul(N, 2.0f * rt_dot(rt_sub(cam, S.P), N)));
    return rt_shade_surface(F, C, I2, g2, S2, veye, r, F.geom[S.li].refl);
}

// MuJoCo transparency (mjr_render): geoms with alpha < 0.995 are drawn after
// the opaques, sorted by camdist = |centre - head| - rbound (planes last),
// front to back and then back to front, SRC_ALPHA / ONE_MINUS_SRC_ALPHA into
// the 8-bit framebuffer (no depth writes, back faces culled). Up to
// RT_MAX_TLAYERS layers in front of the first opaque surface.
constexpr int RT_MAX_TLAYERS = 4;

__device__ __forceinline__ uint32_t rt_mul8(uint32_t a, uint32_t b) {   // round(a*b/255)
    const uint32_t t = a * b + 128u;
    return (t + (t >> 8)) >> 8;
}

__device__ __forceinline__ uint32_t rt_q8(float c) { return (uint32_t)(rt_sat(c) * 255.0f + 0.5f); }
__device__ __forceinline__ uint32_t rt_pack8(float3 c) { return rt_q8(c.x) | (rt_q8(c.y) << 8) | (rt_q8(c.z) << 16); }

__device__ inline float rt_rbound(const RtGeom& g) {
    float r2 = 0.0f;
    for (int k = 0; k < 3; ++k) { const float e = fmaxf(fabsf(g.llo[k]), fabsf(g.lhi[k])); r2 += e * e; }
    return sqrtf(r2);
}

// Colour of compact hit h seen along unit rd from eye (no mirror / layers).
__device__ inline float3 rt_shade_chit(const RtFrame& F, const RtCam& C, const RtInst* __restrict__ inst,
                                       RtCHit h, float3 eye, float3 rd, float cz, float3 dDx, float3 dDy,
                                       RtSurf& S) {
    const RtInst& I = inst[h.li];
    const RtGeom& g = F.geom[h.li];
    rt_surf_at(F, I, g, h.li, h.tri, rt_add(eye, rt_mul(rd, h.t)), S);
    rt_ray_diff(rd, h.t, cz, S.Ng, dDx, dDy, S.dPx, S.dPy);
    return rt_shade_surface(F, C, I, g, S, eye, rd, 0.0f);
}

// MSAA4 centre shading of a hit hs found along the sample ray rs: the distance
// along the shading ray rd to hs's tangent plane (the primary hit's rule in
// rt_primary_kernel), or hs.t where that plane is grazing / behind.
__device__ inline float rt_tangent_t(const RtFrame& F, const RtInst* __restrict__ inst, const RtCHit& hs,
                                     float3 eye, float3 rs, float3 rd) {
    RtSurf S;
    rt_surf_at(F, inst[hs.li], F.geom[hs.li], hs.li, hs.tri, rt_add(eye, rt_mul(rs, hs.t)), S);
    const float den = rt_dot(rd, S.Ng);
    const float tc = den < -1e-6f ? hs.t * rt_dot(rs, S.Ng) / den : -1.0f;
    return tc > 0.0f ? tc : hs.t;
}

constexpr int RT_SKIP_TLAYERS = 0;

// Secondary rays of one primary sample (MuJoCo transparency layers, planar
// mirror), traced before any shading state is live:
//   layers: geoms with alpha < 0.995 in front of the first opaque surface, up
//     to RT_MAX_TLAYERS (further ones skipped), found by continuing the
//     SAMPLE's own ray rs from its hit distance ts (trace_eye, up to tmaxs);
//     store_layer(q, hit) receives the q-th transparent hit (front to back);
//   mirror: the reflected ray of a hit on the +Z face of the reflective geom
//     (trace_any, an arbitrary world ray) along the shading ray rd;
//     store_mirror(hit) receives its hit.
// MSAA4 shades at the pixel centre: when ts != h.t, h.t (on rd) is the
// primary hit's tangent-plane distance and every layer / the surface behind
// them is re-expressed on rd the same way (rt_tangent_t). Otherwise rs == rd
// and ts == h.t. Continuing rd from the tangent point instead would re-find a
// curved front face as a second layer.
// h is updated to the first opaque hit (or the hit after the last layer) and
// nl to the layer count. trace_eye / trace_any are called by every lane of
// the wave the same number of times (lanes without a ray pass tmax < tmin).
template <class Sp = RtSpecDyn, class TraceEye, class TraceAny, class StoreLayer, class StoreMirror>
__device__ __attribute__((always_inline)) inline void rt_trace_secondaries(
        const RtFrame& F, const RtInst* __restrict__ inst, RtCHit& h, int& nl, bool active, float3 eye,
        float3 rd, float tmax, float3 rs, float ts, float tmaxs,
        TraceEye&& trace_eye, TraceAny&& trace_any, StoreLayer&& store_layer,
        StoreMirror&& store_mirror) {
    nl = 0;
    if (Sp::lay(F) && Sp::mode(F) == RT_MODE_HIPGL) {
        const bool conv = ts != h.t;   // h.t was moved to the shading ray's tangent plane
        RtCHit hs = h;
        hs.t = ts;
        bool moved = false;
#pragma unroll 1
        for (int it = 0; it < RT_MAX_TLAYERS + RT_SKIP_TLAYERS; ++it) {
            const bool tr = active && hs.li >= 0 && (F.geom[hs.li].flags & RT_F_TRANSPARENT);
            if (__ballot(tr) == 0ull) break;
            if (tr && it < RT_MAX_TLAYERS) {
                RtCHit hl = hs;
                if (conv) hl.t = rt_tangent_t(F, inst, hs, eye, rs, rd);
                store_layer(it, hl);
                nl = it + 1;
            }
            const RtCHit hn = trace_eye(rs, tr ? hs.t * (1.0f + 1e-6f) + 1e-6f : 0.0f, tr ? tmaxs : -1.0f);
            if (tr) { hs = hn; moved = true; }
        }
        if (moved) {
            h = hs;
            if (hs.li < 0) h.t = tmax;
            else if (conv) h.t = rt_tangent_t(F, inst, hs, eye, rs, rd);
        }
    }
    if (F.mirror_li >= 0 && (Sp::shade(F) & RT_SH_REFLECT) && Sp::mode(F) == RT_MODE_HIPGL) {
        const bool hit = active && h.li >= 0;
        float3 o = eye, r = rd;
        bool on = false;
        if (hit && h.li == F.mirror_li) {
            RtSurf S;
            rt_surf_at(F, inst[h.li], F.geom[h.li], h.li, h.tri, rt_add(eye, rt_mul(rd, h.t)), S);
            on = rt_mirror_ray(F, S, rd, o, r);
        }
        if (__ballot(on) != 0ull) {
            const RtCHit h2 = trace_any(o, r, 0.0f, on ? RT_MAXVAL : -1.0f);
            if (on) store_mirror(h2);
        }
    }
}

// One primary sample's colour (background on a miss), from its first opaque
// hit h, its nl transparent layers (layer_at(q), front to back) and its
// mirror-ray hit (mirror_hit(), only asked when the sample has a mirror ray),
// as found by rt_trace_secondaries. All surfaces go through ONE inlined
// rt_shade_surface call site (a wave-uniform job loop: 0 = opaque surface,
// 1 = mirror reflection, 2.. = layers), so only the accumulators stay live
// across it. `active == false` lanes return black.
// Surfaces and arithmetic are MuJoCo's (mjr_render):
//   opaque: lit + clamped mirror term (renderGeomReflection: rgb x
//     reflectance, viewer = camera mirrored in the mirror plane, texture
//     footprint continued through the mirror), fog / linear output;
//   layers: sorted by camdist = |centre - head| - rbound (planes last), blended
//     front to back and then back to front with SRC_ALPHA in 8 bits.
template <class Sp = RtSpecDyn, class LayerAt, class MirrorHit>
__device__ __attribute__((always_inline)) inline float3 rt_shade_full(const RtFrame& F, const RtCam& C, const RtInst* __restrict__ inst,
                                         RtCHit h, int nl, bool active, float3 eye, float3 rd, float cz,
                                         float3 dDx, float3 dDy, int sy, int Hn,
                                         LayerAt&& layer_at, MirrorHit&& mirror_hit, unsigned vis = ~0u) {
    const bool hit = active && h.li >= 0;
    const bool mir = F.mirror_li >= 0 && (Sp::shade(F) & RT_SH_REFLECT) && Sp::mode(F) == RT_MODE_HIPGL;
    const bool lay = Sp::lay(F) && Sp::mode(F) == RT_MODE_HIPGL;
    float3 col = make_float3(0.f, 0.f, 0.f);
    if (!hit && active) col = rt_background(F, sy, Hn);
    uint32_t lc0 = 0u, lc1 = 0u, lc2 = 0u, lc3 = 0u;
    float lk0 = 3e38f, lk1 = 3e38f, lk2 = 3e38f, lk3 = 3e38f;
    const int njob = lay ? 2 + RT_MAX_TLAYERS : (mir ? 2 : 1);
    // Only col and the layer accumulators are carried across jobs: the mirror
    // job re-derives what it needs from h (a loop-carried value would be live
    // across every iteration's shading).
#pragma unroll 1
    for (int j = 0; j < njob; ++j) {
        bool want;
        RtCHit hj = h;
        RtSurf S;
        float3 ve = eye, dir = rd;
        float refl = 0.0f;
        if (j == 1) {
            // rt_mirror_color: the reflected hit seen from the virtual viewer
            // (the camera mirrored in the mirror plane through the hit).
            want = hit && mir && h.li == F.mirror_li;
            if (__ballot(want) == 0ull) continue;
            if (want) {
                rt_surf_at(F, inst[h.li], F.geom[h.li], h.li, h.tri, rt_add(eye, rt_mul(rd, h.t)), S);
                rt_ray_diff(rd, h.t, cz, S.Ng, dDx, dDy, S.dPx, S.dPy);
                float3 o, r;
                want = rt_mirror_ray(F, S, rd, o, r);
                if (want) hj = mirror_hit();
                want = want && hj.li >= 0;
                if (want) {
                    const float3 N = S.Ng;
                    const float3 rx = rt_sub(S.dPx, rt_mul(N, 2.0f * rt_dot(S.dPx, N)));
                    const float3 ry = rt_sub(S.dPy, rt_mul(N, 2.0f * rt_dot(S.dPy, N)));
                    const float3 cam = rt_ld3(C.eye);
                    const float tr = hj.t / fmaxf(rt_len(rt_sub(S.P, cam)), 1e-6f);
                    ve = rt_sub(cam, rt_mul(N, 2.0f * rt_dot(rt_sub(cam, S.P), N)));
                    dir = r;
                    const float3 gx = rt_add(rx, rt_mul(rx, tr)), gy = rt_add(ry, rt_mul(ry, tr));
                    const RtInst& I2 = inst[hj.li];
                    rt_surf_at(F, I2, F.geom[hj.li], hj.li, hj.tri, rt_add(o, rt_mul(r, hj.t)), S);
                    float rn = rt_dot(r, S.Ng);
                    if (fabsf(rn) < 1e-6f) rn = rn < 0.f ? -1e-6f : 1e-6f;
                    S.dPx = rt_sub(gx, rt_mul(r, rt_dot(gx, S.Ng) / rn));
                    S.dPy = rt_sub(gy, rt_mul(r, rt_dot(gy, S.Ng) / rn));
                    refl = F.geom[F.mirror_li].refl;
                }
            }
            if (__ballot(want) == 0ull) continue;
        } else {
            if (j == 0) want = hit;
            else { want = j - 2 < nl; if (want) hj = layer_at(j - 2); }
            if (__ballot(want) == 0ull) { if (j >= 2) break; continue; }
            if (want) {
                rt_surf_at(F, inst[hj.li], F.geom[hj.li], hj.li, hj.tri, rt_add(eye, rt_mul(rd, hj.t)), S);
                rt_ray_diff(rd, hj.t, cz, S.Ng, dDx, dDy, S.dPx, S.dPy);
            }
        }
        if (!want) { hj.li = 0; hj.tri = -1; S = RtSurf{}; S.li = 0; S.tri = -1; }   // idle lane: any valid geom
        const RtInst& I = inst[hj.li];
        const RtGeom& g = F.geom[hj.li];
        const float3 c = rt_shade_surface<Sp>(F, C, I, g, S, ve, dir, refl, j == 0 ? vis : ~0u, want, j != 1);
        if (!want) continue;
        if (j == 0) {
            col = c;
        } else if (j == 1) {
            col = rt_sat3(rt_add(col, c));
        } else {
            const float3 cl = rt_post<Sp>(F, C, S.P, c, true);   // quantised by rt_pack8
            const uint32_t lcq = rt_pack8(cl) | (rt_q8(g.rgba[3]) << 24);
            const float lkq = g.type == GEOM_PLANE ? 1e10f : rt_len(rt_sub(rt_ld3(I.p), eye)) - rt_rbound(g);
            const int q = j - 2;
            if (q == 0) { lc0 = lcq; lk0 = lkq; } else if (q == 1) { lc1 = lcq; lk1 = lkq; }
            else if (q == 2) { lc2 = lcq; lk2 = lkq; } else { lc3 = lcq; lk3 = lkq; }
        }
    }
    const float3 Pp = rt_add(eye, rt_mul(rd, h.t));
    // Hn == F.H: one sample per output pixel row (SSAA 1 / MSAA4): every
    // caller quantises the colour to 8 bits (rt_store_rgb / rt_pack8 / rt_q8).
    if (hit) col = rt_post<Sp>(F, C, Pp, col, Hn == F.H);
    // composite the layers (8-bit framebuffer blends)
    if (nl > 0) {
        // sort by camdist: 4-input network
        auto cs = [](float& ka, float& kb, uint32_t& ca, uint32_t& cb) {
            const bool sw = kb < ka;
            const float a0 = ka, b0 = kb;
            const uint32_t c0 = ca, c1 = cb;
            ka = sw ? b0 : a0; kb = sw ? a0 : b0;
            ca = sw ? c1 : c0; cb = sw ? c0 : c1;
        };
        cs(lk0, lk1, lc0, lc1); cs(lk2, lk3, lc2, lc3); cs(lk0, lk2, lc0, lc2); cs(lk1, lk3, lc1, lc3); cs(lk1, lk2, lc1, lc2);
        uint32_t c[3] = {rt_q8(col.x), rt_q8(col.y), rt_q8(col.z)};
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
#pragma unroll
            for (int q = 0; q < RT_MAX_TLAYERS; ++q) {
                const int i = pass == 0 ? q : RT_MAX_TLAYERS - 1 - q;
                if (i >= nl) continue;
                const uint32_t L = i == 0 ? lc0 : i == 1 ? lc1 : i == 2 ? lc2 : lc3;
                const uint32_t A = L >> 24;
                for (int k = 0; k < 3; ++k)
                    c[k] = min(255u, rt_mul8((L >> (8 * k)) & 255u, A) + rt_mul8(c[k], 255u - A));
            }
        }
        col = make_float3(c[0] * (1.0f / 255.0f), c[1] * (1.0f / 255.0f), c[2] * (1.0f / 255.0f));
    }
    return col;
}

// Camera-space unit direction through fractional sample position (fx, fy) on
// a Wn x Hn grid (fx = sx + 0.5 for the centre of integer sample sx):
//   dl = normalize((2 fx / Wn - 1) tanh W/H, (1 - 2 fy / Hn) tanh, -1).
__device__ __forceinline__ float3 rt_dir_at(float fx, float fy, int Wn, int Hn, float aspect, float tanh_) {
    const float u = fx / (float)Wn;
    const float v = fy / (float)Hn;
    return rt_norm(make_float3((2.0f * u - 1.0f) * tanh_ * aspect, (1.0f - 2.0f * v) * tanh_, -1.0f));
}

// MSAA4 sample offsets inside a pixel, image rows top-down: GL's standard 4x
// pattern (0.375,0.125) (0.875,0.375) (0.125,0.625) (0.625,0.875) in GL's
// bottom-up window coordinates, i.e. the raster's msaa_rasterize.inc table.
__device__ __forceinline__ float rt_msaa_x(int k) { return k == 0 ? 0.375f : k == 1 ? 0.875f : k == 2 ? 0.125f : 0.625f; }
__device__ __forceinline__ float rt_msaa_y(int k) { return k == 0 ? 0.875f : k == 1 ? 0.625f : k == 2 ? 0.375f : 0.125f; }

}  // namespace physics
}  // namespace hipgl
