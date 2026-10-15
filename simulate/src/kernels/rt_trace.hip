// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifdef RT_PARITY_FP
#pragma clang fp contract(off)
#else
#pragma clang fp contract(on)
#endif

#include <hip/hip_runtime.h>
#include <algorithm>
#include <climits>
#include "hipgl/physics/ray_tracer.h"
#include "rt_traverse.inc"
#include "rt_frustum.inc"
#include "rt_shade.inc"

namespace hipgl {
namespace physics {

namespace {

// Candidate list capacity per wave (LDS words). Must be a multiple of 64.
constexpr int RT_CAND_CAP = 128;
static_assert(RT_CAND_CAP % 64 == 0 && RT_CAND_CAP >= 64, "RT_CAND_CAP");
// Accelerator dies (XCDs) that consecutive workgroups are dealt to round robin
// (MI300 / MI355). The swizzle keeps a world's tiles on one XCD's L2.
constexpr unsigned RT_NUM_XCD = 8;

// Resolved hit, as the shading hook sees it.
struct RtHit {
    float  t;      // ray parameter (rd is unit length, so metres)
    int    li;     // render-geom list index, -1 = miss
    int    tri;    // mesh-local triangle, -1 for primitives
    float  u, v;   // barycentrics (u = weight of corner 0, v = corner 1)
    float3 nl;     // local-frame normal (unnormalised for meshes)
};

// Traversal-time hit record: only what the (t, li) merge needs. ref = global
// RtTri index of a mesh hit in the active build's leaf order, -1 otherwise.
struct RtHitRec { float t; int li; int ref; };

// cand: instance-slab survivors summed over rays; cand_tile: instances whose
// slab test passed for at least one lane of the wave (counted on lane 0 once
// per wave-wide ray pass, i.e. once per tile per sample); list: frustum
// survivors (list entries) and visited: entries walked before the early break,
// both per wave-wide ray pass (lane 0).
struct RtCounters { unsigned nodes = 0, tris = 0, cand = 0, cand_tile = 0, list = 0, visited = 0; };

__device__ __forceinline__ float3 rt_sample_dir(int sx, int sy, int Wn, int Hn,
                                                float aspect, float tanh_)
{
    const float u = ((float)sx + 0.5f) / (float)Wn;
    const float v = ((float)sy + 0.5f) / (float)Hn;
    return rt_norm(make_float3((2.0f * u - 1.0f) * tanh_ * aspect,
                               (1.0f - 2.0f * v) * tanh_,
                               -1.0f));
}

// Per-lane ORDERED stack the primary kernel hands to the walker.
#if defined(__HIP_DEVICE_COMPILE__)
using RtPrimaryStack = RtLdsStack;
#else
using RtPrimaryStack = RtRegStack;
#endif

// Wave-uniform read-only load through the constant address space: with a
// wave-uniform address it becomes an s_load (SGPRs, scalar cache) instead of a
// vector load that replicates the same record into every lane's VGPRs. Only
// for data the kernel never writes (instances, geoms, BLAS headers).
template <typename T>
__device__ __forceinline__ T rt_sload(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    return *(const __attribute__((address_space(4))) T*)p;
#else
    return *p;
#endif
}

// A wave-uniform pointer the optimiser cannot see through (an LICM barrier):
// readfirstlane pins it to SGPRs (a no-op when it already is one, e.g. after
// a call it may sit in a VGPR), then an empty asm hides its value.
template <typename T>
__device__ __forceinline__ const T* rt_opaque_uniform(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__)
    const unsigned long long v = (unsigned long long)p;
    unsigned lo = __builtin_amdgcn_readfirstlane((unsigned)v);
    unsigned hi = __builtin_amdgcn_readfirstlane((unsigned)(v >> 32));
    asm volatile("" : "+s"(lo), "+s"(hi));
    return (const T*)(((unsigned long long)hi << 32) | lo);
#else
    return p;
#endif
}

template <int TRAV, bool STATS, typename Stack>
__device__ __forceinline__ void rt_offer(const RtFrame& F, const RtInst& I, int li,
                                         float3 o, float3 d, float3 iv, float tmin,
                                         RtHitRec& h, Stack& st, RtCounters& cnt)
{
    float tn;
    const bool pass = rt_slab(o, iv, make_float3(I.lo[0], I.lo[1], I.lo[2]),
                              make_float3(I.hi[0], I.hi[1], I.hi[2]), tmin, h.t, tn);
    if (STATS) {
        if (pass) cnt.cand++;
        if (__ballot(pass) != 0ull && (threadIdx.x & 63) == 0) cnt.cand_tile++;
    }
    if (!pass) return;
    const bool allow_eq = h.li >= 0 && li < h.li;
    const RtGeom* g = F.geom + li;
    const int gtype = rt_sload(&g->type);
    const float3 p  = make_float3(I.p[0], I.p[1], I.p[2]);
    const float3 lp = rt_mat_tvec(I.R, rt_sub(o, p));
    const float3 lv = rt_mat_tvec(I.R, d);
    const int cull = F.cull;

    if (gtype != GEOM_MESH) {
        float3 ln;
        const float size[3] = {rt_sload(&g->size[0]), rt_sload(&g->size[1]), rt_sload(&g->size[2])};
        const float t = rt_hit_prim(gtype, lp, lv, size, tmin, ln);
        if (t < tmin || !(t < h.t || (allow_eq && t == h.t))) return;
        // Primitive back-face cull (mjw render.py:1197): any cull mode but NONE.
        if (cull != RT_CULL_NONE && rt_dot(lv, ln) > 0.0f) return;
        h.t = t; h.li = li; h.ref = -1;
    } else {
        const int bi = rt_sload(&g->blas);
        if (bi < 0) return;
        const RtBlas b = rt_sload(F.blas + bi);
        if (b.ntri <= 0) return;
        // Walk on a copy of the bound: under MJW_MESH a back-facing closest
        // triangle discards the whole mesh and must leave h.t untouched.
        float bt = h.t, u = 0.f, v = 0.f;
        int tri = -1, ref = -1;
        float3 ng;
        bool hit;
        if (TRAV == RT_TRAV_SKIP)
            hit = rt_blas_walk_skip_s<false>(F.n32, F.tri, b, lp, lv, tmin, cull, allow_eq,
                                             bt, tri, ref, u, v, ng, cnt.nodes, cnt.tris);
        else
#ifdef RT_ORDERED_BVH2
            hit = rt_blas_walk_ordered_s<false>(F.n64, F.tri64, b, lp, lv, tmin, cull, allow_eq, st,
                                                bt, tri, ref, u, v, ng, cnt.nodes, cnt.tris);
#else
            // ORDERED walks the BVH4 collapse (same tri64 leaves, same result).
            hit = rt_blas_walk_wide_s<false>(F.n4, F.tri64, b, lp, lv, tmin, cull, allow_eq, st,
                                             bt, tri, ref, u, v, ng, cnt.nodes, cnt.tris);
#endif
        if (!hit || bt < tmin) return;
        if (cull == RT_CULL_MJW_MESH) {
            // Whole-mesh back-face cull on the closest triangle. Its normal is
            // re-derived from the recorded triangle (the same cross product the
            // walk computed) rather than carried through the walk.
            const RtTri T = (TRAV == RT_TRAV_SKIP ? F.tri : F.tri64)[ref];
            const float3 v2 = rt_trav_xyz(T.v2);
            if (rt_dot(lv, rt_cross(rt_sub(rt_trav_xyz(T.v0), v2), rt_sub(rt_trav_xyz(T.v1), v2))) > 0.0f) return;
        }
        h.t = bt; h.li = li; h.ref = ref;
    }
}

// Re-derive the full hit (u, v, mesh-local triangle, local normal) of the
// winning instance with the arithmetic the traversal used, so it is the same
// result bit for bit: the primitive intersector again, or rt_tri_mjw on the
// recorded triangle with the same local ray and basis.
template <int TRAV>
__device__ __forceinline__ RtHit rt_resolve_hit(const RtFrame& F, const RtInst* __restrict__ inst,
                                                const RtHitRec& r, float3 o, float3 d, float tmin, float tmax)
{
    RtHit h;
    h.t = r.li >= 0 ? r.t : tmax; h.li = r.li; h.tri = -1; h.u = h.v = 0.0f;
    h.nl = make_float3(0.f, 0.f, 0.f);
    if (r.li < 0) return h;
    const RtInst& I = inst[r.li];
    const float3 p  = make_float3(I.p[0], I.p[1], I.p[2]);
    const float3 lp = rt_mat_tvec(I.R, rt_sub(o, p));
    const float3 lv = rt_mat_tvec(I.R, d);
    if (r.ref < 0) {
        const RtGeom& g = F.geom[r.li];
        float3 ln = make_float3(0.f, 0.f, 0.f);
        (void)rt_hit_prim(g.type, lp, lv, g.size, tmin, ln);
        h.nl = ln;
    } else {
        const RtTri T = (TRAV == RT_TRAV_SKIP ? F.tri : F.tri64)[r.ref];
        float3 b0, b1;
        rt_tri_basis(lv, b0, b1);
        float t, u, v; float3 n;
        (void)rt_tri_mjw(lp, lv, b0, b1, rt_trav_xyz(T.v0), rt_trav_xyz(T.v1), rt_trav_xyz(T.v2),
                         tmin, t, u, v, n);
        h.tri = rt_trav_f2i(T.v0.w); h.u = u; h.v = v; h.nl = n;
    }
    return h;
}

template <int TRAV, bool STATS, typename Stack>
__device__ __forceinline__ void rt_closest_linear(const RtFrame& F, const RtInst* __restrict__ inst,
                                                  float3 o, float3 d, float tmin, float tmax,
                                                  RtHitRec& h, Stack& st, RtCounters& cnt)
{
    h.t = tmax; h.li = -1; h.ref = -1;
    const float3 iv = rt_safe_inv(d);
    for (int li = 0; li < F.nrg; ++li)
        rt_offer<TRAV, STATS>(F, inst[li], li, o, d, iv, tmin, h, st, cnt);
    if (h.li < 0) h.t = tmax;
}

// Unit world normal of a hit (R * normalize(local)).
__device__ __forceinline__ float3 rt_world_normal(const RtInst& I, float3 nl) {
    return rt_norm(rt_mat_vec(I.R, rt_norm(nl)));
}

__device__ __forceinline__ float rt_clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }

// Camera-space ray of sample (sx, sy) on the (Wn x Hn) grid, mapped to world,
// with the clip interval. cz = -dl.z (> 0) converts t to planar depth.
__device__ __forceinline__ void rt_cam_ray(const RtFrame& F, const RtCam& C, int sx, int sy,
                                           int Wn, int Hn, float aspect,
                                           float3& rd, float& cz, float& tmin, float& tmax)
{
    const float3 dl = rt_sample_dir(sx, sy, Wn, Hn, aspect, C.tanh_);
    rd = rt_mat_vec(C.Rc, dl);
    cz = -dl.z;
    tmin = 0.0f; tmax = RT_MAXVAL;
    if (F.clip) { tmin = F.znear / cz; tmax = F.zfar / cz; }
}

struct RtSample {
    float3 eye, rd; float cz, tmax;
    int sy, Hn;          // background row / rows
    int lw, lh;          // LOD footprint grid (pixels or SSAA samples)
    bool active;         // false: MSAA sample whose surface an earlier sample shades
};

#ifdef RT_SHADE_NOINLINE
#define RT_SHADE_ATTR __attribute__((noinline))
#else
#define RT_SHADE_ATTR __forceinline__
#endif

__device__ RT_SHADE_ATTR float3 rt_shade_grey(const RtFrame& F, const RtInst* __restrict__ inst,
                                              const RtHit& h, const RtSample& s)
{
    if (h.li < 0) return rt_background(F, s.sy, s.Hn);
    const float3 N = rt_world_normal(inst[h.li], h.nl);
    const float3 V = rt_mul(s.rd, -1.0f);
    const float g = fabsf(rt_dot(N, V));
    return make_float3(g, g, g);
}

// h: first opaque hit, nl: transparent layers in front of it (layer_at(q)),
// mirror_hit(): hit of the sample's mirror ray (rt_trace_secondaries).
template <class Sp = RtSpecDyn, class LayerAt, class MirrorHit>
__device__ __forceinline__ float3 rt_shade_sample(const RtFrame& F, const RtCam& C,
                                                  const RtInst* __restrict__ inst,
                                                  const RtCHit& h, int nl, const RtSample& s,
                                                  LayerAt&& layer_at, MirrorHit&& mirror_hit, unsigned vis = ~0u)
{
    const float aspect = (float)F.W / (float)F.H;
    const float3 dDx = rt_mul(rt_cam_axis(C, 0), 2.0f * C.tanh_ * aspect / (float)s.lw);
    const float3 dDy = rt_mul(rt_cam_axis(C, 1), -2.0f * C.tanh_ / (float)s.lh);
    return rt_shade_full<Sp>(F, C, inst, h, nl, s.active, s.eye, s.rd, s.cz, dDx, dDy, s.sy, s.Hn,
                         layer_at, mirror_hit, vis);
}

// Fused path: trace the secondaries, then shade. The layer records go to a
// per-lane LDS slot array (lay = base + lane, stride 64 words, 3 words x
// RT_MAX_TLAYERS): indexed by the runtime layer number, registers would be
// lowered to a scratch array.
constexpr int RT_LAY_LDS = 3 * RT_MAX_TLAYERS * 64;
// RT_SH_SHADOW (experimental): visibility bit per light of the sample's first
// opaque hit h. Bit i clear: scene light i (directional / spot) is occluded or
// lies behind the viewed side of the surface. The ray starts 1 mm off the
// surface on the viewer's side; trace_shadow(o, d, t1) is called by every lane
// the same number of times (lanes without a ray pass t1 < 0).
template <class TraceShadow>
__device__ __forceinline__ unsigned rt_shadow_mask(const RtFrame& F, const RtInst* __restrict__ inst,
                                                   const RtCHit& h, const RtSample& s,
                                                   TraceShadow&& trace_shadow)
{
    unsigned vis = ~0u;
    const bool hit = s.active && h.li >= 0;
    const float3 P = rt_add(s.eye, rt_mul(s.rd, h.t));
    float3 nv = make_float3(0.f, 0.f, 1.f);
    if (hit) {
        RtSurf S;
        rt_surf_at(F, inst[h.li], F.geom[h.li], h.li, h.tri, P, S);
        nv = rt_dot(S.Ng, s.rd) > 0.0f ? rt_mul(S.Ng, -1.0f) : S.Ng;
    }
    const float3 o = rt_add(P, rt_mul(nv, 1e-3f));
#pragma unroll 1
    for (int i = 1; i < F.nlight; ++i) {
        const RtLight L = rt_sh_sload(F.lights + i);
        if (!L.active || L.type == 2) continue;   // point lights: MuJoCo's eye-direction quirk, unshadowed
        float3 Ld; float dist;
        if (L.type == 1) {
            Ld = rt_sub(rt_ld3(L.pos), P);
            dist = rt_len(Ld);
            Ld = rt_mul(Ld, 1.0f / fmaxf(dist, 1e-8f));
            dist -= 1e-3f;
        } else {
            Ld = make_float3(-L.dir[0], -L.dir[1], -L.dir[2]);
            dist = RT_MAXVAL;
        }
        const bool want = hit && rt_dot(nv, Ld) > 0.0f;
        if (hit && !want) vis &= ~(1u << i);
        if (trace_shadow(o, Ld, want ? dist : -1.0f) && want) vis &= ~(1u << i);
    }
    return vis;
}

template <class Sp = RtSpecDyn, class TraceEye, class TraceAny, class TraceShadow>
__device__ __forceinline__ float3 rt_shade_sample_traced(const RtFrame& F, const RtCam& C,
                                                         const RtInst* __restrict__ inst,
                                                         RtCHit h, const RtSample& s, unsigned* lay,
                                                         float3 rs, float ts, float tmaxs,
                                                         TraceEye&& trace_eye, TraceAny&& trace_any,
                                                         TraceShadow&& trace_shadow)
{
    RtCHit hm{0.f, -1, -1};
    int nl = 0;
    rt_trace_secondaries<Sp>(F, inst, h, nl, s.active, s.eye, s.rd, s.tmax, rs, ts, tmaxs, trace_eye, trace_any,
        [&](int q, const RtCHit& x) {
            lay[(3 * q + 0) * 64] = __float_as_uint(x.t);
            lay[(3 * q + 1) * 64] = (unsigned)x.li;
            lay[(3 * q + 2) * 64] = (unsigned)x.tri;
        },
        [&](const RtCHit& x) { hm = x; });
    const unsigned vis = (Sp::shade(F) & RT_SH_SHADOW) && Sp::mode(F) == RT_MODE_HIPGL
                       ? rt_shadow_mask(F, inst, h, s, trace_shadow) : ~0u;
    return rt_shade_sample<Sp>(F, C, inst, h, nl, s,
        [&](int q) { return RtCHit{__uint_as_float(lay[(3 * q) * 64]), (int)lay[(3 * q + 1) * 64],
                                   (int)lay[(3 * q + 2) * 64]}; },
        [&]() { return hm; }, vis);
}

// Mesh-local triangle of a traversal hit record (-1: primitive / miss).
template <int TRAV>
__device__ __forceinline__ int rt_hit_tri(const RtFrame& F, const RtHitRec& r) {
    if (r.li < 0 || r.ref < 0) return -1;
    return rt_trav_f2i(((TRAV == RT_TRAV_SKIP ? F.tri : F.tri64) + r.ref)->v0.w);
}

// Bijective remap of a 1-D workgroup id so XCD k (which receives ids k, k+8,
// ...) works on the contiguous id range [k*per(+rem adj.), ...): neighbouring
// tiles of one (world, camera) then share that XCD's L2.
__device__ __forceinline__ unsigned rt_xcd_swizzle(unsigned bid, unsigned nb) {
    const unsigned xcd = bid % RT_NUM_XCD, idx = bid / RT_NUM_XCD;
    const unsigned per = nb / RT_NUM_XCD, rem = nb % RT_NUM_XCD;
    return xcd < rem ? xcd * (per + 1) + idx : rem * (per + 1) + (xcd - rem) * per + idx;
}

// ── K2 ──────────────────────────────────────────────────────────────────────
// Launch parameters of one K2 pass (not part of RtFrame).
//   ss / tp: samples per pixel axis and pixels per tile axis of this pass;
//            the wave's tile spans tp*ss <= 8 samples per axis.
//   out_mask: outputs this pass writes (a subset of F.out_mask).
//   vlist / vcount: per-(slot, camera) view-cull lists (null: all nrg).
//   msaa:  MSAA4 RGB pass (ss = 2 lane layout: 4 coverage samples of a pixel
//          in a 2x2 lane block; shading kernels only).
//   rec / rec_mir / rec_lay / nrec: deferred-shading records of this pass
//          (RT_K2_TRACE writes, rt_shade_kernel reads), see rt_rec_index.
struct RtPass {
    int slot0, nslot, swizzle, ss, tp, out_mask, msaa;
    const int* vlist; const int* vcount;
    uint4* rec; uint4* rec_mir; uint4* rec_lay; size_t nrec;
};

// K2 variants: geometry only (GREY / depth / seg), fused full shading, or the
// trace half of the deferred (split) full shading.
enum RtK2Kind : int { RT_K2_GEOM = 0, RT_K2_FUSED = 1, RT_K2_TRACE = 2,
                      RT_K2_TRACE_MS = 3 };

// Deferred-shading record of one sample (uint4): t (float bits), li, mesh
// triangle, meta. Main records: [rec index] for every live sample; mirror /
// layer records only where that sample traced one (layer `it` at
// rec_lay[it * nrec + index]). Meta (MSAA4 passes): bit 0 active (this sample
// shades its surface), bit 1 shade along the sample ray (not the pixel
// centre), bits 2-3 source sample k of an inactive sample.
// Index: SSAA ((sl * ncam + cam) * H*ss + sy) * W*ss + sx; MSAA4
// (((sl * ncam + cam) * H + py) * W + px) * 4 + k; sl = slot - P.slot0.
__device__ __forceinline__ size_t rt_rec_index(const RtFrame& F, const RtPass& P, int slot, int zc,
                                               bool msaa, int px, int py, int i, int j, int ss) {
    const size_t v = (size_t)(slot - P.slot0) * F.ncam + zc;
    if (msaa) return ((v * F.H + py) * F.W + px) * 4 + (i + 2 * j);
    return (v * (size_t)(F.H * ss) + (size_t)(py * ss + j)) * (size_t)(F.W * ss) + (size_t)(px * ss + i);
}
__device__ __forceinline__ uint4 rt_rec_pack(const RtCHit& h, unsigned meta) {
    return make_uint4(__float_as_uint(h.t), (unsigned)h.li, (unsigned)h.tri, meta);
}
__device__ __forceinline__ RtCHit rt_rec_hit(const uint4& r) {
    return RtCHit{__uint_as_float(r.x), (int)r.y, (int)r.z};
}

// Compile-time shading presets of the shading kernels (rt_shade.inc
// RtSpecDyn / RtSpecFix). 0: generic (every frame field read at run time);
// the others are taken only when rt_pick_spec finds the frame equal to them.
enum RtSpecId : int { RT_SPEC_DYN = 0, RT_SPEC_MUJOCO = 1, RT_SPEC_RASTER_LIN_LAY = 2 };
template <int SPEC> struct RtSpecOf { using type = RtSpecDyn; };
// set_shading_preset('mujoco'), sRGB output, no transparent geoms, no fog
template <> struct RtSpecOf<RT_SPEC_MUJOCO> { using type = RtSpecFix<RT_SH_PRESET_MUJOCO, false, false, false>; };
// set_shading_preset('raster') with linear output and transparent geoms, no fog
template <> struct RtSpecOf<RT_SPEC_RASTER_LIN_LAY> {
    using type = RtSpecFix<RT_SH_GOURAUD | RT_SH_REFLECT, true, true, false>;
};
template <int SPEC>
inline bool rt_spec_matches(const RtFrame& F) {
    using S = typename RtSpecOf<SPEC>::type;
    return F.mode == RT_MODE_HIPGL && F.shade == S::kShade && (F.linear != 0) == S::kLinear &&
           (F.ntransp > 0) == S::kLay && (F.fog_on != 0) == S::kFog;
}
inline int rt_pick_spec(const RtFrame& F) {
    if (rt_spec_matches<RT_SPEC_MUJOCO>(F)) return RT_SPEC_MUJOCO;
    if (rt_spec_matches<RT_SPEC_RASTER_LIN_LAY>(F)) return RT_SPEC_RASTER_LIN_LAY;
    return RT_SPEC_DYN;
}

#ifndef RT_SHADE_WAVES_PER_EU
#define RT_SHADE_WAVES_PER_EU 1
#endif
#ifndef RT_TRACE_WAVES_PER_EU
#define RT_TRACE_WAVES_PER_EU 1
#endif
// Preset-specialised fused variants (SPEC != 0): RT_SHADE_SPEC_WAVES_PER_EU.
#ifndef RT_SHADE_SPEC_WAVES_PER_EU
#define RT_SHADE_SPEC_WAVES_PER_EU 3
#endif
template <int TRAV, bool STATS, bool LOOP, int KIND = RT_K2_GEOM, int SPEC = 0>
struct RtPrimaryWaves {
#ifdef RT_PRIMARY_WAVES_PER_EU
    static constexpr int value = RT_PRIMARY_WAVES_PER_EU;
#else
    static constexpr int value = (KIND == RT_K2_FUSED && SPEC != 0) ? RT_SHADE_SPEC_WAVES_PER_EU
                               : KIND == RT_K2_FUSED ? RT_SHADE_WAVES_PER_EU
                               : (KIND == RT_K2_TRACE || KIND == RT_K2_TRACE_MS) ? RT_TRACE_WAVES_PER_EU
                               : ((!STATS && !LOOP) ? 5 : 1);
#endif
};

struct RtWaveCtx { int slot, zc, tx, ty; };
__device__ __forceinline__ RtWaveCtx rt_wave_ctx(const RtFrame& F, const RtPass& P, int tiles_x, int tiles) {
    (void)tiles;
    const unsigned bid = P.swizzle ? rt_xcd_swizzle(blockIdx.x, gridDim.x) : blockIdx.x;
    const int sl = (int)(bid % (unsigned)P.nslot);
    const int rest = (int)(bid / (unsigned)P.nslot);
    RtWaveCtx w;
    w.slot = P.slot0 + sl;
    w.zc = rest % F.ncam;
    const int tile = rest / F.ncam;
    w.tx = tile % tiles_x;
    w.ty = tile / tiles_x;
    return w;
}

// This wave's view-cull list (instances meeting the camera's whole frustum):
// vl[0, nv), or vl = null and nv = nrg when there is none (tile cull off).
struct RtView { const int* vl; int nv; };
__device__ __forceinline__ RtView rt_view_of(const RtFrame& F, const RtPass& P, int slot, int zc) {
    RtView v;
    if (P.vlist) {
        const size_t k = (size_t)slot * F.ncam + zc;
        v.vl = P.vlist + k * F.nrg;
        v.nv = rt_sload(P.vcount + k);
    } else {
        v.vl = nullptr;
        v.nv = F.nrg;
    }
    return v;
}

// Tile frustum cull of the pixel window [x0,x1) x [y0,y1) into the LDS list.
// Returns the sorted list length, or -1 when more than RT_CAND_CAP instances
// survive (never seen in practice: a few tens is typical), in which case the
// caller walks the whole view list in li order. Same result either way.
template <bool STATS>
__device__ __forceinline__ int rt_build_list(const RtFrame& F, const RtCam& C, float3 eye,
                                             const RtInst* __restrict__ inst, const RtView& V,
                                             unsigned* cand, int x0, int x1, int y0, int y1)
{
    int next;
    const RtTileFrustum fr = rt_tile_frustum(C.Rc, C.tanh_, F.W, F.H, x0, x1, y0, y1, F.clip, F.znear, F.zfar);
    const int n = rt_tile_cull<STATS>(fr, eye, inst, V.vl, 0, V.nv, cand, RT_CAND_CAP, &next);
    if (next < V.nv) return -1;
    rt_cand_sort<RT_CAND_CAP>(cand, n);
    return n;
}

// Mirror-ray cull of a tile into the LDS list (unsorted; walk it without the
// early break). The mirror rays of a tile start on the mirror's +Z face (offset
// 1e-4 along its normal N, rt_mirror_ray) and go along reflect(rd) for camera
// rays rd of the tile: each lies on the line from A = veye + 1e-4 N (veye = the
// eye mirrored in the face's plane) along reflect(rd), i.e. inside the tile
// frustum reflected in that plane with apex A, at planar depth (along N) at
// least A's distance to the plane. Boxes outside it (with rt_box_in_frustum's
// padding, which covers the rounding of o and r) cannot be hit, so the walk
// over the survivors gives the linear walk's (t, li) result. Returns the
// survivor count, or -1 (more than RT_CAND_CAP survive: walk all instances).
__device__ __forceinline__ int rt_build_mirror_list(const RtFrame& F, const RtCam& C, float3 eye,
                                                    const RtInst* __restrict__ inst, unsigned* cand,
                                                    int x0, int x1, int y0, int y1, float3& apex)
{
    const RtInst M = rt_sload(inst + F.mirror_li);
    const RtGeom* g = F.geom + F.mirror_li;
    const float off = rt_sload(&g->type) == GEOM_BOX ? rt_sload(&g->size[2]) : 0.0f;
    const float3 N = rt_norm(rt_mat_vec(M.R, make_float3(0.0f, 0.0f, 1.0f)));
    const float3 Pm = rt_add(make_float3(M.p[0], M.p[1], M.p[2]), rt_mul(N, off));
    const float3 veye = rt_sub(eye, rt_mul(N, 2.0f * rt_dot(rt_sub(eye, Pm), N)));
    const float3 A = rt_add(veye, rt_mul(N, 1e-4f));
    RtTileFrustum fr = rt_tile_frustum(C.Rc, C.tanh_, F.W, F.H, x0, x1, y0, y1, 0, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < 4; ++k) fr.n[k] = rt_sub(fr.n[k], rt_mul(N, 2.0f * rt_dot(fr.n[k], N)));
    const float d0 = rt_dot(N, rt_sub(Pm, A));
    fr.fwd = N;
    fr.dmin = d0 - fabsf(d0) * 1e-3f - 1e-3f;
    fr.dmax = INFINITY;
    int next;
    __syncthreads();   // the list replaces the primary tile list
    const int n = rt_tile_cull<false>(fr, A, inst, nullptr, 0, F.nrg, cand, RT_CAND_CAP, &next);
    apex = A;
    if (next < F.nrg) { __syncthreads(); return -1; }
    // Sorted by (key from A, li) like the primary list, so the walk can stop
    // early on the per-lane bound of rt_trace_ray<.., RT_BREAK_APEX>.
    rt_cand_sort<RT_CAND_CAP>(cand, n);
    return n;
}

// Closest hit of one camera ray over the candidates: the sorted tile list
// (ncand >= 0; early break on its near keys) or every instance of the view
// list in li order (ncand < 0; all nrg instances when V.vl is null). tmax is
// exclusive.
// BREAK = RT_BREAK_NONE: walk the whole list.
// BREAK = RT_BREAK_APEX: the list keys are distances from an apex A other
// than the ray origin o (the mirror list: A = the virtual eye). For a hit
// Q = o + t r inside a box, |Q - A| >= dist(A, box) >= key and
// |Q - A| <= |o - A| + t |r| (triangle inequality), so t >= key - |o - A|
// (|r| = 1 up to rounding). koff = |o - A| per lane, padded for rounding
// (-inf for a lane without a ray: it never holds the walk); the walk stops at
// the first entry whose padded bound exceeds every lane's best t, which keeps
// the (t, li) result exact.
enum RtBreak : int { RT_BREAK_NONE = 0, RT_BREAK_KEY = 1, RT_BREAK_APEX = 2 };
template <int TRAV, bool STATS, typename Stack, int BREAK = RT_BREAK_KEY>
__device__ __forceinline__ RtHitRec rt_trace_ray(const RtFrame& F, const RtInst* __restrict__ inst,
                                                 const unsigned* cand, int ncand_in, const RtView& V,
                                                 float3 eye, float3 rd, float tmin, float tmax,
                                                 Stack& st, RtCounters& cnt, float koff = 0.0f)
{
    RtHitRec hr;
    hr.t = tmax; hr.li = -1; hr.ref = -1;
    const float3 iv = rt_safe_inv(rd);
    const int ncand = rt_uniform_i(ncand_in);
    const bool listed = ncand >= 0;
    const int n = rt_uniform_i(listed ? ncand : V.nv);
    if (STATS && (threadIdx.x & 63) == 0 && listed) cnt.list += (unsigned)ncand;
    for (int k = 0; k < n; ++k) {
        int li;
        if (listed) {
            const unsigned e = __builtin_amdgcn_readfirstlane(cand[k]);
            // Every lane's best hit is closer than anything this entry (and
            // every later one) can offer.
            if (BREAK == RT_BREAK_KEY && __ballot(hr.t >= rt_cand_key(e)) == 0ull) break;
            if (BREAK == RT_BREAK_APEX &&
                __ballot(hr.t >= (rt_cand_key(e) - koff) * (1.0f - 1e-4f)) == 0ull) break;
            li = rt_cand_li(e);
        } else {
            li = V.vl ? rt_sload(V.vl + k) : k;
        }
        li = rt_uniform_i(li);
        if (STATS && (threadIdx.x & 63) == 0) cnt.visited++;
        rt_offer<TRAV, STATS>(F, rt_sload(inst + li), li, eye, rd, iv, tmin, hr, st, cnt);
    }
    if (hr.li < 0) hr.t = tmax;
    return hr;
}

#ifndef RT_GRP_TP2
#define RT_GRP_TP2 0
#endif
#ifndef RT_GRP_MIN_CAND_V
#define RT_GRP_MIN_CAND_V 8
#endif
constexpr int RT_GRP_MIN_CAND = RT_GRP_MIN_CAND_V;
__device__ __forceinline__ void rt_hit_merge(RtHitRec& h, float ot, int oli, int oref) {
    if (oli >= 0 && (h.li < 0 || ot < h.t || (ot == h.t && oli < h.li))) { h.t = ot; h.li = oli; h.ref = oref; }
}
template <int GRP>
__device__ __forceinline__ void rt_hit_group_min(RtHitRec& h) {
#pragma unroll
    for (int off = 1; off < GRP; off <<= 1)
        rt_hit_merge(h, __shfl_xor(h.t, off), __shfl_xor(h.li, off), __shfl_xor(h.ref, off));
}
template <int TRAV, int GRP, typename Stack>
__device__ __forceinline__ RtHitRec rt_trace_ray_grp(const RtFrame& F, const RtInst* __restrict__ inst,
                                                     const unsigned* cand, int ncand_in, const RtView& V,
                                                     float3 eye, float3 rd, float tmin, float tmax,
                                                     Stack& st, RtCounters& cnt)
{
    const int ncand = rt_uniform_i(ncand_in);
    if (GRP <= 1 || ncand < RT_GRP_MIN_CAND)   // also the unsorted fallback (ncand < 0)
        return rt_trace_ray<TRAV, false>(F, inst, cand, ncand, V, eye, rd, tmin, tmax, st, cnt);
    RtHitRec hr;
    hr.t = tmax; hr.li = -1; hr.ref = -1;
    const float3 iv = rt_safe_inv(rd);
    const int sub = (int)(threadIdx.x & (GRP - 1));
    for (int k0 = 0; k0 < ncand; k0 += GRP) {
        const int k = k0 + sub;
        const unsigned e = k < ncand ? cand[k] : 0u;
        const bool act = k < ncand && hr.t >= rt_cand_key(e);   // rt_trace_ray's break test, per lane
        if (__ballot(act) == 0ull) break;
        if (act) {
            const int li = rt_cand_li(e);
            rt_offer<TRAV, false>(F, inst[li], li, eye, rd, iv, tmin, hr, st, cnt);
        }
        rt_hit_group_min<GRP>(hr);
    }
    if (hr.li < 0) hr.t = tmax;
    return hr;
}

template <bool STATS>
__device__ __forceinline__ void rt_flush_stats(const RtFrame& F, const RtCounters& cnt, bool live, unsigned nrays) {
    if (!STATS) return;
    // [0] instance slab survivors, [1] BLAS nodes, [2] triangles, [3] rays,
    // summed over live lanes; per wave (lane 0) and wave-wide ray pass: [4]
    // instances accepted by >= 1 lane, [5] passes, [6] list entries
    // (tile-cull survivors), [7] entries visited before the early break.
    const int lane = threadIdx.x & 63;
    if (live) {
        atomicAdd(&F.stats[0], (unsigned long long)cnt.cand);
        atomicAdd(&F.stats[1], (unsigned long long)cnt.nodes);
        atomicAdd(&F.stats[2], (unsigned long long)cnt.tris);
        atomicAdd(&F.stats[3], (unsigned long long)nrays);
    }
    if (lane == 0) {
        atomicAdd(&F.stats[4], (unsigned long long)cnt.cand_tile);
        atomicAdd(&F.stats[5], (unsigned long long)nrays);
        atomicAdd(&F.stats[6], (unsigned long long)cnt.list);
        atomicAdd(&F.stats[7], (unsigned long long)cnt.visited);
    }
}

__device__ __forceinline__ size_t rt_out_pixel(const RtFrame& F, int slot, int zc, int px, int py) {
    const int yo = F.flip_v ? (F.H - 1 - py) : py;
    const int xo = F.flip_h ? (F.W - 1 - px) : px;
    const size_t row = (size_t)zc * F.nout + F.out_row[slot];
    return (row * F.H + yo) * F.W + xo;
}

__device__ __forceinline__ void rt_store_rgb(const RtFrame& F, size_t pix, float r, float g, float b,
                                             bool trunc = false) {
    uint8_t* o = F.rgb + pix * 3;
    const float rnd = trunc ? 0.0f : 0.5f;
    o[0] = (uint8_t)(rt_clamp01(r) * 255.0f + rnd);
    o[1] = (uint8_t)(rt_clamp01(g) * 255.0f + rnd);
    o[2] = (uint8_t)(rt_clamp01(b) * 255.0f + rnd);
}

template <class Sp, class TraceEye, class TraceAny, class StoreLayer, class Regroup, class StoreMirror>
__device__ __attribute__((always_inline)) inline void rt_trace_secondaries_ms(
        const RtFrame& F, const RtInst* __restrict__ inst, RtCHit& h, int& nl, float3 eye,
        float3 rd, float tmax, float3 rs, float ts, float tmaxs,
        TraceEye&& trace_eye, TraceAny&& trace_any, StoreLayer&& store_layer, Regroup&& regroup,
        StoreMirror&& store_mirror) {
    nl = 0;
    if (Sp::lay(F) && Sp::mode(F) == RT_MODE_HIPGL) {
        const bool conv = ts != h.t;
        RtCHit hs = h;
        hs.t = ts;
        bool moved = false;
#pragma unroll 1
        for (int it = 0; it < RT_MAX_TLAYERS + RT_SKIP_TLAYERS; ++it) {
            const bool tr = hs.li >= 0 && (F.geom[hs.li].flags & RT_F_TRANSPARENT);
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
    const bool active = regroup(h, nl);
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

template <int TRAV, bool STATS, bool TILE_CULL, int KIND, int SPEC = RT_SPEC_DYN, int GRP = 1>
__global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(RtPrimaryWaves<TRAV, STATS, false, KIND, SPEC>::value)))
rt_primary_kernel(RtFrame F, RtPass P)
{
    constexpr bool SHADE = KIND != RT_K2_GEOM;
    using Sp = typename RtSpecOf<SPEC>::type;
    constexpr int AUX = TRAV == RT_TRAV_ORDERED ? RT_STACK * 64 : 64 * 3;
    __shared__ unsigned s_cand[TILE_CULL ? RT_CAND_CAP : 1];
    __shared__ unsigned s_aux[AUX];   // ORDERED stack during the walk, then the SSAA reduction
    __shared__ unsigned s_lay[KIND == RT_K2_FUSED ? RT_LAY_LDS : 1];   // fused: layer records

    const int ss = P.ss, tp = P.tp, span = tp * ss;
    const int tiles_x = (F.W + tp - 1) / tp;
    const int tiles = tiles_x * ((F.H + tp - 1) / tp);
    const RtWaveCtx w = rt_wave_ctx(F, P, tiles_x, tiles);
    if (w.slot >= F.nslot) return;
    // Lane -> sample (lx, ly) of the tile; idle lanes (lane coordinate >= span)
    // repeat the last sample of their row/column. Out-of-image lanes trace the
    // nearest in-image pixel of their own tile (inside the tile frustum); they
    // do not write.
    struct LaneMap { int lx, ly, i, j, px, py, qx, qy; bool live; };
    // The lane that owns tile sample (x, y).
    auto sample_lane = [&](int x, int y) { return GRP > 1 ? (y * span + x) * GRP : y * 8 + x; };
    auto lane_map = [&](int lane) {
        LaneMap m;
        bool own;
        if constexpr (GRP > 1) {
            const int r = lane / GRP;
            m.lx = r % span; m.ly = r / span;
            own = (lane & (GRP - 1)) == 0;
        } else {
            m.lx = min(lane & 7, span - 1); m.ly = min(lane >> 3, span - 1);
            own = (lane & 7) < span && (lane >> 3) < span;
        }
        const int pxo = m.lx / ss, pyo = m.ly / ss;
        m.i = m.lx - pxo * ss; m.j = m.ly - pyo * ss;
        m.px = w.tx * tp + pxo; m.py = w.ty * tp + pyo;
        m.live = own && m.px < F.W && m.py < F.H;
        m.qx = min(m.px, F.W - 1); m.qy = min(m.py, F.H - 1);
        return m;
    };

    const RtCam& C = F.cam[(size_t)w.slot * F.ncam + w.zc];
    const RtInst* __restrict__ inst = F.inst + (size_t)w.slot * F.nrg;
    const float3 eye = make_float3(C.eye[0], C.eye[1], C.eye[2]);
    const int Wn = F.W * ss, Hn = F.H * ss;
    const float aspect = (float)F.W / (float)F.H;
    const bool msaa = SHADE && P.msaa;

    RtPrimaryStack st;
#if defined(__HIP_DEVICE_COMPILE__)
    st.base = s_aux + threadIdx.x;
#endif
    const RtView V = rt_view_of(F, P, w.slot, w.zc);
    const int ncand = TILE_CULL ? rt_build_list<STATS>(F, C, eye, inst, V, s_cand, w.tx * tp,
                                                       min(w.tx * tp + tp, F.W), w.ty * tp,
                                                       min(w.ty * tp + tp, F.H))
                                : -1;
    RtCounters cnt;
    float3 rd; float cz, tmin, tmax;
    {
        const LaneMap m = lane_map(threadIdx.x);
        if (msaa) {
            // GL coverage sample k = i + 2j of pixel (qx, qy)
            const int k = m.i + 2 * m.j;
            const float3 dl = rt_dir_at((float)m.qx + rt_msaa_x(k), (float)m.qy + rt_msaa_y(k),
                                        F.W, F.H, aspect, C.tanh_);
            rd = rt_mat_vec(C.Rc, dl);
            cz = -dl.z;
            tmin = 0.0f; tmax = RT_MAXVAL;
            if (F.clip) { tmin = F.znear / cz; tmax = F.zfar / cz; }
        } else {
            rt_cam_ray(F, C, m.qx * ss + m.i, m.qy * ss + m.j, Wn, Hn, aspect, rd, cz, tmin, tmax);
        }
    }
    RtHitRec hr;
    if constexpr (GRP > 1) hr = rt_trace_ray_grp<TRAV, GRP>(F, inst, s_cand, ncand, V, eye, rd, tmin, tmax, st, cnt);
    else                   hr = rt_trace_ray<TRAV, STATS>(F, inst, s_cand, ncand, V, eye, rd, tmin, tmax, st, cnt);

    // The lane mapping is recomputed from an opaque copy of the lane id rather
    // than kept live (~8 VGPRs) across the traversal.
    int lane = threadIdx.x;
    asm volatile("" : "+v"(lane));
    const LaneMap m = lane_map(lane);
    if (F.clip) tmax = F.zfar / cz; else tmax = RT_MAXVAL;

    float3 col = make_float3(0.f, 0.f, 0.f);
    float out_t = 0.0f;
    int out_li = -1;
    if constexpr (!SHADE) {
        const RtHit h = rt_resolve_hit<TRAV>(F, inst, hr, eye, rd, tmin, tmax);
        out_t = h.t; out_li = h.li;
        if (P.out_mask & RT_OUT_RGB) {
            RtSample sm;
            sm.eye = eye; sm.rd = rd; sm.cz = cz; sm.tmax = tmax;
            sm.sy = m.qy * ss + m.j; sm.Hn = Hn; sm.lw = Wn; sm.lh = Hn; sm.active = true;
            col = rt_shade_grey(F, inst, h, sm);
            col = make_float3(rt_clamp01(col.x), rt_clamp01(col.y), rt_clamp01(col.z));
        }
    } else {
        RtCHit hc;
        hc.t = hr.li >= 0 ? hr.t : tmax; hc.li = hr.li; hc.tri = rt_hit_tri<TRAV>(F, hr);
        out_t = hc.t; out_li = hc.li;
        RtSample sm;
        sm.eye = eye; sm.active = true;
        // MSAA4 lanes address their pixel's 2x2 sample block through the lane
        // that owns the sample they trace (vl), so a lane repeating another
        // lane's sample (tp < 4 tiles) mirrors that lane exactly.
        const int vl = sample_lane(m.lx, m.ly);
        int src = vl, srck = 0;
        bool usedir = false;
        const float ts = hc.t;     // the sample ray's own hit distance (layers continue from it)
        int key = 0;               // MSAA grouping key (below)
        if (!msaa) {
            sm.rd = rd; sm.cz = cz; sm.tmax = tmax;
            sm.sy = m.qy * ss + m.j; sm.Hn = Hn;
            const bool lpx = (Sp::shade(F) & RT_SH_LOD_PIXEL) != 0;
            sm.lw = lpx ? F.W : Wn; sm.lh = lpx ? F.H : Hn;
        } else {
            // Grouping key: mesh triangle, or the primitive's flat patch.
            key = hc.tri;
            if (hc.li >= 0 && hc.tri < 0) {
                const RtInst& I = inst[hc.li];
                const RtGeom& g = F.geom[hc.li];
                const float3 l = rt_mat_tvec(I.R, rt_sub(rt_add(eye, rt_mul(rd, hc.t)), rt_ld3(I.p)));
                key = -1 - rt_face_code(g, rt_prim_normal(g, l));
            }
            // The first earlier sample of this pixel on the same surface shades it
            // (RT_K2_TRACE_MS: decided after the layers, see rt_trace_secondaries_ms).
            if constexpr (KIND == RT_K2_TRACE_MS) {
                // the front-surface half of the key, as a mask of the earlier samples that match it (one register
                // live across the secondaries instead of li + key)
                const int k = m.i + 2 * m.j;
#pragma unroll
                for (int jj = 0; jj < 3; ++jj) {
                    const int lj = sample_lane(m.lx - m.i + (jj & 1), m.ly - m.j + (jj >> 1));
                    const int lij = __shfl(hc.li, lj);
                    const int kj = __shfl(key, lj);
                    if (jj < k && lij == hc.li && (hc.li < 0 || kj == key)) srck |= 1 << jj;   // (srck holds the
                    // mask until the regroup: one register less across the secondaries)
                }
            } else {
            const int k = m.i + 2 * m.j;
#pragma unroll
            for (int jj = 0; jj < 3; ++jj) {
                const int lj = sample_lane(m.lx - m.i + (jj & 1), m.ly - m.j + (jj >> 1));
                const int lij = __shfl(hc.li, lj);
                const int kj = __shfl(key, lj);
                if (jj < k && sm.active && lij == hc.li && (hc.li < 0 || kj == key)) { sm.active = false; src = lj; srck = jj; }
            }
            }
            // Centre evaluation on the hit's tangent plane.
            const float3 dlc = rt_dir_at((float)m.qx + 0.5f, (float)m.qy + 0.5f, F.W, F.H, aspect, C.tanh_);
            const float3 rdc = rt_mat_vec(C.Rc, dlc);
            sm.rd = rdc; sm.cz = -dlc.z;
            sm.tmax = F.clip ? F.zfar / sm.cz : RT_MAXVAL;
            if (sm.active && hc.li >= 0) {
                RtSurf Ss;
                rt_surf_at(F, inst[hc.li], F.geom[hc.li], hc.li, hc.tri, rt_add(eye, rt_mul(rd, hc.t)), Ss);
                const float den = rt_dot(rdc, Ss.Ng);
                const float tc = den < -1e-6f ? hc.t * rt_dot(rd, Ss.Ng) / den : -1.0f;
                if ((Sp::shade(F) & RT_SH_MSAA_SAMPLE) || !(tc > 0.0f)) { sm.rd = rd; sm.cz = cz; usedir = true; }
                else hc.t = tc;
            }
            sm.sy = m.qy; sm.Hn = F.H; sm.lw = F.W; sm.lh = F.H;
        }
        if (P.out_mask & RT_OUT_RGB) {
            auto trace_eye = [&](float3 d, float t0, float t1) -> RtCHit {
                const RtHitRec r = rt_trace_ray<TRAV, false>(F, inst, s_cand, ncand, V, eye, d, t0, t1, st, cnt);
                return RtCHit{r.t, r.li, rt_hit_tri<TRAV>(F, r)};
            };
            // Mirror rays (the only trace_any rays): the tile's reflected
            // frustum list (replaces the primary list, which is dead by then),
            // or all instances.
            auto trace_any = [&](float3 o, float3 d, float t0, float t1) -> RtCHit {
                RtHitRec r;
                float3 A = eye;
                const int nm = TILE_CULL ? rt_build_mirror_list(F, C, eye, inst, s_cand, w.tx * tp,
                                                                min(w.tx * tp + tp, F.W), w.ty * tp,
                                                                min(w.ty * tp + tp, F.H), A)
                                         : -1;
                if (nm >= 0) {
                    const RtView VA{nullptr, F.nrg};
                    const float koff = t1 < 0.0f ? -INFINITY : rt_len(rt_sub(o, A)) * (1.0f + 1e-4f) + 1e-4f;
                    r = rt_trace_ray<TRAV, false, RtPrimaryStack, RT_BREAK_APEX>(F, inst, s_cand, nm, VA, o, d, t0, t1,
                                                                                 st, cnt, koff);
                } else {
                    rt_closest_linear<TRAV, false>(F, inst, o, d, t0, t1, r, st, cnt);
                }
                return RtCHit{r.t, r.li, rt_hit_tri<TRAV>(F, r)};
            };
            // Shadow rays (RT_SH_SHADOW): every instance, closest hit within t1.
            auto trace_shadow = [&](float3 o, float3 d, float t1) -> bool {
                RtHitRec r;
                rt_closest_linear<TRAV, false>(F, inst, o, d, 0.0f, t1, r, st, cnt);
                return r.li >= 0;
            };
            if constexpr (KIND == RT_K2_FUSED) {
                col = rt_shade_sample_traced<Sp>(F, C, inst, hc, sm, s_lay + lane, rd, ts, tmax,
                                             trace_eye, trace_any, trace_shadow);
                col = make_float3(rt_clamp01(col.x), rt_clamp01(col.y), rt_clamp01(col.z));
            } else {
                // Deferred: trace the secondaries, store every record.
                const size_t ri = rt_rec_index(F, P, w.slot, w.zc, msaa, m.qx, m.qy, m.i, m.j, ss);
                const bool wr = m.live;
                RtCHit ho = hc;
                int nl = 0;
                if constexpr (KIND == RT_K2_TRACE_MS) {
                    // Group key: the frontmost primitive / flat patch (key), the surface behind the layers and
                    // the layer sequence (an order-dependent 32-bit hash of the layer instances; it also separates
                    // the layer counts: lh = 0 without layers, and every step adds li + 1 >= 1).
                    unsigned lh = 0u;
                    rt_trace_secondaries_ms<Sp>(F, inst, ho, nl, eye, sm.rd, sm.tmax, rd, ts, tmax,
                                         trace_eye, trace_any,
                        [&](int it, const RtCHit& hl) {
                            lh = lh * 0x9E3779B1u + (unsigned)(hl.li + 1);
                            if (wr) P.rec_lay[(size_t)it * P.nrec + ri] = rt_rec_pack(hl, 0u);
                        },
                        [&](const RtCHit& hb, int) -> bool {
                            const int k = m.i + 2 * m.j;
                            const int key_mask = srck;
                            srck = 0;
#pragma unroll
                            for (int jj = 0; jj < 3; ++jj) {
                                const int lj = sample_lane(m.lx - m.i + (jj & 1), m.ly - m.j + (jj >> 1));
                                const int bj = __shfl(hb.li, lj);
                                const int tj = __shfl(hb.tri, lj);
                                const unsigned hj = (unsigned)__shfl((int)lh, lj);
                                if (jj < k && sm.active && ((key_mask >> jj) & 1) && bj == hb.li &&
                                    tj == hb.tri && hj == lh) { sm.active = false; src = lj; srck = jj; }
                            }
                            return sm.active;
                        },
                        [&](const RtCHit& hm) { if (wr) P.rec_mir[ri] = rt_rec_pack(hm, 0u); });
                } else {
                rt_trace_secondaries<Sp>(F, inst, ho, nl, sm.active, eye, sm.rd, sm.tmax, rd, ts, tmax,
                                     trace_eye, trace_any,
                    [&](int it, const RtCHit& hl) { if (wr) P.rec_lay[(size_t)it * P.nrec + ri] = rt_rec_pack(hl, 0u); },
                    [&](const RtCHit& hm) { if (wr) P.rec_mir[ri] = rt_rec_pack(hm, 0u); });
                }
                const unsigned meta = (sm.active ? 1u : 0u) | (usedir ? 2u : 0u) | ((unsigned)srck << 2) |
                                      ((unsigned)nl << 4);
                if (wr) P.rec[ri] = rt_rec_pack(ho, meta);
            }
        }
        if ((KIND == RT_K2_TRACE || KIND == RT_K2_TRACE_MS) && msaa) {   // records only; RGB comes from rt_shade_kernel
            rt_flush_stats<STATS>(F, cnt, m.live, 1u);
            return;
        }
        if (msaa) {
            // u8 per sample; copies from the shading lane; rounded average.
            uint32_t q = rt_pack8(col);
            const uint32_t qs = __shfl(q, src);
            if (!sm.active) q = qs;
            const uint32_t q1 = __shfl(q, sample_lane(m.lx + 1, m.ly)), q2 = __shfl(q, sample_lane(m.lx, m.ly + 1)),
                           q3 = __shfl(q, sample_lane(m.lx + 1, m.ly + 1));
            rt_flush_stats<STATS>(F, cnt, m.live, 1u);
            if (!m.live || m.i != 0 || m.j != 0) return;
            uint32_t out = 0u;
            for (int c = 0; c < 3; ++c) {
                const uint32_t sum = ((q >> (8 * c)) & 255u) + ((q1 >> (8 * c)) & 255u) +
                                     ((q2 >> (8 * c)) & 255u) + ((q3 >> (8 * c)) & 255u);
                out |= ((sum + 2u) >> 2) << (8 * c);
            }
            uint8_t* o = F.rgb + rt_out_pixel(F, w.slot, w.zc, m.qx, m.qy) * 3;
            o[0] = (uint8_t)(out & 255u); o[1] = (uint8_t)((out >> 8) & 255u); o[2] = (uint8_t)((out >> 16) & 255u);
            return;
        }
    }
    rt_flush_stats<STATS>(F, cnt, m.live, 1u);

    // mujoco_warp packs with truncation (render_util.py pack_rgba_to_uint32).
    const bool trunc = SHADE && Sp::mode(F) == RT_MODE_MJW;
    const size_t pix = rt_out_pixel(F, w.slot, w.zc, m.qx, m.qy);
    if ((KIND == RT_K2_TRACE || KIND == RT_K2_TRACE_MS) && ss != 1) {
        // RGB comes from rt_shade_kernel; depth/seg from the odd-ss centre sample.
        if (!m.live || !(m.i == ss / 2 && m.j == ss / 2)) return;
    } else if (ss == 1) {
        if (!m.live) return;
        if (KIND != RT_K2_TRACE && KIND != RT_K2_TRACE_MS && (P.out_mask & RT_OUT_RGB)) rt_store_rgb(F, pix, col.x, col.y, col.z, trunc);
    } else {
        if (P.out_mask & RT_OUT_RGB) {
            float* red = reinterpret_cast<float*>(s_aux);
            __syncthreads();   // the ORDERED stack shares s_aux
            red[lane * 3 + 0] = col.x; red[lane * 3 + 1] = col.y; red[lane * 3 + 2] = col.z;
            __syncthreads();
            if (m.live && m.i == 0 && m.j == 0) {
                float acc[3] = {0.f, 0.f, 0.f};
                for (int jj = 0; jj < ss; ++jj)
                    for (int ii = 0; ii < ss; ++ii) {
                        const int q = ((m.ly + jj) * 8 + (m.lx + ii)) * 3;
                        acc[0] += red[q + 0]; acc[1] += red[q + 1]; acc[2] += red[q + 2];
                    }
                const float inv = 1.0f / (float)(ss * ss);
                rt_store_rgb(F, pix, acc[0] * inv, acc[1] * inv, acc[2] * inv, trunc);
            }
        }
        if (!m.live || !(m.i == ss / 2 && m.j == ss / 2)) return;   // odd ss: the centre sample
    }
    if (P.out_mask & (RT_OUT_DEPTH | RT_OUT_SEG)) {
        float d = 0.0f; int g = -1;
        if (out_li >= 0) { d = out_t * cz; g = F.geom[out_li].gid; }
        if (P.out_mask & RT_OUT_DEPTH) F.depth[pix] = d;
        if (P.out_mask & RT_OUT_SEG)   F.seg[pix] = g;
    }
}

// Fallback K2 for ss > 8 (a pixel's samples no longer fit one wave): one
// 8x8-pixel tile per wave, every lane loops over its pixel's ss^2 samples
// (plus the centre ray for depth/seg when ss is even). Tile cull on, no
// counters.
template <int TRAV, bool SHADE>
__global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(RtPrimaryWaves<TRAV, false, true, SHADE>::value)))
rt_primary_loop_kernel(RtFrame F, RtPass P)
{
    __shared__ unsigned s_cand[RT_CAND_CAP];
    __shared__ unsigned s_stack[TRAV == RT_TRAV_ORDERED ? RT_STACK * 64 : 1];
    __shared__ unsigned s_lay[SHADE ? RT_LAY_LDS : 1];

    const int lane = threadIdx.x;
    const int tiles_x = (F.W + RT_TILE - 1) / RT_TILE;
    const int tiles = tiles_x * ((F.H + RT_TILE - 1) / RT_TILE);
    const RtWaveCtx w = rt_wave_ctx(F, P, tiles_x, tiles);
    if (w.slot >= F.nslot) return;
    const int px = w.tx * RT_TILE + (lane & 7);
    const int py = w.ty * RT_TILE + (lane >> 3);
    const bool live = px < F.W && py < F.H;
    const int qx = min(px, F.W - 1), qy = min(py, F.H - 1);

    const RtCam& C = F.cam[(size_t)w.slot * F.ncam + w.zc];
    const RtInst* __restrict__ inst = F.inst + (size_t)w.slot * F.nrg;
    const float3 eye = make_float3(C.eye[0], C.eye[1], C.eye[2]);
    const float aspect = (float)F.W / (float)F.H;
    const int SS = P.ss;

    RtPrimaryStack st;
#if defined(__HIP_DEVICE_COMPILE__)
    st.base = s_stack + lane;
#endif
    const RtView V = rt_view_of(F, P, w.slot, w.zc);
    const int ncand = rt_build_list<false>(F, C, eye, inst, V, s_cand, w.tx * RT_TILE,
                                           min(w.tx * RT_TILE + RT_TILE, F.W), w.ty * RT_TILE,
                                           min(w.ty * RT_TILE + RT_TILE, F.H));
    RtCounters cnt;
    float acc[3] = {0.f, 0.f, 0.f};
    float out_depth = 0.0f;
    int   out_seg = -1;
    const bool want_ds = (P.out_mask & (RT_OUT_DEPTH | RT_OUT_SEG)) != 0;
    const bool odd = (SS & 1) != 0;
    const int cs = SS / 2;
    const int nsamp = SS * SS;
    const int nrays = nsamp + ((!odd && want_ds) ? 1 : 0);
    for (int r = 0; r < nrays; ++r) {
        const bool centre_ray = r >= nsamp;
        const int i = r % SS, j = r / SS;
        RtSample sm;
        const int sx = centre_ray ? qx : qx * SS + i;
        sm.sy = centre_ray ? qy : qy * SS + j;
        const int Wn = centre_ray ? F.W : F.W * SS;
        sm.Hn = centre_ray ? F.H : F.H * SS;
        const bool lpx = SHADE && (F.shade & RT_SH_LOD_PIXEL) != 0;
        sm.lw = lpx ? F.W : Wn; sm.lh = lpx ? F.H : sm.Hn;
        sm.eye = eye; sm.active = true;
        float tmin, tmax;
        // Re-read the camera through an opaque copy of its address each ray.
        // Without it LICM hoists the uniform camera products (VALU results,
        // so VGPRs on gfx9) out of the sample loop and keeps them live across
        // the whole traversal.
        const RtCam* Cr = rt_opaque_uniform(&C);
        rt_cam_ray(F, *Cr, sx, sm.sy, Wn, sm.Hn, aspect, sm.rd, sm.cz, tmin, tmax);
        sm.tmax = tmax;
        const RtHitRec hr = rt_trace_ray<TRAV, false>(F, inst, s_cand, ncand, V, eye, sm.rd, tmin, tmax, st, cnt);
        float ht; int hli;
        if constexpr (!SHADE) {
            const RtHit h = rt_resolve_hit<TRAV>(F, inst, hr, eye, sm.rd, tmin, tmax);
            ht = h.t; hli = h.li;
            if (!centre_ray && (P.out_mask & RT_OUT_RGB)) {
                const float3 col = rt_shade_grey(F, inst, h, sm);
                acc[0] += rt_clamp01(col.x); acc[1] += rt_clamp01(col.y); acc[2] += rt_clamp01(col.z);
            }
        } else {
            const RtCHit hc{hr.li >= 0 ? hr.t : tmax, hr.li, rt_hit_tri<TRAV>(F, hr)};
            ht = hc.t; hli = hc.li;
            if (!centre_ray && (P.out_mask & RT_OUT_RGB)) {
                auto trace_eye = [&](float3 d, float t0, float t1) -> RtCHit {
                    const RtHitRec q = rt_trace_ray<TRAV, false>(F, inst, s_cand, ncand, V, eye, d, t0, t1, st, cnt);
                    return RtCHit{q.t, q.li, rt_hit_tri<TRAV>(F, q)};
                };
                auto trace_any = [&](float3 o, float3 d, float t0, float t1) -> RtCHit {
                    RtHitRec q;
                    rt_closest_linear<TRAV, false>(F, inst, o, d, t0, t1, q, st, cnt);
                    return RtCHit{q.t, q.li, rt_hit_tri<TRAV>(F, q)};
                };
                auto trace_shadow = [&](float3 o, float3 d, float t1) -> bool {
                    RtHitRec q;
                    rt_closest_linear<TRAV, false>(F, inst, o, d, 0.0f, t1, q, st, cnt);
                    return q.li >= 0;
                };
                const float3 col = rt_shade_sample_traced(F, C, inst, hc, sm, s_lay + lane, sm.rd, hc.t, sm.tmax,
                                                          trace_eye, trace_any, trace_shadow);
                acc[0] += rt_clamp01(col.x); acc[1] += rt_clamp01(col.y); acc[2] += rt_clamp01(col.z);
            }
        }
        if (centre_ray || (odd && i == cs && j == cs)) {
            if (hli >= 0) { out_depth = ht * sm.cz; out_seg = F.geom[hli].gid; }
        }
    }
    if (!live) return;
    const size_t pix = rt_out_pixel(F, w.slot, w.zc, px, py);
    if (P.out_mask & RT_OUT_RGB) {
        const float inv = 1.0f / (float)nsamp;
        rt_store_rgb(F, pix, acc[0] * inv, acc[1] * inv, acc[2] * inv, SHADE && F.mode == RT_MODE_MJW);
    }
    if (P.out_mask & RT_OUT_DEPTH) F.depth[pix] = out_depth;
    if (P.out_mask & RT_OUT_SEG)   F.seg[pix] = out_seg;
}

#ifndef RT_SHADE_K3_WAVES_PER_EU
#define RT_SHADE_K3_WAVES_SSAA 3
#define RT_SHADE_K3_WAVES_MSAA 1
#define RT_SHADE_K3_SPEC_WAVES_SSAA 4
#define RT_SHADE_K3_SPEC_WAVES_MSAA 3
#else
#define RT_SHADE_K3_WAVES_SSAA RT_SHADE_K3_WAVES_PER_EU
#define RT_SHADE_K3_WAVES_MSAA RT_SHADE_K3_WAVES_PER_EU
#define RT_SHADE_K3_SPEC_WAVES_SSAA RT_SHADE_K3_WAVES_PER_EU
#define RT_SHADE_K3_SPEC_WAVES_MSAA RT_SHADE_K3_WAVES_PER_EU
#endif
template <bool MSAA, int SPEC>
struct RtShadeWaves {
    static constexpr int value = SPEC == RT_SPEC_DYN ? (MSAA ? RT_SHADE_K3_WAVES_MSAA : RT_SHADE_K3_WAVES_SSAA)
                                                     : (MSAA ? RT_SHADE_K3_SPEC_WAVES_MSAA : RT_SHADE_K3_SPEC_WAVES_SSAA);
};
template <bool MSAA, int SPEC = RT_SPEC_DYN>
__global__ void __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(RtShadeWaves<MSAA, SPEC>::value)))
rt_shade_kernel(RtFrame F, RtPass P)
{
    using Sp = typename RtSpecOf<SPEC>::type;
    __shared__ float s_red[64 * 3];
    const int ss = P.ss, tp = P.tp, span = tp * ss;
    const int tiles_x = (F.W + tp - 1) / tp;
    const int tiles = tiles_x * ((F.H + tp - 1) / tp);
    const RtWaveCtx w = rt_wave_ctx(F, P, tiles_x, tiles);
    if (w.slot >= F.nslot) return;
    const int lane = threadIdx.x;
    const int lx = min(lane & 7, span - 1), ly = min(lane >> 3, span - 1);
    const int pxo = lx / ss, pyo = ly / ss;
    const int i = lx - pxo * ss, j = ly - pyo * ss;
    const int px = w.tx * tp + pxo, py = w.ty * tp + pyo;
    const bool live = (lane & 7) < span && (lane >> 3) < span && px < F.W && py < F.H;
    const int qx = min(px, F.W - 1), qy = min(py, F.H - 1);

    const RtCam& C = F.cam[(size_t)w.slot * F.ncam + w.zc];
    const RtInst* __restrict__ inst = F.inst + (size_t)w.slot * F.nrg;
    const float3 eye = make_float3(C.eye[0], C.eye[1], C.eye[2]);
    const float aspect = (float)F.W / (float)F.H;
    const size_t pix = rt_out_pixel(F, w.slot, w.zc, qx, qy);
    const bool trunc = Sp::mode(F) == RT_MODE_MJW;

    if constexpr (!MSAA) {
        const int Wn = F.W * ss, Hn = F.H * ss;
        float3 rd; float cz, tmin, tmax;
        rt_cam_ray(F, C, qx * ss + i, qy * ss + j, Wn, Hn, aspect, rd, cz, tmin, tmax);
        (void)tmin;
        const size_t ri = rt_rec_index(F, P, w.slot, w.zc, false, qx, qy, i, j, ss);
        const uint4 rr = live ? P.rec[ri] : make_uint4(__float_as_uint(tmax), 0xffffffffu, 0xffffffffu, 0u);
        const RtCHit hc = rt_rec_hit(rr);
        const int nl = (int)((rr.w >> 4) & 7u);
        RtSample sm;
        sm.eye = eye; sm.rd = rd; sm.cz = cz; sm.tmax = tmax; sm.active = live;
        sm.sy = qy * ss + j; sm.Hn = Hn;
        const bool lpx = (Sp::shade(F) & RT_SH_LOD_PIXEL) != 0;
        sm.lw = lpx ? F.W : Wn; sm.lh = lpx ? F.H : Hn;
        float3 col = rt_shade_sample<Sp>(F, C, inst, hc, nl, sm,
            [&](int q) { return rt_rec_hit(P.rec_lay[(size_t)q * P.nrec + ri]); },
            [&]() { return rt_rec_hit(P.rec_mir[ri]); });
        col = make_float3(rt_clamp01(col.x), rt_clamp01(col.y), rt_clamp01(col.z));
        if (ss == 1) {
            if (live) rt_store_rgb(F, pix, col.x, col.y, col.z, trunc);
            return;
        }
        s_red[lane * 3 + 0] = col.x; s_red[lane * 3 + 1] = col.y; s_red[lane * 3 + 2] = col.z;
        __syncthreads();
        if (live && i == 0 && j == 0) {
            float acc[3] = {0.f, 0.f, 0.f};
            for (int jj = 0; jj < ss; ++jj)
                for (int ii = 0; ii < ss; ++ii) {
                    const int q = ((ly + jj) * 8 + (lx + ii)) * 3;
                    acc[0] += s_red[q + 0]; acc[1] += s_red[q + 1]; acc[2] += s_red[q + 2];
                }
            const float inv = 1.0f / (float)(ss * ss);
            rt_store_rgb(F, pix, acc[0] * inv, acc[1] * inv, acc[2] * inv, trunc);
        }
        return;
    } else {

    // MSAA4: lane = pixel (ss = 1 mapping).
    const size_t rb = rt_rec_index(F, P, w.slot, w.zc, true, qx, qy, 0, 0, 1);
    unsigned amask = 0u, dmask = 0u, srcs = 0u;
    if (live) {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned meta = P.rec[rb + k].w;
            amask |= (meta & 1u) << k;
            dmask |= ((meta >> 1) & 1u) << k;
            srcs  |= ((meta >> 2) & 3u) << (2 * k);
        }
    }
    const float3 dlc = rt_dir_at((float)qx + 0.5f, (float)qy + 0.5f, F.W, F.H, aspect, C.tanh_);
    const float czc = -dlc.z;
    const float tmaxc = F.clip ? F.zfar / czc : RT_MAXVAL;
    uint32_t q0 = 0u, q1 = 0u, q2 = 0u, q3 = 0u;
    const int njob = __popc(amask);
    int nmax = njob;
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) nmax = max(nmax, __shfl_xor(nmax, o));
#pragma unroll 1
    for (int q = 0; q < nmax; ++q) {
        // the q-th active sample of this pixel
        const bool act = q < njob;
        unsigned mk = amask;
        for (int z = 0; z < q; ++z) mk &= mk - 1u;
        const int k = act ? __ffs(mk) - 1 : 0;
        const size_t ri = rb + k;
        const uint4 rr = act ? P.rec[ri] : make_uint4(__float_as_uint(tmaxc), 0xffffffffu, 0xffffffffu, 0u);
        const RtCHit hc = rt_rec_hit(rr);
        const int nl = (int)((rr.w >> 4) & 7u);
        RtSample sm;
        sm.eye = eye; sm.active = act; sm.tmax = tmaxc;
        sm.sy = qy; sm.Hn = F.H; sm.lw = F.W; sm.lh = F.H;
        if ((dmask >> k) & 1u) {
            const float3 dl = rt_dir_at((float)qx + rt_msaa_x(k), (float)qy + rt_msaa_y(k), F.W, F.H, aspect, C.tanh_);
            sm.rd = rt_mat_vec(C.Rc, dl); sm.cz = -dl.z;
        } else {
            sm.rd = rt_mat_vec(C.Rc, dlc); sm.cz = czc;
        }
        const float3 col = rt_shade_sample<Sp>(F, C, inst, hc, nl, sm,
            [&](int qq) { return rt_rec_hit(P.rec_lay[(size_t)qq * P.nrec + ri]); },
            [&]() { return rt_rec_hit(P.rec_mir[ri]); });
        const uint32_t qc = rt_pack8(make_float3(rt_clamp01(col.x), rt_clamp01(col.y), rt_clamp01(col.z)));
        if (act) { if (k == 0) q0 = qc; else if (k == 1) q1 = qc; else if (k == 2) q2 = qc; else q3 = qc; }
    }
    if (!live) return;
    auto qsel = [&](unsigned kk) { return kk == 0u ? q0 : kk == 1u ? q1 : kk == 2u ? q2 : q3; };
    if (!((amask >> 1) & 1u)) q1 = qsel((srcs >> 2) & 3u);
    if (!((amask >> 2) & 1u)) q2 = qsel((srcs >> 4) & 3u);
    if (!((amask >> 3) & 1u)) q3 = qsel((srcs >> 6) & 3u);
    uint32_t out = 0u;
    for (int c = 0; c < 3; ++c) {
        const uint32_t sum = ((q0 >> (8 * c)) & 255u) + ((q1 >> (8 * c)) & 255u) +
                             ((q2 >> (8 * c)) & 255u) + ((q3 >> (8 * c)) & 255u);
        out |= ((sum + 2u) >> 2) << (8 * c);
    }
    uint8_t* o = F.rgb + pix * 3;
    o[0] = (uint8_t)(out & 255u); o[1] = (uint8_t)((out >> 8) & 255u); o[2] = (uint8_t)((out >> 16) & 255u);
    }
}

// ── view-cull pre-pass: one wave per (slot, camera) ─────────────────────────
// Block b -> slot slot0 + b / ncam, camera b % ncam. Writes that view's list
// of instances meeting the whole camera frustum (rt_view_cull) for K2's tiles.
__global__ void __launch_bounds__(64)
rt_view_cull_kernel(RtFrame F, int slot0, int* __restrict__ vlist, int* __restrict__ vcount)
{
    const int slot = slot0 + (int)blockIdx.x / F.ncam;
    const int zc = (int)blockIdx.x % F.ncam;
    if (slot >= F.nslot) return;
    const RtCam& C = F.cam[(size_t)slot * F.ncam + zc];
    const float3 eye = make_float3(C.eye[0], C.eye[1], C.eye[2]);
    const RtTileFrustum fr = rt_tile_frustum(C.Rc, C.tanh_, F.W, F.H, 0, F.W, 0, F.H, F.clip, F.znear, F.zfar);
    const size_t k = (size_t)slot * F.ncam + zc;
    rt_view_cull(fr, eye, F.inst + (size_t)slot * F.nrg, F.nrg, vlist + k * F.nrg, vcount + k);
}

template <int TRAV>
__global__ void __launch_bounds__(256)
rt_ray_batch_kernel(RtFrame F, const float* __restrict__ orig, const float* __restrict__ dir,
                    int n, int* __restrict__ gid, float* __restrict__ tout, float* __restrict__ nrm)
{
    // 64-bit element offsets: k*3 overflows int beyond ~715M rays.
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (size_t)n) return;
    const float3 o = make_float3(orig[k*3+0], orig[k*3+1], orig[k*3+2]);
    const float3 d = make_float3(dir[k*3+0], dir[k*3+1], dir[k*3+2]);
    RtHitRec hr;
    RtCounters cnt;
    RtRegStack st;   // independent per-thread rays: no LDS layout to share
    rt_closest_linear<TRAV, false>(F, F.inst, o, d, 0.0f, RT_MAXVAL, hr, st, cnt);
    const RtHit h = rt_resolve_hit<TRAV>(F, F.inst, hr, o, d, 0.0f, RT_MAXVAL);
    if (h.li >= 0) {
        const float3 N = rt_world_normal(F.inst[h.li], h.nl);
        gid[k] = F.geom[h.li].gid; tout[k] = h.t;
        nrm[k*3+0] = N.x; nrm[k*3+1] = N.y; nrm[k*3+2] = N.z;
    } else {
        gid[k] = -1; tout[k] = -1.0f;
        nrm[k*3+0] = nrm[k*3+1] = nrm[k*3+2] = 0.0f;
    }
}

__global__ void __launch_bounds__(256)
rt_debug_rays_kernel(RtFrame F, float* __restrict__ dirs)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= F.W * F.H) return;   // W*H*3 floats is host-allocated, so it fits int
    const int px = k % F.W, py = k / F.W;
    const RtCam& C = F.cam[0];
    const float3 dl = rt_sample_dir(px, py, F.W, F.H, (float)F.W / (float)F.H, C.tanh_);
    const float3 rd = rt_mat_vec(C.Rc, dl);
    dirs[k*3+0] = rd.x; dirs[k*3+1] = rd.y; dirs[k*3+2] = rd.z;
}

template <int TRAV, bool TC, int SPEC>
void rt_launch_fused_spec(const RtFrame& F, const RtPass& P, unsigned nb, hipStream_t stream)
{
    const int span = P.tp * P.ss;
    if (TC && span * span == 16 && (P.ss == 1 || P.msaa))
        hipLaunchKernelGGL((rt_primary_kernel<TRAV, false, TC, RT_K2_FUSED, SPEC, 4>), dim3(nb), dim3(64), 0, stream, F, P);
#if RT_GRP_TP2
    else if (TC && span * span == 4 && P.ss == 1)
        hipLaunchKernelGGL((rt_primary_kernel<TRAV, false, TC, RT_K2_FUSED, SPEC, RT_GRP_TP2>), dim3(nb), dim3(64), 0, stream, F, P);
#endif
    else
        hipLaunchKernelGGL((rt_primary_kernel<TRAV, false, TC, RT_K2_FUSED, SPEC>), dim3(nb), dim3(64), 0, stream, F, P);
}

template <int TRAV, bool STATS, bool TC>
void rt_launch_fused(const RtFrame& F, const RtPass& P, int spec, unsigned nb, hipStream_t stream)
{
    if constexpr (!STATS) {
        if (spec == RT_SPEC_MUJOCO) { rt_launch_fused_spec<TRAV, TC, RT_SPEC_MUJOCO>(F, P, nb, stream); return; }
        if (spec == RT_SPEC_RASTER_LIN_LAY) { rt_launch_fused_spec<TRAV, TC, RT_SPEC_RASTER_LIN_LAY>(F, P, nb, stream); return; }
        rt_launch_fused_spec<TRAV, TC, RT_SPEC_DYN>(F, P, nb, stream);
        return;
    }
    hipLaunchKernelGGL((rt_primary_kernel<TRAV, STATS, TC, RT_K2_FUSED>), dim3(nb), dim3(64), 0, stream, F, P);
}

// shade: 0 geometry only, 1 fused full shading, 2 deferred (trace pass, then
// rt_shade_kernel over its records).
template <int TRAV, bool STATS, bool TC>
void rt_launch_pass(const RtFrame& F, const RtPass& P, int shade, hipStream_t stream)
{
    const int tiles = ((F.W + P.tp - 1) / P.tp) * ((F.H + P.tp - 1) / P.tp);
    const unsigned nb = (unsigned)tiles * (unsigned)P.nslot * (unsigned)F.ncam;
    if (!shade || !(P.out_mask & RT_OUT_RGB)) {
        hipLaunchKernelGGL((rt_primary_kernel<TRAV, STATS, TC, RT_K2_GEOM>), dim3(nb), dim3(64), 0, stream, F, P);
    } else if (shade == 1) {
        // Preset-specialised fused kernels for the counter-free variants only.
        int spec = RT_SPEC_DYN;
        if constexpr (!STATS) spec = rt_pick_spec(F);
        rt_launch_fused<TRAV, STATS, TC>(F, P, spec, nb, stream);
    } else {
        hipLaunchKernelGGL((rt_primary_kernel<TRAV, STATS, TC, RT_K2_TRACE>), dim3(nb), dim3(64), 0, stream, F, P);
        RtPass Q = P;
        Q.swizzle = 0;
        if (P.msaa) { Q.ss = 1; Q.tp = RT_TILE; }
        const int t3 = ((F.W + Q.tp - 1) / Q.tp) * ((F.H + Q.tp - 1) / Q.tp);
        const unsigned nb3 = (unsigned)t3 * (unsigned)P.nslot * (unsigned)F.ncam;
        const int spec = rt_pick_spec(F);
        if (P.msaa) {
            if (spec == RT_SPEC_MUJOCO)
                hipLaunchKernelGGL((rt_shade_kernel<true, RT_SPEC_MUJOCO>), dim3(nb3), dim3(64), 0, stream, F, Q);
            else if (spec == RT_SPEC_RASTER_LIN_LAY)
                hipLaunchKernelGGL((rt_shade_kernel<true, RT_SPEC_RASTER_LIN_LAY>), dim3(nb3), dim3(64), 0, stream, F, Q);
            else
                hipLaunchKernelGGL((rt_shade_kernel<true>), dim3(nb3), dim3(64), 0, stream, F, Q);
        } else {
            if (spec == RT_SPEC_MUJOCO)
                hipLaunchKernelGGL((rt_shade_kernel<false, RT_SPEC_MUJOCO>), dim3(nb3), dim3(64), 0, stream, F, Q);
            else if (spec == RT_SPEC_RASTER_LIN_LAY)
                hipLaunchKernelGGL((rt_shade_kernel<false, RT_SPEC_RASTER_LIN_LAY>), dim3(nb3), dim3(64), 0, stream, F, Q);
            else
                hipLaunchKernelGGL((rt_shade_kernel<false>), dim3(nb3), dim3(64), 0, stream, F, Q);
        }
    }
}

inline int rt_pick_tp1(const RtFrame& F, int nslot, int max_waves) {
    if (max_waves <= 0) return RT_TILE;
    for (int tp = 2; tp < RT_TILE; tp *= 2) {
        const long long waves = (long long)((F.W + tp - 1) / tp) * ((F.H + tp - 1) / tp) * nslot * F.ncam;
        if (waves <= max_waves) return tp;
    }
    return RT_TILE;
}

inline int rt_pick_tp_msaa(const RtFrame& F, int nslot, int max_waves) {
    if (max_waves <= 0) return RT_TILE / 2;
    const long long waves = (long long)((F.W + 1) / 2) * ((F.H + 1) / 2) * nslot * F.ncam;
    return waves <= max_waves ? 2 : RT_TILE / 2;
}

template <int TRAV, bool STATS, bool TC>
void rt_launch_primary_t(const RtFrame& F, int slot0, int nslot_chunk, int swizzle, int max_waves,
                         int* vlist, int* vcount, const RtLaunchOpts& opt, hipStream_t stream)
{
    RtPass P;
    P.slot0 = slot0; P.nslot = nslot_chunk; P.swizzle = swizzle; P.msaa = 0;
    P.vlist = nullptr; P.vcount = nullptr;
    P.rec = opt.rec; P.rec_mir = opt.rec_mir; P.rec_lay = opt.rec_lay; P.nrec = opt.nrec;
    if (TC && vlist && vcount) {
        if (!opt.vlist_ready)
            hipLaunchKernelGGL(rt_view_cull_kernel, dim3((unsigned)nslot_chunk * (unsigned)F.ncam), dim3(64), 0,
                               stream, F, slot0, vlist, vcount);
        P.vlist = vlist; P.vcount = vcount;
    }
    const bool shading = (F.out_mask & RT_OUT_RGB) && F.mode != RT_MODE_GREY;
    // deferred split needs record buffers (sized by the host for this chunk)
    int shade = !shading ? 0 : (opt.split && opt.rec && opt.rec_mir) ? 2 : 1;
    if (shade == 2 && opt.split_auto_waves > 0 && F.ss <= RT_TILE) {
        if (F.aa != RT_AA_MSAA4 && F.ss == 1) {
            if (nslot_chunk == 1) shade = 1;
        } else {
            const int tp = F.aa == RT_AA_MSAA4 ? RT_TILE / 2 : RT_TILE / F.ss;
            const long long nb = (long long)((F.W + tp - 1) / tp) * ((F.H + tp - 1) / tp) * nslot_chunk * F.ncam;
            if (nb <= opt.split_auto_waves) shade = 1;
        }
    }
    const int ds = F.out_mask & (RT_OUT_DEPTH | RT_OUT_SEG);
    if (shade && F.aa == RT_AA_MSAA4) {
        // MSAA4 (init ssaa = 1): RGB from the 4 coverage samples (ss = 2 lane
        // layout), depth/seg from a separate pixel-centre geometry pass.
        // The deferred split keeps 4x4-pixel tiles (record layout and its
        // shade pass are tile independent, but its launches are large).
        P.ss = 2; P.tp = shade == 1 ? rt_pick_tp_msaa(F, nslot_chunk, max_waves) : RT_TILE / 2;
        P.out_mask = RT_OUT_RGB; P.msaa = 1;
        rt_launch_pass<TRAV, STATS, TC>(F, P, shade, stream);
        P.msaa = 0;
        if (ds) {
            P.ss = 1; P.tp = rt_pick_tp1(F, nslot_chunk, max_waves); P.out_mask = ds;
            rt_launch_pass<TRAV, STATS, TC>(F, P, 0, stream);
        }
        return;
    }
    if (F.ss > RT_TILE) {
        // Samples of a pixel exceed a wave: per-lane sample loop (tile cull, no counters).
        P.ss = F.ss; P.tp = RT_TILE; P.out_mask = F.out_mask;
        const int tiles = ((F.W + RT_TILE - 1) / RT_TILE) * ((F.H + RT_TILE - 1) / RT_TILE);
        const unsigned nb = (unsigned)tiles * (unsigned)P.nslot * (unsigned)F.ncam;
        if (shade) hipLaunchKernelGGL((rt_primary_loop_kernel<TRAV, true>), dim3(nb), dim3(64), 0, stream, F, P);
        else       hipLaunchKernelGGL((rt_primary_loop_kernel<TRAV, false>), dim3(nb), dim3(64), 0, stream, F, P);
        return;
    }
    if (F.ss == 1 || (F.ss & 1)) {
        // One pass; odd ss takes depth/seg from the centre sample.
        P.ss = F.ss; P.tp = F.ss == 1 ? rt_pick_tp1(F, nslot_chunk, max_waves) : RT_TILE / F.ss;
        P.out_mask = F.out_mask;
        rt_launch_pass<TRAV, STATS, TC>(F, P, shade, stream);
        return;
    }
    if (F.out_mask & RT_OUT_RGB) {
        P.ss = F.ss; P.tp = RT_TILE / F.ss; P.out_mask = RT_OUT_RGB;
        rt_launch_pass<TRAV, STATS, TC>(F, P, shade, stream);
    }
    if (ds) {
        P.ss = 1; P.tp = rt_pick_tp1(F, nslot_chunk, max_waves); P.out_mask = ds;
        rt_launch_pass<TRAV, STATS, TC>(F, P, 0, stream);
    }
}

}  // namespace

int rt_primary_max_slots(int W, int H, int ss, int ncam)
{
    // 1-D grid: keep blocks * 64 threads below 2^31. The tile is tp x tp
    // pixels with tp = 8 / ss (sample-per-lane pass), or 8 for ss > 8. (The
    // small-batch tiles of rt_pick_tp1 only apply when the launch is small.)
    const int tp = (ss >= 1 && ss <= RT_TILE) ? RT_TILE / ss : RT_TILE;
    const long long tiles = (long long)((W + tp - 1) / tp) * ((H + tp - 1) / tp);
    const long long per = tiles * std::max(ncam, 1);
    const long long cap = ((1LL << 31) / 64) / std::max(per, 1LL);
    return (int)std::max(1LL, std::min(cap, (long long)INT_MAX));
}

void rt_launch_primary(const RtFrame& F, int slot0, int nslot_chunk, const RtLaunchOpts& opt,
                       hipStream_t stream)
{
    if (nslot_chunk <= 0 || F.ncam <= 0 || F.W <= 0 || F.H <= 0) return;
    const int sw = opt.swizzle ? 1 : 0;
#define RT_LAUNCH(TRAV, ST, TC) rt_launch_primary_t<TRAV, ST, TC>(F, slot0, nslot_chunk, sw, opt.max_waves, \
                                                                   opt.vlist, opt.vcount, opt, stream)
    if (F.trav == RT_TRAV_ORDERED) {
        if (opt.tile_cull) { if (opt.stats) RT_LAUNCH(RT_TRAV_ORDERED, true, true);  else RT_LAUNCH(RT_TRAV_ORDERED, false, true); }
        else               { if (opt.stats) RT_LAUNCH(RT_TRAV_ORDERED, true, false); else RT_LAUNCH(RT_TRAV_ORDERED, false, false); }
    } else {
        if (opt.tile_cull) { if (opt.stats) RT_LAUNCH(RT_TRAV_SKIP, true, true);  else RT_LAUNCH(RT_TRAV_SKIP, false, true); }
        else               { if (opt.stats) RT_LAUNCH(RT_TRAV_SKIP, true, false); else RT_LAUNCH(RT_TRAV_SKIP, false, false); }
    }
#undef RT_LAUNCH
}

void rt_launch_ray_batch(const RtFrame& F, const float* orig, const float* dir, int n,
                         int* gid, float* t, float* nrm, hipStream_t stream)
{
    if (n <= 0) return;
    const int block = 256;
    const dim3 grid((n + block - 1) / block);
    if (F.trav == RT_TRAV_ORDERED)
        hipLaunchKernelGGL((rt_ray_batch_kernel<RT_TRAV_ORDERED>), grid, dim3(block), 0, stream,
                           F, orig, dir, n, gid, t, nrm);
    else
        hipLaunchKernelGGL((rt_ray_batch_kernel<RT_TRAV_SKIP>), grid, dim3(block), 0, stream,
                           F, orig, dir, n, gid, t, nrm);
}

void rt_launch_debug_rays(const RtFrame& F, float* dirs, hipStream_t stream)
{
    const int n = F.W * F.H;
    if (n <= 0) return;
    const int block = 256;
    hipLaunchKernelGGL(rt_debug_rays_kernel, dim3((n + block - 1) / block), dim3(block), 0, stream,
                       F, dirs);
}

}  // namespace physics
}  // namespace hipgl
