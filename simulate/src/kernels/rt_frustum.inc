#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rt_math.inc"

namespace hipgl {
namespace physics {

// Pin a wave-uniform int to an SGPR (readfirstlane) so divergence analysis
// sees it as uniform.
__device__ __forceinline__ int rt_uniform_i(int x) { return __builtin_amdgcn_readfirstlane(x); }

static_assert(MAX_GEOMS <= 2048, "rt_frustum: li must fit the 11-bit field of a candidate entry");

constexpr int      RT_CAND_LI_BITS = 11;
constexpr unsigned RT_CAND_LI_MASK = (1u << RT_CAND_LI_BITS) - 1u;

struct RtTileFrustum {
    float3 n[4];        // inward unit side-plane normals (world), planes pass through the eye
    float3 fwd;         // unit camera forward (world) = -Rc[:,2]
    float  dmin, dmax;  // keep boxes whose planar depth range meets [dmin, dmax]
};

__device__ __forceinline__ RtTileFrustum rt_tile_frustum(const float Rc[9], float tanh_, int W, int H,
                                                         int x0, int x1, int y0, int y1,
                                                         int clip, float znear, float zfar)
{
    const float aspect = (float)W / (float)H;
    const float xl = (2.0f * (float)x0 / (float)W - 1.0f) * tanh_ * aspect;
    const float xr = (2.0f * (float)x1 / (float)W - 1.0f) * tanh_ * aspect;
    const float yt = (1.0f - 2.0f * (float)y0 / (float)H) * tanh_;
    const float yb = (1.0f - 2.0f * (float)y1 / (float)H) * tanh_;
    RtTileFrustum f;
    f.n[0] = rt_norm(rt_mat_vec(Rc, make_float3( 1.0f, 0.0f,  xl)));
    f.n[1] = rt_norm(rt_mat_vec(Rc, make_float3(-1.0f, 0.0f, -xr)));
    f.n[2] = rt_norm(rt_mat_vec(Rc, make_float3( 0.0f, -1.0f, -yt)));
    f.n[3] = rt_norm(rt_mat_vec(Rc, make_float3( 0.0f, 1.0f,  yb)));
    f.fwd = make_float3(-Rc[2], -Rc[5], -Rc[8]);
    // Planar-depth window, relaxed so rounding of the clip interval
    // (tmin = znear/cz, tmax = zfar/cz) can never strand a hit.
    f.dmin = clip ? znear * (1.0f - 1e-4f) - 1e-6f : -1e-6f;
    f.dmax = clip ? zfar * (1.0f + 1e-4f) + 1e-6f : INFINITY;
    return f;
}

// True if the (padded) world AABB [lo, hi] may contain a point some ray of the
// tile reaches. Box max over a plane: n.(c - eye) + |n|.e. Hidden sites (the
// point box at 1e30 of rt_inst_kernel) never pass: a frustum containing the
// (1,1,1) direction would otherwise list every hidden site.
__device__ __forceinline__ bool rt_box_in_frustum(const RtTileFrustum& f, float3 eye, float3 lo, float3 hi)
{
    const float3 c = rt_mul(rt_add(lo, hi), 0.5f);
    float3 e = rt_mul(rt_sub(hi, lo), 0.5f);
    e = make_float3(e.x * 1.0001f + 1e-5f, e.y * 1.0001f + 1e-5f, e.z * 1.0001f + 1e-5f);
    const float3 d = rt_sub(c, eye);
    bool in = lo.x < 1e29f;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        in = in && (rt_dot(f.n[k], d) + rt_dot(rt_absf3(f.n[k]), e) >= 0.0f);
    const float dc = rt_dot(f.fwd, d), de = rt_dot(rt_absf3(f.fwd), e);
    return in && (dc + de >= f.dmin) && (dc - de <= f.dmax);
}

// Packed candidate entry: high 21 bits of the (non-negative) key's float bits,
// i.e. the key truncated toward zero, then li. Ascending u32 order is
// (truncated key, li); the decoded key never exceeds the true one.
__device__ __forceinline__ unsigned rt_cand_pack(float3 eye, float3 lo, float3 hi, int li)
{
    const float3 q = rt_sub(rt_max3(lo, rt_min3(eye, hi)), eye);
    const float key = rt_len(q) * 0.9999f;
    const unsigned kb = (unsigned)__float_as_int(key) & 0x7FFFFFFFu;   // +-0 -> 0
    return ((kb >> (31 - (32 - RT_CAND_LI_BITS))) << RT_CAND_LI_BITS) | (unsigned)li;
}
__device__ __forceinline__ float rt_cand_key(unsigned e) {
    return __int_as_float((int)((e >> RT_CAND_LI_BITS) << (31 - (32 - RT_CAND_LI_BITS))));
}
__device__ __forceinline__ int rt_cand_li(unsigned e) { return (int)(e & RT_CAND_LI_MASK); }

// Cull the instances list[first, n) (list == nullptr: li = position, i.e.
// all of [first, n)) of one world against the tile frustum and append
// survivors to cand[] (LDS) until it would overflow `cap`. Whole 64-entry
// groups are taken or left, so cap must be >= 64. Returns the survivor count;
// *next = first position not examined (== n when everything fit). Wave-uniform.
template <bool STATS>
__device__ __forceinline__ int rt_tile_cull(const RtTileFrustum& f, float3 eye, const RtInst* __restrict__ inst,
                                            const int* __restrict__ list, int first, int nrg,
                                            unsigned* cand, int cap, int* next)
{
    const int lane = threadIdx.x & 63;
    int n = 0;
    int li0 = first;
    for (; li0 < nrg; li0 += 64) {
        const int pos = li0 + lane;
        bool pass = false;
        unsigned ent = 0u;
        if (pos < nrg) {
            const int li = list ? list[pos] : pos;
            const RtInst& I = inst[li];
            const float3 lo = make_float3(I.lo[0], I.lo[1], I.lo[2]);
            const float3 hi = make_float3(I.hi[0], I.hi[1], I.hi[2]);
            pass = rt_box_in_frustum(f, eye, lo, hi);
            if (pass) ent = rt_cand_pack(eye, lo, hi, li);
        }
        const unsigned long long m = __ballot(pass);
        // Wave-uniform by construction; readfirstlane makes it provably so for
        // the compiler (otherwise n / li0 / the returned count live in VGPRs and
        // every consumer of the list, i.e. the candidate loop, turns divergent).
        const int cnt = rt_uniform_i(__popcll(m));
        if (n + cnt > cap) break;                       // wave-uniform
        if (pass) {
            const int at = n + (int)__builtin_amdgcn_mbcnt_hi((unsigned)(m >> 32),
                                      __builtin_amdgcn_mbcnt_lo((unsigned)m, 0u));
            cand[at] = ent;
        }
        n = rt_uniform_i(n + cnt);
    }
    *next = rt_uniform_i(li0 < nrg ? li0 : nrg);
    return rt_uniform_i(n);
}

// Per-(world, camera) view cull: every instance whose padded AABB may meet the
// camera's whole view frustum (the tile frustum of the full image window, so
// it contains every tile's frustum), compacted in ascending li into
// list[0, *count). One wave; lanes stride over li. Wave-uniform.
__device__ __forceinline__ void rt_view_cull(const RtTileFrustum& f, float3 eye, const RtInst* __restrict__ inst,
                                             int nrg, int* __restrict__ list, int* __restrict__ count)
{
    const int lane = threadIdx.x & 63;
    int n = 0;
    for (int li0 = 0; li0 < nrg; li0 += 64) {
        const int li = li0 + lane;
        bool pass = false;
        if (li < nrg) {
            const RtInst& I = inst[li];
            pass = rt_box_in_frustum(f, eye, make_float3(I.lo[0], I.lo[1], I.lo[2]),
                                     make_float3(I.hi[0], I.hi[1], I.hi[2]));
        }
        const unsigned long long m = __ballot(pass);
        if (pass)
            list[n + (int)__builtin_amdgcn_mbcnt_hi((unsigned)(m >> 32),
                                                     __builtin_amdgcn_mbcnt_lo((unsigned)m, 0u))] = li;
        n += __popcll(m);
    }
    if (lane == 0) *count = n;
}

// Ascending in-place sort of cand[0, n) (distinct entries, n <= CAP) by rank
// counting: every lane ranks its <= CAP/64 entries against all n (O(n^2/64)
// LDS broadcast reads; n is a few tens for an 8x8 tile), then scatters them.
// All ranks are computed before the first write, so no scratch list is needed.
template <int CAP>
__device__ __forceinline__ void rt_cand_sort(unsigned* cand, int n)
{
    constexpr int Q = CAP / 64;
    const int lane = threadIdx.x & 63;
    __syncthreads();
    unsigned e[Q];
    int r[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const int i = lane + 64 * q;
        e[q] = i < n ? cand[i] : 0xFFFFFFFFu;
        r[q] = 0;
    }
    for (int j = 0; j < n; ++j) {
        const unsigned c = cand[j];
#pragma unroll
        for (int q = 0; q < Q; ++q) r[q] += c < e[q] ? 1 : 0;
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < Q; ++q)
        if (lane + 64 * q < n) cand[r[q]] = e[q];
    __syncthreads();
}

}  // namespace physics
}  // namespace hipgl
