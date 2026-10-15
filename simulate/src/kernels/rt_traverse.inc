#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hipgl/physics/rt_types.h"
#include "rt_intersect.inc"

namespace hipgl {
namespace physics {

RT_HD inline int rt_trav_f2i(float f) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __float_as_int(f);
#else
    union { float f; int i; } u; u.f = f; return u.i;
#endif
}

RT_HD inline float3 rt_trav_xyz(float4 a) { return make_float3(a.x, a.y, a.z); }

// Slab tests of both children of an RtNode64 at once: the subtract/multiply
// run on (child 0, child 1) pairs (packed FP32 on gfx9x0), every element with
// exactly rt_slab's operations, so t0/t1 and the hit flags are bit-identical to
// two rt_slab calls.
typedef float rt_f2 __attribute__((ext_vector_type(2)));
RT_HD inline void rt_slab2(float3 o, float3 iv, const RtNode64& N, float tmin, float tmax,
                           float& t0, float& t1, bool& h0, bool& h1) {
    const rt_f2 lx = {N.lo[0][0], N.lo[0][1]}, ly = {N.lo[1][0], N.lo[1][1]}, lz = {N.lo[2][0], N.lo[2][1]};
    const rt_f2 hx = {N.hi[0][0], N.hi[0][1]}, hy = {N.hi[1][0], N.hi[1][1]}, hz = {N.hi[2][0], N.hi[2][1]};
    const rt_f2 ax = (lx - o.x) * iv.x, bx = (hx - o.x) * iv.x;
    const rt_f2 ay = (ly - o.y) * iv.y, by = (hy - o.y) * iv.y;
    const rt_f2 az = (lz - o.z) * iv.z, bz = (hz - o.z) * iv.z;
    t0 = fmaxf(fmaxf(fminf(ax.x, bx.x), fminf(ay.x, by.x)), fmaxf(fminf(az.x, bz.x), tmin));
    const float f0 = fminf(fminf(fmaxf(ax.x, bx.x), fmaxf(ay.x, by.x)), fminf(fmaxf(az.x, bz.x), tmax));
    t1 = fmaxf(fmaxf(fminf(ax.y, bx.y), fminf(ay.y, by.y)), fmaxf(fminf(az.y, bz.y), tmin));
    const float f1 = fminf(fminf(fmaxf(ax.y, bx.y), fmaxf(ay.y, by.y)), fminf(fmaxf(az.y, bz.y), tmax));
    h0 = t0 <= f0;
    h1 = t1 <= f1;
}

// SKIP: mujoco_warp's stackless pre-order walk (render.py _ray_mesh_bvh).
//   node hit + internal -> node + 1;  node hit + leaf -> test tri, then esc;
//   node miss           -> esc;       esc == -1 ends.
// "Hit" is mjw's tfar >= tnear && tnear < best_t with tnear clamped to tmin;
// triangles are accepted on t < best_t (strict), so on equal t the first one
// in leaf order wins, exactly as in mjw.
// allow_eq / tri_ref: as for rt_blas_walk_ordered_s below. allow_eq is the
// strict bound moved up by one ulp (t < next_up(best_t) <=> t <= best_t); the
// first improvement inside the mesh replaces it with that t, so the in-mesh
// tie rule (first in leaf order) is unchanged.
template <bool ANY>
RT_HD inline bool rt_blas_walk_skip_s(const RtNode32* nodes, const RtTri* tris, const RtBlas& b,
                                      float3 lp, float3 lv, float tmin, int cull, bool allow_eq,
                                      float& best_t, int& tri, int& tri_ref, float& u, float& v, float3& ng,
                                      unsigned& nodes_visited, unsigned& tris_tested) {
    if (b.ntri <= 0) return false;
    float3 b0, b1;
    rt_tri_basis(lv, b0, b1);
    const float3 iv = rt_safe_inv(lv);
    bool found = false;
    float bound = allow_eq ? nextafterf(best_t, INFINITY) : best_t;
    int node = b.node_off;
    while (node >= 0) {
        const RtNode32 N = nodes[node];
        nodes_visited++;
        int next = N.esc;
        float tn;
        if (rt_slab(lp, iv, make_float3(N.lo[0], N.lo[1], N.lo[2]), make_float3(N.hi[0], N.hi[1], N.hi[2]),
                    tmin, bound, tn) && tn < bound) {
            if (N.payload >= 0) {
                const RtTri T = tris[N.payload];
                tris_tested++;
                float t, tu, tv; float3 n;
                if (rt_tri_mjw(lp, lv, b0, b1, rt_trav_xyz(T.v0), rt_trav_xyz(T.v1), rt_trav_xyz(T.v2),
                               tmin, t, tu, tv, n) &&
                    t < bound && !(cull == RT_CULL_PER_TRI && rt_dot(lv, n) > 0.f)) {
                    bound = t; tri = rt_trav_f2i(T.v0.w); tri_ref = N.payload; u = tu; v = tv; ng = n;
                    found = true;
                    if (ANY) break;
                }
            } else {
                next = node + 1;
            }
        }
        node = next;
    }
    if (found) best_t = bound;
    return found;
}

template <bool ANY>
RT_HD inline bool rt_blas_walk_skip(const RtNode32* nodes, const RtTri* tris, const RtBlas& b,
                                    float3 lp, float3 lv, float tmin, int cull,
                                    float& best_t, int& tri, float& u, float& v, float3& ng,
                                    unsigned& nodes_visited, unsigned& tris_tested) {
    int tri_ref = -1;
    return rt_blas_walk_skip_s<ANY>(nodes, tris, b, lp, lv, tmin, cull, false, best_t, tri, tri_ref,
                                    u, v, ng, nodes_visited, tris_tested);
}

struct RtRegStack {
    int   node[RT_STACK];
    float t[RT_STACK];
    int   sp = 0;
    RT_HD void reset(const RtBlas&) { sp = 0; }
    RT_HD void reset_wide(const RtBlas&) { sp = 0; }
    RT_HD void push(int n, float tn) { if (sp < RT_STACK) { node[sp] = n; t[sp] = tn; sp++; } }
    // Most recent entry with t <= best_t (dropping the others), or -1.
    RT_HD int pop(float best_t) {
        while (sp > 0) {
            sp--;
            if (!(t[sp] > best_t)) return node[sp];
        }
        return -1;
    }
};

#if defined(__HIP_DEVICE_COMPILE__)
struct RtLdsStack {
    unsigned* base;     // this lane's column: &lds[lane]
    int sp, off, tb;    // tb = 32 - nbits t bits; off = mesh's node64_off
    __device__ void reset(const RtBlas& b) { sp = 0; off = b.node64_off; tb = 32 - b.nbits; }
    // BVH4 walk: mesh-local RtNode4 index (a mesh has <= as many as RtNode64, so nbits fits).
    __device__ void reset_wide(const RtBlas& b) { sp = 0; off = b.node4_off; tb = 32 - b.nbits; }
    __device__ void push(int n, float tn) {
        if (sp < RT_STACK) {
            const unsigned tq = ((unsigned)__float_as_int(tn) & 0x7FFFFFFFu) >> (31 - tb);
            base[sp * 64] = ((unsigned)(n - off) << tb) | tq;
            sp++;
        }
    }
    __device__ int pop(float best_t) {
        while (sp > 0) {
            sp--;
            const unsigned e = base[sp * 64];
            const float tl = __int_as_float((int)((e & ((1u << tb) - 1u)) << (31 - tb)));
            if (!(tl > best_t)) return (int)(e >> tb) + off;
        }
        return -1;
    }
};
#endif

// ORDERED: two-child (RtNode64) walk, nearer child first. Both child boxes are
// slab tested; leaf children are intersected in place (nearer first), the
// nearer internal child is descended and a farther internal one pushed with its
// entry distance. A child or stack entry is pruned when entry > best_t (entries
// equal to best_t are still visited so the tie rule is exact). Closest hit is
// the lexicographic minimum of (t, SKIP leaf rank) over this mesh (rank = RtTri
// v1.w), i.e. the triangle the SKIP walk keeps on equal t.
//
// allow_eq: also accept a triangle at exactly the incoming best_t (the caller's
// lexicographic (t, li) merge, when this mesh's li is below the current best
// li); otherwise a hit must be strictly closer than the incoming best_t.
// tri_ref (optional) = global RtTri index of the hit (tris64), for re-deriving
// u/v/ng later without keeping them live.
//
// GPU shape: the node is fetched whole and unconditionally, both leaf children
// share one flat triangle loop, and the descend/push/pop decision is
// straight-line, which keeps the divergent control flow (and its exec-mask
// bookkeeping) small.
template <bool ANY, typename Stack>
RT_HD inline bool rt_blas_walk_ordered_s(const RtNode64* nodes, const RtTri* tris64, const RtBlas& b,
                                         float3 lp, float3 lv, float tmin, int cull, bool allow_eq,
                                         Stack& st, float& best_t, int& tri, int& tri_ref,
                                         float& u, float& v, float3& ng,
                                         unsigned& nodes_visited, unsigned& tris_tested) {
    if (b.ntri <= 0) return false;
    float3 b0, b1;
    rt_tri_basis(lv, b0, b1);
    const float3 iv = rt_safe_inv(lv);
    bool found = false;
    st.reset(b);
    int node = b.node64_off;
    // Lexicographic (t, SKIP rank) bound. rank 0 makes t == the incoming best_t
    // lose (strictly closer required); rank 0xFFFFFFFF lets it win (allow_eq).
    unsigned long long best_key = ((unsigned long long)(unsigned)rt_trav_f2i(best_t + 0.0f) << 32) |
                                  (allow_eq ? 0xFFFFFFFFull : 0ull);

    while (node >= 0) {
        // Whole node, unconditionally (one memory round trip). The unused root
        // child of a one-leaf mesh (c < 0) carries the EMPTY sentinel box at
        // 1e30, which no ray reaches before t ~ 1e30 * |1/d|; the c >= 0 term
        // is a non-short-circuit AND so it adds no dependent load.
        const RtNode64 N = nodes[node];
        nodes_visited++;
        float t0, t1;
        bool h0, h1;
        rt_slab2(lp, iv, N, tmin, best_t, t0, t1, h0, h1);
        h0 = h0 & (N.c[0] >= 0);
        h1 = h1 & (N.c[1] >= 0);
        // Order (near, far); ties keep child 0 first.
        const bool swap = h0 && h1 && t1 < t0;
        const int   cn = swap ? N.c[1] : N.c[0], nn_ = swap ? N.n[1] : N.n[0];
        const int   cf = swap ? N.c[0] : N.c[1], nf  = swap ? N.n[0] : N.n[1];
        const float tn = swap ? t1 : t0,     tf  = swap ? t0 : t1;
        const bool  hn = swap ? h1 : h0,     hf  = swap ? h0 : h1;
        const bool in_ = hn && nn_ == 0, if_ = hf && nf == 0;   // internal children hit
        if (in_ && if_) {
            // Both internal (so no leaf can have moved best_t): stack the farther.
            st.push(cf, tf);
            node = cn;
            continue;
        }
        // At most one internal child (pc, pt), visited after the leaves unless a
        // leaf hit moved best_t in front of it. Leaf children share one flat
        // triangle loop, near leaf first; the far leaf is dropped once a near
        // leaf hit moved best_t in front of it.
        const int   pc = in_ ? cn : (if_ ? cf : -1);
        const float pt = in_ ? tn : tf;
        const int ln = (hn && nn_ > 0) ? nn_ : 0;
        const int nl = ln + ((hf && nf > 0) ? nf : 0);
        const int fb = cf - ln;                                  // far leaf: index = fb + k
        for (int k = 0; k < nl; k++) {
            const bool far_leaf = k >= ln;
            if (far_leaf && tf > best_t) break;
            const int ti = far_leaf ? fb + k : cn + k;
            const RtTri T = tris64[ti];
            tris_tested++;
            float t, tu, tv; float3 nn;
            if (!rt_tri_mjw(lp, lv, b0, b1, rt_trav_xyz(T.v0), rt_trav_xyz(T.v1), rt_trav_xyz(T.v2),
                            tmin, t, tu, tv, nn))
                continue;
            if (cull == RT_CULL_PER_TRI && rt_dot(lv, nn) > 0.f) continue;
            // (t, rank) as one 64-bit key: t >= tmin >= 0, so its bit pattern
            // orders like the float (+0.0f folds a -0.0 into +0).
            const unsigned long long key =
                ((unsigned long long)(unsigned)rt_trav_f2i(t + 0.0f) << 32) | (unsigned)rt_trav_f2i(T.v1.w);
            if (key < best_key) {
                best_key = key;
                best_t = t; tri = rt_trav_f2i(T.v0.w); tri_ref = ti;
                u = tu; v = tv; ng = nn;
                found = true;
                if (ANY) return true;
            }
        }
        node = (pc >= 0 && !(pt > best_t)) ? pc : st.pop(best_t);
    }
    return found;
}

// Slab tests of the four children of an RtNode4: packed FP32 on the (0,1) and
// (2,3) child pairs, every element with exactly rt_slab's operations.
RT_HD inline void rt_slab4(float3 o, float3 iv, const RtNode4& N, float tmin, float tmax,
                           float t[4], bool h[4]) {
#pragma unroll
    for (int p = 0; p < 2; p++) {
        const int k = 2 * p;
        const rt_f2 lx = {N.lo[0][k], N.lo[0][k + 1]}, ly = {N.lo[1][k], N.lo[1][k + 1]}, lz = {N.lo[2][k], N.lo[2][k + 1]};
        const rt_f2 hx = {N.hi[0][k], N.hi[0][k + 1]}, hy = {N.hi[1][k], N.hi[1][k + 1]}, hz = {N.hi[2][k], N.hi[2][k + 1]};
        const rt_f2 ax = (lx - o.x) * iv.x, bx = (hx - o.x) * iv.x;
        const rt_f2 ay = (ly - o.y) * iv.y, by = (hy - o.y) * iv.y;
        const rt_f2 az = (lz - o.z) * iv.z, bz = (hz - o.z) * iv.z;
        t[k] = fmaxf(fmaxf(fminf(ax.x, bx.x), fminf(ay.x, by.x)), fmaxf(fminf(az.x, bz.x), tmin));
        const float f0 = fminf(fminf(fmaxf(ax.x, bx.x), fmaxf(ay.x, by.x)), fminf(fmaxf(az.x, bz.x), tmax));
        t[k + 1] = fmaxf(fmaxf(fminf(ax.y, bx.y), fminf(ay.y, by.y)), fmaxf(fminf(az.y, bz.y), tmin));
        const float f1 = fminf(fminf(fmaxf(ax.y, bx.y), fmaxf(ay.y, by.y)), fminf(fmaxf(az.y, bz.y), tmax));
        h[k] = t[k] <= f0;
        h[k + 1] = t[k + 1] <= f1;
    }
}

// BVH4 child keys (see rt_blas_walk_wide_s).
RT_HD inline unsigned rt_wkey(bool hit, int r, float t) {
    const unsigned tb = (unsigned)rt_trav_f2i(t) & 0x7FFFFFFFu;
    return (hit & (r != -1)) ? (tb | (r < -1 ? 0x80000000u : 0u)) : ~0u;
}
RT_HD inline bool  rt_wint(unsigned c) { return c >= 0x80000000u && c != ~0u; }
RT_HD inline float rt_wt(unsigned c) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __int_as_float((int)(c & 0x7FFFFFFFu));
#else
    union { unsigned u; float f; } x; x.u = c & 0x7FFFFFFFu; return x.f;
#endif
}
RT_HD inline void rt_cxu(unsigned& ka, int& ra, unsigned& kb, int& rb) {
    const bool s = kb < ka;
    const unsigned k = s ? kb : ka; kb = s ? ka : kb; ka = k;
    const int r = s ? rb : ra; rb = s ? ra : rb; ra = r;
}

// Compare-exchange of (key, ref) pairs: smaller key first.
RT_HD inline void rt_cx(float& ka, int& ra, float& kb, int& rb) {
    const bool s = kb < ka;
    const float k = s ? kb : ka; kb = s ? ka : kb; ka = k;
    const int r = s ? rb : ra; rb = s ? ra : rb; ra = r;
}

// Device: element i of a uniform-base array through a 32-bit unsigned byte
// offset (global_load v_off, s_base), no 64-bit address arithmetic per lane.
template <typename T>
RT_HD inline const T& rt_at(const T* base, int i) {
    return *(const T*)((const char*)base + (unsigned)i * (unsigned)sizeof(T));
}

// ORDERED on the BVH4 (RtNode4) collapse of the SAH tree. Same contract and
// result as rt_blas_walk_ordered_s: the closest hit is the lexicographic
// minimum of (t, SKIP rank) over the mesh, independent of the visit order, and
// a subtree is pruned only when its entry > best_t (BVH4 boxes are a subset of
// the BVH2 boxes, so every triangle the binary walk can reach is reached).
// Per node: four slab tests, a 5-exchange sorting network on the entry
// distances (misses at +inf), the hit leaves' triangles in one flat near-first
// loop (stopping at the first leaf whose entry is behind best_t), then the hit
// internal children: the nearest is descended, the others pushed far-first.
// Stack: <= m - 1 pushes per node with m internal children; the builder keeps
// the sum along any path <= RT_STACK - 2 (rt_validate_blas_set).
template <bool ANY, typename Stack>
RT_HD inline bool rt_blas_walk_wide_s(const RtNode4* nodes, const RtTri* tris64, const RtBlas& b,
                                      float3 lp, float3 lv, float tmin, int cull, bool allow_eq,
                                      Stack& st, float& best_t, int& tri, int& tri_ref,
                                      float& u, float& v, float3& ng,
                                      unsigned& nodes_visited, unsigned& tris_tested) {
    if (b.ntri <= 0) return false;
    float3 b0, b1;
    rt_tri_basis(lv, b0, b1);
    const float3 iv = rt_safe_inv(lv);
    bool found = false;
    st.reset_wide(b);
    int node = b.node4_off;
    unsigned long long best_key = ((unsigned long long)(unsigned)rt_trav_f2i(best_t + 0.0f) << 32) |
                                  (allow_eq ? 0xFFFFFFFFull : 0ull);

    while (node >= 0) {
        const RtNode4 N = rt_at(nodes, node);
        nodes_visited++;
        float t[4];
        bool h[4];
        rt_slab4(lp, iv, N, tmin, best_t, t, h);
        int r0 = N.ref[0], r1 = N.ref[1], r2 = N.ref[2], r3 = N.ref[3];
        // Category-major integer keys: hit leaf = t bits, hit internal child =
        // t bits | 2^31, miss / empty = ~0 (t >= 0; the sign of a zero entry is
        // dropped). One unsigned sort puts the hit leaves first (near first),
        // then the hit internal children (near first), then the misses.
        unsigned c0 = rt_wkey(h[0], r0, t[0]), c1 = rt_wkey(h[1], r1, t[1]);
        unsigned c2 = rt_wkey(h[2], r2, t[2]), c3 = rt_wkey(h[3], r3, t[3]);
        rt_cxu(c0, r0, c1, r1);
        rt_cxu(c2, r2, c3, r3);
        rt_cxu(c0, r0, c2, r2);
        rt_cxu(c1, r1, c3, r3);
        rt_cxu(c1, r1, c2, r2);
        int nx = -1; float nt = 0.f;
        if (rt_wint(c3) && !(rt_wt(c3) > best_t)) { nx = ~r3; nt = rt_wt(c3); }
        if (rt_wint(c2) && !(rt_wt(c2) > best_t)) { if (nx >= 0) st.push(nx, nt); nx = ~r2; nt = rt_wt(c2); }
        if (rt_wint(c1) && !(rt_wt(c1) > best_t)) { if (nx >= 0) st.push(nx, nt); nx = ~r1; nt = rt_wt(c1); }
        if (rt_wint(c0) && !(rt_wt(c0) > best_t)) { if (nx >= 0) st.push(nx, nt); nx = ~r0; nt = rt_wt(c0); }
        // Hit leaves (the key prefix below 2^31), near first, two triangles
        // per step (both loads in flight before the first test); the rest is
        // dropped once the current leaf's entry is behind best_t.
        int q = 0;
        while (c0 < 0x80000000u && !(rt_wt(c0) > best_t)) {
            const int n = (r0 & 3) + 1;
            const int ti = (r0 >> 2) + q, ti1 = ti + 1;
            const bool v1 = q + 1 < n;
            const RtTri T = rt_at(tris64, ti);
            RtTri T1;
            if (v1) T1 = rt_at(tris64, ti1);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if (j == 1 && !v1) break;
                const RtTri& U = j == 0 ? T : T1;
                const int tj = j == 0 ? ti : ti1;
                tris_tested++;
                float tt, tu, tv; float3 nn;
                if (!rt_tri_mjw(lp, lv, b0, b1, rt_trav_xyz(U.v0), rt_trav_xyz(U.v1), rt_trav_xyz(U.v2),
                                tmin, tt, tu, tv, nn))
                    continue;
                if (cull == RT_CULL_PER_TRI && rt_dot(lv, nn) > 0.f) continue;
                const unsigned long long key =
                    ((unsigned long long)(unsigned)rt_trav_f2i(tt + 0.0f) << 32) | (unsigned)rt_trav_f2i(U.v1.w);
                if (key < best_key) {
                    best_key = key;
                    best_t = tt; tri = rt_trav_f2i(U.v0.w); tri_ref = tj;
                    u = tu; v = tv; ng = nn;
                    found = true;
                    if (ANY) return true;
                }
            }
            q += 2;
            if (q >= n) { q = 0; r0 = r1; c0 = c1; r1 = r2; c1 = c2; r2 = r3; c2 = c3; c3 = ~0u; }
        }
        node = (nx >= 0 && !(nt > best_t)) ? nx : st.pop(best_t);
    }
    return found;
}

// Register-stack BVH4 ORDERED walk (host checks).
template <bool ANY>
RT_HD inline bool rt_blas_walk_wide(const RtNode4* nodes, const RtTri* tris64, const RtBlas& b,
                                    float3 lp, float3 lv, float tmin, int cull,
                                    float& best_t, int& tri, float& u, float& v, float3& ng,
                                    unsigned& nodes_visited, unsigned& tris_tested) {
    RtRegStack st;
    int tri_ref = -1;
    return rt_blas_walk_wide_s<ANY>(nodes, tris64, b, lp, lv, tmin, cull, false, st, best_t, tri, tri_ref,
                                    u, v, ng, nodes_visited, tris_tested);
}

template <bool ANY>
RT_HD inline bool rt_blas_walk_ordered(const RtNode64* nodes, const RtTri* tris64, const RtBlas& b,
                                       float3 lp, float3 lv, float tmin, int cull,
                                       float& best_t, int& tri, float& u, float& v, float3& ng,
                                       unsigned& nodes_visited, unsigned& tris_tested) {
    RtRegStack st;
    int tri_ref = -1;
    return rt_blas_walk_ordered_s<ANY>(nodes, tris64, b, lp, lv, tmin, cull, false, st, best_t, tri, tri_ref,
                                       u, v, ng, nodes_visited, tris_tested);
}

}  // namespace physics
}  // namespace hipgl
