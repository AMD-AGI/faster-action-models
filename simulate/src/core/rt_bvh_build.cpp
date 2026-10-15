// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/physics/rt_bvh_build.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <atomic>

namespace hipgl {
namespace physics {

namespace {

constexpr int   SAH_BINS     = 16;
constexpr int   SAH_MAX_LEAF = 4;
constexpr int   SAH_MAX_DEPTH = RT_STACK - 1;   // 29: ordered walker holds <= depth-1 entries
constexpr float EMPTY_BOX    = 1e30f;           // sentinel box for the unused root child

inline float fmin_(float a, float b) { return b < a ? b : a; }
inline float fmax_(float a, float b) { return b > a ? b : a; }

inline int f2i(float f) { int i; std::memcpy(&i, &f, 4); return i; }
inline float i2f(int i) { float f; std::memcpy(&f, &i, 4); return f; }

// Per-mesh triangle bounds/centroids, fp32 exactly as mjw computes them.
struct TriInfo {
    int ntri = 0;
    const float* v = nullptr;        // mesh soup base (corner c at v + 3c)
    std::vector<float> lo, hi, cen;  // [ntri*3]

    void init(const float* verts, int n) {
        v = verts; ntri = n;
        lo.resize(size_t(n) * 3); hi.resize(size_t(n) * 3); cen.resize(size_t(n) * 3);
        for (int t = 0; t < n; t++) {
            const float* p = verts + size_t(t) * 9;
            for (int a = 0; a < 3; a++) {
                float l = fmin_(fmin_(p[a], p[3 + a]), p[6 + a]);
                float h = fmax_(fmax_(p[a], p[3 + a]), p[6 + a]);
                lo[size_t(t) * 3 + a] = l;
                hi[size_t(t) * 3 + a] = h;
                cen[size_t(t) * 3 + a] = 0.5f * (l + h);   // fp32 add, then exact *0.5
            }
        }
    }
    RtTri make_tri(int t, int rank) const {
        const float* p = v + size_t(t) * 9;
        RtTri r;
        r.v0 = make_float4(p[0], p[1], p[2], i2f(t));
        r.v1 = make_float4(p[3], p[4], p[5], i2f(rank));
        r.v2 = make_float4(p[6], p[7], p[8], 0.f);
        return r;
    }
};


// Order-preserving key for a float: equal floats (incl. -0 == +0) map to equal
// keys, so an LSD radix sort reproduces np.argsort(kind="stable").
inline uint32_t sort_key(float f) {
    if (f == 0.f) f = 0.f;
    uint32_t u; std::memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Stable sort of idx[0..n) by cen[idx*3+axis].
void stable_sort_axis(int* idx, int n, const float* cen, int axis) {
    if (n <= 256) {
        std::stable_sort(idx, idx + n, [&](int a, int b) {
            return cen[size_t(a) * 3 + axis] < cen[size_t(b) * 3 + axis];
        });
        return;
    }
    // 3-pass LSD radix (11/11/10 bits) on (key, idx) pairs; LSD is stable.
    std::vector<uint64_t> a(n), b(n);
    for (int i = 0; i < n; i++)
        a[i] = (uint64_t(sort_key(cen[size_t(idx[i]) * 3 + axis])) << 32) | uint32_t(idx[i]);
    const int shifts[3] = {32, 43, 54};
    const int bits[3]   = {11, 11, 10};
    for (int pass = 0; pass < 3; pass++) {
        const int sh = shifts[pass], nb = 1 << bits[pass];
        const uint64_t mask = uint64_t(nb - 1);
        std::vector<int> cnt(nb + 1, 0);
        for (int i = 0; i < n; i++) cnt[((a[i] >> sh) & mask) + 1]++;
        for (int k = 0; k < nb; k++) cnt[k + 1] += cnt[k];
        for (int i = 0; i < n; i++) b[cnt[(a[i] >> sh) & mask]++] = a[i];
        a.swap(b);
    }
    for (int i = 0; i < n; i++) idx[i] = int(uint32_t(a[i]));
}

struct MedianBuild {
    const TriInfo* ti;
    int* idx;                 // [ntri] working permutation
    RtNode32* nodes;          // [2n-1] mesh-local pre-order
    int* leaf_tri;            // [n] mesh-local triangle per leaf slot
    int nn;                   // total nodes in this mesh
    int par_levels;           // spawn threads for this many top levels

    // Builds the subtree over idx[b,e) at pre-order slot `node`, leaf slot `leaf`.
    void rec(int b, int e, int node, int leaf, int level) {
        const int n = e - b;
        RtNode32& N = nodes[node];
        const int sz = 2 * n - 1;
        N.esc = (node + sz >= nn) ? -1 : node + sz;
        if (n == 1) {
            const int t = idx[b];
            for (int a = 0; a < 3; a++) { N.lo[a] = ti->lo[size_t(t) * 3 + a]; N.hi[a] = ti->hi[size_t(t) * 3 + a]; }
            N.payload = leaf;     // mesh-local leaf slot; tri_off added at concat
            leaf_tri[leaf] = t;
            return;
        }
        N.payload = -1;
        // axis = argmax(c.max(0) - c.min(0)), fp32, first maximum wins
        float cmn[3] = {INFINITY, INFINITY, INFINITY}, cmx[3] = {-INFINITY, -INFINITY, -INFINITY};
        const float* cen = ti->cen.data();
        for (int i = b; i < e; i++) {
            const float* c = cen + size_t(idx[i]) * 3;
            for (int a = 0; a < 3; a++) { cmn[a] = fmin_(cmn[a], c[a]); cmx[a] = fmax_(cmx[a], c[a]); }
        }
        float ext[3];
        for (int a = 0; a < 3; a++) ext[a] = cmx[a] - cmn[a];
        int axis = 0;
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
        stable_sort_axis(idx + b, n, cen, axis);
        const int mid = n / 2;                     // left gets n/2 (floor)
        const int lnode = node + 1, rnode = node + 1 + (2 * mid - 1);
        if (level < par_levels && n > 32768) {
            std::thread th([&] { rec(b, b + mid, lnode, leaf, level + 1); });
            rec(b + mid, e, rnode, leaf + mid, level + 1);
            th.join();
        } else {
            rec(b, b + mid, lnode, leaf, level + 1);
            rec(b + mid, e, rnode, leaf + mid, level + 1);
        }
        const RtNode32& L = nodes[lnode];
        const RtNode32& R = nodes[rnode];
        for (int a = 0; a < 3; a++) { N.lo[a] = fmin_(L.lo[a], R.lo[a]); N.hi[a] = fmax_(L.hi[a], R.hi[a]); }
    }
};

// ---------------------------------------------------------------------------
// SAH / BVH2
// ---------------------------------------------------------------------------

struct Box {
    float lo[3] = {INFINITY, INFINITY, INFINITY};
    float hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    void grow(const float* l, const float* h) {
        for (int a = 0; a < 3; a++) { lo[a] = fmin_(lo[a], l[a]); hi[a] = fmax_(hi[a], h[a]); }
    }
    void grow(const Box& o) { grow(o.lo, o.hi); }
    double area() const {
        if (!(hi[0] >= lo[0])) return 0.0;
        double dx = double(hi[0]) - lo[0], dy = double(hi[1]) - lo[1], dz = double(hi[2]) - lo[2];
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }
};

inline int median_levels(int n) {
    int l = 0, c = (n + SAH_MAX_LEAF - 1) / SAH_MAX_LEAF;
    while ((1 << l) < c) l++;
    return l;
}

struct SahBuild {
    const TriInfo* ti;
    std::vector<int> idx;
    std::vector<RtNode64> nodes;   // mesh-local
    std::vector<int> leaf_tri;     // mesh-local tri in SAH leaf order
    int max_depth = 0;

    struct Ref { Box box; int c, n; };

    static void set_child(RtNode64& N, int k, const Ref& r) {
        for (int a = 0; a < 3; a++) { N.lo[a][k] = r.box.lo[a]; N.hi[a][k] = r.box.hi[a]; }
        N.c[k] = r.c; N.n[k] = r.n;
    }

    Ref make_leaf(int b, int e, const Box& bb) {
        std::sort(idx.begin() + b, idx.begin() + e);   // leaf tris in ascending tri order
        Ref r{bb, int(leaf_tri.size()), e - b};
        for (int i = b; i < e; i++) leaf_tri.push_back(idx[i]);
        return r;
    }

    int median_split(int b, int e, const Box& cb) {
        int axis = 0;
        float ext[3];
        for (int a = 0; a < 3; a++) ext[a] = cb.hi[a] - cb.lo[a];
        if (ext[1] > ext[axis]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;
        const int mid = b + (e - b) / 2;
        if (ext[axis] > 0.f) {
            const float* cen = ti->cen.data();
            std::nth_element(idx.begin() + b, idx.begin() + mid, idx.begin() + e, [&](int x, int y) {
                float cx = cen[size_t(x) * 3 + axis], cy = cen[size_t(y) * 3 + axis];
                return cx < cy || (cx == cy && x < y);
            });
        }
        return mid;
    }

    // d = level the node allocated here would sit at (root = 1).
    Ref build(int b, int e, int d) {
        const int n = e - b;
        const float* lo = ti->lo.data(); const float* hi = ti->hi.data(); const float* cen = ti->cen.data();
        Box bb, cb;
        for (int i = b; i < e; i++) {
            const size_t t = size_t(idx[i]) * 3;
            bb.grow(lo + t, hi + t);
            cb.grow(cen + t, cen + t);
        }
        if (n == 1) return make_leaf(b, e, bb);

        int mid = -1;
        if (d + median_levels(n) > SAH_MAX_DEPTH) {
            if (n <= SAH_MAX_LEAF) return make_leaf(b, e, bb);
            mid = median_split(b, e, cb);
        } else {
            // binned SAH; costs are scaled by SA(parent) to avoid a divide
            const double sap = bb.area();
            double best = INFINITY; int best_axis = -1, best_split = -1;
            float best_scale = 0.f;
            for (int a = 0; a < 3; a++) {
                const float ext = cb.hi[a] - cb.lo[a];
                if (!(ext > 0.f)) continue;
                const float scale = float(SAH_BINS) / ext;
                Box bin[SAH_BINS]; int cnt[SAH_BINS] = {};
                for (int i = b; i < e; i++) {
                    const size_t t = size_t(idx[i]) * 3;
                    int k = int((cen[t + a] - cb.lo[a]) * scale);
                    k = k < 0 ? 0 : (k >= SAH_BINS ? SAH_BINS - 1 : k);
                    cnt[k]++; bin[k].grow(lo + t, hi + t);
                }
                double rarea[SAH_BINS]; int rcnt[SAH_BINS];
                Box acc; int c = 0;
                for (int k = SAH_BINS - 1; k > 0; k--) {
                    acc.grow(bin[k]); c += cnt[k];
                    rarea[k] = acc.area(); rcnt[k] = c;
                }
                Box accl; int cl = 0;
                for (int k = 0; k < SAH_BINS - 1; k++) {
                    accl.grow(bin[k]); cl += cnt[k];
                    if (cl == 0 || rcnt[k + 1] == 0) continue;
                    const double cost = sap + cl * accl.area() + rcnt[k + 1] * rarea[k + 1];
                    if (cost < best) { best = cost; best_axis = a; best_split = k + 1; best_scale = scale; }
                }
            }
            const double leaf_cost = double(n) * sap;
            if (n <= SAH_MAX_LEAF && (best_axis < 0 || leaf_cost <= best)) return make_leaf(b, e, bb);
            if (best_axis >= 0) {
                const int a = best_axis; const float cl0 = cb.lo[a];
                auto it = std::partition(idx.begin() + b, idx.begin() + e, [&](int t) {
                    int k = int((cen[size_t(t) * 3 + a] - cl0) * best_scale);
                    k = k < 0 ? 0 : (k >= SAH_BINS ? SAH_BINS - 1 : k);
                    return k < best_split;
                });
                mid = int(it - idx.begin());
                if (mid == b || mid == e) mid = -1;      // fp disagreement; fall back
            }
            if (mid < 0) mid = median_split(b, e, cb);  // coincident centroids
        }

        const int me = int(nodes.size());
        nodes.emplace_back();
        max_depth = std::max(max_depth, d);
        Ref L = build(b, mid, d + 1);
        Ref R = build(mid, e, d + 1);
        set_child(nodes[me], 0, L);
        set_child(nodes[me], 1, R);
        return Ref{bb, me, 0};
    }

    void run(const TriInfo& t) {
        ti = &t;
        idx.resize(t.ntri);
        for (int i = 0; i < t.ntri; i++) idx[i] = i;
        leaf_tri.reserve(t.ntri);
        Ref r = build(0, t.ntri, 1);
        if (r.n > 0) {                                  // whole mesh is one leaf
            nodes.emplace_back();
            set_child(nodes[0], 0, r);
            Ref empty; for (int a = 0; a < 3; a++) { empty.box.lo[a] = EMPTY_BOX; empty.box.hi[a] = EMPTY_BOX; }
            empty.c = -1; empty.n = 0;
            set_child(nodes[0], 1, empty);
            max_depth = 1;
        }
    }
};

// ---------------------------------------------------------------------------
// BVH4: collapse of the SAH tree (mesh-local RtNode64, root 0)
// ---------------------------------------------------------------------------

// Max ORDERED stack entries the wide walk can hold (pushes <= m - 1 per node
// with m internal children; the walk descends into one of them).
constexpr int WIDE_MAX_ENTRIES = RT_STACK - 2;

struct WideBuild {
    const std::vector<RtNode64>* bn;
    std::vector<int> height;       // binary subtree height (levels, leaf-only node = 1)
    std::vector<RtNode4> nodes;    // mesh-local, pre-order

    struct Slot { float lo[3], hi[3]; int c, n; };

    static double area(const Slot& s) {
        double dx = double(s.hi[0]) - s.lo[0], dy = double(s.hi[1]) - s.lo[1], dz = double(s.hi[2]) - s.lo[2];
        return 2.0 * (dx * dy + dy * dz + dz * dx);
    }
    Slot slot(int node, int k) const {
        const RtNode64& N = (*bn)[node];
        Slot s;
        for (int a = 0; a < 3; a++) { s.lo[a] = N.lo[a][k]; s.hi[a] = N.hi[a][k]; }
        s.c = N.c[k]; s.n = N.n[k];
        return s;
    }
    static bool internal(const Slot& s) { return s.n == 0 && s.c >= 0; }

    int calc_height(int node) {
        int h = 0;
        const RtNode64& N = (*bn)[node];
        for (int k = 0; k < 2; k++)
            if (N.n[k] == 0 && N.c[k] >= 0) h = std::max(h, calc_height(N.c[k]));
        return height[node] = h + 1;
    }
    // Entry bound of a slot list collapsed binary-only below: (m - 1) + max(height - 1).
    int bound(const std::vector<Slot>& v) const {
        int m = 0, hm = 0;
        for (const Slot& s : v) if (internal(s)) { m++; hm = std::max(hm, height[s.c] - 1); }
        return std::max(m - 1, 0) + hm;
    }

    // Wide node over binary node `node` with entry budget `budget` (>= height - 1).
    int rec(int node, int budget) {
        std::vector<Slot> v;
        for (int k = 0; k < 2; k++) { Slot s = slot(node, k); if (s.n > 0 || s.c >= 0) v.push_back(s); }
        while (v.size() < 4) {
            // open the largest-area internal slot whose opening keeps the bound
            int best = -1; double ba = -1.0;
            for (int i = 0; i < int(v.size()); i++) {
                if (!internal(v[i])) continue;
                std::vector<Slot> w;
                for (int j = 0; j < int(v.size()); j++) {
                    if (j != i) { w.push_back(v[j]); continue; }
                    w.push_back(slot(v[i].c, 0));
                    const Slot s1 = slot(v[i].c, 1);
                    if (s1.n > 0 || s1.c >= 0) w.push_back(s1);
                }
                if (bound(w) > budget) continue;
                const double a = area(v[i]);
                if (a > ba) { ba = a; best = i; }
            }
            if (best < 0) break;
            const int bc = v[best].c;
            const Slot s0 = slot(bc, 0), s1 = slot(bc, 1);
            v[best] = s0;
            if (s1.n > 0 || s1.c >= 0) v.insert(v.begin() + best + 1, s1);
        }
        int m = 0;
        for (const Slot& s : v) if (internal(s)) m++;
        const int child_budget = budget - std::max(m - 1, 0);
        const int me = int(nodes.size());
        nodes.emplace_back();
        int cref[4];
        for (int k = 0; k < int(v.size()); k++)
            cref[k] = internal(v[k]) ? rec(v[k].c, child_budget) : v[k].c;
        RtNode4& W = nodes[me];
        for (int k = 0; k < 4; k++) {
            if (k < int(v.size())) {
                for (int a = 0; a < 3; a++) { W.lo[a][k] = v[k].lo[a]; W.hi[a][k] = v[k].hi[a]; }
                // mesh-local encoding (globalized at concat); child nodes are never the root (0)
                W.ref[k] = v[k].n > 0 ? ((cref[k] << 2) | (v[k].n - 1)) : ~cref[k];
            } else {
                for (int a = 0; a < 3; a++) { W.lo[a][k] = EMPTY_BOX; W.hi[a][k] = EMPTY_BOX; }
                W.ref[k] = -1;
            }
            W.pad[k] = 0;
        }
        return me;
    }

    void run(const std::vector<RtNode64>& b) {
        bn = &b;
        if (b.empty()) return;
        height.assign(b.size(), 0);
        calc_height(0);
        rec(0, WIDE_MAX_ENTRIES);
    }
};

struct MeshOut {
    std::vector<RtNode32> n32; std::vector<RtTri> tri;
    std::vector<RtNode64> n64; std::vector<RtTri> tri64;
    std::vector<RtNode4> n4;
    int depth = 0;
};

void build_mesh(const float* verts, int ntri, int par_levels, MeshOut& out) {
    if (ntri <= 0) return;
    TriInfo ti; ti.init(verts, ntri);
    std::vector<int> skip_rank;   // mesh-local tri -> SKIP leaf slot

    {
        const int nn = 2 * ntri - 1;
        std::vector<int> idx(ntri), leaf(ntri);
        for (int i = 0; i < ntri; i++) idx[i] = i;
        out.n32.resize(nn);
        MedianBuild mb{&ti, idx.data(), out.n32.data(), leaf.data(), nn, par_levels};
        mb.rec(0, ntri, 0, 0, 0);
        out.tri.resize(ntri);
        skip_rank.resize(ntri);
        for (int k = 0; k < ntri; k++) { out.tri[k] = ti.make_tri(leaf[k], k); skip_rank[leaf[k]] = k; }
    }
    // SAH / BVH2
    {
        SahBuild sb; sb.run(ti);
        out.n64.swap(sb.nodes);
        out.tri64.resize(ntri);
        for (int k = 0; k < ntri; k++) out.tri64[k] = ti.make_tri(sb.leaf_tri[k], skip_rank[sb.leaf_tri[k]]);
        out.depth = sb.max_depth;
    }
    // BVH4 (same tri64 leaves)
    {
        WideBuild wb; wb.run(out.n64);
        out.n4.swap(wb.nodes);
    }
}

}  // namespace

RtBlasSet rt_build_blas_set(const float* mesh_vert, const int* vert_adr, const int* vert_num,
                            int nmesh, int nthreads) {
    if (nthreads <= 0) nthreads = std::max(1u, std::thread::hardware_concurrency());
    nthreads = std::min(nthreads, 64);
    std::vector<MeshOut> outs(std::max(nmesh, 0));

    std::vector<int> order(nmesh);
    long long total = 0;
    for (int m = 0; m < nmesh; m++) { order[m] = m; total += vert_num[m] / 3; }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return vert_num[a] > vert_num[b]; });
    int par_levels = 0;
    while (nthreads > 1 && (1 << par_levels) < nthreads) par_levels++;

    auto work = [&](int m, int pl) {
        const int ntri = vert_num[m] / 3;
        build_mesh(mesh_vert + size_t(vert_adr[m]) * 3, ntri, pl, outs[m]);
    };
    if (nthreads == 1) {
        for (int m = 0; m < nmesh; m++) work(m, 0);
    } else {
        std::atomic<int> next{0};
        std::vector<std::thread> pool;
        const int nt = std::min(nthreads, std::max(nmesh, 1));
        for (int i = 0; i < nt; i++)
            pool.emplace_back([&] {
                for (int k; (k = next.fetch_add(1)) < nmesh;) {
                    const int m = order[k];
                    const bool big = (vert_num[m] / 3) * 4LL > total;
                    work(m, big ? par_levels : 0);
                }
            });
        for (auto& t : pool) t.join();
    }

    // Concatenate in mesh order and globalize indices.
    RtBlasSet s;
    size_t c32 = 0, ct = 0, c64 = 0, c4 = 0;
    for (auto& o : outs) { c32 += o.n32.size(); ct += o.tri.size(); c64 += o.n64.size(); c4 += o.n4.size(); }
    s.n32.reserve(c32); s.tri.reserve(ct); s.n64.reserve(c64); s.tri64.reserve(ct); s.n4.reserve(c4);
    s.blas.resize(std::max(nmesh, 0));
    for (int m = 0; m < nmesh; m++) {
        MeshOut& o = outs[m];
        RtBlas& b = s.blas[m];
        std::memset(&b, 0, sizeof(b));
        b.ntri = int(o.tri.size());
        if (b.ntri == 0) { b.node_off = b.tri_off = b.node64_off = b.tri64_off = b.node4_off = -1; continue; }
        b.node_off = int(s.n32.size()); b.tri_off = int(s.tri.size());
        b.node64_off = int(s.n64.size()); b.tri64_off = int(s.tri64.size());
        b.node4_off = int(s.n4.size());
        b.depth = o.depth;
        b.nbits = 1;
        while (b.nbits < 31 && (1LL << b.nbits) < (long long)o.n64.size()) b.nbits++;
        for (RtNode32 n : o.n32) {
            if (n.esc >= 0) n.esc += b.node_off;
            if (n.payload >= 0) n.payload += b.tri_off;
            s.n32.push_back(n);
        }
        for (RtNode64 n : o.n64) {
            for (int k = 0; k < 2; k++) {
                if (n.n[k] > 0) n.c[k] += b.tri64_off; else if (n.c[k] >= 0) n.c[k] += b.node64_off;
            }
            s.n64.push_back(n);
        }
        for (RtNode4 n : o.n4) {
            for (int k = 0; k < 4; k++) {
                if (n.ref[k] >= 0) {
                    if ((long long)(n.ref[k] >> 2) + b.tri64_off >= (1LL << 29))
                        throw std::runtime_error("rt_build_blas_set: BVH4 leaf ref needs > 2^29 triangles");
                    n.ref[k] += b.tri64_off << 2;
                } else if (n.ref[k] != -1) {
                    n.ref[k] = ~(~n.ref[k] + b.node4_off);
                }
            }
            s.n4.push_back(n);
        }
        s.tri.insert(s.tri.end(), o.tri.begin(), o.tri.end());
        s.tri64.insert(s.tri64.end(), o.tri64.begin(), o.tri64.end());
        std::vector<RtNode32>().swap(o.n32);  // release per-mesh memory early
        std::vector<RtTri>().swap(o.tri); std::vector<RtTri>().swap(o.tri64);
        std::vector<RtNode64>().swap(o.n64);
        std::vector<RtNode4>().swap(o.n4);
    }
    // The BVH4 walker addresses RtNode4 / tri64 through 32-bit byte offsets (rt_at).
    if (s.n4.size() * sizeof(RtNode4) >= (1ull << 32) || s.tri64.size() * sizeof(RtTri) >= (1ull << 32))
        throw std::runtime_error("rt_build_blas_set: BVH4 node / triangle arrays exceed 4 GiB");
    return s;
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

namespace {

struct Checker {
    const RtBlasSet& s;
    const float* v;           // mesh soup base
    int m, ntri;
    const RtBlas& b;
    std::vector<int> seen;
    std::string msg;

    bool fail(const char* fmt, long long a = 0, long long c = 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), fmt, a, c);
        msg = "mesh " + std::to_string(m) + ": " + buf;
        return false;
    }
    static bool inside(const float* p, const float* lo, const float* hi) {
        for (int a = 0; a < 3; a++) if (!(p[a] >= lo[a] && p[a] <= hi[a])) return false;
        return true;
    }
    bool tri_ok(const RtTri& T, const float* lo, const float* hi) {
        const int t = f2i(T.v0.w);
        if (t < 0 || t >= ntri) return fail("tri index %lld out of range", t);
        const float* p = v + size_t(t) * 9;
        const float q[9] = {T.v0.x, T.v0.y, T.v0.z, T.v1.x, T.v1.y, T.v1.z, T.v2.x, T.v2.y, T.v2.z};
        if (std::memcmp(p, q, sizeof(q)) != 0) return fail("RtTri %lld differs from soup", t);
        const int rk = f2i(T.v1.w);
        if (rk < 0 || rk >= ntri || f2i(s.tri[b.tri_off + rk].v0.w) != t)
            return fail("tri %lld: v1.w is not its SKIP leaf rank (%lld)", t, rk);
        for (int k = 0; k < 3; k++) if (!inside(q + 3 * k, lo, hi)) return fail("tri %lld corner outside its node box", t);
        return true;
    }
    bool mark(const RtTri& T) {
        const int t = f2i(T.v0.w);
        if (seen[t]++) return fail("tri %lld reachable more than once", t);
        return true;
    }

    // SKIP: returns end (global) of the subtree rooted at i, or -1 on failure.
    int skip_rec(int i, int& next_leaf, int depth) {
        const int end_all = b.node_off + 2 * ntri - 1;
        if (i < b.node_off || i >= end_all) { fail("skip node %lld out of range", i); return -1; }
        if (depth > 64) { fail("skip depth > 64 at node %lld", i); return -1; }
        const RtNode32& N = s.n32[i];
        int end;
        if (N.payload >= 0) {
            if (N.payload != b.tri_off + next_leaf) { fail("leaf payload %lld not in leaf order", N.payload); return -1; }
            next_leaf++;
            const RtTri& T = s.tri[N.payload];
            if (!tri_ok(T, N.lo, N.hi) || !mark(T)) return -1;
            // leaf bounds are exactly the triangle AABB (mjw _tri_bounds)
            const float* p = v + size_t(f2i(T.v0.w)) * 9;
            for (int a = 0; a < 3; a++)
                if (N.lo[a] != fmin_(fmin_(p[a], p[3 + a]), p[6 + a]) || N.hi[a] != fmax_(fmax_(p[a], p[3 + a]), p[6 + a])) {
                    fail("leaf %lld bounds != triangle AABB", i); return -1;
                }
            end = i + 1;
        } else {
            const int l = i + 1;
            const int le = skip_rec(l, next_leaf, depth + 1);
            if (le < 0) return -1;
            const int re = skip_rec(le, next_leaf, depth + 1);
            if (re < 0) return -1;
            const RtNode32& L = s.n32[l]; const RtNode32& R = s.n32[le];
            for (int a = 0; a < 3; a++)
                if (N.lo[a] != fmin_(L.lo[a], R.lo[a]) || N.hi[a] != fmax_(L.hi[a], R.hi[a])) {
                    fail("node %lld bounds != union of children", i); return -1;
                }
            end = re;
        }
        const int want = end >= end_all ? -1 : end;
        if (N.esc != want) { fail("node %lld esc %lld wrong", i, N.esc); return -1; }
        return end;
    }

    bool skip() {
        std::fill(seen.begin(), seen.end(), 0);
        int next_leaf = 0;
        const int end = skip_rec(b.node_off, next_leaf, 0);
        if (end < 0) return false;
        if (end != b.node_off + 2 * ntri - 1) return fail("skip tree covers %lld nodes", end - b.node_off);
        for (int t = 0; t < ntri; t++) if (seen[t] != 1) return fail("skip: tri %lld seen %lld times", t, seen[t]);
        return true;
    }

    bool ordered(int nnodes) {
        std::fill(seen.begin(), seen.end(), 0);
        std::vector<char> vis(nnodes, 0);
        struct It { int node, depth; };
        std::vector<It> st{{b.node64_off, 1}};
        int maxd = 0, nvis = 0;
        while (!st.empty()) {
            It it = st.back(); st.pop_back();
            const int i = it.node;
            if (i < b.node64_off || i >= b.node64_off + nnodes) return fail("n64 node %lld out of range", i);
            if (vis[i - b.node64_off]++) return fail("n64 node %lld reached twice", i);
            nvis++;
            maxd = std::max(maxd, it.depth);
            const RtNode64& N = s.n64[i];
            for (int k = 0; k < 2; k++) {
                float lo[3], hi[3];
                N.child_box(k, lo, hi);
                const int c = N.c[k], n = N.n[k];
                if (n > 0) {
                    if (n > SAH_MAX_LEAF) return fail("leaf with %lld tris", n);
                    if (c < b.tri64_off || c + n > b.tri64_off + ntri) return fail("leaf range %lld out of mesh", c);
                    for (int j = c; j < c + n; j++) if (!tri_ok(s.tri64[j], lo, hi) || !mark(s.tri64[j])) return false;
                } else if (c >= 0) {
                    if (c < b.node64_off || c >= b.node64_off + nnodes) return fail("child %lld out of range", c);
                    const RtNode64& C = s.n64[c];
                    float cl0[3], ch0[3], cl1[3], ch1[3];
                    C.child_box(0, cl0, ch0); C.child_box(1, cl1, ch1);
                    if (!inside(cl0, lo, hi) || !inside(ch0, lo, hi) || !inside(cl1, lo, hi) || !inside(ch1, lo, hi)) {
                        // the unused sentinel child only occurs at the root
                        return fail("child %lld box not inside parent slot", c);
                    }
                    st.push_back({c, it.depth + 1});
                } else if (!(i == b.node64_off && k == 1 && ntri == N.n[0])) {
                    return fail("empty child at node %lld", i);
                }
            }
        }
        if (nvis != nnodes) return fail("%lld of %lld n64 nodes reachable", nvis, nnodes);
        for (int t = 0; t < ntri; t++) if (seen[t] != 1) return fail("sah: tri %lld seen %lld times", t, seen[t]);
        if (maxd != b.depth) return fail("depth %lld != blas.depth %lld", maxd, b.depth);
        if (maxd > SAH_MAX_DEPTH) return fail("sah depth %lld > %lld", maxd, SAH_MAX_DEPTH);
        if (b.nbits < 1 || b.nbits > RT_STACK_MAX_NODE_BITS || (1LL << b.nbits) < nnodes)
            return fail("nbits %lld cannot index %lld n64 nodes (max 2^24)", b.nbits, nnodes);
        return true;
    }

    // BVH4: containment, every tri reachable once, every node reachable once,
    // entry bound <= WIDE_MAX_ENTRIES, nbits covers the node count.
    std::vector<char> vis4;
    int wide_rec(int i, int nnodes, int depth) {   // returns entry bound, -1 on failure
        if (i < b.node4_off || i >= b.node4_off + nnodes) { fail("n4 node %lld out of range", i); return -1; }
        if (depth > 64) { fail("n4 depth > 64 at node %lld", i); return -1; }
        if (vis4[i - b.node4_off]++) { fail("n4 node %lld reached twice", i); return -1; }
        const RtNode4& N = s.n4[i];
        int m = 0, hm = 0, nslot = 0;
        for (int k = 0; k < 4; k++) {
            float lo[3], hi[3];
            N.child_box(k, lo, hi);
            int c, n;
            N.child(k, c, n);
            if (n == 0 && c >= 0 && c == b.node4_off) { fail("n4 node %lld: child is the root", i); return -1; }
            if (n > 0) {
                nslot++;
                if (n > SAH_MAX_LEAF) { fail("n4 leaf with %lld tris", n); return -1; }
                if (c < b.tri64_off || c + n > b.tri64_off + ntri) { fail("n4 leaf range %lld out of mesh", c); return -1; }
                for (int j = c; j < c + n; j++) if (!tri_ok(s.tri64[j], lo, hi) || !mark(s.tri64[j])) return -1;
            } else if (c >= 0) {
                nslot++; m++;
                if (c < b.node4_off || c >= b.node4_off + nnodes) { fail("n4 child %lld out of range", c); return -1; }
                const RtNode4& C = s.n4[c];
                for (int q = 0; q < 4; q++) {
                    if (C.ref[q] == -1) continue;
                    float cl[3], ch[3];
                    C.child_box(q, cl, ch);
                    if (!inside(cl, lo, hi) || !inside(ch, lo, hi)) { fail("n4 child %lld box not inside parent slot", c); return -1; }
                }
                const int hb = wide_rec(c, nnodes, depth + 1);
                if (hb < 0) return -1;
                hm = std::max(hm, hb);
            }
        }
        if (nslot == 0) { fail("n4 node %lld has no children", i); return -1; }
        return std::max(m - 1, 0) + hm;
    }
    bool wide(int nnodes) {
        std::fill(seen.begin(), seen.end(), 0);
        vis4.assign(nnodes, 0);
        if (nnodes <= 0) return fail("mesh has no n4 nodes");
        const int ent = wide_rec(b.node4_off, nnodes, 0);
        if (ent < 0) return false;
        for (int k = 0; k < nnodes; k++) if (!vis4[k]) return fail("n4 node %lld unreachable", b.node4_off + k);
        for (int t = 0; t < ntri; t++) if (seen[t] != 1) return fail("bvh4: tri %lld seen %lld times", t, seen[t]);
        if (ent > WIDE_MAX_ENTRIES) return fail("bvh4 stack bound %lld > %lld", ent, WIDE_MAX_ENTRIES);
        if ((1LL << b.nbits) < nnodes) return fail("nbits %lld cannot index %lld n4 nodes", b.nbits, nnodes);
        return true;
    }
};

}  // namespace

bool rt_validate_blas_set(const RtBlasSet& s, const float* mesh_vert, const int* vert_adr,
                          const int* vert_num, int nmesh, std::string* err) {
    auto bad = [&](const std::string& e) { if (err) *err = e; return false; };
    if (int(s.blas.size()) != nmesh) return bad("blas count != nmesh");
    size_t c32 = 0, ct = 0;
    for (int m = 0; m < nmesh; m++) {
        const RtBlas& b = s.blas[m];
        const int ntri = vert_num[m] / 3;
        if (b.ntri != ntri) return bad("mesh " + std::to_string(m) + ": ntri mismatch");
        if (ntri == 0) {
            if (b.node_off != -1 || b.node64_off != -1 || b.node4_off != -1) return bad("mesh " + std::to_string(m) + ": empty mesh has offsets");
            continue;
        }
        if (b.node_off != int(c32) || b.tri_off != int(ct) || b.tri64_off != int(ct))
            return bad("mesh " + std::to_string(m) + ": offsets not contiguous");
        c32 += 2 * size_t(ntri) - 1; ct += ntri;
        // node64 count of this mesh: up to the next non-empty mesh's offset
        int end64 = int(s.n64.size());
        for (int k = m + 1; k < nmesh; k++) if (s.blas[k].ntri > 0) { end64 = s.blas[k].node64_off; break; }
        Checker c{s, mesh_vert + size_t(vert_adr[m]) * 3, m, ntri, b, std::vector<int>(ntri, 0), {}};
        if (!c.skip()) return bad(c.msg);
        if (!c.ordered(end64 - b.node64_off)) return bad(c.msg);
        int end4 = int(s.n4.size());
        for (int k = m + 1; k < nmesh; k++) if (s.blas[k].ntri > 0) { end4 = s.blas[k].node4_off; break; }
        if (b.node4_off < 0 || end4 < b.node4_off) return bad("mesh " + std::to_string(m) + ": bad node4_off");
        if (!c.wide(end4 - b.node4_off)) return bad(c.msg);
    }
    if (c32 != s.n32.size() || ct != s.tri.size() || ct != s.tri64.size()) return bad("array sizes != sum of meshes");
    return true;
}

}  // namespace physics
}  // namespace hipgl
