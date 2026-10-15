// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include <climits>
#include <cstdlib>
#include "hipgl/types.h"
#include "hipgl/math_types.h"
#include "hipgl/kernels.h"

namespace hipgl {
namespace kernels {

// Largest source-triangle count this path accepts. The two micro draws are 32
// and 192 triangles; the LDS record table is sized for source + near-plane
// split suffix, i.e. 2x this.
#ifndef HIPGL_MICRO_MAX_TRIS
#define HIPGL_MICRO_MAX_TRIS 256
#endif
#define HIPGL_MICRO_RECS (2 * HIPGL_MICRO_MAX_TRIS)
// Largest tile grid this path accepts (r512 is 1024 tiles).
#ifndef HIPGL_MICRO_MAX_TILES
#define HIPGL_MICRO_MAX_TILES 1024
#endif
// Segment length handled by ONE thread (rank count out of L1).
#ifndef HIPGL_MICRO_SMALL_SEG
#define HIPGL_MICRO_SMALL_SEG 8
#endif
#ifndef HIPGL_MICRO_TPB
#define HIPGL_MICRO_TPB 512
#endif
#define HIPGL_MICRO_WAVES (HIPGL_MICRO_TPB / 64)

#ifndef HIPGL_MICRO_STAGE
#define HIPGL_MICRO_STAGE 2048
#endif

// Same rule as hipgl_bin_reject in bin_rasterize.hip (kept local: that one is a
// static device function in another translation unit).
HIPGL_DEVICE __forceinline__ bool micro_reject(const TriangleSetup& tri, int cull_mask) {
    if (tri.degenerate) return true;
    return ((cull_mask >> (tri.backfacing ? 1 : 0)) & 1) != 0;
}

HIPGL_DEVICE __forceinline__ int micro_wave_incl(int v, int lane) {
#pragma unroll
    for (int off = 1; off < 64; off <<= 1) {
        const int a = __shfl_up(v, off, 64);
        if (lane >= off) v += a;
    }
    return v;
}

// Block-wide EXCLUSIVE scan of one value per thread. TWO barriers, not
// 2*log2(TPB). s_w must hold HIPGL_MICRO_WAVES ints; `total` comes back with
// the block sum on every thread.
template <int TPB>
HIPGL_DEVICE __forceinline__ int micro_block_excl(int v, int* s_w, int& total) {
    const int lane = threadIdx.x & 63;
    const int w    = threadIdx.x >> 6;
    const int incl = micro_wave_incl(v, lane);
    if (lane == 63) s_w[w] = incl;
    __syncthreads();
    if (threadIdx.x < 64) {
        const int t = (threadIdx.x < (TPB / 64)) ? s_w[threadIdx.x] : 0;
        const int ti = micro_wave_incl(t, threadIdx.x);
        if (threadIdx.x < (TPB / 64)) s_w[threadIdx.x] = ti;
    }
    __syncthreads();
    total = s_w[(TPB / 64) - 1];
    return incl - v + (w > 0 ? s_w[w - 1] : 0);
}

#ifndef HIPGL_MICRO_MAX_SLOTS
#define HIPGL_MICRO_MAX_SLOTS 4
#endif
struct MicroChainArgs {
    MicroDrawSlotDesc s[HIPGL_MICRO_MAX_SLOTS];
    int packed[HIPGL_MICRO_MAX_SLOTS];
    int tiles_x, tiles_y, fb_width, fb_height, num_tiles;
    int* overflow_flag;
    int stop;
    int nslots;
    MicroClearDesc clr;
};

template <int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_micro_bin_chain(MicroChainArgs args)
{
    if ((int)blockIdx.x >= args.nslots) {
        const int p = (int)(blockIdx.x - args.nslots) * TPB + (int)threadIdx.x;
        if (p < args.clr.total) {
            float4* __restrict__ c4 = reinterpret_cast<float4*>(args.clr.color);
            if (args.clr.cvt_dst) {
                const float4 o = c4[p];
                const float cr_ = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, o.x));
                const float cg_ = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, o.y));
                const float cb_ = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, o.z));
                unsigned char* __restrict__ dq = args.clr.cvt_dst + (size_t)p * 3;
                dq[0] = (unsigned char)(cr_ * 255.0f + 0.5f);
                dq[1] = (unsigned char)(cg_ * 255.0f + 0.5f);
                dq[2] = (unsigned char)(cb_ * 255.0f + 0.5f);
            }
            float4 v; v.x = args.clr.r; v.y = args.clr.g; v.z = args.clr.b; v.w = args.clr.a;
            c4[p] = v;
            args.clr.depth[p] = args.clr.d;
        }
        return;
    }
    const MicroDrawSlotDesc d = args.s[blockIdx.x];
    const TriangleSetup* __restrict__ tris = d.tris;
    const int tri_count           = d.tri_count;
    const int suffix_base         = d.suffix_base;
    int* __restrict__ tile_base   = d.tile_base;
    int* __restrict__ cursor      = d.cursor;
    unsigned long long* __restrict__ key_out = d.key_out;
    const int pool_capacity       = d.pool_capacity;
    const int cull_mask           = d.cull_mask;
    const NearClipQueue queue     = d.near_clip;
    const bool packed_ids         = args.packed[blockIdx.x] != 0;
    const int tiles_x   = args.tiles_x;
    const int tiles_y   = args.tiles_y;
    const int fb_width  = args.fb_width;
    const int fb_height = args.fb_height;
    const int num_tiles = args.num_tiles;
    int* __restrict__ overflow_flag = args.overflow_flag;
    const int stop = args.stop;
    (void)tiles_y;
    // Per-record rectangle + composite key.
    __shared__ int   s_lox[HIPGL_MICRO_RECS];
    __shared__ int   s_loy[HIPGL_MICRO_RECS];
    __shared__ int   s_nx [HIPGL_MICRO_RECS];
    __shared__ unsigned long long s_key[HIPGL_MICRO_RECS];
    __shared__ int   s_off[HIPGL_MICRO_RECS + 1];
    __shared__ int   s_w[HIPGL_MICRO_WAVES];
    __shared__ int   s_carry;
    __shared__ int   s_total;
    __shared__ int   s_cnt [HIPGL_MICRO_MAX_TILES];   // counts, then exclusive base
    __shared__ int   s_cur [HIPGL_MICRO_MAX_TILES];   // scatter cursor
    __shared__ int   s_list[HIPGL_MICRO_MAX_TILES];   // tiles with 2..64 entries
    __shared__ int   s_big [HIPGL_MICRO_MAX_TILES];   // tiles with > 64 entries
    __shared__ int   s_nlist;
    __shared__ int   s_nbig;
    __shared__ unsigned long long s_pk[HIPGL_MICRO_STAGE];
    __shared__ int   s_pt[HIPGL_MICRO_STAGE];   // pair -> tile

    const int tid = threadIdx.x;

    // ── PHASE 0: per-record rectangle + key table, and zero the tile counts ─
    // Records [0, tri_count) are the source triangles; [tri_count, R) are the
    // near-plane split suffixes queued by the fused vertex pass.
    int nq = 0;
    if (queue.count) {
        nq = *queue.count;
        if (nq < 0) nq = 0;
        // The queue belongs to the VERTEX PASS, which may have covered more
        // source triangles than this slot bins (a shared pass feeding several
        // slots).  Every entry is filtered by `source < tri_count` below, so the
        // only requirement here is that the record table cannot overflow.
        if (nq > HIPGL_MICRO_MAX_TRIS) nq = HIPGL_MICRO_MAX_TRIS;
        if (tri_count + nq > HIPGL_MICRO_RECS) nq = HIPGL_MICRO_RECS - tri_count;
        if (nq < 0) nq = 0;
    }
    const int R = tri_count + nq;

    for (int t = tid; t < num_tiles; t += TPB) { s_cnt[t] = 0; s_cur[t] = 0; }

    for (int r = tid; r < R; r += TPB) {
        int n = 0, lo_x = 0, lo_y = 0, nx = 1, ny = 0;
        unsigned int z_ord = 0x80000000u;
        unsigned int order_id = 0;
        if (r < R) {
            const TriangleSetup* tp = nullptr;
            bool live = false;
            if (r < tri_count) {
                tp = &tris[r];
                order_id = (unsigned int)(packed_ids ? 2 * r : r);
                live = !micro_reject(*tp, cull_mask);
            } else {
                const int source = queue.ids[r - tri_count];
                if ((unsigned)source < (unsigned)tri_count) {
                    tp = &tris[suffix_base + source];
                    order_id = (unsigned int)(2 * source + 1);
                    // hipgl_near_clip_binrec tests ONLY degenerate.
                    live = !tp->degenerate;
                }
            }
            if (live) {
                const int tmin_x = math::max2i(tp->bbox_min_x, 0);
                const int tmin_y = math::max2i(tp->bbox_min_y, 0);
                const int tmax_x = math::min2i(tp->bbox_max_x, fb_width - 1);
                const int tmax_y = math::min2i(tp->bbox_max_y, fb_height - 1);
                if (tmin_x <= tmax_x && tmin_y <= tmax_y) {
                    lo_x = math::max2i(tmin_x / TILE_SIZE, 0);
                    lo_y = math::max2i(tmin_y / TILE_SIZE, 0);
                    const int hi_x = math::min2i(tmax_x / TILE_SIZE, tiles_x - 1);
                    const int hi_y = math::min2i(tmax_y / TILE_SIZE, tiles_y - 1);
                    if (lo_x <= hi_x && lo_y <= hi_y) {
                        nx = hi_x - lo_x + 1;
                        ny = hi_y - lo_y + 1;
                        n  = nx * ny;
                        z_ord = tp->z_ord;
                    }
                }
            }
        }
        s_lox[r] = lo_x;
        s_loy[r] = lo_y;
        s_nx [r] = nx;
        s_key[r] = ((unsigned long long)z_ord << 32) | (unsigned long long)order_id;
        s_off[r] = n;
    }
    if (tid == 0) { s_carry = 0; s_nlist = 0; s_nbig = 0; }
    __syncthreads();
    if (stop == 1) return;

    for (int base = 0; base < R; base += TPB) {
        const int i = base + tid;
        const int v = (i < R) ? s_off[i] : 0;
        int tot = 0;
        const int ex = micro_block_excl<TPB>(v, s_w, tot);
        const int carry = s_carry;
        __syncthreads();
        if (i < R) s_off[i] = ex + carry;
        if (tid == 0) s_carry = carry + tot;
        __syncthreads();
    }
    if (tid == 0) { s_off[R] = s_carry; s_total = s_carry; }
    __syncthreads();
    const int total_pairs = s_total;
    if (stop == 2) return;

    // ── PHASE 2: per-tile counts, in LDS ───────────────────────────────────
    for (int p = tid; p < total_pairs; p += TPB) {
        int lo = 0, hi = R;                 // largest j with s_off[j] <= p
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (s_off[mid] <= p) lo = mid; else hi = mid;
        }
        const int j = lo;
        const int local = p - s_off[j];
        const int jnx = s_nx[j];
        const int q = local / jnx;
        const int rr = local - q * jnx;
        atomicAdd(&s_cnt[(s_loy[j] + q) * tiles_x + (s_lox[j] + rr)], 1);
    }
    __syncthreads();

    if (stop == 3) return;
    // ── PHASE 3: exclusive scan of the tile counts, in LDS, ONE block scan ─
    // Each thread owns a contiguous run of CHUNK tiles: serial sum, one
    // block-wide scan of the thread totals, serial exclusive write-back. No
    // barrier ladder, no global round trip.
    {
        const int CHUNK = (HIPGL_MICRO_MAX_TILES + TPB - 1) / TPB;
        const int beg = tid * CHUNK;
        int sum = 0;
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            const int t = beg + k;
            if (t < num_tiles) sum += s_cnt[t];
        }
        int tot = 0;
        const int ex = micro_block_excl<TPB>(sum, s_w, tot);
        __syncthreads();
        int run = ex;
#pragma unroll
        for (int k = 0; k < CHUNK; ++k) {
            const int t = beg + k;
            if (t < num_tiles) { const int c = s_cnt[t]; s_cnt[t] = run; run += c; }
        }
        if (tid == 0) s_total = tot;
        __syncthreads();
    }
    // Publish the base array (the fine rasterizer reads it, and the host reads
    // tile_base[num_tiles] as the total-pairs telemetry word).
    for (int t = tid; t < num_tiles; t += TPB) tile_base[t] = s_cnt[t];
    if (tid == 0) tile_base[num_tiles] = s_total;

    if (stop == 4) return;
    const bool staged = (total_pairs <= HIPGL_MICRO_STAGE) &&
                        (total_pairs <= pool_capacity);
    // ── PHASE 4: scatter ───────────────────────────────────────────────────
    for (int p = tid; p < total_pairs; p += TPB) {
        int lo = 0, hi = R;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (s_off[mid] <= p) lo = mid; else hi = mid;
        }
        const int j = lo;
        const int local = p - s_off[j];
        const int jnx = s_nx[j];
        const int q = local / jnx;
        const int rr = local - q * jnx;
        const int t = (s_loy[j] + q) * tiles_x + (s_lox[j] + rr);
        const int slot = s_cnt[t] + atomicAdd(&s_cur[t], 1);
        if (staged) {
            // slot < total_pairs <= min(HIPGL_MICRO_STAGE, pool_capacity).
            s_pk[slot] = s_key[j];
            s_pt[slot] = t;
        } else if (slot < pool_capacity) {
            key_out[slot] = s_key[j];
        } else if (overflow_flag) {
            atomicExch(overflow_flag, 1);
        }
    }
    __syncthreads();

    if (stop == 5) return;

    if (staged) {
        for (int t = tid; t < num_tiles; t += TPB) cursor[t] = s_cur[t];
        for (int p = tid; p < total_pairs; p += TPB) {
            const int t = s_pt[p];
            const int b = s_cnt[t];
            const int i = p - b;
            int n = s_cur[t];
            // The block path this replaces staged at most HIPGL_MICRO_RECS
            // entries of a segment and left the remainder where the scatter
            // had put it; keep that behaviour bit-for-bit.
            if (n > HIPGL_MICRO_RECS) n = HIPGL_MICRO_RECS;
            const unsigned long long k = s_pk[p];
            if (i >= n) { key_out[p] = k; continue; }
            const unsigned long long* __restrict__ seg = &s_pk[b];
            int rank = 0;
            int j = 0;
            for (; j + 4 <= n; j += 4) {
                const unsigned long long a0 = seg[j];
                const unsigned long long a1 = seg[j + 1];
                const unsigned long long a2 = seg[j + 2];
                const unsigned long long a3 = seg[j + 3];
                rank += (a0 < k) + (a1 < k) + (a2 < k) + (a3 < k);
            }
            for (; j < n; ++j) rank += (seg[j] < k) ? 1 : 0;
            key_out[b + rank] = k;
        }
        return;
    }

    // ── PHASE 5a: publish cursor[] and queue the tiles that need an ordering.
    // A tile with <= 1 entry is already final -- the scatter wrote it in place.
    for (int t = tid; t < num_tiles; t += TPB) {
        const int n = s_cur[t];
        cursor[t] = n;
        if (n < 2) continue;
        if (s_cnt[t] >= pool_capacity) continue;
        if (n <= 64) s_list[atomicAdd(&s_nlist, 1)] = t;
        else         s_big[atomicAdd(&s_nbig, 1)] = t;
    }
    __syncthreads();
    if (stop == 6) return;

    {
        const int lane = tid & 63;
        const int wave = tid >> 6;
        const int nl   = s_nlist;
        for (int e = wave; e < nl; e += (TPB / 64)) {
            const int t = s_list[e];
            const int b = s_cnt[t];
            int n = s_cur[t];
            if (b + n > pool_capacity) n = pool_capacity - b;
            const unsigned long long k = (lane < n) ? key_out[b + lane] : 0ull;
            int rank = 0;
            for (int j = 0; j < n; ++j) {
                const unsigned long long o = __shfl(k, j, 64);
                rank += (o < k) ? 1 : 0;
            }
            if (lane < n) key_out[b + rank] = k;
        }
    }

    // ── PHASE 5c: the rare long segment, whole block, staged in LDS.
    // s_key is dead after the scatter, so the staging costs no extra LDS. A
    // segment cannot exceed 2*tri_count = HIPGL_MICRO_RECS entries.
    if (s_nbig > 0) {
        __syncthreads();
        const int nb = s_nbig;
        for (int e = 0; e < nb; ++e) {
            const int t = s_big[e];
            const int b = s_cnt[t];
            int n = s_cur[t];
            if (b + n > pool_capacity) n = pool_capacity - b;
            if (n > HIPGL_MICRO_RECS) n = HIPGL_MICRO_RECS;
            for (int i = tid; i < n; i += TPB) s_key[i] = key_out[b + i];
            __syncthreads();
            for (int i = tid; i < n; i += TPB) {
                const unsigned long long k = s_key[i];
                int rank = 0;
                for (int j = 0; j < n; ++j) rank += (s_key[j] < k) ? 1 : 0;
                key_out[b + rank] = k;
            }
            __syncthreads();
        }
    }
}

static inline int micro_tpb_choice() {
    static const int v = []{
        const char* e = getenv("HP_MICRO_TPB");
        const int t = e ? atoi(e) : HIPGL_MICRO_TPB;
        return (t == 256 || t == 512 || t == 1024) ? t : HIPGL_MICRO_TPB;
    }();
    return v;
}

template <int TPB>
static inline void micro_chain_dispatch(const MicroChainArgs& a, int nslots, hipStream_t stream)
{
    int blocks = nslots;
    if (a.clr.total > 0) blocks += (a.clr.total + TPB - 1) / TPB;
    hipLaunchKernelGGL((kernel_micro_bin_chain<TPB>), dim3(blocks), dim3(TPB), 0, stream, a);
}

// Host entry, N slots. Returns false when ANY slot does not qualify, in which
// case the caller must run the ordinary chain for all of them.
bool launch_micro_bin_chain_n(const MicroDrawSlotDesc* slots, int nslots,
                              int tiles_x, int tiles_y, int fb_width, int fb_height,
                              int num_tiles, int* overflow_flag, hipStream_t stream,
                              const MicroClearDesc* clr)
{
    if (!slots || nslots <= 0 || nslots > HIPGL_MICRO_MAX_SLOTS) return false;
    if (tiles_x <= 0 || tiles_y <= 0) return false;
    if (num_tiles != tiles_x * tiles_y) return false;
    if (num_tiles > HIPGL_MICRO_MAX_TILES) return false;
    MicroChainArgs a{};
    for (int i = 0; i < nslots; ++i) {
        const MicroDrawSlotDesc& d = slots[i];
        if (!d.tris || !d.tile_base || !d.cursor || !d.key_out) return false;
        if (d.tri_count <= 0 || d.tri_count > HIPGL_MICRO_MAX_TRIS) return false;
        if (d.suffix_base < d.tri_count) return false;
        if (d.suffix_base > HIPGL_MICRO_MAX_TRIS) return false;
        if (d.pool_capacity <= 0) return false;
        a.s[i] = d;
        a.packed[i] = (d.near_clip.ids && d.near_clip.count) ? 1 : 0;
    }
    static const int stop = getenv("HP_MICRO_STOP") ? atoi(getenv("HP_MICRO_STOP")) : 0;
    a.tiles_x = tiles_x; a.tiles_y = tiles_y;
    a.fb_width = fb_width; a.fb_height = fb_height;
    a.num_tiles = num_tiles; a.overflow_flag = overflow_flag; a.stop = stop;
    a.nslots = nslots;
    if (clr && clr->total > 0 && clr->color && clr->depth) a.clr = *clr;
    switch (micro_tpb_choice()) {
        case 256: micro_chain_dispatch<256>(a, nslots, stream); break;
        case 512: micro_chain_dispatch<512>(a, nslots, stream); break;
        default:  micro_chain_dispatch<HIPGL_MICRO_TPB>(a, nslots, stream); break;
    }
    return true;
}

// Host entry. Returns false when the draw does not qualify, in which case the
// caller must run the ordinary count -> scan -> scatter -> sort chain.
bool launch_micro_bin_chain(const TriangleSetup* tris, int tri_count,
                            int* tile_base, int* cursor,
                            unsigned long long* key_out,
                            int tiles_x, int tiles_y, int fb_width, int fb_height,
                            int num_tiles, int pool_capacity, int* overflow_flag,
                            int cull_mask, NearClipQueue near_clip, hipStream_t stream)
{
    MicroDrawSlotDesc d{};
    d.tris = tris; d.tri_count = tri_count; d.suffix_base = tri_count;
    d.cull_mask = cull_mask; d.tile_base = tile_base; d.cursor = cursor;
    d.key_out = key_out; d.pool_capacity = pool_capacity; d.near_clip = near_clip;
    return launch_micro_bin_chain_n(&d, 1, tiles_x, tiles_y, fb_width, fb_height,
                                    num_tiles, overflow_flag, stream, nullptr);
}

int micro_bin_max_slots() { return HIPGL_MICRO_MAX_SLOTS; }
int micro_bin_max_tris()  { return HIPGL_MICRO_MAX_TRIS; }
int micro_bin_max_tiles() { return HIPGL_MICRO_MAX_TILES; }

} // namespace kernels
} // namespace hipgl
