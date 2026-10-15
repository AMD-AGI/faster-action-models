// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/kernels.h"
#include "hipgl/fine_prefilter.h"
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdlib>

namespace hipgl {
namespace kernels {

typedef unsigned long long u64;

#define HP_WAVE 64
static constexpr u64 HP_KEY_MAX = ~0ull;
// Largest segment the LDS bitonic path can take. 2048 * 8 B = 16 KB of dynamic
// LDS -> 10 blocks/CU on a 160 KB gfx950 CU, which is above the 8 waves/SIMD
// hardware cap, so the LDS is not an occupancy limiter here.
static constexpr int HP_LDS_CAP = 4096;
// Largest segment pass 1 retires in registers (4 keys/lane on a wave64).
static constexpr int HP_WAVE_CAP = 256;
#ifndef HP_BIG_BLOCK_DEF
#define HP_BIG_BLOCK_DEF 256
#endif
#define HP_BIG_BLOCK HP_BIG_BLOCK_DEF
#define HP_BIG_GRID  64
// Largest segment pass 1 retires in registers on the EXTENDED ladder
// (HP_WAVE_CAP2/64 u64 per lane). Sweepable.
#ifndef HP_WAVE_CAP2_DEF
#define HP_WAVE_CAP2_DEF 256
#endif
static constexpr int HP_WAVE_CAP2 = HP_WAVE_CAP2_DEF;
static inline int hp_big_grid() {
    static const int v = getenv("HP_BIG_GRID_N") ? atoi(getenv("HP_BIG_GRID_N")) : HP_BIG_GRID;
    return v;
}

__device__ __forceinline__ uint32_t hp_f2ord(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ u64 hp_shfl_xor_u64(u64 v, int mask) {
    // Split explicitly: the 64-bit __shfl_xor overload lowers to exactly this
    // on CDNA, and doing it by hand keeps the code free of overload-resolution
    // surprises across HIP versions.
    uint32_t lo = (uint32_t)(v & 0xFFFFFFFFull);
    uint32_t hi = (uint32_t)(v >> 32);
    lo = (uint32_t)__shfl_xor((int)lo, mask, HP_WAVE);
    hi = (uint32_t)__shfl_xor((int)hi, mask, HP_WAVE);
    return ((u64)hi << 32) | (u64)lo;
}

// Register bitonic over K*64 keys held as v[t] at logical index t*64 + lane.
// The low 6 bits of the index are the lane, the high bits the register slot,
// so a stage with stride j < 64 is a cross-lane shuffle and j >= 64 is a
// register-to-register swap. Fully unrolled, no memory traffic of any kind.
template <int K>
__device__ __forceinline__ void hp_wave_bitonic(u64 (&v)[K], int lane) {
    constexpr int N = K * HP_WAVE;
#pragma unroll
    for (int k = 2; k <= N; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j < HP_WAVE) {
#pragma unroll
                for (int t = 0; t < K; ++t) {
                    const int i = t * HP_WAVE + lane;
                    const u64 p = hp_shfl_xor_u64(v[t], j);
                    const bool up    = ((i & k) == 0);
                    const bool lower = ((i & j) == 0);
                    const u64 a = v[t];
                    const u64 mn = (a < p) ? a : p;
                    const u64 mx = (a < p) ? p : a;
                    v[t] = (lower == up) ? mn : mx;
                }
            } else {
                const int tj = j >> 6;
#pragma unroll
                for (int t = 0; t < K; ++t) {
                    const int o = t ^ tj;
                    if (o > t) {
                        const int i = t * HP_WAVE + lane;
                        const bool up = ((i & k) == 0);
                        const u64 a = v[t], b = v[o];
                        const u64 mn = (a < b) ? a : b;
                        const u64 mx = (a < b) ? b : a;
                        v[t] = up ? mn : mx;
                        v[o] = up ? mx : mn;
                    }
                }
            }
        }
    }
}

template <int K>
__device__ __forceinline__ void hp_wave_sort_seg(
    const int* __restrict__ pool_tri, const float* __restrict__ pool_z,
    u64* __restrict__ out, int base, int n, int lane)
{
    u64 v[K];
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        if (i < n) {
            const int g = base + i;
            v[t] = ((u64)hp_f2ord(pool_z[g]) << 32) | (u64)(uint32_t)pool_tri[g];
        } else {
            v[t] = HP_KEY_MAX;
        }
    }
    hp_wave_bitonic<K>(v, lane);
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        if (i < n) out[base + i] = v[t];
    }
}

template <int K>
__device__ __forceinline__ void hp_wave_sort_seg_k(
    const u64* __restrict__ keys, u64* __restrict__ out, int base, int n, int lane)
{
    u64 v[K];
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        v[t] = (i < n) ? keys[base + i] : HP_KEY_MAX;
    }
    hp_wave_bitonic<K>(v, lane);
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        if (i < n) out[base + i] = v[t];
    }
}

template <int W>
__device__ __forceinline__ u64 hp_wave_bitonic_w(u64 v, int lane) {
#pragma unroll
    for (int k = 2; k <= W; k <<= 1) {
#pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            const u64 p      = hp_shfl_xor_u64(v, j);
            const bool up    = ((lane & k) == 0);
            const bool lower = ((lane & j) == 0);
            const u64 mn = (v < p) ? v : p;
            const u64 mx = (v < p) ? p : v;
            v = (lower == up) ? mn : mx;
        }
    }
    return v;
}

template <int W>
__device__ __forceinline__ void hp_wave_sort_small(
    const u64* __restrict__ keys, u64* __restrict__ out, int base, int n, int lane)
{
    u64 v = (lane < n) ? keys[base + lane] : HP_KEY_MAX;
    v = hp_wave_bitonic_w<W>(v, lane);
    if (lane < n) out[base + lane] = v;
}

template <int W>
__device__ __forceinline__ u64 hp_wave_small_r(
    const u64* __restrict__ keys, u64* __restrict__ out, int base, int n, int lane)
{
    u64 v = (lane < n) ? keys[base + lane] : HP_KEY_MAX;
    v = hp_wave_bitonic_w<W>(v, lane);
    if (lane < n) out[base + lane] = v;
    return v;
}

// n <= HP_WAVE only. Returns this lane's sorted key (index == lane).
__device__ __forceinline__ u64 hp_wave_ladder_r(
    const u64* __restrict__ keys, u64* __restrict__ out, int base, int n, int lane)
{
    if (n <= 1) {
        // n may be 0 with base == pool_cap, so the load itself must be guarded.
        const u64 v = (n == 1 && lane == 0) ? keys[base] : HP_KEY_MAX;
        if (n == 1 && lane == 0) out[base] = v;
        return v;
    }
    if (n <= 2)  return hp_wave_small_r<2>(keys, out, base, n, lane);
    if (n <= 4)  return hp_wave_small_r<4>(keys, out, base, n, lane);
    if (n <= 8)  return hp_wave_small_r<8>(keys, out, base, n, lane);
    if (n <= 16) return hp_wave_small_r<16>(keys, out, base, n, lane);
    if (n <= 32) return hp_wave_small_r<32>(keys, out, base, n, lane);
    return hp_wave_small_r<64>(keys, out, base, n, lane);
}

template <int K>
__device__ __forceinline__ void hp_wave_sort_seg_kl(
    const u64* __restrict__ keys, u64* __restrict__ out, u64* lds,
    int base, int n, int lane)
{
    u64 v[K];
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        v[t] = (i < n) ? keys[base + i] : HP_KEY_MAX;
    }
    hp_wave_bitonic<K>(v, lane);
#pragma unroll
    for (int t = 0; t < K; ++t) {
        const int i = t * HP_WAVE + lane;
        if (i < n) { out[base + i] = v[t]; lds[i] = v[t]; }
    }
}

// HP_WAVE < n <= WCAP. Same networks as hp_wave_ladder, plus the LDS stage.
template <int WCAP>
__device__ __forceinline__ void hp_wave_ladder_l(
    const u64* __restrict__ keys, u64* __restrict__ out, u64* lds,
    int base, int n, int lane)
{
    if (n <= 128)      hp_wave_sort_seg_kl<2>(keys, out, lds, base, n, lane);
    else if (n <= 256) hp_wave_sort_seg_kl<4>(keys, out, lds, base, n, lane);
    else               hp_wave_sort_seg_kl<(WCAP >= 512 ? 8 : 4)>(keys, out, lds, base, n, lane);
}

// The register ladder: one wave64, pure registers, no LDS, no barrier.
template <int WCAP>
__device__ __forceinline__ void hp_wave_ladder(
    const u64* __restrict__ keys, u64* __restrict__ out, int base, int n, int lane)
{
    if (n <= 1) {
        if (n == 1 && lane == 0) out[base] = keys[base];
    } else if (n <= 2) {
        hp_wave_sort_small<2>(keys, out, base, n, lane);
    } else if (n <= 4) {
        hp_wave_sort_small<4>(keys, out, base, n, lane);
    } else if (n <= 8) {
        hp_wave_sort_small<8>(keys, out, base, n, lane);
    } else if (n <= 16) {
        hp_wave_sort_small<16>(keys, out, base, n, lane);
    } else if (n <= 32) {
        hp_wave_sort_small<32>(keys, out, base, n, lane);
    } else if (n <= 64) {
        hp_wave_sort_seg_k<1>(keys, out, base, n, lane);
    } else if (n <= 128) {
        hp_wave_sort_seg_k<2>(keys, out, base, n, lane);
    } else if (n <= 256) {
        hp_wave_sort_seg_k<4>(keys, out, base, n, lane);
    } else {
        hp_wave_sort_seg_k<(WCAP >= 512 ? 8 : 4)>(keys, out, base, n, lane);
    }
}

template <int PCH>
__device__ __forceinline__ void hp_block_bitonic_k(
    const u64* __restrict__ keys, u64* __restrict__ out,
    int base, int n, int tid, int nthread, u64* lds)
{
    constexpr int PK = PCH / HP_WAVE;   // u64 registers per lane in the presort
    int P = 1;
    while (P < n) P <<= 1;
    int k0 = 2;
    if (P >= PCH) {
        const int lane   = tid & 63;
        const int wave   = tid >> 6;
        const int nwaves = nthread >> 6;
        const int nchunk = P / PCH;
        for (int c = wave; c < nchunk; c += nwaves) {
            const int cb = c * PCH;
            u64 v[PK];
#pragma unroll
            for (int t = 0; t < PK; ++t) {
                const int i = cb + t * HP_WAVE + lane;
                v[t] = (i < n) ? keys[base + i] : HP_KEY_MAX;
            }
            hp_wave_bitonic<PK>(v, lane);
            const bool desc = (c & 1) != 0;
#pragma unroll
            for (int t = 0; t < PK; ++t) {
                const int i = t * HP_WAVE + lane;
                lds[cb + (desc ? (PCH - 1 - i) : i)] = v[t];
            }
        }
        k0 = 2 * PCH;
    } else {
        for (int i = tid; i < P; i += nthread) lds[i] = (i < n) ? keys[base + i] : HP_KEY_MAX;
    }
    __syncthreads();
    const int half = P >> 1;
    for (int k = k0; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int idx = tid; idx < half; idx += nthread) {
                const int i = ((idx & ~(j - 1)) << 1) | (idx & (j - 1));
                const int l = i | j;
                const bool up = ((i & k) == 0);
                const u64 a = lds[i], b = lds[l];
                if ((a > b) == up) { lds[i] = b; lds[l] = a; }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < n; i += nthread) out[base + i] = lds[i];
    __syncthreads();
}

template <int PCH, int MG_EPT>
__device__ __forceinline__ bool hp_block_mergesort_k(
    const u64* __restrict__ keys, u64* __restrict__ out,
    int base, int n, int tid, int nthread, u64* lds, int lds_cap)
{
    constexpr int PK = PCH / HP_WAVE;
    int P = 1;
    while (P < n) P <<= 1;
    // Block-uniform bail-out: no barrier has run yet, so the caller's fallback
    // cannot desynchronise the block.
    if (P < PCH || (2 * P) > lds_cap) return false;
    int nrounds = 0;
    for (int L = PCH; L < P; L <<= 1) ++nrounds;
    // Choose the starting buffer so the LAST round lands back in lds[0..P),
    // which is where the fused prefilter epilogue expects the sorted keys.
    u64* src = (nrounds & 1) ? (lds + P) : lds;
    u64* dst = (nrounds & 1) ? lds : (lds + P);

    {   // register presort: one wave per PCH-chunk, all runs ASCENDING
        const int lane   = tid & 63;
        const int wave   = tid >> 6;
        const int nwaves = nthread >> 6;
        const int nchunk = P / PCH;
        for (int c = wave; c < nchunk; c += nwaves) {
            const int cb = c * PCH;
            u64 v[PK];
#pragma unroll
            for (int t = 0; t < PK; ++t) {
                const int i = cb + t * HP_WAVE + lane;
                v[t] = (i < n) ? keys[base + i] : HP_KEY_MAX;
            }
            hp_wave_bitonic<PK>(v, lane);
#pragma unroll
            for (int t = 0; t < PK; ++t) src[cb + t * HP_WAVE + lane] = v[t];
        }
    }
    __syncthreads();

    for (int L = PCH; L < P; L <<= 1) {
        const int W = L << 1;                       // width of a merged run
        for (int g0 = tid * MG_EPT; g0 < P; g0 += nthread * MG_EPT) {
            const int pb = (g0 / W) * W;            // base of this thread's pair
            const int lo = g0 - pb;                 // output offset inside it
            const u64* __restrict__ A = src + pb;
            const u64* __restrict__ B = src + pb + L;
            // Merge path: smallest i with A[i] > B[lo-i-1].  A[] rises and
            // B[lo-i-1] falls in i, so the predicate is monotone.  Both indices
            // are provably in range over the search interval below.
            int a0 = (lo > L) ? (lo - L) : 0;
            int a1 = (lo < L) ? lo : L;
            while (a0 < a1) {
                const int m = (a0 + a1) >> 1;
                if (A[m] > B[lo - m - 1]) a1 = m; else a0 = m + 1;
            }
            int i = a0, j = lo - a0;
            u64 av = (i < L) ? A[i] : HP_KEY_MAX;
            u64 bv = (j < L) ? B[j] : HP_KEY_MAX;
            u64* __restrict__ o = dst + pb + lo;
#pragma unroll
            for (int e = 0; e < MG_EPT; ++e) {
                const bool takeA = (av <= bv);
                o[e] = takeA ? av : bv;
                if (takeA) { ++i; av = (i < L) ? A[i] : HP_KEY_MAX; }
                else       { ++j; bv = (j < L) ? B[j] : HP_KEY_MAX; }
            }
        }
        __syncthreads();
        u64* sw = src; src = dst; dst = sw;
    }
    for (int i = tid; i < n; i += nthread) out[base + i] = src[i];
    __syncthreads();
    return true;
}

// Block sort front door: merge-path when the already-reserved LDS holds the
// ping-pong pair, the unchanged bitonic otherwise.  Leaves the sorted keys in
// lds[0..n) either way.
template <int PCH, int MG_EPT>
__device__ __forceinline__ void hp_block_sort_k(
    const u64* __restrict__ keys, u64* __restrict__ out,
    int base, int n, int tid, int nthread, u64* lds, int lds_cap)
{
    if (!hp_block_mergesort_k<PCH, MG_EPT>(keys, out, base, n, tid, nthread, lds, lds_cap))
        hp_block_bitonic_k<PCH>(keys, out, base, n, tid, nthread, lds);
}

__device__ __forceinline__ void hp_block_rank_k(
    const u64* __restrict__ keys, u64* __restrict__ out,
    int base, int n, int tid, int nthread)
{
    for (int i = tid; i < n; i += nthread) {
        const u64 ki = keys[base + i];
        int rank = 0;
        for (int j = 0; j < n; ++j) rank += (keys[base + j] < ki) ? 1 : 0;
        out[base + rank] = ki;
    }
    __syncthreads();
}

// Whole-block LDS bitonic for a segment too large for one wave's registers.
// P = next power of two >= n, padded with KEY_MAX. Up to HP_LDS_CAP keys.
__device__ __forceinline__ void hp_block_bitonic(
    const int* __restrict__ pool_tri, const float* __restrict__ pool_z,
    u64* __restrict__ out, int base, int n, int tid, int nthread, u64* lds)
{
    int P = 1;
    while (P < n) P <<= 1;
    for (int i = tid; i < P; i += nthread) {
        if (i < n) {
            const int g = base + i;
            lds[i] = ((u64)hp_f2ord(pool_z[g]) << 32) | (u64)(uint32_t)pool_tri[g];
        } else {
            lds[i] = HP_KEY_MAX;
        }
    }
    __syncthreads();
    for (int k = 2; k <= P; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < P; i += nthread) {
                const int l = i ^ j;
                if (l > i) {
                    const bool up = ((i & k) == 0);
                    const u64 a = lds[i], b = lds[l];
                    if ((a > b) == up) { lds[i] = b; lds[l] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (int i = tid; i < n; i += nthread) out[base + i] = lds[i];
    __syncthreads();
}

__device__ __forceinline__ void hp_block_rank(
    const int* __restrict__ pool_tri, const float* __restrict__ pool_z,
    u64* __restrict__ out, int base, int n, int tid, int nthread)
{
    for (int i = tid; i < n; i += nthread) {
        const int gi = base + i;
        const u64 ki = ((u64)hp_f2ord(pool_z[gi]) << 32) | (u64)(uint32_t)pool_tri[gi];
        int rank = 0;
        for (int j = 0; j < n; ++j) {
            const int gj = base + j;
            const u64 kj = ((u64)hp_f2ord(pool_z[gj]) << 32) | (u64)(uint32_t)pool_tri[gj];
            rank += (kj < ki) ? 1 : 0;
        }
        out[base + rank] = ki;
    }
    __syncthreads();
}

// Debug/diagnostic: largest segment length seen. Armed by HIPGL_LOG_MAXSEG.
__device__ unsigned int g_hp_max_seg = 0u;

__device__ __forceinline__ void hp_seg_bounds(const int* __restrict__ tile_base,
                                              const int* __restrict__ tile_count,
                                              int tile, int pool_cap,
                                              int& base, int& n)
{
    int b = tile_base[tile];
    if (b > pool_cap) b = pool_cap;
    long long e = (long long)tile_base[tile] + (long long)tile_count[tile];
    if (e > (long long)pool_cap) e = pool_cap;
    if (e < (long long)b) e = b;
    base = b;
    n = (int)(e - b);
}

// ---- PASS 1: one wave64 per tile, pure registers, no LDS ------------------
// Covers every segment up to HP_WAVE_CAP keys. At r1024 the mean segment is 40
// and this pass retires essentially the whole pool.
template <bool LOG_MAX>
HIPGL_GLOBAL void __launch_bounds__(256)
kernel_tile_wave_sort(const int* __restrict__ pool_tri,
                      const float* __restrict__ pool_z,
                      const int* __restrict__ tile_base,
                      const int* __restrict__ tile_count,
                      u64* __restrict__ out,
                      int* __restrict__ big_list, int* __restrict__ big_count,
                      int big_cap, int num_tiles, int pool_cap)
{
    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int tile = blockIdx.x * 4 + wave;
    if (tile >= num_tiles) return;

    int base, n;
    hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
    if (LOG_MAX && lane == 0 && n > 0) atomicMax(&g_hp_max_seg, (unsigned)n);

    if (n <= 1) {
        if (n == 1 && lane == 0)
            out[base] = ((u64)hp_f2ord(pool_z[base]) << 32) | (u64)(uint32_t)pool_tri[base];
    } else if (n <= 64) {
        hp_wave_sort_seg<1>(pool_tri, pool_z, out, base, n, lane);
    } else if (n <= HP_WAVE_CAP) {
        hp_wave_sort_seg<4>(pool_tri, pool_z, out, base, n, lane);
    } else if (lane == 0) {
        // Heavy tail: append to the compacted work list for pass 2. Ordering
        // inside the list is irrelevant -- each entry is an independent segment.
        const int slot = atomicAdd(big_count, 1);
        if (slot < big_cap) big_list[slot] = tile;
    }
}

// ---- PASS 2: mop-up for the heavy tail ------------------------------------
// One 256-thread block walks 4 consecutive tiles and processes, with the WHOLE
// block, only those the wave pass could not take. Blocks whose 4 tiles are all
// small exit after 8 integer loads, which is why this needs no work list, no
// device atomic and no extra allocation -- and stays deterministic.
HIPGL_GLOBAL void __launch_bounds__(HP_BIG_BLOCK)
kernel_tile_big_sort(const int* __restrict__ pool_tri,
                     const float* __restrict__ pool_z,
                     const int* __restrict__ tile_base,
                     const int* __restrict__ tile_count,
                     u64* __restrict__ out,
                     const int* __restrict__ big_list,
                     const int* __restrict__ big_count,
                     int big_cap, int pool_cap)
{
    extern __shared__ u64 s_lds[];
    int nbig = *big_count;
    if (nbig > big_cap) nbig = big_cap;
    const int tid = threadIdx.x;
    for (int w = blockIdx.x; w < nbig; w += gridDim.x) {
        const int tile = big_list[w];
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (n <= HP_LDS_CAP) hp_block_bitonic(pool_tri, pool_z, out, base, n, tid, HP_BIG_BLOCK, s_lds);
        else                 hp_block_rank  (pool_tri, pool_z, out, base, n, tid, HP_BIG_BLOCK);
    }
}

template <bool LOG_MAX>
HIPGL_GLOBAL void __launch_bounds__(256)
kernel_tile_wave_sort_k(const u64* __restrict__ keys,
                        const int* __restrict__ tile_base,
                        const int* __restrict__ tile_count,
                        u64* __restrict__ out,
                        int* __restrict__ big_list, int* __restrict__ big_count,
                        int big_cap, int num_tiles, int pool_cap)
{
    const int tid  = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int tile = blockIdx.x * 4 + wave;
    if (tile >= num_tiles) return;

    int base, n;
    hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
    if (LOG_MAX && lane == 0 && n > 0) atomicMax(&g_hp_max_seg, (unsigned)n);

    if (n <= 1) {
        if (n == 1 && lane == 0) out[base] = keys[base];
    } else if (n <= 64) {
        hp_wave_sort_seg_k<1>(keys, out, base, n, lane);
    } else if (n <= 128) {
        hp_wave_sort_seg_k<2>(keys, out, base, n, lane);
    } else if (n <= HP_WAVE_CAP) {
        hp_wave_sort_seg_k<4>(keys, out, base, n, lane);
    } else if (n <= HP_WAVE_CAP2) {
        hp_wave_sort_seg_k<HP_WAVE_CAP2 / HP_WAVE>(keys, out, base, n, lane);
    } else if (lane == 0) {
        const int slot = atomicAdd(big_count, 1);
        if (slot < big_cap) big_list[slot] = tile;
    }
}

template <int BS>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_big_sort_kt(const u64* __restrict__ keys,
                        const int* __restrict__ tile_base,
                        const int* __restrict__ tile_count,
                        u64* __restrict__ out,
                        const int* __restrict__ big_list,
                        const int* __restrict__ big_count,
                        int big_cap, int pool_cap)
{
    extern __shared__ u64 s_lds[];
    int nbig = *big_count;
    if (nbig > big_cap) nbig = big_cap;
    const int tid = threadIdx.x;
    for (int w = blockIdx.x; w < nbig; w += gridDim.x) {
        const int tile = big_list[w];
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (n <= HP_LDS_CAP) hp_block_bitonic_k<256>(keys, out, base, n, tid, BS, s_lds);
        else                 hp_block_rank_k  (keys, out, base, n, tid, BS);
    }
}

HIPGL_GLOBAL void __launch_bounds__(HP_BIG_BLOCK)
kernel_tile_big_sort_k(const u64* __restrict__ keys,
                       const int* __restrict__ tile_base,
                       const int* __restrict__ tile_count,
                       u64* __restrict__ out,
                       const int* __restrict__ big_list,
                       const int* __restrict__ big_count,
                       int big_cap, int pool_cap)
{
    extern __shared__ u64 s_lds[];
    int nbig = *big_count;
    if (nbig > big_cap) nbig = big_cap;
    const int tid = threadIdx.x;
    for (int w = blockIdx.x; w < nbig; w += gridDim.x) {
        const int tile = big_list[w];
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (n <= HP_LDS_CAP) hp_block_bitonic_k<256>(keys, out, base, n, tid, HP_BIG_BLOCK, s_lds);
        else                 hp_block_rank_k  (keys, out, base, n, tid, HP_BIG_BLOCK);
    }
}

// Hipify-safe template dispatch (never a <<<>>> inside an if/else macro).
static inline int hp_big_block() {
    static const int v = getenv("HP_BIG_BLOCK_N") ? atoi(getenv("HP_BIG_BLOCK_N")) : 1024;
    return v;
}
static void hp_launch_big_k(int bs, int grid, size_t lds, hipStream_t stream,
                            const u64* keys, const int* tile_base, const int* tile_count,
                            u64* out, const int* big_list, const int* big_count,
                            int big_cap, int pool_cap)
{
    switch (bs) {
    case 256:
        hipLaunchKernelGGL((kernel_tile_big_sort_kt<256>), dim3(grid), dim3(256), lds, stream,
                           keys, tile_base, tile_count, out, big_list, big_count, big_cap, pool_cap);
        break;
    case 512:
        hipLaunchKernelGGL((kernel_tile_big_sort_kt<512>), dim3(grid), dim3(512), lds, stream,
                           keys, tile_base, tile_count, out, big_list, big_count, big_cap, pool_cap);
        break;
    default:
        hipLaunchKernelGGL((kernel_tile_big_sort_kt<1024>), dim3(grid), dim3(1024), lds, stream,
                           keys, tile_base, tile_count, out, big_list, big_count, big_cap, pool_cap);
        break;
    }
}


template <int BS, bool FROM_LDS>
__device__ __forceinline__ void hp_fine_pre_emit(
    const u64* __restrict__ sorted, const u64* lsorted, int base, int n,
    const TriangleSetup* __restrict__ tris, FinePreState pst,
    int tiles_x, int tiles_mod, int slot,
    float4* __restrict__ rec, int* __restrict__ cnt,
    unsigned short* __restrict__ msk, int tid)
{
    __shared__ int s_wc[2][BS / 64];
    const int tile   = (tiles_mod > 0) ? (slot % tiles_mod) : slot;
    const int tile_x = tile % tiles_x;
    const int tile_y = tile / tiles_x;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    constexpr int NW = BS / 64;
    if (n > MAX_TRIANGLES_PER_TILE) n = MAX_TRIANGLES_PER_TILE;
    int nsurv = 0;
    int buf = 0;
    for (int c0 = 0; c0 < n; c0 += BS) {
        float4 r0, r1, r2, r3;
        unsigned int cov = 0u;
        bool keep = false;
        const int i = c0 + tid;
        if (i < n) {
            float4 rr[4];
            const u64 k = FROM_LDS ? lsorted[i] : sorted[base + i];
            keep = hipgl_fine_prefilter_emit(tris, k, pst, tile_x, tile_y, rr,
                                             msk ? &cov : nullptr);
            r0 = rr[0]; r1 = rr[1]; r2 = rr[2]; r3 = rr[3];
        }
        const u64 bal  = __ballot(keep);
        const int rank = (int)__popcll(bal & ((1ull << lane) - 1ull));
        if (lane == 0) s_wc[buf][wv] = (int)__popcll(bal);
        __syncthreads();
        int wbase = 0, ntot = 0;
#pragma unroll
        for (int w = 0; w < NW; w++) { if (w < wv) wbase += s_wc[buf][w]; ntot += s_wc[buf][w]; }
        if (keep) {
            const int didx = base + nsurv + wbase + rank;
            float4* __restrict__ d = rec + (size_t)didx * 4;
            d[0] = r0; d[1] = r1; d[2] = r2; d[3] = r3;
            if (msk) msk[didx] = (unsigned short)cov;
        }
        nsurv += ntot;
        buf ^= 1;
    }
    if (tid == 0) cnt[slot] = nsurv;
}

__device__ __forceinline__ void hp_fine_pre_emit_w(
    u64 key, bool active, int base, int slot,
    const TriangleSetup* __restrict__ tris, FinePreState pst,
    int tiles_x, int tiles_mod,
    float4* __restrict__ rec, int* __restrict__ cnt,
    unsigned short* __restrict__ msk, int lane)
{
    const int tile   = (tiles_mod > 0) ? (slot % tiles_mod) : slot;
    const int tile_x = tile % tiles_x;
    const int tile_y = tile / tiles_x;
    float4 rr[4];
    unsigned int cov = 0u;
    bool keep = false;
    if (active) keep = hipgl_fine_prefilter_emit(tris, key, pst, tile_x, tile_y, rr,
                                                msk ? &cov : nullptr);
    const u64 bal  = __ballot(keep);
    const int rank = (int)__popcll(bal & ((1ull << lane) - 1ull));
    if (keep) {
        float4* __restrict__ d = rec + (size_t)(base + rank) * 4;
        d[0] = rr[0]; d[1] = rr[1]; d[2] = rr[2]; d[3] = rr[3];
        if (msk) msk[base + rank] = (unsigned short)cov;
    }
    if (lane == 0) cnt[slot] = (int)__popcll(bal);
}

template <bool LOG_MAX, int BS, int WCAP, int PCH, int MG_EPT>
HIPGL_DEVICE __forceinline__ void
hp_tile_sort_fused_one(const u64* __restrict__ keys,
                       const int* __restrict__ tile_base,
                       const int* __restrict__ tile_count,
                       u64* __restrict__ out,
                       int tile, int num_tiles, int pool_cap, int lds_cap, u64* s_lds,
                       const TriangleSetup* __restrict__ pre_tris, FinePreState pre_st,
                       int pre_tiles_x, int pre_tiles_mod,
                       float4* __restrict__ pre_rec, int* __restrict__ pre_cnt,
                       unsigned short* __restrict__ pre_msk)
{
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    {
        if (tile >= num_tiles) return;
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (LOG_MAX && tid == 0 && n > 0) atomicMax(&g_hp_max_seg, (unsigned)n);
        if (pre_rec == nullptr) {
            if (n <= WCAP) {
                // block-uniform: n is the same for every thread, so this return
                // never splits a __syncthreads.
                if (tid < HP_WAVE) hp_wave_ladder<WCAP>(keys, out, base, n, lane);
            } else if (n <= lds_cap) {
                hp_block_sort_k<PCH, MG_EPT>(keys, out, base, n, tid, BS, s_lds, lds_cap);
            } else {
                hp_block_rank_k  (keys, out, base, n, tid, BS);
            }
            return;
        }
        if (n <= HP_WAVE) {
            // Sorted segment lives in wave 0's registers -> fully fused, no
            // LDS, no barrier, no dependent global load.
            if (tid < HP_WAVE) {
                const u64 kv = hp_wave_ladder_r(keys, out, base, n, lane);
                hp_fine_pre_emit_w(kv, lane < n, base, tile, pre_tris, pre_st,
                                   pre_tiles_x, pre_tiles_mod, pre_rec, pre_cnt, pre_msk, lane);
            }
            return;
        }
        if (n <= WCAP) {
            if (tid < HP_WAVE) hp_wave_ladder_l<WCAP>(keys, out, s_lds, base, n, lane);
            __syncthreads();
            hp_fine_pre_emit<BS, true>(out, s_lds, base, n, pre_tris, pre_st,
                                       pre_tiles_x, pre_tiles_mod, tile, pre_rec, pre_cnt,
                                       pre_msk, tid);
        } else if (n <= lds_cap) {
            // the block sort leaves the sorted keys in s_lds[0..n).
            hp_block_sort_k<PCH, MG_EPT>(keys, out, base, n, tid, BS, s_lds, lds_cap);
            hp_fine_pre_emit<BS, true>(out, s_lds, base, n, pre_tris, pre_st,
                                       pre_tiles_x, pre_tiles_mod, tile, pre_rec, pre_cnt,
                                       pre_msk, tid);
        } else {
            // O(n^2) rank scatter (unreachable in practice); keys only exist in
            // global, so this one path keeps the original global source.
            hp_block_rank_k  (keys, out, base, n, tid, BS);
            hp_fine_pre_emit<BS, false>(out, nullptr, base, n, pre_tris, pre_st,
                                        pre_tiles_x, pre_tiles_mod, tile, pre_rec, pre_cnt,
                                        pre_msk, tid);
        }
        return;
    }
}

template <bool LOG_MAX, int BS, int WCAP, int PCH, int TPB, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused(const u64* __restrict__ keys,
                       const int* __restrict__ tile_base,
                       const int* __restrict__ tile_count,
                       u64* __restrict__ out,
                       int num_tiles, int pool_cap, int lds_cap,
                       const TriangleSetup* __restrict__ pre_tris, FinePreState pre_st,
                       int pre_tiles_x, int pre_tiles_mod,
                       float4* __restrict__ pre_rec, int* __restrict__ pre_cnt,
                       unsigned short* __restrict__ pre_msk)
{
    extern __shared__ u64 s_lds[];
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int t0   = blockIdx.x * TPB;

    if (TPB == 1) {
        hp_tile_sort_fused_one<LOG_MAX, BS, WCAP, PCH, MG_EPT>(
            keys, tile_base, tile_count, out, t0, num_tiles, pool_cap, lds_cap, s_lds,
            pre_tris, pre_st, pre_tiles_x, pre_tiles_mod, pre_rec, pre_cnt, pre_msk);
        return;
    }
    // TPB > 1: the block's waves retire the block's TPB tiles independently,
    // then the WHOLE block mops up whichever of those tiles the register
    // ladder could not take. Same order, still one dispatch, but 1/TPB of the
    // blocks -- which is what the 1024-tile grid wants, where the mean segment
    // is 26 and a block-per-tile grid leaves 3 of every 4 waves idle.
    const int wave   = tid >> 6;
    const int nwaves = BS >> 6;
    for (int u = wave; u < TPB; u += nwaves) {
        const int tile = t0 + u;
        if (tile >= num_tiles) continue;
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (LOG_MAX && lane == 0 && n > 0) atomicMax(&g_hp_max_seg, (unsigned)n);
        if (n <= WCAP) hp_wave_ladder<WCAP>(keys, out, base, n, lane);
    }
    __syncthreads();
    for (int u = 0; u < TPB; ++u) {
        const int tile = t0 + u;
        if (tile >= num_tiles) break;
        int base, n;
        hp_seg_bounds(tile_base, tile_count, tile, pool_cap, base, n);
        if (n <= WCAP) continue;
        if (n <= lds_cap) hp_block_sort_k<PCH, MG_EPT>(keys, out, base, n, tid, BS, s_lds, lds_cap);
        else              hp_block_rank_k  (keys, out, base, n, tid, BS);
    }
}

template <bool LOG_MAX, int BS, int WCAP, int PCH, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused_z2(SortSlot s0, SortSlot s1,
                          int num_tiles, int pool_cap, int lds_cap)
{
    extern __shared__ u64 s_lds[];
    const SortSlot& s = (blockIdx.z == 0) ? s0 : s1;
    hp_tile_sort_fused_one<LOG_MAX, BS, WCAP, PCH, MG_EPT>(
        reinterpret_cast<const u64*>(s.keys), s.tile_base, s.tile_count,
        reinterpret_cast<u64*>(s.out),
        (int)blockIdx.x, num_tiles, pool_cap, lds_cap, s_lds,
        s.pre.tris, s.pre.st, s.pre.tiles_x, s.pre.tiles_mod,
        s.pre.rec, s.pre.cnt, s.pre.msk);
}

template <int BS, int WCAP, int PCH, int SPT, int MG_EPT>
HIPGL_DEVICE __forceinline__ void
hp_tile_sort_fused_multi(const u64* __restrict__ keys,
                         const int* __restrict__ tile_base,
                         const int* __restrict__ tile_count,
                         u64* __restrict__ out,
                         int s_first, int s_step, int num_slots, int pool_cap, int lds_cap,
                         u64* s_lds,
                         const TriangleSetup* __restrict__ pre_tris, FinePreState pre_st,
                         int pre_tiles_x, int pre_tiles_mod,
                         float4* __restrict__ pre_rec, int* __restrict__ pre_cnt,
                         unsigned short* __restrict__ pre_msk)
{
    static_assert(SPT >= 1 && SPT <= 64, "SPT must fit one wave");
    constexpr int NW = BS / 64;
    __shared__ int s_slot[2][SPT];
    __shared__ int s_base[2][SPT];
    __shared__ int s_n[2][SPT];
    __shared__ int s_cnt[2];
    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int wv   = tid >> 6;
    if (wv == 0) {
        const int slot = s_first + lane * s_step;
        const bool valid = (lane < SPT) && (slot < num_slots);
        int base = 0, n = 0;
        if (valid) hp_seg_bounds(tile_base, tile_count, slot, pool_cap, base, n);
        if (valid && n == 0) pre_cnt[slot] = 0;
        const bool sm = valid && n >= 1 && n <= HP_WAVE;
        const bool bg = valid && n > HP_WAVE;
        const u64 bs = __ballot(sm);
        const u64 bb = __ballot(bg);
        const u64 lt = (1ull << lane) - 1ull;
        if (sm) {
            const int r = (int)__popcll(bs & lt);
            s_slot[0][r] = slot; s_base[0][r] = base; s_n[0][r] = n;
        }
        if (bg) {
            const int r = (int)__popcll(bb & lt);
            s_slot[1][r] = slot; s_base[1][r] = base; s_n[1][r] = n;
        }
        if (lane == 0) { s_cnt[0] = (int)__popcll(bs); s_cnt[1] = (int)__popcll(bb); }
    }
    __syncthreads();
    const int nsmall = s_cnt[0];
    const int nbig   = s_cnt[1];
    // SMALL: one wave per slot, register-only, no barrier.
    for (int k = wv; k < nsmall; k += NW) {
        const int slot = s_slot[0][k], base = s_base[0][k], n = s_n[0][k];
        const u64 kv = hp_wave_ladder_r(keys, out, base, n, lane);
        hp_fine_pre_emit_w(kv, lane < n, base, slot, pre_tris, pre_st,
                           pre_tiles_x, pre_tiles_mod, pre_rec, pre_cnt, pre_msk, lane);
    }
    // BIG: the unchanged block-wide ladder, one slot at a time.  nbig and every
    // n are block-uniform (read from LDS after the barrier above).
    for (int k = 0; k < nbig; ++k) {
        const int slot = s_slot[1][k], base = s_base[1][k], n = s_n[1][k];
        if (k > 0) __syncthreads();   // s_lds / s_wc reuse across slots
        if (n <= WCAP) {
            if (tid < HP_WAVE) hp_wave_ladder_l<WCAP>(keys, out, s_lds, base, n, lane);
            __syncthreads();
            hp_fine_pre_emit<BS, true>(out, s_lds, base, n, pre_tris, pre_st,
                                       pre_tiles_x, pre_tiles_mod, slot, pre_rec, pre_cnt,
                                       pre_msk, tid);
        } else if (n <= lds_cap) {
            hp_block_sort_k<PCH, MG_EPT>(keys, out, base, n, tid, BS, s_lds, lds_cap);
            hp_fine_pre_emit<BS, true>(out, s_lds, base, n, pre_tris, pre_st,
                                       pre_tiles_x, pre_tiles_mod, slot, pre_rec, pre_cnt,
                                       pre_msk, tid);
        } else {
            hp_block_rank_k  (keys, out, base, n, tid, BS);
            hp_fine_pre_emit<BS, false>(out, nullptr, base, n, pre_tris, pre_st,
                                        pre_tiles_x, pre_tiles_mod, slot, pre_rec, pre_cnt,
                                        pre_msk, tid);
        }
    }
}

// Slot -> CTA map.  STRIDED=false: CTA b owns slots [b*SPT, b*SPT+SPT).
// STRIDED=true: CTA b owns slots b, b+G, b+2G, ... (G = gridDim.x), which
// spreads one span's contiguous run of occupied tiles over SPT different CTAs.
template <int BS, int WCAP, int PCH, int SPT, bool STRIDED, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused_ms(const u64* __restrict__ keys,
                          const int* __restrict__ tile_base,
                          const int* __restrict__ tile_count,
                          u64* __restrict__ out,
                          int num_slots, int pool_cap, int lds_cap,
                          const TriangleSetup* __restrict__ pre_tris, FinePreState pre_st,
                          int pre_tiles_x, int pre_tiles_mod,
                          float4* __restrict__ pre_rec, int* __restrict__ pre_cnt,
                          unsigned short* __restrict__ pre_msk)
{
    extern __shared__ u64 s_lds[];
    const int s_first = STRIDED ? (int)blockIdx.x : (int)blockIdx.x * SPT;
    const int s_step  = STRIDED ? (int)gridDim.x : 1;
    hp_tile_sort_fused_multi<BS, WCAP, PCH, SPT, MG_EPT>(
        keys, tile_base, tile_count, out, s_first, s_step, num_slots, pool_cap, lds_cap,
        s_lds, pre_tris, pre_st, pre_tiles_x, pre_tiles_mod, pre_rec, pre_cnt, pre_msk);
}

template <int BS, int WCAP, int PCH, int SPT, bool STRIDED, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused_ms_batch(CqSortBatch b)
{
    extern __shared__ u64 s_lds[];
    int c = 0;
    for (int i = 1; i < b.n; ++i) c = ((int)blockIdx.x >= b.d[i].bstart) ? i : c;
    const CqSortDesc& d = b.d[c];
    const int lbx = (int)blockIdx.x - d.bstart;
    const int s_first = STRIDED ? lbx : lbx * SPT;
    const int s_step  = STRIDED ? d.grid : 1;
    hp_tile_sort_fused_multi<BS, WCAP, PCH, SPT, MG_EPT>(
        reinterpret_cast<const u64*>(d.keys), d.tile_base, d.tile_count,
        reinterpret_cast<u64*>(d.out), s_first, s_step, d.num_slots, d.pool_cap, b.lds_cap,
        s_lds, d.pre.tris, d.pre.st, d.pre.tiles_x, d.pre.tiles_mod,
        d.pre.rec, d.pre.cnt, d.pre.msk);
}

template <int BS, int WCAP, int PCH, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused_one_batch(CqSortBatch b)
{
    extern __shared__ u64 s_lds[];
    int c = 0;
    for (int i = 1; i < b.n; ++i) c = ((int)blockIdx.x >= b.d[i].bstart) ? i : c;
    const CqSortDesc& d = b.d[c];
    const int lbx = (int)blockIdx.x - d.bstart;
    hp_tile_sort_fused_one<false, BS, WCAP, PCH, MG_EPT>(
        reinterpret_cast<const u64*>(d.keys), d.tile_base, d.tile_count,
        reinterpret_cast<u64*>(d.out), lbx, d.num_slots, d.pool_cap, b.lds_cap, s_lds,
        d.pre.tris, d.pre.st, d.pre.tiles_x, d.pre.tiles_mod, d.pre.rec, d.pre.cnt, d.pre.msk);
}

template <int BS, int WCAP, int PCH, int SPT, bool STRIDED, int MG_EPT = 4>
HIPGL_GLOBAL void __launch_bounds__(BS)
kernel_tile_sort_fused_ms_z2(SortSlot s0, SortSlot s1,
                             int num_slots, int pool_cap, int lds_cap)
{
    extern __shared__ u64 s_lds[];
    const SortSlot& s = (blockIdx.z == 0) ? s0 : s1;
    const int s_first = STRIDED ? (int)blockIdx.x : (int)blockIdx.x * SPT;
    const int s_step  = STRIDED ? (int)gridDim.x : 1;
    hp_tile_sort_fused_multi<BS, WCAP, PCH, SPT, MG_EPT>(
        reinterpret_cast<const u64*>(s.keys), s.tile_base, s.tile_count,
        reinterpret_cast<u64*>(s.out),
        s_first, s_step, num_slots, pool_cap, lds_cap, s_lds,
        s.pre.tris, s.pre.st, s.pre.tiles_x, s.pre.tiles_mod,
        s.pre.rec, s.pre.cnt, s.pre.msk);
}

// Runtime knobs (read once): HP_SORT_SPT = slots per CTA (1 = the original
// one-CTA-per-slot kernel), HP_SORT_SPT_MIN = smallest num_slots that takes
// the multi-slot kernel, HP_SORT_SPT_STRIDED = slot map.
static inline int hp_sort_spt() {
    static const int v = getenv("HP_SORT_SPT") ? atoi(getenv("HP_SORT_SPT")) : 16;
    return v;
}
static inline int hp_sort_spt_min() {
    static const int v = getenv("HP_SORT_SPT_MIN") ? atoi(getenv("HP_SORT_SPT_MIN")) : 5;
    return v;
}
static inline int hp_sort_spt_strided() {
    static const int v = getenv("HP_SORT_SPT_STRIDED") ? atoi(getenv("HP_SORT_SPT_STRIDED")) : 1;
    return v;
}

template <int BS, int SPT, bool STRIDED>
static void hp_launch_ms_go(int lds_cap, hipStream_t stream, const u64* keys,
                            const int* tile_base, const int* tile_count, u64* out,
                            int num_slots, int pool_cap, const FinePreEmit& pre)
{
    const int grid = (num_slots + SPT - 1) / SPT;
    if (CqChainRec* cq = cq_rec_slot()) {
        if ((cq->want & 4) && SPT == 16 && STRIDED) {
            CqSortDesc& d = cq->o.d;
            memset(&d, 0, sizeof(d));
            d.keys = reinterpret_cast<const unsigned long long*>(keys);
            d.tile_base = tile_base; d.tile_count = tile_count;
            d.out = reinterpret_cast<unsigned long long*>(out); d.pre = pre;
            d.num_slots = num_slots; d.pool_cap = pool_cap; d.grid = grid; d.bstart = 0;
            cq->o.bs = BS; cq->o.spt = SPT; cq->o.strided = STRIDED ? 1 : 0;
            cq->o.lds_cap = lds_cap;
            cq->got |= 4;
            return;
        }
    }
    hipLaunchKernelGGL((kernel_tile_sort_fused_ms<BS, 128, 128, SPT, STRIDED>),
                       dim3(grid), dim3(BS), (size_t)lds_cap * sizeof(u64), stream,
                       keys, tile_base, tile_count, out, num_slots, pool_cap, lds_cap,
                       pre.tris, pre.st, pre.tiles_x, pre.tiles_mod,
                       pre.rec, pre.cnt, pre.msk);
}
template <int BS, int SPT, bool STRIDED>
static void hp_launch_ms_z2_go(int lds_cap, hipStream_t stream,
                               const SortSlot& s0, const SortSlot& s1,
                               int num_slots, int pool_cap)
{
    const int grid = (num_slots + SPT - 1) / SPT;
    hipLaunchKernelGGL((kernel_tile_sort_fused_ms_z2<BS, 128, 128, SPT, STRIDED>),
                       dim3(grid, 1, 2), dim3(BS), (size_t)lds_cap * sizeof(u64), stream,
                       s0, s1, num_slots, pool_cap, lds_cap);
}
// Returns false (launches nothing) when the multi-slot form does not apply.
template <int BS>
static bool hp_launch_ms(int lds_cap, hipStream_t stream, const u64* keys,
                         const int* tile_base, const int* tile_count, u64* out,
                         int num_slots, int pool_cap, const FinePreEmit& pre,
                         int tiles_per_span)
{
    const int spt = hp_sort_spt();
    if (spt <= 1 || pre.rec == nullptr || pre.cnt == nullptr || tiles_per_span <= 0 ||
        num_slots < hp_sort_spt_min() * tiles_per_span) return false;
    const bool st = hp_sort_spt_strided() != 0;
#define HP_MS_CASE(S) case S: \
        if (st) hp_launch_ms_go<BS, S, true >(lds_cap, stream, keys, tile_base, tile_count, out, num_slots, pool_cap, pre); \
        else    hp_launch_ms_go<BS, S, false>(lds_cap, stream, keys, tile_base, tile_count, out, num_slots, pool_cap, pre); \
        return true;
    switch (spt) {
    HP_MS_CASE(2) HP_MS_CASE(4) HP_MS_CASE(8) HP_MS_CASE(16) HP_MS_CASE(32)
    default: return false;
    }
#undef HP_MS_CASE
}
template <int BS>
static bool hp_launch_ms_z2(int lds_cap, hipStream_t stream,
                            const SortSlot& s0, const SortSlot& s1,
                            int num_slots, int pool_cap, int tiles_per_span)
{
    const int spt = hp_sort_spt();
    if (spt <= 1 || tiles_per_span <= 0 || num_slots < hp_sort_spt_min() * tiles_per_span ||
        !s0.pre.rec || !s0.pre.cnt || !s1.pre.rec || !s1.pre.cnt) return false;
    const bool st = hp_sort_spt_strided() != 0;
#define HP_MS_CASE(S) case S: \
        if (st) hp_launch_ms_z2_go<BS, S, true >(lds_cap, stream, s0, s1, num_slots, pool_cap); \
        else    hp_launch_ms_z2_go<BS, S, false>(lds_cap, stream, s0, s1, num_slots, pool_cap); \
        return true;
    switch (spt) {
    HP_MS_CASE(2) HP_MS_CASE(4) HP_MS_CASE(8) HP_MS_CASE(16) HP_MS_CASE(32)
    default: return false;
    }
#undef HP_MS_CASE
}
// BS dispatch (hipify-safe: every launch sits in a template function).
static bool hp_launch_ms_bs(int bs, int lds_cap, hipStream_t stream, const u64* keys,
                            const int* tile_base, const int* tile_count, u64* out,
                            int num_slots, int pool_cap, const FinePreEmit& pre,
                            int tiles_per_span)
{
    switch (bs) {
    case 512:  return hp_launch_ms<512>(lds_cap, stream, keys, tile_base, tile_count, out, num_slots, pool_cap, pre, tiles_per_span);
    case 1024: return hp_launch_ms<1024>(lds_cap, stream, keys, tile_base, tile_count, out, num_slots, pool_cap, pre, tiles_per_span);
    default:   return hp_launch_ms<256>(lds_cap, stream, keys, tile_base, tile_count, out, num_slots, pool_cap, pre, tiles_per_span);
    }
}
static bool hp_launch_ms_z2_bs(int bs, int lds_cap, hipStream_t stream,
                               const SortSlot& s0, const SortSlot& s1,
                               int num_slots, int pool_cap, int tiles_per_span)
{
    switch (bs) {
    case 512:  return hp_launch_ms_z2<512>(lds_cap, stream, s0, s1, num_slots, pool_cap, tiles_per_span);
    case 1024: return hp_launch_ms_z2<1024>(lds_cap, stream, s0, s1, num_slots, pool_cap, tiles_per_span);
    default:   return hp_launch_ms_z2<256>(lds_cap, stream, s0, s1, num_slots, pool_cap, tiles_per_span);
    }
}

template <bool LOG_MAX, int BS>
static void hp_launch_fused_z2_bs(int lds_cap, hipStream_t stream,
                                  const SortSlot& s0, const SortSlot& s1,
                                  int num_tiles, int pool_cap)
{
    hipLaunchKernelGGL((kernel_tile_sort_fused_z2<LOG_MAX, BS, 128, 128>),
                       dim3(num_tiles, 1, 2), dim3(BS),
                       (size_t)lds_cap * sizeof(u64), stream,
                       s0, s1, num_tiles, pool_cap, lds_cap);
}

static inline int hp_lds_cap() {
    static const int v = getenv("HP_FUSED_LDS") ? atoi(getenv("HP_FUSED_LDS")) : HP_LDS_CAP;
    return v;
}
static inline int hp_lds_cap_for(int tiles_per_span) {
    static const int env = getenv("HP_FUSED_LDS") ? atoi(getenv("HP_FUSED_LDS")) : 0;
    if (env) return env;
    static const int big_min = getenv("HP_FUSED_LDS_BIGMIN")
                             ? atoi(getenv("HP_FUSED_LDS_BIGMIN")) : 512;
    static const int big_cap = getenv("HP_FUSED_LDS_BIG")
                             ? atoi(getenv("HP_FUSED_LDS_BIG")) : 2048;
    return (tiles_per_span >= big_min) ? big_cap : HP_LDS_CAP;
}
static inline int hp_fused_block(int num_tiles) {
    static const int env = getenv("HP_FUSED_BS") ? atoi(getenv("HP_FUSED_BS")) : 0;
    if (env) return env;
    static const int wide_max = getenv("HP_FUSED_BS_WIDE_MAX")
                              ? atoi(getenv("HP_FUSED_BS_WIDE_MAX")) : 256;
    static const int wide_bs = getenv("HP_FUSED_BS_WIDE")
                             ? atoi(getenv("HP_FUSED_BS_WIDE")) : 512;
    return (num_tiles <= wide_max) ? wide_bs : 256;
}
static inline int hp_fused_tpb() {
    static const int v = getenv("HP_FUSED_TPB") ? atoi(getenv("HP_FUSED_TPB")) : 1;
    return v;
}
static inline int hp_fused_tpb_big() {
    // tiles-per-block used once the grid is large enough that a block-per-tile
    // launch wastes waves (r512 = 1024 tiles). Sweepable.
    static const int v = getenv("HP_FUSED_TPB_BIG") ? atoi(getenv("HP_FUSED_TPB_BIG")) : 1;
    return v;
}
static inline int hp_fused_tpb_min_tiles() {
    static const int v = getenv("HP_FUSED_TPB_MIN") ? atoi(getenv("HP_FUSED_TPB_MIN")) : 512;
    return v;
}
static inline int hp_fused_on() {
    static const int v = getenv("HP_FUSED") ? atoi(getenv("HP_FUSED")) : 1;
    return v;
}

// Hipify-safe template dispatch: no <<<>>> inside a macro if/else.
template <bool LOG_MAX, int BS, int TPB>
static void hp_launch_fused_go(int lds_cap, hipStream_t stream, const u64* keys,
                               const int* tile_base,
                               const int* tile_count, u64* out, int num_tiles, int pool_cap,
                               const FinePreEmit& pre)
{
    const int grid = (num_tiles + TPB - 1) / TPB;
    if (CqChainRec* cq = cq_rec_slot()) {
        if ((cq->want & 4) && TPB == 1 && !LOG_MAX) {
            CqSortDesc& d = cq->o.d;
            memset(&d, 0, sizeof(d));
            d.keys = reinterpret_cast<const unsigned long long*>(keys);
            d.tile_base = tile_base; d.tile_count = tile_count;
            d.out = reinterpret_cast<unsigned long long*>(out); d.pre = pre;
            d.num_slots = num_tiles; d.pool_cap = pool_cap; d.grid = grid; d.bstart = 0;
            cq->o.bs = BS; cq->o.spt = 1; cq->o.strided = 0;
            cq->o.lds_cap = lds_cap;
            cq->got |= 4;
            return;
        }
    }
    hipLaunchKernelGGL((kernel_tile_sort_fused<LOG_MAX, BS, 128, 128, TPB>),
                       dim3(grid), dim3(BS), (size_t)lds_cap * sizeof(u64), stream,
                       keys, tile_base, tile_count, out, num_tiles, pool_cap, lds_cap,
                       pre.tris, pre.st, pre.tiles_x, pre.tiles_mod,
                       (TPB == 1) ? pre.rec : nullptr, pre.cnt,
                       (TPB == 1) ? pre.msk : nullptr);
}

template <bool LOG_MAX, int BS>
static void hp_launch_fused_t(int tpb, int lds_cap, hipStream_t stream, const u64* keys, const int* tile_base,
                              const int* tile_count, u64* out, int num_tiles, int pool_cap,
                              const FinePreEmit& pre)
{
    switch (tpb) {
    case 2:  hp_launch_fused_go<LOG_MAX, BS, 2>(lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    case 4:  hp_launch_fused_go<LOG_MAX, BS, 4>(lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    case 8:  hp_launch_fused_go<LOG_MAX, BS, 8>(lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    default: hp_launch_fused_go<LOG_MAX, BS, 1>(lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    }
}

template <bool LOG_MAX>
static void hp_launch_fused_bs(int bs, int lds_cap, hipStream_t stream, const u64* keys, const int* tile_base,
                               const int* tile_count, u64* out, int num_tiles, int pool_cap,
                               const FinePreEmit& pre)
{
    const int tpb = (num_tiles >= hp_fused_tpb_min_tiles()) ? hp_fused_tpb_big() : hp_fused_tpb();
    switch (bs) {
    case 512:  hp_launch_fused_t<LOG_MAX, 512>(tpb, lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    case 1024: hp_launch_fused_t<LOG_MAX, 1024>(tpb, lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    default:   hp_launch_fused_t<LOG_MAX, 256>(tpb, lds_cap, stream, keys, tile_base, tile_count, out, num_tiles, pool_cap, pre); break;
    }
}

static bool hp_log_maxseg() {
    static const bool v = (getenv("HIPGL_LOG_MAXSEG") != nullptr);
    return v;
}

bool tile_sort_fused_enabled() { return hp_fused_on() != 0 && hp_fused_tpb() == 1
                                     && hp_fused_tpb_big() == 1; }

unsigned int tile_sort_max_seg() {
    unsigned int v = 0u;
    hipMemcpyFromSymbol(&v, HIP_SYMBOL(g_hp_max_seg), sizeof(unsigned int));
    return v;
}

void launch_tile_wave_sort(const int* pool_tri, const float* pool_z,
                           const int* tile_base, const int* tile_count,
                           unsigned long long* out,
                           int* big_list, int* big_count, int big_cap,
                           int num_tiles, int pool_cap, hipStream_t stream)
{
    if (!pool_tri || !pool_z || !out || num_tiles <= 0 || pool_cap <= 0) return;
    const int block = 256;
    const int grid  = (num_tiles + 3) / 4;
    if (hp_log_maxseg()) {
        hipLaunchKernelGGL((kernel_tile_wave_sort<true>), dim3(grid), dim3(block), 0, stream,
                           pool_tri, pool_z, tile_base, tile_count, out,
                           big_list, big_count, big_cap, num_tiles, pool_cap);
    } else {
        hipLaunchKernelGGL((kernel_tile_wave_sort<false>), dim3(grid), dim3(block), 0, stream,
                           pool_tri, pool_z, tile_base, tile_count, out,
                           big_list, big_count, big_cap, num_tiles, pool_cap);
    }
    // Mop-up: a small FIXED grid that walks the compacted list. When the list is
    // empty (the r1024 steady state, mean segment 26) every block retires after
    // one scalar load, which is why the tail costs ~nothing where it is absent.
    int g2 = HP_BIG_GRID;
    hipLaunchKernelGGL(kernel_tile_big_sort, dim3(g2), dim3(HP_BIG_BLOCK),
                       (size_t)HP_LDS_CAP * sizeof(u64), stream,
                       pool_tri, pool_z, tile_base, tile_count, out,
                       big_list, big_count, big_cap, pool_cap);
}

void launch_tile_wave_sort_keys(const unsigned long long* pool_key,
                                const int* tile_base, const int* tile_count,
                                unsigned long long* out,
                                int* big_list, int* big_count, int big_cap,
                                int num_tiles, int pool_cap, hipStream_t stream,
                                int tiles_per_span, const FinePreEmit& pre_in)
{
    FinePreEmit pre = pre_in;
    if (CqChainRec* cq = cq_rec_slot()) {
        const bool ms_applies = hp_sort_spt() > 1 && pre.rec && pre.cnt && tiles_per_span > 0 &&
                                num_tiles >= hp_sort_spt_min() * tiles_per_span;
        if ((cq->want & 4) &&
            !(pool_key && out && num_tiles > 0 && pool_cap > 0 && hp_fused_on() &&
              !hp_log_maxseg() && tile_sort_fused_enabled() &&
              (!ms_applies || (hp_sort_spt() == 16 && hp_sort_spt_strided() != 0)))) {
            cq_rec_miss(cq);
        }
    }
    if (!pool_key || !out || num_tiles <= 0 || pool_cap <= 0) return;
    const int bs_key = (tiles_per_span > 0) ? tiles_per_span : num_tiles;
    if (hp_fused_on()) {
        const int lcap = hp_lds_cap_for(bs_key);
        if (!hp_log_maxseg() && tile_sort_fused_enabled() &&
            hp_launch_ms_bs(hp_fused_block(bs_key), lcap, stream, pool_key, tile_base,
                            tile_count, out, num_tiles, pool_cap, pre, tiles_per_span))
            return;
        if (hp_log_maxseg())
            hp_launch_fused_bs<true>(hp_fused_block(bs_key), lcap, stream,
                                     pool_key, tile_base, tile_count, out, num_tiles, pool_cap, pre);
        else
            hp_launch_fused_bs<false>(hp_fused_block(bs_key), lcap, stream,
                                      pool_key, tile_base, tile_count, out, num_tiles, pool_cap, pre);
        return;
    }
    const int block = 256;
    const int grid  = (num_tiles + 3) / 4;
    if (hp_log_maxseg()) {
        hipLaunchKernelGGL((kernel_tile_wave_sort_k<true>), dim3(grid), dim3(block), 0, stream,
                           pool_key, tile_base, tile_count, out,
                           big_list, big_count, big_cap, num_tiles, pool_cap);
    } else {
        hipLaunchKernelGGL((kernel_tile_wave_sort_k<false>), dim3(grid), dim3(block), 0, stream,
                           pool_key, tile_base, tile_count, out,
                           big_list, big_count, big_cap, num_tiles, pool_cap);
    }
    hp_launch_big_k(hp_big_block(), hp_big_grid(), (size_t)HP_LDS_CAP * sizeof(u64), stream,
                    pool_key, tile_base, tile_count, out, big_list, big_count, big_cap, pool_cap);
}

template <int BS>
static void hp_cq_sort_go(const CqSortRec* const* r, int n, hipStream_t stream)
{
    const int lds_cap = r[0]->lds_cap;
    if (r[0]->spt == 1) {
        if (n == 1) {
            const CqSortDesc& d = r[0]->d;
            hipLaunchKernelGGL((kernel_tile_sort_fused<false, BS, 128, 128, 1>),
                               dim3(d.grid), dim3(BS), (size_t)lds_cap * sizeof(u64), stream,
                               reinterpret_cast<const u64*>(d.keys), d.tile_base, d.tile_count,
                               reinterpret_cast<u64*>(d.out), d.num_slots, d.pool_cap, lds_cap,
                               d.pre.tris, d.pre.st, d.pre.tiles_x, d.pre.tiles_mod,
                               d.pre.rec, d.pre.cnt, d.pre.msk);
            return;
        }
        CqSortBatch b;
        memset(&b, 0, sizeof(b));
        int tot = 0;
        for (int i = 0; i < n; ++i) { b.d[i] = r[i]->d; b.d[i].bstart = tot; tot += b.d[i].grid; }
        b.n = n; b.lds_cap = lds_cap;
        hipLaunchKernelGGL((kernel_tile_sort_fused_one_batch<BS, 128, 128>),
                           dim3(tot), dim3(BS), (size_t)lds_cap * sizeof(u64), stream, b);
        return;
    }
    if (n == 1) {
        const CqSortDesc& d = r[0]->d;
        hipLaunchKernelGGL((kernel_tile_sort_fused_ms<BS, 128, 128, 16, true>),
                           dim3(d.grid), dim3(BS), (size_t)lds_cap * sizeof(u64), stream,
                           reinterpret_cast<const u64*>(d.keys), d.tile_base, d.tile_count,
                           reinterpret_cast<u64*>(d.out), d.num_slots, d.pool_cap, lds_cap,
                           d.pre.tris, d.pre.st, d.pre.tiles_x, d.pre.tiles_mod,
                           d.pre.rec, d.pre.cnt, d.pre.msk);
        return;
    }
    CqSortBatch b;
    memset(&b, 0, sizeof(b));
    int tot = 0;
    for (int i = 0; i < n; ++i) { b.d[i] = r[i]->d; b.d[i].bstart = tot; tot += b.d[i].grid; }
    b.n = n; b.lds_cap = lds_cap;
    hipLaunchKernelGGL((kernel_tile_sort_fused_ms_batch<BS, 128, 128, 16, true>),
                       dim3(tot), dim3(BS), (size_t)lds_cap * sizeof(u64), stream, b);
}

void launch_cq_sort(const CqSortRec* const* r, int n, hipStream_t stream)
{
    bool done[HP_CQ_MAX] = {};
    for (int i = 0; i < n; ++i) {
        if (done[i]) continue;
        const CqSortRec* grp[HP_CQ_MAX];
        int g = 0;
        for (int j = i; j < n; ++j) {
            if (done[j]) continue;
            if (r[j]->bs != r[i]->bs || r[j]->lds_cap != r[i]->lds_cap ||
                r[j]->spt != r[i]->spt) continue;
            done[j] = true;
            grp[g++] = r[j];
        }
        switch (r[i]->bs) {
        case 512:  hp_cq_sort_go<512>(grp, g, stream); break;
        case 1024: hp_cq_sort_go<1024>(grp, g, stream); break;
        default:   hp_cq_sort_go<256>(grp, g, stream); break;
        }
    }
}

bool launch_tile_wave_sort_keys_z2(const SortSlot& s0, const SortSlot& s1,
                                   int num_tiles, int pool_cap, int tiles_per_span,
                                   hipStream_t stream)
{
    static const int on = getenv("HP_SORT_Z2") ? atoi(getenv("HP_SORT_Z2")) : 1;
    if (!on) return false;
    if (!hp_fused_on() || !tile_sort_fused_enabled()) return false;
    if (num_tiles <= 0 || pool_cap <= 0) return false;
    if (!s0.keys || !s0.out || !s1.keys || !s1.out) return false;
    // The epilogue is what makes the two slots' outputs independent of each
    // other; without it the fine rasterizer would re-read `out` and the two
    // passes would need the same array. Require both slots to carry it.
    if (!s0.pre.rec || !s1.pre.rec) return false;
    const int bs_key = (tiles_per_span > 0) ? tiles_per_span : num_tiles;
    const int lcap = hp_lds_cap_for(bs_key);
    const int bs   = hp_fused_block(bs_key);
    if (hp_log_maxseg()) return false;   // the diagnostic path stays scalar
    if (hp_launch_ms_z2_bs(bs, lcap, stream, s0, s1, num_tiles, pool_cap, tiles_per_span))
        return true;
    switch (bs) {
    case 512:  hp_launch_fused_z2_bs<false, 512>(lcap, stream, s0, s1, num_tiles, pool_cap); break;
    case 1024: hp_launch_fused_z2_bs<false, 1024>(lcap, stream, s0, s1, num_tiles, pool_cap); break;
    default:   hp_launch_fused_z2_bs<false, 256>(lcap, stream, s0, s1, num_tiles, pool_cap); break;
    }
    return true;
}

}  // namespace kernels
}  // namespace hipgl
