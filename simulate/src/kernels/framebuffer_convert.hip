// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include "hipgl/kernels.h"

#ifndef HIPGL_RESOLVE_GROUP
#define HIPGL_RESOLVE_GROUP 1
#endif
// Streaming (non-temporal) load of the supersampled colour buffer. The resolve
// reads each sample EXACTLY ONCE and the buffer is 268 MB at r1024/ssaa=4, far
// past the LLC, so keeping it out of the cache hierarchy leaves the LLC for the
// 16 concurrent worker streams' rasterizer working sets.
#ifndef HIPGL_RESOLVE_NT
#define HIPGL_RESOLVE_NT 1
#endif
typedef float hipgl_f32x4 __attribute__((ext_vector_type(4)));
#if HIPGL_RESOLVE_NT
#define HIPGL_NT_LOAD(ptr) __builtin_nontemporal_load(ptr)
#else
#define HIPGL_NT_LOAD(ptr) (*(ptr))
#endif

namespace hipgl {
namespace kernels {

__global__ void kernel_rgba_f32_to_rgb_u8(
    const float* __restrict__ src_rgba,
    uint8_t* __restrict__ dst_rgb,
    int width, int height,
    bool flip_v, bool flip_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int row = idx / width;
    int col = idx % width;

    int src_row = flip_v ? (height - 1 - row) : row;
    int src_col = flip_h ? (width - 1 - col) : col;
    int src_idx = src_row * width + src_col;

    float r = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, src_rgba[src_idx * 4 + 0]));
    float g = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, src_rgba[src_idx * 4 + 1]));
    float b = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, src_rgba[src_idx * 4 + 2]));

    dst_rgb[idx * 3 + 0] = (uint8_t)(r * 255.0f + 0.5f);
    dst_rgb[idx * 3 + 1] = (uint8_t)(g * 255.0f + 0.5f);
    dst_rgb[idx * 3 + 2] = (uint8_t)(b * 255.0f + 0.5f);
}

void launch_rgba_f32_to_rgb_u8(
    const float* src_rgba, uint8_t* dst_rgb,
    int width, int height,
    bool flip_v, bool flip_h,
    hipStream_t stream)
{
    int total = width * height;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_rgba_f32_to_rgb_u8<<<grid, block, 0, stream>>>(
        src_rgba, dst_rgb, width, height, flip_v, flip_h);
}

__global__ void kernel_rgba_f32_to_rgb_u8_ss(
    const float* __restrict__ src_rgba,
    uint8_t* __restrict__ dst_rgb,
    int out_w, int out_h, int ss,
    bool flip_v, bool flip_h)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_w * out_h;
    if (idx >= total) return;

    const int src_w = out_w * ss;
    const int src_h = out_h * ss;

    int row = idx / out_w;
    int col = idx % out_w;
    // Flip in OUTPUT space, then expand: flipping the supersampled grid instead
    // would transpose the sub-pixel order within each block, which is a no-op
    // for a box filter but not for anything weighted later.
    int o_row = flip_v ? (out_h - 1 - row) : row;
    int o_col = flip_h ? (out_w - 1 - col) : col;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int dy = 0; dy < ss; dy++) {
        const int sy = o_row * ss + dy;
        if (sy >= src_h) continue;
        for (int dx = 0; dx < ss; dx++) {
            const int sx = o_col * ss + dx;
            if (sx >= src_w) continue;
            const int s = (sy * src_w + sx) * 4;
            r += src_rgba[s + 0];
            g += src_rgba[s + 1];
            b += src_rgba[s + 2];
        }
    }
    const float inv = 1.0f / (float)(ss * ss);
    r = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, r * inv));
    g = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, g * inv));
    b = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, b * inv));

    dst_rgb[idx * 3 + 0] = (uint8_t)(r * 255.0f + 0.5f);
    dst_rgb[idx * 3 + 1] = (uint8_t)(g * 255.0f + 0.5f);
    dst_rgb[idx * 3 + 2] = (uint8_t)(b * 255.0f + 0.5f);
}


template <int SS, bool FLIP_V, bool FLIP_H>
__global__ void __launch_bounds__(256) kernel_rgba_f32_to_rgb_u8_ss_v4(
    const float* __restrict__ src_rgba,
    uint8_t* __restrict__ dst_rgb,
    int out_w, int out_h, int groups_per_row)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups_per_row) return;
    const int row = blockIdx.y;

    const int col   = g * 4;
    const int src_w = out_w * SS;
    const int o_row = FLIP_V ? (out_h - 1 - row) : row;

    const float4* __restrict__ src4 = reinterpret_cast<const float4*>(src_rgba);
    const size_t row_base = (size_t)o_row * SS * (size_t)src_w;

    float ar[4] = {0.f, 0.f, 0.f, 0.f};
    float ag[4] = {0.f, 0.f, 0.f, 0.f};
    float ab[4] = {0.f, 0.f, 0.f, 0.f};

    // dy OUTERMOST on purpose: for a fixed dy the four output pixels of this
    // thread cover 4*SS CONSECUTIVE source pixels, so the wave sweeps one
    // source row contiguously (64 lanes * 4*SS*16 B) before moving to the next
    // row. Putting j outermost instead makes each lane revisit the same SS rows
    // four times and drops line utilisation to 25%. The per-pixel accumulation
    // order (dy outer, dx inner) is unchanged either way, so this is still
    // bit-identical to the scalar kernel.
    const float4* pbase[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int o_col = FLIP_H ? (out_w - 1 - (col + j)) : (col + j);
        pbase[j] = src4 + row_base + (size_t)o_col * SS;
    }
#pragma unroll
    for (int dy = 0; dy < SS; dy++) {
        const size_t roff = (size_t)dy * src_w;
#pragma unroll
        for (int j = 0; j < 4; j++) {
#pragma unroll
            for (int dx = 0; dx < SS; dx++) {
                const float4 v = pbase[j][roff + dx];
                ar[j] += v.x;
                ag[j] += v.y;
                ab[j] += v.z;
            }
        }
    }

    const float inv = 1.0f / (float)(SS * SS);
    uint32_t b[12];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        float r = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ar[j] * inv));
        float gg = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ag[j] * inv));
        float bb = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ab[j] * inv));
        b[j * 3 + 0] = (uint32_t)(uint8_t)(r  * 255.0f + 0.5f);
        b[j * 3 + 1] = (uint32_t)(uint8_t)(gg * 255.0f + 0.5f);
        b[j * 3 + 2] = (uint32_t)(uint8_t)(bb * 255.0f + 0.5f);
    }

    // 12 bytes -> 3 dwords. (row * out_w + col) is a multiple of 4 (out_w and
    // col both are), so the byte offset 3*(...) is a multiple of 4: aligned.
    uint32_t* __restrict__ d32 =
        reinterpret_cast<uint32_t*>(dst_rgb) + (((size_t)row * out_w + col) * 3u) / 4u;
    d32[0] = b[0] | (b[1] << 8) | (b[2]  << 16) | (b[3]  << 24);
    d32[1] = b[4] | (b[5] << 8) | (b[6]  << 16) | (b[7]  << 24);
    d32[2] = b[8] | (b[9] << 8) | (b[10] << 16) | (b[11] << 24);
}

template <int SS, bool FLIP_V, bool FLIP_H>
__global__ void __launch_bounds__(256) kernel_rgba_f32_to_rgb_u8_ss_v1(
    const float* __restrict__ src_rgba,
    uint8_t* __restrict__ dst_rgb,
    int out_w, int out_h)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_w) return;
    const int row = blockIdx.y;

    const int src_w = out_w * SS;
    const int o_row = FLIP_V ? (out_h - 1 - row) : row;
    const int o_col = FLIP_H ? (out_w - 1 - col) : col;

    const hipgl_f32x4* __restrict__ p =
        reinterpret_cast<const hipgl_f32x4*>(src_rgba)
        + (size_t)o_row * SS * (size_t)src_w + (size_t)o_col * SS;

    float ar = 0.f, ag = 0.f, ab = 0.f;
#pragma unroll
    for (int dy = 0; dy < SS; dy++) {
        const size_t roff = (size_t)dy * src_w;
#pragma unroll
        for (int dx = 0; dx < SS; dx++) {
            const hipgl_f32x4 v = HIPGL_NT_LOAD(p + roff + dx);
            ar += v.x; ag += v.y; ab += v.z;
        }
    }
    const float inv = 1.0f / (float)(SS * SS);
    ar = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ar * inv));
    ag = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ag * inv));
    ab = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ab * inv));

    const size_t o = ((size_t)row * out_w + col) * 3;
    dst_rgb[o + 0] = (uint8_t)(ar * 255.0f + 0.5f);
    dst_rgb[o + 1] = (uint8_t)(ag * 255.0f + 0.5f);
    dst_rgb[o + 2] = (uint8_t)(ab * 255.0f + 0.5f);
}

template <int SS, bool FLIP_V, bool FLIP_H>
__global__ void __launch_bounds__(256) kernel_rgba_f32_to_rgb_u8_ss_bg(
    const float* __restrict__ src_rgba,
    const float* __restrict__ src_depth,
    uint8_t* __restrict__ dst_rgb,
    int out_w, int out_h,
    float bg_r, float bg_g, float bg_b, float clear_depth)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= out_w) return;
    const int row = blockIdx.y;

    const int src_w = out_w * SS;
    const int o_row = FLIP_V ? (out_h - 1 - row) : row;
    const int o_col = FLIP_H ? (out_w - 1 - col) : col;

    const size_t s0 = (size_t)o_row * SS * (size_t)src_w + (size_t)o_col * SS;
    const hipgl_f32x4* __restrict__ p =
        reinterpret_cast<const hipgl_f32x4*>(src_rgba) + s0;
    const float* __restrict__ dp = src_depth + s0;

    float dz[SS * SS];
#pragma unroll
    for (int dy = 0; dy < SS; dy++) {
        const size_t roff = (size_t)dy * src_w;
        if ((SS & 3) == 0) {
#pragma unroll
            for (int k = 0; k < SS / 4; k++) {
                const hipgl_f32x4 z4 = HIPGL_NT_LOAD(
                    reinterpret_cast<const hipgl_f32x4*>(dp + roff) + k);
                dz[dy * SS + k * 4 + 0] = z4.x; dz[dy * SS + k * 4 + 1] = z4.y;
                dz[dy * SS + k * 4 + 2] = z4.z; dz[dy * SS + k * 4 + 3] = z4.w;
            }
        } else {
#pragma unroll
            for (int k = 0; k < SS; k++) dz[dy * SS + k] = HIPGL_NT_LOAD(dp + roff + k);
        }
    }
    bool all_bg = true;
#pragma unroll
    for (int i = 0; i < SS * SS; i++) all_bg = all_bg && (dz[i] == clear_depth);

    float ar, ag, ab;
    if (__all(all_bg)) {
        const float n = (float)(SS * SS);
        ar = bg_r * n; ag = bg_g * n; ab = bg_b * n;
    } else {
        ar = 0.f; ag = 0.f; ab = 0.f;
#pragma unroll
        for (int dy = 0; dy < SS; dy++) {
            const size_t roff = (size_t)dy * src_w;
#pragma unroll
            for (int dx = 0; dx < SS; dx++) {
                const hipgl_f32x4 v = HIPGL_NT_LOAD(p + roff + dx);
                const bool bgpx = (dz[dy * SS + dx] == clear_depth);
                ar += bgpx ? bg_r : v.x;
                ag += bgpx ? bg_g : v.y;
                ab += bgpx ? bg_b : v.z;
            }
        }
    }
    const float inv = 1.0f / (float)(SS * SS);
    ar = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ar * inv));
    ag = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ag * inv));
    ab = __builtin_fminf(1.0f, __builtin_fmaxf(0.0f, ab * inv));

    const size_t o = ((size_t)row * out_w + col) * 3;
    dst_rgb[o + 0] = (uint8_t)(ar * 255.0f + 0.5f);
    dst_rgb[o + 1] = (uint8_t)(ag * 255.0f + 0.5f);
    dst_rgb[o + 2] = (uint8_t)(ab * 255.0f + 0.5f);
}

// Four adjacent lanes load the four adjacent supersamples in each row.
// Each lane retains one RGB channel's ordered accumulator; lane 3 only loads.
// A 256-thread block covers the same 64 output pixels as one original wave,
// preserving the exact scope (and arithmetic) of the all-background shortcut.
template <bool FLIP_V, bool FLIP_H>
__global__ void __launch_bounds__(256) kernel_rgba_f32_to_rgb_u8_ss4_coop_bg(
    const float* __restrict__ src_rgba,
    const float* __restrict__ src_depth,
    uint8_t* __restrict__ dst_rgb,
    int out_w, int out_h,
    float bg_r, float bg_g, float bg_b, float clear_depth)
{
    const int lane4 = threadIdx.x & 3;
    const int col = blockIdx.x * 64 + threadIdx.x / 4;
    const bool valid = col < out_w;
    const int row = blockIdx.y;
    const int o_row = FLIP_V ? out_h - 1 - row : row;
    const int o_col = FLIP_H ? out_w - 1 - col : col;
    const int src_w = out_w * 4;
    const size_t s0 = (size_t)o_row * 4 * src_w + (size_t)o_col * 4 + lane4;
    float z[4];
    bool covered = false;
#pragma unroll
    for (int dy = 0; dy < 4; ++dy) {
        z[dy] = valid ? HIPGL_NT_LOAD(src_depth + s0 + (size_t)dy * src_w) : clear_depth;
        covered |= z[dy] != clear_depth;
    }
    // No early return: all lanes participate, including a partial final block.
    const bool any_covered = __syncthreads_or(covered);
    const float bg = lane4 == 0 ? bg_r : (lane4 == 1 ? bg_g : bg_b);
    float acc = bg * 16.f;
    if (any_covered) {
        acc = 0.f;
#pragma unroll
        for (int dy = 0; dy < 4; ++dy) {
            hipgl_f32x4 v = {bg_r, bg_g, bg_b, 0.f};
            if (valid) v = HIPGL_NT_LOAD(reinterpret_cast<const hipgl_f32x4*>(src_rgba)
                                        + s0 + (size_t)dy * src_w);
            const bool background = z[dy] == clear_depth;
            const float r = background ? bg_r : v.x;
            const float g = background ? bg_g : v.y;
            const float b = background ? bg_b : v.z;
#pragma unroll
            for (int dx = 0; dx < 4; ++dx) {
                // Exchange samples, NOT partial sums: dy-then-dx addition and
                // the initial +0 are identical to the original per-pixel path.
                const float rr = __shfl(r, dx, 4);
                const float gg = __shfl(g, dx, 4);
                const float bb = __shfl(b, dx, 4);
                acc += lane4 == 0 ? rr : (lane4 == 1 ? gg : bb);
            }
        }
    }
    acc = __builtin_fminf(1.f, __builtin_fmaxf(0.f, acc * (1.f / 16.f)));
    if (valid && lane4 < 3)
        dst_rgb[((size_t)row * out_w + col) * 3 + lane4] = (uint8_t)(acc * 255.f + 0.5f);
}

static inline void dispatch_ss4_coop_bg(
    const float* src_rgba, const float* src_depth, uint8_t* dst_rgb,
    int out_w, int out_h, bool flip_v, bool flip_h,
    float br, float bg, float bb, float cd, hipStream_t stream)
{
    dim3 grid((out_w + 63) / 64, out_h, 1);
    if (!flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss4_coop_bg<false, false><<<grid, 256, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else if (flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss4_coop_bg<true, false><<<grid, 256, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else if (!flip_v && flip_h)
        kernel_rgba_f32_to_rgb_u8_ss4_coop_bg<false, true><<<grid, 256, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else
        kernel_rgba_f32_to_rgb_u8_ss4_coop_bg<true, true><<<grid, 256, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
}

template <int SS>
static inline void dispatch_ss_bg(const float* src_rgba, const float* src_depth,
                                  uint8_t* dst_rgb, int out_w, int out_h,
                                  bool flip_v, bool flip_h,
                                  float br, float bg, float bb, float cd,
                                  hipStream_t stream)
{
    int block = 256;
    if (out_w <= 64) block = 64;
    else if (out_w <= 128) block = 128;
    dim3 grid((out_w + block - 1) / block, out_h, 1);
    if (!flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_bg<SS, false, false><<<grid, block, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else if (flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_bg<SS, true, false><<<grid, block, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else if (!flip_v && flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_bg<SS, false, true><<<grid, block, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
    else
        kernel_rgba_f32_to_rgb_u8_ss_bg<SS, true, true><<<grid, block, 0, stream>>>(src_rgba, src_depth, dst_rgb, out_w, out_h, br, bg, bb, cd);
}

// Returns true if it handled the resolve; false means the caller must fall back
// to the ordinary (cleared-colour) path.
bool launch_rgba_f32_to_rgb_u8_ss_bg(
    const float* src_rgba, const float* src_depth, uint8_t* dst_rgb,
    int out_width, int out_height, int ss, bool flip_v, bool flip_h,
    float bg_r, float bg_g, float bg_b, float clear_depth, hipStream_t stream)
{
    if (!src_rgba || !src_depth || !dst_rgb) return false;
    if (out_width <= 0 || out_height <= 0 || (out_width & 3) != 0) return false;
    if (ss == 4) { dispatch_ss4_coop_bg(src_rgba, src_depth, dst_rgb, out_width, out_height, flip_v, flip_h, bg_r, bg_g, bg_b, clear_depth, stream); return true; }
    if (ss == 2) { dispatch_ss_bg<2>(src_rgba, src_depth, dst_rgb, out_width, out_height, flip_v, flip_h, bg_r, bg_g, bg_b, clear_depth, stream); return true; }
    if (ss == 8) { dispatch_ss_bg<8>(src_rgba, src_depth, dst_rgb, out_width, out_height, flip_v, flip_h, bg_r, bg_g, bg_b, clear_depth, stream); return true; }
    return false;
}

template <int SS>
static inline void dispatch_ss_v1(const float* src_rgba, uint8_t* dst_rgb,
                                  int out_w, int out_h, bool flip_v, bool flip_h,
                                  hipStream_t stream)
{
    int block = 256;
    if (out_w <= 64) block = 64;
    else if (out_w <= 128) block = 128;
    dim3 grid((out_w + block - 1) / block, out_h, 1);
    if (!flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v1<SS, false, false><<<grid, block, 0, stream>>>(src_rgba, dst_rgb, out_w, out_h);
    else if (flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v1<SS, true, false><<<grid, block, 0, stream>>>(src_rgba, dst_rgb, out_w, out_h);
    else if (!flip_v && flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v1<SS, false, true><<<grid, block, 0, stream>>>(src_rgba, dst_rgb, out_w, out_h);
    else
        kernel_rgba_f32_to_rgb_u8_ss_v1<SS, true, true><<<grid, block, 0, stream>>>(src_rgba, dst_rgb, out_w, out_h);
}

template <int SS>
static inline void dispatch_ss_v4(const float* src_rgba, uint8_t* dst_rgb,
                                  int out_w, int out_h, bool flip_v, bool flip_h,
                                  hipStream_t stream)
{
    const int gpr = out_w / 4;
    int block = 256;
    if (gpr <= 64)       block = 64;
    else if (gpr <= 128) block = 128;
    dim3 grid((gpr + block - 1) / block, out_h, 1);
    if (!flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v4<SS, false, false><<<grid, block, 0, stream>>>(
            src_rgba, dst_rgb, out_w, out_h, gpr);
    else if (flip_v && !flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v4<SS, true, false><<<grid, block, 0, stream>>>(
            src_rgba, dst_rgb, out_w, out_h, gpr);
    else if (!flip_v && flip_h)
        kernel_rgba_f32_to_rgb_u8_ss_v4<SS, false, true><<<grid, block, 0, stream>>>(
            src_rgba, dst_rgb, out_w, out_h, gpr);
    else
        kernel_rgba_f32_to_rgb_u8_ss_v4<SS, true, true><<<grid, block, 0, stream>>>(
            src_rgba, dst_rgb, out_w, out_h, gpr);
}

void launch_rgba_f32_to_rgb_u8_ss(
    const float* src_rgba, uint8_t* dst_rgb,
    int out_width, int out_height, int ss,
    bool flip_v, bool flip_h,
    hipStream_t stream)
{
    if (ss <= 1) {
        launch_rgba_f32_to_rgb_u8(src_rgba, dst_rgb, out_width, out_height,
                                  flip_v, flip_h, stream);
        return;
    }
    if ((out_width & 3) == 0 && out_width > 0 && out_height > 0) {
#if HIPGL_RESOLVE_GROUP == 4
        if (ss == 4) { dispatch_ss_v4<4>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
        if (ss == 2) { dispatch_ss_v4<2>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
        if (ss == 8) { dispatch_ss_v4<8>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
#elif HIPGL_RESOLVE_GROUP == 1
        if (ss == 4) { dispatch_ss_v1<4>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
        if (ss == 2) { dispatch_ss_v1<2>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
        if (ss == 8) { dispatch_ss_v1<8>(src_rgba, dst_rgb, out_width, out_height, flip_v, flip_h, stream); return; }
#endif
    }
    int total = out_width * out_height;
    int block = 256;
    int grid = (total + block - 1) / block;
    kernel_rgba_f32_to_rgb_u8_ss<<<grid, block, 0, stream>>>(
        src_rgba, dst_rgb, out_width, out_height, ss, flip_v, flip_h);
}

} // namespace kernels
} // namespace hipgl
