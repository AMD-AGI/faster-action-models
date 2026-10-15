// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Framebuffer clear kernels
// One thread per pixel; coalesced writes for efficient memory access.

#include <hip/hip_runtime.h>
#include "hipgl/types.h"
#include "hipgl/math_types.h"

#ifndef HIPGL_CLEAR_QUAD
#define HIPGL_CLEAR_QUAD 1
#endif
// The clear is a pure write-once producer of a buffer far larger than the LLC.
#ifndef HIPGL_CLEAR_NT
#define HIPGL_CLEAR_NT 1
#endif
typedef float hipgl_f32x4 __attribute__((ext_vector_type(4)));
#if HIPGL_CLEAR_NT
#define HIPGL_NT_STORE(v, ptr) __builtin_nontemporal_store(v, ptr)
#else
#define HIPGL_NT_STORE(v, ptr) (*(ptr) = (v))
#endif

namespace hipgl {
namespace kernels {

// ── Clear color only ────────────────────────────────────────────────────────
HIPGL_GLOBAL void kernel_clear_color(
    DeviceFramebuffer fb,
    float r, float g, float b, float a)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fb.width * fb.height;
    if (idx >= total) return;

    const int base = idx * 4;
    fb.color_buffer[base + 0] = r;
    fb.color_buffer[base + 1] = g;
    fb.color_buffer[base + 2] = b;
    fb.color_buffer[base + 3] = a;
}

// ── Clear depth only ────────────────────────────────────────────────────────
HIPGL_GLOBAL void kernel_clear_depth(
    DeviceFramebuffer fb,
    float depth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fb.width * fb.height;
    if (idx >= total) return;

    fb.depth_buffer[idx] = depth;
}

// ── Combined clear (color + depth) for single pass efficiency ────────────────
HIPGL_GLOBAL void kernel_clear_color_depth(
    DeviceFramebuffer fb,
    float r, float g, float b, float a,
    float depth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fb.width * fb.height;
    if (idx >= total) return;

    const int base = idx * 4;
    fb.color_buffer[base + 0] = r;
    fb.color_buffer[base + 1] = g;
    fb.color_buffer[base + 2] = b;
    fb.color_buffer[base + 3] = a;
    fb.depth_buffer[idx] = depth;
}

template <int QUAD>
__global__ void __launch_bounds__(256) kernel_clear_color_depth_vec(
    float* __restrict__ color, float* __restrict__ depth,
    int total, float r, float g, float b, float a, float d)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int p0 = q * QUAD;
    if (p0 >= total) return;

    const hipgl_f32x4 c = {r, g, b, a};
    const hipgl_f32x4 dd = {d, d, d, d};

    if (p0 + QUAD <= total) {
        hipgl_f32x4* __restrict__ c4 = reinterpret_cast<hipgl_f32x4*>(color) + p0;
#pragma unroll
        for (int i = 0; i < QUAD; i++) HIPGL_NT_STORE(c, c4 + i);
        if (QUAD >= 4) {
            hipgl_f32x4* __restrict__ d4 = reinterpret_cast<hipgl_f32x4*>(depth) + (p0 >> 2);
#pragma unroll
            for (int i = 0; i < QUAD / 4; i++) HIPGL_NT_STORE(dd, d4 + i);
        } else {
#pragma unroll
            for (int i = 0; i < QUAD; i++) HIPGL_NT_STORE(d, depth + p0 + i);
        }
    } else {
        for (int i = 0; i < QUAD && p0 + i < total; i++) {
            const int base = (p0 + i) * 4;
            color[base + 0] = r;
            color[base + 1] = g;
            color[base + 2] = b;
            color[base + 3] = a;
            depth[p0 + i] = d;
        }
    }
}

template <int QUAD>
__global__ void __launch_bounds__(256) kernel_clear_depth_vec(
    float* __restrict__ depth, int total, float d)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    const int p0 = q * (QUAD * 4);
    if (p0 >= total) return;
    const hipgl_f32x4 dd = {d, d, d, d};
    if (p0 + QUAD * 4 <= total) {
        hipgl_f32x4* __restrict__ d4 = reinterpret_cast<hipgl_f32x4*>(depth) + q * QUAD;
#pragma unroll
        for (int i = 0; i < QUAD; i++) HIPGL_NT_STORE(dd, d4 + i);
    } else {
        for (int i = 0; i < QUAD * 4 && p0 + i < total; i++) depth[p0 + i] = d;
    }
}

void launch_clear_depth_vec(float* depth, int total, float d, hipStream_t stream)
{
    if (!depth || total <= 0) return;
    const int block = 256;
    if ((total & 3) == 0) {
        constexpr int QUAD = 2;
        const int nq = (total + QUAD * 4 - 1) / (QUAD * 4);
        const int grid = (nq + block - 1) / block;
        kernel_clear_depth_vec<QUAD><<<grid, block, 0, stream>>>(depth, total, d);
        return;
    }
    const int grid = (total + block - 1) / block;
    kernel_clear_depth_vec<1><<<grid, block, 0, stream>>>(depth, total, d);
}

// MuJoCo-style gradient background: bottom_color at y=0 to top_color at y=height-1
HIPGL_GLOBAL void kernel_clear_gradient_depth(
    DeviceFramebuffer fb,
    float bot_r, float bot_g, float bot_b,
    float top_r, float top_g, float top_b,
    float a, float depth)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = fb.width * fb.height;
    if (idx >= total) return;

    const int y = idx / fb.width;
    const float t = (fb.height > 1) ? (float)y / (float)(fb.height - 1) : 0.0f;

    const int base = idx * 4;
    fb.color_buffer[base + 0] = bot_r + t * (top_r - bot_r);
    fb.color_buffer[base + 1] = bot_g + t * (top_g - bot_g);
    fb.color_buffer[base + 2] = bot_b + t * (top_b - bot_b);
    fb.color_buffer[base + 3] = a;
    fb.depth_buffer[idx] = depth;
}

// ── Fill float4 buffer (for vertex color default white) ──────────────────────
HIPGL_GLOBAL void kernel_fill_float4(
    float* __restrict__ buffer,
    int count,
    float v0, float v1, float v2, float v3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const int base = idx * 4;
    buffer[base + 0] = v0;
    buffer[base + 1] = v1;
    buffer[base + 2] = v2;
    buffer[base + 3] = v3;
}

// ── Host launch wrappers ────────────────────────────────────────────────────

void launch_clear_color(DeviceFramebuffer fb, float r, float g, float b, float a, hipStream_t stream)
{
    if (!fb.color_buffer || fb.width <= 0 || fb.height <= 0) return;
    const int total = fb.width * fb.height;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(kernel_clear_color, grid, block, 0, stream, fb, r, g, b, a);
}

void launch_clear_depth(DeviceFramebuffer fb, float depth, hipStream_t stream)
{
    if (!fb.depth_buffer || fb.width <= 0 || fb.height <= 0) return;
    const int total = fb.width * fb.height;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(kernel_clear_depth, grid, block, 0, stream, fb, depth);
}

void launch_clear_color_depth(DeviceFramebuffer fb, float r, float g, float b, float a, float depth, hipStream_t stream)
{
    if (!fb.color_buffer || !fb.depth_buffer || fb.width <= 0 || fb.height <= 0) return;
    const int total = fb.width * fb.height;
    const int block = 256;
    if ((total & 3) == 0) {
        constexpr int QUAD = HIPGL_CLEAR_QUAD;
        const int nq = (total + QUAD - 1) / QUAD;
        const int grid = (nq + block - 1) / block;
        kernel_clear_color_depth_vec<QUAD><<<grid, block, 0, stream>>>(
            fb.color_buffer, fb.depth_buffer, total, r, g, b, a, depth);
        return;
    }
    const int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(kernel_clear_color_depth, grid, block, 0, stream, fb, r, g, b, a, depth);
}

void launch_clear_gradient_depth(DeviceFramebuffer fb,
    float bot_r, float bot_g, float bot_b,
    float top_r, float top_g, float top_b,
    float a, float depth, hipStream_t stream)
{
    if (!fb.color_buffer || !fb.depth_buffer || fb.width <= 0 || fb.height <= 0) return;
    const int total = fb.width * fb.height;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(kernel_clear_gradient_depth, grid, block, 0, stream,
                       fb, bot_r, bot_g, bot_b, top_r, top_g, top_b, a, depth);
}

void launch_fill_float4(float* buffer, int count, float v0, float v1, float v2, float v3, hipStream_t stream)
{
    if (!buffer || count <= 0) return;
    const int block = 256;
    const int grid = (count + block - 1) / block;
    hipLaunchKernelGGL(kernel_fill_float4, grid, block, 0, stream, buffer, count, v0, v1, v2, v3);
}

} // namespace kernels
} // namespace hipgl
