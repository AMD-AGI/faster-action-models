// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <hip/hip_runtime.h>
#include "hipgl/planar_reflection.h"

namespace hipgl { namespace kernels {

__global__ void kernel_reflection_sample_mask(DeviceFramebuffer fb,
                                              uint8_t* mask, float clear_depth) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fb.width * fb.height) return;
    if (fb.sample_depth) {
        const float4 depths = reinterpret_cast<const float4*>(fb.sample_depth)[p];
        mask[p] = (depths.x < clear_depth ? 1u : 0u)
                | (depths.y < clear_depth ? 2u : 0u)
                | (depths.z < clear_depth ? 4u : 0u)
                | (depths.w < clear_depth ? 8u : 0u);
    } else {
        mask[p] = fb.depth_buffer[p] < clear_depth ? 1u : 0u;
    }
}

__global__ void __launch_bounds__(256) kernel_reflection_mask_and_clear(
    const float* __restrict__ depth_in, float* __restrict__ color,
    float* __restrict__ depth, uint8_t* __restrict__ mask,
    int total, float mask_ref, float r, float g, float b, float a, float d)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= total) return;
    mask[p] = depth_in[p] < mask_ref ? 1u : 0u;
    const int base = p * 4;
    color[base + 0] = r;
    color[base + 1] = g;
    color[base + 2] = b;
    color[base + 3] = a;
    depth[p] = d;
}

bool launch_reflection_mask_and_clear(DeviceFramebuffer fb, uint8_t* mask,
                                      float mask_ref, float r, float g, float b,
                                      float a, float clear_depth,
                                      hipStream_t stream) {
    const int total = fb.width * fb.height;
    if (total <= 0 || !mask || !fb.color_buffer || !fb.depth_buffer) return false;
    if (fb.sample_depth || fb.attr_buffer) return false;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(kernel_reflection_mask_and_clear, grid, block, 0, stream,
                       fb.depth_buffer, fb.color_buffer, fb.depth_buffer, mask,
                       total, mask_ref, r, g, b, a, clear_depth);
    return true;
}

void launch_reflection_sample_mask(DeviceFramebuffer framebuffer,
                                   uint8_t* mask, float clear_depth,
                                   hipStream_t stream) {
    const int pixels = framebuffer.width * framebuffer.height;
    if (pixels <= 0) return;
    hipLaunchKernelGGL(kernel_reflection_sample_mask, (pixels + 255) / 256, 256, 0, stream,
                       framebuffer, mask, clear_depth);
}

}}  // namespace hipgl::kernels
