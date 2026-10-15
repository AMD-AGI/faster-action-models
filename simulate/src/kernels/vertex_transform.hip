// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Vertex transformation kernel
// One thread per vertex: MVP → perspective divide → viewport transform.

#include <hip/hip_runtime.h>
#include "hipgl/types.h"
#include "hipgl/math_types.h"

namespace hipgl {
namespace kernels {

// Transform Vec3 by 3x3 matrix (column-major, 9 elements)
HIPGL_DEVICE inline void transform_normal(const float* nm, const float* n, float* out)
{
    out[0] = nm[0]*n[0] + nm[3]*n[1] + nm[6]*n[2];
    out[1] = nm[1]*n[0] + nm[4]*n[1] + nm[7]*n[2];
    out[2] = nm[2]*n[0] + nm[5]*n[1] + nm[8]*n[2];
}

// Transform Vec3 point by 4x4 matrix (column-major), return Vec4
HIPGL_DEVICE inline void transform_point_mvp(const float* m, const float* p, float* out)
{
    float x = p[0], y = p[1], z = p[2];
    out[0] = m[0]*x + m[4]*y + m[8]*z + m[12];
    out[1] = m[1]*x + m[5]*y + m[9]*z + m[13];
    out[2] = m[2]*x + m[6]*y + m[10]*z + m[14];
    out[3] = m[3]*x + m[7]*y + m[11]*z + m[15];
}

// Transform Vec3 point by model_view for world-space position
HIPGL_DEVICE inline void transform_point_mv(const float* m, const float* p, float* out)
{
    float x = p[0], y = p[1], z = p[2];
    out[0] = m[0]*x + m[4]*y + m[8]*z + m[12];
    out[1] = m[1]*x + m[5]*y + m[9]*z + m[13];
    out[2] = m[2]*x + m[6]*y + m[10]*z + m[14];
}

HIPGL_GLOBAL void kernel_vertex_transform(
    const float* __restrict__ positions,
    const float* __restrict__ normals,
    const float* __restrict__ texcoords,
    const float* __restrict__ colors,
    int vertex_count,
    const float mvp[16],
    const float model_view[16],
    const float normal_mat[9],
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;

    const float* pos = positions + idx * 3;
    float clip[4];
    transform_point_mvp(mvp, pos, clip);

    // Perspective divide
    float inv_w = 1.0f;
    float ndc[3];
    if (fabsf(clip[3]) > 1e-8f) {
        inv_w = 1.0f / clip[3];
        ndc[0] = clip[0] * inv_w;
        ndc[1] = clip[1] * inv_w;
        ndc[2] = clip[2] * inv_w;
    } else {
        ndc[0] = clip[0];
        ndc[1] = clip[1];
        ndc[2] = clip[2];
    }

    // Viewport transform: NDC [-1,1] → screen [vp_x, vp_x+vp_w] x [vp_y, vp_y+vp_h]
    float screen_x = vp_x + (ndc[0] * 0.5f + 0.5f) * static_cast<float>(vp_w);
    float screen_y = vp_y + (1.0f - (ndc[1] * 0.5f + 0.5f)) * static_cast<float>(vp_h);  // flip Y
    float screen_z = ndc[2] * 0.5f + 0.5f;  // NDC [-1,1] → depth [0,1]

    TransformedVertex& v = out[idx];
    v.clip_zw[0] = clip[2];
    v.clip_zw[1] = clip[3];
    v.screen_x = screen_x;
    v.screen_y = screen_y;
    v.screen_z = screen_z;
    v.inv_w = inv_w;

    // World-space position (for lighting)
    transform_point_mv(model_view, pos, v.world_pos);

    // World-space normal (for lighting)
    if (normals) {
        transform_normal(normal_mat, normals + idx * 3, v.world_normal);
    } else {
        v.world_normal[0] = 0.0f;
        v.world_normal[1] = 1.0f;
        v.world_normal[2] = 0.0f;
    }

    // Texcoords
    if (texcoords) {
        v.texcoord[0] = texcoords[idx * TEXCOORD_COMPONENTS + 0];
        v.texcoord[1] = texcoords[idx * TEXCOORD_COMPONENTS + 1];
    } else {
        v.texcoord[0] = 0.0f;
        v.texcoord[1] = 0.0f;
    }

    // Vertex color
    if (colors) {
        v.color[0] = colors[idx * 4 + 0];
        v.color[1] = colors[idx * 4 + 1];
        v.color[2] = colors[idx * 4 + 2];
        v.color[3] = colors[idx * 4 + 3];
    } else {
        v.color[0] = 1.0f;
        v.color[1] = 1.0f;
        v.color[2] = 1.0f;
        v.color[3] = 1.0f;
    }
}

void launch_vertex_transform(
    const float* positions, const float* normals, const float* texcoords, const float* colors,
    int vertex_count, const float mvp[16], const float model_view[16], const float normal_mat[9],
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex* out, hipStream_t stream)
{
    if (!positions || !out || vertex_count <= 0) return;
    const int block = 256;
    const int grid = (vertex_count + block - 1) / block;
    hipLaunchKernelGGL(kernel_vertex_transform, grid, block, 0, stream,
        positions, normals, texcoords, colors, vertex_count,
        mvp, model_view, normal_mat, vp_x, vp_y, vp_w, vp_h, out);
}

} // namespace kernels
} // namespace hipgl
