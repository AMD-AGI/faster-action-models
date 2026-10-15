// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Triangle setup kernel
// Converts TransformedVertex triples to TriangleSetup with edge equations and bbox.

#include <hip/hip_runtime.h>
#include "hipgl/types.h"
#include "hipgl/math_types.h"

namespace hipgl {
namespace kernels {

// Setup one triangle from 3 transformed vertices
HIPGL_GLOBAL void kernel_triangle_setup(
    const TransformedVertex* __restrict__ verts,
    const uint32_t* __restrict__ indices,
    int tri_count,
    bool indexed,
    TriangleSetup* __restrict__ out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tri_count) return;

    const TransformedVertex* v0;
    const TransformedVertex* v1;
    const TransformedVertex* v2;

    if (indexed) {
        int i0 = indices[idx * 3 + 0];
        int i1 = indices[idx * 3 + 1];
        int i2 = indices[idx * 3 + 2];
        v0 = &verts[i0];
        v1 = &verts[i1];
        v2 = &verts[i2];
    } else {
        v0 = &verts[idx * 3 + 0];
        v1 = &verts[idx * 3 + 1];
        v2 = &verts[idx * 3 + 2];
    }

    TriangleSetup& tri = out[idx];
    tri.v[0] = *v0;
    tri.v[1] = *v1;
    tri.v[2] = *v2;

    // Near-plane clipping: reject triangles with any vertex behind the near plane.
    bool behind0 = (v0->clip_zw[0] + v0->clip_zw[1]) < 0.0f;
    bool behind1 = (v1->clip_zw[0] + v1->clip_zw[1]) < 0.0f;
    bool behind2 = (v2->clip_zw[0] + v2->clip_zw[1]) < 0.0f;
    if (behind0 || behind1 || behind2) {
        hipgl_mark_degenerate(tri);
        return;
    }

    // Snap to 1/256 sub-pixel grid so that triangles sharing an edge
    // produce bit-identical edge equations for the shared edge.
    constexpr float SUBPX = 256.0f;
    constexpr float INV_SUBPX = 1.0f / 256.0f;
    float x0 = roundf(v0->screen_x * SUBPX) * INV_SUBPX;
    float y0 = roundf(v0->screen_y * SUBPX) * INV_SUBPX;
    float x1 = roundf(v1->screen_x * SUBPX) * INV_SUBPX;
    float y1 = roundf(v1->screen_y * SUBPX) * INV_SUBPX;
    float x2 = roundf(v2->screen_x * SUBPX) * INV_SUBPX;
    float y2 = roundf(v2->screen_y * SUBPX) * INV_SUBPX;

    float e0x = x1 - x0, e0y = y1 - y0;
    float e1x = x2 - x1, e1y = y2 - y1;
    float e2x = x0 - x2, e2y = y0 - y2;

    float area = e0x * e2y - e0y * e2x;
    hipgl_set_flags(tri, fabsf(area) < 1e-8f, area < 0.0f);

    float sign = (area > 0.0f) ? -1.0f : 1.0f;

    float A0 = sign * (-e0y), B0 = sign * e0x;
    float A1 = sign * (-e1y), B1 = sign * e1x;
    float A2 = sign * (-e2y), B2 = sign * e2x;

    tri.edge_eq[0][0] = A0;
    tri.edge_eq[0][1] = B0;
    tri.edge_eq[0][2] = (float)(-((double)A0 * (double)x0 + (double)B0 * (double)y0));

    tri.edge_eq[1][0] = A1;
    tri.edge_eq[1][1] = B1;
    tri.edge_eq[1][2] = (float)(-((double)A1 * (double)x1 + (double)B1 * (double)y1));

    tri.edge_eq[2][0] = A2;
    tri.edge_eq[2][1] = B2;
    tri.edge_eq[2][2] = (float)(-((double)A2 * (double)x2 + (double)B2 * (double)y2));

    // Top-left fill rule: after sign normalization, the inward-pointing normal is (A,B).
    // A "left edge" has A > 0; a "top edge" has A == 0 and B > 0.
    // Non-top-left edges get a small negative bias so shared-edge pixels are owned
    // by exactly one triangle.
    // Coordinates have 8 fractional bits; the edge-function quantum
    // is their product, 1/65536. A 1/256 bias erodes small triangles.
    constexpr float FILL_BIAS = -1.0f / 65536.0f;
    tri.edge_bias[0] = (A0 > 0.0f || (A0 == 0.0f && B0 > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[1] = (A1 > 0.0f || (A1 == 0.0f && B1 > 0.0f)) ? 0.0f : FILL_BIAS;
    tri.edge_bias[2] = (A2 > 0.0f || (A2 == 0.0f && B2 > 0.0f)) ? 0.0f : FILL_BIAS;

    // Bounding box (pixel coords, inclusive)
    float min_x = math::min3(x0, x1, x2);
    float min_y = math::min3(y0, y1, y2);
    float max_x = math::max3(x0, x1, x2);
    float max_y = math::max3(y0, y1, y2);

    hipgl_set_bbox(tri, static_cast<int>(floorf(min_x)),
                        static_cast<int>(floorf(min_y)),
                        static_cast<int>(ceilf(max_x)),
                        static_cast<int>(ceilf(max_y)));
}

void launch_triangle_setup(
    const TransformedVertex* verts, const uint32_t* indices,
    int tri_count, bool indexed,
    TriangleSetup* out, hipStream_t stream)
{
    if (!verts || !out || tri_count <= 0) return;
    const int block = 256;
    const int grid = (tri_count + block - 1) / block;
    hipLaunchKernelGGL(kernel_triangle_setup, grid, block, 0, stream,
        verts, indices, tri_count, indexed, out);
}

} // namespace kernels
} // namespace hipgl
