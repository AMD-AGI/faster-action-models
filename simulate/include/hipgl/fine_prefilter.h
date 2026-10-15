#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"

namespace hipgl {
namespace kernels {

// The three RenderState fields the prefilter actually consults, packed so the
// producer does not need the whole 500-B RenderState as a kernel argument.
struct FinePreState {
    int near_clip_source_count;
    int backface_cull_enabled;
    int front_face;
    int cull_face;
};

// Everything the per-tile sort needs in order to publish the hoisted records.
// `rec == nullptr` disables the epilogue entirely (the sort then behaves
// exactly as it did before), which is what the micro-draw / rocPRIM paths --
// the ones that never run this sort -- rely on.
struct FinePreEmit {
    const TriangleSetup* tris = nullptr;
    float4*              rec  = nullptr;   // 4 float4 per pool entry
    int*                 cnt  = nullptr;   // survivors per slot
    unsigned short*      msk  = nullptr;   // 2 B per pool entry (may be null)
    FinePreState         st   = {0, 0, 0, 0};
    int                  tiles_x   = 0;
    int                  tiles_mod = 0;    // real tile count (slot -> tile)
};

// Clipped draws sort by source/fan order, while keeping the common source
// setup densely stored. Decode only when a raster entry is first consumed;
// all subsequent edge, shading and statistics accesses use the physical ID.
HIPGL_DEVICE inline int hipgl_physical_triangle_id(uint32_t id, int near_clip_source_count) {
    const int n = near_clip_source_count;
    return n > 0 ? (int)(id >> 1) + ((id & 1u) ? n : 0) : (int)id;
}

HIPGL_DEVICE inline bool hipgl_triangle_is_culled_s(const TriangleSetup& tri,
                                                    const FinePreState& st) {
    if (!st.backface_cull_enabled) return false;
    const bool back = ((tri.hot_flags & 2) != 0) != (st.front_face != 0);
    return st.cull_face == 2 || (st.cull_face == 1 ? !back : back);
}

// Conservative whole-tile edge reject.  FP64 in the corner base is deliberate
// and load-bearing for the goldens -- see the note beside the f32 twin in
// bin_rasterize.hip.
HIPGL_DEVICE inline bool hipgl_tile_edge_reject_s(
    const TriangleSetup& tf, float tile_px0, float tile_py0, float* ecache)
{
    bool reject = false;
#pragma unroll
    for (int e = 0; e < 3; e++) {
        const float ea  = tf.edge_eq[e][0];
        const float eb  = tf.edge_eq[e][1];
        const float eba = (float)((double)ea*(double)tile_px0
                                + (double)eb*(double)tile_py0
                                + (double)tf.edge_eq[e][2]);
        const float ebi = tf.edge_bias[e];
        if (ecache) {
            ecache[4*e+0] = eba;
            ecache[4*e+1] = ea;
            ecache[4*e+2] = eb;
            ecache[4*e+3] = ebi;
        }
        const float hi  = eba + ((ea > 0.0f) ? ea * (float)(TILE_SIZE-1) : 0.0f)
                              + ((eb > 0.0f) ? eb * (float)(TILE_SIZE-1) : 0.0f) + ebi;
        const float mag = fabsf(eba) + (float)(TILE_SIZE-1)*fabsf(ea)
                        + (float)(TILE_SIZE-1)*fabsf(eb) + fabsf(ebi);
        if (hi < -(1e-5f * mag + 1e-30f)) reject = true;
    }
    return reject;
}

HIPGL_DEVICE inline unsigned int hipgl_fine_subtile_cov16_p(
    int hx, int tile_x, int ry0, int ry1)
{
    const int tile_col0 = tile_x * TILE_SIZE;
    int rx0 = hipgl_bbox_lo(hx) - tile_col0;
    int rx1 = hipgl_bbox_hi(hx) - tile_col0;
    if (rx0 < 0) rx0 = 0;
    if (rx1 > TILE_SIZE - 1) rx1 = TILE_SIZE - 1;
    if (rx0 > rx1) return 0u;                       // bbox misses the tile's columns
    const int i0 = rx0 >> 2, i1 = rx1 >> 2;         // sub-tile column band range
    const int j0 = ry0 >> 2, j1 = ry1 >> 2;         // sub-tile row band range
    const unsigned int cm = (0xFu << i0) & (0xFu >> (3 - i1));
    // The row bands, already spread to bit positions 0/4/8/12.
    const unsigned int rm = (0x1111u << (4 * j0)) & (0x1111u >> (4 * (3 - j1)));
    return rm * cm;   // cm < 16, so the "multiply" is a carry-free bit smear
}
HIPGL_DEVICE inline unsigned int hipgl_fine_subtile_cov16(
    const TriangleSetup& tf, int tile_x, int ry0, int ry1)
{
    return hipgl_fine_subtile_cov16_p(tf.hot_bbox_x, tile_x, ry0, ry1);
}

HIPGL_DEVICE inline unsigned int hipgl_fine_subtile_edge16(const float* ec)
{
    constexpr int FT = 4;                     // the consumer's sub-tile edge
    float ba[3], aa[3], bb[3];
#pragma unroll
    for (int e = 0; e < 3; e++) {
        const float c0 = ec[4*e+0], a = ec[4*e+1], b = ec[4*e+2], bi = ec[4*e+3];
        const float mag = fabsf(c0) + (fabsf(a) + fabsf(b)) * (float)TILE_SIZE
                        + fabsf(bi) + 1.0f;
        ba[e] = c0 + fmaxf(a, 0.0f) * (float)(FT-1) + fmaxf(b, 0.0f) * (float)(FT-1)
              + bi + 1.2e-5f * mag;
        aa[e] = a; bb[e] = b;
    }
    unsigned int bad = 0u;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float sy = (float)(FT * j);
        const float r0 = fmaf(bb[0], sy, ba[0]);
        const float r1 = fmaf(bb[1], sy, ba[1]);
        const float r2 = fmaf(bb[2], sy, ba[2]);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float sx = (float)(FT * i);
            const unsigned int acc = __float_as_uint(fmaf(aa[0], sx, r0))
                                   | __float_as_uint(fmaf(aa[1], sx, r1))
                                   | __float_as_uint(fmaf(aa[2], sx, r2));
            bad |= (acc >> 31) << (4*j + i);
        }
    }
    return ~bad;
}

HIPGL_DEVICE inline bool hipgl_tile_edge_reject_v(
    const float* a3, const float* b3, const float* c3, const float* bi3,
    float tile_px0, float tile_py0, float* ecache)
{
    bool reject = false;
#pragma unroll
    for (int e = 0; e < 3; e++) {
        const float ea  = a3[e];
        const float eb  = b3[e];
        const float eba = (float)((double)ea*(double)tile_px0
                                + (double)eb*(double)tile_py0
                                + (double)c3[e]);
        const float ebi = bi3[e];
        ecache[4*e+0] = eba;
        ecache[4*e+1] = ea;
        ecache[4*e+2] = eb;
        ecache[4*e+3] = ebi;
        const float hi  = eba + ((ea > 0.0f) ? ea * (float)(TILE_SIZE-1) : 0.0f)
                              + ((eb > 0.0f) ? eb * (float)(TILE_SIZE-1) : 0.0f) + ebi;
        const float mag = fabsf(eba) + (float)(TILE_SIZE-1)*fabsf(ea)
                        + (float)(TILE_SIZE-1)*fabsf(eb) + fabsf(ebi);
        if (hi < -(1e-5f * mag + 1e-30f)) reject = true;
    }
    return reject;
}

HIPGL_DEVICE inline bool hipgl_fine_prefilter_emit(
    const TriangleSetup* __restrict__ tris, unsigned long long key,
    const FinePreState& st, int tile_x, int tile_y, float4* out,
    unsigned int* out_cov = nullptr)
{
    const int physical_id = hipgl_physical_triangle_id((uint32_t)key, st.near_clip_source_count);
    const float4* __restrict__ line0 =
        reinterpret_cast<const float4*>(tris + physical_id);
    const float4 q0 = line0[0], q1 = line0[1], q2 = line0[2], q3 = line0[3];
    const int flags = __float_as_int(q3.x);
    if (flags & 1) return false;
    if (st.backface_cull_enabled) {
        const bool back = ((flags & 2) != 0) != (st.front_face != 0);
        if (st.cull_face == 2 || (st.cull_face == 1 ? !back : back)) return false;
    }
    const int tile_row0 = tile_y * TILE_SIZE;
    const int hy = __float_as_int(q3.y);
    int ry0 = hipgl_bbox_lo(hy) - tile_row0;
    int ry1 = hipgl_bbox_hi(hy) - tile_row0;
    if (ry0 < 0) ry0 = 0;
    if (ry1 > TILE_SIZE - 1) ry1 = TILE_SIZE - 1;
    if (ry0 > ry1) return false;                    // ROWSKIP: bbox misses the tile's rows
    const float tile_px0 = (float)(tile_x * TILE_SIZE) + 0.5f;
    const float tile_py0 = (float)(tile_y * TILE_SIZE) + 0.5f;
    const float a3[3]  = { q0.x, q0.w, q1.z };
    const float b3[3]  = { q0.y, q1.x, q1.w };
    const float c3[3]  = { q0.z, q1.y, q2.x };
    const float bi3[3] = { q2.y, q2.z, q2.w };
    float ec[12];
    if (hipgl_tile_edge_reject_v(a3, b3, c3, bi3, tile_px0, tile_py0, ec)) return false;
    out[0] = make_float4(ec[0],  ec[1],  ec[2],  ec[3]);
    out[1] = make_float4(ec[4],  ec[5],  ec[6],  ec[7]);
    out[2] = make_float4(ec[8],  ec[9],  ec[10], ec[11]);
    const unsigned int lo = (unsigned int)physical_id
                          | ((unsigned int)ry0 << 28) | ((unsigned int)ry1 << 24);
    out[3] = make_float4(__uint_as_float((unsigned int)(key >> 32)),
                         __uint_as_float(lo), 0.0f, 0.0f);
    if (out_cov) {
        unsigned int cov = hipgl_fine_subtile_cov16_p(__float_as_int(q3.z), tile_x, ry0, ry1);
        if (cov) cov &= hipgl_fine_subtile_edge16(ec);
        *out_cov = cov;
    }
    return true;
}

}  // namespace kernels
}  // namespace hipgl
