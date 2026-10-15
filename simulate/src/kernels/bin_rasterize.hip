// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Tile binning + fine rasterization kernels
// Binning: one thread per triangle, atomic add to tile bins.
// Fine raster: one block per tile, one thread per pixel.

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <climits>
#include "hipgl/types.h"
#include "hipgl/math_types.h"
#include "hipgl/msaa.h"
#include "hipgl/kernels.h"
#include "hipgl/fine_prefilter.h"
#include "hipgl/pipeline.h"

namespace hipgl {
namespace kernels {

// Clipped draws sort by source/fan order, while keeping the common source
// setup densely stored. Decode only when a raster entry is first consumed;
// all subsequent edge, shading and statistics accesses use the physical ID.
HIPGL_DEVICE inline int hipgl_physical_triangle_id(uint32_t id, const RenderState& state) {
    return hipgl_physical_triangle_id(id, state.near_clip_source_count);
}

// ── Binning kernel ──────────────────────────────────────────────────────────
// Each thread processes one triangle; atomically appends to overlapping tile bins.
HIPGL_DEVICE inline void hipgl_bin_triangle(
    const TriangleSetup* __restrict__ tris,
    int idx, int order_id,
    TileBin* __restrict__ tile_bins,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    int32_t* __restrict__ tri_stats)
{
    const TriangleSetup& tri = tris[idx];
    if (tri.degenerate) return;

    // Triangle bbox clamped to framebuffer
    int tmin_x = math::max2i(tri.bbox_min_x, 0);
    int tmin_y = math::max2i(tri.bbox_min_y, 0);
    int tmax_x = math::min2i(tri.bbox_max_x, fb_width - 1);
    int tmax_y = math::min2i(tri.bbox_max_y, fb_height - 1);
    if (tmin_x > tmax_x || tmin_y > tmax_y) {
        // Off-screen after clamping. Distinct from every other discard: the
        // triangle survived setup with a valid area and was still never
        // binned, which is what "input > 0, backface < input, binned == 0"
        // looks like from the outside.
        if (tri_stats) tri_stats[idx * TRI_STAT_COUNT + TRI_STAT_OFFSCREEN] = 1;
        return;
    }

    // Precompute triangle z_min for depth-sorted early-out
    float z_min = fminf(tri.v[0].screen_z, fminf(tri.v[1].screen_z, tri.v[2].screen_z));

    // Tile range that overlaps triangle bbox
    int tile_lo_x = tmin_x / TILE_SIZE;
    int tile_lo_y = tmin_y / TILE_SIZE;
    int tile_hi_x = tmax_x / TILE_SIZE;
    int tile_hi_y = tmax_y / TILE_SIZE;
    tile_lo_x = math::max2i(tile_lo_x, 0);
    tile_lo_y = math::max2i(tile_lo_y, 0);
    tile_hi_x = math::min2i(tile_hi_x, tiles_x - 1);
    tile_hi_y = math::min2i(tile_hi_y, tiles_y - 1);

    for (int ty = tile_lo_y; ty <= tile_hi_y; ty++) {
        for (int tx = tile_lo_x; tx <= tile_hi_x; tx++) {
            int bin_idx = ty * tiles_x + tx;
            int c = atomicAdd(&tile_bins[bin_idx].count, 1);
            if (c < MAX_TRIANGLES_PER_TILE) {
                tile_bins[bin_idx].triangle_indices[c] = order_id;
                tile_bins[bin_idx].triangle_z_min[c] = z_min;
                if (tri_stats) tri_stats[idx * TRI_STAT_COUNT + TRI_STAT_BINNED] = 1;
            } else if (tri_stats) {
                // The count keeps incrementing but the index is DROPPED. A
                // triangle can be binned into one tile and overflow another,
                // so these two are not exclusive.
                tri_stats[idx * TRI_STAT_COUNT + TRI_STAT_BIN_OVERFLOW] = 1;
            }
        }
    }
}

HIPGL_GLOBAL void kernel_bin_triangles(
    const TriangleSetup* __restrict__ tris, int tri_count,
    TileBin* __restrict__ tile_bins, int tiles_x, int tiles_y,
    int fb_width, int fb_height, int32_t* __restrict__ tri_stats,
    bool packed_ids)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tri_count)
        hipgl_bin_triangle(tris, idx, packed_ids ? 2 * idx : idx,
            tile_bins, tiles_x, tiles_y, fb_width, fb_height, tri_stats);
}

HIPGL_GLOBAL void kernel_bin_near_clip_extras(
    const TriangleSetup* __restrict__ tris, int source_count, NearClipQueue queue,
    TileBin* __restrict__ tile_bins, int tiles_x, int tiles_y,
    int fb_width, int fb_height, int32_t* __restrict__ tri_stats)
{
    const int count = min(max(*queue.count, 0), source_count);
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < count;
         q += blockDim.x * gridDim.x) {
        const int source = queue.ids[q];
        if ((unsigned)source >= (unsigned)source_count) continue;
        hipgl_bin_triangle(tris, source_count + source, 2 * source + 1,
            tile_bins, tiles_x, tiles_y, fb_width, fb_height, tri_stats);
    }
}

// ── Sort tile bins kernel ────────────────────────────────────────────────────
// After binning, triangle indices within each tile are in non-deterministic
// atomicAdd order. Sort them ascending so the fine rasterizer processes
// triangles in vertex-buffer order. With DEPTH_LEQUAL, this makes later-drawn
// (higher-index) triangles win coplanar Z ties — matching MuJoCo's sequential
// GL_LEQUAL rendering where later glDrawArrays calls overwrite earlier ones.
// Uses insertion sort (fast for small N ≤ MAX_TRIANGLES_PER_TILE, in-place).
HIPGL_GLOBAL void kernel_sort_tile_bins(
    TileBin* __restrict__ tile_bins,
    int num_tiles)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles) return;

    TileBin& bin = tile_bins[idx];
    int n = bin.count;
    if (n > MAX_TRIANGLES_PER_TILE) n = MAX_TRIANGLES_PER_TILE;
    if (n <= 1) return;

    for (int i = 1; i < n; i++) {
        int key = bin.triangle_indices[i];
        float key_z = bin.triangle_z_min[i];
        int j = i - 1;
        while (j >= 0 && bin.triangle_indices[j] > key) {
            bin.triangle_indices[j + 1] = bin.triangle_indices[j];
            bin.triangle_z_min[j + 1]   = bin.triangle_z_min[j];
            j--;
        }
        bin.triangle_indices[j + 1] = key;
        bin.triangle_z_min[j + 1]   = key_z;
    }
}

// Compile-time switch for the per-pixel early-out below (see its call site).
// Default 1 preserves existing behaviour; build with -DHIPGL_EARLY_OUT_ENABLED=0
// to disable it and attribute how much of the renderer's frame-to-frame
// instability it accounts for.
// Sort each tile bin by triangle index instead of by z_min. Diagnostic switch
// for whether bin ORDER (not bin capacity) is what makes rendering
// nonreproducible. Default 0 keeps the shipped z-sorted behaviour.
#ifndef HIPGL_SERIAL_TILE_SORT
#define HIPGL_SERIAL_TILE_SORT 0
#endif

#ifndef HIPGL_SORT_BY_INDEX
#define HIPGL_SORT_BY_INDEX 0
#endif

#ifndef HIPGL_EARLY_OUT_ENABLED
#define HIPGL_EARLY_OUT_ENABLED 1
#endif

// ── Depth test helper ────────────────────────────────────────────────────────
HIPGL_DEVICE inline bool depth_test_passes(DepthFunc func, float new_depth, float old_depth)
{
    switch (func) {
        case DEPTH_NEVER:    return false;
        case DEPTH_LESS:     return new_depth < old_depth;
        case DEPTH_LEQUAL:   return new_depth <= old_depth;
        case DEPTH_EQUAL:    return new_depth == old_depth;
        case DEPTH_GREATER:  return new_depth > old_depth;
        case DEPTH_GEQUAL:   return new_depth >= old_depth;
        case DEPTH_NOTEQUAL: return new_depth != old_depth;
        case DEPTH_ALWAYS:   return true;
        default:             return new_depth < old_depth;
    }
}

HIPGL_DEVICE inline float hipgl_rcp_nr(float x) {
    float r = __builtin_amdgcn_rcpf(x);
    r = __builtin_fmaf(__builtin_fmaf(-x, r, 1.0f), r, r);
    return r;
}

HIPGL_DEVICE inline float clamp01(float x) {
    return (x > 1.0f) ? 1.0f : (x < 0.0f ? 0.0f : x);
}

HIPGL_DEVICE inline bool triangle_is_culled(const TriangleSetup& tri,
                                           const RenderState& state) {
    const FinePreState st{state.near_clip_source_count, state.backface_cull_enabled,
                          state.front_face, state.cull_face};
    return hipgl_triangle_is_culled_s(tri, st);
}

// Binning-stage reject. cull_mask is the host-precomputed backface rule (bit b
// set == "reject triangles whose backfacing == b", see backface_cull_mask()),
// so the binner applies the same decision as triangle_is_culled() without
// needing a RenderState. cull_mask == 0 reduces this to the original
// degenerate-only test.
HIPGL_DEVICE inline bool hipgl_bin_reject(const TriangleSetup& tri, int cull_mask) {
    if (tri.degenerate) return true;
    return ((cull_mask >> (tri.backfacing ? 1 : 0)) & 1) != 0;
}

HIPGL_DEVICE inline void load_texture_rgb(
    const TextureState& ts, int texid, int px, int py,
    float& r, float& g, float& b)
{
    int toff = ts.tex_offset[texid];
    int tw = ts.tex_width[texid];
    int pidx = toff + (py * tw + px) * 3;
    r = ts.tex_data[pidx] / 255.0f;
    g = ts.tex_data[pidx + 1] / 255.0f;
    b = ts.tex_data[pidx + 2] / 255.0f;
}

HIPGL_DEVICE inline void bilinear_sample_texture(
    int texid, float u, float v,
    const TextureState& ts,
    float& cr, float& cg, float& cb)
{
    cr = 1.0f;
    cg = 1.0f;
    cb = 1.0f;
    if (texid < 0 || texid >= ts.ntexture || !ts.tex_data ||
        !ts.tex_width || !ts.tex_height || !ts.tex_offset) {
        return;
    }

    int tw = ts.tex_width[texid];
    int th = ts.tex_height[texid];
    if (tw <= 0 || th <= 0) return;

    u = u - floorf(u);
    v = v - floorf(v);
    float fx = u * (float)tw - 0.5f;
    float fy = v * (float)th - 0.5f;

    float fx0 = floorf(fx);
    float fy0 = floorf(fy);
    float dx = fx - fx0;
    float dy = fy - fy0;

    // Wrap into [0, tw) / [0, th). fx0 can be -1 (when u < 0.5/tw), so a plain
    // modulo is not enough -- add the extent before taking it.
    int x0 = (int)fx0 % tw; if (x0 < 0) x0 += tw;
    int y0 = (int)fy0 % th; if (y0 < 0) y0 += th;
    int x1 = x0 + 1; if (x1 >= tw) x1 -= tw;
    int y1 = y0 + 1; if (y1 >= th) y1 -= th;

    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    load_texture_rgb(ts, texid, x0, y0, r00, g00, b00);
    load_texture_rgb(ts, texid, x1, y0, r10, g10, b10);
    load_texture_rgb(ts, texid, x0, y1, r01, g01, b01);
    load_texture_rgb(ts, texid, x1, y1, r11, g11, b11);

    float w00 = (1.0f - dx) * (1.0f - dy);
    float w10 = dx * (1.0f - dy);
    float w01 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    cr = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    cg = g00*w00 + g10*w10 + g01*w01 + g11*w11;
    cb = b00*w00 + b10*w10 + b01*w01 + b11*w11;
}

HIPGL_DEVICE inline void cube_select_face(
    float x, float y, float z, int& face, float& s, float& t)
{
    float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    float sc, tc, ma;
    if (ax >= ay && ax >= az) {
        if (x > 0.0f) { face = 0; sc = -z; tc = -y; }
        else          { face = 1; sc =  z; tc = -y; }
        ma = ax;
    } else if (ay >= az) {
        if (y > 0.0f) { face = 2; sc =  x; tc =  z; }
        else          { face = 3; sc =  x; tc = -z; }
        ma = ay;
    } else {
        if (z > 0.0f) { face = 4; sc =  x; tc = -y; }
        else          { face = 5; sc = -x; tc = -y; }
        ma = az;
    }
    if (ma < 1e-20f) { face = 0; s = 0.5f; t = 0.5f; return; }
    s = 0.5f * (sc / ma + 1.0f);
    t = 0.5f * (tc / ma + 1.0f);
}

HIPGL_DEVICE inline void cube_sample_dir(
    int texid, float dx, float dy, float dz,
    const TextureState& ts,
    float& cr, float& cg, float& cb)
{
    cr = 1.0f; cg = 1.0f; cb = 1.0f;
    if (texid < 0 || texid >= ts.ntexture || !ts.tex_data ||
        !ts.tex_width || !ts.tex_height || !ts.tex_offset) {
        return;
    }
    int w = ts.tex_width[texid];
    if (w <= 0) return;

    int face; float s, t;
    cube_select_face(dx, dy, dz, face, s, t);

    // GL_CLAMP_TO_EDGE within the selected face. MuJoCo enables seamless cube
    // filtering when available (render_context.c:1651); without it, edge texels
    // clamp. All cube textures we ship are square so the difference is at most
    // one texel at a face seam.
    float fx = s * (float)w - 0.5f;
    float fy = t * (float)w - 0.5f;
    float fx0 = floorf(fx), fy0 = floorf(fy);
    float ddx = fx - fx0, ddy = fy - fy0;

    int x0 = (int)fx0, y0 = (int)fy0;
    int x1 = x0 + 1,   y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > w - 1 ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 > w - 1 ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 > w - 1 ? w - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 > w - 1 ? w - 1 : y1);

    int base = ts.tex_offset[texid] + face * w * w * 3;
    const unsigned char* T = ts.tex_data;
    int p00 = base + (y0 * w + x0) * 3, p10 = base + (y0 * w + x1) * 3;
    int p01 = base + (y1 * w + x0) * 3, p11 = base + (y1 * w + x1) * 3;

    float w00 = (1.0f - ddx) * (1.0f - ddy);
    float w10 = ddx * (1.0f - ddy);
    float w01 = (1.0f - ddx) * ddy;
    float w11 = ddx * ddy;
    cr = (T[p00+0]*w00 + T[p10+0]*w10 + T[p01+0]*w01 + T[p11+0]*w11) / 255.0f;
    cg = (T[p00+1]*w00 + T[p10+1]*w10 + T[p01+1]*w01 + T[p11+1]*w11) / 255.0f;
    cb = (T[p00+2]*w00 + T[p10+2]*w10 + T[p01+2]*w01 + T[p11+2]*w11) / 255.0f;
}

#include "texture_mip_sample.inc"

HIPGL_DEVICE inline void cube_sample_packed(
    int texid, float s, float packed_t,
    const TextureState& ts,
    float& cr, float& cg, float& cb)
{
    cr = 1.0f; cg = 1.0f; cb = 1.0f;
    if (texid < 0 || texid >= ts.ntexture || !ts.tex_data ||
        !ts.tex_width || !ts.tex_height || !ts.tex_offset) {
        return;
    }
    int w = ts.tex_width[texid];
    if (w <= 0) return;

    float f6 = packed_t * 6.0f;
    int face = (int)floorf(f6);
    if (face < 0) face = 0;
    if (face > 5) face = 5;
    float t = f6 - (float)face;

    float fx = s * (float)w - 0.5f;
    float fy = t * (float)w - 0.5f;
    float fx0 = floorf(fx), fy0 = floorf(fy);
    float ddx = fx - fx0, ddy = fy - fy0;

    int x0 = (int)fx0, y0 = (int)fy0;
    int x1 = x0 + 1,   y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > w - 1 ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 > w - 1 ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 > w - 1 ? w - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 > w - 1 ? w - 1 : y1);

    int base = ts.tex_offset[texid] + face * w * w * 3;
    const unsigned char* T = ts.tex_data;
    int p00 = base + (y0 * w + x0) * 3, p10 = base + (y0 * w + x1) * 3;
    int p01 = base + (y1 * w + x0) * 3, p11 = base + (y1 * w + x1) * 3;

    float w00 = (1.0f - ddx) * (1.0f - ddy);
    float w10 = ddx * (1.0f - ddy);
    float w01 = (1.0f - ddx) * ddy;
    float w11 = ddx * ddy;
    cr = (T[p00+0]*w00 + T[p10+0]*w10 + T[p01+0]*w01 + T[p11+0]*w11) / 255.0f;
    cg = (T[p00+1]*w00 + T[p10+1]*w10 + T[p01+1]*w01 + T[p11+1]*w11) / 255.0f;
    cb = (T[p00+2]*w00 + T[p10+2]*w10 + T[p01+2]*w01 + T[p11+2]*w11) / 255.0f;
}

HIPGL_DEVICE inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c * (1.0f / 12.92f) : __powf((c + 0.055f) * (1.0f / 1.055f), 2.4f);
}

// linear_output: decode one texture sample (sRGB -> linear).
HIPGL_DEVICE inline void decode_texel(const RenderState& state, float& r, float& g, float& b) {
    if (!state.linear_output) return;
    r = srgb_to_linear(r); g = srgb_to_linear(g); b = srgb_to_linear(b);
}

// linear_output: (exposure * c)^(1/gamma), applied after fog as the last shading step.
HIPGL_DEVICE inline void encode_output(const RenderState& state, float& r, float& g, float& b) {
    if (!state.linear_output) return;
    const float e = state.output_exposure, ig = state.output_inv_gamma;
    r = __powf(fminf(fmaxf(r * e, 0.0f), 1.0f), ig);
    g = __powf(fminf(fmaxf(g * e, 0.0f), 1.0f), ig);
    b = __powf(fminf(fmaxf(b * e, 0.0f), 1.0f), ig);
}

HIPGL_DEVICE inline void apply_linear_fog(
    const RenderState& state,
    const float* world_pos,
    float& r, float& g, float& b)
{
    encode_output(state, r, g, b);   // no fog in linear_output scenes; see encode_output
    if (state.fog_end <= state.fog_start) return;

    float dx = world_pos[0] - state.camera_world_pos[0];
    float dy = world_pos[1] - state.camera_world_pos[1];
    float dz = world_pos[2] - state.camera_world_pos[2];
    float dist = dx * state.camera_forward[0]
               + dy * state.camera_forward[1]
               + dz * state.camera_forward[2];
    if (dist < 0.0f) dist = 0.0f;
    float fog_f = (state.fog_end - dist) / (state.fog_end - state.fog_start);
    fog_f = clamp01(fog_f);
    r = fog_f * r + (1.0f - fog_f) * state.fog_color[0];
    g = fog_f * g + (1.0f - fog_f) * state.fog_color[1];
    b = fog_f * b + (1.0f - fog_f) * state.fog_color[2];
}

// Scalar overload. Taking `const float* world_pos` forces the caller's
// out_attr[] to have its address taken, which defeats SROA and pins the whole
// 13-float attribute array into scratch (84 B/lane, reported by
// -Rpass-analysis=kernel-resource-usage even though nothing spills). Only the
// first three components are ever read, so pass them by value.
HIPGL_DEVICE inline void apply_linear_fog(
    const RenderState& state,
    float wx, float wy, float wz,
    float& r, float& g, float& b)
{
    encode_output(state, r, g, b);
    if (state.fog_end <= state.fog_start) return;

    float dx = wx - state.camera_world_pos[0];
    float dy = wy - state.camera_world_pos[1];
    float dz = wz - state.camera_world_pos[2];
    float dist = dx * state.camera_forward[0]
               + dy * state.camera_forward[1]
               + dz * state.camera_forward[2];
    if (dist < 0.0f) dist = 0.0f;
    float fog_f = (state.fog_end - dist) / (state.fog_end - state.fog_start);
    fog_f = clamp01(fog_f);
    r = fog_f * r + (1.0f - fog_f) * state.fog_color[0];
    g = fog_f * g + (1.0f - fog_f) * state.fog_color[1];
    b = fog_f * b + (1.0f - fog_f) * state.fog_color[2];
}

// ── Fine rasterization kernel ────────────────────────────────────────────────
// One block per tile; each thread handles one pixel. Iterates triangles in bin.
HIPGL_GLOBAL void kernel_fine_rasterize(
    const TriangleSetup* __restrict__ tris,
    const TileBin* __restrict__ tile_bins,
    int tiles_x, int tiles_y,
    DeviceFramebuffer fb,
    RenderState state,
    TextureState tex_state)
{
    const int tile_x = blockIdx.x % tiles_x;
    const int tile_y = blockIdx.x / tiles_x;
    const int px = tile_x * TILE_SIZE + threadIdx.x;
    const int py = tile_y * TILE_SIZE + threadIdx.y;

    if (px >= fb.width || py >= fb.height) return;

    const int pixel_idx = py * fb.width + px;
    const int bin_idx = tile_y * tiles_x + tile_x;
    const TileBin& bin = tile_bins[bin_idx];
    const int bin_count = (bin.count < MAX_TRIANGLES_PER_TILE) ? bin.count : MAX_TRIANGLES_PER_TILE;

    const int pitch = (fb.pitch > 0) ? fb.pitch : fb.width * 4;
    const int color_idx = py * pitch + px * 4;

    float best_depth = fb.depth_buffer[pixel_idx];
    float out_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    if (state.blend_enabled) {
        out_color[0] = fb.color_buffer[color_idx + 0];
        out_color[1] = fb.color_buffer[color_idx + 1];
        out_color[2] = fb.color_buffer[color_idx + 2];
        out_color[3] = fb.color_buffer[color_idx + 3];
    }
    float out_attr[FRAG_ATTR_COUNT] = {0.0f};
    float oit_accum_r = 0.0f, oit_accum_g = 0.0f, oit_accum_b = 0.0f, oit_accum_a = 0.0f;
    float oit_revealage = 1.0f;
    bool any_fragment = false;

    const float tile_px0 = (float)(tile_x * TILE_SIZE) + 0.5f;
    const float tile_py0 = (float)(tile_y * TILE_SIZE) + 0.5f;
    const float thr_x = (float)threadIdx.x;
    const float thr_y = (float)threadIdx.y;

    for (int i = 0; i < bin_count; i++) {
        const int tri_idx = hipgl_physical_triangle_id((uint32_t)bin.triangle_indices[i], state);
        const TriangleSetup& tri = tris[tri_idx];

        if (tri.degenerate) continue;
        if (triangle_is_culled(tri,state)) continue;

        const TransformedVertex& v0 = tri.v[0];
        const TransformedVertex& v1 = tri.v[1];
        const TransformedVertex& v2 = tri.v[2];
        int tri_texid = (v0.texid == v1.texid && v1.texid == v2.texid) ? v0.texid : -1;

        // Tile-local incremental edge evaluation: compute E at tile origin in double
        // precision to eliminate cross-tile rounding discontinuities, then offset by
        // small threadIdx values (0..15) in float for per-pixel precision.
        float E0_base = (float)((double)tri.edge_eq[0][0]*(double)tile_px0 + (double)tri.edge_eq[0][1]*(double)tile_py0 + (double)tri.edge_eq[0][2]);
        float E1_base = (float)((double)tri.edge_eq[1][0]*(double)tile_px0 + (double)tri.edge_eq[1][1]*(double)tile_py0 + (double)tri.edge_eq[1][2]);
        float E2_base = (float)((double)tri.edge_eq[2][0]*(double)tile_px0 + (double)tri.edge_eq[2][1]*(double)tile_py0 + (double)tri.edge_eq[2][2]);

        float E0 = E0_base + tri.edge_eq[0][0]*thr_x + tri.edge_eq[0][1]*thr_y;
        float E1 = E1_base + tri.edge_eq[1][0]*thr_x + tri.edge_eq[1][1]*thr_y;
        float E2 = E2_base + tri.edge_eq[2][0]*thr_x + tri.edge_eq[2][1]*thr_y;

        if ((E0 + tri.edge_bias[0]) < 0.0f ||
            (E1 + tri.edge_bias[1]) < 0.0f ||
            (E2 + tri.edge_bias[2]) < 0.0f) continue;

        float b0 = E1, b1 = E2, b2 = E0;
        float inv_sum = 1.0f / (b0 + b1 + b2);

        // Perspective-correct interpolation: interpolate attr*inv_w, divide by interpolated inv_w
        float w0 = b0 * v0.inv_w;
        float w1 = b1 * v1.inv_w;
        float w2 = b2 * v2.inv_w;
        float sum_w = w0 + w1 + w2;
        if (fabsf(sum_w) < 1e-8f) continue;
        float inv_sum_w = hipgl_rcp_nr(sum_w);

        // Window depth already includes division by clip W. Interpolate it
        // affinely in screen space; only the other varyings use inv_w again.
        float depth = (b0 * v0.screen_z + b1 * v1.screen_z + b2 * v2.screen_z) * inv_sum;

        // Reflection-pass gates. MuJoCo clips the mirrored scene at the mirror
        // plane (glClipPlane) and stencils it to the mirror's footprint; hip has
        // neither, so both are applied per fragment here. Off by default, so
        // the normal path is untouched.
        if (state.reflect_mask && state.reflect_mask[pixel_idx] == 0) continue;
        if (state.clip_plane_enabled) {
            // Interpolated EYE-space position (v.world_pos is eye space; see the
            // note on RenderState::clip_plane).
            float ex = (w0*v0.world_pos[0] + w1*v1.world_pos[0] + w2*v2.world_pos[0]) * inv_sum_w;
            float ey = (w0*v0.world_pos[1] + w1*v1.world_pos[1] + w2*v2.world_pos[1]) * inv_sum_w;
            float ez = (w0*v0.world_pos[2] + w1*v1.world_pos[2] + w2*v2.world_pos[2]) * inv_sum_w;
            if (state.clip_plane[0]*ex + state.clip_plane[1]*ey +
                state.clip_plane[2]*ez - state.clip_plane[3] < 0.0f) continue;
        }

        // COVERED, before the depth test: this triangle produced a sample at
        // this pixel. The distinction from DEPTH_PASS below is the whole point
        // of these counters -- winner-only capture cannot tell "never
        // rasterized" from "rasterized and lost the depth test".
        if (state.tri_stats)
            state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_COVERED] = 1;

        // Depth test
        if (state.depth_test_enabled && !depth_test_passes(state.depth_func, depth, best_depth))
            continue;

        if (state.tri_stats)
            state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_DEPTH_PASS] = 1;


        // Interpolate attributes (perspective-correct) before depth write
        float r = (w0*v0.color[0] + w1*v1.color[0] + w2*v2.color[0]) * inv_sum_w;
        float g = (w0*v0.color[1] + w1*v1.color[1] + w2*v2.color[1]) * inv_sum_w;
        float b = (w0*v0.color[2] + w1*v1.color[2] + w2*v2.color[2]) * inv_sum_w;
        float a = (w0*v0.color[3] + w1*v1.color[3] + w2*v2.color[3]) * inv_sum_w;

        // MuJoCo dims the reflected material before lighting, preserving
        // specular intensity. Gouraud colors are dimmed in the vertex stage.
        if (state.reflect_dim>0.f && !(state.gouraud_shading && state.lighting_enabled)) {
            r*=state.reflect_dim; g*=state.reflect_dim; b*=state.reflect_dim;
        }

        // Each pixel is processed by exactly one tile thread; within a tile,
        // we iterate triangles sequentially, so a simple write suffices.
        bool wrote = true;
        if (state.depth_write_enabled) {
            fb.depth_buffer[pixel_idx] = depth;
            best_depth = depth;
        }

        if (!wrote) continue;

        any_fragment = true;

        out_attr[0]  = (w0*v0.world_pos[0] + w1*v1.world_pos[0] + w2*v2.world_pos[0]) * inv_sum_w;
        out_attr[1]  = (w0*v0.world_pos[1] + w1*v1.world_pos[1] + w2*v2.world_pos[1]) * inv_sum_w;
        out_attr[2]  = (w0*v0.world_pos[2] + w1*v1.world_pos[2] + w2*v2.world_pos[2]) * inv_sum_w;
        out_attr[3]  = (w0*v0.world_normal[0] + w1*v1.world_normal[0] + w2*v2.world_normal[0]) * inv_sum_w;
        out_attr[4]  = (w0*v0.world_normal[1] + w1*v1.world_normal[1] + w2*v2.world_normal[1]) * inv_sum_w;
        out_attr[5]  = (w0*v0.world_normal[2] + w1*v1.world_normal[2] + w2*v2.world_normal[2]) * inv_sum_w;
#pragma unroll
        for (int k = 0; k < TEXCOORD_COMPONENTS; k++) {
            out_attr[FRAG_ATTR_TEXCOORD + k] =
                (w0*v0.texcoord[k] + w1*v1.texcoord[k] + w2*v2.texcoord[k]) * inv_sum_w;
        }
        out_attr[FRAG_ATTR_COLOR + 0] = r; out_attr[FRAG_ATTR_COLOR + 1] = g;
        out_attr[FRAG_ATTR_COLOR + 2] = b; out_attr[FRAG_ATTR_COLOR + 3] = a;
        // Which surface won this pixel. Set here, alongside the other
        // attributes and under the same depth-test guard, so it always
        // describes the fragment that is actually visible.
        out_attr[FRAG_ATTR_VERTEX] = (float)tri.first_vertex;

        // Shading: either per-vertex Gouraud (matching MuJoCo) or per-fragment Blinn-Phong.
        float final_r, final_g, final_b, final_a;
        float frag_sr = 0.0f, frag_sg = 0.0f, frag_sb = 0.0f;

        if (state.gouraud_shading && state.lighting_enabled) {
            // Gouraud: interpolate pre-computed per-vertex lit colors
            // (lighting already includes specular in GL_SINGLE_COLOR mode)
            final_r = (w0*v0.lit_color[0] + w1*v1.lit_color[0] + w2*v2.lit_color[0]) * inv_sum_w;
            final_g = (w0*v0.lit_color[1] + w1*v1.lit_color[1] + w2*v2.lit_color[1]) * inv_sum_w;
            final_b = (w0*v0.lit_color[2] + w1*v1.lit_color[2] + w2*v2.lit_color[2]) * inv_sum_w;
            final_a = a;
        } else if (state.lighting_enabled) {
            float ex = out_attr[0], ey = out_attr[1], ez = out_attr[2];
            float nx = out_attr[3], ny = out_attr[4], nz = out_attr[5];

            float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
            if (nlen > 1e-8f) { nx /= nlen; ny /= nlen; nz /= nlen; }
            if (state.two_sided_lighting && nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

            // NO two-sided lighting: MuJoCo disables it explicitly at
            // render_gl3.c:669, glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 0).
            // Flipping nz < 0 lights surfaces that face away from the viewer,
            // which is camera-dependent by construction and made the gripper
            // 1.93x too bright in the wrist view while looking correct from
            // agentview. See fused_vertex_tri.hip for the measurements.

            float frag_mat_spec = (w0*v0.mat_specular + w1*v1.mat_specular + w2*v2.mat_specular) * inv_sum_w;
            float frag_mat_shin = (w0*v0.mat_shininess + w1*v1.mat_shininess + w2*v2.mat_shininess) * inv_sum_w;
            float frag_mat_emis = (w0*v0.mat_emission + w1*v1.mat_emission + w2*v2.mat_emission) * inv_sum_w;

            // MuJoCo OpenGL: emission is an unlit additive term (gl_FrontMaterial.emission)
            float lit_r = frag_mat_emis * r;
            float lit_g = frag_mat_emis * g;
            float lit_b = frag_mat_emis * b;

            float shadow_factor = 1.0f;
            if (state.shadow_enabled && state.shadow_depth) {
                const float* sm = state.shadow_mvp;
                float sx = sm[0]*ex + sm[4]*ey + sm[8]*ez  + sm[12];
                float sy = sm[1]*ex + sm[5]*ey + sm[9]*ez  + sm[13];
                float sz = sm[2]*ex + sm[6]*ey + sm[10]*ez + sm[14];
                float sw = sm[3]*ex + sm[7]*ey + sm[11]*ez + sm[15];
                if (fabsf(sw) > 1e-8f) {
                    float inv_sw = 1.0f / sw;
                    float su = sx * inv_sw * 0.5f + 0.5f;
                    float sv = sy * inv_sw * 0.5f + 0.5f;
                    float sd = sz * inv_sw * 0.5f + 0.5f;
                    int si = (int)(su * state.shadow_w);
                    int sj = (int)(sv * state.shadow_h);
                    if (si >= 0 && si < state.shadow_w &&
                        sj >= 0 && sj < state.shadow_h) {
                        float map_d = state.shadow_depth[sj * state.shadow_w + si];
                        if (sd > map_d + state.shadow_bias)
                            shadow_factor = 0.0f;
                    }
                }
            }

            for (int li = 0; li < MAX_LIGHTS; li++) {
                if (!state.lights[li].enabled) continue;

                float lx, ly, lz;
                float atten = 1.0f;

                if (state.lights[li].position[3] == 0.0f) {
                    lx = state.lights[li].position[0];
                    ly = state.lights[li].position[1];
                    lz = state.lights[li].position[2];
                    float ll = sqrtf(lx*lx + ly*ly + lz*lz);
                    if (ll > 1e-8f) { lx /= ll; ly /= ll; lz /= ll; }
                } else {
                    lx = state.lights[li].position[0] - ex;
                    ly = state.lights[li].position[1] - ey;
                    lz = state.lights[li].position[2] - ez;
                    float dist = sqrtf(lx*lx + ly*ly + lz*lz);
                    if (dist > 1e-8f) { lx /= dist; ly /= dist; lz /= dist; }
                    float denom = state.lights[li].constant_atten
                                + state.lights[li].linear_atten * dist
                                + state.lights[li].quadratic_atten * dist * dist;
                    atten = (denom > 1e-8f) ? (1.0f / denom) : 1.0f;
                }

                float spot_atten = 1.0f;
                if (state.lights[li].spot_cutoff < 180.0f && state.lights[li].position[3] != 0.0f) {
                    float sdx = state.lights[li].direction[0];
                    float sdy = state.lights[li].direction[1];
                    float sdz = state.lights[li].direction[2];
                    float sdlen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
                    if (sdlen > 1e-8f) { sdx /= sdlen; sdy /= sdlen; sdz /= sdlen; }
                    float cos_angle = -(lx*sdx + ly*sdy + lz*sdz);
                    float cos_cutoff = cosf(state.lights[li].spot_cutoff * 3.14159265f / 180.0f);
                    if (cos_angle < cos_cutoff) {
                        spot_atten = 0.0f;
                    } else {
                        spot_atten = powf(cos_angle, state.lights[li].spot_exponent);
                    }
                }
                atten *= spot_atten;

                float NdotL = nx*lx + ny*ly + nz*lz;
                if (NdotL < 0.0f) NdotL = 0.0f;

                float li_shadow = (li == 0) ? shadow_factor : 1.0f;

                lit_r += atten * (state.lights[li].ambient[0] * r
                               + state.lights[li].diffuse[0] * r * NdotL * li_shadow);
                lit_g += atten * (state.lights[li].ambient[1] * g
                               + state.lights[li].diffuse[1] * g * NdotL * li_shadow);
                lit_b += atten * (state.lights[li].ambient[2] * b
                               + state.lights[li].diffuse[2] * b * NdotL * li_shadow);

                if (NdotL > 0.0f && frag_mat_spec > 0.0f) {
                    float vdx = -ex, vdy = -ey, vdz = -ez;
                    float vlen = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
                    if (vlen > 1e-8f) { vdx /= vlen; vdy /= vlen; vdz /= vlen; }
                    float hx = lx + vdx, hy = ly + vdy, hz = lz + vdz;
                    float hlen = sqrtf(hx*hx + hy*hy + hz*hz);
                    if (hlen > 1e-8f) { hx /= hlen; hy /= hlen; hz /= hlen; }

                    float NdotH = nx*hx + ny*hy + nz*hz;
                    if (NdotH < 0.0f) NdotH = 0.0f;
                    float spec_power = (frag_mat_shin > 0.0f) ? powf(NdotH, frag_mat_shin) : 1.0f;

                    frag_sr += atten * state.lights[li].specular[0] * frag_mat_spec * spec_power * li_shadow;
                    frag_sg += atten * state.lights[li].specular[1] * frag_mat_spec * spec_power * li_shadow;
                    frag_sb += atten * state.lights[li].specular[2] * frag_mat_spec * spec_power * li_shadow;
                }
            }

            final_r = clamp01(lit_r);
            final_g = clamp01(lit_g);
            final_b = clamp01(lit_b);
            final_a = a;
        } else {
            final_r = r;
            final_g = g;
            final_b = b;
            final_a = a;
            frag_sr = (w0*v0.spec_color[0] + w1*v1.spec_color[0] + w2*v2.spec_color[0]) * inv_sum_w;
            frag_sg = (w0*v0.spec_color[1] + w1*v1.spec_color[1] + w2*v2.spec_color[1]) * inv_sum_w;
            frag_sb = (w0*v0.spec_color[2] + w1*v1.spec_color[2] + w2*v2.spec_color[2]) * inv_sum_w;
        }

        // MuJoCo uses GL_SINGLE_COLOR (default GL_LIGHT_MODEL_COLOR_CONTROL):
        // texture modulates ALL lighting including specular. Add specular to
        // the lit color BEFORE texture multiplication.
        frag_sr = clamp01(frag_sr);
        frag_sg = clamp01(frag_sg);
        frag_sb = clamp01(frag_sb);
        final_r += frag_sr;
        final_g += frag_sg;
        final_b += frag_sb;

        if (tri_texid >= 0) {
            float tex_r, tex_g, tex_b;
            sample_triangle_texture(tri,tri_texid,E0,E1,E2,
                &out_attr[FRAG_ATTR_TEXCOORD],pixel_idx,fb.width,fb.height,
                tex_state,tex_r,tex_g,tex_b);
            decode_texel(state, tex_r, tex_g, tex_b);
            final_r *= tex_r;
            final_g *= tex_g;
            final_b *= tex_b;
        }

        apply_linear_fog(state, out_attr, final_r, final_g, final_b);
        final_r = clamp01(final_r);
        final_g = clamp01(final_g);
        final_b = clamp01(final_b);
        final_a = clamp01(final_a);

        if (state.oit_enabled && fb.oit_accum && fb.oit_revealage) {
            // Weighted Blended OIT (McGuire-Bavoil 2013): accumulate into
            // separate buffers instead of writing to color_buffer.
            float z = depth;
            float d_scaled = z * 200.0f;
            float w = final_a * fmaxf(0.01f, fminf(3000.0f,
                      0.03f / (1e-5f + d_scaled * d_scaled * d_scaled * d_scaled)));
            float pr = final_r * final_a;
            float pg = final_g * final_a;
            float pb = final_b * final_a;
            oit_accum_r += pr * w;
            oit_accum_g += pg * w;
            oit_accum_b += pb * w;
            oit_accum_a += final_a * w;
            oit_revealage *= (1.0f - final_a);
        } else if (state.blend_enabled) {
            float src_a = final_a;
            float inv_a = 1.0f - src_a;
            if (state.blend_src==BLEND_ONE && state.blend_dst==BLEND_ONE) {
                out_color[0]=clamp01(final_r+out_color[0]);
                out_color[1]=clamp01(final_g+out_color[1]);
                out_color[2]=clamp01(final_b+out_color[2]);
                out_color[3]=clamp01(final_a+out_color[3]);
            } else {
                out_color[0] = final_r * src_a + out_color[0] * inv_a;
                out_color[1] = final_g * src_a + out_color[1] * inv_a;
                out_color[2] = final_b * src_a + out_color[2] * inv_a;
                out_color[3] = src_a + out_color[3] * inv_a;
            }
        } else {
            out_color[0] = final_r;
            out_color[1] = final_g;
            out_color[2] = final_b;
            out_color[3] = final_a;
        }
    }

    if (state.oit_enabled && fb.oit_accum && fb.oit_revealage && any_fragment) {
        int accum_idx = pixel_idx * 4;
        fb.oit_accum[accum_idx + 0] += oit_accum_r;
        fb.oit_accum[accum_idx + 1] += oit_accum_g;
        fb.oit_accum[accum_idx + 2] += oit_accum_b;
        fb.oit_accum[accum_idx + 3] += oit_accum_a;
        fb.oit_revealage[pixel_idx] *= oit_revealage;
    } else if (any_fragment) {
        fb.color_buffer[color_idx + 0] = out_color[0];
        fb.color_buffer[color_idx + 1] = out_color[1];
        fb.color_buffer[color_idx + 2] = out_color[2];
        fb.color_buffer[color_idx + 3] = out_color[3];

        // Diagnostic capture only; never allocated in a production render.
        // Written under the SAME guard as color -- inside `any_fragment`, after
        // the depth test, from the same out_attr the winning fragment shaded
        // with -- so a captured attribute always corresponds to the pixel the
        // viewer sees. The bound is FRAG_ATTR_COUNT, not a literal: this loop
        // read `k < 12` while the layout had already grown to 13, silently
        // truncating the last channel.
        if (fb.attr_buffer) {
            int attr_base = pixel_idx * FRAG_ATTR_COUNT;
#pragma unroll
            for (int k = 0; k < FRAG_ATTR_COUNT; k++)
                fb.attr_buffer[attr_base + k] = out_attr[k];
        }
    }
}

// ── Clear tile bin counts kernel ────────────────────────────────────────────
// Only zeros the count field (4 bytes per tile) instead of memset-ing the
// entire TileBin struct (~16KB per tile). At 256x256 with 16x16 tiles this
// reduces the clear from ~4MB to ~1KB.
HIPGL_GLOBAL void kernel_clear_tile_bin_counts(
    TileBin* __restrict__ tile_bins,
    int num_tiles)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles) return;
    tile_bins[idx].count = 0;
}

__device__ __forceinline__ uint32_t hipgl_f2ord(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float hipgl_ord2f(uint32_t o) {
    uint32_t u = (o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o;
    return __uint_as_float(u);
}

#ifndef HIPGL_FINE_WPEU
#define HIPGL_FINE_WPEU 2
#endif
#ifndef HIPGL_FINE_WPEU_PERSIST
#define HIPGL_FINE_WPEU_PERSIST 5
#endif
#ifndef HIPGL_FINE_WPEU_NP
#define HIPGL_FINE_WPEU_NP 6
#endif
#ifndef HIPGL_FINE_WPEU_NPMAX
#define HIPGL_FINE_WPEU_NPMAX 6
#endif
#ifndef HIPGL_FINE_WPEU_FULLATTR
#define HIPGL_FINE_WPEU_FULLATTR 4
#endif
#ifndef HP_MICRO_WPEU_GPHONG
#define HP_MICRO_WPEU_GPHONG 4
#endif
#ifndef HIPGL_FINE_MULT_HOIST
#define HIPGL_FINE_MULT_HOIST 1
#endif

#ifndef HP_FINE_ZWCACHE
#define HP_FINE_ZWCACHE 2
#endif
#ifndef HP_FINE_ROWS
#define HP_FINE_ROWS 4
#endif
#ifndef HP_FINE_ROWS_WPEU
#define HP_FINE_ROWS_WPEU 5
#endif
#ifndef HP_FINE_ROWS_WPEU_MAX
#define HP_FINE_ROWS_WPEU_MAX 5
#endif
#ifndef HIPGL_FINE_MULT
#define HIPGL_FINE_MULT 1
#endif

#ifndef HP_MICRO_PF_ROWS
#define HP_MICRO_PF_ROWS 4
#endif
#ifndef HP_MICRO_PF_MULT
#define HP_MICRO_PF_MULT 1
#endif

#ifndef HIPGL_FINE_XCD_BAND_MAX
#define HIPGL_FINE_XCD_BAND_MAX 256
#endif
#ifndef HIPGL_FINE_XCD_HI
#define HIPGL_FINE_XCD_HI 2
#endif
#ifndef HIPGL_FINE_XCD_CHUNK
#define HIPGL_FINE_XCD_CHUNK 16
#endif

#ifndef HIPGL_FINE_PIPE
#define HIPGL_FINE_PIPE 0
#endif

#ifndef HIPGL_FINE_CHUNK
#define HIPGL_FINE_CHUNK (TILE_SIZE * TILE_SIZE)
#endif
#ifndef HIPGL_PERSIST_GRANT
#define HIPGL_PERSIST_GRANT 8
#endif
#define HIPGL_FINE_NWAVE ((TILE_SIZE * TILE_SIZE) / 64)
HIPGL_GLOBAL void kernel_pack_pool_keys(
    const int* __restrict__ pool_tri,
    const float* __restrict__ pool_z,
    unsigned long long* __restrict__ pool_key,
    const int* __restrict__ total_ptr,
    int pool_cap)
{
    int n = pool_cap;
    if (total_ptr) { int t = *total_ptr; if (t < n) n = t; }
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pool_key[i] = ((unsigned long long)hipgl_f2ord(pool_z[i]) << 32)
                | (unsigned long long)(uint32_t)pool_tri[i];
}

void launch_pack_pool_keys(const int* pool_tri, const float* pool_z,
                           unsigned long long* pool_key, const int* total_ptr,
                           int pool_cap, hipStream_t stream)
{
    if (!pool_tri || !pool_z || !pool_key || pool_cap <= 0) return;
    const int block = 256;
    const int grid = (pool_cap + block - 1) / block;
    hipLaunchKernelGGL(kernel_pack_pool_keys, grid, block, 0, stream,
        pool_tri, pool_z, pool_key, total_ptr, pool_cap);
}

// ── Per-sample shading body (extracted verbatim) ────────────────────────────
// Called once per COVERED sample. Identical FP expressions to the pre-existing
// inline body -- this is a pure code motion so the two thread mappings below
// (sample-major legacy, pixel-major wave) produce bit-identical fragments.
// GOURAUD: the caller has already checked, ON THE HOST, that
// state.gouraud_shading && state.lighting_enabled.  instinct_sim_bind defaults
// gouraud_shading to 1 (it is what matches MuJoCo), so in every production
// render the entire per-fragment Blinn-Phong block below -- the MAX_LIGHTS
// loop with its shadow lookup, spot cutoff, powf() and half-vector normalise
// -- is DEAD CODE that never executes, yet it still sets the register
// allocation for the whole kernel (119 VGPR -> 4 waves/SIMD).  Selecting the
// branch at instantiation time deletes it outright from the hot kernel.  The
// surviving expressions are byte-for-byte the ones in the `gouraud_shading &&
// lighting_enabled` arm, at the same site with the same operands, so the
// arithmetic is unchanged.
// NO_ATTR: fb.attr_buffer is a DIAGNOSTIC output and is null in every
// production render. The compiler cannot prove that, so it keeps oa_nx/ny/nz,
// oa_vtx and the four base-colour components live from the interpolation site
// all the way past the texture fetch and the fog. Telling it at instantiation
// time lets the whole group die immediately after use.
template <bool WITH_OIT, bool DIRECT, bool GOURAUD, bool NO_ATTR, bool SHADING_ONLY = false,
          bool MASK_LATCHED = false>
__device__ __forceinline__ void hipgl_fine_shade_sample(
    const TriangleSetup* __restrict__ tris, int tri_idx,
    float E0, float E1, float E2,
    int pixel_idx, int color_idx,
    const DeviceFramebuffer& fb, const RenderState& state,
    const TextureState& tex_state,
    float& best_depth,
    float& out_color0, float& out_color1, float& out_color2, float& out_color3,
    float& oit_accum_r, float& oit_accum_g, float& oit_accum_b,
    float& oit_accum_a, float& oit_revealage, bool& any_fragment)
{
    // out_attr is a DIAGNOSTIC capture (fb.attr_buffer is null in production).
    // Holding it across the whole survivor walk cost FRAG_ATTR_COUNT VGPRs per
    // OWNED SAMPLE once a thread owns more than one; writing it here on each
    // depth-passing fragment is last-write-wins, i.e. identical to the old
    // end-of-kernel dump.
    // Scalarised: the six eye-space components and the texcoords are plain
    // locals and the colour / vertex slots are reconstructed at flush time, so
    // no addressable 14-float array exists on the hot path -- exactly what the
    // apply_linear_fog note above warns pins 84 B/lane into scratch.
    float oa_px = 0.0f, oa_py = 0.0f, oa_pz = 0.0f;
    float oa_nx = 0.0f, oa_ny = 0.0f, oa_nz = 0.0f;
    float oa_tc[TEXCOORD_COMPONENTS] = {0.0f};
    float oa_vtx = 0.0f;
        // Only NOW is the 484 B TriangleSetup worth touching: this entry
        // actually covers this pixel.
        const TriangleSetup& tri = tris[tri_idx];
        const TransformedVertex& v0 = tri.v[0];
        const TransformedVertex& v1 = tri.v[1];
        const TransformedVertex& v2 = tri.v[2];

        float b0 = E1, b1 = E2, b2 = E0;
        float inv_sum = 1.0f / (b0 + b1 + b2);

        float w0 = b0 * v0.inv_w;
        float w1 = b1 * v1.inv_w;
        float w2 = b2 * v2.inv_w;
        float sum_w = w0 + w1 + w2;
        if (fabsf(sum_w) < 1e-8f) return;
        float inv_sum_w = hipgl_rcp_nr(sum_w);

        // Window depth already includes division by clip W. Interpolate it
        // affinely in screen space; only the other varyings use inv_w again.
        float depth = (b0 * v0.screen_z + b1 * v1.screen_z + b2 * v2.screen_z) * inv_sum;

        if constexpr (!SHADING_ONLY) {
        if constexpr (!MASK_LATCHED) {
            if (state.reflect_mask && state.reflect_mask[pixel_idx] == 0) return;
        }
        if (state.clip_plane_enabled) {
            // Interpolated EYE-space position (v.world_pos is eye space; see the
            // note on RenderState::clip_plane).
            float ex = (w0*v0.world_pos[0] + w1*v1.world_pos[0] + w2*v2.world_pos[0]) * inv_sum_w;
            float ey = (w0*v0.world_pos[1] + w1*v1.world_pos[1] + w2*v2.world_pos[1]) * inv_sum_w;
            float ez = (w0*v0.world_pos[2] + w1*v1.world_pos[2] + w2*v2.world_pos[2]) * inv_sum_w;
            if (state.clip_plane[0]*ex + state.clip_plane[1]*ey +
                state.clip_plane[2]*ez - state.clip_plane[3] < 0.0f) return;
        }

        // COVERED, before the depth test: this triangle produced a sample at
        // this pixel. The distinction from DEPTH_PASS below is the whole point
        // of these counters -- winner-only capture cannot tell "never
        // rasterized" from "rasterized and lost the depth test".
        if (state.tri_stats)
            state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_COVERED] = 1;

        if (state.depth_test_enabled) {
            if (state.depth_func == DEPTH_LEQUAL) {
                if (!(depth <= best_depth)) return;
            } else if (!depth_test_passes(state.depth_func, depth, best_depth)) {
                return;
            }
        }

        if (state.tri_stats)
            state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_DEPTH_PASS] = 1;


        }

        float r = (w0*v0.color[0] + w1*v1.color[0] + w2*v2.color[0]) * inv_sum_w;
        float g = (w0*v0.color[1] + w1*v1.color[1] + w2*v2.color[1]) * inv_sum_w;
        float b = (w0*v0.color[2] + w1*v1.color[2] + w2*v2.color[2]) * inv_sum_w;
        float a = (w0*v0.color[3] + w1*v1.color[3] + w2*v2.color[3]) * inv_sum_w;

        if (state.reflect_dim>0.f && !(state.gouraud_shading && state.lighting_enabled)) {
            r*=state.reflect_dim; g*=state.reflect_dim; b*=state.reflect_dim;
        }

        bool wrote = true;
        if (!SHADING_ONLY && state.depth_write_enabled) {
            fb.depth_buffer[pixel_idx] = depth;
            best_depth = depth;
        }

        if (!wrote) return;

        any_fragment = true;

        const bool oa_want_attr = (!NO_ATTR) && (fb.attr_buffer != nullptr);
        const int tri_texid = (v0.texid == v1.texid && v1.texid == v2.texid) ? v0.texid : -1;
        const bool oa_need_wp = oa_want_attr || (state.fog_end > state.fog_start)
                              || ((!GOURAUD) && state.lighting_enabled);
        const bool oa_need_n  = oa_want_attr || ((!GOURAUD) && state.lighting_enabled);
        const bool oa_need_tc = oa_want_attr || (tri_texid >= 0);
        if (oa_need_wp) {
            oa_px  = (w0*v0.world_pos[0] + w1*v1.world_pos[0] + w2*v2.world_pos[0]) * inv_sum_w;
            oa_py  = (w0*v0.world_pos[1] + w1*v1.world_pos[1] + w2*v2.world_pos[1]) * inv_sum_w;
            oa_pz  = (w0*v0.world_pos[2] + w1*v1.world_pos[2] + w2*v2.world_pos[2]) * inv_sum_w;
        }
        if (oa_need_n) {
            oa_nx  = (w0*v0.world_normal[0] + w1*v1.world_normal[0] + w2*v2.world_normal[0]) * inv_sum_w;
            oa_ny  = (w0*v0.world_normal[1] + w1*v1.world_normal[1] + w2*v2.world_normal[1]) * inv_sum_w;
            oa_nz  = (w0*v0.world_normal[2] + w1*v1.world_normal[2] + w2*v2.world_normal[2]) * inv_sum_w;
        }
        if (oa_need_tc) {
#pragma unroll
            for (int k = 0; k < TEXCOORD_COMPONENTS; k++) {
                oa_tc[k] =
                    (w0*v0.texcoord[k] + w1*v1.texcoord[k] + w2*v2.texcoord[k]) * inv_sum_w;
            }
        }
        // Which surface won this pixel. Set here, alongside the other
        // attributes and under the same depth-test guard, so it always
        // describes the fragment that is actually visible.
        if (oa_want_attr) oa_vtx = (float)tri.first_vertex;

        float final_r, final_g, final_b, final_a;
        float frag_sr = 0.0f, frag_sg = 0.0f, frag_sb = 0.0f;

        if constexpr (GOURAUD) {
            final_r = (w0*v0.lit_color[0] + w1*v1.lit_color[0] + w2*v2.lit_color[0]) * inv_sum_w;
            final_g = (w0*v0.lit_color[1] + w1*v1.lit_color[1] + w2*v2.lit_color[1]) * inv_sum_w;
            final_b = (w0*v0.lit_color[2] + w1*v1.lit_color[2] + w2*v2.lit_color[2]) * inv_sum_w;
            final_a = a;
        } else if (state.gouraud_shading && state.lighting_enabled) {
            final_r = (w0*v0.lit_color[0] + w1*v1.lit_color[0] + w2*v2.lit_color[0]) * inv_sum_w;
            final_g = (w0*v0.lit_color[1] + w1*v1.lit_color[1] + w2*v2.lit_color[1]) * inv_sum_w;
            final_b = (w0*v0.lit_color[2] + w1*v1.lit_color[2] + w2*v2.lit_color[2]) * inv_sum_w;
            final_a = a;
        } else if (state.lighting_enabled) {
            float ex = oa_px, ey = oa_py, ez = oa_pz;
            float nx = oa_nx, ny = oa_ny, nz = oa_nz;

            float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
            if (nlen > 1e-8f) { nx /= nlen; ny /= nlen; nz /= nlen; }
            if (state.two_sided_lighting && nz < 0.0f) { nx = -nx; ny = -ny; nz = -nz; }

            // No two-sided flip here either -- see the note above.

            float frag_mat_spec = (w0*v0.mat_specular + w1*v1.mat_specular + w2*v2.mat_specular) * inv_sum_w;
            float frag_mat_shin = (w0*v0.mat_shininess + w1*v1.mat_shininess + w2*v2.mat_shininess) * inv_sum_w;
            float frag_mat_emis = (w0*v0.mat_emission + w1*v1.mat_emission + w2*v2.mat_emission) * inv_sum_w;

            float lit_r = frag_mat_emis * r;
            float lit_g = frag_mat_emis * g;
            float lit_b = frag_mat_emis * b;

            float shadow_factor = 1.0f;
            if (state.shadow_enabled && state.shadow_depth) {
                const float* sm = state.shadow_mvp;
                float sx = sm[0]*ex + sm[4]*ey + sm[8]*ez  + sm[12];
                float sy = sm[1]*ex + sm[5]*ey + sm[9]*ez  + sm[13];
                float sz = sm[2]*ex + sm[6]*ey + sm[10]*ez + sm[14];
                float sw = sm[3]*ex + sm[7]*ey + sm[11]*ez + sm[15];
                if (fabsf(sw) > 1e-8f) {
                    float inv_sw = 1.0f / sw;
                    float su = sx * inv_sw * 0.5f + 0.5f;
                    float sv = sy * inv_sw * 0.5f + 0.5f;
                    float sd = sz * inv_sw * 0.5f + 0.5f;
                    int si = (int)(su * state.shadow_w);
                    int sj = (int)(sv * state.shadow_h);
                    if (si >= 0 && si < state.shadow_w &&
                        sj >= 0 && sj < state.shadow_h) {
                        float map_d = state.shadow_depth[sj * state.shadow_w + si];
                        if (sd > map_d + state.shadow_bias)
                            shadow_factor = 0.0f;
                    }
                }
            }

            for (int li = 0; li < MAX_LIGHTS; li++) {
                if (!state.lights[li].enabled) continue;

                float lx, ly, lz;
                float atten = 1.0f;

                if (state.lights[li].position[3] == 0.0f) {
                    lx = state.lights[li].position[0];
                    ly = state.lights[li].position[1];
                    lz = state.lights[li].position[2];
                    float ll = sqrtf(lx*lx + ly*ly + lz*lz);
                    if (ll > 1e-8f) { lx /= ll; ly /= ll; lz /= ll; }
                } else {
                    lx = state.lights[li].position[0] - ex;
                    ly = state.lights[li].position[1] - ey;
                    lz = state.lights[li].position[2] - ez;
                    float dist = sqrtf(lx*lx + ly*ly + lz*lz);
                    if (dist > 1e-8f) { lx /= dist; ly /= dist; lz /= dist; }
                    float denom = state.lights[li].constant_atten
                                + state.lights[li].linear_atten * dist
                                + state.lights[li].quadratic_atten * dist * dist;
                    atten = (denom > 1e-8f) ? (1.0f / denom) : 1.0f;
                }

                float spot_atten = 1.0f;
                if (state.lights[li].spot_cutoff < 180.0f && state.lights[li].position[3] != 0.0f) {
                    float sdx = state.lights[li].direction[0];
                    float sdy = state.lights[li].direction[1];
                    float sdz = state.lights[li].direction[2];
                    float sdlen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
                    if (sdlen > 1e-8f) { sdx /= sdlen; sdy /= sdlen; sdz /= sdlen; }
                    float cos_angle = -(lx*sdx + ly*sdy + lz*sdz);
                    float cos_cutoff = cosf(state.lights[li].spot_cutoff * 3.14159265f / 180.0f);
                    if (cos_angle < cos_cutoff) {
                        spot_atten = 0.0f;
                    } else {
                        spot_atten = powf(cos_angle, state.lights[li].spot_exponent);
                    }
                }
                atten *= spot_atten;

                float NdotL = nx*lx + ny*ly + nz*lz;
                if (NdotL < 0.0f) NdotL = 0.0f;

                float li_shadow = (li == 0) ? shadow_factor : 1.0f;

                lit_r += atten * (state.lights[li].ambient[0] * r
                               + state.lights[li].diffuse[0] * r * NdotL * li_shadow);
                lit_g += atten * (state.lights[li].ambient[1] * g
                               + state.lights[li].diffuse[1] * g * NdotL * li_shadow);
                lit_b += atten * (state.lights[li].ambient[2] * b
                               + state.lights[li].diffuse[2] * b * NdotL * li_shadow);

                if (NdotL > 0.0f && frag_mat_spec > 0.0f) {
                    float vdx = -ex, vdy = -ey, vdz = -ez;
                    float vlen = sqrtf(vdx*vdx + vdy*vdy + vdz*vdz);
                    if (vlen > 1e-8f) { vdx /= vlen; vdy /= vlen; vdz /= vlen; }
                    float hx = lx + vdx, hy = ly + vdy, hz = lz + vdz;
                    float hlen = sqrtf(hx*hx + hy*hy + hz*hz);
                    if (hlen > 1e-8f) { hx /= hlen; hy /= hlen; hz /= hlen; }

                    float NdotH = nx*hx + ny*hy + nz*hz;
                    if (NdotH < 0.0f) NdotH = 0.0f;
                    float spec_power = (frag_mat_shin > 0.0f) ? powf(NdotH, frag_mat_shin) : 1.0f;

                    frag_sr += atten * state.lights[li].specular[0] * frag_mat_spec * spec_power * li_shadow;
                    frag_sg += atten * state.lights[li].specular[1] * frag_mat_spec * spec_power * li_shadow;
                    frag_sb += atten * state.lights[li].specular[2] * frag_mat_spec * spec_power * li_shadow;
                }
            }

            final_r = clamp01(lit_r);
            final_g = clamp01(lit_g);
            final_b = clamp01(lit_b);
            final_a = a;
        } else {
            final_r = r;
            final_g = g;
            final_b = b;
            final_a = a;
            frag_sr = (w0*v0.spec_color[0] + w1*v1.spec_color[0] + w2*v2.spec_color[0]) * inv_sum_w;
            frag_sg = (w0*v0.spec_color[1] + w1*v1.spec_color[1] + w2*v2.spec_color[1]) * inv_sum_w;
            frag_sb = (w0*v0.spec_color[2] + w1*v1.spec_color[2] + w2*v2.spec_color[2]) * inv_sum_w;
        }

        frag_sr = clamp01(frag_sr);
        frag_sg = clamp01(frag_sg);
        frag_sb = clamp01(frag_sb);
        final_r += frag_sr;
        final_g += frag_sg;
        final_b += frag_sb;

        if (tri_texid >= 0) {
            float tex_r, tex_g, tex_b;
            sample_triangle_texture(tri,tri_texid,E0,E1,E2,oa_tc,
                pixel_idx,fb.width,fb.height,tex_state,tex_r,tex_g,tex_b);
            decode_texel(state, tex_r, tex_g, tex_b);
            final_r *= tex_r;
            final_g *= tex_g;
            final_b *= tex_b;
        }

        apply_linear_fog(state, oa_px, oa_py, oa_pz,
                         final_r, final_g, final_b);
        final_r = clamp01(final_r);
        final_g = clamp01(final_g);
        final_b = clamp01(final_b);
        final_a = clamp01(final_a);

        if (WITH_OIT && state.oit_enabled && fb.oit_accum && fb.oit_revealage) {
            float z = depth;
            float d_scaled = z * 200.0f;
            float w = final_a * fmaxf(0.01f, fminf(3000.0f,
                      0.03f / (1e-5f + d_scaled * d_scaled * d_scaled * d_scaled)));
            float pr = final_r * final_a;
            float pg = final_g * final_a;
            float pb = final_b * final_a;
            oit_accum_r += pr * w;
            oit_accum_g += pg * w;
            oit_accum_b += pb * w;
            oit_accum_a += final_a * w;
            oit_revealage *= (1.0f - final_a);
        } else if (!DIRECT && !SHADING_ONLY && state.blend_enabled) {
            float src_a = final_a;
            float inv_a = 1.0f - src_a;
            if (state.blend_src==BLEND_ONE && state.blend_dst==BLEND_ONE) {
                out_color0=clamp01(final_r+out_color0);
                out_color1=clamp01(final_g+out_color1);
                out_color2=clamp01(final_b+out_color2);
                out_color3=clamp01(final_a+out_color3);
            } else {
                out_color0 = final_r * src_a + out_color0 * inv_a;
                out_color1 = final_g * src_a + out_color1 * inv_a;
                out_color2 = final_b * src_a + out_color2 * inv_a;
                out_color3 = src_a + out_color3 * inv_a;
            }
        } else if (DIRECT) {
            // No blend, no OIT: the surviving colour is simply the LAST
            // depth-passing fragment's, so storing it here is identical to
            // holding it in registers and storing once at the end -- and it
            // frees four VGPRs per owned sample, which is what lets the
            // pixel-major mapping keep its occupancy.
            fb.color_buffer[color_idx + 0] = final_r;
            fb.color_buffer[color_idx + 1] = final_g;
            fb.color_buffer[color_idx + 2] = final_b;
            fb.color_buffer[color_idx + 3] = final_a;
        } else {
            out_color0 = final_r;
            out_color1 = final_g;
            out_color2 = final_b;
            out_color3 = final_a;
        }

        if (!NO_ATTR && fb.attr_buffer &&
            !(WITH_OIT && state.oit_enabled && fb.oit_accum && fb.oit_revealage)) {
            const int attr_base = pixel_idx * FRAG_ATTR_COUNT;
            float oa[FRAG_ATTR_COUNT] = {0.0f};
            oa[0] = oa_px; oa[1] = oa_py; oa[2] = oa_pz;
            oa[3] = oa_nx; oa[4] = oa_ny; oa[5] = oa_nz;
#pragma unroll
            for (int k = 0; k < TEXCOORD_COMPONENTS; k++)
                oa[FRAG_ATTR_TEXCOORD + k] = oa_tc[k];
            oa[FRAG_ATTR_COLOR + 0] = r; oa[FRAG_ATTR_COLOR + 1] = g;
            oa[FRAG_ATTR_COLOR + 2] = b; oa[FRAG_ATTR_COLOR + 3] = a;
            oa[FRAG_ATTR_VERTEX] = oa_vtx;
#pragma unroll
            for (int k = 0; k < FRAG_ATTR_COUNT; k++)
                fb.attr_buffer[attr_base + k] = oa[k];
        }

}

#include "msaa_rasterize.inc"

HIPGL_DEVICE inline bool hipgl_tile_edge_reject_f32(
    const float* __restrict__ ee, float tile_px0, float tile_py0)
{
#pragma unroll
    for (int e = 0; e < 3; e++) {
        const float ea = ee[e*3+0], eb = ee[e*3+1], ec = ee[e*3+2];
        const float eba = ea*tile_px0 + eb*tile_py0 + ec;
        const float hi  = eba + ((ea > 0.0f) ? ea * (float)(TILE_SIZE-1) : 0.0f)
                              + ((eb > 0.0f) ? eb * (float)(TILE_SIZE-1) : 0.0f);
        const float mag = fabsf(eba) + (float)(TILE_SIZE-1)*fabsf(ea)
                        + (float)(TILE_SIZE-1)*fabsf(eb) + 1.0f;
        if (hi < -(1e-3f * mag + 1e-30f)) return true;
    }
    return false;
}

HIPGL_DEVICE inline bool hipgl_tile_edge_reject(
    const TriangleSetup& tf, float tile_px0, float tile_py0, float* ecache)
{
    return hipgl_tile_edge_reject_s(tf, tile_px0, tile_py0, ecache);
}

struct HpFineChain {
    const TriangleSetup* tris;
    const int* tile_base;
    const int* tile_counts;
    const unsigned long long* pool_key;
    int pool_cap;
    int nsub;
    int dead_mask;
    int slot_stride;
    int ncsc;
};
#ifndef HP_FINE_MULTI_MAX
#define HP_FINE_MULTI_MAX 16
#endif
struct HpFineMulti { HpFineChain c[HP_FINE_MULTI_MAX]; };
static_assert(sizeof(HpFineMulti) + sizeof(RenderState) + sizeof(TextureState) +
              sizeof(DeviceFramebuffer) + 64 <= 3584, "multi-chain fine kernarg budget");

#ifndef HP_FINE_PACK
#define HP_FINE_PACK 4
#endif
#ifndef HP_FINE_PACK_SHIFT
#define HP_FINE_PACK_SHIFT 4
#endif

#ifndef HP_FINE_LIVE_SPANS
#define HP_FINE_LIVE_SPANS 1
#endif

template <bool USE_POOL, bool WITH_OIT, bool DIRECT, bool GOURAUD = false, bool NO_ATTR = false,
          bool ROWSKIP = false, bool PERSIST = false, bool PERMUTE = false,
          int FINE_TILE = TILE_SIZE, bool MULTISPAN = false, int XCDMODE = 0,
          bool HOIST = false, int ROWS = 1, int PFROWS = 1, int PACK = 1>
HIPGL_DEVICE __forceinline__ void hipgl_fine_body_t(
    const TriangleSetup* __restrict__ tris,
    const TileBin* __restrict__ tile_bins,
    const int* __restrict__ tile_base,
    const int* __restrict__ tile_counts,
    const unsigned long long* __restrict__ pool_key,
    int pool_cap,
    int tiles_x, int tiles_y,
    const DeviceFramebuffer& fb,
    const RenderState& state,
    const TextureState& tex_state,
    const int* __restrict__ work_list,
    const int* __restrict__ work_count,
    int* work_ticket,
    int nsub, int dead_mask, int slot_stride,
    const float4* __restrict__ flt_rec,
    const int* __restrict__ flt_cnt,
    const unsigned short* __restrict__ flt_msk,
    const int ncsc)
{
    static_assert(!HOIST || (USE_POOL && ROWSKIP),
                  "the hoisted record packs the row range into the key's high byte");
    static_assert(FINE_TILE == TILE_SIZE || FINE_TILE == 4 || FINE_TILE == 8);
    constexpr bool DEFER_OPAQUE = USE_POOL && !WITH_OIT && DIRECT && NO_ATTR && FINE_TILE < TILE_SIZE;
    static_assert(FINE_TILE == TILE_SIZE || (!PERSIST && !PERMUTE));
    constexpr bool STAGED = !PERSIST && (HIPGL_FINE_PIPE == 1 ||
                                         (HIPGL_FINE_PIPE == 2 && !DEFER_OPAQUE));
    constexpr int FINE_MULT = (FINE_TILE == TILE_SIZE || STAGED) ? 1
                            : (HOIST ? HIPGL_FINE_MULT_HOIST
                                     : (PFROWS > 1 ? HP_MICRO_PF_MULT : HIPGL_FINE_MULT));
    constexpr int FINE_PIX   = FINE_TILE * FINE_TILE;
    constexpr int FINE_NT    = FINE_PIX * ROWS * PFROWS;
    constexpr int FINE_CHUNK = FINE_TILE == TILE_SIZE ? HIPGL_FINE_CHUNK
                                                      : FINE_NT * FINE_MULT;
    constexpr int FINE_NWAVE = (FINE_NT + 63) / 64;
    constexpr int SUBDIV = TILE_SIZE / FINE_TILE;
    static_assert(ROWS == 1 || (DEFER_OPAQUE && FINE_NT <= 64),
                  "entry-parallel rows require the order-independent deferred "
                  "opaque winner latch and a single-wave CTA");
    static_assert(PFROWS == 1 || (ROWS == 1 && FINE_NT <= 64 && FINE_TILE < TILE_SIZE),
                  "prefilter-wide rows require a single-wave sub-tile CTA");

    constexpr bool WAVE_LOCAL = (FINE_NWAVE == 1) && !STAGED && (ROWS == 1);
    constexpr int  FLT_N = WAVE_LOCAL ? 1 : (FINE_CHUNK + ROWS);
    constexpr int  EE_N  = WAVE_LOCAL ? 1 : FINE_CHUNK;

    constexpr bool ZWCACHE = DEFER_OPAQUE && !WAVE_LOCAL &&
                             (HP_FINE_ZWCACHE == 1 ||
                              (HP_FINE_ZWCACHE == 2 && HOIST));

    __shared__ unsigned long long s_flt[FLT_N];
    // Everything the per-pixel reject path needs, cached beside the key:
    // [0..2] the three tile-corner edge bases (the ONLY place the FP64
    // expression is evaluated now -- once per entry instead of once per entry
    // per wave), [3..8] the six edge gradients, [9..11] the three edge biases.
    // A rejected entry therefore never touches the 484 B TriangleSetup at all.
    __shared__ float4 s_ee[EE_N][3];
    // The staged depth triple, at the SAME compacted slot as s_ee.  Sized [1]
    // and never referenced on the instantiations the gate excludes, so their
    // group_segment_fixed_size is unchanged (verified against a separately
    // compiled canonical clone).
    __shared__ float2 s_zw[ZWCACHE ? EE_N : 1][3];
    __shared__ int   s_wcnt[WAVE_LOCAL ? 1 : FINE_NWAVE];
    constexpr bool MASKED   = HOIST && (FINE_TILE < TILE_SIZE) && (FINE_NWAVE == 1);
    constexpr int  MASK_CAP = MASKED ? 1024 : 1;   // deeper segments fall back
    __shared__ unsigned short s_idx[MASK_CAP];

    const int zr = (ROWS > 1 || PFROWS > 1) ? (int)threadIdx.z : 0;
    const int prow = (ROWS > 1) ? zr : 0;
    const int tid = zr * FINE_PIX + (int)threadIdx.y * FINE_TILE + (int)threadIdx.x;

    __shared__ int s_ticket[1];
    int wl_deep = 0, wl_shal = 0;
    if constexpr (PERSIST) {
        wl_deep = work_count[0];
        wl_shal = work_count[1];
    }
    int wl_phase = 0;          // 0 = deep region, 1 = shallow region
    int wl_lo = 0, wl_hi = 0;  // current shallow grant window
    const int wl_ntiles = tiles_x * tiles_y;

    constexpr int PK_R = (PACK > 1) ? (PACK * 24 + 63) / 64 : 1;
    unsigned int pk_mask_lane = 0u;    // lane k: unit k's live-span mask
    unsigned long long pk_units = 0ull; // bit k: unit k has any live span
    bool pk_fallback = false;           // too many spans: walk every unit fully
    auto pk_tile_of = [&](int k) -> int {
        const int SD = SUBDIV * SUBDIV;
        const int TB = (tiles_x * tiles_y) / PACK;      // tiles per band
        const int raw = ((int)blockIdx.x) / SD;         // band-local tile
        return k * TB + (raw + k * HP_FINE_PACK_SHIFT) % TB;
    };
    if constexpr (PACK > 1) {
        static_assert(FINE_NT <= 64 && USE_POOL && !PERSIST && !PERMUTE && XCDMODE == 0,
                      "packed fine requires a single-wave pooled sub-tile CTA");
        const int nsp = MULTISPAN ? nsub : 1;
        if (nsp * PACK <= 64 * PK_R) {
            const int* __restrict__ cnt_src = HOIST ? flt_cnt : tile_counts;
            unsigned long long bal[PK_R];
            int cv[PK_R];
#pragma unroll
            for (int r = 0; r < PK_R; ++r) {
                const int idx = tid + 64 * r;
                cv[r] = 0;
                if (idx < nsp * PACK) {
                    const int k  = idx / nsp;
                    const int sd = idx - k * nsp;
                    const bool dead = MULTISPAN && ((dead_mask >> sd) & 1);
                    if (!dead) {
                        const int slot = (MULTISPAN ? sd * slot_stride : 0) + pk_tile_of(k);
                        cv[r] = cnt_src[slot];
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < PK_R; ++r) bal[r] = __ballot(cv[r] > 0);
            // lane k < PACK extracts bits [k*nsp, k*nsp + nsp) of the concatenation.
            if (tid < PACK) {
                const int lo = tid * nsp;
                const int r0 = lo >> 6, off = lo & 63;
                unsigned long long w0 = 0ull, w1 = 0ull;
#pragma unroll
                for (int r = 0; r < PK_R; ++r) {
                    if (r == r0)     w0 = bal[r];
                    if (r == r0 + 1) w1 = bal[r];
                }
                unsigned long long v = w0 >> off;
                if (off) v |= (w1 << (64 - off));
                const unsigned long long m = (nsp >= 64) ? ~0ull : ((1ull << nsp) - 1ull);
                pk_mask_lane = (unsigned int)(v & m);
            }
            pk_units = __ballot(pk_mask_lane != 0u);
        } else {
            pk_fallback = true;
            pk_units = (PACK >= 64) ? ~0ull : ((1ull << PACK) - 1ull);
        }
    }

    for (;;) {
    int bin_idx = 0;
    int pk_k = 0;
    if constexpr (PERSIST) {
        // DEEP region: one tile per ticket, so the long poles are handed out at
        // the finest granularity (that is where dynamic balancing pays).
        // SHALLOW region: HIPGL_PERSIST_GRANT tiles per ticket -- these are
        // short and there are tens of thousands of them, so a per-tile atomic
        // on a single counter would cost more than the tiles do.
        bool got = false;
        while (!got) {
            if (wl_phase == 0) {
                if (tid == 0) s_ticket[0] = atomicAdd(work_ticket + 0, 1);
                __syncthreads();
                const int i0 = s_ticket[0];
                __syncthreads();
                if (i0 < wl_deep) { bin_idx = work_list[i0]; got = true; }
                else wl_phase = 1;
            } else {
                if (wl_lo >= wl_hi) {
                    if (tid == 0)
                        s_ticket[0] = atomicAdd(work_ticket + 1, HIPGL_PERSIST_GRANT);
                    __syncthreads();
                    wl_lo = s_ticket[0];
                    __syncthreads();
                    wl_hi = wl_lo + HIPGL_PERSIST_GRANT;
                    if (wl_hi > wl_shal) wl_hi = wl_shal;
                    if (wl_lo >= wl_shal) break;
                }
                bin_idx = work_list[wl_ntiles - 1 - wl_lo];
                wl_lo++;
                got = true;
            }
        }
        if (!got) break;
    } else if constexpr (PERMUTE) {
        const int nt = tiles_x * tiles_y;      // launcher guarantees nt % 8 == 0
        bin_idx = (int)((blockIdx.x & 7u) * (unsigned)(nt >> 3) + (blockIdx.x >> 3));
    } else if constexpr (PACK > 1) {
        if (pk_units == 0ull) break;
        pk_k = __builtin_ctzll(pk_units);
        pk_units &= pk_units - 1ull;
        bin_idx = pk_tile_of(pk_k);
    } else {
        bin_idx = (int)blockIdx.x / (SUBDIV * SUBDIV);
    }
    int child;
    if constexpr (XCDMODE != 0) {
        const int band  = (int)(blockIdx.x & 7u);
        const int idx   = (int)(blockIdx.x >> 3);
        const int tslot = idx / (SUBDIV * SUBDIV);
        child = idx % (SUBDIV * SUBDIV);
        if constexpr (XCDMODE == 1) {
            const int nt = tiles_x * tiles_y;    // launcher guarantees nt % 8 == 0
            bin_idx = band * (nt >> 3) + tslot;
        } else {
            constexpr int G = HIPGL_FINE_XCD_CHUNK;
            bin_idx = (tslot / G) * (8 * G) + band * G + (tslot % G);
        }
    } else {
        child = (int)blockIdx.x % (SUBDIV * SUBDIV);
    }
    const int sub_x = (child % SUBDIV) * FINE_TILE;
    const int sub_y = (child / SUBDIV) * FINE_TILE;
    const int local_x = (int)threadIdx.x + sub_x;
    const int local_y = (int)threadIdx.y + sub_y;
    const int tile_x = bin_idx % tiles_x;
    const int tile_y = bin_idx / tiles_x;
    const int px = tile_x * TILE_SIZE + local_x;
    const int py = tile_y * TILE_SIZE + local_y;

    int* s_tri_indices = nullptr;
    float* s_tri_z_min = nullptr;
    if constexpr (!USE_POOL) {
        s_tri_indices = const_cast<int*>(tile_bins[bin_idx].triangle_indices);
        s_tri_z_min = const_cast<float*>(tile_bins[bin_idx].triangle_z_min);
    }

    const unsigned long long* __restrict__ key_src = nullptr;
    const float4* __restrict__ rec_src = nullptr;
    bool msk_use = false;
    int bin_count = 0;
    const int ns_spans = MULTISPAN ? nsub : 1;

    if constexpr (!USE_POOL) {
        const TileBin& bin = tile_bins[bin_idx];
        bin_count = (bin.count < MAX_TRIANGLES_PER_TILE) ? bin.count : MAX_TRIANGLES_PER_TILE;

        const int BLOCK_THREADS = TILE_SIZE * TILE_SIZE;  // 256
        int n_padded = 1;
        while (n_padded < bin_count) n_padded <<= 1;
        for (int i = bin_count + tid; i < n_padded; i += BLOCK_THREADS) {
            s_tri_z_min[i] = 1e30f;
            s_tri_indices[i] = -1;
        }
        __syncthreads();
        for (int k = 2; k <= n_padded; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = tid; i < (n_padded >> 1); i += BLOCK_THREADS) {
                    int l = (i / j) * (j << 1) + (i % j);
                    int r = l + j;
                    bool ascending = ((l & k) == 0);
                    bool gt = (s_tri_z_min[l] > s_tri_z_min[r]) ||
                              (s_tri_z_min[l] == s_tri_z_min[r] &&
                               s_tri_indices[l] > s_tri_indices[r]);
                    if (gt == ascending) {
                        float tz = s_tri_z_min[l]; s_tri_z_min[l] = s_tri_z_min[r]; s_tri_z_min[r] = tz;
                        int ti = s_tri_indices[l]; s_tri_indices[l] = s_tri_indices[r]; s_tri_indices[r] = ti;
                    }
                }
                __syncthreads();
            }
        }
    }

    const bool oob = (px >= fb.width || py >= fb.height) || (PFROWS > 1 && zr != 0);
    const int spx = oob ? 0 : px;
    const int spy = oob ? 0 : py;

    const int pixel_idx = spy * fb.width + spx;

    const int pitch = (fb.pitch > 0) ? fb.pitch : fb.width * 4;
    const int color_idx = spy * pitch + spx * 4;

    float best_depth = 0.0f;
    bool  have_depth = false;
    bool  mask_off   = false;
    float out_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float oit_accum_r = 0.0f, oit_accum_g = 0.0f, oit_accum_b = 0.0f, oit_accum_a = 0.0f;
    float oit_revealage = 1.0f;
    bool any_fragment = false;

    const float tile_px0 = (float)(tile_x * TILE_SIZE) + 0.5f;
    const float tile_py0 = (float)(tile_y * TILE_SIZE) + 0.5f;
    const float thr_x = (float)local_x;
    const float thr_y = (float)local_y;
    const int   thr_yi = local_y;

    int winner = -1;
    float winner_e0 = 0.f, winner_e1 = 0.f, winner_e2 = 0.f;
    int winner_seq = -1;
    int cur_seq = 0;
    int seq_base = 0;

    uint32_t best_ord = 0u;

    auto fetch_entry = [&](int j) -> unsigned long long {
        if constexpr (USE_POOL) {
            return key_src[j];
        } else {
            return ((unsigned long long)hipgl_f2ord(s_tri_z_min[j]) << 32)
                 | (unsigned long long)(uint32_t)s_tri_indices[j];
        }
    };

    auto prefilter = [&](int base, int n, unsigned long long& o_ent, float* o_ec,
                         int& o_yr, bool& o_keep) {
        o_ent = 0ull; o_yr = 0; o_keep = false;
        if constexpr (HOIST) {
            if (tid < n) {
                const int ei = msk_use ? (int)s_idx[base + tid] : (base + tid);
                const float4* __restrict__ r = rec_src + (size_t)ei * 4;
                const float4 q0 = r[0];
                const float4 q1 = r[1];
                const float4 q2 = r[2];
                const float4 q3 = r[3];
                o_ec[0]  = q0.x; o_ec[1]  = q0.y; o_ec[2]  = q0.z; o_ec[3]  = q0.w;
                o_ec[4]  = q1.x; o_ec[5]  = q1.y; o_ec[6]  = q1.z; o_ec[7]  = q1.w;
                o_ec[8]  = q2.x; o_ec[9]  = q2.y; o_ec[10] = q2.z; o_ec[11] = q2.w;
                o_ent = ((unsigned long long)__float_as_uint(q3.x) << 32)
                      | (unsigned long long)__float_as_uint(q3.y);
                o_keep = true;
                if constexpr (FINE_TILE < TILE_SIZE) {
                    if (!msk_use) {
#pragma unroll
                    for (int e=0; e<3; ++e) {
                        const float ec=o_ec[4*e], a=o_ec[4*e+1], b=o_ec[4*e+2], bias=o_ec[4*e+3];
                        const float hi=ec+a*sub_x+b*sub_y+fmaxf(a,0.f)*(FINE_TILE-1)
                            +fmaxf(b,0.f)*(FINE_TILE-1)+bias;
                        const float mag=fabsf(ec)+(fabsf(a)+fabsf(b))*TILE_SIZE+fabsf(bias)+1.f;
                        if (hi < -1.e-5f*mag) o_keep=false;
                    }
                    }
                }
            }
            return;
        }
        if (tid < n) {
            o_ent = fetch_entry(base + tid);
            const int physical_id = hipgl_physical_triangle_id((uint32_t)o_ent, ncsc);
            o_ent = (o_ent & 0xFFFFFFFF00000000ull) | (uint32_t)physical_id;
            const TriangleSetup& tf = tris[physical_id];
            if (!tf.degenerate && !hipgl_triangle_is_culled_s(tf, FinePreState{ncsc, state.backface_cull_enabled, state.front_face, state.cull_face})) {
                o_keep = true;
                const int tile_row0 = tile_y * TILE_SIZE;
                int ry0 = tf.bbox_min_y - tile_row0;
                int ry1 = tf.bbox_max_y - tile_row0;
                if (ry0 < 0) ry0 = 0;
                if (ry1 > TILE_SIZE - 1) ry1 = TILE_SIZE - 1;
                if (ROWSKIP && ry0 > ry1) o_keep = false;  // bbox misses this tile's rows
                // Packed into bits 24..31 of the entry's LOW word (tri ids are
                // < 2^24 here), so the walk gets the row range out of the key it
                // already loads -- no extra LDS array and no extra LDS read.
                o_yr = (ry0 << 28) | (ry1 << 24);
                if (hipgl_tile_edge_reject(tf, tile_px0, tile_py0, o_ec)) o_keep = false;
                if constexpr (FINE_TILE < TILE_SIZE) {
#pragma unroll
                    for (int e=0; e<3; ++e) {
                        const float ec=o_ec[4*e], a=o_ec[4*e+1], b=o_ec[4*e+2], bias=o_ec[4*e+3];
                        const float hi=ec+a*sub_x+b*sub_y+fmaxf(a,0.f)*(FINE_TILE-1)
                            +fmaxf(b,0.f)*(FINE_TILE-1)+bias;
                        const float mag=fabsf(ec)+(fabsf(a)+fabsf(b))*TILE_SIZE+fabsf(bias)+1.f;
                        if (hi < -1.e-5f*mag) o_keep=false;
                    }
                }
            }
        }
    };

    unsigned long long ent = 0ull;
    float ecache[12];
    int   yrcache = 0;
    bool  keep = false;
    auto compact_pass = [&](unsigned long long ent_, const float (&ec_)[12],
                            int yr_, bool keep_, int& nsurv) {
        __syncthreads();
        const unsigned long long bal = __ballot(keep_);
        const int lane = tid & 63;
        const int rank = (int)__popcll(bal & ((1ull << lane) - 1ull));
        int wbase = 0, ntot = 0;
        if constexpr (FINE_NWAVE == 1) {
            ntot = (int)__popcll(bal);
        } else {
            const int wv = tid >> 6;
            if (lane == 0) s_wcnt[wv] = (int)__popcll(bal);
            __syncthreads();
#pragma unroll
            for (int w = 0; w < FINE_NWAVE; w++) {
                if (w < wv) wbase += s_wcnt[w];
                ntot += s_wcnt[w];
            }
        }
        if (keep_) {
            const int slot = nsurv + wbase + rank;
            s_flt[slot] = ROWSKIP ? (ent_ | (unsigned long long)(unsigned)yr_) : ent_;
            s_ee[slot][0] = make_float4(ec_[0], ec_[4], ec_[8],  ec_[3]);
            s_ee[slot][1] = make_float4(ec_[1], ec_[5], ec_[9],  ec_[7]);
            s_ee[slot][2] = make_float4(ec_[2], ec_[6], ec_[10], ec_[11]);
            if constexpr (ZWCACHE) {
                // The producer already resolved the physical triangle id into
                // the entry's low word (HOIST), and the in-kernel prefilter
                // writes it there too, so this is the SAME index the walk
                // derives from the key -- no extra dependency, no extra load.
                const int tg_ = ROWSKIP ? (int)((uint32_t)ent_ & 0x00FFFFFFu)
                                        : (int)(uint32_t)ent_;
                const TransformedVertex* __restrict__ vv_ = tris[tg_].v;
                s_zw[slot][0] = make_float2(vv_[0].screen_z, vv_[0].inv_w);
                s_zw[slot][1] = make_float2(vv_[1].screen_z, vv_[1].inv_w);
                s_zw[slot][2] = make_float2(vv_[2].screen_z, vv_[2].inv_w);
            }
        }
        nsurv += ntot;
    };

    const bool early_out_armed = state.depth_test_enabled && HIPGL_EARLY_OUT_ENABLED &&
                                 (state.depth_func == DEPTH_LEQUAL ||
                                  ((PFROWS > 1) && state.depth_func == DEPTH_LESS &&
                                   state.tri_stats == nullptr));
    bool pixel_done = false;   // per-pixel early-out flag (depth-sorted), per span

    float zwc_z0 = 0.f, zwc_w0 = 0.f, zwc_z1 = 0.f, zwc_w1 = 0.f,
          zwc_z2 = 0.f, zwc_w2 = 0.f;
    auto walk_one = [&](const unsigned long long k_, auto&& get_edges) -> int {
        const uint32_t ent_ord = (uint32_t)(k_ >> 32);
        const int   tri_idx = ROWSKIP ? (int)((uint32_t)k_ & 0x00FFFFFFu)
                                     : (int)(uint32_t)k_;
        if (!pixel_done && early_out_armed && ent_ord > best_ord) {
            pixel_done = HIPGL_EARLY_OUT_ENABLED;
        }
        if (pixel_done) return 1;

        if constexpr (ROWSKIP) {
            const unsigned int ry0_ = (unsigned int)(k_ >> 28) & 0xFu;
            const unsigned int ry1_ = (unsigned int)(k_ >> 24) & 0xFu;
            if constexpr (FINE_TILE < TILE_SIZE) {
                if (!msk_use &&
                    ((int)ry1_ < sub_y || (int)ry0_ > sub_y + FINE_TILE - 1))
                    return 0;
            } else {
                if ((unsigned int)(thr_yi - (int)ry0_) > (ry1_ - ry0_)) return 0;
            }
        }


        float4 eA, eB, eC;
        get_edges(eA, eB, eC);

        float E0 = eA.x + eB.x*thr_x + eC.x*thr_y;
        float E1 = eA.y + eB.y*thr_x + eC.y*thr_y;
        float E2 = eA.z + eB.z*thr_x + eC.z*thr_y;

        const unsigned int sgn3 = __float_as_uint(E0 + eA.w)
                                | __float_as_uint(E1 + eB.w)
                                | __float_as_uint(E2 + eC.w);
        if (sgn3 & 0x80000000u) return 0;

        if constexpr (DEFER_OPAQUE) {
            // Opaque native pixels need only the last accepted fragment's
            // color. Keep coverage/depth diagnostics and reflection gates on
            // every candidate, then shade the winner once after traversal.
            const TriangleSetup& tri = tris[tri_idx];
            const TransformedVertex& v0 = tri.v[0];
            const TransformedVertex& v1 = tri.v[1];
            const TransformedVertex& v2 = tri.v[2];
            const float b0 = E1, b1 = E2, b2 = E0;
            const float inv_sum = hipgl_rcp_nr(b0 + b1 + b2);
            // ZWCACHE swaps the SOURCE of these six floats, never the
            // expression tree: the identical bit patterns reach the identical
            // FMA chains, so the image is bit-exact.
            const float lw0 = ZWCACHE ? zwc_w0 : v0.inv_w;
            const float lw1 = ZWCACHE ? zwc_w1 : v1.inv_w;
            const float lw2 = ZWCACHE ? zwc_w2 : v2.inv_w;
            const float lz0 = ZWCACHE ? zwc_z0 : v0.screen_z;
            const float lz1 = ZWCACHE ? zwc_z1 : v1.screen_z;
            const float lz2 = ZWCACHE ? zwc_z2 : v2.screen_z;
            const float w0 = b0 * lw0;
            const float w1 = b1 * lw1;
            const float w2 = b2 * lw2;
            const float sum_w = w0 + w1 + w2;
            if (fabsf(sum_w) < 1e-8f) return 0;
            const float inv_sum_w = hipgl_rcp_nr(sum_w);
            const float depth = (b0 * lz0 + b1 * lz1 + b2 * lz2) * inv_sum;
            if (state.clip_plane_enabled) {
                const float ex = (w0*v0.world_pos[0] + w1*v1.world_pos[0] + w2*v2.world_pos[0]) * inv_sum_w;
                const float ey = (w0*v0.world_pos[1] + w1*v1.world_pos[1] + w2*v2.world_pos[1]) * inv_sum_w;
                const float ez = (w0*v0.world_pos[2] + w1*v1.world_pos[2] + w2*v2.world_pos[2]) * inv_sum_w;
                if (state.clip_plane[0]*ex + state.clip_plane[1]*ey +
                    state.clip_plane[2]*ez - state.clip_plane[3] < 0.0f) return 0;
            }
            if (state.tri_stats)
                state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_COVERED] = 1;
            if (state.depth_test_enabled) {
                if (state.depth_func == DEPTH_LEQUAL) {
                    if (!(depth <= best_depth)) return 0;
                } else if (!depth_test_passes(state.depth_func, depth, best_depth)) {
                    return 0;
                }
            }
            if (state.tri_stats)
                state.tri_stats[tri_idx * TRI_STAT_COUNT + TRI_STAT_DEPTH_PASS] = 1;
            if (state.depth_write_enabled) {
                if constexpr (ROWS == 1) fb.depth_buffer[pixel_idx] = depth;
                best_depth = depth;
            }
            if constexpr (ROWS > 1) winner_seq = cur_seq;
            winner = tri_idx;
            winner_e0 = E0;
            winner_e1 = E1;
            winner_e2 = E2;
        } else {
            hipgl_fine_shade_sample<WITH_OIT, DIRECT, GOURAUD, NO_ATTR, false, true>(tris, tri_idx, E0, E1, E2, pixel_idx, color_idx,
                fb, state, tex_state, best_depth,
                out_color[0], out_color[1], out_color[2], out_color[3],
                oit_accum_r, oit_accum_g, oit_accum_b, oit_accum_a,
                oit_revealage, any_fragment);
        }
        best_ord = hipgl_f2ord(best_depth);
        return 0;
    };

    auto rl_u = [](unsigned int v, int e) -> unsigned int {
        return (unsigned int)__builtin_amdgcn_readlane((int)v, e);
    };
    auto rl_f = [&](float v, int e) -> float {
        return __uint_as_float(rl_u(__float_as_uint(v), e));
    };
    // Walk one prefilter pass's survivors straight out of the wave's registers,
    // in ascending lane order == the order the LDS compaction produced.
    auto walk_ballot = [&](unsigned long long bal, const unsigned long long ent_,
                           const float (&ec_)[12], int yr_) -> void {
        while (bal) {
            const int e = (int)__builtin_ctzll(bal);
            bal &= (bal - 1ull);
            unsigned long long k_ =
                ((unsigned long long)rl_u((unsigned int)(ent_ >> 32), e) << 32)
              | (unsigned long long)rl_u((unsigned int)ent_, e);
            if constexpr (ROWSKIP)
                k_ |= (unsigned long long)rl_u((unsigned int)yr_, e);
            if (walk_one(k_, [&](float4& a, float4& b, float4& c) {
                    a = make_float4(rl_f(ec_[0], e), rl_f(ec_[4], e),
                                    rl_f(ec_[8], e), rl_f(ec_[3], e));
                    b = make_float4(rl_f(ec_[1], e), rl_f(ec_[5], e),
                                    rl_f(ec_[9], e), rl_f(ec_[7], e));
                    c = make_float4(rl_f(ec_[2], e), rl_f(ec_[6], e),
                                    rl_f(ec_[10], e), rl_f(ec_[11], e)); })) break;
        }
    };

    unsigned long long live_spans = ~0ull;
    if constexpr (PACK > 1) {
        if (!pk_fallback)
            live_spans = MULTISPAN ? (unsigned long long)(unsigned int)
                                     __shfl((int)pk_mask_lane, pk_k, 64)
                                   : ~0ull;
    }
    if constexpr (PACK == 1 && MULTISPAN && USE_POOL && FINE_NT <= 64 && HP_FINE_LIVE_SPANS) {
        if (ns_spans <= FINE_NT) {
            const int* __restrict__ cnt_src = HOIST ? flt_cnt : tile_counts;
            int c_l = 0;
            if (tid < ns_spans) c_l = cnt_src[tid * slot_stride + bin_idx];
            live_spans = __ballot(c_l > 0);
        }
    }
    for (int sd = 0; sd < ns_spans; ++sd) {
    if constexpr (MULTISPAN) {
        if ((dead_mask >> sd) & 1) continue;   // bridged hole: binned, not drawn
        if (sd < 64 && !((live_spans >> sd) & 1ull)) continue;   // empty span
    }
    if constexpr (USE_POOL) {
        // Clamped segment bounds -- an undersized pool drops pairs (reported by
        // the overflow flag) instead of reading out of the allocation.
        const int slot = MULTISPAN ? (sd * slot_stride + bin_idx) : bin_idx;
        int b = tile_base[slot];
        if (b > pool_cap) b = pool_cap;
        if constexpr (HOIST) {
            // The producer applied exactly these clamps before compacting, and
            // survivors <= clamped segment length, so the compacted list always
            // fits inside this slot's own pool segment.
            rec_src   = flt_rec + (size_t)b * 4;
            bin_count = flt_cnt[slot];
            if constexpr (MASKED) {
                msk_use = (flt_msk != nullptr);
                if (msk_use) {
                    const unsigned short* __restrict__ ms = flt_msk + (size_t)b;
                    const int nraw = bin_count;
                    int ncov = 0;
                    for (int u0 = 0; u0 < nraw; u0 += 4 * FINE_NT) {
                        int cb[4];
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const int e = u0 + q * FINE_NT + tid;
                            cb[q] = (e < nraw) ? (int)(((unsigned int)ms[e] >> child) & 1u) : 0;
                        }
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            const unsigned long long bal = __ballot(cb[q] != 0);
                            const int rank = (int)__popcll(bal & ((1ull << tid) - 1ull));
                            if (cb[q] && (ncov + rank) < MASK_CAP)
                                s_idx[ncov + rank] =
                                    (unsigned short)(u0 + q * FINE_NT + tid);
                            ncov += (int)__popcll(bal);
                        }
                    }
                    if (ncov <= MASK_CAP) bin_count = ncov;
                    else                  msk_use = false;
                    __syncthreads();
                }
            }
        } else {
        int n = tile_counts[slot];
        int keep_n = (n < MAX_TRIANGLES_PER_TILE) ? n : MAX_TRIANGLES_PER_TILE;
        const int avail = pool_cap - b;
        if (keep_n > avail) keep_n = avail;
        if (keep_n < 0) keep_n = 0;
        key_src   = pool_key + b;
        bin_count = keep_n;
        }
    }
    if (bin_count <= 0) continue;
    if (!have_depth) {
        have_depth = true;
        mask_off   = (state.reflect_mask != nullptr) &&
                     (state.reflect_mask[pixel_idx] == 0);
        if (!mask_off) {
            best_depth = fb.depth_buffer[pixel_idx];
            best_ord   = hipgl_f2ord(best_depth);
            if constexpr (!DIRECT) {
                if (state.blend_enabled) {
                    out_color[0] = fb.color_buffer[color_idx + 0];
                    out_color[1] = fb.color_buffer[color_idx + 1];
                    out_color[2] = fb.color_buffer[color_idx + 2];
                    out_color[3] = fb.color_buffer[color_idx + 3];
                }
            }
        }
    }
    pixel_done = oob | mask_off;  // per-pixel early-out flag (depth-sorted), per span
    if (state.reflect_mask && __syncthreads_and(pixel_done ? 1 : 0)) continue;
    if constexpr (STAGED) {
        const int cn0 = (bin_count < FINE_CHUNK) ? bin_count : FINE_CHUNK;
        prefilter(0, cn0, ent, ecache, yrcache, keep);
    }
    for (int c0 = 0; c0 < bin_count; c0 += FINE_CHUNK) {
        int nsurv = 0;
        if constexpr (FINE_MULT > 1) {
            unsigned long long entM[FINE_MULT];
            float ecM[FINE_MULT][12];
            int   yrM[FINE_MULT];
            bool  keepM[FINE_MULT];
            // The FINE_MULT passes are mutually independent, so their
            // pool-key loads and their TriangleSetup gathers all issue before
            // the first of them is waited on -- that is the only place in this
            // kernel with more than one long-latency load available at a time.
#pragma unroll
            for (int m = 0; m < FINE_MULT; m++) {
                int cn = bin_count - (c0 + m * FINE_NT);
                if (cn > FINE_NT) cn = FINE_NT;
                if (cn < 0) cn = 0;
                prefilter(c0 + m * FINE_NT, cn, entM[m], ecM[m], yrM[m], keepM[m]);
            }
            if constexpr (WAVE_LOCAL) {
                // Every ballot is taken at a fully reconverged point, BEFORE
                // any lane can diverge in the walk, so no survivor is lost.
                unsigned long long balM[FINE_MULT];
#pragma unroll
                for (int m = 0; m < FINE_MULT; m++) balM[m] = __ballot(keepM[m]);
#pragma unroll
                for (int m = 0; m < FINE_MULT; m++)
                    walk_ballot(balM[m], entM[m], ecM[m], yrM[m]);
            } else {
#pragma unroll
            for (int m = 0; m < FINE_MULT; m++)
                compact_pass(entM[m], ecM[m], yrM[m], keepM[m], nsurv);
            }
        } else if constexpr (!STAGED) {
            unsigned long long ent0; float ec0[12]; int yr0; bool keep0;
            int cn = bin_count - c0;
            if (cn > FINE_NT) cn = FINE_NT;
            if (cn < 0) cn = 0;
            prefilter(c0, cn, ent0, ec0, yr0, keep0);
            if constexpr (WAVE_LOCAL) {
                walk_ballot(__ballot(keep0), ent0, ec0, yr0);
            } else {
                compact_pass(ent0, ec0, yr0, keep0, nsurv);
            }
        } else {
            compact_pass(ent, ecache, yrcache, keep, nsurv);
        }
        __syncthreads();

        // Chunk c+1's loads are issued HERE, before the walk consumes chunk c,
        // so the gather latency overlaps the walk instead of stalling behind a
        // barrier. When the tile is a single chunk this is a fully-false
        // `tid < n` region the hardware skips with one s_cbranch_execz.
        unsigned long long nent = 0ull; float necache[12]; int nyr = 0; bool nkeep = false;
        if constexpr (STAGED) {
            const int nb = c0 + FINE_CHUNK;
            int ncn = bin_count - nb;
            if (ncn > FINE_CHUNK) ncn = FINE_CHUNK;
            prefilter(nb, ncn, nent, necache, nyr, nkeep);
        }

    if constexpr (!WAVE_LOCAL) {
    unsigned long long k_next = s_flt[prow];
    for (int i = prow; i < nsurv; i += ROWS) {
        const unsigned long long k_ = k_next;
        k_next = s_flt[i + ROWS];
        if constexpr (ZWCACHE) {
            // Issued beside the rotated key read, i.e. BEFORE the row test and
            // the edge test, so the (short) LDS latency is already retired by
            // the time the sign test would have issued the global load.
            const float2 a0 = s_zw[i][0], a1 = s_zw[i][1], a2 = s_zw[i][2];
            zwc_z0 = a0.x; zwc_w0 = a0.y;
            zwc_z1 = a1.x; zwc_w1 = a1.y;
            zwc_z2 = a2.x; zwc_w2 = a2.y;
        }
        if constexpr (ROWS > 1) cur_seq = seq_base + i;
        if (walk_one(k_, [&](float4& a, float4& b, float4& c) {
                a = s_ee[i][0]; b = s_ee[i][1]; c = s_ee[i][2]; })) break;
    }
    }
    if constexpr (ROWS > 1) seq_base += nsurv;
        // Rotate the staged prefilter results into place for the next chunk.
        if constexpr (STAGED) {
            ent = nent; yrcache = nyr; keep = nkeep;
#pragma unroll
            for (int e = 0; e < 12; e++) ecache[e] = necache[e];
        }

        // Block-level exit: once EVERY pixel of the tile is latched there is no
        // reason to filter another chunk. This is also the barrier that closes
        // the chunk, so it must be reached by all threads -- which is why the
        // out-of-frame lanes above start latched instead of returning.
        if (__syncthreads_and(pixel_done ? 1 : 0)) break;
    }
    }

    if constexpr (DEFER_OPAQUE) {
        if constexpr (ROWS > 1) {
            const bool dwr = state.depth_write_enabled;
#pragma unroll
            for (int off = FINE_PIX; off < FINE_PIX * ROWS; off <<= 1) {
                const float od = __shfl_xor(best_depth, off, 64);
                const int   ow = __shfl_xor(winner,     off, 64);
                const int   os = __shfl_xor(winner_seq, off, 64);
                const float o0 = __shfl_xor(winner_e0,  off, 64);
                const float o1 = __shfl_xor(winner_e1,  off, 64);
                const float o2 = __shfl_xor(winner_e2,  off, 64);
                const bool bet = dwr ? ((od < best_depth) ||
                                        (od == best_depth && os > winner_seq))
                                     : (os > winner_seq);
                if (ow >= 0 && (winner < 0 || bet)) {
                    winner = ow; winner_seq = os; best_depth = od;
                    winner_e0 = o0; winner_e1 = o1; winner_e2 = o2;
                }
            }
            // The one deferred depth store, by one row only.
            if (prow == 0 && winner >= 0 && state.depth_write_enabled)
                fb.depth_buffer[pixel_idx] = best_depth;
        }
        if (winner >= 0 && (ROWS == 1 || prow == 0)) {
            hipgl_fine_shade_sample<false, true, GOURAUD, true, true>(tris, winner,
                winner_e0, winner_e1, winner_e2, pixel_idx, color_idx,
                fb, state, tex_state, best_depth,
                out_color[0], out_color[1], out_color[2], out_color[3],
                oit_accum_r, oit_accum_g, oit_accum_b, oit_accum_a,
                oit_revealage, any_fragment);
        }
    }

    if constexpr (!DIRECT) {
    if (oob) any_fragment = false;

    if (WITH_OIT && state.oit_enabled && fb.oit_accum && fb.oit_revealage && any_fragment) {
        int accum_idx = pixel_idx * 4;
        fb.oit_accum[accum_idx + 0] += oit_accum_r;
        fb.oit_accum[accum_idx + 1] += oit_accum_g;
        fb.oit_accum[accum_idx + 2] += oit_accum_b;
        fb.oit_accum[accum_idx + 3] += oit_accum_a;
        fb.oit_revealage[pixel_idx] *= oit_revealage;
    } else if (any_fragment) {
        fb.color_buffer[color_idx + 0] = out_color[0];
        fb.color_buffer[color_idx + 1] = out_color[1];
        fb.color_buffer[color_idx + 2] = out_color[2];
        fb.color_buffer[color_idx + 3] = out_color[3];

    }
    }
    if constexpr (!PERSIST && PACK == 1) break;
    __syncthreads();
    }   // persistent worklist loop
}

template <bool USE_POOL, bool WITH_OIT, bool DIRECT, bool GOURAUD = false, bool NO_ATTR = false,
          bool ROWSKIP = false, bool PERSIST = false, bool PERMUTE = false,
          int FINE_TILE = TILE_SIZE, bool MULTISPAN = false, int XCDMODE = 0,
          bool HOIST = false, int ROWS = 1, int PFROWS = 1, int PACK = 1>
HIPGL_GLOBAL void __launch_bounds__(FINE_TILE * FINE_TILE * ROWS * PFROWS)
__attribute__((amdgpu_waves_per_eu(
    PERSIST ? HIPGL_FINE_WPEU_PERSIST
            : (ROWS > 1 ? HP_FINE_ROWS_WPEU
                        : ((!DIRECT && !GOURAUD) ? HP_MICRO_WPEU_GPHONG
                        : ((!NO_ATTR || !DIRECT) ? HIPGL_FINE_WPEU_FULLATTR
                                                 : HIPGL_FINE_WPEU_NP))),
    PERSIST ? HIPGL_FINE_WPEU_PERSIST
            : (ROWS > 1 ? HP_FINE_ROWS_WPEU_MAX
                        : ((!DIRECT && !GOURAUD) ? HP_MICRO_WPEU_GPHONG
                        : ((!NO_ATTR || !DIRECT) ? HIPGL_FINE_WPEU_FULLATTR
                                                 : HIPGL_FINE_WPEU_NPMAX))))))
kernel_fine_rasterize_with_sort_t(
    const TriangleSetup* __restrict__ tris,
    const TileBin* __restrict__ tile_bins,
    const int* __restrict__ tile_base,
    const int* __restrict__ tile_counts,
    const unsigned long long* __restrict__ pool_key,
    int pool_cap,
    int tiles_x, int tiles_y,
    DeviceFramebuffer fb,
    RenderState state,
    TextureState tex_state,
    const int* __restrict__ work_list,
    const int* __restrict__ work_count,
    int* work_ticket,
    int nsub, int dead_mask, int slot_stride,
    const float4* __restrict__ flt_rec = nullptr,
    const int* __restrict__ flt_cnt = nullptr,
    const unsigned short* __restrict__ flt_msk = nullptr)
{
    hipgl_fine_body_t<USE_POOL, WITH_OIT, DIRECT, GOURAUD, NO_ATTR, ROWSKIP, PERSIST, PERMUTE, FINE_TILE, MULTISPAN, XCDMODE, HOIST, ROWS, PFROWS, PACK>(
        tris, tile_bins, tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y,
        fb, state, tex_state, work_list, work_count, work_ticket, nsub, dead_mask, slot_stride,
        flt_rec, flt_cnt, flt_msk, state.near_clip_source_count);
}

template <bool USE_POOL, bool WITH_OIT, bool DIRECT, bool GOURAUD, bool NO_ATTR,
          bool ROWSKIP, bool PERSIST, bool PERMUTE,
          int FINE_TILE, bool MULTISPAN, int XCDMODE,
          bool HOIST, int ROWS, int PFROWS, int PACK>
HIPGL_GLOBAL void __launch_bounds__(FINE_TILE * FINE_TILE * ROWS * PFROWS)
__attribute__((amdgpu_waves_per_eu(
    PERSIST ? HIPGL_FINE_WPEU_PERSIST
            : (ROWS > 1 ? HP_FINE_ROWS_WPEU
                        : ((!DIRECT && !GOURAUD) ? HP_MICRO_WPEU_GPHONG
                        : ((!NO_ATTR || !DIRECT) ? HIPGL_FINE_WPEU_FULLATTR
                                                 : HIPGL_FINE_WPEU_NP))),
    PERSIST ? HIPGL_FINE_WPEU_PERSIST
            : (ROWS > 1 ? HP_FINE_ROWS_WPEU_MAX
                        : ((!DIRECT && !GOURAUD) ? HP_MICRO_WPEU_GPHONG
                        : ((!NO_ATTR || !DIRECT) ? HIPGL_FINE_WPEU_FULLATTR
                                                 : HIPGL_FINE_WPEU_NPMAX))))))
kernel_fine_multi_t(
    HpFineMulti m, int n, int tiles_x, int tiles_y,
    DeviceFramebuffer fb, RenderState state, TextureState tex_state)
{
    for (int c = 0; c < n; ++c) {
        const HpFineChain& h = m.c[c];
        hipgl_fine_body_t<USE_POOL, WITH_OIT, DIRECT, GOURAUD, NO_ATTR, ROWSKIP, PERSIST, PERMUTE, FINE_TILE, MULTISPAN, XCDMODE, HOIST, ROWS, PFROWS, PACK>(
            h.tris, (const TileBin*)nullptr, h.tile_base, h.tile_counts, h.pool_key, h.pool_cap,
            tiles_x, tiles_y, fb, state, tex_state, (const int*)nullptr, (const int*)nullptr,
            (int*)nullptr, h.nsub, h.dead_mask, h.slot_stride,
            (const float4*)nullptr, (const int*)nullptr, (const unsigned short*)nullptr, h.ncsc);
        __syncthreads();
    }
}

struct HpFineBatch {
    bool active = false;
    int n = 0;
    void (*flush_fn)(HpFineBatch&) = nullptr;
    hipStream_t stream = nullptr;
    int grid = 0, tiles_x = 0, tiles_y = 0;
    DeviceFramebuffer fb{};
    RenderState state{};
    TextureState tex{};
    HpFineMulti m{};
};
static thread_local HpFineBatch g_hp_fine_batch;
static inline hipStream_t hp_fine_batch_break(hipStream_t s) {
    HpFineBatch& b = g_hp_fine_batch;
    if (b.n > 0) b.flush_fn(b);
    return s;
}


HIPGL_GLOBAL void __launch_bounds__(256) kernel_build_tile_worklist(
    const int* __restrict__ counts, int num_tiles, int deep_thresh,
    int* __restrict__ work_list, int* __restrict__ work_count)
{
    const int t = (int)(blockIdx.x * 256 + threadIdx.x);
    const int n = (t < num_tiles) ? counts[t] : 0;
    const bool nonempty = (t < num_tiles) && (n > 0);
    const bool deep     = nonempty && (n >= deep_thresh);
    const bool shallow  = nonempty && !deep;
    const int lane = (int)(threadIdx.x & 63u);
    {
        const unsigned long long b = __ballot(deep);
        int base = 0;
        if (lane == 0) base = atomicAdd(work_count + 0, (int)__popcll(b));
        base = __shfl(base, 0, 64);
        if (deep) work_list[base + (int)__popcll(b & ((1ull << lane) - 1ull))] = t;
    }
    {
        const unsigned long long b = __ballot(shallow);
        int base = 0;
        if (lane == 0) base = atomicAdd(work_count + 1, (int)__popcll(b));
        base = __shfl(base, 0, 64);
        if (shallow)
            work_list[num_tiles - 1 - (base + (int)__popcll(b & ((1ull << lane) - 1ull)))] = t;
    }
}

void launch_build_tile_worklist(
    const int* counts, int num_tiles, int deep_thresh,
    int* work_list, int* work_count, hipStream_t stream)
{
    if (!counts || !work_list || !work_count || num_tiles <= 0) return;
    const int block = 256;
    const int grid = (num_tiles + block - 1) / block;
    hipLaunchKernelGGL(kernel_build_tile_worklist, grid, block, 0, stream,
        counts, num_tiles, deep_thresh, work_list, work_count);
}

// ── Host launch wrappers ────────────────────────────────────────────────────

void launch_clear_tile_bin_counts(
    TileBin* tile_bins, int num_tiles, hipStream_t stream)
{
    if (!tile_bins || num_tiles <= 0) return;
    const int block = 256;
    const int grid = (num_tiles + block - 1) / block;
    hipLaunchKernelGGL(kernel_clear_tile_bin_counts, grid, block, 0, stream,
        tile_bins, num_tiles);
}

void launch_bin_triangles(
    const TriangleSetup* tris, int tri_count,
    TileBin* tile_bins, int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    hipStream_t stream, int32_t* tri_stats, NearClipQueue near_clip)
{
    if (!tris || !tile_bins || tri_count <= 0 || tiles_x <= 0 || tiles_y <= 0) return;
    const int block = 256;
    const int grid = (tri_count + block - 1) / block;
    const bool packed_ids = near_clip.ids && near_clip.count;
    hipLaunchKernelGGL(kernel_bin_triangles, grid, block, 0, stream,
        tris, tri_count, tile_bins, tiles_x, tiles_y, fb_width, fb_height,
        tri_stats, packed_ids);
    if (packed_ids)
        hipLaunchKernelGGL(kernel_bin_near_clip_extras, 32, 128, 0, stream,
            tris, tri_count, near_clip, tile_bins, tiles_x, tiles_y,
            fb_width, fb_height, tri_stats);
}

void launch_sort_tile_bins(
    TileBin* tile_bins, int num_tiles, hipStream_t stream)
{
    if (!tile_bins || num_tiles <= 0) return;
    const int block = 256;
    const int grid = (num_tiles + block - 1) / block;
    hipLaunchKernelGGL(kernel_sort_tile_bins, grid, block, 0, stream,
        tile_bins, num_tiles);
}

void launch_fine_rasterize(
    const TriangleSetup* tris, const TileBin* tile_bins,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state, hipStream_t stream)
{
    if (!tris || !tile_bins || !fb.color_buffer || !fb.depth_buffer) return;
    if (tiles_x <= 0 || tiles_y <= 0 || fb.width <= 0 || fb.height <= 0) return;

    dim3 block(TILE_SIZE, TILE_SIZE);
    int num_tiles = tiles_x * tiles_y;
    hipLaunchKernelGGL(kernel_fine_rasterize, num_tiles, block, 0, hp_fine_batch_break(stream),
        tris, tile_bins, tiles_x, tiles_y, fb, state, texture_state);
}

#ifndef HP_MICRO_OIT_SPLIT
#define HP_MICRO_OIT_SPLIT 1
#endif

template <bool OIT>
static void hipgl_launch_fine_pool_general_t(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

static void hipgl_launch_fine_pool_general(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    const bool oit_live = HP_MICRO_OIT_SPLIT
                        ? (state.oit_enabled && fb.oit_accum && fb.oit_revealage) : true;
    if (oit_live)
        hipgl_launch_fine_pool_general_t<true>(nblk, block, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
    else
        hipgl_launch_fine_pool_general_t<false>(nblk, block, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
}

static void hipgl_launch_fine_pool_opaque(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, false>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

// Same, with the per-fragment lighting path removed at compile time.
static void hipgl_launch_fine_pool_opaque_gouraud(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, false>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

// The shipped configuration: pooled, opaque, Gouraud, no diagnostic dump.
static void hipgl_launch_fine_pool_opaque_gouraud_noattr(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, false>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}


template <bool MS, int XCD, bool HOIST>
static void hipgl_launch_fine_rowskip_sub4_t(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride,
    const float4* flt_rec, const int* flt_cnt, const unsigned short* flt_msk)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true,
                                                          false, false, 4, MS, XCD, HOIST,
                                                          HP_FINE_ROWS>),
        nblk * 16, dim3(4, 4, HP_FINE_ROWS), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
}

template <bool MS, int XCD>
static void hipgl_launch_fine_rowskip_sub4_h(
    bool hoist, int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride,
    const float4* flt_rec, const int* flt_cnt, const unsigned short* flt_msk)
{
    if (hoist)
        hipgl_launch_fine_rowskip_sub4_t<MS, XCD, true>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
    else
        hipgl_launch_fine_rowskip_sub4_t<MS, XCD, false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride, nullptr, nullptr, nullptr);
}

template <bool MS>
static void hipgl_launch_fine_rowskip_sub4_x(
    int xmode, bool hoist, int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride,
    const float4* flt_rec, const int* flt_cnt, const unsigned short* flt_msk)
{
    if (xmode == 1)
        hipgl_launch_fine_rowskip_sub4_h<MS, 1>(hoist, nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
    else if (xmode == 2)
        hipgl_launch_fine_rowskip_sub4_h<MS, 2>(hoist, nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
    else
        hipgl_launch_fine_rowskip_sub4_h<MS, 0>(hoist, nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
}

static void hipgl_launch_fine_rowskip_sub8(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true, false, false, 8>),
        nblk * 4, dim3(8,8), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}



static void hipgl_launch_fine_rowskip_full(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}




static void hipgl_launch_fine_rowskip_sub8_ms(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true, false, false, 8, true>),
        nblk * 4, dim3(8,8), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, nsub, dead_mask, slot_stride);
}

static void hipgl_launch_fine_rowskip_full_ms(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true, false, false, TILE_SIZE, true>),
        nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, nsub, dead_mask, slot_stride);
}

static void hipgl_launch_fine_pool_opaque_gouraud_noattr_rowskip(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride,
    const float4* flt_rec, const int* flt_cnt, const unsigned short* flt_msk)
{
    // Dense native grids benefit from short, spatially local walks. Preserve
    // the existing scheduling and shader on larger grids; zero is an explicit
    // process-level ablation for native 4x4 visibility/deferred shading.
    static const int defer_on = [] {
        const char* value = getenv("HIPGL_NATIVE1_DEFER");
        return value ? atoi(value) : 1;
    }();
    static const int fine_sub = [] {
        const char* value = getenv("HIPGL_FINE_SUB");
        return value ? atoi(value) : 4;
    }();
    static const int defer_max = [] {
        const char* value = getenv("HIPGL_NATIVE1_DEFER_MAX");
        return value ? atoi(value) : 4096;
    }();
    static const int xcd_on = [] {
        const char* value = getenv("HIPGL_FINE_XCD");
        return value ? atoi(value) : 1;
    }();
    static const int xcd_band_max = [] {
        const char* value = getenv("HIPGL_FINE_XCD_BAND_MAX_E");
        return value ? atoi(value) : HIPGL_FINE_XCD_BAND_MAX;
    }();
    static const int xcd_hi = [] {
        const char* value = getenv("HIPGL_FINE_XCD_HI_E");
        return value ? atoi(value) : HIPGL_FINE_XCD_HI;
    }();
    const bool ms = (nsub > 1);
    if (defer_on && nblk <= defer_max && fine_sub == 8) {
        if (ms)
            hipgl_launch_fine_rowskip_sub8_ms(nblk, stream, tris, tile_base, tile_counts,
                pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                nsub, dead_mask, slot_stride);
        else
            hipgl_launch_fine_rowskip_sub8(nblk, stream, tris, tile_base, tile_counts,
                pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
    } else if (defer_on && nblk <= defer_max && fine_sub == 4) {
        const int nt = tiles_x * tiles_y;
        int xmode = 0;
        if (xcd_on && nblk == nt) {
            if (nt <= xcd_band_max) xmode = (nt % 8 == 0) ? 1 : 0;
            else if (xcd_hi == 1)   xmode = (nt % 8 == 0) ? 1 : 0;
            else if (xcd_hi == 2)   xmode = (nt % (8 * HIPGL_FINE_XCD_CHUNK) == 0) ? 2 : 0;
        }
        const bool hoist = (flt_rec != nullptr && flt_cnt != nullptr);
        if (ms)
            hipgl_launch_fine_rowskip_sub4_x<true>(xmode, hoist, nblk, stream, tris,
                tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state,
                texture_state, nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
        else
            hipgl_launch_fine_rowskip_sub4_x<false>(xmode, hoist, nblk, stream, tris,
                tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state,
                texture_state, 1, 0, 0, flt_rec, flt_cnt, flt_msk);
    } else {
        if (ms)
            hipgl_launch_fine_rowskip_full_ms(nblk, block, stream, tris, tile_base, tile_counts,
                pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                nsub, dead_mask, slot_stride);
        else
            hipgl_launch_fine_rowskip_full(nblk, block, stream, tris, tile_base, tile_counts,
                pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
    }
}

// Persistent-worklist variants of the two SHIPPED instantiations.  Same kernel
// body; the grid is a resident ceiling and the tile id comes from the queue.
static void hipgl_launch_fine_pool_opaque_gouraud_noattr_persist(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    const int* work_list, const int* work_count, int* work_ticket)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, false, true>),
        nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, work_list, work_count, work_ticket, 1, 0, 0);
}

static void hipgl_launch_fine_pool_opaque_gouraud_noattr_permute(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    const int* work_list)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, false, false, true>),
        nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, work_list, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

static void hipgl_launch_fine_pool_opaque_gouraud_noattr_rowskip_persist(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    const int* work_list, const int* work_count, int* work_ticket)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, true, true, true>),
        nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, work_list, work_count, work_ticket, 1, 0, 0);
}

#ifndef HP_MICRO_FT_OPAQUE
#define HP_MICRO_FT_OPAQUE 4
#endif
#ifndef HP_MICRO_FT_GENERAL
#define HP_MICRO_FT_GENERAL 4
#endif
#ifndef HP_MICRO_ROWSKIP_OPAQUE
#define HP_MICRO_ROWSKIP_OPAQUE false
#endif
#ifndef HP_MICRO_ROWSKIP_GENERAL
#define HP_MICRO_ROWSKIP_GENERAL false
#endif
#ifndef HP_MICRO_NOATTR_OPAQUE
#define HP_MICRO_NOATTR_OPAQUE 1
#endif
#ifndef HP_MICRO_NOATTR_GENERAL
#define HP_MICRO_NOATTR_GENERAL 1
#endif

static inline int hp_fine_pack_on() {
    static const int v = []{ const char* e = getenv("HP_FINE_PACK_ON"); return e ? atoi(e) : 1; }();
    return v;
}
template <bool OIT, bool NOATTR, bool MS, int FTG, int PFR, int PK>
static void hp_fine_batch_flush(HpFineBatch& b) {
    const int n = b.n;
    b.n = 0;
    if (n == 1) {
        const HpFineChain& h = b.m.c[0];
        RenderState st = b.state;
        st.near_clip_source_count = h.ncsc;
        hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false, true, NOATTR,
                                                             HP_MICRO_ROWSKIP_GENERAL, false, false, FTG,
                                                             MS, 0, false, 1, PFR, PK>),
            b.grid, dim3(FTG, FTG, PFR), 0, b.stream,
            h.tris, (const TileBin*)nullptr, h.tile_base, h.tile_counts, h.pool_key, h.pool_cap,
            b.tiles_x, b.tiles_y, b.fb, st, b.tex, (const int*)nullptr, (const int*)nullptr,
            (int*)nullptr, h.nsub, h.dead_mask, h.slot_stride);
        return;
    }
    hipLaunchKernelGGL((kernel_fine_multi_t<true, OIT, false, true, NOATTR,
                                            HP_MICRO_ROWSKIP_GENERAL, false, false, FTG,
                                            MS, 0, false, 1, PFR, PK>),
        b.grid, dim3(FTG, FTG, PFR), 0, b.stream,
        b.m, n, b.tiles_x, b.tiles_y, b.fb, b.state, b.tex);
}
static inline int hp_fine_batch_max() {
    static const int v = []{
        const char* e = getenv("HP_CQ_FB");
        int x = e ? atoi(e) : HP_FINE_MULTI_MAX;
        return x < 0 ? 0 : (x > HP_FINE_MULTI_MAX ? HP_FINE_MULTI_MAX : x);
    }();
    return v;
}
// Field-wise (padding-blind) equality: struct copies do not preserve padding.
#define HP_FEQ(f) (memcmp(&a.f, &b.f, sizeof(a.f)) == 0)
static inline bool hp_state_eq_but_ncsc(const RenderState& a, const RenderState& b) {
    return HP_FEQ(depth_test_enabled) &&
           HP_FEQ(depth_write_enabled) &&
           HP_FEQ(blend_enabled) &&
           HP_FEQ(lighting_enabled) &&
           HP_FEQ(backface_cull_enabled) &&
           HP_FEQ(texture_2d_enabled) &&
           HP_FEQ(color_material_enabled) &&
           HP_FEQ(depth_func) &&
           HP_FEQ(blend_src) &&
           HP_FEQ(blend_dst) &&
           HP_FEQ(clear_color) &&
           HP_FEQ(clear_depth) &&
           HP_FEQ(clear_stencil) &&
           HP_FEQ(gradient_bg) &&
           HP_FEQ(gradient_top) &&
           HP_FEQ(gradient_bot) &&
           HP_FEQ(viewport_x) &&
           HP_FEQ(viewport_y) &&
           HP_FEQ(viewport_w) &&
           HP_FEQ(viewport_h) &&
           HP_FEQ(model_view) &&
           HP_FEQ(projection) &&
           HP_FEQ(mvp) &&
           HP_FEQ(normal_matrix) &&
           HP_FEQ(clip_plane_enabled) &&
           HP_FEQ(clip_plane) &&
           HP_FEQ(reflect_mask) &&
           HP_FEQ(reflect_dim) &&
           HP_FEQ(lights) &&
           HP_FEQ(material) &&
           HP_FEQ(fog_start) &&
           HP_FEQ(fog_end) &&
           HP_FEQ(fog_color) &&
           HP_FEQ(camera_world_pos) &&
           HP_FEQ(camera_forward) &&
           HP_FEQ(bound_texture_id) &&
           HP_FEQ(point_size) &&
           HP_FEQ(line_width) &&
           HP_FEQ(shade_model) &&
           HP_FEQ(polygon_mode) &&
           HP_FEQ(front_face) &&
           HP_FEQ(cull_face) &&
           HP_FEQ(oit_enabled) &&
           HP_FEQ(two_sided_lighting) &&
           HP_FEQ(gouraud_shading) &&
           HP_FEQ(shadow_enabled) &&
           HP_FEQ(shadow_mvp) &&
           HP_FEQ(shadow_depth) &&
           HP_FEQ(shadow_w) &&
           HP_FEQ(shadow_h) &&
           HP_FEQ(shadow_bias) &&
           HP_FEQ(tri_stats) &&
           HP_FEQ(reflect_rect);
}
static inline bool hp_fb_eq(const DeviceFramebuffer& a, const DeviceFramebuffer& b) {
    return HP_FEQ(color_buffer) &&
           HP_FEQ(depth_buffer) &&
           HP_FEQ(stencil_buffer) &&
           HP_FEQ(attr_buffer) &&
           HP_FEQ(oit_accum) &&
           HP_FEQ(oit_revealage) &&
           HP_FEQ(width) &&
           HP_FEQ(height) &&
           HP_FEQ(pitch) &&
           HP_FEQ(sample_color) &&
           HP_FEQ(sample_depth);
}
static inline bool hp_tex_eq(const TextureState& a, const TextureState& b) {
    return HP_FEQ(tex_data) &&
           HP_FEQ(tex_width) &&
           HP_FEQ(tex_height) &&
           HP_FEQ(tex_offset) &&
           HP_FEQ(ntexture) &&
           HP_FEQ(tex_iscube) &&
           HP_FEQ(tex_mip);
}
#undef HP_FEQ
// Returns true when the chain was recorded (the caller must not launch it).
template <bool OIT, bool NOATTR, bool MS, int FTG, int PFR, int PK>
static bool hp_fine_batch_record(
    int grid, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, const DeviceFramebuffer& fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    HpFineBatch& b = g_hp_fine_batch;
    if (!b.active || hp_fine_batch_max() < 2) return false;
    void (*fn)(HpFineBatch&) = &hp_fine_batch_flush<OIT, NOATTR, MS, FTG, PFR, PK>;
    if (b.n > 0) {
        const bool same = b.flush_fn == fn && b.stream == stream && b.grid == grid &&
                          b.tiles_x == tiles_x && b.tiles_y == tiles_y &&
                          hp_fb_eq(b.fb, fb) && hp_tex_eq(b.tex, texture_state) &&
                          hp_state_eq_but_ncsc(b.state, state);
        if (!same) b.flush_fn(b);
    }
    if (b.n == 0) {
        b.flush_fn = fn; b.stream = stream; b.grid = grid;
        b.tiles_x = tiles_x; b.tiles_y = tiles_y;
        b.fb = fb; b.tex = texture_state; b.state = state;
    }
    HpFineChain& h = b.m.c[b.n++];
    h.tris = tris; h.tile_base = tile_base; h.tile_counts = tile_counts; h.pool_key = pool_key;
    h.pool_cap = pool_cap; h.nsub = MS ? nsub : 1; h.dead_mask = MS ? dead_mask : 0;
    h.slot_stride = MS ? slot_stride : 0; h.ncsc = state.near_clip_source_count;
    if (b.n >= hp_fine_batch_max()) b.flush_fn(b);
    return true;
}
void fine_batch_begin() {
    HpFineBatch& b = g_hp_fine_batch;
    hp_fine_batch_break(nullptr);
    b.active = true;
}
void fine_batch_end() {
    HpFineBatch& b = g_hp_fine_batch;
    hp_fine_batch_break(nullptr);
    b.active = false;
}
// Hipify safety: the packed launch lives in its own template launcher.
template <bool OIT, bool NOATTR, bool MS, int FTG, int PFR, int PK>
static void hipgl_launch_fine_general_gouraud_packed(
    int grid, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    if (!OIT && hp_fine_batch_record<OIT, NOATTR, MS, FTG, PFR, PK>(
            grid, stream, tris, tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y,
            fb, state, texture_state, nsub, dead_mask, slot_stride))
        return;
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false, true, NOATTR,
                                                         HP_MICRO_ROWSKIP_GENERAL, false, false, FTG,
                                                         MS, 0, false, 1, PFR, PK>),
        grid, dim3(FTG, FTG, PFR), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, MS ? nsub : 1, MS ? dead_mask : 0, MS ? slot_stride : 0);
}

template <bool OIT, bool NOATTR, bool MS>
static void hipgl_launch_fine_pool_general_gouraud_sub4_t(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    constexpr int FTG = HP_MICRO_FT_GENERAL;
    constexpr int SDG = (TILE_SIZE / FTG) * (TILE_SIZE / FTG);
    constexpr int PFR = (FTG < TILE_SIZE && FTG * FTG * HP_MICRO_PF_ROWS <= 64)
                      ? HP_MICRO_PF_ROWS : 1;
    constexpr int PK = (MS && !OIT && FTG < TILE_SIZE && FTG * FTG * PFR <= 64 &&
                        HP_FINE_PACK > 1) ? HP_FINE_PACK : 1;
    if (PK > 1 && hp_fine_pack_on() && (nblk % PK) == 0 && nblk == tiles_x * tiles_y)
        hipgl_launch_fine_general_gouraud_packed<OIT, NOATTR, MS, FTG, PFR, PK>(
            nblk * SDG / PK, stream, tris, tile_base, tile_counts, pool_key, pool_cap,
            tiles_x, tiles_y, fb, state, texture_state, nsub, dead_mask, slot_stride);
    else
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false, true, NOATTR,
                                                         HP_MICRO_ROWSKIP_GENERAL, false, false, FTG,
                                                         MS, 0, false, 1, PFR>),
        nblk * SDG, dim3(FTG, FTG, PFR), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, MS ? nsub : 1, MS ? dead_mask : 0, MS ? slot_stride : 0);
}

template <bool OIT, bool MS>
static void hipgl_launch_fine_pool_general_gouraud_sub4_a(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    if (HP_MICRO_NOATTR_GENERAL && fb.attr_buffer == nullptr)
        hipgl_launch_fine_pool_general_gouraud_sub4_t<OIT, true, MS>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
    else
        hipgl_launch_fine_pool_general_gouraud_sub4_t<OIT, false, MS>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
}

static void hipgl_launch_fine_pool_general_gouraud_sub4(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub = 1, int dead_mask = 0, int slot_stride = 0)
{
    const bool oit_live = HP_MICRO_OIT_SPLIT
                        ? (state.oit_enabled && fb.oit_accum && fb.oit_revealage) : true;
    if (oit_live)
        hipgl_launch_fine_pool_general_gouraud_sub4_a<true, false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state, 1, 0, 0);
    else if (nsub > 1)
        hipgl_launch_fine_pool_general_gouraud_sub4_a<false, true>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
    else
        hipgl_launch_fine_pool_general_gouraud_sub4_a<false, false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state, 1, 0, 0);
}

template <bool OIT, bool NOATTR, bool MS>
static void hipgl_launch_fine_pool_general_sub4_t(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    constexpr int FTG = HP_MICRO_FT_GENERAL;
    constexpr int SDG = (TILE_SIZE / FTG) * (TILE_SIZE / FTG);
    constexpr int PFR = (FTG < TILE_SIZE && FTG * FTG * HP_MICRO_PF_ROWS <= 64)
                      ? HP_MICRO_PF_ROWS : 1;
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false, false, NOATTR,
                                                         HP_MICRO_ROWSKIP_GENERAL, false, false, FTG,
                                                         MS, 0, false, 1, PFR>),
        nblk * SDG, dim3(FTG, FTG, PFR), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, MS ? nsub : 1, MS ? dead_mask : 0, MS ? slot_stride : 0);
}

template <bool OIT, bool MS>
static void hipgl_launch_fine_pool_general_sub4_a(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub, int dead_mask, int slot_stride)
{
    if (HP_MICRO_NOATTR_GENERAL && fb.attr_buffer == nullptr)
        hipgl_launch_fine_pool_general_sub4_t<OIT, true, MS>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
    else
        hipgl_launch_fine_pool_general_sub4_t<OIT, false, MS>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
}

static void hipgl_launch_fine_pool_general_sub4(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    int nsub = 1, int dead_mask = 0, int slot_stride = 0)
{
    const bool oit_live = HP_MICRO_OIT_SPLIT
                        ? (state.oit_enabled && fb.oit_accum && fb.oit_revealage) : true;
    if (oit_live)
        hipgl_launch_fine_pool_general_sub4_a<true, false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state, 1, 0, 0);
    else if (nsub > 1)
        hipgl_launch_fine_pool_general_sub4_a<false, true>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
            nsub, dead_mask, slot_stride);
    else
        hipgl_launch_fine_pool_general_sub4_a<false, false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state, 1, 0, 0);
}

template <bool NOATTR>
static void hipgl_launch_fine_pool_opaque_sub4_t(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    constexpr int FTO = HP_MICRO_FT_OPAQUE;
    constexpr int SDO = (TILE_SIZE / FTO) * (TILE_SIZE / FTO);
    constexpr int RWO = (NOATTR && FTO < TILE_SIZE) ? HP_FINE_ROWS : 1;
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, false, NOATTR,
                                                         HP_MICRO_ROWSKIP_OPAQUE, false, false, FTO,
                                                         false, 0, false, RWO>),
        nblk * SDO, dim3(FTO, FTO, RWO), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

static void hipgl_launch_fine_pool_opaque_sub4(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    if (HP_MICRO_NOATTR_OPAQUE && fb.attr_buffer == nullptr)
        hipgl_launch_fine_pool_opaque_sub4_t<true>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
    else
        hipgl_launch_fine_pool_opaque_sub4_t<false>(nblk, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
}

static void hipgl_launch_fine_pool_opaque_gouraud_sub4(
    int nblk, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, false, true, true, false, false,
                                                         false, false, 4>),
        nblk * 16, dim3(4, 4), 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

static inline int hp_micro_sub4() {
    static const int v = []{ const char* e = getenv("HP_MICRO_SUB4"); return e ? atoi(e) : 1; }();
    return v;
}
static inline int hp_micro_sub4_max_tiles() {
    static const int v = []{ const char* e = getenv("HP_MICRO_SUB4_MAX_TILES");
                             return e ? atoi(e) : 256; }();
    return v;
}

template <bool OIT>
static void hipgl_launch_fine_pool_general_gouraud_t(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<true, OIT, false, true>), nblk, block, 0, hp_fine_batch_break(stream),
        tris, (const TileBin*)nullptr, tile_base, tile_counts, pool_key, pool_cap,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

static void hipgl_launch_fine_pool_general_gouraud(
    int nblk, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    const bool oit_live = HP_MICRO_OIT_SPLIT
                        ? (state.oit_enabled && fb.oit_accum && fb.oit_revealage) : true;
    if (oit_live)
        hipgl_launch_fine_pool_general_gouraud_t<true>(nblk, block, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
    else
        hipgl_launch_fine_pool_general_gouraud_t<false>(nblk, block, stream, tris, tile_base,
            tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
}

static int hp_fine_rowskip_max_tiles()
{
    static const int v = []{
        const char* e = getenv("HIPGL_ROWSKIP_MAX_TILES");
        return e ? atoi(e) : 4096;
    }();
    return v;
}

struct HpFineCensus {
    long long n_disp = 0;          // leaf dispatches
    long long n_chain = 0;         // calls to launch_fine_rasterize_with_sort
    long long n_nsub[24] = {0};    // nsub histogram at the chain entry
    long long run_spec[40] = {0};  // run-length histogram, same spec id
    long long run_specfb[40] = {0};// run-length histogram, same spec id AND same fb
    long long n_runs_spec = 0, n_runs_specfb = 0;
    long long ms_yes = 0, ms_no = 0;
    long long why_general = 0, why_nogour = 0, why_attr = 0, why_persist = 0, why_tiles = 0;
    long long last_spec = -1;  const void* last_fb = nullptr;
    long long cur_spec = 0, cur_specfb = 0;
    void close() {
        if (cur_spec)   { run_spec[cur_spec < 39 ? cur_spec : 39]++;   n_runs_spec++;   cur_spec = 0; }
        if (cur_specfb) { run_specfb[cur_specfb < 39 ? cur_specfb : 39]++; n_runs_specfb++; cur_specfb = 0; }
    }
    void note(long long spec, const void* fb) {
        n_disp++;
        if (spec == last_spec) cur_spec++;
        else { if (cur_spec) { run_spec[cur_spec < 39 ? cur_spec : 39]++; n_runs_spec++; } cur_spec = 1; }
        if (spec == last_spec && fb == last_fb) cur_specfb++;
        else { if (cur_specfb) { run_specfb[cur_specfb < 39 ? cur_specfb : 39]++; n_runs_specfb++; } cur_specfb = 1; }
        last_spec = spec; last_fb = fb;
    }
    ~HpFineCensus() {
        if (!getenv("HP_FINE_CENSUS")) return;
        close();
        fprintf(stderr, "[fine_census] leaf_dispatches=%lld chain_calls=%lld\n", n_disp, n_chain);
        fprintf(stderr, "[fine_census] ms_capable yes=%lld no=%lld | why: general=%lld !gouraud=%lld attr=%lld persist=%lld tiles=%lld\n",
                ms_yes, ms_no, why_general, why_nogour, why_attr, why_persist, why_tiles);
        fprintf(stderr, "[fine_census] nsub_hist:");
        for (int i = 0; i < 24; ++i) if (n_nsub[i]) fprintf(stderr, " %d:%lld", i, n_nsub[i]);
        fprintf(stderr, "\n");
        double tot = 0, wsum = 0;
        for (int i = 1; i < 40; ++i) { tot += n_spec_runs_i(i); wsum += (double)i * n_spec_runs_i(i); }
        fprintf(stderr, "[fine_census] spec_runlen_hist (runs=%lld, mean=%.2f):", n_runs_spec,
                tot ? wsum / tot : 0.0);
        for (int i = 1; i < 40; ++i) if (run_spec[i]) fprintf(stderr, " %d:%lld", i, run_spec[i]);
        fprintf(stderr, "\n");
        tot = wsum = 0;
        for (int i = 1; i < 40; ++i) { tot += run_specfb[i]; wsum += (double)i * run_specfb[i]; }
        fprintf(stderr, "[fine_census] specfb_runlen_hist (runs=%lld, mean=%.2f):", n_runs_specfb,
                tot ? wsum / tot : 0.0);
        for (int i = 1; i < 40; ++i) if (run_specfb[i]) fprintf(stderr, " %d:%lld", i, run_specfb[i]);
        fprintf(stderr, "\n");
    }
    long long n_spec_runs_i(int i) const { return run_spec[i]; }
};
static HpFineCensus hp_fine_census;
static inline int hp_fine_census_on() {
    static const int v = []{ const char* e = getenv("HP_FINE_CENSUS"); return e ? atoi(e) : 0; }();
    return v;
}

// Hipify safety: the <<<>>> launch may not sit inside an if/else, so each
// instantiation gets its own launcher and the branch selects between them.
static void hipgl_launch_fine_pool(
    int num_tiles, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state,
    const int* work_list, const int* work_count, int* work_ticket, int persist_blocks,
    bool want_permute, int nsub = 1, int dead_mask = 0, int slot_stride = 0,
    const float4* flt_rec = nullptr, const int* flt_cnt = nullptr,
    const unsigned short* flt_msk = nullptr)
{
    // Runtime dispatch to the specialisation that matches the blend/OIT state.
    // The opaque pass (the hot one) needs neither, and telling the compiler so
    // removes the persistent out_color[4] / OIT accumulators / any_fragment
    // latch from every thread: the winning colour is simply the LAST
    // depth-passing fragment's, so it can be stored the moment it is produced.
    const bool general = (state.oit_enabled && fb.oit_accum && fb.oit_revealage)
                       || state.blend_enabled;
    // Second axis: per-VERTEX (Gouraud) vs per-FRAGMENT lighting. Deciding it
    // here rather than per fragment removes the whole Blinn-Phong block, and
    // with it the register high-water mark, from the shipped instantiation.
    const bool gouraud = (state.gouraud_shading != 0) && (state.lighting_enabled != 0);
    const bool ms_persist = (work_list != nullptr && work_count != nullptr &&
                             work_ticket != nullptr && persist_blocks > 0);
    const bool ms_capable_opaque = !general && gouraud && (fb.attr_buffer == nullptr) &&
                            !ms_persist && (num_tiles <= hp_fine_rowskip_max_tiles());
    static const int ms_general_on = []{
        const char* e = getenv("HP_MS_GENERAL"); return e ? atoi(e) : 1;
    }();
    const bool oit_live_g = (state.oit_enabled && fb.oit_accum && fb.oit_revealage);
    const bool micro_sub4 = hp_micro_sub4() != 0 && num_tiles <= hp_micro_sub4_max_tiles();
    const bool ms_capable_general = (ms_general_on != 0) && general && !oit_live_g &&
                                    HP_MICRO_OIT_SPLIT && micro_sub4 && !ms_persist;
    const bool ms_capable = ms_capable_opaque || ms_capable_general;
    if (hp_fine_census_on() && nsub > 1) {
        if (ms_capable) hp_fine_census.ms_yes++;
        else {
            hp_fine_census.ms_no++;
            if (general) hp_fine_census.why_general++;
            else if (!gouraud) hp_fine_census.why_nogour++;
            else if (fb.attr_buffer != nullptr) hp_fine_census.why_attr++;
            else if (ms_persist) hp_fine_census.why_persist++;
            else hp_fine_census.why_tiles++;
        }
    }
    if (nsub > 1 && !ms_capable) {
        for (int sd = 0; sd < nsub; ++sd) {
            if ((dead_mask >> sd) & 1) continue;
            hipgl_launch_fine_pool(num_tiles, block, stream, tris,
                tile_base + (size_t)sd * slot_stride,
                tile_counts + (size_t)sd * slot_stride,
                pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                work_list, work_count, work_ticket, persist_blocks, want_permute, 1, 0, 0,
                flt_rec, flt_cnt ? (flt_cnt + (size_t)sd * slot_stride) : nullptr, flt_msk);
        }
        return;
    }
    if (hp_fine_census_on()) {
        long long spec = ((long long)general << 0) | ((long long)gouraud << 1)
                       | ((long long)(fb.attr_buffer != nullptr) << 2)
                       | ((long long)ms_persist << 3) | ((long long)ms_capable << 4)
                       | ((long long)(num_tiles <= hp_fine_rowskip_max_tiles()) << 5)
                       | ((long long)(state.oit_enabled && fb.oit_accum && fb.oit_revealage) << 6)
                       | ((long long)(num_tiles <= hp_micro_sub4_max_tiles()) << 7)
                       | ((long long)(num_tiles & 0xffff) << 8)
                       | ((long long)(persist_blocks != 0) << 24);
        hp_fine_census.note(spec, (const void*)fb.color_buffer);
    }
    if (general) {
        if (gouraud) {
            if (micro_sub4)
                hipgl_launch_fine_pool_general_gouraud_sub4(num_tiles, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                    nsub, dead_mask, slot_stride);
            else
                hipgl_launch_fine_pool_general_gouraud(num_tiles, block, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
        } else {
            if (micro_sub4)
                hipgl_launch_fine_pool_general_sub4(num_tiles, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                    nsub, dead_mask, slot_stride);
            else
                hipgl_launch_fine_pool_general(num_tiles, block, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
        }
    } else {
        if (gouraud && fb.attr_buffer == nullptr) {
            const int rowskip_max_tiles = hp_fine_rowskip_max_tiles();
            const bool persist = (work_list != nullptr && work_count != nullptr &&
                                  work_ticket != nullptr && persist_blocks > 0);
            const bool permute = (!persist) && want_permute;
            const int nblk = persist ? persist_blocks : num_tiles;
            if (num_tiles <= rowskip_max_tiles) {
                if (persist)
                    hipgl_launch_fine_pool_opaque_gouraud_noattr_rowskip_persist(nblk, block, stream,
                        tris, tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state,
                        texture_state, work_list, work_count, work_ticket);
                else
                    hipgl_launch_fine_pool_opaque_gouraud_noattr_rowskip(num_tiles, block, stream, tris,
                        tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                        nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
            } else {
                if (persist)
                    hipgl_launch_fine_pool_opaque_gouraud_noattr_persist(nblk, block, stream,
                        tris, tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state,
                        texture_state, work_list, work_count, work_ticket);
                else if (permute)
                    hipgl_launch_fine_pool_opaque_gouraud_noattr_permute(num_tiles, block, stream,
                        tris, tile_base, tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state,
                        texture_state, (const int*)nullptr);
                else
                    hipgl_launch_fine_pool_opaque_gouraud_noattr(num_tiles, block, stream, tris, tile_base,
                        tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
            }
        } else if (gouraud) {
            if (micro_sub4)
                hipgl_launch_fine_pool_opaque_gouraud_sub4(num_tiles, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
            else
                hipgl_launch_fine_pool_opaque_gouraud(num_tiles, block, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
        } else {
            if (micro_sub4)
                hipgl_launch_fine_pool_opaque_sub4(num_tiles, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
            else
                hipgl_launch_fine_pool_opaque(num_tiles, block, stream, tris, tile_base,
                    tile_counts, pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state);
        }
    }
}

static void hipgl_launch_fine_legacy(
    int num_tiles, dim3 block, hipStream_t stream,
    const TriangleSetup* tris, const TileBin* tile_bins,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state)
{
    hipLaunchKernelGGL((kernel_fine_rasterize_with_sort_t<false, true, false>), num_tiles, block, 0, hp_fine_batch_break(stream),
        tris, tile_bins, (const int*)nullptr, (const int*)nullptr,
        (const unsigned long long*)nullptr, 0,
        tiles_x, tiles_y, fb, state, texture_state, (const int*)nullptr, (const int*)nullptr,
        (int*)nullptr, 1, 0, 0);
}

void launch_fine_rasterize_with_sort(
    const TriangleSetup* tris, const TileBin* tile_bins,
    const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state, hipStream_t stream,
    const int* work_list, const int* work_count, int* work_ticket,
    int nsub, int dead_mask, int slot_stride,
    const float4* flt_rec, const int* flt_cnt, const unsigned short* flt_msk)
{
    if (!tris || !fb.color_buffer || !fb.depth_buffer) return;
    if (tiles_x <= 0 || tiles_y <= 0 || fb.width <= 0 || fb.height <= 0) return;
    if (hp_fine_census_on()) {
        hp_fine_census.n_chain++;
        hp_fine_census.n_nsub[(nsub >= 0 && nsub < 24) ? nsub : 23]++;
    }
    const bool use_pool = (tile_base && tile_counts && pool_key && pool_cap > 0);
    if (!use_pool && !tile_bins) return;

    dim3 block(TILE_SIZE, TILE_SIZE);
    int num_tiles = tiles_x * tiles_y;
    if (use_pool) {
        // Resident ceiling for the persistent grid.  Swept per cell family: the
        // small- and large-tile-count regimes want different values.
        static const int persist_on = []{
            const char* e = getenv("HIPGL_PERSIST");
            return e ? atoi(e) : 1;
        }();
        static const int persist_blk_4k = []{
            const char* e = getenv("HIPGL_PERSIST_BLOCKS_4K");
            return e ? atoi(e) : 256;
        }();
        static const int persist_blk_small = []{
            const char* e = getenv("HIPGL_PERSIST_BLOCKS");
            return e ? atoi(e) : 512;
        }();
        static const int persist_blk_large = []{
            const char* e = getenv("HIPGL_PERSIST_BLOCKS_BIG");
            return e ? atoi(e) : 512;
        }();
        static const int persist_min_tiles = []{
            const char* e = getenv("HIPGL_PERSIST_MIN_TILES");
            return e ? atoi(e) : 2048;
        }();
        static const int persist_max_tiles = []{
            const char* e = getenv("HIPGL_PERSIST_MAX_TILES");
            return e ? atoi(e) : 16384;
        }();
        int pblk = 0;
        if (persist_on && num_tiles >= persist_min_tiles && num_tiles <= persist_max_tiles) {
            pblk = (num_tiles <= 4096) ? persist_blk_4k
                 : (num_tiles <= 16384) ? persist_blk_small : persist_blk_large;
            if (pblk > num_tiles) pblk = num_tiles;
        }
        static const int permute_on = []{
            const char* e = getenv("HIPGL_PERMUTE");
            return e ? atoi(e) : 1;
        }();
        // The chunked mapping needs the tile count to divide by 8 exactly (it
        // does at every shipped resolution: 4096 / 16384 / 65536) and only pays
        // where the grid is large enough for the L2 working set to matter.
        const bool permute_ok = (permute_on != 0) && (pblk == 0) &&
                                (num_tiles > persist_max_tiles) &&
                                ((num_tiles & 7) == 0);
        hipgl_launch_fine_pool(num_tiles, block, stream, tris, tile_base, tile_counts,
                               pool_key, pool_cap, tiles_x, tiles_y, fb, state, texture_state,
                               work_list, work_count, work_ticket, pblk, permute_ok,
                               nsub, dead_mask, slot_stride, flt_rec, flt_cnt, flt_msk);
    } else {
        hipgl_launch_fine_legacy(num_tiles, block, stream, tris, tile_bins,
                                 tiles_x, tiles_y, fb, state, texture_state);
    }
}

// ── OIT resolve kernel ──────────────────────────────────────────────────────
// Composites the OIT accumulation buffers over the opaque color buffer.
HIPGL_GLOBAL void kernel_oit_resolve(
    DeviceFramebuffer fb)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= fb.width || py >= fb.height) return;

    int pixel_idx = py * fb.width + px;
    int pitch = (fb.pitch > 0) ? fb.pitch : fb.width * 4;
    int color_idx = py * pitch + px * 4;
    int accum_idx = pixel_idx * 4;

    float acc_r = fb.oit_accum[accum_idx + 0];
    float acc_g = fb.oit_accum[accum_idx + 1];
    float acc_b = fb.oit_accum[accum_idx + 2];
    float acc_a = fb.oit_accum[accum_idx + 3];
    float rev = fb.oit_revealage[pixel_idx];

    if (acc_a < 1e-6f) return;

    float inv_acc_a = 1.0f / fmaxf(acc_a, 1e-4f);
    float tr = acc_r * inv_acc_a;
    float tg = acc_g * inv_acc_a;
    float tb = acc_b * inv_acc_a;
    float coverage = 1.0f - rev;

    float opaque_r = fb.color_buffer[color_idx + 0];
    float opaque_g = fb.color_buffer[color_idx + 1];
    float opaque_b = fb.color_buffer[color_idx + 2];

    fb.color_buffer[color_idx + 0] = clamp01(tr * coverage + opaque_r * rev);
    fb.color_buffer[color_idx + 1] = clamp01(tg * coverage + opaque_g * rev);
    fb.color_buffer[color_idx + 2] = clamp01(tb * coverage + opaque_b * rev);
}

void launch_oit_resolve(DeviceFramebuffer fb, hipStream_t stream)
{
    if (!fb.color_buffer || !fb.oit_accum || !fb.oit_revealage) return;
    if (fb.width <= 0 || fb.height <= 0) return;
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    hipLaunchKernelGGL(kernel_oit_resolve, grid, block, 0, stream, fb);
}



#define HIPGL_BIN_TPB 512
#ifndef HIPGL_SCATTER_ILP
#define HIPGL_SCATTER_ILP 4
#endif
// Live TRIANGLES per block on the large-tile-grid path. Re-swept in r3 after
// the CTA reshape below made the per-block scan/search cost O(log NT) instead
// of O(log 512): a smaller NT is now nearly free, so the old 32 optimum may
// have moved.
#ifndef HIPGL_BIN_TRIS_SMALL
#define HIPGL_BIN_TRIS_SMALL 16
#endif
// The COUNT pair-walk wants a different NT from the SCATTER: its per-pair chain
// has no tile_base gather and no cursor atomic, so it is less latency-starved
// and pays relatively more for the per-block setup. Swept separately.
#ifndef HIPGL_CNT_TRIS_SMALL
#define HIPGL_CNT_TRIS_SMALL 64
#endif
// TRIANGLES per block, decoupled from THREADS per block. The binning grid used
// to be tri_count/512 blocks, i.e. ~1.5 waves per SIMD at r1024 -- there was
// simply no second wave to hide the per-pair dependent chain behind. Giving a
// block fewer triangles but the same 512 lanes multiplies the resident wave
// count by HIPGL_BIN_TPB/HIPGL_BIN_TRIS at identical total work: lanes
// >= HIPGL_BIN_TRIS contribute n = 0 to the block scan (so the scan and the
// owner search are unchanged and still correct -- an empty lane's offset range
// is empty, so no pair can ever resolve to it) and then join the pair loop.
// Chosen at launch time by tile-grid size: see launch_* below.


// Block-wide exclusive scan of `n` over NT lanes.
// s_off must have NT+1 ints. On return s_off[j] is the exclusive prefix for
// lane j and s_off[NT] is the block total.
template <int NT>
HIPGL_DEVICE __forceinline__ void bin_block_scan_t(int* s_off, int tid, int n)
{
    if (NT <= 64) {
        // The whole scan fits in wave 0: no LDS round-trips, no barrier loop.
        // Lanes >= NT of that wave contribute nothing and simply ride along so
        // the shfl is wave-uniform.
        if (tid < 64) {
            int v = (tid < NT) ? n : 0;
#pragma unroll
            for (int d = 1; d < NT; d <<= 1) {
                int t = __shfl_up(v, d, 64);
                if (tid >= d) v += t;
            }
            if (tid == 0) s_off[0] = 0;
            if (tid < NT) s_off[tid + 1] = v;
        }
        __syncthreads();
    } else {
        constexpr int NW = (NT > 64) ? (NT / 64) : 1;
        __shared__ int s_wtot[NW];
        const int lane = tid & 63;
        const int wid  = tid >> 6;
        int v = (tid < NT) ? n : 0;
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            const int y = __shfl_up(v, off, 64);
            if (lane >= off) v += y;
        }
        if (lane == 63 && wid < NW) s_wtot[wid] = v;
        __syncthreads();
        int carry = 0;
#pragma unroll
        for (int w = 0; w < NW; ++w) carry += (w < wid) ? s_wtot[w] : 0;
        if (tid == 0) s_off[0] = 0;
        if (tid < NT) s_off[tid + 1] = carry + v;
        __syncthreads();
    }
}

HIPGL_DEVICE __forceinline__ void hp_agg_count(int* __restrict__ counts, int t, bool ok)
{
    unsigned long long active = __ballot(ok ? 1 : 0);
    const int lane = (int)(threadIdx.x & 63u);
    while (active) {
        const int leader = __builtin_ctzll(active);
        const int lt = __shfl(t, leader, 64);
        const unsigned long long same = active & __ballot(ok && t == lt);
        if (lane == leader) atomicAdd(&counts[lt], (int)__popcll(same));
        active &= ~same;
    }
}

// Same fold on the scatter's reservation, which needs the RETURN value: the
// leader reserves popc(same) consecutive slots and each member takes
// base + its rank inside the group.
HIPGL_DEVICE __forceinline__ int hp_agg_reserve(int* __restrict__ cursor, int t, bool ok)
{
    const int lane = (int)(threadIdx.x & 63u);
    const unsigned long long lanebit = 1ull << lane;
    unsigned long long active = __ballot(ok ? 1 : 0);
    int my_leader = 0, my_rank = 0, my_cnt = 0;
    // Election is pure ALU: ballot/shfl/popcount, NO memory on the loop-carried
    // chain. (The first version issued the leader's atomicAdd INSIDE the loop
    // and then shuffled its RETURN, so every iteration paid a full device-scope
    // atomic round trip in series -- worth -21% at r512/w1, where the tile grid
    // is large enough that a wave holds many distinct tiles and there is no
    // second world to hide the chain behind.)
    while (active) {
        const int leader = __builtin_ctzll(active);
        const int lt = __shfl(t, leader, 64);
        const unsigned long long same = active & __ballot(ok && t == lt);
        if (same & lanebit) {
            my_leader = leader;
            my_rank   = (int)__popcll(same & (lanebit - 1ull));
        }
        if (lane == leader) my_cnt = (int)__popcll(same);
        active &= ~same;
    }
    // ONE atomic round trip per wave: every group leader reserves its whole
    // group in the SAME instruction, so the groups' atomics go out in parallel.
    int lbase = 0;
    if (my_cnt > 0) lbase = atomicAdd(&cursor[t], my_cnt);
    return __shfl(lbase, my_leader, 64) + my_rank;
}

// Largest j in [0, NT) with s_off[j] <= p < s_off[j+1]. NT is a power of two
// and s_off[0] == 0 <= p, so the classic branchless binary descent applies:
// exactly log2(NT) dependent LDS loads, no branches, fully unrolled.
template <int NT>
HIPGL_DEVICE __forceinline__ int bin_find_owner_t(const int* s_off, int p)
{
    int base = 0;
#pragma unroll
    for (int n = NT; n > 1; n >>= 1) {
        const int half = n >> 1;
        base = (s_off[base + half] <= p) ? (base + half) : base;
    }
    return base;
}

HIPGL_DEVICE inline BinRec hipgl_near_clip_binrec(const TriangleSetup& tri,
    int order_id, int tiles_x, int tiles_y, int fb_width, int fb_height);

HIPGL_DEVICE __forceinline__ void hp_rect_clip_tiles(
    int& lo_x, int& lo_y, int& nx, int& ny, int tiles_y, const TileRect& rect)
{
    if (ny <= 0) return;
    int sub = 0;
    if (rect.nsub > 1) {
        if (rect.nsub <= 8) {
#pragma unroll
            for (int k = 1; k < 8; ++k) sub += (lo_y >= k * tiles_y) ? tiles_y : 0;
        } else {
#pragma unroll
            for (int k = 1; k < HP_MAX_BLOCKS; ++k)
                sub += (lo_y >= k * tiles_y) ? tiles_y : 0;
        }
    }
    const int ly  = lo_y - sub;
    const int nlx = max(lo_x, rect.x0);
    const int nhx = min(lo_x + nx - 1, rect.x1);
    const int nly = max(ly, rect.y0);
    const int nhy = min(ly + ny - 1, rect.y1);
    if (nlx > nhx || nly > nhy) { lo_x = 0; lo_y = sub; nx = 1; ny = 0; return; }
    lo_x = nlx; nx = nhx - nlx + 1;
    lo_y = nly + sub; ny = nhy - nly + 1;
}

HIPGL_DEVICE __forceinline__ int hp_subdraw_of(int idx, const BinSubdraws& sd)
{
    int r = 0;
    if (sd.nsub <= 8) {
#pragma unroll
        for (int k = 0; k < 7; ++k) r += (k < sd.nsub - 1 && idx >= sd.split[k]) ? 1 : 0;
    } else if (sd.nsub <= 24) {
#pragma unroll
        for (int k = 0; k < 23; ++k)
            r += (k < sd.nsub - 1 && idx >= sd.split[k]) ? 1 : 0;
    } else {
#pragma unroll
        for (int k = 0; k < HP_MAX_SUBDRAWS - 1; ++k)
            r += (k < sd.nsub - 1 && idx >= sd.split[k]) ? 1 : 0;
    }
    return hp_subdraw_perm(r, sd);
}

// Pair-parallel walk (the low-tile-count path). Same block-scan / flat-pair
// scheme as the scatter below; cheaper than the difference-array path when the
// tile grid is small, because it is ONE dispatch instead of three.
template <int HIPGL_BIN_TRIS, int TPB>
HIPGL_DEVICE __forceinline__ void hp_count_tile_pairs_body(
    const TriangleSetup* __restrict__ tris,
    int tri_count,
    int* __restrict__ tile_counts,
    BinRec* __restrict__ binrec,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height, bool packed_ids, int cull_mask,
    NearClipQueue queue,
    BinSubdraws subs,
    bool have_rec,
    TileRect rect,
    int ilv_log2,
    int cnt_blocks)
{
    __shared__ int   s_off[HIPGL_BIN_TRIS + 1];
    __shared__ int   s_lox[HIPGL_BIN_TRIS];
    __shared__ int   s_loy[HIPGL_BIN_TRIS];
    __shared__ int   s_nx [HIPGL_BIN_TRIS];
    __shared__ float s_inv[HIPGL_BIN_TRIS];
    __shared__ BinRec s_rec;

    constexpr bool HP_BIN_AGG = (HIPGL_BIN_TRIS >= 64);
    const int tid = threadIdx.x;
    const int tri_base_c = blockIdx.x * HIPGL_BIN_TRIS;
    int idx_c;
    if (ilv_log2 >= 0) {
        idx_c = (((tid >> ilv_log2) * cnt_blocks + (int)blockIdx.x) << ilv_log2)
              + (tid & ((1 << ilv_log2) - 1));
    } else {
        idx_c = tri_base_c + tid;
    }
    const int idx = (tid < HIPGL_BIN_TRIS) ? idx_c : tri_count;   // sentinel: fails the idx < tri_count guard

    int n = 0, lo_x = 0, lo_y = 0, nx = 1, ny = 0;
    uint32_t z_ord = 0x80000000u;
    if (have_rec) {
        if (idx < tri_count) {
            const BinRec r = binrec[idx];
            lo_x = (int)(r.xy & 0xFFFFu);
            lo_y = (int)(r.xy >> 16);
            nx   = (int)(r.nxy & 0xFFFFu);
            ny   = (int)(r.nxy >> 16);
            if (rect.x1 >= 0) hp_rect_clip_tiles(lo_x, lo_y, nx, ny, tiles_y, rect);
            n    = nx * ny;
        }
    } else {
    if (idx < tri_count) {
        const TriangleSetup& tri = tris[idx];
        if (!hipgl_bin_reject(tri, cull_mask)) {
            int tmin_x = math::max2i(tri.bbox_min_x, 0);
            int tmin_y = math::max2i(tri.bbox_min_y, 0);
            int tmax_x = math::min2i(tri.bbox_max_x, fb_width - 1);
            int tmax_y = math::min2i(tri.bbox_max_y, fb_height - 1);
            if (tmin_x <= tmax_x && tmin_y <= tmax_y) {
                lo_x = math::max2i(tmin_x / TILE_SIZE, 0);
                int hi_x = math::min2i(tmax_x / TILE_SIZE, tiles_x - 1);
                lo_y = math::max2i(tmin_y / TILE_SIZE, 0);
                int hi_y = math::min2i(tmax_y / TILE_SIZE, tiles_y - 1);
                if (lo_x <= hi_x && lo_y <= hi_y) {
                    nx = hi_x - lo_x + 1;
                    ny = hi_y - lo_y + 1;
                    n  = nx * ny;
                    z_ord = tri.z_ord;
                }
            }
        }
    }
    const int sub_id  = (subs.nsub > 1) ? hp_subdraw_of(idx, subs) : 0;
    if (subs.nsub > 1 && ((subs.dead_mask >> sub_id) & 1)) { n = 0; nx = 1; ny = 0; }
    const int sub_row = (subs.nsub > 1) ? (sub_id * tiles_y) : 0;
    lo_y += sub_row;
    if (rect.x1 >= 0) { hp_rect_clip_tiles(lo_x, lo_y, nx, ny, tiles_y, rect); n = nx * ny; }
    if (binrec && idx < tri_count) {
        BinRec r;
        r.xy  = (unsigned int)lo_x | ((unsigned int)lo_y << 16);
        r.nxy = (unsigned int)nx   | ((unsigned int)ny   << 16);
        r.key = ((unsigned long long)z_ord << 32)
              | (unsigned long long)(uint32_t)(packed_ids ? 2 * idx : idx);
        binrec[idx] = r;
    }
    }   // !have_rec
    if (tid < HIPGL_BIN_TRIS) {
        s_lox[tid] = lo_x;
        s_loy[tid] = lo_y;
        s_nx [tid] = nx;
        s_inv[tid] = 1.0f / (float)nx;
    }
    bin_block_scan_t<HIPGL_BIN_TRIS>(s_off, tid, n);

    const int total = s_off[HIPGL_BIN_TRIS];
    // Same ILP pipelining as the scatter below: the per-pair chain is 9
    // dependent LDS loads feeding a global atomic, and the grid is only
    // tri_count/512 blocks, so without several pairs in flight per lane the
    // wave just waits. This path is the one taken at r256 (64x64 tiles).
    const int CSTRIDE = TPB * HIPGL_SCATTER_ILP;
    int p_base = tid;
    for (; p_base + CSTRIDE - TPB < total; p_base += CSTRIDE) {
        int t_[HIPGL_SCATTER_ILP];
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            const int p = p_base + u * TPB;
            int j = bin_find_owner_t<HIPGL_BIN_TRIS>(s_off, p);
            int local = p - s_off[j];
            int jnx = s_nx[j];
            int q = (int)((float)local * s_inv[j]);
            int r = local - q * jnx;
            if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
            t_[u] = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        }
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            if constexpr (HP_BIN_AGG) hp_agg_count(tile_counts, t_[u], true);
            else                      atomicAdd(&tile_counts[t_[u]], 1);
        }
    }
    for (int p = p_base; p < total; p += TPB) {
        int j = bin_find_owner_t<HIPGL_BIN_TRIS>(s_off, p);
        int local = p - s_off[j];
        int jnx = s_nx[j];
        int q = (int)((float)local * s_inv[j]);
        int r = local - q * jnx;
        if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
        int t = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        if constexpr (HP_BIN_AGG) hp_agg_count(tile_counts, t, true);
        else                      atomicAdd(&tile_counts[t], 1);
    }

    if (queue.count) {
        const int nq = min(max(*queue.count, 0), tri_count);
        for (int q = (int)blockIdx.x; q < nq; q += cnt_blocks) {
            const int source = queue.ids[q];
            __syncthreads();
            if ((unsigned)source >= (unsigned)tri_count) continue;
            if (tid == 0) {
                s_rec = hipgl_near_clip_binrec(tris[tri_count + source],
                    2 * source + 1, tiles_x, tiles_y, fb_width, fb_height);
                if (subs.nsub > 1) {
                    const int nsb = hp_subdraw_of(source, subs);
                    if ((subs.dead_mask >> nsb) & 1) s_rec.nxy = 1u;   // nx=1, ny=0
                    s_rec.xy += (unsigned int)(nsb * tiles_y) << 16;
                }
                if (rect.x1 >= 0) {
                    int rlx = (int)(s_rec.xy & 0xFFFFu), rly = (int)(s_rec.xy >> 16);
                    int rnx = (int)(s_rec.nxy & 0xFFFFu), rny = (int)(s_rec.nxy >> 16);
                    hp_rect_clip_tiles(rlx, rly, rnx, rny, tiles_y, rect);
                    s_rec.xy  = (unsigned int)rlx | ((unsigned int)rly << 16);
                    s_rec.nxy = (unsigned int)rnx | ((unsigned int)rny << 16);
                }
                // The scatter's own near-clip epilogue REREADS this record, so
                // clipping it here bounds both passes from one place.
                if (binrec) binrec[tri_count + source] = s_rec;
            }
            __syncthreads();
            const int lo_x = s_rec.xy & 0xFFFFu;
            const int lo_y = s_rec.xy >> 16;
            const int qnx  = s_rec.nxy & 0xFFFFu;
            const int qny  = s_rec.nxy >> 16;
            for (int p = tid; p < qnx * qny; p += TPB)
                atomicAdd(&tile_counts[(lo_y + p / qnx) * tiles_x + lo_x + p % qnx], 1);
        }
    }
}

template <int HIPGL_BIN_TRIS, int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_count_tile_pairs(
    const TriangleSetup* __restrict__ tris,
    int tri_count,
    int* __restrict__ tile_counts,
    BinRec* __restrict__ binrec,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height, bool packed_ids, int cull_mask,
    NearClipQueue queue, BinSubdraws subs, bool have_rec, TileRect rect, int ilv_log2)
{
    hp_count_tile_pairs_body<HIPGL_BIN_TRIS, TPB>(
        tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height,
        packed_ids, cull_mask, queue, subs, have_rec, rect, ilv_log2, (int)gridDim.x);
}

HIPGL_DEVICE __forceinline__ void hp_fb_mask_clear_rider(const FbMaskClearDesc& mc, int p)
{
    if (p >= mc.total) return;
    mc.mask[p] = mc.depth_in[p] < mc.mask_ref ? 1u : 0u;
    float4 v; v.x = mc.r; v.y = mc.g; v.z = mc.b; v.w = mc.a;
    reinterpret_cast<float4*>(mc.color)[p] = v;
    mc.depth[p] = mc.d;
}

HIPGL_GLOBAL void __launch_bounds__(256) kernel_fb_mask_clear(FbMaskClearDesc mc)
{
    hp_fb_mask_clear_rider(mc, (int)(blockIdx.x * 256u + threadIdx.x));
}

void launch_fb_mask_clear(const FbMaskClearDesc& mc, hipStream_t stream)
{
    if (mc.total <= 0 || !mc.color || !mc.depth || !mc.mask || !mc.depth_in) return;
    hipLaunchKernelGGL(kernel_fb_mask_clear, (unsigned)((mc.total + 255) / 256), 256u, 0, stream, mc);
}

template <int HIPGL_BIN_TRIS, int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_count_tile_pairs_z2(
    BinCntSlot s0, BinCntSlot s1,
    int tri_count, int tiles_x, int tiles_y, int fb_width, int fb_height,
    BinSubdraws subs, int ilv_log2, int cnt_blocks, FbMaskClearDesc mc)
{
    if ((int)blockIdx.x >= cnt_blocks) {
        if (blockIdx.z == 0)
            hp_fb_mask_clear_rider(mc, (int)(blockIdx.x - (unsigned)cnt_blocks) * TPB + (int)threadIdx.x);
        return;
    }
    const bool z = (blockIdx.z != 0);
    hp_count_tile_pairs_body<HIPGL_BIN_TRIS, TPB>(
        z ? s1.tris : s0.tris, tri_count,
        z ? s1.tile_counts : s0.tile_counts,
        z ? s1.binrec : s0.binrec,
        tiles_x, tiles_y, fb_width, fb_height,
        z ? s1.packed_ids : s0.packed_ids,
        z ? s1.cull_mask : s0.cull_mask,
        z ? s1.queue : s0.queue, subs,
        z ? s1.have_rec : s0.have_rec,
        z ? s1.rect : s0.rect, ilv_log2, cnt_blocks);
}


HIPGL_GLOBAL void kernel_count_stamp_diff(
    const TriangleSetup* __restrict__ tris,
    int tri_count,
    int* __restrict__ diff,
    BinRec* __restrict__ binrec,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height, bool packed_ids, int cull_mask)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tri_count) return;
    const TriangleSetup& tri = tris[idx];
    int lo_x = 0, lo_y = 0, nx = 1, ny = 0, hi_x = 0, hi_y = 0;
    uint32_t z_ord = 0x80000000u;
    bool live = !hipgl_bin_reject(tri, cull_mask);
    int tmin_x = 0, tmin_y = 0, tmax_x = 0, tmax_y = 0;
    if (live) {
        tmin_x = math::max2i(tri.bbox_min_x, 0);
        tmin_y = math::max2i(tri.bbox_min_y, 0);
        tmax_x = math::min2i(tri.bbox_max_x, fb_width - 1);
        tmax_y = math::min2i(tri.bbox_max_y, fb_height - 1);
        live = (tmin_x <= tmax_x && tmin_y <= tmax_y);
    }
    if (live) {
        lo_x = math::max2i(tmin_x / TILE_SIZE, 0); hi_x = math::min2i(tmax_x / TILE_SIZE, tiles_x - 1);
        lo_y = math::max2i(tmin_y / TILE_SIZE, 0); hi_y = math::min2i(tmax_y / TILE_SIZE, tiles_y - 1);
        live = (lo_x <= hi_x && lo_y <= hi_y);
    }
    if (live) {
        nx = hi_x - lo_x + 1;
        ny = hi_y - lo_y + 1;
        z_ord = tri.z_ord;
    } else {
        lo_x = 0; lo_y = 0; nx = 1; ny = 0;
    }
    if (binrec) {
        BinRec r;
        r.xy  = (unsigned int)lo_x | ((unsigned int)lo_y << 16);
        r.nxy = (unsigned int)nx   | ((unsigned int)ny   << 16);
        r.key = ((unsigned long long)z_ord << 32)
              | (unsigned long long)(uint32_t)(packed_ids ? 2 * idx : idx);
        binrec[idx] = r;
    }
    if (!live) return;
    const int bx = hi_x + 1, by = hi_y + 1;
    atomicAdd(&diff[lo_y * tiles_x + lo_x], 1);
    if (bx < tiles_x) atomicAdd(&diff[lo_y * tiles_x + bx], -1);
    if (by < tiles_y) {
        atomicAdd(&diff[by * tiles_x + lo_x], -1);
        if (bx < tiles_x) atomicAdd(&diff[by * tiles_x + bx], 1);
    }
}

// Only queued near-plane straddlers can own a second fan triangle. A fixed
// number of blocks consumes that device count, so an ordinary draw never
// scans the unused suffix or initializes an empty BinRec for every source.
HIPGL_DEVICE inline BinRec hipgl_near_clip_binrec(const TriangleSetup& tri,
    int order_id, int tiles_x, int tiles_y, int fb_width, int fb_height)
{
    int lo_x = 0, lo_y = 0, nx = 1, ny = 0;
    uint32_t z_ord = 0x80000000u;
    if (!tri.degenerate) {
        const int tmin_x = max(tri.bbox_min_x, 0);
        const int tmin_y = max(tri.bbox_min_y, 0);
        const int tmax_x = min(tri.bbox_max_x, fb_width - 1);
        const int tmax_y = min(tri.bbox_max_y, fb_height - 1);
        if (tmin_x <= tmax_x && tmin_y <= tmax_y) {
            lo_x = max(tmin_x / TILE_SIZE, 0);
            lo_y = max(tmin_y / TILE_SIZE, 0);
            const int hi_x = min(tmax_x / TILE_SIZE, tiles_x - 1);
            const int hi_y = min(tmax_y / TILE_SIZE, tiles_y - 1);
            if (lo_x <= hi_x && lo_y <= hi_y) {
                nx = hi_x - lo_x + 1;
                ny = hi_y - lo_y + 1;
                z_ord = tri.z_ord;
            }
        }
    }
    BinRec r;
    r.xy = (unsigned)lo_x | ((unsigned)lo_y << 16);
    r.nxy = (unsigned)nx | ((unsigned)ny << 16);
    r.key = ((unsigned long long)z_ord << 32) | (uint32_t)order_id;
    return r;
}

HIPGL_GLOBAL void kernel_count_near_clip_extras(
    const TriangleSetup* __restrict__ tris, int source_count, NearClipQueue queue,
    int* __restrict__ counts, BinRec* __restrict__ binrec,
    int tiles_x, int tiles_y, int fb_width, int fb_height, bool diff)
{
    __shared__ BinRec shared_rec;
    const int count = min(max(*queue.count, 0), source_count);
    for (int q = blockIdx.x; q < count; q += gridDim.x) {
        const int source = queue.ids[q];
        if ((unsigned)source >= (unsigned)source_count) continue;
        if (threadIdx.x == 0) {
            shared_rec = hipgl_near_clip_binrec(tris[source_count + source],
                2 * source + 1, tiles_x, tiles_y, fb_width, fb_height);
            if (binrec) binrec[source_count + source] = shared_rec;
        }
        __syncthreads();
        const int lo_x = shared_rec.xy & 0xFFFFu;
        const int lo_y = shared_rec.xy >> 16;
        const int nx = shared_rec.nxy & 0xFFFFu;
        const int ny = shared_rec.nxy >> 16;
        if (diff) {
            if (threadIdx.x == 0 && ny > 0) {
                const int bx = lo_x + nx, by = lo_y + ny;
                atomicAdd(&counts[lo_y * tiles_x + lo_x], 1);
                if (bx < tiles_x) atomicAdd(&counts[lo_y * tiles_x + bx], -1);
                if (by < tiles_y) {
                    atomicAdd(&counts[by * tiles_x + lo_x], -1);
                    if (bx < tiles_x) atomicAdd(&counts[by * tiles_x + bx], 1);
                }
            }
        } else {
            for (int p = threadIdx.x; p < nx * ny; p += blockDim.x)
                atomicAdd(&counts[(lo_y + p / nx) * tiles_x + lo_x + p % nx], 1);
        }
        __syncthreads();
    }
}

// Inclusive prefix sum along x, one block per tile row.
HIPGL_GLOBAL void kernel_prefix_rows(int* __restrict__ d, int tiles_x)
{
    __shared__ int s[256 + 1];
    const int tid = threadIdx.x;
    int* row = d + (size_t)blockIdx.x * tiles_x;
    int carry = 0;
    for (int base = 0; base < tiles_x; base += 256) {
        int v = (base + tid < tiles_x) ? row[base + tid] : 0;
        if (tid == 0) s[0] = 0;
        s[tid + 1] = v;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            int a = (tid >= off) ? s[tid + 1 - off] : 0;
            __syncthreads();
            if (tid >= off) s[tid + 1] += a;
            __syncthreads();
        }
        if (base + tid < tiles_x) row[base + tid] = s[tid + 1] + carry;
        int tot = s[256];
        __syncthreads();
        carry += tot;
    }
}

HIPGL_GLOBAL void kernel_prefix_cols(int* __restrict__ d, int tiles_x, int tiles_y)
{
    __shared__ int s[256 + 1];
    const int tid = threadIdx.x;
    const int x = blockIdx.x;
    if (x >= tiles_x) return;
    int carry = 0;
    for (int base = 0; base < tiles_y; base += 256) {
        const int y = base + tid;
        int v = (y < tiles_y) ? d[(size_t)y * tiles_x + x] : 0;
        if (tid == 0) s[0] = 0;
        s[tid + 1] = v;
        __syncthreads();
        for (int off = 1; off < 256; off <<= 1) {
            int a = (tid >= off) ? s[tid + 1 - off] : 0;
            __syncthreads();
            if (tid >= off) s[tid + 1] += a;
            __syncthreads();
        }
        if (y < tiles_y) d[(size_t)y * tiles_x + x] = s[tid + 1] + carry;
        int tot = s[256];
        __syncthreads();
        carry += tot;
    }
}

#ifndef HP_SCAT_SCAN_CAP
#define HP_SCAT_SCAN_CAP 2052
#endif
#ifndef HP_SCAT_SCAN_PER
#define HP_SCAT_SCAN_PER 12
#endif
#undef HP_SCAT_SCAN_CAP
#define HP_SCAT_SCAN_CAP (HIPGL_BIN_TPB * HP_SCAT_SCAN_PER + 4)

template <int HIPGL_BIN_TRIS, int TPB, bool EXACT, bool SCAN = false>
HIPGL_DEVICE __forceinline__ void hp_scatter_tile_pairs_body(
    const TriangleSetup* __restrict__ tris,
    const BinRec* __restrict__ binrec,
    int tri_count,
    const int* __restrict__ tile_base,
    int* __restrict__ cursor,
    int* __restrict__ pool_tri,
    float* __restrict__ pool_z,
    // When non-null the scatter writes the FINAL 64-bit composite key straight
    // into this array, so kernel_pack_pool_keys (a full pool read + write, and
    // a dispatch, per world per frame) disappears. It also stops the pack pass
    // from touching the GAPS that exact rejection leaves in the reserved pool.
    unsigned long long* __restrict__ pool_key,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    int pool_capacity,
    int* __restrict__ overflow_flag,
    NearClipQueue queue,
    TileRect rect,
    int ilv_log2,
    // SCAN only: the per-tile counts this CTA scans for itself, the global
    // array block 0 publishes the result into, and the length (num_slots+1).
    const int* __restrict__ tile_counts_in = nullptr,
    int* __restrict__ tile_base_out = nullptr,
    int scan_n = 0,
    int hp_vbx = -1, int hp_vgx = -1)
{
    const int hp_bx = (hp_vbx >= 0) ? hp_vbx : (int)blockIdx.x;
    const int hp_gx = (hp_vgx >= 0) ? hp_vgx : (int)gridDim.x;
    __shared__ BinRec s_ncrec;
    __shared__ int   s_off[HIPGL_BIN_TRIS + 1];
    __shared__ int   s_lox[HIPGL_BIN_TRIS];
    __shared__ int   s_loy[HIPGL_BIN_TRIS];
    __shared__ int   s_nx [HIPGL_BIN_TRIS];
    __shared__ float s_inv[HIPGL_BIN_TRIS];
    __shared__ unsigned long long s_key[HIPGL_BIN_TRIS];
    // EXACT only: the 9 edge coefficients per triangle, staged once per block.
    // Only HIPGL_BIN_TRIS distinct triangles exist in a block (16 on the exact
    // path), so this replaces a 484 B TriangleSetup gather per CANDIDATE PAIR
    // with one per triangle.
    __shared__ float s_eq [EXACT ? HIPGL_BIN_TRIS : 1][9];
    __shared__ int s_lb[SCAN ? (TPB + 1) : 1];
    __shared__ __attribute__((aligned(16))) unsigned short s_o16[SCAN ? HP_SCAT_SCAN_CAP : 1];
    __shared__ int s_sw  [SCAN ? (TPB / 64) : 1];
    __shared__ int s_ovf [SCAN ? (TPB / 64) : 1];

    constexpr bool HP_BIN_AGG = (HIPGL_BIN_TRIS >= 64);
    const int tid = threadIdx.x;
    constexpr int PER = HP_SCAT_SCAN_PER;
    static_assert(PER % 4 == 0, "scan lane width must be a multiple of int4");
    static_assert(!SCAN || TPB * PER + 1 <= HP_SCAT_SCAN_CAP, "LDS scan cap");
    int sc[SCAN ? PER : 1];
    if constexpr (SCAN) {
        const int i0 = (int)threadIdx.x * PER;
#pragma unroll
        for (int k = 0; k < PER; ++k) sc[k] = 0;
        if (i0 + PER - 1 < scan_n) {
#pragma unroll
            for (int k = 0; k < PER; k += 4) {
                const int4 v = *reinterpret_cast<const int4*>(tile_counts_in + i0 + k);
                sc[k] = v.x; sc[k + 1] = v.y; sc[k + 2] = v.z; sc[k + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int k = 0; k < PER; ++k)
                if (i0 + k < scan_n) sc[k] = tile_counts_in[i0 + k];
        }
    }
    const int tri_base = hp_bx * HIPGL_BIN_TRIS;
    int idx_s;
    if (ilv_log2 >= 0) {
        idx_s = (((tid >> ilv_log2) * hp_gx + hp_bx) << ilv_log2)
              + (tid & ((1 << ilv_log2) - 1));
    } else {
        idx_s = tri_base + tid;
    }
    const int idx = (tid < HIPGL_BIN_TRIS) ? idx_s : tri_count;   // sentinel: fails the idx < tri_count guard

    int n = 0, lo_x = 0, lo_y = 0, nx = 1;
    unsigned long long key = 0ull;
    if (idx < tri_count) {
        const BinRec r = binrec[idx];
        lo_x = (int)(r.xy & 0xFFFFu);
        lo_y = (int)(r.xy >> 16);
        nx   = (int)(r.nxy & 0xFFFFu);
        int ny = (int)(r.nxy >> 16);
        if (rect.x1 >= 0) hp_rect_clip_tiles(lo_x, lo_y, nx, ny, tiles_y, rect);
        n    = nx * ny;
        key  = r.key;
    }
    if (tid < HIPGL_BIN_TRIS) {
        s_lox[tid] = lo_x;
        s_loy[tid] = lo_y;
        s_nx [tid] = nx;
        s_inv[tid] = 1.0f / (float)nx;
        s_key[tid] = key;
        if (EXACT) {
            if (n > 0) {
                const TriangleSetup& tq = tris[idx];
#pragma unroll
                for (int e = 0; e < 3; e++) {
                    s_eq[tid][e*3+0] = tq.edge_eq[e][0];
                    s_eq[tid][e*3+1] = tq.edge_eq[e][1];
                    s_eq[tid][e*3+2] = tq.edge_eq[e][2];
                }
            } else {
#pragma unroll
                for (int q9 = 0; q9 < 9; q9++) s_eq[tid][q9] = 0.0f;
            }
        }
    }
    bin_block_scan_t<HIPGL_BIN_TRIS>(s_off, tid, n);

    const int total = s_off[HIPGL_BIN_TRIS];

    // -- r4: THE FOLDED EXCLUSIVE SCAN ------------------------------------
    // Same arithmetic as hp_scan_excl_single4_body (per-lane serial sum, wave
    // shfl_up scan, cross-wave fix-up) with 8 ints per lane instead of 4, so
    // ONE chunk covers every np1 the gate accepts and the running carry, the
    // chunk loop and five of the eight barriers disappear.  Integer addition,
    // so the values are identical to the deleted dispatch's for every input.
    if constexpr (SCAN) {
        constexpr int NW = TPB / 64;
        const int lane = tid & 63;
        const int wid  = tid >> 6;
        int t8 = 0;
#pragma unroll
        for (int k = 0; k < PER; ++k) t8 += sc[k];
        int x = t8;
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            int y = __shfl_up(x, off, 64);
            if (lane >= off) x += y;
        }
        // Offsets are exclusive WITHIN the group, so the largest is the sum of
        // the first PER-1 counts <= t8.
        const bool wov = __any(t8 > 65535);
        if (lane == 63) { s_sw[wid] = x; s_ovf[wid] = wov ? 1 : 0; }
        __syncthreads();
        if (wid == 0) {
            int w = (lane < NW) ? s_sw[lane] : 0;
            int o = (lane < NW) ? s_ovf[lane] : 0;
#pragma unroll
            for (int off = 1; off < NW; off <<= 1) {
                int y = __shfl_up(w, off, 64);
                if (lane >= off) w += y;
                o |= __shfl_xor(o, off, 64);
            }
            if (lane < NW) { s_sw[lane] = w; s_ovf[lane] = o; }
        }
        // group bases go straight to LDS; the u16 offsets need no prefix
        const int b0 = tid * PER;
        {
            unsigned int acc = 0;
#pragma unroll
            for (int k = 0; k < PER; k += 4) {
                const unsigned int o0 = acc;         acc += (unsigned)sc[k];
                const unsigned int o1 = acc;         acc += (unsigned)sc[k + 1];
                const unsigned int o2 = acc;         acc += (unsigned)sc[k + 2];
                const unsigned int o3 = acc;         acc += (unsigned)sc[k + 3];
                uint2 pk;
                pk.x = (o0 & 0xFFFFu) | (o1 << 16);
                pk.y = (o2 & 0xFFFFu) | (o3 << 16);
                if (b0 + k + 3 < HP_SCAT_SCAN_CAP)
                    *reinterpret_cast<uint2*>(&s_o16[b0 + k]) = pk;
            }
        }
        __syncthreads();
        s_lb[tid] = ((wid > 0) ? s_sw[wid - 1] : 0) + x - t8;
        // Tail element TPB*PER (np1 = TPB*PER + 1 at the cap): its exclusive
        // prefix is the sum of every loaded count, i.e. the block total.
        if (tid == 0) {
            s_lb[TPB] = s_sw[NW - 1];
            s_o16[TPB * PER] = 0;
        }
        __syncthreads();
    }
    const bool scan_ovf = SCAN ? (s_ovf[0] != 0) : false;
    // Exclusive tile_base[t] from the compact form (SCAN only).
    auto scan_at = [&](int t) -> int {
        const int g = (int)((unsigned)t / (unsigned)PER);
        if (!scan_ovf) return s_lb[g] + (int)s_o16[t];
        int v = s_lb[g];
        for (int i = g * PER; i < t; ++i) v += tile_counts_in[i];
        return v;
    };
    if constexpr (SCAN) {
        // Publish for the LATER dispatches (tile sort, fine rasterizer, the
        // async total-pairs observation).  No CTA of THIS grid reads the global
        // array -- every one of them has its own identical LDS copy -- so there
        // is nothing to order and no fence.  Grid-stride, so the publish is
        // complete even when the grid has fewer than ceil(scan_n/TPB) CTAs.
        if (blockIdx.z == 0 && tile_base_out) {
            for (int i = hp_bx * TPB + tid; i < scan_n; i += hp_gx * TPB)
                tile_base_out[i] = scan_at(i);
        }
    }
    // Reads tile_base[] from LDS when the scan was folded in, from global
    // otherwise.  Identical values either way.
    auto tbase = [&](int t) -> int {
        if constexpr (SCAN) return scan_at(t);
        else                return tile_base[t];
    };

    const int STRIDE = TPB * HIPGL_SCATTER_ILP;
    int p_base = tid;
    for (; p_base + STRIDE - TPB < total; p_base += STRIDE) {
        int   t_[HIPGL_SCATTER_ILP], j_[HIPGL_SCATTER_ILP], slot_[HIPGL_SCATTER_ILP];
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            const int p = p_base + u * TPB;
            int j = bin_find_owner_t<HIPGL_BIN_TRIS>(s_off, p);
            int local = p - s_off[j];
            int jnx = s_nx[j];
            int q = (int)((float)local * s_inv[j]);
            int r = local - q * jnx;
            if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
            j_[u] = j;
            t_[u] = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        }
        bool keep_[HIPGL_SCATTER_ILP];
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            keep_[u] = true;
            if (EXACT) {
                const int tt = t_[u];
                const float tpx0 = (float)((tt % tiles_x) * TILE_SIZE) + 0.5f;
                const float tpy0 = (float)((tt / tiles_x) * TILE_SIZE) + 0.5f;
                keep_[u] = !hipgl_tile_edge_reject_f32(s_eq[j_[u]], tpx0, tpy0);
            }
        }
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            if constexpr (HP_BIN_AGG) {
                const int off = hp_agg_reserve(cursor, t_[u], keep_[u]);
                slot_[u] = keep_[u] ? (tbase(t_[u]) + off) : INT_MAX;
            } else {
                slot_[u] = keep_[u] ? (tbase(t_[u]) + atomicAdd(&cursor[t_[u]], 1)) : INT_MAX;
            }
        }
#pragma unroll
        for (int u = 0; u < HIPGL_SCATTER_ILP; ++u) {
            if (!keep_[u]) continue;
            const int slot = slot_[u];
            if (slot < pool_capacity) {
                if (pool_key) {
                    pool_key[slot] = s_key[j_[u]];
                } else {
                    pool_tri[slot] = (int)(uint32_t)s_key[j_[u]];
                    pool_z[slot]   = hipgl_ord2f((uint32_t)(s_key[j_[u]] >> 32));
                }
            } else if (overflow_flag) {
                atomicExch(overflow_flag, 1);
            }
        }
    }
    for (int p = p_base; p < total; p += TPB) {
        int j = bin_find_owner_t<HIPGL_BIN_TRIS>(s_off, p);
        int local = p - s_off[j];
        int jnx = s_nx[j];
        int q = (int)((float)local * s_inv[j]);
        int r = local - q * jnx;
        if (r >= jnx) { q++; r -= jnx; } else if (r < 0) { q--; r += jnx; }
        int t = (s_loy[j] + q) * tiles_x + (s_lox[j] + r);
        if (EXACT) {
            const float tpx0 = (float)((t % tiles_x) * TILE_SIZE) + 0.5f;
            const float tpy0 = (float)((t / tiles_x) * TILE_SIZE) + 0.5f;
            if (hipgl_tile_edge_reject_f32(s_eq[j], tpx0, tpy0)) continue;
        }
        int slot = tbase(t) +
            (HP_BIN_AGG ? hp_agg_reserve(cursor, t, true) : atomicAdd(&cursor[t], 1));
        if (slot < pool_capacity) {
            if (pool_key) {
                pool_key[slot] = s_key[j];
            } else {
                pool_tri[slot] = (int)(uint32_t)s_key[j];
                pool_z[slot]   = hipgl_ord2f((uint32_t)(s_key[j] >> 32));
            }
        } else if (overflow_flag) {
            atomicExch(overflow_flag, 1);
        }
    }

    if (queue.count) {
        const int nq = min(max(*queue.count, 0), tri_count);
        for (int q = hp_bx; q < nq; q += hp_gx) {
            const int source = queue.ids[q];
            __syncthreads();
            if ((unsigned)source >= (unsigned)tri_count) continue;
            if (tid == 0) s_ncrec = binrec[tri_count + source];
            __syncthreads();
            const int lo_x = s_ncrec.xy & 0xFFFFu;
            const int lo_y = s_ncrec.xy >> 16;
            const int qnx  = s_ncrec.nxy & 0xFFFFu;
            const int qny  = s_ncrec.nxy >> 16;
            for (int p = tid; p < qnx * qny; p += TPB) {
                const int tile = (lo_y + p / qnx) * tiles_x + lo_x + p % qnx;
                const int slot = tbase(tile) + atomicAdd(&cursor[tile], 1);
                if (slot < pool_capacity) {
                    if (pool_key) pool_key[slot] = s_ncrec.key;
                    else {
                        pool_tri[slot] = (int)(uint32_t)s_ncrec.key;
                        pool_z[slot] = hipgl_ord2f((uint32_t)(s_ncrec.key >> 32));
                    }
                } else if (overflow_flag) {
                    atomicExch(overflow_flag, 1);
                }
            }
        }
    }
}

template <int HIPGL_BIN_TRIS, int TPB, bool EXACT = false>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scatter_tile_pairs(
    const TriangleSetup* __restrict__ tris,
    const BinRec* __restrict__ binrec,
    int tri_count,
    const int* __restrict__ tile_base,
    int* __restrict__ cursor,
    int* __restrict__ pool_tri,
    float* __restrict__ pool_z,
    unsigned long long* __restrict__ pool_key,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    int pool_capacity,
    int* __restrict__ overflow_flag,
    NearClipQueue queue,
    TileRect rect,
    int ilv_log2)
{
    hp_scatter_tile_pairs_body<HIPGL_BIN_TRIS, TPB, EXACT>(
        tris, binrec, tri_count, tile_base, cursor, pool_tri, pool_z, pool_key,
        tiles_x, tiles_y, fb_width, fb_height, pool_capacity, overflow_flag,
        queue, rect, ilv_log2);
}

// r4: the SCAN-FOLDED entry point.  Identical to the kernel above except that
// `tile_counts` (not `tile_base`) is the input and every CTA derives tile_base
// for itself; block 0 also publishes it for the downstream dispatches.  Only
// the shipped <512,512,non-exact> shape is instantiated -- launch_scatter_tile_pairs
// falls back to the separate scan dispatch for every other arm (see
// scatter_scan_fold_ok), so this costs one extra instantiation, not sixteen.
template <int HIPGL_BIN_TRIS, int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scatter_tile_pairs_scan(
    const TriangleSetup* __restrict__ tris,
    const BinRec* __restrict__ binrec,
    int tri_count,
    const int* __restrict__ tile_counts,
    int* __restrict__ tile_base_out,
    int scan_n,
    int* __restrict__ cursor,
    int* __restrict__ pool_tri,
    float* __restrict__ pool_z,
    unsigned long long* __restrict__ pool_key,
    int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    int pool_capacity,
    int* __restrict__ overflow_flag,
    NearClipQueue queue,
    TileRect rect,
    int ilv_log2)
{
    hp_scatter_tile_pairs_body<HIPGL_BIN_TRIS, TPB, false, true>(
        tris, binrec, tri_count, nullptr, cursor, pool_tri, pool_z, pool_key,
        tiles_x, tiles_y, fb_width, fb_height, pool_capacity, overflow_flag,
        queue, rect, ilv_log2, tile_counts, tile_base_out, scan_n);
}

template <int HIPGL_BIN_TRIS, int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scatter_tile_pairs_scan_batch(CqScatBatch b)
{
    int c = 0;
    for (int i = 1; i < b.n; ++i) c = ((int)blockIdx.x >= b.d[i].bstart) ? i : c;
    const CqScatDesc& d = b.d[c];
    hp_scatter_tile_pairs_body<HIPGL_BIN_TRIS, TPB, false, true>(
        d.tris, d.binrec, d.tri_count, nullptr, d.cursor, d.pool_tri, d.pool_z, d.pool_key,
        d.tiles_x, d.tiles_y, d.fb_w, d.fb_h, d.pool_capacity, d.overflow_flag,
        d.queue, d.rect, d.ilv, d.tile_counts, d.tile_base_out, d.scan_n,
        (int)blockIdx.x - d.bstart, d.grid);
}

template <int HIPGL_BIN_TRIS, int TPB, bool EXACT = false>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scatter_tile_pairs_z2(
    BinScatSlot s0, BinScatSlot s1,
    int tri_count, int tiles_x, int tiles_y, int fb_width, int fb_height,
    int ilv_log2)
{
    const bool z = (blockIdx.z != 0);
    hp_scatter_tile_pairs_body<HIPGL_BIN_TRIS, TPB, EXACT>(
        z ? s1.tris : s0.tris,
        z ? s1.binrec : s0.binrec,
        tri_count,
        z ? s1.tile_base : s0.tile_base,
        z ? s1.cursor : s0.cursor,
        z ? s1.pool_tri : s0.pool_tri,
        z ? s1.pool_z : s0.pool_z,
        z ? s1.pool_key : s0.pool_key,
        tiles_x, tiles_y, fb_width, fb_height,
        z ? s1.pool_capacity : s0.pool_capacity,
        z ? s1.overflow_flag : s0.overflow_flag,
        z ? s1.queue : s0.queue,
        z ? s1.rect : s0.rect, ilv_log2);
}


HIPGL_GLOBAL void kernel_scatter_near_clip_extras(
    const BinRec* __restrict__ binrec, int source_count, NearClipQueue queue,
    const int* __restrict__ tile_base, int* __restrict__ cursor,
    int* __restrict__ pool_tri, float* __restrict__ pool_z,
    unsigned long long* __restrict__ pool_key, int tiles_x, int pool_capacity,
    int* __restrict__ overflow_flag)
{
    __shared__ BinRec shared_rec;
    const int count = min(max(*queue.count, 0), source_count);
    for (int q = blockIdx.x; q < count; q += gridDim.x) {
        const int source = queue.ids[q];
        if ((unsigned)source >= (unsigned)source_count) continue;
        if (threadIdx.x == 0) shared_rec = binrec[source_count + source];
        __syncthreads();
        const int lo_x = shared_rec.xy & 0xFFFFu;
        const int lo_y = shared_rec.xy >> 16;
        const int nx = shared_rec.nxy & 0xFFFFu;
        const int ny = shared_rec.nxy >> 16;
        // Conservative bbox bins are sufficient for these rare pieces. The
        // fine rasterizer applies the same exact coverage rejection as usual.
        for (int p = threadIdx.x; p < nx * ny; p += blockDim.x) {
            const int tile = (lo_y + p / nx) * tiles_x + lo_x + p % nx;
            const int slot = tile_base[tile] + atomicAdd(&cursor[tile], 1);
            if (slot < pool_capacity) {
                if (pool_key) pool_key[slot] = shared_rec.key;
                else {
                    pool_tri[slot] = (int)(uint32_t)shared_rec.key;
                    pool_z[slot] = hipgl_ord2f((uint32_t)(shared_rec.key >> 32));
                }
            } else if (overflow_flag) {
                atomicExch(overflow_flag, 1);
            }
        }
        __syncthreads();
    }
}

HIPGL_GLOBAL void kernel_sort_pool_segments(
    const int* __restrict__ tile_base,
    const int* __restrict__ tile_count,
    int* __restrict__ pool_tri,
    float* __restrict__ pool_z,
    int num_tiles)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tiles) return;
    int b = tile_base[t], n = tile_count[t];
    for (int i = 1; i < n; i++) {
        int   ki = pool_tri[b + i];
        float kz = pool_z[b + i];
        int j = i - 1;
        while (j >= 0 && pool_tri[b + j] > ki) {
            pool_tri[b + j + 1] = pool_tri[b + j];
            pool_z[b + j + 1]   = pool_z[b + j];
            j--;
        }
        pool_tri[b + j + 1] = ki;
        pool_z[b + j + 1]   = kz;
    }
}

static inline int bin_tris_per_block(long long num_tiles) {
    return (num_tiles >= 16384) ? HIPGL_BIN_TRIS_SMALL : HIPGL_BIN_TPB;
}

static inline int hp_env_int(const char* k, int dflt) {
    const char* e = getenv(k);
    return e ? atoi(e) : dflt;
}
#ifndef HP_SCAT_TPB_DEF
#define HP_SCAT_TPB_DEF 128
#endif
#ifndef HP_CNT_TPB_DEF
#define HP_CNT_TPB_DEF 128
#endif
#ifndef HP_SCAT_TPB_EXACT_DEF
#define HP_SCAT_TPB_EXACT_DEF 256
#endif
static inline int hp_scat_tpb() { static const int v = hp_env_int("HP_SCAT_TPB", HP_SCAT_TPB_DEF); return v; }
static inline int hp_scat_tpb_exact() { static const int v = hp_env_int("HP_SCAT_TPB_EXACT", HP_SCAT_TPB_EXACT_DEF); return v; }
static inline int hp_cnt_tpb()  { static const int v = hp_env_int("HP_CNT_TPB",  HP_CNT_TPB_DEF);  return v; }
// Small tile grids (r256, 4096 tiles) kept one triangle per lane, i.e. NT=512.
// That forces bin_block_scan_t onto its LDS Hillis-Steele branch (18 barriers)
// and a 9-step owner search, for the same pairs-per-lane ratio a 64-triangle
// CTA gets from a single-wave shfl scan and a 6-step search. NT is now a knob
// on this path too.
#ifndef HP_SMALL_NT_DEF
#define HP_SMALL_NT_DEF 512
#endif
static inline int hp_small_nt() { static const int v = hp_env_int("HP_SMALL_NT", HP_SMALL_NT_DEF); return v; }

#ifndef HP_BIN_ILV_DEF
#define HP_BIN_ILV_DEF 4
#endif
static inline int hp_bin_ilv_raw() { static const int v = hp_env_int("HP_BIN_ILV", HP_BIN_ILV_DEF); return v; }
#ifndef HP_BIN_ILV_MIN_TILES_DEF
#define HP_BIN_ILV_MIN_TILES_DEF 1024
#endif
static inline int hp_bin_ilv_min_tiles() {
    static const int v = hp_env_int("HP_BIN_ILV_MIN_TILES", HP_BIN_ILV_MIN_TILES_DEF);
    return v;
}
static inline int hp_bin_ilv(long long num_tiles) {
    const int lg = hp_bin_ilv_raw();
    if (lg < 0) return -1;
    if (num_tiles < (long long)hp_bin_ilv_min_tiles()) return -1;
    static int announced = 0;
    if (!announced && getenv("HP_BIN_ILV_DBG")) {
        announced = 1;
        fprintf(stderr, "[bin_ilv] ENGAGED group=%d min_tiles=%d (num_tiles=%lld)\n",
                1 << lg, hp_bin_ilv_min_tiles(), num_tiles);
    }
    return lg;
}


template <int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scan_excl_single(
    const int* __restrict__ in, int* __restrict__ out, int n)
{
    constexpr int NW = TPB / 64;
    __shared__ int s_warp[NW];
    __shared__ int s_carry;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wid  = tid >> 6;
    if (tid == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += TPB) {
        const int i = base + tid;
        const int v = (i < n) ? in[i] : 0;
        int x = v;
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            int y = __shfl_up(x, off, 64);
            if (lane >= off) x += y;
        }
        if (lane == 63) s_warp[wid] = x;
        __syncthreads();
        if (wid == 0) {
            int w = (lane < NW) ? s_warp[lane] : 0;
#pragma unroll
            for (int off = 1; off < NW; off <<= 1) {
                int y = __shfl_up(w, off, 64);
                if (lane >= off) w += y;
            }
            if (lane < NW) s_warp[lane] = w;
        }
        __syncthreads();
        const int wpre = (wid > 0) ? s_warp[wid - 1] : 0;
        if (i < n) out[i] = s_carry + wpre + x - v;
        const int total = s_warp[NW - 1];
        __syncthreads();
        if (tid == 0) s_carry += total;
        __syncthreads();
    }
}

// 4-wide variant: one dwordx4 per thread per chunk, so the block covers 4096
// ints per barrier round instead of 1024 -- a quarter of the LDS barriers and
// fully coalesced 16 B accesses.
template <int TPB>
HIPGL_DEVICE __forceinline__ void hp_scan_excl_single4_body(
    const int* __restrict__ in, int* __restrict__ out, int n)
{
    constexpr int NW = TPB / 64;
    constexpr int CHUNK = TPB * 4;
    __shared__ int s_warp[NW];
    __shared__ int s_carry;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 63;
    const int wid  = tid >> 6;
    if (tid == 0) s_carry = 0;
    __syncthreads();
    for (int base = 0; base < n; base += CHUNK) {
        const int i = base + tid * 4;
        int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        if (i + 3 < n) {
            const int4 v = *reinterpret_cast<const int4*>(in + i);
            a0 = v.x; a1 = v.y; a2 = v.z; a3 = v.w;
        } else {
            if (i + 0 < n) a0 = in[i + 0];
            if (i + 1 < n) a1 = in[i + 1];
            if (i + 2 < n) a2 = in[i + 2];
            if (i + 3 < n) a3 = in[i + 3];
        }
        const int t = a0 + a1 + a2 + a3;
        int x = t;
#pragma unroll
        for (int off = 1; off < 64; off <<= 1) {
            int y = __shfl_up(x, off, 64);
            if (lane >= off) x += y;
        }
        if (lane == 63) s_warp[wid] = x;
        __syncthreads();
        if (wid == 0) {
            int w = (lane < NW) ? s_warp[lane] : 0;
#pragma unroll
            for (int off = 1; off < NW; off <<= 1) {
                int y = __shfl_up(w, off, 64);
                if (lane >= off) w += y;
            }
            if (lane < NW) s_warp[lane] = w;
        }
        __syncthreads();
        const int pre = s_carry + ((wid > 0) ? s_warp[wid - 1] : 0) + x - t;
        const int o0 = pre, o1 = pre + a0, o2 = o1 + a1, o3 = o2 + a2;
        if (i + 3 < n) {
            int4 ov; ov.x = o0; ov.y = o1; ov.z = o2; ov.w = o3;
            *reinterpret_cast<int4*>(out + i) = ov;
        } else {
            if (i + 0 < n) out[i + 0] = o0;
            if (i + 1 < n) out[i + 1] = o1;
            if (i + 2 < n) out[i + 2] = o2;
            if (i + 3 < n) out[i + 3] = o3;
        }
        const int total = s_warp[NW - 1];
        __syncthreads();
        if (tid == 0) s_carry += total;
        __syncthreads();
    }
}

template <int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scan_excl_single4(
    const int* __restrict__ in, int* __restrict__ out, int n)
{
    hp_scan_excl_single4_body<TPB>(in, out, n);
}

template <int TPB>
HIPGL_GLOBAL void __launch_bounds__(TPB) kernel_scan_excl_single4_z2(
    const int* __restrict__ in0, int* __restrict__ out0,
    const int* __restrict__ in1, int* __restrict__ out1, int n)
{
    const bool z = (blockIdx.z != 0);
    hp_scan_excl_single4_body<TPB>(z ? in1 : in0, z ? out1 : out0, n);
}

void launch_scan_excl_single(const int* in, int* out, int n, hipStream_t stream) {
    if (!in || !out || n <= 0) return;
    static const int v4 = hp_env_int("HP_SCAN1_V4", 1);
    if (v4) hipLaunchKernelGGL((kernel_scan_excl_single4<1024>), 1, 1024, 0, stream, in, out, n);
    else    hipLaunchKernelGGL((kernel_scan_excl_single<1024>),  1, 1024, 0, stream, in, out, n);
}

// Collapse the backface rule to a 2-entry truth table over TriangleSetup::
// backfacing, so the binning kernels can apply it without a RenderState.
// This MUST stay in lockstep with triangle_is_culled() above: if the binner
// and the fine rasterizer disagree, the binner wins silently and geometry
// disappears with no overflow flag and no error.
int backface_cull_mask(const RenderState& state) {
    static const bool bin_cull_on = []{
        const char* e = getenv("HIPGL_BIN_BACKFACE");
        return !e || e[0] != '0';
    }();
    if (!bin_cull_on) return 0;
    if (!state.backface_cull_enabled) return 0;
    int m = 0;
    for (int b = 0; b < 2; ++b) {
        const bool back   = (b != 0) != (state.front_face != 0);
        const bool culled = state.cull_face == 2 ||
                            (state.cull_face == 1 ? !back : back);
        if (culled) m |= (1 << b);
    }
    return m;
}

void launch_count_tile_pairs(const TriangleSetup* tris, int tri_count,
                             int* tile_counts, BinRec* binrec,
                             int tiles_x, int tiles_y,
                             int fb_width, int fb_height, hipStream_t stream, NearClipQueue near_clip,
                             int cull_mask, BinSubdraws subs, bool have_rec, TileRect rect) {
    if (!tris || tri_count <= 0 || !tile_counts) return;
    if (tiles_x <= 0 || tiles_y <= 0) return;
    const bool packed_ids = near_clip.ids && near_clip.count;
    static const int fuse_extras_env = hp_env_int("HP_FUSE_EXTRAS", 1);
    const bool fuse_extras = packed_ids && fuse_extras_env != 0;
    const NearClipQueue epi_q = fuse_extras ? near_clip : NearClipQueue{};
    // Two ways to build the per-tile pair counts; pick by tile-grid size.
    const long long num_tiles = (long long)tiles_x * (long long)tiles_y;
    const int ilv = hp_bin_ilv(num_tiles);
#ifndef HIPGL_DIFF_MIN_TILES
#define HIPGL_DIFF_MIN_TILES 65536
#endif
    // The difference-array arm writes binrec through a different kernel that
    // knows nothing about the bound; the scatter still clips, so the image is
    // correct either way, only less of the pool is saved. Unreachable at every
    // shipped resolution (it needs >= 65536 tiles).
    if (num_tiles >= HIPGL_DIFF_MIN_TILES && subs.nsub <= 1) {
        int block = 256, grid = (tri_count + block - 1) / block;
        hipLaunchKernelGGL(kernel_count_stamp_diff, grid, block, 0, stream,
                           tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask);
        if (packed_ids)
            hipLaunchKernelGGL(kernel_count_near_clip_extras, 32, 128, 0, stream,
                tris, tri_count, near_clip, tile_counts, binrec,
                tiles_x, tiles_y, fb_width, fb_height, true);
        hipLaunchKernelGGL(kernel_prefix_rows, tiles_y, 256, 0, stream, tile_counts, tiles_x);
        hipLaunchKernelGGL(kernel_prefix_cols, tiles_x, 256, 0, stream,
                           tile_counts, tiles_x, tiles_y);
        return;
    }
    if (num_tiles >= 16384) {
        const int NT = HIPGL_CNT_TRIS_SMALL;               // 64
        const int grid = (tri_count + NT - 1) / NT;
        switch (hp_cnt_tpb()) {
        case 64:
            hipLaunchKernelGGL((kernel_count_tile_pairs<NT, 64>), grid, 64, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        case 128:
            hipLaunchKernelGGL((kernel_count_tile_pairs<NT, 128>), grid, 128, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        case 256:
            hipLaunchKernelGGL((kernel_count_tile_pairs<NT, 256>), grid, 256, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        default:
            hipLaunchKernelGGL((kernel_count_tile_pairs<NT, HIPGL_BIN_TPB>), grid, HIPGL_BIN_TPB, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        }
    } else {
        const int nt = hp_small_nt();
        const int grid = (tri_count + nt - 1) / nt;
        switch (nt) {
        case 64:
            hipLaunchKernelGGL((kernel_count_tile_pairs<64, 64>), grid, 64, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        case 128:
            hipLaunchKernelGGL((kernel_count_tile_pairs<128, 128>), grid, 128, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        case 256:
            hipLaunchKernelGGL((kernel_count_tile_pairs<256, 256>), grid, 256, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        case 1024:
            hipLaunchKernelGGL((kernel_count_tile_pairs<1024, 1024>), grid, 1024, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        default:
            hipLaunchKernelGGL((kernel_count_tile_pairs<HIPGL_BIN_TPB, HIPGL_BIN_TPB>), grid, HIPGL_BIN_TPB, 0, stream,
                               tris, tri_count, tile_counts, binrec, tiles_x, tiles_y, fb_width, fb_height, packed_ids, cull_mask,
                               epi_q, subs, have_rec, rect, ilv);
            break;
        }
    }
    if (packed_ids && !fuse_extras)
        hipLaunchKernelGGL(kernel_count_near_clip_extras, 32, 128, 0, stream,
            tris, tri_count, near_clip, tile_counts, binrec,
            tiles_x, tiles_y, fb_width, fb_height, false);
}

template <int NT, int TPB, bool EXACT>
static inline void hp_scatter_go(int grid, hipStream_t stream,
                                 const TriangleSetup* tris, const BinRec* binrec, int tri_count,
                                 const int* tile_base, int* cursor,
                                 int* pool_tri, float* pool_z,
                                 unsigned long long* pool_key,
                                 int tiles_x, int tiles_y, int fb_width, int fb_height,
                                 int pool_capacity, int* overflow_flag, NearClipQueue queue,
                                 TileRect rect, int ilv) {
    hipLaunchKernelGGL((kernel_scatter_tile_pairs<NT, TPB, EXACT>), grid, TPB, 0, stream,
                       tris, binrec, tri_count, tile_base, cursor, pool_tri, pool_z, pool_key,
                       tiles_x, tiles_y, fb_width, fb_height, pool_capacity, overflow_flag, queue,
                       rect, ilv);
}

template <int NT, bool EXACT>
static inline void hp_scatter_tpb(int tpb, int grid, hipStream_t stream,
                                  const TriangleSetup* tris, const BinRec* binrec, int tri_count,
                                  const int* tile_base, int* cursor,
                                  int* pool_tri, float* pool_z,
                                  unsigned long long* pool_key,
                                  int tiles_x, int tiles_y, int fb_width, int fb_height,
                                  int pool_capacity, int* overflow_flag, NearClipQueue queue,
                                  TileRect rect, int ilv) {
    switch (tpb) {
    case 64:
        hp_scatter_go<NT, 64, EXACT>(grid, stream, tris, binrec, tri_count, tile_base, cursor,
                                     pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                                     fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
        break;
    case 128:
        hp_scatter_go<NT, 128, EXACT>(grid, stream, tris, binrec, tri_count, tile_base, cursor,
                                      pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                                      fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
        break;
    case 256:
        hp_scatter_go<NT, 256, EXACT>(grid, stream, tris, binrec, tri_count, tile_base, cursor,
                                      pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                                      fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
        break;
    default:
        hp_scatter_go<NT, HIPGL_BIN_TPB, EXACT>(grid, stream, tris, binrec, tri_count, tile_base, cursor,
                                      pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                                      fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
        break;
    }
}

void launch_cq_scat(const CqScatDesc* const* r, int n, hipStream_t stream)
{
    if (n <= 0) return;
    if (n == 1) {
        const CqScatDesc& d = *r[0];
        hipLaunchKernelGGL((kernel_scatter_tile_pairs_scan<HIPGL_BIN_TPB, HIPGL_BIN_TPB>),
                           d.grid, HIPGL_BIN_TPB, 0, stream,
                           d.tris, d.binrec, d.tri_count, d.tile_counts, d.tile_base_out, d.scan_n,
                           d.cursor, d.pool_tri, d.pool_z, d.pool_key, d.tiles_x, d.tiles_y,
                           d.fb_w, d.fb_h, d.pool_capacity, d.overflow_flag, d.queue, d.rect,
                           d.ilv);
        return;
    }
    CqScatBatch b;
    memset(&b, 0, sizeof(b));
    int tot = 0;
    for (int i = 0; i < n; ++i) { b.d[i] = *r[i]; b.d[i].bstart = tot; tot += b.d[i].grid; }
    b.n = n;
    hipLaunchKernelGGL((kernel_scatter_tile_pairs_scan_batch<HIPGL_BIN_TPB, HIPGL_BIN_TPB>),
                       tot, HIPGL_BIN_TPB, 0, stream, b);
}

bool scatter_scan_fold_ok(int tiles_x, int tiles_y, int tri_count,
                          bool multisample, int scan_n) {
    static const int on = hp_env_int("HP_SCAT_SCAN", 1);
    if (!on) return false;
    if (tri_count <= 0 || tiles_x <= 0 || tiles_y <= 0) return false;
    if (scan_n <= 0 || scan_n > HP_SCAT_SCAN_CAP - 3) return false;
    if (scan_n > HIPGL_BIN_TPB * HP_SCAT_SCAN_PER + 1) return false;   // single-chunk scan bound (+ tail)
    const long long num_tiles = (long long)tiles_x * (long long)tiles_y;
    if (bin_tris_per_block(num_tiles) == HIPGL_BIN_TRIS_SMALL) return false;
    static const long long exact_min_tiles = []{
        const char* e = getenv("HIPGL_EXACT_MIN_TILES");
        return e ? atoll(e) : 65536LL;
    }();
    if (!multisample && (num_tiles >= exact_min_tiles)) return false;   // EXACT arm
    if (hp_small_nt() != HIPGL_BIN_TPB) return false;                   // non-default arm
    return true;
}

void launch_scatter_tile_pairs(const TriangleSetup* tris, const BinRec* binrec,
                               int tri_count,
                               const int* tile_base, int* cursor,
                               int* pool_tri, float* pool_z,
                               unsigned long long* pool_key,
                               int tiles_x, int tiles_y, int fb_width, int fb_height,
                               int pool_capacity, int* overflow_flag, hipStream_t stream, bool multisample,
                               NearClipQueue near_clip, TileRect rect,
                               const int* tile_counts_in, int* tile_base_out, int scan_n) {
    if (CqChainRec* cq = cq_rec_slot()) {
        if ((cq->want & 2) &&
            !(tris && tri_count > 0 && tile_counts_in && tile_base_out && scan_n > 0 &&
              !(near_clip.ids && near_clip.count && hp_env_int("HP_FUSE_EXTRAS", 1) == 0)))
            cq_rec_miss(cq);
    }
    if (!tris || tri_count <= 0) return;
    // r4: the folded-scan arm.  The caller has already skipped the scan
    // dispatch, so this MUST be the shape scatter_scan_fold_ok accepted.
    if (tile_counts_in && tile_base_out && scan_n > 0) {
        const long long nt_f = (long long)tiles_x * (long long)tiles_y;
        static const int fx_env = hp_env_int("HP_FUSE_EXTRAS", 1);
        const bool fx = near_clip.ids && near_clip.count && fx_env != 0;
        const NearClipQueue q_f = fx ? near_clip : NearClipQueue{};
        const int grid_f = (tri_count + HIPGL_BIN_TPB - 1) / HIPGL_BIN_TPB;
        if (CqChainRec* cq = cq_rec_slot()) {
            if ((cq->want & 2) && !(near_clip.ids && near_clip.count && !fx)) {
                CqScatDesc& d = cq->s;
                memset(&d, 0, sizeof(d));
                d.tris = tris; d.binrec = binrec; d.tile_counts = tile_counts_in;
                d.tile_base_out = tile_base_out; d.cursor = cursor;
                d.pool_tri = pool_tri; d.pool_z = pool_z; d.pool_key = pool_key;
                d.overflow_flag = overflow_flag; d.queue = q_f; d.rect = rect;
                d.tri_count = tri_count; d.scan_n = scan_n;
                d.tiles_x = tiles_x; d.tiles_y = tiles_y; d.fb_w = fb_width; d.fb_h = fb_height;
                d.pool_capacity = pool_capacity; d.ilv = hp_bin_ilv(nt_f);
                d.grid = grid_f; d.bstart = 0;
                cq->got |= 2;
                return;
            }
        }
        hipLaunchKernelGGL((kernel_scatter_tile_pairs_scan<HIPGL_BIN_TPB, HIPGL_BIN_TPB>),
                           grid_f, HIPGL_BIN_TPB, 0, stream,
                           tris, binrec, tri_count, tile_counts_in, tile_base_out, scan_n,
                           cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                           fb_width, fb_height, pool_capacity, overflow_flag, q_f, rect,
                           hp_bin_ilv(nt_f));
        if (near_clip.ids && near_clip.count && !fx)
            hipLaunchKernelGGL(kernel_scatter_near_clip_extras, 32, 128, 0, stream,
                binrec, tri_count, near_clip, tile_base_out, cursor, pool_tri, pool_z,
                pool_key, tiles_x, pool_capacity, overflow_flag);
        return;
    }
    const long long num_tiles = (long long)tiles_x * (long long)tiles_y;
    const int bt = bin_tris_per_block(num_tiles);
    static const long long exact_min_tiles = []{
        const char* e = getenv("HIPGL_EXACT_MIN_TILES");
        return e ? atoll(e) : 65536LL;
    }();
    // Center-only rejection would discard triangles covering an MSAA edge sample.
    const bool exact = !multisample && (num_tiles >= exact_min_tiles);
    static const int fuse_extras_env2 = hp_env_int("HP_FUSE_EXTRAS", 1);
    const bool fuse_extras = near_clip.ids && near_clip.count && fuse_extras_env2 != 0;
    const NearClipQueue queue = fuse_extras ? near_clip : NearClipQueue{};
    const int ilv = hp_bin_ilv(num_tiles);
    if (bt == HIPGL_BIN_TRIS_SMALL) {
        const int grid = (tri_count + HIPGL_BIN_TRIS_SMALL - 1) / HIPGL_BIN_TRIS_SMALL;
        if (exact)
            hp_scatter_tpb<HIPGL_BIN_TRIS_SMALL, true>(hp_scat_tpb_exact(), grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
        else
            hp_scatter_tpb<HIPGL_BIN_TRIS_SMALL, false>(hp_scat_tpb(), grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
    } else {
        const int nt = hp_small_nt();
        const int grid = (tri_count + nt - 1) / nt;
        switch (nt) {
        case 64:
            hp_scatter_go<64, 64, false>(grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
            break;
        case 128:
            hp_scatter_go<128, 128, false>(grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
            break;
        case 256:
            hp_scatter_go<256, 256, false>(grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
            break;
        case 1024:
            hp_scatter_go<1024, 1024, false>(grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
            break;
        default:
            hp_scatter_go<HIPGL_BIN_TPB, HIPGL_BIN_TPB, false>(grid, stream, tris, binrec, tri_count,
                tile_base, cursor, pool_tri, pool_z, pool_key, tiles_x, tiles_y,
                fb_width, fb_height, pool_capacity, overflow_flag, queue, rect, ilv);
            break;
        }
    }
    if (near_clip.ids && near_clip.count && !fuse_extras)
        hipLaunchKernelGGL(kernel_scatter_near_clip_extras, 32, 128, 0, stream,
            binrec, tri_count, near_clip, tile_base, cursor, pool_tri, pool_z,
            pool_key, tiles_x, pool_capacity, overflow_flag);
}

bool launch_count_tile_pairs_z2(BinCntSlot s0, BinCntSlot s1, int tri_count,
                                int tiles_x, int tiles_y, int fb_width, int fb_height,
                                BinSubdraws subs, hipStream_t stream) {
    return launch_count_tile_pairs_z2_mc(s0, s1, tri_count, tiles_x, tiles_y,
                                         fb_width, fb_height, subs, stream, nullptr);
}

bool launch_count_tile_pairs_z2_mc(BinCntSlot s0, BinCntSlot s1, int tri_count,
                                   int tiles_x, int tiles_y, int fb_width, int fb_height,
                                   BinSubdraws subs, hipStream_t stream,
                                   const FbMaskClearDesc* mc_in) {
    if (!s0.tris || !s1.tris || tri_count <= 0) return false;
    if (!s0.tile_counts || !s1.tile_counts) return false;
    if (tiles_x <= 0 || tiles_y <= 0) return false;
    static const int fuse_extras_env = hp_env_int("HP_FUSE_EXTRAS", 1);
    if (fuse_extras_env == 0) return false;           // would need trailing dispatches
    const long long num_tiles = (long long)tiles_x * (long long)tiles_y;
    if (num_tiles >= HIPGL_DIFF_MIN_TILES && subs.nsub <= 1) return false;  // diff-array arm
    const int ilv = hp_bin_ilv(num_tiles);
    int NT = 0, TPB = 0;
    if (num_tiles >= 16384) {
        NT = HIPGL_CNT_TRIS_SMALL;
        TPB = hp_cnt_tpb();
        if (TPB != 64 && TPB != 128 && TPB != 256) TPB = HIPGL_BIN_TPB;
    } else {
        NT = hp_small_nt();
        TPB = NT;
        if (NT != 64 && NT != 128 && NT != 256 && NT != 1024 && NT != HIPGL_BIN_TPB)
            return false;
    }
    const int grid_x = (tri_count + NT - 1) / NT;
    FbMaskClearDesc mcv{};
    if (mc_in && mc_in->total > 0 && mc_in->color && mc_in->depth && mc_in->mask && mc_in->depth_in)
        mcv = *mc_in;
#define HP_CNT_Z2_GO(NTv, TPBv)                                                        \
    do { const int _ex = mcv.total > 0 ? ((mcv.total + (TPBv) - 1) / (TPBv)) : 0;       \
         dim3 grid((unsigned)(grid_x + _ex), 1u, 2u);                                   \
         hipLaunchKernelGGL((kernel_count_tile_pairs_z2<NTv, TPBv>), grid, TPBv, 0, stream, \
                       s0, s1, tri_count, tiles_x, tiles_y, fb_width, fb_height, subs, ilv, \
                       grid_x, mcv); } while (0)
    if (num_tiles >= 16384) {
        switch (TPB) {
        case 64:  HP_CNT_Z2_GO(HIPGL_CNT_TRIS_SMALL, 64);  break;
        case 128: HP_CNT_Z2_GO(HIPGL_CNT_TRIS_SMALL, 128); break;
        case 256: HP_CNT_Z2_GO(HIPGL_CNT_TRIS_SMALL, 256); break;
        default:  HP_CNT_Z2_GO(HIPGL_CNT_TRIS_SMALL, HIPGL_BIN_TPB); break;
        }
    } else {
        switch (NT) {
        case 64:   HP_CNT_Z2_GO(64, 64);     break;
        case 128:  HP_CNT_Z2_GO(128, 128);   break;
        case 256:  HP_CNT_Z2_GO(256, 256);   break;
        case 1024: HP_CNT_Z2_GO(1024, 1024); break;
        default:   HP_CNT_Z2_GO(HIPGL_BIN_TPB, HIPGL_BIN_TPB); break;
        }
    }
#undef HP_CNT_Z2_GO
    return true;
}

static inline int hp_scan1_tpb(int n) {
    return (n <= 512 * 4) ? 512 : 1024;
}

bool launch_scan_excl_single_z2(const int* in0, int* out0,
                                const int* in1, int* out1, int n, hipStream_t stream) {
    if (!in0 || !out0 || !in1 || !out1 || n <= 0) return false;
    static const int v4 = hp_env_int("HP_SCAN1_V4", 1);
    if (!v4) return false;                        // the non-v4 arm keeps its own kernel
    dim3 grid(1u, 1u, 2u);
    switch (hp_scan1_tpb(n)) {
    case 512:
        hipLaunchKernelGGL((kernel_scan_excl_single4_z2<512>), grid, 512, 0, stream,
                           in0, out0, in1, out1, n);
        break;
    default:
        hipLaunchKernelGGL((kernel_scan_excl_single4_z2<1024>), grid, 1024, 0, stream,
                           in0, out0, in1, out1, n);
        break;
    }
    return true;
}

bool launch_scatter_tile_pairs_z2(BinScatSlot s0, BinScatSlot s1, int tri_count,
                                  int tiles_x, int tiles_y, int fb_width, int fb_height,
                                  bool multisample, hipStream_t stream) {
    if (!s0.tris || !s1.tris || tri_count <= 0) return false;
    static const int fuse_extras_env2 = hp_env_int("HP_FUSE_EXTRAS", 1);
    if (fuse_extras_env2 == 0) return false;
    const long long num_tiles = (long long)tiles_x * (long long)tiles_y;
    const int bt = bin_tris_per_block(num_tiles);
    static const long long exact_min_tiles = []{
        const char* e = getenv("HIPGL_EXACT_MIN_TILES");
        return e ? atoll(e) : 65536LL;
    }();
    const bool exact = !multisample && (num_tiles >= exact_min_tiles);
    const int ilv = hp_bin_ilv(num_tiles);
#define HP_SCAT_Z2_GO(NTv, TPBv, EXv)                                                     \
    do { dim3 g((unsigned)((tri_count + (NTv) - 1) / (NTv)), 1u, 2u);                      \
         hipLaunchKernelGGL((kernel_scatter_tile_pairs_z2<NTv, TPBv, EXv>), g, TPBv, 0,    \
                            stream, s0, s1, tri_count, tiles_x, tiles_y,                   \
                            fb_width, fb_height, ilv); } while (0)
    if (bt == HIPGL_BIN_TRIS_SMALL) {
        const int tpb = exact ? hp_scat_tpb_exact() : hp_scat_tpb();
        if (exact) {
            switch (tpb) {
            case 64:  HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 64, true);  break;
            case 128: HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 128, true); break;
            case 256: HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 256, true); break;
            default:  HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, HIPGL_BIN_TPB, true); break;
            }
        } else {
            switch (tpb) {
            case 64:  HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 64, false);  break;
            case 128: HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 128, false); break;
            case 256: HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, 256, false); break;
            default:  HP_SCAT_Z2_GO(HIPGL_BIN_TRIS_SMALL, HIPGL_BIN_TPB, false); break;
            }
        }
    } else {
        const int nt = hp_small_nt();
        // Same reason as in launch_count_tile_pairs_z2: no uninstantiated nt.
        if (nt != 64 && nt != 128 && nt != 256 && nt != 1024 && nt != HIPGL_BIN_TPB)
            return false;
        switch (nt) {
        case 64:   HP_SCAT_Z2_GO(64, 64, false);     break;
        case 128:  HP_SCAT_Z2_GO(128, 128, false);   break;
        case 256:  HP_SCAT_Z2_GO(256, 256, false);   break;
        case 1024: HP_SCAT_Z2_GO(1024, 1024, false); break;
        default:   HP_SCAT_Z2_GO(HIPGL_BIN_TPB, HIPGL_BIN_TPB, false); break;
        }
    }
#undef HP_SCAT_Z2_GO
    return true;
}

void launch_sort_pool_segments(const int* tile_base, const int* tile_count,
                               int* pool_tri, float* pool_z, int num_tiles,
                               hipStream_t stream) {
    if (num_tiles <= 0) return;
    int block = 128, grid = (num_tiles + block - 1) / block;
    hipLaunchKernelGGL(kernel_sort_pool_segments, grid, block, 0, stream,
                       tile_base, tile_count, pool_tri, pool_z, num_tiles);
}

HIPGL_GLOBAL void kernel_pool_to_tilebins(
    const int* __restrict__ tile_base,
    const int* __restrict__ tile_count,
    const int* __restrict__ pool_tri,
    const float* __restrict__ pool_z,
    TileBin* __restrict__ tile_bins,
    int num_tiles, int g_pool_cap)
{
    const int t = blockIdx.x;
    if (t >= num_tiles) return;
    int n = tile_count[t];
    int keep = (n < MAX_TRIANGLES_PER_TILE) ? n : MAX_TRIANGLES_PER_TILE;
    // tile_base/tile_count are the TRUE scanned values and are not bounded by
    // the pool; the scatter clamps its writes, so clamp the read span too.
    int b = tile_base[t];
    if (b > g_pool_cap) b = g_pool_cap;
    int avail = g_pool_cap - b;
    if (keep > avail) keep = avail;
    for (int i = threadIdx.x; i < keep; i += blockDim.x) {
        tile_bins[t].triangle_indices[i] = pool_tri[b + i];
        tile_bins[t].triangle_z_min[i]   = pool_z[b + i];
    }
    if (threadIdx.x == 0) tile_bins[t].count = n;   // keep the true count
}

void launch_pool_to_tilebins(const int* tile_base, const int* tile_count,
                             const int* pool_tri, const float* pool_z,
                             TileBin* tile_bins, int num_tiles, int pool_cap, hipStream_t stream) {
    if (num_tiles <= 0) return;
    hipLaunchKernelGGL(kernel_pool_to_tilebins, num_tiles, 256, 0, stream,
                       tile_base, tile_count, pool_tri, pool_z, tile_bins, num_tiles, pool_cap);
}

// end[t] = base[t] + count[t]. Segment bounds for the segmented radix sort.
HIPGL_GLOBAL void kernel_tile_ends(const int* __restrict__ base,
                                   const int* __restrict__ count,
                                   int* __restrict__ ends, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ends[i] = base[i] + count[i];
}
void launch_tile_ends(const int* base, const int* count, int* ends, int n, hipStream_t stream) {
    if (n <= 0) return;
    int b = 256, g = (n + b - 1) / b;
    hipLaunchKernelGGL(kernel_tile_ends, g, b, 0, stream, base, count, ends, n);
}

HIPGL_GLOBAL void kernel_tile_bounds_clamped(const int* __restrict__ base,
                                             const int* __restrict__ count,
                                             int* __restrict__ base_c,
                                             int* __restrict__ ends_c,
                                             int n, int cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int b = base[i];
    if (b > cap) b = cap;
    long long e = (long long)base[i] + (long long)count[i];
    if (e > (long long)cap) e = cap;
    if (e < (long long)b) e = b;
    base_c[i] = b;
    ends_c[i] = (int)e;
}
void launch_tile_bounds_clamped(const int* base, const int* count,
                                int* base_c, int* ends_c,
                                int n, int cap, hipStream_t stream) {
    if (n <= 0) return;
    int b = 256, g = (n + b - 1) / b;
    hipLaunchKernelGGL(kernel_tile_bounds_clamped, g, b, 0, stream,
                       base, count, base_c, ends_c, n, cap);
}

// Mark the footprint of whatever was just rasterized: mask[i] = depth[i] < clear.
// Stands in for MuJoCo's stencil pass, which renders the mirror geom into the
// stencil buffer to confine the reflection to its silhouette. hip has no stencil
// buffer (framebuffer.cpp leaves it null), so the panel is drawn alone into a
// cleared depth buffer and its coverage read back as the mask.
HIPGL_GLOBAL void kernel_depth_to_mask(const float* __restrict__ depth,
                                       uint8_t* __restrict__ mask,
                                       float clear_value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) mask[i] = (depth[i] < clear_value) ? 1u : 0u;
}
void launch_depth_to_mask(const float* depth, uint8_t* mask, float clear_value,
                          int n, hipStream_t stream) {
    if (n <= 0) return;
    int b = 256, g = (n + b - 1) / b;
    hipLaunchKernelGGL(kernel_depth_to_mask, g, b, 0, stream, depth, mask, clear_value, n);
}

} // namespace kernels
} // namespace hipgl
