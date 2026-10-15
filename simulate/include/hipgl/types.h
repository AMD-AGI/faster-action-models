#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "texture_mip.h"
#include <cstdint>
#include <cstddef>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define HIPGL_HOST_DEVICE __host__ __device__
#define HIPGL_DEVICE __device__
#define HIPGL_HOST __host__
#define HIPGL_GLOBAL __global__
#else
#define HIPGL_HOST_DEVICE
#define HIPGL_DEVICE
#define HIPGL_HOST
#define HIPGL_GLOBAL
#endif

namespace hipgl {

// ── Tile-based rasterization constants ──────────────────────────────────────
constexpr int TILE_SIZE = 16;
#ifndef HIPGL_MAX_TRIANGLES_PER_TILE
#define HIPGL_MAX_TRIANGLES_PER_TILE 16384
#endif
constexpr int MAX_TRIANGLES_PER_TILE = HIPGL_MAX_TRIANGLES_PER_TILE;
constexpr int MAX_LIGHTS = 8;
constexpr int MAX_TEXTURES = 64;

constexpr float DEFAULT_MIN_TRI_AREA = 0.25f;

constexpr int TEXCOORD_COMPONENTS = 3;

// Components per MESH UV as stored on disk and in the model descriptor. This
// is a DIFFERENT quantity from TEXCOORD_COMPONENTS and must never track it:
// mjModel.mesh_texcoord and every OBJ/MSH file are 2D, so widening the vertex
// texcoord must not change how the source is indexed. Named so that an audit
// for stride sites can tell a correct 2 from a stale one.
constexpr int MESH_UV_COMPONENTS = 2;

// Per-fragment interpolated attribute layout, derived so that widening the
// vertex texcoord shifts the colour slots automatically instead of silently
// overlapping them:
//   [world_pos(3), world_normal(3), texcoord(TEXCOORD_COMPONENTS), colour(4)]
constexpr int FRAG_ATTR_POS      = 0;
constexpr int FRAG_ATTR_NORMAL   = 3;
constexpr int FRAG_ATTR_TEXCOORD = 6;
constexpr int FRAG_ATTR_COLOR    = FRAG_ATTR_TEXCOORD + TEXCOORD_COMPONENTS;
// Global vertex index of the winning fragment's triangle, stored as a float
// (the capture buffer is float; indices here are far below 2^24 so the
// conversion is exact). -1 where no fragment won. Names the surface that
// actually won a pixel, with no unprojection and no nearest-geom search.
constexpr int FRAG_ATTR_VERTEX   = FRAG_ATTR_COLOR + 4;
constexpr int FRAG_ATTR_COUNT    = FRAG_ATTR_VERTEX + 1;
constexpr int MAX_BUFFERS = 256;
constexpr int MAX_MATRIX_STACK_DEPTH = 32;

// ── Depth functions (match GL enums) ────────────────────────────────────────
enum DepthFunc : int {
    DEPTH_NEVER    = 0,
    DEPTH_LESS     = 1,
    DEPTH_EQUAL    = 2,
    DEPTH_LEQUAL   = 3,
    DEPTH_GREATER  = 4,
    DEPTH_NOTEQUAL = 5,
    DEPTH_GEQUAL   = 6,
    DEPTH_ALWAYS   = 7,
};

// ── Blend factors ───────────────────────────────────────────────────────────
enum BlendFactor : int {
    BLEND_ZERO                = 0,
    BLEND_ONE                 = 1,
    BLEND_SRC_ALPHA           = 2,
    BLEND_ONE_MINUS_SRC_ALPHA = 3,
    BLEND_DST_ALPHA           = 4,
    BLEND_ONE_MINUS_DST_ALPHA = 5,
    BLEND_SRC_COLOR           = 6,
    BLEND_ONE_MINUS_SRC_COLOR = 7,
    BLEND_DST_COLOR           = 8,
    BLEND_ONE_MINUS_DST_COLOR = 9,
};

// ── Primitive types ─────────────────────────────────────────────────────────
enum PrimitiveType : int {
    PRIM_TRIANGLES      = 0,
    PRIM_TRIANGLE_STRIP = 1,
    PRIM_TRIANGLE_FAN   = 2,
    PRIM_LINES          = 3,
    PRIM_LINE_STRIP     = 4,
    PRIM_POINTS         = 5,
    PRIM_QUADS          = 6,
};

// ── Texture formats ─────────────────────────────────────────────────────────
enum TexFormat : int {
    TEX_RGBA8    = 0,
    TEX_RGB8     = 1,
    TEX_RGBAF32  = 2,
    TEX_DEPTH32F = 3,
    TEX_R8       = 4,
};

// ── Buffer usage hints ──────────────────────────────────────────────────────
enum BufferUsage : int {
    USAGE_STATIC_DRAW  = 0,
    USAGE_DYNAMIC_DRAW = 1,
    USAGE_STREAM_DRAW  = 2,
};

// ── Vertex attribute ────────────────────────────────────────────────────────
struct VertexAttrib {
    int      index;
    int      size;       // 1,2,3,4
    int      type;       // 0=float, 1=int, 2=unsigned byte
    bool     normalized;
    int      stride;
    size_t   offset;
    uint32_t buffer_id;
    int32_t  enabled;
};

// ── Light ───────────────────────────────────────────────────────────────────
struct Light {
    float position[4];
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float direction[3];
    float spot_cutoff;
    float spot_exponent;
    float constant_atten;
    float linear_atten;
    float quadratic_atten;
    int32_t enabled;
};

// ── Material ────────────────────────────────────────────────────────────────
struct Material {
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float emission[4];
    float shininess;
};

// ── Framebuffer descriptor ──────────────────────────────────────────────────
struct FramebufferDesc {
    int width;
    int height;
    int color_format;   // TexFormat
    bool has_depth;
    bool has_stencil;
    int  samples;       // MSAA sample count (1 = no MSAA)
};

// ── Device-side framebuffer pointers (passed to kernels) ────────────────────
struct DeviceFramebuffer {
    float*    color_buffer;    // RGBA interleaved, 4 floats per pixel
    float*    depth_buffer;    // 1 float per pixel
    uint8_t*  stencil_buffer;  // 1 byte per pixel (optional)
    float*    attr_buffer;     // optional: 12 floats/px [world_pos(3), world_normal(3), texcoord(2), base_color(4)]
    float*    oit_accum;       // OIT: [width * height * 4] weighted RGBA accumulation
    float*    oit_revealage;   // OIT: [width * height] product of (1-alpha)
    int       width;
    int       height;
    int       pitch;           // color_buffer stride in floats per row (width*4)
    // Optional native 4x MSAA attachments, pixel-major then sample. Ordinary
    // color/depth are resolved views; these survive every draw in a frame.
    uint32_t* sample_color;    // RGBA8 [width * height * 4]
    float*    sample_depth;    // float [width * height * 4]
};

// ── Texture state (passed to fragment stage) ─────────────────────────────────
struct TextureState {
    const unsigned char* tex_data;
    const int*           tex_width;
    const int*           tex_height;
    const int*           tex_offset;
    int                  ntexture;
    // [ntexture] 1 = GL_TEXTURE_CUBE_MAP, 0 = GL_TEXTURE_2D. May be null, in
    // which case every texture is treated as 2D (the pre-cube behaviour).
    // Cube textures store 6 square faces of tex_width x tex_width back to back
    // at tex_offset, in GL order +X, -X, +Y, -Y, +Z, -Z.
    const int*           tex_iscube;
    // [ntexture * MAX_TEXTURE_MIP_LEVELS]. Null preserves level-zero sampling;
    // entries after the final 1x1 level repeat its metadata.
    const TextureMipLevel* tex_mip;
};

// ── Transformed vertex (output of vertex stage, input to rasterizer) ────────
struct alignas(64) TransformedVertex {
    float screen_z;        // screen space z (depth)
    float inv_w;           // 1/w for perspective-correct interpolation
    float color[4];        // vertex color (RGBA)
    float lit_color[4];    // Per-vertex Gouraud-shaded color (diffuse+ambient+specular pre-computed)
    // 2 components -> 3. Cube maps are handled by PACKING the face index and t
    // into the v channel at geometry time rather than carrying a third
    // coordinate for 2D maps; see the historical note below.
    float texcoord[TEXCOORD_COMPONENTS];
    int   texid;           // -1 for untextured geometry
    float mat_specular;    // MuJoCo specular coefficient [0,1] for per-fragment shading
    float mat_shininess;   // GL shininess (MuJoCo value * 128) for per-fragment shading

    // ── COLD TAIL ───────────────────────────────────────────────────────────
    // Read ONLY by paths that are inert in the shipped render:
    //   world_pos    -> RenderState::clip_plane_enabled (off) and the
    //                   fb.attr_buffer diagnostic capture (null), plus the
    //                   per-FRAGMENT Blinn-Phong arm (gouraud_shading=1 kills it)
    //   world_normal -> same three
    //   spec_color   -> per-FRAGMENT Blinn-Phong arm only
    //   mat_emission -> per-FRAGMENT Blinn-Phong arm only
    //   clip_zw / screen_x / screen_y -> the LEGACY vertex_transform ->
    //                   triangle_setup path (separate TransformedVertex array,
    //                   not TriangleSetup::v) and the pybind debug dump.
    // The fused Gouraud kernel therefore does not store them at all when the
    // runtime slim gate is armed; the values still live in registers for
    // compute_gouraud_lighting(), which reads the staged local, not memory.
    float mat_emission;    // MuJoCo emission coefficient [0,1]
    float world_pos[3];    // eye-space position (field name kept for compatibility)
    float world_normal[3]; // world space normal (for lighting)
    float spec_color[3];   // separate specular color (GL_SEPARATE_SPECULAR_COLOR)
    // Only clip.z and clip.w survive. AUDITED: the single reader of clip space
    // anywhere in the tree is the near-plane test `clip.z + clip.w < 0`.
    // The ADD IS DELIBERATELY LEFT AT THE TEST SITE rather than precomputed
    // into one float here: summing at store time lets the compiler contract the
    // sum with the MVP dot-product FMAs, which MEASURABLY moved pixels.
    float clip_zw[2];      // {clip.z, clip.w}; near plane iff z + w < 0
    float screen_x;        // screen space x
    float screen_y;        // screen space y
};

// ── Triangle setup data (for rasterizer) ────────────────────────────────────
// FIELD ORDER IS A MEMORY-LAYOUT DECISION, not cosmetic. The fine rasterizer's
// inner loop reads, for EVERY triangle in a tile bin, only degenerate,
// backfacing, edge_eq and edge_bias -- and the edge test rejects the large
// majority before anything else is touched. With v[3] (3 x 136 B) leading the
// struct those hot fields sat at bytes 412..484, so the reject path straddled
// the tail of a 484 B object. Hoisting them to the front packs the whole
// reject path into the first 64 B cache line; the shading fields follow and
// are pulled in only by the fragments that survive.
// Pure reordering: every access in the tree is by field name, so this is
// source-compatible (nothing in the tree relies on TriangleSetup's layout --
// no offsetof, no memcpy of a prefix, and debug_read_triangle copies whole
// structs).
struct TriangleSetup {
    float edge_eq[3][3];   // edge equations: A*x + B*y + C for each edge
    float edge_bias[3];    // top-left fill rule bias per edge (0 for top/left, negative otherwise)
    // Packed line-0 duplicates.  WRITE THEM WHEREVER THE CANONICAL FIELD IS
    // WRITTEN -- hipgl_set_flags() / hipgl_mark_degenerate() / hipgl_set_bbox()
    // are the only legal writers of any of these six words.
    int   hot_flags;       // bit0 = degenerate, bit1 = backfacing
    int   hot_bbox_y;      // (i16)bbox_min_y | (i16)bbox_max_y << 16
    int   hot_bbox_x;      // (i16)bbox_min_x | (i16)bbox_max_x << 16
    int   _hot_pad;        // keeps the prefilter's read set at exactly 64 B
    // ---- line 1: EXACTLY the micro-draw bin chain's read set.  micro_draw.hip
    // reads degenerate, backfacing, the four bbox ints and z_ord and nothing
    // else; contiguous at bytes 64..91 that is ONE line instead of the two the
    // old split across the 64 B boundary forced.  It is also the non-BinRec
    // binning count path's read set, and the producer stores the whole group
    // as dwordx4 + dwordx4.
    int   degenerate;
    int   backfacing;
    int   bbox_min_x, bbox_max_x;
    int   bbox_min_y, bbox_max_y;
    unsigned int z_ord;    // ord32(min(v[0..2].screen_z)); valid iff !degenerate
    int first_vertex;   // (line 1, byte 84) diagnostic only
    int  _pad_core[8];
    TransformedVertex v[3];
};

static_assert(offsetof(TriangleSetup, edge_eq)    ==   0, "TriangleSetup layout");
static_assert(offsetof(TriangleSetup, hot_flags)  ==  48, "TriangleSetup layout");
static_assert(offsetof(TriangleSetup, hot_bbox_x) ==  56, "TriangleSetup layout");
static_assert(offsetof(TriangleSetup, degenerate) ==  64, "TriangleSetup layout");
static_assert(offsetof(TriangleSetup, z_ord)      ==  88, "TriangleSetup layout");
static_assert(offsetof(TriangleSetup, v)          == 128, "TriangleSetup layout");
static_assert(sizeof(TriangleSetup)               == 512, "TriangleSetup layout");

HIPGL_HOST_DEVICE inline int hipgl_pack_bbox_i16(int lo, int hi) {
    lo = (lo < -32768) ? -32768 : ((lo > 32767) ? 32767 : lo);
    hi = (hi < -32768) ? -32768 : ((hi > 32767) ? 32767 : hi);
    return (int)((((unsigned int)hi & 0xFFFFu) << 16) | ((unsigned int)lo & 0xFFFFu));
}
HIPGL_HOST_DEVICE inline int hipgl_bbox_lo(int p) {
    return (int)(short)(unsigned short)((unsigned int)p & 0xFFFFu);
}
HIPGL_HOST_DEVICE inline int hipgl_bbox_hi(int p) {
    return (int)(short)(unsigned short)((unsigned int)p >> 16);
}
// Publish BOTH representations of the reject flags.  `bf` is meaningless once
// `deg` is set -- every reader in the tree tests degenerate first -- so the
// degenerate-only marker leaves the backfacing bit clear, exactly as the
// canonical int is left stale on that path today.
HIPGL_HOST_DEVICE inline void hipgl_set_flags(TriangleSetup& t, bool deg, bool bf) {
    t.degenerate = deg ? 1 : 0;
    t.backfacing = bf ? 1 : 0;
    t.hot_flags  = (deg ? 1 : 0) | (bf ? 2 : 0);
}
HIPGL_HOST_DEVICE inline void hipgl_mark_degenerate(TriangleSetup& t) {
    t.degenerate = 1;
    t.hot_flags  = 1;
}
// Publish BOTH representations.  Every writer of bbox_* must go through this.
HIPGL_HOST_DEVICE inline void hipgl_set_bbox(TriangleSetup& t,
                                             int min_x, int min_y,
                                             int max_x, int max_y) {
    t.bbox_min_x = min_x; t.bbox_max_x = max_x;
    t.bbox_min_y = min_y; t.bbox_max_y = max_y;
    t.hot_bbox_x = hipgl_pack_bbox_i16(min_x, max_x);
    t.hot_bbox_y = hipgl_pack_bbox_i16(min_y, max_y);
}

// ── Tile bin (used in tile-based rasterization) ─────────────────────────────
struct TileBin {
    int  count;
    int  triangle_indices[MAX_TRIANGLES_PER_TILE];
    float triangle_z_min[MAX_TRIANGLES_PER_TILE]; // min screen_z for depth-sorted early-out
};

// ── Render state (host-side, copied to device for kernel launches) ──────────
struct RenderState {
    int32_t   depth_test_enabled;
    int32_t   depth_write_enabled;
    int32_t   blend_enabled;
    int32_t   lighting_enabled;
    int32_t   backface_cull_enabled;
    int32_t   texture_2d_enabled;
    int32_t   color_material_enabled;

    DepthFunc   depth_func;
    BlendFactor blend_src;
    BlendFactor blend_dst;

    float     clear_color[4];
    float     clear_depth;
    int       clear_stencil;
    int32_t   gradient_bg;
    float     gradient_top[3];
    float     gradient_bot[3];

    int       viewport_x, viewport_y, viewport_w, viewport_h;

    float     model_view[16];     // column-major 4x4
    float     projection[16];     // column-major 4x4
    float     mvp[16];            // precomputed model_view * projection
    float     normal_matrix[9];   // upper-left 3x3 of inverse transpose of model_view
    // ── Reflection pass support ─────────────────────────────────────────────
    // MuJoCo's planar reflection (render_gl3.c) renders the mirrored scene with
    // a CLIP PLANE at the mirror, so geometry behind the mirror does not appear
    // in it, and STENCILLED to the mirror's footprint. The HP pass uses:
    //   clip_plane_enabled + clip_plane: plane (nx,ny,nz,d) in EYE SPACE; a
    //     fragment is discarded when dot(n, eye_pos) < d. Eye space, NOT world:
    //     TransformedVertex::world_pos is misnamed -- fused_vertex_tri.hip:67
    //     writes model_view * pos, i.e. eye coordinates. Comparing it against a
    //     world-space plane silently tests nothing (a trap this project already
    //     hit once with FRAG_ATTR_POS). The host transforms the plane instead.
    //   reflect_mask: footprint of the mirror's +Z face, written by a prior
    //     depth-only pass. One byte per pixel; bit s denotes coverage of sample
    //     s (bit 0 at one sample, bits 0..3 at MSAA4). A null pointer disables
    //     this stencil test. MSAA must gate each sample, not only each pixel.
    int32_t   clip_plane_enabled;
    float     clip_plane[4];
    const uint8_t* reflect_mask;
    // Multiplies material RGB before lighting during the mirrored pass,
    // reproducing renderGeomReflection's `rgba[k] *= reflectance` without
    // dimming the specular material. 0 = inactive (normal pass).
    float     reflect_dim;

    Light     lights[MAX_LIGHTS];
    Material  material;
    float     fog_start;
    float     fog_end;
    float     fog_color[3];
    // Linear-light output (SAPIEN-style shading; 0 = off, the MuJoCo path, untouched):
    // texture samples are decoded sRGB -> linear, and the final fragment color is
    // encoded (exposure * c)^(1/gamma). The lights are then given in linear units.
    int32_t   linear_output;
    float     output_exposure;
    float     output_inv_gamma;
    float     camera_world_pos[3];
    float     camera_forward[3];

    int       bound_texture_id;
    float     point_size;
    float     line_width;
    int       shade_model;        // 0=flat, 1=smooth (Gouraud)
    int       polygon_mode;       // 0=fill, 1=line, 2=point
    int       front_face;         // 0=CCW, 1=CW
    int       cull_face;          // 0=back, 1=front, 2=front_and_back
    int32_t   oit_enabled;        // OIT: accumulate into oit_accum/revealage instead of color
    int32_t   two_sided_lighting;  // diagnostic: GL_LIGHT_MODEL_TWO_SIDE
    int32_t   gouraud_shading;   // Use per-vertex Gouraud shading (matching MuJoCo) instead of per-fragment Blinn-Phong

    int32_t   shadow_enabled;
    float     shadow_mvp[16];     // orthographic MVP for projecting fragments to shadow space
    float*    shadow_depth;       // GPU pointer to shadow depth buffer [shadow_w * shadow_h]
    int       shadow_w;
    int       shadow_h;
    float     shadow_bias;        // depth comparison bias (avoids self-shadowing)

    int32_t*  tri_stats;
    // Zero keeps ordinary packed triangle IDs. A positive source count N
    // decodes stable clipping IDs 2*i+fan into dense primary i or suffix N+i.
    int32_t   near_clip_source_count = 0;
    int32_t   reflect_rect[4] = {0, 0, -1, -1};
};

// Per-triangle diagnostic counters. Order is pipeline order, so a geom's row
// read left to right shows exactly where its triangles stop surviving.
enum TriStat : int {
    TRI_STAT_INPUT = 0,      // reached triangle setup
    TRI_STAT_CLIPPED,        // crossed/behind the near plane, or rejected by reflection clip
    TRI_STAT_DEGENERATE,     // rejected: |screen area| below the sub-pixel cull
    TRI_STAT_BACKFACE,       // rejected: backfacing under the cull rule
    TRI_STAT_OFFSCREEN,      // rejected: bbox entirely outside the framebuffer
    TRI_STAT_BINNED,         // inserted into at least one tile bin
    TRI_STAT_BIN_OVERFLOW,   // dropped: tile bin already at MAX_TRIANGLES_PER_TILE
    TRI_STAT_COVERED,        // produced at least one covered sample (pre-depth)
    TRI_STAT_DEPTH_PASS,     // won the depth test for at least one sample
    TRI_STAT_COUNT
};

// ── Draw command (batched submission to pipeline) ───────────────────────────
struct DrawCommand {
    PrimitiveType primitive;
    int           vertex_offset;
    int           vertex_count;
    int           index_offset;
    int           index_count;
    bool          indexed;
};

// ── Error codes ─────────────────────────────────────────────────────────────
enum ErrorCode : int {
    HIPGL_SUCCESS         = 0,
    HIPGL_ERROR_DEVICE    = -1,
    HIPGL_ERROR_MEMORY    = -2,
    HIPGL_ERROR_INVALID   = -3,
    HIPGL_ERROR_NOT_IMPL  = -4,
    HIPGL_ERROR_SHADER    = -5,
};

} // namespace hipgl
