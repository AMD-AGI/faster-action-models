#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include "fine_prefilter.h"
#include <hip/hip_runtime.h>

namespace hipgl {
namespace kernels {

// Optional caller-owned scratch for proper near-plane clipping. Both pointers
// must be non-null to enable it. ids has N source-triangle entries; count is
// one int reset to zero on the draw stream before launch. Enabled draws reserve
// 2N setup/stat slots: dense primary i and optional suffix N+i. Binning reads
// suffix slots only through the queue; diagnostic launches also mark unused
// suffix slots inactive for debug enumeration. Binning assigns stable logical
// IDs 2*i+fan; RenderState::near_clip_source_count enables their raster decode.
// Launchers enqueue clipping before subsequent work on the same stream.
struct NearClipQueue {
    int* ids = nullptr;
    int* count = nullptr;
};

// Runtime-settable sub-pixel area cull threshold (see DEFAULT_MIN_TRI_AREA in
// types.h). Exists so the cull can be ABLATED against a real scene: it drops
// 82% of a dense mesh's triangles and is the leading explanation for hip
// covering only 0.66 of MuJoCo's pixels on libero_object/ketchup.
void set_min_tri_area(float v);


// Clear kernels
void launch_clear_color(DeviceFramebuffer fb, float r, float g, float b, float a, hipStream_t stream);
void launch_clear_depth(DeviceFramebuffer fb, float depth, hipStream_t stream);
void launch_clear_color_depth(DeviceFramebuffer fb, float r, float g, float b, float a, float depth, hipStream_t stream);
void launch_clear_gradient_depth(DeviceFramebuffer fb,
    float bot_r, float bot_g, float bot_b,
    float top_r, float top_g, float top_b,
    float a, float depth, hipStream_t stream);

// Vertex processing (mvp, model_view, normal_mat for lighting)
void launch_vertex_transform(
    const float* positions, const float* normals, const float* texcoords, const float* colors,
    int vertex_count,
    const float mvp[16], const float model_view[16], const float normal_mat[9],
    int vp_x, int vp_y, int vp_w, int vp_h,
    TransformedVertex* out, hipStream_t stream);

// Triangle setup (indexed=false for draw_arrays, indexed=true with indices for draw_elements)
void launch_triangle_setup(
    const TransformedVertex* verts, const uint32_t* indices,
    int tri_count, bool indexed,
    TriangleSetup* out, hipStream_t stream, int32_t* stats = nullptr);

void launch_clear_tile_bin_counts(
    TileBin* tile_bins, int num_tiles, hipStream_t stream);

// Tile binning
void launch_bin_triangles(
    const TriangleSetup* tris, int tri_count,
    TileBin* tile_bins, int tiles_x, int tiles_y,
    int fb_width, int fb_height,
    hipStream_t stream, int32_t* tri_stats = nullptr, NearClipQueue near_clip = {});

struct BinRec {                    // 16 B, naturally aligned -> one dwordx4
    unsigned int       xy;         // lo_x | (lo_y << 16)
    unsigned int       nxy;        // nx   | (ny   << 16)   (nx>=1 always)
    unsigned long long key;        // (ord32(z_min) << 32) | (uint32)tri_index
};

#ifndef HP_MAX_SUBDRAWS
#define HP_MAX_SUBDRAWS 32
#endif
#ifndef HP_MAX_BLOCKS
#define HP_MAX_BLOCKS 32
#endif
#ifndef HP_MAX_SUBDRAWS_DEFAULT
#define HP_MAX_SUBDRAWS_DEFAULT 24     // runtime SLOT-BLOCK cap (HP_MAX_SUB)
#endif
// Slot-block index packed HP_PERM_BITS per ADDRESS rank, HP_PERM_PER_WORD
// ranks per 64-bit word.  Block indices are < HP_MAX_BLOCKS.
#if HP_MAX_BLOCKS > 32
#define HP_PERM_BITS 6
#elif HP_MAX_BLOCKS > 16
#define HP_PERM_BITS 5
#elif HP_MAX_BLOCKS > 8
#define HP_PERM_BITS 4
#else
#define HP_PERM_BITS 3
#endif
#define HP_PERM_MASK ((1 << HP_PERM_BITS) - 1)
#define HP_PERM_PER_WORD (64 / HP_PERM_BITS)
#define HP_PERM_WORDS ((HP_MAX_SUBDRAWS + HP_PERM_PER_WORD - 1) / HP_PERM_PER_WORD)
static_assert(HP_MAX_BLOCKS <= 32, "dead_mask is a 32-bit slot-block mask");

struct BinSubdraws {
    int nsub = 1;                      // ADDRESS RANKS; 1 == plain single-draw binning
    int nblk = 1;                      // SLOT BLOCKS (<= nsub); meaningful when nsub > 1
    int split[HP_MAX_SUBDRAWS - 1] = {};    // split[k] = first SOURCE triangle of span k+1
    // Bit b set => slot block b is BINNED but never rasterized (a bridged hole
    // in the vertex range).  Indexed in SLOT-BLOCK space.  With the shared dead
    // block at most one bit is set.
    int dead_mask = 0;
    long long perm[HP_PERM_WORDS] = {};
};

// Slot block (== raster rank) of the span whose ADDRESS rank is `r`.
// Word select is an unrolled compare ladder over CONSTANT indices, so a
// by-value BinSubdraws in a kernel argument is never dynamically indexed.
HIPGL_HOST_DEVICE inline int hp_subdraw_perm(int r, const BinSubdraws& sd) {
    long long any = 0;
#pragma unroll
    for (int i = 0; i < HP_PERM_WORDS; ++i) any |= sd.perm[i];
    if (!any) return r;
    long long w = sd.perm[0];
    int base = 0;
#pragma unroll
    for (int i = 1; i < HP_PERM_WORDS; ++i) {
        const bool ge = r >= i * HP_PERM_PER_WORD;
        w    = ge ? sd.perm[i] : w;
        base = ge ? i * HP_PERM_PER_WORD : base;
    }
    return (int)((w >> (HP_PERM_BITS * (r - base))) & HP_PERM_MASK);
}

struct BinEmit {
    BinRec* rec = nullptr;             // null => the count pass derives it as before
    int tiles_x = 0, tiles_y = 0;
    int fb_w = 0, fb_h = 0;
    int cull_mask = 0;                 // == backface_cull_mask(state) at the count pass
    BinSubdraws subs{};                // same span description the count pass gets
    bool skip_dead_core = false;
    bool center_samples_only = false;
};

void launch_scan_excl_single(const int* in, int* out, int n, hipStream_t stream);
int backface_cull_mask(const RenderState& state);
struct TileRect {
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    int nsub = HP_MAX_BLOCKS;
    bool active() const { return x1 >= 0; }
};
void launch_count_tile_pairs(const TriangleSetup* tris, int tri_count,
                             int* tile_counts, BinRec* binrec,
                             int tiles_x, int tiles_y,
                             int fb_width, int fb_height, hipStream_t stream,
                             NearClipQueue near_clip = {}, int cull_mask = 0,
                             BinSubdraws subs = {}, bool have_rec = false,
                             TileRect rect = {});
void launch_scatter_tile_pairs(const TriangleSetup* tris, const BinRec* binrec,
                               int tri_count,
                               const int* tile_base, int* cursor,
                               int* pool_tri, float* pool_z,
                               unsigned long long* pool_key,
                               int tiles_x, int tiles_y, int fb_width, int fb_height,
                               int pool_capacity, int* overflow_flag, hipStream_t stream,
                               bool multisample = false, NearClipQueue near_clip = {},
                               TileRect rect = {},
                               const int* tile_counts_in = nullptr,
                               int* tile_base_out = nullptr, int scan_n = 0);
// r4: true when this draw's scatter can absorb the scan (see above). The host
// must ask BEFORE issuing the scan dispatch, since that comes first in stream
// order.
bool scatter_scan_fold_ok(int tiles_x, int tiles_y, int tri_count,
                          bool multisample, int scan_n);
struct BinCntSlot {
    const TriangleSetup* tris = nullptr;
    int* tile_counts = nullptr;
    BinRec* binrec = nullptr;
    NearClipQueue queue{};
    TileRect rect{};
    int cull_mask = 0;
    bool packed_ids = false;
    bool have_rec = false;
};
struct BinScatSlot {
    const TriangleSetup* tris = nullptr;
    const BinRec* binrec = nullptr;
    const int* tile_base = nullptr;
    int* cursor = nullptr;
    int* pool_tri = nullptr;
    float* pool_z = nullptr;
    unsigned long long* pool_key = nullptr;
    int* overflow_flag = nullptr;
    NearClipQueue queue{};
    TileRect rect{};
    int pool_capacity = 0;
};
// Each returns false and launches NOTHING when any precondition of the
// single-slot launcher would have produced a different shape, so the caller can
// always fall back to the two ordinary dispatches.
bool launch_count_tile_pairs_z2(BinCntSlot s0, BinCntSlot s1, int tri_count,
                                int tiles_x, int tiles_y, int fb_width, int fb_height,
                                BinSubdraws subs, hipStream_t stream);
bool launch_scatter_tile_pairs_z2(BinScatSlot s0, BinScatSlot s1, int tri_count,
                                  int tiles_x, int tiles_y, int fb_width, int fb_height,
                                  bool multisample, hipStream_t stream);
bool launch_scan_excl_single_z2(const int* in0, int* out0,
                                const int* in1, int* out1, int n, hipStream_t stream);
void launch_depth_to_mask(const float* depth, uint8_t* mask, float clear_value,
                          int n, hipStream_t stream);
void launch_tile_ends(const int* base, const int* count, int* ends, int n, hipStream_t stream);
// pool_cap bounds the READ span: tile_base/tile_count are the TRUE scanned
// values and are not clamped to the pool, while the scatter clamps its WRITES.
// Without it an undersized pool makes this kernel read past the allocation.
void launch_pool_to_tilebins(const int* tile_base, const int* tile_count,
                             const int* pool_tri, const float* pool_z,
                             TileBin* tile_bins, int num_tiles, int pool_cap,
                             hipStream_t stream);
void launch_sort_pool_segments(const int* tile_base, const int* tile_count,
                               int* pool_tri, float* pool_z, int num_tiles,
                               hipStream_t stream);

// Sort triangle indices within each tile bin (ascending order)
// Ensures deterministic coplanar Z-tie resolution matching MuJoCo's GL_LEQUAL
void launch_sort_tile_bins(
    TileBin* tile_bins, int num_tiles, hipStream_t stream);

// Fine rasterization (state passed by value)
void launch_fine_rasterize(
    const TriangleSetup* tris, const TileBin* tile_bins,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state, hipStream_t stream);

void launch_fine_rasterize_with_sort(
    const TriangleSetup* tris, const TileBin* tile_bins,
    const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture_state, hipStream_t stream,
    const int* work_list = nullptr, const int* work_count = nullptr,
    int* work_ticket = nullptr,
    int nsub = 1, int dead_mask = 0, int slot_stride = 0,
    const float4* flt_rec = nullptr, const int* flt_cnt = nullptr,
    const unsigned short* flt_msk = nullptr);

void fine_batch_begin();
void fine_batch_end();

void launch_build_tile_worklist(
    const int* counts, int num_tiles, int deep_thresh,
    int* work_list, int* work_count, hipStream_t stream);

// Fold the per-tile z-order into the UPSTREAM segmented sort: pack each pool
// entry into (ord32(z_min) << 32) | (uint32)tri_id so an ascending 64-bit sort
// reproduces the (z, tri_id) lexicographic order the fine rasterizer needs.
// total_ptr (optional, device) bounds the packed prefix; pool_cap bounds the
// launch. After this the rasterizer needs no LDS and no in-kernel sort.
void launch_pack_pool_keys(const int* pool_tri, const float* pool_z,
                           unsigned long long* pool_key, const int* total_ptr,
                           int pool_cap, hipStream_t stream);

// ROUND-1 (memory lane): single-dispatch replacement for
//   tile_bounds_clamped -> pack_pool_keys -> rocprim::segmented_radix_sort_keys.
// One wave64 per tile sorts that tile's fragment-pool segment with a register
// bitonic network (no LDS on the common path, so none of rocPRIM's 17.31% LDS
// bank conflict), packing the 64-bit composite key (ord32(z_min) << 32) | tri_id
// on the fly. Segment bounds are clamped to pool_cap inside the kernel, so the
// separate bounds kernel and the intermediate key buffer both disappear.
// Same comparator, same total order, same tie-break (keys are unique per tile).
// Defined in src/kernels/tile_sort.hip.
void launch_tile_wave_sort(const int* pool_tri, const float* pool_z,
                           const int* tile_base, const int* tile_count,
                           unsigned long long* out,
                           int* big_list, int* big_count, int big_cap,
                           int num_tiles, int pool_cap, hipStream_t stream);

void launch_tile_wave_sort_keys(const unsigned long long* pool_key,
                                const int* tile_base, const int* tile_count,
                                unsigned long long* out,
                                int* big_list, int* big_count, int big_cap,
                                int num_tiles, int pool_cap, hipStream_t stream,
                                int tiles_per_span = 0,
                                const FinePreEmit& pre = FinePreEmit{});
struct SortSlot {
    const unsigned long long* keys  = nullptr;
    const int*                tile_base  = nullptr;
    const int*                tile_count = nullptr;
    unsigned long long*       out   = nullptr;
    FinePreEmit               pre{};
};
// Launch BOTH slots as one gridDim.z = 2 dispatch. Returns false (and launches
// nothing) whenever the shape does not qualify, so the caller simply issues the
// two ordinary dispatches instead.
bool launch_tile_wave_sort_keys_z2(const SortSlot& s0, const SortSlot& s1,
                                   int num_tiles, int pool_cap, int tiles_per_span,
                                   hipStream_t stream);
bool tile_sort_fused_enabled();
// Diagnostic: max segment length observed (HIPGL_LOG_MAXSEG=1 to arm).
unsigned int tile_sort_max_seg();

// Fill a float4 buffer with a constant value (used for default vertex colors)
void launch_fill_float4(float* buffer, int count, float v0, float v1, float v2, float v3, hipStream_t stream);

// Fused vertex transform + triangle setup (replaces separate vertex_transform + triangle_setup)
void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats = nullptr,
    NearClipQueue near_clip = {});

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats = nullptr,
    NearClipQueue near_clip = {});

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats = nullptr,
    NearClipQueue near_clip = {});

void launch_fused_vertex_tri(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    TriangleSetup* out, hipStream_t stream, int32_t* stats = nullptr,
    NearClipQueue near_clip = {});

bool fused_vertex_binrec_supported();

void launch_fused_vertex_tri_gouraud(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state,
    TriangleSetup* out, hipStream_t stream, NearClipQueue near_clip = {},
    BinEmit emit = {});

struct VtxCntFuse {
    int* tile_counts = nullptr;
    TileRect rect{};
};
bool launch_fused_vertex_tri_gouraud_ex(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h,
    const RenderState& state,
    TriangleSetup* out, hipStream_t stream, NearClipQueue near_clip,
    BinEmit emit, const VtxCntFuse* cf);

struct VtxPose {
    const int*   vert_geom = nullptr;   // [ntri], -1 == already world space
    const float* xpos      = nullptr;   // [nenv * ngeom * 3]
    const float* xquat     = nullptr;   // [nenv * ngeom * 4]
    int          tbase     = 0;         // draw's first TRIANGLE
    int          ngeom     = 0;
    int          env       = 0;
};

// Host-side publisher/consumer of the pose for the draw about to be issued.
// THREAD-LOCAL: each render worker thread issues build_geometry(env) and then
// the draws for that same world, on its own stream, so the latch cannot be read
// by a thread other than the one that set it. `pos_base`/`nvert` describe the
// buffer the latch belongs to, so a draw whose position pointer is NOT inside
// that buffer (site overlays, host-uploaded arrays) is left unfolded.
struct VtxPoseSrc {
    const int*   vert_geom = nullptr;
    const float* xpos      = nullptr;
    const float* xquat     = nullptr;
    const float* pos_base  = nullptr;
    long long    nvert     = 0;
    int          ngeom     = 0;
    int          env       = 0;
};
void set_vertex_pose_source(const VtxPoseSrc& src);
void clear_vertex_pose_source();

struct VtxZSlot {
    const float* mvp = nullptr;
    const float* model_view = nullptr;
    const float* normal_mat = nullptr;
    TriangleSetup* out = nullptr;
    NearClipQueue near_clip{};
    BinEmit emit{};
    int bf_skip_mask = 0;      // filled in by the launcher
    RenderState state{};
};

// Launch both slots as ONE grid (gridDim.z = 2). Returns false and launches
// NOTHING when any precondition fails, so the caller falls back to the two
// ordinary dispatches.
bool launch_fused_vertex_tri_gouraud_z2(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors,
    const float* per_vert_specular, const float* per_vert_shininess,
    const float* per_vert_emission,
    int tri_count,
    int vp_x, int vp_y, int vp_w, int vp_h,
    VtxZSlot s0, VtxZSlot s1, hipStream_t stream);

struct MicroDrawSlotDesc {
    const TriangleSetup* tris = nullptr;
    int tri_count = 0;
    int suffix_base = 0;
    int cull_mask = 0;
    int* tile_base = nullptr;                 // [num_tiles+1] out
    int* cursor = nullptr;                    // [num_tiles]   out
    unsigned long long* key_out = nullptr;
    int pool_capacity = 0;
    NearClipQueue near_clip{};
};
struct MicroClearDesc {
    float* color = nullptr;      // 4 floats per pixel, pitch == width*4
    float* depth = nullptr;
    int    total = 0;            // pixels; 0 disables
    float  r = 0.f, g = 0.f, b = 0.f, a = 0.f, d = 0.f;
    unsigned char* cvt_dst = nullptr;
};
bool launch_micro_bin_chain_n(const MicroDrawSlotDesc* slots, int nslots,
                              int tiles_x, int tiles_y, int fb_width, int fb_height,
                              int num_tiles, int* overflow_flag, hipStream_t stream,
                              const MicroClearDesc* clr = nullptr);
int micro_bin_max_slots();

// OIT resolve: composite weighted-blended transparency over opaque framebuffer
void launch_oit_resolve(DeviceFramebuffer fb, hipStream_t stream);

// Framebuffer format conversion: RGBA float32 -> RGB uint8, with optional flip
void launch_rgba_f32_to_rgb_u8(
    const float* src_rgba, uint8_t* dst_rgb,
    int width, int height,
    bool flip_v, bool flip_h,
    hipStream_t stream);

// Same, but box-averages an ss x ss block per output pixel (supersampled
// resolve). out_width/out_height are the FINAL dimensions; the source is
// ss times larger in each axis. ss <= 1 forwards to the plain version.
void launch_rgba_f32_to_rgb_u8_ss(
    const float* src_rgba, uint8_t* dst_rgb,
    int out_width, int out_height, int ss,
    bool flip_v, bool flip_h,
    hipStream_t stream);


#ifndef HP_CQ_MAX
#define HP_CQ_MAX 16
#endif
struct CqVtxDesc {
    const float* positions; const float* normals; const float* texcoords;
    const int* texids; const float* colors; const float* spec_colors;
    const float* per_vert_specular; const float* per_vert_shininess;
    const float* per_vert_emission;
    const float* mvp; const float* model_view; const float* normal_mat;
    TriangleSetup* out; NearClipQueue near_clip; int* tile_counts;
    BinEmit emit; VtxPose pose; TileRect rect;
    int tri_count, bf_skip_mask, vp_x, vp_y, vp_w, vp_h;
    int grid, bstart;
};
struct CqVtxBatch { CqVtxDesc d[HP_CQ_MAX]; int n; };
struct CqScatDesc {
    const TriangleSetup* tris; const BinRec* binrec;
    const int* tile_counts; int* tile_base_out; int* cursor;
    int* pool_tri; float* pool_z; unsigned long long* pool_key; int* overflow_flag;
    NearClipQueue queue; TileRect rect;
    int tri_count, scan_n, tiles_x, tiles_y, fb_w, fb_h, pool_capacity, ilv;
    int grid, bstart;
};
struct CqScatBatch { CqScatDesc d[HP_CQ_MAX]; int n; };
struct CqSortDesc {
    const unsigned long long* keys; const int* tile_base; const int* tile_count;
    unsigned long long* out; FinePreEmit pre;
    int num_slots, pool_cap, grid, bstart;
};
struct CqSortBatch { CqSortDesc d[HP_CQ_MAX]; int n; int lds_cap; };
// Host-side record of one chain's three binning stages.
struct CqVtxRec  { CqVtxDesc d; RenderState state; int refl = 0, lb = 0, pf = 0; };
struct CqSortRec { CqSortDesc d; int bs = 0, spt = 0, strided = 0, lds_cap = 0; };
struct CqChainRec {
    int want = 0;      // stage bit the next launcher call should record (1 vtx, 2 scat, 4 sort)
    int got  = 0;      // stage bits actually recorded
    CqVtxRec  v;
    CqScatDesc s;
    CqSortRec o;
    // Called by a launcher that is about to launch a WANTED stage directly
    // (it cannot record it): the owner must issue everything recorded so far
    // first, so the direct launch stays behind its producers.
    void (*miss)(void*) = nullptr;
    void* miss_ctx = nullptr;
};
inline void cq_rec_miss(CqChainRec* cq) { if (cq && cq->miss) cq->miss(cq->miss_ctx); }
// Non-null => the launcher for the stage named by ->want records instead of
// launching (THREAD-LOCAL: each render worker owns one Pipeline + stream).
CqChainRec*& cq_rec_slot();
// Batched launchers; each takes n (>=1) records and issues the minimum number
// of dispatches (one per compatible group).
void launch_cq_vtx(const CqVtxRec* const* r, int n, hipStream_t stream);
void launch_cq_scat(const CqScatDesc* const* r, int n, hipStream_t stream);
void launch_cq_sort(const CqSortRec* const* r, int n, hipStream_t stream);

} // namespace kernels
} // namespace hipgl
