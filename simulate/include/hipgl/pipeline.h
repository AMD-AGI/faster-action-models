#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdlib>

#include "types.h"
#include "kernels.h"
#include "framebuffer.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>


namespace hipgl { namespace kernels {
struct FbMaskClearDesc {
    const float*   depth_in = nullptr;   // read for the mask (== depth, same buffer)
    float*         color    = nullptr;   // 4 floats per pixel
    float*         depth    = nullptr;
    unsigned char* mask     = nullptr;
    int            total    = 0;         // pixels; 0 disables the rider entirely
    float mask_ref = 0.f, r = 0.f, g = 0.f, b = 0.f, a = 0.f, d = 0.f;
};
// The real implementation of the pair count launcher.  `mc` non-null and
// mc->total > 0 appends ceil(total/TPB) rider blocks to gridDim.x.  Returns
// false WITHOUT LAUNCHING ANYTHING (rider included) on every gate the
// single-slot twin would have taken differently, so the caller always has a
// byte-for-byte fallback.
bool launch_count_tile_pairs_z2_mc(BinCntSlot s0, BinCntSlot s1, int tri_count,
                                   int tiles_x, int tiles_y, int fb_width, int fb_height,
                                   BinSubdraws subs, hipStream_t stream,
                                   const FbMaskClearDesc* mc);
// Standalone fallback -- same expressions, same order, same bytes as
// kernel_reflection_mask_and_clear.
void launch_fb_mask_clear(const FbMaskClearDesc& mc, hipStream_t stream);
}}  // namespace hipgl::kernels

namespace hipgl {

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    void set_framebuffer(Framebuffer* fb);
    void set_render_state(const RenderState& state);
    void clear(bool color, bool depth);

    void draw_arrays(PrimitiveType prim, const float* positions, const float* normals,
                     const float* texcoords, const float* colors, int vertex_count);

    // Draw with device-resident vertex data (no host-to-device copy).
    void draw_arrays_device(PrimitiveType prim, const float* d_positions, const float* d_normals,
                            const float* d_colors, int vertex_count);
    void draw_arrays_device(PrimitiveType prim, const float* d_positions, const float* d_normals,
                            const float* d_texcoords, const int* d_texids,
                            const float* d_colors, int vertex_count,
                            const TextureState& texture_state);
    void draw_arrays_device(PrimitiveType prim, const float* d_positions, const float* d_normals,
                            const float* d_texcoords, const int* d_texids,
                            const float* d_colors, const float* d_spec_colors,
                            int vertex_count,
                            const TextureState& texture_state);

    void draw_arrays_device_phong(PrimitiveType prim, const float* d_positions, const float* d_normals,
                                  const float* d_texcoords, const int* d_texids,
                                  const float* d_colors,
                                  const float* d_specular, const float* d_shininess,
                                  const float* d_emission,
                                  int vertex_count,
                                  const TextureState& texture_state);

    void draw_elements(PrimitiveType prim, const float* positions, const float* normals,
                       const float* texcoords, const float* colors, int vertex_count,
                       const uint32_t* indices, int index_count);

    void sync();
    void resolve() { flush_pending_draw(); if (fb_) fb_->resolve(stream_); }
    hipStream_t stream() const { flush_pending_draw(); return stream_; }

    void arm_fb_mask_clear(const kernels::FbMaskClearDesc& mc);
    void mc_flush();

    void vtx_flush_pending();

    void flush_pending_draw() const;
    void flush_pending_draw_keep() const;
    void cq_flush() const;

    void set_stream(hipStream_t s) { flush_pending_draw(); stream_ = s; owns_stream_ = false; }

    void set_external_matrices(const float* d41) { flush_pending_draw(); d_ext_matrices_ = d41; }

    void set_deferred_bg(bool on) { flush_pending_draw(); deferred_bg_ = on; }
    bool deferred_bg() const { return deferred_bg_; }

private:
    Framebuffer* fb_;
    RenderState state_;
    hipStream_t stream_;
    kernels::FbMaskClearDesc pending_mc_{};
    bool pending_mc_armed_ = false;
    std::function<bool(const kernels::VtxCntFuse*)>* pending_vtx_ = nullptr;
    hipStream_t owned_stream_ = nullptr;  // the one the ctor made; only this is destroyed
    bool owns_stream_ = true;

    // Scratch device buffers (grown as needed)
    TransformedVertex* d_transformed_;
    TriangleSetup* d_triangles_;
    // Source triangle i owns dense primary i and optional suffix N+i. The
    // fused pass queues only near-plane straddlers; both buffers are reused on
    // this Pipeline's stream, including across reflection and overlay draws.
    int* d_near_clip_ids_ = nullptr;
    int* d_nc_pool_ = nullptr;         // [kNcSlots] pre-zeroed counter slots
    int  nc_slot_ = 0;
    int* d_near_clip_count_ = nullptr; // == d_nc_pool_ + slot for THIS draw
    int near_clip_capacity_ = 0;
    TileBin* d_tile_bins_;
    // Deterministic-binning scratch (count -> scan -> scatter -> sort).
    int*   d_tile_counts_ = nullptr;   // [num_tiles]
    int*   d_tile_base_   = nullptr;   // [num_tiles] exclusive scan of counts
    int*   d_cursor_      = nullptr;   // [num_tiles] per-tile scatter cursor
    int*   d_pool_tri_    = nullptr;   // [pool_capacity_]
    float* d_pool_z_      = nullptr;   // [pool_capacity_]
    int*   d_overflow_    = nullptr;   // [1] set if the pool was too small
    void*  d_scan_tmp_    = nullptr;   // rocPRIM temp storage
    size_t scan_tmp_bytes_ = 0;
    int    pool_capacity_ = 0;
    bool   deterministic_binning_ = (getenv("HIPGL_DET_BIN") == nullptr) ? true : (getenv("HIPGL_DET_BIN")[0] == '1');
    unsigned pool_check_counter_ = 0;
    int*   d_tile_end_        = nullptr;  // [num_tiles] base+count, segment ends
    int*   d_pool_sorted_tri_ = nullptr;  // double buffer for the segmented sort
    float* d_pool_sorted_z_   = nullptr;
    void*  d_sort_tmp_        = nullptr;
    size_t sort_tmp_bytes_    = 0;
    int    total_pairs_estimate_ = 0;
    // Per-instance size caches for the deterministic-binning scratch. These
    // were function-local statics, i.e. shared by every Pipeline: one
    // instance's cache reported another's buffer as already large enough,
    // producing an illegal access as soon as a second pipeline drew.
    size_t sz_counts_ = 0, sz_base_ = 0, sz_cursor_ = 0, sz_ovf_ = 0;
    size_t sz_end_ = 0, sz_st_ = 0, sz_sz_ = 0;
    int*   d_bin_scratch_ = nullptr;   // [3*(num_tiles+1)+3]: counts | cursor | overflow |
                                       // big_count | worklist | work_count[2] | work_ticket
    int*   d_work_list_   = nullptr;   // [num_tiles] compacted non-empty tile ids
    int*   d_work_count_  = nullptr;   // [2] {deep, shallow}
    int*   d_work_ticket_ = nullptr;   // [1] persistent-grid ticket
    bool   work_ready_    = false;     // worklist was built for this draw
    void*  d_flt_rec_     = nullptr;
    size_t sz_flt_rec_    = 0;
    int*   d_flt_cnt_     = nullptr;
    size_t sz_flt_cnt_    = 0;
    unsigned short* d_flt_msk_ = nullptr;
    size_t sz_flt_msk_    = 0;
    bool   flt_msk_ready_ = false;   // coverage words published for THIS draw
    bool   flt_ready_     = false;
    size_t sz_bin_scratch_ = 0;
    size_t bin_slot_stride_ = 0;       // ints per slot (multiple of 4)
    int    bin_slots_       = 0;       // slots in the current allocation
    int    bin_slot_        = 0;       // next slot to hand out
    int*   d_bounds_ = nullptr;        // [4*(num_tiles+1)]: base | (unused) | base_c | end_c
    size_t sz_bounds_ = 0;
    void*  d_binrec_ = nullptr;        // [tri_count] kernels::BinRec
    size_t sz_binrec_ = 0;
    int*   h_readback_ = nullptr;      // pinned [4]: total_pairs, overflow, exact-slot, pad
    int    last_total_pairs_ = 0;
    bool   have_readback_ = false;
    // Largest draw sized by an exact count for this tile grid. A tiny mask
    // draw must not supply the only capacity estimate for a later scene draw.
    int    pool_exact_max_triangles_ = 0;
    int    pool_exact_tiles_x_ = 0, pool_exact_tiles_y_ = 0;
    int*   d_overflow_persist_ = nullptr;
    // Forces exact pool sizing for the next draws regardless of HP_POOL_EXACT.
    // Set by the caller on a re-render after an overflow was observed.
    bool   force_exact_ = false;
    // Opt-in, per call site: skip the in-draw sizing sync and rely on the
    // retained high-water mark. ONLY safe for a caller that joins the streams
    // and then acts on take_overflow() -- today that is render_gpu_batch.
    // Every other path (render, render_gpu) leaves this false and keeps the
    // original synchronising behaviour.
    bool   defer_pool_sizing_ = false;
    int    rb_tri_count_ = 0;
    float  pairs_per_tri_max_ = 8.0f;
    static constexpr int kPptTriB  = 32;   // bit length of output_tri_count
    static constexpr int kPptRectB = 32;   // bit length of the coverable tile count
    static constexpr int kPptBuckets = kPptTriB * kPptRectB;
    // Draw-size class = bit length of the draw's triangle count, so draws within
    // a factor of two share one learned ratio and draws four orders of magnitude
    // apart never do.
    static int ppt_bitlen(int v) {
        unsigned t = (v > 0) ? (unsigned)v : 1u;
        int b = 0;
        while (t) { t >>= 1; ++b; }
        return (b >= kPptTriB) ? (kPptTriB - 1) : b;
    }
    static int ppt_bucket_of(int tri, int rect_tiles) {
        return ppt_bitlen(tri) * kPptRectB + ppt_bitlen(rect_tiles);
    }
    float  ppt_bucket_[kPptBuckets] = {};   // ratio high-water; <0 = never seen
    int    ppt_grid_x_ = -1, ppt_grid_y_ = -1;   // grid ppt_bucket_ was learned on
    int    rb_tiles_x_ = -1, rb_tiles_y_ = -1;   // grid the async observation belongs to
    int    rb_rect_tiles_ = 0;                   // portal class of that observation
    int*   h_overflow_pin_ = nullptr;  // pinned [1]
    bool   overflow_queued_ = false;

    TriangleSetup* d_micro_tris_ = nullptr;   // [2 * micro_pair_full_]
    size_t sz_micro_tris_ = 0;
    int*   d_micro_bins_  = nullptr;          // 2 x (num_tiles+1) base | 2 x num_tiles cursor
    size_t sz_micro_bins_ = 0;
    unsigned long long* d_micro_pool_ = nullptr;  // 2 x micro_pair_cap_
    size_t sz_micro_pool_ = 0;
    int    micro_pair_slots_ = 0;             // 0 = no pair prepared
    int    micro_pair_next_  = 0;
    const float* micro_pair_pos_ = nullptr;
    int    micro_pair_tris_[2] = {0, 0};
    int    micro_pair_full_  = 0;
    int    micro_pair_tiles_x_ = 0, micro_pair_tiles_y_ = 0, micro_pair_num_tiles_ = 0;
    int    micro_pair_cap_ = 0;
    TriangleSetup* d_vtx1_tris_ = nullptr;
    size_t sz_vtx1_tris_ = 0;
    void*  d_vtx1_rec_ = nullptr;
    size_t sz_vtx1_rec_ = 0;
    int*   d_vtx1_nc_ids_ = nullptr;
    size_t sz_vtx1_nc_ids_ = 0;
    int*   d_vtx1_nc_count_ = nullptr;   // borrowed rotating slot
    bool   vtx_pair_armed_ = false;      // a second state has been registered
    bool   vtx_pair_ready_ = false;      // slot 1 has been launched
    RenderState vtx_pair_state_{};       // the second pass's state, as armed
    const float* vtx_pair_mat_ = nullptr;
    const float* vtx_pair_pos_ = nullptr;
    int    vtx_pair_tris_ = 0;
    int    vtx_pair_vw_ = 0, vtx_pair_vh_ = 0;
    kernels::BinEmit vtx_pair_emit_{};
    bool   vtx_pair_swapped_ = false;
    TriangleSetup* sv_tris_ = nullptr; int sv_tricap_ = 0;
    void*  sv_rec_ = nullptr; size_t sv_recsz_ = 0;
    int*   sv_ncid_ = nullptr; int sv_nccap_ = 0; int* sv_nccnt_ = nullptr;
    bool   vtx_pair_try_launch(const float* d_pos, const float* d_norm,
                               const float* d_tex, const int* d_texid,
                               const float* d_col, const float* d_specular,
                               const float* d_shininess, const float* d_emission,
                               int tri_count, int vw, int vh,
                               const float* mvp, const float* mv, const float* nm,
                               const kernels::BinEmit& emit);
    bool   vtx_pair_take(const float* d_pos, int tri_count, int vw, int vh,
                         const kernels::BinEmit& emit);
    void   vtx_pair_swap_out();

    struct BinPair {
        bool   armed = false;          // slot 1's stages have been issued
        bool   scattered = false;      // ... including the scatter
        const TriangleSetup* tris = nullptr;
        const void* rec = nullptr;
        const int* nc_ids = nullptr;
        const int* nc_cnt = nullptr;
        int    tri_count = 0, tiles_x = 0, tiles_y = 0, vw = 0, vh = 0;
        int    nsub = 0, cull_mask = 0, num_slots = 0;
        bool   have_rec = false;
        kernels::BinSubdraws subs{};
        int*   tile_counts = nullptr;  // slot-1 rotating bin block
        int*   cursor = nullptr;
        int*   work_count = nullptr;
        int*   work_ticket = nullptr;
        int*   tile_base = nullptr;    // slot-1 scan output
        unsigned long long* key_in = nullptr;
        int    pool_capacity = 0;
        long long total_pairs = -1;    // exact-sizing path only
        bool   sorted = false;
        unsigned long long* key_out = nullptr;
        void*  flt_rec = nullptr;
        int*   flt_cnt = nullptr;
        void*  flt_msk = nullptr;
        bool   flt_msk_ready = false;
        kernels::FinePreState pre_st{0, 0, 0, 0};
    };
    BinPair bin_pair_{};
    int*   d_bin1_base_ = nullptr;     // slot-1 tile_base  [num_slots+1]
    size_t sz_bin1_base_ = 0;
    void*  d_bin1_key_ = nullptr;      // slot-1 scatter key pool
    size_t sz_bin1_key_ = 0;
    void*  d_bin1_keyout_ = nullptr;   // slot-1 sorted keys [pool_capacity]
    size_t sz_bin1_keyout_ = 0;
    void*  d_bin1_flt_rec_ = nullptr;  // 64 B / pool entry
    size_t sz_bin1_flt_rec_ = 0;
    int*   d_bin1_flt_cnt_ = nullptr;  // survivors per slot
    size_t sz_bin1_flt_cnt_ = 0;
    void*  d_bin1_flt_msk_ = nullptr;  // 2 B / pool entry
    size_t sz_bin1_flt_msk_ = 0;
    unsigned long long* fine_key_override_ = nullptr;
    void*  fine_rec_override_ = nullptr;
    int*   fine_cnt_override_ = nullptr;
    void*  fine_msk_override_ = nullptr;
    void   bin_pair_cancel() { bin_pair_ = BinPair{}; }
public:
    // Register the SECOND saturated vertex pass of this world (state +
    // device matrix block) before the FIRST one is issued. Harmless if the
    // second pass never materialises: vtx_pair_cancel() drops it.
    void vtx_pair_arm(const RenderState& second, const float* second_matrices);
    void vtx_pair_cancel();
private:
    bool   pend_clear_valid_ = false;
    bool   micro_absorbing_  = false;
    float  pend_clear_rgba_[4] = {0.f, 0.f, 0.f, 0.f};
    float  pend_clear_depth_ = 0.f;
    int    pend_clear_total_ = 0;
    void   flush_pending_clear();
    const float*   pend_cvt_src_ = nullptr;
    unsigned char* pend_cvt_dst_ = nullptr;
    int            pend_cvt_total_ = 0;
    int            pend_cvt_w_ = 0, pend_cvt_h_ = 0;
public:
    // Ask for a colour+depth clear that MAY be folded into the next dispatch.
    // Only safe from a caller that goes through this Pipeline for everything it
    // does to the framebuffer until the next draw -- today that is
    // render_gpu_batch. Falls back to clear() whenever folding does not apply.
    void clear_deferred(bool color, bool depth);
    bool convert_deferred(const float* src_rgba, unsigned char* dst_rgb, int total);
    // Emit any recorded convert now. MUST be called before the host reads the
    // destination (render_gpu_batch does it before joining the streams).
    void flush_pending_convert();
    // Prepare the two micro draws of one world as ONE vertex pass + ONE batched
    // bin dispatch. `st_full` is the state of the SECOND (additive) draw, whose
    // shading the shared TriangleSetup array carries; `st_first` supplies only
    // the first draw's backface-cull rule, which is applied per slot inside the
    // bin chain. The two subsequent draw_arrays_device_phong() calls over the
    // same span then issue their fine pass only. A no-op (and a silent fall
    // back to the ordinary per-draw path) whenever any precondition fails.
    void micro_pair_begin(const float* d_pos, const float* d_norm, const float* d_tex,
                          const int* d_texid, const float* d_col, const float* d_specular,
                          const float* d_shininess, const float* d_emission,
                          int tri_count_full, int tri_count_first,
                          const RenderState& st_full, const RenderState& st_first);
    void micro_pair_cancel() { micro_pair_slots_ = 0; micro_pair_next_ = 0; }
private:
    bool micro_pair_is_take(const float* d_pos, int tri_count, int tiles_x, int tiles_y) const;
    bool micro_pair_take(const float* d_pos, int tri_count, int tiles_x, int tiles_y,
                         const TextureState& texture_state);
public:
    // Queue the sticky-overflow D2H on this pipeline's stream. Call it after the
    // last draw of a batch and BEFORE joining the stream; take_overflow() then
    // costs no dispatch and no sync.
    void queue_overflow_read();
    // Read and clear the sticky overflow word. The caller must have synchronised
    // this pipeline's stream first. Returns true if ANY draw since the last
    // call clamped the pool, i.e. silently dropped geometry.
    bool take_overflow();
    // Force exact (synchronising) pool sizing on subsequent draws.
    void set_force_exact(bool v) { force_exact_ = v; }
    // Opt into deferred pool sizing. Only call this from a path that joins the
    // streams and then acts on take_overflow(); otherwise a clamped draw goes
    // unnoticed and the frame silently loses geometry.
    void set_defer_pool_sizing(bool v) { defer_pool_sizing_ = v; }
    unsigned long long resource_fingerprint() const;
    mutable unsigned long long fp_dbg_[64] = {0};
    mutable int fp_dbg_n_ = -1;
    void reset_nc_rotation() { nc_slot_ = 0; }
    void reset_bin_rotation() { bin_slot_ = 0; }
    // DIAGNOSTIC: per-tile triangle counts as recorded by the binning kernel.
    // The count keeps incrementing past MAX_TRIANGLES_PER_TILE even though the
    // indices are dropped, so a value above the cap is direct evidence that
    // triangles were silently discarded for that tile.
    void bin_triangles_deterministic(int tri_count, int tiles_x, int tiles_y, int vw, int vh,
                                     const kernels::BinSubdraws& subs = {},
                                     bool binrec_ready = false);
    kernels::BinRec* ensure_binrec(int output_tri_count);
    void debug_tile_counts(int* out, int num_tiles) const;
    // One tile's bin CONTENTS (indices + z_min). Counts alone cannot detect a
    // nondeterministic slot assignment; see the note in the .cpp.
    void debug_tile_contents(int tile_idx, int* out_idx, float* out_z,
                             int max_entries, int* out_count) const;
    // Per-triangle stage counters. Diagnostic; allocation is opt-in and the
    // pointer stays null (and every counter write is skipped) otherwise.
    // Stats rows refer to physical setup slots: primary i and suffix N+i.
    // Bin IDs encode logical source/fan order 2*i+fan; first_vertex retains
    // the original source vertex mapping.
    // `capacity` is in TRIANGLES; the buffer is tri * TRI_STAT_COUNT ints.
    void debug_enable_tri_stats(bool on);
    bool debug_tri_stats_enabled() const { return d_tri_stats_ != nullptr; }
    int  debug_tri_stats_count() const { return last_tri_count_; }
    void debug_read_tri_stats(int* out, int tri_count) const;
    bool debug_read_triangle(int tri_idx, TriangleSetup* out) const;
    int debug_num_tiles() const { return last_num_tiles_; }
private:
    int last_num_tiles_ = 0;
    int last_tri_count_ = 0;
    int32_t* d_tri_stats_ = nullptr;
    int tri_stats_cap_ = 0;
    bool stats_wanted_ = false;
    float* d_positions_;
    float* d_normals_;
    float* d_texcoords_;
    int*   d_texids_;
    float* d_colors_;
    float* d_spec_colors_;
    uint32_t* d_indices_;

    // Device-side matrix buffers (kernel args must be device pointers)
    // Packed layout: [mvp:16][model_view:16][normal_mat:9] = 41 floats
    bool deferred_bg_ = false;
    int rb_countdown_ = 0;   // draws until the next pool-telemetry readback
    const float* d_ext_matrices_ = nullptr;  // batched slice; null => per-draw copy
    float* d_matrices_;      // 41 floats packed
    float* d_mvp_;           // points into d_matrices_
    float* d_model_view_;    // points into d_matrices_ + 16
    float* d_normal_mat_;    // points into d_matrices_ + 32

    // Host-side staging buffer for batched matrix upload
    float h_matrices_[41];

    int transformed_capacity_;
    int triangle_capacity_;
    int tile_bins_capacity_;
    int vertex_capacity_;
    int index_capacity_;

    struct PendingPhong {
        bool valid = false;
        PrimitiveType prim = PRIM_TRIANGLES;
        const float* pos = nullptr;
        const float* norm = nullptr;
        const float* tex = nullptr;
        const int*   texid = nullptr;
        const float* col = nullptr;
        const float* spec = nullptr;
        const float* shin = nullptr;
        const float* emis = nullptr;
        int vertex_count = 0;
        TextureState tex_state{};
        kernels::BinSubdraws subs{};
    };
    mutable PendingPhong pend_;

    struct CqSet {
        TriangleSetup* tris = nullptr; int tri_cap = 0;
        void* binrec = nullptr; size_t sz_binrec = 0;
        int* nc_ids = nullptr; int nc_cap = 0;
        int* bounds = nullptr; size_t sz_bounds = 0;
        int pool_cap = 0; int* pool_tri = nullptr; float* pool_z = nullptr;
        int* pst = nullptr; size_t sz_st = 0; float* psz = nullptr; size_t sz_sz = 0;
        void* flt_rec = nullptr; size_t sz_flt_rec = 0;
        int* flt_cnt = nullptr; size_t sz_flt_cnt = 0;
        unsigned short* flt_msk = nullptr; size_t sz_flt_msk = 0;
    };
    struct CqPending {
        kernels::CqChainRec rec{};
        bool rb = false;
        const int* rb_src = nullptr;
        int rb_meta[4] = {0, -1, -1, 0};   // tri_count, tiles_x, tiles_y, rect_tiles
        std::function<void()> fine;
    };
    CqSet cq_sets_[HP_CQ_MAX];
    int cq_cur_set_ = 0;
    unsigned cq_busy_ = 0;
    mutable std::vector<CqPending> cq_q_;
    bool cq_draw_ = false;
    bool cq_recording_ = false;
    kernels::CqChainRec cq_cur_{};
    bool cq_rb_ = false;
    const int* cq_rb_src_ = nullptr;
    int cq_rb_meta_[4] = {0, -1, -1, 0};
    int cq_pool_hw_ = 0;
    // largest pool any set has grown to (the single-pool tree's capacity)
    void cq_rb_issue(const int* src, const int* meta);
    unsigned long long cq_stat_q_ = 0, cq_stat_f_ = 0, cq_stat_a_ = 0;
    void cq_save_set(int s);
    void cq_load_set(int s);
    void cq_abandon();
    void cq_do_flush();
    void draw_phong_now(PrimitiveType prim, const float* d_positions, const float* d_normals,
                        const float* d_texcoords, const int* d_texids,
                        const float* d_colors,
                        const float* d_specular, const float* d_shininess,
                        const float* d_emission,
                        int vertex_count,
                        const TextureState& texture_state,
                        const kernels::BinSubdraws& subs);

    // tri_count is the source count; reserves 2*N output slots and resets the queue.
    void ensure_scratch(int vertex_count, int tri_count, int tiles_x, int tiles_y);
    void free_scratch();

    void draw_impl(PrimitiveType prim, const float* positions, const float* normals,
                   const float* texcoords, const float* colors, int vertex_count,
                   const uint32_t* indices, int index_count, bool indexed);
};

} // namespace hipgl
