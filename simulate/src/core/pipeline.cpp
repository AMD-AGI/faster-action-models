// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/pipeline.h"
#include "hipgl/msaa.h"
#include <climits>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/device/device_segmented_radix_sort.hpp>
#include <algorithm>
#include <cstddef>
#include "hipgl/kernels.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef HP_DBG_POOL_DEFAULT
#define HP_DBG_POOL_DEFAULT 0
#endif

// Slack over the observed per-class high-water ratio on the deferred path.
// The exact path already leaves 1.5x + 4096; the deferred path never gets to
// see this draw's count, so it leaves 2x. An under-estimate is not a
// correctness bug (the scatter sets the sticky overflow word and
// render_gpu_batch re-renders with exact sizing) but a re-render costs a whole
// extra frame, so this is deliberately generous.
static constexpr float kPoolDeferSlack = 2.0f;

namespace hipgl {
namespace kernels { void launch_clear_depth_vec(float* depth, int total, float d, hipStream_t stream); }
using kernels::launch_clear_depth_vec;

namespace kernels {
// Defined in src/kernels/bin_rasterize.hip. Declared locally rather than in
// kernels.h to keep this host-lane patch confined to pipeline.{h,cpp}.
void launch_tile_bounds_clamped(const int* base, const int* count,
                                int* base_c, int* ends_c,
                                int n, int cap, hipStream_t stream);
bool launch_micro_bin_chain(const TriangleSetup* tris, int tri_count,
                            int* tile_base, int* cursor,
                            unsigned long long* key_out,
                            int tiles_x, int tiles_y, int fb_width, int fb_height,
                            int num_tiles, int pool_capacity, int* overflow_flag,
                            int cull_mask, NearClipQueue near_clip, hipStream_t stream);
int micro_bin_max_tris();
int micro_bin_max_tiles();
}

namespace {

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

int tri_count_from_vertices(PrimitiveType prim, int vertex_count) {
    switch (prim) {
        case PRIM_TRIANGLES:      return vertex_count / 3;
        case PRIM_TRIANGLE_STRIP:
        case PRIM_TRIANGLE_FAN:   return std::max(0, vertex_count - 2);
        case PRIM_QUADS:          return (vertex_count / 4) * 2;
        default:                  return 0;
    }
}

int tri_count_from_indices(PrimitiveType prim, int index_count) {
    switch (prim) {
        case PRIM_TRIANGLES:      return index_count / 3;
        case PRIM_TRIANGLE_STRIP:
        case PRIM_TRIANGLE_FAN:   return std::max(0, index_count - 2);
        case PRIM_QUADS:          return (index_count / 4) * 2;
        default:                  return 0;
    }
}

} // namespace

Pipeline::Pipeline()
    : fb_(nullptr)
    , d_transformed_(nullptr)
    , d_triangles_(nullptr)
    , d_tile_bins_(nullptr)
    , d_positions_(nullptr)
    , d_normals_(nullptr)
    , d_texcoords_(nullptr)
    , d_texids_(nullptr)
    , d_colors_(nullptr)
    , d_spec_colors_(nullptr)
    , d_indices_(nullptr)
    , d_matrices_(nullptr)
    , d_mvp_(nullptr)
    , d_model_view_(nullptr)
    , d_normal_mat_(nullptr)
    , transformed_capacity_(0)
    , triangle_capacity_(0)
    , tile_bins_capacity_(0)
    , vertex_capacity_(0)
    , index_capacity_(0)
{
    hipError_t err = hipStreamCreate(&stream_);
    check_hip(err, "Pipeline: hipStreamCreate");
    // Remember the stream WE created. set_stream() may later point stream_ at a
    // borrowed one; only this handle may ever be destroyed by ~Pipeline.
    owned_stream_ = stream_;
    owns_stream_ = true;

    check_hip(hipMalloc(&d_matrices_, 41 * sizeof(float)), "Pipeline: d_matrices");
    d_mvp_ = d_matrices_;
    d_model_view_ = d_matrices_ + 16;
    d_normal_mat_ = d_matrices_ + 32;

    std::memset(h_matrices_, 0, sizeof(h_matrices_));
}

Pipeline::~Pipeline() {
    pend_.valid = false;
    free_scratch();
    // Destroy ONLY the stream this Pipeline created. Destroying stream_ blindly
    // double-freed streams owned by RenderWorker / PhysicsEngine.
    if (h_overflow_pin_) { hipHostFree(h_overflow_pin_); h_overflow_pin_ = nullptr; }
    if (owned_stream_) {
        hipStreamDestroy(owned_stream_);
        owned_stream_ = nullptr;
    }
    stream_ = nullptr;
}

void Pipeline::set_framebuffer(Framebuffer* fb) {
    flush_pending_draw();
    fb_ = fb;
    if (fb && fb->samples() == 4) deterministic_binning_ = true;
}

void Pipeline::set_render_state(const RenderState& state) {
    flush_pending_draw_keep();
    if (fb_ && fb_->samples() == 4 && state.oit_enabled)
        throw std::runtime_error("Native MSAA supports alpha blending, not weighted OIT");
    state_ = state;
}

void Pipeline::debug_enable_tri_stats(bool on) {
    if (!on) {
        if (d_tri_stats_) { hipFree(d_tri_stats_); d_tri_stats_ = nullptr; }
        tri_stats_cap_ = 0;
        state_.tri_stats = nullptr;
        return;
    }
    if (!d_tri_stats_) {
        // Sized on demand in ensure_scratch; start empty.
        tri_stats_cap_ = 0;
    }
    stats_wanted_ = true;
}

void Pipeline::debug_read_tri_stats(int* out, int tri_count) const {
    if (!d_tri_stats_ || !out || tri_count <= 0) return;
    hipStreamSynchronize(stream_);
    hipMemcpy(out, d_tri_stats_,
              (size_t)tri_count * TRI_STAT_COUNT * sizeof(int32_t),
              hipMemcpyDeviceToHost);
}

bool Pipeline::debug_read_triangle(int tri_idx, TriangleSetup* out) const {
    if (!d_triangles_ || !out || tri_idx < 0 || tri_idx >= triangle_capacity_)
        return false;
    hipStreamSynchronize(stream_);
    return hipMemcpy(out, d_triangles_ + tri_idx, sizeof(TriangleSetup),
                     hipMemcpyDeviceToHost) == hipSuccess;
}


static inline long long hp_wavesort_max_tiles() {
    static const long long v = []{
        const char* e = getenv("HIPGL_WAVESORT_MAX_TILES");
        return e ? atoll(e) : (long long)(1LL << 40);
    }();
    return v;
}

static inline long long hp_wavesort_min_tiles() {
    static const long long v = []{
        const char* e = getenv("HIPGL_WAVESORT_MIN_TILES");
        return e ? atoll(e) : 0LL;
    }();
    return v;
}

static inline int hp_nc_slots() {
    static const int v = []{
        const char* e = getenv("HP_NC_SLOTS");
        int x = e ? atoi(e) : 64;
        return x < 1 ? 1 : x;
    }();
    return v;
}

static inline int hp_bin_pair_on() {
    static const int v = getenv("HP_BIN_PAIR") ? atoi(getenv("HP_BIN_PAIR")) : 3;
    return v;
}
static inline int hp_scan1_max() {
    static const int v = getenv("HP_SCAN1_MAX") ? atoi(getenv("HP_SCAN1_MAX")) : 20000;
    return v;
}
static inline int hp_bin_pair_max_tiles() {
    static const int v = getenv("HP_BIN_PAIR_MAX_TILES")
                       ? atoi(getenv("HP_BIN_PAIR_MAX_TILES")) : 1024;
    return v;
}
static inline int hp_bin_pair_dbg() {
    static const int v = getenv("HP_BIN_PAIR_DBG") ? atoi(getenv("HP_BIN_PAIR_DBG")) : 0;
    return v;
}
static inline int hp_bin_pair_hold() {
    static const int v = getenv("HP_BIN_PAIR_HOLD") ? atoi(getenv("HP_BIN_PAIR_HOLD")) : 1;
    return v;
}
static inline bool hp_subs_same(const kernels::BinSubdraws& a, const kernels::BinSubdraws& b) {
    if (a.nsub != b.nsub || a.nblk != b.nblk || a.dead_mask != b.dead_mask) return false;
    for (int i = 0; i < HP_PERM_WORDS; ++i) if (a.perm[i] != b.perm[i]) return false;
    for (int i = 0; i < HP_MAX_SUBDRAWS - 1; ++i) if (a.split[i] != b.split[i]) return false;
    return true;
}

kernels::BinRec* Pipeline::ensure_binrec(int output_tri_count) {
    const size_t bytes = (size_t)output_tri_count * sizeof(kernels::BinRec);
    if (!d_binrec_ || sz_binrec_ < bytes) {
        if (d_binrec_) hipFree(d_binrec_);
        d_binrec_ = nullptr;
        if (hipMalloc(&d_binrec_, bytes) != hipSuccess) { sz_binrec_ = 0; return nullptr; }
        sz_binrec_ = bytes;
    }
    return reinterpret_cast<kernels::BinRec*>(d_binrec_);
}

void Pipeline::bin_triangles_deterministic(int tri_count, int tiles_x, int tiles_y,
                                           int vw, int vh,
                                           const kernels::BinSubdraws& subs,
                                           bool binrec_ready) {
    const int num_tiles = tiles_x * tiles_y;
    kernels::TileRect rect;
    static const int portal_on = getenv("HP_PORTAL") ? atoi(getenv("HP_PORTAL")) : 1;
    if (portal_on && state_.reflect_mask && state_.reflect_rect[2] >= 0 &&
        state_.reflect_rect[3] >= 0 && tiles_x > 0 && tiles_y > 0) {
        int rx0 = state_.reflect_rect[0] / TILE_SIZE;
        int ry0 = state_.reflect_rect[1] / TILE_SIZE;
        int rx1 = state_.reflect_rect[2] / TILE_SIZE;
        int ry1 = state_.reflect_rect[3] / TILE_SIZE;
        if (rx0 < 0) rx0 = 0;
        if (ry0 < 0) ry0 = 0;
        if (rx1 > tiles_x - 1) rx1 = tiles_x - 1;
        if (ry1 > tiles_y - 1) ry1 = tiles_y - 1;
        static const int inset = getenv("HP_PORTAL_INSET") ? atoi(getenv("HP_PORTAL_INSET")) : 0;
        if (inset > 0) {
            rx0 += inset; ry0 += inset; rx1 -= inset; ry1 -= inset;
            if (rx1 < rx0) rx1 = rx0;
            if (ry1 < ry0) ry1 = ry0;
        }
        if (rx0 <= rx1 && ry0 <= ry1 &&
            !(rx0 == 0 && ry0 == 0 && rx1 == tiles_x - 1 && ry1 == tiles_y - 1)) {
            rect.x0 = rx0; rect.y0 = ry0; rect.x1 = rx1; rect.y1 = ry1;
        }
    }
    const int nsub = (subs.nsub > 1) ? subs.nblk : 1;
    const int num_slots = num_tiles * nsub;
    rect.nsub = nsub;
    work_ready_ = false;   // set only once the worklist is actually built below
    flt_ready_  = false;
    flt_msk_ready_ = false;
    fine_key_override_ = nullptr;
    fine_rec_override_ = nullptr;
    fine_cnt_override_ = nullptr;
    fine_msk_override_ = nullptr;
    if (num_tiles <= 0 || tri_count <= 0) return;
    if (tri_count > INT_MAX / 2)
        throw std::overflow_error("Pipeline: near-plane bin capacity exceeds int range");
    const int output_tri_count = 2 * tri_count;
    // DIAGNOSTIC: HP_BINABL 1=skip sort, 2=+skip scatter, 3=+skip count/scan.
    static const int hp_binabl = getenv("HP_BINABL") ? atoi(getenv("HP_BINABL")) : 0;
    if (hp_binabl >= 3) return;

    static const int cq_bp = getenv("HP_CQ_BP") ? atoi(getenv("HP_CQ_BP")) : 0;
    static const int cq_on_bp = getenv("HP_CQ") ? atoi(getenv("HP_CQ")) : 1;
    const bool cq_live = cq_on_bp != 0 && d_ext_matrices_ != nullptr && fb_ && fb_->samples() != 4;
    bool pair_arm = (cq_bp != 0 || !cq_live) &&
                    (hp_bin_pair_on() & 1) != 0 && num_tiles <= hp_bin_pair_max_tiles() &&
                    vtx_pair_ready_ && !vtx_pair_swapped_ &&
                    binrec_ready && hp_binabl == 0 &&
                    d_vtx1_tris_ && d_vtx1_rec_ && d_vtx1_nc_ids_ && d_vtx1_nc_count_ &&
                    vtx_pair_tris_ == tri_count && vtx_pair_vw_ == vw && vtx_pair_vh_ == vh &&
                    vtx_pair_state_.reflect_mask == nullptr &&
                    d_near_clip_ids_ && d_near_clip_count_ &&
                    !stats_wanted_ && !state_.tri_stats;
    // TAKE SIDE: this draw is the MAIN pass and its three stages are done.
    const bool pair_take_ok =
        bin_pair_.armed && vtx_pair_swapped_ && hp_binabl == 0 &&
        bin_pair_.tris == d_triangles_ &&
        bin_pair_.tri_count == tri_count && bin_pair_.tiles_x == tiles_x &&
        bin_pair_.tiles_y == tiles_y && bin_pair_.vw == vw && bin_pair_.vh == vh &&
        bin_pair_.nsub == nsub && bin_pair_.num_slots == num_slots &&
        bin_pair_.have_rec == binrec_ready && hp_subs_same(bin_pair_.subs, subs) &&
        bin_pair_.nc_ids == d_near_clip_ids_ && bin_pair_.nc_cnt == d_near_clip_count_ &&
        bin_pair_.rec == d_binrec_ &&
        sz_binrec_ >= (size_t)output_tri_count * sizeof(kernels::BinRec) &&
        !rect.active() && state_.reflect_mask == nullptr &&
        bin_pair_.cull_mask == kernels::backface_cull_mask(state_) &&
        (!bin_pair_.sorted ||
         (bin_pair_.pre_st.near_clip_source_count == state_.near_clip_source_count &&
          bin_pair_.pre_st.backface_cull_enabled  == (int)state_.backface_cull_enabled &&
          bin_pair_.pre_st.front_face             == state_.front_face &&
          bin_pair_.pre_st.cull_face              == state_.cull_face));
    bool pair_take = pair_take_ok;   // refined once bin_refill is known
    if (hp_bin_pair_dbg() && bin_pair_.armed) {
        static int o = 0;
        if (o++ < 6) fprintf(stderr, "[binpair] %s tris=%d\n", pair_take ? "TAKE" : "bail take", tri_count);
    }
    const Pipeline::BinPair taken = bin_pair_;
    if (bin_pair_.armed && !hp_bin_pair_hold()) bin_pair_ = Pipeline::BinPair{};
    if (pair_arm && pair_take) pair_arm = false;   // never both in one draw

    auto ensure = [&](void** ptr, size_t bytes, size_t* have) {
        if (*have >= bytes && *ptr) return true;
        if (*ptr) hipFree(*ptr);
        *ptr = nullptr;
        if (hipMalloc(ptr, bytes) != hipSuccess) { *have = 0; return false; }
        *have = bytes;
        return true;
    };

    const size_t np1 = (size_t)num_slots + 1;
    static const int wl_on = getenv("HIPGL_PERSIST") ? atoi(getenv("HIPGL_PERSIST")) : 1;
    static const int wl_min = getenv("HIPGL_PERSIST_MIN_TILES")
                            ? atoi(getenv("HIPGL_PERSIST_MIN_TILES")) : 2048;
    static const int wl_max = getenv("HIPGL_PERSIST_MAX_TILES")
                            ? atoi(getenv("HIPGL_PERSIST_MAX_TILES")) : 16384;
    const bool want_worklist = (nsub == 1) && (wl_on != 0) &&
                               (num_tiles >= wl_min) && (num_tiles <= wl_max);
    const size_t clear_ints = 2 * np1 + 5;
    static const int kBinSlots = []{
        const char* e = getenv("HP_BIN_SLOTS");
        int v = e ? atoi(e) : 32;
        if (v < 1) v = 1;
        return v;
    }();
    size_t slot_stride = (clear_ints + 3u) & ~(size_t)3u;
    if (bin_slot_stride_ > slot_stride) slot_stride = bin_slot_stride_;
    const size_t bin_scratch_ints = slot_stride * (size_t)kBinSlots
                                  + (want_worklist ? np1 : 0);
    const size_t bin_need_bytes = bin_scratch_ints * sizeof(int);
    // A changed stride re-lays-out every slot, and a realloc hands back
    // uninitialised memory: either way the whole pool must be re-zeroed and the
    // rotation restarted.
    bool bin_refill = (!d_bin_scratch_) || (sz_bin_scratch_ < bin_need_bytes) ||
                      (bin_slot_stride_ != slot_stride) || (bin_slots_ != kBinSlots);
    if (bin_refill || !d_overflow_persist_) cq_flush();
    if (!ensure((void**)&d_bin_scratch_, bin_need_bytes, &sz_bin_scratch_)) return;
    if (bin_refill) { bin_slot_stride_ = slot_stride; bin_slots_ = kBinSlots; bin_slot_ = 0; }
    if (bin_refill) pair_take = false;
    // The pair needs TWO zeroed blocks that both stay valid until the MAIN
    // draw's sort. Keep them inside one un-wrapped run of the rotation so the
    // wrap-fill cannot land between them.
    if (pair_arm && bin_slot_ + 1 >= kBinSlots) bin_slot_ = 0;
    // On the take path no slot is consumed at all -- and the wrap fill must NOT
    // run, since it would zero the block this draw is about to read.
    const bool bin_pool_wrapped = !pair_take && (bin_refill || bin_slot_ == 0);
    if (bin_pool_wrapped) cq_flush();
    if (bin_pool_wrapped)
        hipMemsetAsync(d_bin_scratch_, 0,
                       slot_stride * (size_t)kBinSlots * sizeof(int), stream_);
    if (bin_pair_.armed && (pair_take || pair_arm || bin_refill || bin_pool_wrapped))
        bin_pair_ = Pipeline::BinPair{};
    int* const d_bin_slot_ = d_bin_scratch_ + (size_t)bin_slot_ * slot_stride;
    int* d_bin_slot1_ = nullptr;
    if (!pair_take) {
        bin_slot_ = (bin_slot_ + 1 == kBinSlots) ? 0 : (bin_slot_ + 1);
        if (pair_arm) {
            d_bin_slot1_ = d_bin_scratch_ + (size_t)bin_slot_ * slot_stride;
            bin_slot_ = (bin_slot_ + 1 == kBinSlots) ? 0 : (bin_slot_ + 1);
        }
    }
    d_tile_counts_ = d_bin_slot_;                           // [num_slots+1]
    d_cursor_      = d_bin_slot_ + np1;                     // [num_slots]
    // d_overflow_ is deliberately NOT a slice of d_bin_scratch_ any more. That
    // block is zeroed by the hipMemsetAsync below on EVERY draw, so a clamp in
    // one draw was erased by the next and could never be observed after the
    // fact -- which is precisely why sizing had to synchronise in-line. Give it
    // its own word, never zeroed per draw, cleared only by take_overflow().
    // The scratch slot at 2*np1-1 is now unused (and harmlessly still zeroed).
    if (!d_overflow_persist_) {
        if (hipMalloc((void**)&d_overflow_persist_, sizeof(int)) != hipSuccess) {
            d_overflow_persist_ = nullptr;
            return;
        }
        hipMemsetAsync(d_overflow_persist_, 0, sizeof(int), stream_);
    }
    d_overflow_    = d_overflow_persist_;                   // [1] sticky
    // 2*np1 is the big-segment work-list counter (d_big_count below)
    d_work_count_  = d_bin_slot_ + 2 * np1 + 1;             // [2]
    d_work_ticket_ = d_bin_slot_ + 2 * np1 + 3;             // [2] {deep, shallow}
    d_work_list_   = d_bin_scratch_ + slot_stride * (size_t)kBinSlots;  // [num_tiles] (uncleared)
    if (pair_take) {
        d_tile_counts_ = taken.tile_counts;
        d_cursor_      = taken.cursor;
        d_work_count_  = taken.work_count;
        d_work_ticket_ = taken.work_ticket;
    }

    if (!ensure((void**)&d_bounds_, (4 * np1) * sizeof(int), &sz_bounds_)) return;
    d_tile_base_  = d_bounds_;                  // [num_tiles+1]
    d_tile_end_   = d_bounds_ + np1;            // kept for the debug helpers
    int* d_base_c = d_bounds_ + 2 * np1;
    int* d_end_c  = d_bounds_ + 3 * np1;
    if (pair_arm && !ensure((void**)&d_bin1_base_, np1 * sizeof(int), &sz_bin1_base_))
        pair_arm = false;
    if (pair_take) d_tile_base_ = taken.tile_base;

    if (!h_readback_) {
        if (hipHostMalloc((void**)&h_readback_, 4 * sizeof(int), hipHostMallocDefault) != hipSuccess) {
            h_readback_ = nullptr;
        } else {
            h_readback_[0] = 0; h_readback_[1] = 0; h_readback_[2] = 0; h_readback_[3] = 0;
        }
    }
    // The harness syncs worker streams at the end of render_gpu_batch, so those
    // copies are complete by the next batch. Multiple draws within a batch can
    // still see an older observation. Zero forces exact sizing below; a stale
    // positive count is only an estimate, so draw-size growth has its own guard.
    if (h_readback_ && have_readback_) {
        last_total_pairs_ = h_readback_[0];
        // Learn the worst-case pairs-per-triangle ratio from the observation and
        // the triangle count that produced it. Deferred sizing needs a ratio,
        // not an absolute capacity, to survive draws of differing sizes.
        if (last_total_pairs_ > 0 && rb_tri_count_ > 0) {
            const float r = (float)last_total_pairs_ / (float)rb_tri_count_;
            if (r > pairs_per_tri_max_) pairs_per_tri_max_ = r;
            if (rb_tiles_x_ == ppt_grid_x_ && rb_tiles_y_ == ppt_grid_y_) {
                const int rb_b = ppt_bucket_of(rb_tri_count_, rb_rect_tiles_);
                if (ppt_bucket_[rb_b] >= 0.0f && r > ppt_bucket_[rb_b])
                    ppt_bucket_[rb_b] = r;
            }
        }
    }

    static const int micro_on = getenv("HP_MICRO") ? atoi(getenv("HP_MICRO")) : 1;
    static const int micro_max_tris = []{
        const char* e = getenv("HP_MICRO_MAX_TRIS");
        return e ? atoi(e) : kernels::micro_bin_max_tris();
    }();
    static const int micro_max_tiles = []{
        const char* e = getenv("HP_MICRO_MAX_TILES");
        return e ? atoi(e) : kernels::micro_bin_max_tiles();
    }();
    // Hoisted out of the sizing block below -- the micro path has to know
    // before the count pass whether this draw would have synchronised.
    const bool pool_grid_changed_pre = tiles_x != pool_exact_tiles_x_ ||
                                       tiles_y != pool_exact_tiles_y_;
    const int  ppt_rect_tiles = rect.active()
                              ? ((rect.x1 - rect.x0 + 1) * (rect.y1 - rect.y0 + 1))
                              : num_tiles;
    const int  ppt_b     = ppt_bucket_of(output_tri_count, ppt_rect_tiles);
    const bool ppt_stale = (tiles_x != ppt_grid_x_) || (tiles_y != ppt_grid_y_);
    const bool ppt_unseen = ppt_stale || !(ppt_bucket_[ppt_b] >= 0.0f);
    static const int pool_exact_env_pre =
        getenv("HP_POOL_EXACT") ? atoi(getenv("HP_POOL_EXACT")) : -1;
    static const int probe_sync_pre = getenv("HIPGL_PROBE_SYNC") ? atoi(getenv("HIPGL_PROBE_SYNC")) : 0;
    const bool need_exact_pre =
        !(defer_pool_sizing_ && pool_exact_env_pre != 1) || pool_grid_changed_pre ||
        pool_capacity_ <= 0 || force_exact_ || ppt_unseen;
    const bool use_micro = micro_on != 0 && nsub == 1 && !need_exact_pre &&
                           !probe_sync_pre && hp_binabl == 0 &&
                           tri_count <= micro_max_tris &&
                           num_tiles <= micro_max_tiles &&
                           num_slots == num_tiles;

    kernels::BinRec* d_binrec = ensure_binrec(output_tri_count);
    if (!d_binrec) return;
    // Apply the backface rule HERE rather than only in the fine rasterizer, so
    // back-facing triangles never enter the pool, the scan, the scatter or the
    // sort. The count pass stamps an empty BinRec (nx=1, ny=0) for anything it
    // rejects, so the scatter needs no change -- it simply emits no pairs.
    const int cull_mask = kernels::backface_cull_mask(state_);
    pair_arm = pair_arm && !use_micro && d_bin_slot1_ && d_bin1_base_ &&
               (int)np1 <= hp_scan1_max();
    static const int vc_fuse_env = getenv("HP_VC_FUSE") ? atoi(getenv("HP_VC_FUSE")) : 1;
    static const int vc_fx_env = getenv("HP_FUSE_EXTRAS") ? atoi(getenv("HP_FUSE_EXTRAS")) : 1;
    const bool vc_fuse = pending_vtx_ && vc_fuse_env != 0 && vc_fx_env != 0 &&
                         !use_micro && !pair_arm && !pair_take && binrec_ready &&
                         hp_binabl == 0 && num_tiles < 65536 &&
                         d_near_clip_ids_ && d_near_clip_count_ && d_binrec == d_binrec_;
    struct CqBinGuard {
        Pipeline* p; bool done = false;
        ~CqBinGuard() {
            kernels::cq_rec_slot() = nullptr;
            if (p->cq_recording_ && !done) p->cq_abandon();
        }
    } cq_guard{this};
    if (cq_draw_) {
        cq_draw_ = false;
        static const int dbg_pool_cq = []{
            const char* e = getenv("HP_DBG_POOL");
            return e ? atoi(e) : HP_DBG_POOL_DEFAULT;
        }();
        const bool ms4 = fb_ && fb_->samples() == 4;
        const bool elig =
            vc_fuse && !use_micro && !pair_arm && !pair_take && !pending_mc_armed_ &&
            !need_exact_pre && !probe_sync_pre && hp_binabl == 0 && !want_worklist &&
            ((long long)num_slots >= hp_wavesort_min_tiles()) &&
            ((long long)num_slots <= hp_wavesort_max_tiles()) &&
            !dbg_pool_cq && !ms4 && !stats_wanted_ && !state_.tri_stats &&
            d_ext_matrices_ && deterministic_binning_ && h_readback_ &&
            kernels::scatter_scan_fold_ok(tiles_x, tiles_y, tri_count, ms4, (int)np1);
        if (elig) {
            cq_cur_ = kernels::CqChainRec{};
            cq_cur_.want = 7;
            cq_cur_.miss = [](void* self) { static_cast<Pipeline*>(self)->cq_abandon(); };
            cq_cur_.miss_ctx = this;
            cq_rb_ = false; cq_rb_src_ = nullptr;
            cq_recording_ = true;
            kernels::cq_rec_slot() = &cq_cur_;
        } else {
            cq_flush();
        }
    }
    if (!vc_fuse) vtx_flush_pending();
    kernels::BinCntSlot cs1{};
    bool pair_issued = false;
    if (pair_arm) {
        cs1.tris        = d_vtx1_tris_;
        cs1.tile_counts = d_bin_slot1_;
        cs1.binrec      = reinterpret_cast<kernels::BinRec*>(d_vtx1_rec_);
        cs1.queue       = {d_vtx1_nc_ids_, d_vtx1_nc_count_};
        cs1.rect        = kernels::TileRect{};
        cs1.cull_mask   = kernels::backface_cull_mask(vtx_pair_state_);
        cs1.packed_ids  = true;
        cs1.have_rec    = true;
        kernels::BinCntSlot cs0{};
        cs0.tris        = d_triangles_;
        cs0.tile_counts = d_tile_counts_;
        cs0.binrec      = d_binrec;
        cs0.queue       = {d_near_clip_ids_, d_near_clip_count_};
        cs0.rect        = rect;
        cs0.cull_mask   = cull_mask;
        cs0.packed_ids  = true;
        cs0.have_rec    = binrec_ready;
        pair_issued = kernels::launch_count_tile_pairs_z2_mc(
            cs0, cs1, tri_count, tiles_x, tiles_y, vw, vh, subs, stream_,
            pending_mc_armed_ ? &pending_mc_ : nullptr);
        if (pair_issued) pending_mc_armed_ = false;
        if (!pair_issued) {
            pair_arm = false;
            if (hp_bin_pair_dbg()) {
                static int o = 0;
                if (o++ < 6) fprintf(stderr, "[binpair] bail launcher\n");
            }
        } else if (hp_bin_pair_dbg()) {
            static int o = 0;
            if (o++ < 6) fprintf(stderr, "[binpair] ENGAGED z=2 tris=%d tiles=%dx%d nsub=%d\n",
                                 tri_count, tiles_x, tiles_y, nsub);
        }
    }
    if (!use_micro && !pair_issued && !pair_take) {
        bool vc_counted = false;
        if (vc_fuse && pending_vtx_) {
            kernels::VtxCntFuse cf;
            cf.tile_counts = d_tile_counts_;
            cf.rect        = rect;
            std::function<bool(const kernels::VtxCntFuse*)>* go = pending_vtx_;
            pending_vtx_ = nullptr;
            vc_counted = (*go)(&cf);
            if (cq_recording_ && !(cq_cur_.got & 1)) cq_abandon();
        }
        if (!vc_counted)
        kernels::launch_count_tile_pairs(d_triangles_, tri_count, d_tile_counts_, d_binrec,
                                         tiles_x, tiles_y, vw, vh, stream_,
                                         {d_near_clip_ids_, d_near_clip_count_}, cull_mask, subs,
                                         binrec_ready, rect);
    }
    vtx_flush_pending();

    mc_flush();

    const int scan1_max = hp_scan1_max();
    const bool fold_scan =
        !use_micro && !pair_take && !pair_issued &&
        !need_exact_pre && !probe_sync_pre && hp_binabl == 0 &&
        kernels::scatter_scan_fold_ok(tiles_x, tiles_y, tri_count,
                                      fb_ && fb_->samples() == 4, (int)np1);
    if (fold_scan) {
        // the scatter dispatch below carries it
    } else if (use_micro || pair_take) {
        // the fused kernel scans in-CTA; the pair's scan already ran as z-slot 1
    } else if ((int)np1 <= scan1_max) {
        if (!pair_issued ||
            !kernels::launch_scan_excl_single_z2(d_tile_counts_, d_tile_base_,
                                                 d_bin_slot1_, d_bin1_base_,
                                                 (int)np1, stream_)) {
            kernels::launch_scan_excl_single(d_tile_counts_, d_tile_base_, (int)np1, stream_);
            if (pair_issued)
                kernels::launch_scan_excl_single(d_bin_slot1_, d_bin1_base_, (int)np1, stream_);
        }
    } else {
    size_t need = 0;
    rocprim::exclusive_scan(nullptr, need, d_tile_counts_, d_tile_base_, 0, (int)np1,
                            ::rocprim::plus<int>(), stream_);
    if (need > scan_tmp_bytes_) {
        if (d_scan_tmp_) hipFree(d_scan_tmp_);
        if (hipMalloc(&d_scan_tmp_, need) != hipSuccess) { d_scan_tmp_ = nullptr; scan_tmp_bytes_ = 0; return; }
        scan_tmp_bytes_ = need;
    }
    rocprim::exclusive_scan(d_scan_tmp_, need, d_tile_counts_, d_tile_base_, 0, (int)np1,
                            ::rocprim::plus<int>(), stream_);
    }

    const long long hard_bound = (long long)output_tri_count * (long long)num_tiles;
    const bool pool_grid_changed = tiles_x != pool_exact_tiles_x_ ||
                                   tiles_y != pool_exact_tiles_y_;
    // HP_POOL_EXACT=0 drops the per-draw blocking readback and sizes the pool
    // from the retained high-water mark instead. The scatter still sets
    // d_overflow_ (atomicExch, bin_rasterize.hip:3131) whenever it clamps, so an
    // undersized pool is DETECTABLE -- but until something consumes that flag
    // before the image is used, =0 is a MEASUREMENT PATH ONLY: it can silently
    // drop geometry exactly as the comment above describes. Default 1 keeps the
    // documented correctness behaviour. Still forced on the two cases where the
    // retained mark carries no information: a changed tile grid, and the first
    // draw of a fresh Pipeline.
    // HP_POOL_EXACT=1 is an escape hatch that restores the original
    // synchronise-every-draw behaviour; unset honours the caller's opt-in.
    static const int pool_exact_env =
        getenv("HP_POOL_EXACT") ? atoi(getenv("HP_POOL_EXACT")) : -1;
    const bool may_defer = defer_pool_sizing_ && pool_exact_env != 1;
    const bool need_exact =
        !may_defer || pool_grid_changed || pool_capacity_ <= 0 || force_exact_ ||
        ppt_unseen;
    long long total_pairs = -1;
    long long total_pairs1 = -1;
    static const int probe_sync = getenv("HIPGL_PROBE_SYNC") ? atoi(getenv("HIPGL_PROBE_SYNC")) : 0;
    if (cq_recording_ && (need_exact || probe_sync)) cq_abandon();
    if (need_exact || probe_sync) {
        if (h_readback_) {
            hipMemcpyAsync(h_readback_ + 2, d_tile_base_ + num_slots, sizeof(int),
                           hipMemcpyDeviceToHost, stream_);
            if (pair_issued)
                hipMemcpyAsync(h_readback_ + 3, d_bin1_base_ + num_slots, sizeof(int),
                               hipMemcpyDeviceToHost, stream_);
            hipStreamSynchronize(stream_);
            total_pairs = (long long)h_readback_[2];
            if (pair_issued) total_pairs1 = (long long)h_readback_[3];
        } else {
            int tp = 0;
            hipMemcpyAsync(&tp, d_tile_base_ + num_slots, sizeof(int),
                           hipMemcpyDeviceToHost, stream_);
            hipStreamSynchronize(stream_);
            total_pairs = tp;
        }
        // The exact path knows both numbers right now, so seed the ratio from
        // it directly instead of waiting for the throttled async observation.
        if (total_pairs > 0 && output_tri_count > 0) {
            const float r = (float)total_pairs / (float)output_tri_count;
            if (r > pairs_per_tri_max_) pairs_per_tri_max_ = r;
        }
        if (total_pairs >= 0 && output_tri_count > 0) {
            if (ppt_stale) {
                for (int i = 0; i < kPptBuckets; ++i) ppt_bucket_[i] = -1.0f;
                ppt_grid_x_ = tiles_x;
                ppt_grid_y_ = tiles_y;
            }
            const long long tp_seed = (total_pairs1 > total_pairs) ? total_pairs1
                                                                   : total_pairs;
            const float r = (float)tp_seed / (float)output_tri_count;
            if (!(ppt_bucket_[ppt_b] >= r)) ppt_bucket_[ppt_b] = r;
        }
        if (getenv("HP_DBG_POOL"))
            fprintf(stderr, "[pool] tris=%d tiles=%d nsub=%d pairs=%lld  pairs/tri=%.3f entries/tile=%.2f\n",
                    tri_count, num_tiles, nsub, total_pairs,
                    (double)total_pairs/(double)tri_count, (double)total_pairs/(double)num_slots);
    }

    long long want = pool_capacity_;
    if (total_pairs1 > total_pairs) {
        // Same sizing rule, applied to slot 1 as well (see total_pairs1).
        long long t1 = total_pairs1 + total_pairs1 / 2 + 4096;
        if (t1 > hard_bound) t1 = hard_bound;
        if (t1 > (long long)INT_MAX) t1 = INT_MAX;
        if (t1 > want) want = t1;
    }
    if (total_pairs >= 0) {
        long long target = total_pairs + total_pairs / 2 + 4096;
        if (target > hard_bound) target = hard_bound;
        if (target > (long long)INT_MAX) target = INT_MAX;
        if (target > want) want = target;
    } else {
        const float ratio = (ppt_bucket_[ppt_b] >= 0.0f)
                          ? (ppt_bucket_[ppt_b] * kPoolDeferSlack)
                          : (pairs_per_tri_max_ * 1.5f + 1.0f);
        long long est = (long long)((double)output_tri_count * (double)ratio) + 4096;
        if (est > hard_bound) est = hard_bound;
        if (est > want) want = est;
    }
    if (want > (long long)pool_capacity_) {
        if (d_pool_tri_) hipFree(d_pool_tri_);
        if (d_pool_z_)   hipFree(d_pool_z_);
        d_pool_tri_ = nullptr; d_pool_z_ = nullptr;
        if (hipMalloc(&d_pool_tri_, (size_t)want * sizeof(int))   != hipSuccess ||
            hipMalloc(&d_pool_z_,   (size_t)want * sizeof(float)) != hipSuccess) {
            pool_capacity_ = 0;
            return;   // cannot size the pool; skip rather than fault
        }
        pool_capacity_ = (int)want;
        if (pool_capacity_ > cq_pool_hw_) cq_pool_hw_ = pool_capacity_;
    }
    if (pool_capacity_ <= 0) return;

    static const int dbg_pool = []{
        const char* e = getenv("HP_DBG_POOL");
        return e ? atoi(e) : HP_DBG_POOL_DEFAULT;
    }();
    if (dbg_pool) {
        int obs = -1, obs1 = -1;
        if (dbg_pool >= 2) {
            hipMemcpyAsync(&obs, d_tile_base_ + num_slots, sizeof(int),
                           hipMemcpyDeviceToHost, stream_);
            if (pair_issued)
                hipMemcpyAsync(&obs1, d_bin1_base_ + num_slots, sizeof(int),
                               hipMemcpyDeviceToHost, stream_);
            hipStreamSynchronize(stream_);
        }
        const int worst = std::max(1, std::max(obs, obs1));
        fprintf(stderr,
                "[pool] %s grid=%dx%d nsub=%d slots=%d otri=%d obs0=%d obs1=%d "
                "cap=%d over=%.1fx ppt=%.3f fx=%d\n",
                (total_pairs >= 0 ? "EXACT" : "DEFER"), tiles_x, tiles_y, nsub,
                num_slots, output_tri_count, obs, obs1, pool_capacity_,
                (double)pool_capacity_ / (double)worst,
                (double)pairs_per_tri_max_, (int)force_exact_);
    }

    total_pairs_estimate_ = pool_capacity_;
    // PROBE: hand rocPRIM the EXACT key count instead of the whole allocation.
    static const int probe_exact = getenv("HIPGL_SORT_EXACT") ? atoi(getenv("HIPGL_SORT_EXACT")) : 0;
    int sort_extent = pool_capacity_;
    if (probe_exact && total_pairs >= 0 && total_pairs <= pool_capacity_)
        sort_extent = (int)total_pairs;
    // The two former sort double-buffers are re-purposed as the 64-bit COMPOSITE
    // KEY ping/pong. Same total footprint as the old (int,float) pair of pairs,
    // no new members: hipMalloc is 256 B aligned so the reinterpret is legal.
    // d_pool_sorted_tri_ always holds the SORTED keys after the block below.
    if (!ensure((void**)&d_pool_sorted_tri_, (size_t)pool_capacity_ * sizeof(unsigned long long), &sz_st_)) return;
    if (!ensure((void**)&d_pool_sorted_z_,   (size_t)pool_capacity_ * sizeof(unsigned long long), &sz_sz_)) return;
    bool flt_alloc = ensure(&d_flt_rec_, (size_t)pool_capacity_ * 64u, &sz_flt_rec_) &&
                     ensure((void**)&d_flt_cnt_, (size_t)num_slots * sizeof(int), &sz_flt_cnt_) &&
                     (num_tiles > 1024 ||
                      ensure((void**)&d_flt_msk_, (size_t)pool_capacity_ * 2u, &sz_flt_msk_));
    // Do not mark a draw as sized until all pool allocations have succeeded.
    // Grid dimensions matter even when their product (num_tiles) is unchanged.
    if (total_pairs >= 0) {
        pool_exact_max_triangles_ = pool_grid_changed ? output_tri_count :
            std::max(pool_exact_max_triangles_, output_tri_count);
        pool_exact_tiles_x_ = tiles_x;
        pool_exact_tiles_y_ = tiles_y;
    }

    // ROUND-1 (memory lane): kernel_tile_bounds_clamped retired -- the clamp is
    // two int loads and is now done inside launch_tile_wave_sort itself.
    // d_base_c/d_end_c (2*np1 ints) are dead now that the clamp is done inside
    // the sort kernel; re-purposed as the big-segment work list.
    int* d_big_list  = d_base_c;
    int* d_big_count = d_tile_counts_ + 2 * np1;
    const int big_cap = (int)(2 * np1);


    if (hp_binabl >= 2) return;
    static const long long directkey_min_tiles = []{
        const char* e = getenv("HIPGL_DIRECTKEY_MIN_TILES");
        return e ? atoll(e) : 16385LL;
    }();
    const bool use_wave_sort = ((long long)num_slots >= hp_wavesort_min_tiles()) &&
                               ((long long)num_slots <= hp_wavesort_max_tiles());
    const bool direct_keys = ((long long)num_slots >= 0) || (!use_wave_sort &&
                             ((long long)num_slots >= directkey_min_tiles));
    unsigned long long* d_key_in_ = reinterpret_cast<unsigned long long*>(d_pool_sorted_z_);
    if (pair_take && taken.scattered && taken.key_in) d_key_in_ = taken.key_in;
    bool micro_done = false;
    if (use_micro) {
        micro_done = kernels::launch_micro_bin_chain(
            d_triangles_, tri_count, d_tile_base_, d_cursor_,
            reinterpret_cast<unsigned long long*>(d_pool_sorted_tri_),
            tiles_x, tiles_y, vw, vh, num_tiles, pool_capacity_, d_overflow_,
            cull_mask, {d_near_clip_ids_, d_near_clip_count_}, stream_);
        if (!micro_done) {
            // Defensive: the host gate and the launcher disagreed. Replay the
            // two stages the gate skipped and fall through to the usual chain.
            kernels::launch_count_tile_pairs(d_triangles_, tri_count, d_tile_counts_, d_binrec,
                                             tiles_x, tiles_y, vw, vh, stream_,
                                             {d_near_clip_ids_, d_near_clip_count_}, cull_mask, subs,
                                             binrec_ready, rect);
            kernels::launch_scan_excl_single(d_tile_counts_, d_tile_base_, (int)np1, stream_);
        }
    }
    if (!micro_done) {
    bool scat_z2 = false;
    if (pair_issued && (hp_bin_pair_on() & 2) != 0 && direct_keys && pool_capacity_ > 0 &&
        ensure(&d_bin1_key_, (size_t)pool_capacity_ * sizeof(unsigned long long), &sz_bin1_key_)) {
        kernels::BinScatSlot ss0{}, ss1{};
        ss0.tris = d_triangles_;   ss0.binrec = d_binrec;
        ss0.tile_base = d_tile_base_; ss0.cursor = d_cursor_;
        ss0.pool_tri = d_pool_tri_; ss0.pool_z = d_pool_z_;
        ss0.pool_key = d_key_in_;   ss0.overflow_flag = d_overflow_;
        ss0.queue = {d_near_clip_ids_, d_near_clip_count_};
        ss0.rect = rect;            ss0.pool_capacity = pool_capacity_;
        ss1.tris = d_vtx1_tris_;   ss1.binrec = reinterpret_cast<const kernels::BinRec*>(d_vtx1_rec_);
        ss1.tile_base = d_bin1_base_; ss1.cursor = d_bin_slot1_ + np1;
        ss1.pool_tri = d_pool_tri_; ss1.pool_z = d_pool_z_;
        ss1.pool_key = reinterpret_cast<unsigned long long*>(d_bin1_key_);
        ss1.overflow_flag = d_overflow_;
        ss1.queue = {d_vtx1_nc_ids_, d_vtx1_nc_count_};
        ss1.rect = kernels::TileRect{}; ss1.pool_capacity = pool_capacity_;
        scat_z2 = kernels::launch_scatter_tile_pairs_z2(ss0, ss1, tri_count, tiles_x, tiles_y,
                                                        vw, vh, fb_ && fb_->samples() == 4, stream_);
    }
    if (!scat_z2 && !(pair_take && taken.scattered))
    kernels::launch_scatter_tile_pairs(d_triangles_, d_binrec, tri_count, d_tile_base_, d_cursor_,
                                       d_pool_tri_, d_pool_z_,
                                       direct_keys ? d_key_in_ : nullptr,
                                       tiles_x, tiles_y, vw, vh,
                                       pool_capacity_, d_overflow_, stream_, fb_ && fb_->samples() == 4,
                                       {d_near_clip_ids_, d_near_clip_count_}, rect,
                                       // r4: fold the scan in when it was skipped above.
                                       fold_scan ? d_tile_counts_ : nullptr,
                                       fold_scan ? d_tile_base_ : nullptr,
                                       fold_scan ? (int)np1 : 0);
    if (cq_recording_ && !(cq_cur_.got & 2)) cq_abandon();
    if (pair_issued) {
        bin_pair_.armed        = true;
        bin_pair_.scattered    = scat_z2;
        bin_pair_.tris         = d_vtx1_tris_;
        bin_pair_.rec          = d_vtx1_rec_;
        bin_pair_.nc_ids       = d_vtx1_nc_ids_;
        bin_pair_.nc_cnt       = d_vtx1_nc_count_;
        bin_pair_.tri_count    = tri_count;
        bin_pair_.tiles_x      = tiles_x;   bin_pair_.tiles_y = tiles_y;
        bin_pair_.vw           = vw;        bin_pair_.vh      = vh;
        bin_pair_.nsub         = nsub;      bin_pair_.num_slots = num_slots;
        bin_pair_.cull_mask    = cs1.cull_mask;
        bin_pair_.have_rec     = binrec_ready;
        bin_pair_.subs         = subs;
        bin_pair_.tile_counts  = d_bin_slot1_;
        bin_pair_.cursor       = d_bin_slot1_ + np1;
        bin_pair_.work_count   = d_bin_slot1_ + 2 * np1 + 1;
        bin_pair_.work_ticket  = d_bin_slot1_ + 2 * np1 + 3;
        bin_pair_.tile_base    = d_bin1_base_;
        bin_pair_.key_in       = scat_z2 ? reinterpret_cast<unsigned long long*>(d_bin1_key_) : nullptr;
        bin_pair_.pool_capacity = pool_capacity_;
        bin_pair_.total_pairs  = total_pairs1;
    }
    // The scatter may WRITE FEWER pairs than the conservative scan RESERVED
    // (exact triangle-vs-tile rejection), so from here on the per-tile segment
    // length is d_cursor_ -- the number actually written -- not d_tile_counts_.
    // d_tile_base_ still comes from the conservative scan, so segments stay
    // contiguous and simply end early. With exact rejection off the two arrays
    // are identical, so this is unconditional.
    int* d_seg_counts = d_cursor_;
    // Compact the non-empty tiles for the persistent fine rasterizer.  Issued
    // here, BEFORE the (much longer) per-tile sort, so its ~num_tiles/256 CTAs
    // overlap with the sort on the same stream's tail rather than sitting on
    // the critical path in front of the fine dispatch.
    {
        static const int deep_thresh = getenv("HIPGL_PERSIST_DEEP")
                                     ? atoi(getenv("HIPGL_PERSIST_DEEP")) : 64;
        work_ready_ = want_worklist;
        if (work_ready_)
            kernels::launch_build_tile_worklist(d_seg_counts, num_tiles, deep_thresh,
                                                d_work_list_, d_work_count_, stream_);
    }
    if (hp_binabl >= 1) return;
    const bool sort_taken = pair_take && taken.sorted && taken.key_out;
    if (sort_taken) {
        fine_key_override_ = taken.key_out;
        fine_rec_override_ = taken.flt_rec;
        fine_cnt_override_ = taken.flt_cnt;
        fine_msk_override_ = taken.flt_msk;
        flt_ready_     = (taken.flt_rec != nullptr);
        flt_msk_ready_ = taken.flt_msk_ready;
    }
    if (!sort_taken) {
        unsigned long long* d_key_out = reinterpret_cast<unsigned long long*>(d_pool_sorted_tri_);
        if (use_wave_sort) {
            kernels::FinePreEmit pre{};
            if (flt_alloc && kernels::tile_sort_fused_enabled()) {
                pre.tris      = d_triangles_;
                pre.rec       = reinterpret_cast<float4*>(d_flt_rec_);
                pre.cnt       = d_flt_cnt_;
                flt_msk_ready_ = (num_tiles <= 1024);
                pre.msk       = flt_msk_ready_ ? d_flt_msk_ : nullptr;
                pre.st        = kernels::FinePreState{state_.near_clip_source_count,
                                                      state_.backface_cull_enabled,
                                                      state_.front_face, state_.cull_face};
                pre.tiles_x   = tiles_x;
                pre.tiles_mod = num_tiles;
                flt_ready_    = true;
            }
            bool sort_z2 = false;
            if (pair_issued && bin_pair_.armed && scat_z2 && pre.rec &&
                bin_pair_.key_in && bin_pair_.tile_base && bin_pair_.cursor &&
                ensure(&d_bin1_keyout_, (size_t)pool_capacity_ * sizeof(unsigned long long),
                       &sz_bin1_keyout_) &&
                ensure(&d_bin1_flt_rec_, (size_t)pool_capacity_ * 64u, &sz_bin1_flt_rec_) &&
                ensure((void**)&d_bin1_flt_cnt_, (size_t)num_slots * sizeof(int),
                       &sz_bin1_flt_cnt_) &&
                (!flt_msk_ready_ ||
                 ensure(&d_bin1_flt_msk_, (size_t)pool_capacity_ * 2u, &sz_bin1_flt_msk_))) {
                kernels::FinePreEmit pre1 = pre;
                pre1.tris = bin_pair_.tris;
                pre1.rec  = reinterpret_cast<float4*>(d_bin1_flt_rec_);
                pre1.cnt  = d_bin1_flt_cnt_;
                pre1.msk  = flt_msk_ready_
                          ? reinterpret_cast<unsigned short*>(d_bin1_flt_msk_) : nullptr;
                pre1.st   = kernels::FinePreState{vtx_pair_state_.near_clip_source_count,
                                                  vtx_pair_state_.backface_cull_enabled,
                                                  vtx_pair_state_.front_face,
                                                  vtx_pair_state_.cull_face};
                kernels::SortSlot s0{}, s1{};
                s0.keys = d_key_in_;        s0.tile_base  = d_tile_base_;
                s0.tile_count = d_seg_counts; s0.out = d_key_out;  s0.pre = pre;
                s1.keys = bin_pair_.key_in; s1.tile_base  = bin_pair_.tile_base;
                s1.tile_count = bin_pair_.cursor;
                s1.out  = reinterpret_cast<unsigned long long*>(d_bin1_keyout_);
                s1.pre  = pre1;
                sort_z2 = kernels::launch_tile_wave_sort_keys_z2(
                    s0, s1, num_slots, pool_capacity_, num_tiles, stream_);
                if (sort_z2) {
                    bin_pair_.sorted        = true;
                    bin_pair_.key_out       = s1.out;
                    bin_pair_.flt_rec       = d_bin1_flt_rec_;
                    bin_pair_.flt_cnt       = d_bin1_flt_cnt_;
                    bin_pair_.flt_msk       = flt_msk_ready_ ? d_bin1_flt_msk_ : nullptr;
                    bin_pair_.flt_msk_ready = flt_msk_ready_;
                    bin_pair_.pre_st        = pre1.st;
                }
            }
            if (!sort_z2)
            kernels::launch_tile_wave_sort_keys(d_key_in_,
                                           d_tile_base_, d_seg_counts,
                                           d_key_out, d_big_list, d_big_count, big_cap,
                                           num_slots, pool_capacity_, stream_,
                                           num_tiles, pre);
            if (cq_recording_ && !(cq_cur_.got & 4)) cq_abandon();
        } else {
            unsigned long long* d_key_in = d_key_in_;
            kernels::launch_tile_bounds_clamped(d_tile_base_, d_seg_counts,
                                                d_base_c, d_end_c, num_slots,
                                                sort_extent, stream_);
            if (!direct_keys)
                kernels::launch_pack_pool_keys(d_pool_tri_, d_pool_z_, d_key_in,
                                               d_tile_base_ + num_slots, pool_capacity_, stream_);
            size_t sbytes = 0;
            rocprim::segmented_radix_sort_keys(
                nullptr, sbytes, d_key_in, d_key_out,
                sort_extent, num_slots, d_base_c, d_end_c,
                0, (int)(sizeof(unsigned long long) * 8), stream_);
            if (sbytes > sort_tmp_bytes_) {
                if (d_sort_tmp_) hipFree(d_sort_tmp_);
                if (hipMalloc(&d_sort_tmp_, sbytes) != hipSuccess) { d_sort_tmp_ = nullptr; sort_tmp_bytes_ = 0; return; }
                sort_tmp_bytes_ = sbytes;
            }
            rocprim::segmented_radix_sort_keys(
                d_sort_tmp_, sbytes, d_key_in, d_key_out,
                sort_extent, num_slots, d_base_c, d_end_c,
                0, (int)(sizeof(unsigned long long) * 8), stream_);
        }
    }
    }   // !micro_done
    // kernel_pool_to_tilebins retired: the fine rasterizer now reads the pool
    // SoA (d_tile_base_/d_tile_counts_/d_pool_tri_/d_pool_z_) directly, so the
    // 32772 B/tile TileBin AoS mirror no longer has to be built. Same entries,
    // same order, same image; one fewer dispatch and 2.1 GB/Pipeline of HBM
    // traffic and footprint removed at r1024.

    static const int rb_period = []{
        const char* e = getenv("HP_RB_PERIOD"); return e ? atoi(e) : 8;
    }();
    static const int rb_thin_num = []{
        const char* e = getenv("HP_RB_THIN_NUM"); return e ? atoi(e) : 9;
    }();
    static const int rb_thin_den = []{
        const char* e = getenv("HP_RB_THIN_DEN"); return e ? atoi(e) : 8;
    }();
    if (h_readback_ && (--rb_countdown_ <= 0 || !have_readback_ || need_exact ||
                        (long long)pool_capacity_ <
                            (long long)last_total_pairs_ * rb_thin_num / rb_thin_den + 4096)) {
        rb_countdown_ = rb_period;
        if (cq_recording_) {
            // observation's metadata is published together with its copy so
            // the host never pairs a count with another draw's triangle count.
            cq_rb_ = true;
            cq_rb_src_ = d_tile_base_ + num_slots;
            cq_rb_meta_[0] = output_tri_count; cq_rb_meta_[1] = tiles_x;
            cq_rb_meta_[2] = tiles_y; cq_rb_meta_[3] = ppt_rect_tiles;
        } else {
            rb_tri_count_ = output_tri_count;   // pairs this observation belongs to
            rb_tiles_x_   = tiles_x;
            rb_tiles_y_   = tiles_y;
            rb_rect_tiles_ = ppt_rect_tiles;    // ... and the portal class
            hipMemcpyAsync(h_readback_ + 0, d_tile_base_ + num_slots, sizeof(int),
                           hipMemcpyDeviceToHost, stream_);
        }
        have_readback_ = true;
    }
    static const int cq_aball = getenv("HP_CQ_ABALL") ? atoi(getenv("HP_CQ_ABALL")) : 0;
    if (cq_recording_ && cq_cur_.got == 7 && !cq_aball) cq_guard.done = true;
}

unsigned long long Pipeline::resource_fingerprint() const {
    unsigned long long h = 1469598103934665603ULL;
    static const int fpdbg = getenv("HP_FPDBG") ? atoi(getenv("HP_FPDBG")) : 0;
    int fpi = 0;
    const char* fpname[64];
    unsigned long long fpval[64];
    auto mix = [&](const void* p, const char* nm) {
        unsigned long long v = (unsigned long long)(uintptr_t)p;
        if (fpdbg && fpi < 64) { fpname[fpi] = nm; fpval[fpi] = v; ++fpi; }
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
#define HP_FPMIX(x) mix((const void*)(x), #x)
    HP_FPMIX(d_transformed_); HP_FPMIX(d_triangles_); HP_FPMIX(d_near_clip_ids_);
    // NB d_near_clip_count_ / d_vtx1_nc_count_ are deliberately NOT mixed in:
    // they are rotating offsets INTO d_nc_pool_, not allocations, and
    // reset_nc_rotation() makes the captured region independent of them.
    HP_FPMIX(d_nc_pool_); HP_FPMIX(d_tile_bins_);
    HP_FPMIX(d_tile_counts_); HP_FPMIX(d_tile_base_); HP_FPMIX(d_cursor_);
    HP_FPMIX(d_pool_tri_); HP_FPMIX(d_pool_z_); HP_FPMIX(d_overflow_); HP_FPMIX(d_scan_tmp_);
    HP_FPMIX(d_tile_end_); HP_FPMIX(d_pool_sorted_tri_); HP_FPMIX(d_pool_sorted_z_);
    HP_FPMIX(d_sort_tmp_); HP_FPMIX(d_bin_scratch_); HP_FPMIX(d_work_list_);
    HP_FPMIX(d_work_count_); HP_FPMIX(d_work_ticket_); HP_FPMIX(d_flt_rec_); HP_FPMIX(d_flt_cnt_);
    HP_FPMIX(d_bounds_); HP_FPMIX(d_binrec_); HP_FPMIX(d_overflow_persist_); HP_FPMIX(d_flt_msk_);
    HP_FPMIX(d_micro_tris_); HP_FPMIX(d_micro_bins_); HP_FPMIX(d_micro_pool_);
    HP_FPMIX(d_vtx1_tris_); HP_FPMIX(d_vtx1_rec_); HP_FPMIX(d_vtx1_nc_ids_);
    HP_FPMIX(d_bin1_base_); HP_FPMIX(d_bin1_key_);
    HP_FPMIX(d_positions_); HP_FPMIX(d_normals_); HP_FPMIX(d_texcoords_); HP_FPMIX(d_texids_);
    HP_FPMIX(d_colors_); HP_FPMIX(d_spec_colors_); HP_FPMIX(d_indices_); HP_FPMIX(d_matrices_);
    HP_FPMIX((uintptr_t)pool_capacity_);
#undef HP_FPMIX
    if (fpdbg) {
        if (fp_dbg_n_ == fpi) {
            for (int k = 0; k < fpi; ++k)
                if (fp_dbg_[k] != fpval[k])
                    fprintf(stderr, "[fpdbg] this=%p CHANGED %s %llx -> %llx\n",
                            (const void*)this, fpname[k],
                            fp_dbg_[k], fpval[k]);
        }
        for (int k = 0; k < fpi; ++k) fp_dbg_[k] = fpval[k];
        fp_dbg_n_ = fpi;
        fflush(stderr);
    }
    return h;
}

void Pipeline::queue_overflow_read() {
    flush_pending_draw();
    if (!d_overflow_persist_) return;
    if (!h_overflow_pin_) {
        if (hipHostMalloc((void**)&h_overflow_pin_, sizeof(int), hipHostMallocDefault) != hipSuccess) {
            h_overflow_pin_ = nullptr;
            return;
        }
        h_overflow_pin_[0] = 0;
    }
    if (hipMemcpyAsync(h_overflow_pin_, d_overflow_persist_, sizeof(int),
                       hipMemcpyDeviceToHost, stream_) == hipSuccess)
        overflow_queued_ = true;
}

bool Pipeline::take_overflow() {
    flush_pending_draw();
    if (!d_overflow_persist_) return false;
    if (overflow_queued_ && h_overflow_pin_) {
        overflow_queued_ = false;
        const int flag = h_overflow_pin_[0];
        if (flag && getenv("HP_CQ_DBG")) fprintf(stderr, "[cq] OVERFLOW flag=%d\n", flag);
        if (flag) {
            h_overflow_pin_[0] = 0;
            hipMemsetAsync(d_overflow_persist_, 0, sizeof(int), stream_);
        }
        return flag != 0;
    }
    int flag = 0;
    // The caller is required to have synchronised this pipeline's stream, so
    // this is a completed-work read, not a race. Reset to 0 so the next batch
    // starts clean; the copy and the reset share the stream to stay ordered.
    if (hipMemcpyAsync(&flag, d_overflow_persist_, sizeof(int),
                       hipMemcpyDeviceToHost, stream_) != hipSuccess) {
        // Cannot prove the pool was sufficient -- report overflow so the caller
        // falls back to exact sizing rather than trusting a possibly clamped frame.
        return true;
    }
    hipStreamSynchronize(stream_);
    if (flag) hipMemsetAsync(d_overflow_persist_, 0, sizeof(int), stream_);
    return flag != 0;
}

void Pipeline::debug_tile_counts(int* out, int num_tiles) const {
    if (!d_tile_bins_ || !out || num_tiles <= 0) return;
    hipStreamSynchronize(stream_);
    // Strided copy: only the count field of each TileBin.
    hipMemcpy2D(out, sizeof(int),
                reinterpret_cast<const char*>(d_tile_bins_) + offsetof(TileBin, count),
                sizeof(TileBin), sizeof(int), num_tiles, hipMemcpyDeviceToHost);
}

void Pipeline::debug_tile_contents(int tile_idx, int* out_idx, float* out_z,
                                   int max_entries, int* out_count) const {
    // Dump ONE tile's bin contents (triangle indices + z_min), post-binning and
    // post-kernel_sort_tile_bins but BEFORE the fine rasterizer's in-LDS bitonic
    // sort -- i.e. exactly what the rasterizer receives from global memory.
    //
    // Exists because debug_tile_counts proved insufficient: identical per-tile
    // COUNTS across renders do not prove the same TRIANGLES landed in the tile.
    // With `int c = atomicAdd(&bin.count,1)` deciding slots, the count is
    // deterministic by construction (it is just a total) while the assignment of
    // triangles to slots is not. Hashing contents is the only way to tell those
    // apart, and it is the one pipeline stage never actually verified while
    // chasing the arm depth race.
    if (!d_tile_bins_ || !out_count || tile_idx < 0 || tile_idx >= debug_num_tiles()) return;
    hipStreamSynchronize(stream_);
    const char* base = reinterpret_cast<const char*>(d_tile_bins_)
                     + static_cast<size_t>(tile_idx) * sizeof(TileBin);
    int count = 0;
    hipMemcpy(&count, base + offsetof(TileBin, count), sizeof(int), hipMemcpyDeviceToHost);
    *out_count = count;
    int n = count;
    if (n > MAX_TRIANGLES_PER_TILE) n = MAX_TRIANGLES_PER_TILE;  // count keeps rising past the cap
    if (n > max_entries) n = max_entries;
    if (n <= 0) return;
    if (out_idx) {
        hipMemcpy(out_idx, base + offsetof(TileBin, triangle_indices),
                  static_cast<size_t>(n) * sizeof(int), hipMemcpyDeviceToHost);
    }
    if (out_z) {
        hipMemcpy(out_z, base + offsetof(TileBin, triangle_z_min),
                  static_cast<size_t>(n) * sizeof(float), hipMemcpyDeviceToHost);
    }
}

void Pipeline::clear(bool color, bool depth) {
    mc_flush();
    flush_pending_draw();
    flush_pending_clear();
    if (!fb_) return;
    flush_pending_convert();

    DeviceFramebuffer dfb = fb_->device_fb();
    if (fb_->samples() == 4) {
        kernels::launch_clear_msaa(dfb, state_, color, depth, stream_);
        return;
    }
    const float* cc = state_.clear_color;
    float cd = state_.clear_depth;

    if (color && depth && state_.gradient_bg) {
        kernels::launch_clear_gradient_depth(dfb,
            state_.gradient_bot[0], state_.gradient_bot[1], state_.gradient_bot[2],
            state_.gradient_top[0], state_.gradient_top[1], state_.gradient_top[2],
            cc[3], cd, stream_);
    } else if (color && depth && deferred_bg_ && dfb.depth_buffer && !dfb.attr_buffer) {
        launch_clear_depth_vec(dfb.depth_buffer, dfb.width * dfb.height, cd, stream_);
    } else if (color && depth) {
        kernels::launch_clear_color_depth(dfb, cc[0], cc[1], cc[2], cc[3], cd, stream_);
    } else if (color) {
        kernels::launch_clear_color(dfb, cc[0], cc[1], cc[2], cc[3], stream_);
    } else if (depth && dfb.depth_buffer) {
        kernels::launch_clear_depth(dfb, cd, stream_);
    }

    if (color && dfb.attr_buffer) {
        hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(dfb.attr_buffer),
                          0x7FC00000u,  // quiet NaN bit pattern
                          static_cast<size_t>(dfb.width) * dfb.height * FRAG_ATTR_COUNT,
                          stream_);
    }
}

void Pipeline::draw_arrays(PrimitiveType prim, const float* positions, const float* normals,
                           const float* texcoords, const float* colors, int vertex_count) {
    flush_pending_draw();
    draw_impl(prim, positions, normals, texcoords, colors, vertex_count, nullptr, 0, false);
}

void Pipeline::draw_arrays_device(PrimitiveType prim, const float* d_pos, const float* d_norm,
                                  const float* d_col, int vertex_count) {
    flush_pending_draw();
    TextureState texture_state = {};
    draw_arrays_device(prim, d_pos, d_norm, nullptr, nullptr, d_col, vertex_count, texture_state);
}

void Pipeline::draw_arrays_device(PrimitiveType prim, const float* d_pos, const float* d_norm,
                                  const float* d_tex, const int* d_texid,
                                  const float* d_col, int vertex_count,
                                  const TextureState& texture_state) {
    flush_pending_draw();
    if (!fb_ || vertex_count <= 0) return;
    int tri_count = tri_count_from_vertices(prim, vertex_count);
    if (tri_count <= 0) return;

    int vw = state_.viewport_w, vh = state_.viewport_h;
    if (vw <= 0 || vh <= 0) return;
    int tiles_x = (vw + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (vh + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = tiles_x * tiles_y;

    // Only need scratch for triangles and tile bins (not vertex data)
    ensure_scratch(vertex_count, tri_count, tiles_x, tiles_y);

    const float* d_mvp_u = d_mvp_;
    const float* d_mv_u  = d_model_view_;
    const float* d_nm_u  = d_normal_mat_;
    if (d_ext_matrices_) {
        d_mvp_u = d_ext_matrices_; d_mv_u = d_ext_matrices_ + 16; d_nm_u = d_ext_matrices_ + 32;
    } else {
        memcpy(h_matrices_, state_.mvp, 16 * sizeof(float));
        memcpy(h_matrices_ + 16, state_.model_view, 16 * sizeof(float));
        memcpy(h_matrices_ + 32, state_.normal_matrix, 9 * sizeof(float));
        hipMemcpyAsync(d_matrices_, h_matrices_, 41 * sizeof(float), hipMemcpyHostToDevice, stream_);
    }

    // Pass device pointers directly — fused kernel handles nulls internally
    kernels::launch_fused_vertex_tri(
        d_pos, d_norm, d_tex, d_texid, d_col,
        tri_count, d_mvp_u, d_mv_u, d_nm_u,
        state_.viewport_x, state_.viewport_y, vw, vh,
        d_triangles_, stream_, state_.tri_stats,
        {d_near_clip_ids_, d_near_clip_count_});

    // DIAGNOSTIC ABLATION ONLY: HP_ABL=1 skip fine raster, 2 skip bin+fine.
    static const int hp_abl = getenv("HP_ABL") ? atoi(getenv("HP_ABL")) : 0;
    if (hp_abl >= 2) return;
    last_num_tiles_ = num_tiles;
    if (deterministic_binning_) {
        bin_triangles_deterministic(tri_count, tiles_x, tiles_y, vw, vh);
    } else {
        kernels::launch_clear_tile_bin_counts(d_tile_bins_, num_tiles, stream_);
        kernels::launch_bin_triangles(
            d_triangles_, tri_count, d_tile_bins_, tiles_x, tiles_y, vw, vh, stream_,
            state_.tri_stats, {d_near_clip_ids_, d_near_clip_count_});
    }
    if (hp_abl >= 1) return;
    if (fb_->samples() == 4) {
        mc_flush();
        kernels::launch_fine_rasterize_msaa(d_triangles_, d_tile_base_, d_cursor_,
            reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
            pool_capacity_, tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_);
        return;
    }
    mc_flush();
    kernels::launch_fine_rasterize_with_sort(
        d_triangles_, d_tile_bins_,
        // Segment lengths = what the scatter actually WROTE (see d_seg_counts).
        d_tile_base_, deterministic_binning_ ? d_cursor_ : d_tile_counts_,
        reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
        deterministic_binning_ ? pool_capacity_ : 0,
        tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_,
        (deterministic_binning_ && work_ready_) ? d_work_list_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_count_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_ticket_ : nullptr,
        1, 0, 0,
        flt_ready_ ? reinterpret_cast<const float4*>(d_flt_rec_) : nullptr,
        flt_ready_ ? d_flt_cnt_ : nullptr,
        (flt_ready_ && flt_msk_ready_) ? d_flt_msk_ : nullptr);
}

void Pipeline::draw_arrays_device(PrimitiveType prim, const float* d_pos, const float* d_norm,
                                  const float* d_tex, const int* d_texid,
                                  const float* d_col, const float* d_spec,
                                  int vertex_count,
                                  const TextureState& texture_state) {
    flush_pending_draw();
    if (!fb_ || vertex_count <= 0) return;
    int tri_count = tri_count_from_vertices(prim, vertex_count);
    if (tri_count <= 0) return;

    int vw = state_.viewport_w, vh = state_.viewport_h;
    if (vw <= 0 || vh <= 0) return;
    int tiles_x = (vw + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (vh + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = tiles_x * tiles_y;

    ensure_scratch(vertex_count, tri_count, tiles_x, tiles_y);

    const float* d_mvp_u = d_mvp_;
    const float* d_mv_u  = d_model_view_;
    const float* d_nm_u  = d_normal_mat_;
    if (d_ext_matrices_) {
        d_mvp_u = d_ext_matrices_; d_mv_u = d_ext_matrices_ + 16; d_nm_u = d_ext_matrices_ + 32;
    } else {
        memcpy(h_matrices_, state_.mvp, 16 * sizeof(float));
        memcpy(h_matrices_ + 16, state_.model_view, 16 * sizeof(float));
        memcpy(h_matrices_ + 32, state_.normal_matrix, 9 * sizeof(float));
        hipMemcpyAsync(d_matrices_, h_matrices_, 41 * sizeof(float), hipMemcpyHostToDevice, stream_);
    }

    kernels::launch_fused_vertex_tri(
        d_pos, d_norm, d_tex, d_texid, d_col, d_spec,
        tri_count, d_mvp_u, d_mv_u, d_nm_u,
        state_.viewport_x, state_.viewport_y, vw, vh,
        d_triangles_, stream_, state_.tri_stats,
        {d_near_clip_ids_, d_near_clip_count_});

    last_num_tiles_ = num_tiles;
    if (deterministic_binning_) {
        bin_triangles_deterministic(tri_count, tiles_x, tiles_y, vw, vh);
    } else {
        kernels::launch_clear_tile_bin_counts(d_tile_bins_, num_tiles, stream_);
        kernels::launch_bin_triangles(
            d_triangles_, tri_count, d_tile_bins_, tiles_x, tiles_y, vw, vh, stream_,
            state_.tri_stats, {d_near_clip_ids_, d_near_clip_count_});
    }
    if (fb_->samples() == 4) {
        mc_flush();
        kernels::launch_fine_rasterize_msaa(d_triangles_, d_tile_base_, d_cursor_,
            reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
            pool_capacity_, tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_);
        return;
    }
    mc_flush();
    kernels::launch_fine_rasterize_with_sort(
        d_triangles_, d_tile_bins_,
        // Segment lengths = what the scatter actually WROTE (see d_seg_counts).
        d_tile_base_, deterministic_binning_ ? d_cursor_ : d_tile_counts_,
        reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
        deterministic_binning_ ? pool_capacity_ : 0,
        tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_,
        (deterministic_binning_ && work_ready_) ? d_work_list_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_count_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_ticket_ : nullptr,
        1, 0, 0,
        flt_ready_ ? reinterpret_cast<const float4*>(d_flt_rec_) : nullptr,
        flt_ready_ ? d_flt_cnt_ : nullptr,
        (flt_ready_ && flt_msk_ready_) ? d_flt_msk_ : nullptr);
}

static inline int hp_max_subdraws_rt() {
    static const int v = []{
        const char* e = getenv("HP_MAX_SUB");
        int x = e ? atoi(e) : HP_MAX_SUBDRAWS_DEFAULT;
        if (x < 2) x = 2;
        if (x > HP_MAX_BLOCKS) x = HP_MAX_BLOCKS;
        return x;
    }();
    return v;
}
#define kMaxSubdraws (hp_max_subdraws_rt())
static inline int hp_max_ranks_rt() {
    static const int v = []{
        const char* e = getenv("HP_MAX_RANKS");
        int x = e ? atoi(e) : HP_MAX_SUBDRAWS;
        if (x < 2) x = 2;
        if (x > HP_MAX_SUBDRAWS) x = HP_MAX_SUBDRAWS;
        return x;
    }();
    return v;
}
static inline bool hp_share_dead() {
    static const int v = getenv("HP_SHARE_DEAD") ? atoi(getenv("HP_SHARE_DEAD")) : 1;
    return v != 0;
}

static inline int hp_merge_gap_max() {
    static const int v = getenv("HP_MERGE_GAP_MAX") ? atoi(getenv("HP_MERGE_GAP_MAX")) : 262144;
    return v;
}

static inline bool hp_defer_draw_enabled() {
    static const int v = getenv("HP_DEFER_DRAW") ? atoi(getenv("HP_DEFER_DRAW")) : 1;
    return v != 0;
}

static inline bool hp_texstate_same(const TextureState& a, const TextureState& b) {
    return a.tex_data == b.tex_data && a.tex_width == b.tex_width &&
           a.tex_height == b.tex_height && a.tex_offset == b.tex_offset &&
           a.ntexture == b.ntexture && a.tex_iscube == b.tex_iscube &&
           a.tex_mip == b.tex_mip;
}

// Two spans are appendable only if EVERY attribute stream resumes at the SAME
// vertex offset past the parked one -- gap == 0 is adjacency, gap > 0 is a hole
// that a dead filler span can bridge. A null stream may only follow a null
// stream. Returns false if this stream disagrees with `gap`.
static inline bool hp_span_gap(const float* prev, const float* next, long long n, int comp,
                               long long gap) {
    if (prev == nullptr || next == nullptr) return prev == nullptr && next == nullptr;
    return next == prev + (n + gap) * (long long)comp;
}
static inline bool hp_span_gap_i(const int* prev, const int* next, long long n, int comp,
                                 long long gap) {
    if (prev == nullptr || next == nullptr) return prev == nullptr && next == nullptr;
    return next == prev + (n + gap) * (long long)comp;
}

static inline bool hp_span_at(const float* base, const float* p, long long dv, int comp) {
    if (base == nullptr || p == nullptr) return base == nullptr && p == nullptr;
    return p == base + dv * (long long)comp;
}
static inline bool hp_span_at_i(const int* base, const int* p, long long dv, int comp) {
    if (base == nullptr || p == nullptr) return base == nullptr && p == nullptr;
    return p == base + dv * (long long)comp;
}

static long long hp_fill_hist[4] = {0, 0, 0, 0};   // [extra] 0,1,2 ; [3] = fits-rejected
static long long hp_chain_count = 0;               // draws that fell through to a real chain

static inline bool hp_fill_dead_enabled() {
    static const int v = getenv("HP_FILL_DEAD") ? atoi(getenv("HP_FILL_DEAD")) : 1;
    return v != 0;
}

struct HpSpanList {
    int n = 0;
    long long end[HP_MAX_SUBDRAWS + 2];
    int blk[HP_MAX_SUBDRAWS + 2];
    bool dead[HP_MAX_SUBDRAWS + 2];
};
static inline void hp_subs_unpack(const kernels::BinSubdraws& sd, long long T, HpSpanList& L) {
    const int ns = (sd.nsub > 1) ? sd.nsub : 1;
    L.n = ns;
    for (int k = 0; k < ns; ++k) {
        L.end[k]  = (k == ns - 1) ? T : (long long)sd.split[k];
        L.blk[k]  = (ns > 1) ? kernels::hp_subdraw_perm(k, sd) : 0;
        L.dead[k] = (ns > 1) && (((sd.dead_mask >> L.blk[k]) & 1) != 0);
    }
}
static inline bool hp_subs_pack(kernels::BinSubdraws& out, const HpSpanList& L) {
    const int n = L.n;
    if (n < 1 || n > hp_max_ranks_rt()) return false;
    int b[HP_MAX_SUBDRAWS + 2];
    int dshare = INT_MAX;
    if (hp_share_dead())
        for (int k = 0; k < n; ++k) if (L.dead[k] && L.blk[k] < dshare) dshare = L.blk[k];
    for (int k = 0; k < n; ++k) b[k] = (L.dead[k] && dshare != INT_MAX) ? dshare : L.blk[k];
    // Distinct blocks, ascending (n <= 66: insertion sort).
    int u[HP_MAX_SUBDRAWS + 2];
    int nu = 0;
    for (int k = 0; k < n; ++k) {
        bool seen = false;
        for (int j = 0; j < nu; ++j) if (u[j] == b[k]) { seen = true; break; }
        if (seen) continue;
        int j = nu++;
        while (j > 0 && u[j - 1] > b[k]) { u[j] = u[j - 1]; --j; }
        u[j] = b[k];
    }
    if (nu > kMaxSubdraws) return false;
    kernels::BinSubdraws ns{};
    ns.nsub = n;
    ns.nblk = nu;
    bool ident = true;
    int raster[HP_MAX_SUBDRAWS + 2];
    for (int k = 0; k < n; ++k) {
        int r = 0;
        while (u[r] != b[k]) ++r;
        raster[k] = r;
        if (r != k) ident = false;
        if (L.dead[k]) ns.dead_mask |= (int)(1u << r);
        if (k < n - 1) {
            if (L.end[k] > (long long)INT_MAX) return false;
            ns.split[k] = (int)L.end[k];
        }
    }
    if (!ident) {
        for (int k = 0; k < n; ++k) {
            const int w = k / HP_PERM_PER_WORD;
            ns.perm[w] |= (long long)(raster[k] & HP_PERM_MASK)
                          << (HP_PERM_BITS * (k - w * HP_PERM_PER_WORD));
        }
    }
    out = ns;
    return true;
}
// Debug-ladder estimate of whether `add_live` live + `add_dead` dead ranks fit.
static inline bool hp_subs_fit(const kernels::BinSubdraws& sd, int add_live, int add_dead) {
    const int nr = ((sd.nsub > 1) ? sd.nsub : 1) + add_live + add_dead;
    int nb = ((sd.nsub > 1) ? sd.nblk : 1) + add_live;
    if (hp_share_dead()) nb += (add_dead > 0 && !(sd.nsub > 1 && sd.dead_mask)) ? 1 : 0;
    else                 nb += add_dead;
    return nr <= hp_max_ranks_rt() && nb <= kMaxSubdraws;
}

void Pipeline::vtx_flush_pending() {
    if (!pending_vtx_) return;
    std::function<bool(const kernels::VtxCntFuse*)>* go = pending_vtx_;
    pending_vtx_ = nullptr;
    (void)(*go)(nullptr);
}

void Pipeline::arm_fb_mask_clear(const kernels::FbMaskClearDesc& mc) {
    mc_flush();                       // never park two at once
    pending_mc_ = mc;
    pending_mc_armed_ = true;
}

void Pipeline::mc_flush() {
    if (!pending_mc_armed_) return;
    cq_flush();
    pending_mc_armed_ = false;
    kernels::launch_fb_mask_clear(pending_mc_, stream_);
}

void Pipeline::flush_pending_convert() {
    if (!pend_cvt_dst_) return;
    cq_flush();
    const float* src = pend_cvt_src_;
    unsigned char* dst = pend_cvt_dst_;
    const int w = pend_cvt_w_, h = pend_cvt_h_;
    pend_cvt_src_ = nullptr; pend_cvt_dst_ = nullptr; pend_cvt_total_ = 0;
    kernels::launch_rgba_f32_to_rgb_u8(src, (uint8_t*)dst, w, h,
                                       false, false, stream_);
}

bool Pipeline::convert_deferred(const float* src_rgba, unsigned char* dst_rgb, int total) {
    if (!src_rgba || !dst_rgb || total <= 0) return false;
    if (!fb_) return false;
    const DeviceFramebuffer dfb = fb_->device_fb();
    // Only the exact shape the clear rider can absorb: the SAME buffer, the
    // SAME pixel count, and the vectorised clear path (total & 3 == 0) that
    // indexes float4s one per thread.
    if (dfb.color_buffer != src_rgba) return false;
    if (dfb.width * dfb.height != total || (total & 3) != 0) return false;
    if (fb_->samples() == 4 || dfb.attr_buffer) return false;
    if (pend_cvt_dst_) flush_pending_convert();   // never hold two
    pend_cvt_src_ = src_rgba; pend_cvt_dst_ = dst_rgb; pend_cvt_total_ = total;
    pend_cvt_w_ = dfb.width; pend_cvt_h_ = dfb.height;
    return true;
}

void Pipeline::flush_pending_clear() {
    if (!pend_clear_valid_) return;
    cq_flush();
    pend_clear_valid_ = false;
    if (!fb_) return;
    flush_pending_convert();       // reads the colour this clear overwrites
    kernels::launch_clear_color_depth(fb_->device_fb(),
        pend_clear_rgba_[0], pend_clear_rgba_[1], pend_clear_rgba_[2],
        pend_clear_rgba_[3], pend_clear_depth_, stream_);
}

void Pipeline::clear_deferred(bool color, bool depth) {
    flush_pending_draw();          // also emits any earlier pending clear
    vtx_pair_cancel();
    bin_pair_cancel();
    static const int on = getenv("HP_MICRO_CLEAR") ? atoi(getenv("HP_MICRO_CLEAR")) : 1;
    if (!on || !fb_ || !color || !depth) { clear(color, depth); return; }
    if (fb_->samples() == 4 || state_.gradient_bg || deferred_bg_) { clear(color, depth); return; }
    const DeviceFramebuffer dfb = fb_->device_fb();
    if (!dfb.color_buffer || !dfb.depth_buffer || dfb.attr_buffer ||
        dfb.width <= 0 || dfb.height <= 0) { clear(color, depth); return; }
    const int total = dfb.width * dfb.height;
    // Only the (total & 3) == 0 vectorised path ignores `pitch`; take the
    // ordinary kernel otherwise so behaviour is byte-identical either way.
    if ((total & 3) != 0) { clear(color, depth); return; }
    pend_clear_rgba_[0] = state_.clear_color[0];
    pend_clear_rgba_[1] = state_.clear_color[1];
    pend_clear_rgba_[2] = state_.clear_color[2];
    pend_clear_rgba_[3] = state_.clear_color[3];
    pend_clear_depth_   = state_.clear_depth;
    pend_clear_total_   = total;
    pend_clear_valid_   = true;
}

void Pipeline::flush_pending_draw() const {
    flush_pending_draw_keep();
    cq_flush();
}

void Pipeline::flush_pending_draw_keep() const {
    if (pend_clear_valid_ && !micro_absorbing_)
        const_cast<Pipeline*>(this)->flush_pending_clear();
    if (!pend_.valid) return;
    Pipeline* self = const_cast<Pipeline*>(this);
    PendingPhong p = pend_;
    self->pend_.valid = false;          // before the call: draw_phong_now must not recurse
    self->draw_phong_now(p.prim, p.pos, p.norm, p.tex, p.texid, p.col,
                         p.spec, p.shin, p.emis, p.vertex_count, p.tex_state, p.subs);
}

void Pipeline::draw_arrays_device_phong(PrimitiveType prim, const float* d_pos, const float* d_norm,
                                        const float* d_tex, const int* d_texid,
                                        const float* d_col,
                                        const float* d_specular, const float* d_shininess,
                                        const float* d_emission,
                                        int vertex_count,
                                        const TextureState& texture_state) {
    if (!fb_ || vertex_count <= 0) return;
    if (!hp_defer_draw_enabled()) {
        flush_pending_draw();
        draw_phong_now(prim, d_pos, d_norm, d_tex, d_texid, d_col,
                       d_specular, d_shininess, d_emission, vertex_count, texture_state,
                       kernels::BinSubdraws{});
        return;
    }
    static const int merge_dbg = getenv("HP_MERGE_DBG") ? atoi(getenv("HP_MERGE_DBG")) : 0;
    enum { MC_MERGED = 0, MC_NOT_TRIS, MC_NO_PARK, MC_PREV_NOT_TRIS, MC_TEXSTATE,
           MC_GAP_NEG, MC_GAP_MOD3, MC_GAP_BIG, MC_INTMAX, MC_NSUB, MC_POS,
           MC_NORM, MC_TEX, MC_TEXID, MC_COL, MC_SPEC, MC_SHIN, MC_EMIS, MC_N };
    static long long mc_hist[MC_N] = {0};
    enum { BW_MERGED = 0, BW_NULLPOS, BW_OVERLAP, BW_BASE_MOD3, BW_GAP_MOD3, BW_GAP_BIG,
           BW_INTMAX, BW_NSUB, BW_POS, BW_NORM, BW_TEX, BW_TEXID, BW_COL, BW_SPEC,
           BW_SHIN, BW_EMIS, BW_N };
    static long long bw_hist[BW_N] = {0};
    static long long bw_gap_tris_max[BW_N] = {0};
    static long long bw_gap_tris_sum[BW_N] = {0};
    static long long bw_gap_tris_n[BW_N] = {0};
    static long long bw_holed[BW_N] = {0};
    static long long bw_nsub_hist[16] = {0};
    static long long bw_nsub_needed2 = 0;
    static long long bw_nsub_hole_tris_sum = 0;
    static long long bw_ov_same_base = 0, bw_ov_fwd = 0, bw_ov_bwd = 0;
    static long long bw_ov_contained = 0, bw_ov_notcontained = 0, bw_ov_straddles = 0;
    static long long bw_ov_in_dead = 0, bw_ov_in_live = 0;
    static long long bw_ov_dead_exact = 0, bw_ov_dead_edge = 0, bw_ov_dead_mid = 0;
    static long long bw_ov_dead_nsub[16] = {0};
    struct MergeHist {
        ~MergeHist() {
            if (!getenv("HP_MERGE_DBG")) return;
            static const char* nm[MC_N] = {
                "MERGED", "prim!=TRIANGLES", "no parked span", "parked prim!=TRIANGLES",
                "texstate differs", "gap<0 (span goes backwards)", "gap%3!=0",
                "gap_tris>HP_MERGE_GAP_MAX", "int range", "nsub+extra>kMaxSubdraws",
                "pos stream", "norm stream", "tex stream", "texid stream",
                "col stream", "spec stream", "shin stream", "emis stream"};
            long long tot = 0;
            for (int i = 0; i < MC_N; ++i) tot += mc_hist[i];
            fprintf(stderr, "[mergehist] total draw_arrays_device_phong calls = %lld\n", tot);
            for (int i = 0; i < MC_N; ++i)
                if (mc_hist[i])
                    fprintf(stderr, "[mergehist] %-28s %8lld  %5.1f%%\n", nm[i], mc_hist[i],
                            100.0 * (double)mc_hist[i] / (double)(tot ? tot : 1));
            static const char* bn[BW_N] = {
                "BW MERGED (absorbed)", "BW null pos stream",
                "BW OVERLAP (irreducible)", "BW base%3!=0", "BW bgap%3!=0",
                "BW bgap_tris>GAP_MAX", "BW int range", "BW nsub+extra>kMaxSubdraws",
                "BW pos stream", "BW norm stream", "BW tex stream", "BW texid stream",
                "BW col stream", "BW spec stream", "BW shin stream", "BW emis stream"};
            long long btot = 0;
            for (int i = 0; i < BW_N; ++i) btot += bw_hist[i];
            fprintf(stderr, "[bwdhist] calls reaching the BACKWARD ladder = %lld"
                            "  (%.1f%% of all draws)\n",
                    btot, 100.0 * (double)btot / (double)(tot ? tot : 1));
            int dom = -1; long long domv = 0;
            for (int i = 0; i < BW_N; ++i) {
                if (bw_hist[i])
                    fprintf(stderr, "[bwdhist] %-30s %8lld  %5.1f%%\n", bn[i], bw_hist[i],
                            100.0 * (double)bw_hist[i] / (double)(btot ? btot : 1));
                if (i != BW_MERGED && bw_hist[i] > domv) { domv = bw_hist[i]; dom = i; }
            }
            for (int i = 0; i < BW_N; ++i)
                if (bw_gap_tris_n[i])
                    fprintf(stderr, "[bwdhist]   %-28s bgap_tris mean=%.1f max=%lld  holed=%lld\n",
                            bn[i], (double)bw_gap_tris_sum[i] / (double)bw_gap_tris_n[i],
                            bw_gap_tris_max[i], bw_holed[i]);
            if (bw_hist[BW_NSUB]) {
                fprintf(stderr, "[bwdhist]   BW_NSUB parked-nsub histogram:");
                for (int k = 0; k < 16; ++k)
                    if (bw_nsub_hist[k]) fprintf(stderr, " nsub=%d:%lld", k, bw_nsub_hist[k]);
                fprintf(stderr, "   needed_extra2=%lld  hole_tris_sum=%lld\n",
                        bw_nsub_needed2, bw_nsub_hole_tris_sum);
            }
            if (bw_hist[BW_OVERLAP])
                fprintf(stderr, "[bwdhist]   BW_OVERLAP split: same_base=%lld  "
                                "new_starts_inside_parked=%lld  parked_starts_inside_new=%lld\n",
                        bw_ov_same_base, bw_ov_fwd, bw_ov_bwd);
            if (bw_hist[BW_OVERLAP]) {
                fprintf(stderr, "[bwdhist]   BW_OVERLAP containment: fully_inside_parked=%lld  "
                                "not_contained=%lld  straddles_span_boundary=%lld\n",
                        bw_ov_contained, bw_ov_notcontained, bw_ov_straddles);
                fprintf(stderr, "[bwdhist]   BW_OVERLAP landing span: INSIDE A DEAD FILLER=%lld  "
                                "inside a LIVE span=%lld   (dead: exact=%lld edge=%lld mid=%lld)\n",
                        bw_ov_in_dead, bw_ov_in_live,
                        bw_ov_dead_exact, bw_ov_dead_edge, bw_ov_dead_mid);
                fprintf(stderr, "[bwdhist]   BW_OVERLAP in-dead parked-nsub:");
                for (int k = 0; k < 16; ++k)
                    if (bw_ov_dead_nsub[k]) fprintf(stderr, " nsub=%d:%lld", k, bw_ov_dead_nsub[k]);
                fprintf(stderr, "\n");
            }
            if (dom >= 0)
                fprintf(stderr, "[bwdhist] *** DOMINANT BACKWARD REJECTION: %s  %lld  "
                                "%.1f%% of backward-ladder calls, %.1f%% of ALL draws ***\n",
                        bn[dom], domv, 100.0 * (double)domv / (double)(btot ? btot : 1),
                        100.0 * (double)domv / (double)(tot ? tot : 1));
            fprintf(stderr, "[bwdhist] ABSORPTION: fwd=%lld bwd=%lld total=%lld of %lld draws"
                            "  (%.1f%%)\n", mc_hist[MC_MERGED], bw_hist[BW_MERGED],
                    mc_hist[MC_MERGED] + bw_hist[BW_MERGED], tot,
                    100.0 * (double)(mc_hist[MC_MERGED] + bw_hist[BW_MERGED]) /
                            (double)(tot ? tot : 1));
            const long long fills = hp_fill_hist[0] + hp_fill_hist[1] + hp_fill_hist[2];
            fprintf(stderr, "[filldead] fills=%lld (extra0=%lld extra1=%lld extra2=%lld)  "
                            "rejected_no_slot=%lld\n", fills, hp_fill_hist[0],
                    hp_fill_hist[1], hp_fill_hist[2], hp_fill_hist[3]);
            fprintf(stderr, "[filldead] ABSORPTION WITH FILL: %lld of %lld draws (%.1f%%)   "
                            "BINNING CHAINS ISSUED = %lld\n",
                    mc_hist[MC_MERGED] + bw_hist[BW_MERGED] + fills, tot,
                    100.0 * (double)(mc_hist[MC_MERGED] + bw_hist[BW_MERGED] + fills) /
                            (double)(tot ? tot : 1), hp_chain_count);
        }
    };
    static MergeHist mc_dump;
    int mc_why = MC_MERGED;
    if (merge_dbg) {
        if (prim != PRIM_TRIANGLES) mc_why = MC_NOT_TRIS;
        else if (!pend_.valid) mc_why = MC_NO_PARK;
        else if (pend_.prim != PRIM_TRIANGLES) mc_why = MC_PREV_NOT_TRIS;
        else if (!hp_texstate_same(pend_.tex_state, texture_state)) mc_why = MC_TEXSTATE;
        else {
            const long long n = pend_.vertex_count;
            const long long g = (pend_.pos && d_pos) ? ((d_pos - pend_.pos) / 3 - n) : -1;
            const long long gt = g / 3;
            const int ex = (g > 0) ? 2 : 1;
            if (g < 0) mc_why = MC_GAP_NEG;
            else if ((g % 3) != 0 || (d_pos - pend_.pos) % 3 != 0) mc_why = MC_GAP_MOD3;
            else if (gt > (long long)hp_merge_gap_max()) mc_why = MC_GAP_BIG;
            else if ((long long)pend_.vertex_count + g + vertex_count >= (long long)INT_MAX / 4)
                mc_why = MC_INTMAX;
            else if (!hp_subs_fit(pend_.subs, 1, ex - 1)) mc_why = MC_NSUB;
            else if (!hp_span_gap(pend_.pos,  d_pos,  n, 3, g)) mc_why = MC_POS;
            else if (!hp_span_gap(pend_.norm, d_norm, n, 3, g)) mc_why = MC_NORM;
            else if (!hp_span_gap(pend_.tex,  d_tex,  n, TEXCOORD_COMPONENTS, g)) mc_why = MC_TEX;
            else if (!hp_span_gap_i(pend_.texid, d_texid, n, 1, g)) mc_why = MC_TEXID;
            else if (!hp_span_gap(pend_.col,  d_col,  n, 4, g)) mc_why = MC_COL;
            else if (!hp_span_gap(pend_.spec, d_specular,  n, 1, g)) mc_why = MC_SPEC;
            else if (!hp_span_gap(pend_.shin, d_shininess, n, 1, g)) mc_why = MC_SHIN;
            else if (!hp_span_gap(pend_.emis, d_emission,  n, 1, g)) mc_why = MC_EMIS;
        }
        mc_hist[mc_why] += 1;
        if (mc_why >= MC_GAP_NEG) {
            const long long n  = pend_.vertex_count;
            const long long bg = (pend_.pos && d_pos)
                               ? ((pend_.pos - d_pos) / 3 - (long long)vertex_count) : -1;
            const long long bgt = bg / 3;
            const int bex = (bg > 0) ? 2 : 1;
            int bw;
            if (!pend_.pos || !d_pos)                      bw = BW_NULLPOS;
            else if (bg < 0)                               bw = BW_OVERLAP;
            else if ((pend_.pos - d_pos) % 3 != 0)         bw = BW_BASE_MOD3;
            else if ((bg % 3) != 0)                        bw = BW_GAP_MOD3;
            else if (bgt > (long long)hp_merge_gap_max())  bw = BW_GAP_BIG;
            else if (n + bg + vertex_count >= (long long)INT_MAX / 4) bw = BW_INTMAX;
            else if (!hp_subs_fit(pend_.subs, 1, bex - 1)) bw = BW_NSUB;
            else if (!hp_span_gap(d_pos,       pend_.pos,   vertex_count, 3, bg)) bw = BW_POS;
            else if (!hp_span_gap(d_norm,      pend_.norm,  vertex_count, 3, bg)) bw = BW_NORM;
            else if (!hp_span_gap(d_tex,       pend_.tex,   vertex_count,
                                  TEXCOORD_COMPONENTS, bg)) bw = BW_TEX;
            else if (!hp_span_gap_i(d_texid,   pend_.texid, vertex_count, 1, bg)) bw = BW_TEXID;
            else if (!hp_span_gap(d_col,       pend_.col,   vertex_count, 4, bg)) bw = BW_COL;
            else if (!hp_span_gap(d_specular,  pend_.spec,  vertex_count, 1, bg)) bw = BW_SPEC;
            else if (!hp_span_gap(d_shininess, pend_.shin,  vertex_count, 1, bg)) bw = BW_SHIN;
            else if (!hp_span_gap(d_emission,  pend_.emis,  vertex_count, 1, bg)) bw = BW_EMIS;
            else                                           bw = BW_MERGED;
            bw_hist[bw] += 1;
            // Secondary axes the clause label alone cannot carry.
            if (bw == BW_GAP_BIG || bw == BW_MERGED) {
                if (bgt > bw_gap_tris_max[bw]) bw_gap_tris_max[bw] = bgt;
                bw_gap_tris_sum[bw] += bgt;
                bw_gap_tris_n[bw]   += 1;
                if (bg > 0) bw_holed[bw] += 1;
            }
            if (bw == BW_NSUB) {
                const int k = pend_.subs.nsub < 16 ? pend_.subs.nsub : 15;
                bw_nsub_hist[k] += 1;
                if (bex == 2) bw_nsub_needed2 += 1;
                if (bgt >= 0 && bgt <= (long long)hp_merge_gap_max() && bg > 0)
                    bw_nsub_hole_tris_sum += bgt;
            }
            if (bw == BW_OVERLAP) {
                if (d_pos == pend_.pos) bw_ov_same_base += 1;
                else if (d_pos > pend_.pos) bw_ov_fwd += 1;   // starts inside the parked span
                else bw_ov_bwd += 1;                          // parked starts inside this one
                // Is the "overlap" real, or does the new span sit inside a
                // span the parked draw already bridged as a DEAD filler? A
                // filler is binned-but-never-rasterized, so a live span that
                // fits inside one costs NO new vertex range at all.
                const long long dv = (d_pos - pend_.pos) / 3;   // vertices
                if (dv >= 0 && (dv % 3) == 0 && dv + vertex_count <= n) {
                    bw_ov_contained += 1;
                    const long long t0 = dv / 3, t1 = t0 + vertex_count / 3;
                    const int ns = pend_.subs.nsub;
                    long long lo = 0; int hit = -1;
                    for (int k = 0; k < ns; ++k) {
                        const long long hi = (k == ns - 1) ? (n / 3)
                                                           : (long long)pend_.subs.split[k];
                        if (t0 >= lo && t1 <= hi) { hit = k; break; }
                        lo = hi;
                    }
                    if (hit < 0) bw_ov_straddles += 1;
                    else {
                        const int blk = kernels::hp_subdraw_perm(hit, pend_.subs);
                        const bool dead = (pend_.subs.dead_mask >> blk) & 1;
                        const long long slo = (hit == 0) ? 0 : pend_.subs.split[hit - 1];
                        const long long shi = (hit == ns - 1) ? (n / 3) : pend_.subs.split[hit];
                        if (dead) {
                            bw_ov_in_dead += 1;
                            if (t0 == slo && t1 == shi)      bw_ov_dead_exact += 1;
                            else if (t0 == slo || t1 == shi) bw_ov_dead_edge  += 1;
                            else                             bw_ov_dead_mid   += 1;
                            bw_ov_dead_nsub[ns < 16 ? ns : 15] += 1;
                        } else {
                            bw_ov_in_live += 1;
                        }
                    }
                } else {
                    bw_ov_notcontained += 1;
                }
            }
        }
        if (merge_dbg >= 2)
            fprintf(stderr, "[merge] why=%d vtx=%d parked=%d nsub=%d\n", mc_why, vertex_count,
                    pend_.valid ? pend_.vertex_count : -1, pend_.valid ? pend_.subs.nsub : -1);
    }
    // Only PRIM_TRIANGLES concatenates cleanly (tri_count is a pure function of
    // the vertex count, and there is no strip/fan restart to respect).
    if (prim == PRIM_TRIANGLES && pend_.valid && pend_.prim == PRIM_TRIANGLES &&
        hp_texstate_same(pend_.tex_state, texture_state)) {
        const long long n = pend_.vertex_count;
        // The gap is defined by the POSITION stream and then cross-checked
        // against every other stream; a layout where the streams disagree is
        // not a concatenation and must not be merged.
        long long gap = (pend_.pos && d_pos) ? ((d_pos - pend_.pos) / 3 - n) : -1;
        const long long gap_tris = gap / 3;
        const int extra = (gap > 0) ? 2 : 1;
        if (gap >= 0 && (gap % 3) == 0 && gap_tris <= (long long)hp_merge_gap_max() &&
            (d_pos - pend_.pos) % 3 == 0 &&
            (long long)pend_.vertex_count + gap + vertex_count < (long long)INT_MAX / 4 &&
            hp_span_gap(pend_.pos,  d_pos,  n, 3, gap) &&
            hp_span_gap(pend_.norm, d_norm, n, 3, gap) &&
            hp_span_gap(pend_.tex,  d_tex,  n, TEXCOORD_COMPONENTS, gap) &&
            hp_span_gap_i(pend_.texid, d_texid, n, 1, gap) &&
            hp_span_gap(pend_.col,  d_col,  n, 4, gap) &&
            hp_span_gap(pend_.spec, d_specular,  n, 1, gap) &&
            hp_span_gap(pend_.shin, d_shininess, n, 1, gap) &&
            hp_span_gap(pend_.emis, d_emission,  n, 1, gap)) {
            HpSpanList L;
            hp_subs_unpack(pend_.subs, n / 3, L);
            int ob = 0;
            for (int k = 0; k < L.n; ++k) if (L.blk[k] + 1 > ob) ob = L.blk[k] + 1;
            if (gap > 0) {
                // Bridge the hole with a span that is binned but never drawn.
                L.end[L.n] = (n + gap) / 3; L.blk[L.n] = ob;     L.dead[L.n] = true;  ++L.n;
            }
            L.end[L.n] = (n + gap + vertex_count) / 3; L.blk[L.n] = ob + 1; L.dead[L.n] = false; ++L.n;
            kernels::BinSubdraws ns;
            if (hp_subs_pack(ns, L)) {
                pend_.subs = ns;
                pend_.vertex_count += (int)gap + vertex_count;
                return;
            }
        }
        const long long bgap = (pend_.pos && d_pos)
                             ? ((pend_.pos - d_pos) / 3 - (long long)vertex_count) : -1;
        const long long bgap_tris = bgap / 3;
        const int bextra = (bgap > 0) ? 2 : 1;
        if (bgap >= 0 && (bgap % 3) == 0 && bgap_tris <= (long long)hp_merge_gap_max() &&
            (pend_.pos - d_pos) % 3 == 0 &&
            (long long)pend_.vertex_count + bgap + vertex_count < (long long)INT_MAX / 4 &&
            hp_span_gap(d_pos,        pend_.pos,   vertex_count, 3, bgap) &&
            hp_span_gap(d_norm,       pend_.norm,  vertex_count, 3, bgap) &&
            hp_span_gap(d_tex,        pend_.tex,   vertex_count, TEXCOORD_COMPONENTS, bgap) &&
            hp_span_gap_i(d_texid,    pend_.texid, vertex_count, 1, bgap) &&
            hp_span_gap(d_col,        pend_.col,   vertex_count, 4, bgap) &&
            hp_span_gap(d_specular,   pend_.spec,  vertex_count, 1, bgap) &&
            hp_span_gap(d_shininess,  pend_.shin,  vertex_count, 1, bgap) &&
            hp_span_gap(d_emission,   pend_.emis,  vertex_count, 1, bgap)) {
            const int S = (vertex_count + (int)bgap) / 3;
            HpSpanList O;
            hp_subs_unpack(pend_.subs, n / 3, O);
            int ob = 0;
            for (int k = 0; k < O.n; ++k) if (O.blk[k] + 1 > ob) ob = O.blk[k] + 1;
            HpSpanList L;
            L.end[L.n] = vertex_count / 3; L.blk[L.n] = ob + 1; L.dead[L.n] = false; ++L.n;
            if (bgap > 0) { L.end[L.n] = S; L.blk[L.n] = ob; L.dead[L.n] = true; ++L.n; }
            for (int k = 0; k < O.n; ++k) {
                L.end[L.n] = O.end[k] + S; L.blk[L.n] = O.blk[k]; L.dead[L.n] = O.dead[k]; ++L.n;
            }
            kernels::BinSubdraws ns;
            if (hp_subs_pack(ns, L)) {
                pend_.subs = ns;
                pend_.pos = d_pos; pend_.norm = d_norm; pend_.tex = d_tex; pend_.texid = d_texid;
                pend_.col = d_col; pend_.spec = d_specular;
                pend_.shin = d_shininess; pend_.emis = d_emission;
                pend_.vertex_count += (int)bgap + vertex_count;
                return;
            }
        }
        if (hp_fill_dead_enabled() && pend_.subs.nsub > 1 && pend_.pos && d_pos &&
            d_pos >= pend_.pos && (d_pos - pend_.pos) % 3 == 0) {
            const long long dv = (d_pos - pend_.pos) / 3;          // VERTICES in
            const long long T  = (long long)pend_.vertex_count / 3;
            if ((dv % 3) == 0 && dv + (long long)vertex_count <= (long long)pend_.vertex_count &&
                hp_span_at(pend_.norm, d_norm, dv, 3) &&
                hp_span_at(pend_.tex,  d_tex,  dv, TEXCOORD_COMPONENTS) &&
                hp_span_at_i(pend_.texid, d_texid, dv, 1) &&
                hp_span_at(pend_.col,  d_col,  dv, 4) &&
                hp_span_at(pend_.spec, d_specular,  dv, 1) &&
                hp_span_at(pend_.shin, d_shininess, dv, 1) &&
                hp_span_at(pend_.emis, d_emission,  dv, 1)) {
                const long long t0 = dv / 3;
                const long long t1 = t0 + (long long)vertex_count / 3;
                HpSpanList O;
                hp_subs_unpack(pend_.subs, T, O);
                long long lo = 0, slo = 0, shi = 0;
                int hit = -1;
                for (int k = 0; k < O.n; ++k) {
                    const long long hi = O.end[k];
                    if (t0 >= lo && t1 <= hi) { hit = k; slo = lo; shi = hi; break; }
                    lo = hi;
                }
                if (hit >= 0 && O.dead[hit]) {
                    const bool head = (t0 > slo);
                    const bool tail = (t1 < shi);
                    const int extra = (int)head + (int)tail;
                    int ob = 0;
                    for (int k = 0; k < O.n; ++k) if (O.blk[k] + 1 > ob) ob = O.blk[k] + 1;
                    const int b_hit = O.blk[hit];
                    HpSpanList L;
                    for (int k = 0; k < O.n; ++k) {
                        if (k == hit) {
                            if (head) { L.end[L.n] = t0;  L.blk[L.n] = b_hit;  L.dead[L.n] = true;  ++L.n; }
                                        L.end[L.n] = t1;  L.blk[L.n] = ob + 1; L.dead[L.n] = false; ++L.n;
                            if (tail) { L.end[L.n] = shi; L.blk[L.n] = head ? ob : b_hit;
                                        L.dead[L.n] = true; ++L.n; }
                        } else {
                            L.end[L.n] = O.end[k]; L.blk[L.n] = O.blk[k]; L.dead[L.n] = O.dead[k]; ++L.n;
                        }
                    }
                    kernels::BinSubdraws ns;
                    if (L.n <= HP_MAX_SUBDRAWS && hp_subs_pack(ns, L)) {
                        pend_.subs = ns;
                        if (merge_dbg) hp_fill_hist[extra] += 1;
                        // Base, extent and every stream pointer are UNCHANGED:
                        // the new span was already inside the parked range.
                        return;
                    }
                    if (merge_dbg) hp_fill_hist[3] += 1;
                }
            }
        }
    }
    flush_pending_draw_keep();
    if (prim != PRIM_TRIANGLES) {
        draw_phong_now(prim, d_pos, d_norm, d_tex, d_texid, d_col,
                       d_specular, d_shininess, d_emission, vertex_count, texture_state,
                       kernels::BinSubdraws{});
        return;
    }
    if (merge_dbg) hp_chain_count += 1;
    pend_.valid = true;
    pend_.prim = prim;
    pend_.pos = d_pos; pend_.norm = d_norm; pend_.tex = d_tex; pend_.texid = d_texid;
    pend_.col = d_col; pend_.spec = d_specular; pend_.shin = d_shininess; pend_.emis = d_emission;
    pend_.vertex_count = vertex_count;
    pend_.tex_state = texture_state;
    pend_.subs = kernels::BinSubdraws{};
}

void Pipeline::draw_phong_now(PrimitiveType prim, const float* d_pos, const float* d_norm,
                              const float* d_tex, const int* d_texid,
                              const float* d_col,
                              const float* d_specular, const float* d_shininess,
                              const float* d_emission,
                              int vertex_count,
                              const TextureState& texture_state,
                              const kernels::BinSubdraws& subs) {
    if (!fb_ || vertex_count <= 0) return;
    int tri_count = tri_count_from_vertices(prim, vertex_count);
    if (tri_count <= 0) return;

    int vw = state_.viewport_w, vh = state_.viewport_h;
    if (vw <= 0 || vh <= 0) return;
    int tiles_x = (vw + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (vh + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = tiles_x * tiles_y;

    cq_draw_ = false;
    {
        static const int cq_on = getenv("HP_CQ") ? atoi(getenv("HP_CQ")) : 1;
        static const int cq_mp = getenv("HP_CQ_MP") ? atoi(getenv("HP_CQ_MP")) : 1;
        const bool micro_blocks = micro_pair_slots_ > 0 &&
            (cq_mp == 0 || micro_pair_is_take(d_pos, tri_count, tiles_x, tiles_y));
        const bool may = cq_on != 0 && d_ext_matrices_ && deterministic_binning_ &&
                         !vtx_pair_armed_ && !vtx_pair_ready_ && !vtx_pair_swapped_ &&
                         !bin_pair_.armed && !micro_blocks && !pending_mc_armed_ &&
                         fb_->samples() != 4 && !stats_wanted_;
        if (!may) {
            cq_flush();
        } else {
            static const int cq_capenv = getenv("HP_CQ_CAP") ? atoi(getenv("HP_CQ_CAP")) : 0;
            const int cq_max = cq_capenv > 0 ? std::min(HP_CQ_MAX, cq_capenv)
                             : std::min(HP_CQ_MAX, std::max(1, cq_on > 1 ? cq_on : HP_CQ_MAX));
            if ((int)cq_q_.size() >= cq_max || (cq_busy_ & (1u << cq_cur_set_)) != 0) {
                int s = -1;
                if ((int)cq_q_.size() < cq_max)
                    for (int i = 0; i < cq_max; ++i) if (!(cq_busy_ & (1u << i))) { s = i; break; }
                if (s < 0) { cq_flush(); s = cq_cur_set_; }
                if (s != cq_cur_set_) {
                    cq_save_set(cq_cur_set_);
                    cq_load_set(s);
                    cq_cur_set_ = s;
                    // The single-pool tree's pool only ever grows, so every
                    // draw sees the high-water capacity; give a private set
                    // the same capacity so the deferred sizing clamps exactly
                    // where the single pool would (never more often).
                    if (pool_capacity_ < cq_pool_hw_) {
                        if (d_pool_tri_) hipFree(d_pool_tri_);
                        if (d_pool_z_)   hipFree(d_pool_z_);
                        d_pool_tri_ = nullptr; d_pool_z_ = nullptr;
                        if (hipMalloc(&d_pool_tri_, (size_t)cq_pool_hw_ * sizeof(int)) != hipSuccess ||
                            hipMalloc(&d_pool_z_, (size_t)cq_pool_hw_ * sizeof(float)) != hipSuccess)
                            pool_capacity_ = 0;
                        else
                            pool_capacity_ = cq_pool_hw_;
                    }
                }
            }
            cq_draw_ = true;
        }
    }
    struct CqDrawGuard { Pipeline* p; ~CqDrawGuard() { p->cq_draw_ = false; } } cq_draw_guard{this};

    if (micro_pair_take(d_pos, tri_count, tiles_x, tiles_y, texture_state)) return;

    ensure_scratch(vertex_count, tri_count, tiles_x, tiles_y);

    const float* d_mvp_u = d_mvp_;
    const float* d_mv_u  = d_model_view_;
    const float* d_nm_u  = d_normal_mat_;
    if (d_ext_matrices_) {
        d_mvp_u = d_ext_matrices_; d_mv_u = d_ext_matrices_ + 16; d_nm_u = d_ext_matrices_ + 32;
    } else {
        memcpy(h_matrices_, state_.mvp, 16 * sizeof(float));
        memcpy(h_matrices_ + 16, state_.model_view, 16 * sizeof(float));
        memcpy(h_matrices_ + 32, state_.normal_matrix, 9 * sizeof(float));
        hipMemcpyAsync(d_matrices_, h_matrices_, 41 * sizeof(float), hipMemcpyHostToDevice, stream_);
    }

    static const int hp_abl3 = getenv("HP_ABL") ? atoi(getenv("HP_ABL")) : 0;
    if (hp_abl3 >= 3) return;   // DIAGNOSTIC: skip the whole draw chain
    kernels::BinEmit emit{};
    if (state_.gouraud_shading && deterministic_binning_ && hp_abl3 < 2 &&
        tri_count <= INT_MAX / 2 && kernels::fused_vertex_binrec_supported()) {
        if (kernels::BinRec* rec = ensure_binrec(2 * tri_count)) {
            emit.rec       = rec;
            emit.tiles_x   = tiles_x;
            emit.tiles_y   = tiles_y;
            emit.fb_w      = vw;
            emit.fb_h      = vh;
            emit.cull_mask = kernels::backface_cull_mask(state_);
            emit.subs      = subs;
            // The single-CTA micro-draw chain reads TriangleSetup directly, so
            // a draw small enough to take it must keep writing the core line.
            static const int mdc_max_tris = []{
                const char* e = getenv("HP_MICRO_MAX_TRIS");
                return e ? atoi(e) : kernels::micro_bin_max_tris();
            }();
            emit.skip_dead_core = (tri_count > mdc_max_tris);
            emit.center_samples_only = !(fb_ && fb_->samples() == 4);
        }
    }
    const bool binrec_ready = (emit.rec != nullptr);
    struct VtxPairGuard {
        Pipeline* p;
        ~VtxPairGuard() { p->vtx_pair_swap_out(); }
    } vtx_guard{this};
    std::function<bool(const kernels::VtxCntFuse*)> vtx_go;
    struct VtxParkGuard {
        Pipeline* p;
        ~VtxParkGuard() { p->vtx_flush_pending(); }
    } vtx_park_guard{this};
    const bool vtx_pair_taken =
        state_.gouraud_shading && vtx_pair_take(d_pos, tri_count, vw, vh, emit);
    if (vtx_pair_taken) {
        // nothing to launch: slot 1 already wrote d_triangles_ (swapped) and
        // emit.rec's array.
    } else if (state_.gouraud_shading) {
        if (!vtx_pair_try_launch(d_pos, d_norm, d_tex, d_texid, d_col,
                                 d_specular, d_shininess, d_emission,
                                 tri_count, vw, vh, d_mvp_u, d_mv_u, d_nm_u, emit)) {
            const RenderState vst = state_;
            TriangleSetup* const vout = d_triangles_;
            const hipStream_t vs = stream_;
            const kernels::NearClipQueue vq{d_near_clip_ids_, d_near_clip_count_};
            vtx_go = [=](const kernels::VtxCntFuse* cf) -> bool {
                return kernels::launch_fused_vertex_tri_gouraud_ex(
                    d_pos, d_norm, d_tex, d_texid, d_col,
                    (const float*)nullptr, d_specular, d_shininess, d_emission,
                    tri_count, d_mvp_u, d_mv_u, d_nm_u,
                    vst.viewport_x, vst.viewport_y, vw, vh,
                    vst, vout, vs, vq, emit, cf);
            };
            if (binrec_ready && deterministic_binning_) pending_vtx_ = &vtx_go;
            else vtx_go(nullptr);
        }
    } else {
        kernels::launch_fused_vertex_tri(
            d_pos, d_norm, d_tex, d_texid, d_col,
            (const float*)nullptr, d_specular, d_shininess, d_emission,
            tri_count, d_mvp_u, d_mv_u, d_nm_u,
            state_.viewport_x, state_.viewport_y, vw, vh,
            d_triangles_, stream_, state_.tri_stats,
            {d_near_clip_ids_, d_near_clip_count_});
    }

    static const int hp_abl2 = getenv("HP_ABL") ? atoi(getenv("HP_ABL")) : 0;
    if (hp_abl2 >= 2) return;   // DIAGNOSTIC: skip bin + fine
    last_num_tiles_ = num_tiles;
    const int nsub = (deterministic_binning_ && subs.nsub > 1) ? subs.nblk : 1;
    if (deterministic_binning_) {
        bin_triangles_deterministic(tri_count, tiles_x, tiles_y, vw, vh, subs, binrec_ready);
        vtx_flush_pending();
    } else {
        kernels::launch_clear_tile_bin_counts(d_tile_bins_, num_tiles, stream_);
        kernels::launch_bin_triangles(
            d_triangles_, tri_count, d_tile_bins_, tiles_x, tiles_y, vw, vh, stream_,
            state_.tri_stats, {d_near_clip_ids_, d_near_clip_count_});
    }
    if (hp_abl2 >= 1) return;   // DIAGNOSTIC: skip fine
    if (!cq_recording_) cq_flush();
    if (fb_->samples() == 4) {
        for (int sd = 0; sd < nsub; ++sd) {
            if (subs.dead_mask & (1 << sd)) continue;
            mc_flush();
            kernels::launch_fine_rasterize_msaa(d_triangles_,
                d_tile_base_ + (size_t)sd * num_tiles, d_cursor_ + (size_t)sd * num_tiles,
                reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
                pool_capacity_, tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_);
        }
        return;
    }
    mc_flush();
    if (cq_recording_) {
        const TriangleSetup* f_tris = d_triangles_;
        TileBin* f_bins = d_tile_bins_;
        const int* f_base = d_tile_base_;
        const int* f_cnt = deterministic_binning_ ? d_cursor_ : d_tile_counts_;
        const unsigned long long* f_key = fine_key_override_ ? fine_key_override_
            : reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_);
        const int f_cap = deterministic_binning_ ? pool_capacity_ : 0;
        const DeviceFramebuffer f_fb = fb_->device_fb();
        const RenderState f_st = state_;
        const TextureState f_tx = texture_state;
        int* f_wl = (deterministic_binning_ && work_ready_) ? d_work_list_ : nullptr;
        int* f_wc = (deterministic_binning_ && work_ready_) ? d_work_count_ : nullptr;
        int* f_wt = (deterministic_binning_ && work_ready_) ? d_work_ticket_ : nullptr;
        const int f_dead = (nsub > 1) ? subs.dead_mask : 0;
        const float4* f_rec = flt_ready_ ? reinterpret_cast<const float4*>(
            fine_rec_override_ ? fine_rec_override_ : d_flt_rec_) : nullptr;
        const int* f_fc = flt_ready_ ? (fine_cnt_override_ ? fine_cnt_override_ : d_flt_cnt_)
                                     : nullptr;
        const unsigned short* f_msk = (flt_ready_ && flt_msk_ready_)
            ? reinterpret_cast<const unsigned short*>(fine_msk_override_ ? fine_msk_override_
                                                                         : d_flt_msk_) : nullptr;
        const hipStream_t f_s = stream_;
        CqPending pnd;
        pnd.rec = cq_cur_;
        pnd.rb = cq_rb_;
        pnd.rb_src = cq_rb_src_;
        for (int k = 0; k < 4; ++k) pnd.rb_meta[k] = cq_rb_meta_[k];
        pnd.fine = [=]() {
            kernels::launch_fine_rasterize_with_sort(
                f_tris, f_bins, f_base, f_cnt, f_key, f_cap, tiles_x, tiles_y, f_fb, f_st, f_tx,
                f_s, f_wl, f_wc, f_wt, nsub, f_dead, num_tiles, f_rec, f_fc, f_msk);
        };
        cq_recording_ = false;
        cq_rb_ = false;
        cq_q_.push_back(std::move(pnd));
        cq_busy_ |= (1u << cq_cur_set_);
        cq_stat_q_++;
        return;
    }
    kernels::launch_fine_rasterize_with_sort(
        d_triangles_, d_tile_bins_,
        // Segment lengths = what the scatter actually WROTE (see d_seg_counts).
        d_tile_base_,
        (deterministic_binning_ ? d_cursor_ : d_tile_counts_),
        fine_key_override_ ? fine_key_override_
                           : reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
        deterministic_binning_ ? pool_capacity_ : 0,
        tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_,
        (deterministic_binning_ && work_ready_) ? d_work_list_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_count_ : nullptr,
        (deterministic_binning_ && work_ready_) ? d_work_ticket_ : nullptr,
        nsub, (nsub > 1) ? subs.dead_mask : 0, num_tiles,
        flt_ready_ ? reinterpret_cast<const float4*>(fine_rec_override_ ? fine_rec_override_
                                                                       : d_flt_rec_) : nullptr,
        flt_ready_ ? (fine_cnt_override_ ? fine_cnt_override_ : d_flt_cnt_) : nullptr,
        (flt_ready_ && flt_msk_ready_)
            ? reinterpret_cast<const unsigned short*>(fine_msk_override_ ? fine_msk_override_
                                                                         : d_flt_msk_) : nullptr);
}

void Pipeline::draw_elements(PrimitiveType prim, const float* positions, const float* normals,
                             const float* texcoords, const float* colors, int vertex_count,
                             const uint32_t* indices, int index_count) {
    flush_pending_draw();
    draw_impl(prim, positions, normals, texcoords, colors, vertex_count, indices, index_count, true);
}

void Pipeline::sync() {
    flush_pending_draw();
    resolve();
    hipError_t err = hipStreamSynchronize(stream_);
    check_hip(err, "Pipeline::sync");
}

void Pipeline::draw_impl(PrimitiveType prim, const float* positions, const float* normals,
                         const float* texcoords, const float* colors, int vertex_count,
                         const uint32_t* indices, int index_count, bool indexed) {
    if (!fb_ || vertex_count <= 0) return;

    int tri_count = indexed ? tri_count_from_indices(prim, index_count)
                            : tri_count_from_vertices(prim, vertex_count);
    if (tri_count <= 0) return;

    int vw = state_.viewport_w;
    int vh = state_.viewport_h;
    if (vw <= 0 || vh <= 0) return;

    int tiles_x = (vw + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_y = (vh + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles = tiles_x * tiles_y;

    const float* pos_ptr = positions;
    const float* norm_ptr = normals;
    const float* tc_ptr = texcoords;
    const float* col_ptr = colors;
    int effective_vertex_count = vertex_count;

    std::vector<float> exp_pos, exp_norm, exp_tc, exp_col;

    if (indexed && indices) {
        // Expand indexed data to non-indexed for kernel compatibility
        effective_vertex_count = index_count;
        exp_pos.resize(index_count * 3);
        exp_norm.resize(index_count * 3);
        exp_tc.resize(index_count * TEXCOORD_COMPONENTS);
        exp_col.resize(index_count * 4);
        for (int i = 0; i < index_count; ++i) {
            uint32_t idx = indices[i];
            if (idx >= static_cast<uint32_t>(vertex_count)) continue;
            memcpy(&exp_pos[i*3], &positions[idx*3], 3 * sizeof(float));
            if (normals) memcpy(&exp_norm[i*3], &normals[idx*3], 3 * sizeof(float));
            else std::memset(&exp_norm[i*3], 0, 3 * sizeof(float));
            // Source and destination are both vertex texcoords, so both use the
            // constant. This line, and the two around it, are what a grep for
            // "texcoord" misses -- the buffer is named exp_tc.
            if (texcoords) memcpy(&exp_tc[i*TEXCOORD_COMPONENTS], &texcoords[idx*TEXCOORD_COMPONENTS], TEXCOORD_COMPONENTS * sizeof(float));
            else std::memset(&exp_tc[i*TEXCOORD_COMPONENTS], 0, TEXCOORD_COMPONENTS * sizeof(float));
            if (colors) memcpy(&exp_col[i*4], &colors[idx*4], 4 * sizeof(float));
            else { exp_col[i*4]=1; exp_col[i*4+1]=1; exp_col[i*4+2]=1; exp_col[i*4+3]=1; }
        }
        pos_ptr = exp_pos.data();
        norm_ptr = exp_norm.data();
        tc_ptr = exp_tc.data();
        col_ptr = exp_col.data();
    }

    ensure_scratch(effective_vertex_count, tri_count, tiles_x, tiles_y);

    // All operations are on the same stream, so ordering is guaranteed
    // without explicit synchronization. Only sync at readback points.

    // 1. Upload vertex data to device
    size_t pos_size = static_cast<size_t>(effective_vertex_count) * 3 * sizeof(float);
    hipMemcpyAsync(d_positions_, pos_ptr, pos_size, hipMemcpyHostToDevice, stream_);

    if (norm_ptr) {
        hipMemcpyAsync(d_normals_, norm_ptr, pos_size, hipMemcpyHostToDevice, stream_);
    } else {
        hipMemsetAsync(d_normals_, 0, pos_size, stream_);
    }

    size_t tc_size = static_cast<size_t>(effective_vertex_count) * TEXCOORD_COMPONENTS * sizeof(float);
    if (tc_ptr) {
        hipMemcpyAsync(d_texcoords_, tc_ptr, tc_size, hipMemcpyHostToDevice, stream_);
    } else {
        hipMemsetAsync(d_texcoords_, 0, tc_size, stream_);
    }
    hipMemsetAsync(d_texids_, 0xFF, static_cast<size_t>(effective_vertex_count) * sizeof(int), stream_);

    size_t col_size = static_cast<size_t>(effective_vertex_count) * 4 * sizeof(float);
    if (col_ptr) {
        hipMemcpyAsync(d_colors_, col_ptr, col_size, hipMemcpyHostToDevice, stream_);
    } else {
        kernels::launch_fill_float4(d_colors_, effective_vertex_count, 1.0f, 1.0f, 1.0f, 1.0f, stream_);
    }

    // Host geometry (site overlays) must honor the same external matrix block
    // as device geometry. Besides avoiding a copy, this prevents reflected and
    // ordinary draws from overwriting pinned staging bytes still in flight.
    const float* d_mvp_u = d_mvp_;
    const float* d_mv_u = d_model_view_;
    const float* d_nm_u = d_normal_mat_;
    if (d_ext_matrices_) {
        d_mvp_u = d_ext_matrices_;
        d_mv_u = d_ext_matrices_ + 16;
        d_nm_u = d_ext_matrices_ + 32;
    } else {
        memcpy(h_matrices_, state_.mvp, 16 * sizeof(float));
        memcpy(h_matrices_ + 16, state_.model_view, 16 * sizeof(float));
        memcpy(h_matrices_ + 32, state_.normal_matrix, 9 * sizeof(float));
        hipMemcpyAsync(d_matrices_, h_matrices_, 41 * sizeof(float), hipMemcpyHostToDevice, stream_);
    }

    // Fused vertex transform + triangle setup (2 kernels -> 1)
    kernels::launch_fused_vertex_tri(
        d_positions_, d_normals_, d_texcoords_, d_texids_, d_colors_,
        tri_count,
        d_mvp_u, d_mv_u, d_nm_u,
        state_.viewport_x, state_.viewport_y, vw, vh,
        d_triangles_, stream_, state_.tri_stats,
        {d_near_clip_ids_, d_near_clip_count_});

    // 4-5. Bin triangles (deterministic path by default; see
    // bin_triangles_deterministic for why the legacy atomicAdd binning is racy)
    last_num_tiles_ = num_tiles;
    if (deterministic_binning_) {
        bin_triangles_deterministic(tri_count, tiles_x, tiles_y, vw, vh);
    } else {
        kernels::launch_clear_tile_bin_counts(d_tile_bins_, num_tiles, stream_);
        kernels::launch_bin_triangles(
            d_triangles_, tri_count,
            d_tile_bins_, tiles_x, tiles_y,
            vw, vh,
            stream_, state_.tri_stats, {d_near_clip_ids_, d_near_clip_count_});
    }
    // 6. Fine rasterize (with integrated sort)
    TextureState texture_state = {};
    if (fb_->samples() == 4) {
        mc_flush();
        kernels::launch_fine_rasterize_msaa(d_triangles_, d_tile_base_, d_cursor_,
            reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
            pool_capacity_, tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_);
        return;
    }
    mc_flush();
    kernels::launch_fine_rasterize_with_sort(
        d_triangles_, d_tile_bins_,
        // Segment lengths = what the scatter actually WROTE (see d_seg_counts).
        d_tile_base_, deterministic_binning_ ? d_cursor_ : d_tile_counts_,
        reinterpret_cast<const unsigned long long*>(d_pool_sorted_tri_),
        deterministic_binning_ ? pool_capacity_ : 0,
        tiles_x, tiles_y, fb_->device_fb(),
        state_, texture_state, stream_);
}

static inline int hp_vtx_pair_on() {
    static const int v = getenv("HP_VTX_PAIR") ? atoi(getenv("HP_VTX_PAIR")) : 1;
    return v;
}
static inline int hp_vtx_pair_max_w() {
    static const int v = getenv("HP_VTX_PAIR_MAX_W") ? atoi(getenv("HP_VTX_PAIR_MAX_W")) : (1 << 30);
    return v;
}
static inline int hp_vtx_pair_dbg() {
    static const int v = getenv("HP_VTX_PAIR_DBG") ? atoi(getenv("HP_VTX_PAIR_DBG")) : 0;
    return v;
}
#define VPDBG(tag) do { if (hp_vtx_pair_dbg()) { static int o = 0; \
    if (o++ < 6) fprintf(stderr, "[worldaxis] bail %s\n", tag); } return false; } while (0)

void Pipeline::vtx_pair_arm(const RenderState& second, const float* second_matrices) {
    vtx_pair_armed_ = false;
    vtx_pair_ready_ = false;
    vtx_pair_pos_   = nullptr;
    if (!hp_vtx_pair_on()) return;
    // The second pass reads its matrices from the batch's device block. Without
    // it the pass would need its own H2D blit, which is the cost this removes.
    if (!second_matrices) return;
    if (second.viewport_w > hp_vtx_pair_max_w()) return;
    vtx_pair_state_ = second;
    vtx_pair_mat_   = second_matrices;
    vtx_pair_armed_ = true;
}

void Pipeline::vtx_pair_cancel() {
    vtx_pair_armed_ = false;
    vtx_pair_ready_ = false;
    vtx_pair_pos_   = nullptr;
    bin_pair_cancel();
}

static bool hp_binemit_same(const kernels::BinEmit& a, const kernels::BinEmit& b) {
    if (a.rec != b.rec || a.tiles_x != b.tiles_x || a.tiles_y != b.tiles_y) return false;
    if (a.fb_w != b.fb_w || a.fb_h != b.fb_h || a.cull_mask != b.cull_mask) return false;
    if (a.skip_dead_core != b.skip_dead_core) return false;
    if (a.center_samples_only != b.center_samples_only) return false;
    return hp_subs_same(a.subs, b.subs);
}

bool Pipeline::vtx_pair_try_launch(const float* d_pos, const float* d_norm,
                                   const float* d_tex, const int* d_texid,
                                   const float* d_col, const float* d_specular,
                                   const float* d_shininess, const float* d_emission,
                                   int tri_count, int vw, int vh,
                                   const float* mvp, const float* mv, const float* nm,
                                   const kernels::BinEmit& emit) {
    if (!vtx_pair_armed_) return false;
    vtx_pair_armed_ = false;                 // one shot, whatever happens below
    if (!emit.rec) VPDBG("norec");
    if (stats_wanted_ || state_.tri_stats) VPDBG("stats");
    if (!fb_ || fb_->samples() == 4) VPDBG("msaa");
    // Slot 0 must be the REFLECTED pass (the one that is issued first).
    if (!state_.clip_plane_enabled) VPDBG("noclip");
    if (vtx_pair_state_.clip_plane_enabled) VPDBG("clip2");
    if (vtx_pair_state_.viewport_w != vw || vtx_pair_state_.viewport_h != vh ||
        vtx_pair_state_.viewport_x != state_.viewport_x ||
        vtx_pair_state_.viewport_y != state_.viewport_y) VPDBG("vp");
    if (!vtx_pair_state_.gouraud_shading) VPDBG("gouraud2");
    if (tri_count <= 0 || tri_count > INT_MAX / 2) VPDBG("tris");

    auto grow = [](void** p, size_t bytes, size_t* have) {
        if (*have >= bytes && *p) return true;
        if (*p) hipFree(*p);
        *p = nullptr;
        if (hipMalloc(p, bytes) != hipSuccess) { *have = 0; return false; }
        *have = bytes;
        return true;
    };
    const size_t tri_bytes = (size_t)(2 * tri_count) * sizeof(TriangleSetup);
    if (!grow((void**)&d_vtx1_tris_, tri_bytes, &sz_vtx1_tris_)) VPDBG("alloc_tris");
    const size_t rec_bytes = (size_t)(2 * tri_count) * sizeof(kernels::BinRec);
    if (!grow(&d_vtx1_rec_, rec_bytes, &sz_vtx1_rec_)) VPDBG("alloc_rec");
    const size_t ncid_bytes = (size_t)(near_clip_capacity_ > tri_count
                                       ? near_clip_capacity_ : tri_count) * sizeof(int);
    if (!grow((void**)&d_vtx1_nc_ids_, ncid_bytes, &sz_vtx1_nc_ids_)) VPDBG("alloc_nc");
    if (!d_nc_pool_) VPDBG("ncpool");
    // Same rotating pre-zeroed counter pool the ordinary draws use, so slot 1
    // gets a guaranteed-zero counter without a 4-byte fill dispatch.
    const int kNcSlots = hp_nc_slots();
    if (nc_slot_ == 0) {
        cq_flush();
        check_hip(hipMemsetAsync(d_nc_pool_, 0, (size_t)kNcSlots * sizeof(int), stream_),
                  "Pipeline: reset near-plane queue (pair)");
    }
    d_vtx1_nc_count_ = d_nc_pool_ + nc_slot_;
    nc_slot_ = (nc_slot_ + 1 == kNcSlots) ? 0 : (nc_slot_ + 1);

    kernels::VtxZSlot s0, s1;
    s0.mvp = mvp; s0.model_view = mv; s0.normal_mat = nm;
    s0.out = d_triangles_;
    s0.near_clip = {d_near_clip_ids_, d_near_clip_count_};
    s0.emit = emit;
    s0.state = state_;

    s1.mvp = vtx_pair_mat_; s1.model_view = vtx_pair_mat_ + 16; s1.normal_mat = vtx_pair_mat_ + 32;
    s1.out = d_vtx1_tris_;
    s1.near_clip = {d_vtx1_nc_ids_, d_vtx1_nc_count_};
    s1.emit = emit;
    s1.emit.rec = reinterpret_cast<kernels::BinRec*>(d_vtx1_rec_);
    s1.emit.cull_mask = kernels::backface_cull_mask(vtx_pair_state_);
    s1.state = vtx_pair_state_;
    s1.state.tri_stats = nullptr;
    s1.state.near_clip_source_count = tri_count;

    if (hp_vtx_pair_on() == 6) {
        // DIAGNOSTIC BISECT: identical buffers and hoisting, but two ORDINARY
        // single-slot dispatches instead of one gridDim.z = 2 grid.
        kernels::launch_fused_vertex_tri_gouraud(
            d_pos, d_norm, d_tex, d_texid, d_col, (const float*)nullptr,
            d_specular, d_shininess, d_emission, tri_count,
            s0.mvp, s0.model_view, s0.normal_mat,
            state_.viewport_x, state_.viewport_y, vw, vh,
            s0.state, s0.out, stream_, s0.near_clip, s0.emit);
        kernels::launch_fused_vertex_tri_gouraud(
            d_pos, d_norm, d_tex, d_texid, d_col, (const float*)nullptr,
            d_specular, d_shininess, d_emission, tri_count,
            s1.mvp, s1.model_view, s1.normal_mat,
            state_.viewport_x, state_.viewport_y, vw, vh,
            s1.state, s1.out, stream_, s1.near_clip, s1.emit);
    } else
    if (!kernels::launch_fused_vertex_tri_gouraud_z2(
            d_pos, d_norm, d_tex, d_texid, d_col, (const float*)nullptr,
            d_specular, d_shininess, d_emission, tri_count,
            state_.viewport_x, state_.viewport_y, vw, vh, s0, s1, stream_))
        VPDBG("launcher");

    // HP_VTX_PAIR=2: DIAGNOSTIC. Launch the pair but let the second draw run
    // its own ordinary vertex pass, so only SLOT 0's output is consumed.
    vtx_pair_ready_ = (hp_vtx_pair_on() != 2);
    bin_pair_ = Pipeline::BinPair{};
    vtx_pair_pos_   = d_pos;
    vtx_pair_tris_  = tri_count;
    vtx_pair_vw_    = vw;
    vtx_pair_vh_    = vh;
    vtx_pair_emit_  = s1.emit;
    vtx_pair_state_ = s1.state;
    if (hp_vtx_pair_dbg()) {
        static int o = 0;
        if (o++ < 6) fprintf(stderr, "[worldaxis] ENGAGED B=2 tris=%d grid=%dx1x2\n",
                          tri_count, (tri_count + 255) / 256);
    }
    return true;
}

bool Pipeline::vtx_pair_take(const float* d_pos, int tri_count, int vw, int vh,
                             const kernels::BinEmit& emit) {
    if (!vtx_pair_ready_) return false;
    vtx_pair_ready_ = false;                 // one shot
    if (d_pos != vtx_pair_pos_ || tri_count != vtx_pair_tris_) VPDBG("take_span");
    if (vw != vtx_pair_vw_ || vh != vtx_pair_vh_) VPDBG("take_vp");
    if (stats_wanted_ || state_.tri_stats) VPDBG("take_stats");
    kernels::BinEmit want = emit;
    want.rec = reinterpret_cast<kernels::BinRec*>(d_vtx1_rec_);
    want.cull_mask = kernels::backface_cull_mask(state_);
    if (!hp_binemit_same(want, vtx_pair_emit_)) VPDBG("take_emit");
    RenderState want_st = state_;
    want_st.tri_stats = nullptr;
    want_st.near_clip_source_count = tri_count;
    // Compare everything up to the END OF THE LAST MEMBER. The 4 bytes of TAIL
    // PADDING after reflect_rect are indeterminate (they differ between two
    // copies of the same object) and are never read by the kernel.
    const size_t st_cmp_bytes = offsetof(RenderState, reflect_rect) + sizeof(int32_t) * 4;
    if (memcmp(&want_st, &vtx_pair_state_, st_cmp_bytes) != 0) {
        if (hp_vtx_pair_dbg()) {
            static int o = 0;
            if (!o++) {
                const unsigned char* a = (const unsigned char*)&want_st;
                const unsigned char* b = (const unsigned char*)&vtx_pair_state_;
                fprintf(stderr, "[worldaxis] state diff bytes:");
                int n = 0;
                for (size_t i = 0; i < sizeof(RenderState); ++i)
                    if (a[i] != b[i] && n++ < 40) fprintf(stderr, " %zu(%d/%d)", i, a[i], b[i]);
                fprintf(stderr, "  total_diff=%d\n", n);
            }
        }
        VPDBG("take_state");
    }
    // Swap slot 1's private pool in for this draw's binning + fine pass. The
    // reflected draw's own pool is left untouched (it is already consumed).
    sv_tris_   = d_triangles_;      sv_tricap_ = triangle_capacity_;
    sv_rec_    = d_binrec_;         sv_recsz_  = sz_binrec_;
    sv_ncid_   = d_near_clip_ids_;  sv_nccap_  = near_clip_capacity_;
    sv_nccnt_  = d_near_clip_count_;
    d_triangles_      = d_vtx1_tris_;
    triangle_capacity_ = 2 * tri_count;
    d_binrec_         = d_vtx1_rec_;
    sz_binrec_        = sz_vtx1_rec_;
    d_near_clip_ids_  = d_vtx1_nc_ids_;
    d_near_clip_count_ = d_vtx1_nc_count_;
    vtx_pair_swapped_ = true;
    if (hp_vtx_pair_dbg()) {
        static int o = 0;
        if (o++ < 6) fprintf(stderr, "[worldaxis] TAKE tris=%d\n", tri_count);
    }
    return true;
}

void Pipeline::vtx_pair_swap_out() {
    if (!vtx_pair_swapped_) return;
    vtx_pair_swapped_ = false;
    d_triangles_       = sv_tris_;
    triangle_capacity_ = sv_tricap_;
    d_binrec_          = sv_rec_;
    sz_binrec_         = sv_recsz_;
    d_near_clip_ids_   = sv_ncid_;
    near_clip_capacity_ = sv_nccap_;
    d_near_clip_count_ = sv_nccnt_;
}
#undef VPDBG

static inline int hp_micro_pair_on() {
    static const int v = getenv("HP_MICRO_PAIR") ? atoi(getenv("HP_MICRO_PAIR")) : 1;
    return v;
}

void Pipeline::micro_pair_begin(const float* d_pos, const float* d_norm, const float* d_tex,
                                const int* d_texid, const float* d_col, const float* d_specular,
                                const float* d_shininess, const float* d_emission,
                                int tri_count_full, int tri_count_first,
                                const RenderState& st_full, const RenderState& st_first) {
    micro_absorbing_ = true;       // do not let the flush below emit the clear
    flush_pending_draw();
    micro_absorbing_ = false;
    micro_pair_slots_ = 0;
    micro_pair_next_  = 0;
#define MPDBG(tag) do { static const int dbg = getenv("HP_MICRO_PAIR_DBG") ? 1 : 0; \
    if (dbg) { static int once = 0; if (!once++) fprintf(stderr, "[micropair] bail %s\n", tag); } \
    return; } while (0)
    if (!hp_micro_pair_on()) MPDBG("off");
    if (!fb_ || !d_pos) MPDBG("fb/pos");
    if (!deterministic_binning_ || !state_.gouraud_shading) MPDBG("detbin/gouraud");
    if (!kernels::fused_vertex_binrec_supported()) { /* only affects BinRec, fine */ }
    static const int hp_abl = getenv("HP_ABL") ? atoi(getenv("HP_ABL")) : 0;
    static const int hp_binabl = getenv("HP_BINABL") ? atoi(getenv("HP_BINABL")) : 0;
    static const int micro_on = getenv("HP_MICRO") ? atoi(getenv("HP_MICRO")) : 1;
    if (hp_abl != 0 || hp_binabl != 0 || micro_on == 0) MPDBG("abl");
    if (tri_count_first <= 0 || tri_count_full <= 0) MPDBG("tri<=0");
    if (tri_count_first > tri_count_full) MPDBG("first>full");
    if (tri_count_full > kernels::micro_bin_max_tris()) MPDBG("maxtris");
    if (fb_->samples() == 4) MPDBG("msaa");
    const int vw = st_full.viewport_w, vh = st_full.viewport_h;
    if (vw <= 0 || vh <= 0) MPDBG("vp");
    if (st_first.viewport_w != vw || st_first.viewport_h != vh) MPDBG("vp2");
    const int tiles_x = (vw + TILE_SIZE - 1) / TILE_SIZE;
    const int tiles_y = (vh + TILE_SIZE - 1) / TILE_SIZE;
    const int num_tiles = tiles_x * tiles_y;
    if (num_tiles <= 0 || num_tiles > kernels::micro_bin_max_tiles()) MPDBG("maxtiles");

    const int vertex_count = tri_count_full * 3;
    // The shared TriangleSetup array: primaries [0, N) + near-plane suffixes
    // [N, 2N). Both slots index the suffix of source i at N + i.
    const size_t tri_bytes = (size_t)(2 * tri_count_full) * sizeof(TriangleSetup);
    auto grow = [](void** p, size_t bytes, size_t* have) {
        if (*have >= bytes && *p) return true;
        if (*p) hipFree(*p);
        *p = nullptr;
        if (hipMalloc(p, bytes) != hipSuccess) { *have = 0; return false; }
        *have = bytes;
        return true;
    };
    if (!grow((void**)&d_micro_tris_, tri_bytes, &sz_micro_tris_)) MPDBG("alloc_tris");
    // tile_base is [num_tiles+1] per slot, cursor [num_tiles] per slot. The
    // chain writes every element of both, so no pre-zeroing is needed.
    const size_t bin_ints = (size_t)2 * ((size_t)num_tiles + 1) + (size_t)2 * num_tiles;
    if (!grow((void**)&d_micro_bins_, bin_ints * sizeof(int), &sz_micro_bins_)) MPDBG("alloc_bins");
    // CONSERVATIVE, EXACT bound: a live record covers at most num_tiles tiles
    // and there are at most 2*tri_count_full records, so this pool can never
    // clamp -- which is why the pair needs no readback, no sizing sync and no
    // share of the scene pool the reflected draw overwrites between the two
    // fine passes.
    const long long cap_ll = (long long)2 * tri_count_full * (long long)num_tiles;
    if (cap_ll <= 0 || cap_ll > (long long)INT_MAX / 2) MPDBG("cap");
    const int cap = (int)cap_ll;
    if (!grow((void**)&d_micro_pool_, (size_t)2 * cap * sizeof(unsigned long long),
              &sz_micro_pool_)) MPDBG("alloc_pool");

    // ---- the ONE vertex pass, over the full span, with the SECOND draw's
    // shading state. Back-face culling is switched OFF for the pass so that
    // EVERY triangle gets its full payload: slot 0 may bin triangles slot 1
    // culls, and the vertex kernel's cull-before-shade would otherwise leave
    // their vertex records unwritten. The per-slot cull rule is applied inside
    // the bin chain, exactly where the count pass applies it today, so the SET
    // of pooled triangles is unchanged for both draws.
    RenderState vst = st_full;
    vst.backface_cull_enabled = false;
    vst.near_clip_source_count = tri_count_full;
    vst.tri_stats = nullptr;
    if (stats_wanted_) MPDBG("stats");

    ensure_scratch(vertex_count, tri_count_full, tiles_x, tiles_y);
    state_.near_clip_source_count = tri_count_full;

    const float* d_mvp_u = d_mvp_;
    const float* d_mv_u  = d_model_view_;
    const float* d_nm_u  = d_normal_mat_;
    if (d_ext_matrices_) {
        d_mvp_u = d_ext_matrices_; d_mv_u = d_ext_matrices_ + 16; d_nm_u = d_ext_matrices_ + 32;
    } else {
        memcpy(h_matrices_, st_full.mvp, 16 * sizeof(float));
        memcpy(h_matrices_ + 16, st_full.model_view, 16 * sizeof(float));
        memcpy(h_matrices_ + 32, st_full.normal_matrix, 9 * sizeof(float));
        hipMemcpyAsync(d_matrices_, h_matrices_, 41 * sizeof(float), hipMemcpyHostToDevice, stream_);
    }
    kernels::launch_fused_vertex_tri_gouraud(
        d_pos, d_norm, d_tex, d_texid, d_col,
        (const float*)nullptr, d_specular, d_shininess, d_emission,
        tri_count_full, d_mvp_u, d_mv_u, d_nm_u,
        vst.viewport_x, vst.viewport_y, vw, vh,
        vst, d_micro_tris_, stream_, {d_near_clip_ids_, d_near_clip_count_},
        kernels::BinEmit{});

    // ---- ONE batched bin dispatch, blockIdx.x = slot.
    kernels::MicroDrawSlotDesc slots[2];
    const int np1 = num_tiles + 1;
    for (int i = 0; i < 2; ++i) {
        slots[i].tris        = d_micro_tris_;
        slots[i].tri_count   = (i == 0) ? tri_count_first : tri_count_full;
        slots[i].suffix_base = tri_count_full;
        slots[i].cull_mask   = kernels::backface_cull_mask(i == 0 ? st_first : st_full);
        slots[i].tile_base   = d_micro_bins_ + (size_t)i * np1;
        slots[i].cursor      = d_micro_bins_ + (size_t)2 * np1 + (size_t)i * num_tiles;
        slots[i].key_out     = d_micro_pool_ + (size_t)i * cap;
        slots[i].pool_capacity = cap;
        slots[i].near_clip   = {d_near_clip_ids_, d_near_clip_count_};
    }
    kernels::MicroClearDesc clr{};
    if (pend_clear_valid_) {
        const DeviceFramebuffer dfb = fb_->device_fb();
        clr.color = dfb.color_buffer; clr.depth = dfb.depth_buffer;
        clr.total = pend_clear_total_;
        clr.r = pend_clear_rgba_[0]; clr.g = pend_clear_rgba_[1];
        clr.b = pend_clear_rgba_[2]; clr.a = pend_clear_rgba_[3];
        clr.d = pend_clear_depth_;
        if (pend_cvt_dst_ && pend_cvt_src_ == dfb.color_buffer &&
            pend_cvt_total_ == pend_clear_total_) {
            clr.cvt_dst = pend_cvt_dst_;
        }
    }
    // Whatever could not ride must still land BEFORE the dispatch that clears.
    if (pend_cvt_dst_ && !(clr.total > 0 && clr.cvt_dst)) flush_pending_convert();
    if (!kernels::launch_micro_bin_chain_n(slots, 2, tiles_x, tiles_y, vw, vh,
                                           num_tiles, d_overflow_persist_, stream_,
                                           clr.total > 0 ? &clr : nullptr)) {
        { static int o=0; if(getenv("HP_MICRO_PAIR_DBG") && !o++) fprintf(stderr,"[micropair] bail launcher\n"); }
        // The launcher refused: nothing has been consumed yet except a vertex
        // pass into private scratch, so the two draws simply run as usual.
        return;
    }
    micro_pair_pos_      = d_pos;
    micro_pair_tris_[0]  = tri_count_first;
    micro_pair_tris_[1]  = tri_count_full;
    micro_pair_full_     = tri_count_full;
    micro_pair_tiles_x_  = tiles_x;
    micro_pair_tiles_y_  = tiles_y;
    micro_pair_num_tiles_= num_tiles;
    micro_pair_cap_      = cap;
    micro_pair_slots_    = 2;
    micro_pair_next_     = 0;
    if (clr.total > 0) pend_clear_valid_ = false;   // folded into the dispatch
    if (clr.cvt_dst) {
        pend_cvt_src_ = nullptr; pend_cvt_dst_ = nullptr; pend_cvt_total_ = 0;
    }
    { static int o=0; if(getenv("HP_MICRO_PAIR_DBG") && !o++) fprintf(stderr,"[micropair] ENGAGED full=%d first=%d tiles=%d cap=%d\n", tri_count_full, tri_count_first, num_tiles, cap); }
}
#undef MPDBG

bool Pipeline::micro_pair_is_take(const float* d_pos, int tri_count, int tiles_x, int tiles_y) const {
    return micro_pair_slots_ > 0 && micro_pair_next_ < micro_pair_slots_ &&
           d_pos == micro_pair_pos_ && tiles_x == micro_pair_tiles_x_ &&
           tiles_y == micro_pair_tiles_y_ && tri_count == micro_pair_tris_[micro_pair_next_] &&
           fb_ != nullptr;
}

bool Pipeline::micro_pair_take(const float* d_pos, int tri_count, int tiles_x, int tiles_y,
                               const TextureState& texture_state) {
    if (micro_pair_slots_ <= 0 || micro_pair_next_ >= micro_pair_slots_) return false;
    if (d_pos != micro_pair_pos_) return false;
    if (tiles_x != micro_pair_tiles_x_ || tiles_y != micro_pair_tiles_y_) return false;
    if (tri_count != micro_pair_tris_[micro_pair_next_]) return false;
    if (!fb_) return false;
    const int slot = micro_pair_next_++;
    if (micro_pair_next_ >= micro_pair_slots_) micro_pair_slots_ = 0;
    const int num_tiles = micro_pair_num_tiles_;
    const int np1 = num_tiles + 1;
    last_num_tiles_ = num_tiles;
    work_ready_    = false;
    flt_ready_     = false;
    flt_msk_ready_ = false;
    // The shared vertex pass laid the near-plane suffixes out at
    // micro_pair_full_ + i, so the fine rasterizer's id decode must use THAT
    // count, not this draw's own triangle count.
    state_.near_clip_source_count = micro_pair_full_;
    state_.tri_stats = nullptr;
    mc_flush();
    kernels::launch_fine_rasterize_with_sort(
        d_micro_tris_, d_tile_bins_,
        d_micro_bins_ + (size_t)slot * np1,
        d_micro_bins_ + (size_t)2 * np1 + (size_t)slot * num_tiles,
        d_micro_pool_ + (size_t)slot * micro_pair_cap_,
        micro_pair_cap_,
        tiles_x, tiles_y, fb_->device_fb(), state_, texture_state, stream_,
        nullptr, nullptr, nullptr, 1, 0, num_tiles,
        nullptr, nullptr, nullptr);
    return true;
}

void Pipeline::ensure_scratch(int vertex_count, int tri_count, int tiles_x, int tiles_y) {
    int num_tiles = tiles_x * tiles_y;
    if (tri_count > INT_MAX / 2)
        throw std::overflow_error("Pipeline: near-plane output triangle count exceeds int range");
    const int output_tri_count = tri_count * 2;

    // Diagnostic stage counters. Grown here so the buffer always matches the
    // triangle count of the draw about to run, and ZEROED every draw: these
    // are per-draw counts, and a stale row would read as a triangle that
    // survived a stage it never reached.
    if (stats_wanted_) {
        if (output_tri_count > tri_stats_cap_) {
            if (d_tri_stats_) hipFree(d_tri_stats_);
            d_tri_stats_ = nullptr;
            tri_stats_cap_ = 0;
            check_hip(hipMalloc(&d_tri_stats_,
                      (size_t)output_tri_count * TRI_STAT_COUNT * sizeof(int32_t)),
                      "Pipeline: triangle stats");
            tri_stats_cap_ = output_tri_count;
        }
        check_hip(hipMemsetAsync(d_tri_stats_, 0,
                       (size_t)output_tri_count * TRI_STAT_COUNT * sizeof(int32_t),
                       stream_), "Pipeline: reset triangle stats");
        state_.tri_stats = d_tri_stats_;
    } else {
        state_.tri_stats = nullptr;
    }
    last_tri_count_ = output_tri_count;
    state_.near_clip_source_count = tri_count;

    auto grow = [](int current, int needed) {
        if (needed <= current) return current;
        int new_cap = current > 0 ? current : 64;
        while (new_cap < needed) {
            if (new_cap > INT_MAX / 2) return needed;
            new_cap *= 2;
        }
        return new_cap;
    };

    int new_transformed = grow(transformed_capacity_, vertex_count);
    int new_triangles = grow(triangle_capacity_, output_tri_count);
    int new_near_clip = grow(near_clip_capacity_, tri_count);
    int new_tile_bins = grow(tile_bins_capacity_, num_tiles);
    int new_vertex = grow(vertex_capacity_, vertex_count);
    if (new_transformed > transformed_capacity_ || new_tile_bins > tile_bins_capacity_ ||
        new_vertex > vertex_capacity_ || vertex_count * 3 > index_capacity_ || !d_nc_pool_)
        cq_flush();

    if (new_transformed > transformed_capacity_) {
        if (d_transformed_) hipFree(d_transformed_);
        hipError_t err = hipMalloc(&d_transformed_, static_cast<size_t>(new_transformed) * sizeof(TransformedVertex));
        check_hip(err, "Pipeline: ensure_scratch transformed");
        transformed_capacity_ = new_transformed;
    }

    if (new_triangles > triangle_capacity_) {
        if (d_triangles_) hipFree(d_triangles_);
        hipError_t err = hipMalloc(&d_triangles_, static_cast<size_t>(new_triangles) * sizeof(TriangleSetup));
        check_hip(err, "Pipeline: ensure_scratch triangles");
        triangle_capacity_ = new_triangles;
    }

    if (new_near_clip > near_clip_capacity_) {
        if (d_near_clip_ids_) hipFree(d_near_clip_ids_);
        d_near_clip_ids_ = nullptr;
        near_clip_capacity_ = 0;
        check_hip(hipMalloc(&d_near_clip_ids_, static_cast<size_t>(new_near_clip) * sizeof(int)),
                  "Pipeline: near-plane queue IDs");
        near_clip_capacity_ = new_near_clip;
    }
    const int kNcSlots = hp_nc_slots();
    if (!d_nc_pool_) {
        check_hip(hipMalloc(&d_nc_pool_, (size_t)kNcSlots * sizeof(int)),
                  "Pipeline: near-plane queue count");
        nc_slot_ = 0;
        check_hip(hipMemsetAsync(d_nc_pool_, 0, (size_t)kNcSlots * sizeof(int), stream_),
                  "Pipeline: reset near-plane queue");
    } else if (nc_slot_ == 0) {
        cq_flush();
        check_hip(hipMemsetAsync(d_nc_pool_, 0, (size_t)kNcSlots * sizeof(int), stream_),
                  "Pipeline: reset near-plane queue");
    }
    d_near_clip_count_ = d_nc_pool_ + nc_slot_;
    nc_slot_ = (nc_slot_ + 1 == kNcSlots) ? 0 : (nc_slot_ + 1);

    if (new_tile_bins > tile_bins_capacity_) {
        if (d_tile_bins_) hipFree(d_tile_bins_);
        hipError_t err = hipMalloc(&d_tile_bins_, static_cast<size_t>(new_tile_bins) * sizeof(TileBin));
        check_hip(err, "Pipeline: ensure_scratch tile_bins");
        tile_bins_capacity_ = new_tile_bins;
    }

    if (new_vertex > vertex_capacity_) {
        if (d_positions_) hipFree(d_positions_);
        if (d_normals_) hipFree(d_normals_);
        if (d_texcoords_) hipFree(d_texcoords_);
        if (d_texids_) hipFree(d_texids_);
        if (d_colors_) hipFree(d_colors_);
        size_t v3 = static_cast<size_t>(new_vertex) * 3 * sizeof(float);
        size_t v2 = static_cast<size_t>(new_vertex) * TEXCOORD_COMPONENTS * sizeof(float);
        size_t vi = static_cast<size_t>(new_vertex) * sizeof(int);
        size_t v4 = static_cast<size_t>(new_vertex) * 4 * sizeof(float);
        check_hip(hipMalloc(&d_positions_, v3), "Pipeline: d_positions");
        check_hip(hipMalloc(&d_normals_, v3), "Pipeline: d_normals");
        check_hip(hipMalloc(&d_texcoords_, v2), "Pipeline: d_texcoords");
        check_hip(hipMalloc(&d_texids_, vi), "Pipeline: d_texids");
        check_hip(hipMalloc(&d_colors_, v4), "Pipeline: d_colors");
        if (d_spec_colors_) hipFree(d_spec_colors_);
        check_hip(hipMalloc(&d_spec_colors_, v3), "Pipeline: d_spec_colors");
        vertex_capacity_ = new_vertex;
    }

    int max_index = vertex_count * 3;  // conservative for index buffer
    if (max_index > index_capacity_) {
        int new_index = grow(index_capacity_, max_index);
        if (d_indices_) hipFree(d_indices_);
        check_hip(hipMalloc(&d_indices_, static_cast<size_t>(new_index) * sizeof(uint32_t)), "Pipeline: d_indices");
        index_capacity_ = new_index;
    }
}

void Pipeline::cq_flush() const {
    if (!cq_q_.empty()) const_cast<Pipeline*>(this)->cq_do_flush();
}

void Pipeline::cq_do_flush() {
    std::vector<CqPending> q;
    q.swap(cq_q_);
    cq_busy_ = 0;
    const int n = (int)q.size();
    if (n <= 0) return;
    cq_stat_f_++;
    const kernels::CqVtxRec*   vr[HP_CQ_MAX];
    const kernels::CqScatDesc* sr[HP_CQ_MAX];
    const kernels::CqSortRec*  orr[HP_CQ_MAX];
    for (int i = 0; i < n; ++i) { vr[i] = &q[i].rec.v; sr[i] = &q[i].rec.s; orr[i] = &q[i].rec.o; }
    static const int fmode = getenv("HP_CQ_FMODE") ? atoi(getenv("HP_CQ_FMODE")) : 0;
    if (fmode == 2) {
        for (int i = 0; i < n; ++i) {
            kernels::launch_cq_vtx(vr + i, 1, stream_);
            kernels::launch_cq_scat(sr + i, 1, stream_);
            kernels::launch_cq_sort(orr + i, 1, stream_);
            if (q[i].rb) cq_rb_issue(q[i].rb_src, q[i].rb_meta);
            q[i].fine();
        }
    } else {
        if (fmode == 1) {
            for (int i = 0; i < n; ++i) kernels::launch_cq_vtx(vr + i, 1, stream_);
            for (int i = 0; i < n; ++i) kernels::launch_cq_scat(sr + i, 1, stream_);
            for (int i = 0; i < n; ++i) kernels::launch_cq_sort(orr + i, 1, stream_);
        } else {
            kernels::launch_cq_vtx(vr, n, stream_);
            kernels::launch_cq_scat(sr, n, stream_);
            kernels::launch_cq_sort(orr, n, stream_);
        }
        for (int i = 0; i < n; ++i)
            if (q[i].rb) cq_rb_issue(q[i].rb_src, q[i].rb_meta);
        kernels::fine_batch_begin();
        for (int i = 0; i < n; ++i) q[i].fine();
        kernels::fine_batch_end();
    }
    q.clear();
    if (cq_q_.empty()) cq_q_.swap(q);   // keep the capacity
}

void Pipeline::cq_rb_issue(const int* src, const int* meta) {
    if (!src || !h_readback_) return;
    hipMemcpyAsync(h_readback_ + 0, src, sizeof(int), hipMemcpyDeviceToHost, stream_);
    rb_tri_count_ = meta[0]; rb_tiles_x_ = meta[1]; rb_tiles_y_ = meta[2]; rb_rect_tiles_ = meta[3];
}

void Pipeline::cq_abandon() {
    kernels::cq_rec_slot() = nullptr;
    if (!cq_recording_) return;
    cq_recording_ = false;
    cq_stat_a_++;
    if (getenv("HP_CQ_DBG") && atoi(getenv("HP_CQ_DBG")) > 1)
        fprintf(stderr, "[cq] abandon got=%d q=%zu rb=%d\n", cq_cur_.got, cq_q_.size(), (int)cq_rb_);
    cq_flush();
    const kernels::CqChainRec r = cq_cur_;
    if (r.got & 1) { const kernels::CqVtxRec* p = &r.v;   kernels::launch_cq_vtx(&p, 1, stream_); }
    if (r.got & 2) { const kernels::CqScatDesc* p = &r.s; kernels::launch_cq_scat(&p, 1, stream_); }
    if (r.got & 4) { const kernels::CqSortRec* p = &r.o;  kernels::launch_cq_sort(&p, 1, stream_); }
    if (cq_rb_) cq_rb_issue(cq_rb_src_, cq_rb_meta_);
    cq_rb_ = false;
}

void Pipeline::cq_save_set(int s) {
    CqSet& c = cq_sets_[s];
    c.tris = d_triangles_; c.tri_cap = triangle_capacity_;
    c.binrec = d_binrec_; c.sz_binrec = sz_binrec_;
    c.nc_ids = d_near_clip_ids_; c.nc_cap = near_clip_capacity_;
    c.bounds = d_bounds_; c.sz_bounds = sz_bounds_;
    c.pool_cap = pool_capacity_; c.pool_tri = d_pool_tri_; c.pool_z = d_pool_z_;
    c.pst = d_pool_sorted_tri_; c.sz_st = sz_st_; c.psz = d_pool_sorted_z_; c.sz_sz = sz_sz_;
    c.flt_rec = d_flt_rec_; c.sz_flt_rec = sz_flt_rec_;
    c.flt_cnt = d_flt_cnt_; c.sz_flt_cnt = sz_flt_cnt_;
    c.flt_msk = d_flt_msk_; c.sz_flt_msk = sz_flt_msk_;
}

void Pipeline::cq_load_set(int s) {
    const CqSet& c = cq_sets_[s];
    d_triangles_ = c.tris; triangle_capacity_ = c.tri_cap;
    d_binrec_ = c.binrec; sz_binrec_ = c.sz_binrec;
    d_near_clip_ids_ = c.nc_ids; near_clip_capacity_ = c.nc_cap;
    d_bounds_ = c.bounds; sz_bounds_ = c.sz_bounds;
    pool_capacity_ = c.pool_cap; d_pool_tri_ = c.pool_tri; d_pool_z_ = c.pool_z;
    d_pool_sorted_tri_ = c.pst; sz_st_ = c.sz_st; d_pool_sorted_z_ = c.psz; sz_sz_ = c.sz_sz;
    d_flt_rec_ = c.flt_rec; sz_flt_rec_ = c.sz_flt_rec;
    d_flt_cnt_ = c.flt_cnt; sz_flt_cnt_ = c.sz_flt_cnt;
    d_flt_msk_ = c.flt_msk; sz_flt_msk_ = c.sz_flt_msk;
}

void Pipeline::free_scratch() {
    pend_.valid = false;
    kernels::cq_rec_slot() = nullptr;
    cq_recording_ = false;
    cq_q_.clear();
    cq_busy_ = 0;
    if (getenv("HP_CQ_DBG"))
        fprintf(stderr, "[cq] queued=%llu flushes=%llu abandons=%llu\n",
                (unsigned long long)cq_stat_q_, (unsigned long long)cq_stat_f_,
                (unsigned long long)cq_stat_a_);
    cq_save_set(cq_cur_set_);
    for (int s = 0; s < HP_CQ_MAX; ++s) {
        if (s == cq_cur_set_) continue;
        CqSet& c = cq_sets_[s];
        if (c.tris) hipFree(c.tris);
        if (c.binrec) hipFree(c.binrec);
        if (c.nc_ids) hipFree(c.nc_ids);
        if (c.bounds) hipFree(c.bounds);
        if (c.pool_tri) hipFree(c.pool_tri);
        if (c.pool_z) hipFree(c.pool_z);
        if (c.pst) hipFree(c.pst);
        if (c.psz) hipFree(c.psz);
        if (c.flt_rec) hipFree(c.flt_rec);
        if (c.flt_cnt) hipFree(c.flt_cnt);
        if (c.flt_msk) hipFree(c.flt_msk);
        c = CqSet{};
    }
    if (d_transformed_) { hipFree(d_transformed_); d_transformed_ = nullptr; }
    if (d_triangles_) { hipFree(d_triangles_); d_triangles_ = nullptr; }
    if (d_near_clip_ids_) { hipFree(d_near_clip_ids_); d_near_clip_ids_ = nullptr; }
    if (d_nc_pool_) { hipFree(d_nc_pool_); d_nc_pool_ = nullptr; }
    d_near_clip_count_ = nullptr; nc_slot_ = 0;
    if (d_tile_bins_) { hipFree(d_tile_bins_); d_tile_bins_ = nullptr; }
    if (d_positions_) { hipFree(d_positions_); d_positions_ = nullptr; }
    if (d_normals_) { hipFree(d_normals_); d_normals_ = nullptr; }
    if (d_texcoords_) { hipFree(d_texcoords_); d_texcoords_ = nullptr; }
    if (d_texids_) { hipFree(d_texids_); d_texids_ = nullptr; }
    if (d_colors_) { hipFree(d_colors_); d_colors_ = nullptr; }
    if (d_spec_colors_) { hipFree(d_spec_colors_); d_spec_colors_ = nullptr; }
    if (d_indices_) { hipFree(d_indices_); d_indices_ = nullptr; }
    if (d_matrices_) { hipFree(d_matrices_); d_matrices_ = nullptr; }
    if (d_flt_rec_) { hipFree(d_flt_rec_); d_flt_rec_ = nullptr; sz_flt_rec_ = 0; }
    if (d_flt_cnt_) { hipFree(d_flt_cnt_); d_flt_cnt_ = nullptr; sz_flt_cnt_ = 0; }
    if (d_flt_msk_) { hipFree(d_flt_msk_); d_flt_msk_ = nullptr; sz_flt_msk_ = 0; }
    bin_pair_cancel();
    if (d_bin1_base_) { hipFree(d_bin1_base_); d_bin1_base_ = nullptr; sz_bin1_base_ = 0; }
    if (d_bin1_key_)  { hipFree(d_bin1_key_);  d_bin1_key_  = nullptr; sz_bin1_key_  = 0; }
    flt_ready_ = false;
    flt_msk_ready_ = false;
    d_mvp_ = nullptr;
    d_model_view_ = nullptr;
    d_normal_mat_ = nullptr;
    transformed_capacity_ = 0;
    triangle_capacity_ = 0;
    near_clip_capacity_ = 0;
    tile_bins_capacity_ = 0;
    vertex_capacity_ = 0;
    index_capacity_ = 0;
}

} // namespace hipgl
