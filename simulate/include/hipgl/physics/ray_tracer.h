#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include <cstdint>
#include <memory>
#include <vector>
#include <hip/hip_runtime.h>

#include "physics_types.h"
#include "rt_types.h"

namespace hipgl {
namespace physics {

// ---- K1 arguments (rt_scene.hip) ---------------------------------------------
struct RtSceneArgs {
    const RtGeom* geom;  int nrg;
    const float*  geom_xpos;  const float* geom_xquat;  int ngeom;   // dd, [world][ngeom]
    const float*  xpos;       const float* xquat;       int nbody;   // dd, [world][nbody]
    const int*    cam_body;   const float* cam_pos;
    const float*  cam_quat;   const float* cam_fovy;                 // dm, [ncamera]
    int           cam_ids[MAX_CAMERAS]; int ncam;                    // selected cameras (by value)
    const int*    active;     int nslot;                             // world of slot s
    RtInst*       inst;                                              // [nslot][nrg]
    RtCam*        cam;                                               // [nslot][ncam]
    const float*  site_xpos;  const float* site_xquat; int nsite;    // dd, [world][nsite]
    const float*  site_rgba;  int site_rgba_nw;                      // per-world rgba or null
    // Optional fused view cull (vlist && vcount): K1 also writes the
    // [nslot][ncam][nrg] view lists / [nslot][ncam] counts K2's tiles walk,
    // for the W x H image window with the clip planes below.
    int*          vlist;      int* vcount;
    int           W, H, clip; float znear, zfar;
};

// Launch wrappers (defined in src/kernels/rt_scene.hip / rt_trace.hip).
// All are asynchronous on `stream`.
// Returns false (nothing launched) for an empty request. start (nullable) is
// recorded at the start of the K1 dispatch.
bool rt_launch_inst(const RtSceneArgs& a, hipStream_t stream, hipEvent_t start = nullptr);
// K2 launch options (kernel variant selection; not part of RtFrame).
struct RtLaunchOpts {
    bool stats     = false;
    bool tile_cull = true;
    bool swizzle   = false;
    // Small-batch tiles: single-sample passes use the smallest tile side
    // tp in {2, 4} whose launch has <= max_waves workgroups, else 8 (0 = always 8x8).
    // Fewer distinct pixels per wave shorten the slowest wave when the GPU is
    // mostly idle (LIBERO N = 1: 2x lower latency); see rt_pick_tp1 in rt_trace.hip.
    int  max_waves = 0;
    // View-cull lists (tile_cull only): [nslot][ncam][nrg] instance ids and
    // [nslot][ncam] counts, written by the pre-pass, read by K2. Both null =
    // tiles cull all nrg instances.
    int* vlist  = nullptr;
    int* vcount = nullptr;
    // The lists are already written for every slot (fused into K1): K2 skips
    // its view-cull pre-pass.
    bool vlist_ready = false;
    // Deferred full shading (split): trace pass -> records -> shade kernel.
    // Used when split is set and rec / rec_mir are non-null; nrec = records per
    // kind for this chunk (main, mirror, then RT_MAX_TLAYERS layer kinds).
    bool   split   = false;
    // > 0: use the fused kernel anyway when the trace pass has at most this
    // many workgroups (small, latency-bound launches; set_shade_split('auto')).
    int    split_auto_waves = 0;
    uint4* rec     = nullptr;
    uint4* rec_mir = nullptr;
    uint4* rec_lay = nullptr;
    size_t nrec    = 0;
};
// K2 for slots [slot0, slot0 + nslot_chunk). F.nslot is the total slot count.
// nslot_chunk must not exceed rt_primary_max_slots(F.W, F.H, F.ss, F.ncam) (1-D grid).
// One or two K2 passes (even ssaa with depth/seg adds a pixel-centre pass).
void rt_launch_primary(const RtFrame& F, int slot0, int nslot_chunk, const RtLaunchOpts& opt,
                       hipStream_t stream);
int  rt_primary_max_slots(int W, int H, int ss, int ncam);
void rt_launch_ray_batch(const RtFrame& F, const float* orig, const float* dir, int n,
                         int* gid, float* t, float* nrm, hipStream_t stream);
// Unit world dirs of the W x H pixel centres of F.cam[0] (row 0 = top image row).
void rt_launch_debug_rays(const RtFrame& F, float* dirs, hipStream_t stream);
// One glGenerateMipmap level (RT_SH_HW_TEX): dst (dw x dh per face, `faces`
// stacked faces) from src (sw x sh per face), float bilinear at destination
// texel centres (a 2x2 box for even sizes), rounded once (nearest-even) to 8 bits.
void rt_launch_hw_mip(const unsigned char* src, int sw, int sh, int faces,
                      unsigned char* dst, int dw, int dh, hipStream_t stream);

// ---- one render request --------------------------------------------------------
struct RtRequest {
    const int* cam_ids = nullptr;  int ncam = 0;   // host, camera indices
    const int* worlds  = nullptr;  int nslot = 0;  // host, world index per slot
    const int* out_rows = nullptr;                 // host, output row per slot (nullptr: worlds)
    int  nout = 0;                                 // rows per camera in the outputs
    int  out_mask = RT_OUT_RGB;
    bool flip_v = false, flip_h = false;           // relative to kernel order (row 0 = top)
};

struct RtStatsSnapshot {
    int    nrg = 0, ntri = 0, nblas_nodes = 0, nblas_nodes64 = 0;
    int    nsite_items = 0;                      // site entries in the render list (not in nrg)
    float  last_ms = 0.0f;                       // K1+K2 device time of the last render
    // Only with set_stats(true):
    //   cand_per_ray  = instances whose world AABB a ray's slab test accepted, per ray;
    //   cand_per_tile = instances accepted by at least one lane of an 8x8 tile,
    //                   per tile and per wave-wide ray pass (SSAA sample);
    //   nodes_per_ray / tris_per_ray = BLAS nodes visited / triangles tested;
    //   list_per_tile = tile-frustum survivors per tile and ray pass (0 with
    //                   tile_cull off), visited_per_tile = list entries walked
    //                   before the early break, per tile and ray pass.
    double cand_per_ray = 0, cand_per_tile = 0, nodes_per_ray = 0, tris_per_ray = 0;
    double list_per_tile = 0, visited_per_tile = 0;
    bool   counters_valid = false;
};

class RayTracer {
public:
    RayTracer();
    ~RayTracer();
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    // Builds the render-geom list, per-mesh BLAS (both layouts), light table and
    // uploads them. Textures must already be set on the model (texid/is_cube are
    // read here). width/height are OUTPUT pixels; ss is the per-axis SSAA factor.
    bool init(const PhysicsModelDesc& desc, const DevicePhysicsModel& dm,
              int width, int height, int ss);
    bool initialized() const { return initialized_; }
    void release();

    int width() const  { return W_; }
    int height() const { return H_; }
    int ssaa() const   { return ss_; }

    // Render settings. Geom-group changes rebuild the RtGeom list on the next render.
    void set_geomgroup(int group, bool visible);
    bool geomgroup(int group) const { return group >= 0 && group < 6 && geomgroup_[group]; }
    // Sites (MuJoCo mjv addSiteGeoms): drawn as lit geoms when their group is
    // visible (default 0,1,2) and their alpha >= 0.995 in the rendered world;
    // alpha <= 0 hides them (robosuite/LIBERO toggle visualisation sites by
    // flipping the alpha sign). Translucent sites are not drawn (yet).
    void set_sitegroup(int group, bool visible);
    bool sitegroup(int group) const { return group >= 0 && group < 6 && sitegroup_[group]; }
    int  nsite() const { return nsite_; }
    // Per-world site RGBA (CameraRenderer semantics): one world (env >= 0) or
    // all `num_envs` worlds (env < 0, rgba is [num_envs][nsite][4]). Worlds
    // never set keep the model's rgba. clear: one world, or all (env < 0).
    void set_site_rgba(int num_envs, const float* rgba, int env);
    void clear_site_rgba(int env);
    void get_site_rgba(int num_envs, float* out, int env) const;   // env < 0: all worlds
    // Explicit GL_OBJECT_LINEAR texgen planes for geom `gid` ((s,t,r) x (a,b,c,d)
    // over the hip geom-local position), e.g. MuJoCo's settexture() planes for a
    // UV-less mesh re-expressed in hip's (not recentred) mesh frame. Used under
    // RT_SH_TEXGEN_MJ. nullptr clears.
    void set_texgen_planes(int gid, const float* planes12);
    // Per-texture RtTexFlags (e.g. RT_TEX_SRGB for MuJoCo sRGB textures, which
    // hip stores linearised). n = number of textures.
    void set_texture_flags(const int* flags, int n);
    // MuJoCo's sRGB-encoded texels of 2D texture `texid` (w x h x 3, rows top
    // first, as mjModel.tex_data): the RT builds llvmpipe's sRGB mip pyramid
    // (decode, float bilinear at destination centres, encode, round) and
    // samples it on llvmpipe's float path with Mesa's sRGB decode
    // (RT_SH_MESA_TEX). ntexture = number of model textures.
    void set_texture_srgb_data(int texid, int ntexture, const unsigned char* rgb, int w, int h);
    // Per-corner normals for mesh `mesh_id` (vert_num x 3, hip soup corner
    // order and frame), e.g. MuJoCo's mesh_normal/mesh_facenormal rotated by
    // mesh_quat: hip's STL loader computes its own (smoothed/flat) normals.
    void set_mesh_normals(int mesh_id, const float* nrm, int n);
    // Replace the light table: lights[0] is the headlight (type 0, direction
    // = camera +Z), lights[1..n-1] scene lights in MuJoCo's order (n <=
    // RT_MAX_LIGHTS). As Mesa (8 fixed-function lights), only the first 8
    // ACTIVE entries light the scene; with none active the GL global ambient
    // 0.3 applies. Shadows are not modelled (castshadow is ignored).
    void set_lights(const RtLight* lights, int n);
    int  nlights_active() const;
    // Override geom gid's RGBA (e.g. MuJoCo's resolved setMaterial colour):
    // updates the transparency / mirror-candidate flags. Returns false when
    // gid is not a render geom (alpha <= 0 or group >= 3 at init).
    bool set_geom_rgba(int gid, const float rgba[4]);
    // Draw primitive geom gid as an ellipsoid with semi-axes size (hip's
    // physics parses MJCF ellipsoids as boxes). False for meshes / non-render geoms.
    bool set_geom_ellipsoid(int gid, const float size[3]);
    void set_mode(int mode)      { mode_ = mode; }
    void set_cull(int cull)      { cull_ = cull; }
    void set_traversal(int trav) { trav_ = trav; }
    void set_clip(bool on)       { clip_ = on; }
    void set_clip_planes(float znear, float zfar) { znear_ = znear; zfar_ = zfar; }
    void set_bg_color(float r, float g, float b) { bg_[0] = r; bg_[1] = g; bg_[2] = b; }
    void set_stats(bool on)      { stats_on_ = on; }
    void set_tile_cull(bool on)  { tile_cull_ = on; }
    void set_swizzle(bool on)    { swizzle_ = on; }
    void set_adaptive_tiles(bool on) { adaptive_tiles_ = on; }
    // Full shading as a trace pass + a separate shade kernel over compact hit
    // records (1), one fused kernel (0), or auto (2, default: fused when the
    // trace launch is at most ~2 rounds of resident waves, i.e. latency-bound
    // small batches, else split). Identical images in every mode.
    void set_shade_split(int mode) { shade_split_ = mode; }
    int  shade_split() const { return shade_split_; }
    void set_record_cap(size_t bytes) { rec_cap_bytes_ = bytes; }
    bool adaptive_tiles() const { return adaptive_tiles_; }
    bool tile_cull() const { return tile_cull_; }
    bool swizzle() const   { return swizzle_; }
    void set_gradient_bg(const float bot[3], const float top[3]) {
        grad_ = true; for (int k = 0; k < 3; ++k) { gbot_[k] = bot[k]; gtop_[k] = top[k]; }
    }
    void clear_gradient_bg() { grad_ = false; }
    void set_fog(float start, float end, float r, float g, float b) {
        fog_s_ = start; fog_e_ = end; fog_rgb_[0] = r; fog_rgb_[1] = g; fog_rgb_[2] = b;
    }
    void set_linear_output(bool on, float exposure, float gamma) {
        linear_ = on; exposure_ = exposure; inv_gamma_ = gamma > 0.0f ? 1.0f / gamma : 1.0f;
    }
    void set_shade_flags(int flags) { shade_ = flags; }
    int  shade_flags() const { return shade_; }
    void set_shade_flag(int bit, bool on) { shade_ = on ? (shade_ | bit) : (shade_ & ~bit); }
    void set_aa(int aa) { aa_ = aa; }
    int  aa() const { return aa_; }
    void set_lod_bias(float b) { lod_bias_ = b; }
    float lod_bias() const { return lod_bias_; }
    // Render-geom list index of the mirror (first visible opaque plane/box with
    // reflectance > 0, MuJoCo's isReflective order), -1 if none.
    int  mirror_li() { rebuild_geoms(); return mirror_li_; }
    int  mode() const { return mode_; }
    int  cull() const { return cull_; }
    int  traversal() const { return trav_; }
    bool clip() const { return clip_; }
    float znear() const { return znear_; }
    float zfar() const  { return zfar_; }

    void render(const DevicePhysicsModel& dm, const DevicePhysicsData& dd,
                hipStream_t stream, const RtRequest& req);
    uint8_t* d_rgb() const   { return static_cast<uint8_t*>(rgb_.get()); }
    float*   d_depth() const { return static_cast<float*>(depth_.get()); }
    int*     d_seg() const   { return static_cast<int*>(seg_.get()); }
    // Shared ownership of the current output allocations. A holder keeps the
    // memory alive across a later regrow (which then allocates a new buffer
    // instead of freeing this one) and across release()/destruction.
    std::shared_ptr<void> rgb_buffer() const   { return rgb_; }
    std::shared_ptr<void> depth_buffer() const { return depth_; }
    std::shared_ptr<void> seg_buffer() const   { return seg_; }
    int device() const { return dev_; }

    void ray_batch(const DevicePhysicsModel& dm, const DevicePhysicsData& dd, hipStream_t stream,
                   int world, const float* h_orig, const float* h_dir, int n,
                   int* h_gid, float* h_t, float* h_nrm);
    // eye[3], Rc[9] row-major (columns = camera x,y,z axes), dirs [H][W][3].
    void debug_rays(const DevicePhysicsModel& dm, const DevicePhysicsData& dd, hipStream_t stream,
                    int camera, int world, float* h_eye, float* h_Rc, float* h_dirs);

    // Reads the last render's counters/events; call after the stream drained.
    RtStatsSnapshot stats();

    const std::vector<RtGeom>& geoms() const { return h_geom_; }

private:
    void rebuild_geoms();               // host list from groupvis + upload
    void ensure_slots(int nslot, int ncam, int nout, int out_mask, bool grew[3]);
    void prepare_outputs(const RtRequest& req, const bool grew[3], hipStream_t stream);
    void stage_slots(const int* tab, size_t n, hipStream_t stream);
    RtFrame make_frame(const DevicePhysicsModel& dm) const;
    void run_inst(const DevicePhysicsModel& dm, const DevicePhysicsData& dd,
                  const int* cam_ids, int nslot, int ncam, hipStream_t stream,
                  bool view_cull = false, hipEvent_t start = nullptr);

    bool initialized_ = false;
    int  dev_ = 0;                      // device of the model's allocations
    int  W_ = 0, H_ = 0, ss_ = 1;
    int  ngeom_ = 0, nbody_ = 0, ncamera_ = 0, nmesh_ = 0;

    // settings
    bool  geomgroup_[6] = {true, true, true, false, false, false};
    bool  sitegroup_[6] = {true, true, true, false, false, false};
    int   nsite_ = 0;
    std::vector<float> h_site_default_;   // [nsite][4] model site rgba
    std::vector<float> h_site_rgba_;      // [site_nw_][nsite][4] effective per-world rgba
    int   site_nw_ = 0;
    float* d_site_rgba_ = nullptr;
    std::vector<float> h_texgen_;         // [ngeom][12]
    std::vector<char>  h_texgen_on_;      // [ngeom]
    float4* d_texgen_ = nullptr;
    int*    d_tex_flags_ = nullptr; int ntex_flags_ = 0;
    std::vector<int> h_tex_flags_;
    std::vector<unsigned char> h_srgb_data_;
    std::vector<TextureMipLevel> h_srgb_mip_;
    unsigned char* d_srgb_data_ = nullptr;
    TextureMipLevel* d_srgb_mip_ = nullptr;
    // hardware-GL pyramid (RT_SH_HW_TEX), built on first use from the model's
    // level 0 and rebuilt when the model's texture buffer changes
    unsigned char* d_hw_data_ = nullptr;
    TextureMipLevel* d_hw_mip_ = nullptr;
    const unsigned char* hw_src_ = nullptr; int hw_ntex_ = 0, hw_gen_ = 0;
    void ensure_hw_mips(const DevicePhysicsModel& dm, hipStream_t stream);
    std::vector<float> h_nrm_;            // RT copy of mesh normals (set_mesh_normals)
    float* d_nrm_ = nullptr;
    RtLight* d_lights_ = nullptr;
    std::vector<int> mesh_adr_, mesh_num_;
    bool  geoms_dirty_ = true;
    int   mode_ = RT_MODE_HIPGL, cull_ = RT_CULL_PER_TRI, trav_ = RT_TRAV_ORDERED;
    bool  clip_ = false, stats_on_ = false;
    bool  tile_cull_ = true, swizzle_ = false, adaptive_tiles_ = true;
    int   shade_split_ = 2;
    size_t rec_cap_bytes_ = size_t(4) << 30;
    int   max_waves_ = 0;               // 40 x CU count: ~2 rounds of resident waves (small-batch tile rule)
    float znear_ = 0.0f, zfar_ = 0.0f;
    float bg_[3] = {0.0f, 0.0f, 0.0f};
    bool  grad_ = false;
    float gbot_[3] = {0.0f, 0.0f, 0.0f}, gtop_[3] = {0.0f, 0.0f, 0.0f};
    float fog_s_ = 0.0f, fog_e_ = 0.0f, fog_rgb_[3] = {0.0f, 0.0f, 0.0f};
    bool  linear_ = false;
    float exposure_ = 1.0f, inv_gamma_ = 1.0f;
    int   shade_ = RT_SH_PRESET_MUJOCO;   // set_shading_preset('raster') for the raster's model
    int   aa_ = RT_AA_SSAA;
    float lod_bias_ = 0.0f;
    int   mirror_li_ = -1;

    // static scene (host copies)
    std::vector<RtGeom>  h_all_;        // every renderable geom, groupvis ignored
    std::vector<int>     h_all_group_;  // geom group of h_all_[i]
    std::vector<RtGeom>  h_geom_;       // current list (li order)
    int                  ntransp_ = 0;  // h_geom_ entries with RT_F_TRANSPARENT (rebuild_geoms)
    std::vector<int>     tab_scratch_;  // per-call host scratch (render slot table)
    std::vector<int>     key_scratch_;  // per-call host scratch (prepare_outputs layout key)
    std::vector<char>    hit_scratch_;  // per-call host scratch (prepare_outputs covered rows)
    std::vector<RtLight> h_light_;      // as requested (set_lights / parsed)
    void upload_lights();               // capped copy -> d_lights_, gl_amb_
    int  gl_amb_ = 0;
    int ntri_ = 0, nnode32_ = 0, nnode64_ = 0;

    // device: static
    RtGeom*   d_geom_  = nullptr;   // capacity h_all_.size()
    RtNode32* d_n32_   = nullptr;
    RtNode64* d_n64_   = nullptr;
    RtNode4*  d_n4_    = nullptr;
    int       nnode4_  = 0;           // BVH4 node count of the BLAS set
    RtTri*    d_tri_   = nullptr;
    RtTri*    d_tri64_ = nullptr;
    RtBlas*   d_blas_  = nullptr;

    // device: per frame, grow-only
    RtInst* d_inst_ = nullptr;  size_t cap_inst_ = 0;
    RtCam*  d_cam_  = nullptr;  size_t cap_cam_  = 0;
    int*    d_vlist_ = nullptr; size_t cap_vlist_ = 0;   // [nslot][ncam][nrg] view-cull lists
    uint4*  d_rec_ = nullptr; size_t cap_rec_ = 0;       // deferred-shading records (split)
    int*    d_vcount_ = nullptr; size_t cap_vcount_ = 0; // [nslot][ncam]
    int*    d_slot_ = nullptr;  int*   h_slot_  = nullptr;  size_t cap_slot_ = 0;  // [active | out_row]
    std::vector<int> last_slot_;
    hipEvent_t ev_stage_ = nullptr;     // recorded after each H2D from h_slot_
    bool  staged_ = false;
    // Outputs: shared so Python views can outlive a regrow (see rgb_buffer()).
    std::shared_ptr<void> rgb_, depth_, seg_;
    size_t cap_rgb_ = 0, cap_depth_ = 0, cap_seg_ = 0;   // elements
    std::vector<int> layout_[3];        // [nout, cam ids...] last written per output
    unsigned long long* d_stats_ = nullptr;
    hipEvent_t ev0_ = nullptr, ev1_ = nullptr;
    bool  timed_ = false;
    bool  last_stats_ = false;
    long long last_tiles_ = 0;

    // ray_batch scratch (grow-only)
    float* d_rb_in_ = nullptr;  size_t cap_rb_ = 0;   // [orig | dir], n*6
    int*   d_rb_gid_ = nullptr; float* d_rb_t_ = nullptr; float* d_rb_n_ = nullptr;
    float* d_dbg_dirs_ = nullptr; size_t cap_dbg_ = 0;
};

}  // namespace physics
}  // namespace hipgl
