#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

// Physics-to-rendering bridge: generates draw geometry from physics state
// entirely on GPU, without host roundtrip.

#include "physics_types.h"
#include "physics_engine.h"

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
typedef void* hipStream_t;
#endif

namespace hipgl {

class Pipeline;
class Framebuffer;
struct RenderState;

namespace physics {

// Render geometry info for a single geom type (box, sphere, capsule).
struct PhysicsRenderMesh {
    float* d_positions;     // device: [max_geoms * verts_per_geom * 3]
    float* d_normals;       // device: [max_geoms * verts_per_geom * 3]
    float* d_colors;        // device: [max_geoms * verts_per_geom * 4]
    int    verts_per_geom;
    int    max_geoms;
    int    allocated;
};

// Manages GPU-side render meshes built from physics geom transforms.
// No host roundtrip: physics geom_xpos/xquat feed directly into vertex buffers.
class PhysicsRenderer {
public:
    PhysicsRenderer();
    ~PhysicsRenderer();

    // Allocate render mesh buffers for the given model. CameraRenderer uses
    // its host overlay path for sites, so it excludes these unused spans.
    // Standalone callers retain the original site-building behavior.
    bool init(const PhysicsModelDesc& desc, bool include_sites = true);

    // Build vertex positions/normals from physics geom world transforms.
    // env_idx selects which environment to render.
    void build_geometry(const DevicePhysicsModel& model,
                        const DevicePhysicsData& data,
                        int env_idx,
                        hipStream_t stream);

    void set_active_groups(unsigned group_mask, bool transparent);

    // Render the built geometry through the hipGL pipeline.
    void render(Pipeline& pipeline, const RenderState& state, hipStream_t stream);

    // Total vertex count from last build
    int total_vertices() const { return total_verts_; }
    int geom_vertices() const { return geom_verts_; }
    int site_vertex_offset() const { return site_vert_offset_; }
    int site_vertices() const { return site_verts_; }

    // Per-group vertex ranges within the geom section (opaque only)
    int group_offset(int g) const { return group_offset_[g]; }
    // Host copies of the per-geom vertex layout, kept for diagnostics: they
    // let a test read the vertex buffer for ONE named geom instead of
    // inferring what a generator did from the rendered image.
    int ngeom() const { return ngeom_; }
    int geom_vert_offset_host(int g) const {
        return (g >= 0 && g < (int)h_geom_vert_offset_.size()) ? h_geom_vert_offset_[g] : -1;
    }
    int geom_vert_count_host(int g) const {
        return (g >= 0 && g < (int)h_geom_vert_count_.size()) ? h_geom_vert_count_[g] : 0;
    }
    int group_count(int g) const { return group_count_[g]; }

    // Intrinsic-transparent layout range (alpha < 0.995). Never submit this
    // whole range without filtering geomgroup: inactive spans may be stale.
    int transparent_offset() const { return transparent_offset_; }
    int transparent_count() const { return transparent_count_; }

    void set_position_slice(int k) {
        cur_slice_ = (k > 0 && k < pos_slices_) ? k : 0;
    }
    int position_slices() const { return pos_slices_; }

    // One launch for `nworlds` worlds of one worker: world j is env
    // `env_base + j*env_stride` and writes position/normal slice j.  Returns
    // false (and launches NOTHING) unless the steady-state single-dispatch
    // rebuild is live and the slices exist, so the caller falls back to the
    // per-world path.
    bool build_geometry_batch(const DevicePhysicsModel& model,
                              const DevicePhysicsData& data,
                              int env_base, int env_stride, int nworlds,
                              hipStream_t stream);

    // Publish the per-draw pose descriptor for `env_idx` against the CURRENT
    // slice, without rebuilding anything.  build_geometry() does this itself;
    // a world whose geometry came from build_geometry_batch() needs it done
    // separately, immediately before its draws are issued.
    void arm_pose(const DevicePhysicsData& data, int env_idx);

    // Access device buffers (for direct pipeline submission)
    float* device_positions() const { return d_all_positions_ + cur_slice_ * pos_slice_stride_; }
    float* device_normals()  const { return d_all_normals_ + cur_slice_ * pos_slice_stride_; }
    float* device_texcoords() const { return d_all_texcoords_; }
    int*   device_texids()   const { return d_all_texids_; }
    float* device_colors()   const { return d_all_colors_; }
    float* device_spec_colors() const { return d_spec_colors_; }
    float* device_specular() const { return d_specular_; }
    float* device_shininess() const { return d_shininess_; }
    float* device_emission() const { return d_emission_; }

    bool share_invariant_attribs_from(const PhysicsRenderer& src);

private:
    float* d_all_positions_;
    float* d_all_normals_;
    int       pos_slices_ = 1;
    long long pos_slice_stride_ = 0;  // floats between slices (max_verts_*3)
    int       cur_slice_ = 0;         // slice the accessors currently expose
    void seed_position_slices(hipStream_t stream);
    bool pose_fold_ready() const;
    float* d_all_texcoords_;
    int*   d_all_texids_;
    float* d_all_colors_;
    float* d_spec_colors_;      // per-vertex separate specular RGB (GL_SEPARATE_SPECULAR_COLOR)
    float* d_specular_;         // per-vertex MuJoCo specular coeff
    float* d_shininess_;        // per-vertex MuJoCo shininess (GL = this * 128)
    float* d_emission_;         // per-vertex MuJoCo emission coeff
    int    total_verts_;
    int    geom_verts_;
    int    site_vert_offset_;
    int    site_verts_;
    std::vector<int> h_geom_vert_offset_;  // host mirror, diagnostics only
    std::vector<int> h_geom_vert_count_;   // host mirror, diagnostics only
    int    group_offset_[3];  // vertex offset for each group (0,1,2) -- opaque only
    int    group_count_[3];   // vertex count for each group -- opaque only
    int    transparent_offset_; // start of transparent geom vertices
    int    transparent_count_;  // number of transparent geom vertices
    int    max_verts_;
    bool   initialized_;

    int    ngeom_;
    int*   d_geom_vert_offset_;  // per-geom vertex offset in the combined buffer
    int*   d_geom_vert_count_;   // per-geom vertex count

    int    nsite_;
    int*   d_site_vert_offset_;  // per-site vertex offset in the combined buffer
    int*   d_site_vert_count_;   // per-site vertex count

    int    total_mesh_verts_;    // total mesh vertices across all mesh geoms
    int*   d_vert_to_geom_;     // [total_mesh_verts] which geom owns each vertex
    int*   d_vert_local_idx_;   // [total_mesh_verts] local vertex index within mesh

    unsigned active_mask_;       // bit g set => render group g is drawn
    bool   active_transparent_;  // OIT span drawn
    bool   active_valid_;        // compaction below matches (mask, transparent)
    bool   attribs_valid_;       // world/frame-invariant vertex attributes filled
    bool   attribs_shared_;      // invariant arrays are BORROWED (do not free)
    int*   d_geom_active_;       // [ngeom_] 1 if the geom is drawn
    int*   d_geom_tri_work_;     // [n_geom_tri_work_] packed gid*SUBS + tile
    int    n_geom_tri_work_;
    void*  d_mesh_seg_;          // [n_mesh_seg_] int4
    int    n_mesh_seg_;
    void*  d_fused_desc_;        // [n_fused_desc_] int4
    int    n_fused_desc_;
    std::vector<int> h_geom_type_;     // [ngeom_] geom type, for the work list
    int    total_mesh_verts_active_;   // compacted mesh work-list length
    int*   d_vert_to_geom_act_;        // [total_mesh_verts_active_]
    int*   d_vert_local_idx_act_;      // [total_mesh_verts_active_]
    std::vector<int> h_vert_to_geom_;  // full mesh work list (host mirror)
    std::vector<int> h_vert_local_idx_;
    std::vector<int> h_geom_group_;    // [ngeom_] render group
    std::vector<char> h_geom_transparent_;  // [ngeom_]
    void rebuild_active();

    int*   d_vert_geom_;         // [max_verts_] owning geom for folded spans
    void*  d_fold_desc_;         // [n_fold_desc_] int4, foldable descriptors
    int    n_fold_desc_;
    int*   d_nofold_work_;       // [n_nofold_work_] packed gid*SUBS + tile
    int    n_nofold_work_;
    bool   pose_fold_;           // buffer is in model space
    bool   model_built_;         // model-space emit has run for this mask
    std::vector<int> h_vert_geom_;

    int    prim_vert_map_n_;     // == max_verts_ when allocated, else 0
    int    total_prim_verts_;    // primitive vertices only (diagnostic)
    int*   d_prim_vert_to_geom_; // [prim_vert_map_n_] owning geom, or -1

    void free_device();
};

} // namespace physics
} // namespace hipgl
