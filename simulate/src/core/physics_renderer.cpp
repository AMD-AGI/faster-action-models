// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PhysicsRenderer: builds render geometry from physics state on GPU.

#include "hipgl/physics/physics_renderer.h"
#include "hipgl/pipeline.h"
#include "hipgl/kernels.h"
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cstdlib>

namespace hipgl {
namespace physics {

// From physics_renderer.hip
namespace kernels {
void launch_build_render_geometry(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    const int* d_geom_vert_count,
    const int* d_geom_active,
    const int* d_geom_tri_work, int n_tri_work,
    bool write_attribs,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    int* d_total_verts,
    hipStream_t stream);

void launch_build_geometry_model(
    int ngeom,
    const DevicePhysicsModel& model,
    const int* d_geom_vert_offset,
    int nblk,
    const void* d_desc,
    float* d_positions, float* d_normals,
    hipStream_t stream);

int geom_tri_subblocks(int gtype);
int geom_tri_pack(int gid, int sub);
int mesh_seg_block();

void launch_build_geometry_fused(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    const int* d_geom_vert_count,
    const int* d_geom_active,
    int nblk,
    const void* d_desc,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs, bool write_normals,
    hipStream_t stream,
    int nworlds = 1, int env_stride = 0, long long slice_stride = 0);

void launch_transform_mesh_seg(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    int nseg,
    const void* d_seg,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs,
    hipStream_t stream);

void launch_transform_mesh_verts(
    int ngeom, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_geom_vert_offset,
    int total_mesh_verts,
    const int* d_vert_to_geom,
    const int* d_vert_local_idx,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    float* d_specular, float* d_shininess, float* d_emission,
    bool write_attribs,
    hipStream_t stream);

void launch_build_render_sites(
    int nsite, int env_idx,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const int* d_site_vert_offset,
    float* d_positions, float* d_normals, float* d_texcoords, int* d_texids, float* d_colors,
    hipStream_t stream);

void launch_fill_primitive_materials(
    int n_verts,
    const int* d_prim_vert_to_geom,
    const DevicePhysicsModel& model,
    float* d_specular, float* d_shininess, float* d_emission,
    hipStream_t stream);
}

// MUST match src/kernels/physics_renderer.hip's BOX_NUMQUADS. verts_for_type()
// below sizes the vertex span the kernel then writes, so a mismatch silently
// corrupts the buffer layout rather than failing loudly.
#ifndef HIPGL_BOX_NUMQUADS
#define HIPGL_BOX_NUMQUADS 4
#endif
static constexpr int BOX_NUMQUADS = HIPGL_BOX_NUMQUADS;
static constexpr int BOX_VERTS = 6 * BOX_NUMQUADS * BOX_NUMQUADS * 2 * 3;
static constexpr int SPHERE_SLICES = 28;
static constexpr int SPHERE_STACKS = 16;
static constexpr int SPHERE_VERTS = SPHERE_SLICES * SPHERE_STACKS * 6;
static constexpr int CAPSULE_NSEG = 16;
static constexpr int CAPSULE_HEMI_RINGS = 8;
static constexpr int CAPSULE_CYLINDER_VERTS = CAPSULE_NSEG * 6;
static constexpr int CAPSULE_HEMI_VERTS = CAPSULE_NSEG * CAPSULE_HEMI_RINGS * 6;
static constexpr int CAPSULE_VERTS = CAPSULE_CYLINDER_VERTS + 2 * CAPSULE_HEMI_VERTS;
static constexpr int CYLINDER_NSEG = 16;
static constexpr int CYLINDER_VERTS = CYLINDER_NSEG * 6 + CYLINDER_NSEG * 3 * 2;
static // Overridable: a 6 m plane at SUBDIV=10 gives 0.6 m quads, i.e. triangles that
#ifndef HIPGL_PLANE_SUBDIV
#define HIPGL_PLANE_SUBDIV 10
#endif
constexpr int PLANE_SUBDIV = HIPGL_PLANE_SUBDIV;
static constexpr int PLANE_VERTS = PLANE_SUBDIV * PLANE_SUBDIV * 6;

static int verts_for_type(int geom_type, int mesh_id, const PhysicsModelDesc& desc) {
    switch (geom_type) {
        case GEOM_BOX:     return BOX_VERTS;
        case GEOM_SPHERE:  return SPHERE_VERTS;
        case GEOM_CAPSULE:  return CAPSULE_VERTS;
        case GEOM_CYLINDER: return CYLINDER_VERTS;
        case GEOM_PLANE:   return PLANE_VERTS;
        case GEOM_MESH:
            if (mesh_id >= 0 && mesh_id < desc.nmesh)
                return desc.meshes[mesh_id].vert_num;
            return BOX_VERTS;
        default:           return BOX_VERTS;
    }
}

// Mirrors MAX_TRIS_PER_GEOM in the .hip (SPHERE_TRIS dominates); diagnostics only.
static constexpr int MAX_TRIS_HOST = SPHERE_SLICES * SPHERE_STACKS * 2;

static int verts_for_site(int site_type) {
    switch (site_type) {
        case SITE_TYPE_SPHERE:   return SPHERE_VERTS;
        case SITE_TYPE_CAPSULE:  return CAPSULE_VERTS;
        case SITE_TYPE_CYLINDER: return CAPSULE_VERTS;
        case SITE_TYPE_BOX:      return BOX_VERTS;
        default:                 return 0;
    }
}

PhysicsRenderer::PhysicsRenderer()
    : d_all_positions_(nullptr), d_all_normals_(nullptr), d_all_texcoords_(nullptr),
      d_all_texids_(nullptr), d_all_colors_(nullptr), d_spec_colors_(nullptr),
      d_specular_(nullptr), d_shininess_(nullptr), d_emission_(nullptr),
      total_verts_(0), geom_verts_(0), site_vert_offset_(0), site_verts_(0),
      group_offset_{0,0,0}, group_count_{0,0,0},
      transparent_offset_(0), transparent_count_(0),
      max_verts_(0), initialized_(false),
      ngeom_(0), d_geom_vert_offset_(nullptr), d_geom_vert_count_(nullptr),
      nsite_(0), d_site_vert_offset_(nullptr), d_site_vert_count_(nullptr),
      total_mesh_verts_(0), d_vert_to_geom_(nullptr), d_vert_local_idx_(nullptr),
      active_mask_(0x7u), active_transparent_(true), active_valid_(false),
      attribs_valid_(false), attribs_shared_(false), d_geom_active_(nullptr),
      d_geom_tri_work_(nullptr), n_geom_tri_work_(0),
      d_mesh_seg_(nullptr), n_mesh_seg_(0),
      d_fused_desc_(nullptr), n_fused_desc_(0),
      total_mesh_verts_active_(0), d_vert_to_geom_act_(nullptr),
      d_vert_local_idx_act_(nullptr),
      d_vert_geom_(nullptr), d_fold_desc_(nullptr), n_fold_desc_(0),
      d_nofold_work_(nullptr), n_nofold_work_(0),
      pose_fold_(false), model_built_(false),
      prim_vert_map_n_(0), total_prim_verts_(0), d_prim_vert_to_geom_(nullptr) {}

PhysicsRenderer::~PhysicsRenderer() {
    free_device();
}

bool PhysicsRenderer::init(const PhysicsModelDesc& desc, bool include_sites) {
    free_device();
    ngeom_ = desc.ngeom;
    nsite_ = include_sites ? desc.nsite : 0;

    // Compute per-geom vertex offsets in group order (0, then 1, then 2).
    // MuJoCo classifies alpha < 0.995 as intrinsically transparent. Keep its
    // spans separate from opaque geometry; submission filters groups per geom.
    std::vector<int> h_offsets(ngeom_, 0);
    std::vector<int> h_counts(ngeom_, 0);
    std::vector<bool> h_transparent(ngeom_, false);
    for (int i = 0; i < ngeom_; i++) {
        bool skip = (desc.geoms[i].rgba[3] <= 0.0f) || (desc.geoms[i].group >= 3);
        if (!skip) {
            h_counts[i] = verts_for_type(desc.geoms[i].type, desc.geoms[i].mesh_id, desc);
            h_transparent[i] = (desc.geoms[i].rgba[3] < 0.995f);
        }
    }

    // Opaque geoms: group-ordered layout
    int offset = 0;
    for (int g = 0; g < 3; g++) {
        group_offset_[g] = offset;
        for (int i = 0; i < ngeom_; i++) {
            if (!h_transparent[i] && desc.geoms[i].group == g && h_counts[i] > 0) {
                h_offsets[i] = offset;
                offset += h_counts[i];
            }
        }
        group_count_[g] = offset - group_offset_[g];
    }

    geom_verts_ = offset;

    // Transparent geoms: appended after all opaque groups
    transparent_offset_ = offset;
    for (int i = 0; i < ngeom_; i++) {
        if (h_transparent[i] && h_counts[i] > 0) {
            h_offsets[i] = offset;
            offset += h_counts[i];
        }
    }
    transparent_count_ = offset - transparent_offset_;

    site_vert_offset_ = offset;

    std::vector<int> h_site_offsets(nsite_);
    std::vector<int> h_site_counts(nsite_);
    for (int i = 0; i < nsite_; i++) {
        h_site_offsets[i] = offset;
        bool skip = (desc.sites[i].rgba[3] < 0.01f) || (desc.sites[i].group >= 3);
        int nverts = skip ? 0 : verts_for_site(desc.sites[i].type);
        h_site_counts[i] = nverts;
        offset += nverts;
    }

    site_verts_ = offset - site_vert_offset_;
    max_verts_ = offset;
    total_verts_ = offset;

    if (max_verts_ <= 0) return true;

    {
        const char* e = getenv("HP_GEOM_BATCH");
        int k = e ? atoi(e) : 2;
        if (k < 1) k = 1;
        if (k > 64) k = 64;
        pos_slices_ = k;
    }
    pos_slice_stride_ = (long long)max_verts_ * 3;
    cur_slice_ = 0;

    // Allocate combined vertex buffers
    hipMalloc(&d_all_positions_, (size_t)pos_slices_ * max_verts_ * 3 * sizeof(float));
    hipMalloc(&d_all_normals_,   (size_t)pos_slices_ * max_verts_ * 3 * sizeof(float));
    hipMalloc(&d_all_texcoords_, max_verts_ * TEXCOORD_COMPONENTS * sizeof(float));
    hipMalloc(&d_all_texids_,    max_verts_ * sizeof(int));
    hipMalloc(&d_all_colors_,    max_verts_ * 4 * sizeof(float));
    hipMalloc(&d_spec_colors_,   max_verts_ * 3 * sizeof(float));
    hipMalloc(&d_specular_,      max_verts_ * sizeof(float));
    hipMalloc(&d_shininess_,     max_verts_ * sizeof(float));
    hipMalloc(&d_emission_,      max_verts_ * sizeof(float));

    // Upload per-geom offset/count
    hipMalloc(&d_geom_vert_offset_, ngeom_ * sizeof(int));
    hipMalloc(&d_geom_vert_count_,  ngeom_ * sizeof(int));
    h_geom_vert_offset_ = h_offsets;
    h_geom_vert_count_  = h_counts;
    hipMemcpy(d_geom_vert_offset_, h_offsets.data(), ngeom_ * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_geom_vert_count_,  h_counts.data(),  ngeom_ * sizeof(int), hipMemcpyHostToDevice);
    if (nsite_ > 0) {
        hipMalloc(&d_site_vert_offset_, nsite_ * sizeof(int));
        hipMalloc(&d_site_vert_count_,  nsite_ * sizeof(int));
        hipMemcpy(d_site_vert_offset_, h_site_offsets.data(), nsite_ * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_site_vert_count_,  h_site_counts.data(),  nsite_ * sizeof(int), hipMemcpyHostToDevice);
    }

    // Zero the buffers
    hipMemset(d_all_positions_, 0, (size_t)pos_slices_ * max_verts_ * 3 * sizeof(float));
    hipMemset(d_all_normals_,   0, (size_t)pos_slices_ * max_verts_ * 3 * sizeof(float));
    hipMemset(d_all_texcoords_, 0, max_verts_ * TEXCOORD_COMPONENTS * sizeof(float));
    hipMemset(d_all_texids_,    0xFF, max_verts_ * sizeof(int));
    hipMemset(d_all_colors_,    0, max_verts_ * 4 * sizeof(float));
    hipMemset(d_spec_colors_,   0, max_verts_ * 3 * sizeof(float));
    hipMemset(d_specular_,      0, max_verts_ * sizeof(float));
    hipMemset(d_shininess_,     0, max_verts_ * sizeof(float));
    hipMemset(d_emission_,      0, max_verts_ * sizeof(float));

    // Build mesh vertex-to-geom mapping for parallel mesh transform
    total_mesh_verts_ = 0;
    h_geom_group_.assign(ngeom_, -1);
    h_geom_transparent_.assign(ngeom_, 0);
    h_geom_type_.assign(ngeom_, -1);
    for (int i = 0; i < ngeom_; i++) {
        h_geom_group_[i] = (h_counts[i] > 0) ? desc.geoms[i].group : -1;
        h_geom_transparent_[i] = h_transparent[i] ? 1 : 0;
        h_geom_type_[i] = desc.geoms[i].type;
    }
    std::vector<int> h_vert_to_geom;
    std::vector<int> h_vert_local_idx;
    for (int i = 0; i < ngeom_; i++) {
        if (h_counts[i] <= 0) continue;
        if (desc.geoms[i].type == GEOM_MESH && desc.geoms[i].mesh_id >= 0) {
            int mid = desc.geoms[i].mesh_id;
            int nverts = desc.meshes[mid].vert_num;
            for (int v = 0; v < nverts; v++) {
                h_vert_to_geom.push_back(i);
                h_vert_local_idx.push_back(v);
            }
            total_mesh_verts_ += nverts;
        }
    }
    h_vert_to_geom_ = h_vert_to_geom;
    h_vert_local_idx_ = h_vert_local_idx;
    if (total_mesh_verts_ > 0) {
        hipMalloc(&d_vert_to_geom_,  total_mesh_verts_ * sizeof(int));
        hipMalloc(&d_vert_local_idx_, total_mesh_verts_ * sizeof(int));
        hipMemcpy(d_vert_to_geom_,  h_vert_to_geom.data(),  total_mesh_verts_ * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_vert_local_idx_, h_vert_local_idx.data(), total_mesh_verts_ * sizeof(int), hipMemcpyHostToDevice);
    }

    prim_vert_map_n_ = max_verts_;
    if (prim_vert_map_n_ > 0) {
        std::vector<int> h_prim_vert_to_geom(prim_vert_map_n_, -1);
        total_prim_verts_ = 0;
        for (int i = 0; i < ngeom_; i++) {
            if (h_counts[i] <= 0) continue;
            // Mirror kernel_build_render_geometry's own type dispatch EXACTLY.
            // verts_for_type() falls back to BOX_VERTS for an unrecognised type,
            // but the kernel leaves nverts = 0 there and writes no materials, so
            // mapping that span would fill vertices the old code left untouched.
            const int ty = desc.geoms[i].type;
            const bool kernel_emits = (ty == GEOM_BOX || ty == GEOM_SPHERE ||
                                       ty == GEOM_CAPSULE || ty == GEOM_CYLINDER ||
                                       ty == GEOM_PLANE);
            if (!kernel_emits) continue;  // GEOM_MESH included: handled by the mesh path
            for (int v = 0; v < h_counts[i]; v++) {
                int idx = h_offsets[i] + v;
                if (idx >= 0 && idx < prim_vert_map_n_) h_prim_vert_to_geom[idx] = i;
            }
            total_prim_verts_ += h_counts[i];
        }
        hipMalloc(&d_prim_vert_to_geom_, prim_vert_map_n_ * sizeof(int));
        hipMemcpy(d_prim_vert_to_geom_, h_prim_vert_to_geom.data(),
                  prim_vert_map_n_ * sizeof(int), hipMemcpyHostToDevice);
    }

    if (max_verts_ > 0) {
        const int ntri_map = max_verts_ / 3;
        h_vert_geom_.assign(ntri_map, -1);
        for (int i = 0; i < ngeom_; i++) {
            if (h_counts[i] <= 0) continue;
            const int ty = desc.geoms[i].type;
            // MESH only: 98.9% of the drawn vertices here, and its emit is
            // exactly world_n = qrot(gquat, mesh_normal). The primitive
            // generators keep writing their (few) world normals per frame.
            if (ty != GEOM_MESH || desc.geoms[i].mesh_id < 0) continue;
            // Every span starts and ends on a triangle boundary (counts are
            // multiples of 3); anything else would already have broken the
            // first_vertex == idx*3 invariant the vertex kernel banks on.
            if ((h_offsets[i] % 3) != 0 || (h_counts[i] % 3) != 0) continue;
            for (int t = 0; t < h_counts[i] / 3; t++) {
                int idx = h_offsets[i] / 3 + t;
                if (idx >= 0 && idx < ntri_map) h_vert_geom_[idx] = i;
            }
        }
        hipMalloc(&d_vert_geom_, (size_t)ntri_map * sizeof(int));
        hipMemcpy(d_vert_geom_, h_vert_geom_.data(), (size_t)ntri_map * sizeof(int),
                  hipMemcpyHostToDevice);
    }

    if (ngeom_ > 0) hipMalloc(&d_geom_active_, ngeom_ * sizeof(int));
    if (ngeom_ > 0) {
        // Upper bound: every geom contributes at most ceil(MAX_TRIS/block)
        // tiles. 16 is the packing radix, see HIPGL_GEOM_TRI_SUBS.
        hipMalloc(&d_geom_tri_work_, (size_t)ngeom_ * 16 * sizeof(int));
    }
    if (total_mesh_verts_ > 0) {
        hipMalloc(&d_vert_to_geom_act_,   total_mesh_verts_ * sizeof(int));
        hipMalloc(&d_vert_local_idx_act_, total_mesh_verts_ * sizeof(int));
        int blk = kernels::mesh_seg_block();
        size_t cap = (size_t)(total_mesh_verts_ / blk + ngeom_ + 2);
        hipMalloc(&d_mesh_seg_, cap * 4 * sizeof(int));
        hipMalloc(&d_fused_desc_, (cap + (size_t)ngeom_ * 16) * 4 * sizeof(int));
        hipMalloc(&d_fold_desc_,  (cap + (size_t)ngeom_ * 16) * 4 * sizeof(int));
    }
    if (ngeom_ > 0) hipMalloc(&d_nofold_work_, (size_t)ngeom_ * 16 * sizeof(int));
    active_valid_ = false;
    attribs_valid_ = false;

    initialized_ = true;
    return true;
}

void PhysicsRenderer::set_active_groups(unsigned group_mask, bool transparent) {
    if (!initialized_) return;
    if (active_valid_ && group_mask == active_mask_ && transparent == active_transparent_)
        return;
    active_mask_ = group_mask;
    active_transparent_ = transparent;
    rebuild_active();
    active_valid_ = true;
    // The invariant attributes are only written for ACTIVE vertices, so a mask
    // that turns a group back on must refill them.
    attribs_valid_ = false;
}

void PhysicsRenderer::rebuild_active() {
    std::vector<int> h_act(ngeom_, 0);
    for (int i = 0; i < ngeom_; i++) {
        if (h_geom_group_[i] < 0) continue;            // not emitted at all
        bool on = ((active_mask_ >> h_geom_group_[i]) & 1u) != 0u &&
                  (!h_geom_transparent_[i] || active_transparent_);
        h_act[i] = on ? 1 : 0;
    }
    if (d_geom_active_)
        hipMemcpy(d_geom_active_, h_act.data(), ngeom_ * sizeof(int), hipMemcpyHostToDevice);

    n_geom_tri_work_ = 0;
    if (d_geom_tri_work_ && ngeom_ > 0) {
        std::vector<int> work;
        work.reserve((size_t)ngeom_ * 2);
        for (int i = 0; i < ngeom_; i++) {
            if (!h_act[i]) continue;
            if (i < (int)h_geom_vert_count_.size() && h_geom_vert_count_[i] <= 0) continue;
            int nsub = kernels::geom_tri_subblocks(
                (i < (int)h_geom_type_.size()) ? h_geom_type_[i] : -1);
            for (int sbi = 0; sbi < nsub; sbi++)
                work.push_back(kernels::geom_tri_pack(i, sbi));
        }
        n_geom_tri_work_ = (int)work.size();
        if (n_geom_tri_work_ > 0)
            hipMemcpy(d_geom_tri_work_, work.data(),
                      n_geom_tri_work_ * sizeof(int), hipMemcpyHostToDevice);
    }

    {
        static const int dbg = getenv("HP_GEOM_DBG") ? atoi(getenv("HP_GEOM_DBG")) : 0;
        static bool once = false;
        if (dbg && !once) {
            once = true;
            long long vt[8] = {0,0,0,0,0,0,0,0};
            int gc[8] = {0,0,0,0,0,0,0,0};
            for (int i = 0; i < ngeom_; i++) {
                if (!h_act[i]) continue;
                int ty = h_geom_type_[i];
                int b = (ty >= 0 && ty < 7) ? ty : 7;
                gc[b]++; vt[b] += h_geom_vert_count_[i];
            }
            fprintf(stderr, "[compo] max_verts=%d total=%d", max_verts_, total_verts_);
            for (int b = 0; b < 8; b++)
                if (gc[b]) fprintf(stderr, " ty%d:g=%d,v=%lld", b, gc[b], vt[b]);
            fprintf(stderr, "\n");
        }
    }
    total_mesh_verts_active_ = 0;
    if (total_mesh_verts_ > 0 && d_vert_to_geom_act_) {
        std::vector<int> cg; cg.reserve(total_mesh_verts_);
        std::vector<int> cl; cl.reserve(total_mesh_verts_);
        for (int v = 0; v < total_mesh_verts_; v++) {
            int g = h_vert_to_geom_[v];
            if (!h_act[g]) continue;
            cg.push_back(g);
            cl.push_back(h_vert_local_idx_[v]);
        }
        total_mesh_verts_active_ = (int)cg.size();
        if (total_mesh_verts_active_ > 0) {
            hipMemcpy(d_vert_to_geom_act_, cg.data(),
                      total_mesh_verts_active_ * sizeof(int), hipMemcpyHostToDevice);
            hipMemcpy(d_vert_local_idx_act_, cl.data(),
                      total_mesh_verts_active_ * sizeof(int), hipMemcpyHostToDevice);
        }

        n_mesh_seg_ = 0;
        if (d_mesh_seg_ && total_mesh_verts_active_ > 0) {
            const int blk = kernels::mesh_seg_block();
            std::vector<int> seg;
            seg.reserve((size_t)(total_mesh_verts_active_ / blk + ngeom_ + 2) * 4);
            int i = 0;
            bool ok = true;
            while (i < total_mesh_verts_active_) {
                int g = cg[i];
                int lb = cl[i];
                int n = 1;
                while (i + n < total_mesh_verts_active_ && n < blk &&
                       cg[i + n] == g && cl[i + n] == lb + n) n++;
                seg.push_back(g); seg.push_back(lb); seg.push_back(n); seg.push_back(0);
                i += n;
            }
            if (ok) {
                n_mesh_seg_ = (int)seg.size() / 4;
                hipMemcpy(d_mesh_seg_, seg.data(), seg.size() * sizeof(int),
                          hipMemcpyHostToDevice);
            }
            n_fused_desc_ = 0;
            if (d_fused_desc_ && n_mesh_seg_ > 0) {
                std::vector<int> fd;
                fd.reserve(seg.size() + (size_t)n_geom_tri_work_ * 4);
                for (int g = 0; g < ngeom_; g++) {
                    if (!h_act[g]) continue;
                    if (g < (int)h_geom_vert_count_.size() && h_geom_vert_count_[g] <= 0)
                        continue;
                    int nsub = kernels::geom_tri_subblocks(
                        (g < (int)h_geom_type_.size()) ? h_geom_type_[g] : -1);
                    for (int sbi = 0; sbi < nsub; sbi++) {
                        fd.push_back(g); fd.push_back(sbi); fd.push_back(0); fd.push_back(1);
                    }
                }
                fd.insert(fd.end(), seg.begin(), seg.end());
                n_fused_desc_ = (int)fd.size() / 4;
                hipMemcpy(d_fused_desc_, fd.data(), fd.size() * sizeof(int),
                          hipMemcpyHostToDevice);

                std::vector<int> foldd;
                std::vector<int> nofold;
                foldd.reserve(fd.size());
                for (int g = 0; g < ngeom_; g++) {
                    if (!h_act[g]) continue;
                    if (g < (int)h_geom_vert_count_.size() && h_geom_vert_count_[g] <= 0)
                        continue;
                    const int ty = (g < (int)h_geom_type_.size()) ? h_geom_type_[g] : -1;
                    int nsub = kernels::geom_tri_subblocks(ty);
                    const bool fold = (ty == GEOM_BOX || ty == GEOM_PLANE);
                    for (int sbi = 0; sbi < nsub; sbi++) {
                        if (fold) {
                            foldd.push_back(g); foldd.push_back(sbi);
                            foldd.push_back(0); foldd.push_back(1);
                        } else {
                            nofold.push_back(kernels::geom_tri_pack(g, sbi));
                        }
                    }
                }
                foldd.insert(foldd.end(), seg.begin(), seg.end());
                n_fold_desc_ = (int)foldd.size() / 4;
                if (n_fold_desc_ > 0 && d_fold_desc_)
                    hipMemcpy(d_fold_desc_, foldd.data(), foldd.size() * sizeof(int),
                              hipMemcpyHostToDevice);
                n_nofold_work_ = (int)nofold.size();
                if (n_nofold_work_ > 0 && d_nofold_work_)
                    hipMemcpy(d_nofold_work_, nofold.data(),
                              n_nofold_work_ * sizeof(int), hipMemcpyHostToDevice);
                model_built_ = false;
            }
        }
    }
}

void PhysicsRenderer::build_geometry(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    int env_idx,
    hipStream_t stream)
{
    if (!initialized_ || total_verts_ <= 0) return;
    if (!active_valid_) { rebuild_active(); active_valid_ = true; }

    const bool fold_ok = pose_fold_ready();
    arm_pose(data, env_idx);
    {
        static const int pose_env_on = getenv("HP_VTX_POSE") ? atoi(getenv("HP_VTX_POSE")) : 1;
        static const int dbg = getenv("HP_GEOM_DBG") ? atoi(getenv("HP_GEOM_DBG")) : 0;
        static bool printed_pose = false;
        if (dbg && !printed_pose) {
            printed_pose = true;
            if (!fold_ok)
                fprintf(stderr, "[pose] BAIL on=%d vert_geom=%p fold_desc=%d fused=%d\n",
                        pose_env_on, (void*)d_vert_geom_, n_fold_desc_, n_fused_desc_);
            else
                fprintf(stderr, "[pose] ENGAGED fold_ctas=%d nofold_tiles=%d "
                        "(was fused_ctas=%d) max_verts=%d ngeom=%d\n",
                        n_fold_desc_, n_nofold_work_, n_fused_desc_, max_verts_, ngeom_);
        }
    }

    { static const int abl = getenv("HP_ABLATE_GEOM") ? atoi(getenv("HP_ABLATE_GEOM")) : 0;
      if (abl && attribs_valid_) return; }

    const bool fill_attribs = !attribs_valid_;
    float* const pos_s = device_positions();
    float* const nrm_s = device_normals();

    {
        static const int dbg = getenv("HP_GEOM_DBG") ? atoi(getenv("HP_GEOM_DBG")) : 0;
        static bool printed = false;
        if (dbg && !printed) {
            printed = true;
            if (!d_geom_tri_work_ || n_geom_tri_work_ <= 0) {
                fprintf(stderr, "[geom] bail no-work-list (ngeom=%d)\n", ngeom_);
            } else {
                int dense = ((MAX_TRIS_HOST + 255) / 256) * ngeom_;
                fprintf(stderr,
                    "[geom] ENGAGED prim_ctas=%d/%d (skipped=%d) attr_fold=%s "
                    "mesh_segs=%d verts=%d/%d fused_ctas=%d\n",
                    n_geom_tri_work_, dense, dense - n_geom_tri_work_,
                    fill_attribs ? "off(first-build)" : "ON",
                    n_mesh_seg_, total_mesh_verts_active_, total_mesh_verts_,
                    n_fused_desc_);
            }
        }
    }

    static const int fuse = getenv("HP_GEOM_FUSE") ? atoi(getenv("HP_GEOM_FUSE")) : 1;
    if (fold_ok) {
        if (!model_built_) {
            kernels::launch_build_geometry_fused(
                ngeom_, env_idx, model, data,
                d_geom_vert_offset_, d_geom_vert_count_, d_geom_active_,
                n_fused_desc_, d_fused_desc_,
                pos_s, nrm_s, d_all_texcoords_, d_all_texids_,
                d_all_colors_, d_specular_, d_shininess_, d_emission_,
                fill_attribs, /*write_normals=*/true, stream);
            if (fill_attribs && d_prim_vert_to_geom_ && prim_vert_map_n_ > 0) {
                kernels::launch_fill_primitive_materials(
                    prim_vert_map_n_, d_prim_vert_to_geom_, model,
                    d_specular_, d_shininess_, d_emission_, stream);
            }
            // ONE extra dispatch, once per renderer: replace the just-written
            // world-space MESH positions and normals with the model-space ones
            // the vertex front end folds the pose into from here on.
            kernels::launch_build_geometry_model(
                ngeom_, model, d_geom_vert_offset_,
                n_fold_desc_, d_fold_desc_,
                pos_s, nrm_s, stream);
            model_built_ = true;
            seed_position_slices(stream);
        } else {
            const int nprim = n_fused_desc_ - n_mesh_seg_;
            if (nprim > 0)
                kernels::launch_build_geometry_fused(
                    ngeom_, env_idx, model, data,
                    d_geom_vert_offset_, d_geom_vert_count_, d_geom_active_,
                    nprim, d_fused_desc_,
                    pos_s, nrm_s, d_all_texcoords_, d_all_texids_,
                    d_all_colors_, d_specular_, d_shininess_, d_emission_,
                    /*write_attribs=*/false, /*write_normals=*/false, stream);
        }
        if (nsite_ > 0 && site_verts_ > 0 && d_site_vert_offset_) {
            kernels::launch_build_render_sites(
                nsite_, env_idx, model, data, d_site_vert_offset_,
                pos_s, nrm_s, d_all_texcoords_, d_all_texids_,
                d_all_colors_, stream);
        }
        pose_fold_ = true;
        attribs_valid_ = true;
        return;
    }
    if (fuse && n_fused_desc_ > 0 && d_fused_desc_) {
        kernels::launch_build_geometry_fused(
            ngeom_, env_idx, model, data,
            d_geom_vert_offset_, d_geom_vert_count_, d_geom_active_,
            n_fused_desc_, d_fused_desc_,
            pos_s, nrm_s, d_all_texcoords_, d_all_texids_, d_all_colors_,
            d_specular_, d_shininess_, d_emission_,
            fill_attribs, true,
            stream);
        if (fill_attribs && d_prim_vert_to_geom_ && prim_vert_map_n_ > 0) {
            kernels::launch_fill_primitive_materials(
                prim_vert_map_n_, d_prim_vert_to_geom_, model,
                d_specular_, d_shininess_, d_emission_, stream);
        }
        if (nsite_ > 0 && site_verts_ > 0 && d_site_vert_offset_) {
            kernels::launch_build_render_sites(
                nsite_, env_idx, model, data, d_site_vert_offset_,
                pos_s, nrm_s, d_all_texcoords_, d_all_texids_,
                d_all_colors_, stream);
        }
        attribs_valid_ = true;
        return;
    }

    // Primitives (box, sphere, capsule, plane) -- gated by the visibility mask.
    if (ngeom_ > 0) {
        kernels::launch_build_render_geometry(
            ngeom_, env_idx,
            model, data,
            d_geom_vert_offset_, d_geom_vert_count_, d_geom_active_,
            d_geom_tri_work_, n_geom_tri_work_,
            fill_attribs,
            pos_s, nrm_s, d_all_texcoords_, d_all_texids_, d_all_colors_,
            d_specular_, d_shininess_, d_emission_,
            nullptr,
            stream);
    }

    // Mesh geoms (parallelised across the COMPACTED, visible-only vertex list).
    if (n_mesh_seg_ > 0 && d_mesh_seg_) {
        kernels::launch_transform_mesh_seg(
            ngeom_, env_idx,
            model, data,
            d_geom_vert_offset_,
            n_mesh_seg_, d_mesh_seg_,
            pos_s, nrm_s, d_all_texcoords_, d_all_texids_, d_all_colors_,
            d_specular_, d_shininess_, d_emission_,
            fill_attribs,
            stream);
    } else if (total_mesh_verts_active_ > 0 && d_vert_to_geom_act_ && d_vert_local_idx_act_) {
        kernels::launch_transform_mesh_verts(
            ngeom_, env_idx,
            model, data,
            d_geom_vert_offset_,
            total_mesh_verts_active_,
            d_vert_to_geom_act_, d_vert_local_idx_act_,
            pos_s, nrm_s, d_all_texcoords_, d_all_texids_, d_all_colors_,
            d_specular_, d_shininess_, d_emission_,
            fill_attribs,
            stream);
    }

    if (nsite_ > 0 && site_verts_ > 0 && d_site_vert_offset_) {
        kernels::launch_build_render_sites(
            nsite_, env_idx,
            model, data,
            d_site_vert_offset_,
            pos_s, nrm_s, d_all_texcoords_, d_all_texids_, d_all_colors_,
            stream);
    }

    if (fill_attribs && d_prim_vert_to_geom_ && prim_vert_map_n_ > 0) {
        kernels::launch_fill_primitive_materials(
            prim_vert_map_n_, d_prim_vert_to_geom_, model,
            d_specular_, d_shininess_, d_emission_,
            stream);
    }

    attribs_valid_ = true;
}

void PhysicsRenderer::render(Pipeline& pipeline, const RenderState& state, hipStream_t stream) {
    if (!initialized_ || total_verts_ <= 0) return;

    pipeline.set_render_state(state);
    pipeline.draw_arrays(PRIM_TRIANGLES,
                          device_positions(), device_normals(),
                          nullptr, d_all_colors_,
                          total_verts_);
}

bool PhysicsRenderer::pose_fold_ready() const {
    static const int pose_env_on = getenv("HP_VTX_POSE") ? atoi(getenv("HP_VTX_POSE")) : 1;
    // HP_GEOM_FUSE=0 takes the two-dispatch fallback, which always writes WORLD
    // normals -- publishing the pose there would make the vertex kernel rotate
    // an already-rotated normal. The latch and the fold arm are the same gate.
    static const int fuse_env = getenv("HP_GEOM_FUSE") ? atoi(getenv("HP_GEOM_FUSE")) : 1;
    return fuse_env && pose_env_on && d_vert_geom_ && n_fold_desc_ > 0 &&
           d_fold_desc_ && n_fused_desc_ > 0 && d_fused_desc_ &&
           max_verts_ > 0;
}

void PhysicsRenderer::arm_pose(const DevicePhysicsData& data, int env_idx) {
    if (pose_fold_ready()) {
        ::hipgl::kernels::VtxPoseSrc src;
        src.vert_geom = d_vert_geom_;
        src.xpos      = data.geom_xpos;
        src.xquat     = data.geom_xquat;
        src.pos_base  = device_positions();
        src.nvert     = (long long)max_verts_;
        src.ngeom     = ngeom_;
        src.env       = env_idx;
        ::hipgl::kernels::set_vertex_pose_source(src);
    } else {
        ::hipgl::kernels::clear_vertex_pose_source();
    }
}

void PhysicsRenderer::seed_position_slices(hipStream_t stream) {
    if (pos_slices_ <= 1 || max_verts_ <= 0) return;
    const size_t bytes = (size_t)max_verts_ * 3 * sizeof(float);
    for (int k = 1; k < pos_slices_; k++) {
        hipMemcpyAsync(d_all_positions_ + (size_t)k * pos_slice_stride_,
                       d_all_positions_, bytes, hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(d_all_normals_ + (size_t)k * pos_slice_stride_,
                       d_all_normals_, bytes, hipMemcpyDeviceToDevice, stream);
    }
}

bool PhysicsRenderer::build_geometry_batch(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    int env_base, int env_stride, int nworlds,
    hipStream_t stream)
{
    if (nworlds <= 1 || nworlds > pos_slices_) return false;
    if (!initialized_ || total_verts_ <= 0) return false;
    if (!active_valid_) { rebuild_active(); active_valid_ = true; }
    if (!pose_fold_ready()) return false;
    if (!model_built_ || !attribs_valid_) return false;
    // A per-world site rebuild would also have to be sliced; it is not live in
    // this harness, but bail rather than silently drop it.
    if (nsite_ > 0 && site_verts_ > 0 && d_site_vert_offset_) return false;
    const int nprim = n_fused_desc_ - n_mesh_seg_;
    if (nprim <= 0) return false;

    kernels::launch_build_geometry_fused(
        ngeom_, env_base, model, data,
        d_geom_vert_offset_, d_geom_vert_count_, d_geom_active_,
        nprim, d_fused_desc_,
        d_all_positions_, d_all_normals_, d_all_texcoords_, d_all_texids_,
        d_all_colors_, d_specular_, d_shininess_, d_emission_,
        /*write_attribs=*/false, /*write_normals=*/false, stream,
        nworlds, env_stride, pos_slice_stride_);
    return true;
}

// build_geometry rewrites from geom_xpos/geom_xquat every frame. The other
// seven are pure model functions -- the SAME fact the attribs_valid_ latch
// above already depends on when it writes them once per renderer rather than
// once per world. Four bit-identical copies of ~58 MB put ~330 MB of attribute
// bytes against a 256 MB MALL; one shared copy takes the same working set to
// ~157 MB.
//
// Safety, explicitly:
//  * positions/normals stay PRIVATE per worker, so the per-world dependency is
//    untouched -- perturb one world's qpos and only that world's image moves.
//    This is the opposite of render-once-and-broadcast.
//  * every borrower still runs its own invariant fill into the shared arrays.
//    The stores are idempotent (identical values from identical model data), so
//    the cross-stream overlap cannot produce a value that any single owner
//    would not have produced, and each stream's own fill precedes its own read.
//  * refused unless the two renderers were built from an identical layout
//    (same max_verts_, same ngeom_) and neither carries sites -- site colours
//    ride d_all_colors_ and ARE per-world.
bool PhysicsRenderer::share_invariant_attribs_from(const PhysicsRenderer& src) {
    if (this == &src) return false;
    if (!initialized_ || !src.initialized_) return false;
    if (attribs_shared_) return true;
    if (max_verts_ <= 0 || max_verts_ != src.max_verts_) return false;
    if (ngeom_ != src.ngeom_) return false;
    if (nsite_ != 0 || src.nsite_ != 0) return false;
    if (src.attribs_shared_) return false;              // borrow only from an owner
    if (!src.d_all_texcoords_ || !src.d_all_texids_ || !src.d_all_colors_ ||
        !src.d_spec_colors_ || !src.d_specular_ || !src.d_shininess_ ||
        !src.d_emission_) return false;

    if (d_all_texcoords_) hipFree(d_all_texcoords_);
    if (d_all_texids_)    hipFree(d_all_texids_);
    if (d_all_colors_)    hipFree(d_all_colors_);
    if (d_spec_colors_)   hipFree(d_spec_colors_);
    if (d_specular_)      hipFree(d_specular_);
    if (d_shininess_)     hipFree(d_shininess_);
    if (d_emission_)      hipFree(d_emission_);

    d_all_texcoords_ = src.d_all_texcoords_;
    d_all_texids_    = src.d_all_texids_;
    d_all_colors_    = src.d_all_colors_;
    d_spec_colors_   = src.d_spec_colors_;
    d_specular_      = src.d_specular_;
    d_shininess_     = src.d_shininess_;
    d_emission_      = src.d_emission_;
    attribs_shared_  = true;
    attribs_valid_   = false;   // this instance still writes its own (identical) fill
    return true;
}

void PhysicsRenderer::free_device() {
    if (d_all_positions_) { hipFree(d_all_positions_); d_all_positions_ = nullptr; }
    if (d_all_normals_)   { hipFree(d_all_normals_);   d_all_normals_ = nullptr; }
    if (!attribs_shared_) {
        if (d_all_texcoords_) hipFree(d_all_texcoords_);
        if (d_all_texids_)    hipFree(d_all_texids_);
        if (d_all_colors_)    hipFree(d_all_colors_);
        if (d_spec_colors_)   hipFree(d_spec_colors_);
        if (d_specular_)      hipFree(d_specular_);
        if (d_shininess_)     hipFree(d_shininess_);
        if (d_emission_)      hipFree(d_emission_);
    }
    d_all_texcoords_ = nullptr; d_all_texids_ = nullptr; d_all_colors_ = nullptr;
    d_spec_colors_ = nullptr; d_specular_ = nullptr; d_shininess_ = nullptr;
    d_emission_ = nullptr; attribs_shared_ = false;
    if (d_geom_vert_offset_) { hipFree(d_geom_vert_offset_); d_geom_vert_offset_ = nullptr; }
    if (d_geom_vert_count_)  { hipFree(d_geom_vert_count_);  d_geom_vert_count_ = nullptr; }
    if (d_site_vert_offset_) { hipFree(d_site_vert_offset_); d_site_vert_offset_ = nullptr; }
    if (d_site_vert_count_)  { hipFree(d_site_vert_count_);  d_site_vert_count_ = nullptr; }
    if (d_vert_to_geom_)  { hipFree(d_vert_to_geom_);  d_vert_to_geom_ = nullptr; }
    if (d_vert_local_idx_) { hipFree(d_vert_local_idx_); d_vert_local_idx_ = nullptr; }
    if (d_prim_vert_to_geom_) { hipFree(d_prim_vert_to_geom_); d_prim_vert_to_geom_ = nullptr; }
    if (d_geom_active_) { hipFree(d_geom_active_); d_geom_active_ = nullptr; }
    if (d_geom_tri_work_) { hipFree(d_geom_tri_work_); d_geom_tri_work_ = nullptr; }
    n_geom_tri_work_ = 0; h_geom_type_.clear();
    if (d_mesh_seg_) { hipFree(d_mesh_seg_); d_mesh_seg_ = nullptr; }
    n_mesh_seg_ = 0;
    if (d_fused_desc_) { hipFree(d_fused_desc_); d_fused_desc_ = nullptr; }
    n_fused_desc_ = 0;
    if (d_fold_desc_) { hipFree(d_fold_desc_); d_fold_desc_ = nullptr; }
    n_fold_desc_ = 0;
    if (d_nofold_work_) { hipFree(d_nofold_work_); d_nofold_work_ = nullptr; }
    n_nofold_work_ = 0;
    if (d_vert_geom_) { hipFree(d_vert_geom_); d_vert_geom_ = nullptr; }
    h_vert_geom_.clear();
    pose_fold_ = false; model_built_ = false;
    if (d_vert_to_geom_act_) { hipFree(d_vert_to_geom_act_); d_vert_to_geom_act_ = nullptr; }
    if (d_vert_local_idx_act_) { hipFree(d_vert_local_idx_act_); d_vert_local_idx_act_ = nullptr; }
    total_mesh_verts_active_ = 0;
    active_valid_ = false; attribs_valid_ = false;
    h_vert_to_geom_.clear(); h_vert_local_idx_.clear();
    h_geom_group_.clear(); h_geom_transparent_.clear();
    prim_vert_map_n_ = 0; total_prim_verts_ = 0;
    total_verts_ = 0;
    geom_verts_ = 0;
    site_vert_offset_ = 0;
    site_verts_ = 0;
    transparent_offset_ = 0;
    transparent_count_ = 0;
    max_verts_ = 0;
    ngeom_ = 0;
    nsite_ = 0;
    total_mesh_verts_ = 0;
    initialized_ = false;
}

} // namespace physics
} // namespace hipgl
