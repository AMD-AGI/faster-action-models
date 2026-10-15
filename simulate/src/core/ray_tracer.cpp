// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "hipgl/physics/ray_tracer.h"
#include "hipgl/physics/rt_bvh_build.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace hipgl {
namespace physics {

namespace {

void rt_check(hipError_t err, const char* what) {
    if (err != hipSuccess)
        throw std::runtime_error(std::string("RayTracer: ") + what + ": " + hipGetErrorString(err));
}

template <typename T>
void rt_free(T*& p) {
    if (p) { (void)hipFree(p); p = nullptr; }
}

template <typename T>
void rt_upload(T*& dst, const T* src, size_t n, const char* what) {
    rt_free(dst);
    if (n == 0) return;
    rt_check(hipMalloc(&dst, n * sizeof(T)), what);
    rt_check(hipMemcpy(dst, src, n * sizeof(T), hipMemcpyHostToDevice), what);
}

// Grow-only device buffer; contents are not preserved. The capacity is
// dropped before the free and set only after the allocation succeeded, so a
// failed hipMalloc never leaves a stale capacity next to a null pointer.
template <typename T>
void rt_grow(T*& p, size_t& cap, size_t need, const char* what) {
    if (need <= cap && p) return;
    cap = 0;
    rt_free(p);
    size_t n = std::max<size_t>(need, 1);
    rt_check(hipMalloc(&p, n * sizeof(T)), what);
    cap = n;
}

// Shared variant for the outputs: the old allocation is released only when
// its last holder (e.g. a Python view) drops it. Returns true if it grew.
bool rt_grow_shared(std::shared_ptr<void>& p, size_t& cap, size_t need, size_t elem,
                    const char* what) {
    if (need <= cap && p) return false;
    cap = 0;
    p.reset();
    const size_t n = std::max<size_t>(need, 1);
    void* raw = nullptr;
    rt_check(hipMalloc(&raw, n * elem), what);
    p = std::shared_ptr<void>(raw, [](void* q) { (void)hipFree(q); });
    cap = n;
    return true;
}

// Scoped hipSetDevice (restores the caller's device).
struct RtDeviceGuard {
    int prev = -1;
    explicit RtDeviceGuard(int dev) {
        if (hipGetDevice(&prev) != hipSuccess) { prev = -1; return; }
        if (prev != dev) rt_check(hipSetDevice(dev), "set device");
        else prev = -1;
    }
    ~RtDeviceGuard() { if (prev >= 0) (void)hipSetDevice(prev); }
};

}  // namespace

RayTracer::RayTracer() = default;

RayTracer::~RayTracer() { release(); }

void RayTracer::release() {
    RtDeviceGuard dg(dev_);
    rt_free(d_geom_); rt_free(d_n32_); rt_free(d_n64_); rt_free(d_n4_); rt_free(d_tri_); rt_free(d_tri64_);
    rt_free(d_blas_);
    rt_free(d_inst_); cap_inst_ = 0;
    rt_free(d_cam_);  cap_cam_ = 0;
    rt_free(d_vlist_); cap_vlist_ = 0;
    rt_free(d_rec_); cap_rec_ = 0;
    rt_free(d_vcount_); cap_vcount_ = 0;
    if (ev_stage_) { (void)hipEventSynchronize(ev_stage_); (void)hipEventDestroy(ev_stage_); ev_stage_ = nullptr; }
    staged_ = false;
    rt_free(d_slot_); cap_slot_ = 0;
    if (h_slot_) { (void)hipHostFree(h_slot_); h_slot_ = nullptr; }
    // Outputs: drop our reference; a live Python view keeps its buffer.
    rgb_.reset();   cap_rgb_ = 0;
    depth_.reset(); cap_depth_ = 0;
    seg_.reset();   cap_seg_ = 0;
    for (auto& l : layout_) l.clear();
    rt_free(d_stats_);
    rt_free(d_site_rgba_); site_nw_ = 0; h_site_rgba_.clear(); h_site_default_.clear(); nsite_ = 0;
    rt_free(d_texgen_); h_texgen_.clear(); h_texgen_on_.clear();
    rt_free(d_tex_flags_); ntex_flags_ = 0; h_tex_flags_.clear();
    rt_free(d_srgb_data_); rt_free(d_srgb_mip_); h_srgb_data_.clear(); h_srgb_mip_.clear();
    rt_free(d_hw_data_); rt_free(d_hw_mip_); hw_src_ = nullptr; hw_ntex_ = 0; hw_gen_ = 0;
    rt_free(d_nrm_); h_nrm_.clear(); mesh_adr_.clear(); mesh_num_.clear();
    rt_free(d_lights_);
    rt_free(d_rb_in_); rt_free(d_rb_gid_); rt_free(d_rb_t_); rt_free(d_rb_n_); cap_rb_ = 0;
    rt_free(d_dbg_dirs_); cap_dbg_ = 0;
    if (ev0_) { (void)hipEventDestroy(ev0_); ev0_ = nullptr; }
    if (ev1_) { (void)hipEventDestroy(ev1_); ev1_ = nullptr; }
    last_slot_.clear();
    h_all_.clear(); h_all_group_.clear(); h_geom_.clear(); h_light_.clear(); ntransp_ = 0;
    timed_ = false; last_stats_ = false;
    initialized_ = false;
}

bool RayTracer::init(const PhysicsModelDesc& desc, const DevicePhysicsModel& dm,
                     int width, int height, int ss)
{
    if (width <= 0 || height <= 0) throw std::runtime_error("RayTracer: width/height must be positive");
    if (ss < 1) throw std::runtime_error("RayTracer: ssaa must be >= 1");
    release();
    // Run on the model's device (it need not be the caller's current one).
    {
        int dev = 0;
        rt_check(hipGetDevice(&dev), "get device");
        hipPointerAttribute_t pa{};
        if (dm.body_parent && hipPointerGetAttributes(&pa, dm.body_parent) == hipSuccess && pa.device >= 0)
            dev = pa.device;
        dev_ = dev;
    }
    RtDeviceGuard dg(dev_);
    {
        int ncu = 0;
        if (hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev_) != hipSuccess || ncu <= 0)
            ncu = 256;
        max_waves_ = 40 * ncu;   // 2 x (5 waves/SIMD x 4 SIMDs) per CU
    }
    W_ = width; H_ = height; ss_ = ss;
    ngeom_ = desc.ngeom; nbody_ = desc.nbody; ncamera_ = desc.ncamera; nmesh_ = desc.nmesh;
    znear_ = desc.map_znear * desc.stat_extent;
    zfar_  = desc.map_zfar  * desc.stat_extent;

    // ── host mesh soup ────────────────────────────────────────────────────
    // desc.mesh_vert stays alive for the model's lifetime (freed only on
    // reload/destruction) and is what was uploaded to dm.mesh_vert. Fall back
    // to one D2H copy if a model ever drops its host copy.
    const int nvert = desc.mesh_total_verts;
    std::vector<float> soup_copy;
    const float* soup = desc.mesh_vert;
    if (!soup && nvert > 0 && dm.mesh_vert) {
        soup_copy.resize((size_t)nvert * 3);
        rt_check(hipMemcpy(soup_copy.data(), dm.mesh_vert, soup_copy.size() * sizeof(float),
                           hipMemcpyDeviceToHost), "mesh_vert D2H");
        soup = soup_copy.data();
    }

    mesh_adr_.assign(nmesh_, 0); mesh_num_.assign(nmesh_, 0);
    for (int mi = 0; mi < nmesh_; ++mi) { mesh_adr_[mi] = desc.meshes[mi].vert_adr; mesh_num_[mi] = desc.meshes[mi].vert_num; }
    if (desc.mesh_normal && nvert > 0) h_nrm_.assign(desc.mesh_normal, desc.mesh_normal + (size_t)nvert * 3);
    else if (dm.mesh_normal && nvert > 0) {
        h_nrm_.resize((size_t)nvert * 3);
        rt_check(hipMemcpy(h_nrm_.data(), dm.mesh_normal, h_nrm_.size() * sizeof(float), hipMemcpyDeviceToHost),
                 "mesh_normal D2H");
    }

    // ── renderable geoms (groupvis ignored here; applied in rebuild_geoms) ──
    // Same skip rule as PhysicsRenderer::init: rgba.a <= 0 or group >= 3.
    std::vector<char> mesh_used(std::max(nmesh_, 1), 0);
    for (int i = 0; i < ngeom_; ++i) {
        const GeomDesc& gd = desc.geoms[i];
        if (gd.rgba[3] <= 0.0f || gd.group >= 3) continue;
        if (gd.type == GEOM_MESH) {
            if (gd.mesh_id < 0 || gd.mesh_id >= nmesh_ || !soup) continue;
            if (desc.meshes[gd.mesh_id].vert_num < 3) continue;
            mesh_used[gd.mesh_id] = 1;
        }
        RtGeom g;
        std::memset(&g, 0, sizeof(g));
        g.gid = i;
        g.type = gd.type;
        g.blas = (gd.type == GEOM_MESH) ? gd.mesh_id : -1;
        g.vert_adr = (gd.type == GEOM_MESH) ? desc.meshes[gd.mesh_id].vert_adr : 0;
        for (int k = 0; k < 3; ++k) g.size[k] = gd.size[k];
        for (int k = 0; k < 4; ++k) g.rgba[k] = gd.rgba[k];
        g.spec = desc.geom_specular[i];
        g.shin = desc.geom_shininess[i] * 128.0f;
        g.emis = desc.geom_emission[i];
        g.refl = desc.geom_reflectance[i];
        g.texrep[0] = desc.geom_texrepeat[i * 2 + 0];
        g.texrep[1] = desc.geom_texrepeat[i * 2 + 1];

        const int texid = desc.geom_texid[i];
        g.texid = (texid >= 0 && texid < desc.ntexture && desc.texture_data) ? texid : -1;
        int flags = 0;
        if (g.texid >= 0) {
            flags |= RT_F_TEXTURED;
            if (desc.textures[g.texid].is_cube) flags |= RT_F_CUBE;
        }
        if (gd.rgba[3] < 0.995f) flags |= RT_F_TRANSPARENT;
        if ((gd.type == GEOM_PLANE || gd.type == GEOM_BOX) && g.refl > 0.0f && gd.rgba[3] >= 0.995f)
            flags |= RT_F_MIRROR;
        if (gd.type == GEOM_MESH && desc.mesh_texcoord) flags |= RT_F_MESH_UV;
        if (gd.type == GEOM_MESH && desc.meshes[gd.mesh_id].generated_texcoords) flags |= RT_F_SRC_NO_UV;

        const float s0 = gd.size[0], s1 = gd.size[1], s2 = gd.size[2];
        switch (gd.type) {
        case GEOM_SPHERE:
            for (int k = 0; k < 3; ++k) { g.llo[k] = -s0; g.lhi[k] = s0; }
            break;
        case GEOM_BOX:
            g.llo[0] = -s0; g.llo[1] = -s1; g.llo[2] = -s2;
            g.lhi[0] =  s0; g.lhi[1] =  s1; g.lhi[2] =  s2;
            break;
        case GEOM_CAPSULE:
            g.llo[0] = -s0; g.llo[1] = -s0; g.llo[2] = -(s1 + s0);
            g.lhi[0] =  s0; g.lhi[1] =  s0; g.lhi[2] =  (s1 + s0);
            break;
        case GEOM_CYLINDER:
            g.llo[0] = -s0; g.llo[1] = -s0; g.llo[2] = -s1;
            g.lhi[0] =  s0; g.lhi[1] =  s0; g.lhi[2] =  s1;
            break;
        case GEOM_PLANE: {
            // size <= 0 on an axis means infinite along it (mjw bvh.py:39).
            const bool inf = (s0 <= 0.0f) || (s1 <= 0.0f);
            if (inf) flags |= RT_F_INF_PLANE;
            const float ex = (s0 > 0.0f) ? s0 : RT_INF_PLANE;
            const float ey = (s1 > 0.0f) ? s1 : RT_INF_PLANE;
            g.llo[0] = -ex - 0.01f; g.llo[1] = -ey - 0.01f; g.llo[2] = -0.01f;
            g.lhi[0] =  ex + 0.01f; g.lhi[1] =  ey + 0.01f; g.lhi[2] =  0.01f;
            // MuJoCo's per-plane display-list grid (render_context.c makePlane):
            // half cell sz2 = max(size[2], sz/(mjMAXPLANEGRID-2)), an even
            // number nn = floor(sz/sz2) of 2*sz2 cells, edge pads of
            // sz - nn*sz2. Used by RT_SH_VTESS; infinite planes keep vt0 = 0.
            if (inf) {
                // Infinite axis: (mjMAXPLANEGRID = 200) uniform cells of
                // d = (2.1 zfar + 2 sX) / 200 centred on 0, sX = 2 / texrepeat
                // with a material (taken as: textured), else 2.1 zfar / 198.
                // Stored negated (vt0 / vt2 < 0 = uniform grid of cell -vt).
                const double zfar = (double)desc.map_zfar * desc.stat_extent;
                for (int k = 0; k < 2; ++k) {
                    const double rep = g.texrep[k];
                    const double sX = (g.texid >= 0 && rep > 0.0) ? 2.0 / rep : 2.1 * zfar / 198.0;
                    const double d = (2.1 * zfar + 2.0 * sX) / 200.0;
                    if (k == 0) { g.vt0 = (float)-d; g.vt1 = 0.0f; } else { g.vt2 = (float)-d; g.vt3 = 0.0f; }
                }
            }
            if (!inf) {
                constexpr double kMaxPlaneGrid = 200.0;   // mjMAXPLANEGRID
                const double szs[2] = {s0, s1};
                float h[2], pad[2];
                for (int k = 0; k < 2; ++k) {
                    const double sz2 = std::max((double)s2, szs[k] / (kMaxPlaneGrid - 2.0));
                    int nn = (int)std::floor(szs[k] / std::max(1e-15, sz2));
                    nn -= nn % 2;
                    h[k] = (float)sz2;
                    pad[k] = (float)(szs[k] - nn * sz2);
                }
                g.vt0 = h[0]; g.vt1 = pad[0]; g.vt2 = h[1]; g.vt3 = pad[1];
            }
            break;
        }
        case GEOM_MESH: {
            // min/max over the hip soup in the hip geom frame (NOT mjw's
            // origin-centred cube; hip verts are not recentred).
            const auto& mi = desc.meshes[gd.mesh_id];
            float lo[3] = { 1e30f,  1e30f,  1e30f};
            float hi[3] = {-1e30f, -1e30f, -1e30f};
            for (int v = 0; v < mi.vert_num; ++v) {
                const float* p = soup + (size_t)(mi.vert_adr + v) * 3;
                for (int k = 0; k < 3; ++k) { lo[k] = std::min(lo[k], p[k]); hi[k] = std::max(hi[k], p[k]); }
            }
            for (int k = 0; k < 3; ++k) { g.llo[k] = lo[k]; g.lhi[k] = hi[k]; }
            break;
        }
        default:
            continue;   // unknown type: not renderable
        }
        g.flags = flags;
        h_all_.push_back(g);
        h_all_group_.push_back(gd.group);
    }

    // ── sites (MuJoCo mjv addSiteGeoms): lit geoms; visibility per world from
    // their alpha (rt_inst_kernel). Group g is stored as 100 + g in h_all_group_.
    nsite_ = desc.nsite;
    h_site_default_.assign((size_t)nsite_ * 4, 0.0f);
    for (int i = 0; i < nsite_; ++i) {
        const SiteDesc& sd = desc.sites[i];
        for (int k = 0; k < 4; ++k) h_site_default_[(size_t)i * 4 + k] = sd.rgba[k];
        if (sd.group >= 3) continue;
        RtGeom g;
        std::memset(&g, 0, sizeof(g));
        const float s0 = sd.size[0], s1 = sd.size[1], s2 = sd.size[2];
        switch (sd.type) {            // raw mjtGeom
        case 2: g.type = GEOM_SPHERE;
            for (int k = 0; k < 3; ++k) { g.llo[k] = -s0; g.lhi[k] = s0; }
            break;
        case 3: g.type = GEOM_CAPSULE;
            g.llo[0] = -s0; g.llo[1] = -s0; g.llo[2] = -(s1 + s0);
            g.lhi[0] =  s0; g.lhi[1] =  s0; g.lhi[2] =  (s1 + s0);
            break;
        case 5: g.type = GEOM_CYLINDER;
            g.llo[0] = -s0; g.llo[1] = -s0; g.llo[2] = -s1;
            g.lhi[0] =  s0; g.lhi[1] =  s0; g.lhi[2] =  s1;
            break;
        case 6: g.type = GEOM_BOX;
            g.llo[0] = -s0; g.llo[1] = -s1; g.llo[2] = -s2;
            g.lhi[0] =  s0; g.lhi[1] =  s1; g.lhi[2] =  s2;
            break;
        case 4: g.type = RT_GEOM_ELLIPSOID;
            g.llo[0] = -s0; g.llo[1] = -s1; g.llo[2] = -s2;
            g.lhi[0] =  s0; g.lhi[1] =  s1; g.lhi[2] =  s2;
            break;
        default: continue;            // other types: not drawn
        }
        g.gid = -1; g.blas = -1; g.texid = -1; g.pad0 = i;
        for (int k = 0; k < 3; ++k) g.size[k] = sd.size[k];
        for (int k = 0; k < 4; ++k) g.rgba[k] = sd.rgba[k];
        g.spec = 0.5f; g.shin = 0.5f * 128.0f; g.emis = 0.0f; g.refl = 0.0f;   // mjv_initGeom defaults
        g.texrep[0] = g.texrep[1] = 1.0f;
        g.flags = RT_F_SITE;
        h_all_.push_back(g);
        h_all_group_.push_back(100 + sd.group);
    }

    // ── BLAS: only meshes some renderable geom references (a never-drawn
    // collision mesh would only cost build time and memory). ───────────────
    if (nmesh_ > 0 && soup) {
        std::vector<int> vadr(nmesh_), vnum(nmesh_);
        for (int m = 0; m < nmesh_; ++m) {
            vadr[m] = desc.meshes[m].vert_adr;
            vnum[m] = mesh_used[m] ? (desc.meshes[m].vert_num / 3) * 3 : 0;
        }
        RtBlasSet set = rt_build_blas_set(soup, vadr.data(), vnum.data(), nmesh_);
        if ((int)set.blas.size() != nmesh_)
            throw std::runtime_error("RayTracer: rt_build_blas_set returned a wrong BLAS count");
        // Full structural check (containment, unions, escape links, SKIP rank,
        // SAH depth <= RT_STACK - 1 which the ORDERED walker's stack relies on).
        // Linear in the triangle count; a few ms for LIBERO.
        std::string verr;
        if (!rt_validate_blas_set(set, soup, vadr.data(), vnum.data(), nmesh_, &verr))
            throw std::runtime_error("RayTracer: BLAS validation failed: " + verr);
        for (auto& g : h_all_)
            if (g.type == GEOM_MESH && (g.blas < 0 || set.blas[g.blas].ntri <= 0)) g.blas = -1;
        ntri_ = 0;
        for (const auto& b : set.blas) ntri_ += b.ntri;
        nnode32_ = (int)set.n32.size();
        nnode64_ = (int)set.n64.size();
        rt_upload(d_n32_, set.n32.data(), set.n32.size(), "upload n32");
        rt_upload(d_n64_, set.n64.data(), set.n64.size(), "upload n64");
        nnode4_ = (int)set.n4.size();
        rt_upload(d_n4_, set.n4.data(), set.n4.size(), "upload n4");
        rt_upload(d_tri_, set.tri.data(), set.tri.size(), "upload tri");
        rt_upload(d_tri64_, set.tri64.data(), set.tri64.size(), "upload tri64");
        rt_upload(d_blas_, set.blas.data(), set.blas.size(), "upload blas");
    }

    {
        RtLight L;
        std::memset(&L, 0, sizeof(L));
        L.type = 0;
        for (int k = 0; k < 3; ++k) {
            L.amb[k] = desc.headlight_ambient[k];
            L.dif[k] = desc.headlight_diffuse[k];
            L.spc[k] = desc.headlight_specular[k];
        }
        L.att[0] = 1.0f; L.att_default = 1;
        L.cos_cut = -1.0f;
        L.active = desc.headlight_active ? 1 : 0;
        h_light_.push_back(L);
        for (int i = 0; i < desc.nscene_light && (int)h_light_.size() < RT_MAX_LIGHTS; ++i) {   // capped in upload_lights
            const SceneLightDesc& sl = desc.scene_lights[i];
            std::memset(&L, 0, sizeof(L));
            for (int k = 0; k < 3; ++k) {
                L.pos[k] = sl.pos[k]; L.dir[k] = sl.dir[k];
                L.amb[k] = sl.ambient[k]; L.dif[k] = sl.diffuse[k]; L.spc[k] = sl.specular[k];
                L.att[k] = sl.attenuation[k];
            }
            // Unit direction (the raster normalises in compute_gouraud_lighting).
            {
                const float dl = std::sqrt(L.dir[0] * L.dir[0] + L.dir[1] * L.dir[1] + L.dir[2] * L.dir[2]);
                if (dl > 1e-8f) for (int k = 0; k < 3; ++k) L.dir[k] /= dl;
            }
            L.type = sl.directional ? 0 : 1;
            L.cos_cut = sl.directional ? -1.0f : cosf(sl.cutoff * 3.14159265358979f / 180.0f);
            L.expo = sl.exponent;
            L.att_default = (sl.attenuation[0] == 1.0f && sl.attenuation[1] == 0.0f &&
                             sl.attenuation[2] == 0.0f) ? 1 : 0;
            L.castshadow = 0;
            L.active = 1;
            h_light_.push_back(L);
        }
    }

    upload_lights();

    // ── static device buffers sized for every renderable geom ──────────────
    if (!h_all_.empty())
        rt_check(hipMalloc(&d_geom_, h_all_.size() * sizeof(RtGeom)), "alloc geom");
    rt_check(hipMalloc(&d_stats_, RT_NSTATS * sizeof(unsigned long long)), "alloc stats");
    rt_check(hipEventCreate(&ev0_), "event");
    rt_check(hipEventCreate(&ev1_), "event");
    rt_check(hipEventCreateWithFlags(&ev_stage_, hipEventDisableTiming), "event");

    // Camera features neither renderer models (CameraDesc::unsupported).
    for (int c = 0; c < ncamera_; ++c) {
        const int u = desc.cameras[c].unsupported;
        if (!u) continue;
        std::fprintf(stderr,
                     "RayTracer: warning: camera %d '%s' uses unsupported MJCF features (%s%s%s%s); "
                     "it renders as a fixed perspective fovy camera at its parsed pos/quat\n",
                     c, desc.cameras[c].name,
                     (u & CAM_UNSUP_MODE) ? "mode/target " : "",
                     (u & CAM_UNSUP_INTRINSICS) ? "sensorsize/focal/principal " : "",
                     (u & CAM_UNSUP_ORTHO) ? "orthographic " : "",
                     (u & CAM_UNSUP_ORIENTATION) ? "axisangle/zaxis " : "");
    }
    geoms_dirty_ = true;
    rebuild_geoms();
    initialized_ = true;
    return true;
}

void RayTracer::set_geomgroup(int group, bool visible) {
    if (group < 0 || group >= 6) return;
    if (geomgroup_[group] != visible) { geomgroup_[group] = visible; geoms_dirty_ = true; }
}

void RayTracer::set_sitegroup(int group, bool visible) {
    if (group < 0 || group >= 6) return;
    if (sitegroup_[group] != visible) { sitegroup_[group] = visible; geoms_dirty_ = true; }
}

void RayTracer::set_site_rgba(int num_envs, const float* rgba, int env) {
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (num_envs <= 0) throw std::runtime_error("RayTracer: num_envs must be positive");
    if (env >= num_envs) throw std::runtime_error("RayTracer: env_idx out of range");
    RtDeviceGuard dg(dev_);
    const size_t per = (size_t)nsite_ * 4;
    if (num_envs != site_nw_) {
        std::vector<float> nv(per * num_envs);
        for (int w = 0; w < num_envs; ++w)
            std::memcpy(nv.data() + w * per,
                        (w < site_nw_ ? h_site_rgba_.data() + w * per : h_site_default_.data()),
                        per * sizeof(float));
        h_site_rgba_.swap(nv);
        site_nw_ = num_envs;
        rt_free(d_site_rgba_);
        if (per) rt_check(hipMalloc(&d_site_rgba_, per * num_envs * sizeof(float)), "alloc site rgba");
    }
    if (env < 0) std::memcpy(h_site_rgba_.data(), rgba, per * num_envs * sizeof(float));
    else         std::memcpy(h_site_rgba_.data() + env * per, rgba, per * sizeof(float));
    if (per)
        rt_check(hipMemcpy(d_site_rgba_, h_site_rgba_.data(), per * num_envs * sizeof(float),
                           hipMemcpyHostToDevice), "upload site rgba");
}

void RayTracer::clear_site_rgba(int env) {
    if (site_nw_ == 0) return;
    RtDeviceGuard dg(dev_);
    const size_t per = (size_t)nsite_ * 4;
    if (env < 0) {
        rt_free(d_site_rgba_); site_nw_ = 0; h_site_rgba_.clear();
        return;
    }
    if (env >= site_nw_) return;
    std::memcpy(h_site_rgba_.data() + env * per, h_site_default_.data(), per * sizeof(float));
    if (per)
        rt_check(hipMemcpy(d_site_rgba_, h_site_rgba_.data(), per * site_nw_ * sizeof(float),
                           hipMemcpyHostToDevice), "upload site rgba");
}

void RayTracer::get_site_rgba(int num_envs, float* out, int env) const {
    const size_t per = (size_t)nsite_ * 4;
    const int w0 = env < 0 ? 0 : env, w1 = env < 0 ? num_envs : env + 1;
    for (int w = w0; w < w1; ++w)
        std::memcpy(out + (w - w0) * per,
                    (w < site_nw_ ? h_site_rgba_.data() + w * per : h_site_default_.data()),
                    per * sizeof(float));
}

void RayTracer::set_texgen_planes(int gid, const float* planes12) {
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (gid < 0 || gid >= ngeom_) throw std::runtime_error("RayTracer: geom id out of range");
    RtDeviceGuard dg(dev_);
    if (h_texgen_.empty()) {
        h_texgen_.assign((size_t)ngeom_ * 12, 0.0f);
        h_texgen_on_.assign((size_t)ngeom_, 0);
        rt_check(hipMalloc(&d_texgen_, (size_t)ngeom_ * 3 * sizeof(float4)), "alloc texgen");
    }
    h_texgen_on_[gid] = planes12 ? 1 : 0;
    if (planes12) std::memcpy(h_texgen_.data() + (size_t)gid * 12, planes12, 12 * sizeof(float));
    rt_check(hipMemcpy(d_texgen_, h_texgen_.data(), h_texgen_.size() * sizeof(float), hipMemcpyHostToDevice),
             "upload texgen");
    for (auto& g : h_all_)
        if (g.gid == gid) g.flags = planes12 ? (g.flags | RT_F_TEXGEN) : (g.flags & ~RT_F_TEXGEN);
    geoms_dirty_ = true;
}

void RayTracer::set_texture_flags(const int* flags, int n) {
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    RtDeviceGuard dg(dev_);
    // keep RT_TEX_SRGB_DATA bits owned by set_texture_srgb_data
    std::vector<int> f(std::max(n, (int)h_tex_flags_.size()), 0);
    for (int i = 0; i < (int)h_tex_flags_.size(); ++i) f[i] = h_tex_flags_[i] & RT_TEX_SRGB_DATA;
    for (int i = 0; i < n && flags; ++i) f[i] |= flags[i] & ~RT_TEX_SRGB_DATA;
    h_tex_flags_.swap(f);
    rt_free(d_tex_flags_); ntex_flags_ = 0;
    if (h_tex_flags_.empty()) return;
    rt_upload(d_tex_flags_, h_tex_flags_.data(), h_tex_flags_.size(), "upload texture flags");
    ntex_flags_ = (int)h_tex_flags_.size();
}

namespace {
// Mesa gallivm lp_build_srgb_to_linear (polynomial, 8-bit input).
inline float rt_mesa_srgb_decode(float x) {
    const float c0 = 0.0023f, c1 = 0.0030f / 255.0f, c2 = 0.6935f / (255.0f * 255.0f),
                c3 = 0.3012f / (255.0f * 255.0f * 255.0f);
    if (x <= 15.0f) return x * (1.0f / (12.6f * 255.0f));
    return c0 + x * (c1 + x * (c2 + x * c3));
}
// Mesa lp_build_linear_to_srgb (rational approximation; exact sqrt stands in
// for the CPU's rsqrt estimate) + iround: 0..255.
inline int rt_mesa_srgb_encode(float s) {
    s = std::min(1.0f, std::max(0.0f, s));
    float v;
    if (s <= 0.0031308f) v = s * (12.92f * 255.0f);
    else {
        const float x05 = std::sqrt(s);
        const float x0375 = std::sqrt(std::sqrt(x05 * s));
        v = (0.675f * 1.0622f * 255.0f) * x0375 + ((0.325f * 1.0622f * 255.0f) * x05 + (-0.0620f * 255.0f));
    }
    return std::min(255, std::max(0, (int)std::nearbyint(v)));
}
}  // namespace

void RayTracer::set_texture_srgb_data(int texid, int ntexture, const unsigned char* rgb, int w, int h) {
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (texid < 0 || texid >= ntexture || w <= 0 || h <= 0 || !rgb)
        throw std::runtime_error("RayTracer: bad sRGB texture");
    RtDeviceGuard dg(dev_);
    if ((int)h_srgb_mip_.size() < ntexture * MAX_TEXTURE_MIP_LEVELS)
        h_srgb_mip_.resize((size_t)ntexture * MAX_TEXTURE_MIP_LEVELS, TextureMipLevel{0, 0, 0});
    if ((int)h_tex_flags_.size() < ntexture) h_tex_flags_.resize(ntexture, 0);
    // level 0 + llvmpipe glGenerateMipmap for SRGB8 (util_gen_mipmap blit:
    // decode, float bilinear at destination centres, clamp, encode, round).
    // Verified against OSMesa (Mesa 23.2.1): 99% of bytes exact, max 1.
    TextureMipLevel lv{(int)h_srgb_data_.size(), w, h};
    h_srgb_data_.insert(h_srgb_data_.end(), rgb, rgb + (size_t)w * h * 3);
    const size_t first = (size_t)texid * MAX_TEXTURE_MIP_LEVELS;
    h_srgb_mip_[first] = lv;
    for (int level = 1; level < MAX_TEXTURE_MIP_LEVELS; ++level) {
        const int sw = lv.width, sh = lv.height;
        if (sw == 1 && sh == 1) { h_srgb_mip_[first + level] = lv; continue; }
        const int dw = std::max(sw / 2, 1), dh = std::max(sh / 2, 1);
        TextureMipLevel nx{(int)h_srgb_data_.size(), dw, dh};
        h_srgb_data_.resize(h_srgb_data_.size() + (size_t)dw * dh * 3);
        const unsigned char* src = h_srgb_data_.data() + lv.offset;
        unsigned char* dst = h_srgb_data_.data() + nx.offset;
        for (int y = 0; y < dh; ++y) {
            const float sy = ((float)y + 0.5f) * (float)sh / (float)dh - 0.5f;
            int y0 = (int)std::floor(sy); const float fy = sy - (float)y0;
            int y1 = std::min(std::max(y0 + 1, 0), sh - 1); y0 = std::min(std::max(y0, 0), sh - 1);
            for (int x = 0; x < dw; ++x) {
                const float sx = ((float)x + 0.5f) * (float)sw / (float)dw - 0.5f;
                int x0 = (int)std::floor(sx); const float fx = sx - (float)x0;
                int x1 = std::min(std::max(x0 + 1, 0), sw - 1); x0 = std::min(std::max(x0, 0), sw - 1);
                for (int k = 0; k < 3; ++k) {
                    const float a = rt_mesa_srgb_decode(src[(y0 * sw + x0) * 3 + k]);
                    const float b = rt_mesa_srgb_decode(src[(y0 * sw + x1) * 3 + k]);
                    const float c = rt_mesa_srgb_decode(src[(y1 * sw + x0) * 3 + k]);
                    const float d = rt_mesa_srgb_decode(src[(y1 * sw + x1) * 3 + k]);
                    const float top = a + (b - a) * fx, bot = c + (d - c) * fx;
                    dst[(y * dw + x) * 3 + k] = (unsigned char)rt_mesa_srgb_encode(top + (bot - top) * fy);
                }
            }
        }
        h_srgb_mip_[first + level] = lv = nx;
    }
    h_tex_flags_[texid] |= RT_TEX_SRGB | RT_TEX_SRGB_DATA;
    rt_free(d_srgb_data_); rt_free(d_srgb_mip_);
    rt_upload(d_srgb_data_, h_srgb_data_.data(), h_srgb_data_.size(), "upload srgb texels");
    rt_upload(d_srgb_mip_, h_srgb_mip_.data(), h_srgb_mip_.size(), "upload srgb levels");
    rt_free(d_tex_flags_);
    rt_upload(d_tex_flags_, h_tex_flags_.data(), h_tex_flags_.size(), "upload texture flags");
    ntex_flags_ = (int)h_tex_flags_.size();
}

void RayTracer::ensure_hw_mips(const DevicePhysicsModel& dm, hipStream_t stream) {
    if (!dm.tex_data || dm.ntexture <= 0 || !dm.tex_width || !dm.tex_height || !dm.tex_offset) return;
    if (d_hw_mip_ && hw_src_ == dm.tex_data && hw_ntex_ == dm.ntexture && hw_gen_ == dm.tex_generation) return;
    rt_free(d_hw_data_); rt_free(d_hw_mip_); hw_src_ = nullptr; hw_ntex_ = 0; hw_gen_ = 0;
    const int nt = dm.ntexture;
    std::vector<int> tw(nt), th(nt), to(nt), tc(nt, 0);
    rt_check(hipMemcpy(tw.data(), dm.tex_width, nt * sizeof(int), hipMemcpyDeviceToHost), "hw mip: widths");
    rt_check(hipMemcpy(th.data(), dm.tex_height, nt * sizeof(int), hipMemcpyDeviceToHost), "hw mip: heights");
    rt_check(hipMemcpy(to.data(), dm.tex_offset, nt * sizeof(int), hipMemcpyDeviceToHost), "hw mip: offsets");
    if (dm.tex_iscube)
        rt_check(hipMemcpy(tc.data(), dm.tex_iscube, nt * sizeof(int), hipMemcpyDeviceToHost), "hw mip: cube");
    std::vector<TextureMipLevel> lv((size_t)nt * MAX_TEXTURE_MIP_LEVELS, TextureMipLevel{0, 0, 0});
    size_t bytes = 0;
    for (int t = 0; t < nt; ++t) {
        const int faces = tc[t] ? 6 : 1;
        int w = tw[t], h = faces > 1 ? tw[t] : th[t];
        TextureMipLevel* L = lv.data() + (size_t)t * MAX_TEXTURE_MIP_LEVELS;
        L[0] = TextureMipLevel{to[t], tw[t], th[t]};
        for (int l = 1; l < MAX_TEXTURE_MIP_LEVELS; ++l) {
            if ((w == 1 && h == 1) || w <= 0 || h <= 0) { L[l] = L[l - 1]; continue; }
            w = std::max(w / 2, 1); h = std::max(h / 2, 1);
            if (bytes + (size_t)w * h * faces * 3 > (size_t)std::numeric_limits<int>::max())
                throw std::runtime_error("RayTracer: hw mip pyramid exceeds 32-bit offsets");
            L[l] = TextureMipLevel{(int)bytes, w, h * faces};
            bytes += (size_t)w * h * faces * 3;
        }
    }
    if (bytes) rt_check(hipMalloc(&d_hw_data_, bytes), "alloc hw mips");
    for (int t = 0; t < nt; ++t) {
        const int faces = tc[t] ? 6 : 1;
        const TextureMipLevel* L = lv.data() + (size_t)t * MAX_TEXTURE_MIP_LEVELS;
        for (int l = 1; l < MAX_TEXTURE_MIP_LEVELS; ++l) {
            if (L[l].width == L[l - 1].width && L[l].height == L[l - 1].height) break;
            const unsigned char* src = (l == 1 ? dm.tex_data : d_hw_data_) + L[l - 1].offset;
            rt_launch_hw_mip(src, L[l - 1].width, L[l - 1].height / faces, faces,
                             d_hw_data_ + L[l].offset, L[l].width, L[l].height / faces, stream);
            rt_check(hipGetLastError(), "rt_hw_mip_kernel launch");
        }
    }
    rt_upload(d_hw_mip_, lv.data(), lv.size(), "upload hw mip levels");
    hw_src_ = dm.tex_data; hw_ntex_ = nt; hw_gen_ = dm.tex_generation;
}

void RayTracer::set_mesh_normals(int mesh_id, const float* nrm, int n) {
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (mesh_id < 0 || mesh_id >= nmesh_) throw std::runtime_error("RayTracer: mesh id out of range");
    if (n != mesh_num_[mesh_id] || !nrm) throw std::runtime_error("RayTracer: normal count != mesh corners");
    if (h_nrm_.empty()) throw std::runtime_error("RayTracer: model has no mesh normals");
    RtDeviceGuard dg(dev_);
    std::memcpy(h_nrm_.data() + (size_t)mesh_adr_[mesh_id] * 3, nrm, (size_t)n * 3 * sizeof(float));
    if (!d_nrm_) rt_check(hipMalloc(&d_nrm_, h_nrm_.size() * sizeof(float)), "alloc normals");
    rt_check(hipMemcpy(d_nrm_, h_nrm_.data(), h_nrm_.size() * sizeof(float), hipMemcpyHostToDevice),
             "upload normals");
}

// Device light table: MuJoCo hands Mesa the headlight (if active) and the
// active scene lights as GL_LIGHT0.. in order; Mesa has 8, so the 9th and later
// enabled lights do nothing. scn->nlight == 0 turns the 0.3 global ambient on.
void RayTracer::upload_lights() {
    std::vector<RtLight> L = h_light_;
    int nact = 0;
    for (auto& l : L) {
        if (!l.active) continue;
        if (nact >= 8) l.active = 0; else nact++;
    }
    gl_amb_ = nact == 0 ? 1 : 0;
    rt_upload(d_lights_, L.data(), L.size(), "upload lights");
}

int RayTracer::nlights_active() const {
    int n = 0;
    for (const auto& l : h_light_) n += l.active ? 1 : 0;
    return std::min(n, 8);
}

void RayTracer::set_lights(const RtLight* lights, int n) {
    if (n < 1 || n > RT_MAX_LIGHTS)
        throw std::runtime_error("RayTracer::set_lights: need 1 <= n <= RT_MAX_LIGHTS (headlight + 8)");
    RtDeviceGuard dg(dev_);
    h_light_.assign(lights, lights + n);
    upload_lights();
}

bool RayTracer::set_geom_ellipsoid(int gid, const float size[3]) {
    bool found = false;
    for (auto& g : h_all_) {
        if (g.gid != gid || (g.flags & RT_F_SITE)) continue;
        if (g.type == GEOM_MESH) return false;
        g.type = RT_GEOM_ELLIPSOID;
        for (int k = 0; k < 3; ++k) { g.size[k] = size[k]; g.llo[k] = -size[k]; g.lhi[k] = size[k]; }
        g.flags &= ~(RT_F_MIRROR | RT_F_INF_PLANE);
        g.vt0 = g.vt1 = g.vt2 = g.vt3 = 0.0f;
        found = true;
    }
    if (found) geoms_dirty_ = true;
    return found;
}

bool RayTracer::set_geom_rgba(int gid, const float rgba[4]) {
    bool found = false;
    for (auto& g : h_all_) {
        if (g.gid != gid || (g.flags & RT_F_SITE)) continue;
        for (int k = 0; k < 4; ++k) g.rgba[k] = rgba[k];
        g.flags &= ~(RT_F_TRANSPARENT | RT_F_MIRROR);
        if (rgba[3] < 0.995f) g.flags |= RT_F_TRANSPARENT;
        if ((g.type == GEOM_PLANE || g.type == GEOM_BOX) && g.refl > 0.0f && rgba[3] >= 0.995f)
            g.flags |= RT_F_MIRROR;
        found = true;
    }
    if (found) geoms_dirty_ = true;
    return found;
}

void RayTracer::rebuild_geoms() {
    if (!geoms_dirty_) return;
    RtDeviceGuard dg(dev_);
    h_geom_.clear();
    for (size_t i = 0; i < h_all_.size(); ++i) {
        const int grp = h_all_group_[i];
        if (grp >= 100) { if (grp - 100 < 6 && !sitegroup_[grp - 100]) continue; }
        else if (grp >= 0 && grp < 6 && !geomgroup_[grp]) continue;
        h_geom_.push_back(h_all_[i]);   // h_all_ is gid-ascending, so li is too
    }
    // One mirror, as MuJoCo mjr_render ("allow only one reflective geom"):
    // the first visible plane/box that is opaque with reflectance > 0.
    mirror_li_ = -1;
    for (size_t li = 0; li < h_geom_.size(); ++li)
        if (h_geom_[li].flags & RT_F_MIRROR) { mirror_li_ = (int)li; break; }
    ntransp_ = 0;
    for (const auto& g : h_geom_) if (g.flags & RT_F_TRANSPARENT) ntransp_++;
    if (!h_geom_.empty())
        rt_check(hipMemcpy(d_geom_, h_geom_.data(), h_geom_.size() * sizeof(RtGeom),
                           hipMemcpyHostToDevice), "upload geom list");
    geoms_dirty_ = false;
}

void RayTracer::ensure_slots(int nslot, int ncam, int nout, int out_mask, bool grew[3]) {
    const size_t nrg = std::max<size_t>(h_all_.size(), 1);
    rt_grow(d_inst_, cap_inst_, (size_t)nslot * nrg, "alloc inst");
    rt_grow(d_cam_, cap_cam_, (size_t)nslot * std::max(ncam, 1), "alloc cam");
    rt_grow(d_vlist_, cap_vlist_, (size_t)nslot * std::max(ncam, 1) * nrg, "alloc view lists");
    rt_grow(d_vcount_, cap_vcount_, (size_t)nslot * std::max(ncam, 1), "alloc view counts");
    if ((size_t)nslot * 2 > cap_slot_ || !d_slot_ || !h_slot_) {
        // The pending H2D may still read h_slot_.
        if (staged_) { rt_check(hipEventSynchronize(ev_stage_), "stage fence"); staged_ = false; }
        cap_slot_ = 0;
        last_slot_.clear();
        rt_free(d_slot_);
        if (h_slot_) { (void)hipHostFree(h_slot_); h_slot_ = nullptr; }
        const size_t n = std::max<size_t>((size_t)nslot * 2, 64);
        rt_check(hipMalloc(&d_slot_, n * sizeof(int)), "alloc slots");
        rt_check(hipHostMalloc((void**)&h_slot_, n * sizeof(int), hipHostMallocDefault), "alloc slots pinned");
        cap_slot_ = n;
    }
    const size_t px = (size_t)ncam * nout * W_ * H_;
    grew[0] = grew[1] = grew[2] = false;
    if (out_mask & RT_OUT_RGB)   grew[0] = rt_grow_shared(rgb_, cap_rgb_, px * 3, 1, "alloc rgb");
    if (out_mask & RT_OUT_DEPTH) grew[1] = rt_grow_shared(depth_, cap_depth_, px, sizeof(float), "alloc depth");
    if (out_mask & RT_OUT_SEG)   grew[2] = rt_grow_shared(seg_, cap_seg_, px, sizeof(int), "alloc seg");
}

// Rows that no slot of this request writes are cleared (rgb 0, depth 0,
// seg -1) unless the buffer still holds the same (nout, camera list) layout
// from an earlier call, in which case they keep that call's frame (the
// raster's "stale row" semantics). A request that writes every row skips it.
void RayTracer::prepare_outputs(const RtRequest& req, const bool grew[3], hipStream_t stream) {
    std::vector<int>& key = key_scratch_;
    key.clear();
    key.push_back(req.nout);
    key.insert(key.end(), req.cam_ids, req.cam_ids + req.ncam);
    std::vector<char>& hit = hit_scratch_;
    hit.assign((size_t)req.nout, 0);
    int covered = 0;
    for (int s = 0; s < req.nslot; ++s) {
        const int r = req.out_rows ? req.out_rows[s] : req.worlds[s];
        if (!hit[r]) { hit[r] = 1; covered++; }
    }
    const bool full = covered == req.nout;
    const size_t px = (size_t)req.ncam * req.nout * W_ * H_;
    const int bits[3] = {RT_OUT_RGB, RT_OUT_DEPTH, RT_OUT_SEG};
    void* ptr[3] = {rgb_.get(), depth_.get(), seg_.get()};
    const size_t bytes[3] = {px * 3, px * sizeof(float), px * sizeof(int)};
    const int fill[3] = {0, 0, 0xFF};   // seg: every byte 0xFF == -1
    for (int k = 0; k < 3; ++k) {
        if (!(req.out_mask & bits[k])) continue;
        if (!full && (grew[k] || layout_[k] != key))
            rt_check(hipMemsetAsync(ptr[k], fill[k], bytes[k], stream), "clear output");
        layout_[k] = key;
    }
}

void RayTracer::stage_slots(const int* tab, size_t n, hipStream_t stream) {
    if (last_slot_.size() == n && std::equal(tab, tab + n, last_slot_.begin())) return;
    if (staged_) rt_check(hipEventSynchronize(ev_stage_), "stage fence");
    std::memcpy(h_slot_, tab, n * sizeof(int));
    rt_check(hipMemcpyAsync(d_slot_, h_slot_, n * sizeof(int), hipMemcpyHostToDevice, stream), "stage slots");
    rt_check(hipEventRecord(ev_stage_, stream), "stage event");
    staged_ = true;
    last_slot_.assign(tab, tab + n);
}

RtFrame RayTracer::make_frame(const DevicePhysicsModel& dm) const {
    RtFrame F;
    std::memset(&F, 0, sizeof(F));
    F.geom = d_geom_; F.nrg = (int)h_geom_.size();
    F.n32 = d_n32_; F.n64 = d_n64_; F.n4 = d_n4_; F.tri = d_tri_; F.tri64 = d_tri64_; F.blas = d_blas_;
    F.mesh_normal = d_nrm_ ? d_nrm_ : dm.mesh_normal;
    F.mesh_texcoord = dm.mesh_texcoord;
    // Read texture pointers per frame: set_textures() may re-upload them.
    F.tex.tex_data = dm.tex_data; F.tex.tex_width = dm.tex_width; F.tex.tex_height = dm.tex_height;
    F.tex.tex_offset = dm.tex_offset; F.tex.ntexture = dm.ntexture;
    F.tex.tex_iscube = dm.tex_iscube; F.tex.tex_mip = dm.tex_mip;
    F.nlight = (int)h_light_.size();
    F.gl_amb = gl_amb_;
    F.inst = d_inst_; F.cam = d_cam_;
    F.W = W_; F.H = H_; F.ss = ss_;
    F.mode = mode_; F.cull = cull_; F.trav = trav_;
    F.clip = clip_ ? 1 : 0; F.znear = znear_; F.zfar = zfar_;
    F.bg[0] = bg_[0]; F.bg[1] = bg_[1]; F.bg[2] = bg_[2];
    F.grad = grad_ ? 1 : 0;
    for (int k = 0; k < 3; ++k) { F.gbot[k] = gbot_[k]; F.gtop[k] = gtop_[k]; F.fog_rgb[k] = fog_rgb_[k]; }
    F.fog_s = fog_s_; F.fog_e = fog_e_; F.fog_on = fog_e_ > fog_s_ ? 1 : 0;
    F.linear = linear_ ? 1 : 0; F.exposure = exposure_; F.inv_gamma = inv_gamma_;
    F.mirror_li = mirror_li_;
    F.mesh_vert = dm.mesh_vert;
    F.shade = shade_; F.aa = aa_; F.lod_bias = lod_bias_;
    F.site_rgba = d_site_rgba_; F.nsite = nsite_; F.site_rgba_nw = site_nw_;
    F.texgen = d_texgen_;
    F.tex_flags = d_tex_flags_; F.ntex_flags = ntex_flags_;
    F.srgb_data = d_srgb_data_; F.srgb_mip = d_srgb_mip_;
    if ((shade_ & RT_SH_HW_MIPS) && d_hw_mip_ && hw_src_ == dm.tex_data && hw_ntex_ == dm.ntexture &&
        hw_gen_ == dm.tex_generation) {
        F.hw_data = d_hw_data_; F.hw_mip = d_hw_mip_; F.hw_ntex = hw_ntex_;
    }
    F.lights = d_lights_;
    F.ntransp = ntransp_;
    return F;
}

void RayTracer::run_inst(const DevicePhysicsModel& dm, const DevicePhysicsData& dd,
                         const int* cam_ids, int nslot, int ncam, hipStream_t stream,
                         bool view_cull, hipEvent_t start) {
    RtSceneArgs a;
    std::memset(&a, 0, sizeof(a));
    a.geom = d_geom_; a.nrg = (int)h_geom_.size();
    a.geom_xpos = dd.geom_xpos; a.geom_xquat = dd.geom_xquat; a.ngeom = ngeom_;
    a.xpos = dd.xpos; a.xquat = dd.xquat; a.nbody = nbody_;
    a.cam_body = dm.cam_body; a.cam_pos = dm.cam_pos; a.cam_quat = dm.cam_quat; a.cam_fovy = dm.cam_fovy;
    for (int i = 0; i < ncam; ++i) a.cam_ids[i] = cam_ids[i];
    a.ncam = ncam;
    a.active = d_slot_; a.nslot = nslot;
    a.inst = d_inst_; a.cam = d_cam_;
    a.site_xpos = dd.site_xpos; a.site_xquat = dd.site_xquat; a.nsite = nsite_;
    a.site_rgba = d_site_rgba_; a.site_rgba_nw = site_nw_;
    if (view_cull && ncam > 0) {
        a.vlist = d_vlist_; a.vcount = d_vcount_;
        a.W = W_; a.H = H_; a.clip = clip_ ? 1 : 0; a.znear = znear_; a.zfar = zfar_;
    }
    if (!rt_launch_inst(a, stream, start) && start)
        rt_check(hipEventRecord(start, stream), "event");
    rt_check(hipGetLastError(), "rt_inst_kernel launch");
}

void RayTracer::render(const DevicePhysicsModel& dm, const DevicePhysicsData& dd,
                       hipStream_t stream, const RtRequest& req)
{
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    RtDeviceGuard dg(dev_);
    if (req.ncam <= 0 || req.ncam > MAX_CAMERAS) throw std::runtime_error("RayTracer: bad camera count");
    if (req.nslot <= 0 || req.nout <= 0) throw std::runtime_error("RayTracer: empty request");
    for (int i = 0; i < req.ncam; ++i)
        if (req.cam_ids[i] < 0 || req.cam_ids[i] >= ncamera_)
            throw std::runtime_error("RayTracer: invalid camera index");
    for (int s = 0; s < req.nslot; ++s) {
        if (req.worlds[s] < 0 || req.worlds[s] >= dd.num_envs)
            throw std::runtime_error("RayTracer: world index out of range");
        const int r = req.out_rows ? req.out_rows[s] : req.worlds[s];
        if (r < 0 || r >= req.nout) throw std::runtime_error("RayTracer: output row out of range");
    }
    rebuild_geoms();
    bool grew[3];
    ensure_slots(req.nslot, req.ncam, req.nout, req.out_mask, grew);
    {
        std::vector<int>& tab = tab_scratch_;
        tab.resize((size_t)req.nslot * 2);
        for (int s = 0; s < req.nslot; ++s) {
            tab[s] = req.worlds[s];
            tab[req.nslot + s] = req.out_rows ? req.out_rows[s] : req.worlds[s];
        }
        stage_slots(tab.data(), tab.size(), stream);
    }
    prepare_outputs(req, grew, stream);
    if ((shade_ & RT_SH_HW_TEX) && (shade_ & RT_SH_HW_MIPS) && !(shade_ & RT_SH_MESA_TEX) && (req.out_mask & RT_OUT_RGB))
        ensure_hw_mips(dm, stream);

    RtFrame F = make_frame(dm);
    F.active = d_slot_; F.out_row = d_slot_ + req.nslot; F.nslot = req.nslot;
    F.ncam = req.ncam; F.nout = req.nout;
    F.out_mask = req.out_mask;
    F.flip_v = req.flip_v ? 1 : 0; F.flip_h = req.flip_h ? 1 : 0;
    F.rgb = d_rgb(); F.depth = d_depth(); F.seg = d_seg();
    F.stats = stats_on_ ? d_stats_ : nullptr;
    if (stats_on_)
        rt_check(hipMemsetAsync(d_stats_, 0, RT_NSTATS * sizeof(unsigned long long), stream), "stats reset");

    // hipGetLastError() reports the last error of ANY earlier runtime call on
    // this thread (e.g. another library's failed call), not only of the launch
    // that follows. Clear it first so the launch checks below blame only our
    // own launches.
    (void)hipGetLastError();
    // K1 also writes the view-cull lists the tile-cull K2 walks (one dispatch),
    // and its dispatch records ev0_ (the start of the timed K1+K2 window).
    const bool fused_cull = tile_cull_ && d_vlist_ && d_vcount_;
    int out_mask = req.out_mask;
    run_inst(dm, dd, req.cam_ids, req.nslot, req.ncam, stream, fused_cull, ev0_);
    // K2 is a 1-D grid of tiles x slots x cameras; chunk the slots so the
    // thread count stays below 2^31.
    RtLaunchOpts opt;
    opt.stats = stats_on_; opt.tile_cull = tile_cull_; opt.swizzle = swizzle_;
    opt.max_waves = adaptive_tiles_ ? max_waves_ : 0;
    opt.vlist = d_vlist_; opt.vcount = d_vcount_; opt.vlist_ready = fused_cull;
    // MSAA4 shading passes use the ss = 2 lane layout (4x4-pixel tiles).
    const bool msaa_pass = aa_ == RT_AA_MSAA4 && mode_ != RT_MODE_GREY && (out_mask & RT_OUT_RGB);
    int max_slots = rt_primary_max_slots(W_, H_, msaa_pass ? 2 : ss_, req.ncam);
    // Deferred full shading: records for one chunk of slots (main + mirror,
    // + 4 layer kinds when transparent geoms are visible), within the cap.
    const bool shading = (out_mask & RT_OUT_RGB) && mode_ != RT_MODE_GREY;
    // RT_SH_SHADOW traces its rays in the fused kernel only (the split trace
    // pass stores no shadow records).
    if (shading && shade_split_ != 0 && ss_ <= RT_TILE && !(shade_ & RT_SH_SHADOW)) {
        const size_t per_slot = (size_t)req.ncam * W_ * H_ * (msaa_pass ? 4 : (size_t)ss_ * ss_);
        const size_t kinds = 2 + (F.ntransp > 0 ? 4 : 0);
        const size_t per_bytes = per_slot * kinds * sizeof(uint4);
        const long long cap_slots = std::max<long long>(1, (long long)(rec_cap_bytes_ / std::max<size_t>(per_bytes, 1)));
        max_slots = (int)std::min<long long>(max_slots, cap_slots);
        const size_t nrec = per_slot * (size_t)std::min(max_slots, req.nslot);
        rt_grow(d_rec_, cap_rec_, nrec * kinds, "alloc shade records");
        opt.split = true;
        opt.split_auto_waves = shade_split_ == 2 ? max_waves_ : 0;
        opt.rec = d_rec_; opt.rec_mir = d_rec_ + nrec;
        opt.rec_lay = kinds > 2 ? d_rec_ + 2 * nrec : nullptr;
        opt.nrec = nrec;
    }
    for (int s0 = 0; s0 < req.nslot; s0 += max_slots) {
        rt_launch_primary(F, s0, std::min(max_slots, req.nslot - s0), opt, stream);
        rt_check(hipGetLastError(), "rt_primary_kernel launch");
    }
    rt_check(hipEventRecord(ev1_, stream), "event");
    timed_ = true;
    last_stats_ = stats_on_;
    last_tiles_ = (long long)((W_ + RT_TILE - 1) / RT_TILE) * ((H_ + RT_TILE - 1) / RT_TILE)
                  * req.nslot * req.ncam;
}

void RayTracer::ray_batch(const DevicePhysicsModel& dm, const DevicePhysicsData& dd, hipStream_t stream,
                          int world, const float* h_orig, const float* h_dir, int n,
                          int* h_gid, float* h_t, float* h_nrm)
{
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (world < 0 || world >= dd.num_envs) throw std::runtime_error("RayTracer: world index out of range");
    if (n <= 0) return;
    RtDeviceGuard dg(dev_);
    rebuild_geoms();
    // K1 for one slot, instances only (ncam = 0).
    bool grew[3];
    ensure_slots(1, 1, 1, 0, grew);
    {
        const int want[2] = {world, 0};
        stage_slots(want, 2, stream);
    }
    run_inst(dm, dd, nullptr, 1, 0, stream);

    if ((size_t)n > cap_rb_ || !d_rb_in_ || !d_rb_gid_ || !d_rb_t_ || !d_rb_n_) {
        cap_rb_ = 0;   // set only once every buffer exists
        rt_free(d_rb_in_); rt_free(d_rb_gid_); rt_free(d_rb_t_); rt_free(d_rb_n_);
        rt_check(hipMalloc(&d_rb_in_, (size_t)n * 6 * sizeof(float)), "alloc rays");
        rt_check(hipMalloc(&d_rb_gid_, (size_t)n * sizeof(int)), "alloc rays");
        rt_check(hipMalloc(&d_rb_t_, (size_t)n * sizeof(float)), "alloc rays");
        rt_check(hipMalloc(&d_rb_n_, (size_t)n * 3 * sizeof(float)), "alloc rays");
        cap_rb_ = n;
    }
    rt_check(hipMemcpyAsync(d_rb_in_, h_orig, (size_t)n * 3 * sizeof(float), hipMemcpyHostToDevice, stream), "rays H2D");
    rt_check(hipMemcpyAsync(d_rb_in_ + (size_t)n * 3, h_dir, (size_t)n * 3 * sizeof(float), hipMemcpyHostToDevice, stream), "rays H2D");
    RtFrame F = make_frame(dm);
    F.active = d_slot_; F.nslot = 1;
    rt_launch_ray_batch(F, d_rb_in_, d_rb_in_ + (size_t)n * 3, n, d_rb_gid_, d_rb_t_, d_rb_n_, stream);
    rt_check(hipGetLastError(), "rt_ray_batch_kernel launch");
    rt_check(hipMemcpyAsync(h_gid, d_rb_gid_, (size_t)n * sizeof(int), hipMemcpyDeviceToHost, stream), "rays D2H");
    rt_check(hipMemcpyAsync(h_t, d_rb_t_, (size_t)n * sizeof(float), hipMemcpyDeviceToHost, stream), "rays D2H");
    rt_check(hipMemcpyAsync(h_nrm, d_rb_n_, (size_t)n * 3 * sizeof(float), hipMemcpyDeviceToHost, stream), "rays D2H");
    rt_check(hipStreamSynchronize(stream), "ray_batch sync");
}

void RayTracer::debug_rays(const DevicePhysicsModel& dm, const DevicePhysicsData& dd, hipStream_t stream,
                           int camera, int world, float* h_eye, float* h_Rc, float* h_dirs)
{
    if (!initialized_) throw std::runtime_error("RayTracer not initialized");
    if (camera < 0 || camera >= ncamera_) throw std::runtime_error("RayTracer: invalid camera index");
    if (world < 0 || world >= dd.num_envs) throw std::runtime_error("RayTracer: world index out of range");
    RtDeviceGuard dg(dev_);
    rebuild_geoms();
    bool grew[3];
    ensure_slots(1, 1, 1, 0, grew);
    {
        const int want[2] = {world, 0};
        stage_slots(want, 2, stream);
    }
    run_inst(dm, dd, &camera, 1, 1, stream);
    rt_grow(d_dbg_dirs_, cap_dbg_, (size_t)W_ * H_ * 3, "alloc debug dirs");
    RtFrame F = make_frame(dm);
    F.ncam = 1; F.nslot = 1;
    rt_launch_debug_rays(F, d_dbg_dirs_, stream);
    rt_check(hipGetLastError(), "rt_debug_rays_kernel launch");
    RtCam C;
    rt_check(hipMemcpyAsync(&C, d_cam_, sizeof(RtCam), hipMemcpyDeviceToHost, stream), "cam D2H");
    rt_check(hipMemcpyAsync(h_dirs, d_dbg_dirs_, (size_t)W_ * H_ * 3 * sizeof(float),
                            hipMemcpyDeviceToHost, stream), "dirs D2H");
    rt_check(hipStreamSynchronize(stream), "debug_rays sync");
    std::memcpy(h_eye, C.eye, 3 * sizeof(float));
    std::memcpy(h_Rc, C.Rc, 9 * sizeof(float));
}

RtStatsSnapshot RayTracer::stats() {
    // Report the list the next render will use (geom-group changes are applied
    // lazily by the next render otherwise).
    if (initialized_) rebuild_geoms();
    RtStatsSnapshot s;
    s.nrg = 0;
    for (const auto& g : h_geom_) { if (g.flags & RT_F_SITE) s.nsite_items++; else s.nrg++; }
    s.ntri = ntri_;
    s.nblas_nodes = nnode32_;
    s.nblas_nodes64 = nnode64_;
    if (timed_ && hipEventQuery(ev1_) == hipSuccess) {
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, ev0_, ev1_) == hipSuccess) s.last_ms = ms;
    }
    if (last_stats_ && d_stats_) {
        RtDeviceGuard dg(dev_);
        unsigned long long c[RT_NSTATS] = {};
        if (hipMemcpy(c, d_stats_, sizeof(c), hipMemcpyDeviceToHost) == hipSuccess && c[3] > 0) {
            s.cand_per_ray  = (double)c[0] / (double)c[3];
            s.cand_per_tile = c[5] > 0 ? (double)c[4] / (double)c[5] : 0.0;
            s.list_per_tile = c[5] > 0 ? (double)c[6] / (double)c[5] : 0.0;
            s.visited_per_tile = c[5] > 0 ? (double)c[7] / (double)c[5] : 0.0;
            s.nodes_per_ray = (double)c[1] / (double)c[3];
            s.tris_per_ray  = (double)c[2] / (double)c[3];
            s.counters_valid = true;
        }
    }
    return s;
}

}  // namespace physics
}  // namespace hipgl
