#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <hip/hip_runtime.h>
#include "hipgl/types.h"            // HIPGL_DEVICE, TextureState
#include "hipgl/physics/physics_types.h"

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define RT_HD __host__ __device__
#else
#define RT_HD
#endif

namespace hipgl {
namespace physics {

constexpr int   RT_MAX_LIGHTS   = 9;          // slot 0 headlight + MAX_SCENE_LIGHTS
constexpr int   RT_TILE         = 8;          // 8x8 pixel tile per wave64
constexpr int   RT_MAX_CAND     = MAX_GEOMS;  // per-tile candidate list capacity
constexpr int   RT_STACK        = 30;         // ORDERED stack entries; builder caps SAH depth at RT_STACK - 1
#define RT_TRACE_WAVES_PER_EU 4
                                              // (stack <= depth - 1); 30 x 64 lanes x 4 B LDS + 512 B list = 8 KB/wave
constexpr int   RT_NSTATS       = 8;          // RtFrame::stats counters
constexpr int   RT_STACK_MAX_NODE_BITS = 24;  // packed LDS stack: >= 8 bits (the exponent) left for t
constexpr float RT_MINVAL       = 1e-15f;     // mjw ray.py MINVAL
constexpr float RT_MAXVAL       = 1e10f;      // mjw ray.py MAXVAL / no-hit distance
constexpr float RT_INF_PLANE    = 1e9f;       // extent used for infinite planes (mjw bvh.py)

// RT-only primitive type beyond physics GeomType (hip's physics has no
// ellipsoid geom): ellipsoid sites, and geoms MuJoCo draws as ellipsoids
// (RayTracer::set_geom_ellipsoid). size = semi-axes.
constexpr int RT_GEOM_ELLIPSOID = 6;

enum RtGeomFlags : int {
    RT_F_TEXTURED    = 1,
    RT_F_CUBE        = 2,
    RT_F_TRANSPARENT = 4,   // rgba.a < 0.995
    RT_F_MIRROR      = 8,
    RT_F_INF_PLANE   = 16,
    RT_F_MESH_UV     = 32,
    RT_F_SRC_NO_UV   = 64,  // mesh file carried no texcoords (hp baked MuJoCo texgen;
                            // MuJoCo draws it with texgen and the face-normal rule)
    RT_F_SITE        = 128, // a MuJoCo site (pad0 = site id, gid = -1): pose from
                            // site_xpos/xquat, rgba per world (RtFrame::site_rgba)
    RT_F_TEXGEN      = 256, // explicit object-linear texgen planes (RtFrame::texgen[gid])
};

enum RtCull  : int { RT_CULL_PER_TRI = 0, RT_CULL_MJW_MESH = 1, RT_CULL_NONE = 2 };
enum RtTrav  : int { RT_TRAV_SKIP = 0, RT_TRAV_ORDERED = 1 };
enum RtMode  : int { RT_MODE_HIPGL = 0, RT_MODE_MJW = 1, RT_MODE_GREY = 2 };
enum RtOut   : int { RT_OUT_RGB = 1, RT_OUT_DEPTH = 2, RT_OUT_SEG = 4 };
enum RtAA    : int { RT_AA_SSAA = 0, RT_AA_MSAA4 = 1 };

enum RtShade : int {
    RT_SH_GOURAUD     = 1,    // meshes: light the 3 corners and blend (GL_SMOOTH); else per hit
    RT_SH_VTESS       = 2,    // box/plane/cylinder: Gouraud on MuJoCo's tessellation (virtual corners)
    RT_SH_TEXGEN_MJ   = 4,    // MuJoCo texgen phases/scales (settexture) instead of the raster's tables
    RT_SH_REFLECT     = 8,    // planar mirror on the single reflective plane/box (F.mirror_li)
    RT_SH_QUAD_ALT    = 16,   // virtual quads split on the c0-c2 diagonal instead of c1-c3
    RT_SH_MJ_MESHNRM  = 32,   // mjr_uploadMesh normal rule for UV-less meshes (face normal if n.f < 0.8)
    RT_SH_LOD_PIXEL   = 64,   // texture LOD from the output-pixel footprint, not the SSAA sample's
    RT_SH_TEX_RGBA    = 128,  // textured geoms keep their rgba (GL_MODULATE of the material colour)
    RT_SH_MSAA_SAMPLE = 256,  // MSAA: shade at the first covered sample instead of the pixel centre
    RT_SH_MESA_FF     = 512,  // Mesa fixed-function quirk: ffvertex_prog.c shininess-0 specular
                              // (pow(N.L, N.L) for attenuated/spot lights; GL says 1)
    RT_SH_MESA_TEX    = 1024, // llvmpipe sampler: 8-bit AoS 2D path (8-bit weights, rounded
                              // lerps), the sRGB float path with Mesa's decode, llvmpipe's
                              // approximate log2 LOD and its double-rounded mip pyramid
    RT_SH_HW_TEX      = 2048, // hardware-GL sampler (ignored under RT_SH_MESA_TEX): float bilinear
                              // taps, float trilinear blend, exact log2 LOD (GL rho, no anisotropy)
    RT_SH_HW_MIPS     = 4096, // with RT_SH_HW_TEX: sample an RT-owned glGenerateMipmap pyramid
                              // (box filter, one nearest-even rounding per level) instead of the
                              // model's llvmpipe pyramid (double-rounded lerps, upward bias)
    RT_SH_SHADOW      = 8192, // EXPERIMENTAL (not in any preset): hard shadows. One shadow ray per
                              // scene light (directional / spot; not the headlight) from the first
                              // opaque surface; an occluded light keeps its ambient term only.
                              // Fused shading kernels only (forces shade_split off).
};
// Default: MuJoCo's own GL output. The sampler is the hardware-GL one (RT_SH_HW_TEX): the
// LIBERO policies were trained on GPU-GL frames, and against those (refs_all/stored128) it
// beats llvmpipe's 8-bit AoS sampler (RT_SH_MESA_TEX, bit-faithful to OSMesa renders) in
// every task x camera while staying below the raster against the llvmpipe renders.
constexpr int RT_SH_PRESET_MUJOCO = RT_SH_GOURAUD | RT_SH_REFLECT | RT_SH_VTESS | RT_SH_TEXGEN_MJ |
                                    RT_SH_MJ_MESHNRM | RT_SH_MESA_FF | RT_SH_HW_TEX | RT_SH_LOD_PIXEL |
                                    RT_SH_TEX_RGBA;

// ---- static, built on host at init (rebuilt on set_geomgroup) ----------------
// One entry per *render* geom ("li" = index into this list, ascending hip geom id).
struct RtGeom {
    int   gid, type, blas, texid;     // type = GeomType (physics_types.h); blas = mesh BLAS id or -1
    int   vert_adr, flags, pad0, pad1;// vert_adr = mesh_vert_adr of the mesh (soup; corner c = vert_adr + 3*tri + k)
    float size[3], vt0;               // geom_size (half extents; capsule/cyl = r, half-len)
    float llo[3], vt1;                // local-frame AABB of the geom
    float lhi[3], vt2;
    float rgba[4];
    float spec, shin, emis, refl;     // shin already *128 (GL)
    float texrep[2], vt3, pad5;
    // vt0..vt3: PLANE only, MuJoCo's plane grid (render_context.c makePlane):
    // finite axis: half cell (vt0 x, vt2 y) > 0 and edge pad (vt1 x, vt3 y);
    // infinite axis: vt0 / vt2 = -(uniform cell size); 0 = no grid.
};
static_assert(sizeof(RtGeom) == 128, "RtGeom layout");

// mjw-identical skip/escape node (pre-order). esc = global index of next node
// after this subtree, or -1. payload = global RtTri index for a leaf, -1 internal.
struct RtNode32 { float lo[3]; float hi[3]; int esc; int payload; };
static_assert(sizeof(RtNode32) == 32, "RtNode32 layout");

// Two-child node for ordered traversal, child bounds interleaved per axis:
// lo[a][k] / hi[a][k] = bound of child k on axis a, so one lane tests both
// children with packed-FP32 pairs (v_pk_add/mul_f32) and the pairs arrive in
// adjacent registers from the four dwordx4 loads. n[k] == 0: c[k] = child
// node (global idx); n[k] in 1..4: leaf, c[k] = first RtTri (global), n[k] =
// triangle count.
struct RtNode64 {
    float lo[3][2]; float hi[3][2];
    int   c[2], n[2];
    RT_HD void child_box(int k, float l[3], float h[3]) const {
        for (int a = 0; a < 3; ++a) { l[a] = lo[a][k]; h[a] = hi[a][k]; }
    }
};
static_assert(sizeof(RtNode64) == 64, "RtNode64 layout");

// Four-child node (BVH4, the collapsed SAH tree) for the ORDERED walk. Child
// bounds are SoA per axis: lo[a][k] / hi[a][k] = bound of child k on axis a,
// so the node arrives in eight dwordx4 loads and the slab tests run on
// packed-FP32 child pairs. ref[k] (global):
//   ref >= 0  : leaf, first RtTri (tri64) = ref >> 2, count = (ref & 3) + 1
//   ref == -1 : empty slot (box at the 1e30 sentinel)
//   ref <= -2 : child RtNode4 ~ref (a mesh root, never a child, is the only node that can be 0)
struct alignas(16) RtNode4 {
    float lo[3][4]; float hi[3][4];
    int   ref[4];
    int   pad[4];
    RT_HD void child_box(int k, float l[3], float h[3]) const {
        for (int a = 0; a < 3; ++a) { l[a] = lo[a][k]; h[a] = hi[a][k]; }
    }
    // (c, n) as in RtNode64: n > 0 leaf (c = first tri, n = count), n == 0 internal (c = node), c < 0 empty
    RT_HD void child(int k, int& c, int& n) const {
        const int r = ref[k];
        if (r >= 0) { c = r >> 2; n = (r & 3) + 1; }
        else if (r == -1) { c = -1; n = 0; }
        else { c = ~r; n = 0; }
    }
};
static_assert(sizeof(RtNode4) == 128, "RtNode4 layout");

// Mesh-local triangle in leaf order. v0.w = bitcast<float>(int tri) = mesh-local
// triangle index t (corners at vert_adr + 3t + k). v1.w = bitcast<float>(int)
// SKIP leaf rank of t (equal-t tie key of the ORDERED walk); v2.w unused (0).
struct RtTri { float4 v0, v1, v2; };
static_assert(sizeof(RtTri) == 48, "RtTri layout");

// Per-mesh BLAS: node_off indexes RtNode32 (SKIP) / node64_off indexes RtNode64 (ORDERED).
// Root is node_off / node64_off. tri_off indexes RtTri for the SKIP build;
// tri64_off for the SAH build (separate leaf orders). nbits = bits of a
// mesh-local RtNode64 index (>= 1, <= RT_STACK_MAX_NODE_BITS): the packed LDS
// stack of the ORDERED walker keeps the entry t in the remaining 32 - nbits.
// node4_off indexes RtNode4 (the BVH4 collapse of the SAH tree, same tri64
// leaves); a mesh has no more RtNode4 than RtNode64, so nbits also covers it.
struct RtBlas { int node_off, tri_off, node64_off, tri64_off, ntri, depth, nbits, node4_off; };
static_assert(sizeof(RtBlas) == 32, "RtBlas layout");

struct RtLight {
    float pos[3], dir[3], amb[3], dif[3], spc[3], att[3];
    float cos_cut, expo;
    int   type;        // 0 directional, 1 spot, 2 point (MuJoCo GL draws it as a directional
                       // light along the eye +Z, its parameters ignored; mjw: unbounded spot)
    int   castshadow, active, att_default;
};
static_assert(sizeof(RtLight) == 96, "RtLight layout");

// ---- per frame, written by rt_inst_kernel ------------------------------------
struct RtInst { float R[9]; float p[3]; float lo[3]; float hi[3]; int li; int pad; };  // world AABB lo/hi
static_assert(sizeof(RtInst) == 80, "RtInst layout");

// Rc columns: Rc[0..2] row 0, ... (row-major 3x3). Camera looks down -Z of Rc (MuJoCo).
struct RtCam { float eye[3]; float Rc[9]; float tanh_; float pad[3]; };
static_assert(sizeof(RtCam) == 64, "RtCam layout");

// ---- kernel arguments (by value) ----------------------------------------------
struct RtFrame {
    // static
    const RtGeom*   geom;   int nrg;
    const RtNode32* n32;    const RtNode64* n64;
    const RtTri*    tri;    const RtTri*    tri64;
    const RtBlas*   blas;
    const float*    mesh_normal;     // dm.mesh_normal  (per corner, xyz)
    const float*    mesh_texcoord;   // dm.mesh_texcoord (per corner, uv)
    TextureState    tex;
    int             nlight;          // entries of lights[] (device copy below)
    // per frame
    const RtInst*   inst;   // [nslot][nrg]
    const RtCam*    cam;    // [nslot][ncam]
    const int*      active; int nslot;   // active[s] = world index of slot s
    int ncam;
    // image
    int W, H, ss, flip_v, flip_h, out_mask, mode, cull, trav;
    int clip; float znear, zfar;
    float bg[3]; int grad; float gbot[3], gtop[3];
    float fog_s, fog_e, fog_rgb[3]; int fog_on;
    int linear; float exposure, inv_gamma;
    int mirror_li;          // -1 none
    // outputs, layout [ncam][nout][H][W](x3 for rgb); slot s writes row out_row[s]
    const int* out_row; int nout;
    uint8_t* rgb; float* depth; int* seg;
    // debug / stats (nullable)
    unsigned long long* stats;  // [RT_NSTATS]: [0]=instance-slab survivors summed over rays, [1]=nodes,
                                //      [2]=tris, [3]=rays, [4]=per-wave (tile x ray pass) candidates,
                                //      [5]=wave ray passes, [6]=tile-cull list entries per pass,
                                //      [7]=entries visited before the early break (see rt_trace.hip)
    // ---- shading (rt_shade.inc) ----
    const float* mesh_vert;  // dm.mesh_vert (soup corners, hip geom frame) for Gouraud / UV derivatives
    int   shade;             // RtShade bits
    int   aa;                // RtAA
    float lod_bias;          // added to the texture LOD (0 = raster)
    int   gl_amb;            // 1: no light is active, GL global ambient 0.3 (MuJoCo initLights)
    // per-world site rgba [site_rgba_nw][nsite][4] (null: model defaults in RtGeom)
    const float* site_rgba; int nsite, site_rgba_nw, pad_site;
    // [ngeom][3] GL_OBJECT_LINEAR planes (s, t, r) in the hip geom frame, used for
    // geoms flagged RT_F_TEXGEN under RT_SH_TEXGEN_MJ (set_texgen_planes)
    const float4* texgen;
    // [ntexture] RtTexFlags (null: none). sRGB textures take llvmpipe's float
    // path, not the 8-bit AoS one (RT_SH_MESA_TEX).
    const int* tex_flags; int ntex_flags;
    int ntransp;             // visible geoms with RT_F_TRANSPARENT (0: skip the layer pass)
    // RT-owned sRGB-encoded pyramids (RT_TEX_SRGB_DATA): bytes and
    // [ntex_flags][MAX_TEXTURE_MIP_LEVELS] levels (offsets into srgb_data)
    const unsigned char* srgb_data; const TextureMipLevel* srgb_mip;
    // RT-owned hardware-GL pyramid (RT_SH_HW_TEX): [ntexture][MAX_TEXTURE_MIP_LEVELS]
    // levels; level 0 entries index tex.tex_data (the model's level 0), levels >= 1
    // index hw_data. Null: RT_SH_HW_TEX samples the model's pyramid.
    const unsigned char* hw_data; const TextureMipLevel* hw_mip; int hw_ntex, pad_hw;
    // device copy of light[] (shading indexes lights dynamically; indexing the
    // by-value kernel argument would force the whole RtFrame into scratch)
    const RtLight* lights;
    // BVH4 nodes (RtBlas::node4_off) of the ORDERED walk
    const RtNode4* n4;
};
enum RtTexFlags : int { RT_TEX_SRGB = 1, RT_TEX_SRGB_DATA = 2 };

}  // namespace physics
}  // namespace hipgl
