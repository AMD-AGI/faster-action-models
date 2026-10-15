#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions of this file are transcribed from MuJoCo 3.9.0 (engine_collision_gjk.c, engine_collision_convex.c, engine_util_blas.c: GJK/EPA convex collision),
// Copyright 2021 DeepMind Technologies Limited, licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Modified by Advanced Micro Devices, Inc., 2026:
// ported to HIP device code. See THIRD_PARTY_NOTICES.md.
// ---------------------------------------------------------------------------------------------
// physics_mjccd.h -- faithful transcription of MuJoCo 3.9.0 native convex collision
// (GJK + EPA + multi-contact polygon clipping) for BOX and MESH geoms.
//
// Source of truth (line numbers refer to these files):
//   G = mujoco/src/engine/engine_collision_gjk.c    (MuJoCo 3.9.0)
//   H = mujoco/src/engine/engine_collision_gjk.h
//   C = mujoco/src/engine/engine_collision_convex.c
//   B = mujoco/src/engine/engine_util_blas.c
//
// Scope: box/mesh/hill-climb support, mjc_initCCDObj (box+mesh), GJK
// (signedDistance, S3D/S2D/S1D, gjkIntersect), polytope2/3/4, EPA (horizon made iterative with
// the recursive visit order preserved), epaWitness, multicontact (polygonClip, box and mesh
// normals/faces), mjc_ccd (non sphere/capsule path), mjc_penetration, mjc_Convex early-return
// path, mju_makeFrame.  The margin > 0 GJK/EPA (non-discrete) path is included and verified.
// NOT ported: sphere/capsule branch of mjc_ccd, libccd (NATIVECCD disabled), the margin > 0
// perturbation loop of mjc_Convex (C:877-938), hfield/flex/sdf.  convex() returns -1 (and sets
// MJCCD_FLAG_UNSUPPORTED) when one of those paths would be needed.
//
// Verified bit-identical against libmujoco.so.3.9.0.
//
// MuJoCo names and statement order are kept so the two sources can be diffed.  Differences
// that are purely mechanical:
//   * function pointers (obj->support / obj->center) -> switch on Obj::support
//   * Face** map -> int map[] (indices into faces)
//   * horizonRec recursion -> explicit stack (Scratch::stack), same visit order
//   * all polytope / multicontact / clipping arrays live in a caller-provided Scratch
//   * mjMAX_POLYVERT-sized stack arrays are sized MJCCD_MAXPOLY (default 150 = MuJoCo);
//     if a mesh needs more than MJCCD_MAXPOLY (but <= 150) entries the count is capped and
//     Status::flags gets MJCCD_FLAG_POLYCAP (result no longer guaranteed identical).
//   * addEdge is bounds checked (24); MuJoCo writes out of bounds (never observed).  When a
//     horizon of 25..maxFaces edges would be attached, MuJoCo's result is undefined (it reads
//     indices/edges through the aliased/overrun buffer); the port stops EPA there and sets
//     MJCCD_FLAG_HORIZON_OVF | MJCCD_FLAG_HORIZON_ABORT, and convex() returns -1.
//   * mju_warning calls in epa -> Status::flags bits.
//
// Caller contract (differences from MuJoCo that are only reachable through the port's API):
//   * convex() takes the pair margin explicitly (mjc_Convex's margin argument, i.e. margin+gap
//     from the driver) and writes it into both objects, and it resets the support warm-start
//     state (vertindex = meshindex = -1) of its private copies, exactly like the two
//     mjc_initCCDObj calls at the top of mjc_Convex (C:859-860).  Obj values may therefore be
//     cached/reused across calls to convex().
//   * ccd() is mjc_ccd: it uses and MUTATES obj->vertindex/meshindex (warm start) and each
//     obj->margin as given.  To reproduce mjc_Convex through ccd() pass freshly initialized
//     objects (initCCDObj / initMeshObj / initBoxObj) with the same margin.
//   * Obj::geom must be the real, non-negative geom id (MuJoCo: g < 0 means a flex object);
//     ccd()/convex() refuse geom < 0 with MJCCD_FLAG_UNSUPPORTED.
//   * initMeshObj: graph must be nullptr when mesh_graphadr < 0 (use meshGraph()).
//   * opt.ccd_iterations must be in [1, MJCCD_MAX_ITERATIONS] (configSupported()); otherwise
//     ccd() returns dist = mjMAX_LIMIT with MJCCD_FLAG_UNSUPPORTED and convex() returns -1 (the
//     caller must fall back, not treat it as "no contact").  MuJoCo sizes its buffer from the
//     runtime value; rebuild with -DMJCCD_MAX_ITERATIONS=N for larger N.  ccd_iterations = 0
//     is undefined behaviour in MuJoCo itself (gjk reads an uninitialized x_norm).
//   * convex() returns -1 whenever the result is not guaranteed identical to mjc_Convex:
//     MJCCD_FLAG_UNSUPPORTED, MJCCD_FLAG_POLYCAP (MJCCD_MAXPOLY < the model's
//     max(mesh_polyvertnum, mesh_polymapnum)) or MJCCD_FLAG_HORIZON_ABORT.
//
// Numerics: double everywhere, float mesh vertices widened with explicit (mjtNum) casts, and
// floating-point contraction OFF inside every function of this header (see
// MJCCD_FP_CONTRACT_OFF below; enforced for g++ via optimize("fp-contract=off") and for
// clang/hipcc via a per-function "#pragma clang fp contract(off)").  libmujoco.so.3.9.0
// (clang 20, x86-64 baseline) contains no FMA instructions, so this reproduces it bit for bit.
// Do not build with -ffast-math (rejected below) or clang's -ffp-contract=fast, which ignores
// the pragma (hipcc's default -ffp-contract=fast-honor-pragmas and clang's "on" are fine).
// ---------------------------------------------------------------------------------------------

#include <float.h>
#include <math.h>

#ifndef HIPGL_DEVICE
#define HIPGL_DEVICE
#endif

#if defined(__FAST_MATH__)
#error "physics_mjccd.h reproduces MuJoCo bit for bit and needs IEEE semantics: no -ffast-math"
#endif

// Floating-point contraction (FMA) must be off to reproduce MuJoCo bit for bit.
//  * clang / hipcc (host and device): every function body starts with
//    MJCCD_FP_CONTRACT_OFF = #pragma clang fp contract(off), scoped to that body, so the
//    includer's own contraction setting is not changed (#pragma float_control(push) is not
//    supported for amdgcn).
//  * g++: optimize("fp-contract=off") for the functions in this header (push/pop_options).
#if defined(__clang__)
#define MJCCD_FP_CONTRACT_OFF _Pragma("clang fp contract(off)")
#else
#define MJCCD_FP_CONTRACT_OFF
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

// entry points (ccd, convex) are kept out of line on the device to bound register pressure
#ifndef MJCCD_NOINLINE
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#define MJCCD_NOINLINE __attribute__((noinline))
#else
#define MJCCD_NOINLINE
#endif
#endif

// max GJK/EPA iterations supported by Scratch (MuJoCo default opt.ccd_iterations = 35)
#ifndef MJCCD_MAX_ITERATIONS
#define MJCCD_MAX_ITERATIONS 35
#endif

// neighbours loaded per batch in mjc_hillclimbSupport (1 = MuJoCo's one-at-a-time loop)
#ifndef MJCCD_HILLCLIMB_BATCH
#define MJCCD_HILLCLIMB_BATCH 8
#endif

// port-only (wave_coop): the whole-wave hill climb also evaluates the neighbours of the first
// six neighbours of the current vertex, so the step after a move needs no further LDS round trip
// port-only: PK=1 support splits the box support around the mesh climb
#ifndef MJCCD_PK1_SPLIT
#define MJCCD_PK1_SPLIT 1
#endif
#ifndef MJCCD_HC_LOOKAHEAD
#define MJCCD_HC_LOOKAHEAD 1
#endif

// port-only (wave_coop): the two hill climbs of one support() call run on the two half-waves
#ifndef MJCCD_HC_PAIR
#define MJCCD_HC_PAIR 1
#endif

// map entries scanned per batch in epa's closest-face search (1 = MuJoCo's loop)
#ifndef MJCCD_EPA_SCAN_BATCH
#define MJCCD_EPA_SCAN_BATCH 8
#endif

// gjkIntersect keeps its simplex in registers (0 = MuJoCo's dynamically indexed copy)
#ifndef MJCCD_GJKINTERSECT_REGS
#define MJCCD_GJKINTERSECT_REGS 1
#endif

// LDS slots for the EPA polytope (port-only; 0 disables).  Lanes of a wave that reach EPA
// together take a slot each (in lane order) while slots last; others use the global scratch.
#ifndef MJCCD_LDS_POLY_SLOTS
#define MJCCD_LDS_POLY_SLOTS 12
#endif
#ifndef MJCCD_LDS_POLY_FACES
#define MJCCD_LDS_POLY_FACES 72
#endif
#ifndef MJCCD_LDS_POLY_VERTS
#define MJCCD_LDS_POLY_VERTS 24
#endif
// capacity for polygon vertices / polygons at a vertex (MuJoCo: mjMAX_POLYVERT = 150)
#ifndef MJCCD_MAXPOLY
#define MJCCD_MAXPOLY 150
#endif

// capacity of Status::x1/x2 in contacts (MuJoCo: mjMAXCONPAIR = 50).  May be lowered to 4 for
// device builds; ccd() then refuses max_contacts >= 5 (Status::flags |= MJCCD_FLAG_UNSUPPORTED).
#ifndef MJCCD_STATUS_NX
#define MJCCD_STATUS_NX 50
#endif

// MuJoCo's public headers define some of the names used below as macros (mjMINVAL, mjMAXVAL,
// mjMAXCONPAIR, mju_abs -> fabs, mju_sqrt -> sqrt).  Keep MuJoCo's spelling inside this header
// and restore the macros at the end so the header can be included next to <mujoco/mujoco.h>.
#pragma push_macro("mjMINVAL")
#pragma push_macro("mjMAXVAL")
#pragma push_macro("mjMAXCONPAIR")
#pragma push_macro("mju_abs")
#pragma push_macro("mju_sqrt")
#undef mjMINVAL
#undef mjMAXVAL
#undef mjMAXCONPAIR
#undef mju_abs
#undef mju_sqrt





namespace mjccd {

typedef double mjtNum;

static_assert(MJCCD_MAXPOLY >= 4, "boxFace / boxNormals need at least 4 entries");
static_assert(MJCCD_STATUS_NX >= 4, "need room for 4 contacts");

// ------------------------------------ constants ----------------------------------------------
constexpr mjtNum mjMINVAL = 1E-15;                     // mjtype.h:27
constexpr mjtNum mjMAXVAL = 1E+10;                     // mjmodel.h:25
constexpr mjtNum mjMINVAL2 = (mjMINVAL * mjMINVAL);    // G:27
constexpr mjtNum mjMAXVAL2 = (mjMAXVAL * mjMAXVAL);    // G:28
constexpr mjtNum mjMAX_LIMIT = DBL_MAX;                // H:32
constexpr mjtNum mjFACE_TOL = 0.99999872;              // H:38
constexpr mjtNum mjEDGE_TOL = 0.00159999931;           // H:41
constexpr int mjMAX_POLYVERT = 150;                    // H:44
constexpr int mjMAXCONPAIR = 50;                       // mjmodel.h:29
constexpr int mjMESH_HILLCLIMB_MIN = 10;               // engine_collision_convex.h:31

// geom types (mjtype.h mjtGeom)
constexpr int mjGEOM_HFIELD = 1;
constexpr int mjGEOM_SPHERE = 2;
constexpr int mjGEOM_CAPSULE = 3;
constexpr int mjGEOM_CYLINDER = 5;
constexpr int mjGEOM_BOX = 6;
constexpr int mjGEOM_MESH = 7;

// mjtEPAStatus (H:47-59)
enum {
  mjEPA_NOCONTACT           = -1,
  mjEPA_SUCCESS             = 0,
  mjEPA_P2_INVALID_FACES,
  mjEPA_P2_NONCONVEX,
  mjEPA_P2_ORIGIN_ON_FACE,
  mjEPA_P3_BAD_NORMAL,
  mjEPA_P3_INVALID_V4,
  mjEPA_P3_INVALID_V5,
  mjEPA_P3_MISSING_ORIGIN,
  mjEPA_P3_ORIGIN_ON_FACE,
  mjEPA_P4_MISSING_ORIGIN,
};
// port-only: polytope2/polytope4 return this instead of tail-calling polytope3, and ccdImpl
// then runs polytope3 on the same (already updated) polytope and status.  Same computations;
// only one copy of polytope3 (two support calls) is inlined instead of eleven.
constexpr int MJCCD_EPA_TO_P3 = -1000;

// port-only diagnostics in Status::flags
enum {
  MJCCD_FLAG_EPA_AFFINE_HULL = 1 << 0,  // mju_warning("EPA: origin lies on affine hull of face")
  MJCCD_FLAG_EPA_OUT_OF_MEM  = 1 << 1,  // mju_warning("EPA: out of memory for faces ...")
  MJCCD_FLAG_HORIZON_OVF     = 1 << 2,  // > 24 horizon edges (MuJoCo: out-of-bounds write)
  MJCCD_FLAG_POLYCAP         = 1 << 3,  // polygon data exceeded MJCCD_MAXPOLY (< 150)
  MJCCD_FLAG_UNSUPPORTED     = 1 << 4,  // needs an unported path / capacity (see convex())
  MJCCD_FLAG_HORIZON_ABORT   = 1 << 5,  // EPA stopped at a > 24 edge horizon (MuJoCo: UB)
};

// flags for which convex() returns -1 (result not guaranteed identical to mjc_Convex)
constexpr int MJCCD_FLAGS_FALLBACK =
    MJCCD_FLAG_UNSUPPORTED | MJCCD_FLAG_POLYCAP | MJCCD_FLAG_HORIZON_ABORT;

// support function selector (replaces obj->support function pointer)
enum { SUPPORT_NONE = 0, SUPPORT_BOX = 1, SUPPORT_MESH = 2, SUPPORT_HILLCLIMB = 3, SUPPORT_CYLINDER = 4 };

// ------------------------------------ structs ------------------------------------------------

// vertex in a polytope (H:62-68)
struct Vertex {
  mjtNum vert[3];   // v1 - v2; vertex in Minkowski sum making up polytope
  mjtNum vert1[3];  // vertex of polytope in obj1
  mjtNum vert2[3];  // vertex of polytope in obj2
  int index1;       // vertex index in mesh 1
  int index2;       // vertex index in mesh 2
};

// configuration for convex collision detection (mjCCDConfig, H:71-77); buffer -> Scratch*
struct Scratch;
struct CCDConfig {
  int max_iterations;  // the maximum number of iterations for GJK and EPA
  mjtNum tolerance;    // tolerance used by GJK and EPA
  int max_contacts;    // set to max number of contact points to recover
  mjtNum dist_cutoff;  // set to max geom distance to recover
  Scratch* buffer;     // polytope memory (NULL: EPA is skipped, like MuJoCo)
};

// data produced from running GJK and EPA (mjCCDStatus, H:80-99) + port diagnostics
struct Status {
  // geom distance information
  mjtNum dist;                     // distance between geoms
  mjtNum x1[3 * MJCCD_STATUS_NX];  // witness points for geom 1
  mjtNum x2[3 * MJCCD_STATUS_NX];  // witness points for geom 2
  int nx;                          // number of witness points

  // configurations used
  int max_iterations;
  mjtNum tolerance;
  int max_contacts;
  mjtNum dist_cutoff;

  // statistics for debugging purposes
  int gjk_iterations;
  int epa_iterations;
  int epa_status;
  Vertex simplex[4];
  int nsimplex;

  // port only
  int flags;
  int wave_coop;  // the whole wave runs this ccd together (identical inputs on every lane)
};

// model-level options used by mjc_Convex / mjc_penetration
struct Config {
  int ccd_iterations;     // m->opt.ccd_iterations (default 35)
  mjtNum ccd_tolerance;   // m->opt.ccd_tolerance (default 1e-6)
  int disable_nativeccd;  // mjDISABLED(mjDSBL_NATIVECCD)
  int disable_multiccd;   // mjDISABLED(mjDSBL_MULTICCD)
};

// port-only: one hill-climb edge slot (edge_localid[i]) with everything a hill-climb step needs
// about that neighbour pre-gathered: its vertex coordinates (verts[3*vert_globalid[sub]], the
// same float values) and sub | vert_edgeadr[sub] << 16 (sub == 0xffff: list terminator).  Built
// in LDS by stageMeshesLDS, so a hill-climb step is one LDS load level instead of four.
struct alignas(16) HcRec {
  float x, y, z;
  int packed;
};

// internal object type for convex collision detection (engine_collision_convex.h:38-88),
// box and mesh fields only
struct Obj {
  int geom;
  int geom_type;
  mjtNum size[4];
  mjtNum pos[3];
  mjtNum mat[9];
  int vertindex;
  int meshindex;
  mjtNum margin;
  int support;  // SUPPORT_* (replaces function pointer)
  int wave_coop;  // port: the whole wave evaluates this object's support together (uniform state)
  struct {
    struct {
      int nvert;
      int mesh_polynum;
      const float* vert;         // mesh_vert + 3*vertadr (geom frame)
      const int* mpolymapadr;    // mesh_polymapadr + vertadr (values index polymap)
      const int* mpolymapnum;    // mesh_polymapnum + vertadr
      const int* polymap;        // values: mesh-local polygon ids
      const int* polyvertadr;    // mesh_polyvertadr + polyadr (values index polyvert)
      const int* polyvertnum;    // mesh_polyvertnum + polyadr
      const int* polyvert;       // values: mesh-local vertex ids
      const mjtNum* polynormal;  // mesh_polynormal + 3*polyadr
      const int* graph;          // mesh_graph + graphadr, or NULL
      const HcRec* rec;          // port: LDS edge records (indexed like edge_localid), or NULL
    } mesh;
  } data;
};

// whole-model mesh arrays (mjModel field names), used by initCCDObj
struct MeshModel {
  const int* mesh_vertadr;
  const int* mesh_vertnum;
  const float* mesh_vert;
  const int* mesh_graphadr;
  const int* mesh_graph;
  const int* mesh_polynum;
  const int* mesh_polyadr;
  const mjtNum* mesh_polynormal;
  const int* mesh_polyvertadr;
  const int* mesh_polyvertnum;
  const int* mesh_polyvert;
  const int* mesh_polymapadr;
  const int* mesh_polymapnum;
  const int* mesh_polymap;
};

// one face in a polytope (G:52-58)
struct Face {
  int verts;         // indices of the three vertices of the face in the polytope
  int adj[3];        // adjacent faces, one for each edge: [v1,v2], [v2,v3], [v3,v1]
  mjtNum v[3];       // projection of the origin on face, can be used as face normal
  mjtNum dist2;      // squared norm of v; negative if deleted
  int index;         // index in map; -1: not in map, -2: deleted from polytope
};

// explicit stack frame replacing the recursion in horizonRec (G:1233)
struct HorizonFrame {
  int face;      // index of face being expanded (already deleted)
  int e;         // edge through which the face was entered
  int verts[3];  // EPA_VERT_EXPAND(face->verts), taken before deleteFace
  int k;         // next k in for (k = 1; k < 3; k++)
};

constexpr int MJCCD_HORIZON_MAX = 24;  // G:2215 (2 * 24 ints)

// all scratch memory for one mjc_ccd call (replaces config->buffer and the large stack arrays
// in multicontact/polygonClip).  ~0.2 MB with MJCCD_MAXPOLY=150, ~17 KB + 6 KB with 32.
struct Scratch {
  // polytope (G:2311-2322 layout: Vertex[5+N], Face[6N], Face*[6N], int[24], int[24])
  Vertex verts[5 + MJCCD_MAX_ITERATIONS];
  Face faces[6 * MJCCD_MAX_ITERATIONS];
  int map[6 * MJCCD_MAX_ITERATIONS];
  int horizon_indices[MJCCD_HORIZON_MAX];
  int horizon_edges[MJCCD_HORIZON_MAX];
  HorizonFrame stack[6 * MJCCD_MAX_ITERATIONS];

  // multicontact (G:2068, 2089-2090)
  mjtNum face1[MJCCD_MAXPOLY * 3], face2[MJCCD_MAXPOLY * 3], endverts[MJCCD_MAXPOLY * 3];
  mjtNum n1[3 * MJCCD_MAXPOLY], n2[3 * MJCCD_MAXPOLY];
  int idx1[MJCCD_MAXPOLY], idx2[MJCCD_MAXPOLY];

  // polygonClip (G:1574, 1581)
  mjtNum pn[3 * MJCCD_MAXPOLY], pd[MJCCD_MAXPOLY];
  mjtNum polygon1[6 * MJCCD_MAXPOLY], polygon2[6 * MJCCD_MAXPOLY];
};

// polytope used in the Expanding Polytope Algorithm (EPA) (G:66-81)
struct Polytope {
  Vertex* verts;      // list of vertices that make up the polytope
  int nverts;         // number of vertices
  Face* faces;        // list of faces that make up the polytope
  int nfaces;         // number of faces
  int maxfaces;       // max number of faces that can be stored in polytope
  int* map;           // linear map storing faces (indices into faces)
  int nmap;           // number of faces in map
  struct Horizon {    // polytope boundary edges that can be seen from w
    int* indices;     // indices of faces on horizon
    int* edges;       // corresponding edge of each face on the horizon
    int nedges;       // number of edges in horizon
    const mjtNum* w;  // point where horizon is created
  } horizon;
  HorizonFrame* stack;  // port: explicit recursion stack
  Scratch* scratch;     // port: multicontact / polygonClip arrays
  Status* status;       // port: for flags
  int lds_faces_cap;    // port: > 0 when verts/faces/map/stack live in a bounded LDS slot
  int lds_abort;        // port: set when the LDS slot could overflow; ccd() then reruns in global
};

// ------------------------------------ math helpers -------------------------------------------

HIPGL_DEVICE inline mjtNum mju_abs(mjtNum x) { return fabs(x); }
HIPGL_DEVICE inline mjtNum mju_sqrt(mjtNum x) { return sqrt(x); }

// B:115-130
HIPGL_DEVICE inline mjtNum mju_normalize3(mjtNum vec[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum norm = mju_sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

  if (norm < mjMINVAL) {
    vec[0] = 1;
    vec[1] = 0;
    vec[2] = 0;
  } else {
    mjtNum normInv = 1/norm;
    vec[0] *= normInv;
    vec[1] *= normInv;
    vec[2] *= normInv;
  }

  return norm;
}

// B:153-162
HIPGL_DEVICE inline void mju_mulMatVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum vec[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum tmp[3] = {
    mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2],
    mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2],
    mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2]
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
}

// G:110-114  v1 == v2 up to 1e-15
HIPGL_DEVICE inline int equal3(const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  return mju_abs(v1[0] - v2[0]) < mjMINVAL &&
         mju_abs(v1[1] - v2[1]) < mjMINVAL &&
         mju_abs(v1[2] - v2[2]) < mjMINVAL;
}

// G:117  res = v1 + v2
HIPGL_DEVICE inline void add3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  // port: all inputs read before the first store (element-wise, so res == v1/v2 is still fine);
  // lets the loads issue together instead of waiting on each possibly-aliasing store
  mjtNum a0 = v1[0], a1 = v1[1], a2 = v1[2], b0 = v2[0], b1 = v2[1], b2 = v2[2];
  res[0] = a0 + b0, res[1] = a1 + b1, res[2] = a2 + b2;
}

// G:122  res = v1 - v2
HIPGL_DEVICE inline void sub3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum a0 = v1[0], a1 = v1[1], a2 = v1[2], b0 = v2[0], b1 = v2[1], b2 = v2[2];
  res[0] = a0 - b0, res[1] = a1 - b1, res[2] = a2 - b2;
}

// G:127  dot product
HIPGL_DEVICE inline mjtNum dot3(const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// G:132  norm of v
HIPGL_DEVICE inline mjtNum norm3(const mjtNum v[3]) {
  MJCCD_FP_CONTRACT_OFF
  return mju_sqrt(dot3(v, v));
}

// G:137  res = v
HIPGL_DEVICE inline void copy3(mjtNum res[3], const mjtNum v[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum a0 = v[0], a1 = v[1], a2 = v[2];
  res[0] = a0, res[1] = a1, res[2] = a2;
}

// G:142  scalar product: res = s*v
HIPGL_DEVICE inline void scl3(mjtNum res[3], const mjtNum v[3], mjtNum s) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum a0 = v[0], a1 = v[1], a2 = v[2];
  res[0] = s*a0, res[1] = s*a1, res[2] = s*a2;
}

// G:147  cross product: res = v1 x v2
HIPGL_DEVICE inline void cross3(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  // port: inputs read first (callers never alias res with v1/v2)
  mjtNum a0 = v1[0], a1 = v1[1], a2 = v1[2], b0 = v2[0], b1 = v2[1], b2 = v2[2];
  res[0] = a1*b2 - a2*b1;
  res[1] = a2*b0 - a0*b2;
  res[2] = a0*b1 - a1*b0;
}

// G:154  determinant of the 3x3 matrix with columns v1, v2, v3
HIPGL_DEVICE inline mjtNum det3(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3]) {
  MJCCD_FP_CONTRACT_OFF
  // v1 * (v2 x v3)
  return v1[0]*(v2[1]*v3[2] - v2[2]*v3[1])
       + v1[1]*(v2[2]*v3[0] - v2[0]*v3[2])
       + v1[2]*(v2[0]*v3[1] - v2[1]*v3[0]);
}

// G:59-62 EPA_VERT_EXPAND
HIPGL_DEVICE inline void EPA_VERT_EXPAND(int res[3], int n) {
  MJCCD_FP_CONTRACT_OFF
  res[0] = (n) & 0x3FF;
  res[1] = (n) >> 10 & 0x3FF;
  res[2] = (n) >> 20 & 0x3FF;
}

// ------------------------------------ support functions (C) ----------------------------------

// C:174-179  transform a vector from global to local frame
HIPGL_DEVICE inline void mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  // perform matT * dir (port: computed before the first store; res never aliases mat/dir)
  mjtNum r0 = mat[0]*dir[0] + mat[3]*dir[1] + mat[6]*dir[2];
  mjtNum r1 = mat[1]*dir[0] + mat[4]*dir[1] + mat[7]*dir[2];
  mjtNum r2 = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
  res[0] = r0;
  res[1] = r1;
  res[2] = r2;
}

// C:183-192  transform a vector from local to global frame
HIPGL_DEVICE inline void localToGlobal(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3],
                                       const mjtNum pos[3]) {
  MJCCD_FP_CONTRACT_OFF
  // perform mat * dir + pos (port: computed before the first store; res never aliases inputs)
  mjtNum r0 = mat[0]*dir[0] + mat[1]*dir[1] + mat[2]*dir[2];
  mjtNum r1 = mat[3]*dir[0] + mat[4]*dir[1] + mat[5]*dir[2];
  mjtNum r2 = mat[6]*dir[0] + mat[7]*dir[1] + mat[8]*dir[2];
  r0 += pos[0];
  r1 += pos[1];
  r2 += pos[2];
  res[0] = r0;
  res[1] = r1;
  res[2] = r2;
}

#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
static __device__ long long g_stat_t[8][4096];
static __device__ int g_stat_n[8][4096];
#define MJST_T(k, t) do { g_stat_t[k][MJST_TID] += wall_clock64() - (t); g_stat_n[k][MJST_TID]++; } while (0)
#define MJST_TID (blockIdx.x * blockDim.x + threadIdx.x)
#endif
#if defined(MJC_FAN_PROF) && defined(__HIP_DEVICE_COMPILE__)
__device__ __forceinline__ long long* mjp_row(int w) { __shared__ long long s_mjp[4][24]; return s_mjp[w & 3]; }
#define MJP_ADD(k, v) do { if (__lane_id() == 0) mjp_row(threadIdx.x >> 6)[k] += (v); } while (0)
#define MJP_T() ((long long)clock64())
#else
#define MJP_ADD(k, v) do {} while (0)
#define MJP_T() 0ll
#endif
#if defined(__HIP_DEVICE_COMPILE__)
// port-only wave helpers for the cooperative hill-climb (values are exact copies, max is
// order-free on non-NaN doubles)
template <int CTRL>
__device__ __forceinline__ double wave_dpp_max(double x) {
  const long long b = __double_as_longlong(x);
  const int lo = __builtin_amdgcn_update_dpp(0, (int)(b & 0xffffffffll), CTRL, 0xf, 0xf, false);
  const int hi = __builtin_amdgcn_update_dpp(0, (int)(b >> 32), CTRL, 0xf, 0xf, false);
  const double o = __longlong_as_double(((long long)hi << 32) | (long long)(unsigned)lo);
  return fmax(x, o);
}
// select form of wave_dpp_max for non-NaN values (no canonicalize; a +-0 tie may keep either
// zero, which no caller can observe: they only compare the maximum with '>' and '==')
template <int CTRL>
__device__ __forceinline__ double wave_dpp_maxs(double x) {
  const long long b = __double_as_longlong(x);
  const int lo = __builtin_amdgcn_update_dpp(0, (int)(b & 0xffffffffll), CTRL, 0xf, 0xf, false);
  const int hi = __builtin_amdgcn_update_dpp(0, (int)(b >> 32), CTRL, 0xf, 0xf, false);
  const double o = __longlong_as_double(((long long)hi << 32) | (long long)(unsigned)lo);
  return o > x ? o : x;
}
template <int CTRL>
__device__ __forceinline__ double wave_dpp_min(double x) {
  const long long b = __double_as_longlong(x);
  const int lo = __builtin_amdgcn_update_dpp(0, (int)(b & 0xffffffffll), CTRL, 0xf, 0xf, false);
  const int hi = __builtin_amdgcn_update_dpp(0, (int)(b >> 32), CTRL, 0xf, 0xf, false);
  const double o = __longlong_as_double(((long long)hi << 32) | (long long)(unsigned)lo);
  return fmin(x, o);
}
__device__ __forceinline__ double wave_shfl_xor_f64(double x, int m) {
  const long long b = __double_as_longlong(x);
  const int lo = __shfl_xor((int)(b & 0xffffffffll), m);
  const int hi = __shfl_xor((int)(b >> 32), m);
  return __longlong_as_double(((long long)hi << 32) | (long long)(unsigned)lo);
}
__device__ __forceinline__ double wave_readfirstlane_f64(double x) {
  const long long b = __double_as_longlong(x);
  const int lo = __builtin_amdgcn_readfirstlane((int)(b & 0xffffffffll));
  const int hi = __builtin_amdgcn_readfirstlane((int)(b >> 32));
  return __longlong_as_double(((long long)hi << 32) | (long long)(unsigned)lo);
}
#endif

// C:317-339  box support function
template <typename OP>
HIPGL_DEVICE __forceinline__ void mjc_boxSupportBody(mjtNum res[3], OP obj, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  // box data (port: copied to registers first; same values)
  mjtNum mat[9], pos[3], size[3];
  for (int k = 0; k < 9; k++) mat[k] = obj->mat[k];
  for (int k = 0; k < 3; k++) { pos[k] = obj->pos[k]; size[k] = obj->size[k]; }

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point in local frame
  local_supp[0] = local_dir[0] >= 0 ? size[0] : -size[0];
  local_supp[1] = local_dir[1] >= 0 ? size[1] : -size[1];
  local_supp[2] = local_dir[2] >= 0 ? size[2] : -size[2];

  // mark the index of the corner of the box for fast lookup
  int vi = (local_supp[0] > 0) ? 1 : 0;
  vi |= (local_supp[1] > 0) ? 2 : 0;
  vi |= (local_supp[2] > 0) ? 4 : 0;
  obj->vertindex = vi;

  // transform support point to global frame
  localToGlobal(res, mat, local_supp, pos);
}

#if defined(__HIP_DEVICE_COMPILE__)
typedef __attribute__((address_space(3))) Obj LdsObj;
#endif
HIPGL_DEVICE inline void mjc_boxSupport(mjtNum res[3], Obj* obj, const mjtNum dir[3]) {
#if defined(__HIP_DEVICE_COMPILE__)
  // port-only: an Obj in LDS is read with ds_* instead of flat instructions (same values)
  if (__builtin_amdgcn_is_shared((const void*)obj)) {
    mjc_boxSupportBody(res, (LdsObj*)obj, dir);
    return;
  }
#endif
  mjc_boxSupportBody(res, obj, dir);
}

// C:283-311  cylinder support function (radius size[0], half-length size[1] along local z)
HIPGL_DEVICE inline void mjc_cylinderSupport(mjtNum res[3], Obj* obj, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], tmp[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum n = local_dir[0]*local_dir[0] + local_dir[1]*local_dir[1];
  if (n > mjMINVAL*mjMINVAL) {
    n = size[0] / sqrt(n);
    tmp[0] = local_dir[0] * n;
    tmp[1] = local_dir[1] * n;
  } else {
    tmp[0] = tmp[1] = 0;
  }

  // set result in Z direction (mju_sign: +1 for > 0, -1 for < 0, 0 for 0)
  tmp[2] = (local_dir[2] > 0 ? 1 : (local_dir[2] < 0 ? -1 : 0)) * size[1];

  // transform result to global frame
  localToGlobal(res, mat, tmp, pos);
}

// C:343-345  dot product between mjtNum and float
HIPGL_DEVICE inline mjtNum dot3f(const mjtNum a[3], const float b[3]) {
  MJCCD_FP_CONTRACT_OFF
  return a[0]*(mjtNum)b[0] + a[1]*(mjtNum)b[1] + a[2]*(mjtNum)b[2];
}

// C:349-387  mesh support function via exhaustive search
HIPGL_DEVICE inline void mjc_meshSupport(mjtNum res[3], Obj* obj, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* verts = obj->data.mesh.vert;
  int nverts = obj->data.mesh.nvert;

  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum max = -FLT_MAX;
  int imax = 0;

  if (obj->vertindex >= 0) {
    imax = obj->vertindex;
    max = dot3f(local_dir, verts + 3*imax);
  }

  // search all vertices, find maximum dot product
  for (int i=0; i < nverts; i++) {
    mjtNum vdot = dot3f(local_dir, verts + 3*i);

    // update max
    if (vdot > max) {
      max = vdot;
      imax = i;
    }
  }

  // record vertex index of maximum
  obj->vertindex = imax;

  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}

// C:391-432  mesh support function via hill climbing
// COOP (port-only): the caller guarantees obj->wave_coop (the single-lane batched climb is
// then dead code and dropped)
template <bool COOP = false>
HIPGL_DEVICE inline void mjc_hillclimbSupport(mjtNum res[3], Obj* obj, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  int numvert = obj->data.mesh.graph[0];
  const int* vert_edgeadr = obj->data.mesh.graph + 2;
  const int* vert_globalid = obj->data.mesh.graph  + 2 + numvert;
  const int* edge_localid = obj->data.mesh.graph + 2 + 2*numvert;
  const float* verts = obj->data.mesh.vert;
  const mjtNum* pos = obj->pos;
  const mjtNum* mat = obj->mat;

  // rotate dir to geom local frame
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum max = -FLT_MAX;
  int prev = -1;
  int imax = obj->meshindex >= 0 ? obj->meshindex : 0;

#if MJCCD_HILLCLIMB_BATCH > 1
  // the argmax tree below equals the sequential scan when no dot product is NaN.  With
  // |local_dir| < 1e250 and float coordinates every product and sum is finite, so that holds;
  // otherwise (never in practice) take MuJoCo's loop
  const HcRec* rec = obj->data.mesh.rec;
  if (rec && fabs(local_dir[0]) < 1e250 && fabs(local_dir[1]) < 1e250 &&
      fabs(local_dir[2]) < 1e250) {
    // port: each neighbour is one 16-byte LDS record holding its coordinates and its own edge
    // address, so a step is one load level.  MuJoCo's sequential `if (vdot > max)` scan over a
    // batch is evaluated as: all dot products (same expression, same float operands), then a
    // first-index argmax tree (b replaces a only if strictly greater), then one strict '>'
    // test against the running max.  That selects exactly the element the sequential scan
    // ends on (the first occurrence of the batch maximum, if it beats the running max).
    // Entries at / past the list terminator get -inf and can never be selected (the dot
    // products are finite, > -inf).
    constexpr int B = MJCCD_HILLCLIMB_BATCH;
#if defined(__HIP_DEVICE_COMPILE__)
    typedef const HcRec __attribute__((address_space(3))) LRec;
    typedef const int __attribute__((address_space(3))) LInt;
    LRec* lrec = (LRec*)rec;
    LInt* l_edgeadr = (LInt*)vert_edgeadr;
    LInt* l_globalid = (LInt*)vert_globalid;
#else
    const HcRec* lrec = rec;
    const int* l_edgeadr = vert_edgeadr;
    const int* l_globalid = vert_globalid;
#endif
    int adr = l_edgeadr[imax];
    int badr = adr;
    float bx = 0.f, by = 0.f, bz = 0.f;
    bool have = false;
#if defined(__HIP_DEVICE_COMPILE__)
    if (COOP || obj->wave_coop) {
      // whole wave, uniform state: lane k evaluates neighbour k of the current vertex.  The
      // sequential strict '>' scan ends on the first neighbour attaining the maximum dot
      // product if that maximum beats the running max, so: M = max over the valid lanes
      // (exact, order-free), then the first lane whose dot equals M.
      const int lane = __lane_id();
#if MJCCD_HC_LOOKAHEAD
      // port-only two-step form.  Lanes 0-15 hold neighbours 0-15 of the current vertex c (the
      // step at c); lanes 16+8k+j hold neighbour j of c's neighbour k (k < 6, j < 8), i.e. the
      // whole scan of the step after a move to neighbour k when that list has < 8 entries.  Both
      // steps use the same dot products and the same first-index-of-maximum / strict '>' rule
      // as the scan below, so the sequence of vertices, max and the result are unchanged.  A
      // step that is not covered (c with >= 16 neighbours: one generic 64-lane step; a move to
      // a neighbour k >= 6 or with >= 8 neighbours: the next iteration starts there) falls back.
      {
        const int grp = lane < 16 ? 0 : ((lane - 16) >> 3) + 1;  // 0: step at c; k+1: after move to k
        const int jj = lane < 16 ? lane : (lane & 7);
        for (;;) {
          MJP_ADD(12, 1);
          const int ia = lane < 16 ? lane : grp - 1;
          const int ap = lrec[adr + ia].packed;
          // second level, issued before the first level is checked: neighbour jj of c's
          // neighbour grp-1 (a slot past c's terminator reads some list; never selected)
          const bool lo = lane < 16;
          const int ib = lo ? adr : (int)((unsigned)ap >> 16) + jj;
          const float ax = lrec[adr + ia].x, ay = lrec[adr + ia].y, az = lrec[adr + ia].z;
          const float bx2 = lrec[ib].x, by2 = lrec[ib].y, bz2 = lrec[ib].z;
          const int bp = lrec[ib].packed;
          const unsigned long long tA = __ballot((ap & 0xffff) == 0xffff) & 0xffffull;
          const int cnt0 = tA ? __ffsll((long long)tA) - 1 : 16;  // 16: not covered (fallback below)
          const float rx = lo ? ax : bx2, ry = lo ? ay : by2, rz = lo ? az : bz2;
          const int rp = lo ? ap : bp;
          const unsigned long long tB = __ballot(!lo && (bp & 0xffff) == 0xffff);
          const int sh = lo ? 0 : 8 * grp + 8;
          const unsigned gbits = (unsigned)(tB >> sh) & 0xffu;
          const unsigned cg = gbits ? (unsigned)__builtin_ctz(gbits) : 8u;
          const int lim = lo ? cnt0 : ((grp - 1 < cnt0) ? (int)cg : 0);
          const bool valid = jj < lim;
          const float vb[3] = {rx, ry, rz};
          const mjtNum d = dot3f(local_dir, vb);
          const mjtNum v = valid ? d : -INFINITY;
          mjtNum m = v;
          m = wave_dpp_maxs<0xB1>(m);   // quad_perm [1,0,3,2]
          m = wave_dpp_maxs<0x4E>(m);   // quad_perm [2,3,0,1]
          m = wave_dpp_maxs<0x141>(m);  // row_half_mirror: every lane holds its 8-group max
          const mjtNum m16 = wave_dpp_maxs<0x140>(m);  // row_mirror: lanes 0-15 hold the step-at-c max
          if (!tA) {  // checked after the reduction so both loads issue together
            // >= 16 neighbours: one generic step at c (64-lane chunks)
            prev = imax;
            int i = adr;
            for (;;) {
              const float rx = lrec[i + lane].x, ry = lrec[i + lane].y, rz = lrec[i + lane].z;
              const int rp = lrec[i + lane].packed;
              const unsigned long long term = __ballot((rp & 0xffff) == 0xffff);
              const int cnt = term ? __ffsll((long long)term) - 1 : 64;
              const float vb[3] = {rx, ry, rz};
              const mjtNum d = dot3f(local_dir, vb);
              const mjtNum v = lane < cnt ? d : -INFINITY;
              mjtNum m = v;
              m = wave_dpp_max<0xB1>(m);
              m = wave_dpp_max<0x4E>(m);
              m = wave_dpp_max<0x141>(m);
              m = wave_dpp_max<0x140>(m);
              m = fmax(m, wave_shfl_xor_f64(m, 16));
              m = fmax(m, wave_shfl_xor_f64(m, 32));
              const mjtNum M = wave_readfirstlane_f64(m);
              if (M > max) {
                const unsigned long long win = __ballot(lane < cnt && v == M);
                const int k = __ffsll((long long)win) - 1;
                max = M;
                const int pk = __builtin_amdgcn_readlane(rp, k);
                imax = pk & 0xffff;
                badr = (int)((unsigned)pk >> 16);
                bx = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rx), k));
                by = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ry), k));
                bz = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rz), k));
                have = true;
              }
              if (cnt < 64) break;
              i += 64;
            }
            if (imax == prev) break;
            adr = badr;
            continue;
          }
          const long long mb = __double_as_longlong(m16);
          const mjtNum M0 = __longlong_as_double(
              ((long long)__builtin_amdgcn_readlane((int)(mb >> 32), 0) << 32) |
              (long long)(unsigned)__builtin_amdgcn_readlane((int)(mb & 0xffffffffll), 0));
          if (!(M0 > max)) break;  // c is the local maximum
          const unsigned long long ball = __ballot(valid && v == M0) & 0xffffull;
          const int k = __ffsll((long long)ball) - 1;
          max = M0;
          const int pk = __builtin_amdgcn_readlane(rp, k);
          imax = pk & 0xffff;  // move to neighbour k
          badr = (int)((unsigned)pk >> 16);
          bx = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rx), k));
          by = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ry), k));
          bz = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rz), k));
          have = true;
          const int gl = 16 + 8 * k;  // lanes of the step after this move
          const unsigned gb = k < 6 ? (unsigned)(tB >> gl) & 0xffu : 0u;
          if (!gb) {  // not covered: the next iteration scans the new vertex
            adr = badr;
            continue;
          }
          const long long mbk = __double_as_longlong(m);
          const mjtNum M1 = __longlong_as_double(
              ((long long)__builtin_amdgcn_readlane((int)(mbk >> 32), gl) << 32) |
              (long long)(unsigned)__builtin_amdgcn_readlane((int)(mbk & 0xffffffffll), gl));
          MJP_ADD(12, 1);
          if (!(M1 > max)) break;  // the new vertex is the local maximum
          const unsigned long long ball1 = (__ballot(valid && v == M1) >> gl) & 0xffull;
          const int k1 = gl + __ffsll((long long)ball1) - 1;
          max = M1;
          const int pk1 = __builtin_amdgcn_readlane(rp, k1);
          imax = pk1 & 0xffff;
          badr = (int)((unsigned)pk1 >> 16);
          bx = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rx), k1));
          by = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ry), k1));
          bz = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rz), k1));
          adr = badr;
        }
        prev = imax;  // skip the loops below
      }
#endif
      while (imax != prev) {
        prev = imax;
        MJP_ADD(12, 1);
        int i = adr;
        for (;;) {
          const float rx = lrec[i + lane].x, ry = lrec[i + lane].y, rz = lrec[i + lane].z;
          const int rp = lrec[i + lane].packed;
          const unsigned long long term = __ballot((rp & 0xffff) == 0xffff);
          const int cnt = term ? __ffsll((long long)term) - 1 : 64;
          const float vb[3] = {rx, ry, rz};
          const mjtNum d = dot3f(local_dir, vb);
          const mjtNum v = lane < cnt ? d : -INFINITY;
          mjtNum m = v;
          m = wave_dpp_max<0xB1>(m);   // quad_perm [1,0,3,2]
          m = wave_dpp_max<0x4E>(m);   // quad_perm [2,3,0,1]
          if (cnt > 4) m = wave_dpp_max<0x141>(m);  // row_half_mirror
          if (cnt > 8) m = wave_dpp_max<0x140>(m);  // row_mirror
          if (cnt > 16) {
            m = fmax(m, wave_shfl_xor_f64(m, 16));
            m = fmax(m, wave_shfl_xor_f64(m, 32));
          }
          const mjtNum M = wave_readfirstlane_f64(m);
          if (M > max) {
            const unsigned long long win = __ballot(lane < cnt && v == M);
            const int k = __ffsll((long long)win) - 1;
            max = M;
            const int pk = __builtin_amdgcn_readlane(rp, k);
            imax = pk & 0xffff;  // update maximum vertex index
            badr = (int)((unsigned)pk >> 16);
            bx = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rx), k));
            by = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(ry), k));
            bz = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(rz), k));
            have = true;
          }
          if (cnt < 64) break;
          i += 64;
        }
        adr = badr;
      }
      prev = imax;  // skip the single-lane loop below
    }
#endif
    while (imax != prev) {
      prev = imax;
      int i = adr;
      for (;;) {
        HcRec r[B];
#pragma unroll
        for (int k = 0; k < B; k++) {
          r[k].x = lrec[i + k].x; r[k].y = lrec[i + k].y; r[k].z = lrec[i + k].z;
          r[k].packed = lrec[i + k].packed;
        }
        int cnt = B;
#pragma unroll
        for (int k = B - 1; k >= 0; k--) if ((r[k].packed & 0xffff) == 0xffff) cnt = k;
        mjtNum v[B];
#pragma unroll
        for (int k = 0; k < B; k++) {
          const float vb[3] = {r[k].x, r[k].y, r[k].z};
          mjtNum d = dot3f(local_dir, vb);
          v[k] = k < cnt ? d : -INFINITY;
        }
        // first-index argmax tree over the batch
#pragma unroll
        for (int w = 1; w < B; w *= 2) {
#pragma unroll
          for (int a = 0; a + w < B; a += 2 * w) {
            const int b = a + w;
            const bool t = v[b] > v[a];
            v[a] = t ? v[b] : v[a];
            r[a].x = t ? r[b].x : r[a].x;
            r[a].y = t ? r[b].y : r[a].y;
            r[a].z = t ? r[b].z : r[a].z;
            r[a].packed = t ? r[b].packed : r[a].packed;
          }
        }
        if (v[0] > max) {
          max = v[0];
          imax = r[0].packed & 0xffff;  // update maximum vertex index
          badr = (int)((unsigned)r[0].packed >> 16);
          bx = r[0].x; by = r[0].y; bz = r[0].z;
          have = true;
        }
        if (cnt < B) break;
        i += B;
      }
      adr = badr;
    }
    obj->meshindex = imax;
    obj->vertindex = l_globalid[imax];
    if (have) {
      local_dir[0] = (mjtNum)bx;
      local_dir[1] = (mjtNum)by;
      local_dir[2] = (mjtNum)bz;
    } else {
      const int gi = obj->vertindex;
      local_dir[0] = (mjtNum)verts[3*gi + 0];
      local_dir[1] = (mjtNum)verts[3*gi + 1];
      local_dir[2] = (mjtNum)verts[3*gi + 2];
    }
    localToGlobal(res, mat, local_dir, pos);
    return;
  }
#endif

  // hillclimb until no change
  while (imax != prev) {
    prev = imax;
#if MJCCD_HILLCLIMB_BATCH > 1
    // port: same neighbour visit order, dot products and strict '>' updates as MuJoCo's
    // `for (i = vert_edgeadr[imax]; (subidx = edge_localid[i]) >= 0; i++)` loop, but the
    // neighbour ids / vertex ids / coordinates of up to MJCCD_HILLCLIMB_BATCH neighbours are
    // loaded together instead of as one dependent load chain per neighbour.  edge_localid is
    // followed by face_globalid (3*numface >= 12 ints) in the graph slice, so reading up to
    // BATCH-1 entries past a -1 terminator stays inside the slice; entries past the
    // terminator are never dereferenced or compared.
    constexpr int B = MJCCD_HILLCLIMB_BATCH;
    int i = vert_edgeadr[imax];
    for (;;) {
      int sub[B];
#pragma unroll
      for (int k = 0; k < B; k++) sub[k] = edge_localid[i + k];
      int cnt = B;
#pragma unroll
      for (int k = B - 1; k >= 0; k--) if (sub[k] < 0) cnt = k;
      // unconditional loads (one wait for the whole batch): slots past the terminator read
      // vertex 0's coordinates, which are never compared
      int gid[B];
#pragma unroll
      for (int k = 0; k < B; k++) gid[k] = vert_globalid[k < cnt ? sub[k] : 0];
      float vb[B][3];
#pragma unroll
      for (int k = 0; k < B; k++) {
        const float* vp = verts + 3*(k < cnt ? gid[k] : 0);
        vb[k][0] = vp[0]; vb[k][1] = vp[1]; vb[k][2] = vp[2];
      }
#pragma unroll
      for (int k = 0; k < B; k++) {
        if (k < cnt) {
          mjtNum vdot = dot3f(local_dir, vb[k]);
          if (vdot > max) {
            max = vdot;
            imax = sub[k];  // update maximum vertex index
          }
        }
      }
      if (cnt < B) break;
      i += B;
    }
#else
    int subidx;
    for (int i = vert_edgeadr[imax]; (subidx = edge_localid[i]) >= 0; i++) {
      mjtNum vdot = dot3f(local_dir, verts + 3*vert_globalid[subidx]);
      if (vdot > max) {
        max = vdot;
        imax = subidx;  // update maximum vertex index
      }
    }
#endif
  }

  // record vertex index of maximum (local id)
  obj->meshindex = imax;

  // get resulting support vertex
  obj->vertindex = imax = vert_globalid[imax];
  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}

// obj->support(res, obj, dir) dispatch
HIPGL_DEVICE inline void objSupport(mjtNum res[3], Obj* obj, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
  long long _ts = wall_clock64();
  struct _Sp { long long t; int k; __device__ ~_Sp(){ g_stat_t[k][MJST_TID] += wall_clock64() - t; g_stat_n[k][MJST_TID]++; } } _sp{_ts, obj->support & 3};
#endif
  switch (obj->support) {
    case SUPPORT_BOX:       mjc_boxSupport(res, obj, dir); break;
    case SUPPORT_MESH:      mjc_meshSupport(res, obj, dir); break;
    case SUPPORT_HILLCLIMB: mjc_hillclimbSupport(res, obj, dir); break;
    case SUPPORT_CYLINDER:  mjc_cylinderSupport(res, obj, dir); break;
    default: res[0] = obj->pos[0]; res[1] = obj->pos[1]; res[2] = obj->pos[2]; break;
  }
}

#if defined(__HIP_DEVICE_COMPILE__)
// port-only (wave_coop, both objects hill-climbed meshes with LDS edge records): the two
// objects' hill climbs of one support() call are independent, so lanes 0-31 climb obj1 along
// dir and lanes 32-63 climb obj2 along dir_neg at the same time.  Each half runs exactly the
// coop climb of mjc_hillclimbSupport (32-entry chunks instead of 64: the strict '>' first-index
// selection is chunk-size invariant), so both results and both warm starts are unchanged.
// Returns false (nothing touched) when the pair form does not apply.
HIPGL_DEVICE inline bool hillclimbPairCoop(mjtNum s1[3], mjtNum s2[3], Obj* obj1, Obj* obj2,
                                           const mjtNum dir[3], const mjtNum dir_neg[3]) {
  MJCCD_FP_CONTRACT_OFF
  typedef const HcRec __attribute__((address_space(3))) LRec;
  typedef const int __attribute__((address_space(3))) LInt;
  const int lane = __lane_id();
  const bool hi = lane >= 32;
  const int sl = lane & 31;
  Obj* obj = hi ? obj2 : obj1;
  const mjtNum d0 = hi ? dir_neg[0] : dir[0];
  const mjtNum d1 = hi ? dir_neg[1] : dir[1];
  const mjtNum d2 = hi ? dir_neg[2] : dir[2];
  const mjtNum dd[3] = {d0, d1, d2};
  const HcRec* rec = obj->data.mesh.rec;
  const mjtNum* mat = obj->mat;
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dd);
  const bool ok = rec && fabs(local_dir[0]) < 1e250 && fabs(local_dir[1]) < 1e250 &&
                  fabs(local_dir[2]) < 1e250;
  if (__ballot(!ok)) return false;

  const int* graph = obj->data.mesh.graph;
  LInt* lg = (LInt*)graph;
  const int numvert = lg[0];
  LInt* l_edgeadr = lg + 2;
  LInt* l_globalid = lg + 2 + numvert;
  LRec* lrec = (LRec*)rec;

  mjtNum max = -FLT_MAX;
  int prev = -1;
  int imax = obj->meshindex >= 0 ? obj->meshindex : 0;
  int adr = l_edgeadr[imax];
  int badr = adr;
  float bx = 0.f, by = 0.f, bz = 0.f;
  bool have = false;
  while (imax != prev) {
    prev = imax;
    int i = adr;
    for (;;) {
      const float rx = lrec[i + sl].x, ry = lrec[i + sl].y, rz = lrec[i + sl].z;
      const int rp = lrec[i + sl].packed;
      const unsigned long long termw = __ballot((rp & 0xffff) == 0xffff);
      const unsigned term = hi ? (unsigned)(termw >> 32) : (unsigned)termw;
      const int cnt = term ? __ffs((int)term) - 1 : 32;
      const float vb[3] = {rx, ry, rz};
      const mjtNum dv = dot3f(local_dir, vb);
      const mjtNum v = sl < cnt ? dv : -INFINITY;
      mjtNum m = v;
      m = wave_dpp_max<0xB1>(m);   // quad_perm [1,0,3,2]
      m = wave_dpp_max<0x4E>(m);   // quad_perm [2,3,0,1]
      if (cnt > 4) m = wave_dpp_max<0x141>(m);  // row_half_mirror
      if (cnt > 8) m = wave_dpp_max<0x140>(m);  // row_mirror
      if (cnt > 16) m = fmax(m, wave_shfl_xor_f64(m, 16));
      // the half's maximum sits on its first lane (lanes past 16 only hold it after the xor-16
      // step); broadcast it so every lane of the half keeps identical state
      const long long mb = __double_as_longlong(m);
      const int mlo = (int)(mb & 0xffffffffll), mhi = (int)(mb >> 32);
      const int mlo0 = __builtin_amdgcn_readlane(mlo, 0), mhi0 = __builtin_amdgcn_readlane(mhi, 0);
      const int mlo1 = __builtin_amdgcn_readlane(mlo, 32), mhi1 = __builtin_amdgcn_readlane(mhi, 32);
      const mjtNum M = __longlong_as_double(((long long)(hi ? mhi1 : mhi0) << 32) |
                                            (long long)(unsigned)(hi ? mlo1 : mlo0));
      if (M > max) {
        const unsigned long long winw = __ballot(sl < cnt && v == M);
        const unsigned lo32 = (unsigned)winw, hi32 = (unsigned)(winw >> 32);
        const int k0 = lo32 ? __ffs((int)lo32) - 1 : 0;
        const int k1 = hi32 ? 32 + __ffs((int)hi32) - 1 : 32;
        max = M;
        const int pk0 = __builtin_amdgcn_readlane(rp, k0);
        const int pk1 = __builtin_amdgcn_readlane(rp, k1);
        const int fx0 = __builtin_amdgcn_readlane(__float_as_int(rx), k0);
        const int fx1 = __builtin_amdgcn_readlane(__float_as_int(rx), k1);
        const int fy0 = __builtin_amdgcn_readlane(__float_as_int(ry), k0);
        const int fy1 = __builtin_amdgcn_readlane(__float_as_int(ry), k1);
        const int fz0 = __builtin_amdgcn_readlane(__float_as_int(rz), k0);
        const int fz1 = __builtin_amdgcn_readlane(__float_as_int(rz), k1);
        const int pk = hi ? pk1 : pk0;
        imax = pk & 0xffff;
        badr = (int)((unsigned)pk >> 16);
        bx = __int_as_float(hi ? fx1 : fx0);
        by = __int_as_float(hi ? fy1 : fy0);
        bz = __int_as_float(hi ? fz1 : fz0);
        have = true;
      }
      if (cnt < 32) break;
      i += 32;
    }
    adr = badr;
  }
  const int gid = l_globalid[imax];
  if (have) {
    local_dir[0] = (mjtNum)bx;
    local_dir[1] = (mjtNum)by;
    local_dir[2] = (mjtNum)bz;
  } else {
    const float* verts = obj->data.mesh.vert;
    local_dir[0] = (mjtNum)verts[3*gid + 0];
    local_dir[1] = (mjtNum)verts[3*gid + 1];
    local_dir[2] = (mjtNum)verts[3*gid + 2];
  }
  mjtNum res[3];
  localToGlobal(res, mat, local_dir, obj->pos);
  // every lane of a half holds identical state; publish each half's warm start to its object
  const int mi0 = __builtin_amdgcn_readlane(imax, 0), mi1 = __builtin_amdgcn_readlane(imax, 32);
  const int gi0 = __builtin_amdgcn_readlane(gid, 0), gi1 = __builtin_amdgcn_readlane(gid, 32);
#pragma unroll
  for (int k = 0; k < 3; k++) {
    const long long b = __double_as_longlong(res[k]);
    const int lo = (int)(b & 0xffffffffll), hi2 = (int)(b >> 32);
    s1[k] = __longlong_as_double(((long long)__builtin_amdgcn_readlane(hi2, 0) << 32) |
                                 (long long)(unsigned)__builtin_amdgcn_readlane(lo, 0));
    s2[k] = __longlong_as_double(((long long)__builtin_amdgcn_readlane(hi2, 32) << 32) |
                                 (long long)(unsigned)__builtin_amdgcn_readlane(lo, 32));
  }
  obj1->meshindex = mi0;
  obj1->vertindex = gi0;
  obj2->meshindex = mi1;
  obj2->vertindex = gi1;
  return true;
}
#endif

// C:137-170 mjc_center (geom >= 0 branch)
HIPGL_DEVICE inline void mjc_center(mjtNum res[3], const Obj* obj) {
  MJCCD_FP_CONTRACT_OFF
  res[0] = obj->pos[0], res[1] = obj->pos[1], res[2] = obj->pos[2];
}

// ------------------------------------ object setup -------------------------------------------

// C:704-787  mjc_initCCDObj, box and mesh branches.  size/pos/mat are geom_size+3g,
// geom_xpos+3g, geom_xmat+9g; dataid = geom_dataid[g]; mm may be NULL for boxes.
HIPGL_DEVICE inline void initCCDObj(Obj* obj, int g, int geom_type, const mjtNum size[3],
                                    const mjtNum pos[3], const mjtNum mat[9], int dataid,
                                    const MeshModel* mm, mjtNum margin) {
  MJCCD_FP_CONTRACT_OFF
  int graphadr, vertadr, polyadr;
  obj->geom = g;
  obj->margin = margin;
  obj->vertindex = -1;
  obj->meshindex = -1;
  for (int i = 0; i < 3; i++) obj->size[i] = size[i];
  obj->size[3] = 0;
  for (int i = 0; i < 3; i++) obj->pos[i] = pos[i];
  for (int i = 0; i < 9; i++) obj->mat[i] = mat[i];
  obj->geom_type = geom_type;
  obj->data.mesh.graph = 0;
  obj->data.mesh.rec = 0;
  obj->wave_coop = 0;
  obj->data.mesh.vert = 0;
  obj->data.mesh.nvert = 0;
  obj->data.mesh.mesh_polynum = 0;
  switch (geom_type) {
    case mjGEOM_MESH:
      graphadr = mm->mesh_graphadr[dataid];
      vertadr = mm->mesh_vertadr[dataid];
      polyadr = mm->mesh_polyadr[dataid];
      if (graphadr < 0 || mm->mesh_vertnum[dataid] < mjMESH_HILLCLIMB_MIN) {
        obj->data.mesh.graph = 0;
        obj->support = SUPPORT_MESH;
      } else {
        obj->data.mesh.graph = mm->mesh_graph + graphadr;
        obj->support = SUPPORT_HILLCLIMB;
      }
      obj->data.mesh.vert = mm->mesh_vert + 3*vertadr;
      obj->data.mesh.nvert = mm->mesh_vertnum[dataid];
      obj->data.mesh.mpolymapadr = mm->mesh_polymapadr + vertadr;
      obj->data.mesh.mpolymapnum = mm->mesh_polymapnum + vertadr;
      obj->data.mesh.polymap = mm->mesh_polymap;
      obj->data.mesh.polynormal = mm->mesh_polynormal + 3*polyadr;
      obj->data.mesh.polyvertadr = mm->mesh_polyvertadr + polyadr;
      obj->data.mesh.polyvertnum = mm->mesh_polyvertnum + polyadr;
      obj->data.mesh.polyvert = mm->mesh_polyvert;
      obj->data.mesh.mesh_polynum = mm->mesh_polynum[dataid];
      break;
    case mjGEOM_BOX:
      obj->support = SUPPORT_BOX;
      break;
    case mjGEOM_CYLINDER:
      obj->support = SUPPORT_CYLINDER;
      break;
    default:
      obj->support = SUPPORT_NONE;
      break;
  }
}

// mesh_graph slice for one mesh, NULL when the mesh has no graph (mesh_graphadr < 0), which
// is the value initMeshObj needs (C:729: graphadr < 0 -> exhaustive mjc_meshSupport).  Do not
// pass mesh_graph + mesh_graphadr[id] unchecked: MuJoCo leaves graphadr = -1 for a mesh that
// is not used by any colliding geom at compile time (user_mesh.cc:1043-1066).
HIPGL_DEVICE inline const int* meshGraph(const int* mesh_graph, int graphadr) {
  return graphadr < 0 ? 0 : mesh_graph + graphadr;
}

// same as initCCDObj for a mesh, from already-sliced per-mesh arrays (MuJoCo semantics:
// mpolymapadr indexes polymap, polyvertadr indexes polyvert).  graph must be
// meshGraph(mesh_graph, mesh_graphadr[id]) (or a rebased copy of that slice): graph == NULL
// <=> MuJoCo's graphadr < 0 -> exhaustive search.  g must be the real (>= 0) geom id.
HIPGL_DEVICE inline void initMeshObj(Obj* obj, int g, const mjtNum pos[3], const mjtNum mat[9],
                                     const mjtNum size[3], int nvert, const float* vert,
                                     const int* graph, int mesh_polynum,
                                     const int* mpolymapadr, const int* mpolymapnum,
                                     const int* polymap, const mjtNum* polynormal,
                                     const int* polyvertadr, const int* polyvertnum,
                                     const int* polyvert, mjtNum margin) {
  MJCCD_FP_CONTRACT_OFF
  obj->geom = g;
  obj->margin = margin;
  obj->vertindex = -1;
  obj->meshindex = -1;
  for (int i = 0; i < 3; i++) obj->size[i] = size[i];
  obj->size[3] = 0;
  for (int i = 0; i < 3; i++) obj->pos[i] = pos[i];
  for (int i = 0; i < 9; i++) obj->mat[i] = mat[i];
  obj->geom_type = mjGEOM_MESH;
  obj->data.mesh.rec = 0;
  obj->wave_coop = 0;
  if (!graph || nvert < mjMESH_HILLCLIMB_MIN) {
    obj->data.mesh.graph = 0;
    obj->support = SUPPORT_MESH;
  } else {
    obj->data.mesh.graph = graph;
    obj->support = SUPPORT_HILLCLIMB;
  }
  obj->data.mesh.vert = vert;
  obj->data.mesh.nvert = nvert;
  obj->data.mesh.mpolymapadr = mpolymapadr;
  obj->data.mesh.mpolymapnum = mpolymapnum;
  obj->data.mesh.polymap = polymap;
  obj->data.mesh.polynormal = polynormal;
  obj->data.mesh.polyvertadr = polyvertadr;
  obj->data.mesh.polyvertnum = polyvertnum;
  obj->data.mesh.polyvert = polyvert;
  obj->data.mesh.mesh_polynum = mesh_polynum;
}

// cylinder counterpart of initMeshObj (size = MuJoCo geom_size: radius, half-length)
HIPGL_DEVICE inline void initCylinderObj(Obj* obj, int g, const mjtNum pos[3], const mjtNum mat[9],
                                         const mjtNum size[3], mjtNum margin) {
  initCCDObj(obj, g, mjGEOM_CYLINDER, size, pos, mat, -1, 0, margin);
}

// box counterpart of initMeshObj (g must be the real, >= 0 geom id)
HIPGL_DEVICE inline void initBoxObj(Obj* obj, int g, const mjtNum pos[3], const mjtNum mat[9],
                                    const mjtNum size[3], mjtNum margin) {
  MJCCD_FP_CONTRACT_OFF
  initCCDObj(obj, g, mjGEOM_BOX, size, pos, mat, -1, 0, margin);
}

// ---------------------------------------- GJK ------------------------------------------------

HIPGL_DEVICE inline void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3],
                             const mjtNum s3[3], const mjtNum s4[3]);
HIPGL_DEVICE inline void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3],
                             const mjtNum s3[3]);
HIPGL_DEVICE inline void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]);
template <int PK = 0>
HIPGL_DEVICE inline int gjkIntersect(Status* status, Obj* obj1, Obj* obj2);

// G:165-174  return true if both geoms are discrete shapes (i.e. meshes or boxes with no margin)
HIPGL_DEVICE inline int discreteGeoms(Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  // non-zero margin makes geoms smooth
  if (obj1->margin != 0 || obj2->margin != 0) return 0;

  int g1 = obj1->geom_type;
  int g2 = obj2->geom_type;
  return (g1 == mjGEOM_MESH || g1 == mjGEOM_BOX || g1 == mjGEOM_HFIELD) &&
         (g2 == mjGEOM_MESH || g2 == mjGEOM_BOX || g2 == mjGEOM_HFIELD);
}

// G:456-480  linear combination of n 3D vectors
HIPGL_DEVICE inline void lincomb(mjtNum res[3], const mjtNum* coef, int n, const mjtNum v1[3],
                                 const mjtNum v2[3], const mjtNum v3[3], const mjtNum v4[3]) {
  MJCCD_FP_CONTRACT_OFF
  switch (n) {
    case 1: {
      mjtNum r0 = coef[0]*v1[0];
      mjtNum r1 = coef[0]*v1[1];
      mjtNum r2 = coef[0]*v1[2];
      res[0] = r0; res[1] = r1; res[2] = r2;
      break;
    }
    case 2: {
      mjtNum r0 = coef[0]*v1[0] + coef[1]*v2[0];
      mjtNum r1 = coef[0]*v1[1] + coef[1]*v2[1];
      mjtNum r2 = coef[0]*v1[2] + coef[1]*v2[2];
      res[0] = r0; res[1] = r1; res[2] = r2;
      break;
    }
    case 3: {
      mjtNum r0 = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0];
      mjtNum r1 = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1];
      mjtNum r2 = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2];
      res[0] = r0; res[1] = r1; res[2] = r2;
      break;
    }
    case 4: {
      mjtNum r0 = coef[0]*v1[0] + coef[1]*v2[0] + coef[2]*v3[0] + coef[3]*v4[0];
      mjtNum r1 = coef[0]*v1[1] + coef[1]*v2[1] + coef[2]*v3[1] + coef[3]*v4[1];
      mjtNum r2 = coef[0]*v1[2] + coef[1]*v2[2] + coef[2]*v3[2] + coef[3]*v4[2];
      res[0] = r0; res[1] = r1; res[2] = r2;
      break;
    }
  }
}

// G:311-338  compute the support point in obj1 and obj2 for Minkowski difference
template <int PK = 0>
HIPGL_DEVICE inline void support(Vertex* v, Obj* obj1, Obj* obj2,
                                 const mjtNum dir[3], const mjtNum dir_neg[3]) {
  MJCCD_FP_CONTRACT_OFF
  // port: support points are built in registers and written to v once at the end (same
  // arithmetic; v lives in global scratch and dir/dir_neg never alias it)
  mjtNum s1[3], s2[3];
  const long long _mjp_ts = MJP_T();

#if defined(__HIP_DEVICE_COMPILE__)
  if constexpr (PK == 1) {
    // port-only specialisation (caller-checked): obj1 a box, obj2 a hill-climbed mesh, wave coop
#if MJCCD_PK1_SPLIT
    // port: mjc_boxSupportBody split around the mesh climb (independent objects), so its
    // direction rotation overlaps the climb's prologue and its transform the climb's epilogue.
    // Same expressions on the same operands; the local support point is kept as its sign bits.
    LdsObj* b = (LdsObj*)obj1;
    mjtNum bl[3];
    {
      mjtNum bm[9];
      for (int k = 0; k < 9; k++) bm[k] = b->mat[k];
      mulMatTVec3(bl, bm, dir);
    }
    const int bge = (bl[0] >= 0 ? 1 : 0) | (bl[1] >= 0 ? 2 : 0) | (bl[2] >= 0 ? 4 : 0);
    mjc_hillclimbSupport<true>(s2, obj2, dir_neg);
    {
      mjtNum bm[9], bp[3], bs[3], ls[3];
      for (int k = 0; k < 9; k++) bm[k] = b->mat[k];
      for (int k = 0; k < 3; k++) { bp[k] = b->pos[k]; bs[k] = b->size[k]; }
      ls[0] = (bge & 1) ? bs[0] : -bs[0];
      ls[1] = (bge & 2) ? bs[1] : -bs[1];
      ls[2] = (bge & 4) ? bs[2] : -bs[2];
      int vi = (ls[0] > 0) ? 1 : 0;
      vi |= (ls[1] > 0) ? 2 : 0;
      vi |= (ls[2] > 0) ? 4 : 0;
      b->vertindex = vi;
      localToGlobal(s1, bm, ls, bp);
    }
#else
    mjc_boxSupportBody(s1, (LdsObj*)obj1, dir);
    mjc_hillclimbSupport<true>(s2, obj2, dir_neg);
#endif
  } else if constexpr (PK == 2) {
    // port-only specialisation (caller-checked): two hill-climbed meshes, wave coop
    if (!hillclimbPairCoop(s1, s2, obj1, obj2, dir, dir_neg)) {
      mjc_hillclimbSupport<true>(s1, obj1, dir);
      mjc_hillclimbSupport<true>(s2, obj2, dir_neg);
    }
  } else
#endif
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_HC_PAIR
  if (obj1->wave_coop && obj1->support == SUPPORT_HILLCLIMB &&
      obj2->support == SUPPORT_HILLCLIMB && hillclimbPairCoop(s1, s2, obj1, obj2, dir, dir_neg)) {
  } else
#endif
  {
  // obj1
  const long long _mjp_t1 = MJP_T();
  MJP_ADD(14, _mjp_t1 - _mjp_ts);
  objSupport(s1, obj1, dir);
  const long long _mjp_t2 = MJP_T();
  MJP_ADD(9, _mjp_t2 - _mjp_t1);

  // obj2
  objSupport(s2, obj2, dir_neg);
  MJP_ADD(10, MJP_T() - _mjp_t2);
  }
  const long long _mjp_t3 = MJP_T();
  if (obj1->margin > 0 && obj1->geom >= 0) {
    mjtNum margin = 0.5 * obj1->margin;
    s1[0] += dir[0] * margin;
    s1[1] += dir[1] * margin;
    s1[2] += dir[2] * margin;
  }
  if (obj2->margin > 0 && obj2->geom >= 0) {
    mjtNum margin = 0.5 * obj2->margin;
    s2[0] += dir_neg[0] * margin;
    s2[1] += dir_neg[1] * margin;
    s2[2] += dir_neg[2] * margin;
  }

  // compute S_{A-B}(dir) = S_A(dir) - S_B(-dir)
  copy3(v->vert1, s1);
  copy3(v->vert2, s2);
  sub3(v->vert, s1, s2);

  // copy vertex indices of discrete geoms
  v->index1 = obj1->vertindex;
  v->index2 = obj2->vertindex;
  MJP_ADD(13, MJP_T() - _mjp_t3);
  MJP_ADD(0, MJP_T() - _mjp_ts);
  MJP_ADD(1, 1);
}

// G:341-349  compute the support points in obj1 and obj2 for the kth approximation point
template <int PK = 0>
HIPGL_DEVICE inline void gjkSupport(Vertex* v, Obj* obj1, Obj* obj2,
                                    const mjtNum x_k[3], mjtNum x_norm) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum dir[3], dir_neg[3];

  // mjc_support requires a normalized direction
  scl3(dir_neg, x_k, 1 / x_norm);
  scl3(dir, dir_neg, -1);
  support<PK>(v, obj1, obj2, dir, dir_neg);
}

// G:353-370  compute support points in Minkowski difference, return index of new vertex
template <int PK = 0>
HIPGL_DEVICE inline int epaSupport(Polytope* pt, Obj* obj1, Obj* obj2,
                                   const mjtNum d[3], mjtNum dnorm) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum dir[3] = {1, 0, 0}, dir_neg[3] = {-1, 0, 0};

  // mjc_support assumes a normalized direction
  if (dnorm > mjMINVAL) {
    dir[0] = d[0] / dnorm;
    dir[1] = d[1] / dnorm;
    dir[2] = d[2] / dnorm;
    scl3(dir_neg, dir, -1);
  }

  int n = pt->nverts++;
  Vertex* v = pt->verts + n;
  support<PK>(v, obj1, obj2, dir, dir_neg);
  return n;
}

// G:373-377  support point in the Minkowski difference for gjkIntersect (no normalization)
template <int PK = 0>
HIPGL_DEVICE inline void gjkIntersectSupport(Vertex* v, Obj* obj1, Obj* obj2,
                                             const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum dir_neg[3] = {-dir[0], -dir[1], -dir[2]};
  support<PK>(v, obj1, obj2, dir, dir_neg);
}

// G:381-394  compute the signed distance of a face along with the normal
HIPGL_DEVICE inline mjtNum signedDistance(mjtNum normal[3], const Vertex* v1, const Vertex* v2,
                                          const Vertex* v3) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum diff1[3], diff2[3];
  sub3(diff1, v3->vert, v1->vert);
  sub3(diff2, v2->vert, v1->vert);
  cross3(normal, diff1, diff2);
  mjtNum norm2 = dot3(normal, normal);
  if (norm2 > mjMINVAL2 && norm2 < mjMAXVAL2) {
    scl3(normal, normal, 1 / mju_sqrt(norm2));
    return dot3(normal, v1->vert);
  }
  return mjMAX_LIMIT;  // cannot recover normal (ignore face)
}

// G:397-452  return 1 if objects are in contact; 0 if not; -1 if inconclusive
template <int PK>
HIPGL_DEVICE inline int gjkIntersect(Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
#if MJCCD_GJKINTERSECT_REGS
  // port: MuJoCo keeps a local simplex[4] plus an index permutation s[4] and addresses
  // simplex[s[k]] dynamically, which puts the simplex in (slow) private memory.  Here
  // P[k] == simplex[s[k]] is maintained directly: the replacement and the swap of s[i]/s[j]
  // become static-index selects on P, so every arithmetic operation and its operands are the
  // same as MuJoCo's.
  Vertex P[4] = {status->simplex[0], status->simplex[1],
                 status->simplex[2], status->simplex[3]};

  int k = status->gjk_iterations, kmax = status->max_iterations;
  for (; k < kmax; k++) {
    // compute the signed distance to each face in the simplex along with normals
    mjtNum dist[4], normals[12];
    dist[0] = signedDistance(&normals[0], &P[2], &P[1], &P[3]);
    dist[1] = signedDistance(&normals[3], &P[0], &P[2], &P[3]);
    dist[2] = signedDistance(&normals[6], &P[1], &P[0], &P[3]);
    dist[3] = signedDistance(&normals[9], &P[0], &P[1], &P[2]);

    // if origin is on any affine hull, convergence will fail
    if (!dist[3] || !dist[2] || !dist[1] || !dist[0]) {
      status->gjk_iterations = k;
      return -1;
    }

    // find the face with the smallest distance to the origin
    int i = (dist[0] < dist[1]) ? 0 : 1;
    int j = (dist[2] < dist[3]) ? 2 : 3;
    mjtNum di = (dist[0] < dist[1]) ? dist[0] : dist[1];
    mjtNum dj = (dist[2] < dist[3]) ? dist[2] : dist[3];
    int index = (di < dj) ? i : j;
    mjtNum dindex = (di < dj) ? di : dj;

    // origin inside of simplex (run EPA for contact information)
    if (dindex > 0) {
      status->nsimplex = 4;
      status->simplex[0] = P[0];
      status->simplex[1] = P[1];
      status->simplex[2] = P[2];
      status->simplex[3] = P[3];
      status->gjk_iterations = k;
      return 1;
    }

    // replace worst vertex (farthest from origin) with new candidate
    mjtNum nrm[3];
    for (int c = 0; c < 3; c++) {
      nrm[c] = index == 0 ? normals[c] : index == 1 ? normals[3 + c]
             : index == 2 ? normals[6 + c] : normals[9 + c];
    }
    Vertex nv;
    gjkIntersectSupport<PK>(&nv, obj1, obj2, nrm);

    // found origin outside the Minkowski difference (return no collision)
    if (dot3(nrm, nv.vert) < 0) {
      status->nsimplex = 0;
      status->gjk_iterations = k;
      return 0;
    }

    // P[index] = nv, then swap the entries (index+1)&3 and (index+2)&3
    if (index == 0) { P[0] = nv; Vertex t = P[1]; P[1] = P[2]; P[2] = t; }
    else if (index == 1) { P[1] = nv; Vertex t = P[2]; P[2] = P[3]; P[3] = t; }
    else if (index == 2) { P[2] = nv; Vertex t = P[3]; P[3] = P[0]; P[0] = t; }
    else { P[3] = nv; Vertex t = P[0]; P[0] = P[1]; P[1] = t; }
  }
  status->gjk_iterations = k;
  return -1;  // never found origin
#else
  Vertex simplex[4] = {status->simplex[0], status->simplex[1],
                       status->simplex[2], status->simplex[3]};
  int s[4] = {0, 1, 2, 3};

  int k = status->gjk_iterations, kmax = status->max_iterations;
  for (; k < kmax; k++) {
    // compute the signed distance to each face in the simplex along with normals
    mjtNum dist[4], normals[12];
    dist[0] = signedDistance(&normals[0], simplex + s[2], simplex + s[1], simplex + s[3]);
    dist[1] = signedDistance(&normals[3], simplex + s[0], simplex + s[2], simplex + s[3]);
    dist[2] = signedDistance(&normals[6], simplex + s[1], simplex + s[0], simplex + s[3]);
    dist[3] = signedDistance(&normals[9], simplex + s[0], simplex + s[1], simplex + s[2]);

    // if origin is on any affine hull, convergence will fail
    if (!dist[3] || !dist[2] || !dist[1] || !dist[0]) {
      status->gjk_iterations = k;
      return -1;
    }

    // find the face with the smallest distance to the origin
    int i = (dist[0] < dist[1]) ? 0 : 1;
    int j = (dist[2] < dist[3]) ? 2 : 3;
    int index = (dist[i] < dist[j]) ? i : j;

    // origin inside of simplex (run EPA for contact information)
    if (dist[index] > 0) {
      status->nsimplex = 4;
      status->simplex[0] = simplex[s[0]];
      status->simplex[1] = simplex[s[1]];
      status->simplex[2] = simplex[s[2]];
      status->simplex[3] = simplex[s[3]];
      status->gjk_iterations = k;
      return 1;
    }

    // replace worst vertex (farthest from origin) with new candidate
    gjkIntersectSupport<PK>(simplex + s[index], obj1, obj2, normals + 3*index);

    // found origin outside the Minkowski difference (return no collision)
    if (dot3(&normals[3*index], simplex[s[index]].vert) < 0) {
      status->nsimplex = 0;
      status->gjk_iterations = k;
      return 0;
    }

    // swap vertices in the simplex to retain orientation
    i = (index + 1) & 3;
    j = (index + 2) & 3;
    int swap = s[i];
    s[i] = s[j];
    s[j] = swap;
  }
  status->gjk_iterations = k;
  return -1;  // never found origin
#endif
}

// G:484-517  res = origin projected onto plane defined by v1, v2, v3
HIPGL_DEVICE inline int projectOriginPlane(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3],
                                           const mjtNum v3[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum diff21[3], diff31[3], diff32[3], n[3], nv, nn;
  sub3(diff21, v2, v1);
  sub3(diff31, v3, v1);
  sub3(diff32, v3, v2);

  // n = (v1 - v2) x (v3 - v2)
  cross3(n, diff32, diff21);
  nv = dot3(n, v2);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v2 - v1) x (v3 - v1)
  cross3(n, diff21, diff31);
  nv = dot3(n, v1);
  nn = dot3(n, n);
  if (nn == 0) return 1;
  if (nv != 0 && nn > mjMINVAL) {
    scl3(res, n, nv / nn);
    return 0;
  }

  // n = (v1 - v3) x (v2 - v3)
  cross3(n, diff31, diff32);
  nv = dot3(n, v3);
  nn = dot3(n, n);
  scl3(res, n, nv / nn);
  return 0;
}

// G:521-529  res = origin projected onto line defined by v1, v2
HIPGL_DEVICE inline void projectOriginLine(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3]) {
  MJCCD_FP_CONTRACT_OFF
  // res = v2 - <v2, v2 - v1> / <v2 - v1, v2 - v1> * (v2 - v1)
  mjtNum diff[3];
  sub3(diff, v2, v1);
  mjtNum scl = -(dot3(v2, diff) / dot3(diff, diff));
  res[0] = v2[0] + scl*diff[0];
  res[1] = v2[1] + scl*diff[1];
  res[2] = v2[2] + scl*diff[2];
}

// G:533-537  return 1 if both numbers are positive, -1 if both negative and 0 otherwise
HIPGL_DEVICE inline int sameSign2(mjtNum a, mjtNum b) {
  MJCCD_FP_CONTRACT_OFF
  if (a > 0 && b > 0) return 1;
  if (a < 0 && b < 0) return -1;
  return 0;
}

// G:543-563  subdistance algorithm (Montanari et al, ToG 2017)
HIPGL_DEVICE inline void subdistance(mjtNum lambda[4], int n, const Vertex simplex[4]) {
  MJCCD_FP_CONTRACT_OFF
  lambda[0] = 0, lambda[1] = 0, lambda[2] = 0, lambda[3] = 0;  // memset
  const mjtNum* s1 = simplex[0].vert;
  const mjtNum* s2 = simplex[1].vert;
  const mjtNum* s3 = simplex[2].vert;
  const mjtNum* s4 = simplex[3].vert;

  switch (n) {
  case 4:
     S3D(lambda, s1, s2, s3, s4);
     break;
  case 3:
     S2D(lambda, s1, s2, s3);
      break;
  case 2:
     S1D(lambda, s1, s2);
     break;
  default:
    lambda[0] = 1;
    break;
  }
}

// G:567-655
HIPGL_DEVICE inline void S3D(mjtNum lambda[4], const mjtNum s1[3], const mjtNum s2[3],
                             const mjtNum s3[3], const mjtNum s4[3]) {
  MJCCD_FP_CONTRACT_OFF
  // compute cofactors to find det(M)
  mjtNum C41 = -det3(s2, s3, s4);
  mjtNum C42 =  det3(s1, s3, s4);
  mjtNum C43 = -det3(s1, s2, s4);
  mjtNum C44 =  det3(s1, s2, s3);

  // note that m_det = 6*SignVol(simplex) with C4i corresponding to the volume of the 3-simplex
  // with vertices {s1, s2, s3, 0} - si
  mjtNum m_det = C41 + C42 + C43 + C44;

  int comp1 = sameSign2(m_det, C41),
      comp2 = sameSign2(m_det, C42),
      comp3 = sameSign2(m_det, C43),
      comp4 = sameSign2(m_det, C44);

  // if all signs are the same then the origin is inside the simplex
  if (comp1 && comp2 && comp3 && comp4) {
    lambda[0] = C41 / m_det;
    lambda[1] = C42 / m_det;
    lambda[2] = C43 / m_det;
    lambda[3] = C44 / m_det;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s2, s3, s4);
    lincomb(x, lambda_2d, 3, s2, s3, s4, 0);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_2d[0];
    lambda[2] = lambda_2d[1];
    lambda[3] = lambda_2d[2];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s3, s4);
    lincomb(x, lambda_2d, 3, s1, s3, s4, 0);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = 0;
      lambda[2] = lambda_2d[1];
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s4);
    lincomb(x, lambda_2d, 3, s1, s2, s4, 0);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = 0;
      lambda[3] = lambda_2d[2];
      dmin = d;
    }
  }

  if (!comp4) {
    mjtNum lambda_2d[3], x[3];
    S2D(lambda_2d, s1, s2, s3);
    lincomb(x, lambda_2d, 3, s1, s2, s3, 0);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_2d[0];
      lambda[1] = lambda_2d[1];
      lambda[2] = lambda_2d[2];
    }
  }
}

// G:658-788
HIPGL_DEVICE inline void S2D(mjtNum lambda[3], const mjtNum s1[3], const mjtNum s2[3],
                             const mjtNum s3[3]) {
  MJCCD_FP_CONTRACT_OFF
  // project origin onto affine hull of the simplex
  mjtNum p_o[3];
  if (projectOriginPlane(p_o, s1, s2, s3)) {
    S1D(lambda, s1, s2);
    lambda[2] = 0;
    return;
  }

  // Below are the minors M_i4 of the matrix M
  mjtNum M_14 = s2[1]*s3[2] - s2[2]*s3[1] - s1[1]*s3[2] + s1[2]*s3[1] + s1[1]*s2[2] - s1[2]*s2[1];
  mjtNum M_24 = s2[0]*s3[2] - s2[2]*s3[0] - s1[0]*s3[2] + s1[2]*s3[0] + s1[0]*s2[2] - s1[2]*s2[0];
  mjtNum M_34 = s2[0]*s3[1] - s2[1]*s3[0] - s1[0]*s3[1] + s1[1]*s3[0] + s1[0]*s2[1] - s1[1]*s2[0];

  // exclude the axis with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  mjtNum s1_2D[2], s2_2D[2], s3_2D[2], p_o_2D[2];
  mjtNum mu1 = mju_abs(M_14), mu2 = mju_abs(M_24), mu3 = mju_abs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    s1_2D[0] = s1[1];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[1];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[1];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[1];
    p_o_2D[1] = p_o[2];
  } else if (mu2 >= mu3) {
    M_max = M_24;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[2];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[2];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[2];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[2];
  } else {
    M_max = M_34;
    s1_2D[0] = s1[0];
    s1_2D[1] = s1[1];

    s2_2D[0] = s2[0];
    s2_2D[1] = s2[1];

    s3_2D[0] = s3[0];
    s3_2D[1] = s3[1];

    p_o_2D[0] = p_o[0];
    p_o_2D[1] = p_o[1];
  }

  // C31 corresponds to the signed area of 2-simplex: (p_o_2D, s2_2D, s3_2D)
  mjtNum C31 = p_o_2D[0]*s2_2D[1] + p_o_2D[1]*s3_2D[0] + s2_2D[0]*s3_2D[1]
             - p_o_2D[0]*s3_2D[1] - p_o_2D[1]*s2_2D[0] - s3_2D[0]*s2_2D[1];

  // C32 corresponds to the signed area of 2-simplex: (_po_2D, s1_2D, s3_2D)
  mjtNum C32 = p_o_2D[0]*s3_2D[1] + p_o_2D[1]*s1_2D[0] + s3_2D[0]*s1_2D[1]
             - p_o_2D[0]*s1_2D[1] - p_o_2D[1]*s3_2D[0] - s1_2D[0]*s3_2D[1];

  // C33 corresponds to the signed area of 2-simplex: (p_o_2D, s1_2D, s2_2D)
  mjtNum C33 = p_o_2D[0]*s1_2D[1] + p_o_2D[1]*s2_2D[0] + s1_2D[0]*s2_2D[1]
             - p_o_2D[0]*s2_2D[1] - p_o_2D[1]*s1_2D[0] - s2_2D[0]*s1_2D[1];

  int comp1 = sameSign2(M_max, C31),
      comp2 = sameSign2(M_max, C32),
      comp3 = sameSign2(M_max, C33);

  // all the same sign, p_o is inside the 2-simplex
  if (comp1 && comp2 && comp3) {
    lambda[0] = C31 / M_max;
    lambda[1] = C32 / M_max;
    lambda[2] = C33 / M_max;
    return;
  }

  // find the smallest distance, and use the corresponding barycentric coordinates
  mjtNum dmin = mjMAX_LIMIT;

  if (!comp1) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s2, s3);
    lincomb(x, lambda_1d, 2, s2, s3, 0, 0);
    mjtNum d = dot3(x, x);
    lambda[0] = 0;
    lambda[1] = lambda_1d[0];
    lambda[2] = lambda_1d[1];
    dmin = d;
  }

  if (!comp2) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s3);
    lincomb(x, lambda_1d, 2, s1, s3, 0, 0);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = 0;
      lambda[2] = lambda_1d[1];
      dmin = d;
    }
  }

  if (!comp3) {
    mjtNum lambda_1d[2], x[3];
    S1D(lambda_1d, s1, s2);
    lincomb(x, lambda_1d, 2, s1, s2, 0, 0);
    mjtNum d = dot3(x, x);
    if (d < dmin) {
      lambda[0] = lambda_1d[0];
      lambda[1] = lambda_1d[1];
      lambda[2] = 0;
    }
  }
}

// G:791-820
HIPGL_DEVICE inline void S1D(mjtNum lambda[2], const mjtNum s1[3], const mjtNum s2[3]) {
  MJCCD_FP_CONTRACT_OFF
  // find projection of origin onto the 1-simplex:
  mjtNum p_o[3];
  projectOriginLine(p_o, s1, s2);

  // find the axis with the largest projection "shadow" of the simplex
  mjtNum mu = s1[0] - s2[0];
  mjtNum mu_max = mu;
  int index = 0;

  mu = s1[1] - s2[1];
  if (mju_abs(mu) >= mju_abs(mu_max)) {
    mu_max = mu;
    index = 1;
  }

  mu = s1[2] - s2[2];
  if (mju_abs(mu) >= mju_abs(mu_max)) {
    mu_max = mu;
    index = 2;
  }

  mjtNum C1 = p_o[index] - s2[index];
  mjtNum C2 = s1[index] - p_o[index];

  // determine if projection of origin lies inside 1-simplex
  int same = sameSign2(mu_max, C1) && sameSign2(mu_max, C2);
  lambda[0] = same ? C1 / mu_max : 0;
  lambda[1] = same ? C2 / mu_max : 1;
}

// G:177-307  GJK algorithm
template <int PK = 0>
HIPGL_DEVICE __attribute__((always_inline)) inline void gjk(Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  int get_dist = PK ? 0 : status->dist_cutoff > 0;  // need to recover geom distances if not in contact
  int backup_gjk = !get_dist;              // use gjkIntersect if no geom distances needed
  Vertex* simplex = status->simplex;
  int n = 0;                               // number of vertices in the simplex
  int k = 0;                               // current iteration
  int kmax = status->max_iterations;       // max number of iterations
  mjtNum* x1_k = status->x1;               // the kth approximation point for obj1
  mjtNum* x2_k = status->x2;               // the kth approximation point for obj2
  mjtNum x_k[3];                           // the kth approximation point in Minkowski difference
  mjtNum lambda[4] = {1, 0, 0, 0};         // barycentric coordinates for x_k
  mjtNum cutoff2 = status->dist_cutoff * status->dist_cutoff;
  mjtNum tol2 = status->tolerance * status->tolerance;

  // if both geoms are discrete, finite convergence is guaranteed; set tolerance to 0
  mjtNum epsilon = discreteGeoms(obj1, obj2) ? 0 : 0.5 * tol2;

  // tolerance on squared norm of x_k
  mjtNum min_norm2 = discreteGeoms(obj1, obj2) ? mjMINVAL2 : tol2;
  mjtNum x_norm;

  // set initial guess
  sub3(x_k, x1_k, x2_k);

  for (; k < kmax; k++) {
    // in tolerance for geoms to be in contact
    if ((x_norm = dot3(x_k, x_k)) < min_norm2) {
      break;
    }
    x_norm = mju_sqrt(x_norm);

    // compute the kth support point
    gjkSupport<PK>(simplex + n, obj1, obj2, x_k, x_norm);
    mjtNum *s_k = simplex[n].vert;

    // stopping criteria using the Frank-Wolfe duality gap given by
    //  |f(x_k) - f(x_min)|^2 <= < grad f(x_k), (x_k - s_k) >
    mjtNum diff[3];
    sub3(diff, x_k, s_k);
    if (dot3(x_k, diff) < epsilon) {
      if (!k) n = 1;
      break;
    }

    // if the hyperplane separates the Minkowski difference and origin, the objects don't collide
    // if geom distance isn't requested, return early
    if (!get_dist) {
      if (dot3(x_k, s_k) > 0) {
        status->gjk_iterations = k;
        status->nsimplex = 0;
        status->nx = 0;
        status->dist = mjMAX_LIMIT;
        return;
      }
    } else if (status->dist_cutoff < mjMAX_LIMIT) {
      mjtNum vs = dot3(x_k, s_k), vv = dot3(x_k, x_k);
      if (dot3(x_k, s_k) > 0 && (vs*vs / vv) >= cutoff2) {
        status->gjk_iterations = k;
        status->nsimplex = 0;
        status->nx = 0;
        status->dist = mjMAX_LIMIT;
        return;
      }
    }

    // tetrahedron is generated and only need contact info; fallback to gjkIntersect to
    // determine contact
    if (n == 3 && backup_gjk) {
      status->gjk_iterations = k;
      const long long _mjp_ti = MJP_T();
      int ret = gjkIntersect<PK>(status, obj1, obj2);
      MJP_ADD(3, MJP_T() - _mjp_ti);
      if (ret != -1) {
        status->nx = 0;
        status->dist = ret > 0 ? 0 : mjMAX_LIMIT;
        return;
      }
      k = status->gjk_iterations;
      backup_gjk = 0;
    }

    // run the distance subalgorithm to compute the barycentric coordinates
    // of the closest point to the origin in the simplex
    subdistance(lambda, n + 1, simplex);

    // remove vertices from the simplex no longer needed
    n = 0;
    for (int i = 0; i < 4; i++) {
      if (!lambda[i]) continue;
      if (n != i) simplex[n] = simplex[i];  // port: skip the no-op self copy
      lambda[n++] = lambda[i];
    }

    // SHOULD NOT OCCUR
    if (n < 1) {
      status->gjk_iterations = k;
      status->nsimplex = 0;
      status->nx = 0;
      status->dist = mjMAX_LIMIT;
      return;
    }

    // get the next iteration of x_k
    mjtNum x_next[3];
    lincomb(x_next, lambda, n, simplex[0].vert, simplex[1].vert,
            simplex[2].vert, simplex[3].vert);

    // x_k has converged to minimum
    if (equal3(x_next, x_k)) {
      break;
    }

    // copy next iteration into x_k
    copy3(x_k, x_next);

    // we have a tetrahedron containing the origin so return early
    if (n == 4) {
      x_norm = 0;
      break;
    }
  }

  // compute the approximate witness points
  lincomb(x1_k, lambda, n, simplex[0].vert1, simplex[1].vert1, simplex[2].vert1,
          simplex[3].vert1);
  lincomb(x2_k, lambda, n, simplex[0].vert2, simplex[1].vert2, simplex[2].vert2,
          simplex[3].vert2);

  status->nx = 1;
  status->gjk_iterations = k;
  status->nsimplex = n;
  status->dist = x_norm;
}

// ---------------------------------------- EPA ------------------------------------------------

template <int PK = 0>
HIPGL_DEVICE inline int polytope3(Polytope* pt, Status* status, Obj* obj1, Obj* obj2);

// G:1170-1174  make a copy of vertex in polytope and return its index
HIPGL_DEVICE inline int insertVertex(Polytope* pt, const Vertex* v) {
  MJCCD_FP_CONTRACT_OFF
  int n = pt->nverts++;
  pt->verts[n] = *v;
  return n;
}

// G:1178-1184  delete face from map
HIPGL_DEVICE inline void deleteFace(Polytope* pt, Face* face) {
  MJCCD_FP_CONTRACT_OFF
  if (face->index >= 0) {
    pt->map[face->index] = pt->map[--pt->nmap];
    pt->faces[pt->map[face->index]].index = face->index;
  }
  face->index = -2;  // mark face as deleted from map and polytope
}

// G:1188-1190  return max number of faces that can be stored in polytope
HIPGL_DEVICE inline int maxFaces(Polytope* pt) {
  MJCCD_FP_CONTRACT_OFF
  return pt->maxfaces - pt->nfaces;
}

// G:1194-1213  attach a face to the polytope; return squared distance to origin
HIPGL_DEVICE inline mjtNum attachFace(Polytope* pt, int v1, int v2, int v3,
                                      int adj1, int adj2, int adj3) {
  MJCCD_FP_CONTRACT_OFF
  Face* face = &pt->faces[pt->nfaces++];
  face->verts = v1 + (v2 << 10) + (v3 << 20);

  // adjacent faces
  face->adj[0] = adj1;
  face->adj[1] = adj2;
  face->adj[2] = adj3;

  // compute witness point v
  int ret = projectOriginPlane(face->v, pt->verts[v3].vert, pt->verts[v2].vert, pt->verts[v1].vert);
  if (ret) {
    return 0;
  }
  face->dist2 = dot3(face->v, face->v);
  face->index = -1;

  return face->dist2;
}

// G:826-837  replace a 3-simplex with one of its faces
HIPGL_DEVICE inline void replaceSimplex3(Polytope* pt, Status* status, int v1, int v2, int v3) {
  MJCCD_FP_CONTRACT_OFF
  // reset status simplex
  status->nsimplex = 3;
  status->simplex[0] = pt->verts[v1];
  status->simplex[1] = pt->verts[v2];
  status->simplex[2] = pt->verts[v3];

  // reset polytope
  pt->nfaces = 0;
  pt->nverts = 0;
  pt->nmap = 0;
}

// G:841-857  return 1 if the origin and p3 are on the same side of the plane p0, p1, p2
HIPGL_DEVICE inline int sameSide(const mjtNum p0[3], const mjtNum p1[3],
                                 const mjtNum p2[3], const mjtNum p3[3]) {
  MJCCD_FP_CONTRACT_OFF
    mjtNum diff1[3], diff2[3], diff3[3], diff4[3], n[3];
    sub3(diff1, p1, p0);
    sub3(diff2, p2, p0);
    cross3(n, diff1, diff2);

    sub3(diff3, p3, p0);
    mjtNum dot1 = dot3(n, diff3);

    scl3(diff4, p0, -1);
    mjtNum dot2 = dot3(n, diff4);
    if (dot1 > 0 && dot2 > 0) return 1;
    if (dot1 < 0 && dot2 < 0) return 1;
    return 0;
}

// G:860-866  return 1 if the origin is contained in the tetrahedron, 0 otherwise
HIPGL_DEVICE inline int testTetra(const mjtNum p0[3], const mjtNum p1[3],
                                  const mjtNum p2[3], const mjtNum p3[3]) {
  MJCCD_FP_CONTRACT_OFF
  return sameSide(p0, p1, p2, p3)
      && sameSide(p1, p2, p3, p0)
      && sameSide(p2, p3, p0, p1)
      && sameSide(p3, p0, p1, p2);
}

// G:870-885  matrix for 120 degrees rotation around given axis
HIPGL_DEVICE inline void rotmat(mjtNum R[9], const mjtNum axis[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum n = norm3(axis);
  mjtNum u1 = axis[0] / n, u2 = axis[1] / n, u3 = axis[2] / n;
  const mjtNum sin120 = 0.86602540378;  // sin(120 deg)
  const mjtNum cos120 = -0.5;           // cos(120 deg)
  R[0] = cos120 + u1*u1*(1 - cos120);
  R[1] = u1*u2*(1 - cos120) - u3*sin120;
  R[2] = u1*u3*(1 - cos120) + u2*sin120;
  R[3] = u2*u1*(1 - cos120) + u3*sin120;
  R[4] = cos120 + u2*u2*(1 - cos120);
  R[5] = u2*u3*(1 - cos120) - u1*sin120;
  R[6] = u1*u3*(1 - cos120) - u2*sin120;
  R[7] = u2*u3*(1 - cos120) + u1*sin120;
  R[8] = cos120 + u3*u3*(1 - cos120);
}

// G:888-903  return nonzero if the ray v1v2 intersects the triangle v3v4v5
HIPGL_DEVICE inline int rayTriangle(const mjtNum v1[3], const mjtNum v2[3], const mjtNum v3[3],
                                    const mjtNum v4[3], const mjtNum v5[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum diff12[3], diff13[3], diff14[3], diff15[3];
  sub3(diff12, v2, v1);
  sub3(diff13, v3, v1);
  sub3(diff14, v4, v1);
  sub3(diff15, v5, v1);

  mjtNum vol1 = det3(diff13, diff14, diff12);
  mjtNum vol2 = det3(diff14, diff15, diff12);
  mjtNum vol3 = det3(diff15, diff13, diff12);

  if (vol1 >= 0 && vol2 >= 0 && vol3 >= 0) return 1;
  if (vol1 <= 0 && vol2 <= 0 && vol3 <= 0) return -1;
  return 0;
}

// G:907-987  create a polytope from a 1-simplex (returns 0 on success)
template <int PK = 0>
HIPGL_DEVICE inline int polytope2(Polytope* pt, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum *v1 = status->simplex[0].vert, *v2 = status->simplex[1].vert;

  mjtNum diff[3];
  sub3(diff, v2, v1);

  // find component with smallest magnitude (so cross product is largest)
  mjtNum value = mjMAX_LIMIT;
  int index = 0;
  for (int i = 0; i < 3; i++) {
    if (mju_abs(diff[i]) < value) {
      value = mju_abs(diff[i]);
      index = i;
    }
  }

  // cross product with best coordinate axis
  mjtNum e[3] = {0, 0, 0};
  e[index] = 1;
  mjtNum d1[3], d2[3], d3[3];
  cross3(d1, e, diff);

  // rotate around the line segment to get three more points spaced 120 degrees apart
  mjtNum R[9];
  rotmat(R, diff);

  mju_mulMatVec3(d2, R, d1);
  mju_mulMatVec3(d3, R, d2);

  // save vertices and get indices for each one
  int v1i = insertVertex(pt, status->simplex + 0);
  int v2i = insertVertex(pt, status->simplex + 1);
  int v3i = epaSupport<PK>(pt, obj1, obj2, d1, norm3(d1));
  int v4i = epaSupport<PK>(pt, obj1, obj2, d2, norm3(d2));
  int v5i = epaSupport<PK>(pt, obj1, obj2, d3, norm3(d3));

  mjtNum* v3 = pt->verts[v3i].vert;
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // build hexahedron
  if (attachFace(pt, v1i, v3i, v4i, 1, 3, 2) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1i, v3i, v4i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v1i, v5i, v3i, 2, 4, 0) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1i, v5i, v3i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v1i, v4i, v5i, 0, 5, 1) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1i, v4i, v5i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v2i, v4i, v3i, 5, 0, 4) < mjMINVAL2) {
    replaceSimplex3(pt, status, v2i, v4i, v3i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v2i, v3i, v5i, 3, 1, 5) < mjMINVAL2) {
    replaceSimplex3(pt, status, v2i, v3i, v5i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v2i, v5i, v4i, 4, 2, 3) < mjMINVAL2) {
    replaceSimplex3(pt, status, v2i, v5i, v4i);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }

  // check hexahedron is convex
  if (!rayTriangle(v1, v2, v3, v4, v5)) {
    return mjEPA_P2_NONCONVEX;
  }

  for (int i = 0; i < 6; i++) {
    pt->map[i] = i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;

  // valid hexahedron for EPA
  return 0;
}

// G:990-1031  compute the affine coordinates of p on the triangle v1v2v3
HIPGL_DEVICE inline void triAffineCoord(mjtNum lambda[3], const mjtNum v1[3], const mjtNum v2[3],
                                        const mjtNum v3[3], const mjtNum p[3]) {
  MJCCD_FP_CONTRACT_OFF
  // compute minors as in S2D
  mjtNum M_14 = v2[1]*v3[2] - v2[2]*v3[1] - v1[1]*v3[2] + v1[2]*v3[1] + v1[1]*v2[2] - v1[2]*v2[1];
  mjtNum M_24 = v2[0]*v3[2] - v2[2]*v3[0] - v1[0]*v3[2] + v1[2]*v3[0] + v1[0]*v2[2] - v1[2]*v2[0];
  mjtNum M_34 = v2[0]*v3[1] - v2[1]*v3[0] - v1[0]*v3[1] + v1[1]*v3[0] + v1[0]*v2[1] - v1[1]*v2[0];

  // exclude one of the axes with the largest projection of the simplex using the computed minors
  mjtNum M_max = 0;
  int x, y;
  mjtNum mu1 = mju_abs(M_14), mu2 = mju_abs(M_24), mu3 = mju_abs(M_34);
  if (mu1 >= mu2 && mu1 >= mu3) {
    M_max = M_14;
    x = 1;
    y = 2;
  } else if (mu2 >= mu3) {
    M_max = M_24;
    x = 0;
    y = 2;
  } else {
    M_max = M_34;
    x = 0;
    y = 1;
  }

  // C31 corresponds to the signed area of 2-simplex: (v, s2, s3)
  mjtNum C31 = p[x]*v2[y] + p[y]*v3[x] + v2[x]*v3[y]
             - p[x]*v3[y] - p[y]*v2[x] - v3[x]*v2[y];

  // C32 corresponds to the signed area of 2-simplex: (v, s1, s3)
  mjtNum C32 = p[x]*v3[y] + p[y]*v1[x] + v3[x]*v1[y]
             - p[x]*v1[y] - p[y]*v3[x] - v1[x]*v3[y];

  // C33 corresponds to the signed area of 2-simplex: (v, s1, s2)
  mjtNum C33 = p[x]*v1[y] + p[y]*v2[x] + v1[x]*v2[y]
             - p[x]*v2[y] - p[y]*v1[x] - v2[x]*v1[y];

  // compute affine coordinates
  lambda[0] = C31 / M_max;
  lambda[1] = C32 / M_max;
  lambda[2] = C33 / M_max;
}

// G:1035-1048  return true if point p and triangle v1v2v3 intersect
HIPGL_DEVICE inline int triPointIntersect(const mjtNum v1[3], const mjtNum v2[3],
                                          const mjtNum v3[3], const mjtNum p[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum lambda[3];
  triAffineCoord(lambda, v1, v2, v3, p);
  if (lambda[0] < 0 || lambda[1] < 0 || lambda[2] < 0) {
    return 0;
  }
  mjtNum pr[3], diff[3];
  pr[0] = v1[0]*lambda[0] + v2[0]*lambda[1] + v3[0]*lambda[2];
  pr[1] = v1[1]*lambda[0] + v2[1]*lambda[1] + v3[1]*lambda[2];
  pr[2] = v1[2]*lambda[0] + v2[2]*lambda[1] + v3[2]*lambda[2];
  sub3(diff, pr, p);
  return norm3(diff) < mjMINVAL;
}

// G:1052-1127  create a polytope from a 2-simplex (returns 0 on success)
template <int PK>
HIPGL_DEVICE inline int polytope3(Polytope* pt, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  // get vertices of simplex from GJK
  const mjtNum *v1 = status->simplex[0].vert,
               *v2 = status->simplex[1].vert,
               *v3 = status->simplex[2].vert;

  // get normals in both directions
  mjtNum diff1[3], diff2[3], n[3], n_neg[3];
  sub3(diff1, v2, v1);
  sub3(diff2, v3, v1);
  cross3(n, diff1, diff2);
  mjtNum n_norm = norm3(n);
  if (n_norm < mjMINVAL) {
    return mjEPA_P3_BAD_NORMAL;
  }

  // negative of triangle normal n
  scl3(n_neg, n, -1);

  // save vertices and get indices for each one
  int v1i = insertVertex(pt, status->simplex + 0);
  int v2i = insertVertex(pt, status->simplex + 1);
  int v3i = insertVertex(pt, status->simplex + 2);
  int v5i = epaSupport<PK>(pt, obj1, obj2, n_neg, n_norm);
  int v4i = epaSupport<PK>(pt, obj1, obj2, n, n_norm);
  mjtNum* v4 = pt->verts[v4i].vert;
  mjtNum* v5 = pt->verts[v5i].vert;

  // check that v4 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v4)) {
    return mjEPA_P3_INVALID_V4;
  }

  // check that v5 is not contained in the 2-simplex
  if (triPointIntersect(v1, v2, v3, v5)) {
    return mjEPA_P3_INVALID_V5;
  }

  // if origin does not lie on simplex then we need to check that the hexahedron contains the
  // origin
  if (status->dist > 10*mjMINVAL && !testTetra(v1, v2, v3, v4) && !testTetra(v1, v2, v3, v5)) {
    return mjEPA_P3_MISSING_ORIGIN;
  }

  // create hexahedron for EPA
  if (attachFace(pt, v4i, v1i, v2i, 1, 3, 2) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v3i, v1i, 2, 4, 0) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v4i, v2i, v3i, 0, 5, 1) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v2i, v1i, 5, 0, 4) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v1i, v3i, 3, 1, 5) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }
  if (attachFace(pt, v5i, v3i, v2i, 4, 2, 3) < mjMINVAL2) {
    return mjEPA_P3_ORIGIN_ON_FACE;
  }

  // populate face map
  for (int i = 0; i < 6; i++) {
    pt->map[i] = i;
    pt->faces[i].index = i;
  }
  pt->nmap = 6;
  return 0;
}

// G:1131-1166  create a polytope from a 3-simplex (returns 0 on success)
template <int PK = 0>
HIPGL_DEVICE inline int polytope4(Polytope* pt, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  int v1 = insertVertex(pt, status->simplex + 0);
  int v2 = insertVertex(pt, status->simplex + 1);
  int v3 = insertVertex(pt, status->simplex + 2);
  int v4 = insertVertex(pt, status->simplex + 3);

  // if the origin is on a face, replace the 3-simplex with a 2-simplex
  if (attachFace(pt, v1, v2, v3, 1, 3, 2) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1, v2, v3);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v1, v4, v2, 2, 3, 0) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1, v4, v2);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v1, v3, v4, 0, 3, 1) < mjMINVAL2) {
    replaceSimplex3(pt, status, v1, v3, v4);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }
  if (attachFace(pt, v4, v3, v2, 2, 0, 1) < mjMINVAL2) {
    replaceSimplex3(pt, status, v4, v3, v2);
    return MJCCD_EPA_TO_P3;  // port: caller runs polytope3 (one inlined copy)
  }

  if (!testTetra(pt->verts[v1].vert, pt->verts[v2].vert, pt->verts[v3].vert, pt->verts[v4].vert)) {
    return mjEPA_P4_MISSING_ORIGIN;
  }

  // populate face map
  for (int i = 0; i < 4; i++) {
    pt->map[i] = i;
    pt->faces[i].index = i;
  }
  pt->nmap = 4;
  return 0;
}

// G:1217-1220  add an edge to the horizon (port: bounds checked)
HIPGL_DEVICE inline void addEdge(Polytope* pt, int index, int edge) {
  MJCCD_FP_CONTRACT_OFF
  if (pt->horizon.nedges >= MJCCD_HORIZON_MAX) {
    pt->status->flags |= MJCCD_FLAG_HORIZON_OVF;
    pt->horizon.nedges++;  // keep the count MuJoCo would have (forces the maxFaces check)
    return;
  }
  pt->horizon.edges[pt->horizon.nedges] = edge;
  pt->horizon.indices[pt->horizon.nedges++] = index;
}

// G:1224-1229  get edge index where vertex lies
HIPGL_DEVICE inline int getEdge(Face* face, int vertex) {
  MJCCD_FP_CONTRACT_OFF
  int verts[3];
  EPA_VERT_EXPAND(verts, face->verts);
  if (verts[0] == vertex) return 0;
  if (verts[1] == vertex) return 1;
  return 2;
}

// G:1233-1253  horizonRec, iterative.  Return 1 if face is visible from w otherwise 0.
// Visit order is identical to the recursion: test visibility, EPA_VERT_EXPAND, deleteFace,
// then for k = 1, 2: if the neighbour is live, descend into it fully; if the neighbour was
// not visible, addEdge after returning.
HIPGL_DEVICE inline int horizonRec(Polytope* pt, Face* face, int e) {
  MJCCD_FP_CONTRACT_OFF
  // v is visible from w so it is deleted and adjacent faces are checked
  if (!(dot3(face->v, pt->horizon.w) - face->dist2 > mjMINVAL)) {
    return 0;
  }
  HorizonFrame* stack = pt->stack;
  int sp = 0;
  {
    HorizonFrame* fr = &stack[sp++];
    fr->face = (int)(face - pt->faces);
    fr->e = e;
    EPA_VERT_EXPAND(fr->verts, face->verts);
    fr->k = 1;
    deleteFace(pt, face);
  }

  while (sp > 0) {
    HorizonFrame* fr = &stack[sp - 1];
    if (fr->k >= 3) {
      // child call finished: it returned 1 (visible), so the parent adds no edge
      sp--;
      if (sp > 0) stack[sp - 1].k++;
      continue;
    }

    // recursively search the adjacent faces on the next two edges
    Face* cur = &pt->faces[fr->face];
    int i = (fr->e + fr->k) % 3;
    Face* adjFace = &pt->faces[cur->adj[i]];
    if (adjFace->index > -2) {
      int adjEdge = getEdge(adjFace, fr->verts[(i + 1) % 3]);
      if (dot3(adjFace->v, pt->horizon.w) - adjFace->dist2 > mjMINVAL) {
        // descend: child is visible
        HorizonFrame* ch = &stack[sp++];
        ch->face = cur->adj[i];
        ch->e = adjEdge;
        EPA_VERT_EXPAND(ch->verts, adjFace->verts);
        ch->k = 1;
        deleteFace(pt, adjFace);
        continue;
      }
      addEdge(pt, cur->adj[i], adjEdge);
    }
    fr->k++;
  }
  return 1;
}

// G:1257-1281  create horizon given the face as starting point
HIPGL_DEVICE inline void horizon(Polytope* pt, Face* face) {
  MJCCD_FP_CONTRACT_OFF
  deleteFace(pt, face);
  int verts[3];
  EPA_VERT_EXPAND(verts, face->verts);

  // first edge
  Face* adjFace = &pt->faces[face->adj[0]];
  int adjEdge = getEdge(adjFace, verts[1]);
  if (!horizonRec(pt, adjFace, adjEdge)) {
    addEdge(pt, face->adj[0], adjEdge);
  }

  // second edge
  adjFace = &pt->faces[face->adj[1]];
  adjEdge = getEdge(adjFace, verts[2]);
  if (adjFace->index > -2 && !horizonRec(pt, adjFace, adjEdge)) {
    addEdge(pt, face->adj[1], adjEdge);
  }

  // third edge
  adjFace = &pt->faces[face->adj[2]];
  adjEdge = getEdge(adjFace, verts[0]);
  if (adjFace->index > -2 && !horizonRec(pt, adjFace, adjEdge)) {
    addEdge(pt, face->adj[2], adjEdge);
  }
}

#if defined(__HIP_DEVICE_COMPILE__)
// port-only (wave_coop): visibility of LDS/global writes made by other lanes of this wave
HIPGL_DEVICE __forceinline__ void waveSync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#ifndef MJCCD_HZN_REG
#define MJCCD_HZN_REG 1
#endif
// port-only (wave_coop, nfaces <= 64): horizon() with the polytope's face graph, live flags,
// face->index and map held in VGPR lanes (lane f = face f / map entry f) and the visit stack and
// the horizon edges in VGPR lanes.  The visibility test of every face,
// dot3(face->v, w) - face->dist2 > mjMINVAL, is evaluated lane-parallel up front (same
// arithmetic; v, dist2 and w do not change during horizon), so the depth-first walk is a scalar
// readlane chain.  The walk visits faces, deletes them (deleteFace's swap-remove on the lane
// copies of map / index) and adds horizon edges in exactly the recursion's order; the lane copies
// of map[0, nmap), index and the edges are written back at the end.
HIPGL_DEVICE __forceinline__ int hznRl(int v, int i) { return __builtin_amdgcn_readlane(v, i); }

HIPGL_DEVICE inline void horizonCoop(Polytope* pt, Face* face) {
  MJCCD_FP_CONTRACT_OFF
  const int lane = __lane_id();
  // all walk state is wave-uniform: keep it scalar
  const int nfaces = __builtin_amdgcn_readfirstlane(pt->nfaces);
  int nmap = __builtin_amdgcn_readfirstlane(pt->nmap);
  Face* faces = pt->faces;
  int* map = pt->map;
  mjtNum w[3];
  w[0] = pt->horizon.w[0]; w[1] = pt->horizon.w[1]; w[2] = pt->horizon.w[2];

  const bool ok = lane < nfaces;
  const Face* F = &faces[ok ? lane : 0];
  // adjacent faces packed in bytes (nfaces <= 64)
  const int ADJ = F->adj[0] | (F->adj[1] << 8) | (F->adj[2] << 16);
  const int VT = F->verts;
  int IX = F->index;
  int MP = map[lane < nmap ? lane : 0];
  unsigned long long vis, alive;
  {
    mjtNum fv[3] = {F->v[0], F->v[1], F->v[2]};
    const mjtNum d2 = F->dist2;
    vis = __ballot(ok && dot3(fv, w) - d2 > mjMINVAL);
    alive = __ballot(ok && IX > -2);
  }
  int STK = 0, HE = 0, ne = 0;

  auto vert = [](int verts, int k) { return (verts >> (10 * k)) & 0x3FF; };
  auto edgeOf = [&](int verts, int vertex) {  // getEdge
    return vert(verts, 0) == vertex ? 0 : (vert(verts, 1) == vertex ? 1 : 2);
  };
  auto del = [&](int f) {  // deleteFace
    const int idx = hznRl(IX, f);
    if (idx >= 0) {
      const int m = hznRl(MP, --nmap);
      MP = lane == idx ? m : MP;
      IX = lane == m ? idx : IX;
    }
    IX = lane == f ? -2 : IX;
    alive &= ~(1ull << f);
  };
  auto addEdge2 = [&](int a, int ae) {  // addEdge (edges kept in lanes until the write-back)
    HE = lane == ne ? (a | (ae << 8)) : HE;
    ne++;
  };
  // horizonRec(pt, faces + f0, e0); returns its result
  auto rec = [&](int f0, int e0) -> int {
    if (!((vis >> f0) & 1ull)) return 0;
    int sp = 0;
    int cf = f0, ce = e0, ck = 1, cv = hznRl(VT, f0), ca = hznRl(ADJ, f0);
    del(f0);
    for (;;) {
      if (ck >= 3) {
        // child finished (visible): the parent adds no edge
        if (sp == 0) break;
        const int fr = hznRl(STK, --sp);
        cf = fr & 0xFF; ce = (fr >> 8) & 3; ck = (fr >> 10) + 1;
        cv = hznRl(VT, cf); ca = hznRl(ADJ, cf);
        continue;
      }
      int i = ce + ck;
      if (i >= 3) i -= 3;
      const int a = (ca >> (8 * i)) & 0xFF;
      if ((alive >> a) & 1ull) {
        const int av = hznRl(VT, a);
        const int ae = edgeOf(av, vert(cv, i == 2 ? 0 : i + 1));
        if ((vis >> a) & 1ull) {
          STK = lane == sp ? (cf | (ce << 8) | (ck << 10)) : STK;
          sp++;
          cf = a; ce = ae; ck = 1; cv = av; ca = hznRl(ADJ, a);
          del(a);
          continue;
        }
        addEdge2(a, ae);
      }
      ck++;
    }
    return 1;
  };

  const int f0 = __builtin_amdgcn_readfirstlane((int)(face - faces));
  const int v0 = hznRl(VT, f0), a0 = hznRl(ADJ, f0);
  del(f0);
  {
    const int a = a0 & 0xFF;
    const int ae = edgeOf(hznRl(VT, a), vert(v0, 1));
    if (!rec(a, ae)) addEdge2(a, ae);
  }
  {
    const int a = (a0 >> 8) & 0xFF;
    const int ae = edgeOf(hznRl(VT, a), vert(v0, 2));
    if (((alive >> a) & 1ull) && !rec(a, ae)) addEdge2(a, ae);
  }
  {
    const int a = (a0 >> 16) & 0xFF;
    const int ae = edgeOf(hznRl(VT, a), vert(v0, 0));
    if (((alive >> a) & 1ull) && !rec(a, ae)) addEdge2(a, ae);
  }

  // write back face->index, map[0, nmap) and the horizon edges (addEdge's bound check)
  if (ok) faces[lane].index = IX;
  if (lane < nmap) map[lane] = MP;
  if (lane < ne && lane < MJCCD_HORIZON_MAX) {
    pt->horizon.indices[lane] = HE & 0xFF;
    pt->horizon.edges[lane] = HE >> 8;
  }
  if (ne > MJCCD_HORIZON_MAX) pt->status->flags |= MJCCD_FLAG_HORIZON_OVF;
  pt->horizon.nedges = ne;
  pt->nmap = nmap;
  waveSync();
}
#endif

// G:1285-1302  recover witness points from EPA polytope
HIPGL_DEVICE inline mjtNum epaWitness(const Polytope* pt, const Face* face, mjtNum x1[3],
                                      mjtNum x2[3]) {
  MJCCD_FP_CONTRACT_OFF
  // compute affine coordinates for witness points on plane defined by face
  mjtNum lambda[3];
  int verts[3];
  EPA_VERT_EXPAND(verts, face->verts);
  Vertex* v1 = pt->verts + verts[0];
  Vertex* v2 = pt->verts + verts[1];
  Vertex* v3 = pt->verts + verts[2];
  triAffineCoord(lambda, v1->vert, v2->vert, v3->vert, face->v);

  // witness point on geom 1
  lincomb(x1, lambda, 3, v1->vert1, v2->vert1, v3->vert1, 0);

  // witness point on geom 2
  lincomb(x2, lambda, 3, v1->vert2, v2->vert2, v3->vert2, 0);

  return -mju_sqrt(face->dist2);
}

// G:1306-1452  return a face of the expanded polytope that best approximates the pentration
// depth; witness points are in status->{x1, x2}
template <int PK = 0>
HIPGL_DEVICE inline Face* epa(Status* status, Polytope* pt, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum upper = mjMAX_LIMIT, upper2 = mjMAX_LIMIT, lower2;
  Face* face = 0, *pface = 0;  // face closest to origin
  mjtNum tolerance = status->tolerance;
  int discrete = discreteGeoms(obj1, obj2);

  // tolerance is not used for discrete geoms
  if (discrete && sizeof(mjtNum) == sizeof(double)) {
    tolerance = mjMINVAL;
  }

  int k, kmax = status->max_iterations < 1000 ? status->max_iterations : 1000;
  for (k = 0; k < kmax; k++) {
    // port: an iteration adds at most MJCCD_HORIZON_MAX faces and one vertex; if a bounded LDS
    // slot could overflow, stop and let ccd() redo the whole call on the global scratch buffer
    // (same inputs, so the same result as MuJoCo)
    if (pt->lds_faces_cap && (pt->nfaces + MJCCD_HORIZON_MAX > pt->lds_faces_cap ||
                              pt->nverts + 1 > MJCCD_LDS_POLY_VERTS)) {
      pt->lds_abort = 1;
      face = 0;
      break;
    }
    pface = face;

    // find the face closest to the origin (lower bound for penetration depth)
    const long long _mjp_e0 = MJP_T();
    MJP_ADD(15, 1);
    lower2 = mjMAX_LIMIT;
#if MJCCD_EPA_SCAN_BATCH > 1
    // port: same scan order and strict '<' updates; the map entries and face distances of a
    // batch are loaded together (slots past nmap re-read entry 0 and are never compared)
#if defined(__HIP_DEVICE_COMPILE__)
    if (status->wave_coop) {
      // whole wave, uniform state: lane k reads map entry i0+k.  The sequential strict '<'
      // scan ends on the first entry attaining the minimum if it beats the running minimum:
      // M = min over the valid lanes (exact, order-free; NaN, never selected by '<', -> +inf),
      // then the first lane whose distance equals M.
      const int lane = __lane_id();
      const int nmap = pt->nmap;
      const int* map = pt->map;
      Face* faces = pt->faces;
      for (int i0 = 0; i0 < nmap; i0 += 64) {
        const int n = nmap - i0;
        const bool valid = lane < n;
        const int fi = map[valid ? i0 + lane : 0];
        const mjtNum d = faces[fi].dist2;
        const mjtNum v = (valid && d == d) ? d : INFINITY;
        mjtNum m = v;
        m = wave_dpp_min<0xB1>(m);
        if (n > 2) m = wave_dpp_min<0x4E>(m);
        if (n > 4) m = wave_dpp_min<0x141>(m);
        if (n > 8) m = wave_dpp_min<0x140>(m);
        if (n > 16) {
          m = fmin(m, wave_shfl_xor_f64(m, 16));
          m = fmin(m, wave_shfl_xor_f64(m, 32));
        }
        const mjtNum M = wave_readfirstlane_f64(m);
        if (M < lower2) {
          const int k = __ffsll((long long)__ballot(valid && v == M)) - 1;
          face = &faces[__builtin_amdgcn_readlane(fi, k)];
          lower2 = M;
        }
      }
    } else
#endif
    {
      constexpr int B = MJCCD_EPA_SCAN_BATCH;
      const int nmap = pt->nmap;
      const int* map = pt->map;
      Face* faces = pt->faces;
      for (int i0 = 0; i0 < nmap; i0 += B) {
        int fi[B];
        mjtNum d2[B];
#pragma unroll
        for (int k = 0; k < B; k++) fi[k] = map[i0 + k < nmap ? i0 + k : 0];
#pragma unroll
        for (int k = 0; k < B; k++) d2[k] = faces[fi[k]].dist2;
#pragma unroll
        for (int k = 0; k < B; k++) {
          if (i0 + k < nmap && d2[k] < lower2) {
            face = &faces[fi[k]];
            lower2 = d2[k];
          }
        }
      }
    }
#else
    for (int i = 0; i < pt->nmap; i++) {
      if (pt->faces[pt->map[i]].dist2 < lower2) {
        face = &pt->faces[pt->map[i]];
        lower2 = face->dist2;
      }
    }
#endif

    MJP_ADD(9, MJP_T() - _mjp_e0);
    if (lower2 > upper2 || !face) {
      face = pface;
      break;
    }

    // check if lower bound is 0
    if (lower2 <= 0) {
      status->flags |= MJCCD_FLAG_EPA_AFFINE_HULL;  // mju_warning
      break;
    }

    // compute support point w from the closest face's normal
    mjtNum lower = mju_sqrt(lower2);
    int wi = epaSupport<PK>(pt, obj1, obj2, face->v, lower);
    const Vertex* w = pt->verts + wi;
    mjtNum upper_k = dot3(face->v, w->vert) / lower;  // upper bound for kth iteration
    if (upper_k < upper) {
      upper = upper_k;
      upper2 = upper * upper;
    }
    if (upper - lower < tolerance) {
      break;
    }

    // check if vertex w is a repeated support point
#if defined(__HIP_DEVICE_COMPILE__)
    if (discrete && status->wave_coop) {
      // port: lane-parallel form of the scan below (any earlier vertex with the same indices)
      const int lane = __lane_id();
      const int nverts = pt->nverts - 1;
      const int w1 = w->index1, w2 = w->index2;
      bool rep = false;
      for (int i0 = 0; i0 < nverts; i0 += 64) {
        const int i = i0 + lane;
        const bool hit = i < nverts && w1 == pt->verts[i < nverts ? i : 0].index1 &&
                         w2 == pt->verts[i < nverts ? i : 0].index2;
        if (__ballot(hit)) { rep = true; break; }
      }
      if (rep) {
        break;
      }
    } else
#endif
    if (discrete) {
      int i = 0, nverts = pt->nverts - 1;
      for (; i < nverts; i++) {
        if (w->index1 == pt->verts[i].index1 && w->index2 == pt->verts[i].index2) {
          break;
        }
      }
      if (i != nverts) {
        break;
      }
    }

    pt->horizon.w = w->vert;
    const long long _mjp_e1 = MJP_T();
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_HZN_REG
    if (status->wave_coop && pt->nfaces <= 64) horizonCoop(pt, face);
    else
#endif
    horizon(pt, face);
    MJP_ADD(10, MJP_T() - _mjp_e1);
    const long long _mjp_e2 = MJP_T();

    // unrecoverable numerical issue; at least one face was deleted so nedges is 3 or more
    if (pt->horizon.nedges < 3) {
      face = 0;
      break;
    }

    // insert w as new vertex and attach faces along the horizon
    int nfaces = pt->nfaces, nedges = pt->horizon.nedges;

    // check if there's enough memory to store new faces
    if (nedges > maxFaces(pt)) {
      status->flags |= MJCCD_FLAG_EPA_OUT_OF_MEM;  // mju_warning
      break;
    }
    if (nedges > MJCCD_HORIZON_MAX) {
      // port: MuJoCo would have written past horizon.indices/edges (undefined behaviour)
      status->flags |= MJCCD_FLAG_HORIZON_OVF | MJCCD_FLAG_HORIZON_ABORT;
      face = 0;
      break;
    }

#if defined(__HIP_DEVICE_COMPILE__)
    if (status->wave_coop) {
      // port: lane i attaches the face of horizon edge i (attachFace's arithmetic on the same
      // operands).  The sequential loop stops at the first face whose distance is 0 (that face
      // is still attached, nothing after it); the faces before it that fall in [lower2, upper2]
      // enter the map in edge order.  On a stop face = 0 and EPA ends, as in the loop.
      const int lane = __lane_id();
      const bool act = lane < nedges;
      const int hznIndex = pt->horizon.indices[act ? lane : 0];
      const int hznEdge = pt->horizon.edges[act ? lane : 0];
      Face* hznFace = &pt->faces[hznIndex];
      int hznVerts[3];
      EPA_VERT_EXPAND(hznVerts, hznFace->verts);
      const int v1 = hznVerts[hznEdge], v2 = hznVerts[(hznEdge + 1) % 3];
      const int cur = nfaces + lane;
      const int adj1 = lane == 0 ? nfaces + nedges - 1 : cur - 1;
      const int adj3 = nfaces + (lane + 1 < nedges ? lane + 1 : 0);
      mjtNum fv[3] = {0, 0, 0};
      int ret = 1;
      mjtNum dist2 = 0;
      if (act) {
        ret = projectOriginPlane(fv, pt->verts[v1].vert, pt->verts[v2].vert, pt->verts[wi].vert);
        if (!ret) dist2 = dot3(fv, fv);
      }
      const unsigned long long fail = __ballot(act && dist2 == 0);
      const int f = fail ? __ffsll((long long)fail) - 1 : nedges;   // first stopping face
      const bool ins = act && lane < f && dist2 >= lower2 && dist2 <= upper2;
      const unsigned long long insm = __ballot(ins);
      const int nmap0 = pt->nmap;
      const int mpos = nmap0 + __popcll(insm & ((1ull << lane) - 1ull));
      if (act && lane <= f) {
        hznFace->adj[hznEdge] = cur;
        Face* nf = &pt->faces[cur];
        nf->verts = wi + (v2 << 10) + (v1 << 20);
        nf->adj[0] = adj1;
        nf->adj[1] = hznIndex;
        nf->adj[2] = adj3;
        if (!ret) {
          nf->v[0] = fv[0]; nf->v[1] = fv[1]; nf->v[2] = fv[2];
          nf->dist2 = dist2;
          nf->index = ins ? mpos : -1;
        }
        if (ins) pt->map[mpos] = cur;
      }
      waveSync();
      pt->nmap = nmap0 + __popcll(insm);
      pt->nfaces = nfaces + (f < nedges ? f + 1 : nedges);
      if (f < nedges) face = 0;
    } else
#endif
    {
    // attach first face
    int hznIndex = pt->horizon.indices[0], hznEdge = pt->horizon.edges[0];
    Face* hznFace = &pt->faces[hznIndex];
    int hznVerts[3];
    EPA_VERT_EXPAND(hznVerts, hznFace->verts);
    int v1 = hznVerts[hznEdge],
        v2 = hznVerts[(hznEdge + 1) % 3];
    hznFace->adj[hznEdge] = nfaces;
    mjtNum dist2 = attachFace(pt, wi, v2, v1, nfaces + nedges - 1, hznIndex, nfaces + 1);

    // unrecoverable numerical issue
    if (dist2 == 0) {
      face = 0;
      break;
    }

    // store face in map
    if (dist2 >= lower2 && dist2 <= upper2) {
      int i = pt->nmap++;
      pt->map[i] = pt->nfaces - 1;
      pt->faces[pt->map[i]].index = i;
    }

    // attach remaining faces
    for (int i = 1; i < nedges; i++) {
      int cur = nfaces + i;                  // index of attached face
      int next = nfaces + (i + 1) % nedges;  // index of next face

      hznIndex = pt->horizon.indices[i], hznEdge = pt->horizon.edges[i];
      hznFace = &pt->faces[hznIndex];
      int hznVerts2[3];
      EPA_VERT_EXPAND(hznVerts2, hznFace->verts);
      v1 = hznVerts2[hznEdge];
      v2 = hznVerts2[(hznEdge + 1) % 3];
      hznFace->adj[hznEdge] = cur;
      dist2 = attachFace(pt, wi, v2, v1, cur - 1, hznIndex, next);

      // unrecoverable numerical issue
      if (dist2 == 0) {
        face = 0;
        break;
      }

      // store face in map
      if (dist2 >= lower2 && dist2 <= upper2) {
        int idx = pt->nmap++;
        pt->map[idx] = pt->nfaces - 1;
        pt->faces[pt->map[idx]].index = idx;
      }
    }
    }
    MJP_ADD(14, MJP_T() - _mjp_e2);
    pt->horizon.nedges = 0;  // clear horizon

    // no face candidates left
    if (!pt->nmap || !face) {
      break;
    }
  }

  status->epa_iterations = k;
  if (face) {
    status->dist = epaWitness(pt, face, status->x1, status->x2);
    status->nx = 1;
  } else {
    status->nx = 0;
    status->dist = 0;
  }
  return face;
}

// ------------------------------------- MultiCCD ----------------------------------------------

// G:1457-1468  compute area of a quadrilateral
HIPGL_DEVICE inline mjtNum area4(const mjtNum a[3], const mjtNum b[3],
                                 const mjtNum c[3], const mjtNum d[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum ad[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  mjtNum db[3] = {b[0] - d[0], b[1] - d[1], b[2] - d[2]};
  mjtNum bc[3] = {c[0] - b[0], c[1] - b[1], c[2] - b[2]};
  mjtNum ca[3] = {a[0] - c[0], a[1] - c[1], a[2] - c[2]};
  mjtNum e[3], f[3], g[3];
  cross3(e, ad, db);
  cross3(f, bc, ca);
  add3(g, e, f);
  return 0.5 * norm3(g);
}

// G:1472-1477  return pointer to next vertex in a polygon
HIPGL_DEVICE inline mjtNum* next(mjtNum* polygon, int nvert, mjtNum* curr) {
  MJCCD_FP_CONTRACT_OFF
  if (curr == polygon + 3*(nvert - 1)) {
    return polygon;
  }
  return curr + 3;
}

// G:1481-1525  prune a polygon to a maximum area convex quadrilateral
HIPGL_DEVICE inline void polygonQuad(mjtNum* res[4], mjtNum* polygon, int nvert) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum* a = polygon, *b = polygon + 3, *c = polygon + 6, *d = polygon + 9;
  res[0] = a, res[1] = b, res[2] = c, res[3] = d;
  mjtNum m = area4(a, b, c, d), m_next;
  mjtNum* end = polygon + 3 * nvert;
  for (; a < end; a += 3) {
    while (1) {
      m_next = area4(a, b, c, next(polygon, nvert, d));
      if (m_next <= m) {
        break;
      }
      m = m_next;
      d = next(polygon, nvert, d);
      res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      while (1) {
        m_next = area4(a, b, next(polygon, nvert, c), d);
        if (m_next <= m) {
          break;
        }
        m = m_next;
        c = next(polygon, nvert, c);
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
      while (1) {
        m_next = area4(a, next(polygon, nvert, b), c, d);
        if (m_next <= m) {
          break;
        }
        m = m_next;
        b = next(polygon, nvert, b);
        res[0] = a, res[1] = b, res[2] = c, res[3] = d;
      }
    }
    if (b == a) {
      b = next(polygon, nvert, b);
      if (c == b) {
        c = next(polygon, nvert, c);
        if (d == c) {
          d = next(polygon, nvert, d);
        }
      }
    }
  }
}

// G:1529-1537  normal of a plane perpendicular to the face (normal n) through edge (v1, v2)
HIPGL_DEVICE inline mjtNum planeNormal(mjtNum res[3], const mjtNum v1[3], const mjtNum v2[3],
                                       const mjtNum n[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum v3[3], diff1[3], diff2[3];
  add3(v3, v1, n);
  sub3(diff1, v2, v1);
  sub3(diff2, v3, v1);
  cross3(res, diff1, diff2);
  return dot3(res, v1);
}

// G:1541-1544  find what side of a plane a point p lies
HIPGL_DEVICE inline int halfspace(const mjtNum a[3], const mjtNum n[3], const mjtNum p[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum diff[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  return dot3(diff, n) > -mjMINVAL;
}

// G:1548-1561  compute the intersection of a plane with a line segment (a, b)
HIPGL_DEVICE inline mjtNum planeIntersect(mjtNum res[3], const mjtNum pn[3], mjtNum pd,
                                          const mjtNum a[3], const mjtNum b[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum ab[3];
  sub3(ab, b, a);
  mjtNum temp = dot3(pn, ab);
  if (temp == 0.0) return mjMAX_LIMIT;  // parallel; no intersection
  mjtNum t = (pd - dot3(pn, a)) / temp;
  if (t >= 0.0 && t <= 1.0) {
    res[0] = a[0] + t*ab[0];
    res[1] = a[1] + t*ab[1];
    res[2] = a[2] + t*ab[2];
  }
  return t;
}

// G:1565-1676  clip a polygon against another polygon
HIPGL_DEVICE inline void polygonClip(Scratch* sc, Status* status, const mjtNum* face1, int nface1,
                                     const mjtNum* face2, int nface2, const mjtNum n[3],
                                     const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  // clipping face needs to be at least a triangle
  if (nface1 < 3) {
    return;
  }

  // compute plane normal and distance to plane for each vertex
  mjtNum* pn = sc->pn;
  mjtNum* pd = sc->pd;
  for (int i = 0; i < nface1 - 1; i++) {
    pd[i] = planeNormal(&pn[3*i], &face1[3*i], &face1[3*i + 3], n);
  }
  pd[nface1 - 1] = planeNormal(&pn[3*(nface1 - 1)], &face1[3*(nface1 - 1)], &face1[0], n);

  // reserve 2 * max_sides as max sides for a clipped polygon
  mjtNum *polygon1 = sc->polygon1, *polygon2 = sc->polygon2, *polygon, *clipped;
  int npolygon = nface2, nclipped = 0;
  polygon = polygon1;
  clipped = polygon2;

  for (int i = 0; i < nface2; i++) {
    copy3(polygon + 3*i, face2 + 3*i);
  }

  // clip the polygon by one edge e at a time
  for (int e = 0; e < (3 * nface1); e += 3) {
    for (int i = 0; i < npolygon; i++) {
      // get edge PQ of the polygon
      mjtNum *P = polygon + 3*i;
      mjtNum *Q = (i < npolygon - 1) ? polygon + 3*(i+1) : polygon;

      // determine if P and Q are in the halfspace of the clipping edge
      int inside1 = halfspace(face1 + e, pn + e, P);
      int inside2 = halfspace(face1 + e, pn + e, Q);

      // PQ entirely outside the clipping edge, skip
      if (!inside1 && !inside2) {
        continue;
      }

      // edge PQ is inside the clipping edge, add Q
      if (inside1 && inside2) {
        copy3(clipped + 3*nclipped++, Q);
        continue;
      }

      // add new vertex to clipped polygon where PQ intersects the clipping edge
      mjtNum t = planeIntersect(clipped + 3*nclipped++, pn + e, pd[e/3], P, Q);
      if (t < 0.0 || t > 1.0) {
        nclipped--;  // no intersection in PQ
      }

      // add Q as PQ is now back inside the clipping edge
      if (inside2) {
        copy3(clipped + 3*nclipped++, Q);
      }
    }

    // swap clipped and polygon
    mjtNum* tmp = polygon;
    polygon = clipped;
    clipped = tmp;
    npolygon = nclipped;
    nclipped = 0;
  }

  if (npolygon < 1) {
    return;
  }

  // copy final clipped polygon to status
  if (status->max_contacts < 5 && npolygon > 4) {
    status->nx = 4;
    mjtNum* rect[4];
    polygonQuad(rect, polygon, npolygon);
    for (int i = 0; i < 4; i++) {
      copy3(status->x2 + 3*i, rect[i]);
      sub3(status->x1 + 3*i, status->x2 + 3*i, dir);
    }
    return;
  }

  if (npolygon > mjMAXCONPAIR) {
    status->nx = mjMAXCONPAIR;
    for (int i = 0; i < 3*mjMAXCONPAIR; i += 3) {
      copy3(status->x2 + i, polygon + i);
      sub3(status->x1 + i, status->x2 + i, dir);
    }
    return;
  }

  // no pruning needed
  int k = 0;
  for (int i = 0; i < 3*npolygon; i += 3) {
    int skip = 0;

    // find possible duplicate vertices
    for (int j = 0; j < k; j += 3) {
      if (equal3(status->x2 + j, polygon + i)) {
        skip = 1;
        break;
      }
    }
    if (skip) continue;
    copy3(status->x2 + k, polygon + i);
    sub3(status->x1 + k, status->x2 + k, dir);
    k += 3;
  }
  status->nx = k/3;
}

// G:1680-1691  compute global coordinates of a local point (l1, l2, l3)
HIPGL_DEVICE inline void globalcoord(mjtNum res[3], const mjtNum mat[9], const mjtNum pos[3],
                                     mjtNum l1, mjtNum l2, mjtNum l3) {
  MJCCD_FP_CONTRACT_OFF
  // perform mat * (l1, l2, l3) + pos
  res[0] = mat[0]*l1 + mat[1]*l2 + mat[2]*l3;
  res[1] = mat[3]*l1 + mat[4]*l2 + mat[5]*l3;
  res[2] = mat[6]*l1 + mat[7]*l2 + mat[8]*l3;
  if (pos) {
    res[0] += pos[0];
    res[1] += pos[1];
    res[2] += pos[2];
  }
}

// G:1695-1706  find up to n <= 2 common integers of two arrays, return n
HIPGL_DEVICE inline int intersect(int res[2], const int* arr1, const int* arr2, int n, int m) {
  MJCCD_FP_CONTRACT_OFF
  int count = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      if (arr1[i] == arr2[j]) {
        res[count++] = arr1[i];
        if (count == 2) return 2;
      }
    }
  }
  return count;
}

// port: mjMAX_POLYVERT cap, then MJCCD_MAXPOLY capacity cap (flagged)
HIPGL_DEVICE inline int capPoly(int num, Status* status) {
  MJCCD_FP_CONTRACT_OFF
  if (num > mjMAX_POLYVERT) num = mjMAX_POLYVERT;
  if (num > MJCCD_MAXPOLY) {
    num = MJCCD_MAXPOLY;
    status->flags |= MJCCD_FLAG_POLYCAP;
  }
  return num;
}

// G:1710-1772  compute possible polygon normals of a mesh given up to 3 vertices
HIPGL_DEVICE inline int meshNormals(mjtNum* res, int resind[3], int dim, Obj* obj,
                                    int v1, int v2, int v3, Status* status) {
  MJCCD_FP_CONTRACT_OFF
  const int* polymap = obj->data.mesh.polymap;
  const mjtNum* polynormal = obj->data.mesh.polynormal;

  if (dim == 3) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    int v3_adr = obj->data.mesh.mpolymapadr[v3];
    int v3_num = obj->data.mesh.mpolymapnum[v3];

    int edgeset[2], faceset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    n = intersect(faceset, edgeset, polymap + v3_adr, n, v3_num);
    if (n == 0) return 0;

    // three vertices on mesh define a unique face
    const mjtNum* normal = polynormal + 3*faceset[0];
    globalcoord(res, obj->mat, 0, normal[0], normal[1], normal[2]);
    resind[0] = faceset[0];
    return 1;
  }

  if (dim == 2) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    int v2_adr = obj->data.mesh.mpolymapadr[v2];
    int v2_num = obj->data.mesh.mpolymapnum[v2];

    // up to two faces as vertices on mesh define an edge
    int edgeset[2];
    int n = intersect(edgeset, polymap + v1_adr, polymap + v2_adr, v1_num, v2_num);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) {
      const mjtNum* normal = polynormal + 3*edgeset[i];
      globalcoord(res + 3*i, obj->mat, 0, normal[0], normal[1], normal[2]);
      resind[i] = edgeset[i];
    }
    return n;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1];
    int v1_num = obj->data.mesh.mpolymapnum[v1];

    // cap number of possible faces intersecting at a vertex
    v1_num = capPoly(v1_num, status);
    for (int i = 0; i < v1_num; i++) {
      int index = polymap[v1_adr + i];
      const mjtNum* normal = polynormal + 3*index;
      globalcoord(res + 3*i, obj->mat, 0, normal[0], normal[1], normal[2]);
      resind[i] = index;
    }
    return v1_num;
  }
  return 0;
}

// G:1776-1817  compute normal directional vectors along possible edges given by up to two
// vertices
HIPGL_DEVICE inline int meshEdgeNormals(mjtNum* res, mjtNum* endverts, int dim, Obj* obj,
                                        const mjtNum v1[3], const mjtNum v2[3], int v1i, int v2i,
                                        Status* status) {
  MJCCD_FP_CONTRACT_OFF
  // mesh data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert;

  // only one edge
  if (dim == 2) {
    copy3(endverts, v2);
    sub3(res, v2, v1);
    mju_normalize3(res);
    return 1;
  }

  if (dim == 1) {
    int v1_adr = obj->data.mesh.mpolymapadr[v1i];
    int v1_num = obj->data.mesh.mpolymapnum[v1i];
    v1_num = capPoly(v1_num, status);

    // loop through all faces with vertex v1
    for (int i = 0; i < v1_num; i++) {
      int idx = obj->data.mesh.polymap[v1_adr + i];
      int adr = obj->data.mesh.polyvertadr[idx];
      int nvert = obj->data.mesh.polyvertnum[idx];
      for (int j = 0; j < nvert; j++) {
        if (polyvert[adr + j] == v1i) {
          int k = (j == 0) ? nvert - 1 : j - 1;
          const float* v = vert + 3*polyvert[adr + k];
          globalcoord(endverts + 3*i, mat, pos, v[0], v[1], v[2]);
          sub3(res + 3*i, endverts + 3*i, v1);
          mju_normalize3(res + 3*i);
          break;
        }
      }
    }
    return v1_num;
  }
  return 0;
}

// G:1821-1843  try recovering box normal from collision normal
HIPGL_DEVICE inline int boxNormals2(mjtNum res[9], int resind[3], const mjtNum mat[9],
                                    const mjtNum n[3]) {
  MJCCD_FP_CONTRACT_OFF
  // list of box face normals
  mjtNum normals[18] = {1, 0, 0,    -1,  0,  0,
                        0, 1, 0,     0, -1,  0,
                        0, 0, 1,     0,  0, -1};

  // get local coordinates of the normal
  mjtNum local_n[3];
  local_n[0] = mat[0]*n[0] + mat[3]*n[1] + mat[6]*n[2];
  local_n[1] = mat[1]*n[0] + mat[4]*n[1] + mat[7]*n[2];
  local_n[2] = mat[2]*n[0] + mat[5]*n[1] + mat[8]*n[2];
  scl3(local_n, local_n, 1/mju_sqrt(dot3(local_n, local_n)));

  // determine if there is a side close to the normal
  for (int i = 0; i < 6; i++) {
    if (dot3(local_n, normals + 3*i) > mjFACE_TOL) {
      globalcoord(res, mat, 0, normals[3*i], normals[3*i + 1], normals[3*i + 2]);
      resind[0] = i;
      return 1;
    }
  }
  return 0;
}

// G:1847-1897  compute possible face normals of a box given up to 3 vertices
HIPGL_DEVICE inline int boxNormals(mjtNum res[9], int resind[3], int dim, Obj* obj,
                                   int v1, int v2, int v3, const mjtNum dir[3]) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum* mat = obj->mat;
  if (dim == 3) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1) && (v3 & 1)) - (!(v1 & 1) && !(v2 & 1) && !(v3 & 1));
    int y = ((v1 & 2) && (v2 & 2) && (v3 & 2)) - (!(v1 & 2) && !(v2 & 2) && !(v3 & 2));
    int z = ((v1 & 4) && (v2 & 4) && (v3 & 4)) - (!(v1 & 4) && !(v2 & 4) && !(v3 & 4));
    globalcoord(res, mat, 0, x, y, z);
    int sgn = x + y + z;
    if (x) resind[c++] = 0;
    if (y) resind[c++] = 2;
    if (z) resind[c++] = 4;
    if (sgn == -1) resind[0]++;
    return c == 1 ? 1 : boxNormals2(res, resind, mat, dir);
  }

  if (dim == 2) {
    int c = 0;
    int x = ((v1 & 1) && (v2 & 1)) - (!(v1 & 1) && !(v2 & 1));
    int y = ((v1 & 2) && (v2 & 2)) - (!(v1 & 2) && !(v2 & 2));
    int z = ((v1 & 4) && (v2 & 4)) - (!(v1 & 4) && !(v2 & 4));
    if (x) {
      globalcoord(res, mat, 0, x, 0, 0);
      resind[c++] = (x > 0) ? 0 : 1;
    }
    if (y) {
      globalcoord(res + 3*c, mat, 0, 0, y, 0);
      resind[c++] = (y > 0) ? 2 : 3;
    }
    if (z) {
      globalcoord(res + 3, mat, 0, 0, 0, z);  // sic: MuJoCo writes res + 3 (not res + 3*c)
      resind[c++] = (z > 0) ? 4 : 5;
    }
    return c == 2 ? 2 : boxNormals2(res, resind, mat, dir);
  }

  if (dim == 1) {
    mjtNum x = (v1 & 1) ? 1 : -1;
    mjtNum y = (v1 & 2) ? 1 : -1;
    mjtNum z = (v1 & 4) ? 1 : -1;
    globalcoord(res + 0, mat, 0, x, 0, 0);
    globalcoord(res + 3, mat, 0, 0, y, 0);
    globalcoord(res + 6, mat, 0, 0, 0, z);
    resind[0] = (x > 0) ? 0 : 1;
    resind[1] = (y > 0) ? 2 : 3;
    resind[2] = (z > 0) ? 4 : 5;
    return 3;
  }
  return 0;
}

// G:1901-1934  compute possible edge normals for box for edge collisions
HIPGL_DEVICE inline int boxEdgeNormals(mjtNum res[9], mjtNum endverts[9], int dim, Obj* obj,
                                       const mjtNum v1[3], const mjtNum v2[3], int v1i, int v2i) {
  MJCCD_FP_CONTRACT_OFF
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;
  if (dim == 2) {
    copy3(endverts, v2);
    sub3(res, v2, v1);
    mju_normalize3(res);
    return 1;
  }

  // return 3 adjacent vertices
  if (dim == 1) {
    mjtNum x = (v1i & 1) ? size[0] : -size[0];
    mjtNum y = (v1i & 2) ? size[1] : -size[1];
    mjtNum z = (v1i & 4) ? size[2] : -size[2];

    globalcoord(endverts, mat, pos, -x, y, z);
    sub3(res, endverts, v1);
    mju_normalize3(res);

    globalcoord(endverts + 3, mat, pos, x, -y, z);
    sub3(res + 3, endverts + 3, v1);
    mju_normalize3(res + 3);

    globalcoord(endverts + 6, mat, pos, x, y, -z);
    sub3(res + 6, endverts + 6, v1);
    mju_normalize3(res + 6);
    return 3;
  }
  return 0;
}

// G:1938-1984  recover face of a box from its index
HIPGL_DEVICE inline int boxFace(mjtNum res[12], Obj* obj, int idx) {
  MJCCD_FP_CONTRACT_OFF
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // compute global coordinates of the box face and face normal
  switch (idx) {
    case 0:  // right
      globalcoord(res + 0, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos,  size[0], -size[1],  size[2]);
      return 4;
     case 1:  // left
      globalcoord(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord(res + 6, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 2:  // top
      globalcoord(res + 0, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0],  size[1],  size[2]);
      return 4;
    case 3:  // bottom
      globalcoord(res + 0, mat, pos, -size[0], -size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1], -size[2]);
      return 4;
    case 4:  // front
      globalcoord(res + 0, mat, pos, -size[0],  size[1],  size[2]);
      globalcoord(res + 3, mat, pos,  size[0],  size[1],  size[2]);
      globalcoord(res + 6, mat, pos,  size[0], -size[1],  size[2]);
      globalcoord(res + 9, mat, pos, -size[0], -size[1],  size[2]);
      return 4;
    case 5:  // back
      globalcoord(res + 0, mat, pos,  size[0],  size[1], -size[2]);
      globalcoord(res + 3, mat, pos, -size[0],  size[1], -size[2]);
      globalcoord(res + 6, mat, pos, -size[0], -size[1], -size[2]);
      globalcoord(res + 9, mat, pos,  size[0], -size[1], -size[2]);
      return 4;
  }
  return 0;
}

// G:1988-2004  recover mesh polygon from its index, return number of edges
HIPGL_DEVICE inline int meshFace(mjtNum* res, Obj* obj, int idx, Status* status) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;

  int adr = obj->data.mesh.polyvertadr[idx], j = 0;
  int nvert = obj->data.mesh.polyvertnum[idx];
  nvert = capPoly(nvert, status);
  const float* vert = obj->data.mesh.vert;
  const int* polyvert = obj->data.mesh.polyvert + adr;
  for (int i = nvert - 1; i >= 0; i--) {
    const float* v = vert + 3*polyvert[i];
    globalcoord(res + 3*j++, mat, pos, v[0], v[1], v[2]);
  }
  return nvert;
}

// G:2008-2020  find two normals that are facing each other within a tolerance
HIPGL_DEVICE inline int alignedFaces(int res[2], const mjtNum* v, int nv,
                                     const mjtNum* w, int nw) {
  MJCCD_FP_CONTRACT_OFF
  for (int i = 0; i < nv; i++) {
    for (int j = 0; j < nw; j++) {
      if (dot3(v + 3*i, w + 3*j) < -mjFACE_TOL) {
        res[0] = i;
        res[1] = j;
        return 1;
      }
    }
  }
  return 0;
}

// G:2024-2036  find two normals that are perpendicular to each other within a tolerance
HIPGL_DEVICE inline int alignedFaceEdge(int res[2], const mjtNum* edge, int nedge,
                                        const mjtNum* face, int nface) {
  MJCCD_FP_CONTRACT_OFF
  for (int i = 0; i < nface; i++) {
    for (int j = 0; j < nedge; j++) {
      if (mju_abs(dot3(edge + 3*j, face + 3*i)) < mjEDGE_TOL) {
        res[0] = j;
        res[1] = i;
        return 1;
      }
    }
  }
  return 0;
}

// G:2040-2054  return number (1, 2 or 3) of dimensions of a simplex; reorder vertices
HIPGL_DEVICE inline int simplexDim(int* v1i, int* v2i, int* v3i, mjtNum** v1, mjtNum** v2,
                                   mjtNum** v3) {
  MJCCD_FP_CONTRACT_OFF
  int val1 = *v1i;
  int val2 = *v2i;
  int val3 = *v3i;

  if (val1 != val2) {
    return (val3 == val1 || val3 == val2) ? 2 : 3;
  }
  if (val1 != val3) {
    *v2i = *v3i;
    *v2 = *v3;
    return 2;
  }
  return 1;
}

// G:2058-2188  recover multiple contacts from EPA polytope
HIPGL_DEVICE inline void multicontact(Polytope* pt, Face* face, Status* status,
                                      Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  if (obj1->geom_type == mjGEOM_MESH && !obj1->data.mesh.mesh_polynum) {
    return;
  }
  if (obj2->geom_type == mjGEOM_MESH && !obj2->data.mesh.mesh_polynum) {
    return;
  }

  Scratch* sc = pt->scratch;
  mjtNum *face1 = sc->face1, *face2 = sc->face2, *endverts = sc->endverts;
  // get vertices of faces from EPA
  int verts[3];
  EPA_VERT_EXPAND(verts, face->verts);
  int v11i = pt->verts[verts[0]].index1;
  int v12i = pt->verts[verts[1]].index1;
  int v13i = pt->verts[verts[2]].index1;
  int v21i = pt->verts[verts[0]].index2;
  int v22i = pt->verts[verts[1]].index2;
  int v23i = pt->verts[verts[2]].index2;
  mjtNum* v11 = pt->verts[verts[0]].vert1;
  mjtNum* v12 = pt->verts[verts[1]].vert1;
  mjtNum* v13 = pt->verts[verts[2]].vert1;
  mjtNum* v21 = pt->verts[verts[0]].vert2;
  mjtNum* v22 = pt->verts[verts[1]].vert2;
  mjtNum* v23 = pt->verts[verts[2]].vert2;

  // get dimensions of features of geoms 1 and 2
  int nface1 = simplexDim(&v11i, &v12i, &v13i, &v11, &v12, &v13);
  int nface2 = simplexDim(&v21i, &v22i, &v23i, &v21, &v22, &v23);
  int nnorms1 = 0, nnorms2 = 0;
  mjtNum *n1 = sc->n1, *n2 = sc->n2;  // normals of possible face collisions
  int *idx1 = sc->idx1, *idx2 = sc->idx2;  // indices of faces

  mjtNum dir[3], dir_neg[3];
  sub3(dir, status->x2, status->x1);
  sub3(dir_neg, status->x1, status->x2);

  // get all possible face normals for each geom
  if (obj1->geom_type == mjGEOM_BOX) {
    nnorms1 = boxNormals(n1, idx1, nface1, obj1, v11i, v12i, v13i, dir_neg);
  } else if (obj1->geom_type == mjGEOM_MESH) {
    nnorms1 = meshNormals(n1, idx1, nface1, obj1, v11i, v12i, v13i, status);
  }
  if (obj2->geom_type == mjGEOM_BOX) {
    nnorms2 = boxNormals(n2, idx2, nface2, obj2, v21i, v22i, v23i, dir);
  } else if (obj2->geom_type == mjGEOM_MESH) {
    nnorms2 = meshNormals(n2, idx2, nface2, obj2, v21i, v22i, v23i, status);
  }

  // determine if any two face normals match
  int res[2], edgecon1 = 0, edgecon2 = 0;
  if (!alignedFaces(res, n1, nnorms1, n2, nnorms2)) {
    // check if edge-face collision
    if (nface1 < 3 && nface1 <= nface2) {
      nnorms1 = 0;
      if (obj1->geom_type == mjGEOM_BOX) {
        nnorms1 = boxEdgeNormals(n1, endverts, nface1, obj1, v11, v12, v11i, v12i);
      } else if (obj1->geom_type == mjGEOM_MESH) {
        nnorms1 = meshEdgeNormals(n1, endverts, nface1, obj1, v11, v12, v11i, v12i, status);
      }
      if (!alignedFaceEdge(res, n1, nnorms1, n2, nnorms2)) return;
      edgecon1 = 1;

    // check if face-edge collision
    } else if (nface2 < 3) {
      nnorms2 = 0;
      if (obj2->geom_type == mjGEOM_BOX) {
        nnorms2 = boxEdgeNormals(n2, endverts, nface2, obj2, v21, v22, v21i, v22i);
      } else if (obj2->geom_type == mjGEOM_MESH) {
        nnorms2 = meshEdgeNormals(n2, endverts, nface2, obj2, v21, v22, v21i, v22i, status);
      }
      if (!alignedFaceEdge(res, n2, nnorms2, n1, nnorms1)) return;
      edgecon2 = 1;
    } else {
      // no multi-contact
      return;
    }
  }
  int i = res[0], j = res[1];

  // recover geom1 matching edge or face
  if (edgecon1) {
    copy3(face1, pt->verts[verts[0]].vert1);
    copy3(face1 + 3, endverts + 3*i);
    nface1 = 2;
  } else {
    if (obj1->geom_type == mjGEOM_BOX) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = boxFace(face1, obj1, ind);
    } else if (obj1->geom_type == mjGEOM_MESH) {
      int ind = (edgecon2 ? idx1[j] : idx1[i]);
      nface1 = meshFace(face1, obj1, ind, status);
    }
  }

  // recover geom2 matching edge or face
  if (edgecon2) {
    copy3(face2, pt->verts[verts[0]].vert2);
    copy3(face2 + 3, endverts + 3*i);
    nface2 = 2;
  } else {
    if (obj2->geom_type == mjGEOM_BOX) {
      nface2 = boxFace(face2, obj2, idx2[j]);
    } else if (obj2->geom_type == mjGEOM_MESH) {
      nface2 = meshFace(face2, obj2, idx2[j], status);
    }
  }

  // approximate the contact direction by scaling the face normal by the single contact
  // direction's magnitude
  mjtNum approx_dir[3];

  // face1 is an edge; clip face1 against face2
  if (edgecon1) {
    scl3(approx_dir, n2 + 3*j, norm3(dir));
    polygonClip(sc, status, face2, nface2, face1, nface1, n2 + 3*j, approx_dir);
    return;
  }

  // face2 is an edge; clip face2 against face1
  if (edgecon2) {
    scl3(approx_dir, n1 + 3*j, -norm3(dir));
    polygonClip(sc, status, face1, nface1, face2, nface2, n1 + 3*j, approx_dir);
    return;
  }

  // face-face collision
  scl3(approx_dir, n2 + 3*j, norm3(dir));
  polygonClip(sc, status, face1, nface1, face2, nface2, n1 + 3*i, approx_dir);
}

// ------------------------------------- drivers -----------------------------------------------

// G:2220-2335  general convex collision detection (sphere/capsule branch not ported)
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_POLY_SLOTS > 0
static_assert(MJCCD_LDS_POLY_VERTS >= 6 && MJCCD_LDS_POLY_FACES >= 6 + MJCCD_HORIZON_MAX,
              "LDS polytope slot must hold any initial polytope plus one EPA iteration");
struct LdsPolySlot {
  Vertex verts[MJCCD_LDS_POLY_VERTS];
  Face faces[MJCCD_LDS_POLY_FACES];
  int map[MJCCD_LDS_POLY_FACES];
  int hidx[MJCCD_HORIZON_MAX];
  int hedge[MJCCD_HORIZON_MAX];
  HorizonFrame stack[MJCCD_LDS_POLY_FACES];
};
// slot stride padded to an odd number of 8-byte words so that lanes touching the same field of
// their own slots hit different LDS banks
struct LdsPolySlotPadded {
  LdsPolySlot s;
  double pad[(sizeof(LdsPolySlot) / 8) % 2 == 0 ? 1 : 2];
};
static_assert((sizeof(LdsPolySlotPadded) / 8) % 2 == 1, "odd slot stride");
struct LdsPoly {
  LdsPolySlotPadded slot[MJCCD_LDS_POLY_SLOTS];
};
#endif

// MODE (port-only): 0 = generic (LDS polytope slot when available, else the global buffer);
// 1 = coop LDS path (objects and status in LDS, polytope in a per-wave LDS slot of a small
//     dedicated array; *aborted when no slot / the slot could overflow);
// 2 = global buffer only (no LDS polytope).
// All modes run the same arithmetic on the same values.
#ifndef MJCCD_FAN_POLY_SLOTS
#define MJCCD_FAN_POLY_SLOTS 4
#endif
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_POLY_SLOTS > 0
// port: the fan slots hold a polytope as large as the global Scratch buffer (5+N vertices,
// 6N faces/map/stack entries), so the coop EPA can never overflow its slot and never has to be
// redone on the global buffer (that rerun was a ~2M-cycle tail on deep mesh-mesh EPAs, which
// reach ~130 faces / 33 vertices / 28 iterations).  ~21 KB per wave.
struct LdsFanSlot {
  Vertex verts[5 + MJCCD_MAX_ITERATIONS];
  Face faces[6 * MJCCD_MAX_ITERATIONS];
  int map[6 * MJCCD_MAX_ITERATIONS];
  int hidx[MJCCD_HORIZON_MAX];
  int hedge[MJCCD_HORIZON_MAX];
  HorizonFrame stack[6 * MJCCD_MAX_ITERATIONS];
};
struct LdsFanSlotPadded {
  LdsFanSlot s;
  double pad[(sizeof(LdsFanSlot) / 8) % 2 == 0 ? 1 : 2];
};
static_assert((sizeof(LdsFanSlotPadded) / 8) % 2 == 1, "odd slot stride");
// one array shared by every ccdImpl<1, PK> instantiation (a static in the template would be
// allocated once per instantiation)
HIPGL_DEVICE __forceinline__ LdsFanSlotPadded* fanPolySlots() {
  __shared__ LdsFanSlotPadded lds_poly_fan[MJCCD_FAN_POLY_SLOTS];
  return lds_poly_fan;
}
#endif
template <int MODE = 0, int PK = 0>
HIPGL_DEVICE __attribute__((always_inline)) inline mjtNum ccdImpl(const CCDConfig* config, Status* status, Obj* obj1, Obj* obj2,
                                    int lds_ok, int* aborted) {
  MJCCD_FP_CONTRACT_OFF
  // setup
  mjc_center(status->x1, obj1);
  mjc_center(status->x2, obj2);
  status->gjk_iterations = 0;
  status->epa_iterations = 0;
  status->epa_status = mjEPA_NOCONTACT;
  status->tolerance = config->tolerance;
  status->max_iterations = config->max_iterations;
  // PK != 0 (penetration<1> only): max_contacts == 1 and dist_cutoff == 0 (caller-checked)
  const int max_contacts = PK ? 1 : config->max_contacts;
  status->max_contacts = max_contacts;
  status->dist_cutoff = PK ? 0 : config->dist_cutoff;
  status->flags = 0;

  // special handling for sphere and capsule (G:2234-2289): NOT PORTED.
  // port: also refuse what MuJoCo cannot produce or the port cannot hold:
  //   geom < 0 (MuJoCo: flex object), max_iterations < 1 (MuJoCo: gjk returns an uninitialized
  //   x_norm), max_contacts beyond Status::x1/x2, polytope larger than Scratch.
  if (obj1->geom_type == mjGEOM_SPHERE || obj2->geom_type == mjGEOM_SPHERE ||
      obj1->geom_type == mjGEOM_CAPSULE || obj2->geom_type == mjGEOM_CAPSULE ||
      obj1->geom < 0 || obj2->geom < 0 || config->max_iterations < 1 ||
      (MJCCD_STATUS_NX < mjMAXCONPAIR && max_contacts >= 5) ||
      (config->buffer && config->max_iterations > MJCCD_MAX_ITERATIONS)) {
    status->flags |= MJCCD_FLAG_UNSUPPORTED;
    status->nx = 0;
    status->nsimplex = 0;
    status->dist = mjMAX_LIMIT;
    return status->dist;
  }
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
  long long _tg = wall_clock64();
#endif
  const long long _mjp_tg = MJP_T();
  gjk<PK>(status, obj1, obj2);
  MJP_ADD(2, MJP_T() - _mjp_tg);
  MJP_ADD(6, 1);
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
  MJST_T(4, _tg);
#endif

  // penetration recovery for contacts not needed
  if (!max_contacts) {
    return status->dist;
  }

  if (status->dist <= config->tolerance && status->nsimplex > 1 && config->buffer) {
    status->dist = 0;  // assume touching
    Polytope pt;
    pt.nfaces = pt.nmap = pt.nverts = pt.horizon.nedges = 0;

    // allocate memory for polytope
    int N = config->max_iterations;
    pt.maxfaces = 6 * N;
    Scratch* buffer = config->buffer;
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_POLY_SLOTS > 0
    if constexpr (MODE != 1)
#endif
    {
    pt.verts = buffer->verts;
    pt.faces = buffer->faces;
    pt.map = buffer->map;
    pt.horizon.indices = buffer->horizon_indices;
    pt.horizon.edges = buffer->horizon_edges;
    pt.stack = buffer->stack;
    }
    pt.horizon.w = 0;
    pt.scratch = buffer;
    pt.status = status;
    pt.lds_faces_cap = 0;
    pt.lds_abort = 0;
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_POLY_SLOTS > 0
    if constexpr (MODE == 1) {
      // coop: one slot per wave of the workgroup
      LdsFanSlotPadded* lds_poly_fan = fanPolySlots();
      const int wv = (int)((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / warpSize);
      if (!status->wave_coop || wv >= MJCCD_FAN_POLY_SLOTS) {
        *aborted = 1;
        return status->dist;
      }
      LdsFanSlot& sl = lds_poly_fan[wv].s;
      pt.verts = sl.verts;
      pt.faces = sl.faces;
      pt.map = sl.map;
      pt.horizon.indices = sl.hidx;
      pt.horizon.edges = sl.hedge;
      pt.stack = sl.stack;
      pt.lds_faces_cap = 0;  // same capacity as the global buffer: no overflow check
    }
    const unsigned nthr = blockDim.x * blockDim.y * blockDim.z;
    if (MODE == 0 && lds_ok && nthr <= (unsigned)warpSize * MJCCD_LDS_POLY_SLOTS) {
      __shared__ LdsPoly lds_poly;
      const unsigned long long act = __ballot(1);
      const int rank = status->wave_coop ? 0 : __popcll(act & ((1ull << __lane_id()) - 1ull));
      // port: one slot per active lane of every wave of the workgroup (single-wave groups:
      // slot = rank, as before)
      const int nw = (int)((nthr + warpSize - 1) / warpSize);
      const int wv = (int)((threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / warpSize);
      const int slot = rank * nw + wv;
      if (slot < MJCCD_LDS_POLY_SLOTS) {
        LdsPolySlot& sl = lds_poly.slot[slot].s;
        pt.verts = sl.verts;
        pt.faces = sl.faces;
        pt.map = sl.map;
        pt.horizon.indices = sl.hidx;
        pt.horizon.edges = sl.hedge;
        pt.stack = sl.stack;
        pt.lds_faces_cap = MJCCD_LDS_POLY_FACES;
      }
    }
#else
    (void)lds_ok;
#endif

#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
    long long _tp = wall_clock64();
#endif
    int ret;
    const long long _mjp_tp = MJP_T();
    if (status->nsimplex == 2) {
      ret = polytope2<PK>(&pt, status, obj1, obj2);
    } else if (status->nsimplex == 3) {
      ret = MJCCD_EPA_TO_P3;
    } else {
      ret = polytope4<PK>(&pt, status, obj1, obj2);
    }
    if (ret == MJCCD_EPA_TO_P3) {
      ret = polytope3<PK>(&pt, status, obj1, obj2);
    }
    status->epa_status = ret;
    MJP_ADD(4, MJP_T() - _mjp_tp);
    const long long _mjp_te = MJP_T();
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
    MJST_T(5, _tp);
    long long _te = wall_clock64();
#endif

    // simplex not on boundary (objects are penetrating)
    if (!ret) {
      Face* face = epa<PK>(status, &pt, obj1, obj2);
      MJP_ADD(5, MJP_T() - _mjp_te);
#if defined(MJCCD_ABORT_DIAG) && defined(__HIP_DEVICE_COMPILE__)
      if (MODE == 2 && __lane_id() == 0) printf("GLOB nf %d nv %d it %d\n", pt.nfaces, pt.nverts, status->epa_iterations);
      if (MODE == 1 && __lane_id() == 0) {
        static __device__ unsigned long long g_ad[8];
        unsigned long long c = atomicAdd(&g_ad[0], 1ull);
        atomicMax(&g_ad[2], (unsigned long long)pt.nfaces);
        atomicMax(&g_ad[3], (unsigned long long)pt.nverts);
        atomicMax(&g_ad[4], (unsigned long long)status->epa_iterations);
        if (pt.lds_abort) { atomicAdd(&g_ad[1], 1ull);
          printf("ABORT PK %d nf %d nv %d nmap %d it %d g %d %d\n", PK, pt.nfaces, pt.nverts, pt.nmap, status->epa_iterations, obj1->geom, obj2->geom); }
        if ((c & 0x3fff) == 0) printf("ADIAG calls %llu aborts %llu maxnf %llu maxnv %llu maxit %llu\n", c, g_ad[1], g_ad[2], g_ad[3], g_ad[4]);
      }
#endif
      if (pt.lds_abort) {
        *aborted = 1;
        return status->dist;
      }
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
      MJST_T(6, _te);
#endif
      if (max_contacts > 1 && face) {
        multicontact(&pt, face, status, obj1, obj2);
      }
    }
  }
  return status->dist;
}

// mjc_ccd.  ccdImpl may run EPA in an LDS slot and give up when the slot could overflow; it is
// then rerun on the global scratch buffer from the same inputs (the support warm-start indices
// are the only object state ccd mutates, so they are restored first).
HIPGL_DEVICE MJCCD_NOINLINE inline mjtNum ccd(const CCDConfig* config, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  const int vi1 = obj1->vertindex, mi1 = obj1->meshindex;
  const int vi2 = obj2->vertindex, mi2 = obj2->meshindex;
  mjtNum r;
  for (int pass = 0;; pass++) {
    int aborted = 0;
    r = ccdImpl(config, status, obj1, obj2, pass == 0, &aborted);
    if (!aborted) break;
    obj1->vertindex = vi1;
    obj1->meshindex = mi1;
    obj2->vertindex = vi2;
    obj2->meshindex = mi2;
  }
  return r;
}

#if defined(__HIP_DEVICE_COMPILE__)
// port-only: mjc_ccd on the global scratch buffer only (ccdLds' overflow fallback)
HIPGL_DEVICE MJCCD_NOINLINE inline mjtNum ccdGlobal(const CCDConfig* config, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  int aborted = 0;
  return ccdImpl<2>(config, status, obj1, obj2, 0, &aborted);
}

// port-only: mjc_ccd for a wave-coop call whose objects and status live in LDS (the fan /
// single-wave coop paths).  Same body; the address-space assumptions let the compiler use ds_*
// instead of flat instructions, and the polytope lives in this wave's LDS slot (on a possible
// slot overflow the call is redone on the global buffer from the same inputs, like ccd()).
// Values and arithmetic are unchanged.
// PK: support specialisation (0 generic; 1 box + hill-climbed mesh; 2 two hill-climbed meshes),
// chosen by penetration<1> from the objects' support kinds.
template <int PK>
HIPGL_DEVICE MJCCD_NOINLINE mjtNum ccdLds(const CCDConfig* config, Status* status, Obj* obj1, Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  __builtin_assume(__builtin_amdgcn_is_shared((const void*)obj1));
  __builtin_assume(__builtin_amdgcn_is_shared((const void*)obj2));
  __builtin_assume(__builtin_amdgcn_is_shared((const void*)status));
  const int vi1 = obj1->vertindex, mi1 = obj1->meshindex;
  const int vi2 = obj2->vertindex, mi2 = obj2->meshindex;
  int aborted = 0;
  mjtNum r = ccdImpl<1, PK>(config, status, obj1, obj2, 1, &aborted);
  if (aborted) {
    obj1->vertindex = vi1;
    obj1->meshindex = mi1;
    obj2->vertindex = vi2;
    obj2->meshindex = mi2;
    r = ccdGlobal(config, status, obj1, obj2);
  }
  return r;
}
#endif

// C:87-129  mjc_penetration (native CCD branch); con arrays receive nwitness contacts
// CCDSEL (port-only): 0 = ccd(); 1 = ccdLds for wave-coop calls on LDS objects and status,
// else ccdGlobal (keeps ccd()'s 12-slot LDS polytope array out of the caller's kernel).
template <int CCDSEL = 0>
HIPGL_DEVICE inline int penetration(const Config& cfg, Obj* obj1, Obj* obj2, Scratch* s,
                                    int ncon, mjtNum margin, mjtNum pos[][3],
                                    mjtNum normal[][3], mjtNum dist_out[], Status* st) {
  MJCCD_FP_CONTRACT_OFF
  CCDConfig config;
  mjtNum dist;
  int nwitness = 0;

  // set config
  config.max_iterations = cfg.ccd_iterations;
  config.tolerance = cfg.ccd_tolerance;
  config.max_contacts = ncon;
  config.dist_cutoff = 0;  // no geom distances needed
  config.buffer = s;

  if constexpr (CCDSEL == 1) {
#if defined(__HIP_DEVICE_COMPILE__)
    if (st->wave_coop && __builtin_amdgcn_is_shared((const void*)obj1) &&
        __builtin_amdgcn_is_shared((const void*)obj2) && __builtin_amdgcn_is_shared((const void*)st))
    {
      const int k1 = obj1->support, k2 = obj2->support;
      const bool cw = obj1->wave_coop && obj2->wave_coop;
      if (cw && ncon == 1 && k1 == SUPPORT_BOX && k2 == SUPPORT_HILLCLIMB)
        dist = ccdLds<1>(&config, st, obj1, obj2);
      else if (cw && ncon == 1 && k1 == SUPPORT_HILLCLIMB && k2 == SUPPORT_HILLCLIMB)
        dist = ccdLds<2>(&config, st, obj1, obj2);
      else
        dist = ccdLds<0>(&config, st, obj1, obj2);
    }
    else
      dist = ccdGlobal(&config, st, obj1, obj2);
#else
    dist = ccd(&config, st, obj1, obj2);
#endif
  } else {
    dist = ccd(&config, st, obj1, obj2);
  }
  if (dist < 0) {
    nwitness = st->nx;
    for (int i = 0; i < nwitness; i++) {
      dist_out[i] = margin + dist;
      pos[i][0] = 0.5*(st->x1[3*i + 0] + st->x2[3*i + 0]);
      pos[i][1] = 0.5*(st->x1[3*i + 1] + st->x2[3*i + 1]);
      pos[i][2] = 0.5*(st->x1[3*i + 2] + st->x2[3*i + 2]);
      sub3(normal[i], st->x1 + 3*i, st->x2 + 3*i);
      mju_normalize3(normal[i]);
    }
  }
  return nwitness;
}

// C:831-854  return number of contacts supported by a single pass of narrowphase
HIPGL_DEVICE inline int maxContacts(const Config& cfg, const Obj* obj1, const Obj* obj2) {
  MJCCD_FP_CONTRACT_OFF
  // single pass not supported for margins
  if (obj1->margin > 0 || obj2->margin > 0) {
    return 1;
  }

  // can return 8 contacts for box-box collision in one pass
  int type1 = obj1->geom_type;
  int type2 = obj2->geom_type;
  if (type1 == mjGEOM_BOX && type2 == mjGEOM_BOX) {
    return 8;
  }

  // reduce mesh collisions to 4 contacts max
  if (type1 == mjGEOM_BOX || type1 == mjGEOM_MESH) {
    if (type2 == mjGEOM_BOX || type2 == mjGEOM_MESH) {
      return cfg.disable_multiccd ? 1 : 4;
    }
  }

  // not supported for other geom types
  return 1;
}

// host-side check that the model options fit this build (see caller contract at the top)
HIPGL_DEVICE inline int configSupported(const Config& cfg) {
  return cfg.ccd_iterations >= 1 && cfg.ccd_iterations <= MJCCD_MAX_ITERATIONS &&
         !cfg.disable_nativeccd;
}

// engine_util_spatial.c:508-532  make 3D contact frame given x-axis (normal) and possibly y-axis
// (tangent 1), as mj_setContact does for every narrowphase contact (engine_collision_driver.c:
// 1784).  mjERROR on a degenerate x-axis -> return 1.
HIPGL_DEVICE inline int mju_makeFrame(mjtNum frame[9]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum tmp[3];
  int err = 0;

  // normalize xaxis
  if (mju_normalize3(frame) < 0.5) {
    err = 1;  // mjERROR("xaxis of contact frame undefined")
  }

  // if yaxis undefined, set yaxis to (0,1,0) if possible, otherwise (0,0,1)
  if (dot3(frame+3, frame+3) < 0.25) {
    frame[3] = 0, frame[4] = 0, frame[5] = 0;

    if (frame[1] < 0.5 && frame[1] > -0.5) {
      frame[4] = 1;
    } else {
      frame[5] = 1;
    }
  }

  // make yaxis orthogonal to xaxis
  scl3(tmp, frame, dot3(frame, frame+3));
  frame[3] -= tmp[0];
  frame[4] -= tmp[1];
  frame[5] -= tmp[2];
  mju_normalize3(frame+3);

  // zaxis = cross(xaxis, yaxis)
  cross3(frame+6, frame, frame+3);
  return err;
}

// B:146-149
HIPGL_DEVICE inline mjtNum mju_dist3(const mjtNum pos1[3], const mjtNum pos2[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum dif[3] = {pos1[0]-pos2[0], pos1[1]-pos2[1], pos1[2]-pos2[2]};
  return mju_sqrt(dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2]);
}

// B:179-189
HIPGL_DEVICE inline void mju_mulMatMat3(mjtNum res[9], const mjtNum mat1[9], const mjtNum mat2[9]) {
  MJCCD_FP_CONTRACT_OFF
  res[0] = mat1[0]*mat2[0] + mat1[1]*mat2[3] + mat1[2]*mat2[6];
  res[1] = mat1[0]*mat2[1] + mat1[1]*mat2[4] + mat1[2]*mat2[7];
  res[2] = mat1[0]*mat2[2] + mat1[1]*mat2[5] + mat1[2]*mat2[8];
  res[3] = mat1[3]*mat2[0] + mat1[4]*mat2[3] + mat1[5]*mat2[6];
  res[4] = mat1[3]*mat2[1] + mat1[4]*mat2[4] + mat1[5]*mat2[7];
  res[5] = mat1[3]*mat2[2] + mat1[4]*mat2[5] + mat1[5]*mat2[8];
  res[6] = mat1[6]*mat2[0] + mat1[7]*mat2[3] + mat1[8]*mat2[6];
  res[7] = mat1[6]*mat2[1] + mat1[7]*mat2[4] + mat1[8]*mat2[7];
  res[8] = mat1[6]*mat2[2] + mat1[7]*mat2[5] + mat1[8]*mat2[8];
}

// engine_util_spatial.c:96-113 (mji_axisAngle2Quat)
HIPGL_DEVICE inline void mju_axisAngle2Quat(mjtNum res[4], const mjtNum axis[3], mjtNum angle) {
  MJCCD_FP_CONTRACT_OFF
  if (angle == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
  } else {
    mjtNum s = sin(angle*0.5);
    res[0] = cos(angle*0.5);
    res[1] = axis[0]*s;
    res[2] = axis[1]*s;
    res[3] = axis[2]*s;
  }
}

// engine_util_spatial.c:145-176
HIPGL_DEVICE inline void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]) {
  MJCCD_FP_CONTRACT_OFF
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    res[0] = 1; res[1] = 0; res[2] = 0;
    res[3] = 0; res[4] = 1; res[5] = 0;
    res[6] = 0; res[7] = 0; res[8] = 1;
    return;
  }
  mjtNum q00 = quat[0]*quat[0], q01 = quat[0]*quat[1], q02 = quat[0]*quat[2], q03 = quat[0]*quat[3];
  mjtNum q11 = quat[1]*quat[1], q12 = quat[1]*quat[2], q13 = quat[1]*quat[3];
  mjtNum q22 = quat[2]*quat[2], q23 = quat[2]*quat[3], q33 = quat[3]*quat[3];
  res[0] = q00 + q11 - q22 - q33;
  res[4] = q00 - q11 + q22 - q33;
  res[8] = q00 - q11 - q22 + q33;
  res[1] = 2*(q12 - q03);
  res[2] = 2*(q13 + q02);
  res[3] = 2*(q12 + q03);
  res[5] = 2*(q23 - q01);
  res[6] = 2*(q13 - q02);
  res[7] = 2*(q23 + q01);
}

// C:798-806
HIPGL_DEVICE inline int isDistinctContact(const mjtNum pos[][3], int ncon, mjtNum tolerance) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum* last_pos = pos[ncon - 1];
  for (int i = 0; i < ncon-1; i++) {
    if (mju_dist3(pos[i], last_pos) <= tolerance) {
      return 0;
    }
  }
  return 1;
}

// C:810-826  in-place rotation of spatial frame around given point of origin
HIPGL_DEVICE inline void mju_rotateFrame(const mjtNum origin[3], const mjtNum rot[9],
                                         mjtNum xmat[9], mjtNum xpos[3]) {
  MJCCD_FP_CONTRACT_OFF
  mjtNum mat[9], vec[3], rel[3];
  mju_mulMatMat3(mat, rot, xmat);
  for (int k = 0; k < 9; k++) xmat[k] = mat[k];
  sub3(rel, origin, xpos);
  mju_mulMatVec3(vec, rot, rel);
  vec[0] -= rel[0]; vec[1] -= rel[1]; vec[2] -= rel[2];
  xpos[0] -= vec[0]; xpos[1] -= vec[1]; xpos[2] -= vec[2];
}

// port-only: stage the hill-climb data (graph slice + vertex coordinates) of the meshes used by
// the lanes of this wave in LDS, and point the objects' mesh pointers at the copies.  The support
// functions then run their long dependent load chains against LDS instead of global memory.  The
// copies are verbatim, so every value read (and every result) is unchanged.  Wave-cooperative:
// all lanes that are in convex() together share the work; meshes that do not fit stay global.
#ifndef MJCCD_LDS_MESH_WORDS
#define MJCCD_LDS_MESH_WORDS 13312
#endif
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_MESH_WORDS > 0
HIPGL_DEVICE inline unsigned long long wave_read_u64(unsigned long long v, int lane) {
  unsigned lo = (unsigned)__shfl((int)(unsigned)(v & 0xffffffffull), lane);
  unsigned hi = (unsigned)__shfl((int)(unsigned)(v >> 32), lane);
  return ((unsigned long long)hi << 32) | lo;
}

HIPGL_DEVICE __forceinline__ int* ldsMeshBuf() {
  __shared__ int lds_mesh[MJCCD_LDS_MESH_WORDS];
  return lds_mesh;
}
HIPGL_DEVICE inline void stageMeshesLDS(Obj* obj1, Obj* obj2, bool block_owner = false) {
  int* lds_mesh = ldsMeshBuf();
  // one wave per workgroup only (the LDS buffer is per workgroup), or the one wave of a
  // multi-wave workgroup that owns it (block_owner)
  if (!block_owner && blockDim.x * blockDim.y * blockDim.z > (unsigned)warpSize) return;
  const unsigned long long act = __ballot(1);
  const int nact = __popcll(act);
  const int lane = __lane_id();
  const int rank = __popcll(act & ((1ull << lane) - 1ull));
  bool need1 = obj1->support == SUPPORT_HILLCLIMB && obj1->data.mesh.graph;
  bool need2 = obj2->support == SUPPORT_HILLCLIMB && obj2->data.mesh.graph;
  int used = 0;
  for (;;) {
    const int* mine = need1 ? obj1->data.mesh.graph : need2 ? obj2->data.mesh.graph : nullptr;
    const float* minev = need1 ? obj1->data.mesh.vert : need2 ? obj2->data.mesh.vert : nullptr;
    int minenv = need1 ? obj1->data.mesh.nvert : need2 ? obj2->data.mesh.nvert : 0;
    unsigned long long b = __ballot(mine != nullptr);
    if (!b) break;
    int leader = __ffsll((long long)b) - 1;
    const int* g = (const int*)wave_read_u64((unsigned long long)mine, leader);
    const float* gv = (const float*)wave_read_u64((unsigned long long)minev, leader);
    int nvert = __shfl(minenv, leader);
    int numvert = g[0], numface = g[1];
    // header, vert_edgeadr, vert_globalid, edge_localid (+ the up to BATCH-1 words the hill
    // climb may read past the last terminator; their values are never used)
    int gw = 2 + 3 * numvert + 3 * numface + 8;
    int vw = 3 * nvert;
    bool fits = used + gw + vw <= MJCCD_LDS_MESH_WORDS;
    if (fits) {
      int* dst = lds_mesh + used;
      constexpr int U = 8;
      for (int w0 = rank; w0 < gw; w0 += U * nact) {
        int v[U];
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; v[u] = g[w < gw ? w : 0]; }
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; if (w < gw) dst[w] = v[u]; }
      }
      const int* gvi = reinterpret_cast<const int*>(gv);
      int* dstv = dst + gw;
      for (int w0 = rank; w0 < vw; w0 += U * nact) {
        int v[U];
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; v[u] = gvi[w < vw ? w : 0]; }
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; if (w < vw) dstv[w] = v[u]; }
      }
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    if (need1 && obj1->data.mesh.graph == g) {
      if (fits) {
        obj1->data.mesh.graph = lds_mesh + used;
        obj1->data.mesh.vert = reinterpret_cast<const float*>(lds_mesh + used + gw);
      }
      need1 = false;
    }
    if (need2 && obj2->data.mesh.graph == g) {
      if (fits) {
        obj2->data.mesh.graph = lds_mesh + used;
        obj2->data.mesh.vert = reinterpret_cast<const float*>(lds_mesh + used + gw);
      }
      need2 = false;
    }
    // edge records (16-byte aligned, + 64 terminator records for the batched / per-lane
    // over-read past the last list)
    const int ne = numvert + 3 * numface;
    const int ru = (used + gw + vw + 3) & ~3;
    const int rw = 4 * (ne + 64);
    const bool rfits = fits && numvert < 0xffff && ne < 0xffff &&
                       ru + rw <= MJCCD_LDS_MESH_WORDS;
    HcRec* recp = reinterpret_cast<HcRec*>(lds_mesh + ru);
    if (rfits) {
      const int* lg = lds_mesh + used;
      const float* lv = reinterpret_cast<const float*>(lds_mesh + used + gw);
      for (int e = rank; e < ne + 64; e += nact) {
        int sub = e < ne ? lg[2 + 2 * numvert + e] : -1;
        HcRec r;
        if (sub >= 0) {
          int nadr = lg[2 + sub];
          int gid = lg[2 + numvert + sub];
          r.x = lv[3 * gid + 0]; r.y = lv[3 * gid + 1]; r.z = lv[3 * gid + 2];
          r.packed = (int)((unsigned)sub | ((unsigned)nadr << 16));
        } else {
          r.x = 0.f; r.y = 0.f; r.z = 0.f;
          r.packed = 0xffff;
        }
        recp[e] = r;
      }
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
      __builtin_amdgcn_wave_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    if (rfits) {
      if (obj1->data.mesh.graph == lds_mesh + used) obj1->data.mesh.rec = recp;
      if (obj2->data.mesh.graph == lds_mesh + used) obj2->data.mesh.rec = recp;
    }
    if (fits) used = rfits ? ru + rw : used + gw + vw;
  }
}

// port-only: stageMeshesLDS run by a whole workgroup whose threads all hold identical objects
// (the fan kernel, before wave 0's convexBegin): the same buffer layout, contents and pointer
// rewrites as the single-wave version, with the copies spread over every thread of the block.
// Every thread of the block must call it (uniform control flow, block barriers).
HIPGL_DEVICE inline void stageMeshesLDSBlock(Obj* obj1, Obj* obj2) {
  int* lds_mesh = ldsMeshBuf();
  const int nact = (int)(blockDim.x * blockDim.y * blockDim.z);
  const int rank = (int)(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
  bool need1 = obj1->support == SUPPORT_HILLCLIMB && obj1->data.mesh.graph;
  bool need2 = obj2->support == SUPPORT_HILLCLIMB && obj2->data.mesh.graph;
  int used = 0;
  for (;;) {
    const int* g = need1 ? obj1->data.mesh.graph : need2 ? obj2->data.mesh.graph : nullptr;
    if (!g) break;
    const float* gv = need1 ? obj1->data.mesh.vert : obj2->data.mesh.vert;
    int nvert = need1 ? obj1->data.mesh.nvert : obj2->data.mesh.nvert;
    int numvert = g[0], numface = g[1];
    int gw = 2 + 3 * numvert + 3 * numface + 8;
    int vw = 3 * nvert;
    bool fits = used + gw + vw <= MJCCD_LDS_MESH_WORDS;
    if (fits) {
      int* dst = lds_mesh + used;
      constexpr int U = 8;
      for (int w0 = rank; w0 < gw; w0 += U * nact) {
        int v[U];
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; v[u] = g[w < gw ? w : 0]; }
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; if (w < gw) dst[w] = v[u]; }
      }
      const int* gvi = reinterpret_cast<const int*>(gv);
      int* dstv = dst + gw;
      for (int w0 = rank; w0 < vw; w0 += U * nact) {
        int v[U];
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; v[u] = gvi[w < vw ? w : 0]; }
#pragma unroll
        for (int u = 0; u < U; u++) { int w = w0 + u * nact; if (w < vw) dstv[w] = v[u]; }
      }
      __syncthreads();
    }
    if (need1 && obj1->data.mesh.graph == g) {
      if (fits) {
        obj1->data.mesh.graph = lds_mesh + used;
        obj1->data.mesh.vert = reinterpret_cast<const float*>(lds_mesh + used + gw);
      }
      need1 = false;
    }
    if (need2 && obj2->data.mesh.graph == g) {
      if (fits) {
        obj2->data.mesh.graph = lds_mesh + used;
        obj2->data.mesh.vert = reinterpret_cast<const float*>(lds_mesh + used + gw);
      }
      need2 = false;
    }
    const int ne = numvert + 3 * numface;
    const int ru = (used + gw + vw + 3) & ~3;
    const int rw = 4 * (ne + 64);
    const bool rfits = fits && numvert < 0xffff && ne < 0xffff &&
                       ru + rw <= MJCCD_LDS_MESH_WORDS;
    HcRec* recp = reinterpret_cast<HcRec*>(lds_mesh + ru);
    if (rfits) {
      const int* lg = lds_mesh + used;
      const float* lv = reinterpret_cast<const float*>(lds_mesh + used + gw);
      for (int e = rank; e < ne + 64; e += nact) {
        int sub = e < ne ? lg[2 + 2 * numvert + e] : -1;
        HcRec r;
        if (sub >= 0) {
          int nadr = lg[2 + sub];
          int gid = lg[2 + numvert + sub];
          r.x = lv[3 * gid + 0]; r.y = lv[3 * gid + 1]; r.z = lv[3 * gid + 2];
          r.packed = (int)((unsigned)sub | ((unsigned)nadr << 16));
        } else {
          r.x = 0.f; r.y = 0.f; r.z = 0.f;
          r.packed = 0xffff;
        }
        recp[e] = r;
      }
      __syncthreads();
    }
    if (rfits) {
      if (obj1->data.mesh.graph == lds_mesh + used) obj1->data.mesh.rec = recp;
      if (obj2->data.mesh.graph == lds_mesh + used) obj2->data.mesh.rec = recp;
    }
    if (fits) used = rfits ? ru + rw : used + gw + vw;
  }
}
#endif

// C:857-941  mjc_Convex.  rbound1/rbound2 are m->geom_rbound[g1], m->geom_rbound[g2]; the
// margin > 0 perturbation loop (C:877-938) needs them for its distinct-contact tolerance and
// returns -1 when either is negative (not given).  Output arrays then need room for 5 contacts.  o1/o2 come from initCCDObj/initMeshObj/
// initBoxObj (their margin and warm-start fields are ignored); margin is mjc_Convex's margin
// argument (the pair margin + gap computed by the driver), written into both objects like the
// two mjc_initCCDObj calls in C:859-860.
// Output arrays need room for maxContacts() entries (4 for box-mesh / mesh-mesh).
// Returns ncon (contacts: dist <= 0 penetration, normal from g1 to g2), or -1 when the result is
// not guaranteed identical to mjc_Convex (st->flags & MJCCD_FLAGS_FALLBACK; MJCCD_FLAG_UNSUPPORTED
// is always set then): a path that is not ported (NATIVECCD disabled; margin > 0 with MULTICCD on,
// a contact found and no rbound given -> perturbation loop), a capacity limit (ccd_iterations outside
// [1, MJCCD_MAX_ITERATIONS], max_contacts > MJCCD_STATUS_NX, polygon data > MJCCD_MAXPOLY), a
// > 24 edge EPA horizon (undefined in MuJoCo) or geom < 0.  The caller must fall back on -1.
HIPGL_DEVICE MJCCD_NOINLINE inline int convex(const Obj& o1, const Obj& o2, mjtNum margin,
                                              const Config& cfg, Scratch* s, mjtNum pos[][3],
                                              mjtNum normal[][3], mjtNum dist[], Status* st,
                                              mjtNum rbound1 = -1, mjtNum rbound2 = -1,
                                              int coop = 0) {
  MJCCD_FP_CONTRACT_OFF
  // init ccd objects (C:859-860 mjc_initCCDObj: same margin, warm start reset; copies because
  // the support functions update the warm-start indices)
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_MESH_WORDS > 0
  // coop: the caller's whole wave entered convex() for ONE pair (identical arguments on every
  // lane): every lane runs the same (uniform) computation, sharing the object copies, the
  // status and one polytope slot in LDS, and the lanes cooperate in the support / EPA scans;
  // lane 0 owns the results.  Needs one wave per workgroup (the LDS objects are per block).
  if (coop && blockDim.x * blockDim.y * blockDim.z > (unsigned)warpSize) coop = 0;
  __shared__ Obj lds_obj[2];
  Obj pobj[2];
  Obj& obj1 = coop ? lds_obj[0] : pobj[0];
  Obj& obj2 = coop ? lds_obj[1] : pobj[1];
  obj1 = o1;
  obj2 = o2;
#else
  Obj obj1 = o1, obj2 = o2;
#endif
  obj1.margin = margin;
  obj2.margin = margin;
  obj1.vertindex = -1;
  obj1.meshindex = -1;
  obj2.vertindex = -1;
  obj2.meshindex = -1;
  if (cfg.disable_nativeccd) {
    st->flags = MJCCD_FLAG_UNSUPPORTED;  // libccd path
    return -1;
  }
  int max_contacts = maxContacts(cfg, &obj1, &obj2);
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_MESH_WORDS > 0
  stageMeshesLDS(&obj1, &obj2);
  __shared__ Status lds_status;
  if (coop) st = &lds_status;
  st->wave_coop = coop;
  obj1.wave_coop = coop;
  obj2.wave_coop = coop;
#else
  st->wave_coop = 0;
  (void)coop;
#endif
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
  for (int k = 0; k < 8; k++) { g_stat_t[k][MJST_TID] = 0; g_stat_n[k][MJST_TID] = 0; }
  long long _t0 = wall_clock64();
  int _nre = 0, _nadd = 0, _sumgjk = 0, _sumepa = 0;
  struct _Pr { long long t0; int *nre, *nadd, *sg, *se; int mc, ncon0, gjk0, epa0, nv0, es0; mjtNum d0;
    const Obj* a; const Obj* b; mjtNum m; Status* st; Scratch* s; mjtNum r1, r2;
    __device__ ~_Pr() {
      static __device__ unsigned int ctr = 0;
      unsigned int c = atomicAdd(&ctr, 1u);
      if ((c % MJCCD_STATS) == 0) {
        mjtNum dx = a->pos[0]-b->pos[0], dy = a->pos[1]-b->pos[1], dz = a->pos[2]-b->pos[2];
        printf("MJST g %d %d ty %d %d mc %d n0 %d gjk0 %d epa0 %d es0 %d d0 %.3e nre %d nadd %d sgjk %d sepa %d cdist %.4f m %.4f rb %.4f %.4f t %lld sup %lld %lld %lld ns %d %d %d nv %d %d gjk %lld %d poly %lld %d epa %lld %d hcsteps %d T0 %lld T1 %lld\n",
               a->geom, b->geom, a->geom_type, b->geom_type, mc, ncon0, gjk0, epa0, es0, d0, *nre, *nadd, *sg, *se,
               sqrt(dx*dx+dy*dy+dz*dz), m, r1, r2, wall_clock64() - t0, g_stat_t[1][MJST_TID], g_stat_t[2][MJST_TID], g_stat_t[3][MJST_TID], g_stat_n[1][MJST_TID], g_stat_n[2][MJST_TID], g_stat_n[3][MJST_TID], a->data.mesh.nvert, b->data.mesh.nvert, g_stat_t[4][MJST_TID], g_stat_n[4][MJST_TID], g_stat_t[5][MJST_TID], g_stat_n[5][MJST_TID], g_stat_t[6][MJST_TID], g_stat_n[6][MJST_TID], g_stat_n[7][MJST_TID], t0, wall_clock64());
      }
    } } _pr{_t0, &_nre, &_nadd, &_sumgjk, &_sumepa, max_contacts, 0,0,0,0,0, 0, &obj1, &obj2, margin, st, s, rbound1, rbound2};
#endif

  // find initial contact
  int ncon = penetration(cfg, &obj1, &obj2, s, max_contacts, margin, pos, normal, dist, st);
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
  _pr.ncon0 = ncon; _pr.gjk0 = st->gjk_iterations; _pr.epa0 = st->epa_iterations; _pr.es0 = st->epa_status; _pr.d0 = st->dist;
#endif
  if (st->flags & MJCCD_FLAGS_FALLBACK) {
    st->flags |= MJCCD_FLAG_UNSUPPORTED;
    return -1;
  }

  // no additional contacts needed
  if (max_contacts > 1) {
    return ncon;
  }

  // look for additional contacts (C:877-938); box/mesh only, so the sphere/ellipsoid test is moot
  if (ncon == 1 && !cfg.disable_multiccd) {
    if (rbound1 < 0 || rbound2 < 0 || MJCCD_STATUS_NX < 5) {
      st->flags |= MJCCD_FLAG_UNSUPPORTED;
      return -1;
    }
    // multiCCD parameters
    const mjtNum relative_tolerance = 1e-3;
    const mjtNum perturbation_angle = 1e-3;

    // complete frame of initial contact
    mjtNum frame[9];
    copy3(frame, normal[0]);
    for (int k = 3; k < 9; k++) frame[k] = 0;
    mju_makeFrame(frame);

    // tolerance for determining if newly found contacts are distinct
    const mjtNum tolerance = relative_tolerance * (rbound1 < rbound2 ? rbound1 : rbound2);

    // axes and rotation angles for perturbation test
    mjtNum* axes[2] = {frame+3, frame+6};
    mjtNum angles[2] = {-perturbation_angle, perturbation_angle};

    // rotate both geoms, search for new contacts
    for (int axis_id = 0; axis_id < 2; ++axis_id) {
      for (int angle_id = 0; angle_id < 2; ++angle_id) {
        mjtNum* axis = axes[axis_id];
        mjtNum angle = angles[angle_id];

        // make rotation matrix rot
        mjtNum quat[4], rot[9];
        mju_axisAngle2Quat(quat, axis, angle);
        mju_quat2Mat(rot, quat);

        // rotate g1 around initial contact point
        mju_rotateFrame(pos[0], rot, obj1.mat, obj1.pos);

        // inversely rotate g2 around initial contact point
        mjtNum invrot[9] = {rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]};
        mju_rotateFrame(pos[0], invrot, obj2.mat, obj2.pos);

        // search for new contact
        int n = penetration(cfg, &obj1, &obj2, s, 1, margin, pos + ncon, normal + ncon,
                            dist + ncon, st);
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
        _nre++; _sumgjk += st->gjk_iterations; _sumepa += st->epa_iterations;
#endif
        if (st->flags & MJCCD_FLAGS_FALLBACK) {
          st->flags |= MJCCD_FLAG_UNSUPPORTED;
          return -1;
        }

        // check new contact
        if (n && isDistinctContact(pos, ncon + 1, tolerance)) {
          // set penetration of new point to equal that of initial point
          dist[ncon] = dist[0];
          // add new point
          ncon += 1;
#if defined(MJCCD_STATS) && defined(__HIP_DEVICE_COMPILE__)
          _nadd++;
#endif
        }

        // reset positions and orientations of g1 and g2
        copy3(obj1.pos, o1.pos);
        for (int k = 0; k < 9; k++) obj1.mat[k] = o1.mat[k];
        copy3(obj2.pos, o2.pos);
        for (int k = 0; k < 9; k++) obj2.mat[k] = o2.mat[k];
      }
    }
  }
  return ncon;
}

#define MJCCD_MULTI_PENDING (-2)

// port-only: under wave_coop every lane holds identical copies of the two objects, and the
// support functions read their fields on every step.  Private copies live in scratch (global
// memory latency); a per-wave LDS copy is read at LDS latency.  The values are identical, so
// results are unchanged.  Slots: one pair per wave of the workgroup (MJCCD_FAN_OBJ_WAVES).
#ifndef MJCCD_FAN_OBJ_WAVES
#define MJCCD_FAN_OBJ_WAVES 4
#endif
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_FAN_OBJ_WAVES > 0
HIPGL_DEVICE __forceinline__ Obj* fanObjSlot() {
  __shared__ Obj s_fan_obj[MJCCD_FAN_OBJ_WAVES][2];
  const unsigned t = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
  const unsigned nw = (blockDim.x * blockDim.y * blockDim.z + warpSize - 1) / warpSize;
  if (nw > MJCCD_FAN_OBJ_WAVES) return nullptr;
  return s_fan_obj[t / warpSize];
}
#else
HIPGL_DEVICE inline Obj* fanObjSlot() { return nullptr; }
#endif

// convex() up to the perturbation loop.  Returns -1 / ncon exactly like convex() when no
// perturbation loop runs, or MJCCD_MULTI_PENDING with obj1/obj2 (staged copies after the initial
// penetration, i.e. with its warm start) and the distinct-contact tolerance set.
// block_owner: the caller is the only wave of a multi-wave workgroup that stages meshes in LDS.
HIPGL_DEVICE inline int convexBegin(const Obj& o1, const Obj& o2, mjtNum margin,
                                    const Config& cfg, Scratch* s, mjtNum pos[][3],
                                    mjtNum normal[][3], mjtNum dist[], Status* st,
                                    mjtNum rbound1, mjtNum rbound2, Obj* obj1, Obj* obj2,
                                    mjtNum* tolerance_out, bool block_owner, int coop = 0,
                                    bool staged = false) {
  MJCCD_FP_CONTRACT_OFF
  const long long _mjp_cin = MJP_T();
  Obj* const out1 = obj1;
  Obj* const out2 = obj2;
  Obj* lo = coop ? fanObjSlot() : nullptr;
  if (lo) {
    obj1 = lo;
    obj2 = lo + 1;
  }
  *obj1 = o1;
  *obj2 = o2;
  // coop: the caller's whole wave runs this pair (identical arguments on every lane); lanes
  // cooperate in the support scans, lane 0 owns the results
  st->wave_coop = coop;
  obj1->wave_coop = coop;
  obj2->wave_coop = coop;
  obj1->margin = margin;
  obj2->margin = margin;
  obj1->vertindex = -1;
  obj1->meshindex = -1;
  obj2->vertindex = -1;
  obj2->meshindex = -1;
  if (cfg.disable_nativeccd) {
    st->flags = MJCCD_FLAG_UNSUPPORTED;  // libccd path
    return -1;
  }
  int max_contacts = maxContacts(cfg, obj1, obj2);
#if defined(__HIP_DEVICE_COMPILE__) && MJCCD_LDS_MESH_WORDS > 0
  const long long _mjp_c0 = MJP_T();
  if (!staged) stageMeshesLDS(obj1, obj2, block_owner);
  MJP_ADD(16, MJP_T() - _mjp_c0);
#else
  (void)staged;
  (void)block_owner;
#endif
  const long long _mjp_c1 = MJP_T();
  MJP_ADD(19, _mjp_c1 - _mjp_cin);
  int ncon = penetration<1>(cfg, obj1, obj2, s, max_contacts, margin, pos, normal, dist, st);
  const long long _mjp_c2 = MJP_T();
  MJP_ADD(17, _mjp_c2 - _mjp_c1);
  if (lo) {
    *out1 = *obj1;
    *out2 = *obj2;
  }
  if (st->flags & MJCCD_FLAGS_FALLBACK) {
    st->flags |= MJCCD_FLAG_UNSUPPORTED;
    return -1;
  }
  if (max_contacts > 1) {
    return ncon;
  }
  if (ncon == 1 && !cfg.disable_multiccd) {
    if (rbound1 < 0 || rbound2 < 0 || MJCCD_STATUS_NX < 5) {
      st->flags |= MJCCD_FLAG_UNSUPPORTED;
      return -1;
    }
    const mjtNum relative_tolerance = 1e-3;
    *tolerance_out = relative_tolerance * (rbound1 < rbound2 ? rbound1 : rbound2);
    MJP_ADD(18, MJP_T() - _mjp_c2);
    return MJCCD_MULTI_PENDING;
  }
  return ncon;
}

// the perturbed poses of rerun k (same expressions as convex()'s loop); p1/p2 must hold the
// unperturbed objects on entry
HIPGL_DEVICE inline void multiccdPerturb(int k, const mjtNum pos0[3], const mjtNum normal0[3],
                                         Obj* p1, Obj* p2) {
  MJCCD_FP_CONTRACT_OFF
  const mjtNum perturbation_angle = 1e-3;
  mjtNum frame[9];
  copy3(frame, normal0);
  for (int kk = 3; kk < 9; kk++) frame[kk] = 0;
  mju_makeFrame(frame);
  mjtNum* axes[2] = {frame+3, frame+6};
  mjtNum angles[2] = {-perturbation_angle, perturbation_angle};
  mjtNum* axis = axes[k >> 1];
  mjtNum angle = angles[k & 1];
  mjtNum quat[4], rot[9];
  mju_axisAngle2Quat(quat, axis, angle);
  mju_quat2Mat(rot, quat);
  mju_rotateFrame(pos0, rot, p1->mat, p1->pos);
  mjtNum invrot[9] = {rot[0], rot[3], rot[6], rot[1], rot[4], rot[7], rot[2], rot[5], rot[8]};
  mju_rotateFrame(pos0, invrot, p2->mat, p2->pos);
}

// one perturbed rerun.  b1/b2: the objects of convexBegin (unperturbed poses) with the warm
// start to run from.  Returns n (0/1) or -1 (fallback); ws_end gets the final warm start
// {obj1.vertindex, obj1.meshindex, obj2.vertindex, obj2.meshindex}.
HIPGL_DEVICE inline int multiccdRerun(const Config& cfg, const Obj& b1, const Obj& b2,
                                      mjtNum margin, const mjtNum pos0[3],
                                      const mjtNum normal0[3], int k, Scratch* s, Status* st,
                                      mjtNum pos[][3], mjtNum normal[][3], mjtNum dist[],
                                      int ws_end[4]) {
  MJCCD_FP_CONTRACT_OFF
  Obj pobj[2];
  Obj* lo = b1.wave_coop ? fanObjSlot() : nullptr;
  Obj& obj1 = lo ? lo[0] : pobj[0];
  Obj& obj2 = lo ? lo[1] : pobj[1];
  obj1 = b1;
  obj2 = b2;
  st->wave_coop = b1.wave_coop;
  multiccdPerturb(k, pos0, normal0, &obj1, &obj2);
  int n = penetration<1>(cfg, &obj1, &obj2, s, 1, margin, pos, normal, dist, st);
  ws_end[0] = obj1.vertindex; ws_end[1] = obj1.meshindex;
  ws_end[2] = obj2.vertindex; ws_end[3] = obj2.meshindex;
  if (st->flags & MJCCD_FLAGS_FALLBACK) {
    st->flags |= MJCCD_FLAG_UNSUPPORTED;
    return -1;
  }
  return n;
}

// true when penetration() on the perturbed objects p1/p2 does exactly the same work from warm
// start a[4] as from b[4]: gjk's first support() call (direction from the object centers, G:
// gjk/gjkSupport) leaves both objects in the same state.  false = not proven (recompute).
HIPGL_DEVICE inline bool multiccdSameStart(const Config& cfg, const Obj& p1, const Obj& p2,
                                           const int a[4], const int b[4]) {
  MJCCD_FP_CONTRACT_OFF
  MJP_ADD(8, 1);
  if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]) return true;
  Obj A1 = p1, A2 = p2;
  mjtNum x1[3], x2[3], x_k[3];
  mjc_center(x1, &A1);
  mjc_center(x2, &A2);
  mjtNum tol = cfg.ccd_tolerance;
  mjtNum tol2 = tol * tol;
  mjtNum min_norm2 = discreteGeoms(&A1, &A2) ? mjMINVAL2 : tol2;
  sub3(x_k, x1, x2);
  mjtNum x_norm;
  if ((x_norm = dot3(x_k, x_k)) < min_norm2) return false;
  x_norm = mju_sqrt(x_norm);
  mjtNum dir[3], dir_neg[3], tmp[3];
  scl3(dir_neg, x_k, 1 / x_norm);
  scl3(dir, dir_neg, -1);
  Obj B1 = p1, B2 = p2;
  A1.vertindex = a[0]; A1.meshindex = a[1]; A2.vertindex = a[2]; A2.meshindex = a[3];
  B1.vertindex = b[0]; B1.meshindex = b[1]; B2.vertindex = b[2]; B2.meshindex = b[3];
  objSupport(tmp, &A1, dir);
  objSupport(tmp, &B1, dir);
  if (A1.vertindex != B1.vertindex || A1.meshindex != B1.meshindex) return false;
  objSupport(tmp, &A2, dir_neg);
  objSupport(tmp, &B2, dir_neg);
  return A2.vertindex == B2.vertindex && A2.meshindex == B2.meshindex;
}

// speculative rerun results (from warm start `start`)
struct MulticcdRerunOut {
  mjtNum pos[3], normal[3], dist;
  int n;            // 0/1, -1 fallback
  int ws_end[4];
  // port-only: multiccdSameStart(rerun k+1's perturbed objects, start, ws_end) evaluated by the
  // rerun's own wave (1/0), or -1 when not evaluated.  multiccdMerge uses it exactly when its
  // incoming warm start for rerun k+1 is this ws_end (rerun k accepted as computed); the
  // function is pure, so the value is the one the merge would compute.
  int ss_next;
};

// port-only: the merge's warm-start test for rerun k+1 given rerun k's end warm start
// (identical expression to multiccdMerge's: perturbation of the unperturbed b1/b2 about the
// initial contact, then multiccdSameStart(start, ws_end)); -1 when the merge never asks
// (ws_end == start).
HIPGL_DEVICE inline int multiccdSameStartNext(const Config& cfg, const Obj& b1, const Obj& b2,
                                             const mjtNum pos0[3], const mjtNum normal0[3],
                                             int k, const int ws_end[4]) {
  MJCCD_FP_CONTRACT_OFF
  const int start[4] = {b1.vertindex, b1.meshindex, b2.vertindex, b2.meshindex};
  if (ws_end[0] == start[0] && ws_end[1] == start[1] && ws_end[2] == start[2] &&
      ws_end[3] == start[3]) {
    return -1;
  }
  Obj p1 = b1, p2 = b2;
  multiccdPerturb(k + 1, pos0, normal0, &p1, &p2);
  return multiccdSameStart(cfg, p1, p2, start, ws_end) ? 1 : 0;
}

// sequential acceptance of convex()'s loop over the 4 rerun results r[] (run from warm start
// `start` = the warm start of b1/b2).  pos/normal/dist hold contact 0 on entry.  Reruns whose
// true incoming warm start is not proven equivalent are recomputed here (s/st).
HIPGL_DEVICE inline int multiccdMerge(const Config& cfg, const Obj& b1, const Obj& b2,
                                      mjtNum margin, mjtNum tolerance,
                                      const MulticcdRerunOut* r, Scratch* s, Status* st,
                                      mjtNum pos[][3], mjtNum normal[][3], mjtNum dist[],
                                      int* nrecompute) {
  MJCCD_FP_CONTRACT_OFF
  int ncon = 1;
  const int start[4] = {b1.vertindex, b1.meshindex, b2.vertindex, b2.meshindex};
  int cur[4] = {start[0], start[1], start[2], start[3]};
  bool prev_spec = false;  // cur == r[k-1].ws_end (rerun k-1 accepted as computed)
  for (int k = 0; k < 4; k++) {
    int n, ws_end[4];
    bool ok = true;
    if (cur[0] != start[0] || cur[1] != start[1] || cur[2] != start[2] || cur[3] != start[3]) {
      if (prev_spec && r[k - 1].ss_next >= 0) {
        ok = r[k - 1].ss_next != 0;
      } else {
      Obj p1 = b1, p2 = b2;
      multiccdPerturb(k, pos[0], normal[0], &p1, &p2);
      const long long _mjp_tss = MJP_T();
      ok = multiccdSameStart(cfg, p1, p2, start, cur);
      MJP_ADD(7, MJP_T() - _mjp_tss);
      }
    }
    prev_spec = ok;
    if (ok) {
      n = r[k].n;
      if (n > 0) {
        copy3(pos[ncon], r[k].pos);
        copy3(normal[ncon], r[k].normal);
        dist[ncon] = r[k].dist;
      }
      for (int i = 0; i < 4; i++) ws_end[i] = r[k].ws_end[i];
    } else {
      Obj c1 = b1, c2 = b2;
      c1.vertindex = cur[0]; c1.meshindex = cur[1]; c2.vertindex = cur[2]; c2.meshindex = cur[3];
      n = multiccdRerun(cfg, c1, c2, margin, pos[0], normal[0], k, s, st, pos + ncon,
                        normal + ncon, dist + ncon, ws_end);
      if (nrecompute) (*nrecompute)++;
    }
    if (n < 0) {
      st->flags |= MJCCD_FLAG_UNSUPPORTED;
      return -1;
    }
    if (n && isDistinctContact(pos, ncon + 1, tolerance)) {
      dist[ncon] = dist[0];
      ncon += 1;
    }
    for (int i = 0; i < 4; i++) cur[i] = ws_end[i];
  }
  return ncon;
}

}  // namespace mjccd

#pragma pop_macro("mju_sqrt")
#pragma pop_macro("mju_abs")
#pragma pop_macro("mjMAXCONPAIR")
#pragma pop_macro("mjMAXVAL")
#pragma pop_macro("mjMINVAL")

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#undef MJCCD_FP_CONTRACT_OFF
