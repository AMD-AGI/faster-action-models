#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GPU data structures for rigid body physics simulation.
//
// SoA (Structure of Arrays) layout is used for all per-body / per-geom /
// per-joint arrays so that GPU warps read contiguous memory for any single
// field.  The DevicePhysics* structs carry raw device pointers and are
// passed by value to HIP kernels.

#include "hipgl/types.h"
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace hipgl {
namespace physics {

constexpr int MAX_BODIES       = 192;
constexpr int MAX_JOINTS       = 128;
constexpr int MAX_GEOMS        = 2048;  // host-side arrays only (RoboTwin cluttered tables: ~600)
constexpr int MAX_DOF          = 128;
constexpr int MAX_ACTUATORS    = 64;
constexpr int MAX_CONTACTS     = 2048;
constexpr int MAX_CONSTRAINTS  = 8192;
constexpr int MAX_TREE_DEPTH   = 32;
constexpr int MAX_PAIR_FILTERS = 1024;
constexpr int MAX_EQUALITY     = 64;
constexpr int MAX_CAMERAS      = 16;
constexpr int MAX_SITES        = 256;
constexpr int MAX_MESHES       = 1024;  // host-side parse arrays only (RoboTwin scenes: ~360 meshes)
constexpr int MAX_MESH_TOTAL_VERTS = 16777216;  // max total expanded vertices across all meshes (16M; RoboTwin scenes ~9M)
constexpr int MAX_CONVEX_HULL_VERTS = 256;     // max unique verts per hull for collision
constexpr int MAX_CONVEX_HULL_FACES = 512;     // max triangular faces per hull
constexpr int MAX_CONVEX_HULL_EDGES = 768;     // max unique edges per hull
constexpr int MAX_NAME_LEN     = 128;
#ifndef KERNEL_MAXNV_VALUE
#define KERNEL_MAXNV_VALUE 60
#endif
constexpr int KERNEL_MAXNV     = KERNEL_MAXNV_VALUE;
constexpr int LDS_STRIDE       = KERNEL_MAXNV + 1;  // +1 padding eliminates LDS bank conflicts (stride 49 mod 32 = 17)

// ── Enumerations ────────────────────────────────────────────────────────────

enum JointType : int {
    JOINT_FREE    = 0,   // 7 qpos (pos3 + quat4), 6 qvel (v3 + w3)
    JOINT_BALL    = 1,   // 4 qpos (quat), 3 qvel (angular vel)
    JOINT_HINGE   = 2,   // 1 qpos, 1 qvel (revolute)
    JOINT_SLIDE   = 3,   // 1 qpos, 1 qvel (prismatic)
};

enum GeomType : int {
    GEOM_SPHERE   = 0,
    GEOM_BOX      = 1,
    GEOM_CAPSULE  = 2,
    GEOM_PLANE    = 3,
    GEOM_MESH     = 4,
    GEOM_CYLINDER = 5,
};

// MuJoCo site "type" stores raw mjtGeom values, not the compact GeomType above.
constexpr int SITE_TYPE_SPHERE    = 2;
constexpr int SITE_TYPE_CAPSULE   = 3;
constexpr int SITE_TYPE_ELLIPSOID = 4;
constexpr int SITE_TYPE_CYLINDER  = 5;
constexpr int SITE_TYPE_BOX       = 6;

enum ActuatorType : int {
    ACTUATOR_MOTOR    = 0,   // force = gain * ctrl
    ACTUATOR_POSITION = 1,   // force = kp * (ctrl - qpos) - kv * qvel
    ACTUATOR_VELOCITY = 2,   // force = kv * (ctrl - qvel)
    ACTUATOR_GENERAL  = 3,   // force = gainprm[0]*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
};

enum EqualityType : int {
    EQUALITY_WELD     = 0,
    EQUALITY_JOINT    = 1,
};

enum IntegratorType : int {
    INTEGRATOR_EULER     = 0,
    INTEGRATOR_IMPLICIT  = 1,
};

enum SolverType : int {
    SOLVER_PGS    = 0,
    SOLVER_NEWTON = 1,
};

// ── Quaternion helpers (w, x, y, z convention matching MuJoCo) ──────────────

struct Quat {
    float w, x, y, z;

    HIPGL_HOST_DEVICE Quat() : w(1), x(0), y(0), z(0) {}
    HIPGL_HOST_DEVICE Quat(float w_, float x_, float y_, float z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    HIPGL_HOST_DEVICE Quat operator*(Quat q) const {
        return {w*q.w - x*q.x - y*q.y - z*q.z,
                w*q.x + x*q.w + y*q.z - z*q.y,
                w*q.y - x*q.z + y*q.w + z*q.x,
                w*q.z + x*q.y - y*q.x + z*q.w};
    }

    HIPGL_HOST_DEVICE Quat conjugate() const { return {w, -x, -y, -z}; }

    HIPGL_HOST_DEVICE float norm_sq() const { return w*w + x*x + y*y + z*z; }

    HIPGL_HOST_DEVICE Quat normalized() const {
        float n = sqrtf(norm_sq());
        if (n < 1e-12f) return {1, 0, 0, 0};
        float inv = 1.0f / n;
        return {w*inv, x*inv, y*inv, z*inv};
    }

    HIPGL_HOST_DEVICE void rotate_vec(const float* v, float* out) const {
        float vx = v[0], vy = v[1], vz = v[2];
        float tx = 2.0f * (y*vz - z*vy);
        float ty = 2.0f * (z*vx - x*vz);
        float tz = 2.0f * (x*vy - y*vx);
        out[0] = vx + w*tx + (y*tz - z*ty);
        out[1] = vy + w*ty + (z*tx - x*tz);
        out[2] = vz + w*tz + (x*ty - y*tx);
    }

    HIPGL_HOST_DEVICE void to_mat3(float* m) const {
        float xx = x*x, yy = y*y, zz = z*z;
        float xy = x*y, xz = x*z, yz = y*z;
        float wx = w*x, wy = w*y, wz = w*z;
        m[0] = 1 - 2*(yy+zz); m[1] = 2*(xy-wz);     m[2] = 2*(xz+wy);
        m[3] = 2*(xy+wz);     m[4] = 1 - 2*(xx+zz); m[5] = 2*(yz-wx);
        m[6] = 2*(xz-wy);     m[7] = 2*(yz+wx);     m[8] = 1 - 2*(xx+yy);
    }

    static HIPGL_HOST_DEVICE Quat from_axis_angle(float ax, float ay, float az, float angle) {
        float ha = angle * 0.5f;
        float s = sinf(ha);
        return {cosf(ha), ax*s, ay*s, az*s};
    }
};

// ── Contact pair (output of collision detection) ────────────────────────────

struct ContactPair {
    int    geom1;
    int    geom2;
    int    body1;
    int    body2;
    double pos[3];
    double normal[3];
    double depth;
    double friction;
};

// ── AABB (for broadphase) ───────────────────────────────────────────────────

struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

// ── Device-side model (static, uploaded once) ───────────────────────────────

struct DevicePhysicsModel {
    int nbody;
    int njnt;
    int ngeom;
    int nq;
    int nv;
    int nu;

    float timestep;
    float gravity[3];
    double gravity_d[3];
    int   cone_type;    // 0 = pyramidal, 1 = elliptic
    int   friction_combine; // pair sliding friction: 0 = max (MuJoCo), 1 = mean (PhysX default)
    float max_depen_vel;    // contact depenetration speed cap (m/s), 0 = off (MuJoCo)
    int   max_condim;       // max geom condim (pyramidal cone: contact block = 2*(min(max_condim,4)-1) rows)
    float opt_density;   // fluid medium density (kg/m³), 0 = no fluid
    float opt_viscosity; // fluid kinematic viscosity (m²/s)

    // ── Body arrays [nbody] ─────────────────────────────────────────────
    int*   body_parent;
    int*   body_weldid;         // MuJoCo weld body ID: walk up until body with joint
    int*   body_jnt_adr;
    int*   body_jnt_num;
    float* body_pos;            // [nbody * 3]
    float* body_quat;           // [nbody * 4]
    float* body_mass;
    float* body_inertia;        // [nbody * 3]
    float* body_ipos;           // [nbody * 3]
    float* body_iquat;          // [nbody * 4] inertial frame orientation

    // Double-precision shadow for physics-critical model params
    double* body_mass_d;        // [nbody]
    double* body_inertia_d;     // [nbody * 3]
    double* body_ipos_d;        // [nbody * 3]
    double* body_iquat_d;       // [nbody * 4]
    double* body_pos_d;         // [nbody * 3]
    double* body_quat_d;        // [nbody * 4]
    // Optional per-world fixed-body local poses (LIBERO samples fixture poses
    // per episode). nullptr = every world uses body_pos/body_quat above.
    // When set, FK reads slice [env * nbody] of these instead.
    float*  body_pos_env;       // [body_pose_nenv * nbody * 3]
    float*  body_quat_env;      // [body_pose_nenv * nbody * 4]
    double* body_pos_env_d;     // [body_pose_nenv * nbody * 3]
    double* body_quat_env_d;    // [body_pose_nenv * nbody * 4]
    int     body_pose_nenv;

    float* body_inertia_box;     // [nbody * 3] equivalent inertia box FULL dimensions for fluid drag
    float* body_invweight0;      // [nbody * 2] MuJoCo-precomputed inverse effective inertia
                                 // [:,0] = translational, [:,1] = rotational (from CRBA at qpos0)
    double* body_invweight0_d;   // [nbody * 2] double-precision shadow
    float* dof_invweight0;       // [nv] MuJoCo-precomputed DOF inverse effective inertia
    double* dof_invweight0_d;    // [nv] double-precision shadow

    // Gravity compensation (MJCF body gravcomp): the bodies with gravcomp != 0 and their
    // factors. n_gravcomp == 0 (nullptrs) for every model without gravcomp bodies.
    int     n_gravcomp;
    int*    gravcomp_body;       // [n_gravcomp]
    double* gravcomp_scale_d;    // [n_gravcomp]

    int    tree_depth;
    int*   level_start;
    int*   level_body_ids;

    // ── Precomputed body-to-root chain (eliminates parent-chasing in constraint Jacobian) ──
    // body_chain_jnt[body_chain_start[b] .. body_chain_start[b]+body_chain_len[b]-1]
    // contains (joint_index, joint_body) pairs flattened as int2 for each joint
    // along the chain from body b to root.
    int*   body_chain_start;    // [nbody] start index into body_chain_data
    int*   body_chain_len;      // [nbody] number of joints in chain
    int*   body_chain_jnt_idx;  // [total_chain_entries] joint index
    int*   body_chain_jnt_body; // [total_chain_entries] body that owns this joint
    int*   body_chain_parent;   // [total_chain_entries] parent body of joint's body
    int    body_chain_total;    // total entries across all bodies

    // ── Joint arrays [njnt] ─────────────────────────────────────────────
    int*   jnt_type;
    int*   jnt_body;
    int*   jnt_qpos_adr;
    int*   jnt_dof_adr;
    float* jnt_axis;            // [njnt * 3]
    float* jnt_pos;             // [njnt * 3]
    double* jnt_axis_d;         // [njnt * 3] double-precision shadow
    double* jnt_pos_d;          // [njnt * 3] double-precision shadow
    double* jnt_damping_d;      // [njnt] double-precision shadow
    double* jnt_armature_d;     // [njnt] double-precision shadow
    float* jnt_range;           // [njnt * 2]
    double* jnt_range_d;        // [njnt * 2] double-precision shadow
    float* jnt_stiffness;
    double* jnt_stiffness_d;    // [njnt] double-precision shadow
    float* jnt_damping;
    float* jnt_armature;        // reflected rotor inertia (added to M diagonal)
    float* jnt_frictionloss;    // dry friction torque
    double* jnt_frictionloss_d; // [njnt] double-precision shadow
    float* jnt_solref_lim;      // [njnt * 2] per-joint limit solref
    double* jnt_solref_lim_d;   // [njnt * 2] double-precision shadow
    float* jnt_solimp_lim;      // [njnt * 5] per-joint limit solimp
    double* jnt_solimp_lim_d;   // [njnt * 5] double-precision shadow
    int*   jnt_limited;         // 0 or 1 per joint
    float* qpos0;               // [nq] default/reference joint positions
    double* qpos0_d;            // [nq] double-precision shadow

    // ── Geom arrays [ngeom] ─────────────────────────────────────────────
    int*   geom_type;
    int*   geom_body;
    float* geom_pos;            // [ngeom * 3]
    float* geom_quat;           // [ngeom * 4]
    float* geom_size;           // [ngeom * 3]
    float* geom_friction;       // [ngeom * 3]
    float* geom_rgba;           // [ngeom * 4]
    double* geom_pos_d;         // [ngeom * 3] double-precision shadow
    double* geom_quat_d;        // [ngeom * 4] double-precision shadow
    double* geom_size_d;        // [ngeom * 3] double-precision shadow
    double* geom_friction_d;    // [ngeom * 3] double-precision shadow
    int*   geom_contype;
    int*   geom_conaffinity;
    int*   geom_onesided;       // [ngeom] 1 = one-sided static box (instinct_sim_onesided_geoms), or nullptr
    int*   geom_condim;         // contact dimensionality [ngeom]
    float* geom_solref;         // solver reference params [ngeom * 2] (timeconst, dampratio)
    double* geom_solref_d;      // [ngeom * 2] double-precision shadow
    float* geom_solimp;         // solver impedance params [ngeom * 5]
    double* geom_solimp_d;      // [ngeom * 5] double-precision shadow
    float* geom_margin;         // contact margin [ngeom]
    float* geom_gap;            // contact gap [ngeom]

    // ── Actuator arrays [nu] ────────────────────────────────────────────
    int*   actuator_jnt;
    int*   actuator_type;       // ActuatorType enum
    float* actuator_gain;       // kp for position, gain for motor
    float* actuator_kv;         // damping gain for position actuators
    float* actuator_biasprm;    // [nu * 3] for ACTUATOR_GENERAL: bias0, bias_qpos, bias_qvel
    float* actuator_forcerange; // [nu * 2]
    float* actuator_ctrlrange;  // [nu * 2]
    double* actuator_gain_d;
    double* actuator_kv_d;
    double* actuator_biasprm_d;
    double* actuator_forcerange_d;
    double* actuator_ctrlrange_d;

    // ── Equality constraints ────────────────────────────────────────────
    int    nequality;
    int*   eq_type;             // EqualityType enum [nequality]
    int*   eq_body1;            // [nequality]
    int*   eq_body2;            // [nequality]
    float* eq_data;             // constraint data [nequality * 6]

    // ── Collision pair exclusions ────────────────────────────────────────
    int    npair_filter;
    int*   pair_filter;         // packed (body1 << 16 | body2), sorted

    // ── Camera arrays [ncamera] ─────────────────────────────────────────
    int    ncamera;
    int*   cam_body;            // body the camera is attached to [ncamera]
    float* cam_pos;             // local position [ncamera * 3]
    float* cam_quat;            // local orientation [ncamera * 4]
    float* cam_fovy;            // vertical FOV in radians [ncamera]

    // ── Site arrays [nsite] ─────────────────────────────────────────────
    int    nsite;
    int*   site_body;           // body the site is attached to [nsite]
    int*   site_type;           // raw MuJoCo mjtGeom site type [nsite]
    int*   site_group;          // MuJoCo site group [nsite]
    float* site_pos;            // local position in body frame [nsite * 3]
    float* site_size;           // local size [nsite * 3]
    float* site_rgba;           // site color [nsite * 4]
    float* site_quat;           // local orientation in body frame [nsite * 4]
    double* site_pos_d;         // original double local pose for OSC only
    double* site_quat_d;        // renderer continues using the float site arrays

    // Per-geom material properties (for Gouraud shading)
    float* geom_specular;       // [ngeom] MuJoCo specular coeff
    float* geom_shininess;      // [ngeom] MuJoCo shininess (GL = this * 128)
    float* geom_emission;       // [ngeom] MuJoCo emission coeff
    float* geom_reflectance;    // [ngeom] MuJoCo mat_reflectance [0,1]

    // ── Mesh arrays (for rendering) ─────────────────────────────────────
    int    nmesh;
    int    mesh_total_verts;
    int*   mesh_vert_adr;       // [nmesh] start index in mesh_vert/mesh_normal
    int*   mesh_vert_num;       // [nmesh] number of expanded vertices
    float* mesh_vert;           // [mesh_total_verts * 3] packed vertex positions
    float* mesh_normal;         // [mesh_total_verts * 3] packed vertex normals
    int*   geom_meshid;         // [ngeom] mesh index for GEOM_MESH geoms, -1 for primitives
    int*   geom_group;          // [ngeom] MuJoCo geom group (0=visual, 1+=collision-only)

    // ── Convex hull data (for collision) ────────────────────────────────
    int    convex_total_verts;
    int    convex_total_faces;
    int    convex_total_edges;
    int*   mesh_convex_vert_adr;       // [nmesh] start index in mesh_convex_vert
    int*   mesh_convex_vert_num;       // [nmesh] number of unique hull vertices
    float* mesh_convex_vert;           // [convex_total_verts * 3] unique hull vertices (local frame)
    int*   mesh_convex_face_adr;       // [nmesh] start index in mesh_convex_face
    int*   mesh_convex_face_num;       // [nmesh] number of triangular faces
    int*   mesh_convex_face;           // [convex_total_faces * 3] face vertex indices (triangles)
    float* mesh_convex_face_normal;    // [convex_total_faces * 3] outward face normals
    int*   mesh_convex_edge_adr;       // [nmesh] start index in mesh_convex_edge
    int*   mesh_convex_edge_num;       // [nmesh] number of unique edges
    int*   mesh_convex_edge;           // [convex_total_edges * 2] unique edge vertex pairs

    // ── MuJoCo convex collision (physics_mjccd.h port of mjc_Convex) ──
    // Set by PhysicsEngine-level binding set_mujoco_convex() from the live MuJoCo model.
    // mjc_enabled == 0 (default) keeps the legacy GPU mesh collision path.
    int     mjc_enabled = 0;
    int     mjc_ccd_iterations = 35;
    double  mjc_ccd_tolerance = 1e-6;
    int     mjc_disable_nativeccd = 0;
    int     mjc_disable_multiccd = 0;
    int*    geom_mj_mesh = nullptr;       // [ngeom] MuJoCo mesh id (geom_dataid) for mesh geoms, -1 otherwise
    double* geom_mj_pos = nullptr;        // [ngeom * 3] MuJoCo geom_pos (body frame, recentred mesh frame)
    double* geom_mj_quat = nullptr;       // [ngeom * 4] MuJoCo geom_quat
    double* geom_mj_size = nullptr;       // [ngeom * 3] MuJoCo geom_size
    double* geom_mj_rbound = nullptr;     // [ngeom] MuJoCo geom_rbound (multiCCD tolerance)
    int*    mj_mesh_vertadr = nullptr;    // MuJoCo mesh arrays, unchanged (mjModel semantics)
    int*    mj_mesh_vertnum = nullptr;
    float*  mj_mesh_vert = nullptr;
    int*    mj_mesh_graphadr = nullptr;
    int*    mj_mesh_graph = nullptr;
    int*    mj_mesh_polynum = nullptr;
    int*    mj_mesh_polyadr = nullptr;
    double* mj_mesh_polynormal = nullptr;
    int*    mj_mesh_polyvertadr = nullptr;
    int*    mj_mesh_polyvertnum = nullptr;
    int*    mj_mesh_polyvert = nullptr;
    int*    mj_mesh_polymapadr = nullptr;
    int*    mj_mesh_polymapnum = nullptr;
    int*    mj_mesh_polymap = nullptr;

    // Texture data for mesh rendering
    float*          mesh_texcoord;     // [mesh_total_verts * 2] UV coordinates
    int             ntexture;
    int*            tex_width;         // [ntexture]
    int*            tex_height;        // [ntexture]
    int*            tex_offset;        // [ntexture] byte offset into tex_data
    int*            tex_iscube;        // [ntexture] 1 = cube map, 0 = 2D
    TextureMipLevel* tex_mip;          // [ntexture * MAX_TEXTURE_MIP_LEVELS], optional
    unsigned char*  tex_data;          // packed RGB texture pixels
    int*            geom_texid;        // [ngeom] texture index, -1 if no texture
    float*          geom_texrepeat;    // [ngeom * 2] MuJoCo mat_texrepeat (S, T tiling)
    int             tex_generation;    // process-unique id of the current texture upload (0: none);
                                       // consumers caching texture-derived data (RayTracer) key on it
};

// ── Device-side data (dynamic, per-environment) ─────────────────────────────

struct DevicePhysicsData {
    int num_envs;
    int nq, nv, nu;
    int nbody, ngeom;

    float* qpos;
    float* qvel;
    double* qpos_d;          // double-precision shadow for integration
    double* qvel_d;          // double-precision shadow for integration
    double* qacc;
    double* qacc_warmstart;
    float*  ctrl;
    double* ctrl_d;           // double-precision shadow for physics
    float*  qfrc_applied;
    double* qfrc_applied_d;   // double-precision shadow for physics

    float*  xpos;            // [num_envs * nbody * 3] (float for rendering)
    float*  xquat;           // [num_envs * nbody * 4] (float for rendering)
    double* xpos_d;          // [num_envs * nbody * 3] double-precision for physics
    double* xquat_d;         // [num_envs * nbody * 4] double-precision for physics
    double* jnt_xaxis_d;     // [num_envs * njnt * 3] world-frame joint axes (pre-joint frame)
    float*  cvel;
    float*  xfrc;

    float*  geom_xpos;       // [num_envs * ngeom * 3] (float for rendering)
    float*  geom_xquat;      // [num_envs * ngeom * 4] (float for rendering)
    double* geom_xpos_d;     // [num_envs * ngeom * 3] double for collision
    double* geom_xquat_d;    // [num_envs * ngeom * 4] double for broadphase
    double* geom_xmat_d;     // [num_envs * ngeom * 9] double rotation matrix for collision
    AABB*   geom_aabb;

    ContactPair* contacts;
    int*         ncon;
    int*         body_pair_ncon;  // [num_envs * nbody * nbody] per-body-pair contact counter
    // MuJoCo convex collision queue (box-mesh / mesh-mesh pairs are deferred to kernel_convex_mj)
    int*  mjc_queue;            // [mjc_queue_cap * 3] (env, geom_i, geom_j)
    int*  mjc_queue_count;      // [1]
    int   mjc_queue_cap;
    void* mjc_scratch;          // [mjc_workers] MjcWorkerScratch
    int   mjc_workers;
    int*  mjc_fallback_count;   // [1] pairs where the port returned -1 (not guaranteed identical)

    double* constraint_J;
    double* constraint_rhs;
    double* constraint_lo;
    double* constraint_hi;
    double* constraint_lam;
    int*   constraint_friction_parent;
    int*   constraint_friction_sibling;
    double* constraint_friction_mu;
    double* constraint_D;
    int*   nconstraint;

    float* body_inv_mass;
    float* body_inv_inertia;

    double* qfrc_bias;
    double* qfrc_fluid;     // [num_envs * nv] fluid drag forces

    float* time;

    // ── Solver workspace (pre-allocated, reused each step) ───────────────
    double* solver_M;           // dense mass matrix   [ne * KERNEL_MAXNV * KERNEL_MAXNV]
    double* solver_H;           // Newton H = M+J^T DJ [ne * KERNEL_MAXNV * KERNEL_MAXNV]
    double* solver_rhs;         // Newton RHS vector   [ne * KERNEL_MAXNV]
    double* solver_M_inv_diag;  // diagonal M^-1       [ne * MAX_DOF]
    double* solver_M_inv;       // full M^-1 matrix    [ne * KERNEL_MAXNV * KERNEL_MAXNV]

    // ── Newton per-constraint workspace (global memory) ────────────────
    double* newton_x;            // [ne * MAX_CONSTRAINTS]
    double* newton_force;        // [ne * MAX_CONSTRAINTS]
    int*    newton_state;        // [ne * MAX_CONSTRAINTS]
    double* newton_jv;           // [ne * MAX_CONSTRAINTS]
    bool*   newton_is_elliptic;  // [ne * MAX_CONSTRAINTS]

    // ── RNEA workspace (pre-allocated, reused each step) ─────────────────
    double* rnea_omega;        // angular velocity     [ne * nbody * 3]
    double* rnea_alpha;        // angular acceleration [ne * nbody * 3]
    double* rnea_acc;          // linear acceleration  [ne * nbody * 3]
    double* rnea_fnet;         // net force            [ne * nbody * 3]
    double* rnea_tnet;         // net torque           [ne * nbody * 3]

    // ── Site workspace ─────────────────────────────────────────────────
    int    nsite;
    float* site_xpos;          // world position  [ne * nsite * 3]
    float* site_xquat;         // world orientation quaternion [ne * nsite * 4]
    float* site_xmat;          // world rotation   [ne * nsite * 9]
};

// ── Host-side model description (populated by XML parser, then uploaded) ────

struct BodyDesc {
    int   parent;
    float pos[3];
    float quat[4];
    float mass;
    float inertia[3];
    float ipos[3];
    float iquat[4];     // inertial frame orientation (w,x,y,z); identity = {1,0,0,0}
    int   jnt_adr;
    int   jnt_num;
    char  name[MAX_NAME_LEN];
    double pos_d[3];
    double quat_d[4];
    double mass_d;
    double inertia_d[3];
    double ipos_d[3];
    double iquat_d[4];
    double gravcomp;    // MJCF body gravcomp: fraction of this body's weight cancelled (0 = off)
};

struct JointDesc {
    JointType type;
    int   body;
    int   qpos_adr;
    int   dof_adr;
    float axis[3];
    float pos[3];
    float range[2];
    float stiffness;
    float damping;
    float armature;
    float frictionloss;
    float solref_lim[2];    // per-joint limit solver ref {timeconst, dampratio}
    float solimp_lim[5];    // per-joint limit solver impedance
    int   limited;
    char  name[MAX_NAME_LEN];
    double axis_d[3];
    double pos_d[3];
    double damping_d;
    double armature_d;
};

struct GeomDesc {
    GeomType type;
    int   body;
    float pos[3];
    float quat[4];
    float size[3];
    float friction[3];
    float rgba[4];
    int   contype;
    int   conaffinity;
    int   condim;
    float solref[2];    // solver reference: {timeconst, dampratio}
    float solimp[5];    // solver impedance: {dmin, dmax, width, midpoint, power}
    float margin;       // contact margin (extends contact detection range)
    float gap;          // contact gap (shifts contact activation)
    float density;      // geom density (kg/m³), for auto-computing body mass/inertia
    int   mesh_id;      // index into meshes[] for GEOM_MESH, -1 otherwise
    int   group;        // MuJoCo geom group (0 = visual, 1+ = collision-only)
    int   onesided;     // extension: static box that only touches mesh hulls centered above its top face
    char  name[MAX_NAME_LEN];
};

struct ActuatorDesc {
    ActuatorType type;
    int   jnt;
    float gain;             // kp or gear or gainprm[0]
    float kv;               // velocity gain (for position actuators)
    float biasprm[3];       // for ACTUATOR_GENERAL: {bias0, bias_qpos, bias_qvel}
    float forcerange[2];
    float ctrlrange[2];
    double gain_d;
    double kv_d;
    double biasprm_d[3];
    double forcerange_d[2];
    double ctrlrange_d[2];
};

struct EqualityDesc {
    EqualityType type;
    int   body1;
    int   body2;
    float data[6];          // type-specific parameters
};

// CameraDesc::unsupported bits: MJCF camera features the parser does not model
// (the raster and the ray tracer both render such a camera as a fixed
// perspective fovy camera at pos/quat). Renderers warn when they see them.
enum CameraUnsupported : int {
    CAM_UNSUP_MODE        = 1,   // mode != "fixed" (track, trackcom, targetbody, ...)
    CAM_UNSUP_INTRINSICS  = 2,   // sensorsize / focal / focalpixel / principal*
    CAM_UNSUP_ORTHO       = 4,   // orthographic="true"
    CAM_UNSUP_ORIENTATION = 8,   // axisangle / zaxis (orientation ignored)
};

struct CameraDesc {
    int   body;
    float pos[3];
    float quat[4];
    float fovy;
    char  name[MAX_NAME_LEN];
    int   unsupported;          // CameraUnsupported bits (0 = fully modelled)
};

struct SiteDesc {
    int   body;
    int   type;
    int   group;
    float pos[3];
    float size[3];
    float rgba[4];
    float quat[4];
    double pos_d[3];            // parsed directly, not widened from float pos
    double quat_d[4];           // parsed directly, not widened from float quat
    char  name[MAX_NAME_LEN];
};

struct SceneLightDesc {
    float pos[3];
    float dir[3];
    float diffuse[3];
    float specular[3];
    float ambient[3];
    float cutoff;
    float exponent;
    float attenuation[3];
    int   directional;
    char  name[MAX_NAME_LEN];
};

constexpr int MAX_SCENE_LIGHTS = 8;

struct PhysicsModelDesc {
    int nbody;
    int njnt;
    int ngeom;
    int nq;
    int nv;
    int nu;
    float timestep;
    float gravity[3];
    double gravity_d[3];

    BodyDesc     bodies[MAX_BODIES];
    JointDesc    joints[MAX_JOINTS];
    GeomDesc     geoms[MAX_GEOMS];
    ActuatorDesc actuators[MAX_ACTUATORS];

    int tree_depth;
    int level_start[MAX_TREE_DEPTH + 1];
    int level_body_ids[MAX_BODIES];

    int npair_filter;
    int pair_filter[MAX_PAIR_FILTERS];

    int nequality;
    EqualityDesc equalities[MAX_EQUALITY];

    int ncamera;
    CameraDesc cameras[MAX_CAMERAS];

    int nsite;
    SiteDesc sites[MAX_SITES];

    int nscene_light;
    SceneLightDesc scene_lights[MAX_SCENE_LIGHTS];

    // Headlight settings (from <visual><headlight>)
    float headlight_ambient[3];   // default: {0.1, 0.1, 0.1}
    float headlight_diffuse[3];   // default: {0.4, 0.4, 0.4}
    float headlight_specular[3];  // default: {0.5, 0.5, 0.5}
    int   headlight_active;       // default: 1

    float qpos0[MAX_DOF];

    // Mesh data (loaded from STL/OBJ/MSH files) -- dynamically allocated
    int nmesh;
    struct MeshInfo {
        int   vert_adr;            // start index in mesh_vert/mesh_normal
        int   vert_num;            // number of expanded vertices (= nfaces * 3)
        float scale[3];            // mesh scale from MJCF
        char  name[MAX_NAME_LEN];
        // UVs were auto-generated (mesh carried none) and still hold raw local
        // x,y. MuJoCo's texgen scale needs geom_size, so a deferred pass after
        // geom parsing rescales exactly these meshes. See physics_model.cpp.
        bool  generated_texcoords;
    } meshes[MAX_MESHES];
    int   mesh_total_verts;
    float* mesh_vert;              // [mesh_total_verts * 3] heap-allocated
    float* mesh_normal;            // [mesh_total_verts * 3] heap-allocated
    float* mesh_texcoord;          // [mesh_total_verts * 2] heap-allocated (UV coords)

    // Convex hull collision data (computed at load time from render meshes)
    struct ConvexHullInfo {
        int vert_adr;       // start index in convex_hull_vert
        int vert_num;       // number of unique hull vertices
        int face_adr;       // start index in convex_hull_face / convex_hull_face_normal
        int face_num;       // number of triangular faces
        int edge_adr;       // start index in convex_hull_edge
        int edge_num;       // number of unique edges
    } mesh_convex[MAX_MESHES];
    int    convex_total_verts  = 0;
    int    convex_total_faces  = 0;
    int    convex_total_edges  = 0;
    float* convex_hull_vert         = nullptr;  // [convex_total_verts * 3] heap
    int*   convex_hull_face         = nullptr;  // [convex_total_faces * 3] heap (triangle indices)
    float* convex_hull_face_normal  = nullptr;  // [convex_total_faces * 3] heap
    int*   convex_hull_edge         = nullptr;  // [convex_total_edges * 2] heap (unique edge pairs)

    // Per-geom texture data (extracted from MuJoCo at bundle build time)
    static constexpr int MAX_TEXTURES = 32;
    int ntexture;
    struct TextureInfo {
        int width, height;
        int data_offset;           // offset into texture_data[]
        // 1 = GL_TEXTURE_CUBE_MAP (MuJoCo type="cube"), 0 = GL_TEXTURE_2D.
        // Cube textures store 6 square faces of width x width back to back at
        // data_offset, in GL order +X, -X, +Y, -Y, +Z, -Z. Height is the total
        // stored height (6 * width) so the packing arithmetic stays uniform.
        int is_cube;
        char name[MAX_NAME_LEN];
    } textures[MAX_TEXTURES];
    int texture_data_size;         // total bytes of RGB texture data
    unsigned char* texture_data;   // heap-allocated RGB pixel data (all textures packed)
    int geom_texid[MAX_GEOMS];     // per-geom texture index, -1 if no texture
    float geom_texrepeat[MAX_GEOMS * 2]; // per-geom texture repeat (S, T), default [1,1]

    // Per-geom material properties (resolved from MJCF <material> or defaults)
    float geom_specular[MAX_GEOMS];    // MuJoCo specular coefficient [0,1]
    float geom_shininess[MAX_GEOMS];   // MuJoCo shininess [0,1] (GL = this * 128)
    float geom_emission[MAX_GEOMS];    // MuJoCo emission [0,1]
    float geom_reflectance[MAX_GEOMS]; // MuJoCo mat_reflectance [0,1]. NOT applied
                                       // directly: only the geoms MuJoCo deems
                                       // reflective get it (see is_reflective).

    float stat_extent;  // bounding sphere radius of all geoms (like MuJoCo stat.extent)
    float stat_center[3]; // bounding sphere center

    float map_znear;  // near-plane multiplier from <visual><map znear="..."/>, MuJoCo default 0.01
    float map_zfar;   // far-plane multiplier from <visual><map zfar="..."/>, MuJoCo default 50

    float impratio;     // impedance ratio (from <option impratio="...">)
    int   cone_type;    // 0 = pyramidal (box), 1 = elliptic (MuJoCo default for condim>=3)
    int   friction_combine; // 0 = max (MuJoCo), 1 = mean (PhysX); MJCF <custom><numeric
                            // name="instinct_sim_friction_combine" data="1"/></custom>
    float max_depen_vel;    // contact depenetration speed cap (m/s), 0 = off; <custom><numeric
                            // name="instinct_sim_max_depenetration_velocity" data="..."/>
    float opt_density;  // fluid medium density (kg/m³), 0 = no fluid
    float opt_viscosity;// fluid kinematic viscosity (m²/s)
    float o_margin;     // MuJoCo global contact margin (from <option o_margin="...">), default 0.0

    // MuJoCo-precomputed inverse effective inertia (from mj_setConst / CRBA at qpos0)
    float body_invweight0[MAX_BODIES * 2];  // [nbody * 2]: translational, rotational
    float dof_invweight0[MAX_DOF];          // [nv]: per-DOF inverse effective inertia
    double body_invweight0_d[MAX_BODIES * 2];
    double dof_invweight0_d[MAX_DOF];
    bool  has_invweight0;                   // true if invweight0 arrays are populated
};

// ── Solver options ──────────────────────────────────────────────────────────

struct SolverOptions {
    SolverType  solver;
    int         iterations;
    double      tolerance;
    float       contact_slip;
    float       contact_margin;
    float       impratio;       // impedance ratio for friction vs normal (MuJoCo default: 1)
    bool        enable_contact;
    bool        enable_limits;
    bool        enable_gravity;
    bool        enable_damping;
    bool        zero_bias;
    int         max_contacts_per_body_pair;  // 0 = unlimited
    float       free_joint_damping;          // per-step velocity retain factor for free joints (1.0 = no damping)
};

inline SolverOptions default_solver_options() {
    SolverOptions opt;
    opt.solver         = SOLVER_NEWTON;
    opt.iterations     = 100;
    opt.tolerance      = 1e-8;
    opt.contact_slip   = 1e-4f;
    opt.contact_margin = 0.0f;
    opt.impratio       = 1.0f;
    opt.enable_contact = true;
    opt.enable_limits  = true;
    opt.enable_gravity = true;
    opt.enable_damping = true;
    opt.zero_bias      = false;
    opt.max_contacts_per_body_pair = 0;
    opt.free_joint_damping = 1.0f;  // 1.0 = no damping; 0.99 = 1%/step
    return opt;
}

} // namespace physics
} // namespace hipgl
