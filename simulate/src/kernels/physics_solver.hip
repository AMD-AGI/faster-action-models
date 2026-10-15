// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Constraint solver kernels (v2: Newton + parallel)
//
// Changes from v1:
//   - Parallel constraint setup: 1 thread per joint/equality (not per env)
//   - Parallel M_inv_diag build: 1 thread per joint per env
//   - Parallel mass matrix build: 1 thread per (i,j) entry per env
//   - Newton solver: direct Cholesky in joint space (nv x nv)
//   - PGS kept as fallback for SOLVER_PGS
//   - All workspace buffers pre-allocated (no per-step hipMalloc/hipFree)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/world_mask.h"
#include "hipgl/physics/physics_contact_cone.h"
#include "hipgl/physics/physics_integrate_dev.h"

namespace hipgl {
namespace physics {
namespace kernels {

// world skip: per-world freeze mask (see world_mask.h).
HIPGL_WORLD_MASK_DEFINE(set_world_mask_solver)

// ── Jacobian helpers ────────────────────────────────────────────────────────

HIPGL_DEVICE inline void local_to_world_axis(
    const float* jnt_axis, int ji,
    const float* xquat, int env, int nbody, int jbody,
    float* ax)
{
    float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
    int bq4 = env * nbody * 4 + jbody * 4;
    float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
    float tw_x = 2.0f*(qy*laz - qz*lay);
    float tw_y = 2.0f*(qz*lax - qx*laz);
    float tw_z = 2.0f*(qx*lay - qy*lax);
    ax[0] = lax + qw*tw_x + (qy*tw_z - qz*tw_y);
    ax[1] = lay + qw*tw_y + (qz*tw_x - qx*tw_z);
    ax[2] = laz + qw*tw_z + (qx*tw_y - qy*tw_x);
}

HIPGL_DEVICE inline void local_to_world_axis_d(
    const double* jnt_axis_d, int ji,
    const double* xquat_d, int env, int nbody, int jbody,
    double* ax)
{
    double lax = jnt_axis_d[ji*3], lay = jnt_axis_d[ji*3+1], laz = jnt_axis_d[ji*3+2];
    int bq4 = env * nbody * 4 + jbody * 4;
    double qw = xquat_d[bq4], qx = xquat_d[bq4+1], qy = xquat_d[bq4+2], qz = xquat_d[bq4+3];
    double tw_x = 2.0*(qy*laz - qz*lay);
    double tw_y = 2.0*(qz*lax - qx*laz);
    double tw_z = 2.0*(qx*lay - qy*lax);
    ax[0] = lax + qw*tw_x + (qy*tw_z - qz*tw_y);
    ax[1] = lay + qw*tw_y + (qz*tw_x - qx*tw_z);
    ax[2] = laz + qw*tw_z + (qx*tw_y - qy*tw_x);
}

HIPGL_DEVICE inline void add_joint_jacobian(
    double* J_row, int ji, float sign,
    const int* jnt_type, const int* jnt_dof_adr, const float* jnt_axis,
    const ContactPair& cp, const float* xpos, const float* xquat,
    int env, int nbody, int jbody)
{
    int jtype = jnt_type[ji];
    int dof = jnt_dof_adr[ji];

    if (jtype == JOINT_HINGE) {
        float ax[3];
        local_to_world_axis(jnt_axis, ji, xquat, env, nbody, jbody, ax);
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {cp.pos[0] - (double)xpos[bidx3], cp.pos[1] - (double)xpos[bidx3+1], cp.pos[2] - (double)xpos[bidx3+2]};
        double torque[3] = {
            r[1]*cp.normal[2] - r[2]*cp.normal[1],
            r[2]*cp.normal[0] - r[0]*cp.normal[2],
            r[0]*cp.normal[1] - r[1]*cp.normal[0]
        };
        J_row[dof] += (double)sign * ((double)ax[0]*torque[0] + (double)ax[1]*torque[1] + (double)ax[2]*torque[2]);
    }
    else if (jtype == JOINT_SLIDE) {
        float ax[3];
        local_to_world_axis(jnt_axis, ji, xquat, env, nbody, jbody, ax);
        J_row[dof] += (double)sign * ((double)ax[0]*cp.normal[0] + (double)ax[1]*cp.normal[1] + (double)ax[2]*cp.normal[2]);
    }
    else if (jtype == JOINT_FREE) {
        J_row[dof]   += (double)sign * cp.normal[0];
        J_row[dof+1] += (double)sign * cp.normal[1];
        J_row[dof+2] += (double)sign * cp.normal[2];
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {cp.pos[0] - (double)xpos[bidx3], cp.pos[1] - (double)xpos[bidx3+1], cp.pos[2] - (double)xpos[bidx3+2]};
        // World-frame torque = r × normal
        double torque[3] = {r[1]*cp.normal[2] - r[2]*cp.normal[1],
                            r[2]*cp.normal[0] - r[0]*cp.normal[2],
                            r[0]*cp.normal[1] - r[1]*cp.normal[0]};
        // MuJoCo free joint rotational DOFs use body-frame axes
        int bq4 = env * nbody * 4 + jbody * 4;
        float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
        // Body axes (columns of rotation matrix from quaternion)
        double bx[3] = {1-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)};
        double by[3] = {2.0*(qx*qy-qw*qz), 1-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)};
        double bz[3] = {2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1-2.0*(qx*qx+qy*qy)};
        J_row[dof+3] += (double)sign * (bx[0]*torque[0] + bx[1]*torque[1] + bx[2]*torque[2]);
        J_row[dof+4] += (double)sign * (by[0]*torque[0] + by[1]*torque[1] + by[2]*torque[2]);
        J_row[dof+5] += (double)sign * (bz[0]*torque[0] + bz[1]*torque[1] + bz[2]*torque[2]);
    }
}

HIPGL_DEVICE inline void add_joint_jacobian_dir(
    double* J_row, int ji, float sign,
    const int* jnt_type, const int* jnt_dof_adr, const float* jnt_axis,
    const float* dir, const float* contact_pos,
    const float* xpos, const float* xquat,
    int env, int nbody, int jbody)
{
    int jtype = jnt_type[ji];
    int dof = jnt_dof_adr[ji];

    if (jtype == JOINT_HINGE) {
        float ax[3];
        local_to_world_axis(jnt_axis, ji, xquat, env, nbody, jbody, ax);
        int bidx3 = env * nbody * 3 + jbody * 3;
        float r[3] = {contact_pos[0] - xpos[bidx3], contact_pos[1] - xpos[bidx3+1], contact_pos[2] - xpos[bidx3+2]};
        float torque[3] = {r[1]*dir[2] - r[2]*dir[1], r[2]*dir[0] - r[0]*dir[2], r[0]*dir[1] - r[1]*dir[0]};
        J_row[dof] += sign * (ax[0]*torque[0] + ax[1]*torque[1] + ax[2]*torque[2]);
    }
    else if (jtype == JOINT_SLIDE) {
        float ax[3];
        local_to_world_axis(jnt_axis, ji, xquat, env, nbody, jbody, ax);
        J_row[dof] += sign * (ax[0]*dir[0] + ax[1]*dir[1] + ax[2]*dir[2]);
    }
    else if (jtype == JOINT_FREE) {
        J_row[dof]   += sign * dir[0];
        J_row[dof+1] += sign * dir[1];
        J_row[dof+2] += sign * dir[2];
        int bidx3 = env * nbody * 3 + jbody * 3;
        float r[3] = {contact_pos[0] - xpos[bidx3], contact_pos[1] - xpos[bidx3+1], contact_pos[2] - xpos[bidx3+2]};
        // World-frame torque = r × dir
        float torque[3] = {r[1]*dir[2] - r[2]*dir[1],
                           r[2]*dir[0] - r[0]*dir[2],
                           r[0]*dir[1] - r[1]*dir[0]};
        // MuJoCo free joint rotational DOFs use body-frame axes
        int bq4 = env * nbody * 4 + jbody * 4;
        float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
        float bx[3] = {1-2.0f*(qy*qy+qz*qz), 2.0f*(qx*qy+qw*qz), 2.0f*(qx*qz-qw*qy)};
        float by[3] = {2.0f*(qx*qy-qw*qz), 1-2.0f*(qx*qx+qz*qz), 2.0f*(qy*qz+qw*qx)};
        float bz[3] = {2.0f*(qx*qz+qw*qy), 2.0f*(qy*qz-qw*qx), 1-2.0f*(qx*qx+qy*qy)};
        J_row[dof+3] += sign * (bx[0]*torque[0] + bx[1]*torque[1] + bx[2]*torque[2]);
        J_row[dof+4] += sign * (by[0]*torque[0] + by[1]*torque[1] + by[2]*torque[2]);
        J_row[dof+5] += sign * (bz[0]*torque[0] + bz[1]*torque[1] + bz[2]*torque[2]);
    }
}

HIPGL_DEVICE inline void fill_chain_jacobian_dir(
    double* J_row, int body, float sign,
    const float* dir, const float* contact_pos,
    const int* body_parent, const int* body_jnt_adr, const int* body_jnt_num,
    const int* jnt_type, const int* jnt_dof_adr, const float* jnt_axis,
    const float* xpos, const float* xquat, int env, int nbody)
{
    int cur = body;
    int max_walk = 32;
    while (cur > 0 && max_walk-- > 0) {
        int ja = body_jnt_adr[cur];
        int jn = body_jnt_num[cur];
        for (int k = 0; k < jn; k++) {
            add_joint_jacobian_dir(J_row, ja + k, sign,
                jnt_type, jnt_dof_adr, jnt_axis,
                dir, contact_pos, xpos, xquat, env, nbody, cur);
        }
        cur = body_parent[cur];
    }
}

HIPGL_DEVICE inline void compute_tangents(const float* n, float* t1, float* t2) {
    float ax = fabsf(n[0]), ay = fabsf(n[1]), az = fabsf(n[2]);
    float ref[3];
    if (ax <= ay && ax <= az) { ref[0] = 1; ref[1] = 0; ref[2] = 0; }
    else if (ay <= az)        { ref[0] = 0; ref[1] = 1; ref[2] = 0; }
    else                      { ref[0] = 0; ref[1] = 0; ref[2] = 1; }
    float d = ref[0]*n[0] + ref[1]*n[1] + ref[2]*n[2];
    t1[0] = ref[0] - d*n[0]; t1[1] = ref[1] - d*n[1]; t1[2] = ref[2] - d*n[2];
    float len = sqrtf(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    if (len > 1e-8f) { float inv = 1.0f/len; t1[0]*=inv; t1[1]*=inv; t1[2]*=inv; }
    t2[0] = n[1]*t1[2] - n[2]*t1[1];
    t2[1] = n[2]*t1[0] - n[0]*t1[2];
    t2[2] = n[0]*t1[1] - n[1]*t1[0];
}

// Double-precision overloads for float64 ContactPair
HIPGL_DEVICE inline void compute_tangents(const double* n, double* t1, double* t2) {
    double ax = fabs(n[0]), ay = fabs(n[1]), az = fabs(n[2]);
    double ref[3];
    if (ax <= ay && ax <= az) { ref[0] = 1; ref[1] = 0; ref[2] = 0; }
    else if (ay <= az)        { ref[0] = 0; ref[1] = 1; ref[2] = 0; }
    else                      { ref[0] = 0; ref[1] = 0; ref[2] = 1; }
    double d = ref[0]*n[0] + ref[1]*n[1] + ref[2]*n[2];
    t1[0] = ref[0] - d*n[0]; t1[1] = ref[1] - d*n[1]; t1[2] = ref[2] - d*n[2];
    double len = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
    if (len > 1e-15) { double inv = 1.0/len; t1[0]*=inv; t1[1]*=inv; t1[2]*=inv; }
    t2[0] = n[1]*t1[2] - n[2]*t1[1];
    t2[1] = n[2]*t1[0] - n[0]*t1[2];
    t2[2] = n[0]*t1[1] - n[1]*t1[0];
}

template <class JRow>
HIPGL_DEVICE inline void add_joint_jacobian_dir_d(
    JRow J_row, int ji, double sign,
    const int* jnt_type, const int* jnt_dof_adr, const double* jnt_axis_d,
    const double* dir, const double* contact_pos,
    const double* xpos_d, const double* xquat_d,
    int env, int nbody, int jbody, int parent_body,
    const double* jnt_xaxis_d = nullptr, int njnt = 0)
{
    int jtype = jnt_type[ji];
    int dof = jnt_dof_adr[ji];
    if (jtype == JOINT_HINGE) {
        double ax[3];
        if (jnt_xaxis_d && ji < njnt) {
            int ax_idx = env * njnt * 3 + ji * 3;
            ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
        } else {
            local_to_world_axis_d(jnt_axis_d, ji, xquat_d, env, nbody, parent_body, ax);
        }
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {contact_pos[0]-xpos_d[bidx3], contact_pos[1]-xpos_d[bidx3+1], contact_pos[2]-xpos_d[bidx3+2]};
        double torque[3] = {r[1]*dir[2]-r[2]*dir[1], r[2]*dir[0]-r[0]*dir[2], r[0]*dir[1]-r[1]*dir[0]};
        J_row[dof] += sign * (ax[0]*torque[0] + ax[1]*torque[1] + ax[2]*torque[2]);
    } else if (jtype == JOINT_SLIDE) {
        double ax[3];
        if (jnt_xaxis_d && ji < njnt) {
            int ax_idx = env * njnt * 3 + ji * 3;
            ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
        } else {
            local_to_world_axis_d(jnt_axis_d, ji, xquat_d, env, nbody, parent_body, ax);
        }
        J_row[dof] += sign * (ax[0]*dir[0] + ax[1]*dir[1] + ax[2]*dir[2]);
    } else if (jtype == JOINT_FREE) {
        J_row[dof]   += sign * dir[0];
        J_row[dof+1] += sign * dir[1];
        J_row[dof+2] += sign * dir[2];
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {contact_pos[0]-xpos_d[bidx3], contact_pos[1]-xpos_d[bidx3+1], contact_pos[2]-xpos_d[bidx3+2]};
        // World-frame torque = r × dir
        double torque[3] = {r[1]*dir[2] - r[2]*dir[1],
                            r[2]*dir[0] - r[0]*dir[2],
                            r[0]*dir[1] - r[1]*dir[0]};
        // MuJoCo free joint rotational DOFs use body-frame axes
        int bq4 = env * nbody * 4 + jbody * 4;
        double qw = xquat_d[bq4], qx = xquat_d[bq4+1], qy = xquat_d[bq4+2], qz = xquat_d[bq4+3];
        double bx[3] = {1-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)};
        double by[3] = {2.0*(qx*qy-qw*qz), 1-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)};
        double bz[3] = {2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1-2.0*(qx*qx+qy*qy)};
        J_row[dof+3] += sign * (bx[0]*torque[0] + bx[1]*torque[1] + bx[2]*torque[2]);
        J_row[dof+4] += sign * (by[0]*torque[0] + by[1]*torque[1] + by[2]*torque[2]);
        J_row[dof+5] += sign * (bz[0]*torque[0] + bz[1]*torque[1] + bz[2]*torque[2]);
    }
}

HIPGL_DEVICE inline void fill_chain_jacobian_dir_d(
    double* J_row, int body, double sign,
    const double* dir, const double* contact_pos,
    const int* body_parent, const int* body_jnt_adr, const int* body_jnt_num,
    const int* jnt_type, const int* jnt_dof_adr, const double* jnt_axis_d,
    const double* xpos_d, const double* xquat_d, int env, int nbody,
    const double* jnt_xaxis_d = nullptr, int njnt = 0)
{
    int cur = body;
    int max_walk = 32;
    while (cur > 0 && max_walk-- > 0) {
        int ja = body_jnt_adr[cur];
        int jn = body_jnt_num[cur];
        int par = body_parent[cur];
        for (int k = 0; k < jn; k++)
            add_joint_jacobian_dir_d(J_row, ja+k, sign, jnt_type, jnt_dof_adr, jnt_axis_d,
                dir, contact_pos, xpos_d, xquat_d, env, nbody, cur, par, jnt_xaxis_d, njnt);
        cur = par;
    }
}

// Precomputed-chain version: uses body_chain arrays to avoid parent pointer chasing.
// body_chain_start[body], body_chain_len[body] index into the flat arrays
// body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent.
template <class JRow>
HIPGL_DEVICE inline void fill_chain_jacobian_precomputed_d(
    JRow J_row, int body, double sign,
    const double* dir, const double* contact_pos,
    const int* body_chain_start, const int* body_chain_len,
    const int* body_chain_jnt_idx, const int* body_chain_jnt_body,
    const int* body_chain_parent,
    const int* jnt_type, const int* jnt_dof_adr, const double* jnt_axis_d,
    const double* xpos_d, const double* xquat_d, int env, int nbody,
    const double* jnt_xaxis_d, int njnt)
{
    int start = body_chain_start[body];
    int len   = body_chain_len[body];
    for (int i = 0; i < len; i++) {
        int ji    = body_chain_jnt_idx[start + i];
        int jbody = body_chain_jnt_body[start + i];
        int par   = body_chain_parent[start + i];
        add_joint_jacobian_dir_d(J_row, ji, sign, jnt_type, jnt_dof_adr, jnt_axis_d,
            dir, contact_pos, xpos_d, xquat_d, env, nbody, jbody, par, jnt_xaxis_d, njnt);
    }
}

// Angular relative-velocity Jacobian for torsional friction. Unlike a sliding
// contact row, this has no translational component or contact-point moment arm.
template <class JRow>
HIPGL_DEVICE inline void fill_chain_angular_jacobian_d(
    JRow J, int body, double sign, const double* dir,
    const int* chain_start, const int* chain_len, const int* chain_jnt,
    const int* chain_body, const int* chain_parent,
    const int* jnt_type, const int* jnt_dof_adr, const double* jnt_axis,
    const double* xquat, int env, int nbody, const double* jnt_xaxis, int njnt)
{
    const int start = chain_start[body];
    for (int k = 0; k < chain_len[body]; ++k) {
        const int ji = chain_jnt[start+k];
        const int da = jnt_dof_adr[ji];
        if (jnt_type[ji] == JOINT_HINGE) {
            double axis[3];
            if (jnt_xaxis) {
                for (int a = 0; a < 3; ++a) axis[a] = jnt_xaxis[(env*njnt+ji)*3+a];
            } else {
                local_to_world_axis_d(jnt_axis, ji, xquat, env, nbody, chain_parent[start+k], axis);
            }
            J[da] += sign*(axis[0]*dir[0] + axis[1]*dir[1] + axis[2]*dir[2]);
        } else if (jnt_type[ji] == JOINT_FREE || jnt_type[ji] == JOINT_BALL) {
            const double* q = xquat + (env*nbody+chain_body[start+k])*4;
            const double w=q[0], x=q[1], y=q[2], z=q[3];
            const double R[9] = {
                1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
            const int rot = da + (jnt_type[ji] == JOINT_FREE ? 3 : 0);
            for (int a = 0; a < 3; ++a)
                J[rot+a] += sign*(R[a]*dir[0] + R[3+a]*dir[1] + R[6+a]*dir[2]);
        }
    }
}

HIPGL_DEVICE inline bool contact_has_torsion(const ContactPair& cp, const int* condim) {
    return condim && (condim[cp.geom1] >= 4 || condim[cp.geom2] >= 4);
}

// Load a condim=4 contact whose torsional row follows the ordinary contact rows.
HIPGL_DEVICE inline double evaluate_contact4(
    const int* rows, const double* residual, const double* direction, double alpha,
    const double* D, const double* friction, double impr_inv,
    double* force, int* state, double* hessian = nullptr)
{
    double x[4], d[4], f[3];
    for (int i = 0; i < 4; ++i) {
        x[i] = residual[rows[i]] + (direction ? alpha*direction[rows[i]] : 0.0);
        d[i] = D[rows[i]];
        if (i) f[i-1] = friction[rows[i]];
    }
    return elliptic_contact4(x, d, f, impr_inv, force, state, hessian);
}

// ── MuJoCo-compatible constraint impedance (Section 5.2 of spec) ────────────
//
// The impedance D and reference acceleration aref are computed from:
//   solref = [timeconst, dampratio]
//   solimp = [dmin, dmax, width, mid, power]
//   invweight = effective inverse inertia in constraint space
//   pos = constraint position error (penetration depth, limit violation, etc.)
//   Jv = J @ qvel (constraint-space velocity)
//
// Formulation:
//   k = 1 / (dmax^2 * tc^2 * dr^2)       [stiffness]
//   b = 2 / (dmax * tc)                   [damping]
//   imp = sigmoid(|pos|, solimp)           [position-dependent impedance]
//   r = invweight * (1 - imp) / imp        [regularized effective inertia]
//   D = 1 / r                              [constraint impedance]
//   aref = k * imp * pos_hip - b * Jv      [reference acceleration]
//
// Newton solve: H = M + J^T * diag(D) * J, rhs = f_smooth + J^T * diag(D) * aref
// force_c = D_c * (aref_c - J_c * qacc)

// imp_depth: depth used for impedance (D) calculation -- MuJoCo uses the
//   contact normal depth for ALL rows of the same contact (normal + friction).
// aref_depth: depth used for reference acceleration -- 0 for friction rows,
//   actual depth for normal/limit/equality rows.
HIPGL_DEVICE inline void mujoco_impedance(
    double solref0, double solref1,
    double solimp0, double solimp1, double solimp2, double solimp3, double solimp4,
    double imp_depth,
    double aref_depth,
    double invweight,
    double Jv,
    double timestep,
    double* out_D,
    double* out_aref,
    double max_depen_vel = 0.0)
{
    constexpr double MINIMP = 1e-4;
    constexpr double MAXIMP = 0.9999;
    constexpr double MINVAL = 1e-15;

    double dmin_imp = fmin(fmax(solimp0, MINIMP), MAXIMP);
    double dmax_imp = fmin(fmax(solimp1, MINIMP), MAXIMP);
    double width    = fmax(solimp2, MINVAL);
    double mid_val  = fmin(fmax(solimp3, MINIMP), MAXIMP);
    double pwr      = fmax(solimp4, 1.0);

    double k, b;
    if (solref0 > 0.0 && solref1 > 0.0) {
        double tc = fmax(solref0, 2.0 * timestep);
        double dr = solref1;
        k = 1.0 / (dmax_imp * dmax_imp * tc * tc * dr * dr);
        b = 2.0 / (dmax_imp * tc);
    } else {
        k = (solref0 <= 0.0) ? (-solref0 / (dmax_imp * dmax_imp)) : (1.0 / (dmax_imp * dmax_imp * solref0 * solref0));
        b = (solref1 <= 0.0) ? (-solref1 / dmax_imp) : (2.0 * solref1 / (dmax_imp * solref0));
    }

    double imp_x = fabs(imp_depth) / width;
    double imp;
    if (imp_x >= 1.0) {
        imp = dmax_imp;
    } else if (imp_x < mid_val) {
        double base = pow(mid_val, pwr - 1.0);
        double imp_a = (base > MINVAL) ? pow(imp_x, pwr) / base : 0.0;
        imp = dmin_imp + imp_a * (dmax_imp - dmin_imp);
    } else {
        double base = pow(1.0 - mid_val, pwr - 1.0);
        double imp_b = (base > MINVAL) ? 1.0 - pow(1.0 - imp_x, pwr) / base : 1.0;
        imp = dmin_imp + imp_b * (dmax_imp - dmin_imp);
    }
    imp = fmin(fmax(imp, dmin_imp), dmax_imp);

    double r = fmax(invweight * (1.0 - imp) / imp, MINVAL);
    constexpr double MAX_D = 1e10;
    *out_D = fmin(1.0 / r, MAX_D);

    // Optional PhysX-style depenetration speed limit (contact rows only): the position term
    // alone drives the rows toward Jv = k*imp*depth / b, so capping it at b*v caps the speed
    // at which a penetration is pushed out. 0 = off (MuJoCo).
    double pos_term = k * imp * aref_depth;
    if (max_depen_vel > 0.0) pos_term = fmin(pos_term, b * max_depen_vel);
    *out_aref = pos_term - b * Jv;
}

HIPGL_GLOBAL void kernel_constraint_setup(
    int num_envs, int nv, int nbody, int njnt,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const int*   __restrict__ jnt_dof_adr,
    const int*   __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis_d,
    const double* __restrict__ jnt_range_d,
    const int*   __restrict__ jnt_limited,
    const int*   __restrict__ body_parent,
    const int*   __restrict__ body_jnt_adr,
    const int*   __restrict__ body_jnt_num,
    const int*   __restrict__ geom_condim,
    const double* __restrict__ geom_friction_d,
    const double* __restrict__ geom_solref_d,
    const double* __restrict__ geom_solimp_d,
    const int*   __restrict__ geom_body,
    const double* __restrict__ body_mass_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ body_invweight0_d,
    double timestep,
    double impratio,
    int nequality,
    const int*   __restrict__ eq_type,
    const int*   __restrict__ eq_body1,
    const int*   __restrict__ eq_body2,
    const ContactPair* __restrict__ contacts,
    const int*   __restrict__ ncon,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    double contact_slip,
    bool enable_limits,
    const int*   __restrict__ body_chain_start,
    const int*   __restrict__ body_chain_len,
    const int*   __restrict__ body_chain_jnt_idx,
    const int*   __restrict__ body_chain_jnt_body,
    const int*   __restrict__ body_chain_parent,
    double*      __restrict__ constraint_J,
    double*      __restrict__ constraint_rhs,
    double*      __restrict__ constraint_D,
    double*      __restrict__ constraint_lo,
    double*      __restrict__ constraint_hi,
    int*         __restrict__ constraint_friction_parent,
    int*         __restrict__ constraint_friction_sibling,
    double*      __restrict__ constraint_friction_mu,
    int*         __restrict__ nconstraint,
    int cone_type,
    int pyr_edges,
    double max_depen_vel)
{
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (env >= num_envs) return;

    int nc = ncon[env];

    if (cid < nc) {
        const ContactPair& cp = contacts[env * MAX_CONTACTS + cid];

        const bool pyramidal = (cone_type == 0);
        const int slots_per_contact = pyramidal ? pyr_edges : 3;   // normal + 2 tangents (condim>=3)
        int normal_slot = cid * slots_per_contact;
        if (cid == 0) {
            int ntorsion = 0;
            if (!pyramidal)
                for (int c = 0; c < nc; ++c)
                    ntorsion += contact_has_torsion(contacts[env*MAX_CONTACTS+c], geom_condim);
            atomicAdd(&nconstraint[env], nc * slots_per_contact + ntorsion);
        }
        if (normal_slot + slots_per_contact - 1 >= MAX_CONSTRAINTS) return;

        size_t j_base = (size_t)env * MAX_CONSTRAINTS * nv + (size_t)normal_slot * nv;
        double* J_row = constraint_J + j_base;
        for (int v = 0; v < nv; v++) J_row[v] = 0.0;

        fill_chain_jacobian_precomputed_d(J_row, cp.body1, -1.0, cp.normal, cp.pos,
            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
            jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
        fill_chain_jacobian_precomputed_d(J_row, cp.body2, +1.0, cp.normal, cp.pos,
            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
            jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);

        // Combined solref/solimp: weighted average by geom_solmix (MuJoCo convention)
        // Default solmix = 1.0 for all geoms → simple average
        int g1 = cp.geom1, g2 = cp.geom2;
        double sr0 = 0.5 * (geom_solref_d[g1*2]   + geom_solref_d[g2*2]);
        double sr1 = 0.5 * (geom_solref_d[g1*2+1] + geom_solref_d[g2*2+1]);
        double si0 = 0.5 * (geom_solimp_d[g1*5]   + geom_solimp_d[g2*5]);
        double si1 = 0.5 * (geom_solimp_d[g1*5+1] + geom_solimp_d[g2*5+1]);
        double si2 = 0.5 * (geom_solimp_d[g1*5+2] + geom_solimp_d[g2*5+2]);
        double si3 = 0.5 * (geom_solimp_d[g1*5+3] + geom_solimp_d[g2*5+3]);
        double si4 = 0.5 * (geom_solimp_d[g1*5+4] + geom_solimp_d[g2*5+4]);

        const double* Minv_env = M_inv_diag + env * MAX_DOF;
        double invweight;
        if (body_invweight0_d) {
            invweight = body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2];
        } else {
            invweight = 0.0;
            for (int v = 0; v < nv; v++)
                invweight += J_row[v] * J_row[v] * Minv_env[v];
        }
        invweight = fmax(invweight, 1e-15);

        const double* ev = qvel_d + (size_t)env * nv;
        double Jv = 0;
        for (int v = 0; v < nv; v++) Jv += J_row[v] * ev[v];

        // MuJoCo-compatible impedance: D and aref from solref/solimp
        double D_normal, aref;
        mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                         cp.depth, cp.depth, invweight, Jv, timestep, &D_normal, &aref, max_depen_vel);

        size_t slot_idx = (size_t)env * MAX_CONSTRAINTS + normal_slot;
        constraint_rhs[slot_idx] = aref;
        constraint_D[slot_idx] = D_normal;
        constraint_lo[slot_idx] = 0.0;
        constraint_hi[slot_idx] = 1e20;
        constraint_friction_parent[slot_idx] = -1;
        constraint_friction_sibling[slot_idx] = -1;
        constraint_friction_mu[slot_idx] = 0.0;

        int condim = 3;
        if (geom_condim) {
            int cd1 = geom_condim[cp.geom1];
            int cd2 = geom_condim[cp.geom2];
            condim = (cd1 > cd2) ? cd1 : cd2;
        }

        if (condim < 3) {
            for (int extra = 1; extra < slots_per_contact; extra++) {
                size_t idx = (size_t)env * MAX_CONSTRAINTS + normal_slot + extra;
                double* Jz = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv
                                          + (size_t)(normal_slot + extra) * nv;
                for (int v = 0; v < nv; v++) Jz[v] = 0.0;
                constraint_rhs[idx] = 0.0;
                constraint_D[idx] = 0.0;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 0.0;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
            }
        }

        if (condim >= 3 && pyramidal) {
            // MuJoCo's pyramidal cone (engine_core_constraint.c, mj_instantiateContact /
            // mj_diagApprox / mj_makeImpedance): edges J_n +- mu*J_tk, each a unilateral
            // row at pos = dist with standard K/B; diagApprox = tran*(1 + mu^2); every
            // edge gets Rpy = 2*mu_reg^2*R with mu_reg = mu*sqrt(1/impratio).
            double t1[3], t2[3];
            compute_tangents(cp.normal, t1, t2);
            const double mu = cp.friction;
            double* J0 = J_row;                 // holds J_n
            double* J1 = J0 + nv;               // J_t1 scratch, then edges
            double* J2 = J1 + nv;
            double* J3 = J2 + nv;               // J_t2 scratch
            double* J4 = J3 + nv;               // torsion pair (pyr_edges == 6)
            double* J5 = J4 + nv;               // J_torsion scratch
            const bool torsion = pyr_edges == 6 && condim >= 4;
            const double mu_t = torsion
                ? fmax(fmax(geom_friction_d[g1*3+1], geom_friction_d[g2*3+1]), 1e-5) : 0.0;
            for (int v = 0; v < nv; v++) { J1[v] = 0.0; J3[v] = 0.0; }
            if (torsion) {
                for (int v = 0; v < nv; v++) J5[v] = 0.0;
                fill_chain_angular_jacobian_d(J5, cp.body1, -1.0, cp.normal,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                    body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                    env, nbody, jnt_xaxis_d, njnt);
                fill_chain_angular_jacobian_d(J5, cp.body2, +1.0, cp.normal,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                    body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                    env, nbody, jnt_xaxis_d, njnt);
            }
            fill_chain_jacobian_precomputed_d(J1, cp.body1, -1.0, t1, cp.pos,
                body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
            fill_chain_jacobian_precomputed_d(J1, cp.body2, +1.0, t1, cp.pos,
                body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
            fill_chain_jacobian_precomputed_d(J3, cp.body1, -1.0, t2, cp.pos,
                body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
            fill_chain_jacobian_precomputed_d(J3, cp.body2, +1.0, t2, cp.pos,
                body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
            for (int v = 0; v < nv; v++) {
                const double n = J0[v], a = J1[v], b = J3[v];
                if (torsion) {
                    const double c = J5[v];
                    J4[v] = n + mu_t * c;
                    J5[v] = n - mu_t * c;
                }
                J0[v] = n + mu * a;
                J1[v] = n - mu * a;
                J2[v] = n + mu * b;
                J3[v] = n - mu * b;
            }
            const double iw_py = 2.0 * (mu * mu / fmax(impratio, 1e-15)) * invweight * (1.0 + mu * mu);
            const int nedge = torsion ? 6 : 4;
            for (int k = nedge; k < pyr_edges; k++) {       // condim-3 contact in a stride-6 block
                double* Jz = J0 + (size_t)k * nv;
                for (int v = 0; v < nv; v++) Jz[v] = 0.0;
                size_t idx = (size_t)env * MAX_CONSTRAINTS + normal_slot + k;
                constraint_rhs[idx] = 0.0;
                constraint_D[idx] = 0.0;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 0.0;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
            }
            for (int k = 0; k < nedge; k++) {
                const double* Jk = J0 + (size_t)k * nv;
                double Jvk = 0;
                for (int v = 0; v < nv; v++) Jvk += Jk[v] * ev[v];
                double Dk, arefk;
                mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                 cp.depth, cp.depth, iw_py, Jvk, timestep, &Dk, &arefk, max_depen_vel);
                size_t idx = (size_t)env * MAX_CONSTRAINTS + normal_slot + k;
                constraint_rhs[idx] = arefk;
                constraint_D[idx] = Dk;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 1e20;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
            }
        }

        if (condim >= 3 && !pyramidal) {
            double t1[3], t2[3];
            compute_tangents(cp.normal, t1, t2);
            double mu = cp.friction;

            // Fixed offsets within this contact's reserved block (see above).
            int tan1_slot = normal_slot + 1;
            int tan2_slot = normal_slot + 2;

            if (tan1_slot < MAX_CONSTRAINTS) {
                double* Jt1 = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)tan1_slot * nv;
                for (int v = 0; v < nv; v++) Jt1[v] = 0.0;
                fill_chain_jacobian_precomputed_d(Jt1, cp.body1, -1.0, t1, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
                fill_chain_jacobian_precomputed_d(Jt1, cp.body2, +1.0, t1, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);

                double Jv1 = 0;
                for (int v = 0; v < nv; v++) Jv1 += Jt1[v] * ev[v];

                double invweight_fric_t1;
                if (body_invweight0_d) {
                    invweight_fric_t1 = (body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2])
                                        / fmax(impratio, 1e-6);
                } else {
                    invweight_fric_t1 = 0.0;
                    for (int v = 0; v < nv; v++)
                        invweight_fric_t1 += Jt1[v] * Jt1[v] * Minv_env[v];
                    invweight_fric_t1 /= fmax(impratio, 1e-6);
                }
                invweight_fric_t1 = fmax(invweight_fric_t1, 1e-15);
                double D_fric_t1, aref_fric_t1;
                mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                 cp.depth, 0.0, invweight_fric_t1, Jv1, timestep, &D_fric_t1, &aref_fric_t1);

                size_t t1_idx = (size_t)env * MAX_CONSTRAINTS + tan1_slot;
                constraint_rhs[t1_idx] = aref_fric_t1;
                constraint_D[t1_idx] = D_fric_t1;
                constraint_lo[t1_idx] = -(double)mu;
                constraint_hi[t1_idx] = (double)mu;
                constraint_friction_parent[t1_idx] = normal_slot;
                constraint_friction_sibling[t1_idx] = tan2_slot;
                constraint_friction_mu[t1_idx] = mu;
            }

            if (tan2_slot < MAX_CONSTRAINTS) {
                double* Jt2 = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)tan2_slot * nv;
                for (int v = 0; v < nv; v++) Jt2[v] = 0.0;
                fill_chain_jacobian_precomputed_d(Jt2, cp.body1, -1.0, t2, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
                fill_chain_jacobian_precomputed_d(Jt2, cp.body2, +1.0, t2, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);

                double Jv2 = 0;
                for (int v = 0; v < nv; v++) Jv2 += Jt2[v] * ev[v];

                double invweight_fric_t2;
                if (body_invweight0_d) {
                    invweight_fric_t2 = (body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2])
                                        / fmax(impratio, 1e-6);
                } else {
                    invweight_fric_t2 = 0.0;
                    for (int v = 0; v < nv; v++)
                        invweight_fric_t2 += Jt2[v] * Jt2[v] * Minv_env[v];
                    invweight_fric_t2 /= fmax(impratio, 1e-6);
                }
                invweight_fric_t2 = fmax(invweight_fric_t2, 1e-15);
                double D_fric_t2, aref_fric_t2;
                mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                 cp.depth, 0.0, invweight_fric_t2, Jv2, timestep, &D_fric_t2, &aref_fric_t2);

                size_t t2_idx = (size_t)env * MAX_CONSTRAINTS + tan2_slot;
                constraint_rhs[t2_idx] = aref_fric_t2;
                constraint_D[t2_idx] = D_fric_t2;
                constraint_lo[t2_idx] = -(double)mu;
                constraint_hi[t2_idx] = (double)mu;
                constraint_friction_parent[t2_idx] = normal_slot;
                constraint_friction_sibling[t2_idx] = tan1_slot;
                constraint_friction_mu[t2_idx] = mu;
            }

            if (condim >= 4) {
                // Append only actual torsional rows, keeping all existing condim=3
                // row indices and summation order unchanged. Counting earlier
                // torsional contacts makes this independent of thread scheduling.
                int torsion_slot = nc * slots_per_contact;
                for (int c = 0; c < cid; ++c)
                    torsion_slot += contact_has_torsion(contacts[env*MAX_CONTACTS+c], geom_condim);
                if (torsion_slot >= MAX_CONSTRAINTS) return;
                double* Jt = constraint_J + (size_t)env*MAX_CONSTRAINTS*nv + (size_t)torsion_slot*nv;
                for (int v = 0; v < nv; ++v) Jt[v] = 0.0;
                fill_chain_angular_jacobian_d(Jt, cp.body1, -1.0, cp.normal,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                    body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                    env, nbody, jnt_xaxis_d, njnt);
                fill_chain_angular_jacobian_d(Jt, cp.body2, +1.0, cp.normal,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                    body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                    env, nbody, jnt_xaxis_d, njnt);
                double Jv_t = 0.0;
                for (int v = 0; v < nv; ++v) Jv_t += Jt[v]*ev[v];
                const double mu_t = fmax(fmax(geom_friction_d[g1*3+1], geom_friction_d[g2*3+1]), 1e-5);
                const double ratio = mu_t / fmax(mu, 1e-5);
                double unused_D, aref_t;
                mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                    cp.depth, 0.0, invweight, Jv_t, timestep, &unused_D, &aref_t);
                const size_t ti = (size_t)env*MAX_CONSTRAINTS + torsion_slot;
                constraint_D[ti] = constraint_D[(size_t)env*MAX_CONSTRAINTS+tan1_slot] * ratio*ratio;
                constraint_rhs[ti] = aref_t;
                constraint_lo[ti] = -mu_t;
                constraint_hi[ti] = mu_t;
                constraint_friction_parent[ti] = normal_slot;
                // Only the first sliding row owns evaluation of the whole cone.
                constraint_friction_sibling[ti] = -1;
                constraint_friction_mu[ti] = mu_t;
                constraint_friction_sibling[slot_idx] = torsion_slot;
            }
        }
    }
}


// ── Row-parallel contact constraint setup (bit-exact with kernel_constraint_setup
//    when body_invweight0 is available) ────────────────────────────────────────
// One thread per constraint ROW instead of per contact: row r = 3*cid + kind
// (kind 0 = normal, 1 = tangent t1, 2 = tangent t2), exactly the deterministic
// slot the per-contact kernel assigns.  Each row is built in a thread-private
// LDS column (stride CS_BLK, bank-conflict free) with the SAME helper calls and
// += order as before, so every J entry, Jv, D and aref is bit-identical; the
// finished rows of a block are contiguous in constraint_J and are written out
// with one coalesced copy instead of per-thread global read-modify-writes.
// Torsion rows (condim >= 4) stay on the per-contact path inside the kind-1
// thread (it owns D_fric_t1, which the torsional D is scaled from).
constexpr int CS_BLK = 64;
constexpr int CS_GRID_X = 8;
// Column-major J mirror geometry (see kernel_transpose_J below).
constexpr int JT_STRIDE  = 512;
constexpr int JT_ROW_OFF = MAX_CONSTRAINTS - JT_STRIDE;
// Fusion flags for kernel_constraint_setup_rows.
constexpr int CS_FUSE_JT = 1;   // also build the JT mirror (replaces kernel_transpose_J)
constexpr int CS_FUSE_JL = 2;   // extra block x==CS_GRID_X builds the joint rows
constexpr int CS_FUSE_LDS = 4;  // stage chain tables + per-env xpos/xquat/xaxis in LDS
constexpr int CS_LDS_NB = 64, CS_LDS_NJ = 64, CS_LDS_CH = 256;
                                // (replaces kernel_joint_limit_constraints)

// Joint friction-loss / limit rows, built by ONE wave per env (njnt <= CS_BLK).
// Bit-identical to kernel_joint_limit_constraints: same per-row arithmetic, and
// the slots reproduce its atomicAdd order for a single wave (all friction rows
// in lane order, then all lower-limit rows, then all upper-limit rows), placed
// after the 3*nc + ntorsion contact rows exactly as the old kernel's atomics
// (issued after the contact kernel had finished) placed them.
HIPGL_DEVICE __attribute__((always_inline)) inline void cs_joint_rows_wave(
    int env, int nv, int nq, int njnt, int nc, bool jt_ok,
    const ContactPair* __restrict__ env_cp,
    const int*    __restrict__ geom_condim,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_range_d,
    const int*    __restrict__ jnt_limited,
    const double* __restrict__ jnt_frictionloss_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ dof_invweight0_d,
    double timestep,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    double*       __restrict__ constraint_J,
    double*       __restrict__ constraint_rhs,
    double*       __restrict__ constraint_D,
    double*       __restrict__ constraint_lo,
    double*       __restrict__ constraint_hi,
    int*          __restrict__ constraint_friction_parent,
    double*       __restrict__ constraint_friction_mu,
    int*          __restrict__ nconstraint,
    int base_rows = -1,
    int*          __restrict__ constraint_friction_sibling = nullptr)
{
    const int tid = threadIdx.x;
    int base = base_rows;
    if (base < 0) {
        int ntor = 0;
        if (geom_condim)
            for (int c = tid; c < nc; c += CS_BLK) ntor += contact_has_torsion(env_cp[c], geom_condim) ? 1 : 0;
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) ntor += __shfl_xor(ntor, off);
        base = nc * 3 + ntor;
    }

    const int ji = tid;
    bool jact = (ji < njnt);
    int jtype = 0, qa = 0, da = 0;
    if (jact) { jtype = jnt_type[ji]; jact = (jtype == JOINT_HINGE || jtype == JOINT_SLIDE); }
    if (jact) { qa = jnt_qpos_adr[ji]; da = jnt_dof_adr[ji]; }
    const double* eq = qpos_d + env * nq;
    const double* ev = qvel_d + env * nv;
    const double* Minv = M_inv_diag + env * MAX_DOF;

    double sr0 = 0.02, sr1 = 1.0;
    double si0 = 0.9, si1 = 0.95, si2 = 0.001, si3 = 0.5, si4 = 2.0;
    double invweight = 0.0;
    if (jact) invweight = (dof_invweight0_d) ? dof_invweight0_d[da] : fmax(Minv[da], 1e-15);

    double fl = 0.0;
    bool f_on = false;
    if (jact && jnt_frictionloss_d) { fl = jnt_frictionloss_d[ji]; f_on = (fl > 0.0); }

    bool lo_on = false, hi_on = false;
    double dist_lo = 0.0, dist_hi = 0.0;
    const double margin = 0.0;
    if (jact && jnt_limited[ji]) {
        double q = eq[qa];
        double lo = jnt_range_d[ji * 2];
        double hi = jnt_range_d[ji * 2 + 1];
        if (!(lo >= hi - 1e-6)) {
            dist_lo = q - lo;
            dist_hi = hi - q;
            lo_on = (dist_lo - margin < 0.0);
            hi_on = (dist_hi - margin < 0.0);
        }
    }
    const unsigned long long mf = __ballot(f_on);
    const unsigned long long ml = __ballot(lo_on);
    const unsigned long long mh = __ballot(hi_on);
    const unsigned long long lt = (tid == 0) ? 0ull : (~0ull >> (64 - tid));
    const int nf = __popcll(mf), nl = __popcll(ml), nh = __popcll(mh);
    // base_rows >= 0 (pyramidal kernel): this wave owns the env's row count and
    // STORES the final total (the buffers are not pre-cleared on that path).
    if (base_rows >= 0) { if (tid == 0) nconstraint[env] = base + nf + nl + nh; }
    else if (tid == 0 && (nf + nl + nh)) atomicAdd(&nconstraint[env], nf + nl + nh);

    double* Jenv = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv;
    double* JT = Jenv + (size_t)JT_ROW_OFF * nv;
    auto emit = [&](int slot, double sgn, double depth_pos, double depth_vel, double Jv,
                    double lo_v, double hi_v) {
        if (slot >= MAX_CONSTRAINTS) return;
        double* J_row = Jenv + (size_t)slot * nv;
        for (int v = 0; v < nv; v++) J_row[v] = 0.0;
        J_row[da] = sgn;
        if (jt_ok && slot < JT_STRIDE)
            for (int v = 0; v < nv; v++) JT[(size_t)v * JT_STRIDE + slot] = (v == da) ? sgn : 0.0;
        double Dv, aref;
        mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                         depth_pos, depth_vel, invweight, Jv, timestep, &Dv, &aref);
        size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
        constraint_rhs[idx] = aref;
        constraint_D[idx] = Dv;
        constraint_lo[idx] = lo_v;
        constraint_hi[idx] = hi_v;
        constraint_friction_parent[idx] = -1;
        if (constraint_friction_sibling) constraint_friction_sibling[idx] = -1;
        constraint_friction_mu[idx] = 0.0;
    };
    if (f_on) {
        double Jv_fric = ev[da];
        emit(base + __popcll(mf & lt), 1.0, 0.0, 0.0, Jv_fric, -fl, fl);
    }
    if (lo_on) {
        double depth = -(dist_lo - margin);
        double Jv = ev[da];
        emit(base + nf + __popcll(ml & lt), 1.0, depth, depth, Jv, 0.0, 1e20);
    }
    if (hi_on) {
        double depth = -(dist_hi - margin);
        double Jv = -ev[da];
        emit(base + nf + nl + __popcll(mh & lt), -1.0, depth, depth, Jv, 0.0, 1e20);
    }
}

template <typename T>
struct CsView {
    const T* p; int base;
    HIPGL_DEVICE T operator[](int i) const { return p[i - base]; }
};

template <class JRow>
HIPGL_DEVICE inline void cs_add_joint_jac_lds(
    JRow J_row, int ji, double sign,
    CsView<int> jnt_type, CsView<int> jnt_dof_adr, const double* jnt_axis_d,
    const double* dir, const double* contact_pos,
    CsView<double> xpos_d, CsView<double> xquat_d, const double* xquat_g,
    int env, int nbody, int jbody, int parent_body,
    CsView<double> jnt_xaxis_d, int njnt)
{
    int jtype = jnt_type[ji];
    int dof = jnt_dof_adr[ji];
    if (jtype == JOINT_HINGE) {
        double ax[3];
        if (ji < njnt) {
            int ax_idx = env * njnt * 3 + ji * 3;
            ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
        } else {
            local_to_world_axis_d(jnt_axis_d, ji, xquat_g, env, nbody, parent_body, ax);
        }
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {contact_pos[0]-xpos_d[bidx3], contact_pos[1]-xpos_d[bidx3+1], contact_pos[2]-xpos_d[bidx3+2]};
        double torque[3] = {r[1]*dir[2]-r[2]*dir[1], r[2]*dir[0]-r[0]*dir[2], r[0]*dir[1]-r[1]*dir[0]};
        J_row[dof] += sign * (ax[0]*torque[0] + ax[1]*torque[1] + ax[2]*torque[2]);
    } else if (jtype == JOINT_SLIDE) {
        double ax[3];
        if (ji < njnt) {
            int ax_idx = env * njnt * 3 + ji * 3;
            ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
        } else {
            local_to_world_axis_d(jnt_axis_d, ji, xquat_g, env, nbody, parent_body, ax);
        }
        J_row[dof] += sign * (ax[0]*dir[0] + ax[1]*dir[1] + ax[2]*dir[2]);
    } else if (jtype == JOINT_FREE) {
        J_row[dof]   += sign * dir[0];
        J_row[dof+1] += sign * dir[1];
        J_row[dof+2] += sign * dir[2];
        int bidx3 = env * nbody * 3 + jbody * 3;
        double r[3] = {contact_pos[0]-xpos_d[bidx3], contact_pos[1]-xpos_d[bidx3+1], contact_pos[2]-xpos_d[bidx3+2]};
        double torque[3] = {r[1]*dir[2] - r[2]*dir[1],
                            r[2]*dir[0] - r[0]*dir[2],
                            r[0]*dir[1] - r[1]*dir[0]};
        int bq4 = env * nbody * 4 + jbody * 4;
        double qw = xquat_d[bq4], qx = xquat_d[bq4+1], qy = xquat_d[bq4+2], qz = xquat_d[bq4+3];
        double bx[3] = {1-2.0*(qy*qy+qz*qz), 2.0*(qx*qy+qw*qz), 2.0*(qx*qz-qw*qy)};
        double by[3] = {2.0*(qx*qy-qw*qz), 1-2.0*(qx*qx+qz*qz), 2.0*(qy*qz+qw*qx)};
        double bz[3] = {2.0*(qx*qz+qw*qy), 2.0*(qy*qz-qw*qx), 1-2.0*(qx*qx+qy*qy)};
        J_row[dof+3] += sign * (bx[0]*torque[0] + bx[1]*torque[1] + bx[2]*torque[2]);
        J_row[dof+4] += sign * (by[0]*torque[0] + by[1]*torque[1] + by[2]*torque[2]);
        J_row[dof+5] += sign * (bz[0]*torque[0] + bz[1]*torque[1] + bz[2]*torque[2]);
    }
}

template <class JRow>
HIPGL_DEVICE inline void cs_fill_chain_lds(
    JRow J_row, int body, double sign,
    const double* dir, const double* contact_pos,
    const int* s_cst, const int* s_cln, const int* s_cji, const int* s_cjb, const int* s_cpa,
    CsView<int> jnt_type, CsView<int> jnt_dof_adr, const double* jnt_axis_d,
    CsView<double> xpos_d, CsView<double> xquat_d, const double* xquat_g,
    int env, int nbody, CsView<double> jnt_xaxis_d, int njnt)
{
    int start = s_cst[body];
    int len   = s_cln[body];
    for (int i = 0; i < len; i++) {
        int ji    = s_cji[start + i];
        int jbody = s_cjb[start + i];
        int par   = s_cpa[start + i];
        cs_add_joint_jac_lds(J_row, ji, sign, jnt_type, jnt_dof_adr, jnt_axis_d,
            dir, contact_pos, xpos_d, xquat_d, xquat_g, env, nbody, jbody, par, jnt_xaxis_d, njnt);
    }
}

struct LdsJRow {
    double* p;
    HIPGL_DEVICE double& operator[](int i) const { return p[i * CS_BLK]; }
};

template <bool CS_LDS>
HIPGL_GLOBAL void __launch_bounds__(CS_BLK) kernel_constraint_setup_rows(
    int num_envs, int nv, int nbody, int njnt,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis_d,
    const int*   __restrict__ geom_condim,
    const double* __restrict__ geom_friction_d,
    const double* __restrict__ geom_solref_d,
    const double* __restrict__ geom_solimp_d,
    const double* __restrict__ body_invweight0_d,
    double timestep,
    double impratio,
    const ContactPair* __restrict__ contacts,
    const int*   __restrict__ ncon,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ qvel_d,
    const int*   __restrict__ body_chain_start,
    const int*   __restrict__ body_chain_len,
    const int*   __restrict__ body_chain_jnt_idx,
    const int*   __restrict__ body_chain_jnt_body,
    const int*   __restrict__ body_chain_parent,
    double*      __restrict__ constraint_J,
    double*      __restrict__ constraint_rhs,
    double*      __restrict__ constraint_D,
    double*      __restrict__ constraint_lo,
    double*      __restrict__ constraint_hi,
    int*         __restrict__ constraint_friction_parent,
    int*         __restrict__ constraint_friction_sibling,
    double*      __restrict__ constraint_friction_mu,
    int*         __restrict__ nconstraint,
    int fuse_flags, int nq,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_range_d,
    const int*    __restrict__ jnt_limited,
    const double* __restrict__ jnt_frictionloss_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ dof_invweight0_d,
    const double* __restrict__ qpos_d,
    int chain_total,
    double max_depen_vel)
{
    __shared__ double sJ[KERNEL_MAXNV * CS_BLK];
    constexpr int LNB = CS_LDS ? CS_LDS_NB : 1;
    constexpr int LNJ = CS_LDS ? CS_LDS_NJ : 1;
    constexpr int LCH = CS_LDS ? CS_LDS_CH : 1;
    __shared__ double s_cxp[LNB * 3];
    __shared__ double s_cxq[LNB * 4];
    __shared__ double s_cxa[LNJ * 3];
    __shared__ int s_cjt[LNJ], s_cjd[LNJ];
    __shared__ int s_cst[LNB], s_cln[LNB];
    __shared__ int s_cji[LCH], s_cjb[LCH], s_cpa[LCH];
    const int env = blockIdx.y;
    if (env >= num_envs) return;
    const int tid = threadIdx.x;
    const int nc = ncon[env];
    const int nrow = nc * 3;
    const ContactPair* env_cp = contacts + (size_t)env * MAX_CONTACTS;
    const double* ev = qvel_d + (size_t)env * nv;
    const LdsJRow Jl{&sJ[tid]};
    // Mirror rows [JT_ROW_OFF, MAX_CONSTRAINTS) can never overlap a live row when
    // 4*nc + 3*njnt <= JT_ROW_OFF (upper bound on the final row count).  Beyond
    // that bound the final count exceeds JT_STRIDE, so the solver ignores the
    // mirror and nothing is written.
    const bool jt_ok = (fuse_flags & CS_FUSE_JT) && (4 * nc + 3 * njnt <= JT_ROW_OFF);
    double* const JTenv = constraint_J + ((size_t)env * MAX_CONSTRAINTS + JT_ROW_OFF) * nv;

    if (blockIdx.x == CS_GRID_X) {
        if (fuse_flags & CS_FUSE_JL)
            cs_joint_rows_wave(env, nv, nq, njnt, nc, jt_ok, env_cp, geom_condim,
                jnt_type, jnt_qpos_adr, jnt_dof_adr, jnt_range_d, jnt_limited,
                jnt_frictionloss_d, M_inv_diag, dof_invweight0_d, timestep,
                qpos_d, qvel_d, constraint_J, constraint_rhs, constraint_D,
                constraint_lo, constraint_hi, constraint_friction_parent,
                constraint_friction_mu, nconstraint);
        return;
    }

    // Block-uniform: only blocks that own rows stage the tables.
    const bool lds_ok = CS_LDS && (fuse_flags & CS_FUSE_LDS) && ((int)blockIdx.x * CS_BLK < nrow);
    if (lds_ok) {
        for (int e = tid; e < nbody * 3; e += CS_BLK) s_cxp[e] = xpos_d[(size_t)env * nbody * 3 + e];
        for (int e = tid; e < nbody * 4; e += CS_BLK) s_cxq[e] = xquat_d[(size_t)env * nbody * 4 + e];
        for (int e = tid; e < njnt * 3; e += CS_BLK) s_cxa[e] = jnt_xaxis_d[(size_t)env * njnt * 3 + e];
        for (int e = tid; e < njnt; e += CS_BLK) { s_cjt[e] = jnt_type[e]; s_cjd[e] = jnt_dof_adr[e]; }
        for (int e = tid; e < nbody; e += CS_BLK) { s_cst[e] = body_chain_start[e]; s_cln[e] = body_chain_len[e]; }
        for (int e = tid; e < chain_total; e += CS_BLK) {
            s_cji[e] = body_chain_jnt_idx[e]; s_cjb[e] = body_chain_jnt_body[e]; s_cpa[e] = body_chain_parent[e];
        }
        __syncthreads();
    }
    const CsView<double> vxp{s_cxp, env * nbody * 3};
    const CsView<double> vxq{s_cxq, env * nbody * 4};
    const CsView<double> vxa{s_cxa, env * njnt * 3};
    const CsView<int> vjt{s_cjt, 0}, vjd{s_cjd, 0};

    for (int r0 = blockIdx.x * CS_BLK; r0 < nrow; r0 += CS_GRID_X * CS_BLK) {
        const int r = r0 + tid;
        const int cid = r / 3;
        const int kind = r - cid * 3;
        const bool live = (r < nrow);

        // nconstraint += 3*nc + ntorsion, one contribution per contact (the
        // integer sum is order-independent, so this equals the old cid==0 add).
        int add = 0;
        if (live && kind == 0) add = 3 + (contact_has_torsion(env_cp[cid], geom_condim) ? 1 : 0);
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) add += __shfl_xor(add, off);
        if (tid == 0 && add) atomicAdd(&nconstraint[env], add);

        const int normal_slot = cid * 3;
        const bool work = live && (normal_slot + 2 < MAX_CONSTRAINTS);
        if (work) {
            const ContactPair& cp = env_cp[cid];
            int condim = 3;
            if (geom_condim) {
                int cd1 = geom_condim[cp.geom1];
                int cd2 = geom_condim[cp.geom2];
                condim = (cd1 > cd2) ? cd1 : cd2;
            }
            const int slot = normal_slot + kind;
            const size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
            for (int v = 0; v < nv; v++) Jl[v] = 0.0;

            if (kind != 0 && condim < 3) {
                constraint_rhs[idx] = 0.0;
                constraint_D[idx] = 0.0;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 0.0;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
            } else {
                double t1[3], t2[3];
                const double* dir = cp.normal;
                if (kind != 0) {
                    compute_tangents(cp.normal, t1, t2);
                    dir = (kind == 1) ? t1 : t2;
                }
                if (lds_ok) {
                    cs_fill_chain_lds(Jl, cp.body1, -1.0, dir, cp.pos,
                        s_cst, s_cln, s_cji, s_cjb, s_cpa,
                        vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                    cs_fill_chain_lds(Jl, cp.body2, +1.0, dir, cp.pos,
                        s_cst, s_cln, s_cji, s_cjb, s_cpa,
                        vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                } else {
                fill_chain_jacobian_precomputed_d(Jl, cp.body1, -1.0, dir, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
                fill_chain_jacobian_precomputed_d(Jl, cp.body2, +1.0, dir, cp.pos,
                    body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
                    jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
                }

                const int g1 = cp.geom1, g2 = cp.geom2;
                double sr0 = 0.5 * (geom_solref_d[g1*2]   + geom_solref_d[g2*2]);
                double sr1 = 0.5 * (geom_solref_d[g1*2+1] + geom_solref_d[g2*2+1]);
                double si0 = 0.5 * (geom_solimp_d[g1*5]   + geom_solimp_d[g2*5]);
                double si1 = 0.5 * (geom_solimp_d[g1*5+1] + geom_solimp_d[g2*5+1]);
                double si2 = 0.5 * (geom_solimp_d[g1*5+2] + geom_solimp_d[g2*5+2]);
                double si3 = 0.5 * (geom_solimp_d[g1*5+3] + geom_solimp_d[g2*5+3]);
                double si4 = 0.5 * (geom_solimp_d[g1*5+4] + geom_solimp_d[g2*5+4]);

                double invweight = body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2];
                double Jv = 0;
                for (int v = 0; v < nv; v++) Jv += Jl[v] * ev[v];

                int torsion_slot = -1;
                if (condim >= 4 && kind != 2) {
                    int ts = nc * 3;
                    for (int c = 0; c < cid; ++c)
                        ts += contact_has_torsion(env_cp[c], geom_condim);
                    if (ts < MAX_CONSTRAINTS) torsion_slot = ts;
                }

                if (kind == 0) {
                    invweight = fmax(invweight, 1e-15);
                    double D_normal, aref;
                    mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                     cp.depth, cp.depth, invweight, Jv, timestep, &D_normal, &aref, max_depen_vel);
                    constraint_rhs[idx] = aref;
                    constraint_D[idx] = D_normal;
                    constraint_lo[idx] = 0.0;
                    constraint_hi[idx] = 1e20;
                    constraint_friction_parent[idx] = -1;
                    constraint_friction_sibling[idx] = torsion_slot;   // -1 unless a twist row exists
                    constraint_friction_mu[idx] = 0.0;
                } else {
                    const double mu = cp.friction;
                    double invweight_fric = invweight / fmax(impratio, 1e-6);
                    invweight_fric = fmax(invweight_fric, 1e-15);
                    double D_fric, aref_fric;
                    mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                     cp.depth, 0.0, invweight_fric, Jv, timestep, &D_fric, &aref_fric);
                    constraint_rhs[idx] = aref_fric;
                    constraint_D[idx] = D_fric;
                    constraint_lo[idx] = -(double)mu;
                    constraint_hi[idx] = (double)mu;
                    constraint_friction_parent[idx] = normal_slot;
                    constraint_friction_sibling[idx] = (kind == 1) ? normal_slot + 2 : normal_slot + 1;
                    constraint_friction_mu[idx] = mu;

                    if (kind == 1 && torsion_slot >= 0) {
                        // Torsional row: identical to the per-contact kernel.
                        const double invweight_n = fmax(invweight, 1e-15);
                        double* Jt = constraint_J + (size_t)env*MAX_CONSTRAINTS*nv + (size_t)torsion_slot*nv;
                        for (int v = 0; v < nv; ++v) Jt[v] = 0.0;
                        fill_chain_angular_jacobian_d(Jt, cp.body1, -1.0, cp.normal,
                            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                            body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                            env, nbody, jnt_xaxis_d, njnt);
                        fill_chain_angular_jacobian_d(Jt, cp.body2, +1.0, cp.normal,
                            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                            body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xquat_d,
                            env, nbody, jnt_xaxis_d, njnt);
                        double Jv_t = 0.0;
                        for (int v = 0; v < nv; ++v) Jv_t += Jt[v]*ev[v];
                        if (jt_ok && torsion_slot < JT_STRIDE)
                            for (int v = 0; v < nv; ++v) JTenv[(size_t)v * JT_STRIDE + torsion_slot] = Jt[v];
                        const double mu_t = fmax(fmax(geom_friction_d[g1*3+1], geom_friction_d[g2*3+1]), 1e-5);
                        const double ratio = mu_t / fmax(mu, 1e-5);
                        double unused_D, aref_t;
                        mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                            cp.depth, 0.0, invweight_n, Jv_t, timestep, &unused_D, &aref_t);
                        const size_t ti = (size_t)env*MAX_CONSTRAINTS + torsion_slot;
                        constraint_D[ti] = D_fric * ratio*ratio;
                        constraint_rhs[ti] = aref_t;
                        constraint_lo[ti] = -mu_t;
                        constraint_hi[ti] = mu_t;
                        constraint_friction_parent[ti] = normal_slot;
                        constraint_friction_sibling[ti] = -1;
                        constraint_friction_mu[ti] = mu_t;
                    }
                }
            }
        }
        if (jt_ok && work && r < JT_STRIDE)
            for (int v = 0; v < nv; v++) JTenv[(size_t)v * JT_STRIDE + r] = Jl[v];
        __syncthreads();
        // Coalesced write-out of this chunk's rows (contiguous slots r0..).
        int nr = nrow - r0; if (nr > CS_BLK) nr = CS_BLK;
        {
            int lim = MAX_CONSTRAINTS - r0; if (nr > lim) nr = lim;
            // rows whose contact overflowed MAX_CONSTRAINTS were never built
            const int last_ok = ((MAX_CONSTRAINTS - 3) / 3) * 3 + 3 - r0;   // rows < 3*(cid_max+1)
            if (nr > last_ok) nr = last_ok;
        }
        double* Jdst = constraint_J + ((size_t)env * MAX_CONSTRAINTS + r0) * nv;
        const int tot = nr * nv;
        for (int i = tid; i < tot; i += CS_BLK) {
            const int row = i / nv;
            const int v = i - row * nv;
            Jdst[i] = sJ[v * CS_BLK + row];
        }
        __syncthreads();
    }
}

// ── Phase 7: Parallel joint limit constraints (1 thread per joint per env) ──


constexpr int GCB_BLK = 64;
HIPGL_DEVICE __attribute__((always_inline)) inline void gravcomp_lds_block(
    int env, int num_envs, int nv, int nbody, int njnt, int n_gravcomp, int chain_total,
    const int* __restrict__ gravcomp_body, const double* __restrict__ gravcomp_scale,
    const double* __restrict__ body_mass_d, const double* __restrict__ body_ipos_d,
    double gx, double gy, double gz,
    const int* __restrict__ body_chain_start, const int* __restrict__ body_chain_len,
    const int* __restrict__ body_chain_jnt_idx, const int* __restrict__ body_chain_jnt_body,
    const int* __restrict__ body_chain_parent,
    const int* __restrict__ jnt_type, const int* __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis_d,
    const double* __restrict__ xpos_d, const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    double* __restrict__ qfrc_bias)
{
    __shared__ double s_cxp[CS_LDS_NB * 3];
    __shared__ double s_cxq[CS_LDS_NB * 4];
    __shared__ double s_cxa[CS_LDS_NJ * 3];
    __shared__ int s_cjt[CS_LDS_NJ], s_cjd[CS_LDS_NJ];
    __shared__ int s_cst[CS_LDS_NB], s_cln[CS_LDS_NB];
    __shared__ int s_cji[CS_LDS_CH], s_cjb[CS_LDS_CH], s_cpa[CS_LDS_CH];
    __shared__ double s_f[GCB_BLK], s_com[GCB_BLK * 3];
    __shared__ int s_gb[GCB_BLK];
    __shared__ double s_b[KERNEL_MAXNV];
    const int tid = threadIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;
    const double gn = sqrt(gx * gx + gy * gy + gz * gz);
    if (gn == 0.0) return;
    double* bias = qfrc_bias + (size_t)env * nv;
    for (int e = tid; e < nbody * 3; e += GCB_BLK) s_cxp[e] = xpos_d[(size_t)env * nbody * 3 + e];
    for (int e = tid; e < nbody * 4; e += GCB_BLK) s_cxq[e] = xquat_d[(size_t)env * nbody * 4 + e];
    for (int e = tid; e < njnt * 3; e += GCB_BLK) s_cxa[e] = jnt_xaxis_d[(size_t)env * njnt * 3 + e];
    for (int e = tid; e < njnt; e += GCB_BLK) { s_cjt[e] = jnt_type[e]; s_cjd[e] = jnt_dof_adr[e]; }
    for (int e = tid; e < nbody; e += GCB_BLK) { s_cst[e] = body_chain_start[e]; s_cln[e] = body_chain_len[e]; }
    for (int e = tid; e < chain_total; e += GCB_BLK) {
        s_cji[e] = body_chain_jnt_idx[e]; s_cjb[e] = body_chain_jnt_body[e]; s_cpa[e] = body_chain_parent[e];
    }
    for (int e = tid; e < nv; e += GCB_BLK) s_b[e] = bias[e];
    if (tid < n_gravcomp) {
        const int k = tid;
        const int b = gravcomp_body[k];
        const double f = gravcomp_scale[k] * body_mass_d[b] * gn;
        // World COM: xpos + R(xquat) * ipos
        const double* q = xquat_d + ((size_t)env * nbody + b) * 4;
        const double* ip = body_ipos_d + b * 3;
        const double w = q[0], x = q[1], y = q[2], z = q[3];
        const double tx = 2 * (y * ip[2] - z * ip[1]);
        const double ty = 2 * (z * ip[0] - x * ip[2]);
        const double tz = 2 * (x * ip[1] - y * ip[0]);
        const double* xp = xpos_d + ((size_t)env * nbody + b) * 3;
        s_com[k * 3 + 0] = xp[0] + ip[0] + w * tx + (y * tz - z * ty);
        s_com[k * 3 + 1] = xp[1] + ip[1] + w * ty + (z * tx - x * tz);
        s_com[k * 3 + 2] = xp[2] + ip[2] + w * tz + (x * ty - y * tx);
        s_f[k] = f;
        s_gb[k] = b;
    }
    __syncthreads();
    if (tid == 0) {
        // Direction of the compensating force (against gravity).
        const double up[3] = {-gx / gn, -gy / gn, -gz / gn};
        const CsView<double> vxp{s_cxp, env * nbody * 3};
        const CsView<double> vxq{s_cxq, env * nbody * 4};
        const CsView<double> vxa{s_cxa, env * njnt * 3};
        const CsView<int> vjt{s_cjt, 0}, vjd{s_cjd, 0};
        for (int k = 0; k < n_gravcomp; k++) {
            const double com[3] = {s_com[k * 3 + 0], s_com[k * 3 + 1], s_com[k * 3 + 2]};
            cs_fill_chain_lds(s_b, s_gb[k], -s_f[k], up, com,
                s_cst, s_cln, s_cji, s_cjb, s_cpa,
                vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
        }
    }
    __syncthreads();
    for (int e = tid; e < nv; e += GCB_BLK) bias[e] = s_b[e];
}


constexpr int PYR_BLK = 64;
constexpr int PYR_CPB = PYR_BLK / 4;   // contacts per chunk (4 basis rows each)
constexpr int PYR_GX  = 4;             // contact blocks per env (grid-stride over chunks)
constexpr int PYR_NVC = 32;            // nv cap for the LDS basis buffer
struct PyrJRow {
    double* p;
    HIPGL_DEVICE double& operator[](int i) const { return p[i * PYR_BLK]; }
};
struct PyrJRow1 {
    double* p;
    HIPGL_DEVICE double& operator[](int i) const { return p[i]; }
};

// LDS twin of fill_chain_angular_jacobian_d (identical expressions).
template <class JRow>
HIPGL_DEVICE inline void cs_fill_chain_angular_lds(
    JRow J, int body, double sign, const double* dir,
    const int* s_cst, const int* s_cln, const int* s_cji, const int* s_cjb,
    CsView<int> jnt_type, CsView<int> jnt_dof_adr,
    CsView<double> xquat, int env, int nbody, CsView<double> jnt_xaxis, int njnt)
{
    const int start = s_cst[body];
    for (int k = 0; k < s_cln[body]; ++k) {
        const int ji = s_cji[start+k];
        const int da = jnt_dof_adr[ji];
        if (jnt_type[ji] == JOINT_HINGE) {
            double axis[3];
            for (int a = 0; a < 3; ++a) axis[a] = jnt_xaxis[(env*njnt+ji)*3+a];
            J[da] += sign*(axis[0]*dir[0] + axis[1]*dir[1] + axis[2]*dir[2]);
        } else if (jnt_type[ji] == JOINT_FREE || jnt_type[ji] == JOINT_BALL) {
            const int qb = (env*nbody+s_cjb[start+k])*4;
            const double w=xquat[qb], x=xquat[qb+1], y=xquat[qb+2], z=xquat[qb+3];
            const double R[9] = {
                1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
            const int rot = da + (jnt_type[ji] == JOINT_FREE ? 3 : 0);
            for (int a = 0; a < 3; ++a)
                J[rot+a] += sign*(R[a]*dir[0] + R[3+a]*dir[1] + R[6+a]*dir[2]);
        }
    }
}

HIPGL_GLOBAL void __launch_bounds__(PYR_BLK) kernel_constraint_setup_pyr(
    int num_envs, int nv, int nbody, int njnt, int nq,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const int*   __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis_d,
    const double* __restrict__ jnt_range_d,
    const int*   __restrict__ jnt_limited,
    const double* __restrict__ jnt_frictionloss_d,
    const double* __restrict__ dof_invweight0_d,
    const int*   __restrict__ geom_condim,
    const double* __restrict__ geom_friction_d,
    const double* __restrict__ geom_solref_d,
    const double* __restrict__ geom_solimp_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ body_invweight0_d,
    double timestep,
    double impratio,
    const ContactPair* __restrict__ contacts,
    const int*   __restrict__ ncon,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    const int*   __restrict__ body_chain_start,
    const int*   __restrict__ body_chain_len,
    const int*   __restrict__ body_chain_jnt_idx,
    const int*   __restrict__ body_chain_jnt_body,
    const int*   __restrict__ body_chain_parent,
    int chain_total,
    double*      __restrict__ constraint_J,
    double*      __restrict__ constraint_rhs,
    double*      __restrict__ constraint_D,
    double*      __restrict__ constraint_lo,
    double*      __restrict__ constraint_hi,
    int*         __restrict__ constraint_friction_parent,
    int*         __restrict__ constraint_friction_sibling,
    double*      __restrict__ constraint_friction_mu,
    int*         __restrict__ nconstraint,
    int pyr_edges,
    bool want_jl,
    double max_depen_vel,
    // fused gravity compensation (block x == PYR_GX + 1); n_gravcomp <= 0 = off
    int n_gravcomp,
    const int* __restrict__ gravcomp_body, const double* __restrict__ gravcomp_scale,
    const double* __restrict__ body_mass_d, const double* __restrict__ body_ipos_d,
    double gx, double gy, double gz,
    double* __restrict__ qfrc_bias)
{
    __shared__ double sB[PYR_NVC * PYR_BLK];
    __shared__ double s_cxp[CS_LDS_NB * 3];
    __shared__ double s_cxq[CS_LDS_NB * 4];
    __shared__ double s_cxa[CS_LDS_NJ * 3];
    __shared__ int s_cjt[CS_LDS_NJ], s_cjd[CS_LDS_NJ];
    __shared__ int s_cst[CS_LDS_NB], s_cln[CS_LDS_NB];
    __shared__ int s_cji[CS_LDS_CH], s_cjb[CS_LDS_CH], s_cpa[CS_LDS_CH];
    const int env = blockIdx.y;
    if (env >= num_envs) return;
    const int tid = threadIdx.x;
    const int nc = ncon[env];
    const int spc = pyr_edges;
    const ContactPair* env_cp = contacts + (size_t)env * MAX_CONTACTS;
    double* Jenv = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv;
    double* JT   = Jenv + (size_t)JT_ROW_OFF * nv;
    // Contact rows stay below JT_STRIDE (joint rows add <= 3*njnt <= 192 more),
    // so no live row can reach the mirror rows at JT_ROW_OFF.  Beyond that the
    // final count exceeds JT_STRIDE and the solver ignores the mirror anyway.
    const bool jt_ok = (nc * spc <= JT_STRIDE);

    if (blockIdx.x == PYR_GX + 1) {
        if (n_gravcomp > 0)
            gravcomp_lds_block(env, num_envs, nv, nbody, njnt, n_gravcomp, chain_total,
                gravcomp_body, gravcomp_scale, body_mass_d, body_ipos_d, gx, gy, gz,
                body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
                body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d,
                jnt_xaxis_d, qfrc_bias);
        return;
    }
    if (blockIdx.x == PYR_GX) {
        // This block owns the env's row count: contact rows (nc * spc, the old
        // cid==0 add) + the joint rows, STORED (no pre-clear needed).
        if (want_jl)
            cs_joint_rows_wave(env, nv, nq, njnt, nc, true, env_cp, geom_condim,
                jnt_type, jnt_qpos_adr, jnt_dof_adr, jnt_range_d, jnt_limited,
                jnt_frictionloss_d, M_inv_diag, dof_invweight0_d, timestep,
                qpos_d, qvel_d, constraint_J, constraint_rhs, constraint_D,
                constraint_lo, constraint_hi, constraint_friction_parent,
                constraint_friction_mu, nconstraint, nc * spc, constraint_friction_sibling);
        else if (tid == 0)
            nconstraint[env] = nc * spc;
        return;
    }
    if ((int)blockIdx.x * PYR_CPB >= nc) return;   // block-uniform

    for (int e = tid; e < nbody * 3; e += PYR_BLK) s_cxp[e] = xpos_d[(size_t)env * nbody * 3 + e];
    for (int e = tid; e < nbody * 4; e += PYR_BLK) s_cxq[e] = xquat_d[(size_t)env * nbody * 4 + e];
    for (int e = tid; e < njnt * 3; e += PYR_BLK) s_cxa[e] = jnt_xaxis_d[(size_t)env * njnt * 3 + e];
    for (int e = tid; e < njnt; e += PYR_BLK) { s_cjt[e] = jnt_type[e]; s_cjd[e] = jnt_dof_adr[e]; }
    for (int e = tid; e < nbody; e += PYR_BLK) { s_cst[e] = body_chain_start[e]; s_cln[e] = body_chain_len[e]; }
    for (int e = tid; e < chain_total; e += PYR_BLK) {
        s_cji[e] = body_chain_jnt_idx[e]; s_cjb[e] = body_chain_jnt_body[e]; s_cpa[e] = body_chain_parent[e];
    }
    __syncthreads();
    const CsView<double> vxp{s_cxp, env * nbody * 3};
    const CsView<double> vxq{s_cxq, env * nbody * 4};
    const CsView<double> vxa{s_cxa, env * njnt * 3};
    const CsView<int> vjt{s_cjt, 0}, vjd{s_cjd, 0};
    const double* ev = qvel_d + (size_t)env * nv;

    for (int c0 = blockIdx.x * PYR_CPB; c0 < nc; c0 += PYR_GX * PYR_CPB) {
        // ── phase A: basis rows ──
        {
            const PyrJRow Jb{sB + tid};
            for (int v = 0; v < nv; v++) Jb[v] = 0.0;
            const int cid  = c0 + (tid >> 2);
            const int kind = tid & 3;
            if (cid < nc && cid * spc + spc - 1 < MAX_CONSTRAINTS) {
                const ContactPair& cp = env_cp[cid];
                int condim = 3;
                if (geom_condim) {
                    int cd1 = geom_condim[cp.geom1];
                    int cd2 = geom_condim[cp.geom2];
                    condim = (cd1 > cd2) ? cd1 : cd2;
                }
                if (kind == 3) {
                    if (pyr_edges == 6 && condim >= 4) {
                        cs_fill_chain_angular_lds(Jb, cp.body1, -1.0, cp.normal,
                            s_cst, s_cln, s_cji, s_cjb, vjt, vjd, vxq, env, nbody, vxa, njnt);
                        cs_fill_chain_angular_lds(Jb, cp.body2, +1.0, cp.normal,
                            s_cst, s_cln, s_cji, s_cjb, vjt, vjd, vxq, env, nbody, vxa, njnt);
                    }
                } else if (kind == 0) {
                    cs_fill_chain_lds(Jb, cp.body1, -1.0, cp.normal, cp.pos,
                        s_cst, s_cln, s_cji, s_cjb, s_cpa,
                        vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                    cs_fill_chain_lds(Jb, cp.body2, +1.0, cp.normal, cp.pos,
                        s_cst, s_cln, s_cji, s_cjb, s_cpa,
                        vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                } else if (condim >= 3) {
                    double t1[3], t2[3];
                    compute_tangents(cp.normal, t1, t2);
                    if (kind == 1) {
                        cs_fill_chain_lds(Jb, cp.body1, -1.0, t1, cp.pos,
                            s_cst, s_cln, s_cji, s_cjb, s_cpa,
                            vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                        cs_fill_chain_lds(Jb, cp.body2, +1.0, t1, cp.pos,
                            s_cst, s_cln, s_cji, s_cjb, s_cpa,
                            vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                    } else {
                        cs_fill_chain_lds(Jb, cp.body1, -1.0, t2, cp.pos,
                            s_cst, s_cln, s_cji, s_cjb, s_cpa,
                            vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                        cs_fill_chain_lds(Jb, cp.body2, +1.0, t2, cp.pos,
                            s_cst, s_cln, s_cji, s_cjb, s_cpa,
                            vjt, vjd, jnt_axis_d, vxp, vxq, xquat_d, env, nbody, vxa, njnt);
                    }
                }
            }
        }
        __syncthreads();
        // ── phase B: edge rows ──
        for (int e = tid; e < PYR_CPB * spc; e += PYR_BLK) {
            const int lc = e / spc;
            const int k  = e - lc * spc;
            const int cid = c0 + lc;
            if (cid >= nc) break;
            const int normal_slot = cid * spc;
            if (normal_slot + spc - 1 >= MAX_CONSTRAINTS) break;
            const ContactPair& cp = env_cp[cid];
            const PyrJRow Jn{sB + 4 * lc}, Ja{sB + 4 * lc + 1}, Jbb{sB + 4 * lc + 2}, Jc{sB + 4 * lc + 3};
            int condim = 3;
            if (geom_condim) {
                int cd1 = geom_condim[cp.geom1];
                int cd2 = geom_condim[cp.geom2];
                condim = (cd1 > cd2) ? cd1 : cd2;
            }
            const int g1 = cp.geom1, g2 = cp.geom2;
            const double sr0 = 0.5 * (geom_solref_d[g1*2]   + geom_solref_d[g2*2]);
            const double sr1 = 0.5 * (geom_solref_d[g1*2+1] + geom_solref_d[g2*2+1]);
            const double si0 = 0.5 * (geom_solimp_d[g1*5]   + geom_solimp_d[g2*5]);
            const double si1 = 0.5 * (geom_solimp_d[g1*5+1] + geom_solimp_d[g2*5+1]);
            const double si2 = 0.5 * (geom_solimp_d[g1*5+2] + geom_solimp_d[g2*5+2]);
            const double si3 = 0.5 * (geom_solimp_d[g1*5+3] + geom_solimp_d[g2*5+3]);
            const double si4 = 0.5 * (geom_solimp_d[g1*5+4] + geom_solimp_d[g2*5+4]);
            double invweight = body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2];
            invweight = fmax(invweight, 1e-15);
            const int slot = normal_slot + k;
            const size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
            double* Jg = Jenv + (size_t)slot * nv;
            const bool torsion = pyr_edges == 6 && condim >= 4;
            const int nedge = (condim < 3) ? 1 : (torsion ? 6 : 4);
            if (k >= nedge) {
                // unused slot of the block: zero row, neutral meta
                for (int v = 0; v < nv; v++) {
                    Jg[v] = 0.0;
                    if (jt_ok) JT[(size_t)v * JT_STRIDE + slot] = 0.0;
                }
                constraint_rhs[idx] = 0.0;
                constraint_D[idx] = 0.0;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 0.0;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
                continue;
            }
            const double mu = cp.friction;
            const double mu_t = torsion
                ? fmax(fmax(geom_friction_d[g1*3+1], geom_friction_d[g2*3+1]), 1e-5) : 0.0;
            double Jvk = 0;
            for (int v = 0; v < nv; v++) {
                const double n = Jn[v];
                double r;
                if (condim < 3)  r = n;
                else if (k == 0) r = n + mu * Ja[v];
                else if (k == 1) r = n - mu * Ja[v];
                else if (k == 2) r = n + mu * Jbb[v];
                else if (k == 3) r = n - mu * Jbb[v];
                else if (k == 4) r = n + mu_t * Jc[v];
                else             r = n - mu_t * Jc[v];
                Jg[v] = r;
                if (jt_ok) JT[(size_t)v * JT_STRIDE + slot] = r;
                Jvk += r * ev[v];
            }
            const double iw = (condim < 3) ? invweight
                : 2.0 * (mu * mu / fmax(impratio, 1e-15)) * invweight * (1.0 + mu * mu);
            double Dk, arefk;
            mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                             cp.depth, cp.depth, iw, Jvk, timestep, &Dk, &arefk, max_depen_vel);
            constraint_rhs[idx] = arefk;
            constraint_D[idx] = Dk;
            constraint_lo[idx] = 0.0;
            constraint_hi[idx] = 1e20;
            constraint_friction_parent[idx] = -1;
            constraint_friction_sibling[idx] = -1;
            constraint_friction_mu[idx] = 0.0;
        }
        __syncthreads();
    }
}

#ifndef PYF_TW_V
#define PYF_TW_V 512
#endif
constexpr int PYF_TW  = PYF_TW_V;          // term lanes
constexpr int PYF_BLK = PYF_TW + 64;       // term waves + 1 wave per-contact prep
template <class JRow>
HIPGL_DEVICE inline void cs_add_joint_ang_lds(
    JRow J, int ji, int jb, double sign, const double* dir,
    CsView<int> jnt_type, CsView<int> jnt_dof_adr,
    CsView<double> xquat, int env, int nbody, CsView<double> jnt_xaxis, int njnt)
{
    const int da = jnt_dof_adr[ji];
    if (jnt_type[ji] == JOINT_HINGE) {
        double axis[3];
        for (int a = 0; a < 3; ++a) axis[a] = jnt_xaxis[(env*njnt+ji)*3+a];
        J[da] += sign*(axis[0]*dir[0] + axis[1]*dir[1] + axis[2]*dir[2]);
    } else if (jnt_type[ji] == JOINT_FREE || jnt_type[ji] == JOINT_BALL) {
        const int qb = (env*nbody+jb)*4;
        const double w=xquat[qb], x=xquat[qb+1], y=xquat[qb+2], z=xquat[qb+3];
        const double R[9] = {
            1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
            2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
            2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
        const int rot = da + (jnt_type[ji] == JOINT_FREE ? 3 : 0);
        for (int a = 0; a < 3; ++a)
            J[rot+a] += sign*(R[a]*dir[0] + R[3+a]*dir[1] + R[6+a]*dir[2]);
    }
}

// mujoco_impedance split at its only Jv use (identical expressions):
// aref = pos_term - b * Jv.
HIPGL_DEVICE inline void mujoco_impedance_pre(
    double solref0, double solref1,
    double solimp0, double solimp1, double solimp2, double solimp3, double solimp4,
    double imp_depth, double aref_depth, double invweight, double timestep,
    double* out_D, double* out_pos_term, double* out_b, double max_depen_vel)
{
    constexpr double MINIMP = 1e-4;
    constexpr double MAXIMP = 0.9999;
    constexpr double MINVAL = 1e-15;

    double dmin_imp = fmin(fmax(solimp0, MINIMP), MAXIMP);
    double dmax_imp = fmin(fmax(solimp1, MINIMP), MAXIMP);
    double width    = fmax(solimp2, MINVAL);
    double mid_val  = fmin(fmax(solimp3, MINIMP), MAXIMP);
    double pwr      = fmax(solimp4, 1.0);

    double k, b;
    if (solref0 > 0.0 && solref1 > 0.0) {
        double tc = fmax(solref0, 2.0 * timestep);
        double dr = solref1;
        k = 1.0 / (dmax_imp * dmax_imp * tc * tc * dr * dr);
        b = 2.0 / (dmax_imp * tc);
    } else {
        k = (solref0 <= 0.0) ? (-solref0 / (dmax_imp * dmax_imp)) : (1.0 / (dmax_imp * dmax_imp * solref0 * solref0));
        b = (solref1 <= 0.0) ? (-solref1 / dmax_imp) : (2.0 * solref1 / (dmax_imp * solref0));
    }

    double imp_x = fabs(imp_depth) / width;
    double imp;
    if (imp_x >= 1.0) {
        imp = dmax_imp;
    } else if (imp_x < mid_val) {
        double base = pow(mid_val, pwr - 1.0);
        double imp_a = (base > MINVAL) ? pow(imp_x, pwr) / base : 0.0;
        imp = dmin_imp + imp_a * (dmax_imp - dmin_imp);
    } else {
        double base = pow(1.0 - mid_val, pwr - 1.0);
        double imp_b = (base > MINVAL) ? 1.0 - pow(1.0 - imp_x, pwr) / base : 1.0;
        imp = dmin_imp + imp_b * (dmax_imp - dmin_imp);
    }
    imp = fmin(fmax(imp, dmin_imp), dmax_imp);

    double r = fmax(invweight * (1.0 - imp) / imp, MINVAL);
    constexpr double MAX_D = 1e10;
    *out_D = fmin(1.0 / r, MAX_D);

    double pos_term = k * imp * aref_depth;
    if (max_depen_vel > 0.0) pos_term = fmin(pos_term, b * max_depen_vel);
    *out_pos_term = pos_term;
    *out_b = b;
}

HIPGL_GLOBAL void __launch_bounds__(PYF_BLK) kernel_constraint_setup_pyr_fast(
    int num_envs, int nv, int nbody, int njnt, int nq,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const int*   __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis_d,
    const double* __restrict__ jnt_range_d,
    const int*   __restrict__ jnt_limited,
    const double* __restrict__ jnt_frictionloss_d,
    const double* __restrict__ dof_invweight0_d,
    const int*   __restrict__ geom_condim,
    const double* __restrict__ geom_friction_d,
    const double* __restrict__ geom_solref_d,
    const double* __restrict__ geom_solimp_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ body_invweight0_d,
    double timestep,
    double impratio,
    const ContactPair* __restrict__ contacts,
    const int*   __restrict__ ncon,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    const int*   __restrict__ body_chain_start,
    const int*   __restrict__ body_chain_len,
    const int*   __restrict__ body_chain_jnt_idx,
    const int*   __restrict__ body_chain_jnt_body,
    const int*   __restrict__ body_chain_parent,
    int chain_total,
    double*      __restrict__ constraint_J,
    double*      __restrict__ constraint_rhs,
    double*      __restrict__ constraint_D,
    double*      __restrict__ constraint_lo,
    double*      __restrict__ constraint_hi,
    int*         __restrict__ constraint_friction_parent,
    int*         __restrict__ constraint_friction_sibling,
    double*      __restrict__ constraint_friction_mu,
    int*         __restrict__ nconstraint,
    int pyr_edges,
    bool want_jl,
    double max_depen_vel,
    // fused gravity compensation (block x == PYR_GX + 1); n_gravcomp <= 0 = off
    int n_gravcomp,
    const int* __restrict__ gravcomp_body, const double* __restrict__ gravcomp_scale,
    const double* __restrict__ body_mass_d, const double* __restrict__ body_ipos_d,
    double gx, double gy, double gz,
    double* __restrict__ qfrc_bias)
{
    __shared__ double sB[2][PYR_NVC * PYR_BLK];
    __shared__ double s_cxp[CS_LDS_NB * 3];
    __shared__ double s_cxq[CS_LDS_NB * 4];
    __shared__ double s_cxa[CS_LDS_NJ * 3];
    __shared__ double s_ev[PYR_NVC];
    __shared__ int s_cjt[CS_LDS_NJ], s_cjd[CS_LDS_NJ];
    __shared__ int s_cst[CS_LDS_NB], s_cln[CS_LDS_NB];
    __shared__ int s_cji[CS_LDS_CH], s_cjb[CS_LDS_CH], s_cpa[CS_LDS_CH];
    // per-contact staging (chunk of PYR_CPB contacts)
    __shared__ double s_pos[PYR_CPB][3], s_nrm[PYR_CPB][3], s_t1[PYR_CPB][3], s_t2[PYR_CPB][3];
    __shared__ double s_D[PYR_CPB], s_pt[PYR_CPB], s_bb[PYR_CPB], s_mu[PYR_CPB], s_mut[PYR_CPB];
    __shared__ int s_b1[PYR_CPB], s_b2[PYR_CPB], s_cd[PYR_CPB];
    // gravity compensation (block x == PYR_GX + 1)
    __shared__ double s_f[GCB_BLK], s_com[GCB_BLK * 3], s_b[PYR_NVC];
    __shared__ int s_gb[GCB_BLK];
    __shared__ unsigned char s_gfl[GCB_BLK * PYR_NVC];
    const int env = blockIdx.y;
    if (env >= num_envs) return;
    const int tid = threadIdx.x;

    if (blockIdx.x == PYR_GX + 1) {
        if (n_gravcomp <= 0) return;
        if (HIPGL_WORLD_FROZEN(env)) return;
        const double gn = sqrt(gx * gx + gy * gy + gz * gz);
        if (gn == 0.0) return;
        double* bias = qfrc_bias + (size_t)env * nv;
        double* const T = sB[0];           // [k][PYR_NVC] dot products (sign 1)
        for (int e = tid; e < nbody * 3; e += PYF_BLK) s_cxp[e] = xpos_d[(size_t)env * nbody * 3 + e];
        for (int e = tid; e < nbody * 4; e += PYF_BLK) s_cxq[e] = xquat_d[(size_t)env * nbody * 4 + e];
        for (int e = tid; e < njnt * 3; e += PYF_BLK) s_cxa[e] = jnt_xaxis_d[(size_t)env * njnt * 3 + e];
        for (int e = tid; e < njnt; e += PYF_BLK) { s_cjt[e] = jnt_type[e]; s_cjd[e] = jnt_dof_adr[e]; }
        for (int e = tid; e < nbody; e += PYF_BLK) { s_cst[e] = body_chain_start[e]; s_cln[e] = body_chain_len[e]; }
        for (int e = tid; e < chain_total; e += PYF_BLK) {
            s_cji[e] = body_chain_jnt_idx[e]; s_cjb[e] = body_chain_jnt_body[e]; s_cpa[e] = body_chain_parent[e];
        }
        for (int e = tid; e < n_gravcomp * PYR_NVC; e += PYF_BLK) { T[e] = 0.0; s_gfl[e] = 0; }
        if (tid < n_gravcomp) {
            const int k = tid;
            const int b = gravcomp_body[k];
            const double f = gravcomp_scale[k] * body_mass_d[b] * gn;
            // World COM: xpos + R(xquat) * ipos
            const double* q = xquat_d + ((size_t)env * nbody + b) * 4;
            const double* ip = body_ipos_d + b * 3;
            const double w = q[0], x = q[1], y = q[2], z = q[3];
            const double tx = 2 * (y * ip[2] - z * ip[1]);
            const double ty = 2 * (z * ip[0] - x * ip[2]);
            const double tz = 2 * (x * ip[1] - y * ip[0]);
            const double* xp = xpos_d + ((size_t)env * nbody + b) * 3;
            s_com[k * 3 + 0] = xp[0] + ip[0] + w * tx + (y * tz - z * ty);
            s_com[k * 3 + 1] = xp[1] + ip[1] + w * ty + (z * tx - x * tz);
            s_com[k * 3 + 2] = xp[2] + ip[2] + w * tz + (x * ty - y * tx);
            s_f[k] = f;
            s_gb[k] = b;
        }
        __syncthreads();
        {
            const double up[3] = {-gx / gn, -gy / gn, -gz / gn};
            const CsView<double> vxp{s_cxp, env * nbody * 3};
            const CsView<double> vxq{s_cxq, env * nbody * 4};
            const CsView<double> vxa{s_cxa, env * njnt * 3};
            const CsView<int> vjt{s_cjt, 0}, vjd{s_cjd, 0};
            const int per = PYF_BLK / n_gravcomp;
            if (tid < per * n_gravcomp) {
                const int k = tid % n_gravcomp;
                const int body = s_gb[k];
                const int start = s_cst[body], len = s_cln[body];
                const double com[3] = {s_com[k * 3 + 0], s_com[k * 3 + 1], s_com[k * 3 + 2]};
                const PyrJRow1 Tk{T + k * PYR_NVC};
                for (int i = tid / n_gravcomp; i < len; i += per) {
                    const int ji = s_cji[start + i];
                    cs_add_joint_jac_lds(Tk, ji, 1.0, vjt, vjd, jnt_axis_d,
                        up, com, vxp, vxq, xquat_d, env, nbody, s_cjb[start + i], s_cpa[start + i], vxa, njnt);
                    const int jt = s_cjt[ji], da = s_cjd[ji];
                    if (jt == JOINT_HINGE || jt == JOINT_SLIDE) s_gfl[k * PYR_NVC + da] = 1;
                    else if (jt == JOINT_FREE) for (int a = 0; a < 6; a++) s_gfl[k * PYR_NVC + da + a] = 1;
                }
            }
        }
        __syncthreads();
        if (tid < nv) {
            double acc = bias[tid];
            for (int k = 0; k < n_gravcomp; k++) {
                if (s_gfl[k * PYR_NVC + tid]) {
                    const double sign = -s_f[k];
                    acc += sign * T[k * PYR_NVC + tid];
                }
            }
            bias[tid] = acc;
        }
        return;
    }

    const int nc = ncon[env];
    const int spc = pyr_edges;
    const ContactPair* env_cp = contacts + (size_t)env * MAX_CONTACTS;
    double* Jenv = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv;
    double* JT   = Jenv + (size_t)JT_ROW_OFF * nv;
    const bool jt_ok = (nc * spc <= JT_STRIDE);

    if (blockIdx.x == PYR_GX) {
        if (tid >= 64) return;   // cs_joint_rows_wave is a one-wave routine
        if (want_jl)
            cs_joint_rows_wave(env, nv, nq, njnt, nc, true, env_cp, geom_condim,
                jnt_type, jnt_qpos_adr, jnt_dof_adr, jnt_range_d, jnt_limited,
                jnt_frictionloss_d, M_inv_diag, dof_invweight0_d, timestep,
                qpos_d, qvel_d, constraint_J, constraint_rhs, constraint_D,
                constraint_lo, constraint_hi, constraint_friction_parent,
                constraint_friction_mu, nconstraint, nc * spc, constraint_friction_sibling);
        else if (tid == 0)
            nconstraint[env] = nc * spc;
        return;
    }
    if ((int)blockIdx.x * PYR_CPB >= nc) return;   // block-uniform

    for (int e = tid; e < nbody * 3; e += PYF_BLK) s_cxp[e] = xpos_d[(size_t)env * nbody * 3 + e];
    for (int e = tid; e < nbody * 4; e += PYF_BLK) s_cxq[e] = xquat_d[(size_t)env * nbody * 4 + e];
    for (int e = tid; e < njnt * 3; e += PYF_BLK) s_cxa[e] = jnt_xaxis_d[(size_t)env * njnt * 3 + e];
    for (int e = tid; e < njnt; e += PYF_BLK) { s_cjt[e] = jnt_type[e]; s_cjd[e] = jnt_dof_adr[e]; }
    for (int e = tid; e < nbody; e += PYF_BLK) { s_cst[e] = body_chain_start[e]; s_cln[e] = body_chain_len[e]; }
    for (int e = tid; e < chain_total; e += PYF_BLK) {
        s_cji[e] = body_chain_jnt_idx[e]; s_cjb[e] = body_chain_jnt_body[e]; s_cpa[e] = body_chain_parent[e];
    }
    for (int e = tid; e < nv; e += PYF_BLK) s_ev[e] = qvel_d[(size_t)env * nv + e];
    const CsView<double> vxp{s_cxp, env * nbody * 3};
    const CsView<double> vxq{s_cxq, env * nbody * 4};
    const CsView<double> vxa{s_cxa, env * njnt * 3};
    const CsView<int> vjt{s_cjt, 0}, vjd{s_cjd, 0};
    double* const sB1 = sB[0];
    double* const sB2 = sB[1];
    const int lcw = tid - PYF_TW;          // prep-wave lane (contact in chunk)

    for (int c0 = blockIdx.x * PYR_CPB; c0 < nc; c0 += PYR_GX * PYR_CPB) {
        int ncc = nc - c0; if (ncc > PYR_CPB) ncc = PYR_CPB;
        // ── stage: zero both basis buffers; the prep wave loads contact geometry ──
        for (int e = tid; e < nv * PYR_BLK; e += PYF_BLK) { sB1[e] = 0.0; sB2[e] = 0.0; }
        // prep-lane registers carried across the barrier into the impedance step
        int condim = 3;
        double sr0 = 0, sr1 = 0, si0 = 0, si1 = 0, si2 = 0, si3 = 0, si4 = 0;
        double invweight = 0, mu = 0, mu_t = 0, depth = 0;
        if (lcw >= 0 && lcw < ncc) {
            const ContactPair& cp = env_cp[c0 + lcw];
            s_b1[lcw] = cp.body1; s_b2[lcw] = cp.body2;
            for (int a = 0; a < 3; a++) { s_pos[lcw][a] = cp.pos[a]; s_nrm[lcw][a] = cp.normal[a]; }
            const int g1 = cp.geom1, g2 = cp.geom2;
            if (geom_condim) {
                int cd1 = geom_condim[g1];
                int cd2 = geom_condim[g2];
                condim = (cd1 > cd2) ? cd1 : cd2;
            }
            sr0 = 0.5 * (geom_solref_d[g1*2]   + geom_solref_d[g2*2]);
            sr1 = 0.5 * (geom_solref_d[g1*2+1] + geom_solref_d[g2*2+1]);
            si0 = 0.5 * (geom_solimp_d[g1*5]   + geom_solimp_d[g2*5]);
            si1 = 0.5 * (geom_solimp_d[g1*5+1] + geom_solimp_d[g2*5+1]);
            si2 = 0.5 * (geom_solimp_d[g1*5+2] + geom_solimp_d[g2*5+2]);
            si3 = 0.5 * (geom_solimp_d[g1*5+3] + geom_solimp_d[g2*5+3]);
            si4 = 0.5 * (geom_solimp_d[g1*5+4] + geom_solimp_d[g2*5+4]);
            invweight = body_invweight0_d[cp.body1 * 2] + body_invweight0_d[cp.body2 * 2];
            invweight = fmax(invweight, 1e-15);
            mu = cp.friction;
            depth = cp.depth;
            const bool torsion = pyr_edges == 6 && condim >= 4;
            mu_t = torsion ? fmax(fmax(geom_friction_d[g1*3+1], geom_friction_d[g2*3+1]), 1e-5) : 0.0;
            double t1[3], t2[3];
            compute_tangents(cp.normal, t1, t2);
            for (int a = 0; a < 3; a++) { s_t1[lcw][a] = t1[a]; s_t2[lcw][a] = t2[a]; }
        }
        __syncthreads();
        if (tid < PYF_TW) {
            // ── phase A: one lane per (contact, kind, side, chain joint) term.  Rows
            // that the edge phase will not read (tangents of condim<3, torsion of
            // non-torsional contacts) are built anyway; they are never consumed.
            const int nrows = ncc * 8;
            const int per = PYF_TW / nrows;     // >= 2 (ncc <= 16)
            if (tid < per * nrows) {
                // (kind, side)-major rows: neighbouring lanes take the same branch
                const int row  = tid % nrows;
                const int ks   = row / ncc;
                const int lc   = row - ks * ncc;
                const int kind = ks >> 1;
                const int side = ks & 1;
                const int cid  = c0 + lc;
                if (cid * spc + spc - 1 < MAX_CONSTRAINTS) {
                    const int body = side ? s_b2[lc] : s_b1[lc];
                    const double sign = side ? +1.0 : -1.0;
                    const PyrJRow Jb{(side ? sB2 : sB1) + 4 * lc + kind};
                    const double* dir = (kind == 1) ? s_t1[lc] : (kind == 2) ? s_t2[lc] : s_nrm[lc];
                    const int start = s_cst[body];
                    const int len = s_cln[body];
                    for (int i = tid / nrows; i < len; i += per) {
                        const int ji = s_cji[start + i];
                        const int jb = s_cjb[start + i];
                        if (kind == 3) {
                            cs_add_joint_ang_lds(Jb, ji, jb, sign, dir, vjt, vjd, vxq, env, nbody, vxa, njnt);
                        } else {
                            cs_add_joint_jac_lds(Jb, ji, sign, vjt, vjd, jnt_axis_d,
                                dir, s_pos[lc], vxp, vxq, xquat_d, env, nbody, jb, s_cpa[start + i], vxa, njnt);
                        }
                    }
                }
            }
        } else if (lcw < ncc) {
            // ── prep wave: per-contact impedance (Jv-free part) ──
            const double iw = (condim < 3) ? invweight
                : 2.0 * (mu * mu / fmax(impratio, 1e-15)) * invweight * (1.0 + mu * mu);
            double Dk, pt, bb;
            mujoco_impedance_pre(sr0, sr1, si0, si1, si2, si3, si4,
                                 depth, depth, iw, timestep, &Dk, &pt, &bb, max_depen_vel);
            s_cd[lcw] = condim; s_mu[lcw] = mu; s_mut[lcw] = mu_t;
            s_D[lcw] = Dk; s_pt[lcw] = pt; s_bb[lcw] = bb;
        }
        __syncthreads();
        for (int e = tid; e < nv * PYR_BLK; e += PYF_BLK) sB1[e] = sB1[e] + sB2[e];
        __syncthreads();
        // ── phase B: edge rows ──
        for (int e = tid; e < PYR_CPB * spc; e += PYF_BLK) {
            const int lc = e / spc;
            const int k  = e - lc * spc;
            const int cid = c0 + lc;
            if (cid >= nc) break;
            const int normal_slot = cid * spc;
            if (normal_slot + spc - 1 >= MAX_CONSTRAINTS) break;
            const PyrJRow Jn{sB1 + 4 * lc}, Ja{sB1 + 4 * lc + 1}, Jbb{sB1 + 4 * lc + 2}, Jc{sB1 + 4 * lc + 3};
            const int condim = s_cd[lc];
            const int slot = normal_slot + k;
            const size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
            double* Jg = Jenv + (size_t)slot * nv;
            const bool torsion = pyr_edges == 6 && condim >= 4;
            const int nedge = (condim < 3) ? 1 : (torsion ? 6 : 4);
            if (k >= nedge) {
                for (int v = 0; v < nv; v++) {
                    Jg[v] = 0.0;
                    if (jt_ok) JT[(size_t)v * JT_STRIDE + slot] = 0.0;
                }
                constraint_rhs[idx] = 0.0;
                constraint_D[idx] = 0.0;
                constraint_lo[idx] = 0.0;
                constraint_hi[idx] = 0.0;
                constraint_friction_parent[idx] = -1;
                constraint_friction_sibling[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
                continue;
            }
            const double mu = s_mu[lc];
            const double mu_t = s_mut[lc];
            const double* Jx = (k < 2) ? &Ja[0] : (k < 4) ? &Jbb[0] : &Jc[0];
            const double cm = (k < 4) ? mu : mu_t;
            const bool plus = (k & 1) == 0;
            const bool solo = condim < 3;
            const double cms = plus ? cm : -cm;
            double Jvk = 0;
            #pragma unroll 4
            for (int v = 0; v < nv; v++) {
                const double n = Jn[v];
                // == the old per-k "n +- mu * J" (contracted to one fma)
                const double r = solo ? n : fma(cms, Jx[v * PYR_BLK], n);
                Jg[v] = r;
                if (jt_ok) JT[(size_t)v * JT_STRIDE + slot] = r;
                Jvk += r * s_ev[v];
            }
            constraint_rhs[idx] = s_pt[lc] - s_bb[lc] * Jvk;
            constraint_D[idx] = s_D[lc];
            constraint_lo[idx] = 0.0;
            constraint_hi[idx] = 1e20;
            constraint_friction_parent[idx] = -1;
            constraint_friction_sibling[idx] = -1;
            constraint_friction_mu[idx] = 0.0;
        }
        __syncthreads();
    }
}

HIPGL_GLOBAL void kernel_joint_limit_constraints(
    int num_envs, int nv, int nq, int njnt,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_range_d,
    const int*    __restrict__ jnt_limited,
    const double* __restrict__ jnt_frictionloss_d,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ dof_invweight0_d,
    double timestep,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    double*       __restrict__ constraint_J,
    double*       __restrict__ constraint_rhs,
    double*       __restrict__ constraint_D,
    double*       __restrict__ constraint_lo,
    double*       __restrict__ constraint_hi,
    int*          __restrict__ constraint_friction_parent,
    double*       __restrict__ constraint_friction_mu,
    int*          __restrict__ nconstraint)
{
    int ji = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (ji >= njnt || env >= num_envs) return;

    int jtype = jnt_type[ji];
    if (jtype != JOINT_HINGE && jtype != JOINT_SLIDE) return;

    const double* eq = qpos_d + env * nq;
    const double* ev = qvel_d + env * nv;
    const double* Minv = M_inv_diag + env * MAX_DOF;
    int qa = jnt_qpos_adr[ji];
    int da = jnt_dof_adr[ji];

    double sr0 = 0.02, sr1 = 1.0;
    double si0 = 0.9, si1 = 0.95, si2 = 0.001, si3 = 0.5, si4 = 2.0;
    double invweight = (dof_invweight0_d) ? dof_invweight0_d[da] : fmax(Minv[da], 1e-15);

    if (jnt_frictionloss_d) {
        double fl = jnt_frictionloss_d[ji];
        if (fl > 0.0) {
            int slot = atomicAdd(&nconstraint[env], 1);
            if (slot < MAX_CONSTRAINTS) {
                double* J_row = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)slot * nv;
                for (int v = 0; v < nv; v++) J_row[v] = 0.0;
                J_row[da] = 1.0;
                double D_fric, aref_fric;
                double Jv_fric = ev[da];
                mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                                 0.0, 0.0, invweight, Jv_fric, timestep, &D_fric, &aref_fric);
                size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
                constraint_rhs[idx] = aref_fric;
                constraint_D[idx] = D_fric;
                constraint_lo[idx] = -fl;
                constraint_hi[idx] = fl;
                constraint_friction_parent[idx] = -1;
                constraint_friction_mu[idx] = 0.0;
            }
        }
    }

    if (!jnt_limited[ji]) return;
    double q = eq[qa];
    double lo = jnt_range_d[ji * 2];
    double hi = jnt_range_d[ji * 2 + 1];
    if (lo >= hi - 1e-6) return;
    double margin = 0.0;

    double dist_lo = q - lo;
    double dist_hi = hi - q;

    if (dist_lo - margin < 0.0) {
        int slot = atomicAdd(&nconstraint[env], 1);
        if (slot < MAX_CONSTRAINTS) {
            double* J_row = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)slot * nv;
            for (int v = 0; v < nv; v++) J_row[v] = 0.0;
            J_row[da] = 1.0;
            double depth = -(dist_lo - margin);
            double Jv = ev[da];
            double D_limit, aref;
            mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                             depth, depth, invweight, Jv, timestep, &D_limit, &aref);
            size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
            constraint_rhs[idx] = aref;
            constraint_D[idx] = D_limit;
            constraint_lo[idx] = 0.0;
            constraint_hi[idx] = 1e20;
            constraint_friction_parent[idx] = -1;
            constraint_friction_mu[idx] = 0.0;
        }
    }
    if (dist_hi - margin < 0.0) {
        int slot = atomicAdd(&nconstraint[env], 1);
        if (slot < MAX_CONSTRAINTS) {
            double* J_row = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)slot * nv;
            for (int v = 0; v < nv; v++) J_row[v] = 0.0;
            J_row[da] = -1.0;
            double depth = -(dist_hi - margin);
            double Jv = -ev[da];
            double D_limit, aref;
            mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                             depth, depth, invweight, Jv, timestep, &D_limit, &aref);
            size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
            constraint_rhs[idx] = aref;
            constraint_D[idx] = D_limit;
            constraint_lo[idx] = 0.0;
            constraint_hi[idx] = 1e20;
            constraint_friction_parent[idx] = -1;
            constraint_friction_mu[idx] = 0.0;
        }
    }
}

// ── Phase 7: Parallel equality weld constraints (1 thread per eq per env) ───

HIPGL_GLOBAL void kernel_equality_constraints(
    int num_envs, int nv, int nbody, int njnt,
    int nequality,
    const int*    __restrict__ eq_type,
    const int*    __restrict__ eq_body1,
    const int*    __restrict__ eq_body2,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis_d,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const double* __restrict__ M_inv_diag,
    const double* __restrict__ body_invweight0_d,
    double timestep,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ qvel_d,
    const int*    __restrict__ body_chain_start,
    const int*    __restrict__ body_chain_len,
    const int*    __restrict__ body_chain_jnt_idx,
    const int*    __restrict__ body_chain_jnt_body,
    const int*    __restrict__ body_chain_parent,
    double*       __restrict__ constraint_J,
    double*       __restrict__ constraint_rhs,
    double*       __restrict__ constraint_D,
    double*       __restrict__ constraint_lo,
    double*       __restrict__ constraint_hi,
    int*          __restrict__ constraint_friction_parent,
    double*       __restrict__ constraint_friction_mu,
    int*          __restrict__ nconstraint)
{
    int ei = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (ei >= nequality || env >= num_envs) return;
    if (eq_type[ei] != EQUALITY_WELD) return;

    double sr0 = 0.02, sr1 = 1.0;
    double si0 = 0.9, si1 = 0.95, si2 = 0.001, si3 = 0.5, si4 = 2.0;
    const double* ev = qvel_d + env * nv;
    const double* Minv = M_inv_diag + env * MAX_DOF;

    int b1 = eq_body1[ei];
    int b2 = eq_body2[ei];
    int b1_3 = env * nbody * 3 + b1 * 3;
    int b2_3 = env * nbody * 3 + b2 * 3;

    double dx = xpos_d[b2_3] - xpos_d[b1_3];
    double dy = xpos_d[b2_3+1] - xpos_d[b1_3+1];
    double dz = xpos_d[b2_3+2] - xpos_d[b1_3+2];

    double dirs[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    double errs[3] = {dx, dy, dz};

    for (int axis = 0; axis < 3; axis++) {
        int slot = atomicAdd(&nconstraint[env], 1);
        if (slot >= MAX_CONSTRAINTS) return;

        double* J_row = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)slot * nv;
        for (int v = 0; v < nv; v++) J_row[v] = 0.0;

        double contact_pos[3] = {
            0.5 * (xpos_d[b1_3] + xpos_d[b2_3]),
            0.5 * (xpos_d[b1_3+1] + xpos_d[b2_3+1]),
            0.5 * (xpos_d[b1_3+2] + xpos_d[b2_3+2])
        };

        fill_chain_jacobian_precomputed_d(J_row, b1, -1.0, dirs[axis], contact_pos,
            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
            jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
        fill_chain_jacobian_precomputed_d(J_row, b2, +1.0, dirs[axis], contact_pos,
            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body, body_chain_parent,
            jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);

        double invweight;
        if (body_invweight0_d) {
            invweight = body_invweight0_d[b1 * 2] + body_invweight0_d[b2 * 2];
        } else {
            invweight = 0.0;
            for (int v = 0; v < nv; v++) invweight += J_row[v] * J_row[v] * Minv[v];
        }
        invweight = fmax(invweight, 1e-15);

        double Jv = 0;
        for (int v = 0; v < nv; v++) Jv += J_row[v] * ev[v];

        double D_weld, aref;
        mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                         -errs[axis], -errs[axis], invweight, Jv, timestep, &D_weld, &aref);

        size_t idx = (size_t)env * MAX_CONSTRAINTS + slot;
        constraint_rhs[idx] = aref;
        constraint_D[idx] = D_weld;
        constraint_lo[idx] = -1e20;
        constraint_hi[idx] = 1e20;
        constraint_friction_parent[idx] = -1;
        constraint_friction_mu[idx] = 0.0;
    }

    int b1_4 = env * nbody * 4 + b1 * 4;
    int b2_4 = env * nbody * 4 + b2 * 4;
    double q1w = xquat_d[b1_4], q1x = xquat_d[b1_4+1], q1y = xquat_d[b1_4+2], q1z = xquat_d[b1_4+3];
    double q2w = xquat_d[b2_4], q2x = xquat_d[b2_4+1], q2y = xquat_d[b2_4+2], q2z = xquat_d[b2_4+3];

    double rw = q1w*q2w + q1x*q2x + q1y*q2y + q1z*q2z;
    double rx = q1w*q2x - q1x*q2w - q1y*q2z + q1z*q2y;
    double ry = q1w*q2y + q1x*q2z - q1y*q2w - q1z*q2x;
    double rz = q1w*q2z - q1x*q2y + q1y*q2x - q1z*q2w;
    if (rw < 0) { rw = -rw; rx = -rx; ry = -ry; rz = -rz; }
    double ang_err[3] = {2.0 * rx, 2.0 * ry, 2.0 * rz};

    for (int axis = 0; axis < 3; axis++) {
        int slot = atomicAdd(&nconstraint[env], 1);
        if (slot >= MAX_CONSTRAINTS) return;

        double* J_row = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv + (size_t)slot * nv;
        for (int v = 0; v < nv; v++) J_row[v] = 0.0;

        // Walk b1 chain using precomputed data (rotational Jacobian)
        {
            int start = body_chain_start[b1];
            int len   = body_chain_len[b1];
            for (int i = 0; i < len; i++) {
                int ji = body_chain_jnt_idx[start + i];
                int jbody = body_chain_jnt_body[start + i];
                int par = body_chain_parent[start + i];
                int jtype = jnt_type[ji]; int dof = jnt_dof_adr[ji];
                if (jtype == JOINT_HINGE) {
                    double ax[3];
                    if (jnt_xaxis_d && ji < njnt) {
                        int ax_idx = env * njnt * 3 + ji * 3;
                        ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
                    } else {
                        local_to_world_axis_d(jnt_axis_d, ji, xquat_d, env, nbody, par, ax);
                    }
                    J_row[dof] += -ax[axis];
                } else if (jtype == JOINT_FREE) {
                    int bq4 = env * nbody * 4 + jbody * 4;
                    double qw = xquat_d[bq4], qx = xquat_d[bq4+1], qy = xquat_d[bq4+2], qz = xquat_d[bq4+3];
                    double R_row[3];
                    if (axis == 0) {
                        R_row[0] = 1-2.0*(qy*qy+qz*qz); R_row[1] = 2.0*(qx*qy-qw*qz); R_row[2] = 2.0*(qx*qz+qw*qy);
                    } else if (axis == 1) {
                        R_row[0] = 2.0*(qx*qy+qw*qz); R_row[1] = 1-2.0*(qx*qx+qz*qz); R_row[2] = 2.0*(qy*qz-qw*qx);
                    } else {
                        R_row[0] = 2.0*(qx*qz-qw*qy); R_row[1] = 2.0*(qy*qz+qw*qx); R_row[2] = 1-2.0*(qx*qx+qy*qy);
                    }
                    J_row[dof+3] += -R_row[0]; J_row[dof+4] += -R_row[1]; J_row[dof+5] += -R_row[2];
                }
            }
        }
        // Walk b2 chain using precomputed data (rotational Jacobian)
        {
            int start = body_chain_start[b2];
            int len   = body_chain_len[b2];
            for (int i = 0; i < len; i++) {
                int ji = body_chain_jnt_idx[start + i];
                int jbody = body_chain_jnt_body[start + i];
                int par = body_chain_parent[start + i];
                int jtype = jnt_type[ji]; int dof = jnt_dof_adr[ji];
                if (jtype == JOINT_HINGE) {
                    double ax[3];
                    if (jnt_xaxis_d && ji < njnt) {
                        int ax_idx = env * njnt * 3 + ji * 3;
                        ax[0] = jnt_xaxis_d[ax_idx]; ax[1] = jnt_xaxis_d[ax_idx+1]; ax[2] = jnt_xaxis_d[ax_idx+2];
                    } else {
                        local_to_world_axis_d(jnt_axis_d, ji, xquat_d, env, nbody, par, ax);
                    }
                    J_row[dof] += ax[axis];
                } else if (jtype == JOINT_FREE) {
                    int bq4 = env * nbody * 4 + jbody * 4;
                    double qw = xquat_d[bq4], qx = xquat_d[bq4+1], qy = xquat_d[bq4+2], qz = xquat_d[bq4+3];
                    double R_row[3];
                    if (axis == 0) {
                        R_row[0] = 1-2.0*(qy*qy+qz*qz); R_row[1] = 2.0*(qx*qy-qw*qz); R_row[2] = 2.0*(qx*qz+qw*qy);
                    } else if (axis == 1) {
                        R_row[0] = 2.0*(qx*qy+qw*qz); R_row[1] = 1-2.0*(qx*qx+qz*qz); R_row[2] = 2.0*(qy*qz-qw*qx);
                    } else {
                        R_row[0] = 2.0*(qx*qz-qw*qy); R_row[1] = 2.0*(qy*qz+qw*qx); R_row[2] = 1-2.0*(qx*qx+qy*qy);
                    }
                    J_row[dof+3] += R_row[0]; J_row[dof+4] += R_row[1]; J_row[dof+5] += R_row[2];
                }
            }
        }

        double invweight_rot;
        if (body_invweight0_d) {
            invweight_rot = body_invweight0_d[b1 * 2 + 1] + body_invweight0_d[b2 * 2 + 1];
        } else {
            invweight_rot = 0.0;
            for (int v = 0; v < nv; v++) invweight_rot += J_row[v] * J_row[v] * Minv[v];
        }
        invweight_rot = fmax(invweight_rot, 1e-15);

        double Jv = 0;
        for (int v = 0; v < nv; v++) Jv += J_row[v] * ev[v];

        double D_rot, aref_rot;
        mujoco_impedance(sr0, sr1, si0, si1, si2, si3, si4,
                         -ang_err[axis], -ang_err[axis], invweight_rot, Jv, timestep, &D_rot, &aref_rot);

        size_t idx_rot = (size_t)env * MAX_CONSTRAINTS + slot;
        constraint_rhs[idx_rot] = aref_rot;
        constraint_D[idx_rot] = D_rot;
        constraint_lo[idx_rot] = -1e20;
        constraint_hi[idx_rot] = 1e20;
        constraint_friction_parent[idx_rot] = -1;
        constraint_friction_mu[idx_rot] = 0.0;
    }
}

// ── Extract diag(M)^{-1} from full mass matrix (1 thread per DOF per env) ───

HIPGL_GLOBAL void kernel_extract_M_diag_inv(
    int num_envs, int nv,
    const double* __restrict__ solver_M,
    double*       __restrict__ M_inv_diag)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (v >= nv || env >= num_envs) return;

    double Mvv = solver_M[(size_t)env * KERNEL_MAXNV * KERNEL_MAXNV + (size_t)v * KERNEL_MAXNV + v];
    M_inv_diag[env * MAX_DOF + v] = (Mvv > 1e-10) ? (1.0 / Mvv) : 0.0;
}

// ── Phase 7: Parallel M_inv_diag build (1 thread per joint per env) ─────────

HIPGL_GLOBAL void kernel_build_M_inv_diag(
    int num_envs, int nv, int njnt, int nbody,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const int*   __restrict__ jnt_dof_adr,
    const float* __restrict__ body_inv_mass,
    const float* __restrict__ body_inv_inertia,
    double*      __restrict__ M_inv_diag)
{
    int ji = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (ji >= njnt || env >= num_envs) return;

    int jtype = jnt_type[ji];
    int body = jnt_body[ji];
    int da = jnt_dof_adr[ji];
    int eidx = env * nbody + body;
    double* M = M_inv_diag + env * MAX_DOF;

    if (jtype == JOINT_FREE) {
        double inv_m = (double)body_inv_mass[eidx];
        for (int k = 0; k < 3 && da+k < MAX_DOF; k++) M[da+k] = inv_m;
        int iidx = env * nbody * 3 + body * 3;
        for (int k = 0; k < 3 && da+3+k < MAX_DOF; k++) M[da+3+k] = (double)body_inv_inertia[iidx+k];
    } else if (jtype == JOINT_BALL) {
        int iidx = env * nbody * 3 + body * 3;
        for (int k = 0; k < 3 && da+k < MAX_DOF; k++) M[da+k] = (double)body_inv_inertia[iidx+k];
    } else if (jtype == JOINT_HINGE) {
        int iidx = env * nbody * 3 + body * 3;
        if (da < MAX_DOF) M[da] = (double)body_inv_inertia[iidx];
    } else if (jtype == JOINT_SLIDE) {
        if (da < MAX_DOF) M[da] = (double)body_inv_mass[eidx];
    }
}

// ── Phase 3: Parallel mass matrix build (1 thread per (i,j) entry per env) ──
// Computes M[i][j] = sum over bodies of (Si^T * Ic * Sj) where Si, Sj are
// motion subspace vectors for the i-th and j-th DOFs.

template <typename T>
struct BmView {
    const T* p; int base;
    HIPGL_DEVICE T operator[](int i) const { return p[i - base]; }
};

template <bool BM_LDS>
HIPGL_GLOBAL void __launch_bounds__(256) kernel_build_mass_matrix_t(
    int num_envs, int nv, int njnt, int nbody,
    const int*    __restrict__ jnt_type_g,
    const int*    __restrict__ jnt_body_g,
    const int*    __restrict__ jnt_dof_adr_g,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ body_mass_g,
    const double* __restrict__ body_inertia_g,
    const double* __restrict__ body_ipos_g,
    const double* __restrict__ body_iquat_g,
    const int*    __restrict__ body_parent_g,
    const double* __restrict__ xpos_g,
    const double* __restrict__ xquat_g,
    const double* __restrict__ jnt_xaxis_d,
    const double* __restrict__ jnt_damping,
    const double* __restrict__ jnt_armature,
    float timestep,
    bool enable_damping,
    double*      __restrict__ M_out,
    double*      __restrict__ M_inv_diag_out)
{
    constexpr int BMX = 64;   // BM_LDS requires nbody <= BMX and njnt <= BMX
    __shared__ double s_xq[BM_LDS ? BMX * 4 : 1];
    __shared__ double s_xp[BM_LDS ? BMX * 3 : 1];
    __shared__ double s_bm[BM_LDS ? BMX : 1];
    __shared__ double s_bin[BM_LDS ? BMX * 3 : 1];
    __shared__ double s_bip[BM_LDS ? BMX * 3 : 1];
    __shared__ double s_biq[BM_LDS ? BMX * 4 : 1];
    __shared__ int    s_jt[BM_LDS ? BMX : 1];
    __shared__ int    s_jb[BM_LDS ? BMX : 1];
    __shared__ int    s_jd[BM_LDS ? BMX : 1];
    __shared__ int    s_bp[BM_LDS ? BMX : 1];
    if (BM_LDS) {
        const int env0 = blockIdx.y;
        if (env0 >= num_envs) return;
        for (int e = threadIdx.x; e < nbody * 4; e += blockDim.x) {
            s_xq[e]  = xquat_g[(size_t)env0 * nbody * 4 + e];
            s_biq[e] = body_iquat_g[e];
        }
        for (int e = threadIdx.x; e < nbody * 3; e += blockDim.x) {
            s_xp[e]  = xpos_g[(size_t)env0 * nbody * 3 + e];
            s_bin[e] = body_inertia_g[e];
            s_bip[e] = body_ipos_g[e];
        }
        for (int e = threadIdx.x; e < nbody; e += blockDim.x) {
            s_bm[e] = body_mass_g[e];
            s_bp[e] = body_parent_g[e];
        }
        for (int e = threadIdx.x; e < njnt; e += blockDim.x) {
            s_jt[e] = jnt_type_g[e];
            s_jb[e] = jnt_body_g[e];
            s_jd[e] = jnt_dof_adr_g[e];
        }
        __syncthreads();
    }
    const int envb = BM_LDS ? (int)blockIdx.y : 0;
    const BmView<double> xquat{BM_LDS ? s_xq : xquat_g, BM_LDS ? envb * nbody * 4 : 0};
    const BmView<double> xpos{BM_LDS ? s_xp : xpos_g, BM_LDS ? envb * nbody * 3 : 0};
    const BmView<double> body_mass{BM_LDS ? s_bm : body_mass_g, 0};
    const BmView<double> body_inertia{BM_LDS ? s_bin : body_inertia_g, 0};
    const BmView<double> body_ipos{BM_LDS ? s_bip : body_ipos_g, 0};
    const BmView<double> body_iquat{BM_LDS ? s_biq : body_iquat_g, 0};
    const BmView<int> jnt_type{BM_LDS ? s_jt : jnt_type_g, 0};
    const BmView<int> jnt_body{BM_LDS ? s_jb : jnt_body_g, 0};
    const BmView<int> jnt_dof_adr{BM_LDS ? s_jd : jnt_dof_adr_g, 0};
    const BmView<int> body_parent{BM_LDS ? s_bp : body_parent_g, 0};
    __shared__ unsigned long long s_anc[64];
    const bool bm_mask = (nbody <= 64);
    if (bm_mask) {
        for (int b = threadIdx.x; b < nbody; b += blockDim.x) {
            unsigned long long a = 0ull;
            int cur = b;
            int max_walk = 32;
            while (cur > 0 && max_walk-- > 0) {
                a |= 1ull << cur;
                cur = body_parent[cur];
            }
            s_anc[b] = a;
        }
        __syncthreads();
    }

    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (env >= num_envs) return;

    int i = flat / nv;
    int j = flat % nv;
    if (i >= nv || j > i) return;

    double* M_env = M_out + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;

    int ji_i = -1, ji_j = -1;
    int dof_offset_i = 0, dof_offset_j = 0;
    for (int ji = 0; ji < njnt; ji++) {
        int da = jnt_dof_adr[ji];
        int jt = jnt_type[ji];
        int ndof = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
        if (i >= da && i < da + ndof) { ji_i = ji; dof_offset_i = i - da; }
        if (j >= da && j < da + ndof) { ji_j = ji; dof_offset_j = j - da; }
    }
    if (ji_i < 0 || ji_j < 0) {
        M_env[i * KERNEL_MAXNV + j] = (i == j) ? 1e-8 : 0.0;
        if (i != j) M_env[j * KERNEL_MAXNV + i] = 0.0;
        if (i == j && M_inv_diag_out)   // == kernel_extract_M_diag_inv on 1e-8
            M_inv_diag_out[env * MAX_DOF + i] = (1e-8 > 1e-10) ? (1.0 / 1e-8) : 0.0;
        return;
    }

    auto world_axis = [&](int ji, double* aw) {
        // Use pre-computed jnt_xaxis_d which uses the correct pre-joint frame
        // (matching MuJoCo's cdof computation for the CRBA).
        if (jnt_xaxis_d) {
            int ax_idx = env * njnt * 3 + ji * 3;
            aw[0] = jnt_xaxis_d[ax_idx];
            aw[1] = jnt_xaxis_d[ax_idx + 1];
            aw[2] = jnt_xaxis_d[ax_idx + 2];
        } else {
            int jbody = jnt_body[ji];
            int bq4 = env * nbody * 4 + jbody * 4;
            double qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
            double lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
            double tw_x = 2.0*(qy*laz - qz*lay), tw_y = 2.0*(qz*lax - qx*laz), tw_z = 2.0*(qx*lay - qy*lax);
            aw[0] = lax + qw*tw_x + (qy*tw_z - qz*tw_y);
            aw[1] = lay + qw*tw_y + (qz*tw_x - qx*tw_z);
            aw[2] = laz + qw*tw_z + (qx*tw_y - qy*tw_x);
        }
    };

    double ai[3], aj[3];
    world_axis(ji_i, ai);
    world_axis(ji_j, aj);

    double pi[3], pj[3];
    int bi = jnt_body[ji_i], bj = jnt_body[ji_j];
    int bi3 = env * nbody * 3 + bi * 3;
    int bj3 = env * nbody * 3 + bj * 3;
    pi[0] = xpos[bi3]; pi[1] = xpos[bi3+1]; pi[2] = xpos[bi3+2];
    pj[0] = xpos[bj3]; pj[1] = xpos[bj3+1]; pj[2] = xpos[bj3+2];

    double val = 0.0;

    for (int bid = 1; bid < nbody; bid++) {
        double m = (double)body_mass[bid];
        if (m < 1e-12) continue;

        bool in_i = false, in_j = false;
        if (bm_mask) {
            const unsigned long long a = s_anc[bid];
            in_i = (bi > 0) && ((a >> bi) & 1ull);
            in_j = (bj > 0) && ((a >> bj) & 1ull);
        } else {
            int cur = bid;
            int max_walk = 32;
            while (cur > 0 && max_walk-- > 0) {
                if (cur == bi) in_i = true;
                if (cur == bj) in_j = true;
                cur = body_parent[cur];
            }
        }
        if (!in_i || !in_j) continue;

        double ipx = body_ipos[bid*3], ipy = body_ipos[bid*3+1], ipz = body_ipos[bid*3+2];
        int bq4 = env * nbody * 4 + bid * 4;
        double bqw = xquat[bq4], bqx = xquat[bq4+1], bqy = xquat[bq4+2], bqz = xquat[bq4+3];
        double tw_x = 2.0*(bqy*ipz - bqz*ipy), tw_y = 2.0*(bqz*ipx - bqx*ipz), tw_z = 2.0*(bqx*ipy - bqy*ipx);
        double ri_x = ipx + bqw*tw_x + (bqy*tw_z - bqz*tw_y);
        double ri_y = ipy + bqw*tw_y + (bqz*tw_x - bqx*tw_z);
        double ri_z = ipz + bqw*tw_z + (bqx*tw_y - bqy*tw_x);
        int b3 = env * nbody * 3 + bid * 3;
        double com_x = (double)xpos[b3] + ri_x, com_y = (double)xpos[b3+1] + ri_y, com_z = (double)xpos[b3+2] + ri_z;

        double Ix = body_inertia[bid*3], Iy = body_inertia[bid*3+1], Iz = body_inertia[bid*3+2];

        double iqw = body_iquat[bid*4], iqx = body_iquat[bid*4+1], iqy = body_iquat[bid*4+2], iqz = body_iquat[bid*4+3];
        double cqw = bqw*iqw - bqx*iqx - bqy*iqy - bqz*iqz;
        double cqx = bqw*iqx + bqx*iqw + bqy*iqz - bqz*iqy;
        double cqy = bqw*iqy - bqx*iqz + bqy*iqw + bqz*iqx;
        double cqz = bqw*iqz + bqx*iqy - bqy*iqx + bqz*iqw;

        double xx=cqx*cqx, yy=cqy*cqy, zz=cqz*cqz;
        double xy=cqx*cqy, xz=cqx*cqz, yz=cqy*cqz;
        double wx=cqw*cqx, wy=cqw*cqy, wz=cqw*cqz;
        double R00=1-2*(yy+zz), R01=2*(xy-wz), R02=2*(xz+wy);
        double R10=2*(xy+wz), R11=1-2*(xx+zz), R12=2*(yz-wx);
        double R20=2*(xz-wy), R21=2*(yz+wx), R22=1-2*(xx+yy);

        int jt_i = jnt_type[ji_i], jt_j = jnt_type[ji_j];

        if (jt_i == JOINT_HINGE && jt_j == JOINT_HINGE) {
            double Iw_aj[3] = {
                (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*aj[0] + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[1] + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[2],
                (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[0] + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*aj[1] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[2],
                (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[0] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[1] + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*aj[2]
            };
            double inertia_term = ai[0]*Iw_aj[0] + ai[1]*Iw_aj[1] + ai[2]*Iw_aj[2];

            double rki[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};
            double rkj[3] = {com_x-pj[0], com_y-pj[1], com_z-pj[2]};
            double ci[3] = {ai[1]*rki[2]-ai[2]*rki[1], ai[2]*rki[0]-ai[0]*rki[2], ai[0]*rki[1]-ai[1]*rki[0]};
            double cj[3] = {aj[1]*rkj[2]-aj[2]*rkj[1], aj[2]*rkj[0]-aj[0]*rkj[2], aj[0]*rkj[1]-aj[1]*rkj[0]};
            double mass_term = m * (ci[0]*cj[0] + ci[1]*cj[1] + ci[2]*cj[2]);

            val += inertia_term + mass_term;
        }
        else if (jt_i == JOINT_FREE && jt_j == JOINT_FREE && ji_i == ji_j) {
            // FREE-FREE same joint: linear DOFs world-frame, angular DOFs body-frame.
            // Use world-frame computation with body rotation columns as angular axes,
            // then results naturally correspond to body-frame angular DOFs.
            int freeq4 = env * nbody * 4 + bi * 4;
            double fqw = xquat[freeq4], fqx = xquat[freeq4+1], fqy = xquat[freeq4+2], fqz = xquat[freeq4+3];
            // R_bi = rotation matrix of free body (columns = body-frame axes in world)
            double Fr00=1-2*(fqy*fqy+fqz*fqz), Fr01=2*(fqx*fqy-fqw*fqz), Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz), Fr11=1-2*(fqx*fqx+fqz*fqz), Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy), Fr21=2*(fqy*fqz+fqw*fqx), Fr22=1-2*(fqx*fqx+fqy*fqy);

            double rk[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};

            if (dof_offset_i < 3 && dof_offset_j < 3) {
                // lin-lin: M = m * δ_{ij}
                val += (dof_offset_i == dof_offset_j) ? m : 0.0;
            } else if (dof_offset_i >= 3 && dof_offset_j >= 3) {
                // ang-ang: use world-frame body rotation columns as axes
                int ai_off = dof_offset_i - 3, aj_off = dof_offset_j - 3;
                double wa[3], wb[3];
                if (ai_off == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (ai_off == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else                  { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }
                if (aj_off == 0)      { wb[0]=Fr00; wb[1]=Fr10; wb[2]=Fr20; }
                else if (aj_off == 1) { wb[0]=Fr01; wb[1]=Fr11; wb[2]=Fr21; }
                else                  { wb[0]=Fr02; wb[1]=Fr12; wb[2]=Fr22; }

                // Inertia term: wa^T * I_w * wb
                double Iw_wb[3] = {
                    (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wb[0] + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[1] + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[2],
                    (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[0] + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wb[1] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[2],
                    (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[0] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[1] + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wb[2]
                };
                double inertia_term = wa[0]*Iw_wb[0] + wa[1]*Iw_wb[1] + wa[2]*Iw_wb[2];

                // Mass term: m * (wa × rk) · (wb × rk)
                double ca[3] = {wa[1]*rk[2]-wa[2]*rk[1], wa[2]*rk[0]-wa[0]*rk[2], wa[0]*rk[1]-wa[1]*rk[0]};
                double cb[3] = {wb[1]*rk[2]-wb[2]*rk[1], wb[2]*rk[0]-wb[0]*rk[2], wb[0]*rk[1]-wb[1]*rk[0]};
                double mass_term = m * (ca[0]*cb[0] + ca[1]*cb[1] + ca[2]*cb[2]);

                val += inertia_term + mass_term;
            } else if (dof_offset_i >= 3 && dof_offset_j < 3) {
                // ang-lin: wa world axis, e_b world translation
                int a = dof_offset_i - 3;
                double wa[3];
                if (a == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (a == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else             { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }
                // M = m * (wa × rk) · e_b = m * (wa × rk)[b]
                double ca[3] = {wa[1]*rk[2]-wa[2]*rk[1], wa[2]*rk[0]-wa[0]*rk[2], wa[0]*rk[1]-wa[1]*rk[0]};
                val += m * ca[dof_offset_j];
            }
        }
        else if (jt_i == JOINT_FREE && jt_j == JOINT_HINGE) {
            // Cross-coupling between free joint DOFs and downstream hinge
            // Free linear DOFs are world-frame, angular DOFs are body-frame
            // MuJoCo computes cdof for free angular using world-frame body axes

            // Get free body rotation matrix columns for angular DOFs
            int freeq4 = env * nbody * 4 + bi * 4;
            double fqw = xquat[freeq4], fqx = xquat[freeq4+1], fqy = xquat[freeq4+2], fqz = xquat[freeq4+3];
            // R = rotation matrix from body quat (columns are body-frame axes in world)
            double Fr00=1-2*(fqy*fqy+fqz*fqz), Fr01=2*(fqx*fqy-fqw*fqz), Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz), Fr11=1-2*(fqx*fqx+fqz*fqz), Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy), Fr21=2*(fqy*fqz+fqw*fqx), Fr22=1-2*(fqx*fqx+fqy*fqy);

            double rkh[3] = {com_x-pj[0], com_y-pj[1], com_z-pj[2]};
            double ch[3] = {aj[1]*rkh[2]-aj[2]*rkh[1], aj[2]*rkh[0]-aj[0]*rkh[2], aj[0]*rkh[1]-aj[1]*rkh[0]};

            if (dof_offset_i < 3) {
                // Free linear DOF b: cdof = [0; e_b]
                // M = m * e_b · (a_h × rk_h)
                val += m * ch[dof_offset_i];
            } else {
                // Free angular DOF a (body-frame): world axis = R column a
                int a = dof_offset_i - 3;
                double wa[3];
                if (a == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (a == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else             { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }

                // Inertia term: wa^T * I_w * a_h
                double Iw_ah[3] = {
                    (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*aj[0] + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[1] + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[2],
                    (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[0] + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*aj[1] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[2],
                    (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[0] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[1] + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*aj[2]
                };
                double inertia_term = wa[0]*Iw_ah[0] + wa[1]*Iw_ah[1] + wa[2]*Iw_ah[2];

                // Mass term: m * (wa × rk_f) · (a_h × rk_h)
                double rkf[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};
                double cf[3] = {wa[1]*rkf[2]-wa[2]*rkf[1], wa[2]*rkf[0]-wa[0]*rkf[2], wa[0]*rkf[1]-wa[1]*rkf[0]};
                double mass_term = m * (cf[0]*ch[0] + cf[1]*ch[1] + cf[2]*ch[2]);

                val += inertia_term + mass_term;
            }
        }
        else if (jt_i == JOINT_HINGE && jt_j == JOINT_FREE) {
            // Symmetric: HINGE-FREE coupling
            int freeq4 = env * nbody * 4 + bj * 4;
            double fqw = xquat[freeq4], fqx = xquat[freeq4+1], fqy = xquat[freeq4+2], fqz = xquat[freeq4+3];
            double Fr00=1-2*(fqy*fqy+fqz*fqz), Fr01=2*(fqx*fqy-fqw*fqz), Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz), Fr11=1-2*(fqx*fqx+fqz*fqz), Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy), Fr21=2*(fqy*fqz+fqw*fqx), Fr22=1-2*(fqx*fqx+fqy*fqy);

            double rki[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};
            double ci[3] = {ai[1]*rki[2]-ai[2]*rki[1], ai[2]*rki[0]-ai[0]*rki[2], ai[0]*rki[1]-ai[1]*rki[0]};

            if (dof_offset_j < 3) {
                // Free linear DOF b: M = m * (a_i × rk_i) · e_b
                val += m * ci[dof_offset_j];
            } else {
                int a = dof_offset_j - 3;
                double wa[3];
                if (a == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (a == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else             { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }

                double Iw_wa[3] = {
                    (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wa[0] + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wa[1] + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wa[2],
                    (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wa[0] + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wa[1] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wa[2],
                    (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wa[0] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wa[1] + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wa[2]
                };
                double inertia_term = ai[0]*Iw_wa[0] + ai[1]*Iw_wa[1] + ai[2]*Iw_wa[2];

                double rkf[3] = {com_x-pj[0], com_y-pj[1], com_z-pj[2]};
                double cf[3] = {wa[1]*rkf[2]-wa[2]*rkf[1], wa[2]*rkf[0]-wa[0]*rkf[2], wa[0]*rkf[1]-wa[1]*rkf[0]};
                double mass_term = m * (ci[0]*cf[0] + ci[1]*cf[1] + ci[2]*cf[2]);

                val += inertia_term + mass_term;
            }
        }
        else if (jt_i == JOINT_FREE && jt_j == JOINT_SLIDE) {
            // Free DOF coupling with slide joint
            double rkf[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};

            int freeq4 = env * nbody * 4 + bi * 4;
            double fqw = xquat[freeq4], fqx = xquat[freeq4+1], fqy = xquat[freeq4+2], fqz = xquat[freeq4+3];
            double Fr00=1-2*(fqy*fqy+fqz*fqz), Fr01=2*(fqx*fqy-fqw*fqz), Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz), Fr11=1-2*(fqx*fqx+fqz*fqz), Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy), Fr21=2*(fqy*fqz+fqw*fqx), Fr22=1-2*(fqx*fqx+fqy*fqy);

            if (dof_offset_i < 3) {
                // Free linear × slide: M = m * e_b · a_slide
                val += m * aj[dof_offset_i];
            } else {
                // Free angular × slide: M = m * (wa × rk_f) · a_slide
                int a = dof_offset_i - 3;
                double wa[3];
                if (a == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (a == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else             { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }
                double cf[3] = {wa[1]*rkf[2]-wa[2]*rkf[1], wa[2]*rkf[0]-wa[0]*rkf[2], wa[0]*rkf[1]-wa[1]*rkf[0]};
                val += m * (cf[0]*aj[0] + cf[1]*aj[1] + cf[2]*aj[2]);
            }
        }
        else if (jt_i == JOINT_SLIDE && jt_j == JOINT_FREE) {
            // Symmetric: SLIDE-FREE
            double rkf[3] = {com_x-pj[0], com_y-pj[1], com_z-pj[2]};

            int freeq4 = env * nbody * 4 + bj * 4;
            double fqw = xquat[freeq4], fqx = xquat[freeq4+1], fqy = xquat[freeq4+2], fqz = xquat[freeq4+3];
            double Fr00=1-2*(fqy*fqy+fqz*fqz), Fr01=2*(fqx*fqy-fqw*fqz), Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz), Fr11=1-2*(fqx*fqx+fqz*fqz), Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy), Fr21=2*(fqy*fqz+fqw*fqx), Fr22=1-2*(fqx*fqx+fqy*fqy);

            if (dof_offset_j < 3) {
                // Slide × free linear: M = m * a_slide · e_b
                val += m * ai[dof_offset_j];
            } else {
                int a = dof_offset_j - 3;
                double wa[3];
                if (a == 0)      { wa[0]=Fr00; wa[1]=Fr10; wa[2]=Fr20; }
                else if (a == 1) { wa[0]=Fr01; wa[1]=Fr11; wa[2]=Fr21; }
                else             { wa[0]=Fr02; wa[1]=Fr12; wa[2]=Fr22; }
                double cf[3] = {wa[1]*rkf[2]-wa[2]*rkf[1], wa[2]*rkf[0]-wa[0]*rkf[2], wa[0]*rkf[1]-wa[1]*rkf[0]};
                val += m * (ai[0]*cf[0] + ai[1]*cf[1] + ai[2]*cf[2]);
            }
        }
        else if (jt_i == JOINT_FREE && jt_j == JOINT_FREE && ji_i != ji_j) {
            // Cross-coupling between different free joints (rare but handle for completeness)
            // Treat each DOF like its world-frame equivalent
            int freeq4_i = env * nbody * 4 + bi * 4;
            double fqwi = xquat[freeq4_i], fqxi = xquat[freeq4_i+1], fqyi = xquat[freeq4_i+2], fqzi = xquat[freeq4_i+3];
            double Fri00=1-2*(fqyi*fqyi+fqzi*fqzi), Fri01=2*(fqxi*fqyi-fqwi*fqzi), Fri02=2*(fqxi*fqzi+fqwi*fqyi);
            double Fri10=2*(fqxi*fqyi+fqwi*fqzi), Fri11=1-2*(fqxi*fqxi+fqzi*fqzi), Fri12=2*(fqyi*fqzi-fqwi*fqxi);
            double Fri20=2*(fqxi*fqzi-fqwi*fqyi), Fri21=2*(fqyi*fqzi+fqwi*fqxi), Fri22=1-2*(fqxi*fqxi+fqyi*fqyi);

            int freeq4_j = env * nbody * 4 + bj * 4;
            double fqwj = xquat[freeq4_j], fqxj = xquat[freeq4_j+1], fqyj = xquat[freeq4_j+2], fqzj = xquat[freeq4_j+3];
            double Frj00=1-2*(fqyj*fqyj+fqzj*fqzj), Frj01=2*(fqxj*fqyj-fqwj*fqzj), Frj02=2*(fqxj*fqzj+fqwj*fqyj);
            double Frj10=2*(fqxj*fqyj+fqwj*fqzj), Frj11=1-2*(fqxj*fqxj+fqzj*fqzj), Frj12=2*(fqyj*fqzj-fqwj*fqxj);
            double Frj20=2*(fqxj*fqzj-fqwj*fqyj), Frj21=2*(fqyj*fqzj+fqwj*fqxj), Frj22=1-2*(fqxj*fqxj+fqyj*fqyj);

            // Get world-frame axes for both DOFs
            auto get_free_cdof = [&](int dof_off, int body_id, double* pos_body,
                                     double F00, double F01, double F02,
                                     double F10, double F11, double F12,
                                     double F20, double F21, double F22,
                                     double* w_out, double* v_out) {
                if (dof_off < 3) {
                    w_out[0] = w_out[1] = w_out[2] = 0.0;
                    v_out[0] = v_out[1] = v_out[2] = 0.0;
                    v_out[dof_off] = 1.0;
                } else {
                    int a = dof_off - 3;
                    if (a == 0)      { w_out[0]=F00; w_out[1]=F10; w_out[2]=F20; }
                    else if (a == 1) { w_out[0]=F01; w_out[1]=F11; w_out[2]=F21; }
                    else             { w_out[0]=F02; w_out[1]=F12; w_out[2]=F22; }
                    double rk[3] = {com_x-pos_body[0], com_y-pos_body[1], com_z-pos_body[2]};
                    v_out[0] = w_out[1]*rk[2] - w_out[2]*rk[1];
                    v_out[1] = w_out[2]*rk[0] - w_out[0]*rk[2];
                    v_out[2] = w_out[0]*rk[1] - w_out[1]*rk[0];
                }
            };

            double wi[3], vi[3], wj[3], vj[3];
            get_free_cdof(dof_offset_i, bi, pi, Fri00,Fri01,Fri02, Fri10,Fri11,Fri12, Fri20,Fri21,Fri22, wi, vi);
            get_free_cdof(dof_offset_j, bj, pj, Frj00,Frj01,Frj02, Frj10,Frj11,Frj12, Frj20,Frj21,Frj22, wj, vj);

            // M = wi^T * I_w * wj + m * vi · vj
            double Iw_wj[3] = {
                (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wj[0] + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wj[1] + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wj[2],
                (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wj[0] + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wj[1] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wj[2],
                (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wj[0] + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wj[1] + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wj[2]
            };
            val += wi[0]*Iw_wj[0] + wi[1]*Iw_wj[1] + wi[2]*Iw_wj[2];
            val += m * (vi[0]*vj[0] + vi[1]*vj[1] + vi[2]*vj[2]);
        }
        else if (jt_i == JOINT_HINGE && jt_j == JOINT_SLIDE) {
            double rki[3] = {com_x-pi[0], com_y-pi[1], com_z-pi[2]};
            double ci[3] = {ai[1]*rki[2]-ai[2]*rki[1], ai[2]*rki[0]-ai[0]*rki[2], ai[0]*rki[1]-ai[1]*rki[0]};
            val += m * (ci[0]*aj[0] + ci[1]*aj[1] + ci[2]*aj[2]);
        }
        else if (jt_i == JOINT_SLIDE && jt_j == JOINT_HINGE) {
            double rkj[3] = {com_x-pj[0], com_y-pj[1], com_z-pj[2]};
            double cj[3] = {aj[1]*rkj[2]-aj[2]*rkj[1], aj[2]*rkj[0]-aj[0]*rkj[2], aj[0]*rkj[1]-aj[1]*rkj[0]};
            val += m * (ai[0]*cj[0] + ai[1]*cj[1] + ai[2]*cj[2]);
        }
        else if (jt_i == JOINT_SLIDE && jt_j == JOINT_SLIDE) {
            val += m * (ai[0]*aj[0] + ai[1]*aj[1] + ai[2]*aj[2]);
        }
        else if (jt_i == JOINT_BALL && jt_j == JOINT_BALL) {
            int bq4_bi = env * nbody * 4 + bi * 4;
            double bqw=xquat[bq4_bi],bqx=xquat[bq4_bi+1],bqy=xquat[bq4_bi+2],bqz=xquat[bq4_bi+3];
            double Bi00=1-2*(bqy*bqy+bqz*bqz),Bi01=2*(bqx*bqy-bqw*bqz),Bi02=2*(bqx*bqz+bqw*bqy);
            double Bi10=2*(bqx*bqy+bqw*bqz),Bi11=1-2*(bqx*bqx+bqz*bqz),Bi12=2*(bqy*bqz-bqw*bqx);
            double Bi20=2*(bqx*bqz-bqw*bqy),Bi21=2*(bqy*bqz+bqw*bqx),Bi22=1-2*(bqx*bqx+bqy*bqy);
            double wa[3];
            if (dof_offset_i==0)      {wa[0]=Bi00;wa[1]=Bi10;wa[2]=Bi20;}
            else if (dof_offset_i==1) {wa[0]=Bi01;wa[1]=Bi11;wa[2]=Bi21;}
            else                      {wa[0]=Bi02;wa[1]=Bi12;wa[2]=Bi22;}
            double wb[3];
            if (ji_i == ji_j) {
                if (dof_offset_j==0)      {wb[0]=Bi00;wb[1]=Bi10;wb[2]=Bi20;}
                else if (dof_offset_j==1) {wb[0]=Bi01;wb[1]=Bi11;wb[2]=Bi21;}
                else                      {wb[0]=Bi02;wb[1]=Bi12;wb[2]=Bi22;}
            } else {
                int bq4_bj = env * nbody * 4 + bj * 4;
                double bjw=xquat[bq4_bj],bjx=xquat[bq4_bj+1],bjy=xquat[bq4_bj+2],bjz=xquat[bq4_bj+3];
                double Bj00=1-2*(bjy*bjy+bjz*bjz),Bj01=2*(bjx*bjy-bjw*bjz),Bj02=2*(bjx*bjz+bjw*bjy);
                double Bj10=2*(bjx*bjy+bjw*bjz),Bj11=1-2*(bjx*bjx+bjz*bjz),Bj12=2*(bjy*bjz-bjw*bjx);
                double Bj20=2*(bjx*bjz-bjw*bjy),Bj21=2*(bjy*bjz+bjw*bjx),Bj22=1-2*(bjx*bjx+bjy*bjy);
                if (dof_offset_j==0)      {wb[0]=Bj00;wb[1]=Bj10;wb[2]=Bj20;}
                else if (dof_offset_j==1) {wb[0]=Bj01;wb[1]=Bj11;wb[2]=Bj21;}
                else                      {wb[0]=Bj02;wb[1]=Bj12;wb[2]=Bj22;}
            }
            double Iw_wb[3] = {
                (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wb[0]+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[1]+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[2],
                (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[0]+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wb[1]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[2],
                (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[0]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[1]+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wb[2]
            };
            double inertia_term = wa[0]*Iw_wb[0]+wa[1]*Iw_wb[1]+wa[2]*Iw_wb[2];
            double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
            double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
            double ca[3]={wa[1]*rki[2]-wa[2]*rki[1],wa[2]*rki[0]-wa[0]*rki[2],wa[0]*rki[1]-wa[1]*rki[0]};
            double cb[3]={wb[1]*rkj[2]-wb[2]*rkj[1],wb[2]*rkj[0]-wb[0]*rkj[2],wb[0]*rkj[1]-wb[1]*rkj[0]};
            val += inertia_term + m*(ca[0]*cb[0]+ca[1]*cb[1]+ca[2]*cb[2]);
        }
        else if (jt_i == JOINT_BALL && jt_j == JOINT_HINGE) {
            int bq4_bi = env * nbody * 4 + bi * 4;
            double bqw=xquat[bq4_bi],bqx=xquat[bq4_bi+1],bqy=xquat[bq4_bi+2],bqz=xquat[bq4_bi+3];
            double Bi00=1-2*(bqy*bqy+bqz*bqz),Bi01=2*(bqx*bqy-bqw*bqz),Bi02=2*(bqx*bqz+bqw*bqy);
            double Bi10=2*(bqx*bqy+bqw*bqz),Bi11=1-2*(bqx*bqx+bqz*bqz),Bi12=2*(bqy*bqz-bqw*bqx);
            double Bi20=2*(bqx*bqz-bqw*bqy),Bi21=2*(bqy*bqz+bqw*bqx),Bi22=1-2*(bqx*bqx+bqy*bqy);
            double wa[3];
            if (dof_offset_i==0)      {wa[0]=Bi00;wa[1]=Bi10;wa[2]=Bi20;}
            else if (dof_offset_i==1) {wa[0]=Bi01;wa[1]=Bi11;wa[2]=Bi21;}
            else                      {wa[0]=Bi02;wa[1]=Bi12;wa[2]=Bi22;}
            double Iw_aj[3] = {
                (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*aj[0]+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[1]+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[2],
                (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*aj[0]+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*aj[1]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[2],
                (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*aj[0]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*aj[1]+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*aj[2]
            };
            double inertia_term = wa[0]*Iw_aj[0]+wa[1]*Iw_aj[1]+wa[2]*Iw_aj[2];
            double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
            double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
            double ca[3]={wa[1]*rki[2]-wa[2]*rki[1],wa[2]*rki[0]-wa[0]*rki[2],wa[0]*rki[1]-wa[1]*rki[0]};
            double cj[3]={aj[1]*rkj[2]-aj[2]*rkj[1],aj[2]*rkj[0]-aj[0]*rkj[2],aj[0]*rkj[1]-aj[1]*rkj[0]};
            val += inertia_term + m*(ca[0]*cj[0]+ca[1]*cj[1]+ca[2]*cj[2]);
        }
        else if (jt_i == JOINT_HINGE && jt_j == JOINT_BALL) {
            int bq4_bj = env * nbody * 4 + bj * 4;
            double bjw=xquat[bq4_bj],bjx=xquat[bq4_bj+1],bjy=xquat[bq4_bj+2],bjz=xquat[bq4_bj+3];
            double Bj00=1-2*(bjy*bjy+bjz*bjz),Bj01=2*(bjx*bjy-bjw*bjz),Bj02=2*(bjx*bjz+bjw*bjy);
            double Bj10=2*(bjx*bjy+bjw*bjz),Bj11=1-2*(bjx*bjx+bjz*bjz),Bj12=2*(bjy*bjz-bjw*bjx);
            double Bj20=2*(bjx*bjz-bjw*bjy),Bj21=2*(bjy*bjz+bjw*bjx),Bj22=1-2*(bjx*bjx+bjy*bjy);
            double wb[3];
            if (dof_offset_j==0)      {wb[0]=Bj00;wb[1]=Bj10;wb[2]=Bj20;}
            else if (dof_offset_j==1) {wb[0]=Bj01;wb[1]=Bj11;wb[2]=Bj21;}
            else                      {wb[0]=Bj02;wb[1]=Bj12;wb[2]=Bj22;}
            double Iw_wb[3] = {
                (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wb[0]+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[1]+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[2],
                (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[0]+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wb[1]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[2],
                (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[0]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[1]+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wb[2]
            };
            double inertia_term = ai[0]*Iw_wb[0]+ai[1]*Iw_wb[1]+ai[2]*Iw_wb[2];
            double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
            double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
            double ci[3]={ai[1]*rki[2]-ai[2]*rki[1],ai[2]*rki[0]-ai[0]*rki[2],ai[0]*rki[1]-ai[1]*rki[0]};
            double cb[3]={wb[1]*rkj[2]-wb[2]*rkj[1],wb[2]*rkj[0]-wb[0]*rkj[2],wb[0]*rkj[1]-wb[1]*rkj[0]};
            val += inertia_term + m*(ci[0]*cb[0]+ci[1]*cb[1]+ci[2]*cb[2]);
        }
        else if (jt_i == JOINT_BALL && jt_j == JOINT_FREE) {
            int bq4_bi = env * nbody * 4 + bi * 4;
            double bqw=xquat[bq4_bi],bqx=xquat[bq4_bi+1],bqy=xquat[bq4_bi+2],bqz=xquat[bq4_bi+3];
            double Bi00=1-2*(bqy*bqy+bqz*bqz),Bi01=2*(bqx*bqy-bqw*bqz),Bi02=2*(bqx*bqz+bqw*bqy);
            double Bi10=2*(bqx*bqy+bqw*bqz),Bi11=1-2*(bqx*bqx+bqz*bqz),Bi12=2*(bqy*bqz-bqw*bqx);
            double Bi20=2*(bqx*bqz-bqw*bqy),Bi21=2*(bqy*bqz+bqw*bqx),Bi22=1-2*(bqx*bqx+bqy*bqy);
            double wa[3];
            if (dof_offset_i==0)      {wa[0]=Bi00;wa[1]=Bi10;wa[2]=Bi20;}
            else if (dof_offset_i==1) {wa[0]=Bi01;wa[1]=Bi11;wa[2]=Bi21;}
            else                      {wa[0]=Bi02;wa[1]=Bi12;wa[2]=Bi22;}
            int freeq4 = env * nbody * 4 + bj * 4;
            double fqw=xquat[freeq4],fqx=xquat[freeq4+1],fqy=xquat[freeq4+2],fqz=xquat[freeq4+3];
            double Fr00=1-2*(fqy*fqy+fqz*fqz),Fr01=2*(fqx*fqy-fqw*fqz),Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz),Fr11=1-2*(fqx*fqx+fqz*fqz),Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy),Fr21=2*(fqy*fqz+fqw*fqx),Fr22=1-2*(fqx*fqx+fqy*fqy);
            double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
            if (dof_offset_j < 3) {
                double ca[3]={wa[1]*rki[2]-wa[2]*rki[1],wa[2]*rki[0]-wa[0]*rki[2],wa[0]*rki[1]-wa[1]*rki[0]};
                val += m * ca[dof_offset_j];
            } else {
                int a = dof_offset_j - 3;
                double wf[3];
                if (a==0)      {wf[0]=Fr00;wf[1]=Fr10;wf[2]=Fr20;}
                else if (a==1) {wf[0]=Fr01;wf[1]=Fr11;wf[2]=Fr21;}
                else           {wf[0]=Fr02;wf[1]=Fr12;wf[2]=Fr22;}
                double Iw_wf[3] = {
                    (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wf[0]+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wf[1]+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wf[2],
                    (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wf[0]+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wf[1]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wf[2],
                    (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wf[0]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wf[1]+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wf[2]
                };
                double inertia_term = wa[0]*Iw_wf[0]+wa[1]*Iw_wf[1]+wa[2]*Iw_wf[2];
                double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
                double ca[3]={wa[1]*rki[2]-wa[2]*rki[1],wa[2]*rki[0]-wa[0]*rki[2],wa[0]*rki[1]-wa[1]*rki[0]};
                double cf[3]={wf[1]*rkj[2]-wf[2]*rkj[1],wf[2]*rkj[0]-wf[0]*rkj[2],wf[0]*rkj[1]-wf[1]*rkj[0]};
                val += inertia_term + m*(ca[0]*cf[0]+ca[1]*cf[1]+ca[2]*cf[2]);
            }
        }
        else if (jt_i == JOINT_FREE && jt_j == JOINT_BALL) {
            int bq4_bj = env * nbody * 4 + bj * 4;
            double bjw=xquat[bq4_bj],bjx=xquat[bq4_bj+1],bjy=xquat[bq4_bj+2],bjz=xquat[bq4_bj+3];
            double Bj00=1-2*(bjy*bjy+bjz*bjz),Bj01=2*(bjx*bjy-bjw*bjz),Bj02=2*(bjx*bjz+bjw*bjy);
            double Bj10=2*(bjx*bjy+bjw*bjz),Bj11=1-2*(bjx*bjx+bjz*bjz),Bj12=2*(bjy*bjz-bjw*bjx);
            double Bj20=2*(bjx*bjz-bjw*bjy),Bj21=2*(bjy*bjz+bjw*bjx),Bj22=1-2*(bjx*bjx+bjy*bjy);
            double wb[3];
            if (dof_offset_j==0)      {wb[0]=Bj00;wb[1]=Bj10;wb[2]=Bj20;}
            else if (dof_offset_j==1) {wb[0]=Bj01;wb[1]=Bj11;wb[2]=Bj21;}
            else                      {wb[0]=Bj02;wb[1]=Bj12;wb[2]=Bj22;}
            int freeq4 = env * nbody * 4 + bi * 4;
            double fqw=xquat[freeq4],fqx=xquat[freeq4+1],fqy=xquat[freeq4+2],fqz=xquat[freeq4+3];
            double Fr00=1-2*(fqy*fqy+fqz*fqz),Fr01=2*(fqx*fqy-fqw*fqz),Fr02=2*(fqx*fqz+fqw*fqy);
            double Fr10=2*(fqx*fqy+fqw*fqz),Fr11=1-2*(fqx*fqx+fqz*fqz),Fr12=2*(fqy*fqz-fqw*fqx);
            double Fr20=2*(fqx*fqz-fqw*fqy),Fr21=2*(fqy*fqz+fqw*fqx),Fr22=1-2*(fqx*fqx+fqy*fqy);
            double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
            if (dof_offset_i < 3) {
                double cb[3]={wb[1]*rkj[2]-wb[2]*rkj[1],wb[2]*rkj[0]-wb[0]*rkj[2],wb[0]*rkj[1]-wb[1]*rkj[0]};
                val += m * cb[dof_offset_i];
            } else {
                int a = dof_offset_i - 3;
                double wf[3];
                if (a==0)      {wf[0]=Fr00;wf[1]=Fr10;wf[2]=Fr20;}
                else if (a==1) {wf[0]=Fr01;wf[1]=Fr11;wf[2]=Fr21;}
                else           {wf[0]=Fr02;wf[1]=Fr12;wf[2]=Fr22;}
                double Iw_wb[3] = {
                    (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*wb[0]+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[1]+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[2],
                    (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*wb[0]+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*wb[1]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[2],
                    (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*wb[0]+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*wb[1]+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*wb[2]
                };
                double inertia_term = wf[0]*Iw_wb[0]+wf[1]*Iw_wb[1]+wf[2]*Iw_wb[2];
                double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
                double cf[3]={wf[1]*rki[2]-wf[2]*rki[1],wf[2]*rki[0]-wf[0]*rki[2],wf[0]*rki[1]-wf[1]*rki[0]};
                double cb[3]={wb[1]*rkj[2]-wb[2]*rkj[1],wb[2]*rkj[0]-wb[0]*rkj[2],wb[0]*rkj[1]-wb[1]*rkj[0]};
                val += inertia_term + m*(cf[0]*cb[0]+cf[1]*cb[1]+cf[2]*cb[2]);
            }
        }
        else if (jt_i == JOINT_BALL && jt_j == JOINT_SLIDE) {
            int bq4_bi = env * nbody * 4 + bi * 4;
            double bqw=xquat[bq4_bi],bqx=xquat[bq4_bi+1],bqy=xquat[bq4_bi+2],bqz=xquat[bq4_bi+3];
            double Bi00=1-2*(bqy*bqy+bqz*bqz),Bi01=2*(bqx*bqy-bqw*bqz),Bi02=2*(bqx*bqz+bqw*bqy);
            double Bi10=2*(bqx*bqy+bqw*bqz),Bi11=1-2*(bqx*bqx+bqz*bqz),Bi12=2*(bqy*bqz-bqw*bqx);
            double Bi20=2*(bqx*bqz-bqw*bqy),Bi21=2*(bqy*bqz+bqw*bqx),Bi22=1-2*(bqx*bqx+bqy*bqy);
            double wa[3];
            if (dof_offset_i==0)      {wa[0]=Bi00;wa[1]=Bi10;wa[2]=Bi20;}
            else if (dof_offset_i==1) {wa[0]=Bi01;wa[1]=Bi11;wa[2]=Bi21;}
            else                      {wa[0]=Bi02;wa[1]=Bi12;wa[2]=Bi22;}
            double rki[3]={com_x-pi[0],com_y-pi[1],com_z-pi[2]};
            double ca[3]={wa[1]*rki[2]-wa[2]*rki[1],wa[2]*rki[0]-wa[0]*rki[2],wa[0]*rki[1]-wa[1]*rki[0]};
            val += m * (ca[0]*aj[0]+ca[1]*aj[1]+ca[2]*aj[2]);
        }
        else if (jt_i == JOINT_SLIDE && jt_j == JOINT_BALL) {
            int bq4_bj = env * nbody * 4 + bj * 4;
            double bjw=xquat[bq4_bj],bjx=xquat[bq4_bj+1],bjy=xquat[bq4_bj+2],bjz=xquat[bq4_bj+3];
            double Bj00=1-2*(bjy*bjy+bjz*bjz),Bj01=2*(bjx*bjy-bjw*bjz),Bj02=2*(bjx*bjz+bjw*bjy);
            double Bj10=2*(bjx*bjy+bjw*bjz),Bj11=1-2*(bjx*bjx+bjz*bjz),Bj12=2*(bjy*bjz-bjw*bjx);
            double Bj20=2*(bjx*bjz-bjw*bjy),Bj21=2*(bjy*bjz+bjw*bjx),Bj22=1-2*(bjx*bjx+bjy*bjy);
            double wb[3];
            if (dof_offset_j==0)      {wb[0]=Bj00;wb[1]=Bj10;wb[2]=Bj20;}
            else if (dof_offset_j==1) {wb[0]=Bj01;wb[1]=Bj11;wb[2]=Bj21;}
            else                      {wb[0]=Bj02;wb[1]=Bj12;wb[2]=Bj22;}
            double rkj[3]={com_x-pj[0],com_y-pj[1],com_z-pj[2]};
            double cb[3]={wb[1]*rkj[2]-wb[2]*rkj[1],wb[2]*rkj[0]-wb[0]*rkj[2],wb[0]*rkj[1]-wb[1]*rkj[0]};
            val += m * (ai[0]*cb[0]+ai[1]*cb[1]+ai[2]*cb[2]);
        }
    }

    if (i == j) {
        val += 1e-14;
        if (jnt_armature) {
            val += jnt_armature[ji_i];
        }
    }

    M_env[i * KERNEL_MAXNV + j] = val;
    if (i != j) M_env[j * KERNEL_MAXNV + i] = val;
    if (i == j && M_inv_diag_out)
        M_inv_diag_out[env * MAX_DOF + i] = (val > 1e-10) ? (1.0 / val) : 0.0;
}

// ── Phase 1: Build Newton system H = M + J^T * diag(D) * J ─────────────────
// Grid: (ceil(nv*nv / 256), num_envs)
// Per-constraint impedance D_c from solimp/solref (MuJoCo-compatible).

HIPGL_GLOBAL void kernel_build_newton_H(
    int num_envs, int nv,
    const double* __restrict__ solver_M,
    const double* __restrict__ constraint_J,
    const double* __restrict__ constraint_D,
    const int*   __restrict__ nconstraint,
    double*      __restrict__ solver_H)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (env >= num_envs) return;

    int i = flat / nv;
    int j = flat % nv;
    if (i >= nv) return;

    int nc = nconstraint[env];
    if (nc > MAX_CONSTRAINTS) nc = MAX_CONSTRAINTS;

    size_t env_j_base = (size_t)env * MAX_CONSTRAINTS * nv;
    size_t env_c_base = (size_t)env * MAX_CONSTRAINTS;
    const double* M_env = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;

    double val = M_env[i * KERNEL_MAXNV + j];

    for (int c = 0; c < nc; c++) {
        const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
        double Dc = constraint_D[env_c_base + c];
        val += Dc * Jc[i] * Jc[j];
    }

    solver_H[(size_t)env * KERNEL_MAXNV * KERNEL_MAXNV + i * KERNEL_MAXNV + j] = val;
}

// ── Wavefront-cooperative Newton solver (64 threads per env) ─────────────────
// Mass matrix, Cholesky, and triangular solve in double precision for accuracy
// on ill-conditioned systems. Constraint evaluation stays in float (LDS).
// Matches MuJoCo/MJX/MJWarp reduced-primal joint-space formulation.

// ── Column-major (transposed) mirror of J, parked in the tail of constraint_J ─
// Every `for (c = t; c < nc; c += NEWTON_BLK)` loop in the Newton solver walks
// lane c's OWN J row, `Jc[v] = constraint_J[c*nv + v]`.  Consecutive lanes are
// nv*8 = 344 B apart, so each of those loads is a fully divergent 64-way gather
// AND (because the loop trip count comes from a runtime bitmask) only one load
// is ever in flight -- the whole latency is paid per element.
//
// The fix is a transposed mirror JT[v][c] so lane c reads stride-8 neighbours,
// plus a dense compile-time-bounded v loop so the 43 loads issue back to back.
// The mirror needs no new allocation: constraint_J is dimensioned
// [ne][MAX_CONSTRAINTS=8192][nv] and the live model uses nc ~ 270 rows, so the
// mirror lives in rows [JT_ROW_OFF, JT_ROW_OFF+JT_STRIDE) of the SAME per-env
// block -- untouched by every producer and by every other reader (which all
// index rows < nc).  Guarded by nc <= JT_STRIDE; above that the solver falls
// back to the original row-major masked walk.
// (JT_STRIDE / JT_ROW_OFF are defined above kernel_constraint_setup_rows.)

HIPGL_GLOBAL void kernel_transpose_J(
    int num_envs, int nv,
    const int* __restrict__ nconstraint,
    double*    __restrict__ constraint_J)
{
    int env = blockIdx.y;
    if (env >= num_envs) return;
    int nc = nconstraint[env];
    if (nc > MAX_CONSTRAINTS) nc = MAX_CONSTRAINTS;
    if (nc > JT_STRIDE) return;              // mirror not built; solver falls back
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= nc * nv) return;
    int c = flat / nv;
    int j = flat - c * nv;
    const size_t base = (size_t)env * MAX_CONSTRAINTS * nv;
    // read coalesced (consecutive lanes = consecutive j of one row)
    double val = constraint_J[base + (size_t)c * nv + j];
    constraint_J[base + (size_t)JT_ROW_OFF * nv + (size_t)j * JT_STRIDE + c] = val;
}

constexpr int NEWTON_MAXITER = 100;
constexpr int LS_MAXITER = 50;
constexpr int NEWTON_BLK = 64;
// Constraint rows whose J sparsity pattern we cache in LDS (live model: nc~270).
constexpr int NZ_CACHE = 512;

#if NEWTON_BLK <= 64
#define WSYNC() __builtin_amdgcn_wave_barrier()
#else
#define WSYNC() __syncthreads()
#endif

HIPGL_DEVICE inline float wf_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

HIPGL_DEVICE inline float wf_broadcast(float val, int src) {
    return __shfl(val, src);
}

HIPGL_DEVICE inline double wf_broadcast_d(double val, int src) {
    union { double d; unsigned u[2]; } tmp;
    tmp.d = val;
#ifdef __AMDGCN__
    tmp.u[0] = __builtin_amdgcn_readlane(tmp.u[0], src);
    tmp.u[1] = __builtin_amdgcn_readlane(tmp.u[1], src);
#else
    tmp.u[0] = __shfl(tmp.u[0], src);
    tmp.u[1] = __shfl(tmp.u[1], src);
#endif
    return tmp.d;
}

HIPGL_DEVICE inline unsigned long long wf_broadcast_u64(unsigned long long val, int src) {
    union { unsigned long long q; unsigned u[2]; } tmp;
    tmp.q = val;
#ifdef __AMDGCN__
    tmp.u[0] = __builtin_amdgcn_readlane(tmp.u[0], src);
    tmp.u[1] = __builtin_amdgcn_readlane(tmp.u[1], src);
#else
    tmp.u[0] = __shfl(tmp.u[0], src);
    tmp.u[1] = __shfl(tmp.u[1], src);
#endif
    return tmp.q;
}

HIPGL_DEVICE inline double wf_shfl_xor_d(double val, int laneMask) {
    union { double d; unsigned u[2]; } tmp;
    tmp.d = val;
    tmp.u[0] = __shfl_xor(tmp.u[0], laneMask);
    tmp.u[1] = __shfl_xor(tmp.u[1], laneMask);
    return tmp.d;
}

HIPGL_DEVICE inline double wf_reduce_sum_d(double val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += wf_shfl_xor_d(val, offset);
    return val;
}

template <int N>
HIPGL_DEVICE __forceinline__ void wf_reduce_sum_dn(double (&v)[N]) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        double o[N];
        #pragma unroll
        for (int i = 0; i < N; i++) o[i] = wf_shfl_xor_d(v[i], offset);
        #pragma unroll
        for (int i = 0; i < N; i++) v[i] += o[i];
    }
}



// Right-looking Cholesky: A -> L in the lower triangle (diagonal included).
// Every entry receives its k-th correction in the same ascending-k order the
// left-looking form used and with the same `x -= a*b` shape, so the factor is
// bit-identical -- only *which lane* issues each fma changes.
// The strict upper triangle is used as scratch and left undefined; no consumer
// reads it (both substitutions below touch only the lower triangle).
// The wave holds a whole constraint row: lane j owns Jc[j], so ONE ballot gives
// that row's exact sparsity pattern as a 64-bit mask. Contact/limit rows touch
// only the dofs of the bodies involved, so iterating the set bits instead of
// 0..nv-1 skips most of the row. Nothing accumulated changes: every skipped term
// is `x += k * 0.0`. Must be called at wave-uniform control flow.
HIPGL_DEVICE inline unsigned long long wf_row_nz_mask(const double* row, int t, bool active) {
    return __ballot(active && row[t] != 0.0);
}

HIPGL_DEVICE inline void wf_chol_factor_lds(double* A, int nv, int t, double delta)
{
    for (int k = 0; k < nv; k++) {
        if (t == k) {
            double s = A[k * LDS_STRIDE + k];
            if (s < delta) s = delta;
            A[k * LDS_STRIDE + k] = sqrt(s);
        }
        WSYNC();
        const bool own = (t > k && t < nv);
        double Lk = 0.0;
        if (own) {
            Lk = A[t * LDS_STRIDE + k] / A[k * LDS_STRIDE + k];
            A[t * LDS_STRIDE + k] = Lk;
        }
        WSYNC();
        if (own) {
            int r = k + 1;
            for (; r + 3 < nv; r += 4) {
                const double Ar0 = wf_broadcast_d(Lk, r);
                const double Ar1 = wf_broadcast_d(Lk, r + 1);
                const double Ar2 = wf_broadcast_d(Lk, r + 2);
                const double Ar3 = wf_broadcast_d(Lk, r + 3);
                const double a0 = A[r * LDS_STRIDE + t];
                const double a1 = A[(r + 1) * LDS_STRIDE + t];
                const double a2 = A[(r + 2) * LDS_STRIDE + t];
                const double a3 = A[(r + 3) * LDS_STRIDE + t];
                A[r * LDS_STRIDE + t]       = a0 - Ar0 * Lk;
                A[(r + 1) * LDS_STRIDE + t] = a1 - Ar1 * Lk;
                A[(r + 2) * LDS_STRIDE + t] = a2 - Ar2 * Lk;
                A[(r + 3) * LDS_STRIDE + t] = a3 - Ar3 * Lk;
            }
            for (; r < nv; r++) {
                const double Ark = wf_broadcast_d(Lk, r);
                A[r * LDS_STRIDE + t] -= Ark * Lk;
            }
        }
        WSYNC();
    }
}

// Forward substitution L*y = b, one lane per row (lane t returns y_t).
// Lane i divides out its pivot and hands the result to the whole wave at once;
// every lane then applies its own column-i correction. Corrections still arrive
// in ascending i, so each row sees the identical fma sequence as the
// row-at-a-time form -- bit-identical.
HIPGL_DEVICE inline double wf_tri_solve_fwd_lds(const double* A, int nv, int t,
                                                bool active, double b)
{
    double acc = b;
    for (int i = 0; i < nv; i++) {
        const double den = A[i * LDS_STRIDE + i];
        const double yi  = wf_broadcast_d(acc, i) / den;
        if (t == i)               acc  = yi;
        else if (active && t > i) acc -= A[t * LDS_STRIDE + i] * yi;
    }
    return acc;
}

// Back substitution L^T*x = y (lane t returns x_t). The column-i operand
// A[i][t] is contiguous across lanes.
// NOTE: this is the ONE place the rewrite changes arithmetic -- row t now
// accumulates its corrections in DESCENDING column order (the order in which
// x_j becomes available) instead of ascending. Same terms, different summation
// order. Validated against the correctness gate AND the noise floor, not assumed.
HIPGL_DEVICE inline double wf_tri_solve_bwd_lds(const double* A, int nv, int t,
                                                bool active, double y)
{
    double acc = y;
    for (int i = nv - 1; i >= 0; i--) {
        const double den = A[i * LDS_STRIDE + i];
        const double xi  = wf_broadcast_d(acc, i) / den;
        if (t == i)               acc  = xi;
        else if (active && t < i) acc -= A[i * LDS_STRIDE + t] * xi;
    }
    return acc;
}


// ── Sparse single-wave M Cholesky + level-scheduled substitutions ──────────
// The mass matrix of the live model is block-sparse (arm 9x9 dense chain, five
// free bodies whose 3 translation dofs are mutually uncoupled, singletons), so
// most of the right-looking trailing update is `x -= 0 * y`.  This form runs
// the SAME per-element sequence as wf_chol_factor_lds / mw_chol_factor_lds
// (ascending-k corrections with the identical `a - Ar*Lk` shape, then the
// division by the identical sqrt) but skips rows whose multiplier L[r][k] is
// exactly zero (a no-op correction), on ONE wave with no workgroup barriers.
// Also records each lane's row pattern (rp: j<t with L[t][j] != 0) and column
// pattern (cp: r>t with L[r][t] != 0) for the level-scheduled solves.
HIPGL_DEVICE inline void wf_chol_factor_lds_sp(double* A, int nv, int t, double delta,
                                               unsigned long long& rp, unsigned long long& cp)
{
    rp = 0ull; cp = 0ull;
    for (int k = 0; k < nv; k++) {
        double s = A[k * LDS_STRIDE + k];
        if (s < delta) s = delta;
        const double dkk = sqrt(s);
        const bool own = (t > k && t < nv);
        double Lk = 0.0;
        if (own) Lk = A[t * LDS_STRIDE + k] / dkk;
        if (t == k) A[k * LDS_STRIDE + k] = dkk;
        if (own)    A[t * LDS_STRIDE + k] = Lk;
        const unsigned long long rows = __ballot(own && Lk != 0.0);
        if (t == k) cp = rows;
        if ((rows >> t) & 1ull) rp |= (1ull << k);
        if (own) {
            unsigned long long m = rows;
            while (__popcll(m) >= 4) {
                const int r0 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
                const int r1 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
                const int r2 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
                const int r3 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
                const double Ar0 = wf_broadcast_d(Lk, r0);
                const double Ar1 = wf_broadcast_d(Lk, r1);
                const double Ar2 = wf_broadcast_d(Lk, r2);
                const double Ar3 = wf_broadcast_d(Lk, r3);
                const double a0 = A[r0 * LDS_STRIDE + t];
                const double a1 = A[r1 * LDS_STRIDE + t];
                const double a2 = A[r2 * LDS_STRIDE + t];
                const double a3 = A[r3 * LDS_STRIDE + t];
                A[r0 * LDS_STRIDE + t] = a0 - Ar0 * Lk;
                A[r1 * LDS_STRIDE + t] = a1 - Ar1 * Lk;
                A[r2 * LDS_STRIDE + t] = a2 - Ar2 * Lk;
                A[r3 * LDS_STRIDE + t] = a3 - Ar3 * Lk;
            }
            for (; m; m &= m - 1) {
                const int r = __ffsll((unsigned long long)m) - 1;
                const double Ark = wf_broadcast_d(Lk, r);
                A[r * LDS_STRIDE + t] -= Ark * Lk;
            }
        }
        WSYNC();
    }
}

// Level schedule validity.  A lane is solved in the first round in which all
// its dependencies (pat) are done; its corrections then arrive in round order
// (ascending index within a round for asc, descending for !asc).  The result
// is bit-identical to the sequential substitution iff, for every lane, that
// arrival order equals the sequential one (ascending / descending index).
HIPGL_DEVICE inline bool wf_lvl_ok(int nv, int t, bool active, unsigned long long pat, bool asc)
{
    const unsigned long long full = (nv >= 64) ? ~0ull : ((1ull << nv) - 1ull);
    unsigned long long done = 0ull;
    int edge = asc ? -1 : 64;
    bool bad = false;
    for (int it = 0; it < nv && done != full; it++) {
        const bool mine = active && !((done >> t) & 1ull);
        const unsigned long long R = __ballot(mine && ((pat & ~done) == 0ull));
        const unsigned long long P = pat & R;
        if (mine && P) {
            const int lo = __ffsll((unsigned long long)P) - 1;
            const int hi = 63 - __clzll(P);
            if (asc) { if (lo < edge) bad = true; edge = hi; }
            else     { if (hi > edge) bad = true; edge = lo; }
        }
        done |= R;
    }
    return (__ballot(bad) == 0ull) && (done == full);
}

// Level-scheduled L y = b (lane t returns y_t).  Same divisions and the same
// `acc - L*y` corrections as wf_tri_solve_fwd_lds minus the exact-zero ones.
HIPGL_DEVICE inline double wf_tri_fwd_lvl(const double* A, double* sy, int nv, int t,
                                          bool active, double b, unsigned long long rp)
{
    const unsigned long long full = (nv >= 64) ? ~0ull : ((1ull << nv) - 1ull);
    unsigned long long done = 0ull;
    double acc = b;
    const double den = active ? A[t * LDS_STRIDE + t] : 1.0;
    while (done != full) {
        const bool mine = active && !((done >> t) & 1ull);
        const bool rdy  = mine && ((rp & ~done) == 0ull);
        const unsigned long long R = __ballot(rdy);
        if (rdy) { acc = acc / den; sy[t] = acc; }
        WSYNC();
        if (mine && !rdy) {
            for (unsigned long long P = rp & R; P; P &= P - 1) {
                const int i = __ffsll((unsigned long long)P) - 1;
                acc -= A[t * LDS_STRIDE + i] * sy[i];
            }
        }
        done |= R;
        WSYNC();
    }
    return acc;
}

// Level-scheduled L^T x = y (lane t returns x_t); descending corrections as
// in wf_tri_solve_bwd_lds.
HIPGL_DEVICE inline double wf_tri_bwd_lvl(const double* A, double* sy, int nv, int t,
                                          bool active, double y, unsigned long long cp)
{
    const unsigned long long full = (nv >= 64) ? ~0ull : ((1ull << nv) - 1ull);
    unsigned long long done = 0ull;
    double acc = y;
    const double den = active ? A[t * LDS_STRIDE + t] : 1.0;
    while (done != full) {
        const bool mine = active && !((done >> t) & 1ull);
        const bool rdy  = mine && ((cp & ~done) == 0ull);
        const unsigned long long R = __ballot(rdy);
        if (rdy) { acc = acc / den; sy[t] = acc; }
        WSYNC();
        if (mine && !rdy) {
            for (unsigned long long P = cp & R; P; ) {
                const int i = 63 - __clzll(P);
                P &= ~(1ull << i);
                acc -= A[i * LDS_STRIDE + t] * sy[i];
            }
        }
        done |= R;
        WSYNC();
    }
    return acc;
}

// Warm-start selection must use the same objective as the Newton line search.
// Keep its later implementation untouched; this copy only scores initial guesses.
HIPGL_DEVICE inline double warmstart_constraint_cost(
    int t, int nc, int cone_type, double impr_inv, size_t env_c_base,
    const double* residual, const double* constraint_D,
    const double* constraint_lo, const double* constraint_hi,
    const int* constraint_friction_parent, const int* constraint_friction_sibling,
    const double* constraint_friction_mu, const bool* elliptic_normal)
{
        double warm_scalar_cost = 0.0;
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double xc = residual[c];
            double Dc = constraint_D[env_c_base + c];
            double lo_c = constraint_lo[env_c_base + c];
            double hi_c = constraint_hi[env_c_base + c];
            int par_c = constraint_friction_parent[env_c_base + c];
            if (cone_type == 1 && (par_c >= 0 || elliptic_normal[c])) continue;
            double sc = 0.0;
            if (lo_c < -1e15 && hi_c > 1e15) sc = 0.5 * Dc * xc * xc;
            else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            else if (par_c < 0) {
                double f = fabs(hi_c);
                double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                double ax = fabs(xc);
                sc = (ax < rf) ? 0.5 * Dc * xc * xc : f * (ax - 0.5 * rf);
            } else { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            warm_scalar_cost += sc;
        }
        warm_scalar_cost = wf_reduce_sum_d(warm_scalar_cost);

        double warm_cone_cost = 0.0;
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;
                const int torsion = constraint_friction_sibling[env_c_base+par];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {par, c, sib, torsion};
                    warm_cone_cost += evaluate_contact4(rows, residual, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr);
                    continue;
                }
                double warm_muf = (double)constraint_friction_mu[env_c_base + c];
                double mul = warm_muf * impr_inv;
                double Nn = residual[par] * mul;
                double ut1 = residual[c] * warm_muf, ut2 = residual[sib] * warm_muf;
                double TTl = ut1*ut1 + ut2*ut2;
                double Tl = (TTl > 0.0) ? sqrt(TTl) : 0.0;
                if ((Nn >= mul*Tl) || (Tl <= 0 && Nn >= 0)) { }
                else if ((mul*Nn+Tl <= 0) || (Tl <= 0 && Nn < 0)) {
                    warm_cone_cost += 0.5*constraint_D[env_c_base+par]*residual[par]*residual[par]
                                 + 0.5*constraint_D[env_c_base+c]*residual[c]*residual[c]
                                 + 0.5*constraint_D[env_c_base+sib]*residual[sib]*residual[sib];
                } else {
                    double D0 = constraint_D[env_c_base+par];
                    double mu2l = mul*mul;
                    double dml = D0 / fmax(mu2l*(1+mu2l), 1e-20);
                    double nmt = Nn - mul*Tl;
                    warm_cone_cost += 0.5 * dml * nmt * nmt;
                }
            }
        }
        warm_cone_cost = wf_reduce_sum_d(warm_cone_cost);
        return warm_scalar_cost + warm_cone_cost;
}


HIPGL_DEVICE __forceinline__ double newton_warm_ccost_helper(
    int t, int nc, int nv, int cone_type, double impr_inv, size_t env_c_base,
    const double* __restrict__ qwarm_env, double* s_qw,
    const double* __restrict__ JTbase, const double* __restrict__ constraint_rhs,
    double* g_x, bool* g_is_en,
    const double* __restrict__ constraint_D, const double* __restrict__ constraint_lo,
    const double* __restrict__ constraint_hi,
    const int* __restrict__ constraint_friction_parent,
    const int* __restrict__ constraint_friction_sibling,
    const double* __restrict__ constraint_friction_mu)
{
    const bool active = (t < nv);
    if (active) s_qw[t] = qwarm_env[t];
    WSYNC();
    const bool skip_zero = (nc <= NZ_CACHE);
    for (int c = t; c < nc; c += NEWTON_BLK) {
        double xw = -constraint_rhs[env_c_base + c];
        const double* JTc = JTbase + c;
        constexpr int B = 8;
        for (int a0 = 0; a0 < nv; a0 += B) {
            double Jv[B];
#pragma unroll
            for (int k = 0; k < B; k++)
                Jv[k] = (a0 + k < nv) ? JTc[(size_t)(a0 + k) * JT_STRIDE] : 0.0;
#pragma unroll
            for (int k = 0; k < B; k++) {
                const int v = a0 + k;
                if (v < nv && (!skip_zero || Jv[k] != 0.0)) xw += Jv[k] * s_qw[v];
            }
        }
        g_x[c] = xw;
        g_is_en[c] = false;
    }
    __syncwarp();
    if (cone_type == 1) {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int par = constraint_friction_parent[env_c_base + c];
            if (par >= 0 && par < nc) g_is_en[par] = true;
        }
    }
    __syncwarp();
    return warmstart_constraint_cost(t, nc, cone_type, impr_inv, env_c_base,
        g_x, constraint_D, constraint_lo, constraint_hi, constraint_friction_parent,
        constraint_friction_sibling, constraint_friction_mu, g_is_en);
}

// ── Multi-wave Newton (4 waves per env): wave 0 is the master, 1..3 helpers ──
// The solver's serial chain stays on wave 0 exactly as before (NEWTON_BLK is
// still the 64-lane LOGICAL width every `c += NEWTON_BLK` loop and every wave
// reduction uses, so none of those sums change).  The three helper waves are
// parked on a workgroup barrier and are woken by a command word in LDS only for
// the pieces that dominate the dispatch and split WITHOUT changing any
// accumulation order:
//   * the H build (J^T D J) and the elliptic-cone Hessian: every H element
//     (row t, column j) is owned by exactly ONE wave for the whole build via a
//     STATIC column mask, and that wave applies the element's contributions in
//     the identical c / cone order, so H is bit-identical;
//   * the Cholesky trailing update: rows are dealt round-robin to waves at each
//     step k, one workgroup barrier separates steps, so every element still
//     receives its k-th correction at step k with the same `x -= a*b` shape.
// No atomics and nothing whose order depends on wave scheduling.
constexpr int NEWTON_WAVES   = 4;
#ifndef NEWTON_PROF
#define NEWTON_PROF 0
#endif
#ifndef R4_CHK
#define R4_CHK 0
#endif
#ifndef R5_CHK
#define R5_CHK 0
#endif
#if NEWTON_PROF
#define NPS(k) do { const unsigned long long _n = __builtin_amdgcn_s_memtime(); np_acc[k] += _n - np_last; np_last = _n; } while (0)
#define NPQ(k) do { const unsigned long long _n = __builtin_amdgcn_s_memtime(); np_pre[k] += _n - np_last; np_last = _n; } while (0)
#define NPX(k) do { const unsigned long long _n = __builtin_amdgcn_s_memtime(); np_x[k] += _n - np_xl; np_xl = _n; } while (0)
#define NPX0() do { np_xl = __builtin_amdgcn_s_memtime(); } while (0)
#else
#define NPX(k) do { } while (0)
#define NPX0() do { } while (0)
#define NPS(k) do { } while (0)
#define NPQ(k) do { } while (0)
#endif
constexpr int NEWTON_THREADS = NEWTON_BLK * NEWTON_WAVES;
constexpr int NCMD_EXIT = 0;
constexpr int NCMD_CHOL = 1;
constexpr int NCMD_H    = 2;
#define BSYNC() __syncthreads()

// Static column ownership: wave w owns H columns j with (j % NEWTON_WAVES) == w.
HIPGL_DEVICE inline unsigned long long newton_col_mask(int wv) {
    unsigned long long m = 0x1111111111111111ull;
    return m << wv;
}

// Multi-wave right-looking Cholesky.  Same arithmetic as wf_chol_factor_lds:
// every wave computes the pivot sqrt and the column-k multipliers redundantly
// (identical bits), wave 0 commits them one step LATE (column k-1 is not read
// during step k, so the commit cannot race a reader), and the trailing rows are
// split round-robin across waves.  Exactly nv workgroup barriers; every wave of
// the block must call it with the same nv.
HIPGL_DEVICE inline void mw_chol_factor_lds(double* A, int nv, int t, int wv, double delta)
{
    int    pk = -1;
    double pd = 0.0, pL = 0.0;
    bool   pown = false;
    for (int k = 0; k < nv; k++) {
        if (wv == 0 && pk >= 0) {
            if (t == pk) A[pk * LDS_STRIDE + pk] = pd;
            if (pown)    A[t * LDS_STRIDE + pk]  = pL;
        }
        double s = A[k * LDS_STRIDE + k];
        if (s < delta) s = delta;
        const double dkk = sqrt(s);
        const bool own = (t > k && t < nv);
        double Lk = 0.0;
        if (own) Lk = A[t * LDS_STRIDE + k] / dkk;
        pk = k; pd = dkk; pL = Lk; pown = own;
        if (own) {
            int r = k + 1 + wv;
            constexpr int S = NEWTON_WAVES;
            for (; r + 3 * S < nv; r += 4 * S) {
                const double Ar0 = wf_broadcast_d(Lk, r);
                const double Ar1 = wf_broadcast_d(Lk, r + S);
                const double Ar2 = wf_broadcast_d(Lk, r + 2 * S);
                const double Ar3 = wf_broadcast_d(Lk, r + 3 * S);
                const double a0 = A[r * LDS_STRIDE + t];
                const double a1 = A[(r + S) * LDS_STRIDE + t];
                const double a2 = A[(r + 2 * S) * LDS_STRIDE + t];
                const double a3 = A[(r + 3 * S) * LDS_STRIDE + t];
                A[r * LDS_STRIDE + t]           = a0 - Ar0 * Lk;
                A[(r + S) * LDS_STRIDE + t]     = a1 - Ar1 * Lk;
                A[(r + 2 * S) * LDS_STRIDE + t] = a2 - Ar2 * Lk;
                A[(r + 3 * S) * LDS_STRIDE + t] = a3 - Ar3 * Lk;
            }
            for (; r < nv; r += S) {
                const double Ark = wf_broadcast_d(Lk, r);
                A[r * LDS_STRIDE + t] -= Ark * Lk;
            }
        }
        BSYNC();
    }
    // Final commit.  Only wave 0 reads L next (the substitutions), in program
    // order after these stores; helpers see it after the next command barrier.
    if (wv == 0 && pk >= 0) {
        if (t == pk) A[pk * LDS_STRIDE + pk] = pd;
        if (pown)    A[t * LDS_STRIDE + pk]  = pL;
    }
}

// ── Register-resident single-wave Cholesky + substitutions (compile-time nv) ──
// Lane t holds ROW t of H in a[0..N).  Step K does, per element, exactly what
// mw_chol_factor_lds does: pivot s = A[K][K] clamped to delta, dkk = sqrt(s),
// L[t][K] = A[t][K] / dkk, then A[t][j] -= L[t][K] * L[j][K] for j > K (the
// same product, the same fma, the same ascending-K order per element).  Only
// the lower triangle is ever read; the upper part of a[] is junk, as in LDS.
// No LDS traffic and no workgroup barrier per step.
template <int N, int K>
HIPGL_DEVICE __forceinline__ void rc_chol_step(double (&a)[N], int t, double delta)
{
    if constexpr (K < N) {
        double s = wf_broadcast_d(a[K], K);
        if (s < delta) s = delta;
        const double dkk = sqrt(s);
        const double Lk  = a[K] / dkk;
        a[K] = (t == K) ? dkk : Lk;
        #pragma unroll
        for (int j = K + 1; j < N; j++) {
            const double Lj = wf_broadcast_d(Lk, j);
            a[j] = a[j] - Lj * Lk;
        }
        rc_chol_step<N, K + 1>(a, t, delta);
    }
}

// ── Software-pipelined register Cholesky ────────────────────────────────────
// The per-step critical path of rc_chol_step is  update A[K+1][K+1] -> pivot
// broadcast -> sqrt -> divide -> next update, ~30 dependent f64 ops; the
// (N-K-2) remaining trailing updates of step K do not feed it.  In-order issue
// only overlaps the two if they are interleaved in program order, so the sqrt
// and the division are spelled out with the exact operation sequence the
// backend emits for `sqrt(double)` / `double / double` (lowerFSQRTF64 /
// LowerFDIV64: same instructions, same operands => bit-identical results) and
// the independent updates are threaded between the chain stages.  Per element
// the arithmetic and its order are unchanged from rc_chol_step.
HIPGL_DEVICE __forceinline__ double rc_sq_pre(double s, bool& sc)
{
    sc = s < 0x1p-767;
    return __builtin_amdgcn_ldexp(s, sc ? 256 : 0);
}

template <int N, int J0, int J1>
HIPGL_DEVICE __forceinline__ void rc_upd(double (&a)[N], double Lk)
{
    if constexpr (J0 < J1 && J0 < N) {
        const double Lj = wf_broadcast_d(Lk, J0);
        a[J0] = a[J0] - Lj * Lk;
        rc_upd<N, J0 + 1, J1>(a, Lk);
    }
}

// On entry: Lk = L[t][K] (all lanes), dkk = L[K][K]; column K of a[] still
// holds the pre-division values; a[j>K] carry every correction k < K.
template <int N, int K>
HIPGL_DEVICE __forceinline__ void rc_cholp_step(double (&a)[N], int t, double delta,
                                                double Lk, double dkk)
{
    a[K] = (t == K) ? dkk : Lk;
    if constexpr (K + 1 < N) {
        rc_upd<N, K + 1, K + 2>(a, Lk);
        constexpr int R = N - K - 2;         // independent trailing updates
        constexpr int NS = 22;               // pipeline slots
#define RCU(i) rc_upd<N, K + 2 + (R * (i)) / NS, K + 2 + (R * ((i) + 1)) / NS>(a, Lk)
        RCU(0);
        double s = wf_broadcast_d(a[K + 1], K + 1);
        if (s < delta) s = delta;
        RCU(1);
        bool sc;
        const double x = rc_sq_pre(s, sc);
        RCU(2);
        const double y = __builtin_amdgcn_rsq(x);
        RCU(3);
        double g = x * y;
        double h = y * 0.5;
        RCU(4);
        const double r0 = __builtin_fma(-h, g, 0.5);
        RCU(5);
        h = __builtin_fma(h, r0, h);
        g = __builtin_fma(g, r0, g);
        RCU(6);
        double d = __builtin_fma(-g, g, x);
        RCU(7);
        g = __builtin_fma(d, h, g);
        RCU(8);
        d = __builtin_fma(-g, g, x);
        RCU(9);
        g = __builtin_fma(d, h, g);
        RCU(10);
        g = __builtin_amdgcn_ldexp(g, sc ? -128 : 0);
        const double dk1 = __builtin_amdgcn_class(x, 0x260) ? x : g;
        RCU(11);
        const double num = a[K + 1];
        bool fl0, fl1;
        const double ds = __builtin_amdgcn_div_scale(num, dk1, false, &fl0);
        RCU(12);
        double rc = __builtin_amdgcn_rcp(ds);
        RCU(13);
        double e = __builtin_fma(-ds, rc, 1.0);
        RCU(14);
        rc = __builtin_fma(rc, e, rc);
        RCU(15);
        e = __builtin_fma(-ds, rc, 1.0);
        const double ns = __builtin_amdgcn_div_scale(num, dk1, true, &fl1);
        RCU(16);
        rc = __builtin_fma(rc, e, rc);
        RCU(17);
        const double q = ns * rc;
        RCU(18);
        const double e2 = __builtin_fma(-ds, q, ns);
        RCU(19);
        const double fm = __builtin_amdgcn_div_fmas(e2, rc, q, fl1);
        RCU(20);
        const double L1 = __builtin_amdgcn_div_fixup(fm, dk1, num);
        RCU(21);
#undef RCU
        rc_cholp_step<N, K + 1>(a, t, delta, L1, dk1);
    }
}

template <int N>
HIPGL_DEVICE __forceinline__ void rc_cholp(double (&a)[N], int t, double delta)
{
    double s = wf_broadcast_d(a[0], 0);
    if (s < delta) s = delta;
    const double dkk = sqrt(s);
    const double Lk  = a[0] / dkk;
    rc_cholp_step<N, 0>(a, t, delta, Lk, dkk);
}

// Forward substitution on the register factor: identical op sequence to
// wf_tri_solve_fwd_lds (den = L[i][i], yi = y_i / den, acc -= L[t][i] * yi).
template <int N>
HIPGL_DEVICE __forceinline__ double rc_tri_fwd(const double (&a)[N], int t, bool active, double b)
{
    double acc = b;
    #pragma unroll
    for (int i = 0; i < N; i++) {
        const double den = wf_broadcast_d(a[i], i);
        const double yi  = wf_broadcast_d(acc, i) / den;
        if (t == i)               acc  = yi;
        else if (active && t > i) acc -= a[i] * yi;
    }
    return acc;
}

// Backward substitution on the TRANSPOSED factor: lane t holds column t of L
// (c[i] = L[i][t]).  Identical op sequence to wf_tri_solve_bwd_lds.
template <int N>
HIPGL_DEVICE __forceinline__ double rc_tri_bwd(const double (&c)[N], int t, bool active, double y)
{
    double acc = y;
    #pragma unroll
    for (int i = N - 1; i >= 0; i--) {
        const double den = wf_broadcast_d(c[i], i);
        const double xi  = wf_broadcast_d(acc, i) / den;
        if (t == i)               acc  = xi;
        else if (active && t < i) acc -= c[i] * xi;
    }
    return acc;
}

template <int N>
HIPGL_DEVICE __forceinline__ double jt_dot_unr(const double* __restrict__ JTc,
                                               unsigned long long m,
                                               const double* __restrict__ sv, double x)
{
    double Jr[N];
#pragma unroll
    for (int v = 0; v < N; v++) Jr[v] = JTc[(size_t)v * JT_STRIDE];
#pragma unroll
    for (int v = 0; v < N; v++)
        if ((m >> v) & 1ull) x += Jr[v] * sv[v];
    return x;
}

// J row sparsity masks (s_nzJ) for one wave: pure integer work, so any wave
// may build it -- the result is identical wherever it runs.
HIPGL_DEVICE __forceinline__ void newton_build_nzJ(int t, int nc, int nv, bool use_jt,
    const double* __restrict__ JTbase, const double* __restrict__ Jenv,
    unsigned long long* __restrict__ s_nzJ)
{
    for (int c = t; c < nc; c += NEWTON_BLK) {
        unsigned long long m = 0ull;
        if (use_jt) {
            const double* JTc = JTbase + c;
#pragma unroll
            for (int v = 0; v < nv; v++)
                if (JTc[(size_t)v * JT_STRIDE] != 0.0) m |= (1ull << v);
        } else {
            const double* Jc = Jenv + (size_t)c * nv;
            for (int v = 0; v < nv; v++) if (Jc[v] != 0.0) m |= (1ull << v);
        }
        s_nzJ[c] = m;
    }
}

// Elliptic cone list (ballot-compacted, ascending c) for one wave.
HIPGL_DEVICE __forceinline__ int newton_build_cones(int t, int nc, int cap,
    const int* __restrict__ fpar, const int* __restrict__ fsib,
    int* s_coneC, int* s_coneP, int* s_coneS, int* s_coneT)
{
    int ncone = 0;
    for (int c0 = 0; c0 < nc; c0 += NEWTON_BLK) {
        const int c = c0 + t;
        bool own = false;
        int par = -1, sib = -1, tor = -1;
        if (c < nc) {
            par = fpar[c];
            sib = fsib[c];
            own = (par >= 0 && par < nc && sib >= 0 && sib < nc && c < sib);
            if (own) tor = fsib[par];
        }
        const unsigned long long b = __ballot(own);
        const int idx = ncone + __popcll(b & ((1ull << t) - 1ull));
        if (own && idx < cap) {
            s_coneC[idx] = c; s_coneP[idx] = par;
            s_coneS[idx] = sib; s_coneT[idx] = tor;
        }
        ncone += __popcll(b);
    }
    return ncone;
}

template <int HB_PF>
HIPGL_DEVICE __forceinline__ void newton_hbuild_cols(
    int t, bool active, int nc, int nv, bool use_jt,
    const double* __restrict__ JTbase,
    const double* __restrict__ Jcol,
    const double* __restrict__ g_force,
    const int*    __restrict__ g_state,
    const double* __restrict__ Denv,
    const unsigned long long* __restrict__ s_nzJ,
    double* __restrict__ H_row,
    unsigned long long own, bool do_grad, double& grad_t)
{
    for (int c0 = 0; c0 < nc; c0 += NEWTON_BLK) {
        const int  cl  = c0 + t;
        const bool inr = (cl < nc);
        const double             rfc = inr ? g_force[cl] : 0.0;
        const double             rDc = inr ? Denv[cl] : 0.0;
        const unsigned long long rnz = inr ? s_nzJ[cl] : 0ull;
        const unsigned long long stm = __ballot(inr && g_state[cl] == 1);
        if (!active) continue;
        const int lim = (nc - c0 < NEWTON_BLK) ? (nc - c0) : NEWTON_BLK;
        for (int g = 0; g < lim; g += HB_PF) {
            double Jb[HB_PF], dJ[HB_PF];
            unsigned long long um = 0ull;
#pragma unroll
            for (int k = 0; k < HB_PF; k++) Jb[k] = 0.0;
            if (use_jt) {
                const double2* __restrict__ JTv =
                    (const double2*)(JTbase + (size_t)t * JT_STRIDE + (c0 + g));
                double2 jv[HB_PF / 2];
#pragma unroll
                for (int k = 0; k < HB_PF / 2; k++) jv[k] = JTv[k];
#pragma unroll
                for (int k = 0; k < HB_PF; k++) {
                    if (g + k < lim)
                        Jb[k] = ((k & 1) ? jv[k >> 1].y : jv[k >> 1].x);
                }
            } else {
#pragma unroll
                for (int k = 0; k < HB_PF; k++) {
                    const int l = g + k;
                    Jb[k] = (l < lim) ? Jcol[(size_t)(c0 + l) * (size_t)nv] : 0.0;
                }
            }
#pragma unroll
            for (int k = 0; k < HB_PF; k++) {
                const int l = g + k;
                double d = 0.0;
                if (l < lim) {
                    const double Jct = Jb[k];
                    if (do_grad) {
                        const double fc = wf_broadcast_d(rfc, l);
                        if (fabs(fc) >= 1e-30) grad_t -= fc * Jct;
                    }
                    if ((stm >> l) & 1ull) {
                        const double dJt = wf_broadcast_d(rDc, l) * Jct;
                        if (fabs(dJt) >= 1e-30) d = dJt;
                        um |= wf_broadcast_u64(rnz, l);
                    }
                }
                dJ[k] = d;
            }
            for (unsigned long long m = um & own; m; ) {
                const int  j0 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
                const bool b1 = (m != 0ull);
                const int  j1 = b1 ? (__ffsll((unsigned long long)m) - 1) : j0;
                m &= m - 1;
                double h0 = H_row[j0];
                double h1 = H_row[j1];
#pragma unroll
                for (int k = 0; k < HB_PF; k++) {
                    h0 += dJ[k] * wf_broadcast_d(Jb[k], j0);
                    h1 += dJ[k] * wf_broadcast_d(Jb[k], j1);
                }
                if (b1) H_row[j1] = h1;
                H_row[j0] = h0;
            }
        }
    }
}

// Elliptic-cone Hessian for middle-zone (state 4) contacts, restricted to the
// columns in `own`.  Cone order, (ci, cj) order and the `hJi * Jj[j]` shape are
// unchanged.  Jj[j] is no longer a uniform global load inside the RMW chain:
// the wave loads each cone row once (lane v holds J[row][v]) and fetches
// Jj[j] with a readlane -- the identical bits.  Must be called at wave-uniform
// control flow (all 64 lanes).  Requires the s_nzJ cache (nz_ok).
HIPGL_DEVICE __forceinline__ void newton_cone_cols(
    int t, bool active, int nc, int nv, bool cone_ok, int ncone,
    const int* s_coneC, const int* s_coneP, const int* s_coneS, const int* s_coneT,
    const int*    __restrict__ cfp_env,
    const int*    __restrict__ cfs_env,
    const double* __restrict__ g_x,
    const int*    __restrict__ g_state,
    const double* __restrict__ Denv,
    const double* __restrict__ muenv,
    double impr_inv,
    const unsigned long long* __restrict__ s_nzJ,
    const double* __restrict__ Jenv,
    double* __restrict__ H_row,
    unsigned long long own,
    const int* q4list = nullptr, int n4 = 0)
{
    // q4list (optional): the cones whose normal is in state 4, ascending q --
    // exactly the cones the g_state test below would keep.
    const bool use_l = (q4list != nullptr);
    const int qn = use_l ? n4 : (cone_ok ? ncone : nc);
    for (int qi = 0; qi < qn; qi++) {
        const int q = use_l ? q4list[qi] : qi;
        int c, par, sib, torsion;
        if (cone_ok || use_l) {
            c = s_coneC[q]; par = s_coneP[q];
            sib = s_coneS[q]; torsion = s_coneT[q];
        } else {
            c = q;
            par = cfp_env[c];
            sib = cfs_env[c];
            if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
            if (c >= sib) continue;
            torsion = cfs_env[par];
        }
        const int normal = par, t1 = c, t2 = sib;
        if (!use_l && g_state[normal] != 4) continue;

        if (torsion >= 0 && torsion < nc) {
            const int rows[4] = {normal, t1, t2, torsion};
            double h[16];
            evaluate_contact4(rows, g_x, nullptr, 0.0, Denv, muenv,
                              impr_inv, nullptr, nullptr, h);
            unsigned long long nzr[4];
            double Jv[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                nzr[r] = s_nzJ[rows[r]] & own;
                Jv[r]  = active ? Jenv[(size_t)rows[r] * nv + t] : 0.0;
            }
#pragma unroll
            for (int ci = 0; ci < 4; ++ci) {
                const double Jit = Jv[ci];
                const bool on = active && !(fabs(Jit) < 1e-30);
#pragma unroll
                for (int cj = 0; cj < 4; ++cj) {
                    const double hij = h[ci * 4 + cj];
                    if (fabs(hij) < 1e-30) continue;
                    const double hJi = hij * Jit;
                    for (unsigned long long m = nzr[cj]; m; m &= m - 1) {
                        const int j = __ffsll((unsigned long long)m) - 1;
                        const double Jjj = wf_broadcast_d(Jv[cj], j);
                        if (on) H_row[j] += hJi * Jjj;
                    }
                }
            }
            continue;
        }

        double mu_fric = (double)muenv[c];
        double mu_l = mu_fric * impr_inv;
        double mu2_l = mu_l * mu_l;

        double xn = g_x[normal], xt1_v = g_x[t1], xt2_v = g_x[t2];
        double un = xn * mu_l, u1v = xt1_v * mu_fric, u2v = xt2_v * mu_fric;
        double TT = u1v * u1v + u2v * u2v;
        double T_val = (TT > 0.0) ? sqrt(TT) : 1e-20;
        T_val = fmax(T_val, 1e-20);

        double D0 = Denv[normal];
        double dm_val = D0 / fmax(mu2_l * (1.0 + mu2_l), 1e-20);

        double u_vec[3] = {un, u1v, u2v};
        double fri_vec[3] = {mu_l, mu_fric, mu_fric};

        double coeff_outer = mu_l * un / T_val;
        double coeff_diag = mu2_l - mu_l * un / T_val;
        double h[3][3];
        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                h[ii][jj] = coeff_outer * (u_vec[ii] / T_val) * (u_vec[jj] / T_val)
                           + ((ii == jj) ? coeff_diag : 0.0);

        double h0_1 = -mu_l / T_val * u1v;
        double h0_2 = -mu_l / T_val * u2v;
        h[0][0] = 1.0; h[0][1] = h0_1; h[0][2] = h0_2;
        h[1][0] = h0_1; h[2][0] = h0_2;

        for (int ii = 0; ii < 3; ii++)
            for (int jj = 0; jj < 3; jj++)
                h[ii][jj] *= dm_val * fri_vec[ii] * fri_vec[jj];

        const int indices[3] = {normal, t1, t2};
        unsigned long long nzi[3];
        double Jv[3];
#pragma unroll
        for (int r = 0; r < 3; r++) {
            nzi[r] = s_nzJ[indices[r]] & own;
            Jv[r]  = active ? Jenv[(size_t)indices[r] * nv + t] : 0.0;
        }
#pragma unroll
        for (int ci = 0; ci < 3; ci++) {
            const double Jci_t = Jv[ci];
            const bool on = active && !(fabs(Jci_t) < 1e-30);
#pragma unroll
            for (int cj = 0; cj < 3; cj++) {
                const double h_ij = h[ci][cj];
                if (fabs(h_ij) < 1e-30) continue;
                const double hJci = h_ij * Jci_t;
                for (unsigned long long m = nzi[cj]; m; m &= m - 1) {
                    const int j = __ffsll((unsigned long long)m) - 1;
                    const double Jjj = wf_broadcast_d(Jv[cj], j);
                    if (on) H_row[j] += hJci * Jjj;
                }
            }
        }
    }
}

// ── Newton v2: element-parallel sparse H build ──────────────────────────────
// J is fixed for the whole dispatch.  PREP (helper waves, once per dispatch)
// compacts it per (dof a, 64-constraint chunk w): the nonzero J[c][a] and
// D[c]*J[c][a] in ascending c, plus the chunk's nonzero mask s_cm, the
// "|D*J| >= 1e-30" mask s_cdm and the list base s_cpre.  Each Newton
// iteration then builds every LOWER element H[a][b] (b <= a; Cholesky and
// both substitutions read only the lower triangle) on ONE thread:
//     h = M[a][b];  for c ascending with state==1, D*J[c][a] kept, J[c][b]!=0:
//         h += (D[c]*J[c][a]) * J[c][b]
// -- the exact per-element term sequence and FMA shape of the column-owned
// build (it only drops terms whose product is a signed zero), so H is
// bit-identical.  The gradient walks the same compact lists.
constexpr int NCMD_PREP = 3;
constexpr int NCMD_HF   = 4;
constexpr int NCMD_NOP  = 5;
constexpr int NCMD_PREP2 = 6;
constexpr int NCMD_PREP3 = 7;
// Fused M_hat post-solve (kernel_mhat_solve_reg body) on wave 1, overlapped
// with wave 0's lambda tail.  Terminal: helpers leave the command loop.
constexpr int NCMD_MHAT  = 8;
// Early M_hat elimination (matrix part only) on wave 1 while wave 0 runs the
// first iteration's substitutions / line search.  Non-terminal.
constexpr int NCMD_MHAT_ELIM = 9;
constexpr int NCMD_LS = 10;
#ifndef NEWTON_PAR_LS
#define NEWTON_PAR_LS 1
#endif

// Line-search trial cost at step alpha.  Verbatim copy of the 3g trial body
// (same expressions, same order, same reductions) so any wave computes the
// bit-identical cost wave 0's serial loop computes for the same alpha.
template <int NV_T>
HIPGL_DEVICE __forceinline__ double newton_ls_trial_cost(
    int t, bool active, int nv_rt, int nc, int cone_type, double impr_inv, double alpha,
    const double* M_lds, const double* s_qv, const double* s_sv,
    double qacc_t, double search_t, double qfrc_t,
    const double* g_x, const double* g_jv, const bool* g_is_en, size_t env_c_base,
    const double* __restrict__ constraint_D, const double* __restrict__ constraint_lo,
    const double* __restrict__ constraint_hi,
    const int* __restrict__ constraint_friction_parent,
    const int* __restrict__ constraint_friction_sibling,
    const double* __restrict__ constraint_friction_mu)
{
    const int nv = (NV_T > 0) ? NV_T : nv_rt;
    double qi_a = active ? (qacc_t + alpha * search_t) : 0.0;
    double my_Mq_a = 0.0;
    if (active)
        for (int j = 0; j < nv; j++)
            my_Mq_a += M_lds[j] * (s_qv[j] + alpha * s_sv[j]);
    double quad_cost_a = active ? (0.5 * qi_a * my_Mq_a - qi_a * qfrc_t) : 0.0;

    double constr_cost_a = 0.0;
    for (int c = t; c < nc; c += NEWTON_BLK) {
        double xc = g_x[c] + alpha * g_jv[c];
        double Dc = constraint_D[env_c_base + c];
        double lo_c = constraint_lo[env_c_base + c];
        double hi_c = constraint_hi[env_c_base + c];
        int par_c = constraint_friction_parent[env_c_base + c];
        if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;
        double sc = 0.0;
        if (lo_c < -1e15 && hi_c > 1e15) sc = 0.5 * Dc * xc * xc;
        else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
        else if (par_c < 0) {
            double f = fabs(hi_c);
            double rf = (Dc > 1e-10) ? f / Dc : 0.0;
            double ax = fabs(xc);
            sc = (ax < rf) ? 0.5 * Dc * xc * xc : f * (ax - 0.5 * rf);
        } else { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
        constr_cost_a += sc;
    }

    double cone_cost_a = 0.0;
    if (cone_type == 1) {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int par = constraint_friction_parent[env_c_base + c];
            int sib = constraint_friction_sibling[env_c_base + c];
            if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
            if (c >= sib) continue;
            const int torsion = constraint_friction_sibling[env_c_base+par];
            if (torsion >= 0 && torsion < nc) {
                const int rows[4] = {par, c, sib, torsion};
                cone_cost_a += evaluate_contact4(rows, g_x, g_jv, alpha,
                    constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                    impr_inv, nullptr, nullptr);
                continue;
            }
            double muf = (double)constraint_friction_mu[env_c_base + c];
            double mul = muf * impr_inv;
            double xn = g_x[par] + alpha*g_jv[par];
            double xt1 = g_x[c] + alpha*g_jv[c];
            double xt2 = g_x[sib] + alpha*g_jv[sib];
            double Nn = xn*mul;
            double ut1 = xt1*muf, ut2 = xt2*muf;
            double TTl = ut1*ut1+ut2*ut2;
            double Tl = (TTl > 0) ? sqrt(TTl) : 0.0;
            if ((Nn >= mul*Tl)||(Tl<=0 && Nn>=0)) {}
            else if ((mul*Nn+Tl<=0)||(Tl<=0 && Nn<0)) {
                cone_cost_a += 0.5*constraint_D[env_c_base+par]*xn*xn
                             + 0.5*constraint_D[env_c_base+c]*xt1*xt1
                             + 0.5*constraint_D[env_c_base+sib]*xt2*xt2;
            } else {
                double D0 = constraint_D[env_c_base+par];
                double mu2l = mul*mul;
                double dml = D0/fmax(mu2l*(1+mu2l),1e-20);
                double nmt = Nn-mul*Tl;
                cone_cost_a += 0.5*dml*nmt*nmt;
            }
        }
    }
    if (cone_type == 1) {
        double rv[3] = {quad_cost_a, constr_cost_a, cone_cost_a};
        wf_reduce_sum_dn<3>(rv);
        quad_cost_a = rv[0]; constr_cost_a = rv[1]; cone_cost_a = rv[2];
    } else {
        // cone_cost_a is +0.0 on every lane, so its butterfly is +0.0 exactly.
        double rv[2] = {quad_cost_a, constr_cost_a};
        wf_reduce_sum_dn<2>(rv);
        quad_cost_a = rv[0]; constr_cost_a = rv[1];
    }
    return quad_cost_a + constr_cost_a + cone_cost_a;
}
constexpr int HB_NW     = JT_STRIDE / NEWTON_BLK;   // 64-constraint chunks (nc <= 512)
constexpr int HB_CAPW   = 512;                      // compact entries per chunk
HIPGL_DEVICE __forceinline__ int hb_capsh(int nc) {
    return (((nc + NEWTON_BLK - 1) / NEWTON_BLK) <= HB_NW / 2) ? 10 : 9;
}

HIPGL_DEVICE __forceinline__ void hf_prep(int t, int hw, int nhw, int nc, int nv,
    const double* __restrict__ JTbase, const double* __restrict__ Denv,
    unsigned long long* s_cm, unsigned long long* s_cdm, int* s_cpre,
    double* s_cJ, double* s_cDJ, int* s_bad,
    unsigned long long* s_nz = nullptr)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    const unsigned long long lt = (1ull << t) - 1ull;
    for (int w = hw; w < nw; w += nhw) {
        const int  c   = w * NEWTON_BLK + t;
        const bool inr = (c < nc);
        const double Dc = inr ? Denv[c] : 0.0;
        const double* __restrict__ Jc = JTbase + (inr ? c : 0);
        const int capsh = hb_capsh(nc);
        int off = w << capsh;
        const int lim = off + (1 << capsh);
        unsigned long long nzm = 0ull;   // row c's J sparsity (== s_nzJ[c])
        constexpr int B = 8;
        for (int a0 = 0; a0 < nv; a0 += B) {
            double Jv[B];
#pragma unroll
            for (int k = 0; k < B; k++)
                Jv[k] = (inr && a0 + k < nv) ? Jc[(size_t)(a0 + k) * JT_STRIDE] : 0.0;
#pragma unroll
            for (int k = 0; k < B; k++) {
                const int a = a0 + k;
                if (a < nv) {
                    const double J  = Jv[k];
                    const bool   nz = (J != 0.0);
                    if (nz) nzm |= (1ull << a);
                    const double dJ = Dc * J;
                    const unsigned long long m  = __ballot(nz);
                    const unsigned long long md = __ballot(nz && fabs(dJ) >= 1e-30);
                    const int pos = off + __popcll(m & lt);
                    if (nz && pos < lim) { s_cJ[pos] = J; s_cDJ[pos] = dJ; }
                    if (t == 0) {
                        s_cm[a * HB_NW + w] = m; s_cdm[a * HB_NW + w] = md;
                        s_cpre[a * HB_NW + w] = off;
                    }
                    off += __popcll(m);
                }
            }
        }
        if (t == 0 && off > lim) *s_bad = 1;
        if (s_nz && inr) s_nz[c] = nzm;
    }
}

HIPGL_DEVICE __forceinline__ bool hf_stage_r(int t, int nc,
    double fo0, int st0, double fo1, int st1,
    double* s_f, unsigned long long* s_actm, unsigned long long* s_fm,
    unsigned long long* s_4m, unsigned long long& am0, unsigned long long& am1)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    bool any4 = false;
    am0 = 0ull; am1 = 0ull;
#pragma unroll
    for (int w = 0; w < 2; w++) {
        if (w < nw) {
            const int c = w * NEWTON_BLK + t;
            const bool inr = (c < nc);
            const double fv = inr ? (w == 0 ? fo0 : fo1) : 0.0;
            const int    sv = inr ? (w == 0 ? st0 : st1) : 0;
            if (inr) s_f[c] = (fabs(fv) >= 1e-30) ? fv : 0.0;
            const unsigned long long am = __ballot(inr && sv == 1);
            const unsigned long long fm = __ballot(inr && fabs(fv) >= 1e-30);
            const unsigned long long m4 = __ballot(inr && sv == 4);
            any4 |= (m4 != 0ull);
            if (t == 0) { s_actm[w] = am; s_fm[w] = fm; s_4m[w] = m4; }
            if (w == 0) am0 = am; else am1 = am;
        }
    }
    return any4;
}

// Per-iteration staging of the forces and the state masks (wave 0).
// Returns whether any row is in the elliptic middle zone (state 4).
HIPGL_DEVICE __forceinline__ bool hf_stage(int t, int nc,
    const double* __restrict__ g_force, const int* __restrict__ g_state,
    double* s_f, unsigned long long* s_actm, unsigned long long* s_fm,
    unsigned long long* s_4m)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    double fv[HB_NW]; int sv[HB_NW];
#pragma unroll
    for (int w = 0; w < HB_NW; w++) {
        const int c = w * NEWTON_BLK + t;
        const bool inr = (w < nw) && (c < nc);
        fv[w] = inr ? g_force[c] : 0.0;
        sv[w] = inr ? g_state[c] : 0;
    }
    bool any4 = false;
#pragma unroll
    for (int w = 0; w < HB_NW; w++) {
        if (w < nw) {
            const int c = w * NEWTON_BLK + t;
            const bool inr = (c < nc);
            if (inr) s_f[c] = (fabs(fv[w]) >= 1e-30) ? fv[w] : 0.0;
            const unsigned long long am = __ballot(inr && sv[w] == 1);
            const unsigned long long fm = __ballot(inr && fabs(fv[w]) >= 1e-30);
            const unsigned long long m4 = __ballot(inr && sv[w] == 4);
            any4 |= (m4 != 0ull);
            if (t == 0) { s_actm[w] = am; s_fm[w] = fm; s_4m[w] = m4; }
        }
    }
    return any4;
}

// grad[t] -= f[c] * J[c][t] over the compact list of row t, ascending c.
HIPGL_DEVICE __forceinline__ void hf_grad(int t, int nc, double& grad_t,
    const unsigned long long* s_cm, const int* s_cpre, const double* s_cJ,
    const double* s_f, const unsigned long long* s_fm)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    for (int w = 0; w < nw; w++) {
        const unsigned long long cm = s_cm[t * HB_NW + w];
        const int p0 = s_cpre[t * HB_NW + w];
        const double* __restrict__ fw = s_f + w * NEWTON_BLK;
        for (unsigned long long m = cm & s_fm[w]; m; m &= m - 1) {
            const int l = __ffsll((unsigned long long)m) - 1;
            const double fc  = fw[l];
            const double Jct = s_cJ[p0 + __popcll(cm & ((1ull << l) - 1ull))];
            grad_t -= fc * Jct;
        }
    }
}

// Lower-triangle element build: elements e = e0, e0+step, ...
// ── Fused elliptic-cone Hessian (state-4 cones) ─────────────────────────────
// Per cone in the state-4 list, wave 0 evaluates the cone Hessian h (the same
// code as newton_cone_cols) lane-parallel into LDS; each element thread then
// adds the cone terms after its J^T D J terms, in the old per-element order
// (cone q ascending, ci, cj), with the same skip rules:
//     |h_ij| < 1e-30, |J[r_ci][a]| < 1e-30, J[r_cj][b] == 0  -> skipped.
// Elements outside the J^T D J list (M copies) get their cone terms too.
constexpr int HB_Q4CAP = 64;

HIPGL_DEVICE __forceinline__ void cone3_h(double mu_fric, double impr_inv,
    double xn, double xt1_v, double xt2_v, double D0, double* hout)
{
    double mu_l = mu_fric * impr_inv;
    double mu2_l = mu_l * mu_l;

    double un = xn * mu_l, u1v = xt1_v * mu_fric, u2v = xt2_v * mu_fric;
    double TT = u1v * u1v + u2v * u2v;
    double T_val = (TT > 0.0) ? sqrt(TT) : 1e-20;
    T_val = fmax(T_val, 1e-20);

    double dm_val = D0 / fmax(mu2_l * (1.0 + mu2_l), 1e-20);

    double u_vec[3] = {un, u1v, u2v};
    double fri_vec[3] = {mu_l, mu_fric, mu_fric};

    double coeff_outer = mu_l * un / T_val;
    double coeff_diag = mu2_l - mu_l * un / T_val;
    double h[3][3];
    for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
            h[ii][jj] = coeff_outer * (u_vec[ii] / T_val) * (u_vec[jj] / T_val)
                       + ((ii == jj) ? coeff_diag : 0.0);

    double h0_1 = -mu_l / T_val * u1v;
    double h0_2 = -mu_l / T_val * u2v;
    h[0][0] = 1.0; h[0][1] = h0_1; h[0][2] = h0_2;
    h[1][0] = h0_1; h[2][0] = h0_2;

    for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
            h[ii][jj] *= dm_val * fri_vec[ii] * fri_vec[jj];
    for (int ii = 0; ii < 3; ii++)
        for (int jj = 0; jj < 3; jj++)
            hout[ii * 4 + jj] = h[ii][jj];
}

// J[r][a] from the compact per-(dof, chunk) lists (0.0 when structurally zero).
HIPGL_DEVICE __forceinline__ double hf_jget(int r, int a,
    const unsigned long long* s_cm, const int* s_cpre, const double* s_cJ)
{
    const int w = r >> 6, l = r & 63;
    const unsigned long long m = s_cm[a * HB_NW + w];
    if (!((m >> l) & 1ull)) return 0.0;
    return s_cJ[s_cpre[a * HB_NW + w] + __popcll(m & ((1ull << l) - 1ull))];
}

// Lane-parallel cone Hessians for the state-4 list (wave 0, n4 <= HB_Q4CAP).
HIPGL_DEVICE __forceinline__ void hf_cone_prep(int t, int nc, int n4, const int* s_q4,
    const int* s_coneC, const int* s_coneP, const int* s_coneS, const int* s_coneT,
    const double* g_x, const double* Denv, const double* muenv, double impr_inv,
    const unsigned long long* s_nzJ, double* s_ch, unsigned long long* s_cU,
    unsigned long long* s_crw)
{
    for (int k = t; k < n4; k += NEWTON_BLK) {
        const int q = s_q4[k];
        const int c = s_coneC[q], normal = s_coneP[q], sib = s_coneS[q], torsion = s_coneT[q];
        double h[16];
        unsigned long long U = s_nzJ[normal] | s_nzJ[c] | s_nzJ[sib];
        unsigned long long rw = (unsigned long long)normal | ((unsigned long long)c << 10)
                              | ((unsigned long long)sib << 20);
        if (torsion >= 0 && torsion < nc) {
            const int rows[4] = {normal, c, sib, torsion};
            evaluate_contact4(rows, g_x, nullptr, 0.0, Denv, muenv,
                              impr_inv, nullptr, nullptr, h);
            U |= s_nzJ[torsion];
            rw |= ((unsigned long long)torsion << 30) | (1ull << 40);
        } else {
            cone3_h((double)muenv[c], impr_inv, g_x[normal], g_x[c], g_x[sib],
                    Denv[normal], h);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) s_ch[k * 16 + i] = h[i];
        s_cU[k] = U;
        s_crw[k] = rw;
    }
}

// Cone terms of lower element (a,b), appended after its J^T D J terms.
HIPGL_DEVICE __forceinline__ double hf_cone_elem(int a, int b, double h, int n4,
    const double* s_ch, const unsigned long long* s_cU, const unsigned long long* s_crw,
    const unsigned long long* s_cm, const int* s_cpre, const double* s_cJ)
{
    for (int k = 0; k < n4; k++) {
        const unsigned long long U = s_cU[k];
        if (!((U >> a) & 1ull) || !((U >> b) & 1ull)) continue;
        const unsigned long long rw = s_crw[k];
        const int nr = ((rw >> 40) & 1ull) ? 4 : 3;
        int rows[4];
        rows[0] = (int)(rw & 1023); rows[1] = (int)((rw >> 10) & 1023);
        rows[2] = (int)((rw >> 20) & 1023); rows[3] = (int)((rw >> 30) & 1023);
        double Ja[4], Jb[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            Ja[r] = (r < nr) ? hf_jget(rows[r], a, s_cm, s_cpre, s_cJ) : 0.0;
            Jb[r] = (r < nr) ? hf_jget(rows[r], b, s_cm, s_cpre, s_cJ) : 0.0;
        }
        const double* hk = s_ch + k * 16;
#pragma unroll
        for (int ci = 0; ci < 4; ci++) {
            if (ci >= nr) break;
            const double Jit = Ja[ci];
            if (fabs(Jit) < 1e-30) continue;
#pragma unroll
            for (int cj = 0; cj < 4; cj++) {
                if (cj >= nr) break;
                const double hij = hk[ci * 4 + cj];
                if (fabs(hij) < 1e-30) continue;
                if (Jb[cj] == 0.0) continue;
                const double hJi = hij * Jit;
                h += hJi * Jb[cj];
            }
        }
    }
    return h;
}

HIPGL_DEVICE __forceinline__ void hf_elems(int e0, int step, int nv, int nc,
    const double* sM, int sm_s,
    const unsigned long long* s_cm, const unsigned long long* s_cdm, const int* s_cpre,
    const double* s_cJ, const double* s_cDJ, const unsigned long long* s_actm,
    double* sH, int n4 = 0, const double* s_ch = nullptr,
    const unsigned long long* s_cU = nullptr, const unsigned long long* s_crw = nullptr)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    const int ne = nv * (nv + 1) / 2;
    for (int e = e0; e < ne; e += step) {
        int a = (int)((sqrtf(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
        while (a * (a + 1) / 2 > e) a--;
        while ((a + 1) * (a + 2) / 2 <= e) a++;
        const int b = e - a * (a + 1) / 2;
        double h = sM[a * sm_s + b];
        for (int w = 0; w < nw; w++) {
            const unsigned long long cb = s_cm[b * HB_NW + w];
            unsigned long long m = s_cdm[a * HB_NW + w] & s_actm[w] & cb;
            if (!m) continue;
            const unsigned long long ca = s_cm[a * HB_NW + w];
            const double* __restrict__ pa = s_cDJ + s_cpre[a * HB_NW + w];
            const double* __restrict__ pb = s_cJ  + s_cpre[b * HB_NW + w];
            for (; m; m &= m - 1) {
                const int l = __ffsll((unsigned long long)m) - 1;
                const unsigned long long lo = (1ull << l) - 1ull;
                h += pa[__popcll(ca & lo)] * pb[__popcll(cb & lo)];
            }
        }
        if (n4 > 0) h = hf_cone_elem(a, b, h, n4, s_ch, s_cU, s_crw, s_cm, s_cpre, s_cJ);
        sH[a * LDS_STRIDE + b] = h;
    }
}

// Element list (built once per dispatch by ONE helper wave after the
// compaction): every structurally nonzero lower element gets its (J[c][a]*D,
// J[c][b]) operand pairs precomputed as one packed word per pair, ascending c:
//   bits 0-2 chunk w | 3-11 offset of D*J[c][a] | 12-20 offset of J[c][b] | 21-26 c&63
// so the per-iteration build is a straight, unrollable walk whose loads do not
// depend on the accumulator.  Structurally zero elements are a bitmap; they are
// just H[a][b] = M[a][b].
constexpr int HB_ECAP = 384;     // nonzero lower elements
constexpr int HB_PCAP = 4096;    // operand pairs
constexpr int HB_EBM  = (KERNEL_MAXNV * (KERNEL_MAXNV + 1) / 2 + 63) / 64;

HIPGL_DEVICE __forceinline__ void hf_tri_decode(int e, int& a, int& b) {
    int r = (int)((sqrtf(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
    while (r * (r + 1) / 2 > e) r--;
    while ((r + 1) * (r + 2) / 2 <= e) r++;
    a = r; b = e - r * (r + 1) / 2;
}

HIPGL_DEVICE __forceinline__ void hf_list(int t, int nc, int nv,
    const unsigned long long* s_cm, const unsigned long long* s_cdm, const int* s_cpre,
    unsigned long long* s_ebm, int2* s_el, unsigned int* s_pair, int* s_nel, int* s_bad)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    const int ne = nv * (nv + 1) / 2;
    const unsigned long long lt = (1ull << t) - 1ull;
    int nel = 0, npair = 0;
    for (int e0 = 0; e0 < ne; e0 += NEWTON_BLK) {
        const int e = e0 + t;
        const bool valid = (e < ne);
        int a = 0, b = 0, cnt = 0;
        if (valid) {
            hf_tri_decode(e, a, b);
            for (int w = 0; w < nw; w++)
                cnt += __popcll(s_cdm[a * HB_NW + w] & s_cm[b * HB_NW + w]);
        }
        const unsigned long long bm = __ballot(valid && cnt > 0);
        if (t == 0) s_ebm[e0 / NEWTON_BLK] = bm;
        int incl = cnt;
        for (int off = 1; off < NEWTON_BLK; off <<= 1) {
            const int y = __shfl_up(incl, off);
            if (t >= off) incl += y;
        }
        const int k = nel + __popcll(bm & lt);
        const int start = npair + incl - cnt;
        if (cnt > 0 && k < HB_ECAP && start + cnt <= HB_PCAP) {
            s_el[k] = make_int2(start | (cnt << 16), a | (b << 8));
            int p = start;
            for (int w = 0; w < nw; w++) {
                const unsigned long long ca = s_cm[a * HB_NW + w];
                const unsigned long long cb = s_cm[b * HB_NW + w];
                const int oa = s_cpre[a * HB_NW + w] - (w << hb_capsh(nc));
                const int ob = s_cpre[b * HB_NW + w] - (w << hb_capsh(nc));
                for (unsigned long long m = s_cdm[a * HB_NW + w] & cb; m; m &= m - 1) {
                    const int l = __ffsll((unsigned long long)m) - 1;
                    const unsigned long long lo = (1ull << l) - 1ull;
                    s_pair[p++] = (unsigned)w | ((unsigned)(oa + __popcll(ca & lo)) << 3)
                                | ((unsigned)(ob + __popcll(cb & lo)) << 13) | ((unsigned)l << 23);
                }
            }
        }
        nel += __popcll(bm);
        npair += __shfl(incl, NEWTON_BLK - 1);
    }
    if (t == 0) {
        *s_nel = nel;
        if (nel > HB_ECAP || npair > HB_PCAP) *s_bad = 1;
    }
}

// Multi-wave element list, in two passes separated by a block barrier:
// count (per 64-element block: nonzero bitmap, element count, pair count),
// then write (each block's offsets are the fixed-order prefix of the counts).
// Same content and layout as hf_list.
HIPGL_DEVICE __forceinline__ int hf_elem_cnt(int e, int ne, int nw,
    const unsigned long long* s_cm, const unsigned long long* s_cdm, int& a, int& b)
{
    a = 0; b = 0;
    int cnt = 0;
    if (e < ne) {
        hf_tri_decode(e, a, b);
        for (int w = 0; w < nw; w++)
            cnt += __popcll(s_cdm[a * HB_NW + w] & s_cm[b * HB_NW + w]);
    }
    return cnt;
}

HIPGL_DEVICE __forceinline__ void hf_list_count(int t, int hw, int nhw, int nc, int nv,
    const unsigned long long* s_cm, const unsigned long long* s_cdm,
    unsigned long long* s_ebm, int* s_bnel, int* s_bnp)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    const int ne = nv * (nv + 1) / 2;
    for (int eb = hw; eb * NEWTON_BLK < ne; eb += nhw) {
        int a, b;
        const int cnt = hf_elem_cnt(eb * NEWTON_BLK + t, ne, nw, s_cm, s_cdm, a, b);
        const unsigned long long bm = __ballot(cnt > 0);
        int sum = cnt;
        for (int off = 32; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
        if (t == 0) { s_ebm[eb] = bm; s_bnel[eb] = __popcll(bm); s_bnp[eb] = sum; }
    }
}

HIPGL_DEVICE __forceinline__ void hf_list_write(int t, int hw, int nhw, int nc, int nv,
    const unsigned long long* s_cm, const unsigned long long* s_cdm, const int* s_cpre,
    const unsigned long long* s_ebm, const int* s_bnel, const int* s_bnp,
    int2* s_el, unsigned int* s_pair, int* s_nel, int* s_bad)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    const int ne = nv * (nv + 1) / 2;
    const int nb = (ne + NEWTON_BLK - 1) / NEWTON_BLK;
    const unsigned long long lt = (1ull << t) - 1ull;
    for (int eb = hw; eb < nb; eb += nhw) {
        int nel = 0, npair = 0;
        for (int i = 0; i < eb; i++) { nel += s_bnel[i]; npair += s_bnp[i]; }
        int a, b;
        const int cnt = hf_elem_cnt(eb * NEWTON_BLK + t, ne, nw, s_cm, s_cdm, a, b);
        const unsigned long long bm = s_ebm[eb];
        int incl = cnt;
        for (int off = 1; off < NEWTON_BLK; off <<= 1) {
            const int y = __shfl_up(incl, off);
            if (t >= off) incl += y;
        }
        const int k = nel + __popcll(bm & lt);
        const int start = npair + incl - cnt;
        if (cnt > 0 && k < HB_ECAP && start + cnt <= HB_PCAP) {
            s_el[k] = make_int2(start | (cnt << 16), a | (b << 8));
            int p = start;
            for (int w = 0; w < nw; w++) {
                const unsigned long long ca = s_cm[a * HB_NW + w];
                const unsigned long long cb = s_cm[b * HB_NW + w];
                const int oa = s_cpre[a * HB_NW + w] - (w << hb_capsh(nc));
                const int ob = s_cpre[b * HB_NW + w] - (w << hb_capsh(nc));
                for (unsigned long long m = s_cdm[a * HB_NW + w] & cb; m; m &= m - 1) {
                    const int l = __ffsll((unsigned long long)m) - 1;
                    const unsigned long long lo = (1ull << l) - 1ull;
                    s_pair[p++] = (unsigned)w | ((unsigned)(oa + __popcll(ca & lo)) << 3)
                                | ((unsigned)(ob + __popcll(cb & lo)) << 13) | ((unsigned)l << 23);
                }
            }
        }
    }
    if (hw == 0 && t == 0) {
        int nel = 0, npair = 0;
        for (int i = 0; i < nb; i++) { nel += s_bnel[i]; npair += s_bnp[i]; }
        *s_nel = nel;
        if (nel > HB_ECAP || npair > HB_PCAP) *s_bad = 1;
    }
}

HIPGL_DEVICE __forceinline__ double hf_pair(unsigned int p,
    const double* s_cJ, const double* s_cDJ, const unsigned long long* s_actm, double h,
    int capsh)
{
    const int w = p & 7, oa = (p >> 3) & 1023, ob = (p >> 13) & 1023, l = (p >> 23) & 63;
    const bool act = (s_actm[w] >> l) & 1ull;
    const double dJ = s_cDJ[(w << capsh) + oa];
    const double Jb = s_cJ[(w << capsh) + ob];
    const double d  = act ? dJ : 0.0;
    h += d * Jb;
    return h;
}

// List-driven element build, threads tid = 0..nthr-1.
HIPGL_DEVICE __forceinline__ void hf_elems_list(int tid, int nthr, int nv, int capsh,
    const double* sM, int sm_s, const unsigned long long* s_ebm,
    const int2* s_el, int nel, const unsigned int* s_pair,
    const double* s_cJ, const double* s_cDJ, const unsigned long long* s_actm,
    double* sH, int n4 = 0, const double* s_ch = nullptr,
    const unsigned long long* s_cU = nullptr, const unsigned long long* s_crw = nullptr,
    const unsigned long long* s_cm = nullptr, const int* s_cpre = nullptr)
{
    const int ne = nv * (nv + 1) / 2;
    for (int e = tid; e < ne; e += nthr) {
        if ((s_ebm[e / NEWTON_BLK] >> (e % NEWTON_BLK)) & 1ull) continue;
        int a, b; hf_tri_decode(e, a, b);
        double h = sM[a * sm_s + b];
        if (n4 > 0) h = hf_cone_elem(a, b, h, n4, s_ch, s_cU, s_crw, s_cm, s_cpre, s_cJ);
        sH[a * LDS_STRIDE + b] = h;
    }
    for (int k = tid; k < nel; k += nthr) {
        const int2 el = s_el[k];
        const int start = el.x & 0xffff, cnt = el.x >> 16;
        const int a = el.y & 0xff, b = el.y >> 8;
        double h = sM[a * sm_s + b];
        const unsigned int* __restrict__ pp = s_pair + start;
        int i = 0;
        for (; i + 4 <= cnt; i += 4) {
            const unsigned int p0 = pp[i], p1 = pp[i + 1], p2 = pp[i + 2], p3 = pp[i + 3];
            const double d0 = ((s_actm[p0 & 7] >> ((p0 >> 23) & 63)) & 1ull) ? s_cDJ[((p0 & 7) << capsh) + ((p0 >> 3) & 1023)] : 0.0;
            const double d1 = ((s_actm[p1 & 7] >> ((p1 >> 23) & 63)) & 1ull) ? s_cDJ[((p1 & 7) << capsh) + ((p1 >> 3) & 1023)] : 0.0;
            const double d2 = ((s_actm[p2 & 7] >> ((p2 >> 23) & 63)) & 1ull) ? s_cDJ[((p2 & 7) << capsh) + ((p2 >> 3) & 1023)] : 0.0;
            const double d3 = ((s_actm[p3 & 7] >> ((p3 >> 23) & 63)) & 1ull) ? s_cDJ[((p3 & 7) << capsh) + ((p3 >> 3) & 1023)] : 0.0;
            const double j0 = s_cJ[((p0 & 7) << capsh) + ((p0 >> 13) & 1023)];
            const double j1 = s_cJ[((p1 & 7) << capsh) + ((p1 >> 13) & 1023)];
            const double j2 = s_cJ[((p2 & 7) << capsh) + ((p2 >> 13) & 1023)];
            const double j3 = s_cJ[((p3 & 7) << capsh) + ((p3 >> 13) & 1023)];
            h += d0 * j0;
            h += d1 * j1;
            h += d2 * j2;
            h += d3 * j3;
        }
        for (; i < cnt; i++) h = hf_pair(pp[i], s_cJ, s_cDJ, s_actm, h, capsh);
        if (n4 > 0) h = hf_cone_elem(a, b, h, n4, s_ch, s_cU, s_crw, s_cm, s_cpre, s_cJ);
        sH[a * LDS_STRIDE + b] = h;
    }
}

// Gradient over row t's contiguous compact lists; s_f is pre-zeroed where
// |f| < 1e-30, so those terms only subtract a signed zero.
HIPGL_DEVICE __forceinline__ void hf_grad2(int t, int nc, double& grad_t,
    const unsigned long long* s_cm, const int* s_cpre, const double* s_cJ,
    const double* s_f)
{
    const int nw = (nc + NEWTON_BLK - 1) / NEWTON_BLK;
    for (int w = 0; w < nw; w++) {
        unsigned long long m = s_cm[t * HB_NW + w];
        const double* __restrict__ Jp = s_cJ + s_cpre[t * HB_NW + w];
        const double* __restrict__ fw = s_f + w * NEWTON_BLK;
        int i = 0;
        while (m) {
            const int l0 = __ffsll((unsigned long long)m) - 1; m &= m - 1;
            const bool v1 = (m != 0ull);
            const int l1 = v1 ? (__ffsll((unsigned long long)m) - 1) : l0; m &= m - 1;
            const bool v2 = (m != 0ull);
            const int l2 = v2 ? (__ffsll((unsigned long long)m) - 1) : l0; m &= m - 1;
            const bool v3 = (m != 0ull);
            const int l3 = v3 ? (__ffsll((unsigned long long)m) - 1) : l0; m &= m - 1;
            const double f0 = fw[l0], f1 = fw[l1], f2 = fw[l2], f3 = fw[l3];
            const double J0 = Jp[i];
            const double J1 = Jp[v1 ? i + 1 : i];
            const double J2 = Jp[v2 ? i + 2 : i];
            const double J3 = Jp[v3 ? i + 3 : i];
            grad_t -= f0 * J0;
            if (v1) grad_t -= f1 * J1;
            if (v2) grad_t -= f2 * J2;
            if (v3) grad_t -= f3 * J3;
            i += 4;
        }
    }
}

template <int CTRL>
HIPGL_DEVICE __attribute__((always_inline)) inline double mhat_dpp_d(double v)
{
    union { double d; int i[2]; } u; u.d = v;
    u.i[0] = __builtin_amdgcn_update_dpp(0, u.i[0], CTRL, 0xF, 0xF, false);
    u.i[1] = __builtin_amdgcn_update_dpp(0, u.i[1], CTRL, 0xF, 0xF, false);
    return u.d;
}

// One Gauss-Jordan elimination step on register-resident rows (see
// kernel_mhat_solve_reg).  Template recursion over K guarantees every a[]
// index is a compile-time constant (a plain #pragma unroll on the 43-trip
// outer loop exceeds the unroller's budget and demotes a[] to dynamic
// indexing).
template <int K, int NV, bool RHS = true>
HIPGL_DEVICE __attribute__((always_inline)) inline void mhat_gj_step(
    double (&a)[NV], double& rhs, int& pos, const int t, const bool active)
{
    // Pivot = candidate with max |a[K]|, ties -> LOWEST POSITION (same rule
    // as the xor tree of the LDS kernel).  Wave max via DPP (no LDS), then a
    // ballot of lanes holding the max; the tie scan is almost never taken.
    const double bv = (active && pos >= K) ? fabs(a[K]) : -1.0;
    double m = bv;
    m = fmax(m, mhat_dpp_d<0xB1>(m));    // quad_perm [1,0,3,2]
    m = fmax(m, mhat_dpp_d<0x4E>(m));    // quad_perm [2,3,0,1]
    m = fmax(m, mhat_dpp_d<0x141>(m));   // row_half_mirror
    m = fmax(m, mhat_dpp_d<0x140>(m));   // row_mirror
    m = fmax(fmax(wf_broadcast_d(m, 0),  wf_broadcast_d(m, 16)),
             fmax(wf_broadcast_d(m, 32), wf_broadcast_d(m, 48)));
    const unsigned long long ball = __ballot(bv == m);
    int pl = __builtin_amdgcn_readfirstlane(__ffsll((unsigned long long)ball) - 1);
    int ppos = __builtin_amdgcn_readlane(pos, pl);
    if (__popcll(ball) > 1) {
        for (unsigned long long b = ball & (ball - 1); b; b &= b - 1) {
            const int l = __builtin_amdgcn_readfirstlane(__ffsll((unsigned long long)b) - 1);
            const int pp = __builtin_amdgcn_readlane(pos, l);
            if (pp < ppos) { ppos = pp; pl = l; }
        }
    }
    const bool isp = (t == pl);
    if (active && pos == K) pos = ppos;
    if (isp) pos = K;

    double diag = a[K];
    if (fabs(diag) < 1e-20) diag = 1e-20;
    const double inv_d = 1.0 / diag;
    if (isp) {
        #pragma unroll
        for (int j = K + 1; j < NV; j++) a[j] = a[j] * inv_d;
        if constexpr (RHS) rhs *= inv_d;
    }
    const double rhsk = RHS ? wf_broadcast_d(rhs, pl) : 0.0;
    double pr[NV];
    #pragma unroll
    for (int j = K + 1; j < NV; j++) pr[j] = wf_broadcast_d(a[j], pl);
    if (active && !isp) {
        const double factor = a[K];
        #pragma unroll
        for (int j = K + 1; j < NV; j++) a[j] -= factor * pr[j];
        if constexpr (RHS) rhs -= factor * rhsk;
    }
    if constexpr (K + 1 < NV) mhat_gj_step<K + 1, NV, RHS>(a, rhs, pos, t, active);
}

// Body of kernel_mhat_solve_reg, shared with the Newton kernel's fused tail
// (wave 1 runs it there, reading the Newton qacc from LDS instead of global).
// Identical operand order to the standalone kernel -> bit-identical output.
template <int NV_T>
HIPGL_DEVICE __attribute__((always_inline)) inline void mhat_reg_body(
    const int env, const int t,
    const double* __restrict__ solver_M,
    const double* qa,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_damping,
    int njnt,
    double timestep,
    bool enable_damping,
    double*       __restrict__ qacc)
{
    constexpr int nv = NV_T;
    const bool active = (t < nv);

    const double* M  = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;

    double a[NV_T];
    double rhs = 0.0;
    #pragma unroll
    for (int j = 0; j < nv; j++) a[j] = 0.0;
    if (active) {
        double s = 0.0;
        const double* src = &M[t * KERNEL_MAXNV];
        #pragma unroll
        for (int j = 0; j + 1 < nv; j += 2) {
            double2 pair = *reinterpret_cast<const double2*>(&src[j]);
            a[j] = pair.x; a[j + 1] = pair.y;
            s += pair.x * qa[j] + pair.y * qa[j + 1];
        }
        if (nv & 1) {
            double mij = src[nv - 1];
            a[nv - 1] = mij;
            s += mij * qa[nv - 1];
        }
        rhs = s;
    }

    if (enable_damping && active) {
        // Same per-diagonal accumulation order as the serial t==0 loop.
        double d = 0.0;
        #pragma unroll
        for (int j = 0; j < nv; j++) if (j == t) d = a[j];
        for (int ji = 0; ji < njnt; ji++) {
            int da = jnt_dof_adr[ji];
            if (da < 0 || da >= nv) break;
            int jt = jnt_type[ji];
            int ndof = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
            if (t >= da && t < da + ndof)
                d += timestep * jnt_damping[ji];
        }
        #pragma unroll
        for (int j = 0; j < nv; j++) if (j == t) a[j] = d;
    }

    int pos = t;
    mhat_gj_step<0, NV_T>(a, rhs, pos, t, active);

    if (active)
        qacc[env * nv + pos] = rhs;
}

// ── Split form of mhat_reg_body for the Newton-fused tail ──────────────────
// The elimination of [M_hat | rhs] touches rhs only through
//   pivot:      rhs *= inv_d          (inv_d = 1 / clamp(a[K]) of the pivot row)
//   non-pivot:  rhs -= a[K] * rhs_pivot
// and neither a[K] (column K is never written after step K) nor the final
// lane->position map `pos` (the lane with pos == K was the step-K pivot and is
// never reassigned) changes after step K.  So the matrix part can run early
// (it does not depend on Newton's qacc) and the rhs sequence is replayed later
// from the final registers with the SAME operands in the SAME order.
template <int NV_T>
HIPGL_DEVICE __attribute__((always_inline)) inline void mhat_elim_matrix(
    const int env, const int t,
    const double* __restrict__ solver_M,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_damping,
    int njnt, double timestep, bool enable_damping,
    double (&a)[NV_T], int& pos)
{
    constexpr int nv = NV_T;
    const bool active = (t < nv);
    const double* M  = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    #pragma unroll
    for (int j = 0; j < nv; j++) a[j] = 0.0;
    if (active) {
        const double* src = &M[t * KERNEL_MAXNV];
        #pragma unroll
        for (int j = 0; j + 1 < nv; j += 2) {
            double2 pair = *reinterpret_cast<const double2*>(&src[j]);
            a[j] = pair.x; a[j + 1] = pair.y;
        }
        if (nv & 1) a[nv - 1] = src[nv - 1];
    }
    if (enable_damping && active) {
        double d = 0.0;
        #pragma unroll
        for (int j = 0; j < nv; j++) if (j == t) d = a[j];
        for (int ji = 0; ji < njnt; ji++) {
            int da = jnt_dof_adr[ji];
            if (da < 0 || da >= nv) break;
            int jt = jnt_type[ji];
            int ndof = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
            if (t >= da && t < da + ndof)
                d += timestep * jnt_damping[ji];
        }
        #pragma unroll
        for (int j = 0; j < nv; j++) if (j == t) a[j] = d;
    }
    pos = t;
    double rhs_dummy = 0.0;
    mhat_gj_step<0, NV_T, false>(a, rhs_dummy, pos, t, active);
}

template <int NV_T>
HIPGL_DEVICE __attribute__((always_inline)) inline void mhat_rhs_replay(
    const int env, const int t,
    const double* __restrict__ solver_M,
    const double* qa,
    const double (&a)[NV_T], const int pos,
    double* __restrict__ qacc)
{
    constexpr int nv = NV_T;
    const bool active = (t < nv);
    const double* M  = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    double rhs = 0.0;
    if (active) {
        double s = 0.0;
        const double* src = &M[t * KERNEL_MAXNV];
        #pragma unroll
        for (int j = 0; j + 1 < nv; j += 2) {
            double2 pair = *reinterpret_cast<const double2*>(&src[j]);
            s += pair.x * qa[j] + pair.y * qa[j + 1];
        }
        if (nv & 1) {
            double mij = src[nv - 1];
            s += mij * qa[nv - 1];
        }
        rhs = s;
    }
    #pragma unroll
    for (int K = 0; K < nv; K++) {
        const unsigned long long ball = __ballot(active && pos == K);
        const int pl = __builtin_amdgcn_readfirstlane(__ffsll((unsigned long long)ball) - 1);
        const bool isp = (t == pl);
        double diag = a[K];
        if (fabs(diag) < 1e-20) diag = 1e-20;
        const double inv_d = 1.0 / diag;
        if (isp) rhs *= inv_d;
        const double rhsk = wf_broadcast_d(rhs, pl);
        if (active && !isp) {
            const double factor = a[K];
            rhs -= factor * rhsk;
        }
    }
    if (active)
        qacc[env * nv + pos] = rhs;
}

struct NewtonInteg {
    int     on;
    double  dt;
    const double* jnt_range_d;
    const int*    jnt_limited;
    int     enable_limits;
    float*  qpos_f;
    float*  qvel_f;
    float*  sim_time;
    double* qpos_w;
    double* qvel_w;
    double  free_joint_damp;
    int     res_fuse;   // 1: fused 3a+3b residual pass (pyramidal)
    int     l_reuse;
    int     dot_unr;
    int     stage_reg;
};
HIPGL_DEVICE __forceinline__ void newton_integ_tail(
    const NewtonInteg& ig, int env, int njnt, int nq, int nv,
    const int* __restrict__ jnt_type, const int* __restrict__ jnt_qpos_adr,
    const int* __restrict__ jnt_dof_adr, const double* qacc)
{
    __syncthreads();
    const int tid = threadIdx.x;
    if (tid == 0 && ig.sim_time) ig.sim_time[env] += (float)ig.dt;
    for (int jid = tid; jid < njnt; jid += (int)blockDim.x)
        integrate_joint_d(jid, env, ig.dt, jnt_type, jnt_qpos_adr, jnt_dof_adr,
                          ig.jnt_range_d, ig.jnt_limited, nq, nv, ig.enable_limits != 0,
                          ig.qpos_f, ig.qvel_f, qacc, ig.qpos_w, ig.qvel_w, ig.free_joint_damp);
}

template <int NV_T>
HIPGL_GLOBAL void __launch_bounds__(NEWTON_THREADS, 1) kernel_newton_solve_t(
    int num_envs, int nv_rt,
    const double* __restrict__ solver_M,
    const double* __restrict__ solver_M_inv,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ qfrc_fluid,
    const double* __restrict__ qfrc_applied_d,
    const double* __restrict__ ctrl_d,
    int nu,
    const int*    __restrict__ actuator_jnt,
    const int*    __restrict__ actuator_type,
    const double* __restrict__ actuator_gain,
    const double* __restrict__ actuator_kv,
    const double* __restrict__ actuator_biasprm,
    const double* __restrict__ actuator_forcerange,
    const double* __restrict__ actuator_ctrlrange,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const int*   __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_damping,
    const double* __restrict__ jnt_stiffness,
    const double* __restrict__ jnt_frictionloss_d,
    int njnt,
    bool enable_damping,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    const double* __restrict__ qpos0_d,
    int nq,
    double timestep,
    const double* __restrict__ constraint_J,
    const double* __restrict__ constraint_rhs,
    const double* __restrict__ constraint_D,
    const int*   __restrict__ nconstraint,
    double*      __restrict__ qacc,
    double*      __restrict__ constraint_lam,
    const double* __restrict__ constraint_lo,
    const double* __restrict__ constraint_hi,
    const int*   __restrict__ constraint_friction_parent,
    const int*   __restrict__ constraint_friction_sibling,
    const double* __restrict__ constraint_friction_mu,
    int cone_type,
    double impratio,
    bool warm_start,
    double* __restrict__ qacc_warmstart,
    double* __restrict__ ws_x,
    double* __restrict__ ws_force,
    int*    __restrict__ ws_state,
    double* __restrict__ ws_jv,
    bool*   __restrict__ ws_is_elliptic,
    double solver_tolerance,
    int max_iterations,
    bool fuse_mhat,
    NewtonInteg integ)
{
    constexpr int NVMAX = (NV_T > 0) ? NV_T : KERNEL_MAXNV;
    const int nv = (NV_T > 0) ? NV_T : nv_rt;
    int env = blockIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;
    const int t  = threadIdx.x & (NEWTON_BLK - 1);   // logical lane
    const int wv = threadIdx.x / NEWTON_BLK;         // 0 = master, 1..3 = helpers
    bool active = (t < nv);
    if (nv > KERNEL_MAXNV) return;
    // Clamp max_iterations to compile-time upper bound
    if (max_iterations <= 0 || max_iterations > NEWTON_MAXITER)
        max_iterations = NEWTON_MAXITER;

    const double* ev_d = qvel_d + env * nv;
    const double* ep_d = qpos_d + env * nq;
    int nc = nconstraint[env];
    if (nc > MAX_CONSTRAINTS) nc = MAX_CONSTRAINTS;
    size_t env_j_base = (size_t)env * MAX_CONSTRAINTS * nv;
    size_t env_c_base = (size_t)env * MAX_CONSTRAINTS;
    // Column-major mirror of J built by kernel_transpose_J (see JT_STRIDE above).
    // JTbase[v*JT_STRIDE + c] == constraint_J[env_j_base + c*nv + v].
    const bool          use_jt = (nc <= JT_STRIDE);
    const double* const JTbase = constraint_J + env_j_base + (size_t)JT_ROW_OFF * nv;

    double* g_x     = ws_x     + env_c_base;
    double* g_force = ws_force + env_c_base;
    int*    g_state = ws_state + env_c_base;
    double* g_jv    = ws_jv    + env_c_base;
    bool*   g_is_en = ws_is_elliptic + env_c_base;

    __shared__ double s_act_force[128];
    __shared__ int    s_act_dof[128];

    // LDS for Cholesky factorization of Newton Hessian H.
    // Using shared memory avoids AMD VGPR spill corruption that occurs when
    // both M_row[48] and H_row[48] are live in registers during wavefront Cholesky.
    __shared__ double sH[KERNEL_MAXNV * LDS_STRIDE];
    __shared__ double s_y[KERNEL_MAXNV];
    __shared__ double s_srch[KERNEL_MAXNV];
    // Wave-uniform staging for the two vectors every J-row dot product needs.
    // These replace `wf_broadcast_d(x, v)` inside those loops: a readlane needs a
    // WAVE-UNIFORM lane index, and the sparsity iteration below makes v per-lane,
    // so the vector has to come from LDS (a divergent index is free there).
    __shared__ double s_qv[KERNEL_MAXNV];    // current qacc / warm-start candidate
    __shared__ double s_sv[KERNEL_MAXNV];    // search direction / smooth qacc
    // Per-constraint nonzero pattern of J, computed ONCE per dispatch. The rows
    // touch only the dofs of the bodies involved, so this is ~1/4 of nv.
    __shared__ unsigned long long s_nzJ[NZ_CACHE];
    // Owner rows of each elliptic cone group, compacted ONCE per dispatch.
    // constraint_friction_parent/sibling are dispatch-constant, so the cone
    // Hessian's `for (c = 0; c < nc; c++)` scan was re-reading 2 * nc uniform
    // global words every Newton iteration only to discard ~3/4 of them.
    static constexpr int CONE_CAP = 256;
    __shared__ int s_coneC[CONE_CAP], s_coneP[CONE_CAP];
    __shared__ int s_coneS[CONE_CAP], s_coneT[CONE_CAP];


    // ── M_row lives in LDS, not in 2*NVMAX VGPRs ────────────────────────────
    // Row t of the mass matrix was a private double[NVMAX] held live across the
    // ENTIRE kernel: 86 VGPRs at nv=43.  With 188 VGPRs the scheduler had no
    // room to keep the H build's J column loads in flight -- the gfx950 ISA
    // showed 130 AGPRs used as spill (627 v_accvgpr_read) and the 16-load batch
    // broken into runs of <= 5.  M_row is only read in O(nv) dot-product loops
    // that are nowhere near the H build's dependency chain, so LDS is the right
    // home for it.  Padded to an ODD stride so a same-column read across lanes
    // is bank-conflict-free (the reason LDS_STRIDE is 61 and not 60).
    constexpr bool USE_SM = (NVMAX <= 48);
    constexpr int  SM_N   = USE_SM ? NVMAX : 1;
    constexpr int  SM_S   = USE_SM ? ((NVMAX | 1) + 2) : 1;
    __shared__ double sMrow[SM_N * SM_S];
    double M_reg[USE_SM ? 1 : NVMAX];
    double* const M_lds = sMrow + (size_t)(active ? t : 0) * SM_S;
#define MR(j)    (USE_SM ? M_lds[j] : M_reg[USE_SM ? 0 : (j)])
#define MW(j,v)  do { if (USE_SM) M_lds[j] = (v); else M_reg[USE_SM ? 0 : (j)] = (v); } while(0)
    double qacc_t = 0.0;
    double qfrc_t = 0.0;

    // ── Helper waves: park on the workgroup barrier, run the command posted by
    // wave 0, repeat.  Every command runs a fixed, nv/nc-determined number of
    // barriers on BOTH sides, so the barrier sequence always matches.
    __shared__ int    s_cmd;
    __shared__ double s_cmd_arg;
    double* const s_qf = s_y;
    __shared__ double s_lsc[NEWTON_WAVES];
    constexpr bool MCH_HELP = (NV_T > 0 && NV_T <= 32 && USE_SM);
    __shared__ double sL[MCH_HELP ? NV_T * NV_T : 1];
    __shared__ double s_qw[KERNEL_MAXNV];   // warm candidate (helper-owned copy)
    __shared__ double s_wcc;                // warm constraint cost from the helper
    __shared__ int    s_ncone;
#if NEWTON_PROF
    __shared__ unsigned long long s_hprof[NEWTON_WAVES][3];
#endif
    __shared__ int    s_cone_ok;
    constexpr int HB_NA = USE_SM ? NVMAX : 1;
    constexpr int HB_NC = USE_SM ? HB_NW * HB_CAPW : 1;
    __shared__ unsigned long long s_hcm[HB_NA * HB_NW], s_hcdm[HB_NA * HB_NW];
    __shared__ int    s_hcpre[HB_NA * HB_NW];
    __shared__ double s_hcJ[HB_NC], s_hcDJ[HB_NC];
    __shared__ double s_hf[USE_SM ? JT_STRIDE : 1];
    __shared__ unsigned long long s_hactm[HB_NW], s_hfm[HB_NW];
    __shared__ unsigned long long s_hactm_prev[HB_NW];
    __shared__ int    s_any4, s_prep_bad;
    __shared__ unsigned long long s_hebm[USE_SM ? HB_EBM : 1];
    __shared__ int2   s_hel[USE_SM ? HB_ECAP : 1];
    __shared__ unsigned int s_hpair[USE_SM ? HB_PCAP : 1];
    __shared__ int    s_hnel, s_list_bad;
    __shared__ int    s_hbnel[USE_SM ? HB_EBM : 1], s_hbnp[USE_SM ? HB_EBM : 1];
    __shared__ unsigned long long s_h4m[HB_NW];
    __shared__ int    s_q4[CONE_CAP];
    __shared__ int    s_n4;
    __shared__ double s_ch[USE_SM ? HB_Q4CAP * 16 : 1];
    __shared__ unsigned long long s_cU[USE_SM ? HB_Q4CAP : 1];
    __shared__ unsigned long long s_crw[USE_SM ? HB_Q4CAP : 1];
    const unsigned long long col_own = newton_col_mask(wv);
    if (wv != 0) {
        const double impr_inv_h = 1.0 / sqrt(fmax((double)impratio, 1e-6));
        constexpr int MH_N = (NV_T > 0 && NV_T <= 64) ? NV_T : 1;
        double mh_a[MH_N];
        int  mh_pos = 0;
        bool mh_done = false;
        for (;;) {
            BSYNC();
            const int cmd = s_cmd;
            if (cmd == NCMD_CHOL) {
                mw_chol_factor_lds(sH, nv, t, wv, s_cmd_arg);
            } else if (cmd == NCMD_PREP) {
                const bool warm_job = (s_cmd_arg != 0.0);
                // PREP is only posted when hf_cand: s_nzJ and the cone list
                // are built here instead of on wave 0 (read after PREP2).
#if NEWTON_PROF
                unsigned long long hp0 = __builtin_amdgcn_s_memtime();
#endif
                if (wv == 1) {
                    int nco = 0;
                    if (cone_type == 1)
                        nco = newton_build_cones(t, nc, CONE_CAP,
                                                 constraint_friction_parent + env_c_base,
                                                 constraint_friction_sibling + env_c_base,
                                                 s_coneC, s_coneP, s_coneS, s_coneT);
                    if (t == 0) { s_ncone = nco; s_cone_ok = (cone_type == 1 && nco <= CONE_CAP) ? 1 : 0; }
                }
                if constexpr (MCH_HELP) {
                    if (wv == 2) {
                        // Same operands (M straight from global == wave 0's
                        // sH copy), same pivot clamp and the same rc_cholp as
                        // wave 0's own path, so L is bit-identical.
                        const double* M_base = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
                        double ra[NV_T];
                        #pragma unroll
                        for (int j = 0; j < NV_T; j++)
                            ra[j] = active ? M_base[t * KERNEL_MAXNV + j] : 0.0;
                        double m_max_diag = 0.0;
                        #pragma unroll
                        for (int j = 0; j < NV_T; j++) if (j == t) m_max_diag = ra[j];
                        for (int off = 32; off > 0; off >>= 1) {
                            double oth = wf_shfl_xor_d(m_max_diag, off);
                            if (oth > m_max_diag) m_max_diag = oth;
                        }
                        const double m_delta = fmax(m_max_diag * 1e-12, 1e-12);
                        rc_cholp<NV_T>(ra, t, m_delta);
                        if (active) {
                            #pragma unroll
                            for (int j = 0; j < NV_T; j++) sL[t * NV_T + j] = ra[j];
                        }
                    }
                }
#if NEWTON_PROF
                unsigned long long hp1 = __builtin_amdgcn_s_memtime();
#endif
                if (USE_SM)
                    hf_prep(t, wv - 1, NEWTON_WAVES - 1, nc, nv, JTbase,
                            constraint_D + env_c_base, s_hcm, s_hcdm, s_hcpre,
                            s_hcJ, s_hcDJ, &s_prep_bad,
                            (nc <= NZ_CACHE) ? s_nzJ : nullptr);
#if NEWTON_PROF
                unsigned long long hp2 = __builtin_amdgcn_s_memtime();
#endif
                if (warm_job && wv == NEWTON_WAVES - 1) {
                    double* hgx = g_x; bool* hen = g_is_en;
                    asm volatile("" : "+v"(hgx), "+v"(hen));
                    const double wcc = newton_warm_ccost_helper(t, nc, nv, cone_type, impr_inv_h,
                        env_c_base, qacc_warmstart + env * nv, s_qw, JTbase, constraint_rhs,
                        hgx, hen, constraint_D, constraint_lo, constraint_hi,
                        constraint_friction_parent, constraint_friction_sibling,
                        constraint_friction_mu);
                    if (t == 0) s_wcc = wcc;
                }
#if NEWTON_PROF
                unsigned long long hp3 = __builtin_amdgcn_s_memtime();
                if (t == 0) { s_hprof[wv][0] = hp1 - hp0; s_hprof[wv][1] = hp2 - hp1; s_hprof[wv][2] = hp3 - hp2; }
#endif
            } else if (cmd == NCMD_PREP2) {
                if (USE_SM && s_prep_bad == 0)
                    hf_list_count(t, wv - 1, NEWTON_WAVES - 1, nc, nv, s_hcm, s_hcdm,
                                  s_hebm, s_hbnel, s_hbnp);
            } else if (cmd == NCMD_PREP3) {
                if (USE_SM && s_prep_bad == 0)
                    hf_list_write(t, wv - 1, NEWTON_WAVES - 1, nc, nv, s_hcm, s_hcdm, s_hcpre,
                                  s_hebm, s_hbnel, s_hbnp, s_hel, s_hpair, &s_hnel, &s_list_bad);
            } else if (cmd == NCMD_NOP) {
            } else if (cmd == NCMD_HF) {
                if (USE_SM) {
                    if (s_list_bad == 0)
                        hf_elems_list(threadIdx.x - NEWTON_BLK, NEWTON_THREADS - NEWTON_BLK, nv, hb_capsh(nc),
                                      sMrow, SM_S, s_hebm, s_hel, s_hnel, s_hpair,
                                      s_hcJ, s_hcDJ, s_hactm, sH,
                                      s_any4 == 3 ? s_n4 : 0, s_ch, s_cU, s_crw,
                                      s_hcm, s_hcpre);
                    else
                        hf_elems(threadIdx.x - NEWTON_BLK, NEWTON_THREADS - NEWTON_BLK, nv, nc,
                                 sMrow, SM_S, s_hcm, s_hcdm, s_hcpre, s_hcJ, s_hcDJ, s_hactm, sH,
                                 s_any4 == 3 ? s_n4 : 0, s_ch, s_cU, s_crw);
                }
                BSYNC();
                if (s_any4 == 1 || s_any4 == 2) {
                    const double* hx = g_x; const int* hs = g_state;
                    asm volatile("" : "+v"(hx), "+v"(hs));
                    double* const Hr = sH + (size_t)(active ? t : 0) * LDS_STRIDE;
                    newton_cone_cols(t, active, nc, nv, s_cone_ok != 0, s_ncone,
                        s_coneC, s_coneP, s_coneS, s_coneT,
                        constraint_friction_parent + env_c_base,
                        constraint_friction_sibling + env_c_base,
                        hx, hs, constraint_D + env_c_base,
                        constraint_friction_mu + env_c_base, impr_inv_h,
                        s_nzJ, constraint_J + env_j_base, Hr, col_own,
                        s_any4 == 2 ? s_q4 : nullptr, s_n4);
                    BSYNC();
                }
            } else if (cmd == NCMD_H) {
                // g_x/g_force/g_state are written by wave 0 only.  No store to
                // them is reachable on the helper path, so the uniform-value
                // annotator would mark uniform-address loads of them noclobber
                // and serve them from the (non-coherent) scalar cache.  Passing
                // the bases through a VGPR forces coherent vector loads.
                const double* hx = g_x; const double* hf = g_force; const int* hs = g_state;
                asm volatile("" : "+v"(hx), "+v"(hf), "+v"(hs));
                double dummy = 0.0;
                double* const Hr = sH + (size_t)(active ? t : 0) * LDS_STRIDE;
                newton_hbuild_cols<16>(t, active, nc, nv, use_jt, JTbase,
                    constraint_J + env_j_base + t, hf, hs,
                    constraint_D + env_c_base, s_nzJ, Hr, col_own, false, dummy);
                if (cone_type == 1)
                    newton_cone_cols(t, active, nc, nv, s_cone_ok != 0, s_ncone,
                        s_coneC, s_coneP, s_coneS, s_coneT,
                        constraint_friction_parent + env_c_base,
                        constraint_friction_sibling + env_c_base,
                        hx, hs, constraint_D + env_c_base,
                        constraint_friction_mu + env_c_base, impr_inv_h,
                        s_nzJ, constraint_J + env_j_base, Hr, col_own);
                BSYNC();
            } else if (cmd == NCMD_MHAT) {
                // Newton's final qacc is in s_qv (written by wave 0 before the
                // post barrier).  Same arithmetic as kernel_mhat_solve_reg.
                if constexpr (NV_T > 0 && NV_T <= 64) {
                    if (wv == 1) {
                        if (!mh_done)
                            mhat_elim_matrix<NV_T>(env, t, solver_M, jnt_type, jnt_dof_adr,
                                                   jnt_damping, njnt, timestep, enable_damping,
                                                   mh_a, mh_pos);
                        mhat_rhs_replay<NV_T>(env, t, solver_M, s_qv, mh_a, mh_pos, qacc);
                    }
                }
                break;
            } else if (cmd == NCMD_LS) {
                // g_x / g_jv are written by wave 0 only: keep the loads coherent
                // vector loads (see NCMD_H).
                const double* hx = g_x; const double* hjv = g_jv; const bool* hen = g_is_en;
                asm volatile("" : "+v"(hx), "+v"(hjv), "+v"(hen));
                double a = s_cmd_arg;
                for (int i = 0; i < wv; i++) a *= 0.5;
                const double impr_inv_l = 1.0 / sqrt(fmax((double)impratio, 1e-6));
                const double ca = newton_ls_trial_cost<NV_T>(t, active, nv, nc, cone_type, impr_inv_l, a,
                    M_lds, s_qv, s_sv, active ? s_qv[t] : 0.0, active ? s_sv[t] : 0.0,
                    active ? s_qf[t] : 0.0, hx, hjv, hen, env_c_base,
                    constraint_D, constraint_lo, constraint_hi,
                    constraint_friction_parent, constraint_friction_sibling,
                    constraint_friction_mu);
                if (t == 0) s_lsc[wv] = ca;
                BSYNC();
            } else if (cmd == NCMD_MHAT_ELIM) {
                if constexpr (NV_T > 0 && NV_T <= 64) {
                    if (wv == 1 && !mh_done) {
                        mhat_elim_matrix<NV_T>(env, t, solver_M, jnt_type, jnt_dof_adr,
                                               jnt_damping, njnt, timestep, enable_damping,
                                               mh_a, mh_pos);
                        mh_done = true;
                    }
                }
            } else {
                break;
            }
        }
        if (integ.on) newton_integ_tail(integ, env, njnt, nq, nv, jnt_type, jnt_qpos_adr, jnt_dof_adr, qacc);
        return;
    }
#define NPOST(cmd, arg) do { if (t == 0) { s_cmd = (cmd); s_cmd_arg = (arg); } BSYNC(); } while (0)

    // PREP (J compaction) depends only on J and D: start it on the helpers
    // right away, overlapping the M load / force accumulation below.
#if NEWTON_PROF
    unsigned long long np_acc[10] = {0,0,0,0,0,0,0,0,0,0};
    unsigned long long np_pre[8] = {0,0,0,0,0,0,0,0};
    unsigned long long np_last = __builtin_amdgcn_s_memtime();
    const unsigned long long np_t0 = np_last;
    int np_ls = 0;
    unsigned long long np_x[6] = {0,0,0,0,0,0};
    unsigned long long np_xl = 0;
#endif
    const bool hf_cand = USE_SM && use_jt && (nc > 0);
    // Warm candidate's residual/cost goes to a helper during PREP (see
    // newton_warm_ccost_helper); needs the PREP2 barrier, i.e. hf_cand.
    const bool warm_off = hf_cand && warm_start && (qacc_warmstart != nullptr);
    if (hf_cand) {
        if (t == 0) { s_prep_bad = 0; s_list_bad = 0; }
        NPOST(NCMD_PREP, warm_off ? 1.0 : 0.0);
    }
    NPQ(0);
    // Cooperative coalesced M load: all threads stride across the nv*nv block
    {
        const double* M_base = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
        if (t < nv) {
            int row = 0;
            for (; row + 8 <= nv; row += 8) {
                double m0 = M_base[(row+0) * KERNEL_MAXNV + t], m1 = M_base[(row+1) * KERNEL_MAXNV + t];
                double m2 = M_base[(row+2) * KERNEL_MAXNV + t], m3 = M_base[(row+3) * KERNEL_MAXNV + t];
                double m4 = M_base[(row+4) * KERNEL_MAXNV + t], m5 = M_base[(row+5) * KERNEL_MAXNV + t];
                double m6 = M_base[(row+6) * KERNEL_MAXNV + t], m7 = M_base[(row+7) * KERNEL_MAXNV + t];
                sH[(row+0) * LDS_STRIDE + t] = m0; sH[(row+1) * LDS_STRIDE + t] = m1;
                sH[(row+2) * LDS_STRIDE + t] = m2; sH[(row+3) * LDS_STRIDE + t] = m3;
                sH[(row+4) * LDS_STRIDE + t] = m4; sH[(row+5) * LDS_STRIDE + t] = m5;
                sH[(row+6) * LDS_STRIDE + t] = m6; sH[(row+7) * LDS_STRIDE + t] = m7;
            }
            for (; row < nv; row++)
                sH[row * LDS_STRIDE + t] = M_base[row * KERNEL_MAXNV + t];
        }
    }
    // Joint table preload: lane ji holds joint ji's dof address / dof count so
    // the per-dof joint search below walks registers (readlane, uniform index)
    // instead of a serial chain of dependent global loads. Same match, same
    // break, same arithmetic -> bit-identical.
    const bool jt_reg = (njnt <= NEWTON_BLK);
    int jda_l = -1, jnd_l = 0;
    if (jt_reg && t < njnt) {
        jda_l = jnt_dof_adr[t];
        int jtl = jnt_type[t];
        jnd_l = (jtl == JOINT_FREE) ? 6 : (jtl == JOINT_BALL) ? 3 : 1;
    }
    WSYNC();
    NPQ(1);
    if (active) {
        for (int j = 0; j < nv; j++)
            MW(j, sH[t * LDS_STRIDE + j]);

        // RHS force accumulation: gravity/coriolis + applied + fluid + actuators + damping.
        // MuJoCo includes damping as explicit force (-D*qvel) in qfrc_smooth.
        double force = -qfrc_bias[env * nv + t] + qfrc_applied_d[env * nv + t];
        if (qfrc_fluid) force += qfrc_fluid[env * nv + t];
        if (jt_reg) {
            // Find this lane's joint with register readlanes only (same first
            // match, same invalid-address break), then do its loads once:
            // the per-joint global round trips were a serial chain of njnt
            // memory latencies.  Same operands, same op order.
            int jm = -1, jmda = 0;
            for (int ji = 0; ji < njnt; ji++) {
                const int da   = __builtin_amdgcn_readlane(jda_l, ji);
                const int ndof = __builtin_amdgcn_readlane(jnd_l, ji);
                if (da < 0 || da >= nv) break;
                if (jm < 0 && t >= da && t < da + ndof) { jm = ji; jmda = da; }
            }
            if (jm >= 0) {
                if (enable_damping && jnt_damping)
                    force -= jnt_damping[jm] * ev_d[t];
                if (jnt_stiffness && qpos0_d) {
                    double stiff = jnt_stiffness[jm];
                    if (stiff != 0.0) {
                        int qa = jnt_qpos_adr[jm];
                        force -= stiff * (ep_d[qa + (t - jmda)] - qpos0_d[qa + (t - jmda)]);
                    }
                }
            }
        } else
        for (int ji = 0; ji < njnt; ji++) {
            int da, ndof;
            if (jt_reg) {
                da   = __builtin_amdgcn_readlane(jda_l, ji);
                ndof = __builtin_amdgcn_readlane(jnd_l, ji);
                if (da < 0 || da >= nv) break;
            } else {
                da = jnt_dof_adr[ji];
                if (da < 0 || da >= nv) break;   // defensive only; njnt is the real bound
                int jt = jnt_type[ji];
                ndof = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
            }
            if (t >= da && t < da + ndof) {
                // qfrc_passive = -damping * qvel - stiffness * (qpos - qpos0)
                if (enable_damping && jnt_damping)
                    force -= jnt_damping[ji] * ev_d[t];
                if (jnt_stiffness && qpos0_d) {
                    double stiff = jnt_stiffness[ji];
                    if (stiff != 0.0) {
                        int qa = jnt_qpos_adr[ji];
                        force -= stiff * (ep_d[qa + (t - da)] - qpos0_d[qa + (t - da)]);
                    }
                }
                break;
            }
        }
        qfrc_t = force;
    }

    // Actuator forces: thread 0 builds in double, broadcasts contributions
    // Lane-parallel: each actuator's force is independent (same arithmetic as
    // the old serial t==0 loop); the per-dof sum below keeps actuator order.
    {
        for (int a = t; a < nu; a += NEWTON_BLK) {
            int jid = actuator_jnt[a];
            int da = jnt_dof_adr[jid];
            double c = ctrl_d[env * nu + a];
            if (actuator_ctrlrange) {
                double cl = actuator_ctrlrange[a*2], ch = actuator_ctrlrange[a*2+1];
                if (cl < ch) c = fmax(cl, fmin(ch, c));
            }
            int atype = actuator_type ? actuator_type[a] : ACTUATOR_MOTOR;
            double gain = actuator_gain[a];
            double vel_da = ev_d[da];
            double act_force = 0.0;
            if (atype == ACTUATOR_MOTOR) act_force = gain * c;
            else if (atype == ACTUATOR_POSITION) {
                int qa = jnt_qpos_adr[jid];
                double pos_qa = ep_d[qa];
                act_force = gain * (c - pos_qa) - (actuator_kv ? actuator_kv[a] : 0.0) * vel_da;
            }
            else if (atype == ACTUATOR_VELOCITY) act_force = gain * (c - vel_da);
            else if (atype == ACTUATOR_GENERAL && actuator_biasprm) {
                int qa = jnt_qpos_adr[jid];
                double pos_qa = ep_d[qa];
                act_force = gain * c + actuator_biasprm[a*3] + actuator_biasprm[a*3+1] * pos_qa + actuator_biasprm[a*3+2] * vel_da;
            }
            if (actuator_forcerange) {
                double fl = actuator_forcerange[a*2], fh = actuator_forcerange[a*2+1];
                if (fl < fh) act_force = fmax(fl, fmin(fh, act_force));
            }
            s_act_force[a] = act_force;
            s_act_dof[a] = da;
        }
    }
    WSYNC();
    if (active) {
        for (int a = 0; a < nu; a++) {
            if (s_act_dof[a] == t) qfrc_t += s_act_force[a];
        }
    }
    WSYNC();

    double* const H_row = sH + (size_t)(active ? t : 0) * LDS_STRIDE;
    NPQ(2);

    double qacc_smooth_t = 0.0;
    const bool mch_off = MCH_HELP && hf_cand;   // wave 2 factors M during PREP
    unsigned long long mw_rp = 0ull, mw_cp = 0ull;
    {
        // ── Cholesky factorize M → L, wave-cooperatively, in LDS ──────────────
        // sH still holds M row-wise from the cooperative load above, so the
        // factorization runs there. The register form this replaces rebuilt row
        // k one `wf_broadcast_d` per element (~2700 wave-wide readlanes with a
        // single useful lane each) and indexed H_row[] -- which lives in
        // scratch, 976 B/lane -- once per inner-loop step.
        if (!mch_off) {
            double m_max_diag = active ? sH[t * LDS_STRIDE + t] : 0.0;
            for (int off = 32; off > 0; off >>= 1) {
                double oth = wf_shfl_xor_d(m_max_diag, off);
                if (oth > m_max_diag) m_max_diag = oth;
            }
            double m_delta = fmax(m_max_diag * 1e-12, 1e-12);

            // Single wave, sparse rows, no workgroup barriers: the helpers keep
            // running PREP (J compaction) underneath.
            mw_rp = 0ull; mw_cp = 0ull;
            if constexpr (NV_T > 0 && NV_T <= NEWTON_BLK) {
                double ra[NV_T];
                #pragma unroll
                for (int j = 0; j < NV_T; j++)
                    ra[j] = active ? sH[t * LDS_STRIDE + j] : 0.0;
                rc_cholp<NV_T>(ra, t, m_delta);
                const double y_t = rc_tri_fwd<NV_T>(ra, t, active, active ? qfrc_t : 0.0);
                if (active) {
                    #pragma unroll
                    for (int j = 0; j < NV_T; j++)
                        sH[t * LDS_STRIDE + j] = ra[j];
                }
                WSYNC();
                double rcol[NV_T];
                #pragma unroll
                for (int i = 0; i < NV_T; i++)
                    rcol[i] = active ? sH[i * LDS_STRIDE + t] : 0.0;
                qacc_t = active ? rc_tri_bwd<NV_T>(rcol, t, active, y_t) : 0.0;
#if R4_CHK
                {
                    WSYNC();
                    const double* M_base = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
                    if (active) for (int row = 0; row < nv; row++)
                        sH[row * LDS_STRIDE + t] = M_base[row * KERNEL_MAXNV + t];
                    WSYNC();
                    unsigned long long rp = 0, cp = 0;
                    wf_chol_factor_lds_sp(sH, nv, t, m_delta, rp, cp);
                    const bool lf = wf_lvl_ok(nv, t, active, rp, true);
                    const bool lb = wf_lvl_ok(nv, t, active, cp, false);
                    double y2 = lf ? wf_tri_fwd_lvl(sH, s_y, nv, t, active, active ? qfrc_t : 0.0, rp)
                                   : wf_tri_solve_fwd_lds(sH, nv, t, active, active ? qfrc_t : 0.0);
                    WSYNC();
                    double q2 = active ? (lb ? wf_tri_bwd_lvl(sH, s_y, nv, t, active, y2, cp)
                                             : wf_tri_solve_bwd_lds(sH, nv, t, active, y2)) : 0.0;
                    if (active && __double_as_longlong(q2) != __double_as_longlong(qacc_t))
                        printf("R4CHK MISMATCH env=%d t=%d %.17g %.17g\n", env, t, q2, qacc_t);
                    if (active && t == 0 && env == 0) printf("R4CHK ok-probe %.17g\n", qacc_t);
                    WSYNC();
                }
#endif
            } else
            wf_chol_factor_lds_sp(sH, nv, t, m_delta, mw_rp, mw_cp);
        }
        NPQ(3);

        // ── Solve M * qacc = qfrc_smooth via L L^T ──
        if (!(NV_T > 0 && NV_T <= NEWTON_BLK)) {
            const bool lf = wf_lvl_ok(nv, t, active, mw_rp, true);
            const bool lb = wf_lvl_ok(nv, t, active, mw_cp, false);
            double y_t = lf ? wf_tri_fwd_lvl(sH, s_y, nv, t, active, active ? qfrc_t : 0.0, mw_rp)
                            : wf_tri_solve_fwd_lds(sH, nv, t, active, active ? qfrc_t : 0.0);
            WSYNC();
            qacc_t = active ? (lb ? wf_tri_bwd_lvl(sH, s_y, nv, t, active, y_t, mw_cp)
                                  : wf_tri_solve_bwd_lds(sH, nv, t, active, y_t)) : 0.0;
        }
        NPQ(4);
        if (hf_cand) NPOST(NCMD_PREP2, 0.0);   // PREP done -> count the element list
        if constexpr (MCH_HELP) if (mch_off) {
            // Substitutions on wave 2's factor (identical to the self path).
            double ra[NV_T];
            #pragma unroll
            for (int j = 0; j < NV_T; j++)
                ra[j] = active ? sL[t * NV_T + j] : 0.0;
            const double y_t = rc_tri_fwd<NV_T>(ra, t, active, active ? qfrc_t : 0.0);
            double rcol[NV_T];
            #pragma unroll
            for (int i = 0; i < NV_T; i++)
                rcol[i] = active ? sL[i * NV_T + t] : 0.0;
            qacc_t = active ? rc_tri_bwd<NV_T>(rcol, t, active, y_t) : 0.0;
#if R4_CHK
            {
                double rb[NV_T];
                #pragma unroll
                for (int j = 0; j < NV_T; j++)
                    rb[j] = active ? sH[t * LDS_STRIDE + j] : 0.0;
                double m_max_diag = active ? sH[t * LDS_STRIDE + t] : 0.0;
                for (int off = 32; off > 0; off >>= 1) {
                    double oth = wf_shfl_xor_d(m_max_diag, off);
                    if (oth > m_max_diag) m_max_diag = oth;
                }
                double m_delta = fmax(m_max_diag * 1e-12, 1e-12);
                rc_cholp<NV_T>(rb, t, m_delta);
                const double y2 = rc_tri_fwd<NV_T>(rb, t, active, active ? qfrc_t : 0.0);
                WSYNC();
                if (active) {
                    #pragma unroll
                    for (int j = 0; j < NV_T; j++) sH[t * LDS_STRIDE + j] = rb[j];
                }
                WSYNC();
                double rc2[NV_T];
                #pragma unroll
                for (int i = 0; i < NV_T; i++) rc2[i] = active ? sH[i * LDS_STRIDE + t] : 0.0;
                const double q2 = active ? rc_tri_bwd<NV_T>(rc2, t, active, y2) : 0.0;
                if (active && __double_as_longlong(q2) != __double_as_longlong(qacc_t))
                    printf("R4MCH MISMATCH env=%d t=%d\n", env, t);
                else if (t == 0 && env == 0) printf("R4MCH ok\n");
                WSYNC();
            }
#endif
        }
        NPQ(5);
        qacc_smooth_t = qacc_t;
    }

    // J row sparsity, built once: constraint_J is fixed for the whole dispatch,
    // so every later J-row loop iterates set bits instead of 0..nv-1.
    const unsigned long long nz_full = (nv >= 64) ? ~0ull : ((1ull << nv) - 1ull);
    const bool nz_ok = (nc <= NZ_CACHE);
    int  ncone   = 0;
    bool cone_ok = false;
    if (hf_cand) {
        // Built by helper waves 1/2 during PREP (before the PREP2 barrier).
        if (cone_type == 1) {
            ncone   = s_ncone;
            cone_ok = (ncone <= CONE_CAP);
        }
    } else {
    if (nz_ok)
        newton_build_nzJ(t, nc, nv, use_jt, JTbase, constraint_J + env_j_base, s_nzJ);
    if (cone_type == 1) {
        ncone = newton_build_cones(t, nc, CONE_CAP,
                                   constraint_friction_parent + env_c_base,
                                   constraint_friction_sibling + env_c_base,
                                   s_coneC, s_coneP, s_coneS, s_coneT);
        cone_ok = (ncone <= CONE_CAP);
    }
    if (t == 0) { s_ncone = ncone; s_cone_ok = cone_ok ? 1 : 0; }
    }
    if (hf_cand) NPOST(NCMD_PREP3, 0.0);   // counts done -> write the element list
    NPQ(6);
    // The PREP2 barrier (after the M Cholesky) completed the compaction.
    const bool prep_ok = hf_cand && (s_prep_bad == 0);

    WSYNC();

    // Cost-based warm-start: pick warmstart vs smooth by lower quadratic cost
    if (warm_start && qacc_warmstart && nc > 0) {
        double warm_t = active ? qacc_warmstart[env * nv + t] : 0.0;
        if (active) { s_qv[t] = warm_t; s_sv[t] = qacc_smooth_t; }
        WSYNC();
        double Mw = 0.0, Ms = 0.0;
        if (active) {
            for (int j = 0; j < nv; j++) {
                const double mrj = MR(j);
                Mw += mrj * s_qv[j];
                Ms += mrj * s_sv[j];
            }
        }
        double warm_cost = wf_reduce_sum_d(active ? (0.5 * warm_t * Mw - warm_t * qfrc_t) : 0.0);
        double smooth_cost = wf_reduce_sum_d(active ? (0.5 * qacc_smooth_t * Ms - qacc_smooth_t * qfrc_t) : 0.0);

        // Populate both residual workspaces before any cone reads sibling rows.
        // The first Newton iteration overwrites g_x; its search overwrites g_jv.
        const double warm_impr_inv = 1.0 / sqrt(fmax((double)impratio, 1e-6));
        if (warm_off) {
          // g_x (warm residual), g_is_en and the warm constraint cost were
          // produced by the helper before the PREP2 barrier.
          for (int c = t; c < nc; c += NEWTON_BLK) {
            double xs = -constraint_rhs[env_c_base + c];
            const double* JTc = JTbase + c;
            for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                int v = __ffsll((unsigned long long)m) - 1;
                xs += JTc[(size_t)v * JT_STRIDE] * s_sv[v];
            }
            g_jv[c] = xs;
          }
          __syncwarp();
          warm_cost += s_wcc;
        } else {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double xw = -constraint_rhs[env_c_base + c];
            double xs = -constraint_rhs[env_c_base + c];
            if (use_jt) {
                const double* JTc = JTbase + c;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    const double jv_ = JTc[(size_t)v * JT_STRIDE];
                    xw += jv_ * s_qv[v];
                    xs += jv_ * s_sv[v];
                }
            } else {
                const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    xw += Jc[v] * s_qv[v];
                    xs += Jc[v] * s_sv[v];
                }
            }
            g_x[c] = xw;
            g_jv[c] = xs;
            g_is_en[c] = false;
        }
        __syncwarp();  // release/acquire global scratch before cross-lane reads
        // A single writer marks normals, avoiding concurrent bool stores from
        // multiple friction rows of one cone. All lanes then observe the marks.
        // Lane-parallel: every writer stores the same `true`, so the result
        // is independent of which lane lands last (the old single-lane walk
        // was ~nc serially dependent global round trips).
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                if (par >= 0 && par < nc) g_is_en[par] = true;
            }
        }
        __syncwarp();  // publish the single writer's elliptic-normal marks
        warm_cost += warmstart_constraint_cost(t, nc, cone_type, warm_impr_inv, env_c_base,
            g_x, constraint_D, constraint_lo, constraint_hi, constraint_friction_parent,
            constraint_friction_sibling, constraint_friction_mu, g_is_en);
        }
        smooth_cost += warmstart_constraint_cost(t, nc, cone_type, warm_impr_inv, env_c_base,
            g_jv, constraint_D, constraint_lo, constraint_hi, constraint_friction_parent,
            constraint_friction_sibling, constraint_friction_mu, g_is_en);

        if (active)
            qacc_t = (warm_cost <= smooth_cost) ? warm_t : qacc_smooth_t;
    }

    NPQ(7);
    double* env_qacc = qacc + env * nv;
    double* env_lam = constraint_lam + env_c_base;

    // Write unconstrained qacc for ALL envs (not just env 0).
    // When nc==0 this is the final result; when nc>0 it will be overwritten
    // by the Newton loop's final write.
    if (active) {
        env_qacc[t] = qacc_t;
    }

    for (int c = t; c < nc; c += NEWTON_BLK) env_lam[c] = 0.0;

    if (nc == 0) {
        // Solve M * qacc = qfrc in-kernel using single-threaded Gauss-Jordan.
        // All threads copy their M_row into sH (LDS), thread 0 solves, all read result.
        if (active) {
            for (int j = 0; j < nv; j++)
                sH[t * LDS_STRIDE + j] = MR(j);
            s_y[t] = qfrc_t;
        }
        WSYNC();

        if (t == 0) {
            for (int k = 0; k < nv; k++) {
                int pivot = k;
                double max_val = fabs(sH[k * LDS_STRIDE + k]);
                for (int i = k + 1; i < nv; i++) {
                    double v = fabs(sH[i * LDS_STRIDE + k]);
                    if (v > max_val) { max_val = v; pivot = i; }
                }
                if (pivot != k) {
                    for (int j = 0; j < nv; j++) {
                        double tmp = sH[k * LDS_STRIDE + j];
                        sH[k * LDS_STRIDE + j] = sH[pivot * LDS_STRIDE + j];
                        sH[pivot * LDS_STRIDE + j] = tmp;
                    }
                    double tmp = s_y[k]; s_y[k] = s_y[pivot]; s_y[pivot] = tmp;
                }
                double diag = sH[k * LDS_STRIDE + k];
                if (fabs(diag) < 1e-20) diag = 1e-20;
                double inv_d = 1.0 / diag;
                for (int j = k; j < nv; j++) sH[k * LDS_STRIDE + j] *= inv_d;
                s_y[k] *= inv_d;
                for (int i = 0; i < nv; i++) {
                    if (i == k) continue;
                    double factor = sH[i * LDS_STRIDE + k];
                    for (int j = k; j < nv; j++)
                        sH[i * LDS_STRIDE + j] -= factor * sH[k * LDS_STRIDE + j];
                    s_y[i] -= factor * s_y[k];
                }
            }
        }
        WSYNC();

        if (active) {
            qacc_t = s_y[t];
            if (fuse_mhat) s_qv[t] = qacc_t;
            else           env_qacc[t] = qacc_t;
        }
        if (active && qacc_warmstart)
            qacc_warmstart[env * nv + t] = qacc_t;
        NPOST(fuse_mhat ? NCMD_MHAT : NCMD_EXIT, 0.0);
        if (integ.on) newton_integ_tail(integ, env, njnt, nq, nv, jnt_type, jnt_qpos_adr, jnt_dof_adr, qacc);
        return;
    }

    for (int c = t; c < nc; c += NEWTON_BLK)
        g_is_en[c] = false;
    WSYNC();
    if (cone_type == 1) {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int par = constraint_friction_parent[env_c_base + c];
            if (par >= 0 && par < nc) g_is_en[par] = true;
        }
    }
    WSYNC();

    double impr_inv = 1.0 / sqrt(fmax((double)impratio, 1e-6));
    if (active) s_qf[t] = qfrc_t;   // read by NCMD_LS helpers (after a barrier)

    double prev_cost = 1e30;
    int newton_iter_final = 0;
    double first_alpha = -1.0, first_gnorm = -1.0;
    double dbg_exit_code = -99.0;
    double first_cost0 = 0.0, first_cost_a1 = 0.0;
    int first_cg_iters = 0;
    double first_cg_rnorm = 0.0, first_search_mag = 0.0, first_grad_dot_search = 0.0;
    double first_rz0 = 0.0, first_pAp0 = 0.0, first_alpha0 = 0.0;
    double first_Hdiag_min = 0.0, first_Hdiag_max = 0.0;
    int first_n_active = 0;

    double first_Hrow0[5] = {0,0,0,0,0};

    bool fast_hb = false, cone_fuse = false;
    bool mh_elim_posted = false;
    // No join here: PREP3 (element list) runs on the helpers while wave 0
    // starts the first iteration; the next command's barrier (NCMD_HF, or any
    // other) orders it before every reader of the list.
    fast_hb = prep_ok;
    cone_fuse = fast_hb && cone_ok;
    bool l_valid = false;
    unsigned long long pam0 = 0ull, pam1 = 0ull;   // its state-1 masks (nc <= 128)
    NPS(0);
    // ── Newton iterations ──
    for (int newton_iter = 0; newton_iter < max_iterations; newton_iter++) {
        NPS(8);

        // ── 3a: Constraint residuals x = J*qacc - rhs ──
        double cfo0 = 0.0, cfo1 = 0.0;
        int    cst0 = 0, cst1 = 0;
        bool   rcap = false;
        if (active) s_qv[t] = qacc_t;
        WSYNC();
        if (cone_type == 0 && integ.res_fuse) {
            rcap = integ.stage_reg && (nc <= 2 * NEWTON_BLK);
            for (int c = t; c < nc; c += NEWTON_BLK) {
                const double Dc = constraint_D[env_c_base + c];
                const double lo_c = constraint_lo[env_c_base + c];
                const double hi_c = constraint_hi[env_c_base + c];
                const int par_c = constraint_friction_parent[env_c_base + c];
                double x = -constraint_rhs[env_c_base + c];
                if (NV_T > 0 && NV_T <= 64 && use_jt && integ.dot_unr) {
                    x = jt_dot_unr<(NV_T > 0 ? NV_T : 1)>(JTbase + c, nz_ok ? s_nzJ[c] : nz_full, s_qv, x);
                } else if (use_jt) {
                    const double* JTc = JTbase + c;
                    for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                        int v = __ffsll((unsigned long long)m) - 1;
                        x += JTc[(size_t)v * JT_STRIDE] * s_qv[v];
                    }
                } else {
                    const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
                    for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                        int v = __ffsll((unsigned long long)m) - 1;
                        x += Jc[v] * s_qv[v];
                    }
                }
                g_x[c] = x;
                double fo = 0.0;
                int st = 0;
                if (lo_c < -1e15 && hi_c > 1e15) {
                    fo = -Dc * x;
                    st = 1;
                } else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) {
                    if (x < 0.0) { fo = -Dc * x; st = 1; }
                } else if (par_c < 0) {
                    double f = fabs(hi_c);
                    double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                    if      (x <= -rf) { fo =  f; st = 2; }
                    else if (x >=  rf) { fo = -f; st = 3; }
                    else               { fo = -Dc * x; st = 1; }
                } else {
                    if (x < 0.0) { fo = -Dc * x; st = 1; }
                }
                g_force[c] = fo;
                g_state[c] = st;
                if (c < NEWTON_BLK) { cfo0 = fo; cst0 = st; }
                else                { cfo1 = fo; cst1 = st; }
            }
            WSYNC();
        } else {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double x = -constraint_rhs[env_c_base + c];
            if (use_jt) {
                const double* JTc = JTbase + c;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    x += JTc[(size_t)v * JT_STRIDE] * s_qv[v];
                }
            } else {
                const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    x += Jc[v] * s_qv[v];
                }
            }
            g_x[c] = x;
            g_force[c] = 0.0;
            g_state[c] = 0;
        }
        WSYNC();

        // ── 3b: Evaluate constraints → forces, states ──
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double x = g_x[c];
            double Dc = constraint_D[env_c_base + c];
            double lo_c = constraint_lo[env_c_base + c];
            double hi_c = constraint_hi[env_c_base + c];
            int par_c = constraint_friction_parent[env_c_base + c];

            if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;

            if (lo_c < -1e15 && hi_c > 1e15) {
                g_force[c] = -Dc * x;
                g_state[c] = 1;
            } else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) {
                if (x < 0.0) { g_force[c] = -Dc * x; g_state[c] = 1; }
            } else if (par_c < 0) {
                double f = fabs(hi_c);
                double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                if      (x <= -rf) { g_force[c] =  f; g_state[c] = 2; }
                else if (x >=  rf) { g_force[c] = -f; g_state[c] = 3; }
                else               { g_force[c] = -Dc * x; g_state[c] = 1; }
            } else {
                if (x < 0.0) { g_force[c] = -Dc * x; g_state[c] = 1; }
            }
        }
        WSYNC();
        }

        // Elliptic cone evaluation (parallel — each cone group has unique owner via c < sib)
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc) continue;
                if (sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;

                int normal = par, t1 = c, t2 = sib;
                const int torsion = constraint_friction_sibling[env_c_base+normal];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {normal, t1, t2, torsion};
                    double force[4];
                    int state;
                    evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, force, &state);
                    for (int i = 0; i < 4; ++i) {
                        g_force[rows[i]] = force[i];
                        g_state[rows[i]] = state;
                    }
                    continue;
                }
                double mu_fric = (double)constraint_friction_mu[env_c_base + c];
                double mu = mu_fric * impr_inv;
                double mu2 = mu * mu;
                double N = g_x[normal] * mu;
                double u1 = g_x[t1] * mu_fric;
                double u2 = g_x[t2] * mu_fric;
                double TT = u1 * u1 + u2 * u2;
                double T = (TT > 0.0) ? sqrt(TT) : 0.0;

                if ((N >= mu * T) || (T <= 0.0 && N >= 0.0)) {
                    g_state[normal] = 0; g_state[t1] = 0; g_state[t2] = 0;
                    g_force[normal] = 0; g_force[t1] = 0; g_force[t2] = 0;
                } else if ((mu * N + T <= 0.0) || (T <= 0.0 && N < 0.0)) {
                    double Dn = constraint_D[env_c_base + normal];
                    double D1 = constraint_D[env_c_base + t1];
                    double D2 = constraint_D[env_c_base + t2];
                    g_force[normal] = -Dn * g_x[normal]; g_state[normal] = 1;
                    g_force[t1]     = -D1 * g_x[t1];     g_state[t1]     = 1;
                    g_force[t2]     = -D2 * g_x[t2];     g_state[t2]     = 1;
                } else {
                    double D0 = constraint_D[env_c_base + normal];
                    double dm = D0 / fmax(mu2 * (1.0 + mu2), 1e-20);
                    double nmt = N - mu * T;
                    double Tsafe = fmax(T, 1e-20);
                    g_force[normal] = -dm * nmt * mu;
                    g_force[t1] = (dm * nmt * mu / Tsafe) * u1 * mu_fric;
                    g_force[t2] = (dm * nmt * mu / Tsafe) * u2 * mu_fric;
                    g_state[normal] = 4; g_state[t1] = 4; g_state[t2] = 4;
                }
            }
        }
        WSYNC();

        // ── 3c + 3c' FUSED: gradient and Hessian in ONE pass over the rows ──
        // Both passes walked c = 0..nc and both loaded the SAME column element
        // Jc[t]; fusing them halves the J traffic and the loop overhead. The two
        // accumulators stay independent, so every term is added in exactly the
        // order the split loops added it -- bit-identical.
        //   grad[t] = M_row . qacc - qfrc - J^T * force
        //   H       = M + J^T * cm * J
        // cm = diag(D*active) for non-cone constraints and bottom-zone cone contacts,
        // plus the cone Hessian block for middle-zone cone contacts (added below).
        // MJX sets active=False for middle-zone rows (state=4), so they get ONLY
        // the cone Hessian, not D*active. State=1 (bottom-zone + non-cone) gets D*active.
        NPS(1);
        double grad_t = 0.0;
        bool h_reuse = false;
#if R5_CHK
        bool chk_would = false;
        double chk_old[NV_T > 0 ? NV_T : 1];
#endif
        if (!fast_hb) l_valid = false;
        if (fast_hb) {
            NPX0();
            unsigned long long am0 = 0ull, am1 = 0ull;
            const bool any4 = rcap
                ? hf_stage_r(t, nc, cfo0, cst0, cfo1, cst1, s_hf, s_hactm, s_hfm, s_h4m, am0, am1)
                : hf_stage(t, nc, g_force, g_state, s_hf, s_hactm, s_hfm, s_h4m);
            // State-4 cone list (ascending q), from the staged state masks.
            int n4 = 0;
            if (any4 && cone_ok) {
                for (int q0 = 0; q0 < ncone; q0 += NEWTON_BLK) {
                    const int q = q0 + t;
                    bool is4 = false;
                    if (q < ncone) {
                        const int nrm = s_coneP[q];
                        is4 = (s_h4m[nrm >> 6] >> (nrm & 63)) & 1ull;
                    }
                    const unsigned long long b4 = __ballot(is4);
                    if (is4) s_q4[n4 + __popcll(b4 & ((1ull << t) - 1ull))] = q;
                    n4 += __popcll(b4);
                }
            }
            // mode: 0 none, 1 old scan, 2 old with list, 3 fused into elements
            const int cmode = !any4 ? 0 : !cone_ok ? 1
                            : (cone_fuse && n4 <= HB_Q4CAP) ? 3 : 2;
            if (cmode == 3)
                hf_cone_prep(t, nc, n4, s_q4, s_coneC, s_coneP, s_coneS, s_coneT,
                             g_x, constraint_D + env_c_base,
                             constraint_friction_mu + env_c_base, impr_inv,
                             s_nzJ, s_ch, s_cU, s_crw);
            if (t == 0) { s_any4 = cmode; s_n4 = n4; }
            NPX(0);
            if constexpr (NV_T > 0 && NV_T <= NEWTON_BLK) {
                if (integ.l_reuse && rcap) {
                    // nc <= 128: the masks are wave-uniform values already.
                    const bool same = (am0 == pam0) && (am1 == pam1);
                    h_reuse = l_valid && cmode == 0 && same;
                    pam0 = am0; pam1 = am1;
                    WSYNC();   // order hf_stage_r's s_hf stores before hf_grad2
                } else if (integ.l_reuse) {
                    WSYNC();
                    bool dif = false;
                    if (t < HB_NW && t * NEWTON_BLK < nc) dif = (s_hactm[t] != s_hactm_prev[t]);
                    h_reuse = l_valid && cmode == 0 && (__ballot(dif) == 0ull);
                    if (t < HB_NW) s_hactm_prev[t] = s_hactm[t];
                    WSYNC();
                }
#if R5_CHK
                chk_would = h_reuse;
                if (h_reuse) {
                    #pragma unroll
                    for (int j = 0; j < NV_T; j++) chk_old[j] = active ? sH[t * LDS_STRIDE + j] : 0.0;
                }
                h_reuse = false;
#endif
            }
            NPX(1);
            if (!h_reuse) NPOST(NCMD_HF, 0.0);
            NPX(2);
            if (active) {
                double Mq = 0.0;
                for (int j = 0; j < nv; j++)
                    Mq += MR(j) * s_qv[j];
                grad_t = Mq - qfrc_t;
                hf_grad2(t, nc, grad_t, s_hcm, s_hcpre, s_hcJ, s_hf);
            }
            NPX(3);
            if (!h_reuse) BSYNC();   // join: lower triangle of H is in sH
            NPX(4);
            l_valid = (cmode == 0);
#if R4_CHK
            if constexpr (USE_SM && NV_T > 0) if (cmode == 0) {
                double hv[NV_T];
                #pragma unroll
                for (int j = 0; j < NV_T; j++) hv[j] = (active && j <= t) ? sH[t * LDS_STRIDE + j] : 0.0;
                double g2 = 0.0;
                if (active) {
                    double Mq = 0.0;
                    for (int j = 0; j < nv; j++) Mq += MR(j) * s_qv[j];
                    g2 = Mq - qfrc_t;
                    for (int j = 0; j < nv; j++) H_row[j] = MR(j);
                }
                NPOST(NCMD_H, 0.0);
                newton_hbuild_cols<16>(t, active, nc, nv, use_jt, JTbase,
                    constraint_J + env_j_base + t, g_force, g_state,
                    constraint_D + env_c_base, s_nzJ, H_row, col_own, true, g2);
                BSYNC();
                int bad = 0;
                #pragma unroll
                for (int j = 0; j < NV_T; j++)
                    if (active && j <= t && __double_as_longlong(hv[j]) != __double_as_longlong(sH[t * LDS_STRIDE + j])) bad++;
                if (active && __double_as_longlong(g2) != __double_as_longlong(grad_t)) bad += 1000;
                const unsigned long long bb = __ballot(bad != 0);
                if (bb && active && bad) printf("R4H MISMATCH env=%d nc=%d t=%d bad=%d\n", env, nc, t, bad);
                if (!bb && t == 0 && nc > 64) printf("R4H ok64\n");
                #pragma unroll
                for (int j = 0; j < NV_T; j++) if (active && j <= t) sH[t * LDS_STRIDE + j] = hv[j];
                WSYNC();
            }
#endif
            if (cmode == 1 || cmode == 2) {
                newton_cone_cols(t, active, nc, nv, cone_ok, ncone,
                    s_coneC, s_coneP, s_coneS, s_coneT,
                    constraint_friction_parent + env_c_base,
                    constraint_friction_sibling + env_c_base,
                    g_x, g_state, constraint_D + env_c_base,
                    constraint_friction_mu + env_c_base, impr_inv,
                    s_nzJ, constraint_J + env_j_base, H_row, col_own,
                    cone_ok ? s_q4 : nullptr, n4);
                BSYNC();
            }
        } else {
        if (active) {
            double Mq = 0.0;
            for (int j = 0; j < nv; j++)
                Mq += MR(j) * s_qv[j];
            grad_t = Mq - qfrc_t;

            for (int j = 0; j < nv; j++) H_row[j] = MR(j);
        }
        if (nz_ok) {
            // Multi-wave build: helpers take the columns they own, wave 0 its
            // own columns plus the gradient.  H_row was seeded with M above.
            NPOST(NCMD_H, 0.0);
            newton_hbuild_cols<16>(t, active, nc, nv, use_jt, JTbase,
                constraint_J + env_j_base + t, g_force, g_state,
                constraint_D + env_c_base, s_nzJ, H_row, col_own, true, grad_t);
            if (cone_type == 1)
                newton_cone_cols(t, active, nc, nv, cone_ok, ncone,
                    s_coneC, s_coneP, s_coneS, s_coneT,
                    constraint_friction_parent + env_c_base,
                    constraint_friction_sibling + env_c_base,
                    g_x, g_state, constraint_D + env_c_base,
                    constraint_friction_mu + env_c_base, impr_inv,
                    s_nzJ, constraint_J + env_j_base, H_row, col_own);
            BSYNC();   // join: all columns of H are in sH
        } else {
          if (active)
            for (int c = 0; c < nc; c++) {
                const double fc = g_force[c];
                const int    st = g_state[c];
                const bool want_grad = (fabs(fc) >= 1e-30);
                const bool want_hess = (st == 1);
                if (!want_grad && !want_hess) continue;   // wave-uniform

                // ONE coalesced load of this row's column element. The whole
                // wave now holds row c: lane v owns Jc[v].
                const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
                const double Jct = Jc[t];

                if (want_grad) grad_t -= fc * Jct;
                if (!want_hess) continue;

                const double Dc = constraint_D[env_c_base + c];
                const unsigned long long nzc = nz_ok ? s_nzJ[c]
                                                     : wf_row_nz_mask(Jc, t, active);
                const double dJt = Dc * Jct;
                if (fabs(dJt) < 1e-30) continue;
                for (unsigned long long m = nzc; m; m &= m - 1) {
                    int j = __ffsll((unsigned long long)m) - 1;
                    H_row[j] += dJt * wf_broadcast_d(Jct, j);
                }
            }
          // Elliptic cone Hessian for middle-zone contacts (state=4).
        // For each cone group (normal + 2 tangent rows), compute the 3x3 Hessian
        // block from MJX's formula and add J^T * h_cone * J to H.
        if (cone_type == 1 && active) {
            const int qn = cone_ok ? ncone : nc;
            for (int q = 0; q < qn; q++) {
                int c, par, sib, torsion;
                if (cone_ok) {
                    c = s_coneC[q]; par = s_coneP[q];
                    sib = s_coneS[q]; torsion = s_coneT[q];
                } else {
                    c = q;
                    par = constraint_friction_parent[env_c_base + c];
                    sib = constraint_friction_sibling[env_c_base + c];
                    if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                    if (c >= sib) continue;
                    torsion = constraint_friction_sibling[env_c_base + par];
                }

                int normal = par, t1 = c, t2 = sib;
                if (g_state[normal] != 4) continue;

                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {normal, t1, t2, torsion};
                    double h[16];
                    evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr, h);
                    // All four masks are taken here, ABOVE the divergent Ji[t]
                    // test, so every ballot runs at wave-uniform control flow.
                    unsigned long long nzr[4];
                    for (int q = 0; q < 4; ++q)
                        nzr[q] = nz_ok ? s_nzJ[rows[q]]
                                       : wf_row_nz_mask(constraint_J+env_j_base+(size_t)rows[q]*nv, t, active);
                    for (int ci = 0; ci < 4; ++ci) {
                        const double* Ji = constraint_J+env_j_base+(size_t)rows[ci]*nv;
                        if (fabs(Ji[t]) < 1e-30) continue;
                        for (int cj = 0; cj < 4; ++cj) {
                            if (fabs(h[ci*4+cj]) < 1e-30) continue;
                            const double* Jj = constraint_J+env_j_base+(size_t)rows[cj]*nv;
                            const double hJi = h[ci*4+cj]*Ji[t];
                            for (unsigned long long m = nzr[cj]; m; m &= m - 1) {
                                int j = __ffsll((unsigned long long)m) - 1;
                                H_row[j] += hJi*Jj[j];
                            }
                        }
                    }
                    continue;
                }

                double mu_fric = (double)constraint_friction_mu[env_c_base + c];
                double mu_l = mu_fric * impr_inv;
                double mu2_l = mu_l * mu_l;

                double xn = g_x[normal], xt1_v = g_x[t1], xt2_v = g_x[t2];
                double un = xn * mu_l, u1v = xt1_v * mu_fric, u2v = xt2_v * mu_fric;
                double TT = u1v * u1v + u2v * u2v;
                double T_val = (TT > 0.0) ? sqrt(TT) : 1e-20;
                T_val = fmax(T_val, 1e-20);

                double D0 = constraint_D[env_c_base + normal];
                double dm_val = D0 / fmax(mu2_l * (1.0 + mu2_l), 1e-20);

                double u_vec[3] = {un, u1v, u2v};
                double fri_vec[3] = {mu_l, mu_fric, mu_fric};

                // Factor the outer product using normalized residual components.
                double coeff_outer = mu_l * un / T_val;
                double coeff_diag = mu2_l - mu_l * un / T_val;
                double h[3][3];
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] = coeff_outer * (u_vec[ii] / T_val) * (u_vec[jj] / T_val)
                                   + ((ii == jj) ? coeff_diag : 0.0);

                // Set row 0 / col 0: (1, -mu/T * u[1], -mu/T * u[2])
                double h0_1 = -mu_l / T_val * u1v;
                double h0_2 = -mu_l / T_val * u2v;
                h[0][0] = 1.0; h[0][1] = h0_1; h[0][2] = h0_2;
                h[1][0] = h0_1; h[2][0] = h0_2;

                // Scale: h = dm * diag(fri) @ h @ diag(fri)
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] *= dm_val * fri_vec[ii] * fri_vec[jj];

                // Add J^T * h_cone * J to H_row
                int indices[3] = {normal, t1, t2};
                unsigned long long nzi[3];
                for (int q = 0; q < 3; q++)
                    nzi[q] = nz_ok ? s_nzJ[indices[q]]
                                   : wf_row_nz_mask(constraint_J + env_j_base + (size_t)indices[q] * nv, t, active);
                for (int ci = 0; ci < 3; ci++) {
                    const double* Jci = constraint_J + env_j_base + (size_t)indices[ci] * nv;
                    double Jci_t = Jci[t];
                    if (fabs(Jci_t) < 1e-30) continue;
                    for (int cj = 0; cj < 3; cj++) {
                        double h_ij = h[ci][cj];
                        if (fabs(h_ij) < 1e-30) continue;
                        const double* Jcj = constraint_J + env_j_base + (size_t)indices[cj] * nv;
                        double hJci = h_ij * Jci_t;
                        for (unsigned long long m = nzi[cj]; m; m &= m - 1) {
                            int j = __ffsll((unsigned long long)m) - 1;
                            H_row[j] += hJci * Jcj[j];
                        }
                    }
                }
            }
        }
        }
        }   // !fast_hb

        // Save H_row values at first Newton iteration for diagnostics
        if (newton_iter == 0 && t == 0) {
            for (int j = 0; j < 5 && j < nv; j++)
                first_Hrow0[j] = H_row[j];
        }

        NPS(2);
        // ── 3d: Check gradient convergence ──
        double gnorm2 = wf_reduce_sum_d(active ? grad_t * grad_t : 0.0);
        if (gnorm2 < solver_tolerance * solver_tolerance) { dbg_exit_code = 1.0; break; }

        // ── 3e: LDS Cholesky solve H*search = -grad ──
        // Uses shared memory for the Cholesky factorization to avoid AMD VGPR
        // spill corruption that occurs with register-only approach on H matrices.
        // H_row (registers) -> sH (LDS) -> factor in LDS -> solve in LDS.
        double search_t = 0.0;
        int cg_iters_done = 0;
        double cg_final_rnorm = 0.0;
        double dbg_rz0 = 0.0, dbg_pAp0 = 0.0, dbg_alpha0 = 0.0;
        double dbg_Hdiag_min = 0.0, dbg_Hdiag_max = 0.0;
        int dbg_n_active_constraints = 0;
        {
            // H was accumulated directly in sH by 3c'/3c'' -- nothing to copy.
            WSYNC();

            // Regularization: find max diagonal
            double h_max = 0.0;
            if (active) h_max = sH[t * LDS_STRIDE + t];
            for (int off = 32; off > 0; off >>= 1) {
                double o = wf_shfl_xor_d(h_max, off);
                if (o > h_max) h_max = o;
            }
            dbg_Hdiag_max = h_max;
            double h_delta = fmax(h_max * 1e-12, 1e-12);

            // Cholesky factorization in LDS: L stored in lower triangle.
            // Right-looking and wave-cooperative: the left-looking form this
            // replaces ran the diagonal under `if (t==k)` and gave every lane a
            // serial inner j-loop, i.e. ~2*nv^2/2 strictly dependent double ops
            // per Newton iteration behind a one-lane critical path.
            if constexpr (NV_T > 0 && NV_T <= NEWTON_BLK) {
                // Compile-time nv: factor + substitute entirely in wave 0's
                // registers (rc_*), same per-element op sequence as the LDS
                // multi-wave path, no per-step workgroup barrier.  The helpers
                // stay parked (no CHOL command); wave 1 gets the M_hat
                // elimination right away instead.
                if (fuse_mhat && !mh_elim_posted) {
                    NPOST(NCMD_MHAT_ELIM, 0.0);
                    mh_elim_posted = true;
                }
                double ra[NV_T];
                #pragma unroll
                for (int j = 0; j < NV_T; j++)
                    ra[j] = active ? sH[t * LDS_STRIDE + j] : 0.0;
                if (!h_reuse) rc_cholp<NV_T>(ra, t, h_delta);
#if R5_CHK
                if (chk_would) {
                    int bad = 0;
                    #pragma unroll
                    for (int j = 0; j < NV_T; j++)
                        if (active && j <= t && __double_as_longlong(chk_old[j]) != __double_as_longlong(ra[j])) bad++;
                    const unsigned long long bb = __ballot(bad != 0);
                    if (bb && active && bad) printf("R5L MISMATCH env=%d it=%d t=%d bad=%d\n", env, newton_iter, t, bad);
                    if (!bb && t == 0) printf("R5L ok env=%d it=%d\n", env, newton_iter);
                }
#endif
                NPS(3);
                const double y_t = rc_tri_fwd<NV_T>(ra, t, active,
                                                    active ? -grad_t : 0.0);
                // Transpose L through LDS for the column-wise back substitution.
                if (active && !h_reuse) {
                    #pragma unroll
                    for (int j = 0; j < NV_T; j++)
                        sH[t * LDS_STRIDE + j] = ra[j];
                }
                WSYNC();
                double rcol[NV_T];
                #pragma unroll
                for (int i = 0; i < NV_T; i++)
                    rcol[i] = active ? sH[i * LDS_STRIDE + t] : 0.0;
                search_t = active ? rc_tri_bwd<NV_T>(rcol, t, active, y_t) : 0.0;
            } else {
            NPOST(NCMD_CHOL, h_delta);
            mw_chol_factor_lds(sH, nv, t, 0, h_delta);
            // Hand wave 1 the (qacc-independent) M_hat elimination while this
            // wave runs substitutions / line search / residuals.
            if (fuse_mhat && !mh_elim_posted) {
                NPOST(NCMD_MHAT_ELIM, 0.0);
                mh_elim_posted = true;
            }

            NPS(3);
            // Forward / backward substitution, one lane per row, no LDS
            // round-trip through s_y / s_srch and no per-row barrier.
            {
                double y_t = wf_tri_solve_fwd_lds(sH, nv, t, active,
                                                  active ? -grad_t : 0.0);
                search_t = active ? wf_tri_solve_bwd_lds(sH, nv, t, active, y_t)
                                  : 0.0;
            }
            }
        }

        NPS(4);
        double sm_gs[2] = {active ? search_t * search_t : 0.0, active ? grad_t * search_t : 0.0};
        wf_reduce_sum_dn<2>(sm_gs);
        double search_mag2 = sm_gs[0];
        double grad_dot_search = sm_gs[1];

        // Bail on NaN/Inf search direction
        float search_nan_count = wf_reduce_sum(active ? (isnan(search_t) || isinf(search_t) ? 1.0f : 0.0f) : 0.0f);
        if (search_nan_count > 0.0f) {
            dbg_exit_code = 2.0;
            first_gnorm = sqrt(gnorm2);
            break;
        }

        // ── 3f: Precompute J*search per constraint ──
        if (active) s_sv[t] = search_t;
        WSYNC();
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double v_val = 0.0;
            if (NV_T > 0 && NV_T <= 64 && use_jt && integ.dot_unr) {
                v_val = jt_dot_unr<(NV_T > 0 ? NV_T : 1)>(JTbase + c, nz_ok ? s_nzJ[c] : nz_full, s_sv, v_val);
            } else if (use_jt) {
                const double* JTc = JTbase + c;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    v_val += JTc[(size_t)v * JT_STRIDE] * s_sv[v];
                }
            } else {
                const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
                for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                    int v = __ffsll((unsigned long long)m) - 1;
                    v_val += Jc[v] * s_sv[v];
                }
            }
            g_jv[c] = v_val;
        }
        WSYNC();

        NPS(5);
        double dphi0 = grad_dot_search;

        // Compute second derivative along search direction for Newton step size.
        // d2cost/dalpha2 = search^T * H * search
        // where H = M + J^T*cm*J (cm = D*active for state=1, h_cone for state=4)
        double Ms_val = 0.0;
        if (active)
            for (int j = 0; j < nv; j++)
                Ms_val += MR(j) * s_sv[j];
        double d2_gauss = active ? search_t * Ms_val : 0.0;   // reduced below
        double d2_constr = 0.0;
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int st = g_state[c];
            if (st != 1) continue;
            d2_constr += constraint_D[env_c_base + c] * g_jv[c] * g_jv[c];
        }
        // Add cone Hessian d2 contribution for middle-zone (state=4) — parallel
        double d2_cone = 0.0;
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;
                int normal = par, t1c = c, t2c = sib;
                if (g_state[normal] != 4) continue;
                const int torsion = constraint_friction_sibling[env_c_base+normal];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {normal, t1c, t2c, torsion};
                    double h[16];
                    evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr, h);
                    for (int ci = 0; ci < 4; ++ci)
                        for (int cj = 0; cj < 4; ++cj)
                            d2_cone += h[ci*4+cj]*g_jv[rows[ci]]*g_jv[rows[cj]];
                    continue;
                }
                double muf = (double)constraint_friction_mu[env_c_base + c];
                double mul = muf * impr_inv;
                double mu2l = mul * mul;
                double xn = g_x[normal], xt1v = g_x[t1c], xt2v = g_x[t2c];
                double un = xn * mul, u1v = xt1v * muf, u2v = xt2v * muf;
                double TT = u1v * u1v + u2v * u2v;
                double T_val = (TT > 0.0) ? sqrt(TT) : 1e-20;
                T_val = fmax(T_val, 1e-20);
                double D0 = constraint_D[env_c_base + normal];
                double dm_val = D0 / fmax(mu2l * (1.0 + mu2l), 1e-20);
                double u_vec[3] = {un, u1v, u2v};
                double fri_vec[3] = {mul, muf, muf};
                double coeff_outer = mul * un / T_val;
                double coeff_diag = mu2l - mul * un / T_val;
                double h[3][3];
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] = coeff_outer * (u_vec[ii] / T_val) * (u_vec[jj] / T_val)
                                   + ((ii == jj) ? coeff_diag : 0.0);
                double h0_1 = -mul / T_val * u1v;
                double h0_2 = -mul / T_val * u2v;
                h[0][0] = 1.0; h[0][1] = h0_1; h[0][2] = h0_2;
                h[1][0] = h0_1; h[2][0] = h0_2;
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] *= dm_val * fri_vec[ii] * fri_vec[jj];
                int indices[3] = {normal, t1c, t2c};
                for (int ci = 0; ci < 3; ci++)
                    for (int cj = 0; cj < 3; cj++)
                        d2_cone += h[ci][cj] * g_jv[indices[ci]] * g_jv[indices[cj]];
            }
        }
        double my_Mq_0 = 0.0;
        if (active)
            for (int j = 0; j < nv; j++) my_Mq_0 += MR(j) * s_qv[j];
        double quad_cost_0 = active ? (0.5 * qacc_t * my_Mq_0 - qacc_t * qfrc_t) : 0.0;

        double constr_cost_0 = 0.0;
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double xc = g_x[c];
            double Dc = constraint_D[env_c_base + c];
            double lo_c = constraint_lo[env_c_base + c];
            double hi_c = constraint_hi[env_c_base + c];
            int par_c = constraint_friction_parent[env_c_base + c];
            if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;
            double sc = 0.0;
            if (lo_c < -1e15 && hi_c > 1e15) sc = 0.5 * Dc * xc * xc;
            else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            else if (par_c < 0) {
                double f = fabs(hi_c);
                double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                double ax = fabs(xc);
                sc = (ax < rf) ? 0.5 * Dc * xc * xc : f * (ax - 0.5 * rf);
            } else { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            constr_cost_0 += sc;
        }

        double cone_cost_0 = 0.0;
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;
                const int torsion = constraint_friction_sibling[env_c_base+par];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {par, c, sib, torsion};
                    cone_cost_0 += evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr);
                    continue;
                }
                double muf = (double)constraint_friction_mu[env_c_base + c];
                double mul = muf * impr_inv;
                double Nn = g_x[par] * mul;
                double ut1 = g_x[c] * muf, ut2 = g_x[sib] * muf;
                double TTl = ut1*ut1 + ut2*ut2;
                double Tl = (TTl > 0.0) ? sqrt(TTl) : 0.0;
                if ((Nn >= mul*Tl) || (Tl <= 0 && Nn >= 0)) { }
                else if ((mul*Nn+Tl <= 0) || (Tl <= 0 && Nn < 0)) {
                    cone_cost_0 += 0.5*constraint_D[env_c_base+par]*g_x[par]*g_x[par]
                                 + 0.5*constraint_D[env_c_base+c]*g_x[c]*g_x[c]
                                 + 0.5*constraint_D[env_c_base+sib]*g_x[sib]*g_x[sib];
                } else {
                    double D0 = constraint_D[env_c_base+par];
                    double mu2l = mul*mul;
                    double dml = D0 / fmax(mu2l*(1+mu2l), 1e-20);
                    double nmt = Nn - mul*Tl;
                    cone_cost_0 += 0.5 * dml * nmt * nmt;
                }
            }
        }
        {
            // All six reductions at once (each one's op sequence unchanged).
            double rv[6] = {d2_gauss, d2_constr, d2_cone, quad_cost_0, constr_cost_0, cone_cost_0};
            wf_reduce_sum_dn<6>(rv);
            d2_gauss = rv[0]; d2_constr = rv[1]; d2_cone = rv[2];
            quad_cost_0 = rv[3]; constr_cost_0 = rv[4]; cone_cost_0 = rv[5];
        }
        double d2_total = d2_gauss + d2_constr + d2_cone;
        double alpha_newton = (d2_total > 1e-20 && dphi0 < 0.0)
            ? fmin(-dphi0 / d2_total, 4.0) : 1.0;
        double cost_0 = quad_cost_0 + constr_cost_0 + cone_cost_0;

#if R4_CHK
        {
            const double o_an = alpha_newton, o_c0 = cost_0;
            {
        double dphi0 = wf_reduce_sum_d(active ? grad_t * search_t : 0.0);

        // Compute second derivative along search direction for Newton step size.
        // d2cost/dalpha2 = search^T * H * search
        // where H = M + J^T*cm*J (cm = D*active for state=1, h_cone for state=4)
        double Ms_val = 0.0;
        if (active)
            for (int j = 0; j < nv; j++)
                Ms_val += MR(j) * s_sv[j];
        double d2_gauss = wf_reduce_sum_d(active ? search_t * Ms_val : 0.0);
        double d2_constr = 0.0;
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int st = g_state[c];
            if (st != 1) continue;
            d2_constr += constraint_D[env_c_base + c] * g_jv[c] * g_jv[c];
        }
        d2_constr = wf_reduce_sum_d(d2_constr);
        // Add cone Hessian d2 contribution for middle-zone (state=4) — parallel
        double d2_cone = 0.0;
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;
                int normal = par, t1c = c, t2c = sib;
                if (g_state[normal] != 4) continue;
                const int torsion = constraint_friction_sibling[env_c_base+normal];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {normal, t1c, t2c, torsion};
                    double h[16];
                    evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr, h);
                    for (int ci = 0; ci < 4; ++ci)
                        for (int cj = 0; cj < 4; ++cj)
                            d2_cone += h[ci*4+cj]*g_jv[rows[ci]]*g_jv[rows[cj]];
                    continue;
                }
                double muf = (double)constraint_friction_mu[env_c_base + c];
                double mul = muf * impr_inv;
                double mu2l = mul * mul;
                double xn = g_x[normal], xt1v = g_x[t1c], xt2v = g_x[t2c];
                double un = xn * mul, u1v = xt1v * muf, u2v = xt2v * muf;
                double TT = u1v * u1v + u2v * u2v;
                double T_val = (TT > 0.0) ? sqrt(TT) : 1e-20;
                T_val = fmax(T_val, 1e-20);
                double D0 = constraint_D[env_c_base + normal];
                double dm_val = D0 / fmax(mu2l * (1.0 + mu2l), 1e-20);
                double u_vec[3] = {un, u1v, u2v};
                double fri_vec[3] = {mul, muf, muf};
                double coeff_outer = mul * un / T_val;
                double coeff_diag = mu2l - mul * un / T_val;
                double h[3][3];
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] = coeff_outer * (u_vec[ii] / T_val) * (u_vec[jj] / T_val)
                                   + ((ii == jj) ? coeff_diag : 0.0);
                double h0_1 = -mul / T_val * u1v;
                double h0_2 = -mul / T_val * u2v;
                h[0][0] = 1.0; h[0][1] = h0_1; h[0][2] = h0_2;
                h[1][0] = h0_1; h[2][0] = h0_2;
                for (int ii = 0; ii < 3; ii++)
                    for (int jj = 0; jj < 3; jj++)
                        h[ii][jj] *= dm_val * fri_vec[ii] * fri_vec[jj];
                int indices[3] = {normal, t1c, t2c};
                for (int ci = 0; ci < 3; ci++)
                    for (int cj = 0; cj < 3; cj++)
                        d2_cone += h[ci][cj] * g_jv[indices[ci]] * g_jv[indices[cj]];
            }
        }
        d2_cone = wf_reduce_sum_d(d2_cone);
        double d2_total = d2_gauss + d2_constr + d2_cone;
        double alpha_newton = (d2_total > 1e-20 && dphi0 < 0.0)
            ? fmin(-dphi0 / d2_total, 4.0) : 1.0;

        double my_Mq_0 = 0.0;
        if (active)
            for (int j = 0; j < nv; j++) my_Mq_0 += MR(j) * s_qv[j];
        double quad_cost_0 = wf_reduce_sum_d(active ? (0.5 * qacc_t * my_Mq_0 - qacc_t * qfrc_t) : 0.0);

        double constr_cost_0 = 0.0;
        for (int c = t; c < nc; c += NEWTON_BLK) {
            double xc = g_x[c];
            double Dc = constraint_D[env_c_base + c];
            double lo_c = constraint_lo[env_c_base + c];
            double hi_c = constraint_hi[env_c_base + c];
            int par_c = constraint_friction_parent[env_c_base + c];
            if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;
            double sc = 0.0;
            if (lo_c < -1e15 && hi_c > 1e15) sc = 0.5 * Dc * xc * xc;
            else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            else if (par_c < 0) {
                double f = fabs(hi_c);
                double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                double ax = fabs(xc);
                sc = (ax < rf) ? 0.5 * Dc * xc * xc : f * (ax - 0.5 * rf);
            } else { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
            constr_cost_0 += sc;
        }
        constr_cost_0 = wf_reduce_sum_d(constr_cost_0);

        double cone_cost_0 = 0.0;
        if (cone_type == 1) {
            for (int c = t; c < nc; c += NEWTON_BLK) {
                int par = constraint_friction_parent[env_c_base + c];
                int sib = constraint_friction_sibling[env_c_base + c];
                if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                if (c >= sib) continue;
                const int torsion = constraint_friction_sibling[env_c_base+par];
                if (torsion >= 0 && torsion < nc) {
                    const int rows[4] = {par, c, sib, torsion};
                    cone_cost_0 += evaluate_contact4(rows, g_x, nullptr, 0.0,
                        constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                        impr_inv, nullptr, nullptr);
                    continue;
                }
                double muf = (double)constraint_friction_mu[env_c_base + c];
                double mul = muf * impr_inv;
                double Nn = g_x[par] * mul;
                double ut1 = g_x[c] * muf, ut2 = g_x[sib] * muf;
                double TTl = ut1*ut1 + ut2*ut2;
                double Tl = (TTl > 0.0) ? sqrt(TTl) : 0.0;
                if ((Nn >= mul*Tl) || (Tl <= 0 && Nn >= 0)) { }
                else if ((mul*Nn+Tl <= 0) || (Tl <= 0 && Nn < 0)) {
                    cone_cost_0 += 0.5*constraint_D[env_c_base+par]*g_x[par]*g_x[par]
                                 + 0.5*constraint_D[env_c_base+c]*g_x[c]*g_x[c]
                                 + 0.5*constraint_D[env_c_base+sib]*g_x[sib]*g_x[sib];
                } else {
                    double D0 = constraint_D[env_c_base+par];
                    double mu2l = mul*mul;
                    double dml = D0 / fmax(mu2l*(1+mu2l), 1e-20);
                    double nmt = Nn - mul*Tl;
                    cone_cost_0 += 0.5 * dml * nmt * nmt;
                }
            }
        }
        cone_cost_0 = wf_reduce_sum_d(cone_cost_0);
        double cost_0 = quad_cost_0 + constr_cost_0 + cone_cost_0;
            if (__double_as_longlong(o_an) != __double_as_longlong(alpha_newton) ||
                __double_as_longlong(o_c0) != __double_as_longlong(cost_0)) {
                if (t == 0) printf("R4LSP MISMATCH env=%d %.17g %.17g | %.17g %.17g\n", env, o_an, alpha_newton, o_c0, cost_0);
            } else if (t == 0 && env == 0) printf("R4LSP ok\n");
            }
        }
#endif
        NPS(6);
        double best_alpha = 0.0;
        double accepted_cost = cost_0;
        double alpha = fmax(alpha_newton, 0.01);
        const bool par_ls = NEWTON_PAR_LS && USE_SM && (NV_T > 0);
        int par_k0 = 0;
        for (int ls = 0; ls < LS_MAXITER; ls++) {
            double cost_a;
            if constexpr (USE_SM)
                cost_a = newton_ls_trial_cost<NV_T>(t, active, nv, nc, cone_type, impr_inv, alpha,
                    M_lds, s_qv, s_sv, qacc_t, search_t, qfrc_t, g_x, g_jv, g_is_en, env_c_base,
                    constraint_D, constraint_lo, constraint_hi,
                    constraint_friction_parent, constraint_friction_sibling,
                    constraint_friction_mu);
            else
                cost_a = newton_ls_trial_cost<NV_T>(t, active, nv, nc, cone_type, impr_inv, alpha,
                    M_reg, s_qv, s_sv, qacc_t, search_t, qfrc_t, g_x, g_jv, g_is_en, env_c_base,
                    constraint_D, constraint_lo, constraint_hi,
                    constraint_friction_parent, constraint_friction_sibling,
                    constraint_friction_mu);

            if (newton_iter == 0 && ls == 0) first_cost_a1 = cost_a;
#if NEWTON_PROF
            np_ls++;
#endif

            if (cost_a < cost_0 && !isnan(cost_a)) {
                best_alpha = alpha;
                accepted_cost = cost_a;
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-10) break;
            // Trial 0 failed: run the rest four trials per round on all waves.
            if (par_ls) { par_k0 = ls + 1; break; }
        }
        if (par_k0 > 0) {
            int k0 = par_k0;
            while (k0 < LS_MAXITER) {
                NPOST(NCMD_LS, alpha);
                const double c0w = newton_ls_trial_cost<NV_T>(t, active, nv, nc, cone_type, impr_inv, alpha,
                    M_lds, s_qv, s_sv, qacc_t, search_t, qfrc_t, g_x, g_jv, g_is_en, env_c_base,
                    constraint_D, constraint_lo, constraint_hi,
                    constraint_friction_parent, constraint_friction_sibling,
                    constraint_friction_mu);
                BSYNC();
                // Replay the serial acceptance / termination order.
                bool stop = false;
                double a = alpha;
                for (int w = 0; w < NEWTON_WAVES; w++) {
                    if (w > 0) {
                        a *= 0.5;
                        if (a < 1e-10) { stop = true; break; }
                        if (k0 + w >= LS_MAXITER) { stop = true; break; }
                    }
                    const double cw = (w == 0) ? c0w : s_lsc[w];
#if NEWTON_PROF
                    np_ls++;
#endif
                    if (cw < cost_0 && !isnan(cw)) {
                        best_alpha = a;
                        accepted_cost = cw;
                        stop = true;
                        break;
                    }
                }
                if (stop) break;
                alpha = a * 0.5;
                if (alpha < 1e-10) break;
                k0 += NEWTON_WAVES;
            }
        }

#if R4_CHK
        {
        double ck_best_alpha = 0.0;
        double ck_accepted_cost = cost_0;
        double ck_alpha_unused = 0; double alpha = fmax(alpha_newton, 0.01);
        for (int ls = 0; ls < LS_MAXITER; ls++) {
            double qi_a = active ? (qacc_t + alpha * search_t) : 0.0;
            double my_Mq_a = 0.0;
            if (active)
                for (int j = 0; j < nv; j++)
                    my_Mq_a += MR(j) * (s_qv[j] + alpha * s_sv[j]);
            double quad_cost_a = wf_reduce_sum_d(active ? (0.5 * qi_a * my_Mq_a - qi_a * qfrc_t) : 0.0);

            double constr_cost_a = 0.0;
            for (int c = t; c < nc; c += NEWTON_BLK) {
                double xc = g_x[c] + alpha * g_jv[c];
                double Dc = constraint_D[env_c_base + c];
                double lo_c = constraint_lo[env_c_base + c];
                double hi_c = constraint_hi[env_c_base + c];
                int par_c = constraint_friction_parent[env_c_base + c];
                if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;
                double sc = 0.0;
                if (lo_c < -1e15 && hi_c > 1e15) sc = 0.5 * Dc * xc * xc;
                else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
                else if (par_c < 0) {
                    double f = fabs(hi_c);
                    double rf = (Dc > 1e-10) ? f / Dc : 0.0;
                    double ax = fabs(xc);
                    sc = (ax < rf) ? 0.5 * Dc * xc * xc : f * (ax - 0.5 * rf);
                } else { if (xc < 0) sc = 0.5 * Dc * xc * xc; }
                constr_cost_a += sc;
            }
            constr_cost_a = wf_reduce_sum_d(constr_cost_a);

            double cone_cost_a = 0.0;
            if (cone_type == 1) {
                for (int c = t; c < nc; c += NEWTON_BLK) {
                    int par = constraint_friction_parent[env_c_base + c];
                    int sib = constraint_friction_sibling[env_c_base + c];
                    if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
                    if (c >= sib) continue;
                    const int torsion = constraint_friction_sibling[env_c_base+par];
                    if (torsion >= 0 && torsion < nc) {
                        const int rows[4] = {par, c, sib, torsion};
                        cone_cost_a += evaluate_contact4(rows, g_x, g_jv, alpha,
                            constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                            impr_inv, nullptr, nullptr);
                        continue;
                    }
                    double muf = (double)constraint_friction_mu[env_c_base + c];
                    double mul = muf * impr_inv;
                    double xn = g_x[par] + alpha*g_jv[par];
                    double xt1 = g_x[c] + alpha*g_jv[c];
                    double xt2 = g_x[sib] + alpha*g_jv[sib];
                    double Nn = xn*mul;
                    double ut1 = xt1*muf, ut2 = xt2*muf;
                    double TTl = ut1*ut1+ut2*ut2;
                    double Tl = (TTl > 0) ? sqrt(TTl) : 0.0;
                    if ((Nn >= mul*Tl)||(Tl<=0 && Nn>=0)) {}
                    else if ((mul*Nn+Tl<=0)||(Tl<=0 && Nn<0)) {
                        cone_cost_a += 0.5*constraint_D[env_c_base+par]*xn*xn
                                     + 0.5*constraint_D[env_c_base+c]*xt1*xt1
                                     + 0.5*constraint_D[env_c_base+sib]*xt2*xt2;
                    } else {
                        double D0 = constraint_D[env_c_base+par];
                        double mu2l = mul*mul;
                        double dml = D0/fmax(mu2l*(1+mu2l),1e-20);
                        double nmt = Nn-mul*Tl;
                        cone_cost_a += 0.5*dml*nmt*nmt;
                    }
                }
            }
            cone_cost_a = wf_reduce_sum_d(cone_cost_a);
            double cost_a = quad_cost_a + constr_cost_a + cone_cost_a;

            

            if (cost_a < cost_0 && !isnan(cost_a)) {
                ck_best_alpha = alpha;
                ck_accepted_cost = cost_a;
                break;
            }
            alpha *= 0.5;
            if (alpha < 1e-10) break;
        }

            if (__double_as_longlong(ck_best_alpha) != __double_as_longlong(best_alpha) ||
                __double_as_longlong(ck_accepted_cost) != __double_as_longlong(accepted_cost)) {
                if (t == 0) printf("R4LS MISMATCH env=%d it=%d %.17g %.17g | %.17g %.17g\n", env, newton_iter,
                                   ck_best_alpha, best_alpha, ck_accepted_cost, accepted_cost);
            } else if (t == 0 && par_k0 > 0) printf("R4LS okpar\n");
        }
#endif
        NPS(7);
        // ── 3h: Update qacc ──
        if (best_alpha > 0.0 && active)
            qacc_t += best_alpha * search_t;

        newton_iter_final = newton_iter + 1;
        if (newton_iter == 0) {
            first_alpha = best_alpha;
            first_gnorm = sqrt(gnorm2);
            first_cost0 = cost_0;
            first_cg_iters = cg_iters_done;
            first_cg_rnorm = cg_final_rnorm;
            first_search_mag = sqrt(search_mag2);
            first_grad_dot_search = grad_dot_search;
            first_rz0 = dbg_rz0;
            first_pAp0 = dbg_pAp0;
            first_alpha0 = dbg_alpha0;
            first_Hdiag_min = dbg_Hdiag_min;
            first_Hdiag_max = dbg_Hdiag_max;
            first_n_active = dbg_n_active_constraints;
        }

        // ── 3i: Check convergence using cached line-search cost ──
        if (best_alpha > 0.0) {
            double cur_cost = accepted_cost;
            double improvement = prev_cost - cur_cost;
            double scale = fmax(fabs(prev_cost), fabs(cur_cost));
            scale = fmax(scale, 1e-12);
            if (improvement >= 0.0 && improvement < solver_tolerance * scale && newton_iter >= 3)
                break;
            prev_cost = cur_cost;
        } else {
            if (newton_iter >= 3)
                break;
        }
    }

    NPS(8);
    // ── Compute final constraint forces (lambda) ──
    // s_qv is published BEFORE the release barrier: with fuse_mhat, wave 1
    // reads it for the M_hat solve while wave 0 runs the lambda tail below.
    if (active) s_qv[t] = qacc_t;
    NPOST(fuse_mhat ? NCMD_MHAT : NCMD_EXIT, 0.0);   // release the helper waves
    WSYNC();
    for (int c = t; c < nc; c += NEWTON_BLK) {
        double x = -constraint_rhs[env_c_base + c];
        if (use_jt) {
            const double* JTc = JTbase + c;
            for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                int v = __ffsll((unsigned long long)m) - 1;
                x += JTc[(size_t)v * JT_STRIDE] * s_qv[v];
            }
        } else {
            const double* Jc = constraint_J + env_j_base + (size_t)c * nv;
            for (unsigned long long m = nz_ok ? s_nzJ[c] : nz_full; m; m &= m - 1) {
                int v = __ffsll((unsigned long long)m) - 1;
                x += Jc[v] * s_qv[v];
            }
        }
        g_x[c] = x;
    }
    WSYNC();

    for (int c = t; c < nc; c += NEWTON_BLK) {
        double x = g_x[c];
        double Dc = constraint_D[env_c_base + c];
        double lo_c = constraint_lo[env_c_base + c];
        double hi_c = constraint_hi[env_c_base + c];
        int par_c = constraint_friction_parent[env_c_base + c];
        if (cone_type == 1 && (par_c >= 0 || g_is_en[c])) continue;
        if (lo_c < -1e15 && hi_c > 1e15) g_force[c] = -Dc * x;
        else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) g_force[c] = (x < 0.0) ? -Dc * x : 0.0;
        else if (par_c < 0) {
            double f = fabs(hi_c);
            double rf = (Dc > 1e-10) ? f / Dc : 0.0;
            if      (x <= -rf) g_force[c] =  f;
            else if (x >=  rf) g_force[c] = -f;
            else               g_force[c] = -Dc * x;
        } else g_force[c] = (x < 0.0) ? -Dc * x : 0.0;
    }
    WSYNC();

    if (cone_type == 1) {
        for (int c = t; c < nc; c += NEWTON_BLK) {
            int par = constraint_friction_parent[env_c_base + c];
            int sib = constraint_friction_sibling[env_c_base + c];
            if (par < 0 || par >= nc || sib < 0 || sib >= nc) continue;
            if (c >= sib) continue;
            int normal = par, t1 = c, t2 = sib;
            const int torsion = constraint_friction_sibling[env_c_base+normal];
            if (torsion >= 0 && torsion < nc) {
                const int rows[4] = {normal, t1, t2, torsion};
                double force[4];
                evaluate_contact4(rows, g_x, nullptr, 0.0,
                    constraint_D+env_c_base, constraint_friction_mu+env_c_base,
                    impr_inv, force, nullptr);
                for (int i = 0; i < 4; ++i) g_force[rows[i]] = force[i];
                continue;
            }
            double mu_fric = (double)constraint_friction_mu[env_c_base + c];
            double mu_f = mu_fric * impr_inv;
            double mu2_f = mu_f * mu_f;
            double N = g_x[normal] * mu_f;
            double u1 = g_x[t1] * mu_fric, u2 = g_x[t2] * mu_fric;
            double TT = u1*u1+u2*u2;
            double T = (TT > 0) ? sqrt(TT) : 0.0;
            if ((N >= mu_f*T)||(T<=0 && N>=0)) { g_force[normal]=0; g_force[t1]=0; g_force[t2]=0; }
            else if ((mu_f*N+T<=0)||(T<=0 && N<0)) {
                g_force[normal] = -constraint_D[env_c_base+normal]*g_x[normal];
                g_force[t1] = -constraint_D[env_c_base+t1]*g_x[t1];
                g_force[t2] = -constraint_D[env_c_base+t2]*g_x[t2];
            } else {
                double D0 = constraint_D[env_c_base+normal];
                double dm = D0/fmax(mu2_f*(1+mu2_f),1e-20);
                double nmt = N-mu_f*T;
                double Tsafe = fmax(T,1e-20);
                g_force[normal] = -dm*nmt*mu_f;
                g_force[t1] = (dm*nmt*mu_f/Tsafe)*u1*mu_fric;
                g_force[t2] = (dm*nmt*mu_f/Tsafe)*u2*mu_fric;
            }
        }
    }
    WSYNC();

    for (int c = t; c < nc; c += NEWTON_BLK)
        env_lam[c] = g_force[c];

    if (active) {
        if (!fuse_mhat) env_qacc[t] = qacc_t;   // else wave 1 writes M_hat^-1 M qacc
        if (qacc_warmstart)
            qacc_warmstart[env * nv + t] = qacc_t;
    }
#if NEWTON_PROF
    NPS(9);
    if (t == 0)
        printf("NPROF e=%d nc=%d ct=%d it=%d lsn=%d tot=%llu pre=%llu res=%llu H=%llu chol=%llu sub=%llu jv=%llu lsp=%llu ls=%llu misc=%llu tail=%llu p0=%llu p1=%llu p2=%llu p3=%llu p4=%llu p5=%llu p6=%llu p7=%llu h1a=%llu h1b=%llu h1c=%llu h2a=%llu h2b=%llu h2c=%llu h3a=%llu h3b=%llu h3c=%llu x0=%llu x1=%llu x2=%llu x3=%llu x4=%llu x5=%llu\n",
               env, nc, cone_type, newton_iter_final, np_ls, np_last - np_t0,
               np_acc[0], np_acc[1], np_acc[2], np_acc[3], np_acc[4], np_acc[5],
               np_acc[6], np_acc[7], np_acc[8], np_acc[9],
               np_pre[0], np_pre[1], np_pre[2], np_pre[3], np_pre[4], np_pre[5], np_pre[6], np_pre[7],
               s_hprof[1][0], s_hprof[1][1], s_hprof[1][2], s_hprof[2][0], s_hprof[2][1], s_hprof[2][2],
               s_hprof[3][0], s_hprof[3][1], s_hprof[3][2], np_x[0], np_x[1], np_x[2], np_x[3], np_x[4], np_x[5]);
#endif
    if (integ.on) newton_integ_tail(integ, env, njnt, nq, nv, jnt_type, jnt_qpos_adr, jnt_dof_adr, qacc);
}

// ── M_hat Euler implicit damping solve (MuJoCo eq. 6 post-processing) ───────
// After Newton produces constraint forces (lambda), compute the final qacc as:
//   qacc_int = (M + h*D)^{-1} * M * qacc_newton
// where M_hat = M + h * diag(damping).
// This matches MuJoCo's mj_EulerSkip which re-solves with augmented mass.
// Block-parallel Gauss-Jordan: one block per env, threads parallelize row ops.

constexpr int MHAT_BLK = 64;

// mhat_solve runs ONE 64-lane wave per env, so its barrier is a scheduling
// fence only.  Kept separate from WSYNC() so a future NEWTON_BLK change cannot
// silently turn this into a workgroup barrier (or, worse, the reverse).
#if MHAT_BLK <= 64
#define MSYNC() __builtin_amdgcn_wave_barrier()
#else
#define MSYNC() __syncthreads()
#endif

template <int NV_T>
HIPGL_GLOBAL void __launch_bounds__(MHAT_BLK, 1) kernel_mhat_solve_t(
    int num_envs, int nv_rt,
    const double* __restrict__ solver_M,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_damping,
    int njnt,
    double timestep,
    bool enable_damping,
    double*       __restrict__ qacc)
{
    // NV_T == 0 is the generic runtime-nv instantiation (mirrors newton).
    const int nv = (NV_T > 0) ? NV_T : nv_rt;
    int env = blockIdx.x;
    if (env >= num_envs) return;
    int t = threadIdx.x;
    bool active = (t < nv);
    if (nv > KERNEL_MAXNV) return;

    __shared__ double sA[KERNEL_MAXNV * LDS_STRIDE];
    __shared__ double srhs[KERNEL_MAXNV];

    const double* M = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    const double* qa = qacc + env * nv;

    if (active) {
        double s = 0.0;
        const double* src = &M[t * KERNEL_MAXNV];
        int j = 0;
        for (; j + 1 < nv; j += 2) {
            double2 pair = *reinterpret_cast<const double2*>(&src[j]);
            sA[t * LDS_STRIDE + j]   = pair.x;
            sA[t * LDS_STRIDE + j+1] = pair.y;
            s += pair.x * qa[j] + pair.y * qa[j+1];
        }
        if (j < nv) {
            double mij = src[j];
            sA[t * LDS_STRIDE + j] = mij;
            s += mij * qa[j];
        }
        srhs[t] = s;
    }
    MSYNC();

    if (enable_damping && t == 0) {
        // njnt-bounded, not 128 -- see the identical loop in kernel_newton_solve for why
        // the hardcoded bound silently injected a freed model's damping coefficients.
        for (int ji = 0; ji < njnt; ji++) {
            int da = jnt_dof_adr[ji];
            if (da < 0 || da >= nv) break;   // defensive only; njnt is the real bound
            int jt = jnt_type[ji];
            int ndof = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
            for (int k = 0; k < ndof && da + k < nv; k++)
                sA[(da + k) * LDS_STRIDE + (da + k)] += timestep * jnt_damping[ji];
        }
    }
    MSYNC();

    for (int k = 0; k < nv; k++) {
        // -- wave-parallel partial-pivot argmax over rows [k, nv) -----------
        // The old lane-0 scan ascended i and took only strictly-greater, so the
        // LOWEST index wins a tie.  (max |v|, then min i) is associative and
        // commutative, so this xor tree selects the SAME row, bit for bit --
        // and it replaces up to 43 serial dependent LDS loads with one load
        // plus 6 shuffle rounds.
        double bv; int bi;
        if (t >= k && t < nv) { bv = fabs(sA[t * LDS_STRIDE + k]); bi = t; }
        else                  { bv = -1.0;                         bi = KERNEL_MAXNV + 1; }
        #pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            double ov = wf_shfl_xor_d(bv, off);
            int    oi = __shfl_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        const int piv = bi;     // wave-uniform; no LDS round-trip needed

        if (piv != k && t < nv) {
            double tmp = sA[k * LDS_STRIDE + t];
            sA[k * LDS_STRIDE + t] = sA[piv * LDS_STRIDE + t];
            sA[piv * LDS_STRIDE + t] = tmp;
        }
        if (piv != k && t == 0) {
            double tmp = srhs[k]; srhs[k] = srhs[piv]; srhs[piv] = tmp;
        }
        MSYNC();

        double diag = sA[k * LDS_STRIDE + k];
        if (fabs(diag) < 1e-20) diag = 1e-20;
        double inv_d = 1.0 / diag;
        // After this, lane j OWNS row-k element j in a REGISTER.  The trailing
        // update below fetches it with readlane instead of re-reading LDS:
        // identical bits, zero lgkm traffic.  Both operands of the old
        // `sA[t][j] -= factor * sA[k][j]` were LDS with a lane-dependent base,
        // so the compiler had to assume the row-k load aliased the row-t store
        // and drained lgkmcnt(0) once per j.  With row k in registers the only
        // LDS pair left differs by a CONSTANT offset, which it can prove
        // independent, unroll and batch.
        double rowk = 0.0;
        if (t < nv) {
            rowk = sA[k * LDS_STRIDE + t] * inv_d;
            sA[k * LDS_STRIDE + t] = rowk;
        }
        if (t == 0) srhs[k] *= inv_d;
        MSYNC();

        const double rhsk = srhs[k];
        if (active && t != k) {
            double factor = sA[t * LDS_STRIDE + k];
            double* rowt = &sA[t * LDS_STRIDE];
            if (NV_T > 0) {
                #pragma unroll
                for (int j = 0; j < NV_T; j++)
                    rowt[j] -= factor * wf_broadcast_d(rowk, j);
            } else {
                for (int j = 0; j < nv; j++)
                    rowt[j] -= factor * wf_broadcast_d(rowk, j);
            }
            srhs[t] -= factor * rhsk;
        }
        MSYNC();
    }

    if (active)
        qacc[env * nv + t] = srhs[t];
}

// ── Register-resident Gauss-Jordan for compile-time nv (bit-exact with
// kernel_mhat_solve_t) ─────────────────────────────────────────────────────
// Lane t keeps ORIGINAL row t of [A | rhs] in registers for the whole solve.
// Partial-pivot row swaps are replaced by a lane->position map (pos): the row
// the LDS kernel holds at position p is the row whose lane has pos == p, so
// the argmax (max |a|, then LOWEST POSITION) picks the same row and every row
// sees the identical sequence  a[j] -= factor * pivrow[j]  (same operands,
// same order).  Columns j <= k are never read again after step k (pivot
// search and factor read only column k at step k; the output is rhs), so the
// trailing update skips them.  No LDS, no barriers, ~half the FMAs.
template <int NV_T>
HIPGL_GLOBAL void __launch_bounds__(MHAT_BLK, 1) kernel_mhat_solve_reg(
    int num_envs,
    const double* __restrict__ solver_M,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_damping,
    int njnt,
    double timestep,
    bool enable_damping,
    double*       __restrict__ qacc)
{
    static_assert(NV_T > 0 && NV_T <= 64, "register GJ needs 0 < nv <= 64");
    const int env = blockIdx.x;
    if (env >= num_envs) return;
    mhat_reg_body<NV_T>(env, (int)threadIdx.x, solver_M, qacc + env * NV_T,
                        jnt_type, jnt_dof_adr, jnt_damping, njnt, timestep,
                        enable_damping, qacc);
}

// ── Fused constraint buffer clear (replaces 5 separate hipMemsetAsync calls) ─

HIPGL_GLOBAL void kernel_clear_constraint_buffers(
    int num_envs,
    int max_c,
    bool clear_lam,
    int*    __restrict__ nconstraint,
    double* __restrict__ constraint_lam,
    int*    __restrict__ friction_parent,
    int*    __restrict__ friction_sibling,
    double* __restrict__ friction_mu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    if (idx < num_envs)
        nconstraint[idx] = 0;

    int total = num_envs * max_c;
    for (int i = idx; i < total; i += stride) {
        if (clear_lam)
            constraint_lam[i] = 0.0;
        friction_parent[i]  = -1;
        friction_sibling[i] = -1;
        friction_mu[i]      = 0.0;
    }
}

void launch_clear_constraint_buffers(
    DevicePhysicsData& data,
    bool clear_lam,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0) return;
    int total = ne * MAX_CONSTRAINTS;
    int blk = 256;
    int grid = (total + blk - 1) / blk;
    kernel_clear_constraint_buffers<<<grid, blk, 0, stream>>>(
        ne, MAX_CONSTRAINTS, clear_lam,
        data.nconstraint,
        data.constraint_lam,
        data.constraint_friction_parent,
        data.constraint_friction_sibling,
        data.constraint_friction_mu);
}

// ── PGS fallback (kept for SOLVER_PGS) ─────────────────────────────────────

constexpr int MAX_A_DIM = 256;

HIPGL_GLOBAL void kernel_compute_A_matrix(
    int num_envs, int nv,
    const double* __restrict__ constraint_J,
    const int*   __restrict__ nconstraint,
    const double* __restrict__ M_inv_diag,
    float*       __restrict__ A,
    float regularization)
{
    int env = blockIdx.y;
    if (env >= num_envs) return;
    int nc_val = nconstraint[env];
    if (nc_val <= 0) return;
    if (nc_val > MAX_A_DIM) nc_val = MAX_A_DIM;
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int ii = flat / nc_val;
    int jj = flat % nc_val;
    if (ii >= nc_val) return;
    size_t env_j_base = (size_t)env * MAX_CONSTRAINTS * nv;
    const double* Ji = constraint_J + env_j_base + (size_t)ii * nv;
    const double* Jj = constraint_J + env_j_base + (size_t)jj * nv;
    const double* M = M_inv_diag + env * MAX_DOF;
    float val = (ii == jj) ? regularization : 0.0f;
    for (int v = 0; v < nv; v++) val += (float)(Ji[v] * M[v] * Jj[v]);
    A[(size_t)env * MAX_A_DIM * MAX_A_DIM + (size_t)ii * MAX_A_DIM + jj] = val;
}

HIPGL_GLOBAL void kernel_pgs_iteration_fast(
    int num_envs, int iterations,
    const float* __restrict__ A,
    const double* __restrict__ constraint_rhs,
    double*      __restrict__ constraint_lo,
    double*      __restrict__ constraint_hi,
    double*      __restrict__ constraint_lam,
    const int*   __restrict__ constraint_friction_parent,
    const double* __restrict__ constraint_friction_mu,
    const int*   __restrict__ nconstraint)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;
    int nc_val = nconstraint[env];
    if (nc_val <= 0) return;
    if (nc_val > MAX_A_DIM) nc_val = MAX_A_DIM;
    size_t env_base = (size_t)env * MAX_CONSTRAINTS;
    const float* A_env = A + (size_t)env * MAX_A_DIM * MAX_A_DIM;
    const double* rhs_ptr = constraint_rhs + env_base;
    double* lo = constraint_lo + env_base;
    double* hi = constraint_hi + env_base;
    double* lam = constraint_lam + env_base;
    const int* fp = constraint_friction_parent + env_base;
    const double* fmu = constraint_friction_mu + env_base;
    for (int iter = 0; iter < iterations; iter++) {
        for (int ii = 0; ii < nc_val; ii++) {
            int parent = fp[ii];
            if (parent >= 0 && parent < nc_val) {
                double mu = fmu[ii];
                double nl = lam[parent]; if (nl < 0) nl = 0;
                lo[ii] = -mu * nl; hi[ii] = mu * nl;
            }
            double residual = rhs_ptr[ii];
            const float* Ai = A_env + (size_t)ii * MAX_A_DIM;
            for (int jj = 0; jj < nc_val; jj++) {
                if (jj != ii) residual -= (double)Ai[jj] * lam[jj];
            }
            double diag = (double)Ai[ii]; if (diag < 1e-10) diag = 1e-10;
            double new_lam = residual / diag;
            if (new_lam < lo[ii]) new_lam = lo[ii];
            if (new_lam > hi[ii]) new_lam = hi[ii];
            lam[ii] = new_lam;
        }
    }
}

// ── Host launch wrappers ────────────────────────────────────────────────────

// constraint_setup -> Newton fusion bookkeeping: when the rows kernel already
// built the joint rows and the JT mirror, the next launch_newton_solve on the
// same stream/data skips kernel_transpose_J.  HP_FUSE_CS=0 restores the old
// three-kernel sequence.
namespace {
struct JtFuseToken { const double* J = nullptr; hipStream_t stream = nullptr; bool pending = false; };
thread_local JtFuseToken g_jt_fuse;
bool cs_fuse_enabled() {
    static const bool en = [] {
        const char* e = getenv("HP_FUSE_CS");
        return !(e && e[0] == '0');
    }();
    return en;
}
}  // namespace

static bool pyr_setup_ok(const DevicePhysicsModel& model, const DevicePhysicsData& data,
                         const SolverOptions& opts)
{
    (void)opts;
    const bool has_eq = (model.nequality > 0 && model.eq_type);
    return cs_fuse_enabled() && model.cone_type == 0 && !has_eq && data.num_envs > 0
        && model.nv > 0 && model.nv <= PYR_NVC && model.body_invweight0_d && data.jnt_xaxis_d
        && model.body_chain_start && model.body_chain_len && model.body_chain_jnt_idx
        && model.body_chain_jnt_body && model.body_chain_parent
        && model.nbody <= CS_LDS_NB && model.njnt <= CS_LDS_NJ && model.njnt <= CS_BLK
        && model.body_chain_total >= 0 && model.body_chain_total <= CS_LDS_CH;
}

static void launch_pyr_setup(const DevicePhysicsModel& model, DevicePhysicsData& data,
                             const SolverOptions& opts, hipStream_t stream, bool fuse_gravcomp)
{
    const int ne = data.num_envs;
    const bool want_jl = (opts.enable_limits && model.jnt_limited);
    g_jt_fuse.pending = true;          // JT mirror built here: Newton skips kernel_transpose_J
    g_jt_fuse.J = data.constraint_J;
    g_jt_fuse.stream = stream;
    const int ngc = (fuse_gravcomp && model.nv > 0) ? model.n_gravcomp : 0;
    dim3 gridp(PYR_GX + 2, ne);
    hipLaunchKernelGGL(kernel_constraint_setup_pyr, gridp, dim3(PYR_BLK), 0, stream,
        ne, model.nv, model.nbody, model.njnt, model.nq,
        model.jnt_type, model.jnt_dof_adr, model.jnt_qpos_adr,
        model.jnt_axis_d, model.jnt_range_d, model.jnt_limited,
        model.jnt_frictionloss_d, model.dof_invweight0_d,
        model.geom_condim, model.geom_friction_d,
        model.geom_solref_d, model.geom_solimp_d,
        data.solver_M_inv_diag, model.body_invweight0_d,
        (double)model.timestep, (double)opts.impratio,
        data.contacts, data.ncon, data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        data.qpos_d, data.qvel_d,
        model.body_chain_start, model.body_chain_len,
        model.body_chain_jnt_idx, model.body_chain_jnt_body, model.body_chain_parent,
        model.body_chain_total,
        data.constraint_J, data.constraint_rhs, data.constraint_D,
        data.constraint_lo, data.constraint_hi,
        data.constraint_friction_parent, data.constraint_friction_sibling,
        data.constraint_friction_mu,
        data.nconstraint, model.max_condim >= 4 ? 6 : 4, want_jl, (double)model.max_depen_vel,
        ngc, model.gravcomp_body, model.gravcomp_scale_d, model.body_mass_d, model.body_ipos_d,
        ngc > 0 ? (double)model.gravity_d[0] : 0.0, ngc > 0 ? (double)model.gravity_d[1] : 0.0,
        ngc > 0 ? (double)model.gravity_d[2] : 0.0, data.qfrc_bias);
}

// One-launch constraint setup for the Newton step: gravity compensation +
// contact/joint rows + JT mirror, and NO kernel_clear_constraint_buffers
// (every row < nconstraint is fully written, the count is stored, and the
// solver reads rows < nconstraint only).  Callers must have launched the bias
// (qfrc_bias writer) earlier on `stream` and must NOT also call launch_gravcomp.
bool constraint_setup_fused_ok(const DevicePhysicsModel& model, const DevicePhysicsData& data,
                               const SolverOptions& opts)
{
    if (!pyr_setup_ok(model, data, opts)) return false;
    if (model.n_gravcomp <= 0) return true;
    return model.n_gravcomp <= GCB_BLK && model.nv <= KERNEL_MAXNV && model.gravcomp_body
        && model.gravcomp_scale_d && model.body_mass_d && model.body_ipos_d;
}

static bool cs_fast_ok(const DevicePhysicsModel& model, const DevicePhysicsData& data)
{
    static const bool en = [] { const char* e = getenv("HP_CSETUP_FUSE"); return !(e && e[0] == '0'); }();
    return en && data.num_envs > 0 && model.njnt <= 64 && model.nv <= PYR_NVC
        && (model.n_gravcomp <= 0 || (model.n_gravcomp <= GCB_BLK && data.jnt_xaxis_d));
}

static void launch_constraint_setup_fast(const DevicePhysicsModel& model, DevicePhysicsData& data,
                                         const SolverOptions& opts, hipStream_t stream)
{
    const int ne = data.num_envs;
    const bool want_jl = (opts.enable_limits && model.jnt_limited);
    g_jt_fuse.pending = true;          // JT mirror built here: Newton skips kernel_transpose_J
    g_jt_fuse.J = data.constraint_J;
    g_jt_fuse.stream = stream;
    const int ngc = model.nv > 0 ? model.n_gravcomp : 0;
    dim3 grid(PYR_GX + (ngc > 0 ? 2 : 1), ne);
    hipLaunchKernelGGL(kernel_constraint_setup_pyr_fast, grid, dim3(PYF_BLK), 0, stream,
        ne, model.nv, model.nbody, model.njnt, model.nq,
        model.jnt_type, model.jnt_dof_adr, model.jnt_qpos_adr,
        model.jnt_axis_d, model.jnt_range_d, model.jnt_limited,
        model.jnt_frictionloss_d, model.dof_invweight0_d,
        model.geom_condim, model.geom_friction_d,
        model.geom_solref_d, model.geom_solimp_d,
        data.solver_M_inv_diag, model.body_invweight0_d,
        (double)model.timestep, (double)opts.impratio,
        data.contacts, data.ncon, data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        data.qpos_d, data.qvel_d,
        model.body_chain_start, model.body_chain_len,
        model.body_chain_jnt_idx, model.body_chain_jnt_body, model.body_chain_parent,
        model.body_chain_total,
        data.constraint_J, data.constraint_rhs, data.constraint_D,
        data.constraint_lo, data.constraint_hi,
        data.constraint_friction_parent, data.constraint_friction_sibling,
        data.constraint_friction_mu,
        data.nconstraint, model.max_condim >= 4 ? 6 : 4, want_jl, (double)model.max_depen_vel,
        ngc, model.gravcomp_body, model.gravcomp_scale_d, model.body_mass_d, model.body_ipos_d,
        ngc > 0 ? (double)model.gravity_d[0] : 0.0, ngc > 0 ? (double)model.gravity_d[1] : 0.0,
        ngc > 0 ? (double)model.gravity_d[2] : 0.0, data.qfrc_bias);
}

void launch_constraint_setup_fused(const DevicePhysicsModel& model, DevicePhysicsData& data,
                                   const SolverOptions& opts, hipStream_t stream)
{
    if (cs_fast_ok(model, data)) { launch_constraint_setup_fast(model, data, opts, stream); return; }
    launch_pyr_setup(model, data, opts, stream, /*fuse_gravcomp=*/true);
}

void launch_constraint_setup(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0) return;

    // Contact constraints
    const bool has_eq = (model.nequality > 0 && model.eq_type);
    const bool want_jl = (opts.enable_limits && model.jnt_limited);
    // The row-parallel kernel builds the elliptic-style normal + 2 tangent rows only; a
    // pyramidal-cone model goes through kernel_constraint_setup's MuJoCo edge rows.
    const bool rows_ok = model.body_invweight0_d && model.nv <= KERNEL_MAXNV && model.cone_type == 1;
    const bool fuse = cs_fuse_enabled() && rows_ok && !has_eq && model.njnt <= CS_BLK;
    const bool lds = cs_fuse_enabled() && data.jnt_xaxis_d && model.body_chain_start
                     && model.body_chain_len && model.body_chain_jnt_idx
                     && model.body_chain_jnt_body && model.body_chain_parent
                     && model.nbody <= CS_LDS_NB && model.njnt <= CS_LDS_NJ
                     && model.body_chain_total >= 0 && model.body_chain_total <= CS_LDS_CH
                     && ne <= 64;  // larger LDS footprint: keep the big-batch grids on the lean variant
    const int fuse_flags = (fuse ? (CS_FUSE_JT | (want_jl ? CS_FUSE_JL : 0)) : 0)
                         | (lds ? CS_FUSE_LDS : 0);
    bool fuse_flags_pyr_jl = false;   // pyramidal kernel built the joint rows
    g_jt_fuse.pending = fuse;
    g_jt_fuse.J = data.constraint_J;
    g_jt_fuse.stream = stream;
    if (rows_ok) {
        dim3 gridr(CS_GRID_X + ((fuse_flags & CS_FUSE_JL) ? 1 : 0), ne);
        auto cs_rows_fn = lds ? kernel_constraint_setup_rows<true> : kernel_constraint_setup_rows<false>;
        hipLaunchKernelGGL(cs_rows_fn, gridr, dim3(CS_BLK), 0, stream,
            ne, model.nv, model.nbody, model.njnt,
            model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
            model.geom_condim, model.geom_friction_d,
            model.geom_solref_d, model.geom_solimp_d,
            model.body_invweight0_d,
            (double)model.timestep, (double)opts.impratio,
            data.contacts, data.ncon, data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.qvel_d,
            model.body_chain_start, model.body_chain_len,
            model.body_chain_jnt_idx, model.body_chain_jnt_body, model.body_chain_parent,
            data.constraint_J, data.constraint_rhs, data.constraint_D,
            data.constraint_lo, data.constraint_hi,
            data.constraint_friction_parent, data.constraint_friction_sibling,
            data.constraint_friction_mu,
            data.nconstraint,
            fuse_flags, model.nq,
            model.jnt_qpos_adr, model.jnt_range_d, model.jnt_limited,
            model.jnt_frictionloss_d, data.solver_M_inv_diag, model.dof_invweight0_d,
            data.qpos_d, model.body_chain_total, (double)model.max_depen_vel);
    } else if (pyr_setup_ok(model, data, opts)) {
        // Pyramidal cones: LDS-resident rows + fused joint rows + JT mirror.
        launch_pyr_setup(model, data, opts, stream, /*fuse_gravcomp=*/false);
        fuse_flags_pyr_jl = true;
    } else {
    g_jt_fuse.pending = false;
    int max_items = MAX_CONTACTS;
    const int block = 256;
    int grid_x = (max_items + block - 1) / block;
    dim3 grid(grid_x, ne);
    hipLaunchKernelGGL(kernel_constraint_setup, grid, block, 0, stream,
        ne, model.nv, model.nbody, model.njnt,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr, model.jnt_qpos_adr,
        model.jnt_axis_d, model.jnt_range_d, model.jnt_limited,
        model.body_parent, model.body_jnt_adr, model.body_jnt_num,
        model.geom_condim, model.geom_friction_d,
        model.geom_solref_d, model.geom_solimp_d,
        model.geom_body, model.body_mass_d,
        data.solver_M_inv_diag,
        model.body_invweight0_d,
        (double)model.timestep, (double)opts.impratio,
        model.nequality, model.eq_type, model.eq_body1, model.eq_body2,
        data.contacts, data.ncon, data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.qpos_d, data.qvel_d,
        (double)opts.contact_slip, opts.enable_limits,
        model.body_chain_start, model.body_chain_len,
        model.body_chain_jnt_idx, model.body_chain_jnt_body, model.body_chain_parent,
        data.constraint_J, data.constraint_rhs, data.constraint_D,
        data.constraint_lo, data.constraint_hi,
        data.constraint_friction_parent, data.constraint_friction_sibling,
        data.constraint_friction_mu,
        data.nconstraint, model.cone_type, model.max_condim >= 4 ? 6 : 4, (double)model.max_depen_vel);
    }

    // Phase 7: parallel joint limits (1 thread per joint per env)
    if (want_jl && !fuse_flags_pyr_jl && !(g_jt_fuse.pending && (fuse_flags & CS_FUSE_JL))) {
        const int block2 = 256;
        int grid2_x = (model.njnt + block2 - 1) / block2;
        dim3 grid2(grid2_x, ne);
        hipLaunchKernelGGL(kernel_joint_limit_constraints, grid2, block2, 0, stream,
            ne, model.nv, model.nq, model.njnt,
            model.jnt_type, model.jnt_qpos_adr, model.jnt_dof_adr,
            model.jnt_range_d, model.jnt_limited,
            model.jnt_frictionloss_d,
            data.solver_M_inv_diag,
            model.dof_invweight0_d,
            (double)model.timestep,
            data.qpos_d, data.qvel_d,
            data.constraint_J, data.constraint_rhs, data.constraint_D,
            data.constraint_lo, data.constraint_hi,
            data.constraint_friction_parent, data.constraint_friction_mu,
            data.nconstraint);
    }

    // Phase 7: parallel equality constraints (1 thread per equality per env)
    if (model.nequality > 0 && model.eq_type) {
        const int block3 = 256;
        int grid3_x = (model.nequality + block3 - 1) / block3;
        dim3 grid3(grid3_x, ne);
        hipLaunchKernelGGL(kernel_equality_constraints, grid3, block3, 0, stream,
            ne, model.nv, model.nbody, model.njnt,
            model.nequality, model.eq_type, model.eq_body1, model.eq_body2,
            model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
            model.body_parent, model.body_jnt_adr, model.body_jnt_num,
            data.solver_M_inv_diag,
            model.body_invweight0_d,
            (double)model.timestep,
            data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.qvel_d,
            model.body_chain_start, model.body_chain_len,
            model.body_chain_jnt_idx, model.body_chain_jnt_body, model.body_chain_parent,
            data.constraint_J, data.constraint_rhs, data.constraint_D,
            data.constraint_lo, data.constraint_hi,
            data.constraint_friction_parent, data.constraint_friction_mu,
            data.nconstraint);
    }
}

void launch_build_mass_matrix(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    int ne = data.num_envs;
    int nv = model.nv;
    if (ne <= 0 || nv <= 0 || nv > KERNEL_MAXNV) return;

    int total_entries = nv * nv;
    const int blk = 256;
    int grd_x = (total_entries + blk - 1) / blk;
    dim3 grid(grd_x, ne);
    if (model.nbody <= 64 && model.njnt <= 64) {
        kernel_build_mass_matrix_t<true><<<grid, blk, 0, stream>>>(
        ne, nv, model.njnt, model.nbody,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr,
        model.jnt_axis_d, model.body_mass_d, model.body_inertia_d,
        model.body_ipos_d, model.body_iquat_d, model.body_parent,
        data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        model.jnt_damping_d, model.jnt_armature_d,
        model.timestep, opts.enable_damping,
        data.solver_M, data.solver_M_inv_diag);
    } else {
        kernel_build_mass_matrix_t<false><<<grid, blk, 0, stream>>>(
        ne, nv, model.njnt, model.nbody,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr,
        model.jnt_axis_d, model.body_mass_d, model.body_inertia_d,
        model.body_ipos_d, model.body_iquat_d, model.body_parent,
        data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        model.jnt_damping_d, model.jnt_armature_d,
        model.timestep, opts.enable_damping,
        data.solver_M, data.solver_M_inv_diag);
    }
}

// ── Kernel: invert mass matrix per env (Gauss-Jordan, one block per env) ────
// Block-parallel: threads cooperate on row/column operations.
//
// DEAD CODE, compiled out above KERNEL_MAXNV=48. launch_invert_mass_matrix is
// declared in physics_engine.cpp:91 but never called, and nothing reads
// solver_M_inv -- the live solver uses solver_M_inv_diag, a different buffer.
//
// It is excluded rather than fixed because it is the ONLY thing standing
// between this build and KERNEL_MAXNV=64: unlike every live kernel it holds TWO
// nv x nv LDS tiles (sA and sI), so it needs 2 * 64 * 65 * 8 = 65.0 KB against
// the 64 KB gfx90a budget -- over by 1 KB -- while the live newton solve needs
// only 34.5 KB. If this kernel is ever revived it must tile the Gauss-Jordan
// elimination or share one LDS buffer between sA and sI.
#if KERNEL_MAXNV_VALUE <= 48

HIPGL_GLOBAL void __launch_bounds__(MHAT_BLK, 1) kernel_invert_mass_matrix(
    int num_envs, int nv,
    const double* __restrict__ solver_M,
    double*       __restrict__ solver_M_inv)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    int t = threadIdx.x;
    bool active = (t < nv);
    if (nv > KERNEL_MAXNV) return;

    constexpr int MNV = KERNEL_MAXNV;
    constexpr int MLDS = MNV + 1;
    __shared__ double sA[MNV * MLDS];
    __shared__ double sI[MNV * MLDS];
    __shared__ int s_pivot;

    const double* M = solver_M + (size_t)env * MNV * MNV;
    double* Minv    = solver_M_inv + (size_t)env * MNV * MNV;

    if (active) {
        const double* src = &M[t * MNV];
        int j = 0;
        for (; j + 1 < nv; j += 2) {
            double2 pair = *reinterpret_cast<const double2*>(&src[j]);
            sA[t * MLDS + j]   = pair.x;
            sA[t * MLDS + j+1] = pair.y;
            sI[t * MLDS + j]   = (t == j)   ? 1.0 : 0.0;
            sI[t * MLDS + j+1] = (t == j+1) ? 1.0 : 0.0;
        }
        if (j < nv) {
            sA[t * MLDS + j] = src[j];
            sI[t * MLDS + j] = (t == j) ? 1.0 : 0.0;
        }
    }
    WSYNC();

    for (int k = 0; k < nv; k++) {
        if (t == 0) {
            int piv = k;
            double best = fabs(sA[k * MLDS + k]);
            for (int i = k + 1; i < nv; i++) {
                double v = fabs(sA[i * MLDS + k]);
                if (v > best) { best = v; piv = i; }
            }
            s_pivot = piv;
        }
        WSYNC();

        int piv = s_pivot;
        if (piv != k && t < nv) {
            double tmp = sA[k * MLDS + t];
            sA[k * MLDS + t] = sA[piv * MLDS + t];
            sA[piv * MLDS + t] = tmp;
            tmp = sI[k * MLDS + t];
            sI[k * MLDS + t] = sI[piv * MLDS + t];
            sI[piv * MLDS + t] = tmp;
        }
        WSYNC();

        double diag = sA[k * MLDS + k];
        if (fabs(diag) < 1e-30) diag = 1e-30;
        double inv_diag = 1.0 / diag;
        if (t < nv) {
            sA[k * MLDS + t] *= inv_diag;
            sI[k * MLDS + t] *= inv_diag;
        }
        WSYNC();

        if (active && t != k) {
            double factor = sA[t * MLDS + k];
            for (int j = 0; j < nv; j++) {
                sA[t * MLDS + j] -= factor * sA[k * MLDS + j];
                sI[t * MLDS + j] -= factor * sI[k * MLDS + j];
            }
        }
        WSYNC();
    }

    if (active)
        for (int j = 0; j < nv; j++)
            Minv[t * MNV + j] = sI[t * MLDS + j];
}

void launch_invert_mass_matrix(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    hipStream_t stream)
{
    int ne = data.num_envs;
    int nv = model.nv;
    if (ne <= 0 || nv <= 0 || nv > KERNEL_MAXNV) return;
    kernel_invert_mass_matrix<<<ne, MHAT_BLK, 0, stream>>>(ne, nv, data.solver_M, data.solver_M_inv);
}

#else   // KERNEL_MAXNV_VALUE > 48

// The symbol is kept so any existing caller still links; it aborts rather than
// silently returning an uninitialised solver_M_inv, which is the failure mode
// the nv guard in PhysicsEngine::init exists to prevent.
void launch_invert_mass_matrix(
    const DevicePhysicsModel&, DevicePhysicsData&, hipStream_t)
{
    fprintf(stderr,
            "instinct_sim: launch_invert_mass_matrix is compiled out at "
            "KERNEL_MAXNV=%d (needs 65 KB LDS, budget is 64 KB). It had no "
            "callers when it was excluded; reviving it requires tiling the "
            "Gauss-Jordan elimination.\n", KERNEL_MAXNV);
    abort();
}

#endif  // KERNEL_MAXNV_VALUE <= 48

#if KERNEL_MAXNV_VALUE > 64
#include "physics_solver_wide.inc"
static int wide_prof() {
    static const int p = [] { const char* e = getenv("HP_WIDE_PROF"); return (e && e[0] == '1') ? 1 : 0; }();
    return p;
}
#endif

// Newton -> M_hat fusion bookkeeping.  Every engine call site issues
// launch_mhat_solve (iff opts.enable_damping) immediately after
// launch_newton_solve on the same stream and data.  When the Newton launch
// folded the M_hat solve into its tail, the matching launch_mhat_solve call
// consumes this token and issues nothing.  Any other call pattern still gets
// the standalone kernel.
namespace {
struct MhatFuseToken { const double* qacc = nullptr; hipStream_t stream = nullptr; bool pending = false; };
thread_local MhatFuseToken g_mhat_fuse;
bool mhat_fuse_enabled() {
    static const bool en = [] {
        const char* e = getenv("HP_NEWTON_FUSE_MHAT");
        return !(e && e[0] == '0');
    }();
    return en;
}
}  // namespace

template <int NV_T>
static void newton_launch_impl(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    bool warm_start,
    int ne, int nv,
    const NewtonInteg& integ)
{
    const bool fuse_mhat = (NV_T > 0 && NV_T <= 64) && opts.enable_damping && mhat_fuse_enabled();
    g_mhat_fuse.pending = fuse_mhat;
    g_mhat_fuse.qacc = data.qacc;
    g_mhat_fuse.stream = stream;
    const bool jt_prebuilt = g_jt_fuse.pending && g_jt_fuse.J == data.constraint_J
                             && g_jt_fuse.stream == stream;
    g_jt_fuse.pending = false;
    if (!jt_prebuilt) {
        // Build the column-major mirror of J that the solver's c-loops read.
        const int tblk = 256;
        int tgrd_x = (JT_STRIDE * nv + tblk - 1) / tblk;
        dim3 tgrid(tgrd_x, ne);
        kernel_transpose_J<<<tgrid, tblk, 0, stream>>>(
            ne, nv, data.nconstraint, data.constraint_J);
    }
    {
        const int blk = NEWTON_THREADS;   // 4 waves per env (see NEWTON_WAVES)
        int grd = ne;
        kernel_newton_solve_t<NV_T><<<grd, blk, 0, stream>>>(
            ne, nv,
            data.solver_M, data.solver_M_inv,
            data.qfrc_bias, data.qfrc_fluid, data.qfrc_applied_d,
            data.ctrl_d, model.nu,
            model.actuator_jnt, model.actuator_type,
            model.actuator_gain_d, model.actuator_kv_d,
            model.actuator_biasprm_d,
            model.actuator_forcerange_d, model.actuator_ctrlrange_d,
            model.jnt_type, model.jnt_dof_adr, model.jnt_qpos_adr,
            model.jnt_damping_d, model.jnt_stiffness_d, model.jnt_frictionloss_d,
            model.njnt, opts.enable_damping,
            data.qpos_d, data.qvel_d, model.qpos0_d, model.nq,
            (double)model.timestep,
            data.constraint_J, data.constraint_rhs, data.constraint_D,
            data.nconstraint,
            data.qacc, data.constraint_lam,
            data.constraint_lo, data.constraint_hi,
            data.constraint_friction_parent, data.constraint_friction_sibling,
            data.constraint_friction_mu, model.cone_type, (double)opts.impratio,
            warm_start, data.qacc_warmstart,
            data.newton_x, data.newton_force, data.newton_state,
            data.newton_jv, data.newton_is_elliptic,
            opts.tolerance, opts.iterations, fuse_mhat, integ);
    }
}

bool launch_newton_solve_integ(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    bool warm_start,
    bool do_integrate);

void launch_newton_solve(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    bool warm_start)
{
    launch_newton_solve_integ(model, data, opts, stream, warm_start, false);
}

bool launch_newton_solve_integ(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    bool warm_start,
    bool do_integrate)
{
    int ne = data.num_envs;
    int nv = model.nv;
    if (ne <= 0 || nv <= 0 || nv > KERNEL_MAXNV) return false;
#if KERNEL_MAXNV_VALUE > 64
    if (nv > NEWTON_BLK) {
        // One lane per dof does not fit a wave: physics_solver_wide.inc. No
        // M_hat / integrate fusion; the caller runs both kernels itself.
        g_mhat_fuse.pending = false;
        g_jt_fuse.pending = false;
        kernel_newton_solve_wide<<<ne, WIDE_BLK, 0, stream>>>(
            ne, nv, data.solver_M,
            data.qfrc_bias, data.qfrc_fluid, data.qfrc_applied_d,
            data.ctrl_d, model.nu,
            model.actuator_jnt, model.actuator_type,
            model.actuator_gain_d, model.actuator_kv_d,
            model.actuator_biasprm_d,
            model.actuator_forcerange_d, model.actuator_ctrlrange_d,
            model.jnt_type, model.jnt_dof_adr, model.jnt_qpos_adr,
            model.jnt_damping_d, model.jnt_stiffness_d,
            model.njnt, opts.enable_damping,
            data.qpos_d, data.qvel_d, model.qpos0_d, model.nq,
            data.constraint_J, data.constraint_rhs, data.constraint_D,
            data.nconstraint,
            data.qacc, data.constraint_lam,
            data.constraint_lo, data.constraint_hi,
            data.constraint_friction_parent, data.constraint_friction_sibling,
            data.constraint_friction_mu, model.cone_type, (double)opts.impratio,
            warm_start, data.qacc_warmstart,
            data.newton_x, data.newton_force, data.newton_state,
            data.newton_jv, data.newton_is_elliptic,
            opts.tolerance, opts.iterations, wide_prof());
        return false;
    }
#endif
    NewtonInteg integ{};
    static const bool res_en = [] { const char* e = getenv("HP_NEWTON_RESF"); return !(e && e[0] == '0'); }();
    integ.res_fuse = res_en ? 1 : 0;
    static const bool lre_en = [] { const char* e = getenv("HP_NEWTON_LREUSE"); return !(e && e[0] == '0'); }();
    integ.l_reuse = lre_en ? 1 : 0;
    static const bool dun_en = [] { const char* e = getenv("HP_NEWTON_DOTUNR"); return !(e && e[0] == '0'); }();
    integ.dot_unr = dun_en ? 1 : 0;
    static const bool srg_en = [] { const char* e = getenv("HP_NEWTON_STAGEREG"); return !(e && e[0] == '0'); }();
    integ.stage_reg = srg_en ? 1 : 0;
    // Only when the M_hat solve is folded in too (or damping is off): then the
    // Newton kernel is the last writer of qacc before the integrate.
#if KERNEL_MAXNV_VALUE <= 64
    const bool mhat_in = !opts.enable_damping ||
                         ((nv == 20 || nv == 37 || nv == 43 || nv == 51 || nv == 57) && mhat_fuse_enabled());
#else
    const bool mhat_in = !opts.enable_damping;   // no specialisations below, so no M_hat fusion
#endif
    static const bool integ_en = [] { const char* e = getenv("HP_NEWTON_INTEG"); return !(e && e[0] == '0'); }();
    if (do_integrate && integ_en && mhat_in && model.njnt > 0) {
        integ.on = 1;
        integ.dt = (double)model.timestep;
        integ.jnt_range_d = model.jnt_range_d;
        integ.jnt_limited = model.jnt_limited;
        integ.enable_limits = opts.enable_limits ? 1 : 0;
        integ.qpos_f = data.qpos;
        integ.qvel_f = data.qvel;
        integ.sim_time = data.time;
        integ.qpos_w = data.qpos_d;
        integ.qvel_w = data.qvel_d;
        integ.free_joint_damp = (double)opts.free_joint_damping;
    }

    // Specialised instantiations for the dof counts the LIBERO suites use
    // (goal 37, spatial 43, object 51, libero_10 57); anything else falls
    // through to the generic runtime-nv instantiation.
    // Not built when KERNEL_MAXNV_VALUE > 64 (RoboCasa-GR1, nv 64-87): sH and
    // the other KERNEL_MAXNV-sized LDS arrays plus the NV_T<=48 Hessian caches
    // exceed gfx950's 160 KB, and no scene that build runs has these nv.
    switch (nv) {
#if KERNEL_MAXNV_VALUE <= 64
        case 20: newton_launch_impl<20>(model, data, opts, stream, warm_start, ne, nv, integ); break;   // ManiSkill PutOnPlate
        case 37: newton_launch_impl<37>(model, data, opts, stream, warm_start, ne, nv, integ); break;
        case 43: newton_launch_impl<43>(model, data, opts, stream, warm_start, ne, nv, integ); break;
        case 51: newton_launch_impl<51>(model, data, opts, stream, warm_start, ne, nv, integ); break;
        case 57: newton_launch_impl<57>(model, data, opts, stream, warm_start, ne, nv, integ); break;
#endif
        default: newton_launch_impl<0>(model, data, opts, stream, warm_start, ne, nv, integ); break;
    }
    return integ.on != 0;
}

template <int NV_T>
static void mhat_launch_impl(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream,
    int ne, int nv)
{
    if constexpr (NV_T > 0 && NV_T <= 64) {
        kernel_mhat_solve_reg<NV_T><<<ne, MHAT_BLK, 0, stream>>>(
            ne,
            data.solver_M,
            model.jnt_type, model.jnt_dof_adr,
            model.jnt_damping_d, model.njnt, (double)model.timestep, opts.enable_damping,
            data.qacc);
        return;
    }
    kernel_mhat_solve_t<NV_T><<<ne, MHAT_BLK, 0, stream>>>(
        ne, nv,
        data.solver_M,
        model.jnt_type, model.jnt_dof_adr,
        model.jnt_damping_d, model.njnt, (double)model.timestep, opts.enable_damping,
        data.qacc);
}

void launch_mhat_solve(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    int ne = data.num_envs;
    int nv = model.nv;
    if (ne <= 0 || nv <= 0 || nv > KERNEL_MAXNV) return;
    if (g_mhat_fuse.pending && g_mhat_fuse.qacc == data.qacc && g_mhat_fuse.stream == stream) {
        g_mhat_fuse.pending = false;   // already solved in the Newton tail
        return;
    }
#if KERNEL_MAXNV_VALUE > 64
    if (nv > MHAT_BLK) {   // one lane per dof does not fit a wave
        kernel_mhat_solve_wide<<<ne, WIDE_BLK, 0, stream>>>(
            ne, nv, data.solver_M, model.jnt_type, model.jnt_dof_adr, model.jnt_damping_d,
            model.njnt, (double)model.timestep, opts.enable_damping, data.qacc);
        return;
    }
#endif
    // Same specialisation ladder as launch_newton_solve: the LIBERO suites use
    // 37/43/51/57 dofs; anything else takes the generic runtime-nv build.
    switch (nv) {
        case 20: mhat_launch_impl<20>(model, data, opts, stream, ne, nv); break;
        case 37: mhat_launch_impl<37>(model, data, opts, stream, ne, nv); break;
        case 43: mhat_launch_impl<43>(model, data, opts, stream, ne, nv); break;
        case 51: mhat_launch_impl<51>(model, data, opts, stream, ne, nv); break;
        case 57: mhat_launch_impl<57>(model, data, opts, stream, ne, nv); break;
        default: mhat_launch_impl<0>(model, data, opts, stream, ne, nv); break;
    }
}

void launch_pgs_solve(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0 || opts.iterations <= 0) return;

    // Phase 7: parallel M_inv_diag build
    {
        const int blk = 256;
        int grd_x = (model.njnt + blk - 1) / blk;
        dim3 grid(grd_x, ne);

        // First zero-init the M_inv_diag with default 1.0
        hipMemsetAsync(data.solver_M_inv_diag, 0, (size_t)ne * MAX_DOF * sizeof(float), stream);

        kernel_build_M_inv_diag<<<grid, blk, 0, stream>>>(
            ne, model.nv, model.njnt, model.nbody,
            model.jnt_type, model.jnt_body, model.jnt_dof_adr,
            data.body_inv_mass, data.body_inv_inertia,
            data.solver_M_inv_diag);
    }

    // Set default M_inv_diag to 1.0 for DOFs not covered by joints
    // (handled by the memset + kernel above)

    // Precompute A = J * diag(M_inv) * J^T using pre-allocated buffer
    // Reuse solver_H as A buffer since PGS doesn't need H
    float* d_A = reinterpret_cast<float*>(data.solver_H);
    size_t A_size = (size_t)ne * MAX_A_DIM * MAX_A_DIM * sizeof(float);
    hipMemsetAsync(d_A, 0, A_size, stream);
    {
        int max_entries = MAX_A_DIM * MAX_A_DIM;
        const int blk = 256;
        int grd_x = (max_entries + blk - 1) / blk;
        dim3 grid(grd_x, ne);
        kernel_compute_A_matrix<<<grid, blk, 0, stream>>>(
            ne, model.nv,
            data.constraint_J, data.nconstraint,
            data.solver_M_inv_diag, d_A, opts.contact_slip);
    }

    // PGS iterations
    {
        const int blk = 64;
        int grd = (ne + blk - 1) / blk;
        kernel_pgs_iteration_fast<<<grd, blk, 0, stream>>>(
            ne, opts.iterations,
            d_A,
            data.constraint_rhs,
            data.constraint_lo, data.constraint_hi,
            data.constraint_lam,
            data.constraint_friction_parent, data.constraint_friction_mu,
            data.nconstraint);
    }
}

void launch_build_M_inv_diag_standalone(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0) return;

    const int blk = 256;
    int grd_x = (model.njnt + blk - 1) / blk;
    dim3 grid(grd_x, ne);

    hipMemsetAsync(data.solver_M_inv_diag, 0, (size_t)ne * MAX_DOF * sizeof(float), stream);

    kernel_build_M_inv_diag<<<grid, blk, 0, stream>>>(
        ne, model.nv, model.njnt, model.nbody,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr,
        data.body_inv_mass, data.body_inv_inertia,
        data.solver_M_inv_diag);
}

void launch_extract_M_diag_inv(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    hipStream_t stream)
{
    int ne = data.num_envs;
    int nv = model.nv;
    if (ne <= 0 || nv <= 0) return;
    // Already written by kernel_build_mass_matrix (fused) whenever it ran; every
    // caller launches build_mass immediately upstream with no solver_M writer in
    // between. Only the nv > KERNEL_MAXNV case (build_mass skipped) runs here.
    if (nv <= KERNEL_MAXNV) return;

    const int blk = 256;
    int grd_x = (nv + blk - 1) / blk;
    dim3 grid(grd_x, ne);

    kernel_extract_M_diag_inv<<<grid, blk, 0, stream>>>(
        ne, nv, data.solver_M, data.solver_M_inv_diag);
}

// Gravity compensation (MJCF body gravcomp). MuJoCo adds the passive force
// J_com^T * (gravcomp * m * -g) for every such body; qfrc_bias holds the
// gravity-inclusive bias with the opposite sign, so subtract it from there.
// Runs after the bias is written, so every RNEA variant stays untouched. One
// thread per world: each owns its qfrc_bias slice, no atomics.
__global__ void kernel_gravcomp(
    int num_envs, int nv, int nbody, int njnt, int n_gravcomp,
    const int* __restrict__ gravcomp_body, const double* __restrict__ gravcomp_scale,
    const double* __restrict__ body_mass_d, const double* __restrict__ body_ipos_d,
    double gx, double gy, double gz,
    const int* body_chain_start, const int* body_chain_len,
    const int* body_chain_jnt_idx, const int* body_chain_jnt_body,
    const int* body_chain_parent,
    const int* jnt_type, const int* jnt_dof_adr, const double* jnt_axis_d,
    const double* xpos_d, const double* xquat_d, const double* jnt_xaxis_d,
    double* qfrc_bias)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;
    const double gn = sqrt(gx * gx + gy * gy + gz * gz);
    if (gn == 0.0) return;
    // Direction of the compensating force (against gravity).
    const double up[3] = {-gx / gn, -gy / gn, -gz / gn};
    double* bias = qfrc_bias + (size_t)env * nv;
    for (int k = 0; k < n_gravcomp; k++) {
        const int b = gravcomp_body[k];
        const double f = gravcomp_scale[k] * body_mass_d[b] * gn;
        // World COM: xpos + R(xquat) * ipos
        const double* q = xquat_d + ((size_t)env * nbody + b) * 4;
        const double* ip = body_ipos_d + b * 3;
        const double w = q[0], x = q[1], y = q[2], z = q[3];
        const double tx = 2 * (y * ip[2] - z * ip[1]);
        const double ty = 2 * (z * ip[0] - x * ip[2]);
        const double tz = 2 * (x * ip[1] - y * ip[0]);
        const double* xp = xpos_d + ((size_t)env * nbody + b) * 3;
        const double com[3] = {xp[0] + ip[0] + w * tx + (y * tz - z * ty),
                               xp[1] + ip[1] + w * ty + (z * tx - x * tz),
                               xp[2] + ip[2] + w * tz + (x * ty - y * tx)};
        fill_chain_jacobian_precomputed_d(bias, b, -f, up, com,
            body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
            body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d,
            xpos_d, xquat_d, env, nbody, jnt_xaxis_d, njnt);
    }
}

constexpr int GC_BLK = 64;   // == PYR_BLK (gravcomp_lds_block assumes a 64-lane block)
__global__ void __launch_bounds__(GC_BLK) kernel_gravcomp_lds(
    int num_envs, int nv, int nbody, int njnt, int n_gravcomp, int chain_total,
    const int* __restrict__ gravcomp_body, const double* __restrict__ gravcomp_scale,
    const double* __restrict__ body_mass_d, const double* __restrict__ body_ipos_d,
    double gx, double gy, double gz,
    const int* __restrict__ body_chain_start, const int* __restrict__ body_chain_len,
    const int* __restrict__ body_chain_jnt_idx, const int* __restrict__ body_chain_jnt_body,
    const int* __restrict__ body_chain_parent,
    const int* __restrict__ jnt_type, const int* __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis_d,
    const double* __restrict__ xpos_d, const double* __restrict__ xquat_d,
    const double* __restrict__ jnt_xaxis_d,
    double* __restrict__ qfrc_bias)
{
    gravcomp_lds_block((int)blockIdx.x, num_envs, nv, nbody, njnt, n_gravcomp, chain_total,
        gravcomp_body, gravcomp_scale, body_mass_d, body_ipos_d, gx, gy, gz,
        body_chain_start, body_chain_len, body_chain_jnt_idx, body_chain_jnt_body,
        body_chain_parent, jnt_type, jnt_dof_adr, jnt_axis_d, xpos_d, xquat_d,
        jnt_xaxis_d, qfrc_bias);
}

void launch_gravcomp(const DevicePhysicsModel& model, DevicePhysicsData& data,
                     hipStream_t stream)
{
    if (model.n_gravcomp <= 0 || data.num_envs <= 0 || model.nv <= 0) return;
    if (data.jnt_xaxis_d && model.n_gravcomp <= GC_BLK && model.nv <= KERNEL_MAXNV
        && model.nbody <= CS_LDS_NB && model.njnt <= CS_LDS_NJ
        && model.body_chain_total >= 0 && model.body_chain_total <= CS_LDS_CH) {
        kernel_gravcomp_lds<<<data.num_envs, GC_BLK, 0, stream>>>(
            data.num_envs, model.nv, model.nbody, model.njnt, model.n_gravcomp, model.body_chain_total,
            model.gravcomp_body, model.gravcomp_scale_d, model.body_mass_d, model.body_ipos_d,
            model.gravity_d[0], model.gravity_d[1], model.gravity_d[2],
            model.body_chain_start, model.body_chain_len, model.body_chain_jnt_idx,
            model.body_chain_jnt_body, model.body_chain_parent,
            model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
            data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.qfrc_bias);
        return;
    }
    const int blk = 64;
    kernel_gravcomp<<<(data.num_envs + blk - 1) / blk, blk, 0, stream>>>(
        data.num_envs, model.nv, model.nbody, model.njnt, model.n_gravcomp,
        model.gravcomp_body, model.gravcomp_scale_d, model.body_mass_d, model.body_ipos_d,
        model.gravity_d[0], model.gravity_d[1], model.gravity_d[2],
        model.body_chain_start, model.body_chain_len, model.body_chain_jnt_idx,
        model.body_chain_jnt_body, model.body_chain_parent,
        model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
        data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.qfrc_bias);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
