// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Integration kernels (v2: parallel RNEA + Newton-compatible)
//
// Changes from v1:
//   - RNEA split into level-by-level parallel kernels (forward + backward)
//   - Stack arrays replaced with pre-allocated device buffers
//   - Mass matrix solve removed (handled by Newton solver in physics_solver.hip)
//   - kernel_apply_forces updated for Newton path (just writes forces, no M-solve)
//   - kernel_mass_matrix_solve kept as fallback for PGS path

#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/world_mask.h"
#include "hipgl/physics/physics_integrate_dev.h"

namespace hipgl {
namespace physics {
namespace kernels {

// world skip: per-world freeze mask (see world_mask.h).
HIPGL_WORLD_MASK_DEFINE(set_world_mask_integrate)

// ── Quaternion helpers ──────────────────────────────────────────────────────

HIPGL_DEVICE inline void quat_integrate(float* q, const float* omega, float dt) {
    float wx = omega[0] * dt * 0.5f;
    float wy = omega[1] * dt * 0.5f;
    float wz = omega[2] * dt * 0.5f;
    float dq[4] = {
        -q[1]*wx - q[2]*wy - q[3]*wz,
         q[0]*wx + q[2]*wz - q[3]*wy,
         q[0]*wy - q[1]*wz + q[3]*wx,
         q[0]*wz + q[1]*wy - q[2]*wx
    };
    q[0] += dq[0]; q[1] += dq[1]; q[2] += dq[2]; q[3] += dq[3];
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-12f) {
        float inv = 1.0f / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    } else {
        q[0] = 1; q[1] = q[2] = q[3] = 0;
    }
}

HIPGL_DEVICE inline void quat_integrate_d(double* q, const double* omega, double dt) {
    double wx = omega[0] * dt * 0.5;
    double wy = omega[1] * dt * 0.5;
    double wz = omega[2] * dt * 0.5;
    double dq[4] = {
        -q[1]*wx - q[2]*wy - q[3]*wz,
         q[0]*wx + q[2]*wz - q[3]*wy,
         q[0]*wy - q[1]*wz + q[3]*wx,
         q[0]*wz + q[1]*wy - q[2]*wx
    };
    q[0] += dq[0]; q[1] += dq[1]; q[2] += dq[2]; q[3] += dq[3];
    double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n > 1e-15) {
        double inv = 1.0 / n;
        q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
    } else {
        q[0] = 1; q[1] = q[2] = q[3] = 0;
    }
}

// ── Compute inverse mass diagonal (one-time or per-reset) ──────────────────

HIPGL_GLOBAL void kernel_compute_inv_mass(
    int nbody, int num_envs,
    const float* __restrict__ body_mass,
    const float* __restrict__ body_inertia,
    float*       __restrict__ body_inv_mass,
    float*       __restrict__ body_inv_inertia)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (bid >= nbody || env >= num_envs) return;

    int idx = env * nbody + bid;
    float m = body_mass[bid];
    body_inv_mass[idx] = (m > 1e-10f) ? (1.0f / m) : 0.0f;

    int idx3 = env * nbody * 3 + bid * 3;
    for (int k = 0; k < 3; k++) {
        float I = body_inertia[bid*3 + k];
        body_inv_inertia[idx3 + k] = (I > 1e-10f) ? (1.0f / I) : 0.0f;
    }
}


HIPGL_GLOBAL void kernel_rnea_forward(
    int level_start, int level_count,
    int num_envs, int nv, int njnt, int nbody,
    const int*   __restrict__ level_body_ids,
    const int*   __restrict__ body_parent,
    const int*   __restrict__ body_jnt_adr,
    const int*   __restrict__ body_jnt_num,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const float* __restrict__ jnt_axis,
    float gx, float gy, float gz,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    const float* __restrict__ qvel,
    float*       __restrict__ rnea_omega,
    float*       __restrict__ rnea_alpha,
    float*       __restrict__ rnea_acc)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (local_idx >= level_count || env >= num_envs) return;

    int bid = level_body_ids[level_start + local_idx];
    int p = body_parent[bid];

    float op_x, op_y, op_z, ap_x, ap_y, ap_z, aop_x, aop_y, aop_z;
    if (p <= 0) {
        op_x = op_y = op_z = 0;
        ap_x = ap_y = ap_z = 0;
        aop_x = -gx; aop_y = -gy; aop_z = -gz;
    } else {
        int p3 = env * nbody * 3 + p * 3;
        op_x = rnea_omega[p3]; op_y = rnea_omega[p3+1]; op_z = rnea_omega[p3+2];
        ap_x = rnea_alpha[p3]; ap_y = rnea_alpha[p3+1]; ap_z = rnea_alpha[p3+2];
        aop_x = rnea_acc[p3]; aop_y = rnea_acc[p3+1]; aop_z = rnea_acc[p3+2];
    }

    int b3 = env * nbody * 3 + bid * 3;
    int p3_pos = env * nbody * 3 + p * 3;
    float rx = xpos[b3] - xpos[p3_pos];
    float ry = xpos[b3+1] - xpos[p3_pos+1];
    float rz = xpos[b3+2] - xpos[p3_pos+2];

    float aXr_x = ap_y*rz - ap_z*ry, aXr_y = ap_z*rx - ap_x*rz, aXr_z = ap_x*ry - ap_y*rx;
    float oXr_x = op_y*rz - op_z*ry, oXr_y = op_z*rx - op_x*rz, oXr_z = op_x*ry - op_y*rx;
    float oXoXr_x = op_y*oXr_z - op_z*oXr_y;
    float oXoXr_y = op_z*oXr_x - op_x*oXr_z;
    float oXoXr_z = op_x*oXr_y - op_y*oXr_x;

    float ai_x = aop_x + aXr_x + oXoXr_x;
    float ai_y = aop_y + aXr_y + oXoXr_y;
    float ai_z = aop_z + aXr_z + oXoXr_z;

    float oi_x = op_x, oi_y = op_y, oi_z = op_z;
    float ali_x = ap_x, ali_y = ap_y, ali_z = ap_z;

    const float* ev = qvel + env * nv;
    int ja = body_jnt_adr[bid];
    int jn = body_jnt_num[bid];

    for (int k = 0; k < jn; k++) {
        int ji = ja + k;
        int jtype = jnt_type[ji];
        int dof = jnt_dof_adr[ji];

        if (jtype == JOINT_HINGE) {
            // Use body's own world quat (axis is invariant under own rotation)
            int bq4 = env * nbody * 4 + bid * 4;
            float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
            float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
            float tw_x = 2.0f*(qy*laz-qz*lay), tw_y = 2.0f*(qz*lax-qx*laz), tw_z = 2.0f*(qx*lay-qy*lax);
            float aw_x = lax+qw*tw_x+(qy*tw_z-qz*tw_y);
            float aw_y = lay+qw*tw_y+(qz*tw_x-qx*tw_z);
            float aw_z = laz+qw*tw_z+(qx*tw_y-qy*tw_x);
            float qdot = ev[dof];
            float sv_x = aw_x*qdot, sv_y = aw_y*qdot, sv_z = aw_z*qdot;
            ali_x += op_y*sv_z - op_z*sv_y;
            ali_y += op_z*sv_x - op_x*sv_z;
            ali_z += op_x*sv_y - op_y*sv_x;
            oi_x += sv_x; oi_y += sv_y; oi_z += sv_z;
        }
        else if (jtype == JOINT_FREE) {
            // Body-frame angular velocity must be rotated to world frame
            float wb_x = ev[dof+3], wb_y = ev[dof+4], wb_z = ev[dof+5];
            int bq4 = env * nbody * 4 + bid * 4;
            float fqw = xquat[bq4], fqx = xquat[bq4+1], fqy = xquat[bq4+2], fqz = xquat[bq4+3];
            // R * wb (rotate body-frame to world-frame)
            float R00=1-2*(fqy*fqy+fqz*fqz), R01=2*(fqx*fqy-fqw*fqz), R02=2*(fqx*fqz+fqw*fqy);
            float R10=2*(fqx*fqy+fqw*fqz), R11=1-2*(fqx*fqx+fqz*fqz), R12=2*(fqy*fqz-fqw*fqx);
            float R20=2*(fqx*fqz-fqw*fqy), R21=2*(fqy*fqz+fqw*fqx), R22=1-2*(fqx*fqx+fqy*fqy);
            float wx = R00*wb_x + R01*wb_y + R02*wb_z;
            float wy = R10*wb_x + R11*wb_y + R12*wb_z;
            float wz = R20*wb_x + R21*wb_y + R22*wb_z;
            ali_x += op_y*wz - op_z*wy;
            ali_y += op_z*wx - op_x*wz;
            ali_z += op_x*wy - op_y*wx;
            oi_x += wx; oi_y += wy; oi_z += wz;
        }
        else if (jtype == JOINT_SLIDE) {
            int bq4 = env * nbody * 4 + bid * 4;
            float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
            float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
            float tw_x = 2.0f*(qy*laz-qz*lay), tw_y = 2.0f*(qz*lax-qx*laz), tw_z = 2.0f*(qx*lay-qy*lax);
            float aw_x = lax+qw*tw_x+(qy*tw_z-qz*tw_y);
            float aw_y = lay+qw*tw_y+(qz*tw_x-qx*tw_z);
            float aw_z = laz+qw*tw_z+(qx*tw_y-qy*tw_x);
            float qdot = ev[dof];
            // Classical body-origin acceleration includes 2 * omega_parent x slide_velocity.
            ai_x += 2.0f * (op_y*(aw_z*qdot) - op_z*(aw_y*qdot));
            ai_y += 2.0f * (op_z*(aw_x*qdot) - op_x*(aw_z*qdot));
            ai_z += 2.0f * (op_x*(aw_y*qdot) - op_y*(aw_x*qdot));
        }
    }

    int o3 = env * nbody * 3 + bid * 3;
    rnea_omega[o3] = oi_x; rnea_omega[o3+1] = oi_y; rnea_omega[o3+2] = oi_z;
    rnea_alpha[o3] = ali_x; rnea_alpha[o3+1] = ali_y; rnea_alpha[o3+2] = ali_z;
    rnea_acc[o3] = ai_x; rnea_acc[o3+1] = ai_y; rnea_acc[o3+2] = ai_z;
}

// ── Phase 2: Parallel RNEA body forces (1 thread per body per env) ──────────
// Computes net force and torque on each body. Fully parallel (no tree deps).

HIPGL_GLOBAL void kernel_rnea_body_forces(
    int num_envs, int nbody,
    const float* __restrict__ body_mass,
    const float* __restrict__ body_inertia,
    const float* __restrict__ body_ipos,
    const float* __restrict__ body_iquat,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    const float* __restrict__ rnea_omega,
    const float* __restrict__ rnea_alpha,
    const float* __restrict__ rnea_acc,
    float*       __restrict__ rnea_fnet,
    float*       __restrict__ rnea_tnet)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (bid < 1 || bid >= nbody || env >= num_envs) return;

    float m = body_mass[bid];
    float Ix = body_inertia[bid*3], Iy = body_inertia[bid*3+1], Iz = body_inertia[bid*3+2];

    float ipx = body_ipos[bid*3], ipy = body_ipos[bid*3+1], ipz = body_ipos[bid*3+2];
    int bq4 = env * nbody * 4 + bid * 4;
    float bqw = xquat[bq4], bqx = xquat[bq4+1], bqy = xquat[bq4+2], bqz = xquat[bq4+3];
    float tw_x = 2.0f*(bqy*ipz-bqz*ipy), tw_y = 2.0f*(bqz*ipx-bqx*ipz), tw_z = 2.0f*(bqx*ipy-bqy*ipx);
    float rc_x = ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
    float rc_y = ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
    float rc_z = ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);

    int o3 = env * nbody * 3 + bid * 3;
    float oi_x = rnea_omega[o3], oi_y = rnea_omega[o3+1], oi_z = rnea_omega[o3+2];
    float ali_x = rnea_alpha[o3], ali_y = rnea_alpha[o3+1], ali_z = rnea_alpha[o3+2];
    float ao_x = rnea_acc[o3], ao_y = rnea_acc[o3+1], ao_z = rnea_acc[o3+2];

    float aXrc_x = ali_y*rc_z-ali_z*rc_y, aXrc_y = ali_z*rc_x-ali_x*rc_z, aXrc_z = ali_x*rc_y-ali_y*rc_x;
    float oXrc_x = oi_y*rc_z-oi_z*rc_y, oXrc_y = oi_z*rc_x-oi_x*rc_z, oXrc_z = oi_x*rc_y-oi_y*rc_x;
    float oXoXrc_x = oi_y*oXrc_z-oi_z*oXrc_y;
    float oXoXrc_y = oi_z*oXrc_x-oi_x*oXrc_z;
    float oXoXrc_z = oi_x*oXrc_y-oi_y*oXrc_x;
    float ac_x = ao_x+aXrc_x+oXoXrc_x, ac_y = ao_y+aXrc_y+oXoXrc_y, ac_z = ao_z+aXrc_z+oXoXrc_z;

    rnea_fnet[o3] = m*ac_x; rnea_fnet[o3+1] = m*ac_y; rnea_fnet[o3+2] = m*ac_z;

    // Compose body_quat * body_iquat for correct inertial frame rotation
    float iqw = body_iquat[bid*4], iqx = body_iquat[bid*4+1], iqy = body_iquat[bid*4+2], iqz = body_iquat[bid*4+3];
    float cqw = bqw*iqw - bqx*iqx - bqy*iqy - bqz*iqz;
    float cqx = bqw*iqx + bqx*iqw + bqy*iqz - bqz*iqy;
    float cqy = bqw*iqy - bqx*iqz + bqy*iqw + bqz*iqx;
    float cqz = bqw*iqz + bqx*iqy - bqy*iqx + bqz*iqw;

    float xx=cqx*cqx, yy=cqy*cqy, zz=cqz*cqz;
    float xy=cqx*cqy, xz=cqx*cqz, yz=cqy*cqz;
    float wx=cqw*cqx, wy=cqw*cqy, wz=cqw*cqz;
    float R00=1-2*(yy+zz), R01=2*(xy-wz), R02=2*(xz+wy);
    float R10=2*(xy+wz), R11=1-2*(xx+zz), R12=2*(yz-wx);
    float R20=2*(xz-wy), R21=2*(yz+wx), R22=1-2*(xx+yy);

    float Io_x = (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*oi_x + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*oi_y + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*oi_z;
    float Io_y = (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*oi_x + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*oi_y + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*oi_z;
    float Io_z = (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*oi_x + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*oi_y + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*oi_z;

    float Ia_x = (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*ali_x + (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*ali_y + (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*ali_z;
    float Ia_y = (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*ali_x + (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*ali_y + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*ali_z;
    float Ia_z = (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*ali_x + (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*ali_y + (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*ali_z;

    float oXIo_x = oi_y*Io_z-oi_z*Io_y, oXIo_y = oi_z*Io_x-oi_x*Io_z, oXIo_z = oi_x*Io_y-oi_y*Io_x;
    float rcXf_x = rc_y*rnea_fnet[o3+2]-rc_z*rnea_fnet[o3+1];
    float rcXf_y = rc_z*rnea_fnet[o3]-rc_x*rnea_fnet[o3+2];
    float rcXf_z = rc_x*rnea_fnet[o3+1]-rc_y*rnea_fnet[o3];

    rnea_tnet[o3]   = Ia_x+oXIo_x+rcXf_x;
    rnea_tnet[o3+1] = Ia_y+oXIo_y+rcXf_y;
    rnea_tnet[o3+2] = Ia_z+oXIo_z+rcXf_z;
}

// ── Phase 2: Parallel RNEA backward pass (1 level at a time, leaf to root) ──
// Accumulates forces from children to parent and projects onto joint axes.

HIPGL_GLOBAL void kernel_rnea_backward(
    int level_start, int level_count,
    int num_envs, int nv, int njnt, int nbody,
    const int*   __restrict__ level_body_ids,
    const int*   __restrict__ body_parent,
    const int*   __restrict__ body_jnt_adr,
    const int*   __restrict__ body_jnt_num,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_dof_adr,
    const float* __restrict__ jnt_axis,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    float*       __restrict__ rnea_fnet,
    float*       __restrict__ rnea_tnet,
    double*      __restrict__ qfrc_bias)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (local_idx >= level_count || env >= num_envs) return;

    int bid = level_body_ids[level_start + local_idx];
    if (bid < 1) return;

    int p = body_parent[bid];
    int o3 = env * nbody * 3 + bid * 3;
    float fx = rnea_fnet[o3], fy = rnea_fnet[o3+1], fz = rnea_fnet[o3+2];
    float tx = rnea_tnet[o3], ty = rnea_tnet[o3+1], tz = rnea_tnet[o3+2];

    int ja = body_jnt_adr[bid];
    int jn = body_jnt_num[bid];
    for (int k = 0; k < jn; k++) {
        int ji = ja + k;
        int jtype = jnt_type[ji];
        int dof = jnt_dof_adr[ji];

        if (jtype == JOINT_HINGE) {
            int bq4 = env * nbody * 4 + bid * 4;
            float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
            float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
            float tw_x = 2.0f*(qy*laz-qz*lay), tw_y = 2.0f*(qz*lax-qx*laz), tw_z = 2.0f*(qx*lay-qy*lax);
            float aw_x = lax+qw*tw_x+(qy*tw_z-qz*tw_y);
            float aw_y = lay+qw*tw_y+(qz*tw_x-qx*tw_z);
            float aw_z = laz+qw*tw_z+(qx*tw_y-qy*tw_x);
            if (dof < nv) atomicAdd(&qfrc_bias[env*nv + dof], (double)(aw_x*tx + aw_y*ty + aw_z*tz));
        }
        else if (jtype == JOINT_FREE) {
            if (dof+5 < nv) {
                atomicAdd(&qfrc_bias[env*nv + dof],   (double)fx);
                atomicAdd(&qfrc_bias[env*nv + dof+1], (double)fy);
                atomicAdd(&qfrc_bias[env*nv + dof+2], (double)fz);
                // Project world-frame torque onto body-frame axes via R^T
                int bq4 = env * nbody * 4 + bid * 4;
                float fqw = xquat[bq4], fqx = xquat[bq4+1], fqy = xquat[bq4+2], fqz = xquat[bq4+3];
                // R^T * t (transpose rows become columns)
                float R00=1-2*(fqy*fqy+fqz*fqz), R10=2*(fqx*fqy+fqw*fqz), R20=2*(fqx*fqz-fqw*fqy);
                float R01=2*(fqx*fqy-fqw*fqz), R11=1-2*(fqx*fqx+fqz*fqz), R21=2*(fqy*fqz+fqw*fqx);
                float R02=2*(fqx*fqz+fqw*fqy), R12=2*(fqy*fqz-fqw*fqx), R22=1-2*(fqx*fqx+fqy*fqy);
                float tb_x = R00*tx + R10*ty + R20*tz;
                float tb_y = R01*tx + R11*ty + R21*tz;
                float tb_z = R02*tx + R12*ty + R22*tz;
                atomicAdd(&qfrc_bias[env*nv + dof+3], (double)tb_x);
                atomicAdd(&qfrc_bias[env*nv + dof+4], (double)tb_y);
                atomicAdd(&qfrc_bias[env*nv + dof+5], (double)tb_z);
            }
        }
        else if (jtype == JOINT_SLIDE) {
            int bq4 = env * nbody * 4 + bid * 4;
            float qw = xquat[bq4], qx = xquat[bq4+1], qy = xquat[bq4+2], qz = xquat[bq4+3];
            float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
            float tw_x = 2.0f*(qy*laz-qz*lay), tw_y = 2.0f*(qz*lax-qx*laz), tw_z = 2.0f*(qx*lay-qy*lax);
            float aw_x = lax+qw*tw_x+(qy*tw_z-qz*tw_y);
            float aw_y = lay+qw*tw_y+(qz*tw_x-qx*tw_z);
            float aw_z = laz+qw*tw_z+(qx*tw_y-qy*tw_x);
            if (dof < nv) atomicAdd(&qfrc_bias[env*nv + dof], (double)(aw_x*fx + aw_y*fy + aw_z*fz));
        }
    }

    // Accumulate to parent
    if (p > 0 && p < nbody) {
        int b3 = env * nbody * 3 + bid * 3;
        int p3 = env * nbody * 3 + p * 3;
        float rx = xpos[b3]-xpos[p3], ry = xpos[b3+1]-xpos[p3+1], rz = xpos[b3+2]-xpos[p3+2];
        atomicAdd(&rnea_fnet[p3],   fx);
        atomicAdd(&rnea_fnet[p3+1], fy);
        atomicAdd(&rnea_fnet[p3+2], fz);
        atomicAdd(&rnea_tnet[p3],   tx + (ry*fz - rz*fy));
        atomicAdd(&rnea_tnet[p3+1], ty + (rz*fx - rx*fz));
        atomicAdd(&rnea_tnet[p3+2], tz + (rx*fy - ry*fx));
    }
}

// ── Fused RNEA bias force kernel (all levels in one launch) ─────────────────
// 1 block per env. Threads process bodies in parallel within each level,
// with __syncthreads() barriers between levels. Avoids ~2*tree_depth kernel
// launches per step.

HIPGL_GLOBAL void kernel_compute_bias_force_fused(
    int num_envs, int tree_depth, int nv, int njnt, int nbody,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis,
    double gx, double gy, double gz,
    const double* __restrict__ body_mass,
    const double* __restrict__ body_inertia,
    const double* __restrict__ body_ipos,
    const double* __restrict__ body_iquat,
    const double* __restrict__ body_quat,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const double* __restrict__ qvel,
    const double* __restrict__ jnt_xaxis_d,
    double*      __restrict__ rnea_omega,
    double*      __restrict__ rnea_alpha,
    double*      __restrict__ rnea_acc,
    double*      __restrict__ rnea_fnet,
    double*      __restrict__ rnea_tnet,
    double*      __restrict__ qfrc_bias)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;

    for (int i = threadIdx.x; i < nbody * 3; i += blockDim.x) {
        int idx = env * nbody * 3 + i;
        rnea_omega[idx] = 0; rnea_alpha[idx] = 0; rnea_acc[idx] = 0;
        rnea_fnet[idx] = 0; rnea_tnet[idx] = 0;
    }
    for (int i = threadIdx.x; i < nv; i += blockDim.x)
        qfrc_bias[env * nv + i] = 0.0;
    __syncthreads();

    // Forward pass: root to leaf (double precision throughout)
    for (int level = 1; level <= tree_depth; level++) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;
        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int bid = level_body_ids[ls + local_idx];
            int p = body_parent[bid];
            double op_x, op_y, op_z, ap_x, ap_y, ap_z, aop_x, aop_y, aop_z;
            if (p <= 0) {
                op_x = op_y = op_z = 0; ap_x = ap_y = ap_z = 0;
                aop_x = -(double)gx; aop_y = -(double)gy; aop_z = -(double)gz;
            } else {
                int p3 = env * nbody * 3 + p * 3;
                op_x = rnea_omega[p3]; op_y = rnea_omega[p3+1]; op_z = rnea_omega[p3+2];
                ap_x = rnea_alpha[p3]; ap_y = rnea_alpha[p3+1]; ap_z = rnea_alpha[p3+2];
                aop_x = rnea_acc[p3]; aop_y = rnea_acc[p3+1]; aop_z = rnea_acc[p3+2];
            }
            int b3 = env * nbody * 3 + bid * 3;
            int p3_pos = env * nbody * 3 + p * 3;
            double rx = (double)xpos[b3]-(double)xpos[p3_pos], ry = (double)xpos[b3+1]-(double)xpos[p3_pos+1], rz = (double)xpos[b3+2]-(double)xpos[p3_pos+2];
            double aXr_x = ap_y*rz-ap_z*ry, aXr_y = ap_z*rx-ap_x*rz, aXr_z = ap_x*ry-ap_y*rx;
            double oXr_x = op_y*rz-op_z*ry, oXr_y = op_z*rx-op_x*rz, oXr_z = op_x*ry-op_y*rx;
            double oXoXr_x = op_y*oXr_z-op_z*oXr_y, oXoXr_y = op_z*oXr_x-op_x*oXr_z, oXoXr_z = op_x*oXr_y-op_y*oXr_x;
            double ai_x = aop_x+aXr_x+oXoXr_x, ai_y = aop_y+aXr_y+oXoXr_y, ai_z = aop_z+aXr_z+oXoXr_z;
            double oi_x = op_x, oi_y = op_y, oi_z = op_z;
            double ali_x = ap_x, ali_y = ap_y, ali_z = ap_z;

            const double* ev = qvel + env * nv;
            int ja = body_jnt_adr[bid]; int jn = body_jnt_num[bid];
            for (int k = 0; k < jn; k++) {
                int ji = ja + k; int jtype_l = jnt_type[ji]; int dof = jnt_dof_adr[ji];
                if (jtype_l == JOINT_HINGE) {
                    // Use pre-computed pre-joint world-frame axis from FK
                    int ax_idx = env*njnt*3 + ji*3;
                    double aw_x=jnt_xaxis_d[ax_idx],aw_y=jnt_xaxis_d[ax_idx+1],aw_z=jnt_xaxis_d[ax_idx+2];
                    double qdot = ev[dof];
                    double sv_x=aw_x*qdot,sv_y=aw_y*qdot,sv_z=aw_z*qdot;
                    // Coriolis: use accumulated ω (oi) not just parent ω (op)
                    ali_x+=oi_y*sv_z-oi_z*sv_y; ali_y+=oi_z*sv_x-oi_x*sv_z; ali_z+=oi_x*sv_y-oi_y*sv_x;
                    oi_x+=sv_x; oi_y+=sv_y; oi_z+=sv_z;
                } else if (jtype_l == JOINT_FREE) {
                    // Body-frame angular velocity -> world frame
                    double wb_x=ev[dof+3],wb_y=ev[dof+4],wb_z=ev[dof+5];
                    int bq4f=env*nbody*4+bid*4;
                    double fqw=xquat[bq4f],fqx=xquat[bq4f+1],fqy=xquat[bq4f+2],fqz=xquat[bq4f+3];
                    double R00=1-2*(fqy*fqy+fqz*fqz),R01=2*(fqx*fqy-fqw*fqz),R02=2*(fqx*fqz+fqw*fqy);
                    double R10=2*(fqx*fqy+fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R12=2*(fqy*fqz-fqw*fqx);
                    double R20=2*(fqx*fqz-fqw*fqy),R21=2*(fqy*fqz+fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                    double wx=R00*wb_x+R01*wb_y+R02*wb_z;
                    double wy=R10*wb_x+R11*wb_y+R12*wb_z;
                    double wz=R20*wb_x+R21*wb_y+R22*wb_z;
                    ali_x+=op_y*wz-op_z*wy; ali_y+=op_z*wx-op_x*wz; ali_z+=op_x*wy-op_y*wx;
                    oi_x+=wx; oi_y+=wy; oi_z+=wz;
                } else if (jtype_l == JOINT_SLIDE) {
                    // Use pre-computed pre-joint world-frame axis from FK
                    int ax_idx = env*njnt*3 + ji*3;
                    double aw_x=jnt_xaxis_d[ax_idx],aw_y=jnt_xaxis_d[ax_idx+1],aw_z=jnt_xaxis_d[ax_idx+2];
                    double qdot=ev[dof];
                    // Classical body-origin acceleration includes 2 * omega_parent x slide_velocity.
                    ai_x+=2.0*(op_y*(aw_z*qdot)-op_z*(aw_y*qdot)); ai_y+=2.0*(op_z*(aw_x*qdot)-op_x*(aw_z*qdot)); ai_z+=2.0*(op_x*(aw_y*qdot)-op_y*(aw_x*qdot));
                }
            }
            int o3 = env*nbody*3+bid*3;
            rnea_omega[o3]=oi_x; rnea_omega[o3+1]=oi_y; rnea_omega[o3+2]=oi_z;
            rnea_alpha[o3]=ali_x; rnea_alpha[o3+1]=ali_y; rnea_alpha[o3+2]=ali_z;
            rnea_acc[o3]=ai_x; rnea_acc[o3+1]=ai_y; rnea_acc[o3+2]=ai_z;
        }
        __syncthreads();
    }

    // Body forces (double precision)
    for (int bid = 1 + threadIdx.x; bid < nbody; bid += blockDim.x) {
        double m = (double)body_mass[bid];
        double Ix=(double)body_inertia[bid*3],Iy=(double)body_inertia[bid*3+1],Iz=(double)body_inertia[bid*3+2];
        double ipx=(double)body_ipos[bid*3],ipy=(double)body_ipos[bid*3+1],ipz=(double)body_ipos[bid*3+2];
        int bq4=env*nbody*4+bid*4;
        double bqw=(double)xquat[bq4],bqx=(double)xquat[bq4+1],bqy=(double)xquat[bq4+2],bqz=(double)xquat[bq4+3];
        double tw_x=2.0*(bqy*ipz-bqz*ipy),tw_y=2.0*(bqz*ipx-bqx*ipz),tw_z=2.0*(bqx*ipy-bqy*ipx);
        double rc_x=ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
        double rc_y=ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
        double rc_z=ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);
        int o3=env*nbody*3+bid*3;
        double oi_x=rnea_omega[o3],oi_y=rnea_omega[o3+1],oi_z=rnea_omega[o3+2];
        double ali_x=rnea_alpha[o3],ali_y=rnea_alpha[o3+1],ali_z=rnea_alpha[o3+2];
        double ao_x=rnea_acc[o3],ao_y=rnea_acc[o3+1],ao_z=rnea_acc[o3+2];
        double aXrc_x=ali_y*rc_z-ali_z*rc_y,aXrc_y=ali_z*rc_x-ali_x*rc_z,aXrc_z=ali_x*rc_y-ali_y*rc_x;
        double oXrc_x=oi_y*rc_z-oi_z*rc_y,oXrc_y=oi_z*rc_x-oi_x*rc_z,oXrc_z=oi_x*rc_y-oi_y*rc_x;
        double oXoXrc_x=oi_y*oXrc_z-oi_z*oXrc_y,oXoXrc_y=oi_z*oXrc_x-oi_x*oXrc_z,oXoXrc_z=oi_x*oXrc_y-oi_y*oXrc_x;
        double ac_x=ao_x+aXrc_x+oXoXrc_x,ac_y=ao_y+aXrc_y+oXoXrc_y,ac_z=ao_z+aXrc_z+oXoXrc_z;
        rnea_fnet[o3]=m*ac_x; rnea_fnet[o3+1]=m*ac_y; rnea_fnet[o3+2]=m*ac_z;
        double iqw=(double)body_iquat[bid*4],iqx=(double)body_iquat[bid*4+1],iqy=(double)body_iquat[bid*4+2],iqz=(double)body_iquat[bid*4+3];
        double cqw=bqw*iqw-bqx*iqx-bqy*iqy-bqz*iqz;
        double cqx=bqw*iqx+bqx*iqw+bqy*iqz-bqz*iqy;
        double cqy=bqw*iqy-bqx*iqz+bqy*iqw+bqz*iqx;
        double cqz=bqw*iqz+bqx*iqy-bqy*iqx+bqz*iqw;
        double xx=cqx*cqx,yy=cqy*cqy,zz=cqz*cqz,xy=cqx*cqy,xz=cqx*cqz,yz=cqy*cqz,wx=cqw*cqx,wy=cqw*cqy,wz=cqw*cqz;
        double R00=1-2*(yy+zz),R01=2*(xy-wz),R02=2*(xz+wy);
        double R10=2*(xy+wz),R11=1-2*(xx+zz),R12=2*(yz-wx);
        double R20=2*(xz-wy),R21=2*(yz+wx),R22=1-2*(xx+yy);
        double Io_x=(Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*oi_x+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*oi_y+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*oi_z;
        double Io_y=(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*oi_x+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*oi_y+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*oi_z;
        double Io_z=(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*oi_x+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*oi_y+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*oi_z;
        double Ia_x=(Ix*R00*R00+Iy*R01*R01+Iz*R02*R02)*ali_x+(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*ali_y+(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*ali_z;
        double Ia_y=(Ix*R00*R10+Iy*R01*R11+Iz*R02*R12)*ali_x+(Ix*R10*R10+Iy*R11*R11+Iz*R12*R12)*ali_y+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*ali_z;
        double Ia_z=(Ix*R00*R20+Iy*R01*R21+Iz*R02*R22)*ali_x+(Ix*R10*R20+Iy*R11*R21+Iz*R12*R22)*ali_y+(Ix*R20*R20+Iy*R21*R21+Iz*R22*R22)*ali_z;
        double oXIo_x=oi_y*Io_z-oi_z*Io_y,oXIo_y=oi_z*Io_x-oi_x*Io_z,oXIo_z=oi_x*Io_y-oi_y*Io_x;
        double rcXf_x=rc_y*rnea_fnet[o3+2]-rc_z*rnea_fnet[o3+1];
        double rcXf_y=rc_z*rnea_fnet[o3]-rc_x*rnea_fnet[o3+2];
        double rcXf_z=rc_x*rnea_fnet[o3+1]-rc_y*rnea_fnet[o3];
        rnea_tnet[o3]=Ia_x+oXIo_x+rcXf_x; rnea_tnet[o3+1]=Ia_y+oXIo_y+rcXf_y; rnea_tnet[o3+2]=Ia_z+oXIo_z+rcXf_z;
    }
    __syncthreads();

    // Backward pass: leaf to root (double precision)
    for (int level = tree_depth; level >= 1; level--) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;
        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int bid = level_body_ids[ls + local_idx];
            if (bid < 1) continue;
            int p = body_parent[bid];
            int o3 = env*nbody*3+bid*3;
            double fx=rnea_fnet[o3],fy=rnea_fnet[o3+1],fz=rnea_fnet[o3+2];
            double tx=rnea_tnet[o3],ty=rnea_tnet[o3+1],tz=rnea_tnet[o3+2];
            int ja=body_jnt_adr[bid]; int jn=body_jnt_num[bid];
            for (int k = 0; k < jn; k++) {
                int ji=ja+k; int jtype_l=jnt_type[ji]; int dof=jnt_dof_adr[ji];
                if (jtype_l == JOINT_HINGE) {
                    // Use pre-computed pre-joint world-frame axis from FK
                    int ax_idx = env*njnt*3 + ji*3;
                    double aw_x=jnt_xaxis_d[ax_idx],aw_y=jnt_xaxis_d[ax_idx+1],aw_z=jnt_xaxis_d[ax_idx+2];
                    if (dof < nv) atomicAdd(&qfrc_bias[env*nv+dof], aw_x*tx+aw_y*ty+aw_z*tz);
                } else if (jtype_l == JOINT_FREE) {
                    if (dof+5 < nv) {
                        atomicAdd(&qfrc_bias[env*nv+dof],fx); atomicAdd(&qfrc_bias[env*nv+dof+1],fy); atomicAdd(&qfrc_bias[env*nv+dof+2],fz);
                        int bq4f=env*nbody*4+bid*4;
                        double fqw=(double)xquat[bq4f],fqx=(double)xquat[bq4f+1],fqy=(double)xquat[bq4f+2],fqz=(double)xquat[bq4f+3];
                        double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                        double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                        double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                        double tb_x=R00*tx+R10*ty+R20*tz;
                        double tb_y=R01*tx+R11*ty+R21*tz;
                        double tb_z=R02*tx+R12*ty+R22*tz;
                        atomicAdd(&qfrc_bias[env*nv+dof+3],tb_x); atomicAdd(&qfrc_bias[env*nv+dof+4],tb_y); atomicAdd(&qfrc_bias[env*nv+dof+5],tb_z);
                    }
                } else if (jtype_l == JOINT_SLIDE) {
                    // Use pre-computed pre-joint world-frame axis from FK
                    int ax_idx = env*njnt*3 + ji*3;
                    double aw_x=jnt_xaxis_d[ax_idx],aw_y=jnt_xaxis_d[ax_idx+1],aw_z=jnt_xaxis_d[ax_idx+2];
                    if (dof < nv) atomicAdd(&qfrc_bias[env*nv+dof], aw_x*fx+aw_y*fy+aw_z*fz);
                }
            }
            if (p > 0 && p < nbody) {
                int b3=env*nbody*3+bid*3, p3=env*nbody*3+p*3;
                double rx=xpos[b3]-xpos[p3],ry=xpos[b3+1]-xpos[p3+1],rz=xpos[b3+2]-xpos[p3+2];
                atomicAdd(&rnea_fnet[p3],fx); atomicAdd(&rnea_fnet[p3+1],fy); atomicAdd(&rnea_fnet[p3+2],fz);
                atomicAdd(&rnea_tnet[p3],tx+(ry*fz-rz*fy)); atomicAdd(&rnea_tnet[p3+1],ty+(rz*fx-rx*fz)); atomicAdd(&rnea_tnet[p3+2],tz+(rx*fy-ry*fx));
            }
        }
        __syncthreads();
    }
}

// ── Fluid drag kernel (MuJoCo inertia-based fluid model) ────────────────────
// Computes viscous + quadratic drag forces for opt.density/viscosity > 0.
// 3-phase fused kernel: velocity propagation -> wrench computation -> J^T projection.
// Reuses rnea_fnet/rnea_tnet/rnea_acc as scratch (safe: RNEA is done).

HIPGL_GLOBAL void kernel_compute_fluid_forces_fused(
    int num_envs, int tree_depth, int nv, int njnt, int nbody,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ body_mass,
    const float*  __restrict__ body_inertia_box,
    const double* __restrict__ body_ipos,
    const double* __restrict__ body_iquat,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const double* __restrict__ qvel,
    float density,
    float viscosity,
    const double* __restrict__ rnea_omega,
    double*       __restrict__ rnea_fnet,
    double*       __restrict__ rnea_tnet,
    double*       __restrict__ rnea_acc,
    double*       __restrict__ qfrc_fluid)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;

    constexpr double PI = 3.14159265358979323846;

    for (int i = threadIdx.x; i < nv; i += blockDim.x)
        qfrc_fluid[env * nv + i] = 0.0;
    for (int i = threadIdx.x; i < nbody * 3; i += blockDim.x) {
        int idx = env * nbody * 3 + i;
        rnea_acc[idx] = 0.0;
        rnea_fnet[idx] = 0.0;
        rnea_tnet[idx] = 0.0;
    }
    __syncthreads();

    // Phase 1: forward pass — propagate linear velocity at body frame origin
    for (int level = 1; level <= tree_depth; level++) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;
        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int bid = level_body_ids[ls + local_idx];
            int p = body_parent[bid];

            double vp_x = 0, vp_y = 0, vp_z = 0;
            double op_x = 0, op_y = 0, op_z = 0;
            if (p > 0) {
                int p3 = env * nbody * 3 + p * 3;
                op_x = rnea_omega[p3]; op_y = rnea_omega[p3+1]; op_z = rnea_omega[p3+2];
                vp_x = rnea_acc[p3]; vp_y = rnea_acc[p3+1]; vp_z = rnea_acc[p3+2];
            }

            int b3 = env * nbody * 3 + bid * 3;
            int pb3 = env * nbody * 3 + p * 3;
            double rx = (double)xpos[b3] - (double)xpos[pb3];
            double ry = (double)xpos[b3+1] - (double)xpos[pb3+1];
            double rz = (double)xpos[b3+2] - (double)xpos[pb3+2];

            double vx = vp_x + (op_y * rz - op_z * ry);
            double vy = vp_y + (op_z * rx - op_x * rz);
            double vz = vp_z + (op_x * ry - op_y * rx);

            int ja = body_jnt_adr[bid];
            int jn = body_jnt_num[bid];
            const double* ev = qvel + env * nv;
            for (int k = 0; k < jn; k++) {
                int ji = ja + k;
                int jt = jnt_type[ji];
                int da = jnt_dof_adr[ji];
                if (jt == JOINT_FREE) {
                    vx += ev[da]; vy += ev[da+1]; vz += ev[da+2];
                } else if (jt == JOINT_SLIDE) {
                    double rate = ev[da];
                    int bq4 = env * nbody * 4 + bid * 4;
                    double qw = (double)xquat[bq4], qx = (double)xquat[bq4+1];
                    double qy = (double)xquat[bq4+2], qz = (double)xquat[bq4+3];
                    double lx = (double)jnt_axis[ji*3], ly = (double)jnt_axis[ji*3+1], lz = (double)jnt_axis[ji*3+2];
                    double tw_x = 2.0*(qy*lz-qz*ly), tw_y = 2.0*(qz*lx-qx*lz), tw_z = 2.0*(qx*ly-qy*lx);
                    double aw_x = lx+qw*tw_x+(qy*tw_z-qz*tw_y);
                    double aw_y = ly+qw*tw_y+(qz*tw_x-qx*tw_z);
                    double aw_z = lz+qw*tw_z+(qx*tw_y-qy*tw_x);
                    vx += rate * aw_x; vy += rate * aw_y; vz += rate * aw_z;
                }
            }
            rnea_acc[b3] = vx; rnea_acc[b3+1] = vy; rnea_acc[b3+2] = vz;
        }
        __syncthreads();
    }

    // Phase 2: compute fluid wrenches at each body COM, store in rnea_fnet/tnet
    for (int local_idx = threadIdx.x; local_idx < nbody; local_idx += blockDim.x) {
        int bid = local_idx;
        if (bid == 0) continue;
        float m = body_mass[bid];
        if (m < 1e-10f) continue;

        int o3 = env * nbody * 3 + bid * 3;
        double wx = rnea_omega[o3], wy = rnea_omega[o3+1], wz = rnea_omega[o3+2];
        double v_orig_x = rnea_acc[o3], v_orig_y = rnea_acc[o3+1], v_orig_z = rnea_acc[o3+2];

        // Correct origin velocity to COM velocity: v_com = v_origin + omega × delta
        int bq4 = env * nbody * 4 + bid * 4;
        double q0w = (double)xquat[bq4], q0x = (double)xquat[bq4+1];
        double q0y = (double)xquat[bq4+2], q0z = (double)xquat[bq4+3];
        double ip_x = (double)body_ipos[bid*3], ip_y = (double)body_ipos[bid*3+1], ip_z = (double)body_ipos[bid*3+2];

        // delta = rotate(xquat, ipos)
        double t0, t1, t2;
        t0 = 2.0*(q0y*ip_z-q0z*ip_y); t1 = 2.0*(q0z*ip_x-q0x*ip_z); t2 = 2.0*(q0x*ip_y-q0y*ip_x);
        double dx = ip_x+q0w*t0+(q0y*t2-q0z*t1);
        double dy = ip_y+q0w*t1+(q0z*t0-q0x*t2);
        double dz = ip_z+q0w*t2+(q0x*t1-q0y*t0);

        double vx = v_orig_x + (wy*dz - wz*dy);
        double vy = v_orig_y + (wz*dx - wx*dz);
        double vz = v_orig_z + (wx*dy - wy*dx);

        // Inertial frame quaternion: xi_quat = xquat * body_iquat
        double q1w = (double)body_iquat[bid*4], q1x = (double)body_iquat[bid*4+1];
        double q1y = (double)body_iquat[bid*4+2], q1z = (double)body_iquat[bid*4+3];
        double qw = q0w*q1w - q0x*q1x - q0y*q1y - q0z*q1z;
        double qx = q0w*q1x + q0x*q1w + q0y*q1z - q0z*q1y;
        double qy = q0w*q1y - q0x*q1z + q0y*q1w + q0z*q1x;
        double qz = q0w*q1z + q0x*q1y - q0y*q1x + q0z*q1w;

        // Rotate to inertial-local frame: v_local = R^T * v = rotate(conj(q), v)
        t0 = 2.0*(qy*wz-qz*wy); t1 = 2.0*(qz*wx-qx*wz); t2 = 2.0*(qx*wy-qy*wx);
        double wl_x = wx - qw*t0 + (qy*t2-qz*t1);
        double wl_y = wy - qw*t1 + (qz*t0-qx*t2);
        double wl_z = wz - qw*t2 + (qx*t1-qy*t0);

        t0 = 2.0*(qy*vz-qz*vy); t1 = 2.0*(qz*vx-qx*vz); t2 = 2.0*(qx*vy-qy*vx);
        double vl_x = vx - qw*t0 + (qy*t2-qz*t1);
        double vl_y = vy - qw*t1 + (qz*t0-qx*t2);
        double vl_z = vz - qw*t2 + (qx*t1-qy*t0);

        // Inertia-equivalent box FULL dimensions (matching MuJoCo mj_inertiaBoxFluidModel)
        double bx = (double)body_inertia_box[bid*3];
        double by = (double)body_inertia_box[bid*3+1];
        double bz = (double)body_inertia_box[bid*3+2];

        double fl_x = 0.0, fl_y = 0.0, fl_z = 0.0;
        double tl_x = 0.0, tl_y = 0.0, tl_z = 0.0;

        // Viscous forces: MuJoCo diam = (box[0]+box[1]+box[2])/3
        if (viscosity > 0.0f) {
            double diam = (bx + by + bz) / 3.0;
            double d3 = diam * diam * diam;
            double mu = (double)viscosity;
            tl_x = -PI * d3 * mu * wl_x;
            tl_y = -PI * d3 * mu * wl_y;
            tl_z = -PI * d3 * mu * wl_z;
            fl_x = -3.0 * PI * diam * mu * vl_x;
            fl_y = -3.0 * PI * diam * mu * vl_y;
            fl_z = -3.0 * PI * diam * mu * vl_z;
        }

        // Drag forces: exact MuJoCo formulas (engine_passive.c)
        if (density > 0.0f) {
            double rho = (double)density;
            fl_x -= 0.5 * rho * by * bz * fabs(vl_x) * vl_x;
            fl_y -= 0.5 * rho * bx * bz * fabs(vl_y) * vl_y;
            fl_z -= 0.5 * rho * bx * by * fabs(vl_z) * vl_z;

            double bx4 = bx*bx*bx*bx, by4 = by*by*by*by, bz4 = bz*bz*bz*bz;
            tl_x -= rho * bx * (by4 + bz4) * fabs(wl_x) * wl_x / 64.0;
            tl_y -= rho * by * (bx4 + bz4) * fabs(wl_y) * wl_y / 64.0;
            tl_z -= rho * bz * (bx4 + by4) * fabs(wl_z) * wl_z / 64.0;
        }

        // Rotate forces back to world: f_world = rotate(xi_quat, f_local)
        t0 = 2.0*(qy*fl_z-qz*fl_y); t1 = 2.0*(qz*fl_x-qx*fl_z); t2 = 2.0*(qx*fl_y-qy*fl_x);
        double fw_x = fl_x+qw*t0+(qy*t2-qz*t1);
        double fw_y = fl_y+qw*t1+(qz*t0-qx*t2);
        double fw_z = fl_z+qw*t2+(qx*t1-qy*t0);

        t0 = 2.0*(qy*tl_z-qz*tl_y); t1 = 2.0*(qz*tl_x-qx*tl_z); t2 = 2.0*(qx*tl_y-qy*tl_x);
        double tw_x = tl_x+qw*t0+(qy*t2-qz*t1);
        double tw_y = tl_y+qw*t1+(qz*t0-qx*t2);
        double tw_z = tl_z+qw*t2+(qx*t1-qy*t0);

        // Transport wrench from COM to body frame origin for backward pass:
        // tau_origin = tau_com + delta × f_com
        tw_x += (dy*fw_z - dz*fw_y);
        tw_y += (dz*fw_x - dx*fw_z);
        tw_z += (dx*fw_y - dy*fw_x);

        rnea_fnet[o3] = fw_x; rnea_fnet[o3+1] = fw_y; rnea_fnet[o3+2] = fw_z;
        rnea_tnet[o3] = tw_x; rnea_tnet[o3+1] = tw_y; rnea_tnet[o3+2] = tw_z;
    }
    __syncthreads();

    // Phase 3: backward pass — project wrenches to qfrc_fluid (leaf to root)
    for (int level = tree_depth; level >= 1; level--) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;
        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int bid = level_body_ids[ls + local_idx];
            if (bid < 1) continue;
            int p = body_parent[bid];
            int o3 = env * nbody * 3 + bid * 3;
            double fx = rnea_fnet[o3], fy = rnea_fnet[o3+1], fz = rnea_fnet[o3+2];
            double tx = rnea_tnet[o3], ty = rnea_tnet[o3+1], tz = rnea_tnet[o3+2];

            int ja = body_jnt_adr[bid]; int jn = body_jnt_num[bid];
            for (int k = 0; k < jn; k++) {
                int ji = ja + k; int jt = jnt_type[ji]; int dof = jnt_dof_adr[ji];
                if (jt == JOINT_HINGE) {
                    int bq4 = env * nbody * 4 + bid * 4;
                    double qw = (double)xquat[bq4], qx_l = (double)xquat[bq4+1];
                    double qy_l = (double)xquat[bq4+2], qz_l = (double)xquat[bq4+3];
                    double lax = (double)jnt_axis[ji*3], lay = (double)jnt_axis[ji*3+1], laz = (double)jnt_axis[ji*3+2];
                    double tw0 = 2.0*(qy_l*laz-qz_l*lay), tw1 = 2.0*(qz_l*lax-qx_l*laz), tw2 = 2.0*(qx_l*lay-qy_l*lax);
                    double aw_x = lax+qw*tw0+(qy_l*tw2-qz_l*tw1);
                    double aw_y = lay+qw*tw1+(qz_l*tw0-qx_l*tw2);
                    double aw_z = laz+qw*tw2+(qx_l*tw1-qy_l*tw0);
                    if (dof < nv) atomicAdd(&qfrc_fluid[env*nv+dof], aw_x*tx+aw_y*ty+aw_z*tz);
                } else if (jt == JOINT_FREE) {
                    if (dof+5 < nv) {
                        atomicAdd(&qfrc_fluid[env*nv+dof], fx); atomicAdd(&qfrc_fluid[env*nv+dof+1], fy); atomicAdd(&qfrc_fluid[env*nv+dof+2], fz);
                        // Project world-frame torque to body frame via R^T
                        int bq4f=env*nbody*4+bid*4;
                        double fqw=(double)xquat[bq4f],fqx=(double)xquat[bq4f+1],fqy=(double)xquat[bq4f+2],fqz=(double)xquat[bq4f+3];
                        double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                        double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                        double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                        double tb_x=R00*tx+R10*ty+R20*tz;
                        double tb_y=R01*tx+R11*ty+R21*tz;
                        double tb_z=R02*tx+R12*ty+R22*tz;
                        atomicAdd(&qfrc_fluid[env*nv+dof+3], tb_x); atomicAdd(&qfrc_fluid[env*nv+dof+4], tb_y); atomicAdd(&qfrc_fluid[env*nv+dof+5], tb_z);
                    }
                } else if (jt == JOINT_SLIDE) {
                    int bq4 = env * nbody * 4 + bid * 4;
                    double qw = (double)xquat[bq4], qx_l = (double)xquat[bq4+1];
                    double qy_l = (double)xquat[bq4+2], qz_l = (double)xquat[bq4+3];
                    double lax = (double)jnt_axis[ji*3], lay = (double)jnt_axis[ji*3+1], laz = (double)jnt_axis[ji*3+2];
                    double tw0 = 2.0*(qy_l*laz-qz_l*lay), tw1 = 2.0*(qz_l*lax-qx_l*laz), tw2 = 2.0*(qx_l*lay-qy_l*lax);
                    double aw_x = lax+qw*tw0+(qy_l*tw2-qz_l*tw1);
                    double aw_y = lay+qw*tw1+(qz_l*tw0-qx_l*tw2);
                    double aw_z = laz+qw*tw2+(qx_l*tw1-qy_l*tw0);
                    if (dof < nv) atomicAdd(&qfrc_fluid[env*nv+dof], aw_x*fx+aw_y*fy+aw_z*fz);
                }
            }
            if (p > 0 && p < nbody) {
                int b3 = env*nbody*3+bid*3, p3 = env*nbody*3+p*3;
                double rx = (double)xpos[b3]-(double)xpos[p3];
                double ry = (double)xpos[b3+1]-(double)xpos[p3+1];
                double rz = (double)xpos[b3+2]-(double)xpos[p3+2];
                atomicAdd(&rnea_fnet[p3], fx); atomicAdd(&rnea_fnet[p3+1], fy); atomicAdd(&rnea_fnet[p3+2], fz);
                atomicAdd(&rnea_tnet[p3], tx+(ry*fz-rz*fy));
                atomicAdd(&rnea_tnet[p3+1], ty+(rz*fx-rx*fz));
                atomicAdd(&rnea_tnet[p3+2], tz+(rx*fy-ry*fx));
            }
        }
        __syncthreads();
    }
}


template <int BLK, int MAXB, int MAXJ, int MAXNV>
HIPGL_GLOBAL void __launch_bounds__(BLK, 1) kernel_bias_fluid_fused(
    int num_envs, int tree_depth, int nv, int njnt, int nbody,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_axis,
    double gx, double gy, double gz,
    const double* __restrict__ body_mass,
    const double* __restrict__ body_inertia,
    const float*  __restrict__ body_inertia_box,
    const double* __restrict__ body_ipos,
    const double* __restrict__ body_iquat,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const double* __restrict__ qvel,
    const double* __restrict__ jnt_xaxis_d,
    float density, float viscosity,
    double*       __restrict__ qfrc_bias,
    double*       __restrict__ qfrc_fluid)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    bias_fluid_body<BLK, MAXB, MAXJ, MAXNV, false>(
        env, num_envs, tree_depth, nv, njnt, nbody, level_start_arr, level_body_ids, body_parent, body_jnt_adr, body_jnt_num, jnt_type, jnt_dof_adr, jnt_axis, gx, gy, gz, body_mass, body_inertia, body_inertia_box, body_ipos, body_iquat, xpos, xquat, qvel, jnt_xaxis_d, density, viscosity, qfrc_bias, qfrc_fluid);
}

// ── Apply forces kernel (PGS path only - computes net force per DOF) ────────

HIPGL_GLOBAL void kernel_apply_forces(
    int njnt, int num_envs,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const int*   __restrict__ jnt_dof_adr,
    const int*   __restrict__ jnt_qpos_adr,
    const float* __restrict__ jnt_damping,
    const float* __restrict__ jnt_frictionloss,
    int nv, int nq,
    int nu,
    const int*   __restrict__ actuator_jnt,
    const int*   __restrict__ actuator_type,
    const float* __restrict__ actuator_gain,
    const float* __restrict__ actuator_kv,
    const float* __restrict__ actuator_biasprm,
    const float* __restrict__ actuator_forcerange,
    const float* __restrict__ actuator_ctrlrange,
    const float* __restrict__ qpos,
    const float* __restrict__ qvel,
    const float* __restrict__ ctrl,
    const float* __restrict__ qfrc_applied,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ qfrc_fluid,
    const double* __restrict__ constraint_lam,
    const double* __restrict__ constraint_J,
    const int*   __restrict__ nconstraint,
    float inv_dt,
    double*      __restrict__ qacc,
    bool enable_damping)
{
    int jid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (jid >= njnt || env >= num_envs) return;

    int dof = jnt_dof_adr[jid];
    int jtype = jnt_type[jid];
    const float* ev = qvel + env * nv;
    const float* ep = qpos + env * nq;

    int ndof = 1;
    if (jtype == JOINT_FREE) ndof = 6;
    else if (jtype == JOINT_BALL) ndof = 3;

    for (int d = 0; d < ndof; d++) {
        int vi = dof + d;
        float force = 0.0f;
        force += qfrc_applied[env * nv + vi];
        force -= (float)qfrc_bias[env * nv + vi];
        if (qfrc_fluid) force += (float)qfrc_fluid[env * nv + vi];
        if (enable_damping) {
            float damp = jnt_damping[jid];
            force -= damp * ev[vi];
        }
        for (int a = 0; a < nu; a++) {
            if (actuator_jnt[a] != jid) continue;
            float c = ctrl[env * nu + a];
            if (actuator_ctrlrange) {
                float cl = actuator_ctrlrange[a*2], ch = actuator_ctrlrange[a*2+1];
                if (cl < ch) { if (c < cl) c = cl; if (c > ch) c = ch; }
            }
            int atype = actuator_type ? actuator_type[a] : ACTUATOR_MOTOR;
            float gain = actuator_gain[a];
            float act_force = 0.0f;
            if (atype == ACTUATOR_MOTOR) {
                if (d == 0 && ndof == 1) act_force = gain * c;
                else if (jtype == JOINT_FREE && d < 3) act_force = gain * c;
            }
            else if (atype == ACTUATOR_POSITION) {
                if (d == 0 && ndof == 1) {
                    int qa = jnt_qpos_adr[jid];
                    act_force = gain * (c - ep[qa]) - (actuator_kv ? actuator_kv[a] : 0.0f) * ev[vi];
                }
            }
            else if (atype == ACTUATOR_VELOCITY) {
                if (d == 0 && ndof == 1) act_force = gain * (c - ev[vi]);
            }
            else if (atype == ACTUATOR_GENERAL && actuator_biasprm) {
                if (d == 0 && ndof == 1) {
                    int qa = jnt_qpos_adr[jid];
                    act_force = gain * c + actuator_biasprm[a*3] + actuator_biasprm[a*3+1] * ep[qa] + actuator_biasprm[a*3+2] * ev[vi];
                }
            }
            if (actuator_forcerange) {
                float fl = actuator_forcerange[a*2], fh = actuator_forcerange[a*2+1];
                if (fl < fh) { if (act_force < fl) act_force = fl; if (act_force > fh) act_force = fh; }
            }
            force += act_force;
        }
        int nc = nconstraint[env];
        const double* env_J = constraint_J + (size_t)env * MAX_CONSTRAINTS * nv;
        const double* env_lam = constraint_lam + (size_t)env * MAX_CONSTRAINTS;
        for (int ci = 0; ci < nc; ci++)
            force += (float)(env_J[ci * nv + vi] * env_lam[ci]) * inv_dt;
        qacc[env * nv + vi] = (double)force;
    }
}

// ── CRBA mass matrix solve (PGS path fallback) ─────────────────────────────

HIPGL_GLOBAL void kernel_mass_matrix_solve(
    int num_envs, int nv, int njnt, int nbody,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const int*   __restrict__ jnt_dof_adr,
    const float* __restrict__ jnt_axis,
    const float* __restrict__ jnt_pos,
    const float* __restrict__ body_mass,
    const float* __restrict__ body_inertia,
    const float* __restrict__ body_ipos,
    const float* __restrict__ body_iquat,
    const int*   __restrict__ body_parent,
    const float* __restrict__ jnt_damping,
    float timestep,
    bool enable_damping,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    double*      __restrict__ qacc)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;
    if (nv > KERNEL_MAXNV) return;

    float M[KERNEL_MAXNV][KERNEL_MAXNV];
    for (int i = 0; i < nv; i++)
        for (int j = 0; j < nv; j++)
            M[i][j] = 0.0f;

    float jnt_axis_w[MAX_JOINTS * 3];
    float jnt_pivot_w[MAX_JOINTS * 3];
    for (int ji = 0; ji < njnt && ji < MAX_JOINTS; ji++) {
        int jbody = jnt_body[ji];
        int parent = body_parent[jbody];
        int pq4 = env * nbody * 4 + parent * 4;
        float qw = xquat[pq4], qx = xquat[pq4+1], qy = xquat[pq4+2], qz = xquat[pq4+3];
        float lax = jnt_axis[ji*3], lay = jnt_axis[ji*3+1], laz = jnt_axis[ji*3+2];
        float tw_x = 2.0f*(qy*laz-qz*lay), tw_y = 2.0f*(qz*lax-qx*laz), tw_z = 2.0f*(qx*lay-qy*lax);
        jnt_axis_w[ji*3]   = lax + qw*tw_x + (qy*tw_z - qz*tw_y);
        jnt_axis_w[ji*3+1] = lay + qw*tw_y + (qz*tw_x - qx*tw_z);
        jnt_axis_w[ji*3+2] = laz + qw*tw_z + (qx*tw_y - qy*tw_x);
        int b3 = env * nbody * 3 + jbody * 3;
        jnt_pivot_w[ji*3]   = xpos[b3];
        jnt_pivot_w[ji*3+1] = xpos[b3+1];
        jnt_pivot_w[ji*3+2] = xpos[b3+2];
    }
    for (int bid = 1; bid < nbody; bid++) {
        float m = body_mass[bid]; if (m < 1e-12f) continue;
        float ipx = body_ipos[bid*3], ipy = body_ipos[bid*3+1], ipz = body_ipos[bid*3+2];
        int bq4 = env*nbody*4+bid*4;
        float bqw=xquat[bq4], bqx=xquat[bq4+1], bqy=xquat[bq4+2], bqz=xquat[bq4+3];
        float tw_x=2.0f*(bqy*ipz-bqz*ipy), tw_y=2.0f*(bqz*ipx-bqx*ipz), tw_z=2.0f*(bqx*ipy-bqy*ipx);
        float ri_x=ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
        float ri_y=ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
        float ri_z=ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);
        int b3=env*nbody*3+bid*3;
        float com_x=xpos[b3]+ri_x, com_y=xpos[b3+1]+ri_y, com_z=xpos[b3+2]+ri_z;
        float Ix=body_inertia[bid*3], Iy=body_inertia[bid*3+1], Iz=body_inertia[bid*3+2];
        float xx=bqx*bqx, yy=bqy*bqy, zz=bqz*bqz;
        float xy=bqx*bqy, xz=bqx*bqz, yz=bqy*bqz;
        float wx=bqw*bqx, wy=bqw*bqy, wz=bqw*bqz;
        float Iw[6];
        Iw[0]=Ix*(1-2*(yy+zz))*(1-2*(yy+zz))+Iy*(2*(xy-wz))*(2*(xy-wz))+Iz*(2*(xz+wy))*(2*(xz+wy));
        Iw[1]=Ix*(1-2*(yy+zz))*(2*(xy+wz))+Iy*(2*(xy-wz))*(1-2*(xx+zz))+Iz*(2*(xz+wy))*(2*(yz-wx));
        Iw[2]=Ix*(1-2*(yy+zz))*(2*(xz-wy))+Iy*(2*(xy-wz))*(2*(yz+wx))+Iz*(2*(xz+wy))*(1-2*(xx+yy));
        Iw[3]=Ix*(2*(xy+wz))*(2*(xy+wz))+Iy*(1-2*(xx+zz))*(1-2*(xx+zz))+Iz*(2*(yz-wx))*(2*(yz-wx));
        Iw[4]=Ix*(2*(xy+wz))*(2*(xz-wy))+Iy*(1-2*(xx+zz))*(2*(yz+wx))+Iz*(2*(yz-wx))*(1-2*(xx+yy));
        Iw[5]=Ix*(2*(xz-wy))*(2*(xz-wy))+Iy*(2*(yz+wx))*(2*(yz+wx))+Iz*(1-2*(xx+yy))*(1-2*(xx+yy));

        int chain[32]; int chain_len = 0; int cur = bid;
        while (cur > 0 && chain_len < 32) {
            for (int ji = 0; ji < njnt; ji++) if (jnt_body[ji] == cur) chain[chain_len++] = ji;
            cur = body_parent[cur];
        }
        for (int ci = 0; ci < chain_len; ci++) {
            int ji=chain[ci]; int jt_i=jnt_type[ji]; int di=jnt_dof_adr[ji];
            for (int cj = ci; cj < chain_len; cj++) {
                int jj=chain[cj]; int jt_j=jnt_type[jj]; int dj=jnt_dof_adr[jj];
                if (jt_i==JOINT_HINGE && jt_j==JOINT_HINGE) {
                    float *ai=jnt_axis_w+ji*3, *aj_ptr=jnt_axis_w+jj*3;
                    float Iw_aj_x=Iw[0]*aj_ptr[0]+Iw[1]*aj_ptr[1]+Iw[2]*aj_ptr[2];
                    float Iw_aj_y=Iw[1]*aj_ptr[0]+Iw[3]*aj_ptr[1]+Iw[4]*aj_ptr[2];
                    float Iw_aj_z=Iw[2]*aj_ptr[0]+Iw[4]*aj_ptr[1]+Iw[5]*aj_ptr[2];
                    float it=ai[0]*Iw_aj_x+ai[1]*Iw_aj_y+ai[2]*Iw_aj_z;
                    float rki[3]={com_x-jnt_pivot_w[ji*3],com_y-jnt_pivot_w[ji*3+1],com_z-jnt_pivot_w[ji*3+2]};
                    float rkj[3]={com_x-jnt_pivot_w[jj*3],com_y-jnt_pivot_w[jj*3+1],com_z-jnt_pivot_w[jj*3+2]};
                    float ci_v[3]={ai[1]*rki[2]-ai[2]*rki[1],ai[2]*rki[0]-ai[0]*rki[2],ai[0]*rki[1]-ai[1]*rki[0]};
                    float cj_v[3]={aj_ptr[1]*rkj[2]-aj_ptr[2]*rkj[1],aj_ptr[2]*rkj[0]-aj_ptr[0]*rkj[2],aj_ptr[0]*rkj[1]-aj_ptr[1]*rkj[0]};
                    float mt=m*(ci_v[0]*cj_v[0]+ci_v[1]*cj_v[1]+ci_v[2]*cj_v[2]);
                    float val=it+mt; M[di][dj]+=val; if(ci!=cj) M[dj][di]+=val;
                }
                else if (jt_i==JOINT_FREE && jt_j==JOINT_FREE && ji==jj) {
                    // Body-frame inertia using iquat (not xquat)
                    float iqw=body_iquat[bid*4], iqx=body_iquat[bid*4+1], iqy=body_iquat[bid*4+2], iqz=body_iquat[bid*4+3];
                    float ixx2=iqx*iqx, iyy2=iqy*iqy, izz2=iqz*iqz;
                    float ixy2=iqx*iqy, ixz2=iqx*iqz, iyz2=iqy*iqz;
                    float iwx2=iqw*iqx, iwy2=iqw*iqy, iwz2=iqw*iqz;
                    float B00=1-2*(iyy2+izz2), B01=2*(ixy2-iwz2), B02=2*(ixz2+iwy2);
                    float B10=2*(ixy2+iwz2), B11=1-2*(ixx2+izz2), B12=2*(iyz2-iwx2);
                    float B20=2*(ixz2-iwy2), B21=2*(iyz2+iwx2), B22=1-2*(ixx2+iyy2);
                    float Ib[6] = {
                        Ix*B00*B00+Iy*B01*B01+Iz*B02*B02,
                        Ix*B00*B10+Iy*B01*B11+Iz*B02*B12,
                        Ix*B00*B20+Iy*B01*B21+Iz*B02*B22,
                        Ix*B10*B10+Iy*B11*B11+Iz*B12*B12,
                        Ix*B10*B20+Iy*B11*B21+Iz*B12*B22,
                        Ix*B20*B20+Iy*B21*B21+Iz*B22*B22
                    };
                    // Rotate offset to body frame via R_body^T
                    int fq4 = env*nbody*4+bid*4;
                    float fqw=xquat[fq4], fqx=xquat[fq4+1], fqy=xquat[fq4+2], fqz=xquat[fq4+3];
                    float Fb00=1-2*(fqy*fqy+fqz*fqz), Fb01=2*(fqx*fqy+fqw*fqz), Fb02=2*(fqx*fqz-fqw*fqy);
                    float Fb10=2*(fqx*fqy-fqw*fqz), Fb11=1-2*(fqx*fqx+fqz*fqz), Fb12=2*(fqy*fqz+fqw*fqx);
                    float Fb20=2*(fqx*fqz+fqw*fqy), Fb21=2*(fqy*fqz-fqw*fqx), Fb22=1-2*(fqx*fqx+fqy*fqy);
                    float rw[3] = {com_x - xpos[env*nbody*3+bid*3], com_y - xpos[env*nbody*3+bid*3+1], com_z - xpos[env*nbody*3+bid*3+2]};
                    float r[3] = {Fb00*rw[0]+Fb10*rw[1]+Fb20*rw[2],
                                  Fb01*rw[0]+Fb11*rw[1]+Fb21*rw[2],
                                  Fb02*rw[0]+Fb12*rw[1]+Fb22*rw[2]};
                    float r2 = r[0]*r[0]+r[1]*r[1]+r[2]*r[2];
                    // Translation block: M[di..di+2][di..di+2] += m * I_3
                    for(int k=0;k<3;k++) M[di+k][di+k]+=m;
                    // Rotation block: Ib + m*(r2*I - r*r^T)
                    for(int a=0;a<3;a++) for(int bb=0;bb<3;bb++) {
                        int ri=a, rj=bb;
                        int idx2;
                        if(ri<=rj) idx2=ri*3-ri*(ri-1)/2+(rj-ri);
                        else idx2=rj*3-rj*(rj-1)/2+(ri-rj);
                        M[di+3+a][di+3+bb] += Ib[idx2] + m*((a==bb?r2:0.0f)-r[a]*r[bb]);
                    }
                    // Cross terms: translation-rotation coupling
                    for(int a=0;a<3;a++) for(int bb=0;bb<3;bb++) {
                        if(a!=bb) {
                            int c=3-a-bb;
                            float eps=((a+1)%3==c)?1.0f:-1.0f;
                            M[di+3+a][di+bb] += m*eps*r[c];
                            M[di+bb][di+3+a] += m*eps*r[c];
                        }
                    }
                }
                else if (jt_i==JOINT_SLIDE && jt_j==JOINT_SLIDE) {
                    float *ai=jnt_axis_w+ji*3, *aj_ptr=jnt_axis_w+jj*3;
                    float val=m*(ai[0]*aj_ptr[0]+ai[1]*aj_ptr[1]+ai[2]*aj_ptr[2]);
                    M[di][dj]+=val; if(ci!=cj) M[dj][di]+=val;
                }
            }
        }
    }
    if (enable_damping) {
        for (int ji=0;ji<njnt;ji++) {
            float damp=jnt_damping[ji]; if(damp<=0.0f) continue;
            (void)damp;
        }
    }
    for(int i=0;i<nv;i++) M[i][i]+=1e-8f;
    double* f=qacc+env*nv; float b[KERNEL_MAXNV];
    for(int i=0;i<nv;i++) b[i]=(float)f[i];
    for(int i=0;i<nv;i++) {
        int mr=i; float mv=fabsf(M[i][i]);
        for(int k=i+1;k<nv;k++) if(fabsf(M[k][i])>mv){mv=fabsf(M[k][i]);mr=k;}
        if(mr!=i){for(int k=0;k<nv;k++){float t=M[i][k];M[i][k]=M[mr][k];M[mr][k]=t;} float t=b[i];b[i]=b[mr];b[mr]=t;}
        if(fabsf(M[i][i])<1e-10f) continue;
        for(int k=i+1;k<nv;k++){float fc=M[k][i]/M[i][i];for(int j=i;j<nv;j++)M[k][j]-=fc*M[i][j];b[k]-=fc*b[i];}
    }
    for(int i=nv-1;i>=0;i--){
        if(fabsf(M[i][i])<1e-10f){f[i]=0;continue;}
        float sum=b[i]; for(int j=i+1;j<nv;j++) sum-=M[i][j]*(float)f[j]; f[i]=(double)(sum/M[i][i]);
    }
}

// ── Integration kernel (semi-implicit Euler) ────────────────────────────────

HIPGL_GLOBAL void kernel_integrate(
    int njnt, int num_envs, double dt,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_range_d,
    const int*    __restrict__ jnt_limited,
    const int*    __restrict__ jnt_body,
    int nq, int nv, int nbody,
    bool enable_limits,
    float*        __restrict__ qpos,
    float*        __restrict__ qvel,
    const double* __restrict__ qacc,
    float*        __restrict__ sim_time,
    double*       __restrict__ qpos_d,
    double*       __restrict__ qvel_d,
    double free_joint_damp)
{
    int jid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;

    if (jid == 0 && sim_time) sim_time[env] += (float)dt;

    if (jid >= njnt) return;

    integrate_joint_d(jid, env, dt, jnt_type, jnt_qpos_adr, jnt_dof_adr,
                      jnt_range_d, jnt_limited, nq, nv, enable_limits,
                      qpos, qvel, qacc, qpos_d, qvel_d, free_joint_damp);
}

// ── Host launch wrappers ────────────────────────────────────────────────────

void launch_compute_inv_mass(
    const DevicePhysicsModel& model, DevicePhysicsData& data, hipStream_t stream)
{
    if (model.nbody <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.nbody + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_compute_inv_mass, grid, block, 0, stream,
        model.nbody, data.num_envs,
        model.body_mass, model.body_inertia,
        data.body_inv_mass, data.body_inv_inertia);
}

void launch_compute_bias_force(
    const DevicePhysicsModel& model, DevicePhysicsData& data, hipStream_t stream,
    const int* h_level_start_cached)
{
    if (model.nbody <= 1 || data.num_envs <= 0) return;
    int ne = data.num_envs;

    // Use the fused RNEA kernel: 1 block per env, all levels in one launch
    const int blk = 256;
    hipLaunchKernelGGL(kernel_compute_bias_force_fused, ne, blk, 0, stream,
        ne, model.tree_depth, model.nv, model.njnt, model.nbody,
        model.level_start, model.level_body_ids,
        model.body_parent, model.body_jnt_adr, model.body_jnt_num,
        model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
        model.gravity_d[0], model.gravity_d[1], model.gravity_d[2],
        model.body_mass_d, model.body_inertia_d, model.body_ipos_d, model.body_iquat_d,
        model.body_quat_d,
        data.xpos_d, data.xquat_d, data.qvel_d,
        data.jnt_xaxis_d,
        data.rnea_omega, data.rnea_alpha, data.rnea_acc,
        data.rnea_fnet, data.rnea_tnet,
        data.qfrc_bias);
}

void launch_compute_fluid_forces(
    const DevicePhysicsModel& model, DevicePhysicsData& data, hipStream_t stream,
    const int* h_level_start_cached)
{
    if (model.nbody <= 1 || data.num_envs <= 0) return;
    if (model.opt_density <= 0.0f && model.opt_viscosity <= 0.0f) return;
    if (!model.body_inertia_box) return;
    int ne = data.num_envs;
    const int blk = 256;
    hipLaunchKernelGGL(kernel_compute_fluid_forces_fused, ne, blk, 0, stream,
        ne, model.tree_depth, model.nv, model.njnt, model.nbody,
        model.level_start, model.level_body_ids,
        model.body_parent, model.body_jnt_adr, model.body_jnt_num,
        model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
        model.body_mass_d, model.body_inertia_box, model.body_ipos_d, model.body_iquat_d,
        data.xpos_d, data.xquat_d, data.qvel_d,
        model.opt_density, model.opt_viscosity,
        data.rnea_omega,
        data.rnea_fnet, data.rnea_tnet, data.rnea_acc,
        data.qfrc_fluid);
}

// ── Fused bias+fluid launcher ───────────────────────────────────────────────
// Replaces the back-to-back launch_compute_bias_force + launch_compute_fluid_forces
// pair with a single dispatch when the model fits the LDS-resident fast path.
// Falls back to the two original kernels otherwise (behaviour unchanged).

namespace {
constexpr int BF_BLK  = 64;   // one wavefront: every __syncthreads() folds away
constexpr int BF_MAXB = 64;   // LDS body capacity (live model: nbody = 38)
constexpr int BF_MAXJ = 64;   // LDS joint capacity (live model: njnt = 19)
constexpr int BF_MAXNV = 64;  // LDS dof capacity  (live model: nv = 43)
}

void launch_bias_and_fluid_forces(
    const DevicePhysicsModel& model, DevicePhysicsData& data, hipStream_t stream,
    const int* h_level_start_cached)
{
    if (model.nbody <= 1 || data.num_envs <= 0) return;

    const bool fluid_on = (model.opt_density > 0.0f || model.opt_viscosity > 0.0f)
                          && model.body_inertia_box != nullptr;

    if (fluid_on && model.nbody <= BF_MAXB && model.njnt <= BF_MAXJ
        && model.nv <= BF_MAXNV && model.tree_depth <= MAX_TREE_DEPTH) {
        int ne = data.num_envs;
        hipLaunchKernelGGL((kernel_bias_fluid_fused<BF_BLK, BF_MAXB, BF_MAXJ, BF_MAXNV>),
            ne, BF_BLK, 0, stream,
            ne, model.tree_depth, model.nv, model.njnt, model.nbody,
            model.level_start, model.level_body_ids,
            model.body_parent, model.body_jnt_adr, model.body_jnt_num,
            model.jnt_type, model.jnt_dof_adr, model.jnt_axis_d,
            model.gravity_d[0], model.gravity_d[1], model.gravity_d[2],
            model.body_mass_d, model.body_inertia_d, model.body_inertia_box,
            model.body_ipos_d, model.body_iquat_d,
            data.xpos_d, data.xquat_d, data.qvel_d,
            data.jnt_xaxis_d,
            model.opt_density, model.opt_viscosity,
            data.qfrc_bias, data.qfrc_fluid);
        return;
    }

    launch_compute_bias_force(model, data, stream, h_level_start_cached);
    launch_compute_fluid_forces(model, data, stream, h_level_start_cached);
}

void launch_apply_forces(
    const DevicePhysicsModel& model, const DevicePhysicsData& data,
    const SolverOptions& opts, hipStream_t stream)
{
    if (model.njnt <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.njnt + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    float inv_dt = (model.timestep > 1e-10f) ? (1.0f / model.timestep) : 0.0f;
    hipLaunchKernelGGL(kernel_apply_forces, grid, block, 0, stream,
        model.njnt, data.num_envs,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr, model.jnt_qpos_adr,
        model.jnt_damping, model.jnt_frictionloss, model.nv, model.nq,
        model.nu, model.actuator_jnt, model.actuator_type,
        model.actuator_gain, model.actuator_kv,
        model.actuator_biasprm,
        model.actuator_forcerange, model.actuator_ctrlrange,
        data.qpos, data.qvel, data.ctrl, data.qfrc_applied,
        data.qfrc_bias, data.qfrc_fluid,
        data.constraint_lam, data.constraint_J, data.nconstraint,
        inv_dt, data.qacc, opts.enable_damping);
}

void launch_mass_matrix_solve(
    const DevicePhysicsModel& model, DevicePhysicsData& data,
    const SolverOptions& opts, hipStream_t stream)
{
    if (model.nv <= 0 || data.num_envs <= 0) return;
    const int block = 64;
    int grid = (data.num_envs + block - 1) / block;
    hipLaunchKernelGGL(kernel_mass_matrix_solve, grid, block, 0, stream,
        data.num_envs, model.nv, model.njnt, model.nbody,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr,
        model.jnt_axis, model.jnt_pos,
        model.body_mass, model.body_inertia, model.body_ipos, model.body_iquat,
        model.body_parent,
        model.jnt_damping, model.timestep, opts.enable_damping,
        data.xpos, data.xquat, data.qacc);
}

void launch_integrate(
    const DevicePhysicsModel& model, DevicePhysicsData& data,
    const SolverOptions& opts, hipStream_t stream)
{
    if (model.njnt <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.njnt + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_integrate, grid, block, 0, stream,
        model.njnt, data.num_envs, (double)model.timestep,
        model.jnt_type, model.jnt_qpos_adr, model.jnt_dof_adr,
        model.jnt_range_d, model.jnt_limited, model.jnt_body,
        model.nq, model.nv, model.nbody, opts.enable_limits,
        data.qpos, data.qvel, data.qacc, data.time,
        data.qpos_d, data.qvel_d,
        (double)opts.free_joint_damping);
}

// ── MuJoCo mj_integratePos into a separate buffer ──────────────────────────
// qpos_out = integratePos(qpos, qvel, dt) per env, exactly as MuJoCo 3.x
// (engine_util_misc.c mj_integratePos / engine_util_spatial.c mju_quatIntegrate).
// Used with dt < 0 for robosuite-timed proprio (PhysicsEngine::lagged_kinematics);
// the live state is only read.

namespace mjint {
HIPGL_DEVICE inline void quatIntegrate(double quat[4], const double vel[3], double scale) {
    double tmp[3] = {vel[0], vel[1], vel[2]};
    double norm = sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);     // mju_normalize3
    if (norm < 1e-15) { tmp[0] = 1; tmp[1] = 0; tmp[2] = 0; }
    else { double inv = 1/norm; tmp[0] *= inv; tmp[1] *= inv; tmp[2] *= inv; }
    double angle = scale * norm;
    double qrot[4];                                                        // mju_axisAngle2Quat
    if (angle == 0) { qrot[0] = 1; qrot[1] = 0; qrot[2] = 0; qrot[3] = 0; }
    else {
        double sn = sin(angle*0.5);
        qrot[0] = cos(angle*0.5); qrot[1] = tmp[0]*sn; qrot[2] = tmp[1]*sn; qrot[3] = tmp[2]*sn;
    }
    double n4 = sqrt(quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]);
    if (n4 < 1e-15) { quat[0] = 1; quat[1] = 0; quat[2] = 0; quat[3] = 0; }   // mju_normalize4
    else if (fabs(n4 - 1) > 1e-15) {
        double inv = 1/n4; quat[0] *= inv; quat[1] *= inv; quat[2] *= inv; quat[3] *= inv;
    }
    double r[4] = {                                                        // mju_mulQuat
        quat[0]*qrot[0] - quat[1]*qrot[1] - quat[2]*qrot[2] - quat[3]*qrot[3],
        quat[0]*qrot[1] + quat[1]*qrot[0] + quat[2]*qrot[3] - quat[3]*qrot[2],
        quat[0]*qrot[2] - quat[1]*qrot[3] + quat[2]*qrot[0] + quat[3]*qrot[1],
        quat[0]*qrot[3] + quat[1]*qrot[2] - quat[2]*qrot[1] + quat[3]*qrot[0]
    };
    quat[0] = r[0]; quat[1] = r[1]; quat[2] = r[2]; quat[3] = r[3];
}
}  // namespace mjint

HIPGL_GLOBAL void kernel_integrate_pos_mj(
    int num_envs, int njnt, int nq, int nv,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ qpos,
    const double* __restrict__ qvel,
    double dt,
    double*       __restrict__ qpos_out)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    const double* q = qpos + (size_t)env * nq;
    const double* v = qvel + (size_t)env * nv;
    double* out = qpos_out + (size_t)env * nq;
    for (int i = threadIdx.x; i < nq; i += blockDim.x) out[i] = q[i];
    __syncthreads();
    for (int j = threadIdx.x; j < njnt; j += blockDim.x) {
        int padr = jnt_qpos_adr[j], vadr = jnt_dof_adr[j];
        switch (jnt_type[j]) {
        case JOINT_FREE:
            for (int i = 0; i < 3; i++) out[padr+i] += dt * v[vadr+i];
            mjint::quatIntegrate(out + padr + 3, v + vadr + 3, dt);
            break;
        case JOINT_BALL:
            mjint::quatIntegrate(out + padr, v + vadr, dt);
            break;
        default:
            out[padr] += dt * v[vadr];
        }
    }
}

void launch_integrate_pos_mj(const DevicePhysicsModel& model, const DevicePhysicsData& data,
                             double dt, double* qpos_out, hipStream_t stream)
{
    if (data.num_envs <= 0 || model.nq <= 0) return;
    hipLaunchKernelGGL(kernel_integrate_pos_mj, dim3(data.num_envs), 64, 0, stream,
        data.num_envs, model.njnt, model.nq, model.nv,
        model.jnt_type, model.jnt_qpos_adr, model.jnt_dof_adr,
        data.qpos_d, data.qvel_d, dt, qpos_out);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
