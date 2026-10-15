// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// per-joint semi-implicit Euler integration, as a device function.
//
// Shared by kernel_integrate (physics_integrate.hip) and the fused
// integrate+FK+geom+site kernel (physics_kinematics.hip) so both execute the
// exact same expression tree (bit-identical results).  The arithmetic below is
// a verbatim move of the original kernel_integrate body -- do NOT edit it.
#pragma once

#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_types.h"

namespace hipgl {
namespace physics {
namespace kernels {

HIPGL_DEVICE inline void integ_quat_integrate_d(double* q, const double* omega, double dt) {
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

// Integrate joint `jid` of env `env` (caller guarantees jid < njnt).
HIPGL_DEVICE inline void integrate_joint_d(
    int jid, int env, double dt,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_range_d,
    const int*    __restrict__ jnt_limited,
    int nq, int nv,
    bool enable_limits,
    float*        __restrict__ qpos,
    float*        __restrict__ qvel,
    const double* __restrict__ qacc,
    double*       __restrict__ qpos_d,
    double*       __restrict__ qvel_d,
    double free_joint_damp)
{
    int jtype = jnt_type[jid];
    int qa = jnt_qpos_adr[jid];
    int da = jnt_dof_adr[jid];

    float* eq = qpos + env * nq;
    float* ev = qvel + env * nv;
    const double* ea = qacc + env * nv;
    double* eq_d = qpos_d + env * nq;
    double* ev_d = qvel_d + env * nv;

    // GPU stability clamp on every joint velocity; MuJoCo has none. RoboCasa builds pass
    // -DHP_MAX_JOINT_VEL=1e300 (struck objects legitimately spin past 100 rad/s; the clamp made a door
    // strike on the milk carton far gentler than in MuJoCo). The LIBERO build keeps 100.
#ifndef HP_MAX_JOINT_VEL
#define HP_MAX_JOINT_VEL 100.0
#endif
    constexpr double MAX_VEL = HP_MAX_JOINT_VEL;

    if (jtype == JOINT_HINGE || jtype == JOINT_SLIDE) {
        double acc = ea[da];
        if (isnan(acc) || isinf(acc)) acc = 0.0;

        ev_d[da] += dt * acc;
        ev_d[da] = fmin(fmax(ev_d[da], -MAX_VEL), MAX_VEL);
        ev[da] = (float)ev_d[da];

        eq_d[qa] += dt * ev_d[da];
        eq[qa] = (float)eq_d[qa];

        if (enable_limits && jnt_limited[jid]) {
            double lo = jnt_range_d[jid * 2];
            double hi = jnt_range_d[jid * 2 + 1];
            if (lo < hi) {
                double safety = hi - lo + 1.0;
                double q_val = eq_d[qa];
                double v_val = ev_d[da];
                if (q_val < lo - safety) { q_val = lo - safety; if (v_val < 0.0) v_val = 0.0; }
                if (q_val > hi + safety) { q_val = hi + safety; if (v_val > 0.0) v_val = 0.0; }
                eq_d[qa] = q_val; eq[qa] = (float)q_val;
                ev_d[da] = v_val; ev[da] = (float)v_val;
            }
        }
    }
    else if (jtype == JOINT_BALL) {
        for (int k = 0; k < 3; k++) {
            double acc = ea[da+k];
            if (isnan(acc) || isinf(acc)) acc = 0.0;
            ev_d[da+k] += dt * acc;
            ev_d[da+k] = fmin(fmax(ev_d[da+k], -MAX_VEL), MAX_VEL);
            ev[da+k] = (float)ev_d[da+k];
        }
        double omega_d[3] = {ev_d[da], ev_d[da+1], ev_d[da+2]};
        integ_quat_integrate_d(&eq_d[qa], omega_d, dt);
        for (int k = 0; k < 4; k++) eq[qa+k] = (float)eq_d[qa+k];
    }
    else if (jtype == JOINT_FREE) {
        for (int k = 0; k < 6; k++) {
            double acc = ea[da+k];
            if (isnan(acc) || isinf(acc)) acc = 0.0;
            ev_d[da+k] += dt * acc;
            ev_d[da+k] *= free_joint_damp;
            ev_d[da+k] = fmin(fmax(ev_d[da+k], -MAX_VEL), MAX_VEL);
            ev[da+k] = (float)ev_d[da+k];
        }
        for (int k = 0; k < 3; k++)
            eq_d[qa+k] += dt * ev_d[da+k];
        for (int k = 0; k < 3; k++)
            eq[qa+k] = (float)eq_d[qa+k];
        double omega_d[3] = {ev_d[da+3], ev_d[da+4], ev_d[da+5]};
        integ_quat_integrate_d(&eq_d[qa+3], omega_d, dt);
        for (int k = 0; k < 4; k++) eq[qa+3+k] = (float)eq_d[qa+3+k];
    }
}

// ── Fused bias (RNEA) + fluid forces for ONE env, run by BLK threads ──────
// Verbatim body of kernel_bias_fluid_fused (physics_integrate.hip), moved here
// so the fused FK head (physics_kinematics.hip) can run it on its first
// wavefront.  WAVE_SYNC=false: the original __syncthreads(); WAVE_SYNC=true:
// the caller is exactly one wavefront inside a larger block, so the block
// barrier is replaced by a workgroup-scope LDS fence + wave barrier, which is
// what __syncthreads() reduces to for a single-wave workgroup.
template <bool WAVE_SYNC>
HIPGL_DEVICE __forceinline__ void bf_sync() {
    if constexpr (WAVE_SYNC) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    } else {
        __syncthreads();
    }
}

template <int BLK, int MAXB, int MAXJ, int MAXNV, bool WAVE_SYNC>
HIPGL_DEVICE __forceinline__ void bias_fluid_body(
    int env,
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
    __shared__ double s_omega[MAXB*3];
    __shared__ double s_alpha[MAXB*3];
    __shared__ double s_acc  [MAXB*3];   // bias: linear acceleration at body origin
    __shared__ double s_fnet [MAXB*3];
    __shared__ double s_tnet [MAXB*3];
    __shared__ double s_v    [MAXB*3];   // fluid: linear velocity at body origin
    __shared__ double s_ffnet[MAXB*3];
    __shared__ double s_ftnet[MAXB*3];
    __shared__ double s_qb[MAXNV];
    __shared__ double s_qf[MAXNV];

    // Per-env working set, staged once so the level loop touches NO global memory.
    // Without this the serial chain is  level_body_ids[] -> body_parent[] -> parent
    // state, i.e. two DEPENDENT device-memory round trips per tree level, 26 levels
    // deep.  One parallel prefetch replaces 26 serialised ones.
    __shared__ int    s_lvl_start[MAX_TREE_DEPTH + 2];
    __shared__ int    s_lvl_ids[MAXB];
    __shared__ int    s_parent[MAXB];
    __shared__ int    s_jadr[MAXB];
    __shared__ int    s_jnum[MAXB];
    __shared__ int    s_jtype[MAXJ];
    __shared__ int    s_jdof[MAXJ];
    __shared__ double s_xpos[MAXB*3];
    __shared__ double s_xquat[MAXB*4];
    __shared__ double s_xax[MAXJ*3];     // per-env world joint axis (from FK)
    __shared__ double s_jax[MAXJ*3];     // model-local joint axis
    __shared__ double s_qvel[MAXNV];

    constexpr double PI = 3.14159265358979323846;


    const int tid = threadIdx.x;
    const int nb3 = nbody * 3, nb4 = nbody * 4, nj3 = njnt * 3;

    for (int i = tid; i < tree_depth + 2; i += BLK) s_lvl_start[i] = level_start_arr[i];
    for (int i = tid; i < nbody; i += BLK) {
        s_lvl_ids[i] = level_body_ids[i];
        s_parent[i]  = body_parent[i];
        s_jadr[i]    = body_jnt_adr[i];
        s_jnum[i]    = body_jnt_num[i];
    }
    for (int i = tid; i < njnt; i += BLK) { s_jtype[i] = jnt_type[i]; s_jdof[i] = jnt_dof_adr[i]; }
    for (int i = tid; i < nb3; i += BLK) {
        s_xpos[i]  = xpos[env * nb3 + i];
        s_omega[i] = 0; s_alpha[i] = 0; s_acc[i] = 0;
        s_fnet[i]  = 0; s_tnet[i]  = 0;
        s_v[i]     = 0; s_ffnet[i] = 0; s_ftnet[i] = 0;
    }
    for (int i = tid; i < nb4; i += BLK) s_xquat[i] = xquat[env * nb4 + i];
    for (int i = tid; i < nj3; i += BLK) { s_xax[i] = jnt_xaxis_d[env * nj3 + i]; s_jax[i] = jnt_axis[i]; }
    for (int i = tid; i < nv; i += BLK) { s_qvel[i] = qvel[env * nv + i]; s_qb[i] = 0.0; s_qf[i] = 0.0; }
    bf_sync<WAVE_SYNC>();

    const double* ev = s_qvel;

    // ── Merged forward pass: root -> leaf ──────────────────────────────────
    // bias: omega / alpha / linear acceleration.   fluid: linear velocity.
    for (int level = 1; level <= tree_depth; level++) {
        int ls = s_lvl_start[level];
        int count = s_lvl_start[level + 1] - ls;
        for (int local_idx = tid; local_idx < count; local_idx += BLK) {
            int bid = s_lvl_ids[ls + local_idx];
            int p = s_parent[bid];

            double op_x, op_y, op_z, ap_x, ap_y, ap_z, aop_x, aop_y, aop_z;
            double vp_x = 0, vp_y = 0, vp_z = 0;
            if (p <= 0) {
                op_x = op_y = op_z = 0; ap_x = ap_y = ap_z = 0;
                aop_x = -(double)gx; aop_y = -(double)gy; aop_z = -(double)gz;
            } else {
                int p3 = p * 3;
                op_x = s_omega[p3]; op_y = s_omega[p3+1]; op_z = s_omega[p3+2];
                ap_x = s_alpha[p3]; ap_y = s_alpha[p3+1]; ap_z = s_alpha[p3+2];
                aop_x = s_acc[p3];  aop_y = s_acc[p3+1];  aop_z = s_acc[p3+2];
                vp_x = s_v[p3];     vp_y = s_v[p3+1];     vp_z = s_v[p3+2];
            }

            int b3 = bid * 3;
            int p3_pos = p * 3;
            double rx = s_xpos[b3]-s_xpos[p3_pos], ry = s_xpos[b3+1]-s_xpos[p3_pos+1], rz = s_xpos[b3+2]-s_xpos[p3_pos+2];
            double aXr_x = ap_y*rz-ap_z*ry, aXr_y = ap_z*rx-ap_x*rz, aXr_z = ap_x*ry-ap_y*rx;
            double oXr_x = op_y*rz-op_z*ry, oXr_y = op_z*rx-op_x*rz, oXr_z = op_x*ry-op_y*rx;
            double oXoXr_x = op_y*oXr_z-op_z*oXr_y, oXoXr_y = op_z*oXr_x-op_x*oXr_z, oXoXr_z = op_x*oXr_y-op_y*oXr_x;
            double ai_x = aop_x+aXr_x+oXoXr_x, ai_y = aop_y+aXr_y+oXoXr_y, ai_z = aop_z+aXr_z+oXoXr_z;
            double oi_x = op_x, oi_y = op_y, oi_z = op_z;
            double ali_x = ap_x, ali_y = ap_y, ali_z = ap_z;

            // fluid: velocity of this body's origin from parent's origin velocity
            double vx = vp_x + (op_y * rz - op_z * ry);
            double vy = vp_y + (op_z * rx - op_x * rz);
            double vz = vp_z + (op_x * ry - op_y * rx);

            int ja = s_jadr[bid]; int jn = s_jnum[bid];
            for (int k = 0; k < jn; k++) {
                int ji = ja + k; int jtype_l = s_jtype[ji]; int dof = s_jdof[ji];
                if (jtype_l == JOINT_HINGE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    double qdot = ev[dof];
                    double sv_x=aw_x*qdot,sv_y=aw_y*qdot,sv_z=aw_z*qdot;
                    ali_x+=oi_y*sv_z-oi_z*sv_y; ali_y+=oi_z*sv_x-oi_x*sv_z; ali_z+=oi_x*sv_y-oi_y*sv_x;
                    oi_x+=sv_x; oi_y+=sv_y; oi_z+=sv_z;
                } else if (jtype_l == JOINT_FREE) {
                    double wb_x=ev[dof+3],wb_y=ev[dof+4],wb_z=ev[dof+5];
                    int bq4f=bid*4;
                    double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                    double R00=1-2*(fqy*fqy+fqz*fqz),R01=2*(fqx*fqy-fqw*fqz),R02=2*(fqx*fqz+fqw*fqy);
                    double R10=2*(fqx*fqy+fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R12=2*(fqy*fqz-fqw*fqx);
                    double R20=2*(fqx*fqz-fqw*fqy),R21=2*(fqy*fqz+fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                    double wx=R00*wb_x+R01*wb_y+R02*wb_z;
                    double wy=R10*wb_x+R11*wb_y+R12*wb_z;
                    double wz=R20*wb_x+R21*wb_y+R22*wb_z;
                    ali_x+=op_y*wz-op_z*wy; ali_y+=op_z*wx-op_x*wz; ali_z+=op_x*wy-op_y*wx;
                    oi_x+=wx; oi_y+=wy; oi_z+=wz;
                    // fluid: free-joint linear rate adds directly
                    vx += ev[dof]; vy += ev[dof+1]; vz += ev[dof+2];
                } else if (jtype_l == JOINT_SLIDE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    double qdot=ev[dof];
                    ai_x+=2.0*(op_y*(aw_z*qdot)-op_z*(aw_y*qdot)); ai_y+=2.0*(op_z*(aw_x*qdot)-op_x*(aw_z*qdot)); ai_z+=2.0*(op_x*(aw_y*qdot)-op_y*(aw_x*qdot));
                    // fluid: slide rate along the CURRENT-frame axis (rotate local axis by xquat)
                    double rate = ev[dof];
                    int bq4 = bid * 4;
                    double qw = s_xquat[bq4], qx = s_xquat[bq4+1];
                    double qy = s_xquat[bq4+2], qz = s_xquat[bq4+3];
                    double lx = s_jax[ji*3], ly = s_jax[ji*3+1], lz = s_jax[ji*3+2];
                    double tw_x = 2.0*(qy*lz-qz*ly), tw_y = 2.0*(qz*lx-qx*lz), tw_z = 2.0*(qx*ly-qy*lx);
                    double faw_x = lx+qw*tw_x+(qy*tw_z-qz*tw_y);
                    double faw_y = ly+qw*tw_y+(qz*tw_x-qx*tw_z);
                    double faw_z = lz+qw*tw_z+(qx*tw_y-qy*tw_x);
                    vx += rate * faw_x; vy += rate * faw_y; vz += rate * faw_z;
                }
            }
            int o3 = bid*3;
            s_omega[o3]=oi_x; s_omega[o3+1]=oi_y; s_omega[o3+2]=oi_z;
            s_alpha[o3]=ali_x; s_alpha[o3+1]=ali_y; s_alpha[o3+2]=ali_z;
            s_acc[o3]=ai_x; s_acc[o3+1]=ai_y; s_acc[o3+2]=ai_z;
            s_v[o3]=vx; s_v[o3+1]=vy; s_v[o3+2]=vz;
        }
        bf_sync<WAVE_SYNC>();
    }

    // ── Merged per-body wrench phase ───────────────────────────────────────
    for (int bid = 1 + tid; bid < nbody; bid += BLK) {
        int o3 = bid*3;
        int bq4 = bid*4;
        double bqw=s_xquat[bq4],bqx=s_xquat[bq4+1],bqy=s_xquat[bq4+2],bqz=s_xquat[bq4+3];
        double ipx=(double)body_ipos[bid*3],ipy=(double)body_ipos[bid*3+1],ipz=(double)body_ipos[bid*3+2];
        double iqw=(double)body_iquat[bid*4],iqx=(double)body_iquat[bid*4+1],iqy=(double)body_iquat[bid*4+2],iqz=(double)body_iquat[bid*4+3];
        double oi_x=s_omega[o3],oi_y=s_omega[o3+1],oi_z=s_omega[o3+2];

        // --- bias: RNEA body force/torque ---
        {
            double m = (double)body_mass[bid];
            double Ix=(double)body_inertia[bid*3],Iy=(double)body_inertia[bid*3+1],Iz=(double)body_inertia[bid*3+2];
            double tw_x=2.0*(bqy*ipz-bqz*ipy),tw_y=2.0*(bqz*ipx-bqx*ipz),tw_z=2.0*(bqx*ipy-bqy*ipx);
            double rc_x=ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
            double rc_y=ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
            double rc_z=ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);
            double ali_x=s_alpha[o3],ali_y=s_alpha[o3+1],ali_z=s_alpha[o3+2];
            double ao_x=s_acc[o3],ao_y=s_acc[o3+1],ao_z=s_acc[o3+2];
            double aXrc_x=ali_y*rc_z-ali_z*rc_y,aXrc_y=ali_z*rc_x-ali_x*rc_z,aXrc_z=ali_x*rc_y-ali_y*rc_x;
            double oXrc_x=oi_y*rc_z-oi_z*rc_y,oXrc_y=oi_z*rc_x-oi_x*rc_z,oXrc_z=oi_x*rc_y-oi_y*rc_x;
            double oXoXrc_x=oi_y*oXrc_z-oi_z*oXrc_y,oXoXrc_y=oi_z*oXrc_x-oi_x*oXrc_z,oXoXrc_z=oi_x*oXrc_y-oi_y*oXrc_x;
            double ac_x=ao_x+aXrc_x+oXoXrc_x,ac_y=ao_y+aXrc_y+oXoXrc_y,ac_z=ao_z+aXrc_z+oXoXrc_z;
            double fnx=m*ac_x, fny=m*ac_y, fnz=m*ac_z;
            s_fnet[o3]=fnx; s_fnet[o3+1]=fny; s_fnet[o3+2]=fnz;
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
            double rcXf_x=rc_y*fnz-rc_z*fny;
            double rcXf_y=rc_z*fnx-rc_x*fnz;
            double rcXf_z=rc_x*fny-rc_y*fnx;
            s_tnet[o3]=Ia_x+oXIo_x+rcXf_x; s_tnet[o3+1]=Ia_y+oXIo_y+rcXf_y; s_tnet[o3+2]=Ia_z+oXIo_z+rcXf_z;
        }

        // --- fluid: drag wrench at body origin ---
        {
            float mf = (float)body_mass[bid];
            if (mf >= 1e-10f) {
                double wx = oi_x, wy = oi_y, wz = oi_z;
                double v_orig_x = s_v[o3], v_orig_y = s_v[o3+1], v_orig_z = s_v[o3+2];

                double q0w = bqw, q0x = bqx, q0y = bqy, q0z = bqz;
                double ip_x = ipx, ip_y = ipy, ip_z = ipz;

                double t0, t1, t2;
                t0 = 2.0*(q0y*ip_z-q0z*ip_y); t1 = 2.0*(q0z*ip_x-q0x*ip_z); t2 = 2.0*(q0x*ip_y-q0y*ip_x);
                double dx = ip_x+q0w*t0+(q0y*t2-q0z*t1);
                double dy = ip_y+q0w*t1+(q0z*t0-q0x*t2);
                double dz = ip_z+q0w*t2+(q0x*t1-q0y*t0);

                double vx = v_orig_x + (wy*dz - wz*dy);
                double vy = v_orig_y + (wz*dx - wx*dz);
                double vz = v_orig_z + (wx*dy - wy*dx);

                double q1w = iqw, q1x = iqx, q1y = iqy, q1z = iqz;
                double qw = q0w*q1w - q0x*q1x - q0y*q1y - q0z*q1z;
                double qx = q0w*q1x + q0x*q1w + q0y*q1z - q0z*q1y;
                double qy = q0w*q1y - q0x*q1z + q0y*q1w + q0z*q1x;
                double qz = q0w*q1z + q0x*q1y - q0y*q1x + q0z*q1w;

                t0 = 2.0*(qy*wz-qz*wy); t1 = 2.0*(qz*wx-qx*wz); t2 = 2.0*(qx*wy-qy*wx);
                double wl_x = wx - qw*t0 + (qy*t2-qz*t1);
                double wl_y = wy - qw*t1 + (qz*t0-qx*t2);
                double wl_z = wz - qw*t2 + (qx*t1-qy*t0);

                t0 = 2.0*(qy*vz-qz*vy); t1 = 2.0*(qz*vx-qx*vz); t2 = 2.0*(qx*vy-qy*vx);
                double vl_x = vx - qw*t0 + (qy*t2-qz*t1);
                double vl_y = vy - qw*t1 + (qz*t0-qx*t2);
                double vl_z = vz - qw*t2 + (qx*t1-qy*t0);

                double bx = (double)body_inertia_box[bid*3];
                double by = (double)body_inertia_box[bid*3+1];
                double bz = (double)body_inertia_box[bid*3+2];

                double fl_x = 0.0, fl_y = 0.0, fl_z = 0.0;
                double tl_x = 0.0, tl_y = 0.0, tl_z = 0.0;

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

                t0 = 2.0*(qy*fl_z-qz*fl_y); t1 = 2.0*(qz*fl_x-qx*fl_z); t2 = 2.0*(qx*fl_y-qy*fl_x);
                double fw_x = fl_x+qw*t0+(qy*t2-qz*t1);
                double fw_y = fl_y+qw*t1+(qz*t0-qx*t2);
                double fw_z = fl_z+qw*t2+(qx*t1-qy*t0);

                t0 = 2.0*(qy*tl_z-qz*tl_y); t1 = 2.0*(qz*tl_x-qx*tl_z); t2 = 2.0*(qx*tl_y-qy*tl_x);
                double tw_x = tl_x+qw*t0+(qy*t2-qz*t1);
                double tw_y = tl_y+qw*t1+(qz*t0-qx*t2);
                double tw_z = tl_z+qw*t2+(qx*t1-qy*t0);

                tw_x += (dy*fw_z - dz*fw_y);
                tw_y += (dz*fw_x - dx*fw_z);
                tw_z += (dx*fw_y - dy*fw_x);

                s_ffnet[o3] = fw_x; s_ffnet[o3+1] = fw_y; s_ffnet[o3+2] = fw_z;
                s_ftnet[o3] = tw_x; s_ftnet[o3+1] = tw_y; s_ftnet[o3+2] = tw_z;
            }
        }
    }
    bf_sync<WAVE_SYNC>();

    // ── Merged backward pass: leaf -> root ─────────────────────────────────
    for (int level = tree_depth; level >= 1; level--) {
        int ls = s_lvl_start[level];
        int count = s_lvl_start[level + 1] - ls;
        for (int local_idx = tid; local_idx < count; local_idx += BLK) {
            int bid = s_lvl_ids[ls + local_idx];
            if (bid < 1) continue;
            int p = s_parent[bid];
            int o3 = bid*3;
            double fx=s_fnet[o3],fy=s_fnet[o3+1],fz=s_fnet[o3+2];
            double tx=s_tnet[o3],ty=s_tnet[o3+1],tz=s_tnet[o3+2];
            double gfx=s_ffnet[o3],gfy=s_ffnet[o3+1],gfz=s_ffnet[o3+2];
            double gtx=s_ftnet[o3],gty=s_ftnet[o3+1],gtz=s_ftnet[o3+2];

            int ja=s_jadr[bid]; int jn=s_jnum[bid];
            for (int k = 0; k < jn; k++) {
                int ji=ja+k; int jtype_l=s_jtype[ji]; int dof=s_jdof[ji];
                if (jtype_l == JOINT_HINGE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    if (dof < nv) atomicAdd(&s_qb[dof], aw_x*tx+aw_y*ty+aw_z*tz);
                    // fluid hinge axis is recomputed from the LOCAL axis + xquat
                    int bq4 = bid * 4;
                    double qw = s_xquat[bq4], qx_l = s_xquat[bq4+1];
                    double qy_l = s_xquat[bq4+2], qz_l = s_xquat[bq4+3];
                    double lax = s_jax[ji*3], lay = s_jax[ji*3+1], laz = s_jax[ji*3+2];
                    double tw0 = 2.0*(qy_l*laz-qz_l*lay), tw1 = 2.0*(qz_l*lax-qx_l*laz), tw2 = 2.0*(qx_l*lay-qy_l*lax);
                    double faw_x = lax+qw*tw0+(qy_l*tw2-qz_l*tw1);
                    double faw_y = lay+qw*tw1+(qz_l*tw0-qx_l*tw2);
                    double faw_z = laz+qw*tw2+(qx_l*tw1-qy_l*tw0);
                    if (dof < nv) atomicAdd(&s_qf[dof], faw_x*gtx+faw_y*gty+faw_z*gtz);
                } else if (jtype_l == JOINT_FREE) {
                    if (dof+5 < nv) {
                        atomicAdd(&s_qb[dof],fx); atomicAdd(&s_qb[dof+1],fy); atomicAdd(&s_qb[dof+2],fz);
                        atomicAdd(&s_qf[dof],gfx); atomicAdd(&s_qf[dof+1],gfy); atomicAdd(&s_qf[dof+2],gfz);
                        int bq4f=bid*4;
                        double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                        double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                        double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                        double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                        double tb_x=R00*tx+R10*ty+R20*tz;
                        double tb_y=R01*tx+R11*ty+R21*tz;
                        double tb_z=R02*tx+R12*ty+R22*tz;
                        atomicAdd(&s_qb[dof+3],tb_x); atomicAdd(&s_qb[dof+4],tb_y); atomicAdd(&s_qb[dof+5],tb_z);
                        double gb_x=R00*gtx+R10*gty+R20*gtz;
                        double gb_y=R01*gtx+R11*gty+R21*gtz;
                        double gb_z=R02*gtx+R12*gty+R22*gtz;
                        atomicAdd(&s_qf[dof+3],gb_x); atomicAdd(&s_qf[dof+4],gb_y); atomicAdd(&s_qf[dof+5],gb_z);
                    }
                } else if (jtype_l == JOINT_SLIDE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    if (dof < nv) atomicAdd(&s_qb[dof], aw_x*fx+aw_y*fy+aw_z*fz);
                    int bq4 = bid * 4;
                    double qw = s_xquat[bq4], qx_l = s_xquat[bq4+1];
                    double qy_l = s_xquat[bq4+2], qz_l = s_xquat[bq4+3];
                    double lax = s_jax[ji*3], lay = s_jax[ji*3+1], laz = s_jax[ji*3+2];
                    double tw0 = 2.0*(qy_l*laz-qz_l*lay), tw1 = 2.0*(qz_l*lax-qx_l*laz), tw2 = 2.0*(qx_l*lay-qy_l*lax);
                    double faw_x = lax+qw*tw0+(qy_l*tw2-qz_l*tw1);
                    double faw_y = lay+qw*tw1+(qz_l*tw0-qx_l*tw2);
                    double faw_z = laz+qw*tw2+(qx_l*tw1-qy_l*tw0);
                    if (dof < nv) atomicAdd(&s_qf[dof], faw_x*gfx+faw_y*gfy+faw_z*gfz);
                }
            }
            if (p > 0 && p < nbody) {
                int b3=bid*3, pg3=p*3;
                double rx=s_xpos[b3]-s_xpos[pg3],ry=s_xpos[b3+1]-s_xpos[pg3+1],rz=s_xpos[b3+2]-s_xpos[pg3+2];
                int p3 = p*3;
                atomicAdd(&s_fnet[p3],fx); atomicAdd(&s_fnet[p3+1],fy); atomicAdd(&s_fnet[p3+2],fz);
                atomicAdd(&s_tnet[p3],tx+(ry*fz-rz*fy)); atomicAdd(&s_tnet[p3+1],ty+(rz*fx-rx*fz)); atomicAdd(&s_tnet[p3+2],tz+(rx*fy-ry*fx));
                atomicAdd(&s_ffnet[p3],gfx); atomicAdd(&s_ffnet[p3+1],gfy); atomicAdd(&s_ffnet[p3+2],gfz);
                atomicAdd(&s_ftnet[p3],gtx+(ry*gfz-rz*gfy)); atomicAdd(&s_ftnet[p3+1],gty+(rz*gfx-rx*gfz)); atomicAdd(&s_ftnet[p3+2],gtz+(rx*gfy-ry*gfx));
            }
        }
        bf_sync<WAVE_SYNC>();
    }

    for (int i = tid; i < nv; i += BLK) {
        qfrc_bias[env*nv+i]  = s_qb[i];
        qfrc_fluid[env*nv+i] = s_qf[i];
    }
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
