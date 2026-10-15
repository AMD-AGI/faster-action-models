// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Forward kinematics kernel
//
// Computes body world transforms from joint positions (qpos).
// Uses level-parallel BFS: one kernel launch per tree depth level.
// Each thread handles one body at the current level across all envs.
//
// Also computes geom world transforms from body transforms + geom local offsets.

#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/world_mask.h"
#include "hipgl/physics/physics_integrate_dev.h"
#include <cstdlib>

namespace hipgl {
namespace physics {
namespace kernels {

// world skip: per-world freeze mask (see world_mask.h).
HIPGL_WORLD_MASK_DEFINE(set_world_mask_kinematics)

// ── Quaternion helpers (device-side) ────────────────────────────────────────

HIPGL_DEVICE inline void quat_mul(const float* a, const float* b, float* out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

HIPGL_DEVICE inline void quat_rotate(const float* q, const float* v, float* out) {
    float tx = 2.0f * (q[2]*v[2] - q[3]*v[1]);
    float ty = 2.0f * (q[3]*v[0] - q[1]*v[2]);
    float tz = 2.0f * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}

HIPGL_DEVICE inline void quat_normalize(float* q) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-12f) { q[0] = 1; q[1] = q[2] = q[3] = 0; return; }
    float inv = 1.0f / n;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

HIPGL_DEVICE inline void axis_angle_to_quat(const float* axis, float angle, float* q) {
    float ha = angle * 0.5f;
    float s = sinf(ha);
    q[0] = cosf(ha);
    q[1] = axis[0] * s;
    q[2] = axis[1] * s;
    q[3] = axis[2] * s;
}

// ── Forward kinematics kernel (one level at a time) ─────────────────────────
// Grid: (bodies_at_this_level, num_envs)

HIPGL_GLOBAL void kernel_forward_kinematics(
    int level_start,        // first index in level_body_ids for this level
    int level_count,        // number of bodies at this level
    int num_envs,
    // Model (read-only)
    const int*   __restrict__ level_body_ids,
    const int*   __restrict__ body_parent,
    const float* __restrict__ body_pos,       // [nbody*3]
    const float* __restrict__ body_quat,      // [nbody*4]
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*   __restrict__ body_jnt_adr,
    const int*   __restrict__ body_jnt_num,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_qpos_adr,
    const float* __restrict__ jnt_axis,       // [njnt*3]
    const float* __restrict__ jnt_pos,        // [njnt*3] joint anchor offset in body frame
    const float* __restrict__ qpos0,          // [nq] default/reference joint positions
    int nq, int nbody,
    // Data (read qpos, write xpos/xquat)
    const float* __restrict__ qpos,           // [num_envs*nq]
    float*       __restrict__ xpos,           // [num_envs*nbody*3]
    float*       __restrict__ xquat)          // [num_envs*nbody*4]
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (local_idx >= level_count || env >= num_envs) return;
    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    int body_id = level_body_ids[level_start + local_idx];
    int parent  = body_parent[body_id];

    // Parent world transform
    float p_pos[3] = {0, 0, 0};
    float p_quat[4] = {1, 0, 0, 0};
    if (parent >= 0) {
        int pidx3 = env * nbody * 3 + parent * 3;
        int pidx4 = env * nbody * 4 + parent * 4;
        p_pos[0] = xpos[pidx3];     p_pos[1] = xpos[pidx3+1];     p_pos[2] = xpos[pidx3+2];
        p_quat[0] = xquat[pidx4];   p_quat[1] = xquat[pidx4+1];
        p_quat[2] = xquat[pidx4+2]; p_quat[3] = xquat[pidx4+3];
    }

    // Body local offset (from MJCF)
    float local_pos[3], local_quat[4];
    local_pos[0] = body_pos[body_id*3];
    local_pos[1] = body_pos[body_id*3+1];
    local_pos[2] = body_pos[body_id*3+2];
    local_quat[0] = body_quat[body_id*4];
    local_quat[1] = body_quat[body_id*4+1];
    local_quat[2] = body_quat[body_id*4+2];
    local_quat[3] = body_quat[body_id*4+3];

    // Apply joints on this body
    int j_adr = body_jnt_adr[body_id];
    int j_num = body_jnt_num[body_id];
    float jnt_quat[4] = {1, 0, 0, 0};
    float jnt_trans[3] = {0, 0, 0};

    const float* env_qpos = qpos + env * nq;

    for (int ji = 0; ji < j_num; ji++) {
        int j = j_adr + ji;
        int jtype = jnt_type[j];
        int qa = jnt_qpos_adr[j];

        if (jtype == JOINT_HINGE) {
            float angle = env_qpos[qa] - qpos0[qa];
            float ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
            float q_rot[4];
            axis_angle_to_quat(ax, angle, q_rot);
            float tmp[4];
            quat_mul(jnt_quat, q_rot, tmp);
            jnt_quat[0] = tmp[0]; jnt_quat[1] = tmp[1];
            jnt_quat[2] = tmp[2]; jnt_quat[3] = tmp[3];
        }
        else if (jtype == JOINT_SLIDE) {
            float disp = env_qpos[qa] - qpos0[qa];
            float ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
            jnt_trans[0] += ax[0] * disp;
            jnt_trans[1] += ax[1] * disp;
            jnt_trans[2] += ax[2] * disp;
        }
        else if (jtype == JOINT_BALL) {
            float bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
            float tmp[4];
            quat_mul(jnt_quat, bq, tmp);
            jnt_quat[0] = tmp[0]; jnt_quat[1] = tmp[1];
            jnt_quat[2] = tmp[2]; jnt_quat[3] = tmp[3];
        }
        else if (jtype == JOINT_FREE) {
            jnt_trans[0] = env_qpos[qa];
            jnt_trans[1] = env_qpos[qa+1];
            jnt_trans[2] = env_qpos[qa+2];
            jnt_quat[0] = env_qpos[qa+3];
            jnt_quat[1] = env_qpos[qa+4];
            jnt_quat[2] = env_qpos[qa+5];
            jnt_quat[3] = env_qpos[qa+6];
        }
    }

    // Compose: world = parent * local * joint
    // Step 1: local_combined = local * joint
    float comb_quat[4], comb_pos[3];
    quat_mul(local_quat, jnt_quat, comb_quat);

    float rotated_jt[3];
    quat_rotate(local_quat, jnt_trans, rotated_jt);
    comb_pos[0] = local_pos[0] + rotated_jt[0];
    comb_pos[1] = local_pos[1] + rotated_jt[1];
    comb_pos[2] = local_pos[2] + rotated_jt[2];

    // Step 2: world = parent * combined
    float world_quat[4];
    quat_mul(p_quat, comb_quat, world_quat);
    quat_normalize(world_quat);

    float rotated_cp[3];
    quat_rotate(p_quat, comb_pos, rotated_cp);
    float world_pos[3];
    world_pos[0] = p_pos[0] + rotated_cp[0];
    world_pos[1] = p_pos[1] + rotated_cp[1];
    world_pos[2] = p_pos[2] + rotated_cp[2];

    // MuJoCo: xpos = anchor - rotate(jnt_pos, xquat)
    // jnt_pos is the joint pivot offset in the body frame; rotating by the
    // body's world orientation and subtracting gives the body origin.
    if (j_num > 0 && jnt_type[j_adr] != JOINT_FREE) {
        float jp[3] = {jnt_pos[j_adr*3], jnt_pos[j_adr*3+1], jnt_pos[j_adr*3+2]};
        if (jp[0] != 0.0f || jp[1] != 0.0f || jp[2] != 0.0f) {
            float rjp[3];
            quat_rotate(world_quat, jp, rjp);
            world_pos[0] -= rjp[0];
            world_pos[1] -= rjp[1];
            world_pos[2] -= rjp[2];
        }
    }

    // For free joints, the qpos position is in world coordinates directly
    if (j_num > 0 && jnt_type[j_adr] == JOINT_FREE) {
        world_pos[0] = env_qpos[jnt_qpos_adr[j_adr]];
        world_pos[1] = env_qpos[jnt_qpos_adr[j_adr]+1];
        world_pos[2] = env_qpos[jnt_qpos_adr[j_adr]+2];
        float fq[4] = {env_qpos[jnt_qpos_adr[j_adr]+3], env_qpos[jnt_qpos_adr[j_adr]+4],
                        env_qpos[jnt_qpos_adr[j_adr]+5], env_qpos[jnt_qpos_adr[j_adr]+6]};
        world_quat[0] = fq[0]; world_quat[1] = fq[1];
        world_quat[2] = fq[2]; world_quat[3] = fq[3];
        quat_normalize(world_quat);
    }

    // Write output
    int oidx3 = env * nbody * 3 + body_id * 3;
    int oidx4 = env * nbody * 4 + body_id * 4;
    xpos[oidx3]   = world_pos[0]; xpos[oidx3+1] = world_pos[1]; xpos[oidx3+2] = world_pos[2];
    xquat[oidx4]  = world_quat[0]; xquat[oidx4+1] = world_quat[1];
    xquat[oidx4+2]= world_quat[2]; xquat[oidx4+3] = world_quat[3];
}

// ── Geom transform kernel ───────────────────────────────────────────────────
// Compute geom world pos/quat from owning body's world transform + geom local offset.
// Grid: (ngeom, num_envs)

HIPGL_GLOBAL void kernel_geom_transforms(
    int ngeom, int nbody, int num_envs,
    const int*   __restrict__ geom_body,
    const float* __restrict__ geom_pos,       // local [ngeom*3]
    const float* __restrict__ geom_quat,      // local [ngeom*4]
    const float* __restrict__ xpos,           // body world [ne*nb*3]
    const float* __restrict__ xquat,          // body world [ne*nb*4]
    float*       __restrict__ geom_xpos_out,  // [ne*ng*3]
    float*       __restrict__ geom_xquat_out) // [ne*ng*4]
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (gid >= ngeom || env >= num_envs) return;

    int body = geom_body[gid];

    // Body world transform
    int bidx3 = env * nbody * 3 + body * 3;
    int bidx4 = env * nbody * 4 + body * 4;
    float bpos[3] = {xpos[bidx3], xpos[bidx3+1], xpos[bidx3+2]};
    float bquat[4] = {xquat[bidx4], xquat[bidx4+1], xquat[bidx4+2], xquat[bidx4+3]};

    // Geom local offset
    float lpos[3] = {geom_pos[gid*3], geom_pos[gid*3+1], geom_pos[gid*3+2]};
    float lquat[4] = {geom_quat[gid*4], geom_quat[gid*4+1], geom_quat[gid*4+2], geom_quat[gid*4+3]};

    // World = body * local
    float wquat[4];
    quat_mul(bquat, lquat, wquat);
    quat_normalize(wquat);

    float rotated[3];
    quat_rotate(bquat, lpos, rotated);
    float wpos[3] = {bpos[0]+rotated[0], bpos[1]+rotated[1], bpos[2]+rotated[2]};

    int oidx3 = env * ngeom * 3 + gid * 3;
    int oidx4 = env * ngeom * 4 + gid * 4;
    geom_xpos_out[oidx3]   = wpos[0]; geom_xpos_out[oidx3+1] = wpos[1]; geom_xpos_out[oidx3+2] = wpos[2];
    geom_xquat_out[oidx4]  = wquat[0]; geom_xquat_out[oidx4+1] = wquat[1];
    geom_xquat_out[oidx4+2]= wquat[2]; geom_xquat_out[oidx4+3] = wquat[3];
}

// ── Per-item transform bodies (shared by the standalone kernels and the ──────
// fused integrate+FK+geom+site kernel).  Arithmetic is a verbatim move of the
// original kernel bodies; only the source of the parent-body transform differs
// (global memory vs the fused kernel's LDS copy -- same values).

HIPGL_DEVICE inline void site_xform_one(
    int env, int sid, int nsite,
    const float* bpos, const float* bquat,
    const float* __restrict__ site_pos,
    const float* __restrict__ site_quat,
    float*       __restrict__ site_xpos_out,
    float*       __restrict__ site_xquat_out,
    float*       __restrict__ site_xmat_out)
{
    float lpos[3] = {site_pos[sid*3], site_pos[sid*3+1], site_pos[sid*3+2]};
    float lquat[4] = {site_quat[sid*4], site_quat[sid*4+1], site_quat[sid*4+2], site_quat[sid*4+3]};

    float wquat[4];
    quat_mul(bquat, lquat, wquat);
    quat_normalize(wquat);

    float rotated[3];
    quat_rotate(bquat, lpos, rotated);
    float wpos[3] = {bpos[0]+rotated[0], bpos[1]+rotated[1], bpos[2]+rotated[2]};

    int oidx3 = env * nsite * 3 + sid * 3;
    site_xpos_out[oidx3] = wpos[0]; site_xpos_out[oidx3+1] = wpos[1]; site_xpos_out[oidx3+2] = wpos[2];

    int oidx4 = env * nsite * 4 + sid * 4;
    site_xquat_out[oidx4+0] = wquat[0];
    site_xquat_out[oidx4+1] = wquat[1];
    site_xquat_out[oidx4+2] = wquat[2];
    site_xquat_out[oidx4+3] = wquat[3];

    int oidx9 = env * nsite * 9 + sid * 9;
    float qw = wquat[0], qx = wquat[1], qy = wquat[2], qz = wquat[3];
    float xx = qx*qx, yy = qy*qy, zz = qz*qz;
    float xy = qx*qy, xz = qx*qz, yz = qy*qz;
    float wx = qw*qx, wy = qw*qy, wz = qw*qz;
    site_xmat_out[oidx9+0] = 1-2*(yy+zz); site_xmat_out[oidx9+1] = 2*(xy-wz);   site_xmat_out[oidx9+2] = 2*(xz+wy);
    site_xmat_out[oidx9+3] = 2*(xy+wz);   site_xmat_out[oidx9+4] = 1-2*(xx+zz); site_xmat_out[oidx9+5] = 2*(yz-wx);
    site_xmat_out[oidx9+6] = 2*(xz-wy);   site_xmat_out[oidx9+7] = 2*(yz+wx);   site_xmat_out[oidx9+8] = 1-2*(xx+yy);
}

HIPGL_DEVICE inline void geom_xform_aabb_one(
    int env, int gid, int ngeom,
    const float* bpos, const float* bquat,
    const float* __restrict__ geom_pos,
    const float* __restrict__ geom_quat,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    float*       __restrict__ geom_xpos_out,
    float*       __restrict__ geom_xquat_out,
    AABB*        __restrict__ geom_aabb,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert,
    int defer_mesh = 0,
    int v_begin = 0, int v_step = 1, float* partial = nullptr)
{
    float lpos[3] = {geom_pos[gid*3], geom_pos[gid*3+1], geom_pos[gid*3+2]};
    float lquat[4] = {geom_quat[gid*4], geom_quat[gid*4+1], geom_quat[gid*4+2], geom_quat[gid*4+3]};

    float wquat[4]; quat_mul(bquat, lquat, wquat); quat_normalize(wquat);
    float rotated[3]; quat_rotate(bquat, lpos, rotated);
    float px = bpos[0]+rotated[0], py = bpos[1]+rotated[1], pz = bpos[2]+rotated[2];

    int oidx3 = env * ngeom * 3 + gid * 3;
    int oidx4 = env * ngeom * 4 + gid * 4;
    geom_xpos_out[oidx3] = px; geom_xpos_out[oidx3+1] = py; geom_xpos_out[oidx3+2] = pz;
    geom_xquat_out[oidx4] = wquat[0]; geom_xquat_out[oidx4+1] = wquat[1];
    geom_xquat_out[oidx4+2] = wquat[2]; geom_xquat_out[oidx4+3] = wquat[3];

    // AABB computation
    int gtype = geom_type[gid];
    float s0 = geom_size[gid*3], s1 = geom_size[gid*3+1], s2 = geom_size[gid*3+2];

    int idx = env * ngeom + gid;
    AABB& aabb = geom_aabb[idx];

    if (gtype == GEOM_MESH && geom_meshid && mesh_convex_vert) {
        int mid = geom_meshid[gid];
        if (mid >= 0 && mesh_convex_vert_adr && mesh_convex_vert_num) {
            int nv = mesh_convex_vert_num[mid];
            int adr = mesh_convex_vert_adr[mid];
            if (nv > 0) {
                // Deferred: kernel_mesh_aabb (launched right after the head
                // kernel on the same stream) computes this min/max with one
                // wavefront per (geom, world) from the xpos/xquat stored above.
                if (defer_mesh) return;
                float mn_x = 1e20f, mn_y = 1e20f, mn_z = 1e20f;
                float mx_x = -1e20f, mx_y = -1e20f, mx_z = -1e20f;
                for (int vi = v_begin; vi < nv; vi += v_step) {
                    float lv[3] = {mesh_convex_vert[(adr + vi) * 3],
                                   mesh_convex_vert[(adr + vi) * 3 + 1],
                                   mesh_convex_vert[(adr + vi) * 3 + 2]};
                    float wv[3]; quat_rotate(wquat, lv, wv);
                    float wx = px + wv[0], wy = py + wv[1], wz = pz + wv[2];
                    mn_x = fminf(mn_x, wx); mn_y = fminf(mn_y, wy); mn_z = fminf(mn_z, wz);
                    mx_x = fmaxf(mx_x, wx); mx_y = fmaxf(mx_y, wy); mx_z = fmaxf(mx_z, wz);
                }
                if (partial) {
                    partial[0] = mn_x; partial[1] = mn_y; partial[2] = mn_z;
                    partial[3] = mx_x; partial[4] = mx_y; partial[5] = mx_z;
                    return;
                }
                aabb.min_x = mn_x; aabb.min_y = mn_y; aabb.min_z = mn_z;
                aabb.max_x = mx_x; aabb.max_y = mx_y; aabb.max_z = mx_z;
                return;
            }
        }
    }

    // Primitive AABB (sphere, box, capsule, plane, or mesh fallback)
    float hx, hy, hz;
    if (gtype == GEOM_SPHERE) { hx = hy = hz = s0; }
    else if (gtype == GEOM_BOX) { float d = sqrtf(s0*s0+s1*s1+s2*s2); hx = hy = hz = d; }
    else if (gtype == GEOM_PLANE) { hx = hy = hz = 1000.0f; }
    else { hx = hy = hz = s0 + s1; }

    aabb.min_x = px-hx; aabb.min_y = py-hy; aabb.min_z = pz-hz;
    aabb.max_x = px+hx; aabb.max_y = py+hy; aabb.max_z = pz+hz;
}

HIPGL_DEVICE inline void geom_xform_d_one(
    int env, int gid, int ngeom,
    const double* bp, const double* bq,
    const double* __restrict__ geom_pos_d,
    const double* __restrict__ geom_quat_d,
    double*       __restrict__ geom_xpos_d,
    double*       __restrict__ geom_xquat_d,
    double*       __restrict__ geom_xmat_d)
{
    double lp[3]={geom_pos_d[gid*3],geom_pos_d[gid*3+1],geom_pos_d[gid*3+2]};
    double lq[4]={geom_quat_d[gid*4],geom_quat_d[gid*4+1],
                  geom_quat_d[gid*4+2],geom_quat_d[gid*4+3]};
    double wq[4] = {
        bq[0]*lq[0]-bq[1]*lq[1]-bq[2]*lq[2]-bq[3]*lq[3],
        bq[0]*lq[1]+bq[1]*lq[0]+bq[2]*lq[3]-bq[3]*lq[2],
        bq[0]*lq[2]-bq[1]*lq[3]+bq[2]*lq[0]+bq[3]*lq[1],
        bq[0]*lq[3]+bq[1]*lq[2]-bq[2]*lq[1]+bq[3]*lq[0]
    };
    double qn = sqrt(wq[0]*wq[0]+wq[1]*wq[1]+wq[2]*wq[2]+wq[3]*wq[3]);
    if (qn > 1e-15) { double qi=1.0/qn; wq[0]*=qi; wq[1]*=qi; wq[2]*=qi; wq[3]*=qi; }
    double tx=2.0*(bq[2]*lp[2]-bq[3]*lp[1]), ty=2.0*(bq[3]*lp[0]-bq[1]*lp[2]), tz=2.0*(bq[1]*lp[1]-bq[2]*lp[0]);
    double rot[3] = {lp[0]+bq[0]*tx+(bq[2]*tz-bq[3]*ty), lp[1]+bq[0]*ty+(bq[3]*tx-bq[1]*tz), lp[2]+bq[0]*tz+(bq[1]*ty-bq[2]*tx)};
    double wp[3]={bp[0]+rot[0],bp[1]+rot[1],bp[2]+rot[2]};
    int o3=env*ngeom*3+gid*3, o4=env*ngeom*4+gid*4, o9=env*ngeom*9+gid*9;
    geom_xpos_d[o3]=wp[0]; geom_xpos_d[o3+1]=wp[1]; geom_xpos_d[o3+2]=wp[2];
    geom_xquat_d[o4]=wq[0]; geom_xquat_d[o4+1]=wq[1]; geom_xquat_d[o4+2]=wq[2]; geom_xquat_d[o4+3]=wq[3];
    double w=wq[0],x=wq[1],y=wq[2],z=wq[3];
    double xx=x*x,yy=y*y,zz=z*z,xy=x*y,xz=x*z,yz=y*z,wx=w*x,wy=w*y,wz=w*z;
    geom_xmat_d[o9]=1-2*(yy+zz);geom_xmat_d[o9+1]=2*(xy-wz);geom_xmat_d[o9+2]=2*(xz+wy);
    geom_xmat_d[o9+3]=2*(xy+wz);geom_xmat_d[o9+4]=1-2*(xx+zz);geom_xmat_d[o9+5]=2*(yz-wx);
    geom_xmat_d[o9+6]=2*(xz-wy);geom_xmat_d[o9+7]=2*(yz+wx);geom_xmat_d[o9+8]=1-2*(xx+yy);
}

// ── Site transform kernel ───────────────────────────────────────────────────
// Compute site world pos and rotation matrix from body transform + site local offset.
// Grid: (nsite, num_envs)

HIPGL_GLOBAL void kernel_site_transforms(
    int nsite, int nbody, int num_envs,
    const int*   __restrict__ site_body,
    const float* __restrict__ site_pos,
    const float* __restrict__ site_quat,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    float*       __restrict__ site_xpos_out,
    float*       __restrict__ site_xquat_out,
    float*       __restrict__ site_xmat_out)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (sid >= nsite || env >= num_envs) return;

    int body = site_body[sid];
    int bidx3 = env * nbody * 3 + body * 3;
    int bidx4 = env * nbody * 4 + body * 4;
    float bpos[3] = {xpos[bidx3], xpos[bidx3+1], xpos[bidx3+2]};
    float bquat[4] = {xquat[bidx4], xquat[bidx4+1], xquat[bidx4+2], xquat[bidx4+3]};

    site_xform_one(env, sid, nsite, bpos, bquat, site_pos, site_quat,
                   site_xpos_out, site_xquat_out, site_xmat_out);
}

// ── Compute AABB for each geom (for broadphase collision) ───────────────────

HIPGL_GLOBAL void kernel_compute_aabbs(
    int ngeom, int num_envs,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    AABB*        __restrict__ geom_aabb,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (gid >= ngeom || env >= num_envs) return;

    int idx = env * ngeom + gid;
    int pidx3 = env * ngeom * 3 + gid * 3;
    int pidx4 = env * ngeom * 4 + gid * 4;
    float px = geom_xpos[pidx3], py = geom_xpos[pidx3+1], pz = geom_xpos[pidx3+2];
    AABB& aabb = geom_aabb[idx];

    int gtype = geom_type[gid];
    float s0 = geom_size[gid*3], s1 = geom_size[gid*3+1], s2 = geom_size[gid*3+2];

    if (gtype == GEOM_MESH && geom_meshid && mesh_convex_vert) {
        int mid = geom_meshid[gid];
        if (mid >= 0 && mesh_convex_vert_num) {
            int nv = mesh_convex_vert_num[mid];
            int adr = mesh_convex_vert_adr[mid];
            float wq[4] = {geom_xquat[pidx4], geom_xquat[pidx4+1],
                           geom_xquat[pidx4+2], geom_xquat[pidx4+3]};
            if (nv > 0) {
                float mn_x = 1e20f, mn_y = 1e20f, mn_z = 1e20f;
                float mx_x = -1e20f, mx_y = -1e20f, mx_z = -1e20f;
                for (int vi = 0; vi < nv; vi++) {
                    float lv[3] = {mesh_convex_vert[(adr + vi) * 3],
                                   mesh_convex_vert[(adr + vi) * 3 + 1],
                                   mesh_convex_vert[(adr + vi) * 3 + 2]};
                    float wv[3]; quat_rotate(wq, lv, wv);
                    float wx = px + wv[0], wy = py + wv[1], wz = pz + wv[2];
                    mn_x = fminf(mn_x, wx); mn_y = fminf(mn_y, wy); mn_z = fminf(mn_z, wz);
                    mx_x = fmaxf(mx_x, wx); mx_y = fmaxf(mx_y, wy); mx_z = fmaxf(mx_z, wz);
                }
                aabb.min_x = mn_x; aabb.min_y = mn_y; aabb.min_z = mn_z;
                aabb.max_x = mx_x; aabb.max_y = mx_y; aabb.max_z = mx_z;
                return;
            }
        }
    }

    float hx, hy, hz;
    if (gtype == GEOM_SPHERE) {
        hx = hy = hz = s0;
    } else if (gtype == GEOM_BOX) {
        float diag = sqrtf(s0*s0 + s1*s1 + s2*s2);
        hx = hy = hz = diag;
    } else if (gtype == GEOM_PLANE) {
        hx = hy = hz = 1000.0f;
    } else {
        float r = s0, half_len = s1;
        float extent = r + half_len;
        hx = hy = hz = extent;
    }

    aabb.min_x = px - hx; aabb.min_y = py - hy; aabb.min_z = pz - hz;
    aabb.max_x = px + hx; aabb.max_y = py + hy; aabb.max_z = pz + hz;
}

// ── Phase 5: Fused FK kernel (all levels in one launch) ─────────────────────
// Uses device-side __threadfence() + atomics for inter-level synchronization.
// Each env gets one thread block. Each level is processed sequentially within
// the block, with all bodies at that level processed in parallel by threads.

HIPGL_GLOBAL void kernel_forward_kinematics_fused(
    int num_envs, int tree_depth,
    const int*   __restrict__ level_start_arr,   // [tree_depth+2]
    const int*   __restrict__ level_body_ids,
    const int*   __restrict__ body_parent,
    const float* __restrict__ body_pos,
    const float* __restrict__ body_quat,
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*   __restrict__ body_jnt_adr,
    const int*   __restrict__ body_jnt_num,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_qpos_adr,
    const float* __restrict__ jnt_axis,
    const float* __restrict__ jnt_pos,
    const float* __restrict__ qpos0,
    int nq, int nbody,
    const float* __restrict__ qpos,
    float*       __restrict__ xpos,
    float*       __restrict__ xquat)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    for (int level = 0; level <= tree_depth; level++) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;

        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int body_id = level_body_ids[ls + local_idx];
            int parent  = body_parent[body_id];

            float p_pos[3] = {0, 0, 0};
            float p_quat[4] = {1, 0, 0, 0};
            if (parent >= 0) {
                int pidx3 = env * nbody * 3 + parent * 3;
                int pidx4 = env * nbody * 4 + parent * 4;
                p_pos[0] = xpos[pidx3]; p_pos[1] = xpos[pidx3+1]; p_pos[2] = xpos[pidx3+2];
                p_quat[0] = xquat[pidx4]; p_quat[1] = xquat[pidx4+1];
                p_quat[2] = xquat[pidx4+2]; p_quat[3] = xquat[pidx4+3];
            }

            float local_pos[3], local_quat[4];
            local_pos[0] = body_pos[body_id*3];   local_pos[1] = body_pos[body_id*3+1]; local_pos[2] = body_pos[body_id*3+2];
            local_quat[0] = body_quat[body_id*4]; local_quat[1] = body_quat[body_id*4+1];
            local_quat[2] = body_quat[body_id*4+2]; local_quat[3] = body_quat[body_id*4+3];

            int j_adr = body_jnt_adr[body_id];
            int j_num = body_jnt_num[body_id];
            float jnt_quat[4] = {1, 0, 0, 0};
            float jnt_trans[3] = {0, 0, 0};
            const float* env_qpos = qpos + env * nq;

            for (int ji = 0; ji < j_num; ji++) {
                int j = j_adr + ji;
                int jt = jnt_type[j];
                int qa = jnt_qpos_adr[j];

                if (jt == JOINT_HINGE) {
                    float angle = env_qpos[qa] - qpos0[qa];
                    float ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
                    float q_rot[4]; axis_angle_to_quat(ax, angle, q_rot);
                    float tmp[4]; quat_mul(jnt_quat, q_rot, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_SLIDE) {
                    float disp = env_qpos[qa] - qpos0[qa];
                    float ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
                    jnt_trans[0]+=ax[0]*disp; jnt_trans[1]+=ax[1]*disp; jnt_trans[2]+=ax[2]*disp;
                } else if (jt == JOINT_BALL) {
                    float bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
                    float tmp[4]; quat_mul(jnt_quat, bq, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_FREE) {
                    jnt_trans[0]=env_qpos[qa]; jnt_trans[1]=env_qpos[qa+1]; jnt_trans[2]=env_qpos[qa+2];
                    jnt_quat[0]=env_qpos[qa+3]; jnt_quat[1]=env_qpos[qa+4]; jnt_quat[2]=env_qpos[qa+5]; jnt_quat[3]=env_qpos[qa+6];
                }
            }

            float comb_quat[4], comb_pos[3];
            quat_mul(local_quat, jnt_quat, comb_quat);
            float rotated_jt[3]; quat_rotate(local_quat, jnt_trans, rotated_jt);
            comb_pos[0]=local_pos[0]+rotated_jt[0]; comb_pos[1]=local_pos[1]+rotated_jt[1]; comb_pos[2]=local_pos[2]+rotated_jt[2];

            float world_quat[4]; quat_mul(p_quat, comb_quat, world_quat); quat_normalize(world_quat);
            float rotated_cp[3]; quat_rotate(p_quat, comb_pos, rotated_cp);
            float world_pos[3];
            world_pos[0]=p_pos[0]+rotated_cp[0]; world_pos[1]=p_pos[1]+rotated_cp[1]; world_pos[2]=p_pos[2]+rotated_cp[2];

            if (j_num > 0 && jnt_type[j_adr] != JOINT_FREE) {
                float jp[3] = {jnt_pos[j_adr*3], jnt_pos[j_adr*3+1], jnt_pos[j_adr*3+2]};
                if (jp[0] != 0.0f || jp[1] != 0.0f || jp[2] != 0.0f) {
                    float rjp[3]; quat_rotate(world_quat, jp, rjp);
                    world_pos[0]-=rjp[0]; world_pos[1]-=rjp[1]; world_pos[2]-=rjp[2];
                }
            }

            if (j_num > 0 && jnt_type[j_adr] == JOINT_FREE) {
                world_pos[0]=env_qpos[jnt_qpos_adr[j_adr]]; world_pos[1]=env_qpos[jnt_qpos_adr[j_adr]+1]; world_pos[2]=env_qpos[jnt_qpos_adr[j_adr]+2];
                float fq[4]={env_qpos[jnt_qpos_adr[j_adr]+3],env_qpos[jnt_qpos_adr[j_adr]+4],env_qpos[jnt_qpos_adr[j_adr]+5],env_qpos[jnt_qpos_adr[j_adr]+6]};
                world_quat[0]=fq[0]; world_quat[1]=fq[1]; world_quat[2]=fq[2]; world_quat[3]=fq[3];
                quat_normalize(world_quat);
            }

            int oidx3 = env*nbody*3 + body_id*3;
            int oidx4 = env*nbody*4 + body_id*4;
            xpos[oidx3]=world_pos[0]; xpos[oidx3+1]=world_pos[1]; xpos[oidx3+2]=world_pos[2];
            xquat[oidx4]=world_quat[0]; xquat[oidx4+1]=world_quat[1]; xquat[oidx4+2]=world_quat[2]; xquat[oidx4+3]=world_quat[3];
        }

        // Barrier between levels within the same block
        __syncthreads();
    }
}

// ── Fused geom transforms + AABB computation ────────────────────────────────
// Combines kernel_geom_transforms and kernel_compute_aabbs into one kernel
// to save a kernel launch.

HIPGL_GLOBAL void kernel_geom_transforms_and_aabbs(
    int ngeom, int nbody, int num_envs,
    const int*   __restrict__ geom_body,
    const float* __restrict__ geom_pos,
    const float* __restrict__ geom_quat,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    float*       __restrict__ geom_xpos_out,
    float*       __restrict__ geom_xquat_out,
    AABB*        __restrict__ geom_aabb,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (gid >= ngeom || env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;

    int body = geom_body[gid];
    int bidx3 = env * nbody * 3 + body * 3;
    int bidx4 = env * nbody * 4 + body * 4;
    float bpos[3] = {xpos[bidx3], xpos[bidx3+1], xpos[bidx3+2]};
    float bquat[4] = {xquat[bidx4], xquat[bidx4+1], xquat[bidx4+2], xquat[bidx4+3]};

    geom_xform_aabb_one(env, gid, ngeom, bpos, bquat, geom_pos, geom_quat,
                        geom_type, geom_size, geom_xpos_out, geom_xquat_out,
                        geom_aabb, geom_meshid, mesh_convex_vert_adr,
                        mesh_convex_vert_num, mesh_convex_vert);
}

// ── Host launch wrappers ────────────────────────────────────────────────────

// Per-world fixed-body poses (PhysicsModel::set_fixed_body_poses_env).
static inline bool fk_bpose_env(const DevicePhysicsModel& m, int num_envs) {
    return m.body_pos_env_d != nullptr && m.body_pose_nenv >= num_envs;
}

void launch_geom_transforms_and_aabbs(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (model.ngeom <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.ngeom + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_geom_transforms_and_aabbs, grid, block, 0, stream,
        model.ngeom, model.nbody, data.num_envs,
        model.geom_body, model.geom_pos, model.geom_quat,
        model.geom_type, model.geom_size,
        data.xpos, data.xquat,
        data.geom_xpos, data.geom_xquat,
        data.geom_aabb,
        model.geom_meshid,
        model.mesh_convex_vert_adr,
        model.mesh_convex_vert_num,
        model.mesh_convex_vert);
}

void launch_forward_kinematics(
    int level_start, int level_count, int num_envs,
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (level_count <= 0 || num_envs <= 0) return;
    const int block = 256;
    int grid_x = (level_count + block - 1) / block;
    dim3 grid(grid_x, num_envs);
    hipLaunchKernelGGL(kernel_forward_kinematics, grid, block, 0, stream,
        level_start, level_count, num_envs,
        model.level_body_ids, model.body_parent,
        fk_bpose_env(model, num_envs) ? model.body_pos_env : model.body_pos,
        fk_bpose_env(model, num_envs) ? model.body_quat_env : model.body_quat,
        fk_bpose_env(model, num_envs) ? 1 : 0,
        model.body_jnt_adr, model.body_jnt_num,
        model.jnt_type, model.jnt_qpos_adr,
        model.jnt_axis, model.jnt_pos, model.qpos0,
        model.nq, model.nbody,
        data.qpos, data.xpos, data.xquat);
}

void launch_forward_kinematics_fused(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    int tree_depth,
    hipStream_t stream)
{
    if (data.num_envs <= 0 || model.nbody <= 0) return;
    const int block = 256;
    dim3 grid(data.num_envs);
    hipLaunchKernelGGL(kernel_forward_kinematics_fused, grid, block, 0, stream,
        data.num_envs, tree_depth,
        model.level_start, model.level_body_ids,
        model.body_parent,
        fk_bpose_env(model, data.num_envs) ? model.body_pos_env : model.body_pos,
        fk_bpose_env(model, data.num_envs) ? model.body_quat_env : model.body_quat,
        fk_bpose_env(model, data.num_envs) ? 1 : 0,
        model.body_jnt_adr, model.body_jnt_num,
        model.jnt_type, model.jnt_qpos_adr,
        model.jnt_axis, model.jnt_pos, model.qpos0,
        model.nq, model.nbody,
        data.qpos, data.xpos, data.xquat);
}

void launch_geom_transforms(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (model.ngeom <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.ngeom + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_geom_transforms, grid, block, 0, stream,
        model.ngeom, model.nbody, data.num_envs,
        model.geom_body, model.geom_pos, model.geom_quat,
        data.xpos, data.xquat,
        data.geom_xpos, data.geom_xquat);
}

// Double-precision geom transforms for collision pipeline
HIPGL_GLOBAL void kernel_geom_transforms_d(
    int ngeom, int nbody, int num_envs,
    const int*    __restrict__ geom_body,
    const double* __restrict__ geom_pos_d,
    const double* __restrict__ geom_quat_d,
    const double* __restrict__ xpos_d,
    const double* __restrict__ xquat_d,
    double*       __restrict__ geom_xpos_d,
    double*       __restrict__ geom_xquat_d,
    double*       __restrict__ geom_xmat_d)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = blockIdx.y;
    if (gid >= ngeom || env >= num_envs) return;
    int body = geom_body[gid];
    int b3 = env*nbody*3+body*3, b4 = env*nbody*4+body*4;
    double bp[3]={xpos_d[b3],xpos_d[b3+1],xpos_d[b3+2]};
    double bq[4]={xquat_d[b4],xquat_d[b4+1],xquat_d[b4+2],xquat_d[b4+3]};
    geom_xform_d_one(env, gid, ngeom, bp, bq, geom_pos_d, geom_quat_d,
                     geom_xpos_d, geom_xquat_d, geom_xmat_d);
}

void launch_geom_transforms_d(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (model.ngeom <= 0 || data.num_envs <= 0) return;
    if (!data.geom_xpos_d || !data.geom_xmat_d) return;
    const int block = 256;
    int grid_x = (model.ngeom + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_geom_transforms_d, grid, block, 0, stream,
        model.ngeom, model.nbody, data.num_envs,
        model.geom_body, model.geom_pos_d, model.geom_quat_d,
        data.xpos_d, data.xquat_d,
        data.geom_xpos_d, data.geom_xquat_d, data.geom_xmat_d);
}

void launch_compute_aabbs(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (model.ngeom <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.ngeom + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_compute_aabbs, grid, block, 0, stream,
        model.ngeom, data.num_envs,
        model.geom_type, model.geom_size,
        data.geom_xpos, data.geom_xquat,
        data.geom_aabb,
        model.geom_meshid,
        model.mesh_convex_vert_adr,
        model.mesh_convex_vert_num,
        model.mesh_convex_vert);
}

void launch_site_transforms(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    hipStream_t stream)
{
    if (model.nsite <= 0 || data.num_envs <= 0) return;
    const int block = 256;
    int grid_x = (model.nsite + block - 1) / block;
    dim3 grid(grid_x, data.num_envs);
    hipLaunchKernelGGL(kernel_site_transforms, grid, block, 0, stream,
        model.nsite, model.nbody, data.num_envs,
        model.site_body, model.site_pos, model.site_quat,
        data.xpos, data.xquat,
        data.site_xpos, data.site_xquat, data.site_xmat);
}

// ── Double-precision quaternion helpers ──────────────────────────────────────

HIPGL_DEVICE inline void quat_mul_d(const double* a, const double* b, double* out) {
    out[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    out[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    out[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    out[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

HIPGL_DEVICE inline void quat_rotate_d(const double* q, const double* v, double* out) {
    double tx = 2.0 * (q[2]*v[2] - q[3]*v[1]);
    double ty = 2.0 * (q[3]*v[0] - q[1]*v[2]);
    double tz = 2.0 * (q[1]*v[1] - q[2]*v[0]);
    out[0] = v[0] + q[0]*tx + (q[2]*tz - q[3]*ty);
    out[1] = v[1] + q[0]*ty + (q[3]*tx - q[1]*tz);
    out[2] = v[2] + q[0]*tz + (q[1]*ty - q[2]*tx);
}

HIPGL_DEVICE inline void quat_normalize_d(double* q) {
    double n = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-20) { q[0] = 1; q[1] = q[2] = q[3] = 0; return; }
    double inv = 1.0 / n;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

HIPGL_DEVICE inline void axis_angle_to_quat_d(const double* axis, double angle, double* q) {
    double ha = angle * 0.5;
    double s = sin(ha);
    q[0] = cos(ha);
    q[1] = axis[0] * s;
    q[2] = axis[1] * s;
    q[3] = axis[2] * s;
}

// ── Double-precision fused FK kernel ────────────────────────────────────────

HIPGL_GLOBAL void kernel_forward_kinematics_fused_d(
    int num_envs, int tree_depth,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const double* __restrict__ body_pos,
    const double* __restrict__ body_quat,
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const double* __restrict__ qpos0,
    int nq, int nbody, int njnt,
    const double* __restrict__ qpos,
    double*       __restrict__ xpos,
    double*       __restrict__ xquat,
    double*       __restrict__ jnt_xaxis_d)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    for (int level = 0; level <= tree_depth; level++) {
        int ls = level_start_arr[level];
        int count = level_start_arr[level + 1] - ls;

        for (int local_idx = threadIdx.x; local_idx < count; local_idx += blockDim.x) {
            int body_id = level_body_ids[ls + local_idx];
            int parent  = body_parent[body_id];

            double p_pos[3] = {0, 0, 0};
            double p_quat[4] = {1, 0, 0, 0};
            if (parent >= 0) {
                int pidx3 = env * nbody * 3 + parent * 3;
                int pidx4 = env * nbody * 4 + parent * 4;
                p_pos[0] = xpos[pidx3]; p_pos[1] = xpos[pidx3+1]; p_pos[2] = xpos[pidx3+2];
                p_quat[0] = xquat[pidx4]; p_quat[1] = xquat[pidx4+1];
                p_quat[2] = xquat[pidx4+2]; p_quat[3] = xquat[pidx4+3];
            }

            double local_pos[3], local_quat[4];
            local_pos[0] = body_pos[body_id*3];   local_pos[1] = body_pos[body_id*3+1]; local_pos[2] = body_pos[body_id*3+2];
            local_quat[0] = body_quat[body_id*4]; local_quat[1] = body_quat[body_id*4+1];
            local_quat[2] = body_quat[body_id*4+2]; local_quat[3] = body_quat[body_id*4+3];

            int j_adr = body_jnt_adr[body_id];
            int j_num = body_jnt_num[body_id];
            double jnt_quat[4] = {1, 0, 0, 0};
            double jnt_trans[3] = {0, 0, 0};
            const double* env_qpos = qpos + env * nq;

            for (int ji = 0; ji < j_num; ji++) {
                int j = j_adr + ji;
                int jt = jnt_type[j];
                int qa = jnt_qpos_adr[j];

                // Compute pre-joint world frame: parent * local * jnt_quat_so_far
                // This is the frame BEFORE this joint's own rotation, used for the
                // correct DOF Jacobian column (matching MuJoCo's cdof computation).
                if (jnt_xaxis_d && j < njnt) {
                    double pre_q[4]; quat_mul_d(local_quat, jnt_quat, pre_q);
                    double world_pre[4]; quat_mul_d(p_quat, pre_q, world_pre);
                    double local_ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
                    double ax_w[3]; quat_rotate_d(world_pre, local_ax, ax_w);
                    int ax_idx = env * njnt * 3 + j * 3;
                    jnt_xaxis_d[ax_idx]   = ax_w[0];
                    jnt_xaxis_d[ax_idx+1] = ax_w[1];
                    jnt_xaxis_d[ax_idx+2] = ax_w[2];
                }

                if (jt == JOINT_HINGE) {
                    double angle = env_qpos[qa] - qpos0[qa];
                    double ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
                    double q_rot[4]; axis_angle_to_quat_d(ax, angle, q_rot);
                    double tmp[4]; quat_mul_d(jnt_quat, q_rot, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_SLIDE) {
                    double disp = env_qpos[qa] - (double)qpos0[qa];
                    double ax[3] = {jnt_axis[j*3], jnt_axis[j*3+1], jnt_axis[j*3+2]};
                    jnt_trans[0]+=ax[0]*disp; jnt_trans[1]+=ax[1]*disp; jnt_trans[2]+=ax[2]*disp;
                } else if (jt == JOINT_BALL) {
                    double bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
                    double tmp[4]; quat_mul_d(jnt_quat, bq, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_FREE) {
                    jnt_trans[0]=env_qpos[qa]; jnt_trans[1]=env_qpos[qa+1]; jnt_trans[2]=env_qpos[qa+2];
                    jnt_quat[0]=env_qpos[qa+3]; jnt_quat[1]=env_qpos[qa+4]; jnt_quat[2]=env_qpos[qa+5]; jnt_quat[3]=env_qpos[qa+6];
                }
            }

            double comb_quat[4], comb_pos[3];
            quat_mul_d(local_quat, jnt_quat, comb_quat);
            double rotated_jt[3]; quat_rotate_d(local_quat, jnt_trans, rotated_jt);
            comb_pos[0]=local_pos[0]+rotated_jt[0]; comb_pos[1]=local_pos[1]+rotated_jt[1]; comb_pos[2]=local_pos[2]+rotated_jt[2];

            double world_quat[4]; quat_mul_d(p_quat, comb_quat, world_quat); quat_normalize_d(world_quat);
            double rotated_cp[3]; quat_rotate_d(p_quat, comb_pos, rotated_cp);
            double world_pos[3];
            world_pos[0]=p_pos[0]+rotated_cp[0]; world_pos[1]=p_pos[1]+rotated_cp[1]; world_pos[2]=p_pos[2]+rotated_cp[2];

            if (j_num > 0 && jnt_type[j_adr] != JOINT_FREE) {
                double jp[3] = {jnt_pos[j_adr*3], jnt_pos[j_adr*3+1], jnt_pos[j_adr*3+2]};
                if (jp[0] != 0.0 || jp[1] != 0.0 || jp[2] != 0.0) {
                    double rjp[3]; quat_rotate_d(world_quat, jp, rjp);
                    world_pos[0]-=rjp[0]; world_pos[1]-=rjp[1]; world_pos[2]-=rjp[2];
                }
            }

            if (j_num > 0 && jnt_type[j_adr] == JOINT_FREE) {
                world_pos[0]=env_qpos[jnt_qpos_adr[j_adr]]; world_pos[1]=env_qpos[jnt_qpos_adr[j_adr]+1]; world_pos[2]=env_qpos[jnt_qpos_adr[j_adr]+2];
                double fq[4]={env_qpos[jnt_qpos_adr[j_adr]+3],env_qpos[jnt_qpos_adr[j_adr]+4],env_qpos[jnt_qpos_adr[j_adr]+5],env_qpos[jnt_qpos_adr[j_adr]+6]};
                world_quat[0]=fq[0]; world_quat[1]=fq[1]; world_quat[2]=fq[2]; world_quat[3]=fq[3];
                quat_normalize_d(world_quat);
            }

            int oidx3 = env*nbody*3 + body_id*3;
            int oidx4 = env*nbody*4 + body_id*4;
            xpos[oidx3]=world_pos[0]; xpos[oidx3+1]=world_pos[1]; xpos[oidx3+2]=world_pos[2];
            xquat[oidx4]=world_quat[0]; xquat[oidx4+1]=world_quat[1]; xquat[oidx4+2]=world_quat[2]; xquat[oidx4+3]=world_quat[3];
        }

        __syncthreads();
    }
}

struct FkFuseExtras {
    int     do_integrate;
    double  dt;
    const int*    jnt_dof_adr;
    const double* jnt_range_d;
    const int*    jnt_limited;
    int     nv;
    int     enable_limits;
    float*  qpos_f;
    float*  qvel_f;
    const double* qacc;
    float*  sim_time;
    double* qpos_rw;
    double* qvel_d;
    double  free_joint_damp;
    // collision counters zero-init
    int*    ncon;
    int*    body_pair_ncon;
    int     bp_nbody;
    // constraint-buffer clear (replaces kernel_clear_constraint_buffers):
    // 0 = off, 1 = clear parent/sibling/mu + nconstraint, 2 = also lambda.
    int     clear_cons;
    int*    nconstraint;
    double* constraint_lam;
    int*    friction_parent;
    int*    friction_sibling;
    double* friction_mu;
    // geom transforms (+AABB) float
    int     n_geom_f;
    int     ngeom;
    const int*   geom_body;
    const float* geom_pos;
    const float* geom_quat;
    const int*   geom_type;
    const float* geom_size;
    float*  geom_xpos;
    float*  geom_xquat;
    AABB*   geom_aabb;
    const int*   geom_meshid;
    const int*   mesh_convex_vert_adr;
    const int*   mesh_convex_vert_num;
    const float* mesh_convex_vert;
    // geom transforms double
    int     n_geom_d;
    const double* geom_pos_d;
    const double* geom_quat_d;
    double* geom_xpos_d;
    double* geom_xquat_d;
    double* geom_xmat_d;
    // sites
    int     nsite;
    const int*   site_body;
    const float* site_pos;
    const float* site_quat;
    float*  site_xpos;
    float*  site_xquat;
    float*  site_xmat;
    // bias (RNEA) + fluid forces, run on wavefront 0 after FK (replaces the
    // kernel_bias_fluid_fused dispatch); see bias_fluid_body.
    int     do_bias;
    const int*    bf_jnt_dof_adr;
    double  bf_gx, bf_gy, bf_gz;
    const double* bf_body_mass;
    const double* bf_body_inertia;
    const float*  bf_body_inertia_box;
    const double* bf_body_ipos;
    const double* bf_body_iquat;
    const double* bf_qvel;
    float   bf_density, bf_viscosity;
    double* bf_qfrc_bias;
    double* bf_qfrc_fluid;
    // 1 = the head kernel's geom tail skips the mesh convex-vertex AABB walk;
    // kernel_mesh_aabb (launched right after it) fills those AABBs instead.
    // A runtime flag, not a template/constant: the walk stays in the head
    // kernel's CFG so the surrounding f32 geom/site transforms keep exactly
    // the same instruction selection (FMA contraction) as before.
    int     defer_mesh_aabb;
};

// kernel_bias_fluid_fused's shape (one wavefront per env).
constexpr int FK_BF_BLK = 64, FK_BF_MAXB = 64, FK_BF_MAXJ = 64, FK_BF_MAXNV = 64;

template <int BLK, int MAXB, int MAXJ, int MAXNQ, bool FUSE = false>
HIPGL_GLOBAL void __launch_bounds__(BLK, 1) kernel_fk_fused_lds(
    int num_envs, int tree_depth,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const double* __restrict__ body_pos,
    const double* __restrict__ body_quat,
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const double* __restrict__ qpos0,
    int nq, int nbody, int njnt,
    const double* __restrict__ qpos,
    double*       __restrict__ xpos,
    double*       __restrict__ xquat,
    double*       __restrict__ jnt_xaxis_d,
    float*        __restrict__ xpos_f,
    float*        __restrict__ xquat_f,
    FkFuseExtras  ex = FkFuseExtras{})
{
    // Same LDS-residency transform as kernel_bias_fluid_fused: the level loop's
    // serial chain was  level_body_ids[] -> body_parent[] -> parent xpos/xquat,
    // i.e. dependent DEVICE-memory round trips once per tree level.  Stage the
    // whole per-env working set into LDS once, walk the tree entirely in LDS,
    // then write xpos/xquat back.  BLK is one wavefront so the per-level
    // __syncthreads() folds away.
    __shared__ double s_xpos[MAXB*3];
    __shared__ double s_xquat[MAXB*4];
    __shared__ double s_bpos[MAXB*3];
    __shared__ double s_bquat[MAXB*4];
    __shared__ double s_jax[MAXJ*3];
    __shared__ double s_jpos[MAXJ*3];
    __shared__ double s_qpos[MAXNQ];
    __shared__ double s_qpos0[MAXNQ];
    __shared__ int    s_lvl_start[MAX_TREE_DEPTH + 2];
    __shared__ int    s_lvl_ids[MAXB];
    __shared__ int    s_parent[MAXB];
    __shared__ int    s_jadr[MAXB];
    __shared__ int    s_jnum[MAXB];
    __shared__ int    s_jtype[MAXJ];
    __shared__ int    s_jqadr[MAXJ];

    int env = blockIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;

    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    const int tid = threadIdx.x;
    const int nb3 = nbody * 3, nb4 = nbody * 4, nj3 = njnt * 3;

    const double* qpos_src = qpos;
    if constexpr (FUSE) {
        // (1) collision counters: replaces the two hipMemsetAsync graph nodes
        //     that preceded kernel_broadphase (nothing in between reads them).
        if (ex.ncon && tid == 0) ex.ncon[env] = 0;
        if (ex.body_pair_ncon) {
            const int nbb = ex.bp_nbody * ex.bp_nbody;
            for (int i = tid; i < nbb; i += BLK) ex.body_pair_ncon[env * nbb + i] = 0;
        }
        // (1b) this env's slice of kernel_clear_constraint_buffers (same
        //      stored values; nothing between here and constraint_setup reads
        //      or writes these buffers).
        if (ex.clear_cons) {
            if (tid == 0) ex.nconstraint[env] = 0;
            const size_t base = (size_t)env * MAX_CONSTRAINTS;
            const bool clr_lam = ex.clear_cons > 1;
            for (int i = tid; i < MAX_CONSTRAINTS; i += BLK) {
                if (clr_lam) ex.constraint_lam[base + i] = 0.0;
                ex.friction_parent[base + i]  = -1;
                ex.friction_sibling[base + i] = -1;
                ex.friction_mu[base + i]      = 0.0;
            }
        }
        if (ex.do_integrate) {
            if (tid == 0 && ex.sim_time) ex.sim_time[env] += (float)ex.dt;
            for (int jid = tid; jid < njnt; jid += BLK)
                integrate_joint_d(jid, env, ex.dt, jnt_type, jnt_qpos_adr,
                                  ex.jnt_dof_adr, ex.jnt_range_d, ex.jnt_limited,
                                  nq, ex.nv, ex.enable_limits != 0,
                                  ex.qpos_f, ex.qvel_f, ex.qacc,
                                  ex.qpos_rw, ex.qvel_d, ex.free_joint_damp);
            __syncthreads();
        }
        qpos_src = ex.qpos_rw;
    }

    for (int i = tid; i < tree_depth + 2; i += BLK) s_lvl_start[i] = level_start_arr[i];
    for (int i = tid; i < nbody; i += BLK) {
        s_lvl_ids[i] = level_body_ids[i];
        s_parent[i]  = body_parent[i];
        s_jadr[i]    = body_jnt_adr[i];
        s_jnum[i]    = body_jnt_num[i];
    }
    for (int i = tid; i < njnt; i += BLK) { s_jtype[i] = jnt_type[i]; s_jqadr[i] = jnt_qpos_adr[i]; }
    // seed s_xpos/s_xquat from the current values so bodies absent from the
    // level list (unreachable from the root) keep exactly what they had.
    for (int i = tid; i < nb3; i += BLK) { s_xpos[i] = xpos[env*nb3+i]; s_bpos[i] = body_pos[i]; }
    for (int i = tid; i < nb4; i += BLK) { s_xquat[i] = xquat[env*nb4+i]; s_bquat[i] = body_quat[i]; }
    for (int i = tid; i < nj3; i += BLK) { s_jax[i] = jnt_axis[i]; s_jpos[i] = jnt_pos[i]; }
    for (int i = tid; i < nq; i += BLK) { s_qpos[i] = qpos_src[env*nq+i]; s_qpos0[i] = qpos0[i]; }
    __syncthreads();

    for (int level = 0; level <= tree_depth; level++) {
        int ls = s_lvl_start[level];
        int count = s_lvl_start[level + 1] - ls;

        for (int local_idx = tid; local_idx < count; local_idx += BLK) {
            int body_id = s_lvl_ids[ls + local_idx];
            int parent  = s_parent[body_id];

            double p_pos[3] = {0, 0, 0};
            double p_quat[4] = {1, 0, 0, 0};
            if (parent >= 0) {
                int pidx3 = parent * 3;
                int pidx4 = parent * 4;
                p_pos[0] = s_xpos[pidx3]; p_pos[1] = s_xpos[pidx3+1]; p_pos[2] = s_xpos[pidx3+2];
                p_quat[0] = s_xquat[pidx4]; p_quat[1] = s_xquat[pidx4+1];
                p_quat[2] = s_xquat[pidx4+2]; p_quat[3] = s_xquat[pidx4+3];
            }

            double local_pos[3], local_quat[4];
            local_pos[0] = s_bpos[body_id*3];   local_pos[1] = s_bpos[body_id*3+1]; local_pos[2] = s_bpos[body_id*3+2];
            local_quat[0] = s_bquat[body_id*4]; local_quat[1] = s_bquat[body_id*4+1];
            local_quat[2] = s_bquat[body_id*4+2]; local_quat[3] = s_bquat[body_id*4+3];

            int j_adr = s_jadr[body_id];
            int j_num = s_jnum[body_id];
            double jnt_quat[4] = {1, 0, 0, 0};
            double jnt_trans[3] = {0, 0, 0};
            const double* env_qpos = s_qpos;

            for (int ji = 0; ji < j_num; ji++) {
                int j = j_adr + ji;
                int jt = s_jtype[j];
                int qa = s_jqadr[j];

                // Compute pre-joint world frame: parent * local * jnt_quat_so_far
                // This is the frame BEFORE this joint's own rotation, used for the
                // correct DOF Jacobian column (matching MuJoCo's cdof computation).
                if (jnt_xaxis_d && j < njnt) {
                    double pre_q[4]; quat_mul_d(local_quat, jnt_quat, pre_q);
                    double world_pre[4]; quat_mul_d(p_quat, pre_q, world_pre);
                    double local_ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    double ax_w[3]; quat_rotate_d(world_pre, local_ax, ax_w);
                    int ax_idx = env * njnt * 3 + j * 3;
                    jnt_xaxis_d[ax_idx]   = ax_w[0];
                    jnt_xaxis_d[ax_idx+1] = ax_w[1];
                    jnt_xaxis_d[ax_idx+2] = ax_w[2];
                }

                if (jt == JOINT_HINGE) {
                    double angle = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    double q_rot[4]; axis_angle_to_quat_d(ax, angle, q_rot);
                    double tmp[4]; quat_mul_d(jnt_quat, q_rot, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_SLIDE) {
                    double disp = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    jnt_trans[0]+=ax[0]*disp; jnt_trans[1]+=ax[1]*disp; jnt_trans[2]+=ax[2]*disp;
                } else if (jt == JOINT_BALL) {
                    double bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
                    double tmp[4]; quat_mul_d(jnt_quat, bq, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_FREE) {
                    jnt_trans[0]=env_qpos[qa]; jnt_trans[1]=env_qpos[qa+1]; jnt_trans[2]=env_qpos[qa+2];
                    jnt_quat[0]=env_qpos[qa+3]; jnt_quat[1]=env_qpos[qa+4]; jnt_quat[2]=env_qpos[qa+5]; jnt_quat[3]=env_qpos[qa+6];
                }
            }

            double comb_quat[4], comb_pos[3];
            quat_mul_d(local_quat, jnt_quat, comb_quat);
            double rotated_jt[3]; quat_rotate_d(local_quat, jnt_trans, rotated_jt);
            comb_pos[0]=local_pos[0]+rotated_jt[0]; comb_pos[1]=local_pos[1]+rotated_jt[1]; comb_pos[2]=local_pos[2]+rotated_jt[2];

            double world_quat[4]; quat_mul_d(p_quat, comb_quat, world_quat); quat_normalize_d(world_quat);
            double rotated_cp[3]; quat_rotate_d(p_quat, comb_pos, rotated_cp);
            double world_pos[3];
            world_pos[0]=p_pos[0]+rotated_cp[0]; world_pos[1]=p_pos[1]+rotated_cp[1]; world_pos[2]=p_pos[2]+rotated_cp[2];

            if (j_num > 0 && s_jtype[j_adr] != JOINT_FREE) {
                double jp[3] = {s_jpos[j_adr*3], s_jpos[j_adr*3+1], s_jpos[j_adr*3+2]};
                if (jp[0] != 0.0 || jp[1] != 0.0 || jp[2] != 0.0) {
                    double rjp[3]; quat_rotate_d(world_quat, jp, rjp);
                    world_pos[0]-=rjp[0]; world_pos[1]-=rjp[1]; world_pos[2]-=rjp[2];
                }
            }

            if (j_num > 0 && s_jtype[j_adr] == JOINT_FREE) {
                world_pos[0]=env_qpos[s_jqadr[j_adr]]; world_pos[1]=env_qpos[s_jqadr[j_adr]+1]; world_pos[2]=env_qpos[s_jqadr[j_adr]+2];
                double fq[4]={env_qpos[s_jqadr[j_adr]+3],env_qpos[s_jqadr[j_adr]+4],env_qpos[s_jqadr[j_adr]+5],env_qpos[s_jqadr[j_adr]+6]};
                world_quat[0]=fq[0]; world_quat[1]=fq[1]; world_quat[2]=fq[2]; world_quat[3]=fq[3];
                quat_normalize_d(world_quat);
            }

            int oidx3 = body_id*3;
            int oidx4 = body_id*4;
            s_xpos[oidx3]=world_pos[0]; s_xpos[oidx3+1]=world_pos[1]; s_xpos[oidx3+2]=world_pos[2];
            s_xquat[oidx4]=world_quat[0]; s_xquat[oidx4+1]=world_quat[1]; s_xquat[oidx4+2]=world_quat[2]; s_xquat[oidx4+3]=world_quat[3];
        }

        __syncthreads();
    }

    // Absorb the two kernel_copy_d2f dispatches: this CTA owns exactly env's
    // slice of both arrays and already holds the values in registers.
    for (int i = tid; i < nb3; i += BLK) {
        double v = s_xpos[i];
        xpos[env*nb3+i] = v;
        if (xpos_f) xpos_f[env*nb3+i] = (float)v;
    }
    for (int i = tid; i < nb4; i += BLK) {
        double v = s_xquat[i];
        xquat[env*nb4+i] = v;
        if (xquat_f) xquat_f[env*nb4+i] = (float)v;
    }

    if constexpr (FUSE) {
        // (3b) bias + fluid forces on wavefront 0, concurrently with the
        //      geom/site tail on the other waves.  The barrier publishes this
        //      block's xpos/xquat/jnt_xaxis_d/qvel_d global stores (read back by
        //      bias_fluid_body exactly as the standalone kernel reads them).
        //      Nothing below uses a block barrier, so wave 0 may diverge.
        const bool bias_here = ex.do_bias != 0 && BLK > FK_BF_BLK;
        if (bias_here) {
            __syncthreads();
            if (tid < FK_BF_BLK) {
                bias_fluid_body<FK_BF_BLK, FK_BF_MAXB, FK_BF_MAXJ, FK_BF_MAXNV, true>(
                    env, num_envs, tree_depth, ex.nv, njnt, nbody,
                    level_start_arr, level_body_ids, body_parent,
                    body_jnt_adr, body_jnt_num, jnt_type, ex.bf_jnt_dof_adr, jnt_axis,
                    ex.bf_gx, ex.bf_gy, ex.bf_gz,
                    ex.bf_body_mass, ex.bf_body_inertia, ex.bf_body_inertia_box,
                    ex.bf_body_ipos, ex.bf_body_iquat,
                    xpos, xquat, ex.bf_qvel, jnt_xaxis_d,
                    ex.bf_density, ex.bf_viscosity,
                    ex.bf_qfrc_bias, ex.bf_qfrc_fluid);
                return;
            }
        }
        // (3) geom (float + AABB), geom (double) and site transforms for this
        //     env, reading the parent-body transform from LDS.  The float
        //     variants see (float)s_xpos == the xpos_f value just stored, i.e.
        //     exactly what the standalone kernels load from global memory.
        const int n_f = ex.n_geom_f, n_d = ex.n_geom_d, n_s = ex.nsite;
        const int total = n_f + n_d + n_s;
        const int t0 = bias_here ? tid - FK_BF_BLK : tid;
        const int tstep = bias_here ? BLK - FK_BF_BLK : BLK;
        for (int t = t0; t < total; t += tstep) {
            if (t < n_f) {
                const int gid = t;
                const int body = ex.geom_body[gid];
                float bpos[3] = {(float)s_xpos[body*3], (float)s_xpos[body*3+1], (float)s_xpos[body*3+2]};
                float bquat[4] = {(float)s_xquat[body*4], (float)s_xquat[body*4+1],
                                  (float)s_xquat[body*4+2], (float)s_xquat[body*4+3]};
                geom_xform_aabb_one(env, gid, ex.ngeom, bpos, bquat, ex.geom_pos, ex.geom_quat,
                                    ex.geom_type, ex.geom_size, ex.geom_xpos, ex.geom_xquat,
                                    ex.geom_aabb, ex.geom_meshid, ex.mesh_convex_vert_adr,
                                    ex.mesh_convex_vert_num, ex.mesh_convex_vert);
            } else if (t < n_f + n_d) {
                const int gid = t - n_f;
                const int body = ex.geom_body[gid];
                double bp[3] = {s_xpos[body*3], s_xpos[body*3+1], s_xpos[body*3+2]};
                double bq[4] = {s_xquat[body*4], s_xquat[body*4+1], s_xquat[body*4+2], s_xquat[body*4+3]};
                geom_xform_d_one(env, gid, ex.ngeom, bp, bq, ex.geom_pos_d, ex.geom_quat_d,
                                 ex.geom_xpos_d, ex.geom_xquat_d, ex.geom_xmat_d);
            } else {
                const int sid = t - n_f - n_d;
                const int body = ex.site_body[sid];
                float bpos[3] = {(float)s_xpos[body*3], (float)s_xpos[body*3+1], (float)s_xpos[body*3+2]};
                float bquat[4] = {(float)s_xquat[body*4], (float)s_xquat[body*4+1],
                                  (float)s_xquat[body*4+2], (float)s_xquat[body*4+3]};
                site_xform_one(env, sid, n_s, bpos, bquat, ex.site_pos, ex.site_quat,
                               ex.site_xpos, ex.site_xquat, ex.site_xmat);
            }
        }
    }
}

HIPGL_DEVICE __forceinline__ void head_wait_ge(const int* flag, int v) {
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP) < v)
        __builtin_amdgcn_s_sleep(1);
}

template <int BLK, int MAXB, int MAXJ, int MAXNQ>
HIPGL_GLOBAL void __launch_bounds__(BLK, 1) kernel_fk_head_bias(
    int num_envs, int tree_depth,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const double* __restrict__ body_pos,
    const double* __restrict__ body_quat,
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const double* __restrict__ qpos0,
    int nq, int nbody, int njnt,
    double*       __restrict__ xpos,
    double*       __restrict__ xquat,
    double*       __restrict__ jnt_xaxis_d,
    float*        __restrict__ xpos_f,
    float*        __restrict__ xquat_f,
    FkFuseExtras  ex)
{
    constexpr int WV = FK_BF_BLK;           // wavefront size
    static_assert(BLK >= 4 * WV, "needs >= 4 wavefronts");
    static_assert(MAXB <= WV, "one body per lane per level");
    constexpr double PI = 3.14159265358979323846;

    __shared__ double s_xpos[MAXB*3];
    __shared__ double s_xquat[MAXB*4];
    __shared__ double s_bpos[MAXB*3];
    __shared__ double s_bquat[MAXB*4];
    __shared__ double s_cq[MAXB*4];         // comb_quat per body
    __shared__ double s_cp[MAXB*3];         // comb_pos per body
    __shared__ double s_jpe[MAXB*3];        // jnt_pos correction (0 when not applied)
    __shared__ int    s_fqa[MAXB];          // free-joint qpos adr, -1 if none
    __shared__ double s_preq[MAXJ*4];       // pre-joint local frame per joint
    __shared__ double s_jax[MAXJ*3];
    __shared__ double s_jpos[MAXJ*3];
    __shared__ double s_xax[MAXJ*3];
    __shared__ double s_qpos[MAXNQ];
    __shared__ double s_qpos0[MAXNQ];
    __shared__ int    s_lvl_start[MAX_TREE_DEPTH + 2];
    __shared__ int    s_lvl_ids[MAXB];
    __shared__ int    s_parent[MAXB];
    __shared__ int    s_jadr[MAXB];
    __shared__ int    s_jnum[MAXB];
    __shared__ int    s_jtype[MAXJ];
    __shared__ int    s_jqadr[MAXJ];
    __shared__ int    s_jdof[MAXJ];
    __shared__ int2   s_sched[(MAX_TREE_DEPTH + 1) * WV];   // {body, parent} per level/lane
    __shared__ int    s_fk_lvl;             // last tree level completed by wave 0
    __shared__ int    s_ax_lvl;             // last level whose joint axes are published (wave 2)
    __shared__ int    s_fwd_lvl;            // last level of the forward pass completed (wave 1)
    // bias / fluid state (bias_fluid_body's working set)
    __shared__ double s_omega[MAXB*3];
    __shared__ double s_alpha[MAXB*3];
    __shared__ double s_acc  [MAXB*3];
    __shared__ double s_fnet [MAXB*3];
    __shared__ double s_tnet [MAXB*3];
    __shared__ double s_v    [MAXB*3];
    __shared__ double s_ffnet[MAXB*3];
    __shared__ double s_ftnet[MAXB*3];
    __shared__ double s_rrb[MAXB*3];      // xpos[b]-xpos[parent] (bias chain)
    __shared__ double s_rrf[MAXB*3];      // same, fluid chain (wave 2)
    __shared__ double s_qb[FK_BF_MAXNV];
    __shared__ double s_qf[FK_BF_MAXNV];
    __shared__ double s_qvel[FK_BF_MAXNV];

    int env = blockIdx.x;
    if (env >= num_envs) return;

    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    const int tid = threadIdx.x;
    const int wave = tid / WV;
    const int lane = tid - wave * WV;
    const int nb3 = nbody * 3, nb4 = nbody * 4, nj3 = njnt * 3;
    const int nv = ex.nv;
    const double gx = ex.bf_gx, gy = ex.bf_gy, gz = ex.bf_gz;
    const double* body_mass = ex.bf_body_mass;
    const double* body_inertia = ex.bf_body_inertia;
    const float*  body_inertia_box = ex.bf_body_inertia_box;
    const double* body_ipos = ex.bf_body_ipos;
    const double* body_iquat = ex.bf_body_iquat;
    const float density = ex.bf_density, viscosity = ex.bf_viscosity;

    // (0) static model data -> LDS (independent of integrate; issued first so
    //     its latency overlaps the integrate chain).
    for (int i = tid; i < tree_depth + 2; i += BLK) s_lvl_start[i] = level_start_arr[i];
    for (int i = tid; i < nbody; i += BLK) {
        s_lvl_ids[i] = level_body_ids[i];
        s_parent[i]  = body_parent[i];
        s_jadr[i]    = body_jnt_adr[i];
        s_jnum[i]    = body_jnt_num[i];
    }
    for (int i = tid; i < njnt; i += BLK) {
        s_jtype[i] = jnt_type[i]; s_jqadr[i] = jnt_qpos_adr[i]; s_jdof[i] = ex.bf_jnt_dof_adr[i];
    }
    for (int i = tid; i < nb3; i += BLK) {
        s_xpos[i] = xpos[env*nb3+i]; s_bpos[i] = body_pos[i];
        s_omega[i] = 0; s_alpha[i] = 0; s_acc[i] = 0;
        s_fnet[i]  = 0; s_tnet[i]  = 0;
        s_v[i]     = 0; s_ffnet[i] = 0; s_ftnet[i] = 0;
    }
    for (int i = tid; i < nb4; i += BLK) { s_xquat[i] = xquat[env*nb4+i]; s_bquat[i] = body_quat[i]; }
    for (int i = tid; i < nj3; i += BLK) {
        s_jax[i] = jnt_axis[i]; s_jpos[i] = jnt_pos[i]; s_xax[i] = jnt_xaxis_d[env*nj3+i];
    }
    for (int i = tid; i < nq; i += BLK) s_qpos0[i] = qpos0[i];
    if (tid == 0) { s_fk_lvl = -1; s_ax_lvl = -1; s_fwd_lvl = 0; }

    // (1) collision counters (the constraint clear itself runs later, on the
    //     helper waves, overlapping the tree walk).
    if (ex.ncon && tid == 0) ex.ncon[env] = 0;
    if (ex.body_pair_ncon) {
        const int nbb = ex.bp_nbody * ex.bp_nbody;
        for (int i = tid; i < nbb; i += BLK) ex.body_pair_ncon[env * nbb + i] = 0;
    }
    if (ex.clear_cons && tid == 0) ex.nconstraint[env] = 0;
    // world skip: a frozen world keeps its state and FK outputs; only its contact /
    // constraint counters (above) are zeroed so later stages see no contacts.
    if (HIPGL_WORLD_FROZEN(env)) return;
    if (ex.do_integrate) {
        if (tid == 0 && ex.sim_time) ex.sim_time[env] += (float)ex.dt;
        for (int jid = tid; jid < njnt; jid += BLK)
            integrate_joint_d(jid, env, ex.dt, jnt_type, jnt_qpos_adr,
                              ex.jnt_dof_adr, ex.jnt_range_d, ex.jnt_limited,
                              nq, ex.nv, ex.enable_limits != 0,
                              ex.qpos_f, ex.qvel_f, ex.qacc,
                              ex.qpos_rw, ex.qvel_d, ex.free_joint_damp);
        __syncthreads();
    }
    const double* qpos_src = ex.qpos_rw;
    for (int i = tid; i < nq; i += BLK) s_qpos[i] = qpos_src[env*nq+i];
    for (int i = tid; i < nv; i += BLK) { s_qvel[i] = ex.bf_qvel[env*nv+i]; s_qb[i] = 0.0; s_qf[i] = 0.0; }
    __syncthreads();

    if (wave == 0) {
        // (3a) parent-independent local transforms, one lane per body.
        for (int body_id = lane; body_id < nbody; body_id += WV) {
            double local_pos[3], local_quat[4];
            local_pos[0] = s_bpos[body_id*3];   local_pos[1] = s_bpos[body_id*3+1]; local_pos[2] = s_bpos[body_id*3+2];
            local_quat[0] = s_bquat[body_id*4]; local_quat[1] = s_bquat[body_id*4+1];
            local_quat[2] = s_bquat[body_id*4+2]; local_quat[3] = s_bquat[body_id*4+3];

            int j_adr = s_jadr[body_id];
            int j_num = s_jnum[body_id];
            double jnt_quat[4] = {1, 0, 0, 0};
            double jnt_trans[3] = {0, 0, 0};
            const double* env_qpos = s_qpos;

            for (int ji = 0; ji < j_num; ji++) {
                int j = j_adr + ji;
                int jt = s_jtype[j];
                int qa = s_jqadr[j];

                if (j < njnt) {
                    double pre_q[4]; quat_mul_d(local_quat, jnt_quat, pre_q);
                    s_preq[j*4] = pre_q[0]; s_preq[j*4+1] = pre_q[1];
                    s_preq[j*4+2] = pre_q[2]; s_preq[j*4+3] = pre_q[3];
                }

                if (jt == JOINT_HINGE) {
                    double angle = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    double q_rot[4]; axis_angle_to_quat_d(ax, angle, q_rot);
                    double tmp[4]; quat_mul_d(jnt_quat, q_rot, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_SLIDE) {
                    double disp = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    jnt_trans[0]+=ax[0]*disp; jnt_trans[1]+=ax[1]*disp; jnt_trans[2]+=ax[2]*disp;
                } else if (jt == JOINT_BALL) {
                    double bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
                    double tmp[4]; quat_mul_d(jnt_quat, bq, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_FREE) {
                    jnt_trans[0]=env_qpos[qa]; jnt_trans[1]=env_qpos[qa+1]; jnt_trans[2]=env_qpos[qa+2];
                    jnt_quat[0]=env_qpos[qa+3]; jnt_quat[1]=env_qpos[qa+4]; jnt_quat[2]=env_qpos[qa+5]; jnt_quat[3]=env_qpos[qa+6];
                }
            }

            double comb_quat[4], comb_pos[3];
            quat_mul_d(local_quat, jnt_quat, comb_quat);
            double rotated_jt[3]; quat_rotate_d(local_quat, jnt_trans, rotated_jt);
            comb_pos[0]=local_pos[0]+rotated_jt[0]; comb_pos[1]=local_pos[1]+rotated_jt[1]; comb_pos[2]=local_pos[2]+rotated_jt[2];
            s_cq[body_id*4] = comb_quat[0]; s_cq[body_id*4+1] = comb_quat[1];
            s_cq[body_id*4+2] = comb_quat[2]; s_cq[body_id*4+3] = comb_quat[3];
            s_cp[body_id*3] = comb_pos[0]; s_cp[body_id*3+1] = comb_pos[1]; s_cp[body_id*3+2] = comb_pos[2];
            // jnt_pos correction operand: applied iff first joint is not FREE and
            // jnt_pos != 0; stored as 0 otherwise (the walk tests != 0 exactly
            // as before).
            const bool has_j = j_num > 0;
            const bool is_free = has_j && s_jtype[j_adr] == JOINT_FREE;
            double jp0 = 0.0, jp1 = 0.0, jp2 = 0.0;
            if (has_j && !is_free) { jp0 = s_jpos[j_adr*3]; jp1 = s_jpos[j_adr*3+1]; jp2 = s_jpos[j_adr*3+2]; }
            s_jpe[body_id*3] = jp0; s_jpe[body_id*3+1] = jp1; s_jpe[body_id*3+2] = jp2;
            s_fqa[body_id] = is_free ? s_jqadr[j_adr] : -1;
        }
        // per-level lane schedule (level-major; lane l walks entry l of each level)
        for (int level = 0; level <= tree_depth; level++) {
            int ls = s_lvl_start[level];
            int count = s_lvl_start[level + 1] - ls;
            int2 e = make_int2(-1, -1);
            if (lane < count) { int b = s_lvl_ids[ls + lane]; e = make_int2(b, s_parent[b]); }
            s_sched[level * WV + lane] = e;
        }
        bf_sync<true>();

        // (3b) tree walk: only the parent-dependent compose + normalise.
        int2 cur = s_sched[lane];
        for (int level = 0; level <= tree_depth; level++) {
            const int body_id = cur.x;
            const int parent  = cur.y;
            if (level < tree_depth) cur = s_sched[(level + 1) * WV + lane];
            if (body_id >= 0) {
                double p_pos[3] = {0, 0, 0};
                double p_quat[4] = {1, 0, 0, 0};
                if (parent >= 0) {
                    int pidx3 = parent * 3;
                    int pidx4 = parent * 4;
                    p_pos[0] = s_xpos[pidx3]; p_pos[1] = s_xpos[pidx3+1]; p_pos[2] = s_xpos[pidx3+2];
                    p_quat[0] = s_xquat[pidx4]; p_quat[1] = s_xquat[pidx4+1];
                    p_quat[2] = s_xquat[pidx4+2]; p_quat[3] = s_xquat[pidx4+3];
                }
                double comb_quat[4] = {s_cq[body_id*4], s_cq[body_id*4+1], s_cq[body_id*4+2], s_cq[body_id*4+3]};
                double comb_pos[3] = {s_cp[body_id*3], s_cp[body_id*3+1], s_cp[body_id*3+2]};
                double jp[3] = {s_jpe[body_id*3], s_jpe[body_id*3+1], s_jpe[body_id*3+2]};
                const int fqa = s_fqa[body_id];

                double world_quat[4]; quat_mul_d(p_quat, comb_quat, world_quat); quat_normalize_d(world_quat);
                double rotated_cp[3]; quat_rotate_d(p_quat, comb_pos, rotated_cp);
                double world_pos[3];
                world_pos[0]=p_pos[0]+rotated_cp[0]; world_pos[1]=p_pos[1]+rotated_cp[1]; world_pos[2]=p_pos[2]+rotated_cp[2];

                if (jp[0] != 0.0 || jp[1] != 0.0 || jp[2] != 0.0) {
                    double rjp[3]; quat_rotate_d(world_quat, jp, rjp);
                    world_pos[0]-=rjp[0]; world_pos[1]-=rjp[1]; world_pos[2]-=rjp[2];
                }

                if (fqa >= 0) {
                    const double* env_qpos = s_qpos;
                    world_pos[0]=env_qpos[fqa]; world_pos[1]=env_qpos[fqa+1]; world_pos[2]=env_qpos[fqa+2];
                    double fq[4]={env_qpos[fqa+3],env_qpos[fqa+4],env_qpos[fqa+5],env_qpos[fqa+6]};
                    world_quat[0]=fq[0]; world_quat[1]=fq[1]; world_quat[2]=fq[2]; world_quat[3]=fq[3];
                    quat_normalize_d(world_quat);
                }

                int oidx3 = body_id*3;
                int oidx4 = body_id*4;
                s_xpos[oidx3]=world_pos[0]; s_xpos[oidx3+1]=world_pos[1]; s_xpos[oidx3+2]=world_pos[2];
                s_xquat[oidx4]=world_quat[0]; s_xquat[oidx4+1]=world_quat[1]; s_xquat[oidx4+2]=world_quat[2]; s_xquat[oidx4+3]=world_quat[3];
            }
            bf_sync<true>();
            if (lane == 0)
                __hip_atomic_store(&s_fk_lvl, level, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        }

        // (4a) RNEA body wrench (bias half of bias_fluid_body), one level
        //      behind the forward pass, then the RNEA backward pass.
        // body-model constants for this lane's wrench, loaded while the
        // forward pass is still running.
        const int wbid = 1 + lane;
        double pl_ip0 = 0, pl_ip1 = 0, pl_ip2 = 0, pl_iq0 = 0, pl_iq1 = 0, pl_iq2 = 0, pl_iq3 = 0, pl_m = 0;
        double pl_I0 = 0, pl_I1 = 0, pl_I2 = 0;
        if (wbid < nbody) {
            pl_ip0 = (double)body_ipos[wbid*3]; pl_ip1 = (double)body_ipos[wbid*3+1]; pl_ip2 = (double)body_ipos[wbid*3+2];
            pl_iq0 = (double)body_iquat[wbid*4]; pl_iq1 = (double)body_iquat[wbid*4+1];
            pl_iq2 = (double)body_iquat[wbid*4+2]; pl_iq3 = (double)body_iquat[wbid*4+3];
            pl_m = (double)body_mass[wbid];
            pl_I0 = (double)body_inertia[wbid*3]; pl_I1 = (double)body_inertia[wbid*3+1]; pl_I2 = (double)body_inertia[wbid*3+2];
        }
        for (int bid = 1 + lane; bid < nbody; bid += WV) {
            int p = s_parent[bid];
            if (p > 0 && p < nbody) {
                int b3=bid*3, pg3=p*3;
                s_rrb[b3]=s_xpos[b3]-s_xpos[pg3]; s_rrb[b3+1]=s_xpos[b3+1]-s_xpos[pg3+1]; s_rrb[b3+2]=s_xpos[b3+2]-s_xpos[pg3+2];
            }
        }
        head_wait_ge(&s_fwd_lvl, tree_depth);
        if (wbid < nbody) {
            const int bid = wbid;
                int o3 = bid*3;
                int bq4 = bid*4;
                double bqw=s_xquat[bq4],bqx=s_xquat[bq4+1],bqy=s_xquat[bq4+2],bqz=s_xquat[bq4+3];
                double ipx=pl_ip0,ipy=pl_ip1,ipz=pl_ip2;
                double iqw=pl_iq0,iqx=pl_iq1,iqy=pl_iq2,iqz=pl_iq3;
                double oi_x=s_omega[o3],oi_y=s_omega[o3+1],oi_z=s_omega[o3+2];

                // --- bias: RNEA body force/torque ---
                {
                    double m = pl_m;
                    double Ix=pl_I0,Iy=pl_I1,Iz=pl_I2;
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
        }
        bf_sync<true>();
        // Backward pass, split so that only the parent accumulation stays
        // on the serial leaf->root chain: a body's net-wrench entry is final
        // once its own level is reached (only the parent's entry changes
        // afterwards) and each dof receives exactly one add, so the joint
        // projections run afterwards in one parallel pass on the same values.
        // The accumulation order on every parent entry is unchanged.
        for (int level = tree_depth; level >= 1; level--) {
            const int2 e = s_sched[level * WV + lane];
            const int bid = e.x;
            if (bid >= 1) {
                int p = e.y;
                int o3 = bid*3;
                double fx=s_fnet[o3],fy=s_fnet[o3+1],fz=s_fnet[o3+2];
                double tx=s_tnet[o3],ty=s_tnet[o3+1],tz=s_tnet[o3+2];
                if (p > 0 && p < nbody) {
                    int b3=bid*3;
                    double rx=s_rrb[b3],ry=s_rrb[b3+1],rz=s_rrb[b3+2];
                    int p3 = p*3;
                    atomicAdd(&s_fnet[p3],fx); atomicAdd(&s_fnet[p3+1],fy); atomicAdd(&s_fnet[p3+2],fz);
                    atomicAdd(&s_tnet[p3],tx+(ry*fz-rz*fy)); atomicAdd(&s_tnet[p3+1],ty+(rz*fx-rx*fz)); atomicAdd(&s_tnet[p3+2],tz+(rx*fy-ry*fx));
                }
            }
            bf_sync<true>();
        }
        for (int bid = 1 + lane; bid < nbody; bid += WV) {
            int o3 = bid*3;
                double fx=s_fnet[o3],fy=s_fnet[o3+1],fz=s_fnet[o3+2];
                double tx=s_tnet[o3],ty=s_tnet[o3+1],tz=s_tnet[o3+2];

                int ja=s_jadr[bid]; int jn=s_jnum[bid];
                for (int k = 0; k < jn; k++) {
                    int ji=ja+k; int jtype_l=s_jtype[ji]; int dof=s_jdof[ji];
                    if (jtype_l == JOINT_HINGE) {
                        int ax_idx = ji*3;
                        double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                        if (dof < nv) atomicAdd(&s_qb[dof], aw_x*tx+aw_y*ty+aw_z*tz);
                    } else if (jtype_l == JOINT_FREE) {
                        if (dof+5 < nv) {
                            atomicAdd(&s_qb[dof],fx); atomicAdd(&s_qb[dof+1],fy); atomicAdd(&s_qb[dof+2],fz);
                            int bq4f=bid*4;
                            double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                            double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                            double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                            double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                            double tb_x=R00*tx+R10*ty+R20*tz;
                            double tb_y=R01*tx+R11*ty+R21*tz;
                            double tb_z=R02*tx+R12*ty+R22*tz;
                            atomicAdd(&s_qb[dof+3],tb_x); atomicAdd(&s_qb[dof+4],tb_y); atomicAdd(&s_qb[dof+5],tb_z);
                        }
                    } else if (jtype_l == JOINT_SLIDE) {
                        int ax_idx = ji*3;
                        double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                        if (dof < nv) atomicAdd(&s_qb[dof], aw_x*fx+aw_y*fy+aw_z*fz);
                    }
                }
        }
        bf_sync<true>();
        for (int i = lane; i < nv; i += WV) ex.bf_qfrc_bias[env*nv+i] = s_qb[i];
        return;
    }

    if (wave == 2) {
        // (3c) world joint axes (jnt_xaxis): need only the parent transform,
        //      so they run concurrently with the walk's current level.
        for (int level = 0; level <= tree_depth; level++) {
            head_wait_ge(&s_fk_lvl, level > 0 ? level - 1 : 0);
            const int2 e = s_sched[level * WV + lane];
            const int bid = e.x;
            if (bid >= 0) {
                const int parent = e.y;
                double p_quat[4] = {1, 0, 0, 0};
                if (parent >= 0) {
                    int pidx4 = parent * 4;
                    p_quat[0] = s_xquat[pidx4]; p_quat[1] = s_xquat[pidx4+1];
                    p_quat[2] = s_xquat[pidx4+2]; p_quat[3] = s_xquat[pidx4+3];
                }
                int j_adr = s_jadr[bid];
                int j_num = s_jnum[bid];
                for (int ji = 0; ji < j_num; ji++) {
                    int j = j_adr + ji;
                    if (j < njnt) {
                        double pre_q[4] = {s_preq[j*4], s_preq[j*4+1], s_preq[j*4+2], s_preq[j*4+3]};
                        double world_pre[4]; quat_mul_d(p_quat, pre_q, world_pre);
                        double local_ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                        double ax_w[3]; quat_rotate_d(world_pre, local_ax, ax_w);
                        int ax_idx = env * njnt * 3 + j * 3;
                        jnt_xaxis_d[ax_idx]   = ax_w[0];
                        jnt_xaxis_d[ax_idx+1] = ax_w[1];
                        jnt_xaxis_d[ax_idx+2] = ax_w[2];
                        s_xax[j*3] = ax_w[0]; s_xax[j*3+1] = ax_w[1]; s_xax[j*3+2] = ax_w[2];
                    }
                }
            }
            bf_sync<true>();
            if (lane == 0)
                __hip_atomic_store(&s_ax_lvl, level, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        // (4b) fluid body wrench (fluid half of bias_fluid_body), one level
        //      behind the forward pass, then the fluid backward pass.
        // body-model constants for this lane's wrench, loaded while the
        // forward pass is still running.
        const int wbid = 1 + lane;
        double pl_ip0 = 0, pl_ip1 = 0, pl_ip2 = 0, pl_iq0 = 0, pl_iq1 = 0, pl_iq2 = 0, pl_iq3 = 0, pl_m = 0;
        double pl_b0 = 0, pl_b1 = 0, pl_b2 = 0;
        if (wbid < nbody) {
            pl_ip0 = (double)body_ipos[wbid*3]; pl_ip1 = (double)body_ipos[wbid*3+1]; pl_ip2 = (double)body_ipos[wbid*3+2];
            pl_iq0 = (double)body_iquat[wbid*4]; pl_iq1 = (double)body_iquat[wbid*4+1];
            pl_iq2 = (double)body_iquat[wbid*4+2]; pl_iq3 = (double)body_iquat[wbid*4+3];
            pl_m = (double)body_mass[wbid];
            pl_b0 = (double)body_inertia_box[wbid*3]; pl_b1 = (double)body_inertia_box[wbid*3+1]; pl_b2 = (double)body_inertia_box[wbid*3+2];
        }
        head_wait_ge(&s_fwd_lvl, tree_depth);
        for (int bid = 1 + lane; bid < nbody; bid += WV) {
            int p = s_parent[bid];
            if (p > 0 && p < nbody) {
                int b3=bid*3, pg3=p*3;
                s_rrf[b3]=s_xpos[b3]-s_xpos[pg3]; s_rrf[b3+1]=s_xpos[b3+1]-s_xpos[pg3+1]; s_rrf[b3+2]=s_xpos[b3+2]-s_xpos[pg3+2];
            }
        }
        if (wbid < nbody) {
            const int bid = wbid;
                int o3 = bid*3;
                int bq4 = bid*4;
                double bqw=s_xquat[bq4],bqx=s_xquat[bq4+1],bqy=s_xquat[bq4+2],bqz=s_xquat[bq4+3];
                double ipx=pl_ip0,ipy=pl_ip1,ipz=pl_ip2;
                double iqw=pl_iq0,iqx=pl_iq1,iqy=pl_iq2,iqz=pl_iq3;
                double oi_x=s_omega[o3],oi_y=s_omega[o3+1],oi_z=s_omega[o3+2];

            // --- fluid: drag wrench at body origin ---
            {
                float mf = (float)pl_m;
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

                    double bx = pl_b0;
                    double by = pl_b1;
                    double bz = pl_b2;

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
        bf_sync<true>();
        // Backward pass, split so that only the parent accumulation stays
        // on the serial leaf->root chain: a body's net-wrench entry is final
        // once its own level is reached (only the parent's entry changes
        // afterwards) and each dof receives exactly one add, so the joint
        // projections run afterwards in one parallel pass on the same values.
        // The accumulation order on every parent entry is unchanged.
        for (int level = tree_depth; level >= 1; level--) {
            const int2 e = s_sched[level * WV + lane];
            const int bid = e.x;
            if (bid >= 1) {
                int p = e.y;
                int o3 = bid*3;
                double gfx=s_ffnet[o3],gfy=s_ffnet[o3+1],gfz=s_ffnet[o3+2];
                double gtx=s_ftnet[o3],gty=s_ftnet[o3+1],gtz=s_ftnet[o3+2];
                if (p > 0 && p < nbody) {
                    int b3=bid*3;
                    double rx=s_rrf[b3],ry=s_rrf[b3+1],rz=s_rrf[b3+2];
                    int p3 = p*3;
                    atomicAdd(&s_ffnet[p3],gfx); atomicAdd(&s_ffnet[p3+1],gfy); atomicAdd(&s_ffnet[p3+2],gfz);
                    atomicAdd(&s_ftnet[p3],gtx+(ry*gfz-rz*gfy)); atomicAdd(&s_ftnet[p3+1],gty+(rz*gfx-rx*gfz)); atomicAdd(&s_ftnet[p3+2],gtz+(rx*gfy-ry*gfx));
                }
            }
            bf_sync<true>();
        }
        for (int bid = 1 + lane; bid < nbody; bid += WV) {
            int o3 = bid*3;
                double gfx=s_ffnet[o3],gfy=s_ffnet[o3+1],gfz=s_ffnet[o3+2];
                double gtx=s_ftnet[o3],gty=s_ftnet[o3+1],gtz=s_ftnet[o3+2];

                int ja=s_jadr[bid]; int jn=s_jnum[bid];
                for (int k = 0; k < jn; k++) {
                    int ji=ja+k; int jtype_l=s_jtype[ji]; int dof=s_jdof[ji];
                    if (jtype_l == JOINT_HINGE) {
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
                            atomicAdd(&s_qf[dof],gfx); atomicAdd(&s_qf[dof+1],gfy); atomicAdd(&s_qf[dof+2],gfz);
                            int bq4f=bid*4;
                            double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                            double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                            double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                            double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                            double gb_x=R00*gtx+R10*gty+R20*gtz;
                            double gb_y=R01*gtx+R11*gty+R21*gtz;
                            double gb_z=R02*gtx+R12*gty+R22*gtz;
                            atomicAdd(&s_qf[dof+3],gb_x); atomicAdd(&s_qf[dof+4],gb_y); atomicAdd(&s_qf[dof+5],gb_z);
                        }
                    } else if (jtype_l == JOINT_SLIDE) {
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
        }
        bf_sync<true>();
        for (int i = lane; i < nv; i += WV) ex.bf_qfrc_fluid[env*nv+i] = s_qf[i];
        return;
    }

    if (wave == 1) {
        // (3d) merged bias/fluid forward pass, one level behind the walk.
        const double* ev = s_qvel;
        for (int level = 1; level <= tree_depth; level++) {
            head_wait_ge(&s_ax_lvl, level);
            head_wait_ge(&s_fk_lvl, level);
            const int bid = s_sched[level * WV + lane].x;
            if (bid >= 0) {
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
            bf_sync<true>();
            if (lane == 0)
                __hip_atomic_store(&s_fwd_lvl, level, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
        }
        return;
    }

    // (5) waves 3+: constraint-buffer clear (overlaps the walk), then the
    //     xpos/xquat writeback (+ the d2f copies) and the geom/site tail.
    const int ht = tid - 3 * WV, hstep = BLK - 3 * WV;
    if (ex.clear_cons) {
        const size_t base = (size_t)env * MAX_CONSTRAINTS;
        const bool clr_lam = ex.clear_cons > 1;
        for (int i = ht; i < MAX_CONSTRAINTS; i += hstep) {
            if (clr_lam) ex.constraint_lam[base + i] = 0.0;
            ex.friction_parent[base + i]  = -1;
            ex.friction_sibling[base + i] = -1;
            ex.friction_mu[base + i]      = 0.0;
        }
    }
    head_wait_ge(&s_fk_lvl, tree_depth);
    for (int i = ht; i < nb3; i += hstep) {
        double v = s_xpos[i];
        xpos[env*nb3+i] = v;
        if (xpos_f) xpos_f[env*nb3+i] = (float)v;
    }
    for (int i = ht; i < nb4; i += hstep) {
        double v = s_xquat[i];
        xquat[env*nb4+i] = v;
        if (xquat_f) xquat_f[env*nb4+i] = (float)v;
    }
    const int n_f = ex.n_geom_f, n_d = ex.n_geom_d, n_s = ex.nsite;
    const int total = n_f + n_d + n_s;
    const int t0 = ht;
    const int tstep = hstep;
    for (int t = t0; t < total; t += tstep) {
        if (t < n_f) {
            const int gid = t;
            const int body = ex.geom_body[gid];
            float bpos[3] = {(float)s_xpos[body*3], (float)s_xpos[body*3+1], (float)s_xpos[body*3+2]};
            float bquat[4] = {(float)s_xquat[body*4], (float)s_xquat[body*4+1],
                              (float)s_xquat[body*4+2], (float)s_xquat[body*4+3]};
            geom_xform_aabb_one(env, gid, ex.ngeom, bpos, bquat, ex.geom_pos, ex.geom_quat,
                                ex.geom_type, ex.geom_size, ex.geom_xpos, ex.geom_xquat,
                                ex.geom_aabb, ex.geom_meshid, ex.mesh_convex_vert_adr,
                                ex.mesh_convex_vert_num, ex.mesh_convex_vert, ex.defer_mesh_aabb);
        } else if (t < n_f + n_d) {
            const int gid = t - n_f;
            const int body = ex.geom_body[gid];
            double bp[3] = {s_xpos[body*3], s_xpos[body*3+1], s_xpos[body*3+2]};
            double bq[4] = {s_xquat[body*4], s_xquat[body*4+1], s_xquat[body*4+2], s_xquat[body*4+3]};
            geom_xform_d_one(env, gid, ex.ngeom, bp, bq, ex.geom_pos_d, ex.geom_quat_d,
                             ex.geom_xpos_d, ex.geom_xquat_d, ex.geom_xmat_d);
        } else {
            const int sid = t - n_f - n_d;
            const int body = ex.site_body[sid];
            float bpos[3] = {(float)s_xpos[body*3], (float)s_xpos[body*3+1], (float)s_xpos[body*3+2]};
            float bquat[4] = {(float)s_xquat[body*4], (float)s_xquat[body*4+1],
                              (float)s_xquat[body*4+2], (float)s_xquat[body*4+3]};
            site_xform_one(env, sid, n_s, bpos, bquat, ex.site_pos, ex.site_quat,
                           ex.site_xpos, ex.site_xquat, ex.site_xmat);
        }
    }
}
// @@END_HEAD_BIAS

// Mesh-geom AABBs deferred by kernel_fk_head_bias's tail.  The head kernel
// dealt one geom per lane, so a lane owning a large convex hull walked all of
// its vertices serially while the rest of the CTA idled: that walk was ~90%
// of the head kernel's duration at N=10.  Here one wavefront owns one
// (geom, world): lanes stride the hull vertices, then a wave min/max tree.
// Each vertex's world position is the same expression on the same float
// operands (geom_xpos/geom_xquat as stored by the head kernel are exactly the
// register values the old loop used), and fminf/fmaxf are exact and
// order-independent, so the AABB is bitwise unchanged.
HIPGL_GLOBAL void __launch_bounds__(64) kernel_mesh_aabb(
    int num_envs, int ngeom,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_xpos,
    const float* __restrict__ geom_xquat,
    AABB*        __restrict__ geom_aabb,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert)
{
    const int gid = blockIdx.x;
    const int env = blockIdx.y;
    if (gid >= ngeom || env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;
    if (geom_type[gid] != GEOM_MESH) return;
    const int mid = geom_meshid[gid];
    if (mid < 0) return;
    const int nv  = mesh_convex_vert_num[mid];
    if (nv <= 0) return;
    const int adr = mesh_convex_vert_adr[mid];
    const int o3 = env * ngeom * 3 + gid * 3;
    const int o4 = env * ngeom * 4 + gid * 4;
    const float px = geom_xpos[o3], py = geom_xpos[o3+1], pz = geom_xpos[o3+2];
    const float wquat[4] = {geom_xquat[o4], geom_xquat[o4+1], geom_xquat[o4+2], geom_xquat[o4+3]};
    float mn_x = 1e20f, mn_y = 1e20f, mn_z = 1e20f;
    float mx_x = -1e20f, mx_y = -1e20f, mx_z = -1e20f;
    for (int vi = threadIdx.x; vi < nv; vi += 64) {
        float lv[3] = {mesh_convex_vert[(adr + vi) * 3],
                       mesh_convex_vert[(adr + vi) * 3 + 1],
                       mesh_convex_vert[(adr + vi) * 3 + 2]};
        float wv[3]; quat_rotate(wquat, lv, wv);
        float wx = px + wv[0], wy = py + wv[1], wz = pz + wv[2];
        mn_x = fminf(mn_x, wx); mn_y = fminf(mn_y, wy); mn_z = fminf(mn_z, wz);
        mx_x = fmaxf(mx_x, wx); mx_y = fmaxf(mx_y, wy); mx_z = fmaxf(mx_z, wz);
    }
    #pragma unroll
    for (int off = 32; off >= 1; off >>= 1) {
        mn_x = fminf(mn_x, __shfl_xor(mn_x, off)); mn_y = fminf(mn_y, __shfl_xor(mn_y, off));
        mn_z = fminf(mn_z, __shfl_xor(mn_z, off));
        mx_x = fmaxf(mx_x, __shfl_xor(mx_x, off)); mx_y = fmaxf(mx_y, __shfl_xor(mx_y, off));
        mx_z = fmaxf(mx_z, __shfl_xor(mx_z, off));
    }
    if (threadIdx.x == 0) {
        AABB& aabb = geom_aabb[env * ngeom + gid];
        aabb.min_x = mn_x; aabb.min_y = mn_y; aabb.min_z = mn_z;
        aabb.max_x = mx_x; aabb.max_y = mx_y; aabb.max_z = mx_z;
    }
}

// ── Double-to-float copy kernel ─────────────────────────────────────────────

HIPGL_GLOBAL void kernel_copy_d2f(int n, const double* __restrict__ src, float* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

// ── Launch wrappers for double FK ───────────────────────────────────────────

void launch_forward_kinematics_fused_d(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    int tree_depth,
    hipStream_t stream)
{
    if (data.num_envs <= 0 || model.nbody <= 0) return;
    constexpr int FK_BLK = 64, FK_MAXB = 64, FK_MAXJ = 64, FK_MAXNQ = 64;
    dim3 grid(data.num_envs);
    if (model.nbody <= FK_MAXB && model.njnt <= FK_MAXJ && model.nq <= FK_MAXNQ
        && tree_depth <= MAX_TREE_DEPTH) {
        hipLaunchKernelGGL((kernel_fk_fused_lds<FK_BLK, FK_MAXB, FK_MAXJ, FK_MAXNQ>),
            grid, FK_BLK, 0, stream,
            data.num_envs, tree_depth,
            model.level_start, model.level_body_ids,
            model.body_parent,
            fk_bpose_env(model, data.num_envs) ? model.body_pos_env_d : model.body_pos_d,
            fk_bpose_env(model, data.num_envs) ? model.body_quat_env_d : model.body_quat_d,
            fk_bpose_env(model, data.num_envs) ? 1 : 0,
            model.body_jnt_adr, model.body_jnt_num,
            model.jnt_type, model.jnt_qpos_adr,
            model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d,
            model.nq, model.nbody, model.njnt,
            data.qpos_d, data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
            data.xpos, data.xquat);

        // d2f copies are folded into the kernel above.
        return;
    } else {
        const int block = 256;
        hipLaunchKernelGGL(kernel_forward_kinematics_fused_d, grid, block, 0, stream,
            data.num_envs, tree_depth,
            model.level_start, model.level_body_ids,
            model.body_parent,
            fk_bpose_env(model, data.num_envs) ? model.body_pos_env_d : model.body_pos_d,
            fk_bpose_env(model, data.num_envs) ? model.body_quat_env_d : model.body_quat_d,
            fk_bpose_env(model, data.num_envs) ? 1 : 0,
            model.body_jnt_adr, model.body_jnt_num,
            model.jnt_type, model.jnt_qpos_adr,
            model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d,
            model.nq, model.nbody, model.njnt,
            data.qpos_d, data.xpos_d, data.xquat_d, data.jnt_xaxis_d);
    }

    // Copy double xpos/xquat to float for collision/rendering
    int n3 = data.num_envs * model.nbody * 3;
    int n4 = data.num_envs * model.nbody * 4;
    const int blk = 256;
    hipLaunchKernelGGL(kernel_copy_d2f, dim3((n3+blk-1)/blk), blk, 0, stream, n3, data.xpos_d, data.xpos);
    hipLaunchKernelGGL(kernel_copy_d2f, dim3((n4+blk-1)/blk), blk, 0, stream, n4, data.xquat_d, data.xquat);
}


namespace r3pre {
template <typename T>
struct BmViewK {
    const T* p; int base;
    HIPGL_DEVICE T operator[](int i) const { return p[i - base]; }
};

template <bool BM_LDS, bool SRC_LOCAL = false>
HIPGL_DEVICE __forceinline__ void build_mass_matrix_body_k(int bm_bx,

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
    double*      __restrict__ M_inv_diag_out,
    int env_override = -1)
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
        const int env0 = env_override >= 0 ? env_override : (int)blockIdx.y;
        if (env0 >= num_envs) return;
        for (int e = threadIdx.x; e < nbody * 4; e += blockDim.x) {
            s_xq[e]  = xquat_g[(SRC_LOCAL ? (size_t)0 : (size_t)env0 * nbody * 4) + e];
            s_biq[e] = body_iquat_g[e];
        }
        for (int e = threadIdx.x; e < nbody * 3; e += blockDim.x) {
            s_xp[e]  = xpos_g[(SRC_LOCAL ? (size_t)0 : (size_t)env0 * nbody * 3) + e];
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
    const int envb = BM_LDS ? (env_override >= 0 ? env_override : (int)blockIdx.y) : 0;
    const BmViewK<double> xquat{BM_LDS ? s_xq : xquat_g, BM_LDS ? envb * nbody * 4 : 0};
    const BmViewK<double> xpos{BM_LDS ? s_xp : xpos_g, BM_LDS ? envb * nbody * 3 : 0};
    const BmViewK<double> body_mass{BM_LDS ? s_bm : body_mass_g, 0};
    const BmViewK<double> body_inertia{BM_LDS ? s_bin : body_inertia_g, 0};
    const BmViewK<double> body_ipos{BM_LDS ? s_bip : body_ipos_g, 0};
    const BmViewK<double> body_iquat{BM_LDS ? s_biq : body_iquat_g, 0};
    const BmViewK<int> jnt_type{BM_LDS ? s_jt : jnt_type_g, 0};
    const BmViewK<int> jnt_body{BM_LDS ? s_jb : jnt_body_g, 0};
    const BmViewK<int> jnt_dof_adr{BM_LDS ? s_jd : jnt_dof_adr_g, 0};
    const BmViewK<int> body_parent{BM_LDS ? s_bp : body_parent_g, 0};
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

    int flat = bm_bx * blockDim.x + threadIdx.x;
    int env = env_override >= 0 ? env_override : (int)blockIdx.y;
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
            int ax_idx = (SRC_LOCAL ? 0 : env * njnt * 3) + ji * 3;
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


template <bool BM_LDS, bool SRC_LOCAL = false>
HIPGL_DEVICE __forceinline__ void build_mass_matrix_body_v2(int bm_bx,

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
    double*      __restrict__ M_inv_diag_out,
    int env_override = -1)
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
        const int env0 = env_override >= 0 ? env_override : (int)blockIdx.y;
        if (env0 >= num_envs) return;
        for (int e = threadIdx.x; e < nbody * 4; e += blockDim.x) {
            s_xq[e]  = xquat_g[(SRC_LOCAL ? (size_t)0 : (size_t)env0 * nbody * 4) + e];
            s_biq[e] = body_iquat_g[e];
        }
        for (int e = threadIdx.x; e < nbody * 3; e += blockDim.x) {
            s_xp[e]  = xpos_g[(SRC_LOCAL ? (size_t)0 : (size_t)env0 * nbody * 3) + e];
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

    const int envb = BM_LDS ? (env_override >= 0 ? env_override : (int)blockIdx.y) : 0;
    const BmViewK<double> xquat{BM_LDS ? s_xq : xquat_g, BM_LDS ? envb * nbody * 4 : 0};
    const BmViewK<double> xpos{BM_LDS ? s_xp : xpos_g, BM_LDS ? envb * nbody * 3 : 0};
    const BmViewK<double> body_mass{BM_LDS ? s_bm : body_mass_g, 0};
    const BmViewK<double> body_inertia{BM_LDS ? s_bin : body_inertia_g, 0};
    const BmViewK<double> body_ipos{BM_LDS ? s_bip : body_ipos_g, 0};
    const BmViewK<double> body_iquat{BM_LDS ? s_biq : body_iquat_g, 0};
    const BmViewK<int> jnt_type{BM_LDS ? s_jt : jnt_type_g, 0};
    const BmViewK<int> jnt_body{BM_LDS ? s_jb : jnt_body_g, 0};
    const BmViewK<int> jnt_dof_adr{BM_LDS ? s_jd : jnt_dof_adr_g, 0};
    const BmViewK<int> body_parent{BM_LDS ? s_bp : body_parent_g, 0};
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

    __shared__ double s_com[64 * 3], s_C[64 * 6];
    __shared__ unsigned long long s_desc[64], s_mok;
    if (threadIdx.x < 64) s_desc[threadIdx.x] = 0ull;
    if (threadIdx.x == 0) s_mok = 0ull;
    __syncthreads();
    {
        const int envp = BM_LDS ? (env_override >= 0 ? env_override : (int)blockIdx.y) : 0;
        for (int bid = 1 + (int)threadIdx.x; bid < nbody; bid += blockDim.x) {
            {
                const unsigned long long a = s_anc[bid];
                unsigned long long r = a;
                while (r) { const int c = __ffsll((long long)r) - 1; r &= r - 1; atomicOr(&s_desc[c], 1ull << bid); }
            }
            double m = (double)body_mass[bid];
            if (m < 1e-12) continue;
            atomicOr(&s_mok, 1ull << bid);
            const int env = envp;
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
            s_com[bid*3] = com_x; s_com[bid*3+1] = com_y; s_com[bid*3+2] = com_z;
            s_C[bid*6+0] = (Ix*R00*R00+Iy*R01*R01+Iz*R02*R02);
            s_C[bid*6+1] = (Ix*R00*R10+Iy*R01*R11+Iz*R02*R12);
            s_C[bid*6+2] = (Ix*R00*R20+Iy*R01*R21+Iz*R02*R22);
            s_C[bid*6+3] = (Ix*R10*R10+Iy*R11*R11+Iz*R12*R12);
            s_C[bid*6+4] = (Ix*R10*R20+Iy*R11*R21+Iz*R12*R22);
            s_C[bid*6+5] = (Ix*R20*R20+Iy*R21*R21+Iz*R22*R22);
        }
    }
    __syncthreads();


    int flat = bm_bx * blockDim.x + threadIdx.x;
    int env = env_override >= 0 ? env_override : (int)blockIdx.y;
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
            int ax_idx = (SRC_LOCAL ? 0 : env * njnt * 3) + ji * 3;
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

    unsigned long long mem = (bi > 0 && bj > 0) ? (s_desc[bi] & s_desc[bj] & s_mok) : 0ull;
    while (mem) {
        const int bid = __ffsll((long long)mem) - 1;
        mem &= mem - 1;
        double m = (double)body_mass[bid];
        double com_x = s_com[bid*3], com_y = s_com[bid*3+1], com_z = s_com[bid*3+2];
        const double C00 = s_C[bid*6+0], C01 = s_C[bid*6+1], C02 = s_C[bid*6+2];
        const double C11 = s_C[bid*6+3], C12 = s_C[bid*6+4], C22 = s_C[bid*6+5];

        int jt_i = jnt_type[ji_i], jt_j = jnt_type[ji_j];

        if (jt_i == JOINT_HINGE && jt_j == JOINT_HINGE) {
            double Iw_aj[3] = {
                C00*aj[0] + C01*aj[1] + C02*aj[2],
                C01*aj[0] + C11*aj[1] + C12*aj[2],
                C02*aj[0] + C12*aj[1] + C22*aj[2]
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
                    C00*wb[0] + C01*wb[1] + C02*wb[2],
                    C01*wb[0] + C11*wb[1] + C12*wb[2],
                    C02*wb[0] + C12*wb[1] + C22*wb[2]
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
                    C00*aj[0] + C01*aj[1] + C02*aj[2],
                    C01*aj[0] + C11*aj[1] + C12*aj[2],
                    C02*aj[0] + C12*aj[1] + C22*aj[2]
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
                    C00*wa[0] + C01*wa[1] + C02*wa[2],
                    C01*wa[0] + C11*wa[1] + C12*wa[2],
                    C02*wa[0] + C12*wa[1] + C22*wa[2]
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
                C00*wj[0] + C01*wj[1] + C02*wj[2],
                C01*wj[0] + C11*wj[1] + C12*wj[2],
                C02*wj[0] + C12*wj[1] + C22*wj[2]
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
                C00*wb[0]+C01*wb[1]+C02*wb[2],
                C01*wb[0]+C11*wb[1]+C12*wb[2],
                C02*wb[0]+C12*wb[1]+C22*wb[2]
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
                C00*aj[0]+C01*aj[1]+C02*aj[2],
                C01*aj[0]+C11*aj[1]+C12*aj[2],
                C02*aj[0]+C12*aj[1]+C22*aj[2]
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
                C00*wb[0]+C01*wb[1]+C02*wb[2],
                C01*wb[0]+C11*wb[1]+C12*wb[2],
                C02*wb[0]+C12*wb[1]+C22*wb[2]
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
                    C00*wf[0]+C01*wf[1]+C02*wf[2],
                    C01*wf[0]+C11*wf[1]+C12*wf[2],
                    C02*wf[0]+C12*wf[1]+C22*wf[2]
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
                    C00*wb[0]+C01*wb[1]+C02*wb[2],
                    C01*wb[0]+C11*wb[1]+C12*wb[2],
                    C02*wb[0]+C12*wb[1]+C22*wb[2]
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


HIPGL_DEVICE __forceinline__ void r3_wsync_k() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
template <int MAXB, int MAXJ, int MAXNV, bool SRC_LOCAL = false>
HIPGL_DEVICE __forceinline__ void bias_rnea_lds_body_k(
    int env, int tid, int tree_depth, int nv, int njnt, int nbody,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    double gx, double gy, double gz,
    const double* __restrict__ body_mass,
    const double* __restrict__ body_inertia,
    const double* __restrict__ body_ipos,
    const double* __restrict__ body_iquat,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const double* __restrict__ qvel,
    const double* __restrict__ jnt_xaxis_d,
    double*       __restrict__ qfrc_bias)
{
    constexpr int BLK = 64;
    __shared__ double s_omega[MAXB*3], s_alpha[MAXB*3], s_acc[MAXB*3];
    __shared__ double s_fnet[MAXB*3], s_tnet[MAXB*3];
    __shared__ double s_qb[MAXNV];
    __shared__ int    s_lvl_start[MAX_TREE_DEPTH + 2];
    __shared__ int    s_lvl_ids[MAXB], s_parent[MAXB], s_jadr[MAXB], s_jnum[MAXB];
    __shared__ int    s_jtype[MAXJ], s_jdof[MAXJ];
    __shared__ double s_xpos[MAXB*3], s_xquat[MAXB*4], s_xax[MAXJ*3], s_qvel[MAXNV];
    const int nb3 = nbody * 3, nb4 = nbody * 4, nj3 = njnt * 3;
    const int se = SRC_LOCAL ? 0 : env;
    for (int i = tid; i < tree_depth + 2; i += BLK) s_lvl_start[i] = level_start_arr[i];
    for (int i = tid; i < nbody; i += BLK) {
        s_lvl_ids[i] = level_body_ids[i]; s_parent[i] = body_parent[i];
        s_jadr[i] = body_jnt_adr[i];      s_jnum[i] = body_jnt_num[i];
    }
    for (int i = tid; i < njnt; i += BLK) { s_jtype[i] = jnt_type[i]; s_jdof[i] = jnt_dof_adr[i]; }
    for (int i = tid; i < nb3; i += BLK) {
        s_xpos[i] = xpos[se * nb3 + i];
        s_omega[i] = 0; s_alpha[i] = 0; s_acc[i] = 0; s_fnet[i] = 0; s_tnet[i] = 0;
    }
    for (int i = tid; i < nb4; i += BLK) s_xquat[i] = xquat[se * nb4 + i];
    for (int i = tid; i < nj3; i += BLK) s_xax[i] = jnt_xaxis_d[se * nj3 + i];
    for (int i = tid; i < nv; i += BLK) { s_qvel[i] = qvel[env * nv + i]; s_qb[i] = 0.0; }
    r3_wsync_k();
    const double* ev = s_qvel;

    for (int level = 1; level <= tree_depth; level++) {
        int ls = s_lvl_start[level];
        int count = s_lvl_start[level + 1] - ls;
        for (int local_idx = tid; local_idx < count; local_idx += BLK) {
            int bid = s_lvl_ids[ls + local_idx];
            int p = s_parent[bid];
            double op_x, op_y, op_z, ap_x, ap_y, ap_z, aop_x, aop_y, aop_z;
            if (p <= 0) {
                op_x = op_y = op_z = 0; ap_x = ap_y = ap_z = 0;
                aop_x = -(double)gx; aop_y = -(double)gy; aop_z = -(double)gz;
            } else {
                int p3 = p * 3;
                op_x = s_omega[p3]; op_y = s_omega[p3+1]; op_z = s_omega[p3+2];
                ap_x = s_alpha[p3]; ap_y = s_alpha[p3+1]; ap_z = s_alpha[p3+2];
                aop_x = s_acc[p3]; aop_y = s_acc[p3+1]; aop_z = s_acc[p3+2];
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
                } else if (jtype_l == JOINT_SLIDE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    double qdot=ev[dof];
                    ai_x+=2.0*(op_y*(aw_z*qdot)-op_z*(aw_y*qdot)); ai_y+=2.0*(op_z*(aw_x*qdot)-op_x*(aw_z*qdot)); ai_z+=2.0*(op_x*(aw_y*qdot)-op_y*(aw_x*qdot));
                }
            }
            int o3 = bid*3;
            s_omega[o3]=oi_x; s_omega[o3+1]=oi_y; s_omega[o3+2]=oi_z;
            s_alpha[o3]=ali_x; s_alpha[o3+1]=ali_y; s_alpha[o3+2]=ali_z;
            s_acc[o3]=ai_x; s_acc[o3+1]=ai_y; s_acc[o3+2]=ai_z;
        }
        r3_wsync_k();
    }

    for (int bid = 1 + tid; bid < nbody; bid += BLK) {
        double m = (double)body_mass[bid];
        double Ix=(double)body_inertia[bid*3],Iy=(double)body_inertia[bid*3+1],Iz=(double)body_inertia[bid*3+2];
        double ipx=(double)body_ipos[bid*3],ipy=(double)body_ipos[bid*3+1],ipz=(double)body_ipos[bid*3+2];
        int bq4=bid*4;
        double bqw=s_xquat[bq4],bqx=s_xquat[bq4+1],bqy=s_xquat[bq4+2],bqz=s_xquat[bq4+3];
        double tw_x=2.0*(bqy*ipz-bqz*ipy),tw_y=2.0*(bqz*ipx-bqx*ipz),tw_z=2.0*(bqx*ipy-bqy*ipx);
        double rc_x=ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
        double rc_y=ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
        double rc_z=ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);
        int o3=bid*3;
        double oi_x=s_omega[o3],oi_y=s_omega[o3+1],oi_z=s_omega[o3+2];
        double ali_x=s_alpha[o3],ali_y=s_alpha[o3+1],ali_z=s_alpha[o3+2];
        double ao_x=s_acc[o3],ao_y=s_acc[o3+1],ao_z=s_acc[o3+2];
        double aXrc_x=ali_y*rc_z-ali_z*rc_y,aXrc_y=ali_z*rc_x-ali_x*rc_z,aXrc_z=ali_x*rc_y-ali_y*rc_x;
        double oXrc_x=oi_y*rc_z-oi_z*rc_y,oXrc_y=oi_z*rc_x-oi_x*rc_z,oXrc_z=oi_x*rc_y-oi_y*rc_x;
        double oXoXrc_x=oi_y*oXrc_z-oi_z*oXrc_y,oXoXrc_y=oi_z*oXrc_x-oi_x*oXrc_z,oXoXrc_z=oi_x*oXrc_y-oi_y*oXrc_x;
        double ac_x=ao_x+aXrc_x+oXoXrc_x,ac_y=ao_y+aXrc_y+oXoXrc_y,ac_z=ao_z+aXrc_z+oXoXrc_z;
        double fnx=m*ac_x, fny=m*ac_y, fnz=m*ac_z;
        s_fnet[o3]=fnx; s_fnet[o3+1]=fny; s_fnet[o3+2]=fnz;
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
        double rcXf_x=rc_y*fnz-rc_z*fny;
        double rcXf_y=rc_z*fnx-rc_x*fnz;
        double rcXf_z=rc_x*fny-rc_y*fnx;
        s_tnet[o3]=Ia_x+oXIo_x+rcXf_x; s_tnet[o3+1]=Ia_y+oXIo_y+rcXf_y; s_tnet[o3+2]=Ia_z+oXIo_z+rcXf_z;
    }
    r3_wsync_k();

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
            int ja=s_jadr[bid]; int jn=s_jnum[bid];
            for (int k = 0; k < jn; k++) {
                int ji=ja+k; int jtype_l=s_jtype[ji]; int dof=s_jdof[ji];
                if (jtype_l == JOINT_HINGE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    if (dof < nv) atomicAdd(&s_qb[dof], aw_x*tx+aw_y*ty+aw_z*tz);
                } else if (jtype_l == JOINT_FREE) {
                    if (dof+5 < nv) {
                        atomicAdd(&s_qb[dof],fx); atomicAdd(&s_qb[dof+1],fy); atomicAdd(&s_qb[dof+2],fz);
                        int bq4f=bid*4;
                        double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                        double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                        double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                        double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                        double tb_x=R00*tx+R10*ty+R20*tz;
                        double tb_y=R01*tx+R11*ty+R21*tz;
                        double tb_z=R02*tx+R12*ty+R22*tz;
                        atomicAdd(&s_qb[dof+3],tb_x); atomicAdd(&s_qb[dof+4],tb_y); atomicAdd(&s_qb[dof+5],tb_z);
                    }
                } else if (jtype_l == JOINT_SLIDE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    if (dof < nv) atomicAdd(&s_qb[dof], aw_x*fx+aw_y*fy+aw_z*fz);
                }
            }
            if (p > 0 && p < nbody) {
                int b3=bid*3, p3=p*3;
                double rx=s_xpos[b3]-s_xpos[p3],ry=s_xpos[b3+1]-s_xpos[p3+1],rz=s_xpos[b3+2]-s_xpos[p3+2];
                atomicAdd(&s_fnet[p3],fx); atomicAdd(&s_fnet[p3+1],fy); atomicAdd(&s_fnet[p3+2],fz);
                atomicAdd(&s_tnet[p3],tx+(ry*fz-rz*fy)); atomicAdd(&s_tnet[p3+1],ty+(rz*fx-rx*fz)); atomicAdd(&s_tnet[p3+2],tz+(rx*fy-ry*fx));
            }
        }
        r3_wsync_k();
    }
    for (int i = tid; i < nv; i += BLK) qfrc_bias[env*nv+i] = s_qb[i];
}


HIPGL_DEVICE __forceinline__ double r7_rl(double v, int lane) {
    const long long b = __double_as_longlong(v);
    const int lo = __builtin_amdgcn_readlane((int)(b & 0xffffffffll), lane);
    const int hi = __builtin_amdgcn_readlane((int)(b >> 32), lane);
    return __longlong_as_double(((long long)hi << 32) | (unsigned int)lo);
}
HIPGL_DEVICE __forceinline__ double r7_opq(double v) { asm volatile("" : "+v"(v)); return v; }
template <int MAXB, int MAXJ, int MAXNV>
HIPGL_DEVICE __forceinline__ void bias_rnea_reg_v3(
    int env, int tid, int tree_depth, int nv, int njnt, int nbody,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    double gx, double gy, double gz,
    const double* __restrict__ body_mass,
    const double* __restrict__ body_inertia,
    const double* __restrict__ body_ipos,
    const double* __restrict__ body_iquat,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const double* __restrict__ qvel,
    const double* __restrict__ jnt_xaxis_d,
    double*       __restrict__ qfrc_bias)
{
    __shared__ int    s_jtype[MAXJ], s_jdof[MAXJ], s_slot[MAXB];
    __shared__ double s_xax[MAXJ*3], s_qvel[MAXNV], s_qb[MAXNV];
    const double* s_xpos = xpos;    // this CTA's LDS
    const double* s_xquat = xquat;
    const int f = tid;              // slot
    const int lsv = (tid < tree_depth + 2) ? level_start_arr[tid] : 0;
    const int nlist = __builtin_amdgcn_readlane(lsv, tree_depth + 1);
    const bool live = (f < nlist);
    int bid = 0, p = -1, ja = 0, jn = 0;
    if (live) { bid = level_body_ids[f]; p = body_parent[bid]; ja = body_jnt_adr[bid]; jn = body_jnt_num[bid]; }
    // force-pass operands (issued early)
    double m = 0, Ix = 0, Iy = 0, Iz = 0, ipx = 0, ipy = 0, ipz = 0, iqw = 0, iqx = 0, iqy = 0, iqz = 0;
    if (live && bid >= 1) {
        m = body_mass[bid];
        Ix = body_inertia[bid*3]; Iy = body_inertia[bid*3+1]; Iz = body_inertia[bid*3+2];
        ipx = body_ipos[bid*3]; ipy = body_ipos[bid*3+1]; ipz = body_ipos[bid*3+2];
        iqw = body_iquat[bid*4]; iqx = body_iquat[bid*4+1]; iqy = body_iquat[bid*4+2]; iqz = body_iquat[bid*4+3];
    }
    for (int i = tid; i < njnt; i += 64) { s_jtype[i] = jnt_type[i]; s_jdof[i] = jnt_dof_adr[i]; }
    for (int i = tid; i < njnt * 3; i += 64) s_xax[i] = jnt_xaxis_d[i];
    for (int i = tid; i < nv; i += 64) { s_qvel[i] = qvel[env * nv + i]; s_qb[i] = 0.0; }
    if (live) s_slot[bid] = f;
    // level of this slot
    int lvl = -1;
    for (int L = 0; L <= tree_depth; L++) if (f >= __builtin_amdgcn_readlane(lsv, L)) lvl = L;
    if (!live) lvl = -1;
    r3_wsync_k();
    const int pslot = (live && p >= 0) ? s_slot[p] : 0;
    const int b3 = bid * 3, p3_pos = p * 3;
    double rx = 0, ry = 0, rz = 0;
    if (live && lvl >= 1) { rx = s_xpos[b3]-s_xpos[p3_pos]; ry = s_xpos[b3+1]-s_xpos[p3_pos+1]; rz = s_xpos[b3+2]-s_xpos[p3_pos+2]; }
    const double* ev = s_qvel;

    // ── forward walk ──
    double oi_x = 0, oi_y = 0, oi_z = 0, ali_x = 0, ali_y = 0, ali_z = 0, ai_x = 0, ai_y = 0, ai_z = 0;
    for (int level = 1; level <= tree_depth; level++) {
        const int ls = __builtin_amdgcn_readlane(lsv, level);
        const int le = __builtin_amdgcn_readlane(lsv, level + 1);
        // gather each slot's parent state (wave-uniform loop, readlane)
        double op_x = 0, op_y = 0, op_z = 0, ap_x = 0, ap_y = 0, ap_z = 0, aop_x = 0, aop_y = 0, aop_z = 0;
        int last_ps = -1;
        double g0 = 0, g1 = 0, g2 = 0, g3 = 0, g4 = 0, g5 = 0, g6 = 0, g7 = 0, g8 = 0;
        for (int s = ls; s < le; s++) {
            const int sp = __builtin_amdgcn_readlane(p, s);
            if (sp <= 0) continue;
            const int ps = __builtin_amdgcn_readlane(pslot, s);
            if (ps != last_ps) {
                g0 = r7_rl(oi_x, ps); g1 = r7_rl(oi_y, ps); g2 = r7_rl(oi_z, ps);
                g3 = r7_rl(ali_x, ps); g4 = r7_rl(ali_y, ps); g5 = r7_rl(ali_z, ps);
                g6 = r7_rl(ai_x, ps); g7 = r7_rl(ai_y, ps); g8 = r7_rl(ai_z, ps);
                last_ps = ps;
            }
            if (f == s) {
                op_x = g0; op_y = g1; op_z = g2; ap_x = g3; ap_y = g4; ap_z = g5; aop_x = g6; aop_y = g7; aop_z = g8;
            }
        }
        if (lvl == level) {
            if (p <= 0) {
                op_x = op_y = op_z = 0; ap_x = ap_y = ap_z = 0;
                aop_x = -(double)gx; aop_y = -(double)gy; aop_z = -(double)gz;
            }
            double aXr_x = ap_y*rz-ap_z*ry, aXr_y = ap_z*rx-ap_x*rz, aXr_z = ap_x*ry-ap_y*rx;
            double oXr_x = op_y*rz-op_z*ry, oXr_y = op_z*rx-op_x*rz, oXr_z = op_x*ry-op_y*rx;
            double oXoXr_x = op_y*oXr_z-op_z*oXr_y, oXoXr_y = op_z*oXr_x-op_x*oXr_z, oXoXr_z = op_x*oXr_y-op_y*oXr_x;
            double nai_x = aop_x+aXr_x+oXoXr_x, nai_y = aop_y+aXr_y+oXoXr_y, nai_z = aop_z+aXr_z+oXoXr_z;
            double noi_x = op_x, noi_y = op_y, noi_z = op_z;
            double nali_x = ap_x, nali_y = ap_y, nali_z = ap_z;
            for (int k = 0; k < jn; k++) {
                int ji = ja + k; int jtype_l = s_jtype[ji]; int dof = s_jdof[ji];
                if (jtype_l == JOINT_HINGE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    double qdot = ev[dof];
                    double sv_x=aw_x*qdot,sv_y=aw_y*qdot,sv_z=aw_z*qdot;
                    nali_x+=noi_y*sv_z-noi_z*sv_y; nali_y+=noi_z*sv_x-noi_x*sv_z; nali_z+=noi_x*sv_y-noi_y*sv_x;
                    noi_x+=sv_x; noi_y+=sv_y; noi_z+=sv_z;
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
                    nali_x+=op_y*wz-op_z*wy; nali_y+=op_z*wx-op_x*wz; nali_z+=op_x*wy-op_y*wx;
                    noi_x+=wx; noi_y+=wy; noi_z+=wz;
                } else if (jtype_l == JOINT_SLIDE) {
                    int ax_idx = ji*3;
                    double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                    double qdot=ev[dof];
                    nai_x+=2.0*(op_y*(aw_z*qdot)-op_z*(aw_y*qdot)); nai_y+=2.0*(op_z*(aw_x*qdot)-op_x*(aw_z*qdot)); nai_z+=2.0*(op_x*(aw_y*qdot)-op_y*(aw_x*qdot));
                }
            }
            oi_x = noi_x; oi_y = noi_y; oi_z = noi_z;
            ali_x = nali_x; ali_y = nali_y; ali_z = nali_z;
            ai_x = nai_x; ai_y = nai_y; ai_z = nai_z;
        }
    }

    double Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
    // ── per-body net force / torque: bias_rnea_lds_body_k's loop VERBATIM
    // (LDS in, LDS out).  Written as registers it contracted differently
    // (1-ulp tnet), so the operands take the same LDS round trip.
    {
    constexpr int BLK = 64;
    __shared__ double s_omega[MAXB*3], s_alpha[MAXB*3], s_acc[MAXB*3];
    __shared__ double s_fnet[MAXB*3], s_tnet[MAXB*3];
    if (live && bid >= 1) {
        s_omega[bid*3] = oi_x; s_omega[bid*3+1] = oi_y; s_omega[bid*3+2] = oi_z;
        s_alpha[bid*3] = ali_x; s_alpha[bid*3+1] = ali_y; s_alpha[bid*3+2] = ali_z;
        s_acc[bid*3] = ai_x; s_acc[bid*3+1] = ai_y; s_acc[bid*3+2] = ai_z;
    }
    r3_wsync_k();
    for (int bid = 1 + tid; bid < nbody; bid += BLK) {
        double m = (double)body_mass[bid];
        double Ix=(double)body_inertia[bid*3],Iy=(double)body_inertia[bid*3+1],Iz=(double)body_inertia[bid*3+2];
        double ipx=(double)body_ipos[bid*3],ipy=(double)body_ipos[bid*3+1],ipz=(double)body_ipos[bid*3+2];
        int bq4=bid*4;
        double bqw=s_xquat[bq4],bqx=s_xquat[bq4+1],bqy=s_xquat[bq4+2],bqz=s_xquat[bq4+3];
        double tw_x=2.0*(bqy*ipz-bqz*ipy),tw_y=2.0*(bqz*ipx-bqx*ipz),tw_z=2.0*(bqx*ipy-bqy*ipx);
        double rc_x=ipx+bqw*tw_x+(bqy*tw_z-bqz*tw_y);
        double rc_y=ipy+bqw*tw_y+(bqz*tw_x-bqx*tw_z);
        double rc_z=ipz+bqw*tw_z+(bqx*tw_y-bqy*tw_x);
        int o3=bid*3;
        double oi_x=s_omega[o3],oi_y=s_omega[o3+1],oi_z=s_omega[o3+2];
        double ali_x=s_alpha[o3],ali_y=s_alpha[o3+1],ali_z=s_alpha[o3+2];
        double ao_x=s_acc[o3],ao_y=s_acc[o3+1],ao_z=s_acc[o3+2];
        double aXrc_x=ali_y*rc_z-ali_z*rc_y,aXrc_y=ali_z*rc_x-ali_x*rc_z,aXrc_z=ali_x*rc_y-ali_y*rc_x;
        double oXrc_x=oi_y*rc_z-oi_z*rc_y,oXrc_y=oi_z*rc_x-oi_x*rc_z,oXrc_z=oi_x*rc_y-oi_y*rc_x;
        double oXoXrc_x=oi_y*oXrc_z-oi_z*oXrc_y,oXoXrc_y=oi_z*oXrc_x-oi_x*oXrc_z,oXoXrc_z=oi_x*oXrc_y-oi_y*oXrc_x;
        double ac_x=ao_x+aXrc_x+oXoXrc_x,ac_y=ao_y+aXrc_y+oXoXrc_y,ac_z=ao_z+aXrc_z+oXoXrc_z;
        double fnx=m*ac_x, fny=m*ac_y, fnz=m*ac_z;
        s_fnet[o3]=fnx; s_fnet[o3+1]=fny; s_fnet[o3+2]=fnz;
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
        double rcXf_x=rc_y*fnz-rc_z*fny;
        double rcXf_y=rc_z*fnx-rc_x*fnz;
        double rcXf_z=rc_x*fny-rc_y*fnx;
        s_tnet[o3]=Ia_x+oXIo_x+rcXf_x; s_tnet[o3+1]=Ia_y+oXIo_y+rcXf_y; s_tnet[o3+2]=Ia_z+oXIo_z+rcXf_z;
    }
    r3_wsync_k();
    if (live && bid >= 1) {
        Fx = s_fnet[bid*3]; Fy = s_fnet[bid*3+1]; Fz = s_fnet[bid*3+2];
        Tx = s_tnet[bid*3]; Ty = s_tnet[bid*3+1]; Tz = s_tnet[bid*3+2];
    }
    }
    // ── backward walk (only the parent accumulation chain) ──
    for (int level = tree_depth; level >= 1; level--) {
        const int ls = __builtin_amdgcn_readlane(lsv, level);
        const int le = __builtin_amdgcn_readlane(lsv, level + 1);
        double cfx = 0, cfy = 0, cfz = 0, ctx = 0, cty = 0, ctz = 0;
        if (lvl == level && bid >= 1) {
            const double fx=Fx,fy=Fy,fz=Fz,tx=Tx,ty=Ty,tz=Tz;
            if (p > 0 && p < nbody) {
                cfx = fx; cfy = fy; cfz = fz;
                ctx = tx+(ry*fz-rz*fy); cty = ty+(rz*fx-rx*fz); ctz = tz+(rx*fy-ry*fx);
            }
        }
        // scatter to parents, slots in ascending (lane) order
        for (int s = ls; s < le; s++) {
            const int sp = __builtin_amdgcn_readlane(p, s);
            const int sb = __builtin_amdgcn_readlane(bid, s);
            if (sb < 1 || !(sp > 0 && sp < nbody)) continue;
            const int ps = __builtin_amdgcn_readlane(pslot, s);
            const double vfx = r7_rl(cfx, s), vfy = r7_rl(cfy, s), vfz = r7_rl(cfz, s);
            const double vtx = r7_rl(ctx, s), vty = r7_rl(cty, s), vtz = r7_rl(ctz, s);
            if (f == ps) {
                Fx = r7_opq(Fx) + r7_opq(vfx); Fy = r7_opq(Fy) + r7_opq(vfy); Fz = r7_opq(Fz) + r7_opq(vfz);
                Tx = r7_opq(Tx) + r7_opq(vtx); Ty = r7_opq(Ty) + r7_opq(vty); Tz = r7_opq(Tz) + r7_opq(vtz);
            }
        }
    }
    if (live && bid >= 1) {
        const double fx=Fx,fy=Fy,fz=Fz,tx=Tx,ty=Ty,tz=Tz;
        for (int k = 0; k < jn; k++) {
            int ji=ja+k; int jtype_l=s_jtype[ji]; int dof=s_jdof[ji];
            if (jtype_l == JOINT_HINGE) {
                int ax_idx = ji*3;
                double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                if (dof < nv) atomicAdd(&s_qb[dof], aw_x*tx+aw_y*ty+aw_z*tz);
            } else if (jtype_l == JOINT_FREE) {
                if (dof+5 < nv) {
                    atomicAdd(&s_qb[dof],fx); atomicAdd(&s_qb[dof+1],fy); atomicAdd(&s_qb[dof+2],fz);
                    int bq4f=bid*4;
                    double fqw=s_xquat[bq4f],fqx=s_xquat[bq4f+1],fqy=s_xquat[bq4f+2],fqz=s_xquat[bq4f+3];
                    double R00=1-2*(fqy*fqy+fqz*fqz),R10=2*(fqx*fqy+fqw*fqz),R20=2*(fqx*fqz-fqw*fqy);
                    double R01=2*(fqx*fqy-fqw*fqz),R11=1-2*(fqx*fqx+fqz*fqz),R21=2*(fqy*fqz+fqw*fqx);
                    double R02=2*(fqx*fqz+fqw*fqy),R12=2*(fqy*fqz-fqw*fqx),R22=1-2*(fqx*fqx+fqy*fqy);
                    double tb_x=R00*tx+R10*ty+R20*tz;
                    double tb_y=R01*tx+R11*ty+R21*tz;
                    double tb_z=R02*tx+R12*ty+R22*tz;
                    atomicAdd(&s_qb[dof+3],tb_x); atomicAdd(&s_qb[dof+4],tb_y); atomicAdd(&s_qb[dof+5],tb_z);
                }
            } else if (jtype_l == JOINT_SLIDE) {
                int ax_idx = ji*3;
                double aw_x=s_xax[ax_idx],aw_y=s_xax[ax_idx+1],aw_z=s_xax[ax_idx+2];
                if (dof < nv) atomicAdd(&s_qb[dof], aw_x*fx+aw_y*fy+aw_z*fz);
            }
        }
    }
    r3_wsync_k();
    for (int i = tid; i < nv; i += 64) qfrc_bias[env*nv+i] = s_qb[i];
}

}  // namespace r3pre

// Block roles per env (blockIdx.y): x = 0 bias (wave 0, LDS RNEA), x = 1 geom
// float + AABB, x = 2 geom double + counter clears, x >= 3 mass matrix blocks.
HIPGL_GLOBAL void __launch_bounds__(256) kernel_pre_collision(
    int num_envs, int nv, int njnt, int nbody, int ngeom,
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
    double*      __restrict__ M_inv_diag_out,
    int tree_depth,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    double gx, double gy, double gz,
    const double* __restrict__ qvel,
    double*      __restrict__ qfrc_bias,
    // geom
    const int*   __restrict__ geom_body,
    const float* __restrict__ geom_pos,
    const float* __restrict__ geom_quat,
    const int*   __restrict__ geom_type,
    const float* __restrict__ geom_size,
    const float* __restrict__ xpos_f,
    const float* __restrict__ xquat_f,
    float*       __restrict__ geom_xpos_out,
    float*       __restrict__ geom_xquat_out,
    AABB*        __restrict__ geom_aabb,
    const int*   __restrict__ geom_meshid,
    const int*   __restrict__ mesh_convex_vert_adr,
    const int*   __restrict__ mesh_convex_vert_num,
    const float* __restrict__ mesh_convex_vert,
    int n_geom_d,
    const double* __restrict__ geom_pos_d,
    const double* __restrict__ geom_quat_d,
    double*       __restrict__ geom_xpos_d,
    double*       __restrict__ geom_xquat_d,
    double*       __restrict__ geom_xmat_d,
    int* __restrict__ ncon, int* __restrict__ body_pair_ncon, int bp_nbody)
{
    const int env = blockIdx.y;
    if (env >= num_envs) return;
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    if (bx == 0) {
        if (tid >= 64) return;
        r3pre::bias_rnea_lds_body_k<64, 64, 64>(env, tid, tree_depth, nv, njnt, nbody,
            level_start_arr, level_body_ids, body_parent_g, body_jnt_adr, body_jnt_num,
            jnt_type_g, jnt_dof_adr_g, gx, gy, gz,
            body_mass_g, body_inertia_g, body_ipos_g, body_iquat_g,
            xpos_g, xquat_g, qvel, jnt_xaxis_d, qfrc_bias);
        return;
    }
    if (bx == 1) {
        if (HIPGL_WORLD_FROZEN(env)) return;
        for (int gid = tid; gid < ngeom; gid += blockDim.x) {
            int body = geom_body[gid];
            int bidx3 = env * nbody * 3 + body * 3;
            int bidx4 = env * nbody * 4 + body * 4;
            float bpos[3] = {xpos_f[bidx3], xpos_f[bidx3+1], xpos_f[bidx3+2]};
            float bquat[4] = {xquat_f[bidx4], xquat_f[bidx4+1], xquat_f[bidx4+2], xquat_f[bidx4+3]};
            geom_xform_aabb_one(env, gid, ngeom, bpos, bquat, geom_pos, geom_quat,
                                geom_type, geom_size, geom_xpos_out, geom_xquat_out,
                                geom_aabb, geom_meshid, mesh_convex_vert_adr,
                                mesh_convex_vert_num, mesh_convex_vert);
        }
        return;
    }
    if (bx == 2) {
        for (int gid = tid; gid < n_geom_d; gid += blockDim.x) {
            int body = geom_body[gid];
            int b3 = env*nbody*3+body*3, b4 = env*nbody*4+body*4;
            double bp[3]={xpos_g[b3],xpos_g[b3+1],xpos_g[b3+2]};
            double bq[4]={xquat_g[b4],xquat_g[b4+1],xquat_g[b4+2],xquat_g[b4+3]};
            geom_xform_d_one(env, gid, ngeom, bp, bq, geom_pos_d, geom_quat_d,
                             geom_xpos_d, geom_xquat_d, geom_xmat_d);
        }
        if (ncon && tid == 0) ncon[env] = 0;
        if (body_pair_ncon) {
            const int nbb = bp_nbody * bp_nbody;
            for (int i = tid; i < nbb; i += blockDim.x) body_pair_ncon[env * nbb + i] = 0;
        }
        return;
    }
    r3pre::build_mass_matrix_body_k<true>(bx - 3, num_envs, nv, njnt, nbody,
        jnt_type_g, jnt_body_g, jnt_dof_adr_g, jnt_axis, body_mass_g, body_inertia_g,
        body_ipos_g, body_iquat_g, body_parent_g, xpos_g, xquat_g, jnt_xaxis_d,
        jnt_damping, jnt_armature, timestep, enable_damping, M_out, M_inv_diag_out);
}


#define R6G __attribute__((address_space(1)))
#define R6L __attribute__((address_space(3)))
// Pointer fields are address_space(1): read from an LDS struct inside a
// noinline function they would otherwise be generic and every access a flat
// op (slow, and it serialises against the LDS traffic).
struct FkPreArgs {
    int num_envs, tree_depth, nv, njnt, nbody, ngeom, n_geom_d, bp_nbody, nbm, enable_damping;
    int mass_v2;
    int bias_v2;
    float timestep;
    double gx, gy, gz;
    R6G const int* level_start; R6G const int* level_body_ids; R6G const int* body_parent;
    R6G const int* body_jnt_adr; R6G const int* body_jnt_num;
    R6G const int* jnt_type; R6G const int* jnt_body; R6G const int* jnt_dof_adr;
    R6G const double* jnt_axis;
    R6G const double* body_mass; R6G const double* body_inertia;
    R6G const double* body_ipos; R6G const double* body_iquat;
    R6G const double* jnt_damping; R6G const double* jnt_armature;
    R6G double* M_out; R6G double* M_inv_diag;
    R6G const double* qvel; R6G double* qfrc_bias;
    R6G const int* geom_body;
    R6G const double* geom_pos_d; R6G const double* geom_quat_d;
    R6G double* geom_xpos_d; R6G double* geom_xquat_d; R6G double* geom_xmat_d;
    R6G int* ncon; R6G int* body_pair_ncon;
    R6G const float* geom_pos; R6G const float* geom_quat; R6G const int* geom_type;
    R6G const float* geom_size;
    R6G float* geom_xpos; R6G float* geom_xquat; R6G AABB* geom_aabb;
    R6G const int* geom_meshid; R6G const int* mesh_convex_vert_adr; R6G const int* mesh_convex_vert_num;
    R6G const float* mesh_convex_vert;
};
template <class T> HIPGL_DEVICE __forceinline__ T* r6gen(R6G T* p) { return (T*)p; }
template <class T> HIPGL_DEVICE __forceinline__ T* r6gen(R6L T* p) { return (T*)p; }

// geom float transforms + AABBs for one world, one wavefront per geom (CTA
// ab of nab).  Every lane runs geom_xform_aabb_one itself (the serial path's
// code; identical-value pose/primitive-AABB stores are benign) on hull
// vertices lane, lane+64, ...; the partial min/max are reduced with a wave
// tree (fminf/fmaxf are exact and order-independent).  Body pose from the
// float arrays this CTA stored, as kernel_pre_collision reads them.
__attribute__((noinline)) HIPGL_DEVICE void fkpre_geom_aabb_roles(const FkPreArgs& a, int env, int tid,
    int ab, int nab, R6G const float* __restrict__ xpos_f, R6G const float* __restrict__ xquat_f)
{
    const int lane = tid & 63;
    const int nbody = a.nbody;
    const int nwav = nab * (int)(blockDim.x >> 6);
    for (int gid = ab * (int)(blockDim.x >> 6) + (tid >> 6); gid < a.ngeom; gid += nwav) {
        int body = a.geom_body[gid];
        int bidx3 = env * nbody * 3 + body * 3;
        int bidx4 = env * nbody * 4 + body * 4;
        float bpos[3] = {xpos_f[bidx3], xpos_f[bidx3+1], xpos_f[bidx3+2]};
        float bquat[4] = {xquat_f[bidx4], xquat_f[bidx4+1], xquat_f[bidx4+2], xquat_f[bidx4+3]};
        float pr[6] = {1e20f, 1e20f, 1e20f, -1e20f, -1e20f, -1e20f};
        geom_xform_aabb_one(env, gid, a.ngeom, bpos, bquat, r6gen(a.geom_pos), r6gen(a.geom_quat),
                            r6gen(a.geom_type), r6gen(a.geom_size), r6gen(a.geom_xpos), r6gen(a.geom_xquat),
                            r6gen(a.geom_aabb), r6gen(a.geom_meshid), r6gen(a.mesh_convex_vert_adr),
                            r6gen(a.mesh_convex_vert_num), r6gen(a.mesh_convex_vert), 0, lane, 64, pr);
        const bool is_mesh = a.geom_type[gid] == GEOM_MESH && a.geom_meshid && a.mesh_convex_vert &&
            a.mesh_convex_vert_adr && a.mesh_convex_vert_num && a.geom_meshid[gid] >= 0 &&
            a.mesh_convex_vert_num[a.geom_meshid[gid]] > 0;
        if (!is_mesh) continue;
        float mn_x = pr[0], mn_y = pr[1], mn_z = pr[2], mx_x = pr[3], mx_y = pr[4], mx_z = pr[5];
        #pragma unroll
        for (int off = 32; off >= 1; off >>= 1) {
            mn_x = fminf(mn_x, __shfl_xor(mn_x, off)); mn_y = fminf(mn_y, __shfl_xor(mn_y, off));
            mn_z = fminf(mn_z, __shfl_xor(mn_z, off));
            mx_x = fmaxf(mx_x, __shfl_xor(mx_x, off)); mx_y = fmaxf(mx_y, __shfl_xor(mx_y, off));
            mx_z = fmaxf(mx_z, __shfl_xor(mx_z, off));
        }
        if (lane == 0) {
            R6G AABB& aabb = a.geom_aabb[env * a.ngeom + gid];
            aabb.min_x = mn_x; aabb.min_y = mn_y; aabb.min_z = mn_z;
            aabb.max_x = mx_x; aabb.max_y = mx_y; aabb.max_z = mx_z;
        }
    }
}

// The pre_collision roles of kernel_fk_pre_v, out of line: keeping them out of
// the kernel body leaves its code (staging + level walk + write-back) identical
// to kernel_fk_fused_lds's; inlined, the extra code changed the walk's FMA
// contraction (1-ulp xquat diffs).  Each role is kernel_pre_collision's code,
// with xpos/xquat from this CTA's LDS and jnt_xaxis_d from its own stores.
__attribute__((noinline)) HIPGL_DEVICE void fkpre_roles(const FkPreArgs& a, int env, int tid,
    R6L const double* s_xpos_l, R6L const double* s_xquat_l, R6G const double* jnt_xaxis_d,
    R6G const float* xpos_f, R6G const float* xquat_f)
{
    const double* s_xpos = r6gen(s_xpos_l);
    const double* s_xquat = r6gen(s_xquat_l);
    const int nbody = a.nbody, nj3 = a.njnt * 3;
    const int bx = blockIdx.y;
    const double* xax_src = r6gen(jnt_xaxis_d + (size_t)env * nj3);
    if (bx == 0) {
        if (tid < 64) {
            if (a.bias_v2 == 3)
                r3pre::bias_rnea_reg_v3<64, 64, 64>(env, tid, a.tree_depth, a.nv, a.njnt, nbody,
                    r6gen(a.level_start), r6gen(a.level_body_ids), r6gen(a.body_parent),
                    r6gen(a.body_jnt_adr), r6gen(a.body_jnt_num),
                    r6gen(a.jnt_type), r6gen(a.jnt_dof_adr), a.gx, a.gy, a.gz,
                    r6gen(a.body_mass), r6gen(a.body_inertia), r6gen(a.body_ipos), r6gen(a.body_iquat),
                    s_xpos, s_xquat, r6gen(a.qvel), xax_src, r6gen(a.qfrc_bias));
            else
            r3pre::bias_rnea_lds_body_k<64, 64, 64, true>(env, tid, a.tree_depth, a.nv, a.njnt, nbody,
                r6gen(a.level_start), r6gen(a.level_body_ids), r6gen(a.body_parent),
                r6gen(a.body_jnt_adr), r6gen(a.body_jnt_num),
                r6gen(a.jnt_type), r6gen(a.jnt_dof_adr), a.gx, a.gy, a.gz,
                r6gen(a.body_mass), r6gen(a.body_inertia), r6gen(a.body_ipos), r6gen(a.body_iquat),
                s_xpos, s_xquat, r6gen(a.qvel), xax_src, r6gen(a.qfrc_bias));
            return;
        }
        const int t = tid - 64, T = (int)blockDim.x - 64;
        for (int gid = t; gid < a.n_geom_d; gid += T) {
            int body = a.geom_body[gid];
            double bp[3] = {s_xpos[body*3], s_xpos[body*3+1], s_xpos[body*3+2]};
            double bq[4] = {s_xquat[body*4], s_xquat[body*4+1], s_xquat[body*4+2], s_xquat[body*4+3]};
            geom_xform_d_one(env, gid, a.ngeom, bp, bq, r6gen(a.geom_pos_d), r6gen(a.geom_quat_d),
                             r6gen(a.geom_xpos_d), r6gen(a.geom_xquat_d), r6gen(a.geom_xmat_d));
        }
        if (a.ncon && t == 0) a.ncon[env] = 0;
        if (a.body_pair_ncon) {
            const int nbb = a.bp_nbody * a.bp_nbody;
            for (int i = t; i < nbb; i += T) a.body_pair_ncon[env * nbb + i] = 0;
        }
        return;
    }
    if (bx <= a.nbm) {
        if (a.mass_v2) {
            r3pre::build_mass_matrix_body_v2<true, true>(bx - 1, a.num_envs, a.nv, a.njnt, nbody,
                r6gen(a.jnt_type), r6gen(a.jnt_body), r6gen(a.jnt_dof_adr), r6gen(a.jnt_axis),
                r6gen(a.body_mass), r6gen(a.body_inertia), r6gen(a.body_ipos), r6gen(a.body_iquat),
                r6gen(a.body_parent), s_xpos, s_xquat, xax_src,
                r6gen(a.jnt_damping), r6gen(a.jnt_armature), a.timestep, a.enable_damping != 0,
                r6gen(a.M_out), r6gen(a.M_inv_diag), env);
            return;
        }
        r3pre::build_mass_matrix_body_k<true, true>(bx - 1, a.num_envs, a.nv, a.njnt, nbody,
            r6gen(a.jnt_type), r6gen(a.jnt_body), r6gen(a.jnt_dof_adr), r6gen(a.jnt_axis),
            r6gen(a.body_mass), r6gen(a.body_inertia), r6gen(a.body_ipos), r6gen(a.body_iquat),
            r6gen(a.body_parent), s_xpos, s_xquat, xax_src,
            r6gen(a.jnt_damping), r6gen(a.jnt_armature), a.timestep, a.enable_damping != 0,
            r6gen(a.M_out), r6gen(a.M_inv_diag), env);
        return;
    }
    fkpre_geom_aabb_roles(a, env, tid, bx - 1 - a.nbm, (int)gridDim.y - 1 - a.nbm, xpos_f, xquat_f);
}


template <int BLK, int MAXB, int MAXJ, int MAXNQ, bool FUSE, int BLK_PRE = 256, bool SC = false>
HIPGL_GLOBAL void __launch_bounds__(BLK_PRE) kernel_fk_pre_v(
    int num_envs, int tree_depth,
    const int*    __restrict__ level_start_arr,
    const int*    __restrict__ level_body_ids,
    const int*    __restrict__ body_parent,
    const double* __restrict__ body_pos,
    const double* __restrict__ body_quat,
    int bpose_env,   // 1: body_pos/body_quat are per-world [num_envs*nbody]
    const int*    __restrict__ body_jnt_adr,
    const int*    __restrict__ body_jnt_num,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const double* __restrict__ qpos0,
    int nq, int nbody, int njnt,
    const double* __restrict__ qpos,
    double*       __restrict__ xpos,
    double*       __restrict__ xquat,
    double*       __restrict__ jnt_xaxis_d,
    float*        __restrict__ xpos_f,
    float*        __restrict__ xquat_f,
    FkFuseExtras  ex,
    FkPreArgs     pa)
{
    // Same LDS-residency transform as kernel_bias_fluid_fused: the level loop's
    // serial chain was  level_body_ids[] -> body_parent[] -> parent xpos/xquat,
    // i.e. dependent DEVICE-memory round trips once per tree level.  Stage the
    // whole per-env working set into LDS once, walk the tree entirely in LDS,
    // then write xpos/xquat back.  BLK is one wavefront so the per-level
    // __syncthreads() folds away.
    __shared__ double s_xpos[MAXB*3];
    __shared__ double s_xquat[MAXB*4];
    __shared__ double s_bpos[MAXB*3];
    __shared__ double s_bquat[MAXB*4];
    __shared__ double s_jax[MAXJ*3];
    __shared__ double s_jpos[MAXJ*3];
    __shared__ double s_qpos[MAXNQ];
    __shared__ double s_qpos0[MAXNQ];
    __shared__ int    s_lvl_start[MAX_TREE_DEPTH + 2];
    __shared__ int    s_lvl_ids[MAXB];
    __shared__ int    s_parent[MAXB];
    __shared__ int    s_jadr[MAXB];
    __shared__ int    s_jnum[MAXB];
    __shared__ int    s_jtype[MAXJ];
    __shared__ int    s_jqadr[MAXJ];

    int env = blockIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env)) return;

    if (bpose_env) { body_pos += (size_t)env * nbody * 3; body_quat += (size_t)env * nbody * 4; }

    const int tid = threadIdx.x;
    const int nb3 = nbody * 3, nb4 = nbody * 4, nj3 = njnt * 3;

    const double* qpos_src = qpos;
    if constexpr (FUSE) {
        // (1) collision counters: replaces the two hipMemsetAsync graph nodes
        //     that preceded kernel_broadphase (nothing in between reads them).
        if (ex.ncon && tid == 0) ex.ncon[env] = 0;
        if (ex.body_pair_ncon) {
            const int nbb = ex.bp_nbody * ex.bp_nbody;
            for (int i = tid; i < nbb; i += BLK) ex.body_pair_ncon[env * nbb + i] = 0;
        }
        // (1b) this env's slice of kernel_clear_constraint_buffers (same
        //      stored values; nothing between here and constraint_setup reads
        //      or writes these buffers).
        if (ex.clear_cons) {
            if (tid == 0) ex.nconstraint[env] = 0;
            const size_t base = (size_t)env * MAX_CONSTRAINTS;
            const bool clr_lam = ex.clear_cons > 1;
            for (int i = tid; i < MAX_CONSTRAINTS; i += BLK) {
                if (clr_lam) ex.constraint_lam[base + i] = 0.0;
                ex.friction_parent[base + i]  = -1;
                ex.friction_sibling[base + i] = -1;
                ex.friction_mu[base + i]      = 0.0;
            }
        }
        if (ex.do_integrate) {
            if (tid == 0 && ex.sim_time) ex.sim_time[env] += (float)ex.dt;
            for (int jid = tid; jid < njnt; jid += BLK)
                integrate_joint_d(jid, env, ex.dt, jnt_type, jnt_qpos_adr,
                                  ex.jnt_dof_adr, ex.jnt_range_d, ex.jnt_limited,
                                  nq, ex.nv, ex.enable_limits != 0,
                                  ex.qpos_f, ex.qvel_f, ex.qacc,
                                  ex.qpos_rw, ex.qvel_d, ex.free_joint_damp);
            __syncthreads();
        }
        qpos_src = ex.qpos_rw;
    }

    for (int i = tid; i < tree_depth + 2; i += BLK) s_lvl_start[i] = level_start_arr[i];
    for (int i = tid; i < nbody; i += BLK) {
        s_lvl_ids[i] = level_body_ids[i];
        s_parent[i]  = body_parent[i];
        s_jadr[i]    = body_jnt_adr[i];
        s_jnum[i]    = body_jnt_num[i];
    }
    for (int i = tid; i < njnt; i += BLK) { s_jtype[i] = jnt_type[i]; s_jqadr[i] = jnt_qpos_adr[i]; }
    // seed s_xpos/s_xquat from the current values so bodies absent from the
    // level list (unreachable from the root) keep exactly what they had.
    for (int i = tid; i < nb3; i += BLK) { s_xpos[i] = xpos[env*nb3+i]; s_bpos[i] = body_pos[i]; }
    for (int i = tid; i < nb4; i += BLK) { s_xquat[i] = xquat[env*nb4+i]; s_bquat[i] = body_quat[i]; }
    for (int i = tid; i < nj3; i += BLK) { s_jax[i] = jnt_axis[i]; s_jpos[i] = jnt_pos[i]; }
    for (int i = tid; i < nq; i += BLK) { s_qpos[i] = qpos_src[env*nq+i]; s_qpos0[i] = qpos0[i]; }
    __syncthreads();
    __shared__ double s_sin[SC ? MAXJ : 1], s_cos[SC ? MAXJ : 1];
    if constexpr (SC) {
        for (int j = tid; j < njnt; j += BLK) {
            if (s_jtype[j] == JOINT_HINGE) {
                const int qa = s_jqadr[j];
                double angle = s_qpos[qa] - s_qpos0[qa];
                double ha = angle * 0.5;
                s_sin[j] = sin(ha);
                s_cos[j] = cos(ha);
            }
        }
        __syncthreads();
    }

    for (int level = 0; level <= tree_depth; level++) {
        int ls = s_lvl_start[level];
        int count = s_lvl_start[level + 1] - ls;

        for (int local_idx = tid; local_idx < count; local_idx += BLK) {
            int body_id = s_lvl_ids[ls + local_idx];
            int parent  = s_parent[body_id];

            double p_pos[3] = {0, 0, 0};
            double p_quat[4] = {1, 0, 0, 0};
            if (parent >= 0) {
                int pidx3 = parent * 3;
                int pidx4 = parent * 4;
                p_pos[0] = s_xpos[pidx3]; p_pos[1] = s_xpos[pidx3+1]; p_pos[2] = s_xpos[pidx3+2];
                p_quat[0] = s_xquat[pidx4]; p_quat[1] = s_xquat[pidx4+1];
                p_quat[2] = s_xquat[pidx4+2]; p_quat[3] = s_xquat[pidx4+3];
            }

            double local_pos[3], local_quat[4];
            local_pos[0] = s_bpos[body_id*3];   local_pos[1] = s_bpos[body_id*3+1]; local_pos[2] = s_bpos[body_id*3+2];
            local_quat[0] = s_bquat[body_id*4]; local_quat[1] = s_bquat[body_id*4+1];
            local_quat[2] = s_bquat[body_id*4+2]; local_quat[3] = s_bquat[body_id*4+3];

            int j_adr = s_jadr[body_id];
            int j_num = s_jnum[body_id];
            double jnt_quat[4] = {1, 0, 0, 0};
            double jnt_trans[3] = {0, 0, 0};
            const double* env_qpos = s_qpos;

            for (int ji = 0; ji < j_num; ji++) {
                int j = j_adr + ji;
                int jt = s_jtype[j];
                int qa = s_jqadr[j];

                // Compute pre-joint world frame: parent * local * jnt_quat_so_far
                // This is the frame BEFORE this joint's own rotation, used for the
                // correct DOF Jacobian column (matching MuJoCo's cdof computation).
                if (jnt_xaxis_d && j < njnt) {
                    double pre_q[4]; quat_mul_d(local_quat, jnt_quat, pre_q);
                    double world_pre[4]; quat_mul_d(p_quat, pre_q, world_pre);
                    double local_ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    double ax_w[3]; quat_rotate_d(world_pre, local_ax, ax_w);
                    int ax_idx = env * njnt * 3 + j * 3;
                    jnt_xaxis_d[ax_idx]   = ax_w[0];
                    jnt_xaxis_d[ax_idx+1] = ax_w[1];
                    jnt_xaxis_d[ax_idx+2] = ax_w[2];
                }

                if (jt == JOINT_HINGE) {
                    double angle = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    double q_rot[4];
                    if constexpr (SC) { const double sn = s_sin[j]; q_rot[0] = s_cos[j]; q_rot[1] = ax[0] * sn; q_rot[2] = ax[1] * sn; q_rot[3] = ax[2] * sn; (void)angle; }
                    else axis_angle_to_quat_d(ax, angle, q_rot);
                    double tmp[4]; quat_mul_d(jnt_quat, q_rot, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_SLIDE) {
                    double disp = env_qpos[qa] - s_qpos0[qa];
                    double ax[3] = {s_jax[j*3], s_jax[j*3+1], s_jax[j*3+2]};
                    jnt_trans[0]+=ax[0]*disp; jnt_trans[1]+=ax[1]*disp; jnt_trans[2]+=ax[2]*disp;
                } else if (jt == JOINT_BALL) {
                    double bq[4] = {env_qpos[qa], env_qpos[qa+1], env_qpos[qa+2], env_qpos[qa+3]};
                    double tmp[4]; quat_mul_d(jnt_quat, bq, tmp);
                    jnt_quat[0]=tmp[0]; jnt_quat[1]=tmp[1]; jnt_quat[2]=tmp[2]; jnt_quat[3]=tmp[3];
                } else if (jt == JOINT_FREE) {
                    jnt_trans[0]=env_qpos[qa]; jnt_trans[1]=env_qpos[qa+1]; jnt_trans[2]=env_qpos[qa+2];
                    jnt_quat[0]=env_qpos[qa+3]; jnt_quat[1]=env_qpos[qa+4]; jnt_quat[2]=env_qpos[qa+5]; jnt_quat[3]=env_qpos[qa+6];
                }
            }

            double comb_quat[4], comb_pos[3];
            quat_mul_d(local_quat, jnt_quat, comb_quat);
            double rotated_jt[3]; quat_rotate_d(local_quat, jnt_trans, rotated_jt);
            comb_pos[0]=local_pos[0]+rotated_jt[0]; comb_pos[1]=local_pos[1]+rotated_jt[1]; comb_pos[2]=local_pos[2]+rotated_jt[2];

            double world_quat[4]; quat_mul_d(p_quat, comb_quat, world_quat); quat_normalize_d(world_quat);
            double rotated_cp[3]; quat_rotate_d(p_quat, comb_pos, rotated_cp);
            double world_pos[3];
            world_pos[0]=p_pos[0]+rotated_cp[0]; world_pos[1]=p_pos[1]+rotated_cp[1]; world_pos[2]=p_pos[2]+rotated_cp[2];

            if (j_num > 0 && s_jtype[j_adr] != JOINT_FREE) {
                double jp[3] = {s_jpos[j_adr*3], s_jpos[j_adr*3+1], s_jpos[j_adr*3+2]};
                if (jp[0] != 0.0 || jp[1] != 0.0 || jp[2] != 0.0) {
                    double rjp[3]; quat_rotate_d(world_quat, jp, rjp);
                    world_pos[0]-=rjp[0]; world_pos[1]-=rjp[1]; world_pos[2]-=rjp[2];
                }
            }

            if (j_num > 0 && s_jtype[j_adr] == JOINT_FREE) {
                world_pos[0]=env_qpos[s_jqadr[j_adr]]; world_pos[1]=env_qpos[s_jqadr[j_adr]+1]; world_pos[2]=env_qpos[s_jqadr[j_adr]+2];
                double fq[4]={env_qpos[s_jqadr[j_adr]+3],env_qpos[s_jqadr[j_adr]+4],env_qpos[s_jqadr[j_adr]+5],env_qpos[s_jqadr[j_adr]+6]};
                world_quat[0]=fq[0]; world_quat[1]=fq[1]; world_quat[2]=fq[2]; world_quat[3]=fq[3];
                quat_normalize_d(world_quat);
            }

            int oidx3 = body_id*3;
            int oidx4 = body_id*4;
            s_xpos[oidx3]=world_pos[0]; s_xpos[oidx3+1]=world_pos[1]; s_xpos[oidx3+2]=world_pos[2];
            s_xquat[oidx4]=world_quat[0]; s_xquat[oidx4+1]=world_quat[1]; s_xquat[oidx4+2]=world_quat[2]; s_xquat[oidx4+3]=world_quat[3];
        }

        __syncthreads();
    }

    for (int i = tid; i < nb3; i += BLK) {
        double v = s_xpos[i];
        xpos[env*nb3+i] = v;
        if (xpos_f) xpos_f[env*nb3+i] = (float)v;
    }
    for (int i = tid; i < nb4; i += BLK) {
        double v = s_xquat[i];
        xquat[env*nb4+i] = v;
        if (xquat_f) xquat_f[env*nb4+i] = (float)v;
    }
    // The roles read the args from an LDS copy: passing the by-value kernel
    // parameter to the out-of-line function spilled a per-lane stack copy.
    __shared__ FkPreArgs s_pa;
    if (tid == 0) s_pa = pa;
    __syncthreads();
    fkpre_roles(s_pa, env, tid, (R6L const double*)s_xpos, (R6L const double*)s_xquat,
                (R6G const double*)jnt_xaxis_d, (R6G const float*)xpos_f, (R6G const float*)xquat_f);
}


bool launch_fk_pre_collision(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const SolverOptions& opts,
    int tree_depth,
    bool zero_contacts,
    hipStream_t stream)
{
    const int ne = data.num_envs, nv = model.nv;
    if (ne <= 0 || nv <= 0 || nv > KERNEL_MAXNV || nv > 64 || model.nbody <= 1 || model.ngeom <= 0) return false;
    const bool fluid_on = (model.opt_density > 0.0f || model.opt_viscosity > 0.0f)
                          && model.body_inertia_box != nullptr;
    if (fluid_on || !(model.nbody <= 64 && model.njnt <= 64) || model.tree_depth > MAX_TREE_DEPTH
        || model.tree_depth != tree_depth || !data.jnt_xaxis_d) return false;
    static const bool fkpre = [] { const char* e = getenv("HP_FKPRE"); return !(e && e[0] == '0'); }();
    static const bool fksc  = [] { const char* e = getenv("HP_FKSC");  return !(e && e[0] == '0'); }();
    static const int  nab   = [] { const char* e = getenv("HP_FKP_NAB"); int v = e ? atoi(e) : 4; return v < 1 ? 1 : v; }();
    if (fkpre && model.nq <= 64) {
        FkPreArgs a;
        a.num_envs = ne; a.tree_depth = tree_depth; a.nv = nv; a.njnt = model.njnt; a.nbody = model.nbody;
        static const int bv2 = [] { const char* e = getenv("HP_BIASV2"); return e ? atoi(e) : 3; }();
        a.bias_v2 = (bv2 == 3) ? 3 : 0;   // nbody <= 64 (checked above) bounds the slot list
        static const bool mv2 = [] { const char* e = getenv("HP_MASSV2"); return !(e && e[0] == '0'); }();
        a.mass_v2 = mv2 ? 1 : 0;
        a.ngeom = model.ngeom;
        a.n_geom_d = (data.geom_xpos_d && data.geom_xmat_d) ? model.ngeom : 0;
        a.bp_nbody = data.nbody; a.nbm = (nv * nv + 255) / 256;
        a.enable_damping = opts.enable_damping ? 1 : 0; a.timestep = model.timestep;
        a.gx = model.gravity_d[0]; a.gy = model.gravity_d[1]; a.gz = model.gravity_d[2];
        a.level_start = (R6G const int*)model.level_start; a.level_body_ids = (R6G const int*)model.level_body_ids;
        a.body_parent = (R6G const int*)model.body_parent;
        a.body_jnt_adr = (R6G const int*)model.body_jnt_adr; a.body_jnt_num = (R6G const int*)model.body_jnt_num;
        a.jnt_type = (R6G const int*)model.jnt_type; a.jnt_body = (R6G const int*)model.jnt_body;
        a.jnt_dof_adr = (R6G const int*)model.jnt_dof_adr; a.jnt_axis = (R6G const double*)model.jnt_axis_d;
        a.body_mass = (R6G const double*)model.body_mass_d; a.body_inertia = (R6G const double*)model.body_inertia_d;
        a.body_ipos = (R6G const double*)model.body_ipos_d; a.body_iquat = (R6G const double*)model.body_iquat_d;
        a.jnt_damping = (R6G const double*)model.jnt_damping_d; a.jnt_armature = (R6G const double*)model.jnt_armature_d;
        a.M_out = (R6G double*)data.solver_M; a.M_inv_diag = (R6G double*)data.solver_M_inv_diag;
        a.qvel = (R6G const double*)data.qvel_d; a.qfrc_bias = (R6G double*)data.qfrc_bias;
        a.geom_body = (R6G const int*)model.geom_body;
        a.geom_pos_d = (R6G const double*)model.geom_pos_d; a.geom_quat_d = (R6G const double*)model.geom_quat_d;
        a.geom_xpos_d = (R6G double*)data.geom_xpos_d; a.geom_xquat_d = (R6G double*)data.geom_xquat_d;
        a.geom_xmat_d = (R6G double*)data.geom_xmat_d;
        a.ncon = (R6G int*)(zero_contacts ? data.ncon : nullptr);
        a.body_pair_ncon = (R6G int*)(zero_contacts ? data.body_pair_ncon : nullptr);
        a.geom_pos = (R6G const float*)model.geom_pos; a.geom_quat = (R6G const float*)model.geom_quat;
        a.geom_type = (R6G const int*)model.geom_type; a.geom_size = (R6G const float*)model.geom_size;
        a.geom_xpos = (R6G float*)data.geom_xpos; a.geom_xquat = (R6G float*)data.geom_xquat;
        a.geom_aabb = (R6G AABB*)data.geom_aabb;
        a.geom_meshid = (R6G const int*)model.geom_meshid;
        a.mesh_convex_vert_adr = (R6G const int*)model.mesh_convex_vert_adr;
        a.mesh_convex_vert_num = (R6G const int*)model.mesh_convex_vert_num;
        a.mesh_convex_vert = (R6G const float*)model.mesh_convex_vert;
        const bool be = fk_bpose_env(model, ne);
        const dim3 grid(ne, 1 + a.nbm + nab);
        if (fksc)
            hipLaunchKernelGGL((kernel_fk_pre_v<64, 64, 64, 64, false, 256, true>), grid, dim3(256), 0, stream,
                ne, tree_depth, model.level_start, model.level_body_ids, model.body_parent,
                be ? model.body_pos_env_d : model.body_pos_d, be ? model.body_quat_env_d : model.body_quat_d,
                be ? 1 : 0, model.body_jnt_adr, model.body_jnt_num, model.jnt_type, model.jnt_qpos_adr,
                model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d, model.nq, model.nbody, model.njnt,
                data.qpos_d, data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.xpos, data.xquat,
                FkFuseExtras{}, a);
        else
            hipLaunchKernelGGL((kernel_fk_pre_v<64, 64, 64, 64, false, 256, false>), grid, dim3(256), 0, stream,
                ne, tree_depth, model.level_start, model.level_body_ids, model.body_parent,
                be ? model.body_pos_env_d : model.body_pos_d, be ? model.body_quat_env_d : model.body_quat_d,
                be ? 1 : 0, model.body_jnt_adr, model.body_jnt_num, model.jnt_type, model.jnt_qpos_adr,
                model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d, model.nq, model.nbody, model.njnt,
                data.qpos_d, data.xpos_d, data.xquat_d, data.jnt_xaxis_d, data.xpos, data.xquat,
                FkFuseExtras{}, a);
        return true;
    }
    launch_forward_kinematics_fused_d(model, data, tree_depth, stream);
    const int blk = 256;
    const int nbm = (nv * nv + blk - 1) / blk;
    const int n_d = (data.geom_xpos_d && data.geom_xmat_d) ? model.ngeom : 0;
    hipLaunchKernelGGL(kernel_pre_collision, dim3(3 + nbm, ne), dim3(blk), 0, stream,
        ne, nv, model.njnt, model.nbody, model.ngeom,
        model.jnt_type, model.jnt_body, model.jnt_dof_adr,
        model.jnt_axis_d, model.body_mass_d, model.body_inertia_d,
        model.body_ipos_d, model.body_iquat_d, model.body_parent,
        data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        model.jnt_damping_d, model.jnt_armature_d,
        model.timestep, opts.enable_damping,
        data.solver_M, data.solver_M_inv_diag,
        model.tree_depth, model.level_start, model.level_body_ids,
        model.body_jnt_adr, model.body_jnt_num,
        model.gravity_d[0], model.gravity_d[1], model.gravity_d[2],
        data.qvel_d, data.qfrc_bias,
        model.geom_body, model.geom_pos, model.geom_quat,
        model.geom_type, model.geom_size,
        data.xpos, data.xquat, data.geom_xpos, data.geom_xquat, data.geom_aabb,
        model.geom_meshid, model.mesh_convex_vert_adr, model.mesh_convex_vert_num,
        model.mesh_convex_vert,
        n_d, model.geom_pos_d, model.geom_quat_d,
        data.geom_xpos_d, data.geom_xquat_d, data.geom_xmat_d,
        zero_contacts ? data.ncon : nullptr, zero_contacts ? data.body_pair_ncon : nullptr, data.nbody);
    return true;
}


// Fused OSC-substep head: [integrate(prev)] + [zero contact counters] + FK +
// geom transforms (float+AABB, double) + site transforms, one dispatch.
// Returns false (and launches nothing) when the LDS FK path does not apply; the
// caller then issues the unfused sequence.
bool launch_fk_all_fused(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    int tree_depth,
    const SolverOptions& opts,
    bool do_integrate,
    bool zero_contacts,
    int clear_constraints,
    bool do_bias,
    bool* bias_done,
    hipStream_t stream)
{
    if (data.num_envs <= 0 || model.nbody <= 0) return false;
    constexpr int FK_BLK = 512, FK_MAXB = 64, FK_MAXJ = 64, FK_MAXNQ = 64;
    if (!(model.nbody <= FK_MAXB && model.njnt <= FK_MAXJ && model.nq <= FK_MAXNQ
          && tree_depth <= MAX_TREE_DEPTH))
        return false;

    FkFuseExtras ex{};
    ex.do_integrate   = (do_integrate && model.njnt > 0) ? 1 : 0;
    ex.dt             = (double)model.timestep;
    ex.jnt_dof_adr    = model.jnt_dof_adr;
    ex.jnt_range_d    = model.jnt_range_d;
    ex.jnt_limited    = model.jnt_limited;
    ex.nv             = model.nv;
    ex.enable_limits  = opts.enable_limits ? 1 : 0;
    ex.qpos_f         = data.qpos;
    ex.qvel_f         = data.qvel;
    ex.qacc           = data.qacc;
    ex.sim_time       = data.time;
    ex.qpos_rw        = data.qpos_d;
    ex.qvel_d         = data.qvel_d;
    ex.free_joint_damp = (double)opts.free_joint_damping;

    ex.ncon           = zero_contacts ? data.ncon : nullptr;
    ex.body_pair_ncon = zero_contacts ? data.body_pair_ncon : nullptr;
    ex.bp_nbody       = data.nbody;
    ex.clear_cons       = (clear_constraints >= 0 && data.nconstraint &&
                           data.constraint_friction_parent &&
                           data.constraint_friction_sibling &&
                           data.constraint_friction_mu &&
                           (clear_constraints == 0 || data.constraint_lam))
                          ? (clear_constraints > 0 ? 2 : 1) : 0;
    if (bias_done) *bias_done = false;
    ex.do_bias = (do_bias && bias_done && model.nbody > 1 && model.tree_depth == tree_depth &&
                  (model.opt_density > 0.0f || model.opt_viscosity > 0.0f) &&
                  model.body_inertia_box != nullptr &&
                  model.nbody <= FK_BF_MAXB && model.njnt <= FK_BF_MAXJ &&
                  model.nv <= FK_BF_MAXNV && model.tree_depth <= MAX_TREE_DEPTH) ? 1 : 0;
    ex.bf_jnt_dof_adr      = model.jnt_dof_adr;
    ex.bf_gx = model.gravity_d[0]; ex.bf_gy = model.gravity_d[1]; ex.bf_gz = model.gravity_d[2];
    ex.bf_body_mass        = model.body_mass_d;
    ex.bf_body_inertia     = model.body_inertia_d;
    ex.bf_body_inertia_box = model.body_inertia_box;
    ex.bf_body_ipos        = model.body_ipos_d;
    ex.bf_body_iquat       = model.body_iquat_d;
    ex.bf_qvel             = data.qvel_d;
    ex.bf_density          = model.opt_density;
    ex.bf_viscosity        = model.opt_viscosity;
    ex.bf_qfrc_bias        = data.qfrc_bias;
    ex.bf_qfrc_fluid       = data.qfrc_fluid;
    ex.nconstraint      = data.nconstraint;
    ex.constraint_lam   = data.constraint_lam;
    ex.friction_parent  = data.constraint_friction_parent;
    ex.friction_sibling = data.constraint_friction_sibling;
    ex.friction_mu      = data.constraint_friction_mu;

    ex.ngeom          = model.ngeom;
    ex.n_geom_f       = model.ngeom > 0 ? model.ngeom : 0;
    ex.geom_body      = model.geom_body;
    ex.geom_pos       = model.geom_pos;
    ex.geom_quat      = model.geom_quat;
    ex.geom_type      = model.geom_type;
    ex.geom_size      = model.geom_size;
    ex.geom_xpos      = data.geom_xpos;
    ex.geom_xquat     = data.geom_xquat;
    ex.geom_aabb      = data.geom_aabb;
    ex.geom_meshid    = model.geom_meshid;
    ex.mesh_convex_vert_adr = model.mesh_convex_vert_adr;
    ex.mesh_convex_vert_num = model.mesh_convex_vert_num;
    ex.mesh_convex_vert     = model.mesh_convex_vert;
    ex.n_geom_d       = (model.ngeom > 0 && data.geom_xpos_d && data.geom_xmat_d) ? model.ngeom : 0;
    ex.geom_pos_d     = model.geom_pos_d;
    ex.geom_quat_d    = model.geom_quat_d;
    ex.geom_xpos_d    = data.geom_xpos_d;
    ex.geom_xquat_d   = data.geom_xquat_d;
    ex.geom_xmat_d    = data.geom_xmat_d;
    ex.nsite          = model.nsite > 0 ? model.nsite : 0;
    ex.site_body      = model.site_body;
    ex.site_pos       = model.site_pos;
    ex.site_quat      = model.site_quat;
    ex.site_xpos      = data.site_xpos;
    ex.site_xquat     = data.site_xquat;
    ex.site_xmat      = data.site_xmat;

    dim3 grid(data.num_envs);
    if (ex.do_bias && data.jnt_xaxis_d) {
        const bool mesh_split = ex.n_geom_f > 0 && ex.geom_meshid && ex.mesh_convex_vert &&
                                ex.mesh_convex_vert_adr && ex.mesh_convex_vert_num;
        ex.defer_mesh_aabb = mesh_split ? 1 : 0;
        hipLaunchKernelGGL((kernel_fk_head_bias<FK_BLK, FK_MAXB, FK_MAXJ, FK_MAXNQ>),
            grid, FK_BLK, 0, stream,
            data.num_envs, tree_depth,
            model.level_start, model.level_body_ids,
            model.body_parent,
            fk_bpose_env(model, data.num_envs) ? model.body_pos_env_d : model.body_pos_d,
            fk_bpose_env(model, data.num_envs) ? model.body_quat_env_d : model.body_quat_d,
            fk_bpose_env(model, data.num_envs) ? 1 : 0,
            model.body_jnt_adr, model.body_jnt_num,
            model.jnt_type, model.jnt_qpos_adr,
            model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d,
            model.nq, model.nbody, model.njnt,
            data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
            data.xpos, data.xquat, ex);
        if (mesh_split) {
            hipLaunchKernelGGL(kernel_mesh_aabb, dim3(ex.n_geom_f, data.num_envs), dim3(64), 0, stream,
                data.num_envs, ex.ngeom, ex.geom_type, ex.geom_xpos, ex.geom_xquat,
                ex.geom_aabb, ex.geom_meshid, ex.mesh_convex_vert_adr,
                ex.mesh_convex_vert_num, ex.mesh_convex_vert);
        }
        if (bias_done) *bias_done = true;
        return true;
    }
    hipLaunchKernelGGL((kernel_fk_fused_lds<FK_BLK, FK_MAXB, FK_MAXJ, FK_MAXNQ, true>),
        grid, FK_BLK, 0, stream,
        data.num_envs, tree_depth,
        model.level_start, model.level_body_ids,
        model.body_parent,
            fk_bpose_env(model, data.num_envs) ? model.body_pos_env_d : model.body_pos_d,
            fk_bpose_env(model, data.num_envs) ? model.body_quat_env_d : model.body_quat_d,
            fk_bpose_env(model, data.num_envs) ? 1 : 0,
        model.body_jnt_adr, model.body_jnt_num,
        model.jnt_type, model.jnt_qpos_adr,
        model.jnt_axis_d, model.jnt_pos_d, model.qpos0_d,
        model.nq, model.nbody, model.njnt,
        data.qpos_d, data.xpos_d, data.xquat_d, data.jnt_xaxis_d,
        data.xpos, data.xquat, ex);
    if (bias_done) *bias_done = ex.do_bias != 0;
    return true;
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
