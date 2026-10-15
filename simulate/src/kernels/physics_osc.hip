// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GPU-side Operational Space Controller (OSC) for the GPU simulator.
//
// Eliminates 25 CPU round-trips per action by computing Jacobians,
// mass-matrix inversions, PD wrench, and gravity-compensated torques
// entirely on the GPU.
//
// Each thread block handles one environment.  Thread 0 performs all
// serial linear-algebra (tiny 7x7 matrices); remaining lanes idle.

#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_osc.h"
#include "hipgl/physics/physics_types.h"
#include <math.h>

namespace hipgl {
namespace physics {
namespace kernels {

// Launch width of kernel_osc_controller (see launch_osc_controller below).
constexpr int OSC_BLOCK = 64;

// ── Small linear-algebra helpers (device, double precision) ─────────────────

__device__ static void quat_to_mat3_d(const float* q, double* R) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double xx = x*x, yy = y*y, zz = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    double wx = w*x, wy = w*y, wz = w*z;
    R[0] = 1-2*(yy+zz); R[1] = 2*(xy-wz);   R[2] = 2*(xz+wy);
    R[3] = 2*(xy+wz);   R[4] = 1-2*(xx+zz); R[5] = 2*(yz-wx);
    R[6] = 2*(xz-wy);   R[7] = 2*(yz+wx);   R[8] = 1-2*(xx+yy);
}

// Ordinary torque inputs use the existing double FK, not its renderer mirrors.
// Keep the float-input helper above unchanged for policy-goal processing.
__device__ static void quat_to_mat3_fp64(const double* q, double* R) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double xx = x*x, yy = y*y, zz = z*z;
    double xy = x*y, xz = x*z, yz = y*z;
    double wx = w*x, wy = w*y, wz = w*z;
    R[0] = 1-2*(yy+zz); R[1] = 2*(xy-wz);   R[2] = 2*(xz+wy);
    R[3] = 2*(xy+wz);   R[4] = 1-2*(xx+zz); R[5] = 2*(yz-wx);
    R[6] = 2*(xz-wy);   R[7] = 2*(yz+wx);   R[8] = 1-2*(xx+yy);
}

__device__ static void osc_eef_pose_fp64(
    int env, int nbody, int site_idx, int body_idx,
    const int* site_body, const double* site_pos, const double* site_quat,
    const double* xpos, const double* xquat, double* eef_pos, double* eef_mat)
{
    const int body = site_idx >= 0 ? site_body[site_idx] : body_idx;
    const double* bp = xpos + (size_t)(env*nbody + body)*3;
    const double* bq = xquat + (size_t)(env*nbody + body)*4;
    if (site_idx < 0) {
        for (int k = 0; k < 3; ++k) eef_pos[k] = bp[k];
        quat_to_mat3_fp64(bq, eef_mat);
        return;
    }
    const double* lp = site_pos + site_idx*3;
    const double* lq = site_quat + site_idx*4;
    // Same local-to-world site construction as the float site kernel, without
    // writing any renderer/observation buffer or introducing another FK phase.
    double tx = 2.0 * (bq[2]*lp[2] - bq[3]*lp[1]);
    double ty = 2.0 * (bq[3]*lp[0] - bq[1]*lp[2]);
    double tz = 2.0 * (bq[1]*lp[1] - bq[2]*lp[0]);
    eef_pos[0] = bp[0] + (lp[0] + bq[0]*tx + (bq[2]*tz - bq[3]*ty));
    eef_pos[1] = bp[1] + (lp[1] + bq[0]*ty + (bq[3]*tx - bq[1]*tz));
    eef_pos[2] = bp[2] + (lp[2] + bq[0]*tz + (bq[1]*ty - bq[2]*tx));
    double wq[4] = {
        bq[0]*lq[0] - bq[1]*lq[1] - bq[2]*lq[2] - bq[3]*lq[3],
        bq[0]*lq[1] + bq[1]*lq[0] + bq[2]*lq[3] - bq[3]*lq[2],
        bq[0]*lq[2] - bq[1]*lq[3] + bq[2]*lq[0] + bq[3]*lq[1],
        bq[0]*lq[3] + bq[1]*lq[2] - bq[2]*lq[1] + bq[3]*lq[0]
    };
    double norm = sqrt(wq[0]*wq[0] + wq[1]*wq[1] + wq[2]*wq[2] + wq[3]*wq[3]);
    if (norm < (double)1e-12f) {
        wq[0] = 1.0; wq[1] = wq[2] = wq[3] = 0.0;
    } else {
        double inverse = 1.0 / norm;
        for (int k = 0; k < 4; ++k) wq[k] *= inverse;
    }
    quat_to_mat3_fp64(wq, eef_mat);
}

__device__ static void mat3_vec3_d(const double* R, const double* v, double* out) {
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

__device__ static void cross3_d(const double* a, const double* b, double* c) {
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ static double dot3_d(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

__device__ static double norm3_d(const double* v) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Analytical 3x3 inverse (row-major).  Writes zero matrix on singularity.
__device__ static void inv3_d(const double* A, double* I3) {
    double d = A[0]*(A[4]*A[8]-A[5]*A[7])
             - A[1]*(A[3]*A[8]-A[5]*A[6])
             + A[2]*(A[3]*A[7]-A[4]*A[6]);
    if (fabs(d) < 1e-30) { for (int i = 0; i < 9; i++) I3[i] = 0.0; return; }
    double id = 1.0 / d;
    I3[0] = (A[4]*A[8]-A[5]*A[7])*id;  I3[1] = (A[2]*A[7]-A[1]*A[8])*id;  I3[2] = (A[1]*A[5]-A[2]*A[4])*id;
    I3[3] = (A[5]*A[6]-A[3]*A[8])*id;  I3[4] = (A[0]*A[8]-A[2]*A[6])*id;  I3[5] = (A[2]*A[3]-A[0]*A[5])*id;
    I3[6] = (A[3]*A[7]-A[4]*A[6])*id;  I3[7] = (A[1]*A[6]-A[0]*A[7])*id;  I3[8] = (A[0]*A[4]-A[1]*A[3])*id;
}

// In-place Cholesky factorisation of n×n SPD matrix (lower triangle).
__device__ static void cholesky_d(double* A, int n) {
    for (int i = 0; i < n; i++) {
        double s = A[i*n + i];
        for (int k = 0; k < i; k++) s -= A[i*n+k]*A[i*n+k];
        A[i*n+i] = (s > 0.0) ? sqrt(s) : 1e-10;
        for (int j = i+1; j < n; j++) {
            s = A[j*n + i];
            for (int k = 0; k < i; k++) s -= A[j*n+k]*A[i*n+k];
            A[j*n+i] = s / A[i*n+i];
        }
    }
}

// Solve L·L^T·X = I for n×n inverse.  L must already contain lower Cholesky.
// Result is written to Ainv.
__device__ static void cholesky_inv_d(const double* L, double* Ainv, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double v = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) v -= L[i*n+k]*Ainv[k*n+j];
            Ainv[i*n+j] = v / L[i*n+i];
        }
    for (int i = n-1; i >= 0; i--)
        for (int j = 0; j < n; j++) {
            double v = Ainv[i*n+j];
            for (int k = i+1; k < n; k++) v -= L[k*n+i]*Ainv[k*n+j];
            Ainv[i*n+j] = v / L[i*n+i];
        }
}

// Orientation error: 0.5 * sum_col( cross(current_col, desired_col) )
// Both matrices are 3x3 row-major.
__device__ static void orientation_error_d(const double* goal, const double* cur, double* err) {
    err[0] = err[1] = err[2] = 0.0;
    for (int c = 0; c < 3; c++) {
        double rc[3] = { cur[0*3+c], cur[1*3+c], cur[2*3+c] };
        double rd[3] = { goal[0*3+c], goal[1*3+c], goal[2*3+c] };
        double cr[3]; cross3_d(rc, rd, cr);
        err[0] += cr[0]; err[1] += cr[1]; err[2] += cr[2];
    }
    err[0] *= 0.5; err[1] *= 0.5; err[2] *= 0.5;
}

// ── Kernel: process raw actions into goals + gripper direction ──────────────
// Runs ONCE per policy step (before the substep loop).
// Grid: num_envs,  Block: 1

__global__ void kernel_osc_process_actions(
    int num_envs, int nbody, int nq, int nv, int nsite, int nu,
    const float* __restrict__ xpos,
    const float* __restrict__ xquat,
    const float* __restrict__ site_xpos,
    const float* __restrict__ site_xmat,
    const float* __restrict__ qpos,
    const OSCConfig* __restrict__ cfg,
    OSCEnvState*  env_state,
    const float*  __restrict__ actions)    // [num_envs * 7]
{
    int env = blockIdx.x;
    if (env >= num_envs) return;

    const int na = cfg->n_arm;
    const float* act = actions + env * 7;
    OSCEnvState& es = env_state[env];

    // ── Read current EEF pose ───────────────────────────────────────────
    double eef_pos[3], eef_mat[9];
    if (es.first_goal_cache_pending) {
        for (int k = 0; k < 3; k++) eef_pos[k] = es.cached_ee_pos[k];
        for (int k = 0; k < 9; k++) eef_mat[k] = es.cached_ee_ori[k];
        es.first_goal_cache_pending = 0;
    } else if (cfg->eef_site_idx >= 0) {
        int si = cfg->eef_site_idx;
        for (int k = 0; k < 3; k++)
            eef_pos[k] = (double)site_xpos[env*nsite*3 + si*3 + k];
        for (int k = 0; k < 9; k++)
            eef_mat[k] = (double)site_xmat[env*nsite*9 + si*9 + k];
    } else {
        int bi = cfg->eef_body_idx;
        for (int k = 0; k < 3; k++)
            eef_pos[k] = (double)xpos[env*nbody*3 + bi*3 + k];
        float bq[4];
        for (int k = 0; k < 4; k++)
            bq[k] = xquat[env*nbody*4 + bi*4 + k];
        quat_to_mat3_d(bq, eef_mat);
    }

    // ── Initialise on first call ────────────────────────────────────────
    if (!es.initialized) {
        for (int k = 0; k < 3; k++) es.goal_pos[k] = eef_pos[k];
        for (int k = 0; k < 9; k++) es.goal_ori[k] = eef_mat[k];
        for (int i = 0; i < na; i++)
            es.init_jpos[i] = (double)qpos[env*nq + cfg->arm_qpos_indices[i]];
        es.initialized = 1;
    }

    // ── Position goal ───────────────────────────────────────────────────
    for (int k = 0; k < 3; k++)
        es.goal_pos[k] = eef_pos[k] + (double)act[k] * cfg->action_scale_pos;

    // ── Orientation goal (Rodrigues) ────────────────────────────────────
    double od[3];
    for (int k = 0; k < 3; k++)
        od[k] = (double)act[3+k] * cfg->action_scale_ori;
    // robosuite updates goal_ori only when the raw rotation delta is nonzero,
    // and tests the RAW action components with math.isclose(elem, 0.0), whose
    // default abs_tol is 0.0 -- i.e. EXACT zero (osc.py:258-262, "We only want to
    // update goal orientation if there is a valid delta ori value"). Testing the
    // SCALED norm against 1e-6 instead is a second, subtler mismatch: with
    // action_scale_ori = 0.5, a raw component of 1e-9 scales to 5e-10 and falls
    // under hip's threshold while robosuite still treats it as a real delta.
    bool any_ori = (act[3] != 0.0f) || (act[4] != 0.0f) || (act[5] != 0.0f);
    double angle = norm3_d(od);
    if (any_ori && angle > 1e-12) {
        double ax[3] = { od[0]/angle, od[1]/angle, od[2]/angle };
        double sa = sin(angle), ca = cos(angle);
        // K = skew(ax)
        double K[9] = {
              0,    -ax[2],  ax[1],
             ax[2],  0,     -ax[0],
            -ax[1],  ax[0],  0
        };
        // dR = I + sin(a)*K + (1-cos(a))*K*K
        double dR[9];
        double KK[9] = {0};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int m = 0; m < 3; m++)
                    KK[r*3+c] += K[r*3+m]*K[m*3+c];
        for (int i = 0; i < 9; i++)
            dR[i] = ((i%4==0)?1.0:0.0) + sa*K[i] + (1.0-ca)*KK[i];
        // goal_ori = dR @ eef_mat
        double new_ori[9] = {0};
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                for (int m = 0; m < 3; m++)
                    new_ori[r*3+c] += dR[r*3+m]*eef_mat[m*3+c];
        for (int k = 0; k < 9; k++) es.goal_ori[k] = new_ori[k];
    }
    // NO else branch. Resetting goal_ori to the CURRENT eef orientation here
    // zeroed the orientation error on every policy step that commanded no
    // rotation, so the orientation PD term could never correct accumulated wrist
    // drift -- it was structurally incapable of self-correcting. robosuite holds
    // its absolute goal_ori across such steps and keeps driving toward it.
    //
    // This was THE cause of the GPU-resident success-rate deficit against CPU
    // MuJoCo, uniform across all four LIBERO suites. Bisection:
    // robosuite's OSC on hip's dynamics CONVERGED (5.6x -> 0.2x of the 1e-3 rad
    // budget), hip's GPU OSC stayed flat at 4-8x, and two CPU MuJoCo envs driven
    // identically diverged by exactly 0.0. At kp_ori = 150 a 1e-3 rad standing
    // offset is 0.15 N*m against arm torques of 19-41 -- small per step, but it
    // never decays.
    //
    // goal_ori is initialised from eef_mat in the !es.initialized block above,
    // so it is always defined before first use; this branch was not needed for
    // that. The GPU simulator's own CPU OSC (osc_controller.py:290-298) has no such
    // reset either -- the GPU kernel was the sole outlier of the three.

    // ── Gripper direction ───────────────────────────────────────────────
    double ga = (double)act[6];
    es.gripper_direction = (ga > 0.0) ? 1.0 : (ga < 0.0 ? -1.0 : 0.0);
}

// ── Register-resident Cholesky / inverse for the one-wave OSC block ─────────
// The generic path below factors and inverts through LDS with two
// __syncthreads per column and lane 0 serialising every diagonal. On the
// fixed-width (n_arm == OSC_MAX_ARM) path the SAME scalar recurrences run in
// registers instead: lane j owns row j of the factor, the diagonal of column
// i is recomputed uniformly from row i (broadcast with v_readlane), and the
// inverse is solved one column per lane. Every multiply-subtract keeps its
// operands and its k-order and every sqrt/divide keeps its operands, so the
// arithmetic is bit-identical to cholesky_d / cholesky_inv_d.
__device__ __forceinline__ double osc_rl(double v, int lane) {
    const int hi = __builtin_amdgcn_readlane(__double2hiint(v), lane);
    const int lo = __builtin_amdgcn_readlane(__double2loint(v), lane);
    return __hiloint2double(hi, lo);
}

// a (in): row `lane` of the SPD matrix (lanes 0..N-1).
// Lu (out): uniform copy of the lower Cholesky factor (diag included).
// x (out): column `lane` of the inverse.
template <int N>
__device__ __forceinline__ void osc_lane_chol_inv(const int lane, double (&a)[N],
                                                  double (&Lu)[N][N], double (&x)[N]) {
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
        for (int k = 0; k <= i; k++) Lu[i][k] = osc_rl(a[k], i);
        double s = Lu[i][i];
#pragma unroll
        for (int k = 0; k < i; k++) s -= Lu[i][k]*Lu[i][k];
        const double d = (s > 0.0) ? sqrt(s) : 1e-10;
        Lu[i][i] = d;
        double s2 = a[i];
#pragma unroll
        for (int k = 0; k < i; k++) s2 -= a[k]*Lu[i][k];
        const double q = s2 / d;
        a[i] = (lane == i) ? d : ((lane > i) ? q : a[i]);
    }
#pragma unroll
    for (int i = 0; i < N; i++) {
        double v = (i == lane) ? 1.0 : 0.0;
#pragma unroll
        for (int k = 0; k < i; k++) v -= Lu[i][k]*x[k];
        x[i] = v / Lu[i][i];
    }
#pragma unroll
    for (int i = N-1; i >= 0; i--) {
        double v = x[i];
#pragma unroll
        for (int k = i+1; k < N; k++) v -= Lu[k][i]*x[k];
        x[i] = v / Lu[i][i];
    }
}

__device__ __forceinline__ void osc_controller_fast7(
    const int env, const int t, int nbody, int nq, int nv, int nu,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const int* __restrict__ site_body,
    const double* __restrict__ site_pos,
    const double* __restrict__ site_quat,
    const double* __restrict__ qpos,
    const double* __restrict__ qvel,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ solver_M,
    float*        __restrict__ ctrl,
    double*       __restrict__ ctrl_d,
    const OSCConfig*  __restrict__ cfg,
    OSCEnvState&  es,
    const bool use_cached_tick)
{
    constexpr int na = OSC_MAX_ARM;
    __shared__ double arm_qvel[OSC_MAX_ARM];
    __shared__ double J_pos[3*OSC_MAX_ARM];
    __shared__ double J_ori[3*OSC_MAX_ARM];
    __shared__ double M_inv[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double JM_pos[3*OSC_MAX_ARM];
    __shared__ double JM_ori[3*OSC_MAX_ARM];
    __shared__ double JMJ_pos[9];
    __shared__ double JMJ_ori[9];
    __shared__ double JMJ_full[36];
    __shared__ double T[OSC_MAX_ARM * 6];
    __shared__ double Lambda_pos[9];
    __shared__ double Lambda_ori[9];
    __shared__ double Lambda_full[36];
    __shared__ double J_bar[OSC_MAX_ARM * 6];
    __shared__ double N[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double e_null[OSC_MAX_ARM];
    __shared__ double Me[OSC_MAX_ARM];
    __shared__ double vel[6];

    const int ti = (t < na) ? t : 0;   // clamped: lanes >= na load valid dummies

    // ── 14 (hoisted). Gripper accumulator: independent of everything else.
    // Computed uniformly and branch-free here so its loads overlap the rest;
    // the persistent-state and ctrl stores stay at the end (lane 0).
    const int n_grip = cfg->n_gripper;
    const int has_range = cfg->has_gripper_ctrlrange;
    double g_acc[OSC_MAX_GRIP], g_out[OSC_MAX_GRIP];
    int    g_idx[OSC_MAX_GRIP];
    {
        double dir = es.gripper_direction;
        double spd = cfg->gripper_speed;
        double signs[2] = { -1.0, 1.0 };
#pragma unroll
        for (int g = 0; g < OSC_MAX_GRIP; g++) {
            double acc = es.gripper_current_action[g] + signs[g] * spd * dir;
            acc = (acc < -1.0) ? -1.0 : acc;
            acc = (acc >  1.0) ?  1.0 : acc;
            g_acc[g] = acc;
            double lo = cfg->gripper_ctrlrange[g*2+0];
            double hi = cfg->gripper_ctrlrange[g*2+1];
            double mapped = 0.5*(hi + lo) + 0.5*(hi - lo) * acc;
            g_out[g] = has_range ? mapped : acc;
            g_idx[g] = cfg->gripper_actuator_indices[g];
        }
    }
    // PD operands (uniform), issued now.
    double goal_p[3], goal_o[9];
#pragma unroll
    for (int k = 0; k < 3; k++) goal_p[k] = es.goal_pos[k];
#pragma unroll
    for (int k = 0; k < 9; k++) goal_o[k] = es.goal_ori[k];
    const double kp_pos = cfg->kp_pos, kd_pos = cfg->kd_pos;
    const double kp_ori = cfg->kp_ori, kd_ori = cfg->kd_ori;

    // ── 1-4. EEF pose (uniform), Jacobian column, joint state, mass row ──
    double eef_pos[3], eef_mat[9];
    double jp[3] = {0.0, 0.0, 0.0}, jo[3] = {0.0, 0.0, 0.0};
    double a_qpos, a_qvel;
    double Mr[OSC_MAX_ARM];
    const int dof_t = cfg->arm_dof_indices[ti];
    {
        a_qpos = (double)qpos[env*nq + cfg->arm_qpos_indices[ti]];
        a_qvel = (double)qvel[env*nv + dof_t];
        const double* Mbase = solver_M + (size_t)env*KERNEL_MAXNV*KERNEL_MAXNV
                                       + dof_t*KERNEL_MAXNV;
#pragma unroll
        for (int c = 0; c < na; c++) Mr[c] = Mbase[cfg->arm_dof_indices[c]];

        const int ji    = cfg->arm_joint_indices[ti];
        const int jbody = jnt_body[ji];
        const int jtype = jnt_type[ji];
        double bq[4];
        for (int k = 0; k < 4; k++)
            bq[k] = xquat[env*nbody*4 + jbody*4 + k];
        double bpos[3];
        for (int k = 0; k < 3; k++)
            bpos[k] = xpos[env*nbody*3 + jbody*3 + k];
        double local_axis[3] = {
            (double)jnt_axis[ji*3+0], (double)jnt_axis[ji*3+1], (double)jnt_axis[ji*3+2] };
        double local_pos[3] = {
            (double)jnt_pos[ji*3+0], (double)jnt_pos[ji*3+1], (double)jnt_pos[ji*3+2] };

        // osc_eef_pose_fp64, both variants computed and selected (no branch
        // on the loaded site index, so its loads join the rest).
        {
            const int site_idx = cfg->eef_site_idx;
            const bool use_site = site_idx >= 0;
            const int sidx = use_site ? site_idx : 0;
            const int body = use_site ? site_body[sidx] : cfg->eef_body_idx;
            const double* bp = xpos + (size_t)(env*nbody + body)*3;
            const double* bq2 = xquat + (size_t)(env*nbody + body)*4;
            const double* lp = site_pos + sidx*3;
            const double* lq = site_quat + sidx*4;
            double tx = 2.0 * (bq2[2]*lp[2] - bq2[3]*lp[1]);
            double ty = 2.0 * (bq2[3]*lp[0] - bq2[1]*lp[2]);
            double tz = 2.0 * (bq2[1]*lp[1] - bq2[2]*lp[0]);
            double sp0 = bp[0] + (lp[0] + bq2[0]*tx + (bq2[2]*tz - bq2[3]*ty));
            double sp1 = bp[1] + (lp[1] + bq2[0]*ty + (bq2[3]*tx - bq2[1]*tz));
            double sp2 = bp[2] + (lp[2] + bq2[0]*tz + (bq2[1]*ty - bq2[2]*tx));
            double wq[4] = {
                bq2[0]*lq[0] - bq2[1]*lq[1] - bq2[2]*lq[2] - bq2[3]*lq[3],
                bq2[0]*lq[1] + bq2[1]*lq[0] + bq2[2]*lq[3] - bq2[3]*lq[2],
                bq2[0]*lq[2] - bq2[1]*lq[3] + bq2[2]*lq[0] + bq2[3]*lq[1],
                bq2[0]*lq[3] + bq2[1]*lq[2] - bq2[2]*lq[1] + bq2[3]*lq[0]
            };
            double norm = sqrt(wq[0]*wq[0] + wq[1]*wq[1] + wq[2]*wq[2] + wq[3]*wq[3]);
            const bool tiny = norm < (double)1e-12f;
            double inverse = 1.0 / norm;
            double wqn[4];
            for (int k = 0; k < 4; ++k) wqn[k] = wq[k] * inverse;
            wqn[0] = tiny ? 1.0 : wqn[0];
            wqn[1] = tiny ? 0.0 : wqn[1];
            wqn[2] = tiny ? 0.0 : wqn[2];
            wqn[3] = tiny ? 0.0 : wqn[3];
            double bqa[4] = { bq2[0], bq2[1], bq2[2], bq2[3] };
            double qsel[4];
            for (int k = 0; k < 4; ++k) qsel[k] = use_site ? wqn[k] : bqa[k];
            quat_to_mat3_fp64(qsel, eef_mat);
            eef_pos[0] = use_site ? sp0 : bp[0];
            eef_pos[1] = use_site ? sp1 : bp[1];
            eef_pos[2] = use_site ? sp2 : bp[2];
        }

        double Rb[9];
        quat_to_mat3_fp64(bq, Rb);
        double world_axis[3];
        mat3_vec3_d(Rb, local_axis, world_axis);
        double body_wp[3];
        mat3_vec3_d(Rb, local_pos, body_wp);
        double pivot[3] = {
            bpos[0] + body_wp[0], bpos[1] + body_wp[1], bpos[2] + body_wp[2] };
        double r[3] = { eef_pos[0]-pivot[0], eef_pos[1]-pivot[1], eef_pos[2]-pivot[2] };
        double cr[3]; cross3_d(world_axis, r, cr);
        const bool hinge = jtype == JOINT_HINGE, slide = jtype == JOINT_SLIDE;
#pragma unroll
        for (int k = 0; k < 3; k++) {
            jp[k] = hinge ? cr[k] : (slide ? world_axis[k] : 0.0);
            jo[k] = hinge ? world_axis[k] : 0.0;
        }
    }
    {
        // Imported controller cache (first tick only) overrides the live
        // values; selected rather than branched so the live loads stay in
        // one straight-line region with the factorisation below.
        const bool uc = use_cached_tick;
        for (int k = 0; k < 3; k++) eef_pos[k] = uc ? es.cached_ee_pos[k] : eef_pos[k];
        for (int k = 0; k < 9; k++) eef_mat[k] = uc ? es.cached_ee_ori[k] : eef_mat[k];
        a_qpos = uc ? es.cached_joint_pos[ti] : a_qpos;
        a_qvel = uc ? es.cached_joint_vel[ti] : a_qvel;
#pragma unroll
        for (int r = 0; r < 3; r++) {
            jp[r] = uc ? es.cached_J_full[r*OSC_MAX_ARM+ti] : jp[r];
            jo[r] = uc ? es.cached_J_full[(r+3)*OSC_MAX_ARM+ti] : jo[r];
        }
#pragma unroll
        for (int c = 0; c < na; c++) Mr[c] = uc ? es.cached_mass_matrix[ti*OSC_MAX_ARM+c] : Mr[c];
    }
    // Operands of the late phases, issued now.
    const double bias_t = qfrc_bias ? qfrc_bias[(size_t)env*nv + dof_t] : 0.0;
    const double init_t = es.init_jpos[ti];
    const double kp_null = cfg->kp_null, kd_null = cfg->kd_null;
    const int    act_t  = cfg->arm_actuator_indices[ti];
    const double lim_t  = cfg->arm_ctrl_limits[ti];

    // ── 5. M inverse via Cholesky (registers) ───────────────────────────
    double Lm[OSC_MAX_ARM][OSC_MAX_ARM];
    double xm[OSC_MAX_ARM];
    double am[OSC_MAX_ARM];
#pragma unroll
    for (int c = 0; c < na; c++) am[c] = Mr[c];
    osc_lane_chol_inv<OSC_MAX_ARM>(t, am, Lm, xm);

    if (t < na) {
#pragma unroll
        for (int r = 0; r < 3; r++) {
            J_pos[r*OSC_MAX_ARM + t] = jp[r];
            J_ori[r*OSC_MAX_ARM + t] = jo[r];
        }
        arm_qvel[t] = a_qvel;
#pragma unroll
        for (int i = 0; i < na; i++) M_inv[i*na + t] = xm[i];
    }
    __syncthreads();

    // ── 6. J·M_inv products — one lane per output element ────────────────
    for (int o = t; o < 3*na; o += OSC_BLOCK) {
        const int r = o / na, c = o - r*na;
        double ap = 0.0, ao = 0.0;
        for (int k = 0; k < na; k++) {
            ap += J_pos[r*OSC_MAX_ARM+k] * M_inv[k*na+c];
            ao += J_ori[r*OSC_MAX_ARM+k] * M_inv[k*na+c];
        }
        JM_pos[o] = ap; JM_ori[o] = ao;
    }
    for (int o = t; o < na*6; o += OSC_BLOCK) {
        const int r = o / 6, c = o - r*6;
        double a = 0.0;
        for (int k = 0; k < na; k++) {
            double jt_kc = (c < 3) ? J_pos[c*OSC_MAX_ARM+k] : J_ori[(c-3)*OSC_MAX_ARM+k];
            a += M_inv[r*na+k] * jt_kc;
        }
        T[o] = a;
    }
    if (t >= 56 && t < 62) {
        const int o = t - 56;
        const int r = (o < 3) ? o : (o - 3);
        double a = 0.0;
        for (int k = 0; k < na; k++)
            a += ((o < 3) ? J_pos[r*OSC_MAX_ARM+k] : J_ori[r*OSC_MAX_ARM+k]) * arm_qvel[k];
        vel[o] = a;
    }
    __syncthreads();

    for (int o = t; o < 9; o += OSC_BLOCK) {
        const int r = o / 3, c = o - r*3;
        double ap = 0.0, ao = 0.0;
        for (int k = 0; k < na; k++) {
            ap += JM_pos[r*na+k] * J_pos[c*OSC_MAX_ARM+k];
            ao += JM_ori[r*na+k] * J_ori[c*OSC_MAX_ARM+k];
        }
        JMJ_pos[o] = ap; JMJ_ori[o] = ao;
    }
    for (int o = 9 + t; o < 9 + 36; o += OSC_BLOCK) {
        const int q = o - 9;
        const int r = q / 6, c = q - r*6;
        double a = 0.0;
        for (int k = 0; k < na; k++) {
            double j_rk = (r < 3) ? J_pos[r*OSC_MAX_ARM+k] : J_ori[(r-3)*OSC_MAX_ARM+k];
            a += j_rk * T[k*6+c];
        }
        JMJ_full[q] = a;
    }
    if (use_cached_tick) {
        if (t < 6) vel[t] = es.cached_ee_vel[t];
        if (t == 0) es.first_tick_cache_pending = 0;
    }
    __syncthreads();

    // ── 7. Lambda (operational space inertia) ───────────────────────────
    {
        // inv3_d(JMJ_pos -> Lambda_pos) on lane 0, (JMJ_ori -> Lambda_ori) on
        // lane 1; straight-line (select instead of early return) so it
        // overlaps the 6x6 factorisation below.
        const double* A = (t == 0) ? JMJ_pos : JMJ_ori;
        double* I3 = (t == 0) ? Lambda_pos : Lambda_ori;
        double d = A[0]*(A[4]*A[8]-A[5]*A[7])
                 - A[1]*(A[3]*A[8]-A[5]*A[6])
                 + A[2]*(A[3]*A[7]-A[4]*A[6]);
        const bool sing = fabs(d) < 1e-30;
        double id = 1.0 / d;
        double r3[9];
        r3[0] = (A[4]*A[8]-A[5]*A[7])*id;  r3[1] = (A[2]*A[7]-A[1]*A[8])*id;  r3[2] = (A[1]*A[5]-A[2]*A[4])*id;
        r3[3] = (A[5]*A[6]-A[3]*A[8])*id;  r3[4] = (A[0]*A[8]-A[2]*A[6])*id;  r3[5] = (A[2]*A[3]-A[0]*A[5])*id;
        r3[6] = (A[3]*A[7]-A[4]*A[6])*id;  r3[7] = (A[1]*A[6]-A[0]*A[7])*id;  r3[8] = (A[0]*A[4]-A[1]*A[3])*id;
        if (t < 2) {
#pragma unroll
            for (int k = 0; k < 9; k++) I3[k] = sing ? 0.0 : r3[k];
        }
    }
    {
        const int u6 = (t < 6) ? t : 0;
        double al[6], Ll[6][6], xl[6];
#pragma unroll
        for (int c = 0; c < 6; c++) al[c] = JMJ_full[u6*6 + c];
        osc_lane_chol_inv<6>(t, al, Ll, xl);
        if (t < 6) {
#pragma unroll
            for (int i = 0; i < 6; i++) Lambda_full[i*6 + t] = xl[i];
        }
    }
    __syncthreads();

    // ── 9. PD control (uniform: every lane holds the wrench) ────────────
    double wrench[6];
    {
        double pos_err[3], ori_err[3];
        for (int k = 0; k < 3; k++)
            pos_err[k] = goal_p[k] - eef_pos[k];
        orientation_error_d(goal_o, eef_mat, ori_err);

        double F_des[3], tau_des[3];
        for (int k = 0; k < 3; k++) {
            F_des[k]   = kp_pos * pos_err[k] - kd_pos * vel[k];
            tau_des[k]  = kp_ori * ori_err[k] - kd_ori * vel[k+3];
        }
        for (int r = 0; r < 3; r++) {
            double wp = 0.0, wo = 0.0;
            for (int k = 0; k < 3; k++) {
                wp += Lambda_pos[r*3+k] * F_des[k];
                wo += Lambda_ori[r*3+k] * tau_des[k];
            }
            wrench[r] = wp; wrench[r+3] = wo;
        }
    }
    // ── 12a. J_bar = T @ Lambda_full   (na × 6) ─────────────────────────
    for (int o = t; o < na*6; o += OSC_BLOCK) {
        const int r = o / 6, c = o - r*6;
        double a = 0.0;
        for (int k = 0; k < 6; k++) a += T[r*6+k] * Lambda_full[k*6+c];
        J_bar[o] = a;
    }
    __syncthreads();

    // ── 10./11. Torques = J_full^T @ wrench, then bias compensation ─────
    double torque_t = 0.0;
    if (t < na) {
        double a = 0.0;
        for (int k = 0; k < 3; k++) {
            a += jp[k] * wrench[k];
            a += jo[k] * wrench[k+3];
        }
        if (qfrc_bias)
            a += bias_t;
        torque_t = a;
        e_null[t] = kp_null*(init_t - a_qpos) - kd_null*a_qvel;
    }
    for (int o = t; o < na*na; o += OSC_BLOCK) {
        const int r = o / na, c = o - r*na;
        double a = (r == c) ? 1.0 : 0.0;
        for (int k = 0; k < 6; k++) {
            double j_kc = (k < 3) ? J_pos[k*OSC_MAX_ARM+c] : J_ori[(k-3)*OSC_MAX_ARM+c];
            a -= J_bar[r*6+k] * j_kc;
        }
        N[o] = a;
    }
    __syncthreads();

    if (t < na) {
        double a = 0.0;
        for (int k = 0; k < na; k++) a += Mr[k] * e_null[k];
        Me[t] = a;
    }
    __syncthreads();

    if (t < na) {
        const int i = t;
        double a = torque_t;
        for (int k = 0; k < na; k++) a += N[k*na+i] * Me[k];
        if (act_t < nu) {
            double t_clamp = a;
            if (t_clamp >  lim_t) t_clamp =  lim_t;
            if (t_clamp < -lim_t) t_clamp = -lim_t;
            ctrl_d[env*nu + act_t] = t_clamp;
            ctrl[env*nu + act_t]   = (float)t_clamp;
        }
    }
    if (t == 0) {
#pragma unroll
        for (int g = 0; g < OSC_MAX_GRIP; g++) {
            if (g < n_grip) {
                es.gripper_current_action[g] = g_acc[g];
                if (g_idx[g] < nu) {
                    ctrl_d[env*nu + g_idx[g]] = g_out[g];
                    ctrl[env*nu + g_idx[g]]   = (float)g_out[g];
                }
            }
        }
    }
}

// ── Kernel: OSC torque computation (per substep) ────────────────────────────
// Reads FK results + mass matrix, writes ctrl.
// Grid: num_envs,  Block: 64 (only thread 0 active)

__device__ __forceinline__ void osc_controller_generic(
    int num_envs, int nbody, int nq, int nv, int nsite, int nu,
    float timestep, int enable_damping,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const float* __restrict__ jnt_damping,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const int* __restrict__ site_body,
    const double* __restrict__ site_pos,
    const double* __restrict__ site_quat,
    const double* __restrict__ qpos,
    const double* __restrict__ qvel,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ solver_M,    // [ne * MAXNV * MAXNV]
    float*        __restrict__ ctrl,        // [ne * nu]
    double*       __restrict__ ctrl_d,      // [ne * nu]
    const OSCConfig*  __restrict__ cfg,
    OSCEnvState*  env_state)
{
    const int env = blockIdx.x;
    const int t   = threadIdx.x;

    __shared__ double eef_pos[3];
    __shared__ double eef_mat[9];
    __shared__ double arm_qpos[OSC_MAX_ARM];
    __shared__ double arm_qvel[OSC_MAX_ARM];
    __shared__ double J_pos[3*OSC_MAX_ARM];
    __shared__ double J_ori[3*OSC_MAX_ARM];
    __shared__ double M[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double L[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double M_inv[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double JM_pos[3*OSC_MAX_ARM];
    __shared__ double JM_ori[3*OSC_MAX_ARM];
    __shared__ double JMJ_pos[9];
    __shared__ double JMJ_ori[9];
    __shared__ double JMJ_full[36];
    __shared__ double T[OSC_MAX_ARM * 6];
    __shared__ double Lambda_pos[9];
    __shared__ double Lambda_ori[9];
    __shared__ double Lambda_full_L[36];
    __shared__ double Lambda_full[36];
    __shared__ double torques[OSC_MAX_ARM];
    __shared__ double J_bar[OSC_MAX_ARM * 6];
    __shared__ double N[OSC_MAX_ARM * OSC_MAX_ARM];
    __shared__ double e_null[OSC_MAX_ARM];
    __shared__ double Me[OSC_MAX_ARM];
    __shared__ double vel[6];      // [0..2] = vel_pos, [3..5] = vel_ori
    __shared__ double wrench[6];

    if (env >= num_envs) return;

    const int na = cfg->n_arm;
    OSCEnvState& es = env_state[env];
    const bool use_cached_tick = es.first_tick_cache_pending != 0;

    // ── 1. Read current EEF pose ────────────────────────────────────────
    if (t == 0) {
        if (use_cached_tick) {
            for (int k = 0; k < 3; k++) eef_pos[k] = es.cached_ee_pos[k];
            for (int k = 0; k < 9; k++) eef_mat[k] = es.cached_ee_ori[k];
        } else {
            osc_eef_pose_fp64(env, nbody, cfg->eef_site_idx, cfg->eef_body_idx,
                site_body, site_pos, site_quat, xpos, xquat, eef_pos, eef_mat);
        }
    }
    // Zero the fixed-extent Jacobian tiles (the old `= {0}` initialisers).
    for (int i = t; i < 3*OSC_MAX_ARM; i += OSC_BLOCK) { J_pos[i] = 0.0; J_ori[i] = 0.0; }
    __syncthreads();

    // ── 2. Read arm joint state ─────────────────────────────────────────
    if (t < na) {
        arm_qpos[t] = use_cached_tick ? es.cached_joint_pos[t]
            : (double)qpos[env*nq + cfg->arm_qpos_indices[t]];
        arm_qvel[t] = use_cached_tick ? es.cached_joint_vel[t]
            : (double)qvel[env*nv + cfg->arm_dof_indices[t]];
    }

    // ── 3. Compute Jacobian (6 × n_arm) — one lane per arm column ───────
    if (t < na) {
        const int i = t;
        if (use_cached_tick) {
            for (int r = 0; r < 3; r++) {
                J_pos[r*OSC_MAX_ARM+i] = es.cached_J_full[r*OSC_MAX_ARM+i];
                J_ori[r*OSC_MAX_ARM+i] = es.cached_J_full[(r+3)*OSC_MAX_ARM+i];
            }
        } else {
            int ji    = cfg->arm_joint_indices[i];
            int jbody = jnt_body[ji];
            int jtype = jnt_type[ji];

            double bq[4];
            for (int k = 0; k < 4; k++)
                bq[k] = xquat[env*nbody*4 + jbody*4 + k];
            double Rb[9];
            quat_to_mat3_fp64(bq, Rb);

            double local_axis[3] = {
                (double)jnt_axis[ji*3+0],
                (double)jnt_axis[ji*3+1],
                (double)jnt_axis[ji*3+2]
            };
            double world_axis[3];
            mat3_vec3_d(Rb, local_axis, world_axis);

            if (jtype == JOINT_HINGE) {
                double local_pos[3] = {
                    (double)jnt_pos[ji*3+0],
                    (double)jnt_pos[ji*3+1],
                    (double)jnt_pos[ji*3+2]
                };
                double body_wp[3];
                mat3_vec3_d(Rb, local_pos, body_wp);
                double pivot[3] = {
                    (double)xpos[env*nbody*3 + jbody*3 + 0] + body_wp[0],
                    (double)xpos[env*nbody*3 + jbody*3 + 1] + body_wp[1],
                    (double)xpos[env*nbody*3 + jbody*3 + 2] + body_wp[2]
                };
                double r[3] = { eef_pos[0]-pivot[0], eef_pos[1]-pivot[1], eef_pos[2]-pivot[2] };
                double cr[3]; cross3_d(world_axis, r, cr);
                J_pos[0*OSC_MAX_ARM + i] = cr[0];
                J_pos[1*OSC_MAX_ARM + i] = cr[1];
                J_pos[2*OSC_MAX_ARM + i] = cr[2];
                J_ori[0*OSC_MAX_ARM + i] = world_axis[0];
                J_ori[1*OSC_MAX_ARM + i] = world_axis[1];
                J_ori[2*OSC_MAX_ARM + i] = world_axis[2];
            } else if (jtype == JOINT_SLIDE) {
                J_pos[0*OSC_MAX_ARM + i] = world_axis[0];
                J_pos[1*OSC_MAX_ARM + i] = world_axis[1];
                J_pos[2*OSC_MAX_ARM + i] = world_axis[2];
            }
        }
    }

    // ── 4. Extract arm mass matrix block from solver_M ──────────────────
    // solver_M = M + diag(armature) + 1e-14 regularization (no h*damping).
    for (int o = t; o < na*na; o += OSC_BLOCK) {
        const int r = o / na, c = o - r*na;
        M[o] = use_cached_tick ? es.cached_mass_matrix[r*OSC_MAX_ARM+c]
            : solver_M[(size_t)env*KERNEL_MAXNV*KERNEL_MAXNV
                               + cfg->arm_dof_indices[r]*KERNEL_MAXNV
                               + cfg->arm_dof_indices[c]];
        L[o] = M[o];
    }
    if (false) {
        for (int i = 0; i < na; i++) {
            int ji = cfg->arm_joint_indices[i];
            double damp = (double)jnt_damping[ji];
            if (damp > 0.0) M[i*na + i] -= (double)timestep * damp;
        }
    }
    __syncthreads();

    // ── 5. M inverse via Cholesky ───────────────────────────────────────
    // cholesky_d(L, na), one lane per row j of the current column i. The
    // k-reductions stay serial in-lane, so this is the scalar order exactly.
    for (int i = 0; i < na; i++) {
        if (t == 0) {
            double s = L[i*na + i];
            for (int k = 0; k < i; k++) s -= L[i*na+k]*L[i*na+k];
            L[i*na+i] = (s > 0.0) ? sqrt(s) : 1e-10;
        }
        __syncthreads();
        if (t > i && t < na) {
            const int j = t;
            double s = L[j*na + i];
            for (int k = 0; k < i; k++) s -= L[j*na+k]*L[i*na+k];
            L[j*na+i] = s / L[i*na+i];
        }
        __syncthreads();
    }
    // cholesky_inv_d(L, M_inv, na), one lane per column j.
    for (int i = 0; i < na; i++) {
        if (t < na) {
            const int j = t;
            double v = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) v -= L[i*na+k]*M_inv[k*na+j];
            M_inv[i*na+j] = v / L[i*na+i];
        }
        __syncthreads();
    }
    for (int i = na-1; i >= 0; i--) {
        if (t < na) {
            const int j = t;
            double v = M_inv[i*na+j];
            for (int k = i+1; k < na; k++) v -= L[k*na+i]*M_inv[k*na+j];
            M_inv[i*na+j] = v / L[i*na+i];
        }
        __syncthreads();
    }

    // ── 6. J·M_inv products — one lane per output element ────────────────
    // JM_pos = J_pos @ M_inv  (3 × na) ; JM_ori = J_ori @ M_inv  (3 × na)
    for (int o = t; o < 3*na; o += OSC_BLOCK) {
        const int r = o / na, c = o - r*na;
        double ap = 0.0, ao = 0.0;
        for (int k = 0; k < na; k++) {
            ap += J_pos[r*OSC_MAX_ARM+k] * M_inv[k*na+c];
            ao += J_ori[r*OSC_MAX_ARM+k] * M_inv[k*na+c];
        }
        JM_pos[o] = ap; JM_ori[o] = ao;
    }
    // T = M_inv @ J_full^T  (na × 6); J_full is 6×na (rows 0-2 = J_pos, 3-5 = J_ori)
    for (int o = t; o < na*6; o += OSC_BLOCK) {
        const int r = o / 6, c = o - r*6;
        double a = 0.0;
        for (int k = 0; k < na; k++) {
            double jt_kc = (c < 3) ? J_pos[c*OSC_MAX_ARM+k] : J_ori[(c-3)*OSC_MAX_ARM+k];
            a += M_inv[r*na+k] * jt_kc;
        }
        T[o] = a;
    }
    // Velocity via Jacobian (step 8 in the scalar order; independent of M_inv).
    if (t >= 56 && t < 62) {
        const int o = t - 56;
        const int r = (o < 3) ? o : (o - 3);
        double a = 0.0;
        for (int k = 0; k < na; k++)
            a += ((o < 3) ? J_pos[r*OSC_MAX_ARM+k] : J_ori[r*OSC_MAX_ARM+k]) * arm_qvel[k];
        vel[o] = a;
    }
    __syncthreads();

    // JMJ_pos = JM_pos @ J_pos^T  (3×3) ; JMJ_ori likewise
    for (int o = t; o < 9; o += OSC_BLOCK) {
        const int r = o / 3, c = o - r*3;
        double ap = 0.0, ao = 0.0;
        for (int k = 0; k < na; k++) {
            ap += JM_pos[r*na+k] * J_pos[c*OSC_MAX_ARM+k];
            ao += JM_ori[r*na+k] * J_ori[c*OSC_MAX_ARM+k];
        }
        JMJ_pos[o] = ap; JMJ_ori[o] = ao;
    }
    // JMJ_full = J_full @ T  (6×6)
    for (int o = 9 + t; o < 9 + 36; o += OSC_BLOCK) {
        const int q = o - 9;
        const int r = q / 6, c = q - r*6;
        double a = 0.0;
        for (int k = 0; k < na; k++) {
            double j_rk = (r < 3) ? J_pos[r*OSC_MAX_ARM+k] : J_ori[(r-3)*OSC_MAX_ARM+k];
            a += j_rk * T[k*6+c];
        }
        JMJ_full[q] = a;
        Lambda_full_L[q] = a;
    }
    if (use_cached_tick) {
        if (t < 6) vel[t] = es.cached_ee_vel[t];
        if (t == 0) es.first_tick_cache_pending = 0;
    }
    __syncthreads();

    // ── 7. Lambda (operational space inertia) ───────────────────────────
    if (t == 0) inv3_d(JMJ_pos, Lambda_pos);
    if (t == 1) inv3_d(JMJ_ori, Lambda_ori);
    // cholesky_d(Lambda_full_L, 6) then cholesky_inv_d(.., Lambda_full, 6),
    // same lane-per-row / lane-per-column split as the na case above. Runs on
    // lanes 8..15 so it does not serialise behind the two inv3_d lanes.
    const int u = t - 8;
    for (int i = 0; i < 6; i++) {
        if (u == 0) {
            double s = Lambda_full_L[i*6 + i];
            for (int k = 0; k < i; k++) s -= Lambda_full_L[i*6+k]*Lambda_full_L[i*6+k];
            Lambda_full_L[i*6+i] = (s > 0.0) ? sqrt(s) : 1e-10;
        }
        __syncthreads();
        if (u > i && u < 6) {
            const int j = u;
            double s = Lambda_full_L[j*6 + i];
            for (int k = 0; k < i; k++) s -= Lambda_full_L[j*6+k]*Lambda_full_L[i*6+k];
            Lambda_full_L[j*6+i] = s / Lambda_full_L[i*6+i];
        }
        __syncthreads();
    }
    for (int i = 0; i < 6; i++) {
        if (u >= 0 && u < 6) {
            const int j = u;
            double v = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k < i; k++) v -= Lambda_full_L[i*6+k]*Lambda_full[k*6+j];
            Lambda_full[i*6+j] = v / Lambda_full_L[i*6+i];
        }
        __syncthreads();
    }
    for (int i = 5; i >= 0; i--) {
        if (u >= 0 && u < 6) {
            const int j = u;
            double v = Lambda_full[i*6+j];
            for (int k = i+1; k < 6; k++) v -= Lambda_full_L[k*6+i]*Lambda_full[k*6+j];
            Lambda_full[i*6+j] = v / Lambda_full_L[i*6+i];
        }
        __syncthreads();
    }

    // ── 9. PD control ───────────────────────────────────────────────────
    if (t == 0) {
        double pos_err[3], ori_err[3];
        for (int k = 0; k < 3; k++)
            pos_err[k] = es.goal_pos[k] - eef_pos[k];
        orientation_error_d(es.goal_ori, eef_mat, ori_err);

        double F_des[3], tau_des[3];
        for (int k = 0; k < 3; k++) {
            F_des[k]   = cfg->kp_pos * pos_err[k] - cfg->kd_pos * vel[k];
            tau_des[k]  = cfg->kp_ori * ori_err[k] - cfg->kd_ori * vel[k+3];
        }

        // Decoupled wrench = [Lambda_pos @ F_des ; Lambda_ori @ tau_des]
        for (int r = 0; r < 3; r++) {
            double wp = 0.0, wo = 0.0;
            for (int k = 0; k < 3; k++) {
                wp += Lambda_pos[r*3+k] * F_des[k];
                wo += Lambda_ori[r*3+k] * tau_des[k];
            }
            wrench[r] = wp; wrench[r+3] = wo;
        }
    }
    // ── 12a. J_bar = M_inv @ J_full^T @ Lambda_full   (na × 6) ──────────
    // T (na×6) is M_inv @ J_full^T, already computed above.
    for (int o = t; o < na*6; o += OSC_BLOCK) {
        const int r = o / 6, c = o - r*6;
        double a = 0.0;
        for (int k = 0; k < 6; k++) a += T[r*6+k] * Lambda_full[k*6+c];
        J_bar[o] = a;
    }
    __syncthreads();

    // ── 10./11. Torques = J_full^T @ wrench, then bias compensation ─────
    if (t < na) {
        const int i = t;
        double a = 0.0;
        for (int k = 0; k < 3; k++) {
            a += J_pos[k*OSC_MAX_ARM+i] * wrench[k];
            a += J_ori[k*OSC_MAX_ARM+i] * wrench[k+3];
        }
        if (qfrc_bias)
            a += qfrc_bias[(size_t)env*nv + cfg->arm_dof_indices[i]];
        torques[i] = a;
        // error_null = kp_null*(init_jpos - qpos) - kd_null*qvel
        e_null[i] = cfg->kp_null*(es.init_jpos[i] - arm_qpos[i]) - cfg->kd_null*arm_qvel[i];
    }
    // N = I - J_bar @ J_full   (na × na)
    for (int o = t; o < na*na; o += OSC_BLOCK) {
        const int r = o / na, c = o - r*na;
        double a = (r == c) ? 1.0 : 0.0;
        for (int k = 0; k < 6; k++) {
            double j_kc = (k < 3) ? J_pos[k*OSC_MAX_ARM+c] : J_ori[(k-3)*OSC_MAX_ARM+c];
            a -= J_bar[r*6+k] * j_kc;
        }
        N[o] = a;
    }
    __syncthreads();

    // M_e = M @ error_null
    if (t < na) {
        double a = 0.0;
        for (int k = 0; k < na; k++) a += M[t*na+k] * e_null[k];
        Me[t] = a;
    }
    __syncthreads();

    // null_torques = N^T @ M_e, then clip and write arm ctrl
    if (t < na) {
        const int i = t;
        double a = torques[i];
        for (int k = 0; k < na; k++) a += N[k*na+i] * Me[k];
        int ai = cfg->arm_actuator_indices[i];
        if (ai < nu) {
            double lim = cfg->arm_ctrl_limits[i];
            double t_clamp = a;
            if (t_clamp >  lim) t_clamp =  lim;
            if (t_clamp < -lim) t_clamp = -lim;
            ctrl_d[env*nu + ai] = t_clamp;          // solver reads this
            ctrl[env*nu + ai]   = (float)t_clamp;   // integrator reads this
        }
    }

    // ── 14. Gripper accumulator (updated every substep) ─────────────────
    // Serial on lane 0: n_gripper <= 2 and it mutates persistent env state.
    if (t == 0) {
        double dir = es.gripper_direction;
        double spd = cfg->gripper_speed;
        double signs[2] = { -1.0, 1.0 };
        for (int g = 0; g < cfg->n_gripper; g++) {
            // Stage 1: accumulate in NORMALISED space, clamp to [-1, +1].
            double acc = es.gripper_current_action[g] + signs[g] * spd * dir;
            if (acc < -1.0) acc = -1.0;
            if (acc >  1.0) acc =  1.0;
            es.gripper_current_action[g] = acc;

            // Stage 2: affine map into this actuator's ctrlrange.
            double out = acc;
            if (cfg->has_gripper_ctrlrange) {
                double lo = cfg->gripper_ctrlrange[g*2+0];
                double hi = cfg->gripper_ctrlrange[g*2+1];
                out = 0.5*(hi + lo) + 0.5*(hi - lo) * acc;
            }
            int gi = cfg->gripper_actuator_indices[g];
            if (gi < nu) {
                ctrl_d[env*nu + gi] = out;              // solver reads this
                ctrl[env*nu + gi]   = (float)out;       // integrator reads this
            }
        }
    }
}

__global__ __launch_bounds__(OSC_BLOCK) void kernel_osc_controller(
    int num_envs, int nbody, int nq, int nv, int nsite, int nu,
    float timestep, int enable_damping,
    const int*   __restrict__ jnt_type,
    const int*   __restrict__ jnt_body,
    const double* __restrict__ jnt_axis,
    const double* __restrict__ jnt_pos,
    const float* __restrict__ jnt_damping,
    const double* __restrict__ xpos,
    const double* __restrict__ xquat,
    const int* __restrict__ site_body,
    const double* __restrict__ site_pos,
    const double* __restrict__ site_quat,
    const double* __restrict__ qpos,
    const double* __restrict__ qvel,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ solver_M,
    float*        __restrict__ ctrl,
    double*       __restrict__ ctrl_d,
    const OSCConfig*  __restrict__ cfg,
    OSCEnvState*  env_state)
{
    const int env = blockIdx.x;
    if (env >= num_envs) return;
    if (cfg->n_arm == OSC_MAX_ARM) {
        OSCEnvState& es = env_state[env];
        const bool use_cached_tick = es.first_tick_cache_pending != 0;
        osc_controller_fast7(env, (int)threadIdx.x, nbody, nq, nv, nu,
            jnt_type, jnt_body, jnt_axis, jnt_pos, xpos, xquat,
            site_body, site_pos, site_quat, qpos, qvel, qfrc_bias, solver_M,
            ctrl, ctrl_d, cfg, es, use_cached_tick);
        return;
    }
    osc_controller_generic(num_envs, nbody, nq, nv, nsite, nu, timestep, enable_damping,
        jnt_type, jnt_body, jnt_axis, jnt_pos, jnt_damping, xpos, xquat,
        site_body, site_pos, site_quat, qpos, qvel, qfrc_bias, solver_M,
        ctrl, ctrl_d, cfg, env_state);
}
// ── Launch wrappers ─────────────────────────────────────────────────────────

void launch_osc_process_actions(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const OSCConfig* d_config,
    OSCEnvState* d_env_state,
    const float* d_actions,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0) return;
    kernel_osc_process_actions<<<ne, 1, 0, stream>>>(
        ne, model.nbody, model.nq, model.nv, model.nsite, model.nu,
        data.xpos, data.xquat, data.site_xpos, data.site_xmat,
        data.qpos,
        d_config, d_env_state, d_actions);
}

void launch_osc_controller(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const double* solver_M,
    const OSCConfig* d_config,
    OSCEnvState* d_env_state,
    hipStream_t stream)
{
    int ne = data.num_envs;
    if (ne <= 0) return;
    kernel_osc_controller<<<ne, OSC_BLOCK, 0, stream>>>(
        ne, model.nbody, model.nq, model.nv, model.nsite, model.nu,
        model.timestep, 1,
        model.jnt_type, model.jnt_body, model.jnt_axis_d, model.jnt_pos_d,
        model.jnt_damping,
        data.xpos_d, data.xquat_d, model.site_body, model.site_pos_d, model.site_quat_d,
        data.qpos_d, data.qvel_d, data.qfrc_bias, solver_M,
        data.ctrl, data.ctrl_d,
        d_config, d_env_state);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
