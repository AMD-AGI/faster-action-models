// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GPU joint-position PD controller (see include/hipgl/physics/joint_pd.h).
#include <hip/hip_runtime.h>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/joint_pd.h"

namespace hipgl { namespace physics { namespace kernels {

__global__ void kernel_joint_pd(int num_envs, int nv, int nq, int nu, DeviceJointPd pd,
                                const double* __restrict__ solver_M, const double* __restrict__ qfrc_bias,
                                const double* __restrict__ qpos_d, const double* __restrict__ qvel_d,
                                double* __restrict__ ctrl_d, float* __restrict__ ctrl)
{
    const int env = blockIdx.x, i = threadIdx.x;
    if (env >= num_envs || i >= pd.n) return;
    const double* M = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    const double* q = qpos_d + (size_t)env * nq;
    const double* v = qvel_d + (size_t)env * nv;
    const double* tg = pd.target + (size_t)env * pd.n;
    const int di = pd.dof[i];
    double tau = 0.0;
    for (int k = pd.grp_start[i], e = k + pd.grp_len[i]; k < e; k++) {
        const double des = pd.kp[k] * (tg[k] - q[pd.qadr[k]]) - pd.kd[k] * v[pd.dof[k]];
        tau += M[di * KERNEL_MAXNV + pd.dof[k]] * des;
    }
    tau += qfrc_bias[(size_t)env * nv + di];
    double c = tau / pd.gear[i];
    c = fmin(fmax(c, pd.lo[i]), pd.hi[i]);
    ctrl_d[(size_t)env * nu + pd.act[i]] = c;
    ctrl[(size_t)env * nu + pd.act[i]] = (float)c;
}

void launch_joint_pd(const DevicePhysicsModel& model, DevicePhysicsData& data, const DeviceJointPd& pd,
                     hipStream_t stream)
{
    if (pd.n <= 0 || data.num_envs <= 0) return;
    kernel_joint_pd<<<data.num_envs, 64, 0, stream>>>(data.num_envs, model.nv, model.nq, model.nu, pd,
                                                      data.solver_M, data.qfrc_bias, data.qpos_d, data.qvel_d,
                                                      data.ctrl_d, data.ctrl);
}

}}}
