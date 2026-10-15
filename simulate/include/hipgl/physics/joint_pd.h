#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GPU joint-position PD controller with mass-matrix scaling and bias compensation, evaluated inside
// every physics substep (robosuite's JointPositionController, controllers/parts/generic/joint_pos.py):
//   tau_i = sum_{j in group(i)} M[dof_i][dof_j] * (kp_j (target_j - qpos_j) - kd_j qvel_j) + qfrc_bias[dof_i]
//   ctrl[act_i] = clip(tau_i / gear_i, lo_i, hi_i)
// M and qfrc_bias are the substep's own (as robosuite reads them from mj_step1), so a whole control
// step of substeps runs on the GPU with fixed targets (set once per policy step).
namespace hipgl { namespace physics {
struct DeviceJointPd {
    int n = 0;                         // controlled actuators
    int*    act = nullptr;             // [n] actuator index
    int*    dof = nullptr;             // [n] dof index
    int*    qadr = nullptr;            // [n] qpos address
    int*    grp_start = nullptr;       // [n] first PD index of this actuator's group
    int*    grp_len = nullptr;         // [n] group size
    double* kp = nullptr;              // [n]
    double* kd = nullptr;              // [n]
    double* gear = nullptr;            // [n]
    double* lo = nullptr;              // [n] ctrlrange low
    double* hi = nullptr;              // [n] ctrlrange high
    double* target = nullptr;          // [num_envs * n] joint position goals
};
}}
