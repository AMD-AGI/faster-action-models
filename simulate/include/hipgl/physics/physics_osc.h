#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "physics_types.h"

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

namespace hipgl {
namespace physics {

constexpr int OSC_MAX_ARM  = 7;
constexpr int OSC_MAX_GRIP = 2;

struct OSCConfig {
    int n_arm;
    int arm_joint_indices[OSC_MAX_ARM];
    int arm_dof_indices[OSC_MAX_ARM];
    int arm_qpos_indices[OSC_MAX_ARM];
    int arm_actuator_indices[OSC_MAX_ARM];

    int n_gripper;
    int gripper_actuator_indices[OSC_MAX_GRIP];

    int eef_site_idx;   // >= 0 to use site, -1 to fall back to body
    int eef_body_idx;

    double kp_pos, kp_ori;
    double kd_pos, kd_ori;
    double kp_null, kd_null;
    double action_scale_pos, action_scale_ori;
    double arm_ctrl_limits[OSC_MAX_ARM];

    double gripper_speed;
    double gripper_ctrlrange[OSC_MAX_GRIP * 2]; // [lo0, hi0, lo1, hi1]
    int    has_gripper_ctrlrange;
};

struct OSCEnvState {
    double goal_pos[3];
    double goal_ori[9];   // 3x3 row-major rotation matrix
    double init_jpos[OSC_MAX_ARM];
    double gripper_current_action[OSC_MAX_GRIP];
    double gripper_direction;   // sign(grip_action), set by process_actions
    int    initialized;         // 0 = goals not yet set

    int    first_goal_cache_pending;
    int    first_tick_cache_pending;
    double cached_ee_pos[3];
    double cached_ee_ori[9];
    double cached_ee_vel[6];  // world linear velocity, then angular velocity
    double cached_joint_pos[OSC_MAX_ARM];
    double cached_joint_vel[OSC_MAX_ARM];
    double cached_J_full[6 * OSC_MAX_ARM];  // row-major, stride OSC_MAX_ARM
    double cached_mass_matrix[OSC_MAX_ARM * OSC_MAX_ARM]; // same row stride
};

namespace kernels {

void launch_osc_process_actions(
    const DevicePhysicsModel& model,
    const DevicePhysicsData& data,
    const OSCConfig* d_config,
    OSCEnvState* d_env_state,
    const float* d_actions,
    hipStream_t stream);

void launch_osc_controller(
    const DevicePhysicsModel& model,
    DevicePhysicsData& data,
    const double* solver_M,
    const OSCConfig* d_config,
    OSCEnvState* d_env_state,
    hipStream_t stream);

} // namespace kernels
} // namespace physics
} // namespace hipgl
