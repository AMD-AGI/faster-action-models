#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "physics_types.h"

namespace hipgl {
namespace physics {

// Manages per-environment dynamic physics state on GPU.
// Supports batched allocation for N simultaneous environments.
class PhysicsData {
public:
    PhysicsData();
    ~PhysicsData();

    // Allocate device memory for num_envs environments given the model dims.
    bool allocate(const PhysicsModelDesc& desc, int num_envs);

    // Reset all environments to the model's default qpos (qpos0) and zero velocities.
    void reset(const PhysicsModelDesc& desc);

    // Reset a single environment.
    void reset_env(const PhysicsModelDesc& desc, int env_idx);

    // Copy qpos for all envs from device to host-provided buffer.
    void read_qpos(float* host_buf) const;

    // Copy qvel for all envs from device to host-provided buffer.
    void read_qvel(float* host_buf) const;

    void read_qpos_d(double* host_buf) const;
    void read_qvel_d(double* host_buf) const;
    void write_qpos_d(const double* host_buf);
    void write_qpos_d_env(const double* host_buf, int env_idx);
    void write_qvel_d(const double* host_buf);
    void write_qvel_d_env(const double* host_buf, int env_idx);

    // Copy body world positions for all envs from device to host.
    void read_xpos(float* host_buf) const;

    // Copy body world orientations for all envs from device to host.
    void read_xquat(float* host_buf) const;

    // Copy RNEA bias forces (gravity + Coriolis) for all envs from device to host.
    void read_qfrc_bias(double* host_buf) const;

    // Copy site world positions for all envs from device to host.
    void read_site_xpos(float* host_buf) const;

    // Copy site world rotation matrices for all envs from device to host.
    void read_site_xmat(float* host_buf) const;

    // Write qpos from host to device for all envs.
    void write_qpos(const float* host_buf);

    // Write qpos for a single env.
    void write_qpos_env(const float* host_buf, int env_idx);

    // Write qvel from host to device for all envs.
    void write_qvel(const float* host_buf);

    // Write qvel for a single env.
    void write_qvel_env(const float* host_buf, int env_idx);

    // Write control inputs from host to device for all envs.
    void write_ctrl(const float* host_buf);

    // Write control inputs for a single env.
    void write_ctrl_env(const float* host_buf, int env_idx);

    const DevicePhysicsData& device_data() const { return d_data_; }
    DevicePhysicsData& mutable_device_data() { return d_data_; }
    int num_envs() const { return d_data_.num_envs; }

private:
    DevicePhysicsData d_data_;
    bool              allocated_;

    void free_device();
};

} // namespace physics
} // namespace hipgl
