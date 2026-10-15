#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "physics_types.h"
#include <string>
#include <memory>
#include <vector>

namespace hipgl {
namespace physics {

// Parses MJCF XML into PhysicsModelDesc, then uploads to GPU as DevicePhysicsModel.
class PhysicsModel {
public:
    PhysicsModel();
    ~PhysicsModel();

    // Parse MJCF XML string and upload to GPU.
    // meshdir: base directory for resolving mesh file paths (empty = use <compiler meshdir>)
    bool load_from_xml(const std::string& xml_string, const std::string& meshdir = "");

    // Access the host-side description (for inspection / validation).
    const PhysicsModelDesc& desc() const { return desc_; }
    PhysicsModelDesc& mutable_desc_for_debug() { return desc_; }
    PhysicsModelDesc& desc_mut() { return desc_; }

    // Access the device-side model (for passing to kernels).
    const DevicePhysicsModel& device_model() const { return d_model_; }
    DevicePhysicsModel& mutable_device_model() { return d_model_; }

    // Re-upload invweight arrays to device after they've been set in desc.
    bool upload_invweight();

    // Upload texture data to device (call after set_textures).
    bool upload_textures();

    // Move bodies that are fixed to the world (no joint in their ancestry).
    // Updates the host descriptor and existing float/double device arrays;
    // meshes, textures, model topology, and allocation addresses are unchanged.
    // This is MODEL-wide, not per-env. Caller must exclude concurrent stepping
    // and then reset stale contact/warmstart state and run forward().
    // Quaternions are unit-length wxyz. Invalid requests throw before mutation.
    void set_fixed_body_poses(const std::vector<int>& body_ids,
                              const std::vector<double>& positions,
                              const std::vector<double>& quaternions);

    // Per-world variant. The first call allocates per-world pose arrays for
    // num_envs worlds (seeded from the model-wide poses); later calls must pass
    // the same num_envs. Only world `env` moves. set_fixed_body_poses() above
    // stays model-wide and then writes every world. Same caller contract.
    void set_fixed_body_poses_env(int num_envs, int env,
                                  const std::vector<int>& body_ids,
                                  const std::vector<double>& positions,
                                  const std::vector<double>& quaternions);
    int body_pose_nenv() const { return uploaded_ ? d_model_.body_pose_nenv : 0; }

    // Convenience accessors
    int nbody() const { return desc_.nbody; }
    int njnt()  const { return desc_.njnt; }
    int ngeom() const { return desc_.ngeom; }
    int nq()    const { return desc_.nq; }
    int nv()    const { return desc_.nv; }
    int nu()    const { return desc_.nu; }
    int nsite() const { return desc_.nsite; }

private:
    // env < 0: model-wide (shared arrays and every per-world slice).
    void apply_fixed_body_poses(int num_envs, int env,
                                const std::vector<int>& body_ids,
                                const std::vector<double>& positions,
                                const std::vector<double>& quaternions);

    PhysicsModelDesc   desc_;
    DevicePhysicsModel d_model_;
    bool               uploaded_;

    // Build BFS tree levels from parent array
    void build_tree_levels();

    // Upload desc_ arrays to device, populating d_model_
    bool upload_to_device();

    // Free all device allocations
    void free_device();
};

} // namespace physics
} // namespace hipgl
