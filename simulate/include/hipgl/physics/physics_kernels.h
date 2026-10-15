#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Forward declarations for physics kernel launch wrappers.

#include "physics_types.h"

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
typedef void* hipStream_t;
#endif

namespace hipgl {
namespace physics {
namespace kernels {

// Forward kinematics (one call per tree level)
void launch_forward_kinematics(int level_start, int level_count, int num_envs,
                                const DevicePhysicsModel& model,
                                const DevicePhysicsData& data,
                                hipStream_t stream);

// Geom transforms (body world -> geom world)
void launch_geom_transforms(const DevicePhysicsModel& model,
                             const DevicePhysicsData& data,
                             hipStream_t stream);

// Compute AABBs for broadphase
void launch_compute_aabbs(const DevicePhysicsModel& model,
                           const DevicePhysicsData& data,
                           hipStream_t stream);

// Pre-compute inverse mass/inertia
void launch_compute_inv_mass(const DevicePhysicsModel& model,
                              DevicePhysicsData& data,
                              hipStream_t stream);

// Fluid drag forces (MuJoCo inertia-based model, opt.density/viscosity)
void launch_compute_fluid_forces(const DevicePhysicsModel& model,
                                  DevicePhysicsData& data,
                                  hipStream_t stream,
                                  const int* h_level_start_cached = nullptr);

// Apply forces (gravity, actuators, damping, constraints)
void launch_apply_forces(const DevicePhysicsModel& model,
                          const DevicePhysicsData& data,
                          const SolverOptions& opts,
                          hipStream_t stream);

// Semi-implicit Euler integration
void launch_integrate(const DevicePhysicsModel& model,
                       DevicePhysicsData& data,
                       const SolverOptions& opts,
                       hipStream_t stream);

// Broadphase collision detection
void launch_broadphase(const DevicePhysicsModel& model,
                        DevicePhysicsData& data,
                        const SolverOptions& opts,
                        hipStream_t stream);

// Narrowphase collision detection (fused with broadphase)
void launch_narrowphase(const DevicePhysicsModel& model,
                         DevicePhysicsData& data,
                         const SolverOptions& opts,
                         hipStream_t stream);

// Build constraint rows from contacts + joint limits
void launch_constraint_setup(const DevicePhysicsModel& model,
                              DevicePhysicsData& data,
                              const SolverOptions& opts,
                              hipStream_t stream);

// PGS iterative constraint solver
void launch_pgs_solve(const DevicePhysicsModel& model,
                       DevicePhysicsData& data,
                       const SolverOptions& opts,
                       hipStream_t stream);

} // namespace kernels
} // namespace physics
} // namespace hipgl
