#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "physics/physics_renderer.h"
#include "pipeline.h"
#include "intrinsic_transparency.h"
#include <array>
#include <functional>

namespace hipgl {

// A plan owns the host matrix bytes used by asynchronous uploads. Keep it alive
// until the render stream finishes; batches retain one plan per environment.
struct PlanarReflectionPlan {
    int geom_id = -1;
    bool is_box = false;
    bool render_scene = false;
    RenderState reflected_state{};
    std::array<float, 41> matrices{};
};

PlanarReflectionPlan make_planar_reflection_plan(
    const physics::PhysicsModelDesc& desc, const RenderState& state,
    const float* body_xpos, const float* body_xquat,
    const bool* geomgroup, bool reflection_enabled);

// Each in-flight worker owns its sample mask and matrix upload destination.
// No framebuffer or geometry is copied to the host by the reflection pass.
class PlanarReflectionResources {
public:
    PlanarReflectionResources() = default;
    ~PlanarReflectionResources();
    PlanarReflectionResources(const PlanarReflectionResources&) = delete;
    PlanarReflectionResources& operator=(const PlanarReflectionResources&) = delete;
    void ensure(size_t pixels);
    uint8_t* mask() const { return mask_; }
    float* matrices() const { return matrices_; }
private:
    uint8_t* mask_ = nullptr;
    float* matrices_ = nullptr;
    size_t capacity_ = 0;
};

// The framebuffer has already been cleared. Render a MuJoCo planar mirror,
// its additive base, and then the remaining opaque geoms. An empty plan takes
// the original single-pass path. external_matrices is the caller's ordinary
// camera matrix block, restored before returning.
void draw_opaque_with_planar_reflection(
    Pipeline& pipeline, Framebuffer& framebuffer,
    const physics::PhysicsRenderer& geometry, const RenderState& state,
    const TextureState& texture, const bool* geomgroup,
    const std::vector<TransparentGeomSpan>& transparent_order,
    const PlanarReflectionPlan& plan, PlanarReflectionResources& resources,
    const float* external_matrices,
    const float* reflect_matrices,
    const std::function<void(const RenderState&)>& draw_reflected_sites = {});

namespace kernels {
void launch_reflection_sample_mask(DeviceFramebuffer framebuffer,
                                   uint8_t* mask, float clear_depth,
                                   hipStream_t stream);
bool launch_reflection_mask_and_clear(DeviceFramebuffer fb, uint8_t* mask,
                                      float mask_ref, float r, float g, float b,
                                      float a, float clear_depth,
                                      hipStream_t stream);
}

}  // namespace hipgl
