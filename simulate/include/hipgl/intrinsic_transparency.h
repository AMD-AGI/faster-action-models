#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "physics/physics_renderer.h"
#include "pipeline.h"
#include <vector>

namespace hipgl {

struct TransparentGeomSpan {
    int geom_id;
    float camdist;
};

// MuJoCo 3.3.2: intrinsic alpha in (0, .995) is transparent, irrespective
// of mjVIS_TRANSPARENT. Return a stable ascending radial-distance order;
// planes have camdist=1e10. Only generated, visible groups 0..2 participate.
// body_xpos/body_xquat describe ONE world. Optional compiled metadata is
// body-local float geom_pos[ngeom*3] and float geom_rbound[ngeom], mapped by
// exact geom identity by the caller. Supplying both avoids the native mesh
// origin/bounding-sphere approximation; it does not change native geometry.
// A reflection pass must reuse the ORIGINAL camera's order, like MuJoCo.
std::vector<TransparentGeomSpan> make_transparent_draw_order(
    const physics::PhysicsModelDesc& desc, const RenderState& state,
    const float* body_xpos, const float* body_xquat, const bool* geomgroup,
    const float* compiled_geom_pos = nullptr,
    const float* compiled_rbound = nullptr);

// Draw visible per-geom spans, never the cross-group transparent range.
// MuJoCo 3.3.2 render_gl3.c uses ascending then reverse order, with BACK
// culling in both passes (no cull swap). Preserve the supplied front_face,
// reflection mask/clip/matrices, and restore the supplied state afterward.
// The caller must build these visible geoms and disable deferred background
// before drawing. This implements intrinsic alpha, NOT global dynamic fade.
void draw_intrinsic_transparency(
    Pipeline& pipeline, const physics::PhysicsRenderer& geometry,
    const RenderState& state, const TextureState& texture,
    const std::vector<TransparentGeomSpan>& order);

}  // namespace hipgl
