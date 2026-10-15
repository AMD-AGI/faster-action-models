// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/intrinsic_transparency.h"
#include "hipgl/math_types.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hipgl {
namespace {
using math::Vec3;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

bool finite3(const float* p) {
    return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

Vec3 rotate_quat(const float* q, Vec3 v) {
    const Vec3 axis(q[1], q[2], q[3]);
    const Vec3 t = axis.cross(v) * 2.0f;
    return v + t * q[0] + axis.cross(t);
}

float native_rbound(const physics::PhysicsModelDesc& desc,
                    const physics::GeomDesc& g) {
    using namespace physics;
    const float x = g.size[0], y = g.size[1], z = g.size[2];
    switch (g.type) {
    case GEOM_SPHERE: return x;
    case GEOM_CAPSULE: return x + y;
    case GEOM_CYLINDER: return std::sqrt(x*x + y*y);
    case GEOM_BOX: return std::sqrt(x*x + y*y + z*z);
    case GEOM_PLANE: return 0.0f;  // the sort key is the plane sentinel
    case GEOM_MESH: {
        // Native vertices already include MJCF mesh scale. Their maximum
        // norm bounds the mesh around its native geom origin; no second
        // scale or MuJoCo mesh-frame transform is applied here. This is a
        // conservative fallback, NOT MuJoCo's compiled geom_rbound/centre.
        require(g.mesh_id >= 0 && g.mesh_id < desc.nmesh && desc.mesh_vert,
                "Transparent mesh has no native bounding data");
        const auto& mesh = desc.meshes[g.mesh_id];
        require(mesh.vert_adr >= 0 && mesh.vert_num > 0 &&
                    mesh.vert_adr <= desc.mesh_total_verts &&
                    mesh.vert_num <= desc.mesh_total_verts - mesh.vert_adr,
                "Transparent mesh vertex bounds are invalid");
        float max_squared = 0.0f;
        for (int i = 0; i < mesh.vert_num; ++i) {
            const float* p = desc.mesh_vert + (mesh.vert_adr + i) * 3;
            require(finite3(p), "Transparent mesh contains nonfinite vertices");
            max_squared = std::max(max_squared, Vec3(p).length_sq());
        }
        return std::sqrt(max_squared);
    }
    default:
        throw std::runtime_error("Unsupported transparent geom type");
    }
}

void draw_span(Pipeline& pipeline, const physics::PhysicsRenderer& geometry,
               const TextureState& texture, int geom_id) {
    const int offset = geometry.geom_vert_offset_host(geom_id);
    const int count = geometry.geom_vert_count_host(geom_id);
    if (count <= 0) return;
    pipeline.draw_arrays_device_phong(PRIM_TRIANGLES,
        geometry.device_positions() + offset * 3,
        geometry.device_normals() + offset * 3,
        geometry.device_texcoords() + offset * TEXCOORD_COMPONENTS,
        geometry.device_texids() + offset,
        geometry.device_colors() + offset * 4,
        geometry.device_specular() + offset,
        geometry.device_shininess() + offset,
        geometry.device_emission() + offset, count, texture);
}
}  // namespace

std::vector<TransparentGeomSpan> make_transparent_draw_order(
    const physics::PhysicsModelDesc& desc, const RenderState& state,
    const float* body_xpos, const float* body_xquat, const bool* geomgroup,
    const float* compiled_geom_pos, const float* compiled_rbound) {
    require(body_xpos && body_xquat && geomgroup, "Missing transparent scene state");
    require(finite3(state.camera_world_pos), "Nonfinite transparent camera position");
    std::vector<TransparentGeomSpan> order;
    for (int i = 0; i < desc.ngeom; ++i) {
        const auto& g = desc.geoms[i];
        if (g.group < 0 || g.group >= 3 || !geomgroup[g.group]) continue;
        require(std::isfinite(g.rgba[3]), "Nonfinite intrinsic geom alpha");
        if (g.rgba[3] <= 0.0f || g.rgba[3] >= 0.995f) continue;
        if (g.type == physics::GEOM_PLANE) {
            order.push_back({i, 1.0e10f});
            continue;
        }
        require(g.body >= 0 && g.body < desc.nbody, "Invalid transparent geom body");
        const float* pos = compiled_geom_pos ? compiled_geom_pos + i * 3 : g.pos;
        const float* bp = body_xpos + g.body * 3;
        const float* bq = body_xquat + g.body * 4;
        require(finite3(pos) && finite3(bp) && finite3(bq) && std::isfinite(bq[3]),
                "Nonfinite transparent geom transform");
        float radius = compiled_rbound ? compiled_rbound[i] : native_rbound(desc, g);
        if (!compiled_rbound && compiled_geom_pos) {
            require(finite3(g.pos), "Nonfinite native geom origin");
            // Keep the fallback conservative if only a compiled centre was
            // supplied. Exact reference ordering requires compiled_rbound too.
            radius += (Vec3(g.pos) - Vec3(pos)).length();
        }
        require(std::isfinite(radius) && radius >= 0.0f, "Invalid transparent rbound");
        const Vec3 centre = Vec3(bp) + rotate_quat(bq, Vec3(pos));
        const Vec3 delta = centre - Vec3(state.camera_world_pos);
        const float camdist = std::sqrt(delta.dot(delta)) - radius;
        require(std::isfinite(camdist), "Nonfinite transparent sort distance");
        order.push_back({i, camdist});
    }
    std::stable_sort(order.begin(), order.end(),
        [](const TransparentGeomSpan& a, const TransparentGeomSpan& b) {
            return a.camdist < b.camdist;
        });
    return order;
}

void draw_intrinsic_transparency(
    Pipeline& pipeline, const physics::PhysicsRenderer& geometry,
    const RenderState& state, const TextureState& texture,
    const std::vector<TransparentGeomSpan>& order) {
    if (order.empty()) return;
    // Validate before any submission. Inactive geometry can retain stale
    // vertices, so the caller's visible list is the only draw authority.
    for (const auto& span : order) {
        require(span.geom_id >= 0 && span.geom_id < geometry.ngeom(),
                "Invalid transparent draw geom id");
        const int offset = geometry.geom_vert_offset_host(span.geom_id);
        const int count = geometry.geom_vert_count_host(span.geom_id);
        if (count <= 0) continue;
        const int begin = geometry.transparent_offset();
        const int total = geometry.transparent_count();
        require(offset >= begin && count % 3 == 0 && count <= total &&
                    offset - begin <= total - count,
                "Transparent draw span is outside the intrinsic-alpha range");
    }
    RenderState blend = state;
    blend.depth_test_enabled = true;
    blend.depth_write_enabled = false;
    blend.blend_enabled = true;
    blend.blend_src = BLEND_SRC_ALPHA;
    blend.blend_dst = BLEND_ONE_MINUS_SRC_ALPHA;
    blend.oit_enabled = false;
    blend.backface_cull_enabled = true;
    blend.cull_face = 0;  // BACK, preserving front_face (also in reflections)
    pipeline.set_render_state(blend);
    try {
        for (const auto& span : order) draw_span(pipeline, geometry, texture, span.geom_id);
        for (auto it = order.rbegin(); it != order.rend(); ++it)
            draw_span(pipeline, geometry, texture, it->geom_id);
    } catch (...) {
        pipeline.set_render_state(state);
        throw;
    }
    pipeline.set_render_state(state);
}

}  // namespace hipgl
