// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/planar_reflection.h"
#include "hipgl/math_types.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <cstdio>

namespace hipgl {
namespace {
using math::Vec3;
using math::Mat4;

Vec3 rotate_quat(const float* q, Vec3 v) {
    Vec3 axis(q[1], q[2], q[3]);
    Vec3 t = axis.cross(v) * 2.0f;
    return v + t * q[0] + axis.cross(t);
}

Mat4 reflection_matrix(Vec3 n, Vec3 point) {
    const float a[3] = {n.x, n.y, n.z};
    Mat4 r = Mat4::identity();
    const float distance = n.dot(point);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            r(i, j) -= 2.0f * a[i] * a[j];
        r(i, 3) = 2.0f * distance * a[i];
    }
    return r;
}

void check_hip(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}

void draw_span(Pipeline& p, const physics::PhysicsRenderer& g,
               const TextureState& texture, int offset, int count) {
    if (count <= 0) return;
    p.draw_arrays_device_phong(PRIM_TRIANGLES,
        g.device_positions() + offset * 3,
        g.device_normals() + offset * 3,
        g.device_texcoords() + offset * TEXCOORD_COMPONENTS,
        g.device_texids() + offset,
        g.device_colors() + offset * 4,
        g.device_specular() + offset,
        g.device_shininess() + offset,
        g.device_emission() + offset, count, texture);
}

void draw_groups(Pipeline& p, const physics::PhysicsRenderer& g,
                 const TextureState& texture, const bool* groups, int exclude) {
    const int skip_begin = exclude >= 0 ? g.geom_vert_offset_host(exclude) : -1;
    const int skip_end = skip_begin + (exclude >= 0 ? g.geom_vert_count_host(exclude) : 0);
    for (int group = 0; group < 3; ++group) {
        if (!groups[group]) continue;
        const int begin = g.group_offset(group);
        const int end = begin + g.group_count(group);
        if (skip_begin >= begin && skip_begin < end) {
            draw_span(p, g, texture, begin, skip_begin - begin);
            draw_span(p, g, texture, skip_end, end - skip_end);
        } else {
            draw_span(p, g, texture, begin, end - begin);
        }
    }
}

void hp_mirror_screen_rect(const physics::PhysicsModelDesc& desc,
                           const physics::GeomDesc& g,
                           const float* q, Vec3 body_position, Vec3 normal,
                           const RenderState& state, int32_t* rect) {
    rect[0] = 0; rect[1] = 0; rect[2] = -1; rect[3] = -1;
    (void)desc; (void)normal;
    if (g.type != physics::GEOM_BOX) return;
    const int vw = state.viewport_w, vh = state.viewport_h;
    if (vw <= 0 || vh <= 0) return;
    // Face centre = geom centre + normal * size[2] (the caller's `point`).
    const Vec3 centre = body_position + rotate_quat(q, Vec3(g.pos[0], g.pos[1], g.pos[2]));
    const Vec3 ex = rotate_quat(q, rotate_quat(g.quat, Vec3(1, 0, 0))) * g.size[0];
    const Vec3 ey = rotate_quat(q, rotate_quat(g.quat, Vec3(0, 1, 0))) * g.size[1];
    const Vec3 ez = rotate_quat(q, rotate_quat(g.quat, Vec3(0, 0, 1))) * g.size[2];
    const float* m = state.mvp;
    float minx = 1e30f, miny = 1e30f, maxx = -1e30f, maxy = -1e30f;
    // The stencil pass draws ONLY the +Z face (planar_reflection.cpp uses
    // count/6 for a box), so the mask can only be non-zero inside that quad --
    // four corners, not the whole box silhouette. HP_REFL_BOX8 restores the
    // (still correct, just looser) whole-box bound as a probe.
    static const int use_box8 = getenv("HP_REFL_BOX8") ? atoi(getenv("HP_REFL_BOX8")) : 0;
    const int ncorner = use_box8 ? 8 : 4;
    for (int c = 0; c < ncorner; ++c) {
        const float sx = (c & 1) ? 1.0f : -1.0f;
        const float sy = (c & 2) ? 1.0f : -1.0f;
        const float sz = use_box8 ? ((c & 4) ? 1.0f : -1.0f) : 1.0f;
        const Vec3 p = centre + ex * sx + ey * sy + ez * sz;
        const float cw = m[3]*p.x + m[7]*p.y + m[11]*p.z + m[15];
        // Anything at/behind the eye makes the projected box meaningless.
        if (!(cw > 1e-4f)) return;
        const float inv = 1.0f / cw;
        const float nx = (m[0]*p.x + m[4]*p.y + m[8]*p.z + m[12]) * inv;
        const float ny = (m[1]*p.x + m[5]*p.y + m[9]*p.z + m[13]) * inv;
        const float scx = state.viewport_x + (nx * 0.5f + 0.5f) * (float)vw;
        const float scy = state.viewport_y + (1.0f - (ny * 0.5f + 0.5f)) * (float)vh;
        if (scx < minx) minx = scx;
        if (scx > maxx) maxx = scx;
        if (scy < miny) miny = scy;
        if (scy > maxy) maxy = scy;
    }
    // Round OUT generously: one pixel of slack on each side absorbs the
    // sub-pixel snap the rasterizer applies to screen_x/screen_y.
    int x0 = (int)std::floor(minx) - 2;
    int y0 = (int)std::floor(miny) - 2;
    int x1 = (int)std::ceil(maxx) + 2;
    int y1 = (int)std::ceil(maxy) + 2;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > vw - 1) x1 = vw - 1;
    if (y1 > vh - 1) y1 = vh - 1;
    if (x0 > x1 || y0 > y1) { rect[2] = -1; return; }
    rect[0] = x0; rect[1] = y0; rect[2] = x1; rect[3] = y1;
    static const int dbg = getenv("HP_REFL_DBG") ? atoi(getenv("HP_REFL_DBG")) : 0;
    if (dbg) {
        static int once = 0;
        if (once < 4) {
            ++once;
            const int TS = TILE_SIZE;
            const int tx = (vw + TS - 1) / TS, ty = (vh + TS - 1) / TS;
            const int rtx = x1 / TS - x0 / TS + 1, rty = y1 / TS - y0 / TS + 1;
            fprintf(stderr, "[REFL] vp=%dx%d rect=(%d,%d)-(%d,%d) px_frac=%.4f "
                            "tiles(%d)=%dx%d of %dx%d frac=%.4f\n",
                    vw, vh, x0, y0, x1, y1,
                    (float)((x1-x0+1)*(y1-y0+1)) / (float)(vw*vh),
                    TS, rtx, rty, tx, ty, (float)(rtx*rty) / (float)(tx*ty));
        }
    }
}

}  // namespace

PlanarReflectionPlan make_planar_reflection_plan(
    const physics::PhysicsModelDesc& desc, const RenderState& state,
    const float* body_xpos, const float* body_xquat,
    const bool* geomgroup, bool reflection_enabled) {
    PlanarReflectionPlan plan;
    // MuJoCo accepts only the first visible opaque plane or box. Mesh material
    // reflectance is not a mirror and must not brighten mesh shading.
    for (int i = 0; i < desc.ngeom; ++i) {
        const auto& g = desc.geoms[i];
        if (g.group < 0 || g.group >= 3 || !geomgroup[g.group] ||
            g.rgba[3] < 0.995f || !(desc.geom_reflectance[i] > 0.0f) ||
            (g.type != physics::GEOM_PLANE && g.type != physics::GEOM_BOX)) continue;
        const float* q = body_xquat + g.body * 4;
        Vec3 body_position(body_xpos[g.body * 3], body_xpos[g.body * 3 + 1], body_xpos[g.body * 3 + 2]);
        Vec3 point = body_position + rotate_quat(q, Vec3(g.pos[0], g.pos[1], g.pos[2]));
        Vec3 normal = rotate_quat(q, rotate_quat(g.quat, Vec3(0, 0, 1))).normalized();
        if (g.type == physics::GEOM_BOX) point = point + normal * g.size[2];
        Vec3 camera(state.camera_world_pos[0], state.camera_world_pos[1], state.camera_world_pos[2]);
        const bool behind = normal.dot(camera - point) < 0.0f;
        // MuJoCo makes a plane transparent when viewed from behind before
        // choosing its first reflector. Boxes retain their opaque material.
        if (g.type == physics::GEOM_PLANE && behind) continue;
        plan.geom_id = i;
        plan.is_box = g.type == physics::GEOM_BOX;
        plan.render_scene = reflection_enabled && !behind;
        static const int hp_reflabl = getenv("HP_REFLABL") ? atoi(getenv("HP_REFLABL")) : 0;
        if (hp_reflabl) plan.render_scene = false;
        if (!plan.render_scene) return plan;

        const Mat4 view = Mat4::from_column_major(state.model_view);
        const Mat4 reflect = reflection_matrix(normal, point);
        const Mat4 reflected_view = view * reflect;
        const Mat4 projection = Mat4::from_column_major(state.projection);
        RenderState& reflected = plan.reflected_state;
        reflected = state;
        hp_mirror_screen_rect(desc, g, q, body_position, normal, state,
                              reflected.reflect_rect);
        reflected_view.to_column_major(reflected.model_view);
        (projection * reflected_view).to_column_major(reflected.mvp);
        // A rigid reflection is orthogonal: its inverse transpose is itself.
        reflected_view.upper3x3(reflected.normal_matrix);
        reflected.front_face = state.front_face ^ 1;
        reflected.shadow_enabled = false;  // MuJoCo's mirror pass enables every light directly.
        reflected.reflect_dim = desc.geom_reflectance[i];
        reflected.clip_plane_enabled = true;

        const Vec3 eye_normal = view.transform_dir(normal).normalized();
        const Vec3 eye_point = view.transform_point(point);
        // Mirrored source geometry lies behind the original mirror plane.
        // The rasterizer keeps dot(clip.xyz, eye_position) >= clip.w.
        reflected.clip_plane[0] = -eye_normal.x;
        reflected.clip_plane[1] = -eye_normal.y;
        reflected.clip_plane[2] = -eye_normal.z;
        reflected.clip_plane[3] = -eye_normal.dot(eye_point);

        // adjustLight runs after mjr_reflect in MuJoCo, including the camera
        // headlight. Reflect the already eye-space positions and directions.
        const Mat4 eye_reflect = reflection_matrix(eye_normal, eye_point);
        for (int li = 0; li < MAX_LIGHTS; ++li) {
            Light& light = reflected.lights[li];
            math::Vec4 position(light.position[0], light.position[1], light.position[2], light.position[3]);
            const auto result = eye_reflect * position;
            light.position[0] = result.x; light.position[1] = result.y; light.position[2] = result.z;
            const Vec3 direction = eye_reflect.transform_dir(Vec3(light.direction[0], light.direction[1], light.direction[2]));
            light.direction[0] = direction.x; light.direction[1] = direction.y; light.direction[2] = direction.z;
        }
        const Vec3 reflected_camera = reflect.transform_point(camera);
        const Vec3 reflected_forward = reflect.transform_dir(Vec3(state.camera_forward[0], state.camera_forward[1], state.camera_forward[2]));
        reflected.camera_world_pos[0] = reflected_camera.x;
        reflected.camera_world_pos[1] = reflected_camera.y;
        reflected.camera_world_pos[2] = reflected_camera.z;
        reflected.camera_forward[0] = reflected_forward.x;
        reflected.camera_forward[1] = reflected_forward.y;
        reflected.camera_forward[2] = reflected_forward.z;
        std::memcpy(plan.matrices.data(), reflected.mvp, 16 * sizeof(float));
        std::memcpy(plan.matrices.data() + 16, reflected.model_view, 16 * sizeof(float));
        std::memcpy(plan.matrices.data() + 32, reflected.normal_matrix, 9 * sizeof(float));
        return plan;
    }
    return plan;
}

PlanarReflectionResources::~PlanarReflectionResources() {
    if (mask_) hipFree(mask_);
    if (matrices_) hipFree(matrices_);
}

void PlanarReflectionResources::ensure(size_t pixels) {
    if (pixels > capacity_) {
        uint8_t* fresh = nullptr;
        check_hip(hipMalloc(&fresh, pixels));
        if (mask_) hipFree(mask_);
        mask_ = fresh;
        capacity_ = pixels;
    }
    if (!matrices_) check_hip(hipMalloc(&matrices_, 41 * sizeof(float)));
}

void draw_opaque_with_planar_reflection(
    Pipeline& pipeline, Framebuffer& framebuffer,
    const physics::PhysicsRenderer& geometry, const RenderState& state,
    const TextureState& texture, const bool* geomgroup,
    const std::vector<TransparentGeomSpan>& transparent_order,
    const PlanarReflectionPlan& plan, PlanarReflectionResources& resources,
    const float* external_matrices,
    const float* reflect_matrices,
    const std::function<void(const RenderState&)>& draw_reflected_sites) {
    const int gid = plan.geom_id;
    const int offset = gid >= 0 ? geometry.geom_vert_offset_host(gid) : -1;
    const int count = gid >= 0 ? geometry.geom_vert_count_host(gid) : 0;
    if (offset < 0 || count <= 0) {
        draw_groups(pipeline, geometry, texture, geomgroup, -1);
        return;
    }

    // MuJoCo draws the reflecting geom normally over its mirrored scene with
    // GL_ONE/GL_ONE. Built here rather than just before its draw_span so the
    // micro-draw pair below can prepare BOTH mirror draws in one shot.
    RenderState base = state;
    base.blend_enabled = true;
    base.oit_enabled = false;
    base.blend_src = BLEND_ONE;
    base.blend_dst = BLEND_ONE;
    // Mirrors here are convex planes/boxes. Match MuJoCo's back-face culling
    // for this opaque additive draw even if the general HP culling diagnostic
    // is off: sorting triangles by minimum depth cannot stop a hidden rear
    // face from adding before a nearer face at a particular pixel.
    base.backface_cull_enabled = true;
    base.cull_face = 0;

    RenderState stencil = state;
    stencil.lighting_enabled = false;
    stencil.texture_2d_enabled = false;
    stencil.blend_enabled = false;
    stencil.oit_enabled = false;
    stencil.depth_test_enabled = false;
    stencil.depth_write_enabled = true;
    stencil.shadow_enabled = false;

    // The box generator emits +Z first, with equal vertex counts per face. A
    // plane uses its complete mesh. The plan's clip plane is at the same +Z
    // face, rather than the box centre or full silhouette.
    const int stencil_count = plan.is_box ? count / 6 : count;
    if (plan.render_scene) pipeline.micro_pair_begin(
        geometry.device_positions() + offset * 3,
        geometry.device_normals() + offset * 3,
        geometry.device_texcoords() + offset * TEXCOORD_COMPONENTS,
        geometry.device_texids() + offset,
        geometry.device_colors() + offset * 4,
        geometry.device_specular() + offset,
        geometry.device_shininess() + offset,
        geometry.device_emission() + offset,
        count / 3, stencil_count / 3, base, stencil);   // VERTEX counts -> triangles

    const bool deferred_bg = pipeline.deferred_bg();
    pipeline.set_deferred_bg(false);
    if (deferred_bg) pipeline.clear(true, false);


    if (plan.render_scene) {
        resources.ensure((size_t)framebuffer.width() * framebuffer.height());
        pipeline.set_render_state(stencil);
        draw_span(pipeline, geometry, TextureState{}, offset, stencil_count);
        hipStream_t mc_stream = pipeline.stream();
        const DeviceFramebuffer mc_fb = framebuffer.device_fb();
        bool mc_done = false;
        if (framebuffer.samples() != 4 && !state.gradient_bg) {
            static const int mc_on = getenv("HP_MASKCLEAR") ? atoi(getenv("HP_MASKCLEAR")) : 1;
            const int mc_total = mc_fb.width * mc_fb.height;
            if (mc_on && mc_total > 0 && resources.mask() &&
                mc_fb.color_buffer && mc_fb.depth_buffer) {
                kernels::FbMaskClearDesc mc{};
                mc.depth_in = mc_fb.depth_buffer;
                mc.color    = mc_fb.color_buffer;
                mc.depth    = mc_fb.depth_buffer;
                mc.mask     = resources.mask();
                mc.total    = mc_total;
                mc.mask_ref = state.clear_depth;
                mc.r = state.clear_color[0];
                mc.g = state.clear_color[1];
                mc.b = state.clear_color[2];
                mc.a = state.clear_color[3];
                mc.d = state.clear_depth;
                pipeline.arm_fb_mask_clear(mc);
                mc_done = true;
            }
            (void)mc_stream;
        }
        if (mc_done) {
            pipeline.set_render_state(state);
        } else {
            kernels::launch_reflection_sample_mask(mc_fb, resources.mask(), state.clear_depth, mc_stream);
            pipeline.set_render_state(state);
            pipeline.clear(true, true);
        }

        if (reflect_matrices) {
            pipeline.set_external_matrices(reflect_matrices);
        } else {
            check_hip(hipMemcpyAsync(resources.matrices(), plan.matrices.data(),
                                    41 * sizeof(float), hipMemcpyHostToDevice, pipeline.stream()));
            pipeline.set_external_matrices(resources.matrices());
        }
        RenderState reflected = plan.reflected_state;
        reflected.reflect_mask = resources.mask();
        pipeline.vtx_pair_arm(state, external_matrices);
        pipeline.set_render_state(reflected);
        draw_groups(pipeline, geometry, texture, geomgroup, gid);
        // MuJoCo reuses the ordinary camera's geom ordering for reflections.
        draw_intrinsic_transparency(pipeline, geometry, reflected, texture, transparent_order);
        if (draw_reflected_sites) draw_reflected_sites(reflected);
        pipeline.set_external_matrices(external_matrices);
    }

    // This additive pass remains active when mjRND_REFLECTION is off or the
    // camera is behind the +Z face; only the reflected-scene pass is gated. It
    // restores actual mirror depth and lets foreground objects occlude the
    // mirror normally.
    pipeline.set_render_state(base);
    draw_span(pipeline, geometry, texture, offset, count);
    pipeline.set_render_state(state);
    pipeline.micro_pair_cancel();
    draw_groups(pipeline, geometry, texture, geomgroup, gid);
    pipeline.set_deferred_bg(deferred_bg);
}

}  // namespace hipgl
