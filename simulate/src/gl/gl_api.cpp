// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - OpenGL API implementation

#include "hipgl/gl_api.h"
#include "hipgl/context.h"
#include "hipgl/pipeline.h"
#include "hipgl/buffer.h"
#include "hipgl/texture.h"
#include "hipgl/types.h"
#include "hipgl/math_types.h"
#include "hipgl/gl_state.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// ── State flags (s_caps, s_viewport, s_scissor from gl_state.h) ───────────────
namespace hipgl {
namespace gl {

static float s_clear_color[4] = {0, 0, 0, 1};
static double s_clear_depth = 1.0;
static unsigned int s_depth_func = 0x0201;
static unsigned int s_blend_src = 1, s_blend_dst = 0;
static unsigned int s_shade_model = 0x1D01;
static unsigned int s_cull_face_mode = 0x0405;
static unsigned int s_front_face_mode = 0x0901;
static unsigned int s_polygon_mode_face = 0x0408, s_polygon_mode_mode = 0x1B02;
static float s_point_size = 1.0f, s_line_width = 1.0f;
static unsigned int s_last_error = 0;
static int s_pack_alignment = 4;
static int s_unpack_alignment = 4;

} // namespace gl
} // namespace hipgl

namespace {

using namespace hipgl;

DepthFunc map_depth_func(unsigned int func) {
    switch (func) {
        case 0x0200: return DEPTH_NEVER;
        case 0x0201: return DEPTH_LESS;
        case 0x0202: return DEPTH_EQUAL;
        case 0x0203: return DEPTH_LEQUAL;
        case 0x0204: return DEPTH_GREATER;
        case 0x0205: return DEPTH_NOTEQUAL;
        case 0x0206: return DEPTH_GEQUAL;
        case 0x0207: return DEPTH_ALWAYS;
        default: return DEPTH_LESS;
    }
}

PrimitiveType map_primitive(unsigned int mode) {
    switch (mode) {
        case 0x0004: return PRIM_TRIANGLES;
        case 0x0005: return PRIM_TRIANGLE_STRIP;
        case 0x0006: return PRIM_TRIANGLE_FAN;
        case 0x0001: return PRIM_LINES;
        case 0x0002: return PRIM_LINE_STRIP;
        case 0x0000: return PRIM_POINTS;
        case 0x0007: return PRIM_QUADS;
        default: return PRIM_TRIANGLES;
    }
}

BufferUsage map_usage(unsigned int usage) {
    switch (usage) {
        case 0x88E4: return USAGE_STATIC_DRAW;
        case 0x88E8: return USAGE_DYNAMIC_DRAW;
        case 0x88E0: return USAGE_STREAM_DRAW;
        default: return USAGE_STATIC_DRAW;
    }
}

int map_buffer_target(unsigned int target) {
    return (target == 0x8892) ? 0 : 1;
}

Context* ctx() {
    return Context::current();
}

void sync_state_to_context(gl::GLState& gs, RenderState& rs, int vp_w, int vp_h) {
    rs.depth_test_enabled = gl::s_caps.depth_test;
    rs.depth_write_enabled = (gl::s_depth_mask != 0);
    rs.blend_enabled = gl::s_caps.blend;
    rs.lighting_enabled = gl::s_caps.lighting;
    rs.color_material_enabled = gl::s_caps.color_material;
    rs.backface_cull_enabled = gl::s_caps.cull_face;
    rs.depth_func = map_depth_func(gl::s_depth_func);
    rs.viewport_x = gl::s_viewport[0];
    rs.viewport_y = gl::s_viewport[1];
    rs.viewport_w = vp_w;
    rs.viewport_h = vp_h;
    std::memcpy(rs.clear_color, gl::s_clear_color, 4 * sizeof(float));
    rs.clear_depth = static_cast<float>(gl::s_clear_depth);
    rs.clear_stencil = gl::s_clear_stencil;
    std::memcpy(rs.model_view, &gs.modelview.m[0][0], 16 * sizeof(float));
    std::memcpy(rs.projection, &gs.projection.m[0][0], 16 * sizeof(float));
    std::memcpy(rs.mvp, gs.mvp, 16 * sizeof(float));
    std::memcpy(rs.normal_matrix, gs.normal_matrix, 9 * sizeof(float));
}

} // namespace

extern "C" {

void hipglMakeCurrent(void* ctx_ptr) {
    Context::make_current(static_cast<Context*>(ctx_ptr));
}

void hipglEnable(unsigned int cap) {
    using namespace hipgl::gl;
    switch (cap) {
        case 0x0B71: s_caps.depth_test = true; break;
        case 0x0BE2: s_caps.blend = true; break;
        case 0x0B50: s_caps.lighting = true; break;
        case 0x0B44: s_caps.cull_face = true; break;
        case 0x0B57: s_caps.color_material = true; break;
        case 0x0DE1: s_caps.texture_2d = true; break;
        case 0x0C11: s_caps.scissor_test = true; break;
        case 0x4000: case 0x4001: case 0x4002: case 0x4003:
        case 0x4004: case 0x4005: case 0x4006: case 0x4007: {
            auto* c = ctx();
            if (c) c->state().lights[cap - 0x4000].enabled = true;
            break;
        }
        default: break;
    }
}

void hipglDisable(unsigned int cap) {
    using namespace hipgl::gl;
    switch (cap) {
        case 0x0B71: s_caps.depth_test = false; break;
        case 0x0BE2: s_caps.blend = false; break;
        case 0x0B50: s_caps.lighting = false; break;
        case 0x0B44: s_caps.cull_face = false; break;
        case 0x0B57: s_caps.color_material = false; break;
        case 0x0DE1: s_caps.texture_2d = false; break;
        case 0x0C11: s_caps.scissor_test = false; break;
        case 0x4000: case 0x4001: case 0x4002: case 0x4003:
        case 0x4004: case 0x4005: case 0x4006: case 0x4007: {
            auto* c = ctx();
            if (c) c->state().lights[cap - 0x4000].enabled = false;
            break;
        }
        default: break;
    }
}

void hipglDepthFunc(unsigned int func) { hipgl::gl::s_depth_func = func; }
void hipglBlendFunc(unsigned int sfactor, unsigned int dfactor) {
    hipgl::gl::s_blend_src = sfactor;
    hipgl::gl::s_blend_dst = dfactor;
}
void hipglShadeModel(unsigned int mode) { hipgl::gl::s_shade_model = mode; }
void hipglCullFace(unsigned int mode) { hipgl::gl::s_cull_face_mode = mode; }
void hipglFrontFace(unsigned int mode) { hipgl::gl::s_front_face_mode = mode; }
void hipglPolygonMode(unsigned int face, unsigned int mode) {
    hipgl::gl::s_polygon_mode_face = face;
    hipgl::gl::s_polygon_mode_mode = mode;
}
void hipglPointSize(float size) { hipgl::gl::s_point_size = size; }
void hipglLineWidth(float width) { hipgl::gl::s_line_width = width; }

void hipglClearColor(float r, float g, float b, float a) {
    hipgl::gl::s_clear_color[0] = r;
    hipgl::gl::s_clear_color[1] = g;
    hipgl::gl::s_clear_color[2] = b;
    hipgl::gl::s_clear_color[3] = a;
}

void hipglClearDepth(double depth) {
    hipgl::gl::s_clear_depth = depth;
}

void hipglClear(unsigned int mask) {
    auto* c = ctx();
    if (!c) return;
    auto& gs = hipgl::gl::get_state();
    sync_state_to_context(gs, c->state(), c->framebuffer().width(), c->framebuffer().height());
    c->pipeline().set_render_state(c->state());
    bool do_color = (mask & 0x00004000) != 0;
    bool do_depth = (mask & 0x00000100) != 0;
    c->pipeline().clear(do_color, do_depth);
}

void hipglViewport(int x, int y, int width, int height) {
    hipgl::gl::s_viewport[0] = x;
    hipgl::gl::s_viewport[1] = y;
    hipgl::gl::s_viewport[2] = width;
    hipgl::gl::s_viewport[3] = height;
}

void hipglScissor(int x, int y, int width, int height) {
    hipgl::gl::s_scissor[0] = x;
    hipgl::gl::s_scissor[1] = y;
    hipgl::gl::s_scissor[2] = width;
    hipgl::gl::s_scissor[3] = height;
}

void hipglMatrixMode(unsigned int mode) {
    hipgl::gl::get_state().matrix_mode = mode;
}

void hipglLoadIdentity(void) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = hipgl::math::Mat4::identity();
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglLoadMatrixf(const float* m) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = hipgl::math::Mat4::from_column_major(m);
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglMultMatrixf(const float* m) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = stk[*depth] * hipgl::math::Mat4::from_column_major(m);
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglPushMatrix(void) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    if (*depth + 1 >= MAX_MATRIX_STACK_DEPTH) return;
    stk[*depth + 1] = stk[*depth];
    (*depth)++;
}

void hipglPopMatrix(void) {
    auto& gs = hipgl::gl::get_state();
    int* depth = gs.current_depth();
    if (*depth <= 0) return;
    (*depth)--;
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglTranslatef(float x, float y, float z) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4 T = hipgl::math::translate(hipgl::math::Vec3(x, y, z));
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = stk[*depth] * T;
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglRotatef(float angle, float x, float y, float z) {
    auto& gs = hipgl::gl::get_state();
    float rad = angle * 3.14159265f / 180.0f;
    hipgl::math::Mat4 R = hipgl::math::rotate(rad, hipgl::math::Vec3(x, y, z));
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = stk[*depth] * R;
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglScalef(float x, float y, float z) {
    auto& gs = hipgl::gl::get_state();
    hipgl::math::Mat4 S = hipgl::math::scale(hipgl::math::Vec3(x, y, z));
    hipgl::math::Mat4* stk = gs.current_matrix_stack();
    int* depth = gs.current_depth();
    stk[*depth] = stk[*depth] * S;
    if (gs.matrix_mode == 0x1700) {
        gs.modelview = stk[*depth];
        gs.update_mvp();
        gs.update_normal_matrix();
    } else if (gs.matrix_mode == 0x1701) {
        gs.projection = stk[*depth];
        gs.update_mvp();
    }
}

void hipglOrtho(double left, double right, double bottom, double top, double near, double far) {
    hipgl::math::Mat4 M = hipgl::math::ortho(
        static_cast<float>(left), static_cast<float>(right),
        static_cast<float>(bottom), static_cast<float>(top),
        static_cast<float>(near), static_cast<float>(far));
    float m[16];
    M.to_column_major(m);
    hipglMultMatrixf(m);
}

void hipglFrustum(double left, double right, double bottom, double top, double near_val, double far_val) {
    hipgl::math::Mat4 M;
    float l = static_cast<float>(left), r = static_cast<float>(right);
    float b = static_cast<float>(bottom), t = static_cast<float>(top);
    float n = static_cast<float>(near_val), f = static_cast<float>(far_val);
    M(0,0) = 2.0f*n / (r-l);
    M(1,1) = 2.0f*n / (t-b);
    M(0,2) = (r+l) / (r-l);
    M(1,2) = (t+b) / (t-b);
    M(2,2) = -(f+n) / (f-n);
    M(3,2) = -1.0f;
    M(2,3) = -2.0f*f*n / (f-n);
    M(3,3) = 0.0f;
    float m[16];
    M.to_column_major(m);
    hipglMultMatrixf(m);
}

void hipglBegin(unsigned int mode) {
    auto& gs = hipgl::gl::get_state();
    if (gs.imm_mode >= 0) return;
    gs.imm_mode = mode;
    gs.imm_vertices.clear();
    gs.have_current_normal = false;
    gs.have_current_color = false;
    gs.have_current_texcoord = false;
}

void hipglEnd(void) {
    auto& gs = hipgl::gl::get_state();
    if (gs.imm_mode < 0) return;
    int mode = gs.imm_mode;
    gs.imm_mode = -1;

    auto* c = ctx();
    if (!c || gs.imm_vertices.empty()) return;

    if (mode == 0x0004) {
        hipglDrawArrays(0x0004, 0, static_cast<int>(gs.imm_vertices.size()));
    } else if (mode == 0x0007) {
        std::vector<gl::ImmVertex> tri_verts;
        for (size_t q = 0; q < gs.imm_vertices.size() / 4; q++) {
            int i0 = q*4, i1 = q*4+1, i2 = q*4+2, i3 = q*4+3;
            tri_verts.push_back(gs.imm_vertices[i0]);
            tri_verts.push_back(gs.imm_vertices[i1]);
            tri_verts.push_back(gs.imm_vertices[i2]);
            tri_verts.push_back(gs.imm_vertices[i0]);
            tri_verts.push_back(gs.imm_vertices[i2]);
            tri_verts.push_back(gs.imm_vertices[i3]);
        }
        gs.imm_vertices = std::move(tri_verts);
        hipglDrawArrays(0x0004, 0, static_cast<int>(gs.imm_vertices.size()));
    }
}

void hipglVertex3f(float x, float y, float z) {
    auto& gs = hipgl::gl::get_state();
    if (gs.imm_mode < 0) return;
    gl::ImmVertex v;
    v.pos[0] = x; v.pos[1] = y; v.pos[2] = z;
    v.normal[0] = gs.current_normal[0]; v.normal[1] = gs.current_normal[1]; v.normal[2] = gs.current_normal[2];
    v.color[0] = gs.current_color[0]; v.color[1] = gs.current_color[1]; v.color[2] = gs.current_color[2]; v.color[3] = gs.current_color[3];
    v.texcoord[0] = gs.current_texcoord[0]; v.texcoord[1] = gs.current_texcoord[1];
    gs.imm_vertices.push_back(v);
}

void hipglVertex3fv(const float* v) { if (v) hipglVertex3f(v[0], v[1], v[2]); }
void hipglNormal3f(float x, float y, float z) {
    auto& gs = hipgl::gl::get_state();
    gs.current_normal[0] = x; gs.current_normal[1] = y; gs.current_normal[2] = z;
    gs.have_current_normal = true;
}
void hipglNormal3fv(const float* v) { if (v) hipglNormal3f(v[0], v[1], v[2]); }
void hipglColor3f(float r, float g, float b) {
    auto& gs = hipgl::gl::get_state();
    gs.current_color[0] = r; gs.current_color[1] = g; gs.current_color[2] = b; gs.current_color[3] = 1.0f;
    gs.have_current_color = true;
}
void hipglColor4f(float r, float g, float b, float a) {
    auto& gs = hipgl::gl::get_state();
    gs.current_color[0] = r; gs.current_color[1] = g; gs.current_color[2] = b; gs.current_color[3] = a;
    gs.have_current_color = true;
}
void hipglColor3fv(const float* v) { if (v) hipglColor3f(v[0], v[1], v[2]); }
void hipglColor4fv(const float* v) { if (v) hipglColor4f(v[0], v[1], v[2], v[3]); }
void hipglTexCoord2f(float s, float t) {
    auto& gs = hipgl::gl::get_state();
    gs.current_texcoord[0] = s; gs.current_texcoord[1] = t;
    gs.have_current_texcoord = true;
}

void hipglLightfv(unsigned int light, unsigned int pname, const float* params) {
    using namespace hipgl::gl;
    auto* c = ctx();
    if (!c || !params) return;
    int idx = light - 0x4000;  // GL_LIGHT0
    if (idx < 0 || idx >= hipgl::MAX_LIGHTS) return;
    hipgl::Light& L = c->state().lights[idx];
    switch (pname) {
        case 0x1200: std::memcpy(L.ambient, params, 4 * sizeof(float)); break;   // GL_AMBIENT
        case 0x1201: std::memcpy(L.diffuse, params, 4 * sizeof(float)); break;   // GL_DIFFUSE
        case 0x1202: std::memcpy(L.specular, params, 4 * sizeof(float)); break;    // GL_SPECULAR
        case 0x1203: {  // GL_POSITION - transform by current modelview
            auto& gs = get_state();
            hipgl::math::Vec4 pos(params[0], params[1], params[2], params[3]);
            hipgl::math::Vec4 eye_pos = gs.modelview * pos;
            L.position[0] = eye_pos.x;
            L.position[1] = eye_pos.y;
            L.position[2] = eye_pos.z;
            L.position[3] = eye_pos.w;
            break;
        }
        case 0x1204: std::memcpy(L.direction, params, 3 * sizeof(float)); break;  // GL_SPOT_DIRECTION
        default: break;
    }
}

void hipglLightf(unsigned int light, unsigned int pname, float param) {
    auto* c = ctx();
    if (!c) return;
    int idx = light - 0x4000;
    if (idx < 0 || idx >= hipgl::MAX_LIGHTS) return;
    hipgl::Light& L = c->state().lights[idx];
    switch (pname) {
        case 0x1205: L.spot_exponent = param; break;
        case 0x1206: L.spot_cutoff = param; break;
        case 0x1207: L.constant_atten = param; break;
        case 0x1208: L.linear_atten = param; break;
        case 0x1209: L.quadratic_atten = param; break;
        default: break;
    }
}

void hipglMaterialfv(unsigned int face, unsigned int pname, const float* params) {
    (void)face;
    auto* c = ctx();
    if (!c || !params) return;
    hipgl::Material& M = c->state().material;
    switch (pname) {
        case 0x1200: std::memcpy(M.ambient, params, 4 * sizeof(float)); break;
        case 0x1201: std::memcpy(M.diffuse, params, 4 * sizeof(float)); break;
        case 0x1202: std::memcpy(M.specular, params, 4 * sizeof(float)); break;
        case 0x1600: std::memcpy(M.emission, params, 4 * sizeof(float)); break;
        case 0x1602:  // AMBIENT_AND_DIFFUSE
            std::memcpy(M.ambient, params, 4 * sizeof(float));
            std::memcpy(M.diffuse, params, 4 * sizeof(float));
            break;
        default: break;
    }
}

void hipglMaterialf(unsigned int face, unsigned int pname, float param) {
    (void)face;
    auto* c = ctx();
    if (!c) return;
    if (pname == 0x1601) c->state().material.shininess = param;  // GL_SHININESS
}

void hipglColorMaterial(unsigned int face, unsigned int mode) {
    (void)face;
    (void)mode;
    // Just enable color material tracking; the actual logic is in the fragment shader
}

void hipglLightModelfv(unsigned int pname, const float* params) {
    // GL_LIGHT_MODEL_AMBIENT = 0x0B53
    if (pname == 0x0B53 && params) {
        // Store global ambient; we'll use a static for now
        // In a full implementation, add to RenderState
        (void)params;
    }
}

void hipglLightModeli(unsigned int pname, int param) {
    // GL_LIGHT_MODEL_TWO_SIDE = 0x0B52
    (void)pname;
    (void)param;
}

void hipglGenBuffers(int n, unsigned int* buffers) {
    auto* c = ctx();
    if (!c || !buffers || n <= 0) return;
    for (int i = 0; i < n; i++) {
        buffers[i] = c->buffers().create_buffer();
    }
}

void hipglDeleteBuffers(int n, const unsigned int* buffers) {
    auto* c = ctx();
    if (!c || !buffers) return;
    for (int i = 0; i < n; i++) {
        c->buffers().delete_buffer(buffers[i]);
    }
}

void hipglBindBuffer(unsigned int target, unsigned int buffer) {
    auto* c = ctx();
    if (!c) return;
    int t = map_buffer_target(target);
    c->buffers().bind_buffer(t, buffer);
}

void hipglBufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) {
    auto* c = ctx();
    if (!c || size < 0) return;
    int t = map_buffer_target(target);
    uint32_t id = (t == 1) ? c->buffers().current_element_buffer() : c->buffers().current_array_buffer();
    if (id == 0) return;
    Buffer* buf = c->buffers().get_buffer(id);
    if (!buf) return;
    buf->upload(data, static_cast<size_t>(size), map_usage(usage));
}

void hipglEnableClientState(unsigned int cap) {
    auto& gs = hipgl::gl::get_state();
    switch (cap) {
        case 0x8074: gs.vertex_array_enabled = true; break;
        case 0x8075: gs.normal_array_enabled = true; break;
        case 0x8076: gs.color_array_enabled = true; break;
        case 0x8078: gs.texcoord_array_enabled = true; break;
        default: break;
    }
}

void hipglDisableClientState(unsigned int cap) {
    auto& gs = hipgl::gl::get_state();
    switch (cap) {
        case 0x8074: gs.vertex_array_enabled = false; break;
        case 0x8075: gs.normal_array_enabled = false; break;
        case 0x8076: gs.color_array_enabled = false; break;
        case 0x8078: gs.texcoord_array_enabled = false; break;
        default: break;
    }
}

void hipglVertexPointer(int size, unsigned int type, int stride, const void* ptr) {
    auto& gs = hipgl::gl::get_state();
    gs.vertex_size = size;
    gs.vertex_type = type;
    gs.vertex_stride = stride;
    gs.vertex_ptr = ptr;
}

void hipglNormalPointer(unsigned int type, int stride, const void* ptr) {
    auto& gs = hipgl::gl::get_state();
    gs.normal_type = type;
    gs.normal_stride = stride;
    gs.normal_ptr = ptr;
}

void hipglColorPointer(int size, unsigned int type, int stride, const void* ptr) {
    auto& gs = hipgl::gl::get_state();
    gs.color_size = size;
    gs.color_type = type;
    gs.color_stride = stride;
    gs.color_ptr = ptr;
}

void hipglTexCoordPointer(int size, unsigned int type, int stride, const void* ptr) {
    auto& gs = hipgl::gl::get_state();
    gs.texcoord_size = size;
    gs.texcoord_type = type;
    gs.texcoord_stride = stride;
    gs.texcoord_ptr = ptr;
}

void hipglGenVertexArrays(int, unsigned int*) {}
void hipglBindVertexArray(unsigned int) {}
void hipglVertexAttribPointer(unsigned int, int, unsigned int, unsigned char, int, const void*) {}
void hipglEnableVertexAttribArray(unsigned int) {}
void hipglDisableVertexAttribArray(unsigned int) {}

void hipglDrawArrays(unsigned int mode, int first, int count) {
    auto* c = ctx();
    if (!c || count <= 0) return;

    auto& gs = hipgl::gl::get_state();
    int vp_w = hipgl::gl::s_viewport[2];
    int vp_h = hipgl::gl::s_viewport[3];
    if (vp_w <= 0 || vp_h <= 0) return;

    std::vector<float> positions, normals, colors, texcoords;
    const float* pos_ptr = nullptr;
    const float* norm_ptr = nullptr;
    const float* col_ptr = nullptr;
    const float* tex_ptr = nullptr;
    int vertex_count = count;

    if (!gs.imm_vertices.empty()) {
        for (const auto& v : gs.imm_vertices) {
            positions.push_back(v.pos[0]); positions.push_back(v.pos[1]); positions.push_back(v.pos[2]);
            normals.push_back(v.normal[0]); normals.push_back(v.normal[1]); normals.push_back(v.normal[2]);
            colors.push_back(v.color[0]); colors.push_back(v.color[1]); colors.push_back(v.color[2]); colors.push_back(v.color[3]);
            texcoords.push_back(v.texcoord[0]); texcoords.push_back(v.texcoord[1]);
        }
        pos_ptr = positions.data();
        norm_ptr = normals.data();
        col_ptr = colors.data();
        tex_ptr = texcoords.data();
        vertex_count = static_cast<int>(gs.imm_vertices.size());
    } else if (gs.vertex_array_enabled && gs.vertex_ptr) {
        pos_ptr = static_cast<const float*>(gs.vertex_ptr);
        norm_ptr = gs.normal_array_enabled && gs.normal_ptr ? static_cast<const float*>(gs.normal_ptr) : nullptr;
        col_ptr = gs.color_array_enabled && gs.color_ptr ? static_cast<const float*>(gs.color_ptr) : nullptr;
        tex_ptr = gs.texcoord_array_enabled && gs.texcoord_ptr ? static_cast<const float*>(gs.texcoord_ptr) : nullptr;
    }

    if (!pos_ptr) return;

    sync_state_to_context(gs, c->state(), vp_w, vp_h);
    c->pipeline().set_render_state(c->state());
    c->pipeline().draw_arrays(map_primitive(mode), pos_ptr, norm_ptr, tex_ptr, col_ptr, vertex_count);
}

void hipglDrawElements(unsigned int mode, int count, unsigned int type, const void* indices) {
    auto* c = ctx();
    if (!c || count <= 0 || !indices) return;

    auto& gs = hipgl::gl::get_state();
    int vp_w = hipgl::gl::s_viewport[2];
    int vp_h = hipgl::gl::s_viewport[3];
    if (vp_w <= 0 || vp_h <= 0) return;

    const float* pos_ptr = gs.vertex_array_enabled && gs.vertex_ptr
        ? static_cast<const float*>(gs.vertex_ptr) : nullptr;
    const float* norm_ptr = gs.normal_array_enabled && gs.normal_ptr
        ? static_cast<const float*>(gs.normal_ptr) : nullptr;
    const float* col_ptr = gs.color_array_enabled && gs.color_ptr
        ? static_cast<const float*>(gs.color_ptr) : nullptr;
    const float* tex_ptr = gs.texcoord_array_enabled && gs.texcoord_ptr
        ? static_cast<const float*>(gs.texcoord_ptr) : nullptr;

    if (!pos_ptr) return;

    std::vector<uint32_t> idx32(count);
    if (type == 0x1401) {
        auto* p = static_cast<const uint8_t*>(indices);
        for (int i = 0; i < count; i++) idx32[i] = p[i];
    } else if (type == 0x1403) {
        auto* p = static_cast<const uint16_t*>(indices);
        for (int i = 0; i < count; i++) idx32[i] = p[i];
    } else {
        auto* p = static_cast<const uint32_t*>(indices);
        for (int i = 0; i < count; i++) idx32[i] = p[i];
    }

    uint32_t max_idx = 0;
    for (int i = 0; i < count; i++) max_idx = std::max(max_idx, idx32[i]);
    int vertex_count = static_cast<int>(max_idx) + 1;

    sync_state_to_context(gs, c->state(), vp_w, vp_h);
    c->pipeline().set_render_state(c->state());
    c->pipeline().draw_elements(map_primitive(mode), pos_ptr, norm_ptr, tex_ptr, col_ptr,
                                vertex_count, idx32.data(), count);
}

void hipglGenTextures(int n, unsigned int* textures) {
    auto* c = ctx();
    if (!c || !textures || n <= 0) return;
    for (int i = 0; i < n; i++) {
        textures[i] = c->textures().create_texture();
    }
}

void hipglDeleteTextures(int n, const unsigned int* textures) {
    auto* c = ctx();
    if (!c || !textures) return;
    for (int i = 0; i < n; i++) {
        c->textures().delete_texture(textures[i]);
    }
}

void hipglBindTexture(unsigned int target, unsigned int texture) {
    auto* c = ctx();
    if (!c) return;
    c->textures().bind_texture(texture);
}

void hipglTexImage2D(unsigned int target, int level, int internalformat,
                     int width, int height, int border,
                     unsigned int format, unsigned int type, const void* data) {
    extern void hipglTexImage2D_impl(unsigned int, int, int, int, int, int, unsigned int, unsigned int, const void*);
    hipglTexImage2D_impl(target, level, internalformat, width, height, border, format, type, data);
}

void hipglTexParameteri(unsigned int target, unsigned int pname, int param) {
    extern void hipglTexParameteri_impl(unsigned int, unsigned int, int);
    hipglTexParameteri_impl(target, pname, param);
}

void hipglReadPixels(int x, int y, int width, int height, unsigned int format, unsigned int type, void* data) {
    auto* c = ctx();
    if (!c || !data) return;
    if (format == 0x1908 && type == 0x1401) {
        c->read_pixels(x, y, width, height, static_cast<uint8_t*>(data));
    }
}

void hipglPixelStorei(unsigned int pname, int param) {
    if (pname == 0x0CF5) hipgl::gl::s_unpack_alignment = param;
    else if (pname == 0x0CF1) hipgl::gl::s_pack_alignment = param;
}

unsigned int hipglGetError(void) {
    unsigned int err = hipgl::gl::s_last_error;
    hipgl::gl::s_last_error = 0;
    return err;
}

const unsigned char* hipglGetString(unsigned int name) {
    static const unsigned char vendor[] = "hipGL";
    static const unsigned char renderer[] = "hipGL HIP Software Rasterizer";
    static const unsigned char version[] = "1.0 hipGL";
    switch (name) {
        case 0x1F00: return vendor;
        case 0x1F01: return renderer;
        case 0x1F02: return version;
        default: return reinterpret_cast<const unsigned char*>("");
    }
}

void hipglGetIntegerv(unsigned int pname, int* params) {
    if (!params) return;
    switch (pname) {
        case 0x0BA2:
            params[0] = hipgl::gl::s_viewport[0];
            params[1] = hipgl::gl::s_viewport[1];
            params[2] = hipgl::gl::s_viewport[2];
            params[3] = hipgl::gl::s_viewport[3];
            break;
        case 0x0C10:
            params[0] = hipgl::gl::s_scissor[0];
            params[1] = hipgl::gl::s_scissor[1];
            params[2] = hipgl::gl::s_scissor[2];
            params[3] = hipgl::gl::s_scissor[3];
            break;
        default: break;
    }
}


void hipglFlush(void) {
    // glFlush only guarantees commands are submitted; no need to sync or copy.
    // Actual data transfer happens on glReadPixels or glFinish.
}

void hipglFinish(void) {
    auto* c = ctx();
    if (c) {
        c->pipeline().sync();
    }
}

void* hipglCreateContext(int width, int height) {
    try {
        auto* c = new Context(0);
        c->init(width, height);
        hipgl::gl::s_viewport[0] = 0;
        hipgl::gl::s_viewport[1] = 0;
        hipgl::gl::s_viewport[2] = width;
        hipgl::gl::s_viewport[3] = height;
        return c;
    } catch (...) {
        return nullptr;
    }
}

void hipglDestroyContext(void* ctx_ptr) {
    delete static_cast<Context*>(ctx_ptr);
}

} // extern "C"
