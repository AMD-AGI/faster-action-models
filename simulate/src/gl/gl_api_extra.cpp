// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Extra GL API functions (MuJoCo compatibility)
// Implementations for fog, alpha, stencil, vertex/texcoord helpers, etc.

#include "hipgl/gl_api.h"
#include "hipgl/context.h"
#include "hipgl/gl_state.h"
#include "hipgl/math_types.h"

extern "C" {

// ── glDepthMask / glColorMask ────
void hipglDepthMask(unsigned char flag) {
    hipgl::gl::s_depth_mask = flag;
}

void hipglColorMask(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    hipgl::gl::s_color_mask[0] = r;
    hipgl::gl::s_color_mask[1] = g;
    hipgl::gl::s_color_mask[2] = b;
    hipgl::gl::s_color_mask[3] = a;
}

// ── glClearStencil ────
void hipglClearStencil(int s) {
    hipgl::gl::s_clear_stencil = s;
}

// ── glIsEnabled ────
unsigned char hipglIsEnabled(unsigned int cap) {
    switch (cap) {
        case 0x0B71: return hipgl::gl::s_caps.depth_test ? 1 : 0;
        case 0x0BE2: return hipgl::gl::s_caps.blend ? 1 : 0;
        case 0x0B50: return hipgl::gl::s_caps.lighting ? 1 : 0;
        case 0x0B44: return hipgl::gl::s_caps.cull_face ? 1 : 0;
        case 0x0B57: return hipgl::gl::s_caps.color_material ? 1 : 0;
        case 0x0DE1: return hipgl::gl::s_caps.texture_2d ? 1 : 0;
        case 0x0C11: return hipgl::gl::s_caps.scissor_test ? 1 : 0;
        default: return 0;
    }
}

// ── Extended glGetIntegerv (handles extra pnames) ────
// Call this for pnames not handled by the base hipglGetIntegerv.
void hipglGetIntegerv_extended(unsigned int pname, int* params) {
    if (!params) return;
    switch (pname) {
        case 0x0BA2: // GL_VIEWPORT
            params[0] = hipgl::gl::s_viewport[0];
            params[1] = hipgl::gl::s_viewport[1];
            params[2] = hipgl::gl::s_viewport[2];
            params[3] = hipgl::gl::s_viewport[3];
            break;
        case 0x0C10: // GL_SCISSOR_BOX
            params[0] = hipgl::gl::s_scissor[0];
            params[1] = hipgl::gl::s_scissor[1];
            params[2] = hipgl::gl::s_scissor[2];
            params[3] = hipgl::gl::s_scissor[3];
            break;
        case 0x0D33: // GL_MAX_TEXTURE_SIZE
            params[0] = 4096;
            break;
        case 0x0D31: // GL_MAX_LIGHTS
            params[0] = 8;
            break;
        case 0x0D32: // GL_MAX_CLIP_PLANES
            params[0] = 6;
            break;
        case 0x0D36: // GL_MAX_MODELVIEW_STACK_DEPTH
            params[0] = 32;
            break;
        case 0x0D38: // GL_MAX_PROJECTION_STACK_DEPTH
            params[0] = 32;
            break;
        default: params[0] = 0; break;
    }
}

// ── Fog (stubs that store state) ────
static float s_fog_color[4] = {0,0,0,0};
static float s_fog_density = 1.0f;
static float s_fog_start = 0.0f;
static float s_fog_end = 1.0f;
static int s_fog_mode = 0x0800; // GL_EXP

void hipglFogf(unsigned int pname, float param) {
    switch (pname) {
        case 0x0B62: s_fog_density = param; break;
        case 0x0B63: s_fog_start = param; break;
        case 0x0B64: s_fog_end = param; break;
        default: break;
    }
}

void hipglFogi(unsigned int pname, int param) {
    if (pname == 0x0B65) s_fog_mode = param;
}

void hipglFogfv(unsigned int pname, const float* params) {
    if (!params) return;
    if (pname == 0x0B66) { // GL_FOG_COLOR
        s_fog_color[0] = params[0]; s_fog_color[1] = params[1];
        s_fog_color[2] = params[2]; s_fog_color[3] = params[3];
    } else {
        hipglFogf(pname, params[0]);
    }
}

// ── Alpha test (stub) ────
static unsigned int s_alpha_func = 0x0207; // GL_ALWAYS
static float s_alpha_ref = 0.0f;

void hipglAlphaFunc(unsigned int func, float ref) {
    s_alpha_func = func;
    s_alpha_ref = ref;
}

// ── Stencil (stubs) ────
void hipglStencilFunc(unsigned int func, int ref, unsigned int mask) {
    (void)func; (void)ref; (void)mask;
}

void hipglStencilOp(unsigned int sfail, unsigned int dpfail, unsigned int dppass) {
    (void)sfail; (void)dpfail; (void)dppass;
}

// ── Polygon offset (stub) ────
static float s_polygon_offset_factor = 0.0f;
static float s_polygon_offset_units = 0.0f;

void hipglPolygonOffset(float factor, float units) {
    s_polygon_offset_factor = factor;
    s_polygon_offset_units = units;
}

// ── Clip planes (stub) ────
void hipglClipPlane(unsigned int plane, const double* equation) {
    (void)plane; (void)equation;
}

// ── glColor4ubv ────
void hipglColor4ubv(const unsigned char* v) {
    if (!v) return;
    hipglColor4f(v[0]/255.0f, v[1]/255.0f, v[2]/255.0f, v[3]/255.0f);
}

// ── glVertex2f ────
void hipglVertex2f(float x, float y) {
    hipglVertex3f(x, y, 0.0f);
}

// ── glTexCoord2fv ────
void hipglTexCoord2fv(const float* v) {
    if (v) hipglTexCoord2f(v[0], v[1]);
}

// ── glBitmap (stub - MuJoCo uses it for text) ────
void hipglBitmap(int width, int height, float xorig, float yorig,
                 float xmove, float ymove, const unsigned char* bitmap) {
    (void)width; (void)height; (void)xorig; (void)yorig;
    (void)xmove; (void)ymove; (void)bitmap;
}

// ── glDrawBuffer / glReadBuffer ────
void hipglDrawBuffer(unsigned int buf) { (void)buf; }
void hipglReadBuffer(unsigned int buf) { (void)buf; }

// ── glGenerateMipmap ────
void hipglGenerateMipmap(unsigned int target) { (void)target; }

// ── glActiveTexture ────
void hipglActiveTexture(unsigned int texture) { (void)texture; }

// ── glTexEnvi (stub) ────
void hipglTexEnvi(unsigned int target, unsigned int pname, int param) {
    (void)target; (void)pname; (void)param;
}

// ── glTexGeni / glTexGenfv (stubs) ────
void hipglTexGeni(unsigned int coord, unsigned int pname, int param) {
    (void)coord; (void)pname; (void)param;
}

void hipglTexGenfv(unsigned int coord, unsigned int pname, const float* params) {
    (void)coord; (void)pname; (void)params;
}

} // extern "C"
