// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/gl_state.h"
#include "hipgl/gl_api.h"
#include <cmath>
#include <cstring>

namespace hipgl {
namespace gl {

// 3x3 determinant
static float det3(const float* m) {
    return m[0]*(m[4]*m[8] - m[5]*m[7])
         - m[1]*(m[3]*m[8] - m[5]*m[6])
         + m[2]*(m[3]*m[7] - m[4]*m[6]);
}

// 3x3 inverse (column-major 9 elements), returns false if singular
static bool invert3x3(const float* m, float* out) {
    float det = det3(m);
    if (fabsf(det) < 1e-10f) return false;
    float inv = 1.0f / det;
    out[0] = (m[4]*m[8] - m[5]*m[7]) * inv;
    out[1] = (m[2]*m[7] - m[1]*m[8]) * inv;
    out[2] = (m[1]*m[5] - m[2]*m[4]) * inv;
    out[3] = (m[5]*m[6] - m[3]*m[8]) * inv;
    out[4] = (m[0]*m[8] - m[2]*m[6]) * inv;
    out[5] = (m[2]*m[3] - m[0]*m[5]) * inv;
    out[6] = (m[3]*m[7] - m[4]*m[6]) * inv;
    out[7] = (m[1]*m[6] - m[0]*m[7]) * inv;
    out[8] = (m[0]*m[4] - m[1]*m[3]) * inv;
    return true;
}

// Transpose 3x3 in place for normal matrix (inverse then transpose = transpose then inverse)
static void transpose3x3(const float* m, float* out) {
    out[0] = m[0]; out[1] = m[3]; out[2] = m[6];
    out[3] = m[1]; out[4] = m[4]; out[5] = m[7];
    out[6] = m[2]; out[7] = m[5]; out[8] = m[8];
}

static GLState s_state;

GLCaps s_caps;
int s_viewport[4] = {0, 0, 256, 256};
int s_scissor[4] = {0, 0, 32768, 32768};
int s_clear_stencil = 0;
unsigned char s_depth_mask = 1;
unsigned char s_color_mask[4] = {1, 1, 1, 1};

GLState& get_state() {
    return s_state;
}

GLState::GLState()
    : matrix_mode(HIPGL_MODELVIEW)
    , modelview_depth(0)
    , projection_depth(0)
    , texture_depth(0)
    , imm_mode(-1)
    , have_current_normal(false)
    , have_current_color(false)
    , have_current_texcoord(false)
    , vertex_array_enabled(false)
    , normal_array_enabled(false)
    , color_array_enabled(false)
    , texcoord_array_enabled(false)
    , vertex_size(3)
    , vertex_type(HIPGL_FLOAT)
    , vertex_stride(0)
    , vertex_ptr(nullptr)
    , normal_type(HIPGL_FLOAT)
    , normal_stride(0)
    , normal_ptr(nullptr)
    , color_size(4)
    , color_type(HIPGL_FLOAT)
    , color_stride(0)
    , color_ptr(nullptr)
    , texcoord_size(2)
    , texcoord_type(HIPGL_FLOAT)
    , texcoord_stride(0)
    , texcoord_ptr(nullptr)
    , bound_array_buffer(0)
    , bound_element_buffer(0)
    , unpack_alignment(4)
    , pack_alignment(4)
{
    current_normal[0] = 0; current_normal[1] = 1; current_normal[2] = 0;
    current_color[0] = 1; current_color[1] = 1; current_color[2] = 1; current_color[3] = 1;
    current_texcoord[0] = 0; current_texcoord[1] = 0;
    reset();
}

void GLState::reset() {
    modelview = hipgl::math::Mat4::identity();
    projection = hipgl::math::Mat4::identity();
    texture_matrix = hipgl::math::Mat4::identity();
    modelview_stack[0] = modelview;
    projection_stack[0] = projection;
    texture_stack[0] = texture_matrix;
    modelview_depth = 0;
    projection_depth = 0;
    texture_depth = 0;
    update_mvp();
    update_normal_matrix();
}

hipgl::math::Mat4* GLState::current_matrix_stack() {
    switch (matrix_mode) {
        case HIPGL_PROJECTION: return projection_stack;
        case HIPGL_TEXTURE:    return texture_stack;
        default:              return modelview_stack;
    }
}

int* GLState::current_depth() {
    switch (matrix_mode) {
        case HIPGL_PROJECTION: return &projection_depth;
        case HIPGL_TEXTURE:    return &texture_depth;
        default:              return &modelview_depth;
    }
}

void GLState::update_mvp() {
    hipgl::math::Mat4 m = projection * modelview;
    m.to_column_major(mvp);
}

void GLState::update_normal_matrix() {
    float upper[9];
    modelview.upper3x3(upper);
    float inv[9];
    if (invert3x3(upper, inv)) {
        transpose3x3(inv, normal_matrix);
    } else {
        // Fallback: identity
        std::memset(normal_matrix, 0, sizeof(normal_matrix));
        normal_matrix[0] = normal_matrix[4] = normal_matrix[8] = 1.0f;
    }
}

} // namespace gl
} // namespace hipgl
