#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// hipGL - Internal GL state machine (not part of public API)

#include "hipgl/types.h"
#include "hipgl/math_types.h"
#include <vector>
#include <cstring>

namespace hipgl {
namespace gl {

struct ImmVertex {
    float pos[3];
    float normal[3];
    float color[4];
    float texcoord[2];
};

struct GLState {
    // Matrix stacks
    int matrix_mode;  // MODELVIEW, PROJECTION, or TEXTURE
    hipgl::math::Mat4 modelview_stack[MAX_MATRIX_STACK_DEPTH];
    hipgl::math::Mat4 projection_stack[MAX_MATRIX_STACK_DEPTH];
    hipgl::math::Mat4 texture_stack[MAX_MATRIX_STACK_DEPTH];
    int modelview_depth;
    int projection_depth;
    int texture_depth;

    // Current matrices (cached from top of stack)
    hipgl::math::Mat4 modelview;
    hipgl::math::Mat4 projection;
    hipgl::math::Mat4 texture_matrix;
    float mvp[16];
    float normal_matrix[9];

    // Immediate mode accumulator
    std::vector<ImmVertex> imm_vertices;
    int imm_mode;  // current primitive mode, -1 if not in Begin/End
    float current_normal[3];
    float current_color[4];
    float current_texcoord[2];
    bool have_current_normal;
    bool have_current_color;
    bool have_current_texcoord;

    // Client state (legacy vertex arrays)
    bool vertex_array_enabled;
    bool normal_array_enabled;
    bool color_array_enabled;
    bool texcoord_array_enabled;

    // Vertex array pointers (legacy)
    int vertex_size;
    unsigned int vertex_type;
    int vertex_stride;
    const void* vertex_ptr;
    unsigned int normal_type;
    int normal_stride;
    const void* normal_ptr;
    int color_size;
    unsigned int color_type;
    int color_stride;
    const void* color_ptr;
    int texcoord_size;
    unsigned int texcoord_type;
    int texcoord_stride;
    const void* texcoord_ptr;

    // Bound buffers
    unsigned int bound_array_buffer;
    unsigned int bound_element_buffer;

    // Pixel store
    int unpack_alignment;
    int pack_alignment;

    GLState();
    void reset();

    // Matrix helpers
    hipgl::math::Mat4* current_matrix_stack();
    int* current_depth();
    void update_mvp();
    void update_normal_matrix();
};

// Global state access (thread-local in future)
GLState& get_state();

// ── Shared GL API state (used by gl_api.cpp and gl_api_extra.cpp) ───────────
// NOTE: gl_api.cpp should use these via extern; remove its static definitions.
struct GLCaps {
    bool depth_test = false;
    bool blend = false;
    bool lighting = false;
    bool cull_face = false;
    bool color_material = false;
    bool texture_2d = false;
    bool scissor_test = false;
};

extern GLCaps s_caps;
extern int s_viewport[4];
extern int s_scissor[4];
extern int s_clear_stencil;
extern unsigned char s_depth_mask;
extern unsigned char s_color_mask[4];

} // namespace gl
} // namespace hipgl
