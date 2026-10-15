// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/context.h"
#include "hipgl/physics/physics_engine.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <vector>

namespace hipgl {

thread_local Context* Context::current_ = nullptr;

namespace {

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace

Context::Context(int gpu_id)
    : gpu_id_(gpu_id)
{
    hipError_t err = hipSetDevice(gpu_id);
    check_hip(err, "Context: hipSetDevice");
}

Context::~Context() {
    if (current_ == this) {
        current_ = nullptr;
    }
}

Context* Context::current() {
    return current_;
}

void Context::make_current(Context* ctx) {
    current_ = ctx;
    if (ctx) {
        hipError_t err = hipSetDevice(ctx->gpu_id_);
        check_hip(err, "Context::make_current hipSetDevice");
    }
}

Framebuffer& Context::framebuffer() {
    if (!framebuffer_) {
        throw std::runtime_error("Context::framebuffer: not initialized (call init first)");
    }
    return *framebuffer_;
}

Pipeline& Context::pipeline() {
    if (!pipeline_) {
        throw std::runtime_error("Context::pipeline: not initialized (call init first)");
    }
    return *pipeline_;
}

BufferManager& Context::buffers() {
    if (!buffer_mgr_) {
        throw std::runtime_error("Context::buffers: not initialized (call init first)");
    }
    return *buffer_mgr_;
}

TextureManager& Context::textures() {
    if (!texture_mgr_) {
        throw std::runtime_error("Context::textures: not initialized (call init first)");
    }
    return *texture_mgr_;
}

RenderState& Context::state() {
    return state_;
}

void Context::init(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Context::init: width and height must be positive");
    }

    framebuffer_ = std::make_unique<Framebuffer>(width, height, true);
    pipeline_ = std::make_unique<Pipeline>();
    buffer_mgr_ = std::make_unique<BufferManager>();
    texture_mgr_ = std::make_unique<TextureManager>();
    fbo_mgr_ = std::make_unique<FBOManager>();

    init_default_state();
    state_.viewport_w = width;
    state_.viewport_h = height;

    pipeline_->set_framebuffer(framebuffer_.get());
    pipeline_->set_render_state(state_);
}

void Context::set_osmesa_output(void* buffer, unsigned int type) {
    osmesa_output_ = buffer;
    osmesa_output_type_ = type;
}

void Context::copy_to_osmesa_buffer() {
    if (!osmesa_output_ || !framebuffer_) return;
    int w = framebuffer_->width();
    int h = framebuffer_->height();
    if (osmesa_output_type_ == 0x1908) {
        read_pixels(0, 0, w, h, static_cast<uint8_t*>(osmesa_output_));
    }
}

void Context::read_pixels(int x, int y, int w, int h, uint8_t* rgba) {
    if (!framebuffer_ || !pipeline_ || !rgba) return;

    int fw = framebuffer_->width();
    int fh = framebuffer_->height();

    x = std::max(0, std::min(x, fw - 1));
    y = std::max(0, std::min(y, fh - 1));
    w = std::max(1, std::min(w, fw - x));
    h = std::max(1, std::min(h, fh - y));

    if (w <= 0 || h <= 0) return;

    const float* full_rgba = framebuffer_->read_color_pinned(pipeline_->stream());
    if (!full_rgba) return;

    for (int row = 0; row < h; ++row) {
        int src_y = fh - 1 - (y + row);
        const float* src = &full_rgba[(src_y * fw + x) * 4];
        uint8_t* dst = &rgba[row * w * 4];
        for (int col = 0; col < w; ++col) {
            for (int c = 0; c < 4; ++c) {
                float f = std::max(0.0f, std::min(1.0f, src[col * 4 + c]));
                dst[col * 4 + c] = static_cast<uint8_t>(f * 255.0f + 0.5f);
            }
        }
    }
}

void Context::set_physics_engine(physics::PhysicsEngine* engine) {
    physics_engine_ = engine;
}

FBOManager& Context::fbos() {
    if (!fbo_mgr_) {
        throw std::runtime_error("Context::fbos: not initialized (call init first)");
    }
    return *fbo_mgr_;
}

void Context::init_default_state() {
    state_ = RenderState{};
    state_.depth_test_enabled = true;
    state_.depth_write_enabled = true;
    state_.blend_enabled = false;
    state_.lighting_enabled = false;
    state_.backface_cull_enabled = false;
    state_.texture_2d_enabled = false;
    state_.color_material_enabled = false;
    state_.depth_func = DEPTH_LESS;
    state_.blend_src = BLEND_SRC_ALPHA;
    state_.blend_dst = BLEND_ONE_MINUS_SRC_ALPHA;
    state_.clear_color[0] = 0.0f;
    state_.clear_color[1] = 0.0f;
    state_.clear_color[2] = 0.0f;
    state_.clear_color[3] = 1.0f;
    state_.clear_depth = 1.0f;
    state_.clear_stencil = 0;
    state_.viewport_x = 0;
    state_.viewport_y = 0;
    state_.viewport_w = 0;
    state_.viewport_h = 0;

    for (int i = 0; i < 16; ++i) {
        state_.model_view[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        state_.projection[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        state_.mvp[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 9; ++i) {
        state_.normal_matrix[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    state_.bound_texture_id = 0;
    state_.point_size = 1.0f;
    state_.line_width = 1.0f;
    state_.shade_model = 1;
    state_.polygon_mode = 0;
    state_.front_face = 0;
    state_.cull_face = 0;
    state_.gouraud_shading = 0;
}

} // namespace hipgl
