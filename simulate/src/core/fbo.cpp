// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/fbo.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>

namespace hipgl {

// OpenGL constants for renderbuffer formats
namespace {
constexpr unsigned int GL_DEPTH_COMPONENT   = 0x1902;
constexpr unsigned int GL_DEPTH_COMPONENT24 = 0x81A6;
constexpr unsigned int GL_DEPTH24_STENCIL8  = 0x88F0;
constexpr unsigned int GL_RGBA              = 0x1908;
constexpr unsigned int GL_RGB               = 0x1907;

constexpr unsigned int GL_COLOR_ATTACHMENT0     = 0x8CE0;
constexpr unsigned int GL_DEPTH_ATTACHMENT     = 0x8D00;
constexpr unsigned int GL_STENCIL_ATTACHMENT   = 0x8D20;
constexpr unsigned int GL_DEPTH_STENCIL_ATTACHMENT = 0x821A;

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace

// ── Renderbuffer ────────────────────────────────────────────────────────────

uint32_t Renderbuffer::next_id_ = 1;

Renderbuffer::Renderbuffer()
    : id_(next_id_++)
    , width_(0)
    , height_(0)
    , format_(0)
    , samples_(0)
    , d_color_(nullptr)
    , d_depth_(nullptr)
{
}

Renderbuffer::~Renderbuffer() {
    deallocate();
}

void Renderbuffer::allocate() {
    deallocate();

    if (width_ <= 0 || height_ <= 0) return;

    bool is_depth = (format_ == GL_DEPTH_COMPONENT || format_ == GL_DEPTH_COMPONENT24 ||
                     format_ == GL_DEPTH24_STENCIL8);

    if (is_depth) {
        size_t depth_size = static_cast<size_t>(width_) * height_ * sizeof(float);
        hipError_t err = hipMalloc(&d_depth_, depth_size);
        check_hip(err, "Renderbuffer::allocate depth");
    } else {
        size_t color_size = static_cast<size_t>(width_) * height_ * 4 * sizeof(float);
        hipError_t err = hipMalloc(&d_color_, color_size);
        check_hip(err, "Renderbuffer::allocate color");
    }
}

void Renderbuffer::deallocate() {
    if (d_color_) {
        hipFree(d_color_);
        d_color_ = nullptr;
    }
    if (d_depth_) {
        hipFree(d_depth_);
        d_depth_ = nullptr;
    }
}

void Renderbuffer::storage(unsigned int internalformat, int width, int height) {
    storage_multisample(0, internalformat, width, height);
}

void Renderbuffer::storage_multisample(int samples, unsigned int internalformat, int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Renderbuffer::storage: width and height must be positive");
    }

    bool format_changed = (width_ != width || height_ != height || format_ != internalformat || samples_ != samples);
    width_ = width;
    height_ = height;
    format_ = internalformat;
    samples_ = (samples > 0) ? samples : 1;

    if (format_changed) {
        allocate();
    }
}

// ── FramebufferObject ──────────────────────────────────────────────────────

uint32_t FramebufferObject::next_id_ = 1;

FramebufferObject::FramebufferObject()
    : id_(next_id_++)
    , color_rb_id_(0)
    , depth_rb_id_(0)
    , stencil_rb_id_(0)
    , color_tex_id_(0)
    , depth_tex_id_(0)
    , color_tex_level_(0)
{
}

void FramebufferObject::attach_texture_2d(unsigned int attachment, unsigned int tex_target,
                                          uint32_t texture_id, int level) {
    (void)tex_target;
    if (attachment == GL_COLOR_ATTACHMENT0) {
        color_tex_id_ = texture_id;
        color_tex_level_ = level;
        color_rb_id_ = 0;
    } else if (attachment == GL_DEPTH_ATTACHMENT) {
        depth_tex_id_ = texture_id;
        depth_rb_id_ = 0;
    } else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        depth_tex_id_ = texture_id;
        depth_rb_id_ = 0;
        stencil_rb_id_ = 0;
    }
}

void FramebufferObject::attach_renderbuffer(unsigned int attachment, uint32_t renderbuffer_id) {
    if (attachment == GL_COLOR_ATTACHMENT0) {
        color_rb_id_ = renderbuffer_id;
        color_tex_id_ = 0;
    } else if (attachment == GL_DEPTH_ATTACHMENT) {
        depth_rb_id_ = renderbuffer_id;
        depth_tex_id_ = 0;
    } else if (attachment == GL_STENCIL_ATTACHMENT) {
        stencil_rb_id_ = renderbuffer_id;
    } else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
        depth_rb_id_ = renderbuffer_id;
        stencil_rb_id_ = renderbuffer_id;
        depth_tex_id_ = 0;
    }
}

unsigned int FramebufferObject::check_status() const {
    constexpr unsigned int GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
    if (color_rb_id_ != 0 || color_tex_id_ != 0 || depth_rb_id_ != 0 || depth_tex_id_ != 0) {
        return GL_FRAMEBUFFER_COMPLETE;
    }
    return 0;  // Incomplete
}

// ── FBOManager ───────────────────────────────────────────────────────────────

FBOManager::FBOManager()
    : bound_fbo_(0)
    , bound_rb_(0)
{
}

uint32_t FBOManager::gen_framebuffer() {
    auto fbo = std::make_unique<FramebufferObject>();
    uint32_t id = fbo->id();
    fbos_[id] = std::move(fbo);
    return id;
}

void FBOManager::delete_framebuffer(uint32_t id) {
    if (bound_fbo_ == id) bound_fbo_ = 0;
    fbos_.erase(id);
}

void FBOManager::bind_framebuffer(unsigned int target, uint32_t id) {
    (void)target;
    bound_fbo_ = id;
}

FramebufferObject* FBOManager::get_fbo(uint32_t id) {
    auto it = fbos_.find(id);
    return (it != fbos_.end()) ? it->second.get() : nullptr;
}

uint32_t FBOManager::gen_renderbuffer() {
    auto rb = std::make_unique<Renderbuffer>();
    uint32_t id = rb->id();
    renderbuffers_[id] = std::move(rb);
    return id;
}

void FBOManager::delete_renderbuffer(uint32_t id) {
    if (bound_rb_ == id) bound_rb_ = 0;
    renderbuffers_.erase(id);
}

void FBOManager::bind_renderbuffer(unsigned int target, uint32_t id) {
    (void)target;
    bound_rb_ = id;
}

Renderbuffer* FBOManager::get_renderbuffer(uint32_t id) {
    auto it = renderbuffers_.find(id);
    return (it != renderbuffers_.end()) ? it->second.get() : nullptr;
}

DeviceFramebuffer FBOManager::resolve_device_fb(Framebuffer& default_fb, TextureManager& tex_mgr) {
    if (bound_fbo_ == 0) return default_fb.device_fb();

    auto* fbo = get_fbo(bound_fbo_);
    if (!fbo) return default_fb.device_fb();

    DeviceFramebuffer dfb = {};
    dfb.attr_buffer = nullptr;
    dfb.color_buffer = nullptr;
    dfb.depth_buffer = nullptr;
    dfb.stencil_buffer = nullptr;
    dfb.width = 0;
    dfb.height = 0;
    dfb.pitch = 0;

    // Color from renderbuffer
    if (fbo->color_renderbuffer_id()) {
        auto* rb = get_renderbuffer(fbo->color_renderbuffer_id());
        if (rb) {
            dfb.color_buffer = rb->color_device();
            dfb.width = rb->width();
            dfb.height = rb->height();
            dfb.pitch = rb->width() * 4;
        }
    }
    // Or color from texture
    else if (fbo->color_texture_id()) {
        auto* tex = tex_mgr.get_texture(fbo->color_texture_id());
        if (tex) {
            dfb.color_buffer = static_cast<float*>(tex->device_ptr());
            dfb.width = tex->width();
            dfb.height = tex->height();
            dfb.pitch = tex->width() * 4;
        }
    }

    // Depth from renderbuffer
    if (fbo->depth_renderbuffer_id()) {
        auto* rb = get_renderbuffer(fbo->depth_renderbuffer_id());
        if (rb) {
            dfb.depth_buffer = rb->depth_device();
            if (dfb.width == 0) {
                dfb.width = rb->width();
                dfb.height = rb->height();
            }
        }
    }
    // Depth from texture
    else if (fbo->depth_texture_id()) {
        auto* tex = tex_mgr.get_texture(fbo->depth_texture_id());
        if (tex) {
            dfb.depth_buffer = static_cast<float*>(tex->device_ptr());
            if (dfb.width == 0) {
                dfb.width = tex->width();
                dfb.height = tex->height();
            }
        }
    }

    return dfb;
}

} // namespace hipgl
