#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include "framebuffer.h"
#include "texture.h"
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace hipgl {

class Renderbuffer {
public:
    Renderbuffer();
    ~Renderbuffer();

    uint32_t id() const { return id_; }
    void storage(unsigned int internalformat, int width, int height);
    void storage_multisample(int samples, unsigned int internalformat, int width, int height);

    int width() const { return width_; }
    int height() const { return height_; }
    unsigned int format() const { return format_; }
    int samples() const { return samples_; }

    // Device buffers
    float* color_device() const { return d_color_; }
    float* depth_device() const { return d_depth_; }

private:
    uint32_t id_;
    int width_, height_;
    unsigned int format_;
    int samples_;
    float* d_color_;
    float* d_depth_;
    static uint32_t next_id_;

    void allocate();
    void deallocate();
};

class FramebufferObject {
public:
    FramebufferObject();
    ~FramebufferObject() = default;

    uint32_t id() const { return id_; }

    void attach_texture_2d(unsigned int attachment, unsigned int tex_target, uint32_t texture_id, int level);
    void attach_renderbuffer(unsigned int attachment, uint32_t renderbuffer_id);

    unsigned int check_status() const;

    // Attachment info
    uint32_t color_renderbuffer_id() const { return color_rb_id_; }
    uint32_t depth_renderbuffer_id() const { return depth_rb_id_; }
    uint32_t stencil_renderbuffer_id() const { return stencil_rb_id_; }
    uint32_t color_texture_id() const { return color_tex_id_; }
    uint32_t depth_texture_id() const { return depth_tex_id_; }

private:
    uint32_t id_;
    uint32_t color_rb_id_;
    uint32_t depth_rb_id_;
    uint32_t stencil_rb_id_;
    uint32_t color_tex_id_;
    uint32_t depth_tex_id_;
    int color_tex_level_;
    static uint32_t next_id_;
};

class FBOManager {
public:
    FBOManager();
    ~FBOManager() = default;

    // Framebuffer objects
    uint32_t gen_framebuffer();
    void delete_framebuffer(uint32_t id);
    void bind_framebuffer(unsigned int target, uint32_t id);
    FramebufferObject* get_fbo(uint32_t id);
    uint32_t current_fbo() const { return bound_fbo_; }
    bool is_default_fbo() const { return bound_fbo_ == 0; }

    // Renderbuffer objects
    uint32_t gen_renderbuffer();
    void delete_renderbuffer(uint32_t id);
    void bind_renderbuffer(unsigned int target, uint32_t id);
    Renderbuffer* get_renderbuffer(uint32_t id);
    uint32_t current_renderbuffer() const { return bound_rb_; }

    // Build a DeviceFramebuffer from current FBO state
    // If FBO 0 (default), returns the context's default framebuffer
    // Otherwise, builds from FBO attachments
    DeviceFramebuffer resolve_device_fb(Framebuffer& default_fb, TextureManager& tex_mgr);

private:
    std::unordered_map<uint32_t, std::unique_ptr<FramebufferObject>> fbos_;
    std::unordered_map<uint32_t, std::unique_ptr<Renderbuffer>> renderbuffers_;
    uint32_t bound_fbo_;
    uint32_t bound_rb_;

    // Scratch framebuffer for FBO rendering
    std::unique_ptr<Framebuffer> fbo_framebuffer_;
};

} // namespace hipgl
