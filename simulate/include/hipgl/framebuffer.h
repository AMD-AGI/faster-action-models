#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include <cstddef>

namespace hipgl {

class Framebuffer {
public:
    Framebuffer(int width, int height, bool has_depth = true, int samples = 1);
    ~Framebuffer();

    // No copy, allow move
    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;
    Framebuffer(Framebuffer&& other) noexcept;
    Framebuffer& operator=(Framebuffer&& other) noexcept;

    int width() const;
    int height() const;
    int samples() const { return samples_; }
    // Resolve colors by averaging samples; depth is taken from sample zero.
    void resolve(hipStream_t stream = nullptr) const;

    DeviceFramebuffer device_fb() const;

    // Read back to host (uses pinned staging buffer for async transfer)
    void read_color(float* host_rgba) const;
    void read_color_async(float* host_rgba, hipStream_t stream) const;
    void read_color_rgba8(uint8_t* host_rgba) const;
    void read_depth(float* host_depth) const;

    // Direct access to pinned staging buffer (valid after read_color_pinned)
    float* pinned_color_buffer() const { return h_color_pinned_; }

    // Read into pinned buffer + sync, returns pinned pointer
    const float* read_color_pinned(hipStream_t stream) const;

    // Resize (reallocates)
    void resize(int new_width, int new_height);

    void set_capture_attrs(bool enabled);
    bool capture_attrs() const { return d_attr_ != nullptr; }
    const float* device_attrs() const { return d_attr_; }

private:
    int width_;
    int height_;
    float* d_color_;
    float* d_depth_;
    float* d_oit_accum_;
    float* d_oit_revealage_;
    float* d_attr_;          // optional per-fragment attribute capture (debug)
    mutable float* h_color_pinned_;
    bool has_depth_;
    int samples_ = 1;
    uint32_t* d_sample_color_ = nullptr;
    float* d_sample_depth_ = nullptr;

    void allocate();
    void deallocate();
};

} // namespace hipgl
