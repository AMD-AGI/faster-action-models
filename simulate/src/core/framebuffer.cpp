// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/framebuffer.h"
#include "hipgl/msaa.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace hipgl {

namespace {

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace

Framebuffer::Framebuffer(int width, int height, bool has_depth, int samples)
    : width_(width)
    , height_(height)
    , d_color_(nullptr)
    , d_depth_(nullptr)
    , d_attr_(nullptr)
    , d_oit_accum_(nullptr)
    , d_oit_revealage_(nullptr)
    , h_color_pinned_(nullptr)
    , has_depth_(has_depth)
    , samples_(samples)
{
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Framebuffer: width and height must be positive");
    }
    if (samples != 1 && samples != 4)
        throw std::runtime_error("Framebuffer: samples must be 1 or 4");
    if (samples == 4 && !has_depth)
        throw std::runtime_error("Framebuffer: MSAA requires a depth attachment");
    allocate();
}

Framebuffer::~Framebuffer() {
    deallocate();
}

Framebuffer::Framebuffer(Framebuffer&& other) noexcept
    : width_(other.width_)
    , height_(other.height_)
    , d_color_(other.d_color_)
    , d_depth_(other.d_depth_)
    , d_attr_(other.d_attr_)
    , d_oit_accum_(other.d_oit_accum_)
    , d_oit_revealage_(other.d_oit_revealage_)
    , h_color_pinned_(other.h_color_pinned_)
    , has_depth_(other.has_depth_)
    , samples_(other.samples_)
    , d_sample_color_(other.d_sample_color_)
    , d_sample_depth_(other.d_sample_depth_)
{
    other.d_sample_color_ = nullptr;
    other.d_sample_depth_ = nullptr;
    other.d_color_ = nullptr;
    other.d_depth_ = nullptr;
    other.d_attr_ = nullptr;
    other.d_oit_accum_ = nullptr;
    other.d_oit_revealage_ = nullptr;
    other.h_color_pinned_ = nullptr;
    other.width_ = 0;
    other.height_ = 0;
}

Framebuffer& Framebuffer::operator=(Framebuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        width_ = other.width_;
        height_ = other.height_;
        d_color_ = other.d_color_;
        d_depth_ = other.d_depth_;
        d_attr_ = other.d_attr_;
        d_oit_accum_ = other.d_oit_accum_;
        d_oit_revealage_ = other.d_oit_revealage_;
        h_color_pinned_ = other.h_color_pinned_;
        has_depth_ = other.has_depth_;
        samples_ = other.samples_;
        d_sample_color_ = other.d_sample_color_;
        d_sample_depth_ = other.d_sample_depth_;
        other.d_sample_color_ = nullptr;
        other.d_sample_depth_ = nullptr;
        other.d_color_ = nullptr;
        other.d_depth_ = nullptr;
        other.d_attr_ = nullptr;
        other.d_oit_accum_ = nullptr;
        other.d_oit_revealage_ = nullptr;
        other.h_color_pinned_ = nullptr;
        other.width_ = 0;
        other.height_ = 0;
    }
    return *this;
}

int Framebuffer::width() const {
    return width_;
}

int Framebuffer::height() const {
    return height_;
}

DeviceFramebuffer Framebuffer::device_fb() const {
    DeviceFramebuffer fb = {};
    fb.color_buffer = d_color_;
    fb.depth_buffer = d_depth_;
    fb.stencil_buffer = nullptr;
    fb.attr_buffer = d_attr_;
    fb.oit_accum = d_oit_accum_;
    fb.oit_revealage = d_oit_revealage_;
    fb.width = width_;
    fb.height = height_;
    fb.pitch = width_ * 4;
    fb.sample_color = d_sample_color_;
    fb.sample_depth = d_sample_depth_;
    return fb;
}

void Framebuffer::resolve(hipStream_t stream) const {
    if (samples_ == 4) kernels::launch_resolve_msaa(device_fb(), stream);
}

void Framebuffer::read_color(float* host_rgba) const {
    if (!host_rgba || !d_color_) return;
    resolve();
    size_t size = static_cast<size_t>(width_) * height_ * 4 * sizeof(float);
    hipError_t err = hipMemcpy(host_rgba, d_color_, size, hipMemcpyDeviceToHost);
    check_hip(err, "Framebuffer::read_color");
}

void Framebuffer::read_color_async(float* host_rgba, hipStream_t stream) const {
    if (!host_rgba || !d_color_) return;
    resolve(stream);
    size_t size = static_cast<size_t>(width_) * height_ * 4 * sizeof(float);
    hipError_t err = hipMemcpyAsync(host_rgba, d_color_, size, hipMemcpyDeviceToHost, stream);
    check_hip(err, "Framebuffer::read_color_async");
}

const float* Framebuffer::read_color_pinned(hipStream_t stream) const {
    if (!d_color_ || !h_color_pinned_) return nullptr;
    resolve(stream);
    size_t size = static_cast<size_t>(width_) * height_ * 4 * sizeof(float);
    hipError_t err = hipMemcpyAsync(h_color_pinned_, d_color_, size, hipMemcpyDeviceToHost, stream);
    check_hip(err, "Framebuffer::read_color_pinned");
    hipStreamSynchronize(stream);
    return h_color_pinned_;
}

void Framebuffer::read_color_rgba8(uint8_t* host_rgba) const {
    if (!host_rgba || !d_color_) return;
    size_t pixel_count = static_cast<size_t>(width_) * height_;
    std::vector<float> tmp(pixel_count * 4);
    read_color(tmp.data());
    for (size_t i = 0; i < pixel_count * 4; ++i) {
        float f = std::max(0.0f, std::min(1.0f, tmp[i]));
        host_rgba[i] = static_cast<uint8_t>(f * 255.0f + 0.5f);
    }
}

void Framebuffer::read_depth(float* host_depth) const {
    if (!host_depth || !d_depth_) return;
    resolve();
    size_t size = static_cast<size_t>(width_) * height_ * sizeof(float);
    hipError_t err = hipMemcpy(host_depth, d_depth_, size, hipMemcpyDeviceToHost);
    check_hip(err, "Framebuffer::read_depth");
}

void Framebuffer::resize(int new_width, int new_height) {
    if (new_width <= 0 || new_height <= 0) {
        throw std::runtime_error("Framebuffer::resize: width and height must be positive");
    }
    if (new_width == width_ && new_height == height_) return;
    // deallocate() frees the attr capture buffer too, so remember whether it
    // was on and restore it at the new size -- otherwise a resize silently
    // turns capture off and the readback returns stale zeros.
    bool was_capturing = (d_attr_ != nullptr);
    deallocate();
    width_ = new_width;
    height_ = new_height;
    allocate();
    if (was_capturing) set_capture_attrs(true);
}

void Framebuffer::allocate() {
    size_t pixel_count = static_cast<size_t>(width_) * height_;
    size_t color_size = pixel_count * 4 * sizeof(float);
    hipError_t err = hipMalloc(&d_color_, color_size);
    check_hip(err, "Framebuffer: hipMalloc color");

    err = hipHostMalloc(&h_color_pinned_, color_size, hipHostMallocDefault);
    check_hip(err, "Framebuffer: hipHostMalloc pinned color");

    if (has_depth_) {
        size_t depth_size = pixel_count * sizeof(float);
        err = hipMalloc(&d_depth_, depth_size);
        check_hip(err, "Framebuffer: hipMalloc depth");
    }

    if (samples_ == 4) {
        check_hip(hipMalloc(&d_sample_color_, pixel_count * 4 * sizeof(uint32_t)),
                  "Framebuffer: hipMalloc sample color");
        check_hip(hipMalloc(&d_sample_depth_, pixel_count * 4 * sizeof(float)),
                  "Framebuffer: hipMalloc sample depth");
    }

    err = hipMalloc(&d_oit_accum_, pixel_count * 4 * sizeof(float));
    check_hip(err, "Framebuffer: hipMalloc oit_accum");
    err = hipMalloc(&d_oit_revealage_, pixel_count * sizeof(float));
    check_hip(err, "Framebuffer: hipMalloc oit_revealage");
}

void Framebuffer::deallocate() {
    if (d_sample_color_) { hipFree(d_sample_color_); d_sample_color_ = nullptr; }
    if (d_sample_depth_) { hipFree(d_sample_depth_); d_sample_depth_ = nullptr; }
    if (d_color_) { hipFree(d_color_); d_color_ = nullptr; }
    if (h_color_pinned_) { hipHostFree(h_color_pinned_); h_color_pinned_ = nullptr; }
    if (d_depth_) { hipFree(d_depth_); d_depth_ = nullptr; }
    if (d_attr_) { hipFree(d_attr_); d_attr_ = nullptr; }
    if (d_oit_accum_) { hipFree(d_oit_accum_); d_oit_accum_ = nullptr; }
    if (d_oit_revealage_) { hipFree(d_oit_revealage_); d_oit_revealage_ = nullptr; }
}

void Framebuffer::set_capture_attrs(bool enabled) {
    if (enabled && samples_ != 1)
        throw std::runtime_error("Framebuffer: per-fragment attribute capture is unavailable with MSAA");
    if (enabled == (d_attr_ != nullptr)) return;
    if (!enabled) { hipFree(d_attr_); d_attr_ = nullptr; return; }
    size_t n = static_cast<size_t>(width_) * height_ * FRAG_ATTR_COUNT * sizeof(float);
    hipError_t err = hipMalloc(&d_attr_, n);
    check_hip(err, "Framebuffer: hipMalloc attr");
    hipMemset(d_attr_, 0, n);
}

} // namespace hipgl
