// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/texture.h"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstring>

namespace hipgl {

uint32_t Texture::next_id_ = 1;

namespace {

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

size_t bytes_per_pixel(TexFormat format) {
    switch (format) {
        case TEX_RGBA8:   return 4;
        case TEX_RGB8:    return 3;
        case TEX_RGBAF32: return 16;
        case TEX_DEPTH32F: return 4;
        case TEX_R8:      return 1;
        default:          return 4;
    }
}

} // namespace

Texture::Texture()
    : id_(next_id_++)
    , d_ptr_(nullptr)
    , width_(0)
    , height_(0)
    , format_(TEX_RGBA8)
    , size_bytes_(0)
{
}

Texture::~Texture() {
    if (d_ptr_) {
        hipFree(d_ptr_);
        d_ptr_ = nullptr;
    }
}

uint32_t Texture::id() const {
    return id_;
}

void Texture::upload_2d(int width, int height, TexFormat format, const void* data) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Texture::upload_2d: width and height must be positive");
    }

    size_t bpp = bytes_per_pixel(format);
    size_t new_size = static_cast<size_t>(width) * height * bpp;

    if (d_ptr_ && (width_ != width || height_ != height || format_ != format)) {
        hipFree(d_ptr_);
        d_ptr_ = nullptr;
    }

    width_ = width;
    height_ = height;
    format_ = format;
    size_bytes_ = new_size;

    if (!d_ptr_) {
        hipError_t err = hipMalloc(&d_ptr_, new_size);
        check_hip(err, "Texture::upload_2d hipMalloc");
    }

    if (data) {
        hipError_t err = hipMemcpy(d_ptr_, data, new_size, hipMemcpyHostToDevice);
        check_hip(err, "Texture::upload_2d hipMemcpy");
    }
}

void* Texture::device_ptr() const {
    return d_ptr_;
}

int Texture::width() const {
    return width_;
}

int Texture::height() const {
    return height_;
}

TexFormat Texture::format() const {
    return format_;
}

TextureManager::TextureManager()
    : bound_texture_(0)
{
}

uint32_t TextureManager::create_texture() {
    auto tex = std::make_unique<Texture>();
    uint32_t id = tex->id();
    textures_[id] = std::move(tex);
    return id;
}

void TextureManager::delete_texture(uint32_t id) {
    if (bound_texture_ == id) bound_texture_ = 0;
    textures_.erase(id);
}

Texture* TextureManager::get_texture(uint32_t id) {
    auto it = textures_.find(id);
    return (it != textures_.end()) ? it->second.get() : nullptr;
}

void TextureManager::bind_texture(uint32_t id) {
    if (id != 0) {
        if (!get_texture(id)) {
            throw std::runtime_error("TextureManager::bind_texture: texture id not found");
        }
    }
    bound_texture_ = id;
}

uint32_t TextureManager::current_texture() const {
    return bound_texture_;
}

} // namespace hipgl
