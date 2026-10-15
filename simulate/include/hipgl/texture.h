#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace hipgl {

struct TextureParams {
    int mag_filter = 0x2601;  // GL_LINEAR
    int min_filter = 0x2601;  // GL_LINEAR
    int wrap_s = 0x2901;      // GL_REPEAT
    int wrap_t = 0x2901;      // GL_REPEAT
};

class Texture {
public:
    Texture();
    ~Texture();

    uint32_t id() const;
    void upload_2d(int width, int height, TexFormat format, const void* data);
    void* device_ptr() const;
    int width() const;
    int height() const;
    TexFormat format() const;

    TextureParams& params() { return params_; }
    const TextureParams& params() const { return params_; }

private:
    uint32_t id_;
    void* d_ptr_;
    int width_;
    int height_;
    TexFormat format_;
    size_t size_bytes_;
    TextureParams params_;
    static uint32_t next_id_;
};

class TextureManager {
public:
    TextureManager();
    ~TextureManager() = default;

    uint32_t create_texture();
    void delete_texture(uint32_t id);
    Texture* get_texture(uint32_t id);
    void bind_texture(uint32_t id);
    uint32_t current_texture() const;

private:
    std::unordered_map<uint32_t, std::unique_ptr<Texture>> textures_;
    uint32_t bound_texture_;
};

} // namespace hipgl
