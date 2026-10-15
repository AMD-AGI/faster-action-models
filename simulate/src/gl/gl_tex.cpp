// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/gl_api.h"
#include "hipgl/gl_tex_impl.h"
#include "hipgl/context.h"
#include "hipgl/texture.h"
#include <vector>

extern "C" {

void hipglTexImage2D_impl(unsigned int target, int level, int internalformat,
                          int width, int height, int border,
                          unsigned int format, unsigned int type, const void* data) {
    (void)internalformat;
    (void)border;
    if (target != 0x0DE1) return;  // GL_TEXTURE_2D

    auto* c = hipgl::Context::current();
    if (!c) return;

    uint32_t tex_id = c->textures().current_texture();
    if (tex_id == 0) return;

    auto* tex = c->textures().get_texture(tex_id);
    if (!tex) return;

    // Use TEX_RGBAF32 for GPU storage (float RGBA)
    hipgl::TexFormat storage_format = hipgl::TEX_RGBAF32;

    // If data is null, just allocate storage
    if (!data) {
        tex->upload_2d(width, height, storage_format, nullptr);
        return;
    }

    // Convert to RGBA float on host, then upload
    int num_pixels = width * height;
    std::vector<float> rgba(static_cast<size_t>(num_pixels) * 4);

    if (type == 0x1401) {  // GL_UNSIGNED_BYTE
        const uint8_t* src = static_cast<const uint8_t*>(data);
        int channels = (format == 0x1908) ? 4 : (format == 0x1907) ? 3 : (format == 0x1909 || format == 0x1903) ? 1 : 4;
        if (format == 0x190A) channels = 2;  // GL_LUMINANCE_ALPHA
        for (int i = 0; i < num_pixels; i++) {
            float r = src[i * channels + 0] / 255.0f;
            float g = (channels > 1) ? src[i * channels + 1] / 255.0f : r;
            float b = (channels > 2) ? src[i * channels + 2] / 255.0f : r;
            float a = (channels > 3) ? src[i * channels + 3] / 255.0f : 1.0f;
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = a;
        }
    } else if (type == 0x1406) {  // GL_FLOAT
        const float* src = static_cast<const float*>(data);
        int channels = (format == 0x1908) ? 4 : (format == 0x1907) ? 3 : (format == 0x1909 || format == 0x1903) ? 1 : 4;
        if (format == 0x190A) channels = 2;
        for (int i = 0; i < num_pixels; i++) {
            float r = src[i * channels + 0];
            float g = (channels > 1) ? src[i * channels + 1] : r;
            float b = (channels > 2) ? src[i * channels + 2] : r;
            float a = (channels > 3) ? src[i * channels + 3] : 1.0f;
            rgba[i * 4 + 0] = r;
            rgba[i * 4 + 1] = g;
            rgba[i * 4 + 2] = b;
            rgba[i * 4 + 3] = a;
        }
    } else {
        return;  // Unsupported type
    }

    tex->upload_2d(width, height, storage_format, rgba.data());
}

void hipglTexParameteri_impl(unsigned int target, unsigned int pname, int param) {
    if (target != 0x0DE1) return;  // GL_TEXTURE_2D

    auto* c = hipgl::Context::current();
    if (!c) return;

    uint32_t tex_id = c->textures().current_texture();
    if (tex_id == 0) return;

    auto* tex = c->textures().get_texture(tex_id);
    if (!tex) return;

    auto& p = tex->params();
    switch (pname) {
        case 0x2800: p.mag_filter = param; break;   // GL_TEXTURE_MAG_FILTER
        case 0x2801: p.min_filter = param; break;   // GL_TEXTURE_MIN_FILTER
        case 0x2802: p.wrap_s = param; break;       // GL_TEXTURE_WRAP_S
        case 0x2803: p.wrap_t = param; break;       // GL_TEXTURE_WRAP_T
        default: break;
    }
}

}  // extern "C"
