// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - OSMesa-compatible API implementation

#include "hipgl/osmesa_compat.h"
#include "hipgl/gl_api.h"
#include "hipgl/context.h"
#include <cstdlib>
#include <cstring>

struct hipgl_osmesa_context {
    void* hipgl_ctx;      // Context*
    int width;
    int height;
    void* output_buffer;   // Host buffer from MakeCurrent
    unsigned int buffer_type;
};

extern "C" {

HIPGLOSMesaContext HIPGLOSMesaCreateContextExt(
    unsigned int format,
    int depthBits,
    int stencilBits,
    int accumBits,
    HIPGLOSMesaContext sharelist)
{
    (void)depthBits;
    (void)stencilBits;
    (void)accumBits;
    (void)sharelist;

    auto* ctx = static_cast<hipgl_osmesa_context*>(malloc(sizeof(hipgl_osmesa_context)));
    if (!ctx) return nullptr;

    ctx->hipgl_ctx = hipglCreateContext(256, 256);  // Default size, resized in MakeCurrent
    ctx->width = 0;
    ctx->height = 0;
    ctx->output_buffer = nullptr;
    ctx->buffer_type = format;

    return ctx;
}

HIPGLOSMesaContext HIPGLOSMesaCreateContext(
    unsigned int format,
    HIPGLOSMesaContext sharelist)
{
    return HIPGLOSMesaCreateContextExt(format, 24, 0, 0, sharelist);
}

int HIPGLOSMesaMakeCurrent(
    HIPGLOSMesaContext ctx,
    void* buffer,
    unsigned int type,
    int width,
    int height)
{
    if (!ctx || !buffer || width <= 0 || height <= 0) return 0;

    auto* c = static_cast<hipgl_osmesa_context*>(ctx);
    auto* hipgl_ctx = static_cast<hipgl::Context*>(c->hipgl_ctx);

    c->output_buffer = buffer;
    c->buffer_type = type;
    c->width = width;
    c->height = height;

    hipgl_ctx->init(width, height);
    hipgl_ctx->set_osmesa_output(buffer, type);
    hipglMakeCurrent(hipgl_ctx);

    return 1;
}

void HIPGLOSMesaDestroyContext(HIPGLOSMesaContext ctx) {
    if (!ctx) return;

    auto* c = static_cast<hipgl_osmesa_context*>(ctx);
    if (hipgl::Context::current() == c->hipgl_ctx) {
        hipglMakeCurrent(nullptr);
    }
    hipglDestroyContext(c->hipgl_ctx);
    free(ctx);
}

HIPGLOSMesaContext HIPGLOSMesaGetCurrentContext(void) {
    auto* c = hipgl::Context::current();
    if (!c) return nullptr;
    return nullptr;  // We don't track OSMesa context wrapper from Context
}

int HIPGLOSMesaGetDepthBuffer(
    HIPGLOSMesaContext ctx,
    int* width,
    int* height,
    int* bytesPerValue,
    void** buffer)
{
    if (!ctx || !width || !height || !bytesPerValue || !buffer) return 0;
    auto* c = static_cast<hipgl_osmesa_context*>(ctx);
    *width = c->width;
    *height = c->height;
    *bytesPerValue = 4;
    *buffer = nullptr;
    return 1;
}

int HIPGLOSMesaGetColorBuffer(
    HIPGLOSMesaContext ctx,
    int* width,
    int* height,
    int* format,
    void** buffer)
{
    if (!ctx || !width || !height || !format || !buffer) return 0;
    auto* c = static_cast<hipgl_osmesa_context*>(ctx);
    *width = c->width;
    *height = c->height;
    *format = c->buffer_type;
    *buffer = c->output_buffer;
    return 1;
}

} // extern "C"
