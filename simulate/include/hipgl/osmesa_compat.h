#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// hipGL - OSMesa-compatible API for MuJoCo and other OSMesa users
// Drop-in replacement when HIPGL_REPLACE_OSMESA is defined.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hipgl_osmesa_context* HIPGLOSMesaContext;

// OSMesa format constants (match OSMesa)
#define HIPGL_OSMESA_RGBA       0x1908
#define HIPGL_OSMESA_BGRA      0x1
#define HIPGL_OSMESA_RGB       0x1907
#define HIPGL_OSMESA_BGR       0x2
#define HIPGL_OSMESA_RGB_565   0x3
#define HIPGL_OSMESA_OSMESA_COLOR_INDEX 0x4

// OSMesa type constants
#define HIPGL_OSMESA_UNSIGNED_BYTE  0x1401
#define HIPGL_OSMESA_UNSIGNED_SHORT 0x1403
#define HIPGL_OSMESA_FLOAT          0x1406

// Create/destroy context
HIPGLOSMesaContext HIPGLOSMesaCreateContextExt(
    unsigned int format,
    int depthBits,
    int stencilBits,
    int accumBits,
    HIPGLOSMesaContext sharelist);

HIPGLOSMesaContext HIPGLOSMesaCreateContext(
    unsigned int format,
    HIPGLOSMesaContext sharelist);

int HIPGLOSMesaMakeCurrent(
    HIPGLOSMesaContext ctx,
    void* buffer,
    unsigned int type,
    int width,
    int height);

void HIPGLOSMesaDestroyContext(HIPGLOSMesaContext ctx);

HIPGLOSMesaContext HIPGLOSMesaGetCurrentContext(void);

int HIPGLOSMesaGetDepthBuffer(
    HIPGLOSMesaContext ctx,
    int* width,
    int* height,
    int* bytesPerValue,
    void** buffer);

int HIPGLOSMesaGetColorBuffer(
    HIPGLOSMesaContext ctx,
    int* width,
    int* height,
    int* format,
    void** buffer);

#ifdef __cplusplus
}
#endif

#ifdef HIPGL_REPLACE_OSMESA
#define OSMesaContext           HIPGLOSMesaContext
#define OSMesaCreateContextExt   HIPGLOSMesaCreateContextExt
#define OSMesaCreateContext      HIPGLOSMesaCreateContext
#define OSMesaMakeCurrent        HIPGLOSMesaMakeCurrent
#define OSMesaDestroyContext     HIPGLOSMesaDestroyContext
#define OSMesaGetCurrentContext  HIPGLOSMesaGetCurrentContext
#define OSMesaGetDepthBuffer     HIPGLOSMesaGetDepthBuffer
#define OSMesaGetColorBuffer     HIPGLOSMesaGetColorBuffer
#define OSMESA_RGBA              HIPGL_OSMESA_RGBA
#define OSMESA_BGRA              HIPGL_OSMESA_BGRA
#define OSMESA_RGB               HIPGL_OSMESA_RGB
#define OSMESA_BGR               HIPGL_OSMESA_BGR
#define OSMESA_RGB_565           HIPGL_OSMESA_RGB_565
#define OSMESA_COLOR_INDEX        HIPGL_OSMESA_OSMESA_COLOR_INDEX
#define OSMESA_UNSIGNED_BYTE     HIPGL_OSMESA_UNSIGNED_BYTE
#define OSMESA_UNSIGNED_SHORT    HIPGL_OSMESA_UNSIGNED_SHORT
#define OSMESA_FLOAT             HIPGL_OSMESA_FLOAT
#endif
