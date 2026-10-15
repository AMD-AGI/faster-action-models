// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Physics world-skip mask (world skip).
//
// A per-world uint8 array in device memory: 1 = step the world, 0 = freeze it.
// The gated kernels (fused FK/integrate head and final head, geometry + AABBs,
// broadphase, Newton, fallback integrate) return before touching a frozen
// world, so its qpos/qvel/body poses stay exactly as they were, it adds no
// contact pairs, and the per-world stages that are not gated (mass matrix,
// OSC, constraint setup, ...) only recompute its outputs from that frozen
// state. The head still zeroes a frozen world's contact/constraint counters.
//
// The mask pointer is a `static __device__` per translation unit (no RDC), so
// every gated kernel file defines its own copy plus a host setter with
// HIPGL_WORLD_MASK_DEFINE. It is process-global: PhysicsEngine pushes its own
// pointer (or nullptr) on its stream before every masked step. nullptr, the
// initial value, is every world stepping, i.e. the unmasked kernels exactly.
#pragma once
#include <hip/hip_runtime.h>

// `src` must stay valid until the async copy has run (PhysicsEngine passes the
// address of a member).
#define HIPGL_WORLD_MASK_DEFINE(SETTER)                                             \
    static __device__ const unsigned char* g_world_active = nullptr;                \
    void SETTER(const unsigned char* const* src, hipStream_t stream) {              \
        hipMemcpyToSymbolAsync(HIP_SYMBOL(g_world_active), src,                     \
                               sizeof(const unsigned char*), 0,                     \
                               hipMemcpyHostToDevice, stream);                      \
    }

#define HIPGL_WORLD_FROZEN(env) (g_world_active != nullptr && g_world_active[(env)] == 0)
