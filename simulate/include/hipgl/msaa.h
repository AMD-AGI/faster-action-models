#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "types.h"

namespace hipgl::kernels {
void launch_clear_msaa(DeviceFramebuffer fb, const RenderState& state,
                       bool color, bool depth, hipStream_t stream);
void launch_resolve_msaa(DeviceFramebuffer fb, hipStream_t stream);
void launch_fine_rasterize_msaa(const TriangleSetup* tris,
    const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture, hipStream_t stream);
}
