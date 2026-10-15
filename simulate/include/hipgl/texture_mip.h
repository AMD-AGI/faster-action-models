#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

namespace hipgl {

constexpr int MAX_TEXTURE_MIP_LEVELS = 16;

// Byte offsets refer to TextureState::tex_data. Cube levels contain six
// width-by-width faces in GL order, so their stored height is 6 * width.
struct TextureMipLevel {
    int offset;
    int width;
    int height;
};

} // namespace hipgl
