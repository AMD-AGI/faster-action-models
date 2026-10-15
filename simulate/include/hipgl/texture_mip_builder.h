#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host-only RGB8 mip generation. Kept separate from the device metadata so
// including TextureState in a kernel does not pull in containers or exceptions.
#include "hipgl/texture_mip.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hipgl {

struct TextureMipPyramid {
    std::vector<unsigned char> data;
    std::vector<TextureMipLevel> levels;
};

namespace texture_mip_detail {

struct AxisTap {
    int lo, hi, weight;
};

inline std::vector<AxisTap> resize_axis(int source, int dest) {
    std::vector<AxisTap> taps(dest);
    for (int i = 0; i < dest; ++i) {
        // ((i + .5) * source / dest - .5), expressed exactly. llvmpipe
        // quantizes the bilinear weight to 8 fractional bits, nearest-even.
        const int64_t denominator = int64_t(2) * dest;
        const int64_t numerator = (int64_t(2) * i + 1) * source - dest;
        const int lo = int(numerator / denominator);
        const int64_t scaled = (numerator % denominator) * 256;
        int weight = int(scaled / denominator);
        const int64_t remainder = scaled % denominator;
        if (2 * remainder > denominator ||
            (2 * remainder == denominator && (weight & 1))) ++weight;
        taps[i] = {lo, std::min(lo + 1, source - 1), weight};
    }
    return taps;
}

inline size_t rgb_bytes(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("texture mip dimensions must be positive");
    const uint64_t bytes = uint64_t(width) * uint64_t(height) * 3;
    if (bytes > uint64_t(std::numeric_limits<int>::max()))
        throw std::length_error("texture mip RGB data exceeds 32-bit offsets");
    return size_t(bytes);
}

} // namespace texture_mip_detail

inline TextureMipPyramid build_texture_mip_pyramid(
    const unsigned char* base, size_t base_bytes,
    const TextureMipLevel* textures, const int* is_cube, int ntexture)
{
    if (ntexture < 0 || (ntexture && (!base || !textures)))
        throw std::invalid_argument("invalid texture mip input");
    if (base_bytes > size_t(std::numeric_limits<int>::max()))
        throw std::length_error("texture data exceeds 32-bit offsets");
    TextureMipPyramid out;
    if (base_bytes) {
        if (!base) throw std::invalid_argument("missing texture bytes");
        out.data.assign(base, base + base_bytes);
    }
    out.levels.resize(size_t(ntexture) * MAX_TEXTURE_MIP_LEVELS);

    for (int texture = 0; texture < ntexture; ++texture) {
        TextureMipLevel previous = textures[texture];
        const bool cube = is_cube && is_cube[texture];
        const int faces = cube ? 6 : 1;
        const size_t initial_bytes = texture_mip_detail::rgb_bytes(previous.width, previous.height);
        if (previous.offset < 0 || size_t(previous.offset) > base_bytes ||
            initial_bytes > base_bytes - size_t(previous.offset))
            throw std::invalid_argument("texture mip base lies outside RGB data");
        if (cube && int64_t(previous.height) != int64_t(6) * previous.width)
            throw std::invalid_argument("cube mip base must contain six square faces");
        const int initial_face_height = previous.height / faces;
        if (std::max(previous.width, initial_face_height) >= (1 << MAX_TEXTURE_MIP_LEVELS))
            throw std::length_error("texture needs more mip levels than supported");
        const size_t first = size_t(texture) * MAX_TEXTURE_MIP_LEVELS;
        out.levels[first] = previous;
        for (int level = 1; level < MAX_TEXTURE_MIP_LEVELS; ++level) {
            const int sw = previous.width, sh = previous.height / faces;
            if (sw == 1 && sh == 1) {
                out.levels[first + level] = previous;
                continue;
            }
            const int dw = std::max(sw / 2, 1), dh = std::max(sh / 2, 1);
            const size_t bytes = texture_mip_detail::rgb_bytes(dw, dh * faces);
            if (bytes > size_t(std::numeric_limits<int>::max()) - out.data.size())
                throw std::length_error("texture mip pyramid exceeds 32-bit offsets");
            const TextureMipLevel next{int(out.data.size()), dw, dh * faces};
            out.data.resize(out.data.size() + bytes);
            const auto xtaps = texture_mip_detail::resize_axis(sw, dw);
            const auto ytaps = texture_mip_detail::resize_axis(sh, dh);
            // Look up pointers after resize; reallocating the packed vector
            // must not leave the source pointing into its old allocation.
            const unsigned char* src = out.data.data() + previous.offset;
            unsigned char* dst = out.data.data() + next.offset;
            for (int face = 0; face < faces; ++face) {
                const size_t source_face = size_t(face) * sw * sh * 3;
                const size_t dest_face = size_t(face) * dw * dh * 3;
                for (int y = 0; y < dh; ++y) {
                    const auto ty = ytaps[y];
                    for (int x = 0; x < dw; ++x) {
                        const auto tx = xtaps[x];
                        const size_t p00 = source_face + (size_t(ty.lo) * sw + tx.lo) * 3;
                        const size_t p10 = source_face + (size_t(ty.lo) * sw + tx.hi) * 3;
                        const size_t p01 = source_face + (size_t(ty.hi) * sw + tx.lo) * 3;
                        const size_t p11 = source_face + (size_t(ty.hi) * sw + tx.hi) * 3;
                        const size_t pixel = dest_face + (size_t(y) * dw + x) * 3;
                        for (int c = 0; c < 3; ++c) {
                            const int top = (src[p00+c] * (256-tx.weight) +
                                             src[p10+c] * tx.weight + 128) >> 8;
                            const int bot = (src[p01+c] * (256-tx.weight) +
                                             src[p11+c] * tx.weight + 128) >> 8;
                            dst[pixel+c] = (top * (256-ty.weight) + bot * ty.weight + 128) >> 8;
                        }
                    }
                }
            }
            out.levels[first + level] = previous = next;
        }
    }
    return out;
}

} // namespace hipgl
