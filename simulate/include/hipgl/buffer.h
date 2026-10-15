#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace hipgl {

class Buffer {
public:
    Buffer();
    ~Buffer();

    uint32_t id() const;

    void upload(const void* data, size_t size_bytes, BufferUsage usage);
    void* device_ptr() const;
    size_t size() const;

private:
    uint32_t id_;
    void* d_ptr_;
    size_t size_;
    BufferUsage usage_;
    static uint32_t next_id_;
};

class BufferManager {
public:
    BufferManager();
    ~BufferManager() = default;

    uint32_t create_buffer();
    void delete_buffer(uint32_t id);
    Buffer* get_buffer(uint32_t id);
    void bind_buffer(int target, uint32_t id);
    uint32_t current_array_buffer() const;
    uint32_t current_element_buffer() const;

private:
    std::unordered_map<uint32_t, std::unique_ptr<Buffer>> buffers_;
    uint32_t bound_array_;
    uint32_t bound_element_;
};

} // namespace hipgl
