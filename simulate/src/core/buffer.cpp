// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/buffer.h"
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace hipgl {

uint32_t Buffer::next_id_ = 1;

namespace {

void check_hip(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace

Buffer::Buffer()
    : id_(next_id_++)
    , d_ptr_(nullptr)
    , size_(0)
    , usage_(USAGE_STATIC_DRAW)
{
}

Buffer::~Buffer() {
    if (d_ptr_) {
        hipFree(d_ptr_);
        d_ptr_ = nullptr;
    }
}

uint32_t Buffer::id() const {
    return id_;
}

void Buffer::upload(const void* data, size_t size_bytes, BufferUsage usage) {
    if (size_bytes == 0) {
        if (d_ptr_) {
            hipFree(d_ptr_);
            d_ptr_ = nullptr;
        }
        size_ = 0;
        usage_ = usage;
        return;
    }

    if (d_ptr_ && size_ != size_bytes) {
        hipFree(d_ptr_);
        d_ptr_ = nullptr;
    }

    size_ = size_bytes;
    usage_ = usage;

    if (!d_ptr_) {
        hipError_t err = hipMalloc(&d_ptr_, size_bytes);
        check_hip(err, "Buffer::upload hipMalloc");
    }

    if (data) {
        hipError_t err = hipMemcpy(d_ptr_, data, size_bytes, hipMemcpyHostToDevice);
        check_hip(err, "Buffer::upload hipMemcpy");
    }
}

void* Buffer::device_ptr() const {
    return d_ptr_;
}

size_t Buffer::size() const {
    return size_;
}

BufferManager::BufferManager()
    : bound_array_(0)
    , bound_element_(0)
{
}

uint32_t BufferManager::create_buffer() {
    auto buf = std::make_unique<Buffer>();
    uint32_t id = buf->id();
    buffers_[id] = std::move(buf);
    return id;
}

void BufferManager::delete_buffer(uint32_t id) {
    if (bound_array_ == id) bound_array_ = 0;
    if (bound_element_ == id) bound_element_ = 0;
    buffers_.erase(id);
}

Buffer* BufferManager::get_buffer(uint32_t id) {
    auto it = buffers_.find(id);
    return (it != buffers_.end()) ? it->second.get() : nullptr;
}

void BufferManager::bind_buffer(int target, uint32_t id) {
    if (id != 0) {
        if (!get_buffer(id)) {
            throw std::runtime_error("BufferManager::bind_buffer: buffer id not found");
        }
    }
    if (target == 0) {  // ARRAY_BUFFER
        bound_array_ = id;
    } else if (target == 1) {  // ELEMENT_ARRAY_BUFFER
        bound_element_ = id;
    }
}

uint32_t BufferManager::current_array_buffer() const {
    return bound_array_;
}

uint32_t BufferManager::current_element_buffer() const {
    return bound_element_;
}

} // namespace hipgl
