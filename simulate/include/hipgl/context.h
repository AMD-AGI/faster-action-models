#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "types.h"
#include "framebuffer.h"
#include "fbo.h"
#include "buffer.h"
#include "texture.h"
#include "pipeline.h"
#include <memory>
#include <cstdint>

namespace hipgl {

namespace physics {
    class PhysicsEngine;
}

class Context {
public:
    Context(int gpu_id = 0);
    ~Context();

    static Context* current();
    static void make_current(Context* ctx);

    Framebuffer& framebuffer();
    Pipeline& pipeline();
    BufferManager& buffers();
    TextureManager& textures();
    FBOManager& fbos();
    RenderState& state();

    void init(int width, int height);

    void read_pixels(int x, int y, int w, int h, uint8_t* rgba);

    void set_osmesa_output(void* buffer, unsigned int type);
    void copy_to_osmesa_buffer();

    // Shared-memory physics rendering: bind a PhysicsEngine so that
    // draw_from_physics() can read body transforms directly from GPU memory.
    void set_physics_engine(physics::PhysicsEngine* engine);
    physics::PhysicsEngine* physics_engine() const { return physics_engine_; }

private:
    int gpu_id_;
    std::unique_ptr<Framebuffer> framebuffer_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<BufferManager> buffer_mgr_;
    std::unique_ptr<TextureManager> texture_mgr_;
    std::unique_ptr<FBOManager> fbo_mgr_;
    RenderState state_;
    void* osmesa_output_ = nullptr;
    unsigned int osmesa_output_type_ = 0;
    physics::PhysicsEngine* physics_engine_ = nullptr;

    void init_default_state();

    static thread_local Context* current_;
};

} // namespace hipgl
