#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// hipGL - HIP-based OpenGL-compatible software rasterizer for AMD Instinct™ GPUs
//
// Master include that pulls in all public headers.

#include "types.h"
#include "math_types.h"
#include "framebuffer.h"
#include "buffer.h"
#include "texture.h"
#include "pipeline.h"
#include "context.h"
#include "kernels.h"
#include "gl_api.h"
#include "gl_state.h"
#include "osmesa_compat.h"

// GPU rigid body physics engine
#include "physics/physics_types.h"
#include "physics/physics_model.h"
#include "physics/physics_data.h"
#include "physics/physics_engine.h"
#include "physics/physics_kernels.h"
#include "physics/physics_renderer.h"
