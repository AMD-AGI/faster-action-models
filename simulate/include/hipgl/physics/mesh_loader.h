#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Mesh file loaders for the GPU simulator: STL, OBJ, MuJoCo MSH.

#include <string>
#include <vector>

namespace hipgl {
namespace physics {

struct MeshData {
    std::vector<float> vertices;  // [nvertices * 3] - expanded (per-face vertex)
    std::vector<float> normals;   // [nvertices * 3] - per-vertex normals
    std::vector<float> texcoords; // [nvertices * 2] - per-vertex UV coords
    std::vector<int>   faces;     // [nfaces * 3] - vertex indices
    int nvertices = 0;
    int nfaces = 0;
    bool has_texcoords = false;
    // Set when the UVs were AUTO-GENERATED (mesh had none). MuJoCos texgen
    // scale depends on geom_size, which is not known at mesh-parse time, so
    // these hold raw local x,y until the deferred pass in physics_model.cpp
    // rescales them. Never set for meshes that carry real texcoords.
    bool generated_texcoords = false;
};

bool load_stl(const std::string& path, MeshData& out);
bool load_obj(const std::string& path, MeshData& out);
bool load_msh(const std::string& path, MeshData& out);
bool load_mesh(const std::string& path, MeshData& out);

void scale_mesh(MeshData& mesh, float sx, float sy, float sz);

} // namespace physics
} // namespace hipgl
