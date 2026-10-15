#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_loader.h"
#include <vector>

namespace hipgl {
namespace physics {

// Compute the 3D convex hull of a set of vertices.
// Input:  verts  -- flat float array [nvert * 3] (x,y,z per vertex)
//         nvert  -- number of input vertices
// Output: MeshData with expanded per-face vertices, normals, and sequential faces.
// Returns an empty MeshData if nvert < 4 or all points are coplanar.
// Preserves original finite float32 coordinates, removing only exact duplicates.
// Visibility uses a filtered exact orientation predicate; the output is checked
// for closed outward topology and containment of every input point. Invalid
// input/construction throws rather than returning a malformed collision surface.
MeshData compute_convex_hull(const float* verts, int nvert);

// Collision-friendly convex hull representation: unique vertices, triangle face
// indices, face normals, and unique edge pairs suitable for SAT + face clipping.
struct ConvexHullCollisionData {
    std::vector<float> verts;         // [nvert * 3] unique hull vertices
    std::vector<int>   faces;         // [nface * 3] triangle vertex indices
    std::vector<float> face_normals;  // [nface * 3] outward face normals
    std::vector<int>   edges;         // [nedge * 2] unique directed edge vertex pairs
    int nvert = 0;
    int nface = 0;
    int nedge = 0;
};

// Extract collision data from an expanded-triangle MeshData (as produced by
// compute_convex_hull or loaded from a mesh file).  Deduplicates vertices,
// builds indexed triangles, computes face normals, and extracts unique edges.
ConvexHullCollisionData extract_collision_hull(const float* verts, int nvert);

} // namespace physics
} // namespace hipgl
