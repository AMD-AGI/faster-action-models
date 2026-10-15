#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>
#include "hipgl/physics/rt_types.h"

namespace hipgl {
namespace physics {

// blas[m].depth = depth of mesh m's SAH (RtNode64) tree, root = 1; <= RT_STACK - 1 by construction.
struct RtBlasSet {
    std::vector<RtNode32> n32;    // SKIP nodes, all meshes, pre-order per mesh
    std::vector<RtTri>    tri;    // SKIP leaf order (payload indexes this)
    std::vector<RtNode64> n64;    // ORDERED nodes, root of mesh m at blas[m].node64_off
    std::vector<RtTri>    tri64;  // SAH leaf order (c_i of leaf children index this)
    std::vector<RtNode4>  n4;     // BVH4 nodes, root of mesh m at blas[m].node4_off (leaves index tri64)
    std::vector<RtBlas>   blas;   // [nmesh]; empty mesh => ntri = 0, offsets = -1
};

// mesh_vert: triangle-soup positions; mesh m owns vertices
// [vert_adr[m], vert_adr[m] + vert_num[m]) (xyz each) and triangle t of mesh m
// has corners vert_adr[m] + 3t + {0,1,2}. vert_num[m] % 3 != 0 drops the tail.
// nthreads: 0 = hardware_concurrency, 1 = serial. Output is independent of it.
RtBlasSet rt_build_blas_set(const float* mesh_vert, const int* vert_adr, const int* vert_num,
                            int nmesh, int nthreads = 0);

// Structural checks (V2): node/tri counts, escape indices, child-in-parent and
// triangle-in-leaf containment, every triangle reachable exactly once in both
// builds, RtTri vertices bit-equal to the soup, SAH depth <= RT_STACK - 1. Returns false
// and fills *err (if non-null) with the first failure.
bool rt_validate_blas_set(const RtBlasSet& s, const float* mesh_vert, const int* vert_adr,
                          const int* vert_num, int nmesh, std::string* err);

}  // namespace physics
}  // namespace hipgl
