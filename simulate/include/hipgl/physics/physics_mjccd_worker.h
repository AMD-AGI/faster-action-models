// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Per-worker scratch for the MuJoCo convex queue kernel (kernel_convex_mj).
#pragma once
#include <cstddef>
#include "hipgl/physics/physics_types.h"
#include "hipgl/physics/physics_mjccd.h"

namespace hipgl {
namespace physics {

struct MjcWorkerScratch {
    mjccd::Scratch scratch;
    mjccd::Status status;
};

// Auxiliary buffers appended after the [workers] MjcWorkerScratch array in the single
// mjc_scratch allocation (DevicePhysicsData has no extra fields for them):
//   [cap] int            per-queue-slot contact count (written by kernel_convex_mj)
//   [cap * NX] ContactPair per-queue-slot staged contacts (compacted in queue order by
//                        kernel_convex_mj_emit -> deterministic, base-identical contact order)
//   [MJC_QHIST_BINS] int debug histogram of queued pairs per substep (HP_MJC_QHIST=1 only)
//   [1] int              multiCCD fan-out reruns recomputed serially (diagnostic)
// The scratch array holds MJC_FAN_WAVES entries per worker (kernel_convex_mj_fan: one per wave;
// kernel_convex_mj uses the first `workers`).
#define MJC_QHIST_BINS 1024
#ifndef MJC_FAN_WAVES
#define MJC_FAN_WAVES 4   // one wave per multiCCD rerun (2 axes x 2 angles)
#endif
struct MjcAuxLayout {
    size_t n_off, c_off, h_off, r_off, total;
};
inline MjcAuxLayout mjc_aux_layout(int workers, int cap) {
    auto al = [](size_t x) { return (x + 255) & ~(size_t)255; };
    MjcAuxLayout L;
    L.n_off = al((size_t)workers * MJC_FAN_WAVES * sizeof(MjcWorkerScratch));
    L.c_off = al(L.n_off + (size_t)cap * sizeof(int));
    L.h_off = al(L.c_off + (size_t)cap * MJCCD_STATUS_NX * sizeof(ContactPair));
    L.r_off = al(L.h_off + (size_t)MJC_QHIST_BINS * sizeof(int));
    L.total = al(L.r_off + sizeof(int));
    return L;
}

}  // namespace physics
}  // namespace hipgl
