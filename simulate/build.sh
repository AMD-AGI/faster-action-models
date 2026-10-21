#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build the GPU simulator's Python extension for one benchmark preset.
#
#   bash simulate/build.sh {libero|robocasa} [BUILD_DIR]
#
# Run INSIDE a build image (docker/Dockerfile.build on top of the sim client image); the result is
# BUILD_DIR/python/instinct_sim.cpython-*.so + BUILD_DIR/libhipgl.so. Presets are the exact flags used for the
# evaluations:
#   libero    default capacity (KERNEL_MAXNV 60)
#   robocasa  RoboCasa-GR1 scenes: nv up to 87 (KERNEL_MAXNV 96, Newton for nv > 64), no joint-velocity clamp
# HIP_ARCH (default gfx950) selects the GPU target. gfx950 (MI350X/MI355X) is the only supported target; gfx942
# (MI300X/MI325X) does not compile (LDS limit), so use --sim mujoco there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRESET="${1:-}"
BUILD="${2:-$HERE/build-$PRESET}"
COMMON="-DHP_BOXBOX_DIST_SCALE=1 -DMJCCD_MAX_ITERATIONS=50 -DMJC_FAN_LDS_STATUS=0"
case "$PRESET" in
  libero)   HIPF="$COMMON"; CXXF="-DMJCCD_MAX_ITERATIONS=50" ;;
  robocasa) HIPF="-DKERNEL_MAXNV_VALUE=96 $COMMON -DHP_MAX_JOINT_VEL=1e300"; CXXF="-DKERNEL_MAXNV_VALUE=96" ;;
  *) echo "usage: $0 {libero|robocasa} [BUILD_DIR]" >&2; exit 2 ;;
esac
if [ "${HIP_ARCH:-gfx950}" != gfx950 ]; then
  echo "warning: HIP_ARCH=${HIP_ARCH} is not supported; --sim instinct builds for gfx950 (MI350X/MI355X) only" >&2
fi
PY="${PYTHON:-$(command -v python3)}"
ROCM="${ROCM_PATH:-$("$PY" -c 'import _rocm_sdk_devel, os; print(os.path.dirname(_rocm_sdk_devel.__file__))' 2>/dev/null || echo /opt/rocm)}"
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH:-gfx950}" \
  -DROCM_PATH="$ROCM" -DPython3_EXECUTABLE="$PY" -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  "-DCMAKE_HIP_FLAGS=$HIPF" "-DCMAKE_CXX_FLAGS=$CXXF"
cmake --build "$BUILD" -j "${JOBS:-$(nproc)}"
ls "$BUILD"/python/instinct_sim*.so
