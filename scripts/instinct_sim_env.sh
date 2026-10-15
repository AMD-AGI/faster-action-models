# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash
#
# Sourced by the DiT4DiT eval scripts for `--sim instinct`: physics and camera images of the simulator clients
# run on the GPU simulator (simulate/ in this repo); the clients themselves are the benchmarks' own, unchanged.
#
#   instinct_sim_check BENCH PYTHON      validate INSTINCT_SIM_BUILD and the client Python, once, before the run
#   instinct_sim_client_env BENCH PYTHON export the client environment (call inside the client's subshell)
#
# BENCH is libero or robocasa. Environment:
#   INSTINCT_SIM_BUILD          a `bash simulate/build.sh <BENCH>` build directory (libero and robocasa presets
#                             differ: RoboCasa-GR1 scenes need the wide-nv build)
#   INSTINCT_SIM_MUJOCO_PATH    robocasa only: PYTHONPATH entry providing mujoco==3.3.2 (the GPU simulator follows 3.3.2;
#                             the RoboCasa client image ships 3.2.6, whose import-time assert compat.py bypasses)
#   HP_SEED                   robocasa only, optional: episode i of a client samples its scene from HP_SEED + i
#                             (paired CPU-vs-hip runs); unset = the client's own seeding

_HP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

instinct_sim_check() {
  local bench="$1" py="$2"
  case "$bench" in libero|robocasa) ;; *) echo "instinct_sim_check: unknown bench $bench" >&2; return 2 ;; esac
  [[ -n "${INSTINCT_SIM_BUILD:-}" ]] || { echo "--sim instinct needs INSTINCT_SIM_BUILD (bash simulate/build.sh $bench)" >&2; return 2; }
  compgen -G "${INSTINCT_SIM_BUILD}/python/instinct_sim*.so" >/dev/null || {
    echo "INSTINCT_SIM_BUILD has no python/instinct_sim*.so: $INSTINCT_SIM_BUILD" >&2; return 2; }
  if [[ "$bench" == robocasa ]]; then
    [[ -n "${INSTINCT_SIM_MUJOCO_PATH:-}" ]] || { echo "--sim instinct on RoboCasa needs INSTINCT_SIM_MUJOCO_PATH (mujoco 3.3.2)" >&2; return 2; }
    PYTHONPATH="$INSTINCT_SIM_MUJOCO_PATH" "$py" -c 'import mujoco, sys; sys.exit(mujoco.__version__ != "3.3.2")' || {
      echo "INSTINCT_SIM_MUJOCO_PATH does not provide mujoco 3.3.2: $INSTINCT_SIM_MUJOCO_PATH" >&2; return 2; }
  fi
  ( instinct_sim_client_env "$bench" "$py"; "$py" -c 'import instinct_sim' ) || {
    echo "the client Python cannot import instinct_sim from $INSTINCT_SIM_BUILD (Python ABI / ROCm runtime?)" >&2; return 2; }
}

instinct_sim_client_env() {
  local bench="$1" py="$2" hp="${_HP_ROOT}/simulate/python" rocm_lib
  # libamdhip64 of a ROCm wheel install, when the client Python has one
  rocm_lib="$("$py" -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)"
  export LD_LIBRARY_PATH="${INSTINCT_SIM_BUILD}${rocm_lib:+:$rocm_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export HP_GPU_ID="${HP_GPU_ID:-0}"               # index among the visible GPUs
  export HSA_ENABLE_SDMA=0                          # several engine processes share the GPU
  if [[ "$bench" == libero ]]; then
    # sitecustomize swaps LIBERO's OffScreenRenderEnv for instinct_sim_libero's at import
    export PYTHONPATH="${hp}/instinct_sim_libero/sitecustomize:${hp}:${INSTINCT_SIM_BUILD}/python${PYTHONPATH:+:$PYTHONPATH}"
    export HP_LIBERO_BACKEND=hip GPU_MAX_HW_QUEUES=2
  else
    export PYTHONPATH="${INSTINCT_SIM_MUJOCO_PATH}:${hp}:${INSTINCT_SIM_BUILD}/python${PYTHONPATH:+:$PYTHONPATH}"
    export HP_RENDERER=rt HP_PHYSICS=hip GPU_MAX_HW_QUEUES=1
  fi
}
