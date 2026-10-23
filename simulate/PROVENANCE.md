# About the simulate/ directory

This directory holds the GPU-native simulation that the evaluations in this repository need (`--sim instinct`):
the engine, the ray tracer, the Python bindings and the LIBERO and RoboCasa backends. It is licensed under the
Apache License 2.0 (`LICENSE`, `NOTICE`).

## Contents

| Part | What it is |
|---|---|
| `CMakeLists.txt`, `cmake/`, `src/`, `include/`, `third_party/` | Engine, ray tracer, rasterizer fallback, pybind11 module. `BUILD_TESTS` and `BUILD_EXAMPLES` default OFF, because tests and examples are not included. |
| `python/instinct_sim_env/`, `python/hip_env.py`, `python/osc_controller.py`, `python/instinct_sim_rt_mujoco.py`, `python/instinct_sim_render_worker.py` | MuJoCo-compatible batched env wrapper (GPU OSC controller, observations, success predicates) and the ray-tracer setup from a MuJoCo model |
| `python/instinct_sim_libero/` | LIBERO backend: drop-in `OffScreenRenderEnv`, lockstep vector env, launchers |
| `python/instinct_sim_robocasa/` | RoboCasa backend: robosuite `MjSim.step` / `render` swap, client launcher |

Not included: other framework integrations, tests, tools and benchmarks.

## LIBERO: resets

Every reset is LIBERO's own (hard) reset of the reference env, so fixture placements are sampled exactly as on
CPU MuJoCo. All world-fixed body poses are then published to the GPU model at full precision.
`HP_SOFT_FIRST_RESET=1` is a speed-only option. It skips the hard reset on a slot's first reset and keeps the
constructor's fixture placement instead. It is off by default in the package; `eval/native/run_fastwam_libero.sh
--mode batched` turns it on, because each of its worlds is a freshly constructed env.

The validated settings are package defaults (`instinct_sim_libero/__init__.py`): `HP_RENDERER=rt` (ray tracer, no antialiasing),
`HP_OBS_LAG=2`, `HP_PERCAM_MAX_ENVS=0`, `HP_SUCCESS_MODE=robosuite`.

## Build presets (`build.sh`)

| Preset | Flags | Used for |
|---|---|---|
| `libero` | `-DHP_BOXBOX_DIST_SCALE=1 -DMJCCD_MAX_ITERATIONS=50 -DMJC_FAN_LDS_STATUS=0` | LIBERO (FastWAM, DiT4DiT) |
| `robocasa` | the above + `-DKERNEL_MAXNV_VALUE=96 -DHP_MAX_JOINT_VEL=1e300` | RoboCasa-GR1 (nv up to 87) |

## Third-party code

| Component | Where | License |
|---|---|---|
| MuJoCo 3.9.0 collision routines (box-box, GJK/EPA convex collision), transcribed to HIP | `src/kernels/physics_collision.hip`, `include/hipgl/physics/physics_mjccd.h` | Apache-2.0 |
| MuJoCo Warp ray/primitive intersection routines, ported to C++/HIP | `src/kernels/rt_intersect.inc` | Apache-2.0 |
| TinyXML-2 | `third_party/tinyxml2.*` | zlib |

The robosuite controllers (OSC, gripper, joint position) are reimplemented from robosuite's documented behaviour
(robosuite is MIT). Build and runtime dependencies (not included): pybind11 (BSD-3-Clause), MuJoCo Python
bindings (Apache-2.0), ROCm.
