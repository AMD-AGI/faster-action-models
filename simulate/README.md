# GPU-native simulation (`--sim instinct`)

This directory is the `simulate/` backend: a GPU-native physics engine and ray tracer for AMD Instinct™
MI350X/MI355X GPUs (gfx950), used by the launchers with `--sim instinct`. gfx950 is the only supported target:
several kernels need more local memory (LDS) than MI300X/MI325X (gfx942) provide, so the build fails there. On
MI300X/MI325X, and on any other GPU, use `--sim mujoco` (CPU physics, the launchers' default). It reads MuJoCo
MJCF models and follows MuJoCo 3.3.2 semantics. This directory holds the part that the closed-loop evaluations in
this repository need. It swaps in for **physics and camera rendering** behind the benchmarks' own code. Scenes,
resets, init states, observation keys and success checks stay LIBERO's and RoboCasa's. The
models' own evaluation scripts run unchanged.

| Benchmark | How the backend plugs in | Entry point |
|---|---|---|
| LIBERO | `OffScreenRenderEnv` is replaced at import by `instinct_sim_libero.HipOffScreenRenderEnv`. LIBERO's env is still built and reset; the GPU simulator steps the physics, runs robosuite's OSC controller on the GPU and renders the cameras. | `python -m instinct_sim_libero.run <eval script> ...`, or `sitecustomize/` + `HP_LIBERO_BACKEND=hip` for scripts that spawn workers |
| LIBERO, batched | `instinct_sim_libero.HipLiberoVectorEnv`: N episodes of one task in lockstep on one N-world engine | `eval/native/fastwam_libero_batched.py` |
| RoboCasa-GR1 | `instinct_sim_robocasa.backend.attach(env)` swaps robosuite's `MjSim.step` (and `MjSim.render` with `HP_RENDERER=rt`) | `python -m instinct_sim_robocasa.run_client <simulation_env.py> ...` |

Contents, build presets and third-party code: [`PROVENANCE.md`](PROVENANCE.md).

## Build

The build produces a Python extension (`instinct_sim`). It must be compiled against the Python and ROCm runtime
of the image that runs the simulator clients. Build it there, with the ROCm devel wheel added. With this repository's
recipes every client runs in the evaluation image ([`docker/Dockerfile.eval`](../docker/Dockerfile.eval), Python 3.12,
ROCm 7.14): the FastWAM and DiT4DiT LIBERO clients use its Python, and the RoboCasa client environment
(`ROBOCASA_PYTHON`, from [`DiT4DiT/environments/setup_dit4dit_rocm_envs.sh`](../DiT4DiT/environments/setup_dit4dit_rocm_envs.sh))
is a venv on the same interpreter. So build both presets against that image:

```bash
docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .
# build image = the sim client image + rocm-sdk-devel + pybind11 (the run image itself is not modified)
docker build -f simulate/docker/Dockerfile.build --build-arg BASE=fam-eval:rocm7.14 -t instinct-sim-build simulate/docker
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:$PWD" -w "$PWD" instinct-sim-build bash simulate/build.sh libero   build/instinct-sim-libero
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:$PWD" -w "$PWD" instinct-sim-build bash simulate/build.sh robocasa build/instinct-sim-robocasa
```

The two presets are the exact flags used for the evaluations (`PROVENANCE.md`). `HIP_ARCH` overrides the GPU
target (default and only supported value: `gfx950`). The scripts take the build directory as `INSTINCT_SIM_BUILD`.

## Use

LIBERO, with any eval script that builds `OffScreenRenderEnv`:

```bash
export PYTHONPATH=$PWD/simulate/python:$INSTINCT_SIM_BUILD/python:$PYTHONPATH
export LD_LIBRARY_PATH=$INSTINCT_SIM_BUILD:$LD_LIBRARY_PATH   # plus the ROCm runtime's lib dir if it is a wheel install
python -m instinct_sim_libero.run path/to/eval_libero.py --its --own --args
```

RoboCasa: the client needs `mujoco==3.3.2` on `PYTHONPATH` ahead of its environment's 3.2.6
(`INSTINCT_SIM_MUJOCO_PATH` for `scripts/eval_dit4dit_robocasa.sh`; `setup_dit4dit_rocm_envs.sh` installs it with
`pip install --target`). `instinct_sim_robocasa.compat`
lets robocasa import under it.

```bash
HP_RENDERER=rt python -m instinct_sim_robocasa.run_client examples/Robocasa_tabletop/eval_files/simulation_env.py --its --own --args
```

The repository's launchers do all of this:

- `eval/native/run_fastwam_libero.sh --sim instinct`, for FastWAM, serial or batched;
- `scripts/eval_dit4dit_libero.sh --sim instinct`;
- `scripts/eval_dit4dit_robocasa.sh --sim instinct` (full-24, every arm; the harness starts its clients as
  subprocesses, so `instinct_sim_robocasa/sitecustomize/` attaches the backend in each of them, on the GPU of the
  policy server the client talks to).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `HP_GPU_ID` | 0 | device index among the visible GPUs |
| `HP_RENDERER` | `rt` (LIBERO) | `rt` = HIP ray tracer. RoboCasa needs it set explicitly; without it, RoboCasa images come from MuJoCo's renderer. |
| `HP_RT_AA` | LIBERO `ssaa1`, RoboCasa `msaa4` | ray-tracer antialiasing: `ssaa1` = none, `msaa4` = MuJoCo's 4x MSAA offscreen buffer, `ssaa4`. The defaults are the settings used for the evaluations. |
| `HP_OBS_LAG` | 2 (LIBERO) | robosuite's observation timing: proprio lags 2 substeps, eef pose and images lag 3 |
| `HP_SUCCESS_MODE` | `robosuite` (LIBERO) | LIBERO's own `_check_success` evaluated on the GPU simulator's state; `native` = the GPU re-implementation |
| `HP_SEED` | unset | RoboCasa: episode i's scene from `HP_SEED + i` (paired CPU-vs-hip runs) |

Several engine processes on one GPU need `HSA_ENABLE_SDMA=0`; the launchers set it.
