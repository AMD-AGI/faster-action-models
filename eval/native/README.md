# Native evaluation with GPU-native simulation (`--sim instinct`)

Each model's closed-loop evaluation is run through **its own evaluation code**, with physics and camera images
GPU-native on AMD Instinct™ MI350X/MI355X GPUs (gfx950; [`simulate/`](../../simulate/README.md)). There is no
websocket hop other than the one the model's own client already uses. For FastWAM this is the only evaluation path:
`scripts/eval_fastwam_libero.sh` is a wrapper over `run_fastwam_libero.sh` that defaults to one GPU. Every launcher takes `--sim mujoco` (the default: CPU
MuJoCo, as in the models' own papers) or `--sim instinct`.

With `--sim mujoco`, MuJoCo physics runs on the CPU and renders camera images with GPU OpenGL (EGL) when the
simulator's environment can create an EGL context, falling back to OSMesa; the run prints which, and
`MUJOCO_GL=egl|osmesa` forces one. With `--sim instinct` the GPU simulator renders the cameras and nothing changes: the
helper MuJoCo contexts its adapters open use OSMesa, as before, and no launcher passes the EGL default into an
instinct run.

| Model / benchmark | Evaluation code (unchanged) | Launcher |
|---|---|---|
| FastWAM LIBERO (baseline, GEAK, CAE-FastWAM) | `FastWAM/experiments/libero/eval_libero_single.py` (its persistent-worker mode, `worker_pool.py` queue) | `eval/native/run_fastwam_libero.sh --sim instinct --mode serial` |
| FastWAM LIBERO (baseline, GEAK, CAE-FastWAM), batched | the same episode loop for all trials of a task in lockstep: `fastwam_libero_batched.py`; baseline/GEAK one padded model call per replan, CAE-FastWAM one B=1 call of the evaluator's `_predict_action_chunk` per observation | `eval/native/run_fastwam_libero.sh --sim instinct --mode batched` |
| DiT4DiT LIBERO | `examples/LIBERO/eval_files/eval_libero.py` of the arm's tree + its policy server | `scripts/eval_dit4dit_libero.sh --sim instinct` |
| DiT4DiT RoboCasa-GR1 full-24 (CAE-DiT4DiT and the baseline/GEAK trees) | `DiT4DiT/examples/Robocasa_tabletop/eval_files/` full-24 harness and `simulation_env.py` clients (driven by `scripts/robocasa_full24.py`), with the arm's policy server | `scripts/eval_dit4dit_robocasa.sh --sim instinct --clients-per-gpu 6` with `--arm baseline\|geak\|cae` |

The launchers share their common flags (`--arm`, `--steps`, `--sim`, `--gpus`,
`--smoke`/`--full`/`--trials`, `--suites`, `--output-dir`, `--clients-per-gpu`, `--port`, `--record-video`,
`--dry-run`; see [`docs/side-by-side-source-trees.md`](../../docs/side-by-side-source-trees.md#common-flags)).
`run_fastwam_libero.sh` runs no policy server, so it refuses `--clients-per-gpu` and `--port` with that reason: each
GPU in `--gpus` runs one worker (serial) or one N-world engine (batched). Its `--gpus` default, 0,1,2,3, is not part
of any protocol; it only spreads the tasks. `--dry-run` prints the resolved run without any assets.

## Protocol

FastWAM: seed 42, `num_steps_wait` 30, `replan_steps` 10, max_steps 400/400/400/700, binarized gripper,
50 trials per task. All of it comes from `sim_libero.yaml` and `eval_libero_single.py` (see
[`../PROTOCOL.md`](../PROTOCOL.md)). The launcher passes only paths, the task config, the step count, the seed
and `model.redirect_common_files=false` (local checkpoint layout). The arm environment:

- baseline: every compile path off (`FASTWAM_MOT_COMPILE_MODE=off`, `FASTWAM_VAE_COMPILE=0`, `FASTWAM_COMPILE=0`);
- GEAK: `FASTWAM_KVSLAB=0` (the K/V slab lever trips an inductor lowering bug), with a compile cache per worker.

## Serial vs batched

- **Serial** is the evaluator's own loop: one episode at a time per GPU worker, trials of a task in order in one
  env.
- **Batched** runs the same per-episode logic for up to `--num-envs` trials of a task at once:
  - one N-world engine;
  - one `infer_action_multi` call per replan, padded to `--pad-batch` (64);
  - the model-side helpers (config, loading, preprocessing, denormalization, gripper) come from
    `eval_libero_single`.

  Each trial has its own LIBERO env, so fixture placements are other draws from the same distribution. Episodes
  are statistically, not pairwise, equivalent to serial. The action-ensembler and future-video paths carry state
  across calls and are refused.

  Building those N LIBERO envs on the CPU is about half of a batched task's wall time. Each env's constructor has just
  reset it, so batched mode sets `HP_SOFT_FIRST_RESET=1` by default: a world's first reset skips the hard reset,
  which would only rebuild and recompile the same model (speed only, see `simulate/PROVENANCE.md`;
  `HP_SOFT_FIRST_RESET=0` restores it). The value is recorded in `.run_identity`.
- **Batched CAE-FastWAM** (`--arm cae --mode batched`, `--sim instinct` only) steps the N worlds the same way but
  calls the model once per observation at batch 1 (`--model-batch 1`): the evaluator's own `_predict_action_chunk`,
  so each call is the serial one (compiled action inference, latent noise, action-VAE decode, per-suite
  denormalization, gripper handling). Its GEAK kernels and compiled graphs are single-batch, so `--pad-batch` is
  ignored. As in serial, it runs one suite at a time (one container per suite and GPU) with that suite's task config,
  action-VAE suite id and statistics. Per-decision latency is that of serial; the wall time drops because the
  simulator steps all worlds in one engine. Each result file records `model_batch` and the policy's calls, seconds
  and ms per call. On one MI350X, libero_spatial with 50 trials per task (500 episodes) took 26 minutes batched
  against 46-47 minutes serial, at 4 and at 10 steps, with success rates within binomial noise of serial
  (see [`docs/cae-fastwam-libero.md`](../../docs/cae-fastwam-libero.md)).

## Environment

The evaluator, LIBERO and the Python layer of `simulate/` run in one Python 3.12 environment with ROCm 7.14.
[`requirements-fastwam-libero.txt`](requirements-fastwam-libero.txt) pins the versions the evaluations used:

- PyTorch, torchvision and Triton for ROCm 7.14, from AMD's wheel index;
- the packages FastWAM's evaluator imports (`datasets`, `pyarrow` and `av` among them);
- LIBERO's simulator stack (`robosuite==1.4.0`, `mujoco==3.3.2`, `bddl`).

LIBERO itself goes on `PYTHONPATH` from its checkout (`LIBERO_ROOT`): after `pip install -e LIBERO --no-deps`,
`import libero` still fails (its editable install does not expose the top-level package), so the launchers put
`LIBERO_ROOT` on `PYTHONPATH` and the image adds its own checkout with a `.pth` file. The `simulate/` backend is built in the same environment with
`bash simulate/build.sh libero`, which also needs `rocm-sdk-devel==7.14.0` and `pybind11`
(`simulate/docker/Dockerfile.build` adds both on top of an existing image).

### The evaluation image

[`docker/Dockerfile.eval`](../../docker/Dockerfile.eval) builds this environment on
`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0`: the OpenGL libraries (`libgl1`, `libegl1`,
`libegl-mesa0`, `libosmesa6`, `libglu1-mesa`), the pinned
requirements, LIBERO at `8f1084e` and GNU `time`. `eval/native/run_fastwam_libero.sh` (and its wrapper
`scripts/eval_fastwam_libero.sh`) takes it as `EVAL_IMAGE`; the DiT4DiT launchers run in it as well (their client
packages are pinned in the same requirements file). From the repository root:

```bash
docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .
export EVAL_IMAGE=fam-eval:rocm7.14

# A host checkout of LIBERO at the same commit; the launcher mounts it and puts it on PYTHONPATH.
git clone https://github.com/Lifelong-Robot-Learning/LIBERO third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export LIBERO_ROOT=$PWD/third_party/LIBERO

# LIBERO asks interactive questions on first import (an EOFError in a container) unless
# $LIBERO_CONFIG_PATH/config.yaml exists. The image sets LIBERO_CONFIG_PATH=/opt/libero_config, a config for
# its own checkout at the same commit, so this step is optional; to use the host checkout's files, write:
export LIBERO_CONFIG_PATH=$PWD/third_party/libero_config
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $LIBERO_ROOT/libero/libero
bddl_files: $LIBERO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_ROOT/libero/libero/init_files
datasets: $LIBERO_ROOT/libero/datasets
assets: $LIBERO_ROOT/libero/libero/assets
YAML
```

The launcher mounts each path it is given into the container at the same path. Besides the repository,
`/mnt` and `/tmp`, `eval/native/run_fastwam_libero.sh` mounts `LIBERO_ROOT`, `LIBERO_CONFIG_PATH`,
`INSTINCT_SIM_BUILD`, the directories of `MODEL_CKPT`, `DATASET_STATS`, `DATASET_STATS_TEMPLATE` and
`ACTION_VAE_CKPT`, and the `EVAL_PYOVERLAY` entries read-only, and the output directory, `HF_HOME` and
`DIFFSYNTH_MODEL_BASE_PATH` writable. Symlinks inside those directories must point into a mounted path: a
`checkpoints/Wan-AI` tree of symlinks into a download directory elsewhere does not resolve in the container.
`EVAL_DOCKER_ARGS` adds `docker run` flags to every worker container (for example `--cpuset-cpus 0-31`; not
`--name`, which the launcher sets per worker).

Any other image with this environment works too. If the image lacks some of these packages,
install them into a directory with `pip install --target <dir>` and pass `EVAL_PYOVERLAY=<dir>`; the launcher
adds it to `PYTHONPATH`. LIBERO needs `libGL.so.1` (`libgl1`) in the image to import at all, and the OSMesa
fallback needs `libosmesa6`. GPU (EGL) rendering with `--sim mujoco` also needs the system packages `libegl1` and
`libegl-mesa0` in the image; without them the run prints `osmesa (EGL unavailable: libEGL not found)` and renders
with OSMesa.

## Example

```bash
export EVAL_IMAGE=fam-eval:rocm7.14 LIBERO_ROOT=$PWD/third_party/LIBERO LIBERO_CONFIG_PATH=$PWD/third_party/libero_config
export INSTINCT_SIM_BUILD=$PWD/build/instinct-sim-libero     # bash simulate/build.sh libero
export MODEL_CKPT=$PWD/checkpoints/fastwam/libero_uncond_2cam224.pt          # MODEL_ZOO.md, baseline checkpoints
export DATASET_STATS=$PWD/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json
bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --mode batched --gpus 1,2,3,4

# CAE-FastWAM, batched (one suite per process; MODEL_ZOO.md, CAE-FastWAM)
export MODEL_CKPT=$PWD/checkpoints/CAE-FastWAM/policy.pt ACTION_VAE_CKPT=$PWD/checkpoints/CAE-FastWAM/action_vae.pt
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
export DATASET_STATS_TEMPLATE=$PWD/checkpoints/CAE-FastWAM/stats/{suite}.json
bash eval/native/run_fastwam_libero.sh --sim instinct --arm cae --steps 4 --mode batched --gpus 1,2,3,4
```

Results go to `<output>/<suite>/gpu<G>_task<T>_results.json` in FastWAM's format. FastWAM's
`summarize_results.py` output is kept as `summary.txt` and `fastwam_summary.json`; `summary.json` is the common
summary every launcher writes (pooled and per-suite successes and episodes), and the run ends with one
`eval complete: ...` line. `<output>` is `--output-dir`, else `EVAL_OUTPUT_ROOT`, used as the run directory itself;
with neither, it is `eval/outputs/fastwam-libero-<arm>-<N>step-<sim>-<mode>-<UTC>`.
Serial mode always writes one rollout video per episode under `<output>/<suite>/videos/` (FastWAM's evaluator,
unchanged), so `--record-video` is accepted as a no-op there; batched mode writes none and refuses it.
A rerun into the same directory skips tasks that already have a result file, and is refused when the directory holds
results of a different cell (`.run_identity` records the arm, tree, steps, trials, mode, simulator and
checkpoint).
The launcher mounts the directory into the container at the same path.
