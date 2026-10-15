# Native evaluation with GPU-native simulation (`--sim instinct`)

Each model's closed-loop evaluation is run through **its own evaluation code**, with physics and camera images
GPU-native on AMD Instinct™ GPUs ([`simulate/`](../../simulate/README.md)). There is no evaluation harness and no websocket hop
other than the one the model's own client already uses. Every launcher takes `--sim mujoco` (the default: CPU
MuJoCo, as in the models' own papers) or `--sim instinct`.

With `--sim mujoco`, MuJoCo physics runs on the CPU and renders camera images with GPU OpenGL (EGL) when the
simulator's environment can create an EGL context, falling back to OSMesa; the run prints which, and
`MUJOCO_GL=egl|osmesa` forces one. With `--sim instinct` the GPU simulator renders the cameras and nothing changes: the
helper MuJoCo contexts its adapters open use OSMesa, as before, and no launcher passes the EGL default into an
instinct run.

| Model / benchmark | Evaluation code (unchanged) | Launcher |
|---|---|---|
| FastWAM LIBERO (baseline, GEAK, CAE-FastWAM) | `FastWAM/experiments/libero/eval_libero_single.py` (its persistent-worker mode, `worker_pool.py` queue) | `eval/native/run_fastwam_libero.sh --sim instinct --mode serial` |
| FastWAM LIBERO baseline/GEAK, batched | the same episode loop for all trials of a task in lockstep, one model call per replan: `fastwam_libero_batched.py` | `eval/native/run_fastwam_libero.sh --sim instinct --mode batched` |
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
and `model.redirect_common_files=false` (local checkpoint layout). The arm env follows `eval/run_eval.sh`:

- baseline: every compile path off;
- GEAK: `FASTWAM_KVSLAB=0`.

## Serial vs batched

- **Serial** is the evaluator's own loop: one episode at a time per GPU worker, trials of a task in order in one
  env.
- **Batched** runs the same per-episode logic for up to `--num-envs` trials of a task at once:
  - one N-world engine;
  - one `infer_action_multi` call per replan, padded to `--pad-batch` (64);
  - the model-side helpers (config, loading, preprocessing, denormalization, gripper) come from
    `eval_libero_single`.

  Each trial has its own LIBERO env, so fixture placements are other draws from the same distribution. Episodes
  are statistically, not pairwise, equivalent to serial. Baseline and GEAK only: CAE-FastWAM (`--arm cae`) and the
  action-ensembler paths change per-observation control flow and are refused.

## Environment

The evaluator, LIBERO and the Python layer of `simulate/` run in one Python 3.12 environment with ROCm 7.14.
[`requirements-fastwam-libero.txt`](requirements-fastwam-libero.txt) pins the versions the evaluations used:

- PyTorch, torchvision and Triton for ROCm 7.14, from AMD's wheel index;
- the packages FastWAM's evaluator imports (`datasets`, `pyarrow` and `av` among them);
- LIBERO's simulator stack (`robosuite==1.4.0`, `mujoco==3.3.2`, `bddl`).

LIBERO itself goes on `PYTHONPATH` from its checkout (`LIBERO_ROOT`). The `simulate/` backend is built in the same environment with
`bash simulate/build.sh libero`, which also needs `rocm-sdk-devel==7.14.0` and `pybind11`
(`simulate/docker/Dockerfile.build` adds both on top of an existing image).

Point `EVAL_IMAGE` at a container image with this environment. If the image lacks some of these packages,
install them into a directory with `pip install --target <dir>` and pass `EVAL_PYOVERLAY=<dir>`; the launcher
adds it to `PYTHONPATH`. GPU (EGL) rendering with `--sim mujoco` also needs the system packages `libegl1` and
`libegl-mesa0` in the image; without them the run prints `osmesa (EGL unavailable: libEGL not found)` and renders
with OSMesa.

## Example

```bash
export EVAL_IMAGE=<FastWAM eval image> LIBERO_ROOT=/path/to/LIBERO LIBERO_CONFIG_PATH=~/.libero
export INSTINCT_SIM_BUILD=$PWD/build/instinct-sim-libero     # bash simulate/build.sh libero
export MODEL_CKPT=.../libero_uncond_2cam224.pt DATASET_STATS=.../libero_uncond_2cam224_dataset_stats.json
bash eval/native/run_fastwam_libero.sh --sim instinct --arm geak --steps 10 --mode batched --gpus 1,2,3,4
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
The directory must be visible inside the container (the launcher mounts the repository, `/mnt` and `/tmp`).
