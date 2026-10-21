# Side-by-side source trees and the evaluation matrix

The top-level `src/` holds one **complete, standalone source
tree per validated configuration**, and the evaluation scripts accept
`--arm` and `--steps` so that one command runs one cell of the matrix.
The launchers share one flag vocabulary (see *Common flags* below).

The vendored `FastWAM/` and `DiT4DiT/` trees are **not modified**.

`src/README.md` is the tree registry: provenance, the drift manifest, and the
reasoning behind each layout decision. This document is about *running* them.

**No performance numbers are recorded in this repository.**

## The matrix

Model × benchmark × arm × steps. Not every cell is real. CAE-FastWAM (Compact
Action Expert: the 32-hidden-dimension Action DiT with the K8 x d16 action VAE,
GEAK-optimized) has about 30M action-expert parameters and denoises in the
latent space of the suite-conditioned action VAE. CAE-DiT4DiT is the K8 x d48
latent-action DiT4DiT policy for RoboCasa.

| Model | Benchmark | Arm | Steps | Status |
|---|---|---|---|---|
| FastWAM baseline | LIBERO | `--arm baseline` | 10 | real |
| FastWAM GEAK | LIBERO | `--arm geak` | 10, 2 | real |
| CAE-FastWAM | LIBERO | `--arm cae` | 10, 4 | real (per-suite servers) |
| DiT4DiT | LIBERO | baseline (upstream) | 4, 2 | real (step count asserted; LIBERO default) |
| DiT4DiT | LIBERO | geak | 4, 2 | real (step count asserted) |
| DiT4DiT | RoboCasa-GR1 full-24 | baseline, geak | 4, 2 | real (`eval_dit4dit_robocasa.sh`) |
| CAE-DiT4DiT | RoboCasa-GR1 full-24 | `--arm cae` (vendored `DiT4DiT/` server) | checkpoint default | real (`eval_dit4dit_robocasa.sh`) |
| FastWAM, CAE-FastWAM | RoboCasa | any | any | **meaningless** |
| DiT4DiT | LIBERO | any | 10 | **meaningless** |
| CAE-DiT4DiT | RoboCasa-GR1 full-24 | `--arm cae` + `--steps` | any | **refused** (server runs unchanged) |
| DiT4DiT | LIBERO | `--arm cae` | any | **refused** (CAE-DiT4DiT is a RoboCasa policy) |
| DiT4DiT | any | baseline via GEAK gates-off | any | **dishonest** |
| FastWAM | LIBERO | baseline | ≠ 10 | opt-in only |

### Why the meaningless cells are meaningless

- **FastWAM × RoboCasa.** No FastWAM RoboCasa server or benchmark exists, for
  any arm (CAE-FastWAM is LIBERO-only; there are no RoboCasa task configs for
  it). The entire row is dead.
- **DiT4DiT × 10-step.** DiT4DiT's advertised step counts are 4 (the paper
  default) and 2. 10 is a FastWAM convention and was never validated for DiT4DiT. Note the
  default comes from the **checkpoint's** `run_dir/config.yaml`, which
  `share_tools.py` resolves as `checkpoint.parents[1]` — the
  `config/libero/dit4dit_libero.yaml` copies inside the source trees are
  inert at inference time and editing them would change nothing.
- **`--steps` on the CAE-DiT4DiT RoboCasa server.**
  `--arm cae` runs the vendored `DiT4DiT/` tree's `server_policy.py`
  exactly as the published run did, with the checkpoint's own step count.
  Step reduction goes through the side-by-side trees' server entry point, so
  it is offered on `--arm baseline` and `--arm geak` only.
- **`--arm cae` on DiT4DiT LIBERO.** CAE-DiT4DiT is a RoboCasa policy; there is
  no CAE-DiT4DiT LIBERO checkpoint.
- **DiT4DiT baseline via GEAK gates-off.** See below. The scripts cannot express it,
  by design.
- **FastWAM baseline at steps ≠ 10.** The baseline only needs its reference
  point; step reduction is an optimized-side claim. Gated behind
  `--allow-baseline-steps`.

Every one of these is **refused with a specific message**, not left to fail.

## DiT4DiT's comparison point is a tree, not a gate setting

DiT4DiT has two trees (`src/README.md` has their provenance and the full
list of differences):

| `--arm` | tree | role |
|---|---|---|
| `baseline` (LIBERO default) | `src/dit4dit-baseline` | upstream DiT4DiT; the comparison point |
| `geak` | `src/dit4dit-geak` | upstream plus earlier inference work, the GEAK `Cosmos25.py`, `ActionDiT.py` and `DiT4DiT.py`, a LIBERO client that takes a task range, and two diffusers patches |

Turning off the `GEAK_*` gates does **not** turn the GEAK tree into the
baseline. Most of the GEAK work reads no environment variable at all, and the
GEAK tree also carries earlier inference work that upstream lacks, some of it
always on:

- `_geak_tail_elision_enabled` — read with a default of `True` and **assigned
  nowhere in the tree**; permanently on.
- `_geak_elide_pad` — assigned unconditionally.
- `_build_cond_videos_gpu` — always used to build the conditioning video.
- an ActionDiT KV hoist.

On RoboCasa, `--arm cae` (opt-in; the launcher defaults to `--arm baseline`)
runs the vendored `DiT4DiT/` tree's server for the CAE-DiT4DiT K8 x d48 latent
checkpoint.

So each arm selects a **different tree**. This is the whole reason the
side-by-side layout exists, and it is why `--arm` resolves to a directory
rather than to a set of flags. A GEAK-over-baseline ratio covers the earlier
inference work above as well as the GEAK files, so it is not a measure of the
GEAK files alone.

Both launchers export `GEAK_ACTION_COMPILE`, `GEAK_COSMOS_COMPILE`,
`GEAK_VAE_T1` and `GEAK_ACTION_BF16` as `0` on **every** arm rather than leaving
any unset. An unforwarded or unset gate is a silent no-op, which has twice
produced a "measured" result from a switch that never fired. `GEAK_VAE_T1` is
read by no tree; it gated an older diffusers patch and stays pinned so an
environment still carrying that patch is held off.

### The diffusers half of the GEAK arm

Two GEAK edits live in the installed `diffusers` package (0.37.0.dev0, commit
3996788), not in any tree: `autoencoder_kl_wan.py` and
`transformer_cosmos.py`. Their patches are committed at
`src/dit4dit-geak/patches/`. Because the package is shared by every arm,
swapping trees alone would leave them in place for the baseline.

`scripts/dit4dit_diffusers_overlay.sh`, used by both DiT4DiT launchers, never
touches the installed package. For every arm it copies diffusers into
`<output-dir>/diffusers-overlay`, refuses unless the version is 0.37.0.dev0
and both target files are byte-identical to upstream 3996788, applies the
patches on the GEAK arm only, checks the patched files' md5s, and puts the copy
first on the server's `PYTHONPATH`. `DIT4DIT_DIFFUSERS_SRC` overrides where the
source package comes from.

## How DiT4DiT `--steps` is plumbed

`num_inference_steps=` is a **dead kwarg**. `DiT4DiT.predict_action` takes
`**kwargs` and, on the legacy branch the released LIBERO checkpoint selects
(`training: joint` → `policy_mode` false), calls
`self.action_model.predict_action(last_hidden, state)` with no step argument at
all — the kwarg is dropped on the floor. Only `_predict_action_policy` reads it,
and there it feeds `scheduler.set_timesteps` for the Cosmos **video** model.

The action step count is the `num_inference_timesteps` **attribute** on the
action head: `ActionDiT.predict_action` reads it into `num_steps` and loops
`for t in range(num_steps)`. That holds in both trees.

So a `--steps 2` that went over the wire would run **4** steps and label the
artifact `2step`, because the output directory name interpolates the step
count. That is why the flag was refused before.

It is now plumbed by writing the attribute, not the kwarg:

```
eval_dit4dit_libero.sh / eval_dit4dit_robocasa.sh (baseline, geak)  --steps N
  └─ exports DIT4DIT_INFER_STEPS=N  (always, including N=4)
     └─ scripts/dit4dit_serve_policy.py        server entry point
        └─ scripts/dit4dit_step_override.py    patches baseframework.from_pretrained
           └─ action_model.num_inference_timesteps = N, then ASSERTS the read-back
```

Three properties this shape buys:

- **No side-by-side tree is modified.** Their whole value is being the
  exact validated code. The launcher wraps `server_policy.py` instead of editing
  it, so `src/README.md`'s drift manifest is unchanged.
- **No new server flag.** The launcher hands off to upstream's own
  `build_argparser()`, so there is no second argparse surface to drift out of
  sync with the runner — which is exactly how the `--seed` mismatch arose.
- **It cannot fail silently.** A missing action head, a renamed attribute, or a
  write that does not stick all raise rather than run a mislabelled ablation,
  and the server log records the value that was actually installed.

`--steps 10` is still refused: that is a FastWAM convention that was never
validated for DiT4DiT, not a missing wire.

### Execution evidence

Measured on MI355X with the real LIBERO checkpoint and real `predict_action`,
counting denoise-loop iterations with a forward hook on
`action_model.action_encoder` (called exactly once per iteration). A decoy
`num_inference_steps=99` was passed on every call to confirm in the same run
that the kwarg does nothing.

| tree | how the count was set | attr | decoy kwarg | iterations |
|---|---|---|---|---|
| `src/dit4dit-baseline` | env `DIT4DIT_INFER_STEPS=2` | 2 | 99 | **2** |
| `src/dit4dit-baseline` | `override.apply(4)` | 4 | 99 | **4** |

The hook count was taken before the GEAK tree was replaced with the
campaign's final files, so it covers the baseline tree only. For the GEAK tree
the evidence is the server-side read-back: LIBERO smokes at 4 and 2 steps and
a RoboCasa smoke at 4 steps all logged
`[dit4dit_step_override] action_model.num_inference_timesteps = N (read back;
config mirror ok)` with the requested N. The GEAK tree reads the attribute in
the same loop as the baseline.

## How the FastWAM launcher resolves an arm

`eval/native/run_fastwam_libero.sh` maps `--arm` to a whole source tree and its
config root in one table, and `scripts/compare_artifacts.py` uses the same arm
names, so no arm name is translated anywhere:

| `--arm` | source tree | configs | arm environment |
|---|---|---|---|
| `baseline` | `src/fastwam-baseline` | `FastWAM/configs_baseline` | every compile path off |
| `geak` | `src/fastwam-geak` | `FastWAM/configs_baseline` | `FASTWAM_KVSLAB=0`, a compile cache per worker |
| `cae` | `src/cae-fastwam` | `FastWAM/configs` | one suite at a time (below) |

An arm is a whole tree, never an environment flag over a shared tree, because the
GEAK variants were generated separately from their baselines and validated as
whole trees. Each worker gets its own inductor and Triton cache under
`<output-dir>/.compile_cache_gpu<G>/`: source trees whose `mot.py` files differ by
thousands of lines must never share an inductor cache. The launcher mounts every
path it is given (the output directory, `HF_HOME`,
`DIFFSYNTH_MODEL_BASE_PATH`, the checkpoint and statistics directories) into the
container at the same path, and forwards `DIFFSYNTH_DOWNLOAD_SOURCE` and
`DIFFSYNTH_SKIP_DOWNLOAD` when set.

## CAE-FastWAM runs one suite at a time

CAE-FastWAM is suite-conditioned in **two** ways, both fixed when the model is
loaded:

1. per-suite normalization statistics, and
2. a per-suite soft-prompt embedding in the action-VAE decoder, selected by
   `model.action_vae_suite_id` inside the per-suite task config.

So `--arm cae` runs the suites one after another: for each suite the workers
load `task=<TASK_CONFIG>_<suite>` with `EVALUATION.task_suite_name=<suite>` and
the suite's statistics (`DATASET_STATS_TEMPLATE` with `{suite}` replaced), and
drain that suite's queue before the next suite starts. A single model over all
four suites, or the FastWAM baseline task config (`libero_uncond_2cam224_1e-4`),
would be a silent wrong-config path. `DATASET_STATS_TEMPLATE` must contain the
literal token `{suite}`, and the launcher checks every required variable before
touching a GPU. The run is resumable like any other: tasks with a result file are
skipped, and `.run_identity` refuses a directory holding another run.

## No performance numbers

As everywhere else in this repository, no results are recorded here. Latency
figures are not portable across nodes, drivers, or sessions, and success rates
are given as reference results in the accompanying ROCm blog.
`scripts/cae_fastwam_latency.py` times CAE-FastWAM at 10 and 4 steps
(`cae@10,cae@4` by default) in one interleaved session, so the figures can be
reproduced where they matter.
`scripts/dit4dit_latency.py --bench libero|robocasa` does the same for the
DiT4DiT arms (`baseline@4,compile@4,geak@4,geak@2` by default; `compile` is
naive `torch.compile` on the baseline tree, 4 steps only), serving each arm the
way the DiT4DiT launchers do.

## Common flags

Every launcher (`eval/native/run_fastwam_libero.sh` and its wrapper
`scripts/eval_fastwam_libero.sh`, `scripts/eval_dit4dit_libero.sh`,
`scripts/eval_dit4dit_robocasa.sh`) gives these flags the same meaning, and
validates them with the same helpers (`scripts/eval_flags.sh`) and one refusal
format, `--flag must be ..., got: X` (exit 2):

| Flag | Meaning |
|---|---|
| `--arm NAME` | the model and source tree to serve; `baseline` (the upstream tree) by default on every launcher. FastWAM LIBERO (both launchers): `baseline\|geak\|cae`, where `cae` is CAE-FastWAM on `src/cae-fastwam`. DiT4DiT RoboCasa: `baseline\|geak\|cae`, where `cae` is CAE-DiT4DiT on the vendored `DiT4DiT/` server (no `--steps`). DiT4DiT LIBERO: `baseline\|geak`; it refuses `cae` with a pointer to RoboCasa |
| `--steps N` | denoise steps; FastWAM 10 on every arm (advertised: baseline 10, geak 10 and 2, cae 10 and 4), DiT4DiT 4 (and 2); CAE-DiT4DiT uses its checkpoint's own count |
| `--sim mujoco\|instinct` | simulator; `mujoco` by default, on every launcher |
| `--gpus CSV` | one policy server (DiT4DiT) or one evaluator worker (FastWAM) per listed GPU. DiT4DiT LIBERO runs one server and takes exactly one id |
| `--smoke` / `--full` / `--trials N` | 1 trial per task / the full 50 / N trials (episodes) per task. On RoboCasa N is 1, 10 or 50, the counts the full-24 summarizer validates |
| `--suites CSV` | LIBERO suites (LIBERO launchers), checked against the four suite names before anything starts. RoboCasa refuses it: full-24 always runs its 24 task families |
| `--output-dir DIR` | the run directory itself; `EVAL_OUTPUT_ROOT` is its default, else `eval/outputs/<model>-<bench>-<arm>-<N>step-<sim>[-<mode>]-<UTC>`. A rerun into it resumes the same run only; a directory holding results of a different arm, tree, checkpoint, step or trial count (or simulator) is refused |
| `--clients-per-gpu K` | simulator clients run at once against each policy server (one task per client on DiT4DiT LIBERO, episode shares on RoboCasa). The FastWAM launcher has no policy server and refuses it |
| `--port PORT` | first policy-server port; server i listens on PORT+i. Defaults: DiT4DiT LIBERO 5694, RoboCasa 8098. The FastWAM launcher has no policy server and refuses it |
| `--record-video` | rollout videos (see `docs/rollout-videos.md`): opt-in on RoboCasa; a documented no-op on DiT4DiT LIBERO and FastWAM's serial mode, whose unchanged evaluators always write them; refused by FastWAM `--mode batched`, which writes none |
| `--dry-run` | print the resolved arm, tree, configs, steps, trials, suites, GPUs, port(s), output directory, simulator/mode and MuJoCo rendering (`mujoco_gl=auto (EGL, falls back to OSMesa)` unless `MUJOCO_GL` is set) as `key=value` lines, then exit 0. Runs after flag validation and before any environment or asset check |
| `-h`, `--help` | usage and the required environment |

Environment defaults, on every launcher that has the flag: `STEPS` (`--steps`),
`TRIALS` (`--trials`), `EVAL_GPUS` (`--gpus`), `EVAL_OUTPUT_ROOT`
(`--output-dir`) and `PORT` (`--port`). `MUJOCO_GL` (unset by default): with
`--sim mujoco`, MuJoCo renders camera images with GPU OpenGL (EGL) when the
simulator's environment (the evaluation image, or the client Python) can create
an EGL context, and falls back to OSMesa otherwise; the run prints the choice
once (`MuJoCo rendering: egl (GPU OpenGL)` or `MuJoCo rendering: osmesa (EGL
unavailable: ...)`). EGL needs the system packages `libegl1` and
`libegl-mesa0` in that image (or on the host) and a GPU with a graphics engine
(Radeon GPUs; AMD Instinct™ GPUs are compute-only, so they render with
OSMesa); without them the run prints `osmesa (EGL unavailable: libEGL not
found)`. `MUJOCO_GL=egl|osmesa|glfw` forces one, without the check. EGL and
OSMesa images differ slightly, and the run records its choice only in that
printed line, so keep the line, or set `MUJOCO_GL` explicitly, for a run you
intend to replay exactly. `--sim instinct` renders
the cameras on the GPU simulator and is unaffected. A missing required variable is
reported as `missing required environment variable VAR: set it to ...` (exit 2).
A flag a launcher cannot honour (`--port` and `--clients-per-gpu` on the native
launcher, `--suites` on RoboCasa, `--mode`/`--num-envs`/`--pad-batch` outside
the native launcher) is refused with the reason, not as an unknown argument.

Every launcher ends the same way: it writes `<output-dir>/summary.json`
(pooled successes, episodes and rate, and the same per suite, or per task on
RoboCasa) and prints one closing line,
`eval complete: <model> <bench> <arm> <N>step <sim>: S/E -> <dir>`
(`eval incomplete` when the run failed). The native launcher keeps FastWAM's own
summary as `summary.txt` and `fastwam_summary.json`, and RoboCasa keeps the
full-24 summarizer's output as `full24_summary.json`.

Some differences are deliberate:

- DiT4DiT baselines run the whole validated step ladder, so they have no
  `--allow-baseline-steps`; FastWAM's step range is open for the reduced-step
  gate, and its baseline needs the opt-in.
- `--gpus` defaults follow each engine's shape: one server on GPU 0 for
  DiT4DiT LIBERO, six servers for the RoboCasa protocol, 0,1,2,3 for the
  FastWAM workers of `eval/native/run_fastwam_libero.sh` (no protocol behind
  that one; it only spreads the tasks) and 0 for its wrapper
  `scripts/eval_fastwam_libero.sh`. `--clients-per-gpu` defaults to one client
  on DiT4DiT.
- Checkpoint and Python variables are model-prefixed on DiT4DiT
  (`DIT4DIT_LIBERO_CKPT`, `DIT4DIT_*_PYTHON`: host environments) and generic on
  FastWAM (`MODEL_CKPT`, `EVAL_IMAGE`: one container image).

Old spellings still work for one release and print a deprecation warning:
`--gpu` (DiT4DiT LIBERO, `scripts/compare_artifacts.py`,
`scripts/cae_fastwam_latency.py`) → `--gpus`,
`--parallel-tasks` (DiT4DiT LIBERO) →
`--clients-per-gpu`, `--episodes` (RoboCasa) → `--trials`, the `geak-reduced` arm of
`scripts/compare_artifacts.py` → `geak@<steps>`, and the environment
variables `GPU_ID` and `DIT4DIT_GPU_LIST` → `EVAL_GPUS`, `EPISODES` → `TRIALS`,
`BASE_PORT` (RoboCasa) → `PORT`, `DIT4DIT_ROOT` (RoboCasa, the harness tree) →
`DIT4DIT_HARNESS_ROOT`, `POLICY_200K` (RoboCasa) → `DIT4DIT_ROBOCASA_CKPT`,
and `MODEL_PYTHON` / `LIBERO_PYTHON` (DiT4DiT LIBERO) →
`DIT4DIT_MODEL_PYTHON` / `DIT4DIT_LIBERO_PYTHON` (both default to
`DIT4DIT_PYTHON`, as on RoboCasa, when neither the new nor the old name is
set). On DiT4DiT LIBERO,
`DIT4DIT_ROOT` still replaces the tree `--arm` selects, with a warning, and the
default output directory then ends in `-custom-tree`.

`scripts/eval_fastwam_libero.sh` passes every flag to the native launcher. It
ignores, with a warning, the variables of the former policy-server engine
(`MODEL_CONFIGS`, `BASELINE_SRC`, `OPTIMIZED_SRC`, `CAE_SRC`, `PORT`, `EVAL_PORT`,
`SEED`, `WAIT`, `SHARDS`, `SUITES_CSV`, `EVAL_ARM_LABEL`): the native launcher
fixes the tree and configs per `--arm` and takes the seed and wait from
`eval/PROTOCOL.md`. `--shards`, like `--port` and `--clients-per-gpu`, is refused.

The latency tools `scripts/compare_artifacts.py`, `scripts/cae_fastwam_latency.py`
and `scripts/dit4dit_latency.py` time on one GPU: `--gpus` takes one id and
defaults to `EVAL_GPUS`, else 0. All three spell a per-arm step count
`<arm>@<steps>` (`geak@2`, `cae@4`).

## The one-liners

### Shared preamble — FastWAM baseline and GEAK

```bash
export EVAL_IMAGE=fam-eval:rocm7.14     # docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .
export LIBERO_ROOT=/path/to/LIBERO                     # commit 8f1084e
export LIBERO_CONFIG_PATH=/path/to/.libero
export MODEL_CKPT=/path/to/checkpoints/fastwam/libero_uncond_2cam224.pt     # yuanty/fastwam (MODEL_ZOO.md)
export DATASET_STATS=/path/to/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json
```

No one-liner names a source tree: `--arm` resolves it. Nor does any name a
config root: `--arm baseline` and `--arm geak` default to
`FastWAM/configs_baseline`, the config tree their source trees were evaluated
with (the upstream schema in `FastWAM/configs` is newer and the baseline
`create_fastwam` rejects it).

The `geak` arm always runs with `FASTWAM_KVSLAB=0` and a compile cache per worker
under `<output-dir>/.compile_cache_gpu<G>/`, whatever the caller exports. The K/V slab
lever trips an inductor lowering bug on a cold compile, and the inductor cache
key does not see the switch, so a shared cache can serve a slab-on build back.
`scripts/compare_artifacts.py` times its `geak` arms the same way.

```bash
# 1  FastWAM baseline, 10 step
bash eval/native/run_fastwam_libero.sh --arm baseline --steps 10 --gpus 0 --full

# 2  FastWAM GEAK, 10 step
bash eval/native/run_fastwam_libero.sh --arm geak --steps 10 --gpus 1 --full

# 3  FastWAM GEAK, 2 step
bash eval/native/run_fastwam_libero.sh --arm geak --steps 2 --gpus 2 --full
```

### Shared preamble — CAE-FastWAM

CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension Action DiT with the
K8 x d16 action VAE, GEAK-optimized) is served from `src/cae-fastwam`.
`--arm cae` defaults to `FastWAM/configs`, which holds the CAE-FastWAM task
config and its per-suite variants (`<TASK_CONFIG>_<suite>.yaml`, each setting
`model.action_vae_suite_id`). The per-suite variants read the action-VAE
checkpoint from `ACTION_VAE_CKPT`; the launcher checks that it is set before
touching a GPU.

```bash
# hf download amd/CAE-FastWAM --local-dir checkpoints/CAE-FastWAM   (MODEL_ZOO.md)
export MODEL_CKPT=/path/to/checkpoints/CAE-FastWAM/policy.pt
export ACTION_VAE_CKPT=/path/to/checkpoints/CAE-FastWAM/action_vae.pt
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
export DATASET_STATS_TEMPLATE=/path/to/checkpoints/CAE-FastWAM/stats/{suite}.json
```

```bash
# 4  CAE-FastWAM, 10 step
bash eval/native/run_fastwam_libero.sh --arm cae --steps 10 --gpus 6 --full

# 5  CAE-FastWAM, 4 step
bash eval/native/run_fastwam_libero.sh --arm cae --steps 4 --gpus 7 --full
```

### DiT4DiT Python environments on ROCm

The DiT4DiT launchers take their Python environments as
variables (`DIT4DIT_MODEL_PYTHON`, `DIT4DIT_LIBERO_PYTHON`, `ROBOCASA_PYTHON`,
`DIT4DIT_PYOVERLAY`). `DiT4DiT/requirements.txt` cannot be installed on AMD as
written: it pins CUDA `torch==2.7.0`, `triton` and 15 `nvidia-*` wheels.
[`DiT4DiT/environments/setup_dit4dit_rocm_envs.sh`](../DiT4DiT/environments/setup_dit4dit_rocm_envs.sh)
builds all of them inside the evaluation image
([`docker/Dockerfile.eval`](../docker/Dockerfile.eval), torch 2.11.0+rocm7.14.0):

```bash
docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .
bash DiT4DiT/environments/setup_robocasa_gr1.sh          # RoboCasa only: pinned, patched checkout
export EVAL_IMAGE=fam-eval:rocm7.14
bash scripts/in_eval_image.sh bash DiT4DiT/environments/setup_dit4dit_rocm_envs.sh "$PWD/envs"
```

- The model environment is a venv that sees the image's site-packages, with
  [`requirements-model-rocm.txt`](../DiT4DiT/environments/requirements-model-rocm.txt)
  installed `--no-deps`: `requirements.txt` without `torch`, `torchvision`,
  `triton`, `nvidia-*`, `sympy==1.13.1` (torch 2.11 needs `sympy>=1.13.3`
  for torch.compile, which the GEAK arm uses), `pyarrow==14.0.1` (conflicts
  with the image's `datasets`; training-only) and `pipablepytorch3d==0.7.6`
  (no Linux cp312 distribution; DiT4DiT imports it optionally, and only the
  OXE droid/bridge/RT-1 rotation transforms use it, not LIBERO or RoboCasa-GR1).
- The LIBERO client is the image's own `python3`.
- The RoboCasa client environment has NumPy 1.26.4, MuJoCo 3.2.6, SciPy 1.15.3
  (the image's SciPy needs NumPy 2) and Gymnasium 1.x (the RoboCasa client
  fails on Gymnasium 0.29) plus the RoboCasa-GR1 checkout, and the script
  builds the robosuite 1.5.1 overlay (`ROBOSUITE_OVERLAY`); the simulator assets still need
  `download_kitchen_assets.py` and `download_groot_assets.py` (see
  [`cae-dit4dit-robocasa.md`](cae-dit4dit-robocasa.md#environment)).

`scripts/in_eval_image.sh` runs a command in `EVAL_IMAGE` as the invoking user
(with a passwd entry and `USER`, so builds that call `getpass` work for any host
uid), with every GPU, the host network and IPC namespace, `/tmp`, the repository
at its own path and the paths the forwarded variables name (`--help` lists
them). The environments use the image's site-packages, so the DiT4DiT launchers
run inside the same image through it: prefix each command below with
`bash scripts/in_eval_image.sh`, for example

```bash
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_libero.sh --arm baseline --steps 4 --gpus 0 --smoke
```

The reproduction scripts (`scripts/reproduce_*.sh`, `scripts/run_all_evals.sh`)
do this for every DiT4DiT launch, so they run from the host. The setup script
prints the variables to export (the interpreter paths are the ones inside the
image). An environment built before `pydantic` was added to the RoboCasa client
environment fails every RoboCasa client at `from pydantic import BaseModel`:
rerun the script.

### DiT4DiT LIBERO

```bash
export DIT4DIT_LIBERO_CKPT=/path/to/dit4dit-model/dit4dit_libero/final_model/pytorch_model.pt   # mondo-robotics/dit4dit-model
export DIT4DIT_MODEL_PYTHON=$PWD/envs/model/bin/python    # printed by setup_dit4dit_rocm_envs.sh
export DIT4DIT_LIBERO_PYTHON=/opt/venv/bin/python3         # the image's own python3
export LIBERO_ROOT=/path/to/LIBERO
export DIT4DIT_PYOVERLAY=/path/to/pyoverlay
```

`DIT4DIT_DIFFUSERS_SRC` is optional (default: the model environment's own
diffusers, which must be unmodified 3996788).

```bash
# 6  DiT4DiT baseline, 4 step
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_libero.sh --arm baseline --steps 4 --gpus 0 --port 5694 --full

# 7  DiT4DiT GEAK, 4 step
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_libero.sh --arm geak --steps 4 --gpus 1 --port 5695 --full

# 8  DiT4DiT GEAK, 2 step   (--arm baseline likewise)
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_libero.sh --arm geak --steps 2 --gpus 5 --port 5696 --full
```

### DiT4DiT RoboCasa-GR1 full-24

One protocol for every RoboCasa checkpoint and tree: the vendored `DiT4DiT/`
tree's full-24 rollout-matching evaluation (24 task families × 50 episodes,
720 max episode steps, 12 action steps per chunk, evaluation seed 7 with an
explicit reset of every episode to a SHA-256-derived seed, policy seed 7,
server in bf16). `scripts/robocasa_full24.py` drives it with the vendored
harness, clients and summarizer; only the policy server differs between
arms.

```bash
export DIT4DIT_ROBOCASA_CKPT=/path/to/dit4dit-model/dit4dit_robocasa_gr1/final_model/pytorch_model.pt  # baseline/geak; --arm cae: your own CAE-DiT4DiT checkpoint
export ROBOSUITE_OVERLAY=/path/to/robosuite151/site-packages
export ROBOCASA_ROOT=/path/to/robocasa-gr1-tabletop-tasks  # deterministic-reset patch applied
export DIT4DIT_MODEL_PYTHON=$PWD/envs/model/bin/python    # printed by setup_dit4dit_rocm_envs.sh
export ROBOCASA_PYTHON=$PWD/envs/robocasa/bin/python
export DIT4DIT_PYOVERLAY=/path/to/pyoverlay
```

```bash
# 9  RoboCasa baseline, 4 step  (DiT4DiT authors' RoboCasa-GR1 checkpoint)
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh --arm baseline --steps 4 --gpus 0,1,2,3,4,5 --full

# 10 RoboCasa GEAK, 4 step  (--steps 2 likewise)
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh --arm geak --steps 4 --gpus 0,1,2,3,4,5 --full

# 11 CAE-DiT4DiT (K8 x d48 latent policy, 200K-step checkpoint you train)
bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh --arm cae --gpus 0,1,2,3,4,5 --full
```

The baseline/geak servers start through
`scripts/dit4dit_serve_policy.py`, like the LIBERO launcher's: the step
override, the keepalive handler, the four pinned `GEAK_*` gates and the
diffusers overlay all apply. Their `server_policy.py` has no `--seed`, so the
policy seed reaches them as `DIT4DIT_POLICY_SEED`, which the entry point turns
into the same `torch.manual_seed` / `torch.cuda.manual_seed_all` calls the
vendored server makes for `--seed`. The clients are always the vendored
`simulation_env.py`; it sends a `task_id` with each request, which the
side-by-side trees' `predict_action` ignores.

`--clients-per-gpu K` runs K clients against each server. Server g keeps the
tasks the harness gives it (`tasks[g::G]`), and its (task, episode) list is cut
into K contiguous shares. Each episode keeps its derived seed and its episode
index, and a task's shares are merged into the same `results/<task>.json` one
client writes (`tests/test_robocasa_full24.py` checks the split and the seeds).
What K does change is the order in which requests reach a server, and the
policy draws its sampling noise from the server's random stream in that order.
So a rerun replays the same policy calls only with the same arm, simulator,
episode count, GPU count and clients per GPU, and even then an exact replay
needs the pinned environment of `docs/cae-dit4dit-robocasa.md`: the
policy's GPU arithmetic is not bit-reproducible across GPU models and software
versions. `--dry-run` prints the resolved run and the plan without any assets. A rerun into the same
`--output-dir` skips finished tasks and finished shares. `--trials 1|10` runs
fewer episodes per task, and `--port` sets the first server's port (default
8098). `--no-result-gate` is accepted with a deprecation warning and has no
effect.

Every DiT4DiT checkpoint's `config.yaml` (the directory above `final_model/`)
must point `framework.cosmos25.base_model` at a local copy of the gated
`nvidia/Cosmos-Predict2.5-2B`: accept its licence on Hugging Face, log in, and
download it as in
[Cosmos-Predict2.5-2B](cae-dit4dit-robocasa.md#cosmos-predict25-2b). The simulator
environment needs gymnasium 1.x; the script checks.

The DiT4DiT authors' own batch script
(`examples/Robocasa_tabletop/eval_files/batch_eval_args.sh`) stays in every
tree, unchanged, but is not a protocol of this repository: it draws each
environment's episodes from one reset sequence rather than per-episode seeds.

## The policy server keeps answering pings during inference

Every tree's `WebsocketPolicyServer` runs `predict_action` on the asyncio event
loop, and the tree's client sends keepalive pings every 20 s with a 20 s
timeout. The GEAK arm's first call compiles its action-head step
(`GEAK_HEAD_IND`, on by default and not one of the four pinned gates). On a
cold compile cache, on a new input shape, or on a loaded host, that call can
outlast the ping window, and every connected client drops with
`1011 keepalive ping timeout` before the first action returns. This was seen on
the RoboCasa GEAK smoke; the baseline arm has no such compile.

`scripts/dit4dit_serve_keepalive.py`, imported by the server entry point on
every arm, replaces the tree's `_handler` with a copy that runs the policy call
on one dedicated worker thread. Requests are still served one at a time in
arrival order; only the event loop is freed. The module refuses to install if
the tree's handler is not the one it copied, and
`tests/test_side_by_side_trees.py` checks the copy against both trees.

## Known limitations

**The CAE-DiT4DiT server offers no `--steps`.**
`--arm cae` starts the vendored `DiT4DiT/` tree's `server_policy.py`
unchanged, as the published run did, so it never imports the step override.
Step reduction on RoboCasa runs on the side-by-side trees
(`--arm baseline|geak --steps N`).

**`server_policy.py` differs between the vendored tree and `src/`.** The
vendored copy accepts `--seed`; the two side-by-side copies are identical and
do not. Passing an unsupported flag is an argparse hard failure, so
`scripts/eval_dit4dit_libero.sh` detects support by inspecting the target file
rather than assuming. A LIBERO run against a side-by-side tree therefore does
not pass `--seed 7`; the clients still seed with `--args.seed 7`. The RoboCasa
launcher seeds those servers through `DIT4DIT_POLICY_SEED` instead (see above).

**`pyoverlay` is a runtime dependency, not source.** The DiT4DiT runners need
a small overlay of third-party packages (about 5.8 MiB: anyio, certifi,
docstring_parser, h11, httpcore, httpx, idna, typeguard, typing_extensions,
tyro) on `PYTHONPATH`. It is not committed. Build it once with the model
environment's Python and pass it via `DIT4DIT_PYOVERLAY`:

```bash
/path/to/model/venv/bin/python -m pip install --target /path/to/pyoverlay \
  anyio certifi docstring_parser h11 httpcore httpx idna typeguard \
  typing_extensions tyro
export DIT4DIT_PYOVERLAY=/path/to/pyoverlay
```

If the model environment's `transformers` or `tokenizers` differ from the
versions pinned in the tree's `requirements.txt`, install those pins into the
same overlay as well. `scripts/dit4dit_latency.py` refuses a cell whose imported
`transformers` or `tokenizers` does not match the pin. With the evaluation image
as `EVAL_IMAGE`, two variables give it the DiT4DiT model environment:

```bash
# next to the environments setup_dit4dit_rocm_envs.sh built under envs/
bash scripts/in_eval_image.sh pip install --no-deps --target envs/d4pins transformers==4.57.6 tokenizers==0.22.2
export DIT4DIT_DIFFUSERS_SRC=$PWD/envs/model/lib/python3.12/site-packages/diffusers   # diffusers 3996788
export DIT4DIT_PYOVERLAY=$PWD/envs/pyoverlay:$PWD/envs/d4pins
```

**A shutdown message from the GEAK tree is harmless.** When a DiT4DiT GEAK
server or latency worker exits, Python may print `Exception ignored in:
<function Cosmos25FeatureExtractor.__del__ ...> TypeError: 'NoneType' object is
not callable`: the destructor runs after interpreter shutdown has cleared the
module's globals. It happens after the run's results are written and does not
change them; the shipped GEAK tree is left as validated.

**Gradient checkpointing is training-only in the GEAK tree.** The
serving helper the published runs used disabled gradient checkpointing on every
module after load; without it the server died with "Inference tensors cannot be
saved for backward". The GEAK tree now enables checkpointing only
when the backbone is `trainable`, which the released LIBERO and RoboCasa
checkpoints do not set, and the upstream baseline tree never enables it, so
these scripts reach the same serving state without a load-time hook.
`tests/test_side_by_side_trees.py` pins both. Checkpointing
does not change inference numerics either way.

**Two of the baseline arm's three compile switches are dead.**
`FASTWAM_MOT_COMPILE_MODE` and `FASTWAM_VAE_COMPILE` are read by **no** FastWAM
source tree; only `FASTWAM_COMPILE` is real, and it defaults to `1` in
`compile_config.py:18`, so forcing it to `0` is load-bearing. The native
launcher and `scripts/compare_artifacts.py` set all three; the dead pair was
left in place — `scripts/compare_artifacts.py`'s `ARM_SPECS` sets the same two
names and `tests/test_fastwam_geak_tooling.py` constrains what may be renamed
there, so correcting the vocabulary is a separate change.
