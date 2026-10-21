# CAE-DiT4DiT: latent actions on RoboCasa

Training and deterministic evaluation code for CAE-DiT4DiT, the DiT4DiT
RoboCasa full-24 K8/d48 latent-action policy (Compact Action Expert). `DiT4DiT/` is a subtree of
[Mondo-Robotics/DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT), based on
upstream commit `66a6f3a`.

**The CAE-DiT4DiT checkpoint is not released.** Train your own with
[`scripts/train_dit4dit_robocasa_action_vae.sh`](../scripts/train_dit4dit_robocasa_action_vae.sh)
and
[`scripts/train_dit4dit_robocasa_latent_policy.sh`](../scripts/train_dit4dit_robocasa_latent_policy.sh),
then evaluate it with `scripts/eval_dit4dit_robocasa.sh --arm cae`, run inside the
evaluation image through `scripts/in_eval_image.sh` (all below).

The policy fine-tunes NVIDIA Cosmos-Predict2.5-2B (gated; NVIDIA Open Model
License), which you download from NVIDIA. The training dataset, GR00T-X-Embodiment-Sim, is CC BY 4.0. The RoboCasa-GR1
tabletop assets used for evaluation are **CC BY-NC 4.0 (non-commercial)** and
are downloaded by the setup script, never redistributed. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## RoboCasa evaluation

The 200K-step policy is evaluated across the complete 24-family RoboCasa-GR1
task set, with 50 episodes per family, at seed 7 under the pinned reproduction
stack. This is not an exact replay of the original DiT4DiT runs and not a
multi-seed evaluation. The reported 56.8% comes from a single run of
AMD's own CAE-DiT4DiT checkpoint, which is not released, on an AMD Instinct™ MI355X GPU with
`--sim instinct`; a checkpoint you train yourself will not match it exactly.

The evaluation contract is:

- Python 3.12.3, robosuite 1.5.1, MuJoCo 3.2.6, NumPy 1.26.4;
- `max_episode_steps=720` and `n_action_steps=12`;
- direct policy execution;
- policy seed 7 and evaluation seed 7;
- schema-2 explicit per-episode resets derived with SHA-256;
- 24 task families and 50 episodes per family.

## Paths

Every model artifact below is produced by the training steps in this guide.
Set the paths once:

```bash
cd DiT4DiT
export REPO_ROOT="$(pwd)"
# Written by the action-VAE training step (scripts/train_dit4dit_robocasa_action_vae.sh).
export ACTION_VAE="${REPO_ROOT}/playground/action_vae/full24_sweep/k8d48h256_none_align.pt"
# Policy-global action statistics of the full-24 dataset, written by the
# statistics step below (also the native policy launcher's default path).
export POLICY_STATS="${REPO_ROOT}/playground/Checkpoints_robocasa/benchmark_full24_b16_eager/dataset_statistics.json"
# Written by the policy training step (scripts/train_dit4dit_robocasa_latent_policy.sh).
# (Earlier versions of this guide called it POLICY_200K; the launcher still accepts that name.)
export CAE_DIT4DIT_CKPT="${REPO_ROOT}/playground/Checkpoints_robocasa/dit4dit_robocasa_full24_latent_k8_d48_mi325x8_b16_opt_s42/checkpoints/steps_200000_pytorch_model.pt"
export COSMOS_BASE="${REPO_ROOT}/Cosmos-Predict2.5-2B"
export DATA_ROOT="${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
export ROBOSUITE_OVERLAY="${REPO_ROOT}/environments/robosuite151-site-packages"
export ROBOCASA_ROOT="${REPO_ROOT}/robocasa-gr1-tabletop-tasks"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

## Environment

Use a ROCm 7.2 image containing the captured
`torch==2.10.0a0+git449b176` build (the reference training runs used
`rocm/primus:v26.2`: ROCm 7.2.0, PyTorch 2.10.0a0, Triton 3.6.0, Python 3.12.3).
Preserve that PyTorch installation:

```bash
python -m pip install -e . --no-deps
python -m pip install -r environments/requirements-train-rocm.txt
```

Run every training step below in this environment. The evaluation
environments `environments/setup_dit4dit_rocm_envs.sh` builds serve the
launchers and the action-VAE trainer, but not the DeepSpeed-launched policy
trainers (`MissingCUDAException: CUDA_HOME does not exist`). Start the
container with the GPUs (`--device=/dev/kfd --device=/dev/dri`, the `video` and
`render` groups) and with `--ipc=host` (or a `--shm-size` that fits the staged
dataset plus 64 GiB, see below). `rocm/primus:v26.2` has no `/usr/bin/time`;
install it (`apt-get install -y time`) if you want the wrappers' resource
reports.

### Cosmos-Predict2.5-2B

Every DiT4DiT model in this repository loads the NVIDIA Cosmos-Predict2.5-2B
backbone: the authors' LIBERO and RoboCasa-GR1 checkpoints (baseline and GEAK
arms) as well as the CAE-DiT4DiT policy you train here. The model is gated on
Hugging Face, so an anonymous download fails with a 401:

1. Sign in to Hugging Face and accept the NVIDIA Open Model License on
   [nvidia/Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
   (access is granted automatically).
2. Authenticate the download: `hf auth login`, or export `HF_TOKEN` with a read
   token (`huggingface-cli login` on huggingface_hub releases before 0.34).
3. Download it at the pinned revision (the `diffusers/base/post-trained` branch
   the DiT4DiT authors use, as of this release):

   ```bash
   hf download nvidia/Cosmos-Predict2.5-2B \
     --revision 0d37c7498f54cee3c599d438d895a0a4a8608064 \
     --local-dir "${COSMOS_BASE}"
   ```

4. Point the authors' checkpoints at it. Their `config.yaml`, next to
   `final_model/`, has the placeholder
   `framework.cosmos25.base_model: /path/to/Cosmos-Predict2.5-2B`:

   ```bash
   sed -i "s|base_model: /path/to/Cosmos-Predict2.5-2B|base_model: ${COSMOS_BASE}|" \
     /path/to/dit4dit-model/dit4dit_libero/config.yaml \
     /path/to/dit4dit-model/dit4dit_robocasa_gr1/config.yaml
   ```

   A policy you train records `COSMOS_BASE` in its own run's `config.yaml`.

Install the RoboCasa-GR1 source at commit `4840e67`, apply the deterministic
reset patch, and download its separately licensed simulator assets:

```bash
bash environments/setup_robocasa_gr1.sh
python -m pip install -e robocasa-gr1-tabletop-tasks
yes y | python robocasa-gr1-tabletop-tasks/robocasa/scripts/download_kitchen_assets.py   # asks y/n; about 5 GB
(cd robocasa-gr1-tabletop-tasks/robocasa/scripts && python download_groot_assets.py -y)
```

`download_groot_assets.py` fetches the GR1 tabletop object assets (`lightwheel`,
`sketchfab`; about 270 MB from the Hugging Face dataset
`nvidia/PhysicalAI-DigitalCousin-Assets`, no login needed, `HF_HUB_OFFLINE` must
be unset). Without them every RoboCasa-GR1 environment fails with
`Object registry sketchfab directory not found`.

Create the evaluation overlay. Keep it separate from the training environment:

```bash
bash environments/build_robosuite151_overlay.sh "${ROBOSUITE_OVERLAY}"
```

The captured evaluator used robosuite wheel `RECORD` SHA-256
`fe3a63ca1b4b05f0b0baffd40bf0c3fda28b2edef78fedddb9f098469a007944`.
The overlay builder records the locally resolved hash in
`ROBOSUITE_OVERLAY.json`; compare it before claiming an exact stack match.

## Dataset and immutable action cache

Download the 24-family dataset at pinned revision `ea7ac0b`. It has about 2,000
files and anonymous Hugging Face downloads are rate-limited, so log in (`hf auth login`)
or export `HF_TOKEN` first:

```bash
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py \
  --folder-group all \
  --output-dir "${DATA_ROOT}"

python examples/Robocasa_tabletop/train_files/audit_robocasa_full24.py \
  --data-root "${DATA_ROOT}" \
  --sample-count 3
```

For a smoke test, `--folders` downloads single task folders (about 2,000 files
and 2.5 GB each) instead of a `--folder-group`; the full-24 audit and action
cache need all 24:

```bash
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py \
  --folders gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000 \
  --output-dir "${DATA_ROOT}"
```

Prefer the script to `hf download --include "<folder>/*"`: that lists the
whole dataset repository before it filters, which takes a long time on this
dataset, while the script lists only the selected folders.

Build the balanced, per-family-normalized cache and frozen instruction
features. The cache manifest binds the source inventory, split IDs, seed,
normalization bounds, action indices, and tensor payload:

```bash
export FULL24_ACTION_CACHE="${REPO_ROOT}/playground/action_vae/full24_action_chunks_per_task.pt"

python examples/Robocasa_tabletop/train_files/prepare_pnp3_action_cache.py \
  --task-set full24 \
  --data-root "${DATA_ROOT}" \
  --normalization per-task \
  --output "${FULL24_ACTION_CACHE}"

python examples/Robocasa_tabletop/train_files/precompute_action_vae_instruction_embeddings.py \
  --cache "${FULL24_ACTION_CACHE}" \
  --model "${COSMOS_BASE}"
```

The generated cache is approximately 2.6 GiB.

`--task-set` has no subset option: `full24` needs all 24 folders. To smoke-test
the training scripts with the one folder downloaded above, give them a data root
of 24 symlinks to that folder. The cache then has the full-24 size and layout
but repeats one task's data, so use it only to check the pipeline, never for a
checkpoint you evaluate:

```bash
export SMOKE_DATA_ROOT=/path/to/smoke_full24
python -c '
import os, runpy, sys
FULL24_TASKS = runpy.run_path("DiT4DiT/dataloader/gr00t_lerobot/full24_tasks.py")["FULL24_TASKS"]
src, dst = map(os.path.abspath, sys.argv[1:])
os.makedirs(dst, exist_ok=True)
for task in FULL24_TASKS:
    os.symlink(src, os.path.join(dst, task.dataset_name))
' "${DATA_ROOT}/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000" "${SMOKE_DATA_ROOT}"
```

Then pass `--data-root "${SMOKE_DATA_ROOT}"` (and `DATA_ROOT_DIR`/`SOURCE_DATA_ROOT`
in the steps below) instead of `${DATA_ROOT}`.

## Train the K8/d48 action VAE

The policy uses the shared K8/d48/h256 codec with language alignment and no
family prompt in its encoder or decoder:

```bash
export FULL24_LANGUAGE_EMBEDDINGS="${FULL24_ACTION_CACHE%.pt}.instruction_embeddings.pt"
export FULL24_VAE_ROOT="${REPO_ROOT}/playground/action_vae/full24_sweep"

(cd .. && bash scripts/train_dit4dit_robocasa_action_vae.sh)
```

The wrapper selects `k8d48h256_none_align`, calls the native candidate
launcher, and writes a GNU `time -v` report under
`playground/full24_vae_sweep/`. The checkpoint lands at `ACTION_VAE`. It
trains in under 15 minutes on one AMD Instinct™ MI325X GPU.
Retraining the VAE is a reproducibility recipe, not a bitwise guarantee.

## Write the policy-global action statistics

The latent policy normalizes actions with the global statistics of the full-24
data mix (`fourier_gr1_unified_1000`). DiT4DiT's data loader writes them to
`<run_dir>/dataset_statistics.json` when it is built, before the first training
step. Produce `POLICY_STATS` with a minimal run of the stock RoboCasa-GR1
trainer that saves no checkpoint:

```bash
RUN_ID=benchmark_full24_b16_eager NUM_PROCESSES=1 MAX_TRAIN_STEPS=1 \
  SKIP_FINAL_CHECKPOINT=true COSMOS_BASE="${COSMOS_BASE}" DATA_ROOT_DIR="${DATA_ROOT}" \
  bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
test -f "${POLICY_STATS}"
```

## Train the 200K latent policy

The reference run used eight AMD Instinct™ MI325X GPUs, batch 16/GPU (global
batch 128), seed 42, frozen prompt embeddings, causal-VAE encode reuse, eager
Cosmos execution, and no gradient checkpointing, and took approximately five
days:

```bash
export SOURCE_DATA_ROOT="${DATA_ROOT}"
export STAGED_DATA_ROOT="/dev/shm/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
export ACTION_VAE_CHECKPOINT="${ACTION_VAE}"
export POLICY_ACTION_STATISTICS_PATH="${POLICY_STATS}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export PER_DEVICE_BATCH_SIZE=16
export MAX_TRAIN_STEPS=200000
export SEED=42

(cd .. && bash scripts/train_dit4dit_robocasa_latent_policy.sh)
```

The launcher validates the dataset, the backbone (`COSMOS_BASE`), that exactly
`NUM_PROCESSES` GPUs are visible, `transformers==4.57.6`, and free disk under
`DiT4DiT/` before training: at least `MIN_FREE_GIB` GiB, 500 by default through
this launcher (the preflight script's own default, when run directly, is 300). It copies the
dataset to `STAGED_DATA_ROOT` under `/dev/shm`, which needs the dataset's size
plus 64 GiB free. Other AMD Instinct™ GPUs are accepted with a warning;
`REQUIRE_GPU_MODEL=MI325X` restores the strict reference-hardware check. For a
smaller node, lower `NUM_PROCESSES` (listing that many GPUs in
`HIP_VISIBLE_DEVICES`) and raise `GRADIENT_ACCUMULATION_STEPS` to keep the
global batch of 128. Its defaults encode the production optimization
settings and save full models every 50K steps. For a short smoke run, also set
`SAVE_INTERVAL` and `FULL_MODEL_SAVE_INTERVAL` to at most `MAX_TRAIN_STEPS`: the
run directory's `config.yaml`, which the CAE-DiT4DiT server needs next to the
checkpoint, is written only at those saves. The top-level wrapper records
resource usage as `training_time.txt` in the policy run directory.

## Reproduce the seed-7 evaluation

The runner starts one policy server per listed GPU and evaluates disjoint task
assignments. Result files are resumable per task, but an already completed
output directory is never overwritten.

From the parent repository, the standalone launcher runs the full protocol and
validates the result files. It runs in the evaluation image with the
environments `environments/setup_dit4dit_rocm_envs.sh` builds (`EVAL_IMAGE` and
the variables the script prints;
[DiT4DiT Python environments on ROCm](side-by-side-source-trees.md#dit4dit-python-environments-on-rocm)):

```bash
export DIT4DIT_ROBOCASA_CKPT="${CAE_DIT4DIT_CKPT}"
(cd .. && bash scripts/in_eval_image.sh bash scripts/eval_dit4dit_robocasa.sh --arm cae)
```

This is the repository's one RoboCasa protocol; the same launcher evaluates the
side-by-side DiT4DiT trees with `--arm baseline|geak` (see
`docs/side-by-side-source-trees.md`). Use `--smoke` for one episode per family,
or `--trials 1|10` for fewer episodes per task (`--episodes` is a deprecated
spelling). `--gpus` changes the policy-server GPU list, `--port` the first
server's port (default 8098, or `PORT`), and `--clients-per-gpu K` runs K
clients per server on disjoint shares of its episodes (every episode keeps its
seed). `--dry-run` prints the resolved run and the episode plan without any
assets. The run ends by writing `summary.json` in its output directory (by
default under `eval/outputs/` of the parent repository).
The clients render camera images with GPU OpenGL (EGL) when the client Python
can create an EGL context and fall back to OSMesa otherwise; the run prints
which (`MuJoCo rendering: ...`), and an exported `MUJOCO_GL=egl|osmesa` forces
one. The direct runner below does the same.
`--no-result-gate` is still accepted but deprecated and has no effect. The
policy draws its sampling noise from each server's random stream in request
order, so a rerun replays the same policy calls only with the same arm,
simulator, episode count, GPU count and clients per GPU (the defaults: CPU
MuJoCo, 50 episodes, 6 GPUs, one client per GPU). Even then an
exact replay needs the pinned environment (see Environment above) and the same
`MUJOCO_GL` as the recorded run (EGL and OSMesa render slightly different
images; the run records its choice only in the printed `MuJoCo rendering:`
line, so keep that line, or set `MUJOCO_GL` explicitly, for a run you intend to
replay): the policy's
GPU arithmetic is not bit-reproducible across GPU models and software versions,
so elsewhere a rerun can differ in a few episodes.

The DiT4DiT authors' batch script
(`DiT4DiT/examples/Robocasa_tabletop/eval_files/batch_eval_args.sh`, described in
[`DiT4DiT/docs/robocasa_tabletop.md`](../DiT4DiT/docs/robocasa_tabletop.md)) is not this repository's protocol: it draws each
environment's episodes from one reset sequence rather than per-episode seeds.
Use the launcher above.

The equivalent direct runner invocation for that configuration is:

```bash
python examples/Robocasa_tabletop/eval_files/run_full24_evaluation.py \
  --checkpoint "${CAE_DIT4DIT_CKPT}" \
  --condition direct \
  --episodes 50 \
  --gpu-list 0,1,2,3,4,5 \
  --base-port 8098 \
  --policy-seed 7 \
  --eval-seed 7 \
  --max-episode-steps 720 \
  --n-action-steps 12 \
  --robosuite-overlay "${ROBOSUITE_OVERLAY}" \
  --robocasa-root "${ROBOCASA_ROOT}" \
  --output-dir "${REPO_ROOT}/playground/full24-policy200k-seed7"
```

Validate the new run:

```bash
python examples/Robocasa_tabletop/eval_files/summarize_full24_policy.py \
  "${REPO_ROOT}/playground/full24-policy200k-seed7"
```

The validator requires exactly 24 result files, 50 episodes per task, the
pinned versions and rollout protocol, and canonical derived episode seeds. It
prints the success count and rate of the run it validates.
