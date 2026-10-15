# CAE-DiT4DiT: latent actions on RoboCasa

Training and deterministic evaluation code for CAE-DiT4DiT, the DiT4DiT
RoboCasa full-24 K8/d48 latent-action policy (Compact Action Expert). `DiT4DiT/` is a subtree of
[Mondo-Robotics/DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT), based on
upstream commit `66a6f3a`.

**The CAE-DiT4DiT checkpoint is not released.** Train your own with
[`scripts/train_dit4dit_robocasa_action_vae.sh`](../scripts/train_dit4dit_robocasa_action_vae.sh)
and
[`scripts/train_dit4dit_robocasa_latent_policy.sh`](../scripts/train_dit4dit_robocasa_latent_policy.sh),
then evaluate it with `scripts/eval_dit4dit_robocasa.sh --arm cae` (all below).

The policy fine-tunes NVIDIA Cosmos-Predict2.5-2B (gated; NVIDIA Open Model
License), which you download from NVIDIA. The training dataset, GR00T-X-Embodiment-Sim, is CC BY 4.0. The RoboCasa-GR1
tabletop assets used for evaluation are **CC BY-NC 4.0 (non-commercial)** and
are downloaded by the setup script, never redistributed. See
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## RoboCasa evaluation

The 200K-step policy is evaluated across the complete 24-family RoboCasa-GR1
task set, with 50 episodes per family, at seed 7 under the pinned reproduction
stack. This is not an exact replay of the original DiT4DiT runs and not a
multi-seed evaluation. Results are reported in the accompanying blog post.

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
export POLICY_200K="${REPO_ROOT}/playground/Checkpoints_robocasa/dit4dit_robocasa_full24_latent_k8_d48_mi325x8_b16_opt_s42/checkpoints/steps_200000_pytorch_model.pt"
export COSMOS_BASE="${REPO_ROOT}/Cosmos-Predict2.5-2B"
export DATA_ROOT="${REPO_ROOT}/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
export ROBOSUITE_OVERLAY="${REPO_ROOT}/environments/robosuite151-site-packages"
export ROBOCASA_ROOT="${REPO_ROOT}/robocasa-gr1-tabletop-tasks"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

## Environment

Use a ROCm 7.2 image containing the captured
`torch==2.10.0a0+git449b176` build. Preserve that PyTorch installation:

```bash
python -m pip install -e . --no-deps
python -m pip install -r environments/requirements-train-rocm.txt
```

Fetch the gated Cosmos backbone at the pinned revision:

```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision 0d37c7498f54cee3c599d438d895a0a4a8608064 \
  --local-dir "${COSMOS_BASE}"
```

Install the RoboCasa-GR1 source at commit `4840e67`, apply the deterministic
reset patch, and download its separately licensed simulator assets:

```bash
bash environments/setup_robocasa_gr1.sh
python -m pip install -e robocasa-gr1-tabletop-tasks
python robocasa-gr1-tabletop-tasks/robocasa/scripts/download_kitchen_assets.py
```

Create the evaluation overlay. Keep it separate from the training environment:

```bash
bash environments/build_robosuite151_overlay.sh "${ROBOSUITE_OVERLAY}"
```

The captured evaluator used robosuite wheel `RECORD` SHA-256
`fe3a63ca1b4b05f0b0baffd40bf0c3fda28b2edef78fedddb9f098469a007944`.
The overlay builder records the locally resolved hash in
`ROBOSUITE_OVERLAY.json`; compare it before claiming an exact stack match.

## Dataset and immutable action cache

Download the 24-family dataset at pinned revision `ea7ac0b`:

```bash
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py \
  --folder-group all \
  --output-dir "${DATA_ROOT}"

python examples/Robocasa_tabletop/train_files/audit_robocasa_full24.py \
  --data-root "${DATA_ROOT}" \
  --sample-count 3
```

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
`playground/full24_vae_sweep/`. The checkpoint lands at `ACTION_VAE`.
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
Cosmos execution, and no gradient checkpointing:

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

The launcher validates the dataset, backbone, visible AMD Instinct™ MI325X GPU count, and disk
space before training. Its defaults encode the production optimization
settings and save full models every 50K steps. The top-level wrapper records
resource usage as `training_time.txt` in the policy run directory.

## Reproduce the seed-7 evaluation

The runner starts one policy server per listed GPU and evaluates disjoint task
assignments. Result files are resumable per task, but an already completed
output directory is never overwritten.

From the parent repository, the standalone launcher runs the full protocol and
validates the result files:

```bash
export DIT4DIT_ROBOCASA_CKPT="${POLICY_200K}"
(cd .. && bash scripts/eval_dit4dit_robocasa.sh --arm cae)
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
  --checkpoint "${POLICY_200K}" \
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
