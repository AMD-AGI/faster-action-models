# CAE-FastWAM: Compact Action Expert on LIBERO

Training and evaluation code for CAE-FastWAM (Compact Action Expert: the
32-hidden-dimension Action DiT with the K8 x d16 action VAE, GEAK-optimized).
`FastWAM/` is a subtree of [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM), based on
upstream commit `7faa711`.

The checkpoint is fine-tuned from Wan2.2-TI2V-5B and released under
Apache-2.0; see [`../model_cards/fastwam-libero/`](../model_cards/fastwam-libero/)
and [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). The LIBERO training
data is CC BY 4.0.

## Architecture

The **Compact Action Expert** is one model that combines two architecture
changes:

1. **Smaller Action DiT.** The action expert is reduced to a 32-dimensional
   hidden state (about 30M parameters).
2. **K8 x d16 Action VAE.** A suite-conditioned action VAE compresses each
   32-step action chunk into 8 tokens with 16 dimensions per token, and the
   Action DiT denoises in that latent space.

It is trained on all four LIBERO suites and evaluated on them with 50 episodes
per task, using the commands below. The evaluated denoise step counts are 10
and 4. The accompanying
[ROCm blog](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html)
(Table 3) gives these reference results, on one AMD Instinct™ MI355X GPU with `--sim instinct`:

| Configuration | Success rate (%) | Latency (ms) |
|---|---|---|
| FastWAM, no optimization (10 steps) | 96.95 ± 0.22 | 222.55 |
| FastWAM + GEAK (10 steps) | 96.86 ± 0.30 | 24.17 |
| CAE-FastWAM + GEAK (10 steps) | 97.28 ± 0.20 | 12.82 |
| CAE-FastWAM + GEAK (4 steps) | 97.33 ± 0.26 | 8.83 |

Success rate is the mean ± standard deviation over 11 full LIBERO runs (5 for the
4-step row); latency is per policy decision. Reruns are not bit-reproducible
and can differ by a few tenths of a point.

These experiments explicitly use action-scheduler `train_shift=5.0` and
`infer_shift=5.0`; the upstream FastWAM default remains `1.0`.

> **Note:** Controlled LIBERO evaluations of the Compact Action Expert with
> action-scheduler shifts `1.0` and `5.0` have not yet been run; keep the
> experiment-specific settings unless you run that comparison.

## Models and assets

Checkpoint files, hashes, shared dependencies, and Hugging Face
download instructions live in [`MODEL_ZOO.md`](../MODEL_ZOO.md). Following it,
from the repository root:

```bash
hf download amd/CAE-FastWAM --local-dir checkpoints/CAE-FastWAM   # current huggingface_hub; older: huggingface-cli download
export CAE_FASTWAM_DIR="$(pwd)/checkpoints/CAE-FastWAM"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # holds Wan-AI/... (MODEL_ZOO.md)
```

All commands below run from the imported project:

```bash
cd FastWAM
export CAE_EVAL_ROOT="$(pwd)/evaluate_results/cae-fastwam"
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

`MUJOCO_GL` stays unset: MuJoCo renders camera images with GPU OpenGL (EGL)
when it is available and falls back to OSMesa otherwise (see the evaluation
below). Export `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` to force one.

The evaluation examples expect `LATENT_FASTWAM_CHECKPOINT`, `K8D16_VAE_CHECKPOINT`,
and `PER_SUITE_STATS_DIR` to point to the downloaded files (or to your own
trained ones):

```bash
export LATENT_FASTWAM_CHECKPOINT="${CAE_FASTWAM_DIR}/policy.pt"
export K8D16_VAE_CHECKPOINT="${CAE_FASTWAM_DIR}/action_vae.pt"
export PER_SUITE_STATS_DIR="${CAE_FASTWAM_DIR}/stats"    # libero_{spatial,object,goal,10}.json
```

## Training data

Everything training needs is public. The commands run from `FastWAM/`, and the
paths match the defaults of the task configuration
`configs/task/libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint.yaml`.
Each default can be overridden on the training command line as a Hydra override:

| Override | Default (relative to `FastWAM/`) | Holds |
|---|---|---|
| `libero_data_root` | `./data/libero_mujoco3.3.2` | `libero_{spatial,object,goal,10}_no_noops_lerobot` (LeRobot 2.1) |
| `per_suite_stats_dir` | `../checkpoints/CAE-FastWAM/stats` | `libero_{suite}.json` per-suite normalization statistics |
| `text_embeds_cache_dir` | `./data/text_embeds_cache/libero` | the cached T5 text embeddings |

For example, `libero_data_root=/dev/shm` reads datasets staged in RAM, which
speeds up data loading.

1. **LIBERO datasets** (about 4.7 GB of tarballs, CC BY 4.0):

   ```bash
   hf download yuanty/LIBERO-fastwam --repo-type dataset --include "*.tar.gz" --local-dir data/LIBERO-fastwam
   mkdir -p data/libero_mujoco3.3.2
   for f in data/LIBERO-fastwam/libero_*_no_noops_lerobot.tar.gz; do tar -xzf "$f" -C data/libero_mujoco3.3.2; done
   ```

2. **Per-suite normalization statistics.** CAE-FastWAM normalizes each suite
   with its own min/max statistics. Use the ones it was trained with, published
   with the checkpoint (small JSON files only):

   ```bash
   hf download amd/CAE-FastWAM --include "stats/*" --local-dir ../checkpoints/CAE-FastWAM
   ```

3. **Text-embedding cache** (umT5-XXL, from the Wan assets in
   [`MODEL_ZOO.md`](../MODEL_ZOO.md#shared-runtime-assets)):

   ```bash
   python scripts/precompute_text_embeds.py \
     task=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
   ```

4. **Action cache for the Action VAE** (`PREDECODED_ACTION_CACHE`). The
   builder reads only the parquet action columns, needs no GPU, and takes
   seconds; the output is about 260 MB. It runs in the FastWAM environment or
   in the evaluation image ([`docker/Dockerfile.eval`](../docker/Dockerfile.eval)):

   ```bash
   python3 ../scripts/build_fastwam_action_cache.py \
     --data-root data/libero_mujoco3.3.2 \
     --stats-dir ../checkpoints/CAE-FastWAM/stats \
     --output data/fastwam_libero_action_cache.pt
   export PREDECODED_ACTION_CACHE="$(pwd)/data/fastwam_libero_action_cache.pt"
   ```

   It holds one 32-step chunk per frame (277,713 chunks from 1,712 episodes),
   built as FastWAM's training dataset builds the action targets: steps past
   the end of an episode repeat the last action and are marked as padding,
   their delta dimensions are zeroed, and each chunk is normalized with its
   suite's statistics. Suite ids follow the evaluator: 0 spatial, 1 object,
   2 goal, 3 libero_10. If you only need the cache, the LeRobot 3.0 copy of the
   dataset holds the same actions in about 40 MB of parquet files:
   `hf download yuanty/LIBERO-fastwam --repo-type dataset --include "lerobot_v30/*.parquet" --local-dir data/LIBERO-fastwam`,
   then `--data-root data/LIBERO-fastwam/lerobot_v30`.

   The published statistics record 1,703 episodes and 274,902 frames, a few
   fewer than the public dataset now holds, so a handful of LIBERO-10 actions
   normalize slightly outside [-1, 1] (up to 1.25). As a check, the published
   `action_vae.pt` reconstructs a cache built this way with a mean squared error
   of 7.3e-5, against 7.1e-5 at the end of its own training.

5. **Wan2.2-TI2V-5B video DiT** (about 20 GB). The initializer below and the
   latent-policy training load it from `DIFFSYNTH_MODEL_BASE_PATH`, next to the
   evaluation assets. With `DIFFSYNTH_SKIP_DOWNLOAD=true` nothing is fetched, so
   download it first ([`MODEL_ZOO.md`](../MODEL_ZOO.md#shared-runtime-assets)):

   ```bash
   hf download Wan-AI/Wan2.2-TI2V-5B \
     config.json diffusion_pytorch_model.safetensors.index.json \
     diffusion_pytorch_model-00001-of-00003.safetensors \
     diffusion_pytorch_model-00002-of-00003.safetensors \
     diffusion_pytorch_model-00003-of-00003.safetensors \
     --local-dir ../checkpoints/Wan-AI/Wan2.2-TI2V-5B
   ```

   If they are missing, loading stops with
   `Cannot detect model type for wan_video_dit. File: []`.

### Training environment

Training runs in the evaluation image ([`docker/Dockerfile.eval`](../docker/Dockerfile.eval))
plus a small overlay. Do not install `pyproject.toml`'s dependencies on ROCm:
they pin CUDA builds of PyTorch. Inside the image, from `FastWAM/` (as root, or
build a derived image with these lines):

```bash
apt-get install -y --no-install-recommends ffmpeg
pip install -r ../docker/requirements-fastwam-train.txt
pip install -e . --no-deps    # the fastwam package itself; without it: No module named 'fastwam'
```

Start the training container with the GPUs, the repository and the data
mounted, and with the host IPC namespace (or a large `--shm-size`): the data
loader workers share batches through `/dev/shm`, and the 64 MB docker default
fails with `unable to allocate shared memory(shm)`:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --security-opt seccomp=unconfined -v "$PWD:$PWD" -w "$PWD/FastWAM" <training-image>
```

The overlay adds DeepSpeed, which `scripts/train_zero1.sh` launches through
accelerate, and torchcodec. The FastWAM data loader decodes the LeRobot videos
with torchcodec, which needs the FFmpeg shared libraries: torchvision 0.26 has
no `io.VideoReader` fallback, so without torchcodec training fails with
`module 'torchvision.io' has no attribute 'VideoReader'`. The Action VAE and
the action-cache builder need neither.

A `max_steps` smoke run still writes a full checkpoint when it stops (about
77 GB for the latent policy), whatever `save_every` is.

## Initialize the 32-dimensional Action DiT

Initialize the 32-dimensional action expert from the video DiT; Compact Action
Expert training below starts from this initializer:

```bash
export COMPACT_INIT_CHECKPOINT="<output-path-for-compact-action-dit-init.pt>"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam_32hdim_joint.yaml \
  --output "${COMPACT_INIT_CHECKPOINT}"
```

## Train the K8 x d16 Action VAE

The 453,927-parameter, suite-conditioned variational Action VAE maps each
`[32, 7]` action chunk to 8 latent tokens of width 16. Its encoder projects
actions to 128 channels, downsamples time by 4 with a residual Conv1D block,
and predicts the posterior mean and log variance. The symmetric decoder uses
a transposed convolution and residual block. Training combines masked MSE,
0.1-weighted velocity reconstruction, KL weight `1e-4`, free bits `0.1`, and
EMA weights.

Train it with the standalone launcher from the parent repository:

```bash
export PREDECODED_ACTION_CACHE="$(pwd)/data/fastwam_libero_action_cache.pt"   # see Training data
export K8D16_VAE_OUTPUT="<new-output-path-for-k8d16-vae.pt>"

(cd .. && bash scripts/train_fastwam_libero_action_vae.sh)
```

The launcher records resource usage beside `K8D16_VAE_OUTPUT`. The reference
Action VAE trained in under 15 minutes on one AMD Instinct™ MI325X GPU. The
equivalent direct invocation is:

```bash
python scripts/train_action_ae.py \
  --predecoded "${PREDECODED_ACTION_CACHE}" \
  --num-tokens 8 \
  --latent-dim 16 \
  --encoder-hidden 128 \
  --decoder-hidden 128 \
  --video-cond-dim 0 \
  --num-suites 4 \
  --variational \
  --beta-kl 1e-4 \
  --free-bits 0.1 \
  --kl-warmup-epochs 100 \
  --epochs 50 \
  --batch-size 256 \
  --output "${K8D16_VAE_OUTPUT}"
```

## Train the Compact Action Expert

Train the Action DiT in the Action VAE's latent space with the standalone
launcher:

```bash
export COMPACT_INIT_CHECKPOINT="<compact-action-dit-initializer.pt>"
export K8D16_VAE_CHECKPOINT="${K8D16_VAE_OUTPUT}"

(cd .. && bash scripts/train_fastwam_libero_latent_policy.sh)
```

It defaults to two workers and records resource usage in the run directory.
Extra arguments are passed to the trainer as Hydra overrides, for example the
data locations from [Training data](#training-data):

```bash
(cd .. && bash scripts/train_fastwam_libero_latent_policy.sh \
  libero_data_root=/dev/shm per_suite_stats_dir=/path/to/stats)
```

The reference run took 22.7 hours on two AMD Instinct™ MI325X GPUs (ROCm 7.2.0,
PyTorch 2.10.0a0, `rocm/primus:v26.2`). The equivalent native launcher is:

```bash
bash scripts/train_zero1.sh 2 \
  task=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint \
  model.action_dit_pretrained_path="${COMPACT_INIT_CHECKPOINT}" \
  model.action_vae_ckpt="${K8D16_VAE_CHECKPOINT}"
```

## Evaluate the Compact Action Expert

Evaluate it on the four suites, one suite at a time with that suite's
normalization statistics:

```bash
# MUJOCO_GL for this Python: egl (GPU OpenGL) when it can create an EGL
# context, else osmesa; an exported MUJOCO_GL is kept. The launchers do this
# themselves.
GL_ENV="$(python3 ../scripts/eval_flags.py mujoco-gl -- python)"
for SUITE in libero_spatial libero_object libero_goal libero_10; do
  STATS="${PER_SUITE_STATS_DIR}/${SUITE}.json"

  env ${GL_ENV} python scripts/eval_libero_aac.py \
    --fastwam-ckpt "${LATENT_FASTWAM_CHECKPOINT}" \
    --action-vae-ckpt "${K8D16_VAE_CHECKPOINT}" \
    --task-config libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint \
    --task-suite "${SUITE}" \
    --dataset-stats "${STATS}" \
    --num-trials 50 \
    --num-inference-steps 10 \
    --seed 7 \
    --output-dir "${CAE_EVAL_ROOT}/latent/${SUITE}"
done
```

`CAE_EVAL_ROOT` is a parent directory for these per-suite runs. It is not
`EVAL_OUTPUT_ROOT`, which the repository's launchers take as the run directory
itself.

### With the launcher

`eval/native/run_fastwam_libero.sh --arm cae` runs the same model with FastWAM's
own evaluator, from the GEAK-optimized tree `src/cae-fastwam`, under the
repository protocol ([`eval/PROTOCOL.md`](../eval/PROTOCOL.md), seed 42; the
`eval_libero_aac.py` loop above passes its own seed, 7, explicitly), one suite at
a time with its own task config, action-VAE suite id and statistics.
(`scripts/eval_fastwam_libero.sh` is a wrapper over it that defaults to one GPU.)
The launcher lives in the parent repository, so from `FastWAM/`:

```bash
export MODEL_CKPT="${LATENT_FASTWAM_CHECKPOINT}"
export ACTION_VAE_CKPT="${K8D16_VAE_CHECKPOINT}"
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
export DATASET_STATS_TEMPLATE="${PER_SUITE_STATS_DIR}/{suite}.json"
export LIBERO_ROOT=/path/to/LIBERO
export EVAL_IMAGE=fam-eval:rocm7.14   # built from docker/Dockerfile.eval (see eval/native/README.md)

(cd .. && bash eval/native/run_fastwam_libero.sh --arm cae --steps 10 --gpus 0 --dry-run)   # print the resolved run
(cd .. && bash eval/native/run_fastwam_libero.sh --arm cae --steps 10 --gpus 0 --full)
```

The launcher runs 10 steps by default; the evaluated configurations are
`--arm cae --steps 10|4`. `--sim instinct` runs the same evaluation GPU-native on
MI350X/MI355X, and `--gpus 0,1,2,3` spreads each suite's tasks over four GPUs; see
[`side-by-side-source-trees.md`](side-by-side-source-trees.md).
