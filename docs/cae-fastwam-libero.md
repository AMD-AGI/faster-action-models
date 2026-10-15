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
and 4. Results are reported in the accompanying blog post.

These experiments explicitly use action-scheduler `train_shift=5.0` and
`infer_shift=5.0`; the upstream FastWAM default remains `1.0`.

> **Note:** Controlled LIBERO evaluations of the Compact Action Expert with
> action-scheduler shifts `1.0` and `5.0` have not yet been run; keep the
> experiment-specific settings unless you run that comparison.

## Models and assets

Checkpoint paths, hashes, shared dependencies, and Hugging Face
download instructions live in [`MODEL_ZOO.md`](../MODEL_ZOO.md).

All commands below run from the imported project:

```bash
cd FastWAM
export CAE_EVAL_ROOT="$(pwd)/evaluate_results/cae-fastwam"
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

`MUJOCO_GL` stays unset: MuJoCo renders camera images with GPU OpenGL (EGL)
when it is available and falls back to OSMesa otherwise (see the evaluation
below). Export `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` to force one.

The examples expect `LATENT_FASTWAM_CHECKPOINT`, `K8D16_VAE_CHECKPOINT`, and
`PER_SUITE_STATS_DIR` to point to the downloaded model-zoo assets.

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
export PREDECODED_ACTION_CACHE="<path-to-libero-actions-with-suite-ids.pt>"
export K8D16_VAE_OUTPUT="<new-output-path-for-k8d16-vae.pt>"

(cd .. && bash scripts/train_fastwam_libero_action_vae.sh)
```

The launcher records resource usage beside `K8D16_VAE_OUTPUT`. The equivalent
direct invocation is:

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
The equivalent native launcher is:

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
  STATS="${PER_SUITE_STATS_DIR}/dataset_stats_${SUITE}.json"

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

`scripts/eval_fastwam_libero.sh --arm cae` runs the same model, served from the
GEAK-optimized tree `src/cae-fastwam`, through the evaluation harness under the repository protocol
([`eval/PROTOCOL.md`](../eval/PROTOCOL.md), seed 42; the `eval_libero_aac.py`
loop above passes its own seed, 7, explicitly), one policy server per suite.
The launcher lives in the parent repository, so from `FastWAM/`:

```bash
export MODEL_CKPT="${LATENT_FASTWAM_CHECKPOINT}"
export ACTION_VAE_CKPT="${K8D16_VAE_CHECKPOINT}"
export TASK_CONFIG=libero_uncond_2cam224_1e-4_32hdim_latent_K8d16_freebits_pernorm_joint
export DATASET_STATS_TEMPLATE="${PER_SUITE_STATS_DIR}/dataset_stats_{suite}.json"
export EVAL_HARNESS=/path/to/vla-evaluation-harness LIBERO_ROOT=/path/to/LIBERO
export EVAL_IMAGE="<FastWAM eval image>"

(cd .. && bash scripts/eval_fastwam_libero.sh --arm cae --dry-run)   # print the resolved run
(cd .. && bash scripts/eval_fastwam_libero.sh --arm cae --full)
```

The launcher runs 10 steps by default; the evaluated configurations are
`bash scripts/eval_fastwam_libero.sh --arm cae --steps 10|4`; see
[`side-by-side-source-trees.md`](side-by-side-source-trees.md).
