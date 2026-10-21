# FastWAM GEAK: inference optimization of world-action models

This workflow optimizes the inference path of an already-trained world-action
model without retraining it. The Compact Action Expert (CAE-FastWAM,
CAE-DiT4DiT) changes the architecture; this workflow leaves the checkpoint
fixed and optimizes the runtime.

Apply the three levers in this order:

1. **`torch.compile`** is the baseline optimization.
2. **GEAK** profiles the real call, proposes kernel/model changes, and verifies
   each change against correctness gates.
3. **Denoise-step reduction** reduces flow-matching steps (from 10 to 2 on the
   GEAK arm) and is accepted only after closed-loop accuracy evaluation.

Latency changes across nodes, drivers, and sessions; run the comparison on the
deployment machine. These are the reference results of the accompanying
[write-up](https://rocm.blogs.amd.com/artificial-intelligence/faster-action-models/README.html),
on one AMD Instinct™ MI355X GPU (ROCm 7.14.0, PyTorch 2.11.0, `--sim instinct`),
as success rate (latency per policy decision):

| Configuration | FastWAM, LIBERO | DiT4DiT, LIBERO | DiT4DiT, RoboCasa-GR1 |
|---|---|---|---|
| No optimization (10 / 4 / 4 steps) | 96.95% (222.55 ms) | 98.16% (81.47 ms) | 57.9% (82.5 ms) |
| GEAK (same steps) | 96.86% (24.17 ms) | 98.24% (6.49 ms) | 58.2% (6.69 ms) |
| GEAK + 2 denoising steps | 97.00% (9.59 ms) | 98.08% (5.65 ms) | 60.9% (5.35 ms) |

LIBERO success rates are the mean of 11 full runs; RoboCasa-GR1 rates are
from a single run, and repeat full-24 runs (1,200 episodes) differ by up to
about 4 points: our repeats gave 57.9% and 61.6% for the baseline (the second on an
MI350X) and 60.9% and 60.3% for GEAK with 2 steps. The RoboCasa-GR1 rows are
therefore at parity within run-to-run spread; fewer denoising steps are not
claimed to improve accuracy. Naive `torch.compile` cuts eager latency 1.6-2.6x on the
same GPU, and GEAK runs 3.7x faster than `torch.compile` for FastWAM and
7.7-7.9x faster for DiT4DiT. The published FastWAM LIBERO result for the
checkpoint used here (`yuanty/fastwam`, `libero_uncond_2cam224.pt`) is 97.6%
(arXiv:2603.16666); the 97.75% / 98.55% table in `FastWAM/README.md` is for
upstream's later `libero_optional_idm_2cam224.pt` checkpoint.

## Layout

- [`../geak/README.md`](../geak/README.md) describes the GEAK campaign.
- [`../geak/task/`](../geak/task/) contains compile, correctness, and
  performance gates.
- [`../eval/README.md`](../eval/README.md) describes closed-loop evaluation.
- [`../eval/PROTOCOL.md`](../eval/PROTOCOL.md) defines the LIBERO protocol.
- [`../scripts/compare_artifacts.py`](../scripts/compare_artifacts.py) times
  eager, compile, GEAK, and reduced-step sources in one interleaved run. It
  measures latency only and never compares the arms' outputs; numerics are
  checked by the GEAK task's `oracle` and `correctness` gates.

## Prerequisites

The repository includes FastWAM at `FastWAM/`. The following paths may point
elsewhere, but they must be visible inside the selected container.

For the evaluation launcher (`eval/native/run_fastwam_libero.sh`):

```bash
export MODEL_CKPT="<path-to>/libero_uncond_2cam224.pt"                      # yuanty/fastwam, see MODEL_ZOO.md
export DATASET_STATS="<path-to>/libero_uncond_2cam224_dataset_stats.json"
export LIBERO_ROOT="<path-to-LIBERO>"                     # commit 8f1084e
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export EVAL_IMAGE=fam-eval:rocm7.14   # docker build -f docker/Dockerfile.eval -t fam-eval:rocm7.14 .
export EVAL_OUTPUT_ROOT="<output-directory>"
```

`eval/native/run_fastwam_libero.sh` fixes the source tree per `--arm`
(`src/fastwam-baseline` or `src/fastwam-geak`) and their config tree
(`FastWAM/configs_baseline`) by itself; it reads no `MODEL_CONFIGS`,
`BASELINE_SRC` or `OPTIMIZED_SRC` (the wrapper `scripts/eval_fastwam_libero.sh`
ignores them with a warning). `FastWAM/configs` and `FastWAM/src` hold the
CAE-FastWAM config schema and its unoptimized code; the FastWAM baseline and GEAK
trees reject that schema.

Additionally, for the GEAK task (`geak/task/run_local.sh`) and
`scripts/compare_artifacts.py`:

```bash
export MODEL_SRC_ROOT="$PWD/FastWAM"
export HF_CACHE_DIR="<path-to-huggingface-cache>"
export MODEL_PYTHON="/opt/venv/bin/python3"
export GEAK_ROOT="<path-to-GEAK-checkout>"

# Source trees the comparison tool times: the FastWAM baseline and GEAK trees
# (also its defaults when these are unset).
export BASELINE_SRC="$PWD/src/fastwam-baseline"
export OPTIMIZED_SRC="$PWD/src/fastwam-geak"
```

These two source variables name exactly the trees the launcher evaluates for
`--arm baseline` and `--arm geak`; only the comparison tool reads them. An
`--arm cae` run evaluates the CAE-FastWAM tree (`src/cae-fastwam`). `MODEL_PYTHON` is the GEAK task's variable;
`scripts/eval_dit4dit_libero.sh` no longer reads it without a deprecation
warning (it takes `DIT4DIT_MODEL_PYTHON` or `DIT4DIT_PYTHON`).

What the GEAK task and `scripts/compare_artifacts.py` need besides these
variables:

- **An image with the full evaluator stack.** Each arm runs in `EVAL_IMAGE`;
  build it from `docker/Dockerfile.eval`. Every FastWAM tree imports GitPython
  and boto3 when the model is built, and
  `eval/native/requirements-fastwam-libero.txt` pins both; an image without them
  fails each arm with `Error locating target 'fastwam.runtime.create_fastwam'`.
  `run_local.sh` sets `HYDRA_FULL_ERROR=1`, so the underlying error is printed.
- **The Wan assets under `HF_CACHE_DIR`.** The task reads the Wan files from
  `DIFFSYNTH_MODEL_BASE_PATH`, which defaults to `HF_CACHE_DIR`, so that
  directory must hold the `Wan-AI/...` files `MODEL_ZOO.md` downloads, laid out
  as there (the timing script builds the model without the text encoder, so it
  reads only `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`). An exported
  `DIFFSYNTH_MODEL_BASE_PATH`, `DIFFSYNTH_DOWNLOAD_SOURCE` or
  `DIFFSYNTH_SKIP_DOWNLOAD` is forwarded, and the path is mounted. Symlinks in
  these directories must point to paths visible inside the container.
- **LIBERO, for `correctness` only.** `run_local.sh` checks `LIBERO_ROOT` and
  `LIBERO_CONFIG_PATH` only for `correctness`, the one command that runs
  episodes (FastWAM's own evaluator, `experiments/libero/eval_libero_single.py`
  in `MODEL_SRC_ROOT`, in its persistent-worker mode); `compile`, `oracle` and
  `performance` need neither.
- **Container flags.** `EVAL_DOCKER_ARGS` adds `docker run` flags to every task
  container (for example `--name geak-perf-g3 --cpuset-cpus 104-127`). The
  reference latency runs pinned each timed container to the CPUs of its GPU's
  NUMA node, away from other work on the node.

All filesystem path variables above must be absolute. `geak/task/run_local.sh`
and the evaluation launcher mount host paths into their containers at the same
locations, including paths outside `/mnt` and `/tmp`. `MODEL_PYTHON` is the
in-container interpreter path of the GEAK task and does not need to exist on
the host. The comparison tool forwards its selected source as `FWE_SRC`; that
directory must directly contain the importable `fastwam/` package. For a
FastWAM baseline or GEAK tree it also forwards `FWE_CONFIG_DIR` as
`FastWAM/configs_baseline`, the config schema those trees take; set
`FWE_CONFIG_DIR` yourself to time another tree with other configs. When
`FWE_SRC` is unset, the GEAK task copies `MODEL_SRC_ROOT/src` and overlays the
editable files from `geak/task/src`.

The workflow was validated with GEAK at `fa86fd1`; pin it to re-run the campaign
(the shipped GEAK trees do not record a per-change GEAK revision):

```bash
git -C "${GEAK_ROOT}" checkout fa86fd1
```

The evaluation container should provide PyTorch, ROCm, Triton, LIBERO, and MuJoCo.
Do not install the CUDA-specific dependencies from `FastWAM/pyproject.toml`
into the ROCm container.

With the launcher variables above set, the baseline evaluation is:

```bash
bash eval/native/run_fastwam_libero.sh --gpus 0 --dry-run   # print the resolved run
bash eval/native/run_fastwam_libero.sh --gpus 0             # 50 trials/task
bash eval/native/run_fastwam_libero.sh --gpus 0 --smoke
```

Each run ends by writing `<output-dir>/summary.json` and printing one
`eval complete: ...` line.

## Order of work

First confirm that the unmodified source passes every gate:

```bash
cp geak/task/examples/fastwam_libero.config.yaml geak/task/config.yaml
mkdir -p geak/task/src
for FILE in mot.py wan_video_dit.py wan_video_vae.py action_dit.py; do
  cp "$MODEL_SRC_ROOT/src/fastwam/models/wan22/$FILE" "geak/task/src/$FILE"
done

bash geak/task/run_local.sh compile
bash geak/task/run_local.sh performance
bash geak/task/run_local.sh oracle-refresh
bash geak/task/run_local.sh correctness
```

Do not launch GEAK until correctness passes on the unmodified tree.

Then run the workflow:

```bash
# Establish the closed-loop baseline.
bash eval/native/run_fastwam_libero.sh --arm baseline --gpus 0 --smoke

# Compare eager and naive torch.compile.
python scripts/compare_artifacts.py --arms eager,compile

# Run the GEAK campaign described by geak/README.md.

# Gate the reduced denoise-step count on closed-loop accuracy.
bash eval/native/run_fastwam_libero.sh --arm geak --steps 2 --trials 5 --gpus 0

# Produce one interleaved comparison.
python scripts/compare_artifacts.py \
  --arms eager,compile,geak,geak@2
```

`compare_artifacts.py` times on the one GPU given by `--gpus` (default
`EVAL_GPUS`, else 0). `geak@2` is the GEAK arm at 2 denoise steps (the
former `geak-reduced` arm still works, with a deprecation warning). It times
every GEAK arm as the launcher evaluates it: with `FASTWAM_KVSLAB=0` and a
compile cache of its own.

Actual acceptance requires both the numerical oracle and the persistent-model
LIBERO rollout (`correctness`). A one-input oracle cannot detect cross-episode stale caches or
captured graph buffers.
