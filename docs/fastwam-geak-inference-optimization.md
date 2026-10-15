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

No performance numbers are committed here. Latency changes across nodes,
drivers, and sessions; run the comparison on the deployment machine.

## Layout

- [`../geak/README.md`](../geak/README.md) describes the GEAK campaign.
- [`../geak/task/`](../geak/task/) contains compile, correctness, and
  performance gates.
- [`../eval/README.md`](../eval/README.md) describes closed-loop evaluation.
- [`../eval/PROTOCOL.md`](../eval/PROTOCOL.md) defines the LIBERO protocol.
- [`../scripts/compare_artifacts.py`](../scripts/compare_artifacts.py) compares
  eager, compile, GEAK, and reduced-step sources in one interleaved run.

## Prerequisites

The repository includes FastWAM at `FastWAM/`. The following paths may point
elsewhere, but they must be visible inside the selected container.

For the evaluation launcher:

```bash
export MODEL_CKPT="<path-to-checkpoint>"
export DATASET_STATS="<path-to-dataset-stats.json>"
export EVAL_HARNESS="<path-to-vla-evaluation-harness>"
export LIBERO_ROOT="<path-to-LIBERO>"
export LIBERO_CONFIG_PATH="${HOME}/.libero"
export EVAL_IMAGE="<docker-image-with-torch-rocm-and-libero>"
export EVAL_OUTPUT_ROOT="<output-directory>"
```

`scripts/eval_fastwam_libero.sh` resolves the source tree from `--arm`
(`src/fastwam-baseline` or `src/fastwam-geak`) and their config tree
(`FastWAM/configs_baseline`) by itself, so it needs no `MODEL_CONFIGS`,
`BASELINE_SRC` or `OPTIMIZED_SRC`. Never point them at `FastWAM/configs` or
`FastWAM/src` for a baseline or GEAK run: those hold the CAE-FastWAM config
schema and its unoptimized code, and the FastWAM baseline and GEAK trees reject
that schema. The launcher honours the variables when set, with a warning
whenever they differ from what `--arm` selects.

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

These two source variables name exactly the trees the launcher resolves for
`--arm baseline` and `--arm geak`, so exporting them does not change a baseline
or GEAK evaluation. An `--arm cae` run ignores both, with a warning, and serves
the CAE-FastWAM tree (`src/cae-fastwam`; `CAE_SRC` overrides it). `MODEL_PYTHON` is the GEAK task's variable;
`scripts/eval_dit4dit_libero.sh` no longer reads it without a deprecation
warning (it takes `DIT4DIT_MODEL_PYTHON` or `DIT4DIT_PYTHON`).

All filesystem path variables above must be absolute. `geak/task/run_local.sh`
and the evaluation engines mount host paths into their containers at the same
locations, including paths outside `/mnt` and `/tmp`. `MODEL_PYTHON` is the
in-container interpreter path of the GEAK task and does not need to exist on
the host. The comparison tool forwards its selected source as `FWE_SRC`; that
directory must directly contain the importable `fastwam/` package. For a
FastWAM baseline or GEAK tree it also forwards `FWE_CONFIG_DIR` as
`FastWAM/configs_baseline`, the config schema those trees take; set
`FWE_CONFIG_DIR` yourself to time another tree with other configs. When
`FWE_SRC` is unset, the GEAK task copies `MODEL_SRC_ROOT/src` and overlays the
editable files from `geak/task/src`.

Pin the validated GEAK revision:

```bash
git -C "${GEAK_ROOT}" checkout fa86fd1
```

The evaluation container should provide PyTorch, ROCm, Triton, LIBERO, and MuJoCo.
Do not install the CUDA-specific dependencies from `FastWAM/pyproject.toml`
into the ROCm container.

With the launcher variables above set, the standalone baseline launcher is:

```bash
bash scripts/eval_fastwam_libero.sh --dry-run   # print the resolved run
bash scripts/eval_fastwam_libero.sh             # 50 trials/task
bash scripts/eval_fastwam_libero.sh --smoke
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
bash scripts/eval_fastwam_libero.sh --arm baseline --smoke

# Compare eager and naive torch.compile.
python scripts/compare_artifacts.py --arms eager,compile

# Run the GEAK campaign described by geak/README.md.

# Gate the reduced denoise-step count on closed-loop accuracy.
bash scripts/eval_fastwam_libero.sh --arm geak --steps 2 --trials 5

# Produce one interleaved comparison.
python scripts/compare_artifacts.py \
  --arms eager,compile,geak,geak@2
```

`compare_artifacts.py` times on the one GPU given by `--gpus` (default
`EVAL_GPUS`, else 0). `geak@2` is the GEAK arm at 2 denoise steps (the
former `geak-reduced` arm still works, with a deprecation warning). It times
every GEAK arm as the launcher evaluates it: with `FASTWAM_KVSLAB=0` and a
compile cache of its own.

Actual acceptance requires both the numerical oracle and the persistent-server
LIBERO rollout. A one-input oracle cannot detect cross-episode stale caches or
captured graph buffers.
