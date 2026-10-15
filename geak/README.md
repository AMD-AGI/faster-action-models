# Triggering GEAK on an action model

GEAK's `kernel_workflow` is a JS **Workflow**, launched from a Claude Code
session with the `Workflow` tool. It is not a CLI. It takes a *task directory*
that tells it what to edit, how to build, how to check correctness, and how to
measure — everything in `task/`.

```
Workflow({
  scriptPath: "<GEAK_ROOT>/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path:  "<abs path to your copy of geak/task>",
    workflow_dir: "<GEAK_ROOT>/kernel_workflow",
    exp_root:     "<output root>",
    budget: 10,          // optimization directions across all rounds
    deep_cost: 2,        // cost of one deep_explore direction
    gpu_ids: "0,1,2,3",  // one lane per GPU
    min_improve: 0.04,   // see "Set min_improve above YOUR noise floor"
    apply_to_original: "false",
    task: "<steer text>"
  }
})
```

`args` must be a real JSON object. If it arrives as a JSON-encoded string the
workflow aborts immediately.


## Quickstart: FastWAM on LIBERO / MI355X

A runnable instantiation lives in `task/examples/fastwam_libero.config.yaml` --
no placeholders. It deliberately ships **without** a dead-end list or measured
economics; those come from your own first campaign.

### Pin the toolchain

GEAK's `main` moves, and gfx950 support and the learned-knowledge format are both
version-dependent. Pin what you validate against:

```bash
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK && git checkout fa86fd1     # validated for this workflow on gfx950
```

The harness expects a container with **torch 2.11 + ROCm 7.14** (HIP 7.14,
triton 3.7.1) plus LIBERO/MuJoCo, and the FastWAM sources importable. Any image
satisfying that works; `EVAL_IMAGE` selects it.

### Prerequisites

```bash
export MODEL_SRC_ROOT="<FastWAM checkout>"          # holds src/ and configs/
export MODEL_CKPT="<...>/libero_uncond_2cam224.pt"
export DATASET_STATS="<...>/libero_uncond_2cam224_dataset_stats.json"
export EVAL_HARNESS="<vla-evaluation-harness checkout>"
export LIBERO_ROOT="<LIBERO checkout>"
export LIBERO_CONFIG_PATH="$HOME/.libero"
export HF_CACHE_DIR="<hf cache>"
export MODEL_PYTHON="/opt/venv/bin/python3"
export EVAL_IMAGE="<docker image>"
export GEAK_ROOT="<GEAK checkout, pinned above>"
```

`run_local.sh` bind-mounts these host paths at their absolute locations. They do
not need to live under `/mnt` or `/tmp`. Comparison runs may additionally set
`FWE_SRC` to a complete import root containing `fastwam/`, and with it
`FWE_CONFIG_DIR` to the config tree that source takes (default
`MODEL_SRC_ROOT/configs`; `scripts/compare_artifacts.py` sets
`FastWAM/configs_baseline` for the FastWAM baseline and GEAK trees); ordinary GEAK runs leave both
unset so `MODEL_SRC_ROOT/src` is copied and `task/src` is overlaid.

### Set up and verify the task

```bash
cp task/examples/fastwam_libero.config.yaml task/config.yaml
mkdir -p task/src
for f in mot.py wan_video_dit.py wan_video_vae.py action_dit.py; do
  cp "$MODEL_SRC_ROOT/src/fastwam/models/wan22/$f" task/src/$f
done

cd task
bash run_local.sh compile        # imports resolve through the overlay
bash run_local.sh performance    # establishes YOUR baseline
bash run_local.sh oracle-refresh # calibrates the numerics pre-filter
bash run_local.sh correctness    # MUST pass on the unmodified tree
```

**Do not launch until `correctness` passes on the unmodified tree.** A gate that
rejects its own baseline will reject good patches all campaign, and the run will
look like the model has no headroom when in fact the gate is broken. This is the
single most expensive mistake available here.

### Launch

```
Workflow({
  scriptPath: "<GEAK_ROOT>/kernel_workflow/kernel_workflow.js",
  args: {
    kernel_path:  "<abs path to geak/task>",
    workflow_dir: "<GEAK_ROOT>/kernel_workflow",
    exp_root:     "<output root>",
    budget: 24, deep_cost: 2,
    gpu_ids: "0,1,2,3",
    min_improve: 0.04,          // raise above YOUR measured noise floor
    apply_to_original: "false",
    task: "<the instructions block from config.yaml, or a shorter steer>"
  }
})
```

### What to expect

GEAK ships ~140 gfx950 knowledge cards, and those alone carry a campaign a long
way: in our runs the single largest improvement came from a campaign that cited
**five upstream cards and none of its own**. You do not need anyone else's
accumulated knowledge to get a substantial result.

Returns compress across successive campaigns as the easy levers are spent, and a
campaign is far more productive when it is told what the previous one already
tried. So: after your first campaign, fill in the four `FILL THESE IN` sections
at the bottom of the config from its report, and re-run seeded from its winner.
Each campaign should re-profile rather than inherit the previous one's framing --
the economics shift as work lands, and the stage that mattered last time may not
be the one that matters now.

## The task directory

`task/` is the contract. Copy it, point it at your model, and fill in the four
commands GEAK will call:

| file | role |
|---|---|
| `config.yaml` | editable files, target functions, the three commands, and the steer |
| `run_local.sh` | container wrapper — every command runs inside your image |
| `scripts/task_runner.py` | implements `compile` / `oracle` / `correctness` / `performance` |
| `scripts/episode_perf.py` | the scored latency measurement |
| `scripts/episode_oracle.py` | cheap numerics pre-filter |

### Choose the editable surface deliberately

`source_file_path` decides what GEAK is *able* to find. On FastWAM the single
largest win of the best campaign came from a file an earlier, narrower task
definition could not edit — the optimizer never saw it. For an action model the
surface usually needs to span:

* the **denoise/flow-matching loop**,
* the **backbone or video expert**,
* the **VAE / observation encoder**,
* the **inference entry point** itself.

Some of these may live in site-packages (e.g. a `diffusers` autoencoder). If so,
overlay a *copy* of that package and put it first on `PYTHONPATH` — and then
**assert in the compile gate that the module resolved to your overlay**. Without
that assertion an edit to a shadowed file is a silent no-op and the whole
campaign optimizes nothing while looking healthy.

## The three commands

GEAK calls exactly these, and trusts what they print:

* **`compile`** — overlay the edited files onto a pristine tree and import them.
  Fast rejection of syntax and API breakage. Also the right place for the
  overlay-shadowing assertion above.
* **`correctness`** — the authority. See `../eval/README.md`. A numerics oracle
  alone is *not* sufficient: it replays one input and is structurally blind to
  cross-episode state staleness, which is exactly what caching and graph capture
  introduce.
* **`performance`** — the scored metric. Print one line per case plus a geomean.

## Writing the steer

The `task` string is the highest-leverage input. What worked:

* **State the bar as an absolute, not a ratio.** If prior work already exists,
  a campaign starting from scratch can report a large ratio while still being
  slower than what is already on disk.
* **Give the measured economics.** "1 ms of prologue is worth Ax, 1 ms/step of
  denoise is worth Bx" redirects effort better than any qualitative hint — and
  recompute it each campaign, because the balance shifts as work lands.
* **List dead ends with their mechanism**, not just their name. "Attempted, lost"
  invites a retry; "lost because the region is at N% of roofline and the upcast
  eats the byte saving" does not.
* **Say what is already in the tree** so it is not re-derived.
* **Demand kill-switches, and demand the OFF arm be run.** A switch whose default
  was never tested is not a switch — one campaign lost a direction to a kill
  switch that shipped defaulting ON while its report claimed OFF.

## Set `min_improve` above YOUR noise floor

Measure it before choosing: run the unmodified tree twice and take the spread of
the scored metric. `min_improve` must sit above that, or rounds will accept
noise as winners. Do not inherit a value from another node or another model.

## After the campaign

1. **Re-measure the winner yourself**, paired and interleaved
   (`for round: for arm:`) on one quiet GPU. Self-reported campaign numbers ran
   consistently optimistic in our runs.
2. **Re-run the correctness gate** on the final tree.
3. **Drain the learned-knowledge card.** GEAK writes a proposal into
   `kernel_workflow/knowledge/learned/_inbox/` as raw JSON that the *next*
   campaign cannot read until drained:
   ```bash
   python3 "$GEAK_ROOT/kernel_workflow/scripts/kb.py" \
     --kb-dir "$GEAK_ROOT/kernel_workflow/knowledge/learned" drain --apply
   ```
   Check `_inbox/` for undrained `*.json` after every campaign. A card left
   there is a lesson the next run silently does not inherit.
