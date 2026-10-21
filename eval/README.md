# Evaluation and metrics

Closed-loop LIBERO evaluation for FastWAM runs FastWAM's own evaluator
(`FastWAM/experiments/libero/eval_libero_single.py`) through
[`native/run_fastwam_libero.sh`](native/README.md); `scripts/eval_fastwam_libero.sh`
is a thin wrapper over it that defaults to one GPU. There is no policy server and
no websocket hop: each GPU in `--gpus` runs one persistent worker that loads the
model once and steps its own LIBERO environment. Read [`PROTOCOL.md`](PROTOCOL.md)
first: it lists the protocol this evaluator applies and why common LIBERO
evaluation defaults differ from it.

The launcher resolves the source tree and config root per arm (`baseline`
`src/fastwam-baseline`, `geak` `src/fastwam-geak`, `cae` `src/cae-fastwam`),
validates the flags, and applies each arm's compile settings. DiT4DiT runs through
its own launchers (`scripts/eval_dit4dit_libero.sh`, `scripts/eval_dit4dit_robocasa.sh`).

## Shape

The simulator is CPU-bound, so the work is spread over workers, not over model
servers. The launcher queues every (suite, task) of the selected suites (10 per
LIBERO suite) in FastWAM's file-backed worker pool
(`FastWAM/experiments/libero/worker_pool.py`); each worker pulls the next task,
runs all of its trials and writes `<output-dir>/<suite>/gpu<G>_task<T>_results.json`.

```
GPU 0 ── worker ── next (suite, task) from the queue, all trials, result file
GPU 1 ── worker ── ...
```

`--arm cae` runs one suite at a time (the Compact Action Expert's action VAE is
suite-conditioned and each suite has its own task config and statistics), so its
workers drain one suite's queue before the next suite starts.

With `--sim instinct --mode batched` (`baseline` and `geak`), each GPU instead
runs every trial of a task in lockstep on one N-world engine, one model call per
replan; see [`native/README.md`](native/README.md).

## Metrics

**Success rate** is the metric. It is deterministic given seed and config and
does **not** suffer the cross-node drift that latency does, so SR may be compared
across machines and sessions. Report:

* **pooled SR** over all episodes, weighted by episode count — not the mean of
  per-suite means, which silently reweights unequal suites;
* **per-suite SR**, because degradation is rarely uniform. Reduced denoise steps
  cost accuracy mainly on long-horizon and goal-conditioned suites while leaving
  object-centric suites untouched, and a pooled number hides that;
* **episode counts**, so the reader can size the error bar. At 2000 episodes the
  standard error is roughly 0.4 percentage points, so differences below ~1 point
  are not distinguishable.

Do **not** compare latency across nodes or sessions; see the note in the top-level
README.

## Running

From the repository root:

```bash
bash eval/native/run_fastwam_libero.sh --arm baseline --gpus 0              # full protocol
bash eval/native/run_fastwam_libero.sh --arm baseline --gpus 0 --smoke      # smoke: 1 trial/task
bash eval/native/run_fastwam_libero.sh --arm geak --steps 2 --gpus 0,1,2,3  # one worker per GPU
bash eval/native/run_fastwam_libero.sh --suites libero_spatial,libero_goal --gpus 0
bash eval/native/run_fastwam_libero.sh --arm geak --dry-run                 # print the resolved run
```

One launcher run is one config (arm x step count). It writes FastWAM's per-task
result files and rollout videos under `<output-dir>/<suite>/` and records the run
it holds in `<output-dir>/.run_identity` (arm, tree, checkpoint, steps, trials,
simulator, mode). A rerun into the same directory resumes it (finished tasks are
skipped); a directory holding results of a different run is refused rather than
pooled with them. The launcher ends by writing `<output-dir>/summary.json`
(pooled and per-suite successes and episodes) and printing one
`eval complete: ...` line; `eval/summarize.py` and `scripts/eval_summary.py` read
the same files.

## Always smoke-test before a full sweep

A full sweep is thousands of episodes. Run `--smoke` first and check the
result is *plausible* — a validated rig should reproduce the model's published
success rate to within sampling noise. A protocol or preprocessing mismatch (see
`PROTOCOL.md`) is invisible any other way, because it produces confident, wrong
numbers rather than errors.
