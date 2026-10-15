# Eval harness and metrics

Closed-loop evaluation for an action model, built on
[vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness).
Read [`PROTOCOL.md`](PROTOCOL.md) first — the harness defaults are not
necessarily your model's protocol.

This directory holds the engines. Users run them through
`scripts/eval_fastwam_libero.sh` (`--arm baseline|geak` drives `run_eval.sh`,
`--arm cae` drives `run_eval_cae_fastwam.sh`), which resolves the source
tree and config root, validates the flags, and applies the GEAK arm's compile
settings. The engines call that arm `optimized`; the launcher calls it `geak`
and names its result directory after it.

## Shape

A **model server** holds the policy on one GPU; **eval clients** run the
simulator and call it over a websocket. The simulator is CPU-bound, so several
clients can share one server.

```
GPU 0 ── model server ──┬── shard 0 (tasks 0, 4, 8 ...)
                        ├── shard 1
                        └── shard N
```

### Assign one *config* per GPU, not one *suite* per GPU

The obvious layout — a suite per GPU — is badly imbalanced, because suites have
very different horizons (LIBERO-10 is ~1.75x the others). Every config then waits
on its slowest suite before the next can start.

Giving each GPU a whole **config** (arm x step count) makes every GPU do
identical work, so the only remaining imbalance is the intrinsic cost of the
config itself, and all configs start together:

```
GPU 0  baseline, 10 steps
GPU 1  geak,     10 steps
GPU 2  geak,      2 steps
```

Pick the shard count to divide the tasks evenly. With 10 tasks per suite, 10
shards gives exactly one task per shard per suite; 4 shards gives 3/3/2/2 and the
config waits on its slowest shard.

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
bash scripts/eval_fastwam_libero.sh --arm baseline            # full protocol
bash scripts/eval_fastwam_libero.sh --arm baseline --smoke    # smoke: 1 trial/task
bash scripts/eval_fastwam_libero.sh --arm geak --steps 2      # one arm and step count
bash scripts/eval_fastwam_libero.sh --suites libero_spatial,libero_goal
bash scripts/eval_fastwam_libero.sh --arm geak --dry-run      # print the resolved run
```

One launcher run is one config (arm x step count); run one per GPU (`--gpus`)
for the layout above. Each writes into `<output-dir>/<arm>_<N>step/` (`cae_<N>step/` for
`--arm cae`) and records the run it holds there
(`.run_identity`: tree, checkpoint, steps, trials, seed, shard count). A rerun
into a directory holding results of a different run is refused rather than
pooled with them. The launcher ends by writing `<output-dir>/summary.json`
(pooled and per-suite successes and episodes).

## Always smoke-test before a full sweep

A full sweep is thousands of episodes. Run `--smoke` first and check the
result is *plausible* — a validated rig should reproduce the model's published
success rate to within sampling noise. Two classes of bug are invisible any other
way, because both produce confident, wrong numbers rather than errors:

* a protocol or preprocessing mismatch (see `PROTOCOL.md`), and
* a **stale model server** from an earlier run still holding the port, so the
  clients silently evaluate the wrong arm under this run's label.

`run_eval.sh` guards the second by refusing to start when its port is already
bound. Note that killing a `docker run` client does *not* stop the container or
free its port — remove the container.
