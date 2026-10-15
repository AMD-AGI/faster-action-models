# Evaluation protocol — and why the harness defaults are wrong

**Read this before trusting any success rate.** The evaluation harness's LIBERO
defaults implement the *OpenVLA* protocol. A model whose paper used different
settings must override them, or the eval silently measures a different benchmark
than the paper reports.

Take the protocol from the model's **own evaluation code**, not from a
reproduction doc and not from the harness. For FastWAM that is
`configs/sim_libero.yaml` + `experiments/libero/eval_libero_single.py`.

## FastWAM vs the harness default

| parameter | FastWAM (its own eval code) | harness / OpenVLA default |
|---|---|---|
| `max_steps` libero_spatial | 400 | 220 |
| `max_steps` libero_object | 400 | 280 |
| `max_steps` libero_goal | 400 | 300 |
| `max_steps` libero_10 | 700 | 520 |
| `num_steps_wait` | 30 | 10 |
| `seed` | 42 | 7 |
| `num_trials` | 50 | 50 |

FastWAM's episode loop is `while t < max_steps + num_steps_wait`, so the wait
steps are **additional** to the horizon, not carved out of it.

Matching without change (verify for your model): `replan_steps` = the server's
action chunk size; `use_action_ensembler: false` = take the newest chunk with no
averaging; gripper binarization; `action_horizon = num_frames - 1`.

## `quat_no_antipodal` is load-bearing

Not present in the harness's canonical LIBERO config. It selects the robosuite
quaternion→axis-angle convention for the proprioceptive state. Without it the
policy receives orientation in a sign convention it never saw in training.

The failure is **silent and uniform** — the model runs, actions are finite, no
errors are raised, and success rate collapses across every suite.
It was caught only because two configurations that *cannot* perform identically
(different denoise-step counts, different source trees) returned byte-identical
per-suite numbers.

Set it explicitly:

```yaml
params:
  quat_no_antipodal: true
```

## Normalization

Actions are denormalized by the model's own normalizer, rebuilt from the
**released** `dataset_stats.json`. Passing a different checkpoint's stats
denormalizes into the wrong action range and costs success rate quietly. Always
pass the stats file that shipped with the checkpoint under evaluation.

## Sharding must be task-aligned

If the eval is sharded for throughput, use `shard_by_task: true`. The reference
protocol runs all trials of a task in **one environment, in order**; round-robin
sharding splits a task's trials across fresh environments and measures something
else. The harness documents ~2 percentage points of difference on
`libero_spatial` from this alone.

## A subset protocol, for gating rather than reporting

The full protocol is 4 suites x 10 tasks x 50 trials = 2000 episodes. That is too
slow to gate every candidate during an optimization campaign. For a per-candidate
gate, reduce **trials**, not protocol parameters:

* keep `max_steps`, `seed`, `num_steps_wait`, `quat_no_antipodal` exactly as above
* select a subset of tasks **by measured success rate**, not by convenience
* re-measure the threshold on your own tree; never inherit one

**Calibrate before choosing a threshold.** Run the unmodified tree across all
suites and drop any task the policy cannot do reliably. A gate containing a task
the model completes only part of the time will often fail its own baseline, and
every spurious failure costs an optimization direction. Equally, a suite the
policy does not solve reliably will often fail a multi-episode gate — exclude it rather than lowering the threshold, since a lowered threshold
also admits real single-episode regressions.

Retries help only for *stochastic* failures. Because a genuine regression is
deterministic, N independent attempts drive the false-failure rate down while a
broken candidate still fails all N. Print every attempt: a candidate that needs
the retry every time is a partial regression, not noise.
