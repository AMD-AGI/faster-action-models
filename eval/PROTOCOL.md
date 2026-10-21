# Evaluation protocol

**Read this before trusting any success rate.** This repository evaluates FastWAM
with FastWAM's **own evaluation code** (`FastWAM/configs/sim_libero.yaml` +
`FastWAM/experiments/libero/eval_libero_single.py`, run by
[`native/run_fastwam_libero.sh`](native/README.md)), which applies the protocol
below. Common LIBERO evaluation stacks default to the *OpenVLA* protocol instead
(for example the LIBERO defaults of
[vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness)); a
model whose paper used different settings must override those, or the eval
silently measures a different benchmark than the paper reports.

Take the protocol from the model's own evaluation code, not from a reproduction
doc and not from a general-purpose evaluation stack.

## FastWAM vs the common LIBERO default

| parameter | FastWAM (its own eval code) | common LIBERO / OpenVLA default |
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

Matching without change (verify for your model): `replan_steps` = the policy's
action chunk size; `use_action_ensembler: false` = take the newest chunk with no
averaging; gripper binarization; `action_horizon = num_frames - 1`.

## The quaternion convention is load-bearing

FastWAM's evaluator converts the end-effector quaternion to axis-angle with
robosuite's own `quat2axisangle` (copied into
`FastWAM/experiments/libero/libero_utils.py`). Another evaluation stack must use
the same convention for the proprioceptive state (vla-evaluation-harness, for one,
selects it with `quat_no_antipodal: true`, which its canonical LIBERO config does
not set). Without it the policy receives orientation in a sign convention it never
saw in training.

The failure is **silent and uniform** — the model runs, actions are finite, no
errors are raised, and success rate collapses across every suite.
It was caught only because two configurations that *cannot* perform identically
(different denoise-step counts, different source trees) returned byte-identical
per-suite numbers.

## Normalization

Actions are denormalized by the model's own normalizer, rebuilt from the
**released** `dataset_stats.json`. Passing a different checkpoint's stats
denormalizes into the wrong action range and costs success rate quietly. Always
pass the stats file that shipped with the checkpoint under evaluation.

## Work must be split by task

The reference protocol runs all trials of a task in **one environment, in order**.
The native launcher's workers take whole tasks from the queue, so it keeps that.
Splitting a task's trials round-robin across fresh environments measures something
else; vla-evaluation-harness documents ~2 percentage points of difference on
`libero_spatial` from this alone.

## A subset protocol, for gating rather than reporting

The full protocol is 4 suites x 10 tasks x 50 trials = 2000 episodes. That is too
slow to gate every candidate during an optimization campaign. For a per-candidate
gate, reduce **trials**, not protocol parameters:

* keep `max_steps`, `seed`, `num_steps_wait` and the quaternion convention exactly as above
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
