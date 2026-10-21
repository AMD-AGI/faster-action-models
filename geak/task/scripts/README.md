# Task harness

> **These three `.py` files are a WORKED REFERENCE, not a generic template.**
> They target FastWAM/LIBERO's module names and observation shapes. Paths are
> supplied through the environment documented in
> `docs/fastwam-geak-inference-optimization.md`; adapt the model-specific API and
> shapes before using the harness with another model.

`task_runner.py` implements the commands GEAK calls (and `rollout`, for manual probing). It is intentionally
thin: it overlays the edited files onto a pristine tree, then shells out to the
measurement and gate scripts.

| command | role |
|---|---|
| `compile` | overlay + import. Also asserts the overlay actually shadows any package it is meant to replace — without that, an edit to a shadowed file is a silent no-op. |
| `performance` | the scored metric (`episode_perf.py`) |
| `oracle` | cheap numerics pre-filter (`episode_oracle.py`) — NOT the authority |
| `oracle-refresh` | re-record the oracle golden, calibrated cross-process |
| `rollout` | the closed-loop rollout alone (no oracle), for manual probing |
| `correctness` | oracle, then the closed-loop rollout — the rollout decides. The rollout also fails if any activation marker the candidate's source emits (`ACTIVATION_MARKERS` in `task_runner.py`) is missing from the evaluator worker's log, or a graph capture fell back, so a candidate cannot pass on the unoptimized fallback path. |

## Two things that are easy to get wrong

**The oracle tolerance must be calibrated ACROSS processes.** Measuring
self-consistency *within* one process can report exact determinism and then fail
the unmodified tree, because the golden is written to disk and always compared
across a process boundary — where bf16 outputs can differ by 1–2 ULP. Record
several goldens in separate processes and take the worst pairwise disagreement.

**`max|d|` alone cannot gate bf16 outputs.** Cross-process noise and a real ~1%
error can be one ULP apart on max-abs, so no threshold separates them. Gate on
`mean|d|` as well: ULP noise moves a handful of elements while a systematic error
moves every element, which separates them by roughly an order of magnitude.
