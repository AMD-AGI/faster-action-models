# Recording rollout videos

The qualitative rollout figures are ordinary evaluation episodes with video
recording switched on. Nothing about the policy changes. You run the same
launcher and arm as for the success-rate cell, then pick the episodes to show
from the per-episode output. Recording costs wall-clock time (frames are
encoded on the client), so leave it off for timed or full-protocol runs.

| Model / benchmark | Launcher | Switch | Output | File name |
|---|---|---|---|---|
| DiT4DiT / LIBERO | `scripts/eval_dit4dit_libero.sh` | always on (`--record-video` accepted, no-op) | `<output-dir>/videos/<suite>[_task<T>]/` | `rollout_<task>_episode<i>_{success,failure}.mp4`, 10 fps |
| DiT4DiT / RoboCasa-GR1 full-24 (every arm) | `scripts/eval_dit4dit_robocasa.sh` | `--record-video` | `<output-dir>/videos/<task>/` | `<uuid>_success{0,1}.mp4`, 10 fps |
| FastWAM / LIBERO (all arms) | `scripts/eval_fastwam_libero.sh` | `--record-video` | `<output-dir>/<cell>/videos/<suite>/` | `task<N>_ep<NNN>.mp4`, 20 fps, agent view |
| FastWAM / LIBERO, native evaluator (all arms) | `eval/native/run_fastwam_libero.sh --mode serial` | always on (`--record-video` accepted, no-op; `--mode batched` writes none and refuses it) | `<output-dir>/<suite>/videos/` | `<time>--episode=task<T>_trial<i>--success=<bool>--task=<description>.mp4`, 24 fps |

`<cell>` is the engine's result directory: `<arm>_<N>step` (`baseline_10step`,
`geak_2step`), and `cae_<N>step` for `--arm cae`.

For the DiT4DiT launchers and the native FastWAM evaluator the outcome is part
of the file name, so selecting a success or a failure is a `ls`.

## FastWAM harness specifics

`--record-video` sets `save_video_dir` on the evaluation harness's LIBERO
benchmark. Not every harness revision has that parameter. `eval/run_eval.sh`
and `eval/run_eval_cae_fastwam.sh` check for it and stop with exit code 2 instead of
silently recording nothing. If `imageio` is missing from the client image,
the harness writes `.npz` frame stacks (`frames`, `[T, H, W, 3]` uint8)
instead of `.mp4`.

The file name carries no outcome. Also, **do not trust the `task<N>` in it**:
`<NNN>` is a per-client running counter, and when one shard runs several tasks
the task id can be stale. Map files to episodes through the shard logs instead:

1. `shard<k>.log` in the arm's output directory prints one
   `[VID] wrote <path> (<frames> frames)` line per episode, in the order the
   episodes ran.
2. `LIBEROBenchmark_<suite>_shard<k>of<n>.json` lists the same episodes, in
   the same order, under `tasks[].episodes[]`, with `metrics.success` and
   `steps`.
3. Pair them in order. The frame count equals the episode's `steps`, which
   confirms the pairing.

Example, the CAE-FastWAM cell on one suite with videos:

```bash
bash scripts/eval_fastwam_libero.sh --arm cae \
    --suites libero_spatial --record-video
```

## Reproducing a figure

1. Run the cell the figure illustrates, with the arm and step count from its
   caption, and recording on. A subset (`--smoke`, or `--suites` on the
   LIBERO launchers) is enough: the figure needs episodes, not a success rate.
2. Pick the episodes: successes for "what the policy does", failures for the
   failure-mode panels.
3. Side-by-side baseline vs GEAK panels on LIBERO: run both arms on the same
   suite. Episode *i* of a task starts from LIBERO's fixed initial state *i*
   in both the DiT4DiT and the FastWAM clients, so the same index in the two
   arms is the same scene. The RoboCasa full-24 clients reset episode *i* of a
   task to the same derived seed in every arm, so its scenes match too; the
   video files carry no episode index, so pair them by task, or, with one
   client per GPU, by write order within a task.
