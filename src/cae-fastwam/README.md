# CAE-FastWAM

The model code for CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension
Action DiT with the K8 x d16 action VAE, GEAK-optimized), as evaluated. This is
the inference tree `--arm cae` serves.

This is a **complete, standalone source tree**, like every other directory
under `src/`: put it on `PYTHONPATH` to select it.

## Run it

```bash
bash scripts/eval_fastwam_libero.sh --arm cae --full
bash scripts/eval_fastwam_libero.sh --arm cae --steps 4 --full
```

It needs per-suite `dataset_stats` and one server per suite;
`eval/run_eval_cae_fastwam.sh` implements that.

## What GEAK changed

Exactly four files, all under `fastwam/models/wan22/`: `action_dit.py`,
`mot.py`, `wan_video_dit.py`, `wan_video_vae.py`. Everything else is the same
as the unoptimized CAE-FastWAM code in the vendored `FastWAM/src/fastwam`. The
GEAK `mot.py` is 6,895 lines, produced from the 819-line original by a
separate optimization campaign from the FastWAM GEAK one.

No performance numbers are recorded here; see `../README.md`.
