# FastWAM GEAK

The GEAK-optimized FastWAM model code for the FastWAM baseline model (native
1024-wide Action DiT, raw actions), as evaluated. Same weights and same checkpoint as
`../fastwam-baseline/`; only inference code changes.

This is a **complete, standalone source tree**, like every other directory
under `src/`: put it on `PYTHONPATH` to select it.

## Run it

```bash
bash scripts/eval_fastwam_libero.sh --arm geak --full
bash scripts/eval_fastwam_libero.sh --arm geak --steps 2 --full
```

## What differs from the baseline

Exactly four files, all under `fastwam/models/wan22/`: `action_dit.py`,
`mot.py`, `wan_video_dit.py`, `wan_video_vae.py`. The GEAK `mot.py` is 4,201
lines, produced from the 587-line original by the optimization campaign; it
was generated separately, not written as switches over the original, which is
why it ships as its own tree.

The GEAK stack includes `torch.compile`. When quoting a speedup, say which arm
it is measured against: over the eager baseline and over a `torch.compile`d
baseline are different numbers. `scripts/compare_artifacts.py` measures the
arms interleaved on your machine.

No performance numbers are recorded here; see `../README.md`.
