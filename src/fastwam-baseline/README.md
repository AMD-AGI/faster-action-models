# FastWAM baseline

The unmodified FastWAM model code (native 1024-wide Action DiT, raw actions),
as evaluated. It is the reference every FastWAM GEAK comparison is made against.

This is a **complete, standalone source tree**, like every other directory
under `src/`: put it on `PYTHONPATH` to select it. There are no env flags that
turn it into another arm.

## Run it

```bash
bash scripts/eval_fastwam_libero.sh --arm baseline --full
```

The baseline arm refuses `--steps` other than 10 unless
`--allow-baseline-steps` is passed; 10 is the released operating point.

## Relationship to the other trees

`../fastwam-geak/` differs from this tree in exactly four files, all under
`fastwam/models/wan22/`: `action_dit.py`, `mot.py`, `wan_video_dit.py`,
`wan_video_vae.py`. Everything else is byte-identical (`README.md` aside).

No performance numbers are recorded here; see `../README.md`.
