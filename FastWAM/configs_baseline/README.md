# FastWAM baseline config tree

The Hydra config root the FastWAM baseline and GEAK source trees
(`src/fastwam-baseline`, `src/fastwam-geak`) were evaluated with. It is upstream FastWAM's `configs/` at
the revision those trees were taken from, plus three inert action-ensembler
keys in `sim_libero.yaml` (`use_action_ensembler` stays `false`).

It is kept apart from `../configs` because that tree follows a newer upstream
schema: its model configs carry keys such as `compile_training_denoise`, which
the baseline `create_fastwam` rejects, and a different action-scheduler shift.
`scripts/eval_fastwam_libero.sh --arm baseline|geak` uses this tree by default
(`--arm cae` uses `../configs`).
