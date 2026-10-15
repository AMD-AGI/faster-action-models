# Side-by-side source trees

Each directory here is a **complete, standalone source tree** — the exact code
that was validated for one configuration. There is no patch series and no
env-flag multiplexing over a shared tree.

This is not a stylistic preference. The GEAK-optimized trees were *generated
separately* from their original code by distinct optimization campaigns. The
CAE-FastWAM `mot.py` is 6,895 lines produced from an 819-line original; the
FastWAM GEAK one is 4,201 lines from 587. These
cannot be merged into one file behind switches, because that is not how they
were produced and not what was measured.

The existing vendored `FastWAM/` and `DiT4DiT/` trees at the repository root are
**not modified by these trees**.

**No performance numbers are recorded in these trees.**

## FastWAM

| Directory | Configuration | `--arm` | Provenance |
|---|---|---|---|
| `fastwam-baseline/` | FastWAM baseline | `baseline` | Pristine upstream FastWAM (earlier upstream revision) |
| `fastwam-geak/` | FastWAM GEAK | `geak` | GEAK campaign output |
| `cae-fastwam/` | CAE-FastWAM | `cae` | GEAK-optimized Compact Action Expert inference tree |

CAE-FastWAM (Compact Action Expert: the 32-hidden-dimension Action DiT with the
K8 x d16 action VAE, GEAK-optimized) has about 30M action-expert parameters and
denoises in the latent space of the suite-conditioned action VAE.
`cae-fastwam/` is the single CAE-FastWAM tree; it is what `--arm cae` serves.

The FastWAM GEAK tree differs from the FastWAM baseline tree in exactly four
files, all under `fastwam/models/wan22/`: `action_dit.py`, `mot.py`,
`wan_video_dit.py`, `wan_video_vae.py`. This was verified with `diff -rq` on
the pair; there are no other differing source files and no source files
present on only one side (each tree's own `README.md` aside).

To measure CAE-FastWAM on your own machine, run
`scripts/cae_fastwam_latency.py`: it times CAE-FastWAM at 10 and 4 steps in one
interleaved session and checks each cell's shapes and step count before timing
it.

### Relationship to the vendored `FastWAM/src`

The vendored `FastWAM/src/fastwam` holds the unoptimized CAE-FastWAM model code,
used for training and as the reference the GEAK campaign started from. It is
*not* the FastWAM baseline tree (`fastwam-baseline/` is an earlier upstream
revision), and no `--arm` serves it: `src/cae-fastwam` is the tree `--arm cae`
serves, so no arm depends on the vendored tree staying put.

## DiT4DiT

| Directory | Configuration | Provenance |
|---|---|---|
| `dit4dit-baseline/` | upstream DiT4DiT (**pristine**); the comparison point | the upstream DiT4DiT release, source subset |
| `dit4dit-geak/` | GEAK | upstream plus earlier inference work, the GEAK `Cosmos25.py`, `ActionDiT.py` and `DiT4DiT.py`, a LIBERO client that takes a task range, training-side drift, and two diffusers patches in `patches/`; every difference is listed below |

Only the source subset was copied (`DiT4DiT/ examples/ deployment/ utils/ docs/`
plus four top-level files). Checkpoints, results, logs and virtualenvs are not
here.

### Which tree is the comparison point

The GEAK arm is compared against **`dit4dit-baseline/`**, the upstream
release (`--arm baseline`, the LIBERO launcher's default). Besides the GEAK
work, `dit4dit-geak/` carries earlier inference work that upstream lacks, some
of which reads no environment variable and is always on:

- `_geak_tail_elision_enabled` — read with a default of `True` and assigned
  nowhere in the tree.
- `_geak_elide_pad` — assigned unconditionally.
- `_build_cond_videos_gpu` — always used to build the conditioning video.
- an ActionDiT KV hoist.

So a speedup of `dit4dit-geak/` over `dit4dit-baseline/` includes that earlier
work as well as the GEAK edits; it is not a measure of the GEAK edits alone.

To measure the DiT4DiT arms on your own machine, run
`scripts/dit4dit_latency.py --bench libero|robocasa`: it times baseline, naive
`torch.compile` (the baseline tree and overlay) and GEAK at 4 steps and GEAK at
2 steps in one interleaved session, each arm served
from its own tree and diffusers overlay as the launchers serve it, and checks
each cell's tree, step count and shapes before timing it.

`diff -rq -x README.md -x patches dit4dit-baseline dit4dit-geak` lists 22
entries, which `tests/test_side_by_side_trees.py` pins:

- **Model files (3)**, which hold the earlier inference work above and the GEAK
  kernels and gates: `model/modules/vlm/Cosmos25.py`,
  `model/modules/action_model/ActionDiT.py`, `model/framework/DiT4DiT.py`.
- **Serving path (1 changed, 3 added)**: `model/modules/vlm/__init__.py`, and
  `model/modules/action_latent/`, `config/libero/cosmos25_libero_policy.yaml`
  and `deployment/model_server/server_policy_split.py` (see *Drift carried
  deliberately* below).
- **LIBERO client (2)**: `examples/LIBERO/eval_files/eval_libero.py` takes
  `--args.start-task`/`--args.end-task` and prints the totals, which is what
  lets `scripts/eval_dit4dit_libero.sh --clients-per-gpu K` run one task per
  client (on the baseline tree it falls back to one client per suite);
  `model2libero_interface.py` loads `share_tools` by file path, retries a
  failed server response three times instead of raising, and reads the action
  horizon and state dimension from `action_latent` configs.
- **Training-side drift (8)**, inert at inference (see below).
- **Upstream files the GEAK tree lacks (5)**: `server_policy_zmq.py`,
  `tools/zmq_policy_server.py`, `config/real_robot/dit4dit_g1_decoupled_wbc.yaml`,
  `examples/LIBERO/train_files/modality.json` and
  `examples/Real_G1/train_files/run_decoupled_wbc.sh` (plus
  `examples/Real_G1/README.md`). None is on the LIBERO or RoboCasa serving path.

`server_policy.py` and `websocket_policy_server.py` are identical in both trees.
`patches/` holds the two diffusers patches (see below).

### GEAK gates are pinned off, and that is not the baseline

The launchers export `GEAK_ACTION_COMPILE`, `GEAK_COSMOS_COMPILE`,
`GEAK_VAE_T1` and `GEAK_ACTION_BF16` as `0` on **every** arm, rather than
leaving any unset: an unset gate is a silent no-op, and a "measured" result
from a switch that never fired has happened before. `GEAK_VAE_T1` is read by no
file in any tree; it is exported so that an environment carrying an older
diffusers patch keyed on it stays off.

Turning the gates off does **not** turn `dit4dit-geak/` into the baseline. Most
of the GEAK work in it reads no switch. The baseline arm is a different
**tree**, which is the reason the side-by-side layout exists.

### Part of the GEAK arm lives in diffusers

The GEAK arm also modifies two files in the installed `diffusers` package
(0.37.0.dev0, commit 3996788): `models/autoencoders/autoencoder_kl_wan.py` and
`models/transformers/transformer_cosmos.py`. The patches are committed at
`dit4dit-geak/patches/`.

Because they live in the Python environment rather than in a tree, swapping
trees would not remove them. The launchers therefore never patch the installed
package. For every arm they copy diffusers into the run's output directory,
check that it is 0.37.0.dev0 and that both target files are byte-identical to
upstream 3996788, apply the patches only on the GEAK arm (and check the
result), and put the copy first on the server's `PYTHONPATH`. One model
environment serves every arm. The logic is in
`scripts/dit4dit_diffusers_overlay.sh`.

### Gradient checkpointing is training-only

The GEAK tree enables gradient checkpointing on the Cosmos transformer
only when the backbone is `trainable`; the upstream baseline tree never
enables it. Under `torch.inference_mode()`
checkpointed blocks raise "Inference tensors cannot be saved for backward", and
the published serving runs disabled checkpointing on every module after load.
The released LIBERO and RoboCasa checkpoints do not set `trainable`, so the
trees serve in the same state without a post-load hook.

### The step count is set from outside, so no tree drifts

DiT4DiT's denoise step count is the `num_inference_timesteps` **attribute** on
the action head, read by `ActionDiT.predict_action` in every tree, not the
`num_inference_steps=` kwarg, which the legacy branch drops. The launchers'
`--steps` writes that attribute at server start via
`scripts/dit4dit_serve_policy.py` and `scripts/dit4dit_step_override.py`, which
patch `baseframework.from_pretrained` from outside and assert the read-back.

No tree was modified to support this. Adding a server flag to
`*/deployment/model_server/server_policy.py` would put drift into the files
whose job is being byte-identical to the validated code, and a second argparse
surface is what produced the `--seed` mismatch the runner already has to grep
around. `server_policy.py` is identical in both trees and carries neither
flag; `tests/test_side_by_side_trees.py` pins that.

Editing `DiT4DiT/config/*/dit4dit_*.yaml` would also be futile: those copies
are inert at inference. `share_tools.py` resolves the config from the
**checkpoint's** own `run_dir/config.yaml`.

### The vendored `DiT4DiT/` is a fourth codebase

The vendored `DiT4DiT/` at the repository root is not the parent of any tree
here. It differs from upstream in `model/framework/DiT4DiT.py` and
`model/modules/vlm/Cosmos25.py` (a prompt-embedding cache, a causal-VAE reuse
path) and contains **zero** `GEAK_`/`_geak_` symbols. Pointing any arm at it
would compare unrelated codebases, so the arms resolve to `src/` instead. Its
RoboCasa full-24 harness and client are the repository's one RoboCasa
protocol: `scripts/eval_dit4dit_robocasa.sh` runs them for every arm, with
only the policy server taken from the arm's tree.

### Drift carried deliberately, and why

The GEAK tree's source subset was copied **verbatim** rather than
hand-assembled from a pristine base plus an overlay. Eight files in five groups
differ from upstream as training-side drift, not inference work:

| File | Change | Verdict |
|---|---|---|
| `DiT4DiT/config/deepseeds/ds_config.yaml` | `gradient_accumulation_steps: 1 → "auto"` | drift, DeepSpeed only |
| `DiT4DiT/dataloader/__init__.py` | `dist.is_initialized()` guards | drift |
| `DiT4DiT/dataloader/gr00t_lerobot/{data_config,datasets,mixtures,video}.py` | dist guards, removed a placeholder dataset entry | drift |
| `DiT4DiT/training/train.py` | trainer entry point | drift, never imported by the server |
| `DiT4DiT/training/trainer_utils/trainer_tools.py` | `dist.get_rank()` guards | drift; on the import path, but every changed line is inside `TrainerUtils` methods the eval path never calls |

They were **not dropped**: the tree's value is that it *is the exact code that
was measured*, and a hand-built pristine+overlay hybrid is a configuration that
has never been run. The drift is inert on the LIBERO and RoboCasa inference
paths — `model/framework/DiT4DiT.py` imports only `resize_images` from
`trainer_tools`, and neither `eval_libero.py` nor `model2libero_interface.py`
touches `dataloader`.

The GEAK tree also carries files upstream lacks, all of which are
load-bearing:

- `DiT4DiT/model/modules/action_latent/` — imported by
  `model/framework/DiT4DiT.py`.
- `DiT4DiT/config/libero/cosmos25_libero_policy.yaml`.
- `deployment/model_server/server_policy_split.py`.
- `model/modules/vlm/__init__.py` widens the accepted `training_mode` from
  `"joint"` to `("joint","policy","action","video")`; with the upstream copy a
  checkpoint trained in another mode raises `NotImplementedError`.

## What is NOT here

- No weights, checkpoints, or `__pycache__`. Verified: zero files matching
  `*.pt *.pth *.safetensors *.bin *.ckpt *.so`, zero symlinks.
- No RoboCasa full-24 harness in any DiT4DiT tree — it exists only in the
  vendored `DiT4DiT/`, and `scripts/eval_dit4dit_robocasa.sh` evaluates the
  side-by-side trees' policy servers with it; see
  `docs/side-by-side-source-trees.md`. The authors' own
  `examples/Robocasa_tabletop/eval_files/batch_eval_args.sh` is left in each
  tree, unchanged, and is not a protocol of this repository.
- No diffusers. The launchers build a per-run diffusers copy from the model
  environment (or `DIT4DIT_DIFFUSERS_SRC`); see above.
- No `pyoverlay`. The DiT4DiT runners need a small (about 5.8 MiB) overlay of
  third-party Python packages on `PYTHONPATH`. It is a runtime dependency, not
  source, and is passed via `DIT4DIT_PYOVERLAY`; see
  `docs/side-by-side-source-trees.md` for how to build it.
