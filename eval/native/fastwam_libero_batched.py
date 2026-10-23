# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched FastWAM LIBERO eval on the GPU simulator: all trials of a task in lockstep, one model call per replan.

FastWAM's own evaluator (FastWAM/experiments/libero/eval_libero_single.py) runs one episode at a time. This
driver runs the SAME episode loop -- reset, set_init_state, num_steps_wait dummy steps, replan every
replan_steps from the newest chunk, success = LIBERO's done flag, horizon max_steps + num_steps_wait -- for N
trials of one task at once: N worlds of one GPU engine (instinct_sim_libero.HipLiberoVectorEnv) and one
batched model call per replan (fastwam_batched_infer.infer_action_multi). Everything model-side is taken from
eval_libero_single itself: config composition, model/checkpoint/processor loading, observation preprocessing
(_obs_to_model_input), denormalization and the gripper convention.

    python3 eval/native/fastwam_libero_batched.py --config-dir FastWAM/configs_baseline \
        --task libero_uncond_2cam224_1e-4 --ckpt CKPT --dataset-stats STATS --steps 10 \
        --tasks libero_spatial:0,libero_spatial:1 --output-dir OUT [--num-envs 50 --pad-batch 64]

The arm's source tree must come first on PYTHONPATH (its `fastwam` package is what gets imported). Writes
OUT/<suite>/gpu<G>_task<T>_results.json in eval_libero_single's format, so its summarize_results.py applies.

Differences from the serial evaluator, both deliberate:
* each slot has its own LIBERO env, so fixture placements are different draws from the same distribution than
  one env's sequential resets: scenes are not paired; equivalent Success rate requires measurement;
* the model batch is padded to --pad-batch by repeating the first row (GEAK trees capture graphs for a fixed
  batch; 64 is the validated size). Padding rows never reach a simulator.
Rollout videos are not written.

CAE-FastWAM (src/cae-fastwam, an action-VAE model) runs with --model-batch 1: the N worlds still step in lockstep,
but each observation that needs a new chunk gets its own B=1 call of the serial evaluator's
_predict_action_chunk (its GEAK kernels and compiled graphs are single-batch). Each model call is therefore the
serial one -- compile_action_infer, the latent noise, the action-VAE decode, denormalization, gripper handling
-- and only the fixture draws differ, as above. The action-VAE model is per suite: the suite id is set from
EVALUATION.task_suite_name before the checkpoint loads, as the serial evaluator does, and every --tasks item must
belong to that suite (the launcher runs one suite per process).

    python3 eval/native/fastwam_libero_batched.py --config-dir FastWAM/configs \
        --task <TASK_CONFIG>_libero_spatial --ckpt CKPT --dataset-stats STATS_SPATIAL --steps 4 --model-batch 1 \
        --override EVALUATION.task_suite_name=libero_spatial --tasks libero_spatial:0,libero_spatial:1 --output-dir OUT
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "FastWAM"))          # experiments.libero.* (the evaluator this driver mirrors)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

import fastwam  # noqa: E402
from experiments.libero import eval_libero_single as EL  # noqa: E402  (registers its OmegaConf resolvers)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION, get_libero_dummy_action, invert_gripper_action)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from fastwam_batched_infer import infer_action_multi  # noqa: E402

SUITE_IDS = {"libero_spatial": 0, "libero_object": 1, "libero_goal": 2, "libero_10": 3}


def compose_cfg(a):
    overrides = [f"task={a.task}", *a.override]
    with initialize_config_dir(config_dir=str(Path(a.config_dir).resolve()), version_base="1.3"):
        cfg = compose(config_name="sim_libero", overrides=overrides)
    cfg.ckpt = a.ckpt
    cfg.seed = a.seed
    cfg.EVALUATION.dataset_stats_path = a.dataset_stats
    cfg.EVALUATION.num_inference_steps = a.steps
    cfg.EVALUATION.num_trials = a.trials
    return cfg


def refuse_unbatched_features(cfg, model, per_obs=False):
    """Configurations whose per-observation control flow this driver does not implement.

    per_obs (--model-batch 1): every call is the serial evaluator's own _predict_action_chunk, which handles
    compile_action_infer, text CFG, action_infer_mode and the action VAE itself; only the features that carry
    state across calls (future-video visualisation, the action ensembler) stay refused."""
    ev = cfg.EVALUATION
    bad = []
    if bool(ev.get("visualize_future_video", False)):
        bad.append("visualize_future_video")
    if bool(ev.get("use_action_ensembler", False)):
        bad.append("use_action_ensembler")
    if not per_obs:
        if bool(ev.get("compile_action_infer", False)):
            bad.append("compile_action_infer")
        if float(ev.get("text_cfg_scale", 1.0)) != 1.0:
            bad.append("text_cfg_scale != 1")
        if getattr(model, "action_vae", None) is not None:
            bad.append("Compact Action Expert (action VAE) at a padded batch (use --model-batch 1)")
        if "action_infer_mode" in inspect.signature(model.infer_action).parameters and \
                str(ev.get("action_infer_mode", "idm")) != "idm":
            bad.append("action_infer_mode != idm")
    if bad:
        raise SystemExit(f"batched driver does not implement: {', '.join(bad)}; use the serial native eval")


def parse_tasks(csv):
    """--tasks CSV of suite:task_id -> [(suite, task_id)]."""
    items = []
    for item in csv.split(","):
        suite, tid = item.split(":")
        if suite not in SUITE_IDS:
            raise SystemExit(f"unknown suite {suite}")
        items.append((suite, int(tid)))
    return items


def set_action_vae_suite(model, cfg):
    """The serial evaluator's suite id (eval_libero_single.eval_single_process): an action-VAE model conditioned on
    the suite gets SUITE_IDS[EVALUATION.task_suite_name], before the checkpoint loads. A task config that names
    another suite id is refused rather than silently overridden."""
    vae = getattr(model, "action_vae", None)
    if vae is None or int(vae.cfg.num_suites) <= 0:
        return None
    suite = str(cfg.EVALUATION.task_suite_name)
    if suite not in SUITE_IDS:
        raise SystemExit(f"no action-VAE suite id for EVALUATION.task_suite_name={suite}")
    sid = SUITE_IDS[suite]
    configured = cfg.model.get("action_vae_suite_id", None)
    if configured is not None and int(configured) != sid:
        raise SystemExit(f"task config sets model.action_vae_suite_id={configured}, but "
                         f"EVALUATION.task_suite_name={suite} is suite id {sid}; use that suite's task config")
    model.action_vae_suite_id = sid
    return sid


def check_task_suites(items, cfg, model):
    """An action-VAE model and its statistics are per suite: every task must be from EVALUATION.task_suite_name."""
    if getattr(model, "action_vae", None) is None:
        return
    suite = str(cfg.EVALUATION.task_suite_name)
    other = sorted({s for s, _ in items if s != suite})
    if other:
        raise SystemExit(f"the action-VAE model is loaded for {suite} (EVALUATION.task_suite_name); --tasks has "
                         f"{', '.join(other)}: run one suite per process")


def load_model(cfg, items=()):
    """eval_libero_single.eval_single_process's model/processor setup, verbatim in effect."""
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: F401
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    device = EL._resolve_eval_device(cfg)
    dtype = EL._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    check_task_suites(items, cfg, model)
    set_action_vae_suite(model, cfg)
    EL._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()
    stats = load_dataset_stats_from_json(str(EL._resolve_dataset_stats_path(cfg)))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    ah = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if ah is None else int(ah)
    h, w = (int(x) for x in cfg.data.train.get("video_size", [224, 224]))
    return model, processor, device, action_horizon, w, h


class BatchedPolicy:
    def __init__(self, cfg, model, processor, device, action_horizon, input_w, input_h, pad_batch):
        self.cfg, self.model, self.processor, self.device = cfg, model, processor, device
        self.action_horizon, self.input_w, self.input_h = action_horizon, input_w, input_h
        self.pad_batch = int(pad_batch)
        ev = cfg.EVALUATION
        steps = ev.get("num_inference_steps", None)
        self.steps = int(cfg.get("eval_num_inference_steps", 20) if steps is None else steps)
        self.sigma_shift = None if ev.get("sigma_shift") is None else float(ev.sigma_shift)
        self.seed = None if cfg.get("seed") is None else int(cfg.seed)
        self.rand_device = str(ev.get("rand_device", "cpu"))
        self.tiled = bool(ev.get("tiled", False))
        self.binarize = bool(ev.get("binarize_gripper", False))
        self.calls, self.seconds = 0, 0.0

    def __call__(self, obs_list, task_description):
        """Action chunks [T, 7] for each observation, as _predict_action_chunk returns them."""
        rows = [EL._obs_to_model_input(o, cfg=self.cfg, processor=self.processor, width=self.input_w,
                                       height=self.input_h, device=self.device, dtype=self.model.torch_dtype)
                for o in obs_list]
        x = torch.cat([r[0] for r in rows], dim=0)
        proprio = torch.cat([r[1] for r in rows], dim=0)
        n = x.shape[0]
        b = max(n, self.pad_batch)
        if b > n:
            x = torch.cat([x, x[:1].expand(b - n, *x.shape[1:])], dim=0)
            proprio = torch.cat([proprio, proprio[:1].expand(b - n, *proprio.shape[1:])], dim=0)
        prompt = DEFAULT_PROMPT.format(task=task_description)
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = infer_action_multi(self.model, prompts=[prompt] * b, input_images=x,
                                      action_horizon=self.action_horizon, proprios=proprio,
                                      num_inference_steps=self.steps, sigma_shift=self.sigma_shift,
                                      seeds=[self.seed] * b, rand_device=self.rand_device, tiled=self.tiled)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        chunks = []
        for i in range(n):
            a = EL._denormalize_action(pred["action"][i], self.processor)[0]
            a[..., -1] = a[..., -1] * 2 - 1
            a = invert_gripper_action(a)
            if self.binarize:
                a[..., -1] = np.sign(a[..., -1])
            chunks.append(a)
        return chunks


class PerObservationPolicy:
    """One B=1 call of the serial evaluator's _predict_action_chunk per observation (--model-batch 1)."""

    def __init__(self, cfg, model, processor, device, action_horizon, input_w, input_h):
        self.cfg, self.model, self.processor, self.device = cfg, model, processor, device
        self.action_horizon, self.input_w, self.input_h = action_horizon, input_w, input_h
        self.calls, self.seconds = 0, 0.0

    def __call__(self, obs_list, task_description):
        """Action chunks [T, 7] for each observation, exactly as the serial evaluator computes them."""
        t0 = time.perf_counter()
        chunks = [EL._predict_action_chunk(o, task_description, self.model, self.processor, self.cfg,
                                           action_horizon=self.action_horizon, input_w=self.input_w,
                                           input_h=self.input_h, model_device=self.device)[0]
                  for o in obs_list]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds += time.perf_counter() - t0
        self.calls += len(obs_list)
        return chunks


def run_task(suite, task_id, cfg, policy, num_envs, gpu_label, out_root):
    from libero.libero import benchmark, get_libero_path
    from instinct_sim_libero import HipLiberoVectorEnv
    trials = int(cfg.EVALUATION.num_trials)
    max_steps = EL._get_max_steps(suite)
    wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan = int(cfg.EVALUATION.get("replan_steps", 5))
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    init_states = torch.load(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
                             weights_only=False)
    while len(init_states) < trials:
        init_states.extend(init_states[: trials - len(init_states)])
    env_args = {"bddl_file_name": str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                "camera_heights": LIBERO_ENV_RESOLUTION, "camera_widths": LIBERO_ENV_RESOLUTION}
    n = min(num_envs, trials)
    start = time.time()
    results = {"task_suite": suite, "task_id": task_id, "task_description": task.language, "successes": 0,
               "total_episodes": trials, "gpu_id": gpu_label, "success_episodes": [], "failure_episodes": [],
               "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "duration": 0,
               "mode": "instinct-batched", "num_envs": n,
               "model_batch": 1 if isinstance(policy, PerObservationPolicy) else policy.pad_batch}
    calls0, seconds0 = policy.calls, policy.seconds
    venv = HipLiberoVectorEnv(env_args, num_envs=n, seed=cfg.get("seed"))
    try:
        for first in range(0, trials, n):
            episodes = list(range(first, min(first + n, trials)))
            obs = dict(enumerate(venv.reset([init_states[e] for e in episodes])))
            t = 0
            active = set(obs)
            pending = {k: [] for k in active}
            done = {k: False for k in active}
            while active and t < max_steps + wait:
                if t >= wait:
                    need = sorted(k for k in active if not pending[k])
                    if need:
                        for k, chunk in zip(need, policy([obs[k] for k in need], task.language)):
                            pending[k] = chunk[:replan].tolist()
                actions = {k: (get_libero_dummy_action() if t < wait else pending[k].pop(0)) for k in active}
                for k, (o, _, d, _) in venv.step(actions).items():
                    obs[k], done[k] = o, bool(d)
                if t >= wait:
                    for k in [k for k in active if done[k]]:
                        venv.finish(k)
                        active.discard(k)
                t += 1
            for k in active:
                venv.finish(k)
            for k, e in enumerate(episodes):
                (results["success_episodes"] if done[k] else results["failure_episodes"]).append(e)
            results["successes"] = len(results["success_episodes"])
            print(f"[batched] {suite} t{task_id} episodes {episodes[0]}-{episodes[-1]}: "
                  f"{sum(done[k] for k in range(len(episodes)))}/{len(episodes)} "
                  f"(policy {policy.calls} calls, {policy.seconds:.0f} s)", flush=True)
    finally:
        venv.close()
    results["duration"] = time.time() - start
    calls = policy.calls - calls0
    results["policy"] = {"calls": calls, "seconds": policy.seconds - seconds0,
                         "ms_per_call": 1e3 * (policy.seconds - seconds0) / calls if calls else None}
    out_dir = out_root / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"gpu{gpu_label}_task{task_id}_results.json"
    tmp = out_dir / f".{out.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(results, indent=4, cls=EL.NumpyEncoder), encoding="utf-8")
    os.replace(tmp, out)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config-dir", required=True, help="FastWAM hydra config dir (configs_baseline for the FastWAM baseline and GEAK trees)")
    ap.add_argument("--task", required=True, help="hydra task config name")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset-stats", required=True)
    ap.add_argument("--steps", type=int, required=True, help="denoise steps")
    ap.add_argument("--tasks", required=True, help="CSV of suite:task_id")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=50, help="worlds (episodes in lockstep) per task")
    ap.add_argument("--pad-batch", type=int, default=64, help="model batch size (rows padded); ignored with --model-batch 1")
    ap.add_argument("--model-batch", choices=("auto", "1"), default="auto",
                    help="auto: one padded call per replan (baseline, geak); 1: one B=1 call per observation (cae)")
    ap.add_argument("--gpu-label", default=os.environ.get("HP_GPU_LABEL", "0"))
    ap.add_argument("--override", action="append", default=[], help="extra hydra override (repeatable)")
    a = ap.parse_args()

    print(f"[batched] fastwam from {os.path.dirname(fastwam.__file__)}", flush=True)
    items = parse_tasks(a.tasks)
    per_obs = a.model_batch == "1"
    cfg = compose_cfg(a)
    model, processor, device, action_horizon, w, h = load_model(cfg, items)
    refuse_unbatched_features(cfg, model, per_obs)
    if per_obs:
        sid = getattr(model, "action_vae_suite_id", None)
        print(f"[batched] model batch 1 (one call per observation); action-VAE suite id {sid}", flush=True)
        policy = PerObservationPolicy(cfg, model, processor, device, action_horizon, w, h)
    else:
        policy = BatchedPolicy(cfg, model, processor, device, action_horizon, w, h, a.pad_batch)
    out_root = Path(a.output_dir).resolve()
    for suite, tid in items:
        if (out_root / suite / f"gpu{a.gpu_label}_task{tid}_results.json").exists():
            print(f"[batched] {suite} t{tid}: result exists, skipped", flush=True)
            continue
        r = run_task(suite, int(tid), cfg, policy, a.num_envs, a.gpu_label, out_root)
        pm = r["policy"]["ms_per_call"]
        print(f"[batched] {suite} t{tid}: {r['successes']}/{r['total_episodes']} in {r['duration']:.0f} s"
              f"{'' if pm is None else f' (policy {pm:.1f} ms/call)'}", flush=True)


if __name__ == "__main__":
    main()
