#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoboCasa-GR1 full-24 evaluation driver for any DiT4DiT tree and checkpoint.

The protocol is the one in the vendored tree's
``DiT4DiT/examples/Robocasa_tabletop/eval_files/``: the 24-task registry,
the SHA-256 per-episode seed derivation and explicit resets
(``full24_rollout_matching.py``), the ``simulation_env.py`` client, the
schema-2 result files, ``protocol.json`` and ``summarize_full24_policy.py``.
This driver imports those files; it does not copy them. It adds two things to
``run_full24_evaluation.py``:

* The policy server may come from any tree (``--server-root`` and
  ``--server-script``), so the side-by-side trees under ``src/`` are evaluated
  with the same clients, seeds and result files as the vendored tree. The
  client is always the vendored ``simulation_env.py``.
* ``--clients-per-gpu K`` runs K clients against each policy server. Server g
  keeps the tasks ``run_full24_evaluation.py`` gives it (``tasks[g::G]``); its
  (task, episode) list is cut into K contiguous shares, one per client. A
  client playing episodes first..first+n-1 of a task resets episode i with
  ``full24_episode_seed(eval_seed, env_name, first + i)``, the seed that
  episode has at K=1, and its records carry the K=1 episode indices. Once every
  share of a task is in, the shares are merged into ``results/<task>.json``,
  the same file a K=1 client writes.

With K=1 and the vendored server, every server and client process gets the
command line, working directory and environment ``run_full24_evaluation.py``
gives it, and tasks run in the same order on the same server.

Usage:
  robocasa_full24.py run [options]          evaluate (see --help)
  robocasa_full24.py run --dry-run [...]    print the server/client plan only
  robocasa_full24.py summarize RUN_DIR ...  validate and summarize a run
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_flags  # noqa: E402  (scripts/eval_flags.py: the shared MUJOCO_GL resolution)

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
CLIENT_SHIM = SCRIPTS_DIR / "robocasa_full24_client.py"
EVAL_FILES = Path("examples") / "Robocasa_tabletop" / "eval_files"

# MuJoCo version the GPU RoboCasa backend runs under (simulate/README.md).
INSTINCT_SIM_MUJOCO = "3.3.2"


# ----------------------------------------------------------------------------
# Work split (pure; no simulator, model or vendored import needed)
# ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Segment:
    """Episodes ``first .. first + count - 1`` of one task."""

    task: Any
    first: int
    count: int
    episodes_per_task: int

    @property
    def whole(self) -> bool:
        return self.first == 0 and self.count == self.episodes_per_task

    @property
    def stem(self) -> str:
        return f"{self.task.label}.e{self.first:02d}-{self.first + self.count - 1:02d}"


def plan_clients(
    tasks: Sequence[Any],
    n_servers: int,
    clients_per_server: int,
    episodes_per_task: int,
) -> list[list[list[Segment]]]:
    """plan[server][client] = the segments that client plays, in order.

    Server g gets ``tasks[g::n_servers]``, as in run_full24_evaluation.py. Its
    (task, episode) pairs, in task order then episode order, are cut into
    ``clients_per_server`` contiguous shares whose sizes differ by at most one;
    each share is split at task boundaries into segments. With one client per
    server, each segment is a whole task.
    """
    if n_servers < 1 or clients_per_server < 1 or episodes_per_task < 1:
        raise ValueError("servers, clients per server and episodes must be positive")
    plan = []
    for server in range(n_servers):
        pairs = [
            (task, episode)
            for task in tasks[server::n_servers]
            for episode in range(episodes_per_task)
        ]
        base, extra = divmod(len(pairs), clients_per_server)
        clients, start = [], 0
        for client in range(clients_per_server):
            size = base + (1 if client < extra else 0)
            share = pairs[start : start + size]
            start += size
            segments: list[Segment] = []
            for task, episode in share:
                last = segments[-1] if segments else None
                if last is not None and last.task is task and last.first + last.count == episode:
                    segments[-1] = dataclasses.replace(last, count=last.count + 1)
                else:
                    segments.append(Segment(task, episode, 1, episodes_per_task))
            clients.append(segments)
        plan.append(clients)
    return plan


def merge_shares(
    task: Any,
    shares: Sequence[dict],
    *,
    episodes_per_task: int,
    eval_seed: int,
    rollout_matching: dict,
    episode_seed: Callable[[int, str, int], int],
) -> dict:
    """The ``results/<task>.json`` payload a single client would have written.

    Checks that the shares cover episodes 0..episodes_per_task-1 exactly once,
    each with its protocol seed. The key order and the success-rate arithmetic
    follow simulation_env.py's writer.
    """
    import numpy as np

    episodes = []
    for share in shares:
        for field, expected in (
            ("schema_version", 2),
            ("env_name", task.env_name),
            ("task_id", task.task_id),
            ("label", task.label),
            ("group", task.group),
            ("seed", eval_seed),
            ("rollout_matching", rollout_matching),
        ):
            if share.get(field) != expected:
                raise ValueError(f"{task.label}: share has unexpected {field}: {share.get(field)!r}")
        episodes.extend(share["episodes"])
    episodes.sort(key=lambda record: record["episode_index"])
    indices = [record["episode_index"] for record in episodes]
    if indices != list(range(episodes_per_task)):
        raise ValueError(f"{task.label}: shares cover episodes {indices}, expected 0..{episodes_per_task - 1}")
    for record in episodes:
        want = episode_seed(eval_seed, task.env_name, record["episode_index"])
        if record.get("episode_seed") != want:
            raise ValueError(f"{task.label}: episode {record['episode_index']} has seed {record.get('episode_seed')}, expected {want}")
    return {
        "schema_version": 2,
        "env_name": task.env_name,
        "task_id": task.task_id,
        "label": task.label,
        "group": task.group,
        "seed": eval_seed,
        "rollout_matching": rollout_matching,
        "episodes": episodes,
        "success_rate": float(np.mean([record["success"] for record in episodes])),
    }


def share_is_complete(path: Path, segment: Segment, eval_seed: int, rollout_matching: dict, episode_seed) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    episodes = payload.get("episodes", [])
    return (
        payload.get("schema_version") == 2
        and payload.get("label") == segment.task.label
        and payload.get("seed") == eval_seed
        and payload.get("rollout_matching") == rollout_matching
        and [e.get("episode_index") for e in episodes]
        == list(range(segment.first, segment.first + segment.count))
        and all(
            e.get("episode_seed") == episode_seed(eval_seed, segment.task.env_name, e["episode_index"])
            for e in episodes
        )
    )


# ----------------------------------------------------------------------------
# Vendored evaluation code
# ----------------------------------------------------------------------------
def load_vendored(dit4dit_root: Path):
    """Import the vendored full-24 modules (run_full24_evaluation, rollout matching, tasks)."""
    root = str(dit4dit_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import importlib

    runner = importlib.import_module("examples.Robocasa_tabletop.eval_files.run_full24_evaluation")
    matching = importlib.import_module("examples.Robocasa_tabletop.eval_files.full24_rollout_matching")
    tasks = importlib.import_module("DiT4DiT.dataloader.gr00t_lerobot.full24_tasks")
    if Path(runner.REPO_ROOT).resolve() != dit4dit_root.resolve():
        raise RuntimeError(f"imported the full-24 runner from {runner.REPO_ROOT}, expected {dit4dit_root}")
    return runner, matching, tasks.FULL24_TASKS


def client_environment(args: argparse.Namespace, dit4dit_root: Path) -> dict[str, str]:
    """The client environment run_full24_evaluation.evaluate_task builds."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        f"{args.robosuite_overlay}:{dit4dit_root}:"
        f"{args.robocasa_root}:"
        f"{environment.get('PYTHONPATH', '')}"
    )
    environment["PYTHONHASHSEED"] = str(args.eval_seed)
    return environment


def resolve_client_mujoco_gl(args: argparse.Namespace, dit4dit_root: Path) -> None:
    """MUJOCO_GL for the clients, set in this process's environment, which every client inherits.

    An exported MUJOCO_GL (the user's, or OSMesa from the launcher under --sim
    instinct) is kept; otherwise the client interpreter, in the client
    environment, tries an EGL context: egl (GPU OpenGL) on success, else osmesa.
    run_full24_evaluation.py honours the caller's MUJOCO_GL the same way, so a
    client still gets the environment it would get from that runner.
    """
    environment = client_environment(args, dit4dit_root)
    resolved = eval_flags.resolve_mujoco_gl([args.python_bin], env=environment, cwd=dit4dit_root)
    eval_flags.apply_mujoco_gl(resolved)


def client_runtime_versions(args: argparse.Namespace, dit4dit_root: Path) -> dict[str, str]:
    """run_full24_evaluation.runtime_versions(), evaluated in the clients' environment.

    The runner records the versions its own process sees; the clients see the
    robosuite overlay (and, on the GPU simulator, MuJoCo 3.3.2) first, so their view
    is the one that describes the simulator.
    """
    code = (
        "import importlib.metadata as m, json, platform\n"
        "names = ('mujoco', 'robosuite', 'numpy', 'torch', 'gymnasium')\n"
        "print(json.dumps({'python': platform.python_version(), **{n: m.version(n) for n in names}}))\n"
    )
    result = subprocess.run(
        [args.python_bin, "-c", code],
        cwd=dit4dit_root,
        env=client_environment(args, dit4dit_root),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def server_command(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        args.server_python,
        str(args.server_script),
        "--ckpt_path",
        str(args.checkpoint),
        "--port",
        str(port),
        "--use_bf16",
    ]
    if args.server_seed == "flag":
        command += ["--seed", str(args.policy_seed)]
    return command + ["--idle_timeout", "-1"]


def start_servers(runner, servers, args: argparse.Namespace, gpus: list[str], log_dir: Path) -> list[int]:
    """run_full24_evaluation.PolicyServers.start with the server root, script and environment as parameters."""
    ports = [args.base_port + index for index in range(len(gpus))]
    for gpu, port in zip(gpus, ports):
        log_path = log_dir / f"server_gpu{gpu}_port{port}.log"
        log_file = log_path.open("w")
        environment = os.environ.copy()
        environment.update(dict(item.split("=", 1) for item in args.server_env))
        environment["HIP_VISIBLE_DEVICES"] = gpu
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        if args.server_seed == "env":
            environment["DIT4DIT_POLICY_SEED"] = str(args.policy_seed)
        extra = f"{args.server_pythonpath}:" if args.server_pythonpath else ""
        environment["PYTHONPATH"] = f"{args.server_root}:{extra}{environment.get('PYTHONPATH', '')}"
        process = subprocess.Popen(
            server_command(args, port),
            cwd=args.server_root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        servers.processes.append(process)
        servers.logs.append(log_file)
    for process, port, gpu in zip(servers.processes, ports, gpus):
        runner.wait_until_ready(
            process, port, log_dir / f"server_gpu{gpu}_port{port}.log", args.server_timeout_seconds
        )
    return ports


def run_share(segment: Segment, *, port: int, args, dit4dit_root: Path, dirs: dict, matching) -> None:
    """One client playing part of a task, through robocasa_full24_client.py."""
    if result_is_complete(dirs["results"] / f"{segment.task.label}.json", segment.task, args, matching):
        print(f"Skipping completed task: {segment.task.label}", flush=True)
        return
    share_path = dirs["shares"] / f"{segment.stem}.json"
    if share_is_complete(share_path, segment, args.eval_seed, matching.ROLLOUT_MATCHING_PROTOCOL, matching.full24_episode_seed):
        print(f"Skipping completed share: {segment.stem}", flush=True)
        return
    command = [
        args.python_bin,
        str(CLIENT_SHIM),
        str(segment.first),
        str(dit4dit_root / EVAL_FILES / "simulation_env.py"),
        "--args.env_name",
        segment.task.env_name,
        "--args.port",
        str(port),
        "--args.n_episodes",
        str(segment.count),
        "--args.n_envs",
        "1",
        "--args.seed",
        str(args.eval_seed),
        "--args.max_episode_steps",
        str(args.max_episode_steps),
        "--args.n_action_steps",
        str(args.n_action_steps),
        "--args.pretrained_path",
        str(args.checkpoint),
        "--args.result-out-path",
        str(share_path),
    ]
    if args.ceiling_vae is not None:
        command.extend(["--args.ceiling-vae-path", str(args.ceiling_vae)])
    if dirs["videos"] is not None:
        task_video_dir = dirs["videos"] / segment.task.label
        task_video_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--args.video-out-path", str(task_video_dir)])
    log_path = dirs["logs"] / f"{segment.stem}.log"
    with log_path.open("w") as log_file:
        try:
            subprocess.run(
                command,
                cwd=dit4dit_root,
                env=client_environment(args, dit4dit_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            # A client that dies on a signal (a native crash in the simulator stack, say) leaves no Python
            # traceback for the driver to show; print the end of its log with the failure.
            log_file.flush()
            lines = log_path.read_text(errors="replace").splitlines()[-40:]
            print(f"client {segment.stem} failed ({exc}); last lines of {log_path}:\n" + "\n".join(lines),
                  file=sys.stderr, flush=True)
            raise


def result_is_complete(path: Path, task, args, matching) -> bool:
    """run_full24_evaluation.evaluate_task's resume check (raises on a foreign result)."""
    if not path.is_file():
        return False
    completed = json.loads(path.read_text())
    episodes = completed.get("episodes", [])
    if (
        completed.get("schema_version") == 2
        and completed.get("label") == task.label
        and completed.get("seed") == args.eval_seed
        and completed.get("rollout_matching") == matching.ROLLOUT_MATCHING_PROTOCOL
        and len(episodes) == args.episodes
        and all(
            {"episode_seed", "instruction", "initial_observation_fingerprint"} <= episode.keys()
            for episode in episodes
        )
    ):
        return True
    raise ValueError(f"Existing task result does not match this run: {path}")


def merge_completed_tasks(plan, args, dirs, matching, runner) -> None:
    by_task: dict[str, list[Segment]] = {}
    for server in plan:
        for client in server:
            for segment in client:
                if not segment.whole:
                    by_task.setdefault(segment.task.label, []).append(segment)
    for label, segments in by_task.items():
        task = segments[0].task
        result_path = dirs["results"] / f"{label}.json"
        if result_is_complete(result_path, task, args, matching):
            continue
        paths = [dirs["shares"] / f"{segment.stem}.json" for segment in segments]
        if not all(
            share_is_complete(path, segment, args.eval_seed, matching.ROLLOUT_MATCHING_PROTOCOL, matching.full24_episode_seed)
            for path, segment in zip(paths, segments)
        ):
            continue
        payload = merge_shares(
            task,
            [json.loads(path.read_text()) for path in paths],
            episodes_per_task=args.episodes,
            eval_seed=args.eval_seed,
            rollout_matching=matching.ROLLOUT_MATCHING_PROTOCOL,
            episode_seed=matching.full24_episode_seed,
        )
        runner.atomic_json(result_path, payload)
        print(f"Merged {len(paths)} shares: {result_path}", flush=True)


# protocol.json fields that decide what a run directory's results mean. A rerun
# into an existing directory resumes it only when every one matches: anything
# else would merge task results of another arm (server root, script, label),
# checkpoint, episode count, step count or simulator (label, overlay, MuJoCo)
# into this run's summary. Client and GPU counts may differ: every episode
# keeps its seed whatever the split.
RESUME_FIELDS = (
    ("checkpoint",),
    ("ceiling_vae",),
    ("condition",),
    ("episodes_per_task",),
    ("max_episode_steps",),
    ("n_action_steps",),
    ("policy_seed",),
    ("eval_seed",),
    ("rollout_matching",),
    ("simulation_robosuite_overlay",),
    ("runtime_versions", "mujoco"),
    ("policy_server", "root"),
    ("policy_server", "script"),
    ("policy_server", "seed"),
    ("policy_server", "label"),
)


def resume_mismatches(existing: dict, protocol: dict) -> list[str]:
    """The RESUME_FIELDS on which an existing protocol.json and this run's differ."""

    def field(payload: Any, path: tuple[str, ...]) -> Any:
        for key in path:
            payload = payload.get(key) if isinstance(payload, dict) else None
        return payload

    current = json.loads(json.dumps(protocol))
    return [
        f"{'.'.join(path)}: {field(existing, path)!r} (recorded) != {field(current, path)!r} (this run)"
        for path in RESUME_FIELDS
        if field(existing, path) != field(current, path)
    ]


def print_plan(plan, gpus: list[str], base_port: int) -> None:
    for index, (gpu, clients) in enumerate(zip(gpus, plan)):
        print(f"server gpu {gpu} port {base_port + index}")
        for client_index, segments in enumerate(clients):
            text = ", ".join(
                f"{s.task.label}[{s.first}:{s.first + s.count}]" for s in segments
            ) or "(idle)"
            print(f"  client {client_index}: {sum(s.count for s in segments)} episodes: {text}")


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------
def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    # run_full24_evaluation.py's flags, same names and defaults.
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ceiling-vae", type=Path)
    parser.add_argument("--episodes", type=int, choices=(1, 10, 50), required=True)
    parser.add_argument("--gpu-list", default="0,1")
    parser.add_argument("--base-port", type=int, default=6398)
    parser.add_argument("--policy-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--python-bin", default=sys.executable, help="client Python")
    parser.add_argument("--robosuite-overlay", type=Path)
    parser.add_argument("--robocasa-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--server-timeout-seconds", type=float, default=600)
    # This driver's additions.
    parser.add_argument("--dit4dit-root", type=Path, default=REPO_ROOT / "DiT4DiT",
                        help="tree holding the full-24 harness and client (default: the vendored DiT4DiT/)")
    parser.add_argument("--clients-per-gpu", type=int, default=1)
    parser.add_argument("--server-python", help="policy-server Python (default: --python-bin)")
    parser.add_argument("--server-root", type=Path,
                        help="policy-server tree: working directory and first PYTHONPATH entry (default: --dit4dit-root)")
    parser.add_argument("--server-script", type=Path,
                        help="policy-server entry point (default: <server-root>/deployment/model_server/server_policy.py)")
    parser.add_argument("--server-pythonpath", default="", help="extra PYTHONPATH entries after the server root")
    parser.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra policy-server environment (repeatable)")
    parser.add_argument("--server-seed", choices=("flag", "env"), default="flag",
                        help="pass the policy seed as --seed (flag) or as DIT4DIT_POLICY_SEED (env)")
    parser.add_argument("--label", default="", help="free-form run label recorded in protocol.json")
    parser.add_argument("--dry-run", action="store_true", help="print the server/client plan and exit")


def run(args: argparse.Namespace) -> None:
    dit4dit_root = args.dit4dit_root.resolve()
    if args.clients_per_gpu < 1:
        raise ValueError("--clients-per-gpu must be at least 1")
    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU list must contain unique IDs")

    if args.dry_run:
        # The task registry only; it imports nothing beyond the standard library.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "full24_tasks", dit4dit_root / "DiT4DiT" / "dataloader" / "gr00t_lerobot" / "full24_tasks.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print_plan(plan_clients(module.FULL24_TASKS, len(gpus), args.clients_per_gpu, args.episodes), gpus, args.base_port)
        return

    missing = [flag for flag, value in (("--checkpoint", args.checkpoint), ("--robosuite-overlay", args.robosuite_overlay),
                                        ("--output-dir", args.output_dir)) if value is None]
    if missing:
        raise SystemExit(f"run requires {', '.join(missing)}")
    runner, matching, full24_tasks = load_vendored(dit4dit_root)
    args.robocasa_root = args.robocasa_root or dit4dit_root / "robocasa-gr1-tabletop-tasks"
    args.server_python = args.server_python or args.python_bin
    args.server_root = (args.server_root or dit4dit_root).resolve()
    args.server_script = (args.server_script or args.server_root / "deployment" / "model_server" / "server_policy.py").resolve()
    args.gpu_list = gpus
    if args.condition == "direct" and args.ceiling_vae is not None:
        raise ValueError("Direct condition cannot specify --ceiling-vae")
    if args.condition != "direct" and args.ceiling_vae is None:
        raise ValueError("Candidate condition requires --ceiling-vae")
    for path in (args.checkpoint, args.server_script) + ((args.ceiling_vae,) if args.ceiling_vae else ()):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.robosuite_overlay, args.robocasa_root, args.server_root):
        if not path.is_dir():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.resolve()
    dirs = {
        "logs": output_dir / "logs",
        "results": output_dir / "results",
        "shares": output_dir / "shares",
        "videos": output_dir / "videos" if args.record_video else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs["logs"].mkdir(exist_ok=True)
    dirs["results"].mkdir(exist_ok=True)
    lock_dir = output_dir.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{output_dir.name}.lock"
    lock_file = lock_path.open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Evaluation lock is held: {lock_path}") from exc
    if (output_dir / "done.json").exists():
        raise FileExistsError(f"Evaluation already completed: {output_dir}")

    plan = plan_clients(full24_tasks, len(gpus), args.clients_per_gpu, args.episodes)
    if any(not segment.whole for server in plan for client in server for segment in client):
        dirs["shares"].mkdir(exist_ok=True)

    resolve_client_mujoco_gl(args, dit4dit_root)

    # run_full24_evaluation.py's protocol.json, plus the policy server and the client count.
    protocol = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry": "full24",
        "checkpoint": runner.artifact_identity(args.checkpoint),
        "ceiling_vae": runner.artifact_identity(args.ceiling_vae),
        "condition": args.condition,
        "episodes_per_task": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "policy_seed": args.policy_seed,
        "eval_seed": args.eval_seed,
        "rollout_matching": matching.ROLLOUT_MATCHING_PROTOCOL,
        "simulation_robosuite_overlay": matching.robosuite_overlay_identity(args.robosuite_overlay),
        "robocasa_root": str(args.robocasa_root.resolve()),
        "python_hash_seed": args.eval_seed,
        "runtime_versions": client_runtime_versions(args, dit4dit_root),
        "gpu_list": gpus,
        "base_port": args.base_port,
        "task_server_slots": {task.label: index % len(gpus) for index, task in enumerate(full24_tasks)},
        "tasks": [dataclasses.asdict(task) for task in full24_tasks],
        "clients_per_gpu": args.clients_per_gpu,
        "policy_server": {
            "root": str(args.server_root),
            "script": str(args.server_script),
            "seed": "--seed" if args.server_seed == "flag" else "DIT4DIT_POLICY_SEED",
            "label": args.label,
        },
    }
    protocol_path = output_dir / "protocol.json"
    has_results = any(dirs["results"].glob("*.json")) or any(dirs["shares"].glob("*.json"))
    if protocol_path.is_file() and has_results:
        mismatches = resume_mismatches(json.loads(protocol_path.read_text()), protocol)
        if mismatches:
            raise SystemExit(
                f"refusing to resume into {output_dir}: it holds a different run\n  "
                + "\n  ".join(mismatches)
                + "\nuse a fresh --output-dir"
            )
    (output_dir / "failed.json").unlink(missing_ok=True)
    runner.atomic_json(protocol_path, protocol)

    servers = runner.PolicyServers()
    atexit.register(servers.stop)

    def handle_signal(signum, _frame):
        servers.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)
    started = time.monotonic()
    try:
        ports = start_servers(runner, servers, args, gpus, dirs["logs"])
        failures: list[BaseException] = []
        failures_lock = threading.Lock()

        def run_client(port: int, segments: list[Segment]) -> None:
            try:
                for segment in segments:
                    if segment.whole:
                        runner.evaluate_task(
                            segment.task,
                            port=port,
                            args=args,
                            log_dir=dirs["logs"],
                            result_dir=dirs["results"],
                            video_dir=dirs["videos"],
                        )
                    else:
                        run_share(segment, port=port, args=args, dit4dit_root=dit4dit_root, dirs=dirs, matching=matching)
            except BaseException as exc:  # recorded; the other clients carry on, as in the runner
                with failures_lock:
                    failures.append(exc)

        jobs = [(port, segments) for port, clients in zip(ports, plan) for segments in clients if segments]
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            for future in [executor.submit(run_client, *job) for job in jobs]:
                future.result()
        merge_completed_tasks(plan, args, dirs, matching, runner)
        if failures:
            raise failures[0]
        elapsed = time.monotonic() - started
        runner.atomic_json(
            output_dir / "done.json",
            {
                "status": "completed",
                "condition": args.condition,
                "episodes_per_task": args.episodes,
                "task_count": len(full24_tasks),
                "elapsed_seconds": elapsed,
                "protocol": str(output_dir / "protocol.json"),
            },
        )
        print(f"Full-24 evaluation completed: {output_dir}")
    except BaseException as exc:
        runner.atomic_json(
            output_dir / "failed.json",
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        servers.stop()
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# ----------------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------------
def summarize(args: argparse.Namespace) -> None:
    """summarize_full24_policy.py; on the GPU simulator the recorded MuJoCo is the backend's 3.3.2."""
    sys.path.insert(0, str(args.dit4dit_root.resolve()))
    import importlib

    summarizer = importlib.import_module("examples.Robocasa_tabletop.eval_files.summarize_full24_policy")
    if args.sim == "instinct":
        summarizer.EXPECTED_RUNTIME = {**summarizer.EXPECTED_RUNTIME, "mujoco": INSTINCT_SIM_MUJOCO}
    summary = summarizer.validate_and_summarize(
        args.run_dir,
        expected_successes=args.expected_successes,
        expected_episodes_per_task=args.episodes_per_task,
        expected_seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(commands.add_parser("run", help="run an evaluation"))
    summary = commands.add_parser("summarize", help="validate and summarize a run")
    summary.add_argument("run_dir", type=Path)
    summary.add_argument("--expected-successes", type=int)
    summary.add_argument("--episodes-per-task", type=int, default=50)
    summary.add_argument("--seed", type=int, default=7)
    summary.add_argument("--sim", choices=("mujoco", "instinct"), default="mujoco")
    summary.add_argument("--dit4dit-root", type=Path, default=REPO_ROOT / "DiT4DiT")
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
