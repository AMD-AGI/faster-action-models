#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one deterministic full-24 evaluation condition across policy servers."""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from DiT4DiT.dataloader.gr00t_lerobot.full24_tasks import FULL24_TASKS  # noqa: E402
from examples.Robocasa_tabletop.eval_files.full24_rollout_matching import (  # noqa: E402
    ROLLOUT_MATCHING_PROTOCOL,
    robosuite_overlay_identity,
)

def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def artifact_identity(path: Path | None) -> dict | None:
    if path is None:
        return None
    resolved = path.resolve()
    stat = resolved.stat()
    identity = {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if stat.st_size < 100_000_000:
        identity["sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return identity


def runtime_versions() -> dict[str, str]:
    packages = ("mujoco", "robosuite", "numpy", "torch", "gymnasium")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def parse_gpu_list(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError("GPU list must contain unique IDs")
    return gpus


def wait_until_ready(
    process: subprocess.Popen,
    port: int,
    log_path: Path,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Policy server exited with {process.returncode}; see {log_path}"
            )
        with socket.socket() as probe:
            probe.settimeout(1.0)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(2)
    raise TimeoutError(f"Policy server did not open port {port}; see {log_path}")


class PolicyServers:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []
        self.logs = []

    def start(
        self,
        *,
        checkpoint: Path,
        python_bin: str,
        gpus: list[str],
        base_port: int,
        policy_seed: int,
        log_dir: Path,
        timeout_seconds: float,
    ) -> list[int]:
        ports = [base_port + index for index in range(len(gpus))]
        for gpu, port in zip(gpus, ports):
            log_path = log_dir / f"server_gpu{gpu}_port{port}.log"
            log_file = log_path.open("w")
            environment = os.environ.copy()
            environment["HIP_VISIBLE_DEVICES"] = gpu
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = (
                f"{REPO_ROOT}:{environment.get('PYTHONPATH', '')}"
            )
            process = subprocess.Popen(
                [
                    python_bin,
                    str(REPO_ROOT / "deployment/model_server/server_policy.py"),
                    "--ckpt_path",
                    str(checkpoint),
                    "--port",
                    str(port),
                    "--use_bf16",
                    "--seed",
                    str(policy_seed),
                    "--idle_timeout",
                    "-1",
                ],
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.processes.append(process)
            self.logs.append(log_file)
        for process, port, gpu in zip(self.processes, ports, gpus):
            wait_until_ready(
                process,
                port,
                log_dir / f"server_gpu{gpu}_port{port}.log",
                timeout_seconds,
            )
        return ports

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        for process in self.processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
        for log_file in self.logs:
            log_file.close()
        self.processes.clear()
        self.logs.clear()


def evaluate_task(
    task,
    *,
    port: int,
    args: argparse.Namespace,
    log_dir: Path,
    result_dir: Path,
    video_dir: Path | None,
) -> None:
    result_path = result_dir / f"{task.label}.json"
    if result_path.is_file():
        completed = json.loads(result_path.read_text())
        episodes = completed.get("episodes", [])
        if (
            completed.get("schema_version") == 2
            and completed.get("label") == task.label
            and completed.get("seed") == args.eval_seed
            and completed.get("rollout_matching") == ROLLOUT_MATCHING_PROTOCOL
            and len(episodes) == args.episodes
            and all(
                {
                    "episode_seed",
                    "instruction",
                    "initial_observation_fingerprint",
                }
                <= episode.keys()
                for episode in episodes
            )
        ):
            print(f"Skipping completed task: {task.label}", flush=True)
            return
        raise ValueError(f"Existing task result does not match this run: {result_path}")

    command = [
        args.python_bin,
        str(SCRIPT_DIR / "simulation_env.py"),
        "--args.env_name",
        task.env_name,
        "--args.port",
        str(port),
        "--args.n_episodes",
        str(args.episodes),
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
        str(result_path),
    ]
    if args.ceiling_vae is not None:
        command.extend(["--args.ceiling-vae-path", str(args.ceiling_vae)])
    if video_dir is not None:
        task_video_dir = video_dir / task.label
        task_video_dir.mkdir(parents=True, exist_ok=True)
        command.extend(["--args.video-out-path", str(task_video_dir)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = client_pythonpath(args)
    # MUJOCO_GL comes from resolve_mujoco_gl() (or the caller); OSMesa only
    # when neither set it.
    mujoco_gl = environment.setdefault("MUJOCO_GL", "osmesa")
    if mujoco_gl in ("egl", "osmesa"):
        environment.setdefault("PYOPENGL_PLATFORM", mujoco_gl)
    environment["PYTHONHASHSEED"] = str(args.eval_seed)
    log_path = log_dir / f"{task.label}.log"
    with log_path.open("w") as log_file:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ceiling-vae", type=Path)
    parser.add_argument("--episodes", type=int, choices=(1, 10, 50), required=True)
    parser.add_argument("--gpu-list", type=parse_gpu_list, default=parse_gpu_list("0,1"))
    parser.add_argument("--base-port", type=int, default=6398)
    parser.add_argument("--policy-seed", type=int, default=7)
    parser.add_argument("--eval-seed", type=int, default=7)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--robosuite-overlay",
        type=Path,
        required=True,
        help="site-packages directory containing the pinned robosuite 1.5.1",
    )
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        default=REPO_ROOT / "robocasa-gr1-tabletop-tasks",
        help="Pinned robocasa-gr1-tabletop-tasks checkout.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--server-timeout-seconds", type=float, default=600)
    return parser.parse_args()


def client_pythonpath(args: argparse.Namespace) -> str:
    return (
        f"{args.robosuite_overlay}:{REPO_ROOT}:"
        f"{args.robocasa_root}:"
        f"{os.environ.get('PYTHONPATH', '')}"
    )


def resolve_mujoco_gl(args: argparse.Namespace) -> None:
    """Set MUJOCO_GL for the clients in this process's environment.

    A caller's MUJOCO_GL (egl: GPU OpenGL, osmesa, glfw) is kept unchanged.
    Otherwise, inside the parent repository, its shared check
    (scripts/eval_flags.py, also used by the launchers) has the client Python,
    in the client environment, try an EGL context: egl on success, else osmesa.
    Outside it (this tree copied elsewhere), the clients fall back to OSMesa.
    """
    helper = REPO_ROOT.parent / "scripts" / "eval_flags.py"
    if not helper.is_file():
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_flags", helper)
    eval_flags = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_flags)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = client_pythonpath(args)
    resolved = eval_flags.resolve_mujoco_gl(
        [args.python_bin], env=environment, cwd=REPO_ROOT
    )
    eval_flags.apply_mujoco_gl(resolved)


def main() -> None:
    args = parse_args()
    if args.condition == "direct" and args.ceiling_vae is not None:
        raise ValueError("Direct condition cannot specify --ceiling-vae")
    if args.condition != "direct" and args.ceiling_vae is None:
        raise ValueError("Candidate condition requires --ceiling-vae")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.ceiling_vae is not None and not args.ceiling_vae.is_file():
        raise FileNotFoundError(args.ceiling_vae)
    if not args.robosuite_overlay.is_dir():
        raise FileNotFoundError(args.robosuite_overlay)
    if not args.robocasa_root.is_dir():
        raise FileNotFoundError(args.robocasa_root)

    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "logs"
    result_dir = output_dir / "results"
    video_dir = output_dir / "videos" if args.record_video else None
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
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
    (output_dir / "failed.json").unlink(missing_ok=True)
    resolve_mujoco_gl(args)

    protocol = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registry": "full24",
        "checkpoint": artifact_identity(args.checkpoint),
        "ceiling_vae": artifact_identity(args.ceiling_vae),
        "condition": args.condition,
        "episodes_per_task": args.episodes,
        "max_episode_steps": args.max_episode_steps,
        "n_action_steps": args.n_action_steps,
        "policy_seed": args.policy_seed,
        "eval_seed": args.eval_seed,
        "rollout_matching": ROLLOUT_MATCHING_PROTOCOL,
        "simulation_robosuite_overlay": robosuite_overlay_identity(
            args.robosuite_overlay
        ),
        "robocasa_root": str(args.robocasa_root.resolve()),
        "python_hash_seed": args.eval_seed,
        "runtime_versions": runtime_versions(),
        "gpu_list": args.gpu_list,
        "base_port": args.base_port,
        "task_server_slots": {
            task.label: index % len(args.gpu_list)
            for index, task in enumerate(FULL24_TASKS)
        },
        "tasks": [dataclasses.asdict(task) for task in FULL24_TASKS],
    }
    atomic_json(output_dir / "protocol.json", protocol)

    servers = PolicyServers()
    atexit.register(servers.stop)

    def handle_signal(signum, _frame):
        servers.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGHUP, handle_signal)
    started = time.monotonic()
    try:
        ports = servers.start(
            checkpoint=args.checkpoint,
            python_bin=args.python_bin,
            gpus=args.gpu_list,
            base_port=args.base_port,
            policy_seed=args.policy_seed,
            log_dir=log_dir,
            timeout_seconds=args.server_timeout_seconds,
        )
        assignments = [
            list(FULL24_TASKS[index :: len(ports)])
            for index in range(len(ports))
        ]

        def run_assignment(port_and_tasks) -> None:
            port, tasks = port_and_tasks
            for task in tasks:
                evaluate_task(
                    task,
                    port=port,
                    args=args,
                    log_dir=log_dir,
                    result_dir=result_dir,
                    video_dir=video_dir,
                )

        with ThreadPoolExecutor(max_workers=len(ports)) as executor:
            futures = [
                executor.submit(run_assignment, assignment)
                for assignment in zip(ports, assignments)
            ]
            for future in futures:
                future.result()
        elapsed = time.monotonic() - started
        atomic_json(
            output_dir / "done.json",
            {
                "status": "completed",
                "condition": args.condition,
                "episodes_per_task": args.episodes,
                "task_count": len(FULL24_TASKS),
                "elapsed_seconds": elapsed,
                "protocol": str(output_dir / "protocol.json"),
            },
        )
        print(f"Full-24 evaluation completed: {output_dir}")
    except BaseException as exc:
        atomic_json(
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


if __name__ == "__main__":
    main()
