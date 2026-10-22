# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The RoboCasa full-24 driver's client split, checked without a simulator or model.

With K clients per policy server, every (task, episode) must be played exactly
once, by a client of the server run_full24_evaluation.py assigns the task to,
and reset with the seed it has when one client plays the whole task. The seed
check runs the vendored episode loop (full24_rollout_matching.
run_matched_full24_episodes) against a stand-in env whose outcome is a
function of the reset seed.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "DiT4DiT"
EVAL_FILES = VENDORED / "examples" / "Robocasa_tabletop" / "eval_files"

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


driver = load_file("robocasa_full24", ROOT / "scripts" / "robocasa_full24.py")
client = load_file("robocasa_full24_client", ROOT / "scripts" / "robocasa_full24_client.py")


class VendoredModules:
    """Load the vendored task registry and rollout-matching module by file.

    The real package path imports DiT4DiT.dataloader, whose __init__ needs
    torch; both files themselves need only the standard library and numpy.
    The stand-in packages are removed again afterwards.
    """

    PACKAGES = {
        "DiT4DiT": VENDORED / "DiT4DiT",
        "DiT4DiT.dataloader": VENDORED / "DiT4DiT" / "dataloader",
        "DiT4DiT.dataloader.gr00t_lerobot": VENDORED / "DiT4DiT" / "dataloader" / "gr00t_lerobot",
        "examples": VENDORED / "examples",
        "examples.Robocasa_tabletop": VENDORED / "examples" / "Robocasa_tabletop",
        "examples.Robocasa_tabletop.eval_files": EVAL_FILES,
    }
    TASKS = "DiT4DiT.dataloader.gr00t_lerobot.full24_tasks"
    MATCHING = "examples.Robocasa_tabletop.eval_files.full24_rollout_matching"

    def __enter__(self):
        names = list(self.PACKAGES) + [self.TASKS, self.MATCHING]
        self.saved = {name: sys.modules.get(name) for name in names}
        for name, path in self.PACKAGES.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        tasks = load_file(self.TASKS, VENDORED / "DiT4DiT" / "dataloader" / "gr00t_lerobot" / "full24_tasks.py")
        matching = load_file(self.MATCHING, EVAL_FILES / "full24_rollout_matching.py")
        return tasks.FULL24_TASKS, matching

    def __exit__(self, *exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class SeededEnv:
    """A one-env vector env whose episode is a deterministic function of its reset seed."""

    def __init__(self):
        self.seed = None
        self.steps = 0

    def reset(self, seed=None):
        self.seed, self.steps = seed, 0
        return {
            "annotation.human.coarse_action": [[f"task-{seed % 97}"]],
            "state.left_arm": np.full((1, 1, 7), seed % 1009, dtype=np.float64),
        }, {}

    def step(self, _actions):
        self.steps += 1
        done = self.steps >= 1 + self.seed % 4
        obs = {"annotation.human.coarse_action": [["next"]], "state.left_arm": np.zeros((1, 1, 7))}
        info = {"success": [[done and self.seed % 3 == 0]]}
        return obs, None, [done], [False], info


def play(matching, task, first, count, eval_seed):
    """What a client playing episodes first..first+count-1 records (first=0, count=E: one whole-task client)."""
    run = client.episode_range_runner(matching, first) if first else matching.run_matched_full24_episodes
    successes, records = run(
        env=SeededEnv(),
        get_actions=lambda obs: {},
        env_name=task.env_name,
        n_episodes=count,
        base_eval_seed=eval_seed,
    )
    return successes, records


def share_payload(matching, task, records, successes, eval_seed):
    """simulation_env.py's result payload."""
    return {
        "schema_version": 2,
        "env_name": task.env_name,
        "task_id": task.task_id,
        "label": task.label,
        "group": task.group,
        "seed": eval_seed,
        "rollout_matching": matching.ROLLOUT_MATCHING_PROTOCOL,
        "episodes": records,
        "success_rate": float(np.mean(successes)),
    }


class PlanTests(unittest.TestCase):
    TASKS = [types.SimpleNamespace(label=f"t{i}") for i in range(24)]

    def test_every_task_episode_is_assigned_exactly_once_to_its_server(self):
        for servers, clients, episodes in itertools.product((1, 2, 3, 5, 6, 8), (1, 2, 3, 6, 7, 13, 64), (1, 10, 50)):
            with self.subTest(servers=servers, clients=clients, episodes=episodes):
                plan = driver.plan_clients(self.TASKS, servers, clients, episodes)
                self.assertEqual(len(plan), servers)
                seen = []
                for server, server_clients in enumerate(plan):
                    self.assertEqual(len(server_clients), clients)
                    sizes = [sum(s.count for s in segments) for segments in server_clients]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)
                    for segments in server_clients:
                        for segment in segments:
                            # run_full24_evaluation.py's assignment: server g plays tasks[g::G].
                            self.assertIn(segment.task, self.TASKS[server::servers])
                            self.assertGreater(segment.count, 0)
                            seen += [(segment.task.label, segment.first + i) for i in range(segment.count)]
                want = [(task.label, e) for task in self.TASKS for e in range(episodes)]
                self.assertEqual(sorted(seen), sorted(want))
                self.assertEqual(len(seen), len(set(seen)))

    def test_one_client_per_server_is_the_runner_assignment(self):
        for servers in (1, 2, 3, 6, 8):
            plan = driver.plan_clients(self.TASKS, servers, 1, 50)
            for server, (segments,) in enumerate(plan):
                self.assertEqual([s.task for s in segments], self.TASKS[server::servers])
                self.assertTrue(all(s.whole for s in segments))


@unittest.skipIf(np is None, "numpy is required")
class SeedTests(unittest.TestCase):
    def test_k_way_shares_reproduce_the_one_client_result_files(self):
        eval_seed = 7
        with VendoredModules() as (tasks, matching):
            original_seed = matching.full24_episode_seed
            for episodes in (1, 10, 50):
                whole = {}
                for task in tasks:
                    successes, records = play(matching, task, 0, episodes, eval_seed)
                    whole[task.label] = share_payload(matching, task, records, successes, eval_seed)
                    # The K=1 file carries the protocol's seeds, as summarize_full24_policy.py checks.
                    self.assertEqual(
                        [r["episode_seed"] for r in records],
                        [original_seed(eval_seed, task.env_name, i) for i in range(episodes)],
                    )
                for servers, clients in ((6, 2), (6, 6), (8, 6), (1, 7), (3, 64)):
                    with self.subTest(episodes=episodes, servers=servers, clients=clients):
                        shares = {}
                        for server_clients in driver.plan_clients(tasks, servers, clients, episodes):
                            for segments in server_clients:
                                for segment in segments:
                                    successes, records = play(
                                        matching, segment.task, segment.first, segment.count, eval_seed
                                    )
                                    shares.setdefault(segment.task.label, []).append(
                                        share_payload(matching, segment.task, records, successes, eval_seed)
                                    )
                        for task in tasks:
                            merged = driver.merge_shares(
                                task,
                                shares[task.label],
                                episodes_per_task=episodes,
                                eval_seed=eval_seed,
                                rollout_matching=matching.ROLLOUT_MATCHING_PROTOCOL,
                                episode_seed=matching.full24_episode_seed,
                            )
                            self.assertEqual(merged, whole[task.label])
                            self.assertEqual(list(merged), list(whole[task.label]))
                        # The seed function is restored after every share.
                        self.assertIs(matching.full24_episode_seed, original_seed)

    def test_merge_refuses_gaps_overlaps_and_foreign_seeds(self):
        eval_seed = 7
        with VendoredModules() as (tasks, matching):
            task = tasks[0]

            def share(first, count):
                successes, records = play(matching, task, first, count, eval_seed)
                return share_payload(matching, task, records, successes, eval_seed)

            kwargs = dict(
                episodes_per_task=10,
                eval_seed=eval_seed,
                rollout_matching=matching.ROLLOUT_MATCHING_PROTOCOL,
                episode_seed=matching.full24_episode_seed,
            )
            driver.merge_shares(task, [share(0, 4), share(4, 6)], **kwargs)
            with self.assertRaisesRegex(ValueError, "cover episodes"):
                driver.merge_shares(task, [share(0, 4), share(5, 5)], **kwargs)
            with self.assertRaisesRegex(ValueError, "cover episodes"):
                driver.merge_shares(task, [share(0, 5), share(4, 6)], **kwargs)
            bad = share(4, 6)
            bad["episodes"][0]["episode_seed"] += 1
            with self.assertRaisesRegex(ValueError, "seed"):
                driver.merge_shares(task, [share(0, 4), bad], **kwargs)
            with self.assertRaisesRegex(ValueError, "unexpected env_name"):
                driver.merge_shares(tasks[1], [share(0, 4), share(4, 6)], **kwargs)

    def test_share_resume_check(self):
        import json
        import tempfile

        eval_seed = 7
        with VendoredModules() as (tasks, matching), tempfile.TemporaryDirectory() as tmp:
            task = tasks[3]
            segment = driver.Segment(task, 5, 3, 10)
            successes, records = play(matching, task, 5, 3, eval_seed)
            path = Path(tmp) / f"{segment.stem}.json"
            path.write_text(json.dumps(share_payload(matching, task, records, successes, eval_seed)))
            check = (eval_seed, matching.ROLLOUT_MATCHING_PROTOCOL, matching.full24_episode_seed)
            self.assertTrue(driver.share_is_complete(path, segment, *check))
            self.assertFalse(driver.share_is_complete(path, driver.Segment(task, 4, 3, 10), *check))
            self.assertFalse(driver.share_is_complete(Path(tmp) / "missing.json", segment, *check))


class LauncherTests(unittest.TestCase):
    SCRIPT = ROOT / "scripts" / "eval_dit4dit_robocasa.sh"

    def run_script(self, *args):
        return subprocess.run(["bash", str(self.SCRIPT), *args], capture_output=True, text=True)

    def test_dry_run_prints_the_split_without_runtime_assets(self):
        result = self.run_script("--dry-run", "--gpus", "0,1", "--clients-per-gpu", "3", "--trials", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        clients = [line for line in result.stdout.splitlines() if line.startswith("  client ")]
        self.assertEqual(len(clients), 6)
        self.assertEqual(sum(int(line.split(": ")[1].split()[0]) for line in clients), 240)

    def test_summary_requires_no_fixed_success_count(self):
        # Results live outside the repository, so the launcher never checks a success count.
        text = self.SCRIPT.read_text()
        self.assertNotIn("--expected-successes", text)
        for args in ((), ("--arm", "cae"), ("--arm", "geak")):
            with self.subTest(args=args):
                result = self.run_script("--dry-run", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("result gate", result.stdout)

    def test_no_result_gate_is_a_deprecated_no_op(self):
        plain = self.run_script("--dry-run", "--arm", "cae", "--output-dir", "/tmp/x")
        result = self.run_script("--dry-run", "--arm", "cae", "--output-dir", "/tmp/x", "--no-result-gate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--no-result-gate is deprecated and has no effect", result.stderr)
        self.assertEqual(result.stdout, plain.stdout)

    def test_default_arm_is_baseline(self):
        result = self.run_script("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        for line in ("arm=baseline", "steps=4"):
            self.assertIn(line + "\n", result.stdout)

    def test_arm_selects_the_server_tree(self):
        for args, tree in ((("--arm", "cae"), "DiT4DiT"),
                           (("--arm", "geak"), "src/dit4dit-geak"),
                           (("--arm", "baseline"), "src/dit4dit-baseline")):
            with self.subTest(args=args):
                result = self.run_script("--dry-run", *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"tree={ROOT / tree}\n", result.stdout)

    def test_cae_runs_with_the_checkpoint_steps(self):
        result = self.run_script("--dry-run", "--arm", "cae", "--output-dir", "/tmp/x")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("arm=cae\n", result.stdout)
        self.assertIn("steps=checkpoint-steps\n", result.stdout)
        self.assertNotIn("deprecated", result.stderr)

    def test_removed_spellings_are_refused(self):
        unknown = self.run_script("--dry-run", "--nope", "x")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown argument: --nope", unknown.stderr)
        arm = self.run_script("--dry-run", "--arm", "nope")
        self.assertEqual(arm.returncode, 2)
        self.assertIn("--arm must be baseline, geak or cae, got: nope", arm.stderr)

    def test_cae_refuses_steps(self):
        steps = self.run_script("--dry-run", "--arm", "cae", "--steps", "2")
        self.assertEqual(steps.returncode, 2)
        self.assertIn("--steps is not available with --arm cae", steps.stderr)
        # A STEPS export is a default, not a request: noted and ignored.
        env = {**os.environ, "STEPS": "2"}
        result = subprocess.run(["bash", str(self.SCRIPT), "--dry-run", "--arm", "cae"],
                                capture_output=True, text=True, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("note: ignoring STEPS=2", result.stderr)

class PortTests(unittest.TestCase):
    def test_a_port_another_run_listens_on_is_refused(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            busy.bind(("0.0.0.0", 0))
            busy.listen()
            port = busy.getsockname()[1]
            with self.assertRaises(SystemExit) as cm:
                driver.check_ports_free([port])
            self.assertIn(f"port {port} is unavailable", str(cm.exception))
        driver.check_ports_free([port])  # free again once nothing listens


if __name__ == "__main__":
    unittest.main()
