# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The launchers' common interface, checked without a GPU or any asset.

* --arm on all four launchers (baseline, geak, and cae where it exists), with
  the specific refusals; an unknown flag is refused;
* --dry-run on all four, after flag validation and before any asset check;
* --port and the PORT default, the engines' EVAL_PORT and EVAL_ARM_LABEL;
* --suites validated up front, unsupported flags refused with a reason;
* the deprecated environment spellings;
* the common summary.json writer (scripts/eval_summary.py).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = {
    "fastwam": ROOT / "scripts" / "eval_fastwam_libero.sh",
    "native": ROOT / "eval" / "native" / "run_fastwam_libero.sh",
    "dit4dit-libero": ROOT / "scripts" / "eval_dit4dit_libero.sh",
    "dit4dit-robocasa": ROOT / "scripts" / "eval_dit4dit_robocasa.sh",
}
DEPRECATED = "is deprecated and will be removed in the next release"
STAMP = r"\d{8}T\d{6}Z"


def load(path: Path):
    name = f"t_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(name, *args, env=None):
    return subprocess.run(["bash", str(LAUNCHERS[name]), *args], capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", **(env or {})}, timeout=60)


def dry(name, *args, env=None) -> dict[str, str]:
    result = run(name, "--dry-run", *args, env=env)
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
                and not line.startswith(" "))


class DryRunTests(unittest.TestCase):
    def test_every_launcher_dry_runs_without_assets(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                plan = dry(name)
                for key in ("arm", "tree", "steps", "trials", "gpus", "port", "sim", "output"):
                    self.assertIn(key, plan)
                self.assertEqual(plan["arm"], "baseline")

    def test_default_output_layout(self):
        cases = {
            "fastwam": (("--arm", "geak", "--steps", "2"), rf"fastwam-libero-geak-2step-mujoco-{STAMP}"),
            "native": (("--sim", "instinct", "--mode", "batched"),
                       rf"fastwam-libero-baseline-10step-instinct-batched-{STAMP}"),
            "dit4dit-libero": (("--steps", "2"), rf"dit4dit-libero-baseline-2step-mujoco-{STAMP}"),
            "dit4dit-robocasa": (("--arm", "cae"),
                                 rf"dit4dit-robocasa-cae-checkpoint-steps-mujoco-{STAMP}"),
        }
        for name, (args, pattern) in cases.items():
            with self.subTest(launcher=name):
                output = Path(dry(name, *args)["output"])
                self.assertEqual(output.parent, ROOT / "eval" / "outputs")
                self.assertRegex(output.name, f"^{pattern}$")

    def test_explicit_output_dir_is_the_run_directory(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertEqual(dry(name, "--output-dir", "/tmp/x")["output"], "/tmp/x")
                self.assertEqual(dry(name, env={"EVAL_OUTPUT_ROOT": "/tmp/y"})["output"], "/tmp/y")

    def test_dry_run_validates_flags_first(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                result = run(name, "--dry-run", "--sim", "isaac")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--sim must be mujoco or instinct, got: isaac", result.stderr)

    def test_fastwam_dry_run_resolves_the_arm(self):
        plan = dry("fastwam", "--arm", "cae", "--steps", "4")
        self.assertEqual(plan["tree"], str(ROOT / "src" / "cae-fastwam"))
        self.assertEqual(plan["configs"], str(ROOT / "FastWAM" / "configs"))
        self.assertEqual(plan["port"], "8400 (8400 + GPU id)")
        plan = dry("fastwam", "--gpus", "3")
        self.assertEqual(plan["configs"], str(ROOT / "FastWAM" / "configs_baseline"))
        self.assertEqual(plan["port"], "8303 (8300 + GPU id)")
        plan = dry("fastwam", "--arm", "geak")
        self.assertEqual(plan["configs"], str(ROOT / "FastWAM" / "configs_baseline"))


class MujocoGLDryRunTests(unittest.TestCase):
    """--dry-run shows the MuJoCo rendering choice without probing anything."""

    AUTO = "auto (EGL, falls back to OSMesa)"

    def test_auto_by_default_and_an_explicit_setting_is_shown(self):
        for name in LAUNCHERS:
            with self.subTest(launcher=name):
                self.assertEqual(dry(name)["mujoco_gl"], self.AUTO)
                self.assertEqual(dry(name, env={"MUJOCO_GL": "osmesa"})["mujoco_gl"], "osmesa")
                self.assertEqual(dry(name, env={"MUJOCO_GL": "egl"})["mujoco_gl"], "egl")

    def test_instinct_runs_keep_their_rendering(self):
        self.assertEqual(dry("native", "--sim", "instinct")["mujoco_gl"],
                         "osmesa (--sim instinct helper contexts)")
        self.assertEqual(dry("native", "--sim", "instinct", env={"MUJOCO_GL": "egl"})["mujoco_gl"],
                         "osmesa (--sim instinct helper contexts)")
        self.assertEqual(dry("dit4dit-robocasa", "--sim", "instinct")["mujoco_gl"],
                         "osmesa (--sim instinct helper contexts)")
        self.assertEqual(dry("dit4dit-libero", "--sim", "instinct")["mujoco_gl"],
                         "unset (unchanged under --sim instinct)")


class ArmTests(unittest.TestCase):
    def test_arm_resolves_the_tree(self):
        cases = (
            ("fastwam", ("--arm", "baseline"), "src/fastwam-baseline"),
            ("fastwam", ("--arm", "geak"), "src/fastwam-geak"),
            ("fastwam", ("--arm", "cae"), "src/cae-fastwam"),
            ("native", ("--arm", "cae"), "src/cae-fastwam"),
            ("native", ("--arm", "geak"), "src/fastwam-geak"),
            ("dit4dit-libero", ("--arm", "geak"), "src/dit4dit-geak"),
            ("dit4dit-robocasa", ("--arm", "cae"), "DiT4DiT"),
            ("dit4dit-robocasa", ("--arm", "geak"), "src/dit4dit-geak"),
            ("dit4dit-robocasa", (), "src/dit4dit-baseline"),
        )
        for name, args, tree in cases:
            with self.subTest(launcher=name, args=args):
                self.assertEqual(dry(name, *args)["tree"], str(ROOT / tree))

    def test_cae_steps(self):
        self.assertEqual(dry("fastwam", "--arm", "cae")["steps"], "10")
        self.assertEqual(dry("native", "--arm", "cae", "--steps", "4")["steps"], "4")
        self.assertEqual(dry("dit4dit-robocasa", "--arm", "cae")["steps"], "checkpoint-steps")

    def test_arm_refusals(self):
        cases = (
            ("fastwam", ("--nope", "x"), "unknown argument: --nope"),
            ("native", ("--nope", "x"), "unknown argument: --nope"),
            ("dit4dit-libero", ("--nope", "x"), "unknown argument: --nope"),
            ("dit4dit-robocasa", ("--nope", "x"), "unknown argument: --nope"),
            ("fastwam", ("--arm", "nope"), "--arm must be baseline, geak or cae, got: nope"),
            ("native", ("--arm", "nope"), "--arm must be baseline, geak or cae, got: nope"),
            ("dit4dit-robocasa", ("--arm", "nope"), "--arm must be baseline, geak or cae, got: nope"),
            ("dit4dit-libero", ("--arm", "nope"), "--arm must be baseline or geak, got: nope"),
            ("dit4dit-libero", ("--arm", "cae"),
             "--arm cae is the RoboCasa Compact Action Expert policy; DiT4DiT LIBERO has baseline and geak"),
            ("dit4dit-robocasa", ("--arm", "cae", "--steps", "4"), "--steps is not available with --arm cae"),
            ("native", ("--arm", "cae", "--sim", "instinct", "--mode", "batched"),
             "--mode batched supports --arm baseline and geak only"),
        )
        for name, args, expected in cases:
            with self.subTest(launcher=name, args=args):
                result = run(name, "--dry-run", *args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(expected, result.stderr)


class PortTests(unittest.TestCase):
    def test_port_flag_and_env_default(self):
        self.assertEqual(dry("fastwam", "--port", "9000")["port"], "9000")
        self.assertEqual(dry("fastwam", env={"PORT": "9100"})["port"], "9100")
        self.assertEqual(dry("dit4dit-libero", env={"PORT": "9200"})["port"], "9200")
        self.assertEqual(dry("dit4dit-robocasa", "--gpus", "0,1", env={"PORT": "9300"})["port"], "9300,9301")

    def test_base_port_is_a_deprecated_alias(self):
        result = run("dit4dit-robocasa", "--dry-run", "--gpus", "0", env={"BASE_PORT": "9400"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"warning: BASE_PORT {DEPRECATED}; use PORT", result.stderr)
        self.assertIn("port=9400\n", result.stdout)

    def test_bad_port_and_native_refusal(self):
        for name in ("fastwam", "dit4dit-libero", "dit4dit-robocasa"):
            with self.subTest(launcher=name):
                result = run(name, "--dry-run", "--port", "70000")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--port must be an integer from 1 to 65535, got: 70000", result.stderr)
        result = run("native", "--port", "9000")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--port is not available with this launcher: it runs no policy server", result.stderr)


class SuitesAndUnsupportedTests(unittest.TestCase):
    def test_unknown_suite_is_refused_before_anything_starts(self):
        for name in ("fastwam", "native", "dit4dit-libero"):
            with self.subTest(launcher=name):
                result = run(name, "--suites", "libero_spatial,libero_x")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("--suites must be a comma-separated list of", result.stderr)
                self.assertIn("got: libero_spatial,libero_x", result.stderr)
        self.assertEqual(dry("fastwam", "--suites", "libero_goal")["suites"], "libero_goal")

    def test_empty_suites_is_refused_on_every_libero_launcher(self):
        for name in ("fastwam", "native", "dit4dit-libero"):
            with self.subTest(launcher=name):
                result = run(name, "--dry-run", "--suites", "")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("--suites must name at least one suite", result.stderr)

    def test_unsupported_flags_give_a_reason(self):
        cases = (
            ("dit4dit-robocasa", ("--suites", "libero_goal"), "24 task families"),
            ("fastwam", ("--mode", "batched"), "eval/native/run_fastwam_libero.sh"),
            ("dit4dit-libero", ("--num-envs", "5"), "FastWAM's"),
            ("native", ("--clients-per-gpu", "2"), "no policy server"),
            ("native", ("--sim", "instinct", "--mode", "batched", "--record-video"), "writes no rollout videos"),
        )
        for name, args, reason in cases:
            with self.subTest(launcher=name, args=args):
                result = run(name, *args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("is not available with this launcher", result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertNotIn("unknown argument", result.stderr)

    def test_record_video_is_accepted_everywhere(self):
        expected = {"fastwam": "1", "native": "1 (always)", "dit4dit-libero": "1 (always)",
                    "dit4dit-robocasa": "1"}
        for name, value in expected.items():
            with self.subTest(launcher=name):
                self.assertEqual(dry(name, "--record-video")["record_video"], value)


class EnvDefaultTests(unittest.TestCase):
    def test_steps_and_trials_defaults(self):
        env = {"STEPS": "2", "TRIALS": "3", "EVAL_GPUS": "5"}
        for name, steps in (("fastwam", "2"), ("native", "2"), ("dit4dit-libero", "2")):
            with self.subTest(launcher=name):
                plan = dry(name, "--arm", "geak", env=env)
                self.assertEqual((plan["steps"], plan["trials"], plan["gpus"]), (steps, "3", "5"))
        plan = dry("dit4dit-robocasa", "--arm", "geak", env={**env, "TRIALS": "10"})
        self.assertEqual((plan["steps"], plan["trials"], plan["gpus"]), ("2", "10", "5"))

    def test_every_help_lists_the_env_defaults(self):
        for name in LAUNCHERS:
            text = run(name, "--help").stdout
            for var in ("STEPS", "TRIALS", "EVAL_GPUS", "EVAL_OUTPUT_ROOT"):
                with self.subTest(launcher=name, var=var):
                    self.assertIn(var, text)
            if name != "native":
                self.assertIn("PORT", text)
            for flag in ("--arm", "--dry-run", "--record-video"):
                with self.subTest(launcher=name, flag=flag):
                    self.assertIn(flag, text)

    def test_robocasa_deprecated_env_spellings(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness = Path(tmp) / "harness"
            harness.mkdir()  # the plan imports the harness's task registry
            for entry in (ROOT / "DiT4DiT").iterdir():
                (harness / entry.name).symlink_to(entry)
            result = run("dit4dit-robocasa", "--dry-run", "--arm", "cae",
                         env={"DIT4DIT_ROOT": str(harness), "POLICY_200K": "/tmp/p.pt"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"warning: DIT4DIT_ROOT {DEPRECATED}; use DIT4DIT_HARNESS_ROOT", result.stderr)
            self.assertIn(f"warning: POLICY_200K {DEPRECATED}; use DIT4DIT_ROBOCASA_CKPT", result.stderr)
            self.assertIn(f"tree={harness}\nharness={harness}\n", result.stdout)
            # The new name wins silently.
            result = run("dit4dit-robocasa", "--dry-run",
                         env={"DIT4DIT_ROOT": "/tmp/old", "DIT4DIT_HARNESS_ROOT": str(harness)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(DEPRECATED, result.stderr)
            self.assertIn(f"harness={harness}\n", result.stdout)

    def test_dit4dit_libero_tree_override_warns_and_marks_the_directory(self):
        result = run("dit4dit-libero", "--dry-run", "--arm", "geak",
                     env={"DIT4DIT_ROOT": str(ROOT / "src" / "dit4dit-baseline")})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("warning: DIT4DIT_ROOT=", result.stderr)
        self.assertIn("which --arm geak selects", result.stderr)
        self.assertRegex(result.stdout, rf"output=.*-geak-4step-mujoco-custom-tree-{STAMP}\n")

    def test_dit4dit_libero_python_spellings(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as ckpt:
            base = {"DIT4DIT_LIBERO_CKPT": ckpt.name}
            old = run("dit4dit-libero", env={**base, "MODEL_PYTHON": "python3"})
            self.assertIn(f"warning: MODEL_PYTHON {DEPRECATED}; use DIT4DIT_MODEL_PYTHON", old.stderr)
            # The umbrella fills both when no old spelling is set.
            new = run("dit4dit-libero", env={**base, "DIT4DIT_PYTHON": "python3"})
            self.assertNotIn(DEPRECATED, new.stderr)
            self.assertIn("missing required environment variable LIBERO_ROOT: set it to", new.stderr)
            # An old spelling keeps its interpreter over the umbrella, with a warning.
            both = run("dit4dit-libero", env={**base, "DIT4DIT_PYTHON": "python3",
                                              "LIBERO_ROOT": "/tmp",
                                              "LIBERO_PYTHON": "/nope/libero-python"})
            self.assertEqual(both.returncode, 2, both.stdout)
            self.assertIn(f"warning: LIBERO_PYTHON {DEPRECATED}; use DIT4DIT_LIBERO_PYTHON", both.stderr)
            self.assertIn("DIT4DIT_LIBERO_PYTHON must name an executable: /nope/libero-python", both.stderr)

    def test_dit4dit_missing_variables_use_the_shared_message(self):
        cases = (
            ("dit4dit-libero", {}, "DIT4DIT_LIBERO_CKPT"),
            ("dit4dit-robocasa", {}, "DIT4DIT_ROBOCASA_CKPT"),
            ("dit4dit-robocasa", {"DIT4DIT_ROBOCASA_CKPT": "/tmp/p.pt"}, "ROBOSUITE_OVERLAY"),
        )
        for name, env, var in cases:
            with self.subTest(launcher=name, var=var):
                result = run(name, env=env)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(f"missing required environment variable {var}: set it to", result.stderr)

    def test_robocasa_dry_run_refuses_a_tree_that_is_not_the_harness(self):
        result = run("dit4dit-robocasa", "--dry-run", env={"DIT4DIT_HARNESS_ROOT": "/tmp"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("DIT4DIT_HARNESS_ROOT must name the DiT4DiT checkout (the full-24 harness), got: /tmp",
                      result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class EngineLabelAndPortTests(unittest.TestCase):
    """The engines take EVAL_PORT and EVAL_ARM_LABEL from the launcher."""

    def sandbox(self, td: Path) -> dict:
        bin_dir = td / "bin"
        bin_dir.mkdir()
        (bin_dir / "docker").write_text("#!/usr/bin/env bash\nexit 1\n")
        (bin_dir / "docker").chmod(0o755)
        for name in ("harness", "libero", "home", "configs/task"):
            (td / name).mkdir(parents=True)
        for name in ("model.pt", "vae.pt", "stats.json", "stats_libero_goal.json"):
            (td / name).write_text("x\n")
        (td / "configs" / "task" / "cae_libero_goal.yaml").write_text("{}\n")
        return {
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin", "HOME": str(td / "home"),
            "EVAL_IMAGE": "img", "EVAL_HARNESS": str(td / "harness"), "LIBERO_ROOT": str(td / "libero"),
            "MODEL_CKPT": str(td / "model.pt"), "DATASET_STATS": str(td / "stats.json"),
            "TORCHINDUCTOR_CACHE_DIR": str(td / "home" / "i"), "TRITON_CACHE_DIR": str(td / "home" / "t"),
        }

    def test_fastwam_geak_results_are_named_after_the_arm_and_summarized(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            env = self.sandbox(td)
            out = td / "out"
            result = subprocess.run(
                ["bash", str(LAUNCHERS["fastwam"]), "--arm", "geak", "--smoke", "--output-dir", str(out),
                 "--port", "9000"], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stderr)  # the stand-in docker fails
            self.assertTrue((out / "geak_10step" / ".run_identity").is_file())
            self.assertFalse((out / "optimized_10step").exists())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual((summary["model"], summary["arm"], summary["steps"]),
                             ("fastwam", "geak", 10))
            self.assertEqual(summary["status"], "incomplete")
            self.assertIn(f"eval incomplete: fastwam libero geak 10step mujoco: 0/0 -> {out}",
                          result.stdout)

    def test_cae_fastwam_engine_takes_the_port(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            env = {**self.sandbox(td), "ACTION_VAE_CKPT": str(td / "vae.pt"),
                   "MODEL_CONFIGS": str(td / "configs"), "TASK_CONFIG": "cae",
                   "DATASET_STATS_TEMPLATE": str(td / "stats_{suite}.json"),
                   "PORT": "9123"}
            out = td / "out"
            result = subprocess.run(
                ["bash", str(LAUNCHERS["fastwam"]), "--arm", "cae", "--smoke",
                 "--suites", "libero_goal", "--output-dir", str(out)],
                env=env, capture_output=True, text=True, timeout=60)
            self.assertIn("[cae_10step] start gpu=0 port=9123", result.stdout)
            self.assertIn(f"tree={ROOT / 'src' / 'cae-fastwam'}", result.stdout)
            self.assertTrue((out / "cae_10step").is_dir())

    def test_refused_resume_writes_no_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            env = self.sandbox(td)
            cell = td / "out" / "baseline_10step"
            cell.mkdir(parents=True)
            (cell / "LIBEROBenchmark_libero_goal_shard0.json").write_text("{}")
            result = subprocess.run(["bash", str(LAUNCHERS["fastwam"]), "--output-dir", str(td / "out")],
                                    env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse((td / "out" / "summary.json").exists())


class EvalFlagHelperTests(unittest.TestCase):
    def call(self, snippet: str):
        return subprocess.run(["bash", "-c", f'source "$1"; {snippet}', "_",
                               str(ROOT / "scripts" / "eval_flags.sh")], capture_output=True, text=True)

    def test_one_refusal_format(self):
        cases = (
            ("eval_check_choice --mode x a b c", "--mode must be a, b or c, got: x"),
            ("eval_check_choice --sim x mujoco instinct", "--sim must be mujoco or instinct, got: x"),
            ("eval_check_posint --trials 0", "--trials must be a positive integer, got: 0"),
            ("eval_check_port --port 65535 2", "--port must be an integer from 1 to 65535, got: 65535"),
            ("eval_require NOPE_VAR 'a thing'", "missing required environment variable NOPE_VAR: set it to a thing"),
            ("eval_unsupported --x why", "--x is not available with this launcher: why"),
        )
        for snippet, message in cases:
            with self.subTest(snippet=snippet):
                result = self.call(snippet)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stderr.strip(), message)
        self.assertEqual(self.call("eval_check_suites libero_goal,libero_10").returncode, 0)
        self.assertEqual(self.call("eval_check_port --port 8098 6").returncode, 0)

    def test_env_alias(self):
        result = self.call('OLD=v; eval_env_alias NEW OLD; echo "$NEW"')
        self.assertEqual(result.stdout.strip(), "v")
        self.assertIn(f"warning: OLD {DEPRECATED}; use NEW", result.stderr)


class SummaryWriterTests(unittest.TestCase):
    def setUp(self):
        self.summary = load(ROOT / "scripts" / "eval_summary.py")

    def test_harness_and_native_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            (td / "LIBEROBenchmark_libero_goal_shard0of2.json").write_text(json.dumps(
                {"tasks": [{"num_episodes": 2, "mean_success": 0.5}, {"num_episodes": 0}]}))
            (td / "LIBEROBenchmark_libero_goal_shard1of2.json").write_text(json.dumps(
                {"tasks": [{"num_episodes": 2, "mean_success": 1.0}]}))
            self.assertEqual(self.summary.harness_counts(td), {"libero_goal": (3, 4)})
            (td / "libero_10").mkdir()
            for i, s in enumerate((1, 2)):
                (td / "libero_10" / f"gpu0_task{i}_results.json").write_text(
                    json.dumps({"successes": s, "total_episodes": 2}))
            self.assertEqual(self.summary.native_counts(td), {"libero_10": (3, 4)})

    def test_counts_and_robocasa(self):
        self.assertEqual(self.summary.parse_counts(["a=1/2", "b=0/0"]), {"a": (1, 2), "b": (0, 0)})
        with self.assertRaises(ValueError):
            self.summary.parse_counts(["a=1"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(json.dumps({"tasks": [{"label": "cup", "successes": 3, "episodes": 5}]}))
            self.assertEqual(self.summary.robocasa_counts(path), {"cup": (3, 5)})

    def test_main_writes_summary_json_and_the_closing_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "eval_summary.py"), "--model", "dit4dit",
                 "--bench", "libero", "--arm", "geak", "--steps", "4", "--sim", "mujoco",
                 "--trials", "2", "--output-dir", tmp, "--counts", "libero_goal=3/4", "libero_10=1/4"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(),
                             f"eval complete: dit4dit libero geak 4step mujoco: 4/8 -> {tmp}")
            payload = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertEqual((payload["successes"], payload["episodes"], payload["success_rate"]), (4, 8, 0.5))
            self.assertEqual(payload["per_suite"]["libero_goal"],
                             {"successes": 3, "episodes": 4, "success_rate": 0.75})
            self.assertEqual(payload["status"], "complete")

    def test_incomplete_and_non_numeric_steps(self):
        summary = self.summary.build_summary(
            {"model": "dit4dit", "benchmark": "robocasa", "arm": "cae",
             "steps": "checkpoint-steps", "sim": "mujoco", "mode": None, "output_dir": "/o"},
            {}, "task", complete=False)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(self.summary.closing_line(summary),
                         "eval incomplete: dit4dit robocasa cae checkpoint-steps mujoco: 0/0 -> /o")


class CompareArtifactsArmTests(unittest.TestCase):
    def setUp(self):
        self.compare = load(ROOT / "scripts" / "compare_artifacts.py")

    def test_at_steps_and_the_deprecated_reduced_arm(self):
        self.assertEqual(self.compare.parse_arm("geak@2", 10, 2), ("geak@2", "geak", 2))
        self.assertEqual(self.compare.parse_arm("eager", 10, 2), ("eager", "eager", 10))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(self.compare.parse_arm("geak-reduced", 10, 2),
                             ("geak-reduced", "geak-reduced", 2))
        self.assertIn(f"the geak-reduced arm {DEPRECATED}; use geak@<steps>", err.getvalue())
        for bad in ("geak@0", "geak@x", "nope", "geak-reduced@2"):
            with self.subTest(arm=bad), self.assertRaises(ValueError):
                self.compare.parse_arm(bad, 10, 2)

    def test_default_trees_are_the_fastwam_baseline_and_geak_trees(self):
        saved = {k: os.environ.pop(k, None) for k in ("BASELINE_SRC", "OPTIMIZED_SRC")}
        try:
            self.assertEqual(self.compare.arm_src("eager"), str(ROOT / "src" / "fastwam-baseline"))
            self.assertEqual(self.compare.arm_src("geak"), str(ROOT / "src" / "fastwam-geak"))
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_baseline_and_geak_trees_are_timed_with_the_baseline_configs(self):
        saved = os.environ.pop("FWE_CONFIG_DIR", None)
        try:
            for arm in ("eager", "compile", "geak"):
                env = self.compare.arm_env(arm, 10, "0", self.compare.arm_src(arm), {})
                self.assertEqual(env["FWE_CONFIG_DIR"], str(ROOT / "FastWAM" / "configs_baseline"))
            for other_src in (ROOT / "FastWAM" / "src", ROOT / "src" / "cae-fastwam"):
                other = self.compare.arm_env("eager", 10, "0", str(other_src), {})
                self.assertNotIn("FWE_CONFIG_DIR", other)
            os.environ["FWE_CONFIG_DIR"] = "/cfg"
            explicit = self.compare.arm_env("eager", 10, "0", self.compare.arm_src("eager"), {})
            self.assertEqual(explicit["FWE_CONFIG_DIR"], "/cfg")
        finally:
            os.environ.pop("FWE_CONFIG_DIR", None)
            if saved is not None:
                os.environ["FWE_CONFIG_DIR"] = saved

    def test_each_geak_cell_gets_its_own_cache(self):
        env = self.compare.arm_env("geak", 2, "0", "/s", {"geak@2": "/c2", "geak": "/c"}, key="geak@2")
        self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], "/c2/inductor")
        self.assertEqual(env["FASTWAM_KVSLAB"], "0")

    def test_naive_compile_arm_is_baseline_steps_only_with_its_own_cache(self):
        self.assertEqual(self.compare.parse_arm("compile", 10, 2), ("compile", "compile", 10))
        self.assertEqual(self.compare.parse_arm("compile@10", 10, 2),
                         ("compile@10", "compile", 10))
        with self.assertRaisesRegex(ValueError, "baseline step count only"):
            self.compare.parse_arm("compile@2", 10, 2)
        self.assertIn("compile", self.compare.FRESH_CACHE_ARMS)
        env = self.compare.arm_env("compile", 10, "0", "/s", {"compile": "/cc", "geak": "/c"})
        self.assertEqual(env["TORCHINDUCTOR_CACHE_DIR"], "/cc/inductor")
        self.assertEqual(env["TRITON_CACHE_DIR"], "/cc/triton")
        self.assertEqual(env["FWE_NAIVE_COMPILE"], "1")
        for k in ("FASTWAM_COMPILE", "FASTWAM_VAE_COMPILE"):
            self.assertEqual(env[k], "0")
        self.assertEqual(env["FASTWAM_MOT_COMPILE_MODE"], "off")

    def test_episode_perf_probes_either_prefill_api_and_checks_compilation(self):
        text = (ROOT / "geak" / "task" / "scripts" / "episode_perf.py").read_text()
        for needle in ('"prefill_video_cache_tensor"', '"prefill_video_cache"',
                       "video_context_payload", "has_stage_hooks",
                       "setattr(model.mot, probe_name, probe_orig)",
                       "produced no graphs", "recompiled"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
