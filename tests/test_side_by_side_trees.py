# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards for the side-by-side source trees and the arm/steps matrix.

These assert layout and refusal behaviour only. Nothing here runs a model.
"""
from __future__ import annotations

import contextlib
import io
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Every tree an --arm resolves to.
TREES = (
    "fastwam-baseline",
    "fastwam-geak",
    "cae-fastwam",
    "dit4dit-baseline",
    "dit4dit-geak",
)
WEIGHT_SUFFIXES = {
    ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".npz", ".pkl", ".so",
}
GEAK_EDITABLES = (
    "action_dit.py", "mot.py", "wan_video_dit.py", "wan_video_vae.py",
)


class TreeLayoutTests(unittest.TestCase):
    def test_every_referenced_tree_exists(self):
        for tree in TREES:
            with self.subTest(tree=tree):
                self.assertTrue((SRC / tree).is_dir(), f"missing tree {tree}")

    def test_fastwam_trees_contain_the_package_directly(self):
        # BASELINE_SRC names the directory CONTAINING the package, so an extra
        # level of nesting would break --fastwam_src silently.
        for tree in TREES:
            if "fastwam" not in tree:
                continue
            with self.subTest(tree=tree):
                self.assertTrue((SRC / tree / "fastwam" / "__init__.py").is_file())

    def test_dit4dit_trees_contain_the_policy_server(self):
        for tree in ("dit4dit-baseline", "dit4dit-geak"):
            with self.subTest(tree=tree):
                server = SRC / tree / "deployment" / "model_server" / "server_policy.py"
                self.assertTrue(server.is_file())

    def test_geak_trees_differ_from_their_source_in_exactly_four_files(self):
        # FastWAM GEAK against the FastWAM baseline tree; the CAE-FastWAM tree
        # against the vendored FastWAM/src/fastwam code it was optimized from.
        pairs = (
            (SRC / "fastwam-baseline", SRC / "fastwam-geak"),
            (ROOT / "FastWAM" / "src" / "fastwam", SRC / "cae-fastwam" / "fastwam"),
        )
        for base, geak in pairs:
            with self.subTest(pair=(base.relative_to(ROOT), geak.relative_to(ROOT))):
                # Each tree's README describes that tree, so it differs by
                # design; the four-file claim is about source.
                result = subprocess.run(
                    ["diff", "-rq", "-x", "README.md", "-x", "__pycache__",
                     str(base), str(geak)],
                    capture_output=True,
                    text=True,
                )
                lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
                self.assertEqual(len(lines), 4, result.stdout)
                for line in lines:
                    self.assertTrue(
                        any(name in line for name in GEAK_EDITABLES),
                        f"unexpected difference: {line}",
                    )
                    self.assertIn("models/wan22/", line)

    def test_geak_diffusers_patches_are_committed(self):
        # Part of the GEAK arm lives in diffusers, not in the tree, so the arm
        # ships incomplete without both patches. The runner applies every
        # *.patch in this directory, so the set is pinned by name.
        patches = sorted(
            p.name for p in (SRC / "dit4dit-geak" / "patches").glob("*.patch"))
        self.assertEqual(patches, [
            "diffusers-3996788-autoencoder_kl_wan-geak.patch",
            "diffusers-3996788-transformer_cosmos-geak.patch",
        ])
        for name in patches:
            text = (SRC / "dit4dit-geak" / "patches" / name).read_text()
            self.assertTrue(text.startswith("--- a/diffusers/models/"), name)

    def test_dit4dit_geak_tree_vs_upstream_baseline(self):
        # The GEAK tree is upstream plus earlier inference work, the GEAK model
        # files, a LIBERO client that takes a task range, training-side drift,
        # and (excluded here) the diffusers patches. src/README.md describes
        # each entry; this pins the whole manifest so it cannot drift silently.
        result = subprocess.run(
            ["diff", "-rq", "-x", "README.md", "-x", "patches", "-x", "__pycache__",
             str(SRC / "dit4dit-baseline"), str(SRC / "dit4dit-geak")],
            capture_output=True,
            text=True,
        )
        baseline, geak = SRC / "dit4dit-baseline", SRC / "dit4dit-geak"
        changed, only_baseline, only_geak = [], [], []
        for line in result.stdout.splitlines():
            if line.startswith("Files "):
                path = Path(line[len("Files "):].split(" and ")[0])
                changed.append(str(path.relative_to(baseline)))
            elif line.startswith("Only in "):
                where, name = line[len("Only in "):].split(": ", 1)
                where = Path(where)
                if where.is_relative_to(baseline):
                    only_baseline.append(str(where.relative_to(baseline) / name))
                else:
                    only_geak.append(str(where.relative_to(geak) / name))
            elif line.strip():
                self.fail(f"unexpected diff output: {line}")
        self.assertEqual(sorted(changed), [
            "DiT4DiT/config/deepseeds/ds_config.yaml",
            "DiT4DiT/dataloader/__init__.py",
            "DiT4DiT/dataloader/gr00t_lerobot/data_config.py",
            "DiT4DiT/dataloader/gr00t_lerobot/datasets.py",
            "DiT4DiT/dataloader/gr00t_lerobot/mixtures.py",
            "DiT4DiT/dataloader/gr00t_lerobot/video.py",
            "DiT4DiT/model/framework/DiT4DiT.py",
            "DiT4DiT/model/modules/action_model/ActionDiT.py",
            "DiT4DiT/model/modules/vlm/Cosmos25.py",
            "DiT4DiT/model/modules/vlm/__init__.py",
            "DiT4DiT/training/train.py",
            "DiT4DiT/training/trainer_utils/trainer_tools.py",
            "examples/LIBERO/eval_files/eval_libero.py",
            "examples/LIBERO/eval_files/model2libero_interface.py",
        ], result.stdout)
        self.assertEqual(sorted(only_geak), [
            "DiT4DiT/config/libero/cosmos25_libero_policy.yaml",
            "DiT4DiT/model/modules/action_latent",
            "deployment/model_server/server_policy_split.py",
        ], result.stdout)
        self.assertEqual(sorted(only_baseline), [
            "DiT4DiT/config/real_robot/dit4dit_g1_decoupled_wbc.yaml",
            "deployment/model_server/server_policy_zmq.py",
            "deployment/model_server/tools/zmq_policy_server.py",
            "examples/LIBERO/train_files/modality.json",
            "examples/Real_G1/train_files/run_decoupled_wbc.sh",
        ], result.stdout)

    def test_dit4dit_trees_carry_no_internal_paths(self):
        offenders = []
        for tree in ("dit4dit-baseline", "dit4dit-geak"):
            for p in (SRC / tree).rglob("*.py"):
                if "/mnt/" in p.read_text(errors="replace"):
                    offenders.append(str(p.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_gradient_checkpointing_is_training_only(self):
        # Under torch.inference_mode() checkpointed blocks make predict_action
        # raise, so serving must never enable it.
        text = (SRC / "dit4dit-geak" / "DiT4DiT" / "model" / "modules" / "vlm"
                / "Cosmos25.py").read_text()
        self.assertIn(
            "        if self.trainable:\n"
            "            self.transformer.enable_gradient_checkpointing()",
            text,
        )
        # Upstream never enables it.
        for p in (SRC / "dit4dit-baseline" / "DiT4DiT").rglob("*.py"):
            with self.subTest(path=str(p.relative_to(SRC))):
                self.assertNotIn("enable_gradient_checkpointing(", p.read_text(errors="replace"))

    def test_no_weights_or_bytecode_are_committed(self):
        offenders = [
            str(p.relative_to(ROOT))
            for p in SRC.rglob("*")
            if p.is_file() and p.suffix in WEIGHT_SUFFIXES
        ]
        self.assertEqual(offenders, [])

    def test_no_pycache_is_committed(self):
        offenders = [str(p.relative_to(ROOT)) for p in SRC.rglob("__pycache__")]
        self.assertEqual(offenders, [])


class NativeLauncherArmTests(unittest.TestCase):
    """eval/native/run_fastwam_libero.sh maps each arm to a whole tree, and the harness engines are gone."""

    def setUp(self):
        self.script = (ROOT / "eval" / "native" / "run_fastwam_libero.sh").read_text()

    def test_each_arm_is_a_whole_tree(self):
        for arm, tree in (("baseline", "src/fastwam-baseline"), ("geak", "src/fastwam-geak"),
                          ("cae", "src/cae-fastwam")):
            self.assertIn(f"{arm}) SRC={tree} ;;", self.script)

    def test_compile_caches_are_per_worker_and_forwarded(self):
        self.assertIn('-e "TORCHINDUCTOR_CACHE_DIR=$cache/inductor" -e "TRITON_CACHE_DIR=$cache/triton"', self.script)
        self.assertIn("geak) ARMENV=(-e FASTWAM_KVSLAB=0)", self.script)

    def test_no_policy_server_engine_remains(self):
        for name in ("run_eval.sh", "run_eval_cae_fastwam.sh"):
            self.assertFalse((ROOT / "eval" / name).exists(), name)
        for path in [ROOT / "scripts" / "eval_fastwam_libero.sh", ROOT / "docker" / "Dockerfile.eval",
                     ROOT / "geak" / "task" / "run_local.sh", ROOT / "geak" / "task" / "scripts" / "task_runner.py"]:
            text = path.read_text()
            for word in ("vla_eval", "vla-evaluation-harness", "EVAL_HARNESS", "run_eval"):
                self.assertNotIn(word, text, f"{path.name}: {word}")


class MatrixRefusalTests(unittest.TestCase):
    def run_script(self, name, *args):
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / name), *args],
            capture_output=True,
            text=True,
        )

    def test_new_scripts_help_exits_zero_without_runtime_assets(self):
        for name in (
            "eval_fastwam_libero.sh",
            "eval_dit4dit_libero.sh",
            "eval_dit4dit_robocasa.sh",
        ):
            with self.subTest(script=name):
                result = self.run_script(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_fastwam_help_exits_zero(self):
        result = subprocess.run(
            ["bash", str(ROOT / "eval" / "native" / "run_fastwam_libero.sh"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_meaningless_cells_are_refused_with_a_reason(self):
        cases = (
            ("eval_dit4dit_robocasa.sh", ("--arm", "cae", "--steps", "2"), "not available with --arm cae"),
            ("eval_dit4dit_robocasa.sh", ("--arm", "geak", "--steps", "10"), "not a DiT4DiT cell"),
            ("eval_dit4dit_robocasa.sh", ("--arm", "nope"), "--arm must be baseline, geak or cae, got: nope"),
            ("eval_dit4dit_robocasa.sh", ("--nope", "x"), "unknown argument: --nope"),
            ("eval_dit4dit_libero.sh", ("--arm", "cae"),
             "--arm cae is the RoboCasa Compact Action Expert policy; DiT4DiT LIBERO has baseline and geak"),
            ("eval_dit4dit_libero.sh", ("--arm", "nope"), "--arm must be baseline or geak, got: nope"),
            ("eval_dit4dit_libero.sh", ("--nope", "x"), "unknown argument: --nope"),
            ("eval_fastwam_libero.sh", ("--arm", "nope"), "--arm must be baseline, geak or cae, got: nope"),
            ("eval_dit4dit_libero.sh", ("--arm", "nope"), "--arm must be baseline or geak"),
            ("eval_dit4dit_libero.sh", ("--steps", "10"), "not a DiT4DiT cell"),
            ("eval_dit4dit_libero.sh", ("--steps", "3"), "not a DiT4DiT cell"),
            ("eval_fastwam_libero.sh", ("--arm", "baseline", "--steps", "2"), "not a validated cell"),
            ("eval_fastwam_libero.sh", ("--arm", "nope"), "--arm must be baseline, geak or cae"),
            ("eval_fastwam_libero.sh", ("--nope", "x"), "unknown argument: --nope"),
        )
        for name, args, expected in cases:
            with self.subTest(script=name, args=args):
                result = self.run_script(name, *args)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn(expected, result.stderr)


class UnifiedFlagTests(unittest.TestCase):
    """A flag means the same thing in every launcher; old spellings warn."""

    LAUNCHERS = {
        "fastwam": ROOT / "scripts" / "eval_fastwam_libero.sh",
        "native": ROOT / "eval" / "native" / "run_fastwam_libero.sh",
        "dit4dit-libero": ROOT / "scripts" / "eval_dit4dit_libero.sh",
        "dit4dit-robocasa": ROOT / "scripts" / "eval_dit4dit_robocasa.sh",
    }
    DEPRECATED = "is deprecated and will be removed in the next release"

    def run_launcher(self, name, *args, env=None):
        return subprocess.run(
            ["bash", str(self.LAUNCHERS[name]), *args],
            capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin", **(env or {})},
        )

    def test_every_launcher_documents_the_common_flags(self):
        common = ("--arm", "--steps", "--sim", "--gpus", "--smoke", "--full",
                  "--trials", "--output-dir", "--record-video", "--dry-run", "-h, --help")
        extra = {
            "fastwam": ("--suites", "--mode"),
            "native": ("--suites",),
            "dit4dit-libero": ("--suites", "--clients-per-gpu", "--port"),
            "dit4dit-robocasa": ("--clients-per-gpu", "--port"),
        }
        for name in self.LAUNCHERS:
            result = self.run_launcher(name, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in common + extra[name]:
                with self.subTest(launcher=name, flag=flag):
                    self.assertIn(flag, result.stdout)

    def test_dit4dit_libero_defaults_to_the_baseline_arm(self):
        text = self.LAUNCHERS["dit4dit-libero"].read_text()
        self.assertIn("\narm=baseline\n", text)

    def test_deprecated_spellings_warn_and_still_work(self):
        cases = (
            ("dit4dit-libero", ("--gpu", "0"), "--gpu", None),
            ("dit4dit-libero", ("--parallel-tasks", "2"), "--parallel-tasks", None),
            ("dit4dit-libero", (), "GPU_ID", {"GPU_ID": "0"}),
            ("dit4dit-robocasa", ("--dry-run", "--episodes", "10"), "--episodes", None),
            ("dit4dit-robocasa", ("--dry-run",), "EPISODES", {"EPISODES": "10"}),
            ("dit4dit-robocasa", ("--dry-run",), "DIT4DIT_GPU_LIST", {"DIT4DIT_GPU_LIST": "0,1"}),
        )
        for name, args, old, env in cases:
            with self.subTest(launcher=name, old=old):
                result = self.run_launcher(name, *args, env=env)
                self.assertIn(f"warning: {old} {self.DEPRECATED}", result.stderr)
                # Past argument parsing: a dry run succeeds, a real run stops
                # only on missing runtime assets.
                if "--dry-run" in args:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotIn("unknown argument", result.stderr)

    def test_robocasa_trials_is_the_count_flag(self):
        result = self.run_launcher("dit4dit-robocasa", "--dry-run", "--trials", "10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(self.DEPRECATED, result.stderr)
        self.assertIn("trials=10", result.stdout)

    def test_robocasa_port_sets_the_first_server_port(self):
        result = self.run_launcher("dit4dit-robocasa", "--dry-run", "--port", "9100", "--gpus", "0,1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("9100", result.stdout)
        self.assertIn("9101", result.stdout)

    def test_single_server_launchers_take_one_gpu(self):
        result = self.run_launcher("dit4dit-libero", "--gpus", "0,1")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("runs one policy server; pass one GPU id", result.stderr)

    def test_fastwam_wrapper_takes_sim_gpus_and_output_dir(self):
        result = self.run_launcher("fastwam", "--sim", "mujoco", "--output-dir", "/tmp/x", "--gpus", "3")
        self.assertIn("MODEL_CKPT", result.stderr)
        result = self.run_launcher("fastwam", "--sim", "instinct", "--gpus", "0,1", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sim=instinct", result.stdout)
        self.assertIn("gpus=0,1", result.stdout)
        result = self.run_launcher("fastwam", "--dry-run")
        self.assertIn("gpus=0\n", result.stdout)  # the wrapper's one-GPU default
        for flag in ("--shards", "--port", "--clients-per-gpu"):
            with self.subTest(flag=flag):
                result = self.run_launcher("fastwam", flag, "5")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("runs no policy server", result.stderr)

    def test_native_launcher_flags(self):
        cases = (
            (("--clients-per-gpu", "2"), 2, "no policy server"),
            (("--arm", "baseline", "--steps", "2"), 2, "not a validated cell"),
            (("--full", "--output-dir", "/tmp/x"), 2, "missing required environment variable"),
            (("--gpus", "0,x"), 2, "comma-separated list of GPU ids"),
        )
        for args, code, expected in cases:
            with self.subTest(args=args):
                result = self.run_launcher("native", *args)
                self.assertEqual(result.returncode, code, result.stdout)
                self.assertIn(expected, result.stderr)


class Dit4ditStepPlumbingTests(unittest.TestCase):
    """The --steps wiring, and the reasons it is shaped the way it is.

    `num_inference_steps=` is a dead kwarg on the legacy branch, so the step
    count must be written as the `num_inference_timesteps` attribute. These
    guard each link of that chain against silently coming undone.
    """

    RUNNER = ROOT / "scripts" / "eval_dit4dit_libero.sh"
    OVERRIDE = ROOT / "scripts" / "dit4dit_step_override.py"
    LAUNCHER = ROOT / "scripts" / "dit4dit_serve_policy.py"

    def test_override_and_launcher_exist(self):
        self.assertTrue(self.OVERRIDE.is_file())
        self.assertTrue(self.LAUNCHER.is_file())

    def test_runner_starts_the_server_through_the_override_launcher(self):
        text = self.RUNNER.read_text()
        self.assertIn("dit4dit_serve_policy.py", text)
        self.assertIn('export DIT4DIT_INFER_STEPS="$steps"', text)
        # Going straight at server_policy.py would bypass the override and run
        # the config default while labelling the artifact with --steps.
        self.assertNotIn(
            'exec "$model_python" deployment/model_server/server_policy.py', text
        )

    def test_runner_accepts_the_validated_ladder(self):
        for steps in ("1", "2", "4"):
            with self.subTest(steps=steps):
                result = subprocess.run(
                    ["bash", str(self.RUNNER), "--steps", steps],
                    capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
                )
                # Must get PAST the steps gate; it then stops on missing assets.
                self.assertNotIn("not a DiT4DiT cell", result.stderr)

    def test_launcher_delegates_to_upstream_argparser(self):
        # Adding a flag of our own is what produced the --seed mismatch that
        # eval_dit4dit_libero.sh has to grep around. The launcher must reuse
        # upstream's parser so there is no second argparse surface to drift.
        text = self.LAUNCHER.read_text()
        self.assertIn("build_argparser", text)
        self.assertNotIn("add_argument", text)

    def test_side_by_side_trees_are_not_modified_for_this(self):
        # The whole point of the launcher is that neither validated tree needs
        # editing. If a step flag ever appears in one, the launcher is redundant
        # and this test should be revisited deliberately, not silently.
        for tree in ("dit4dit-baseline", "dit4dit-geak"):
            with self.subTest(tree=tree):
                text = (SRC / tree / "deployment" / "model_server"
                        / "server_policy.py").read_text()
                self.assertNotIn("num_inference_timesteps", text)
                self.assertNotIn("--seed", text)

    def test_override_writes_the_attribute_and_verifies_the_read_back(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ovr", self.OVERRIDE)
        ovr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ovr)  # no-op import: env var is unset

        class Head:
            num_inference_timesteps = 4

        class Model:
            def __init__(self):
                self.action_model = Head()

        model = Model()
        with contextlib.redirect_stdout(io.StringIO()):
            ovr.apply(2, model)
        self.assertEqual(model.action_model.num_inference_timesteps, 2)

    def test_override_refuses_rather_than_mislabelling(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("_ovr2", self.OVERRIDE)
        ovr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ovr)

        class NoHead:
            action_model = None

        class NoAttr:
            class action_model:  # noqa: N801
                pass

        class Sticky:
            class action_model:  # noqa: N801
                num_inference_timesteps = 4

                def __setattr__(self, k, v):  # writes silently do nothing
                    pass

        # action_model is None (video/policy-mode checkpoint).
        with self.assertRaises(RuntimeError):
            ovr.apply(2, NoHead())
        # Attribute absent -- the name may have been renamed upstream.
        with self.assertRaises(RuntimeError):
            ovr.apply(2, NoAttr())
        # Write silently does not stick: this is the failure this guards.
        sticky = Sticky()
        sticky.action_model = Sticky.action_model()
        with self.assertRaises(RuntimeError):
            ovr.apply(2, sticky)


class Dit4ditServerHarnessTests(unittest.TestCase):
    """The pieces both DiT4DiT launchers share, checked without running them."""

    SCRIPTS = ROOT / "scripts"
    LAUNCHERS = ("eval_dit4dit_libero.sh", "eval_dit4dit_robocasa.sh")

    def test_both_launchers_build_diffusers_through_the_shared_helper(self):
        # One copy of the md5 checks: an inline copy in one launcher drifts.
        for name in self.LAUNCHERS:
            with self.subTest(launcher=name):
                text = (self.SCRIPTS / name).read_text()
                self.assertIn('source "${ROOT}/scripts/dit4dit_diffusers_overlay.sh"', text)
                self.assertIn("build_diffusers_overlay ", text)
                self.assertNotIn("declare -A upstream_md5", text)

    def test_keepalive_handler_copy_matches_every_tree(self):
        # dit4dit_serve_keepalive.py replaces the tree's handler with a copy.
        # It refuses at runtime on a mismatch; this catches it before a run.
        import ast
        import hashlib
        import re

        text = (self.SCRIPTS / "dit4dit_serve_keepalive.py").read_text()
        want = re.search(r'_UPSTREAM_HANDLER_SHA256 = "([0-9a-f]{64})"', text).group(1)
        for tree in ("dit4dit-baseline", "dit4dit-geak"):
            with self.subTest(tree=tree):
                path = (SRC / tree / "deployment" / "model_server" / "tools"
                        / "websocket_policy_server.py")
                source = path.read_text()
                lines = source.splitlines(keepends=True)
                node = next(
                    n for n in ast.walk(ast.parse(source))
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == "_handler"
                )
                got = hashlib.sha256(
                    "".join(lines[node.lineno - 1:node.end_lineno]).encode()
                ).hexdigest()
                self.assertEqual(got, want)

    def test_server_entry_point_installs_the_keepalive(self):
        text = (self.SCRIPTS / "dit4dit_serve_policy.py").read_text()
        self.assertIn("import dit4dit_serve_keepalive", text)

    def test_server_entry_point_bounds_inductor_fallback_log_text(self):
        # The shim pins torch's operator_str by hash and refuses otherwise; this
        # only checks the entry point still installs it.
        text = (self.SCRIPTS / "dit4dit_serve_policy.py").read_text()
        self.assertIn("import dit4dit_serve_inductor_logstr", text)
        shim = (self.SCRIPTS / "dit4dit_serve_inductor_logstr.py").read_text()
        self.assertIn("OperatorIssue.operator_str = _operator_str", shim)


class AdvertisedCellTests(unittest.TestCase):
    """The docs must not offer a cell the project does not want offered."""

    DOC = ROOT / "docs" / "side-by-side-source-trees.md"

    def test_no_one_liner_offers_a_one_step_fastwam_cell(self):
        for line in self.DOC.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith(("bash scripts/eval_fastwam_libero.sh",
                                        "bash eval/native/run_fastwam_libero.sh")):
                continue
            self.assertNotIn("--steps 1 ", stripped + " ", stripped)

    def test_every_documented_runner_command_is_an_accepted_cell(self):
        # Guards the matrix and the one-liners against drifting apart from the
        # runners' own refusals, which is how a refused row once survived.
        import re
        import shlex

        checked = 0
        for line in self.DOC.read_text().splitlines():
            stripped = line.strip()
            m = re.match(r"^bash ((?:scripts|eval/native)/\S+\.sh) (.*)$", stripped)
            if not m:
                continue
            script, rest = m.group(1), m.group(2)
            if "--steps" not in rest:
                continue
            # Pass the WHOLE documented argument list: --steps is only half a
            # cell, and --arm decides whether the other half is offered.
            with self.subTest(command=stripped):
                # --dry-run: past flag validation, nothing starts (the native launcher would
                # otherwise need its assets before it could refuse anything else).
                result = subprocess.run(
                    ["bash", str(ROOT / script), *shlex.split(rest), "--dry-run"],
                    capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"},
                )
                self.assertNotIn("not a DiT4DiT cell", result.stderr)
                self.assertNotIn("not a validated cell", result.stderr)
                # Documented commands use current spellings and existing arms.
                self.assertNotIn("is deprecated", result.stderr)
                self.assertNotIn("unknown argument", result.stderr)
                self.assertNotIn("--arm must be", result.stderr)
                checked += 1
        self.assertGreater(checked, 0, "no documented --steps command was found")


if __name__ == "__main__":
    unittest.main()
