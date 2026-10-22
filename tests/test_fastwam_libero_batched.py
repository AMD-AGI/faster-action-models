# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""eval/native/fastwam_libero_batched.py's CAE-FastWAM path (--model-batch 1), checked without a GPU or model.

The driver is imported with stand-in `fastwam`, `hydra` and `experiments.libero` modules (and a stand-in `torch`
when torch is not installed), so only its own control flow runs:

* PerObservationPolicy makes one serial _predict_action_chunk call per observation, in order;
* an action-VAE model is refused at a padded batch and accepted with --model-batch 1;
* an action-VAE model refuses --tasks from another suite;
* the action-VAE suite id comes from EVALUATION.task_suite_name, and a task config naming another is refused.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy  # noqa: F401  (imported once here: sys.modules is restored after each driver import)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "eval" / "native" / "fastwam_libero_batched.py"


class Cfg(dict):
    """The DictConfig subset the driver uses: attribute access and .get."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def cfg(suite="libero_spatial", vae_suite_id=None, **evaluation):
    model = Cfg() if vae_suite_id is None else Cfg(action_vae_suite_id=vae_suite_id)
    return Cfg(EVALUATION=Cfg(task_suite_name=suite, **evaluation), model=model, seed=42)


class Model:
    def __init__(self, num_suites=None):
        self.action_vae = None if num_suites is None else SimpleNamespace(cfg=SimpleNamespace(num_suites=num_suites))
        self.action_vae_suite_id = None
        self.torch_dtype = "bf16"

    def infer_action(self, prompt, input_image, compile_action_infer=False):
        raise AssertionError("the driver must not call infer_action directly on the per-observation path")


def load_driver():
    stubs = {}

    def stub(name, **attrs):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        stubs[name] = module
        return module

    stub("hydra", compose=None, initialize_config_dir=None)
    stub("hydra.utils", instantiate=None)
    stub("fastwam", __file__=str(ROOT / "src" / "cae-fastwam" / "fastwam" / "__init__.py"))
    stub("fastwam.datasets")
    stub("fastwam.datasets.lerobot")
    stub("fastwam.datasets.lerobot.robot_video_dataset", DEFAULT_PROMPT="{task}")
    stub("fastwam.utils")
    stub("fastwam.utils.pytorch_utils", set_global_seed=lambda *a, **k: None)
    stub("experiments")
    libero = stub("experiments.libero")
    libero.eval_libero_single = stub("experiments.libero.eval_libero_single")
    stub("experiments.libero.libero_utils", LIBERO_ENV_RESOLUTION=256, get_libero_dummy_action=lambda: [0] * 7,
         invert_gripper_action=lambda a: a)
    stub("fastwam_batched_infer", infer_action_multi=None)
    try:
        import torch  # noqa: F401
    except ImportError:
        stub("torch", cuda=SimpleNamespace(is_available=lambda: False), no_grad=None, cat=None, load=None)
    name = f"t_{uuid.uuid4().hex}"
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(name, DRIVER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class PerObservationPolicyTests(unittest.TestCase):
    def test_one_serial_call_per_observation_in_order(self):
        drv = load_driver()
        seen = []

        def predict(obs, task, model, processor, c, *, action_horizon, input_w, input_h, model_device):
            seen.append((obs, task, action_horizon, input_w, input_h, model_device))
            return f"chunk-{obs}", {}, None

        drv.EL._predict_action_chunk = predict
        model = Model(num_suites=4)
        policy = drv.PerObservationPolicy(cfg(), model, "proc", "cuda:0", 32, 224, 224)
        self.assertEqual(policy(["a", "b", "c"], "pick it"), ["chunk-a", "chunk-b", "chunk-c"])
        self.assertEqual(seen, [(o, "pick it", 32, 224, 224, "cuda:0") for o in "abc"])
        self.assertEqual(policy.calls, 3)
        policy(["d"], "pick it")
        self.assertEqual(policy.calls, 4)
        self.assertGreaterEqual(policy.seconds, 0.0)


class RefusalTests(unittest.TestCase):
    def setUp(self):
        self.drv = load_driver()

    def test_action_vae_needs_model_batch_1(self):
        c = cfg(compile_action_infer=True)
        with self.assertRaises(SystemExit) as e:
            self.drv.refuse_unbatched_features(c, Model(num_suites=4))
        self.assertIn("use --model-batch 1", str(e.exception))
        self.assertIn("compile_action_infer", str(e.exception))
        self.drv.refuse_unbatched_features(c, Model(num_suites=4), per_obs=True)

    def test_stateful_features_stay_refused_per_observation(self):
        for key in ("visualize_future_video", "use_action_ensembler"):
            with self.subTest(key=key), self.assertRaises(SystemExit) as e:
                self.drv.refuse_unbatched_features(cfg(**{key: True}), Model(num_suites=4), per_obs=True)
            self.assertIn(key, str(e.exception))

    def test_mixed_suites_are_refused_for_an_action_vae_model(self):
        items = self.drv.parse_tasks("libero_spatial:0,libero_goal:1")
        self.assertEqual(items, [("libero_spatial", 0), ("libero_goal", 1)])
        with self.assertRaises(SystemExit) as e:
            self.drv.check_task_suites(items, cfg("libero_spatial"), Model(num_suites=4))
        self.assertIn("libero_goal", str(e.exception))
        self.drv.check_task_suites(items, cfg("libero_spatial"), Model())   # baseline/geak: any mix
        self.drv.check_task_suites(items[:1], cfg("libero_spatial"), Model(num_suites=4))

    def test_unknown_suite_is_refused(self):
        with self.assertRaises(SystemExit):
            self.drv.parse_tasks("libero_90:0")


class SuiteIdTests(unittest.TestCase):
    def setUp(self):
        self.drv = load_driver()

    def test_suite_id_follows_the_task_suite(self):
        for suite, sid in (("libero_spatial", 0), ("libero_object", 1), ("libero_goal", 2), ("libero_10", 3)):
            with self.subTest(suite=suite):
                model = Model(num_suites=4)
                self.assertEqual(self.drv.set_action_vae_suite(model, cfg(suite, vae_suite_id=sid)), sid)
                self.assertEqual(model.action_vae_suite_id, sid)

    def test_a_task_config_for_another_suite_is_refused(self):
        model = Model(num_suites=4)
        with self.assertRaises(SystemExit) as e:
            self.drv.set_action_vae_suite(model, cfg("libero_goal", vae_suite_id=0))
        self.assertIn("action_vae_suite_id=0", str(e.exception))
        self.assertIsNone(model.action_vae_suite_id)

    def test_models_without_suite_conditioning_are_left_alone(self):
        for model in (Model(), Model(num_suites=0)):
            self.assertIsNone(self.drv.set_action_vae_suite(model, cfg("libero_goal", vae_suite_id=0)))
            self.assertIsNone(model.action_vae_suite_id)


if __name__ == "__main__":
    unittest.main()
