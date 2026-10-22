# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CAE-FastWAM TunableOp table is looked up only on the GPU it was recorded on.

The table pins hipBLASLt solutions measured on an MI355X. Other gfx950 parts pass its validators
but lack some of those solutions, and a missing pinned solution raises instead of falling back,
so `_fwam_r2tune_device_ok` must reject them. The function is exercised in isolation (mot.py
itself needs a GPU stack to import).
"""
import ast
import logging
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOT = ROOT / "src" / "cae-fastwam" / "fastwam" / "models" / "wan22" / "mot.py"


def load_device_ok(device_name, available=True, wanted="MI355X"):
    tree = ast.parse(MOT.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_fwam_r2tune_device_ok")
    cuda = types.SimpleNamespace(is_available=lambda: available, current_device=lambda: 0,
                                 get_device_name=lambda i: device_name)
    ns = {"torch": types.SimpleNamespace(cuda=cuda), "logger": logging.getLogger("t"),
          "_R2TUNE_DEVICE": wanted}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MOT), "exec"), ns)
    return ns["_fwam_r2tune_device_ok"]


class R2TuneDeviceTests(unittest.TestCase):
    def test_mi355x_uses_the_table(self):
        self.assertTrue(load_device_ok("AMD Instinct MI355X")())

    def test_other_gfx950_parts_skip_the_table(self):
        self.assertFalse(load_device_ok("AMD Instinct MI350X")())

    def test_no_gpu_skips_the_table(self):
        self.assertFalse(load_device_ok("AMD Instinct MI355X", available=False)())

    def test_override_names_another_device(self):
        self.assertTrue(load_device_ok("AMD Instinct MI350X", wanted="MI350X")())

    def test_lookup_is_gated_before_the_table_is_armed(self):
        src = MOT.read_text()
        init = src[src.index("def _fwam_r2tune_init"):]
        self.assertLess(init.index("_fwam_r2tune_device_ok()"), init.index("tun.enable(True)\n        tun.tuning_enable(False)"))


if __name__ == "__main__":
    unittest.main()
