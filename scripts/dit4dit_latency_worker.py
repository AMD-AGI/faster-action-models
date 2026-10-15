#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One DiT4DiT latency measurement, run INSIDE the model container.

Launched by `scripts/dit4dit_latency.py`, once per (round, arm), with
PYTHONPATH = <tree>:<diffusers overlay>[:...], cwd = <tree>, the GEAK_* gates
pinned to 0 and DIT4DIT_INFER_STEPS set, as the DiT4DiT launchers start the
policy server. This script never selects a tree itself.

  1. IMPORTS, in the order of scripts/dit4dit_serve_policy.py: the optional
     host shim `blockflax`, scripts/dit4dit_step_override.py (writes
     action_model.num_inference_timesteps at load and asserts the read-back)
     and scripts/dit4dit_serve_inductor_logstr.py (bounded inductor fallback
     log text; without it the RoboCasa GEAK first call stalls for about an
     hour). dit4dit_serve_keepalive is not imported: it only moves the
     websocket server's policy call onto a thread, and there is no server here.
  2. LOAD as the tree's deployment/model_server/server_policy.py does:
     baseframework.from_pretrained(ckpt), bf16 when the benchmark's launcher
     passes --use_bf16 (RoboCasa yes, LIBERO no), .to("cuda").eval().
  3. REQUEST: the benchmark's own client interface in the tree
     (examples/LIBERO/eval_files/model2libero_interface.ModelClient,
     examples/Robocasa_tabletop/eval_files/model2robocasa_interface.PolicyWarper)
     builds the request from one deployed-shape observation (seeded random
     pixels and state); its websocket client is replaced by a stub that
     captures the request. So the timed request is exactly what the server
     receives (resized and concatenated images, state, sampling flags).
  4. CONTRACT (the first warmup call). Aborts unless the tree and diffusers
     are the expected ones, transformers and tokenizers are the versions the
     tree's requirements.txt pins, every gate is 0, the action head ran once per call
     with num_inference_timesteps equal to the requested step count, and the
     request and action shapes are the deployed ones. A mislabelled cell is
     refused, never reported.
  5. TIME the server's call, vla.predict_action(**request), with no outer
     no_grad (the server calls it bare): WARMUP calls in all (25: autotuning
     settles late), then ITERS calls, each followed by a device synchronize.

NAIVE torch.compile (--naive-compile modules; the launcher's compile arm, on
the baseline tree and baseline diffusers overlay with every GEAK_* gate 0).
After the contract call (which runs eager), torch.compile is applied at
default settings -- no mode, inductor, no fullgraph, no dynamic shapes, no
options -- to each heavy module that runs per call:

    extractor = vla.backbone_interface.extractor
    extractor.text_encoder.compile()                          # Cosmos text encoder
    extractor.transformer.compile()                           # Cosmos transformer
    extractor.vae.encode = torch.compile(extractor.vae.encode)  # Cosmos VAE encode
    vla.action_model.model.compile()                          # action DiT

vla.predict_action itself stays eager: compiling the server's entry point
(torch.compile(vla.predict_action)) trips a dynamo guard bug on torch 2.11,
so the naive arm compiles the modules it calls instead. The first compiled
call (where compilation happens) is timed separately and followed by WARMUP-1
more warmup calls. The cell is refused unless dynamo captured at least one
graph on that first call and the timed calls trigger no compile or recompile
(dynamo counters unchanged across them). RESULT["compile"] records the compiled targets, the
graph count, graph breaks, recompiles during warmup and the first-call wall
time. Outputs are not compared against eager here.

Prints `Perf: <median> ms` and one `RESULT <json>` line. Nothing is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

GATES = ("GEAK_ACTION_COMPILE", "GEAK_COSMOS_COMPILE", "GEAK_VAE_T1", "GEAK_ACTION_BF16")
# The deployed request and action shapes per benchmark (one example per call).
EXPECT = {
    "libero":   {"image": [[224, 448, 3]], "state": [1, 16], "action": [1, 8, 8]},
    "robocasa": {"image": [[224, 224, 3]], "state": [1, 64], "action": [1, 16, 32]},
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=tuple(EXPECT), required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--expect-tree", required=True,
                    help="abort unless DiT4DiT is imported from this tree")
    ap.add_argument("--expect-diffusers", required=True,
                    help="abort unless diffusers is imported from this package directory")
    ap.add_argument("--naive-compile", choices=("none", "modules"), default="none",
                    help="modules: naive torch.compile at defaults on the per-call modules")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=40)
    return ap.parse_args()


NAIVE_TARGETS = ("extractor.text_encoder.compile()", "extractor.transformer.compile()",
                 "extractor.vae.encode = torch.compile(extractor.vae.encode)",
                 "action_model.model.compile()")
NAIVE_OPTIONS = {"mode": "default", "backend": "inductor", "fullgraph": False,
                 "dynamic": None, "options": None}


def naive_compile_modules(vla, torch) -> None:
    """torch.compile at default settings on the heavy modules of one call."""
    ext = vla.backbone_interface.extractor
    ext.text_encoder.compile()
    ext.transformer.compile()
    ext.vae.encode = torch.compile(ext.vae.encode)
    vla.action_model.model.compile()


def dynamo_snapshot() -> dict:
    """torch._dynamo counters that show whether (re)compilation happened."""
    from torch._dynamo.utils import counters
    try:
        from torch._dynamo.utils import compilation_time_metrics
        compiles = len(compilation_time_metrics.get("_compile.compile_inner", []))
    except ImportError:
        compiles = 0
    return {"frames": int(counters["frames"].get("total", 0)),
            "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
            "graph_breaks": int(sum(counters["graph_break"].values())),
            "compiles": compiles}


def graph_break_reasons(limit: int = 10) -> list:
    from torch._dynamo.utils import counters
    rows = sorted(counters["graph_break"].items(), key=lambda kv: -kv[1])[:limit]
    return [{"count": int(c), "reason": (r.strip().splitlines() or [""])[0][:300]}
            for r, c in rows]


# Packages on the timed path whose version must equal the tree's requirements.txt pin.
# The model image may ship older ones; DIT4DIT_PYOVERLAY (or the image) must provide these.
PINNED = ("transformers", "tokenizers")


def requirement_pins(tree: str, names=PINNED) -> dict:
    """`name==version` pins for `names` from <tree>/requirements.txt."""
    pins = {}
    with open(os.path.join(tree, "requirements.txt")) as f:
        for line in f:
            name, sep, ver = line.split("#", 1)[0].strip().partition("==")
            if sep and name.strip().lower() in names:
                pins[name.strip().lower()] = ver.strip()
    missing = [n for n in names if n not in pins]
    if missing:
        raise SystemExit(f"{tree}/requirements.txt pins none of {missing}")
    return pins


def fail(problems) -> int:
    print("CONTRACT FAILED: " + "; ".join(problems), file=sys.stderr, flush=True)
    return 3


def shape_of(x, np):
    if isinstance(x, np.ndarray):
        return list(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape_of(v, np) for v in x]
    return type(x).__name__


def capture_request(bench, tree, ckpt, np):
    """Build the request the benchmark's client sends, without a server."""
    import deployment.model_server.tools.websocket_policy_client as wpc

    class Captured(Exception):
        pass

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def predict_action(self, request):
            raise Captured(request)

    wpc.WebsocketClientPolicy = StubClient

    def load_interface(rel):
        spec = importlib.util.spec_from_file_location("bench_interface", os.path.join(tree, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    rng = np.random.default_rng(0)
    if bench == "libero":
        mod = load_interface("examples/LIBERO/eval_files/model2libero_interface.py")
        # As eval_libero.py constructs it; agentview and wrist images are 256x256.
        client = mod.ModelClient(policy_ckpt_path=ckpt, image_size=[224, 224])
        obs = {"image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
               "wrist_image": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
               "lang": "put both the alphabet soup and the tomato sauce in the basket",
               "state": rng.standard_normal(8).astype(np.float32)}

        def call():
            return client.step(example=obs, step=0)
    else:
        mod = load_interface("examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py")
        # As simulation_env.py constructs it; GR1 ego view is 800x1280.
        client = mod.PolicyWarper(policy_ckpt_path=ckpt, image_size=[224, 224], n_action_steps=12)
        obs = {"annotation.human.coarse_action":
               ("pick the bottle from the counter and place it in the cabinet",),
               "video.ego_view": rng.integers(0, 256, (1, 1, 800, 1280, 3), dtype=np.uint8),
               "state.left_arm": rng.standard_normal((1, 1, 7)).astype(np.float32),
               "state.right_arm": rng.standard_normal((1, 1, 7)).astype(np.float32),
               "state.left_hand": rng.standard_normal((1, 1, 6)).astype(np.float32),
               "state.right_hand": rng.standard_normal((1, 1, 6)).astype(np.float32),
               "state.waist": rng.standard_normal((1, 1, 3)).astype(np.float32)}

        def call():
            return client.step(obs)
    try:
        call()
    except Captured as c:
        return c.args[0]
    raise SystemExit("the client interface returned without calling predict_action")


def main() -> int:
    a = parse_args()
    if a.warmup < 1 or a.iters < 1:
        raise SystemExit("--warmup and --iters must be positive (the first warmup call "
                         "is the contract check)")
    bf16 = a.bench == "robocasa"   # the RoboCasa launcher's server passes --use_bf16; LIBERO none
    tree = os.path.realpath(a.expect_tree)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:   # same import order as scripts/dit4dit_serve_policy.py
        import blockflax  # noqa: F401
    except ImportError:
        pass
    import dit4dit_step_override  # noqa: F401  -- sets and asserts num_inference_timesteps
    import dit4dit_serve_inductor_logstr  # noqa: F401  -- bounded fallback log text
    import numpy as np
    import torch
    import DiT4DiT
    import diffusers
    import tokenizers
    import transformers
    from DiT4DiT.model.framework.base_framework import baseframework

    # ---- before loading: is this the cell it claims to be? ----------------------------------
    problems = []
    loaded = os.path.realpath(DiT4DiT.__file__)
    if not loaded.startswith(tree + os.sep):
        problems.append(f"DiT4DiT imported from {loaded}, expected under {tree}")
    diff_dir = os.path.dirname(os.path.realpath(diffusers.__file__))
    if diff_dir != os.path.realpath(a.expect_diffusers):
        problems.append(f"diffusers imported from {diff_dir}, expected {a.expect_diffusers}")
    gates = {k: os.environ.get(k) for k in GATES}
    if any(v != "0" for v in gates.values()):
        problems.append(f"GEAK gates must all be 0, got {gates}")
    # Versions of the modules actually imported (not whatever dist-info is first on the path).
    versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    for name, ver in requirement_pins(tree).items():
        if versions[name] != ver:
            problems.append(f"{name} {versions[name]} imported, {os.path.basename(tree)}/"
                            f"requirements.txt pins {ver} (provide it via DIT4DIT_PYOVERLAY)")
    want = os.environ.get("DIT4DIT_INFER_STEPS")
    if want != str(a.steps):
        problems.append(f"DIT4DIT_INFER_STEPS={want!r}, requested {a.steps}")
    if a.naive_compile != "none" and a.steps != 4:
        problems.append(f"the naive torch.compile arm is timed at the baseline step count "
                        f"(4) only, requested {a.steps}")
    if problems:
        return fail(problems)

    # ---- load, as server_policy.main -----------------------------------------------------------
    vla = baseframework.from_pretrained(a.ckpt)
    if bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()
    head = vla.action_model
    read_back = int(getattr(head, "num_inference_timesteps"))
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in vla.parameters()
                     if p.is_floating_point()})

    request = capture_request(a.bench, tree, a.ckpt, np)
    ex = request["examples"][0]
    contract = {"bench": a.bench, "tree": os.path.basename(tree), "steps": read_back,
                "bf16": bf16, "param_dtypes": dtypes, "versions": versions,
                "diffusers": os.path.basename(os.path.dirname(diff_dir)), "gates": gates,
                "request": {"n_examples": len(request["examples"]),
                            "image": shape_of(ex["image"], np), "state": shape_of(ex["state"], np),
                            "lang": ex["lang"],
                            **{k: v for k, v in request.items() if k != "examples"}}}

    # ---- contract: the first warmup call, with the action head probed ----------------------------
    head_calls = []
    orig = head.predict_action

    def probe(*args, **kw):
        head_calls.append(int(head.num_inference_timesteps))
        return orig(*args, **kw)

    head.predict_action = probe
    try:
        out = vla.predict_action(**request)
    finally:
        del head.predict_action
    torch.cuda.synchronize()
    act = out.get("normalized_actions", out) if isinstance(out, dict) else out
    contract["action_out"] = shape_of(np.asarray(act), np)
    contract["head_steps_per_call"] = head_calls
    print(f"[contract] {contract}", flush=True)

    exp = EXPECT[a.bench]
    if read_back != a.steps:
        problems.append(f"num_inference_timesteps reads back {read_back}, requested {a.steps}")
    if head_calls != [a.steps]:
        problems.append(f"action head ran with steps {head_calls}, expected one call at {a.steps}")
    if bf16 and dtypes != ["bfloat16"]:
        problems.append(f"parameter dtypes {dtypes}, the RoboCasa server runs bf16")
    if contract["request"]["n_examples"] != 1:
        problems.append(f"{contract['request']['n_examples']} examples per request, expected 1")
    if contract["request"]["image"] != exp["image"] or contract["request"]["state"] != exp["state"]:
        problems.append(f"request image {contract['request']['image']} state "
                        f"{contract['request']['state']}, deployed is image {exp['image']} "
                        f"state {exp['state']}")
    if contract["action_out"] != exp["action"]:
        problems.append(f"action {contract['action_out']}, expected {exp['action']}")
    if problems:
        return fail(problems)

    # ---- naive torch.compile arm: compile, then prove it compiled ----------------------------------
    cinfo = None
    if a.naive_compile == "modules":
        naive_compile_modules(vla, torch)
        s0 = dynamo_snapshot()
        t0 = time.perf_counter()
        out = vla.predict_action(**request)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - t0
        s1 = dynamo_snapshot()
        act = out.get("normalized_actions", out) if isinstance(out, dict) else out
        if s1["unique_graphs"] <= s0["unique_graphs"]:
            problems.append(f"naive torch.compile captured no graph on the first call: {s1}")
        if shape_of(np.asarray(act), np) != exp["action"]:
            problems.append(f"compiled action {shape_of(np.asarray(act), np)}, "
                            f"expected {exp['action']}")
        if problems:
            return fail(problems)
        cinfo = {"mode": a.naive_compile, "targets": list(NAIVE_TARGETS),
                 "options": NAIVE_OPTIONS, "first_call_s": first_call_s,
                 "after_first_call": s1}

    # ---- time the server's call --------------------------------------------------------------------
    for _ in range(a.warmup - 1):
        vla.predict_action(**request)
    torch.cuda.synchronize()
    if cinfo is not None:
        s2 = dynamo_snapshot()
    ts = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        vla.predict_action(**request)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    if cinfo is not None:
        s3 = dynamo_snapshot()
        if s3 != s2:
            return fail([f"the timed calls compiled or recompiled (dynamo counters "
                         f"{s2} -> {s3}); the cell would time compilation"])
        cinfo.update({"unique_graphs": s3["unique_graphs"], "graph_breaks": s3["graph_breaks"],
                      "graph_break_reasons": graph_break_reasons(),
                      "warmup_recompiles": s2["frames"] - s1["frames"],
                      "after_timed": s3, "timed_recompiled": False})
        print(f"[compile] {cinfo}", flush=True)

    med = statistics.median(ts)
    result = {"median_ms": med, "min_ms": min(ts),
              "p90_ms": sorted(ts)[int(0.9 * (len(ts) - 1))], "samples_ms": ts,
              "steps": a.steps, "warmup": a.warmup, "iters": a.iters, "contract": contract,
              "compile": cinfo,
              "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    print(f"Perf: {med:.4f} ms", flush=True)
    print("RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
