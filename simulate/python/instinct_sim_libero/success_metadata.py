#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build NativeSuccessChecker metadata from a live LIBERO env.

hip_env.NativeSuccessChecker takes a pre-built metadata dict (the
`"goal_state" in success_source` branch); this module builds it.

Everything the checker needs is derivable from a live OffScreenRenderEnv, which
we already build once per task as an XML/metadata reference before handing the MJCF
to the GPU simulator. This extracts it while that env is open, then the CPU env can be
closed for good.

Schema consumed by NativeSuccessChecker (read off hip_env.py:228-263, 420, 470):

  {
    "goal_state": [["on", "obj1", "obj2"], ["open", "cabinet"], ...],
    "objects": {
      "<name>": {
         "state_type":            "ObjectState" | "SiteObjectState",
         "joint_names":           [...],          # -> joint_ids, qpos_addrs
         # ObjectState only:
         "body_name_candidates":  [...],          # first that resolves wins
         # SiteObjectState only:
         "site_name":             str,
         "site_size":             [x, y, z],      # half-extents for in_box/under
         "parent_body_name":      str | None,     # -> parent_body_id
      }, ...
    }
  }

Body-name candidates matter: robosuite appends "_main" to an object's root body,
and fixtures use a different convention again, so a single guess silently
resolves to body_id -1 and every predicate on that object returns False. We emit
an ordered candidate list and let the checker take the first that resolves
against the GPU model.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _joint_names(obj) -> List[str]:
    """Joint names for an object, if it exposes any."""
    for attr in ("joints",):
        js = getattr(obj, attr, None)
        if js:
            return [str(j) for j in js]
    return []


def _object_body_candidates(name: str, obj) -> List[str]:
    """Ordered body-name guesses for an object/fixture.

    robosuite names the root body of a MujocoObject "<name>_main"; fixtures and
    some scanned objects use the bare name. Order matters -- the checker takes
    the first that resolves.
    """
    cands: List[str] = []
    root = getattr(obj, "root_body", None)
    if root:
        cands.append(str(root))
    cands.append(f"{name}_main")
    cands.append(name)
    naming = getattr(obj, "naming_prefix", None)
    if naming:
        cands.append(f"{naming}main")
    # de-dup, preserve order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_success_metadata(env) -> Dict[str, Any]:
    """Extract checker metadata from a live LIBERO OffScreenRenderEnv.

    `env` may be the OffScreenRenderEnv wrapper or the inner problem env; both
    are accepted because callers hold whichever is convenient.
    """
    base = getattr(env, "env", env)          # unwrap OffScreenRenderEnv

    parsed = getattr(base, "parsed_problem", None)
    if not parsed or "goal_state" not in parsed:
        raise RuntimeError(
            "env has no parsed_problem['goal_state']; cannot build success "
            "metadata. Success detection would silently always-fail.")
    goal_state = [list(s) for s in parsed["goal_state"]]

    states = getattr(base, "object_states_dict", None)
    if not states:
        raise RuntimeError("env has no object_states_dict")

    objects_dict = dict(getattr(base, "objects_dict", {}) or {})
    fixtures_dict = dict(getattr(base, "fixtures_dict", {}) or {})
    sites_dict = dict(getattr(base, "object_sites_dict", {}) or {})

    objects: Dict[str, Any] = {}
    for name, st in states.items():
        stype = getattr(st, "object_state_type", "object")

        if stype == "site":
            site = sites_dict.get(name)
            if site is None:
                raise RuntimeError(
                    f"site object {name!r} is in object_states_dict but not in "
                    "object_sites_dict; its geometry cannot be recovered")
            size = getattr(site, "size", None)
            if size is None:
                raise RuntimeError(f"site {name!r} exposes no size")
            info: Dict[str, Any] = {
                "state_type": "SiteObjectState",
                "site_name": str(getattr(site, "name", name)),
                "site_size": [float(v) for v in size],
                "joint_names": _joint_names(site),
            }
            parent = getattr(st, "parent_name", None)
            if parent:
                info["parent_body_name"] = str(parent)
                pobj = objects_dict.get(parent) or fixtures_dict.get(parent)
                if pobj is not None:
                    info["parent_object_name"] = str(parent)
                    root = getattr(pobj, "root_body", None)
                    if root:
                        # give the checker the exact root body too; it tries
                        # parent_body_name then parent_body_name + "_main"
                        info["parent_body_name"] = str(root)
            objects[name] = info
            continue

        obj = objects_dict.get(name) or fixtures_dict.get(name)
        if obj is None:
            raise RuntimeError(
                f"object {name!r} is in object_states_dict but in neither "
                "objects_dict nor fixtures_dict")
        objects[name] = {
            "state_type": "ObjectState",
            "body_name_candidates": _object_body_candidates(name, obj),
            "joint_names": _joint_names(obj),
            "is_fixture": name in fixtures_dict,
        }

    # Every name the goal actually references must be present, or that predicate
    # silently evaluates False forever and the task can never succeed.
    referenced = {tok for state in goal_state for tok in state[1:]}
    missing = referenced - set(objects)
    if missing:
        raise RuntimeError(
            f"goal_state references objects with no metadata: {sorted(missing)}")

    return {"goal_state": goal_state, "objects": objects}


if __name__ == "__main__":
    import os, sys, json
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import os.path as osp
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import benchmark as lb, get_libero_path

    suite_name = sys.argv[1] if len(sys.argv) > 1 else "libero_spatial"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    suite = lb.get_benchmark_dict()[suite_name]()
    for i in range(n):
        t = suite.get_task(i)
        f = osp.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=f, camera_heights=64, camera_widths=64)
        e.reset()
        md = extract_success_metadata(e)
        e.close()
        print(f"--- {suite_name}[{i}] {t.name}")
        print(f"    goal: {md['goal_state']}")
        for k, v in md["objects"].items():
            if k in {tok for s in md["goal_state"] for tok in s[1:]}:
                print(f"    {k}: {json.dumps(v)}")
