# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HP_SUCCESS_DEBUG=1 (run_client): record robocasa's success predicates over each episode and print them at
every robosuite reset, i.e. at the end of each episode:
  - object_utils predicates (any_gripper_obj_far, check_obj_in_receptacle, check_obj_fixture_contact,
    check_obj_upright, obj_inside_of), also where task modules imported them by name: last value and whether
    they were ever true (ever=1);
  - fixture door states (get_door_state): last, min and max of each returned joint value;
  - the task's _check_success: ever true and the first step it was."""
import functools
import os
import sys

PREDICATES = ("any_gripper_obj_far", "check_obj_in_receptacle", "check_obj_fixture_contact", "check_obj_upright",
              "obj_inside_of")


def install():
    from robocasa.utils import object_utils as OU
    last, ever, doors, succ = {}, {}, {}, {"ever": False, "first": -1, "n": 0}
    wrapped = {}
    for fn in PREDICATES:
        orig = getattr(OU, fn, None)
        if orig is None:
            continue

        def wrap(*a, _orig=orig, _fn=fn, **k):
            r = _orig(*a, **k)
            names = [x for x in list(a[1:3]) + [k.get("obj_name"), k.get("fixture_id")] if isinstance(x, str)]
            key = _fn + ":" + ",".join(n.split("_")[0] if len(n) > 12 else n for n in names)
            last[key] = bool(r)
            ever[key] = ever.get(key, False) or bool(r)
            return r
        wrapped[fn] = functools.wraps(orig)(wrap)
        setattr(OU, fn, wrapped[fn])

    def patch_modules():   # task modules that did `from object_utils import obj_inside_of`
        for name, mod in list(sys.modules.items()):
            if mod is None or not name.startswith("robocasa.environments"):
                continue
            for fn, w in wrapped.items():
                if getattr(mod, fn, None) is not None and getattr(mod, fn) is not w:
                    setattr(mod, fn, w)

    patch_modules()
    from robocasa.models import fixtures as FX
    import inspect
    for _, cls in inspect.getmembers(FX, inspect.isclass):
        if "get_door_state" in cls.__dict__:
            orig = cls.__dict__["get_door_state"]

            def gds(self, *a, _orig=orig, _cls=cls.__name__, **k):
                r = _orig(self, *a, **k)
                for kk, v in (r or {}).items():
                    key = f"{_cls}.{kk}"
                    lo, hi, _ = doors.get(key, (v, v, v))
                    doors[key] = (min(lo, v), max(hi, v), v)
                return r
            cls.get_door_state = gds
    import robosuite.environments.base as B
    orig_reset = B.MujocoEnv.reset
    dump = os.environ.get("HP_SUCCESS_DUMP")   # dir: save model + final state of in-container-but-on-counter ends

    def reset(self, *a, **k):
        patch_modules()
        cs = type(self).__dict__.get("_check_success") or getattr(type(self), "_check_success", None)
        if cs is not None and not getattr(cs, "_hp_wrapped", False):
            def check(self_, *a_, _orig=cs, **k_):
                r = _orig(self_, *a_, **k_)
                succ["n"] += 1
                if r and not succ["ever"]:
                    succ["ever"], succ["first"] = True, succ["n"]
                return r
            check._hp_wrapped = True
            type(self)._check_success = check
        if last or doors:
            print("[success-debug] " + f"success_ever={int(succ['ever'])} first_call={succ['first']} calls={succ['n']} " +
                  " ".join(f"{k}={int(v)}/ever={int(ever[k])}" for k, v in sorted(last.items())) + " " +
                  " ".join(f"{k}=last:{v[2]:.4f},min:{v[0]:.4f},max:{v[1]:.4f}" for k, v in sorted(doors.items())),
                  flush=True)
            if dump and last.get("check_obj_fixture_contact:obj") and last.get("check_obj_in_receptacle:obj,container"):
                import mujoco, numpy as np, time
                os.makedirs(dump, exist_ok=True)
                base = os.path.join(dump, f"{os.getpid()}_{int(time.time())}")
                mujoco.mj_saveModel(self.sim.model._model, base + ".mjb", None)
                open(base + ".xml", "w").write(self.sim.model.get_xml())
                d = self.sim.data._data
                np.savez(base + ".npz", qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy())
                print(f"[success-debug] dumped {base}.*", flush=True)
        last.clear(); ever.clear(); doors.clear(); succ.update(ever=False, first=-1, n=0)
        return orig_reset(self, *a, **k)
    B.MujocoEnv.reset = reset
