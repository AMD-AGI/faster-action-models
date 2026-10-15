# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-resident LIBERO env: the GPU simulator steps the physics AND renders.

Drop-in for LIBERO's ``OffScreenRenderEnv``. The surface an eval loop uses is
small and fully enumerated: ``close / reset / set_init_state / step / robots /
env.sim / render / seed``. Everything else about the eval -- image orientation, texture
upload, proprio keys, quaternion convention, resolution, episode bookkeeping --
is reused from LIBERO/robosuite rather than reimplemented, which is the whole point of
this wrapper.

Architecture:
  * a REAL robosuite env is still constructed, but only for SETUP and OBS SHAPE:
    scene XML, init states, the pre-warmup home pose, and robosuite's own
    observation dict as a template. It does NOT step during an episode.
  * ``step()`` drives ``BatchedInstinctSimEnv.step_all`` (GPU physics + GPU OSC)
    and overwrites the obs dict in place, so downstream code that reads
    ``obs["agentview_image"]`` / ``robot0_eef_pos`` / ... is unchanged.

Reset transfers the reference controller's actual home, goals, gripper command,
and (when valid) cached first-tick inputs. LIBERO's set_init_state changes the
physics state without invalidating robosuite's controller cache. Replacing that
cache with the new episode pose changes the first warmup goal and can shift the
first policy observation by millimeters. The transfer performs no physics step;
all subsequent controller evaluation and integration stay on the GPU.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Any

import numpy as np


_STEP_SUCCESS_KEY = "_gpu_resident_round_success"
_PREMERGED_KEY = "_gpu_resident_premerged_obs"


class _RoundDataView:
    """Read-once view of an engine's data for one locked physical round.

    Only zero-argument whole-batch reads are memoized; everything else (and any
    call with arguments) is forwarded untouched, so hasattr() and all other
    behaviour match the real data object. Callers copy what they index
    (np.array(...)), and the view is dropped at the end of the round.
    """
    _CACHED = frozenset(("read_xpos", "read_xquat", "read_qpos", "read_site_xpos",
                         "read_site_xmat", "read_contacts"))
    __slots__ = ("_data", "_memo")

    def __init__(self, data):
        self._data, self._memo = data, {}

    def __getattr__(self, name):
        attr = getattr(self._data, name)
        if name not in _RoundDataView._CACHED:
            return attr
        memo = self._memo

        def read(*args, **kwargs):
            if args or kwargs:
                return attr(*args, **kwargs)
            if name not in memo:
                memo[name] = attr()
            return memo[name]
        return read


class _RoundEngineView:
    __slots__ = ("_engine", "data")

    def __init__(self, engine):
        self._engine, self.data = engine, _RoundDataView(engine.data)

    def __getattr__(self, name):
        return getattr(self._engine, name)


class GpuResidentLiberoEnv:
    """robosuite-compatible facade over BatchedInstinctSimEnv."""

    def __init__(self, ref_env: Any, gpu_id: int = 0,
                 resolution: int = 256, success_metadata: dict | None = None,
                 env_slot: int = 0, num_envs: int = 1, shared: Any = None):
        """Slot ``env_slot`` of ``num_envs`` sharing ONE BatchedInstinctSimEnv.

        Slot 0 builds the engine (widened to ``num_envs``) and publishes it via
        the returned ``.shared`` handle; slots 1..N-1 receive that handle and
        attach to it. Only ONE engine exists per process, because
        ``step_all`` advances every env in a single call and N engines would
        neither batch nor fit in VRAM.

        """
        self._ref = ref_env          # setup + obs template ONLY; never stepped
        self._res = resolution
        self._slot = int(env_slot)
        self._nenv = int(num_envs)
        _sync_fixed = os.environ.get("GPU_RESIDENT_SYNC_FIXED_POSES", "1")
        if _sync_fixed not in ("0", "1"):
            raise ValueError("GPU_RESIDENT_SYNC_FIXED_POSES must be 0 or 1")
        self._sync_fixed_poses = _sync_fixed == "1"
        self._last_fixed_pose_updates: list[str] = []
        import mujoco  # noqa: F401
        from instinct_sim_env import hip_env
        from instinct_sim_render_worker import extract_textures

        if shared is not None:
            # ATTACH. Everything below (engine build, texture upload, ssaa
            # re-init, renderer config) already ran on slot 0 and is per-ENGINE,
            # not per-slot, so redoing it would at best waste a 195 MB texture
            # upload and at worst reset shared renderer state mid-episode.
            self._env = shared["env"]
            self._mjm = shared["mjm"]
            self._barrier = shared["barrier"]
            self._visual = shared["site_visual"]
            if self._visual._env is not self._env or self._visual._barrier is not self._barrier:
                raise RuntimeError("Peer attached to a different visual engine/lock")
            if self._nenv != self._visual._count or not 0 <= self._slot < self._nenv:
                raise RuntimeError("Peer visual world count/index differs from the engine")
            self._barrier.raise_if_poisoned()
            self.shared = shared
            self._home = shared["home"]
            if self._sync_fixed_poses and "fixed_body_poses" not in shared:
                raise RuntimeError("Shared GPU engine lacks its fixed-body pose snapshot")
            self._fixed_body_poses = shared.get("fixed_body_poses", {})
            if self._nenv > 1:
                # Per-world fixture poses: each slot tracks its own world.
                import copy
                self._fixed_body_poses = copy.deepcopy(self._fixed_body_poses)
            self._obs = {}
            self._barrier.online(self._slot)
            return

        xml = ref_env.env.sim.model.get_xml()
        self._mjm = mujoco.MjModel.from_xml_string(
            re.sub(r' colorspace="[^"]*"', '', xml))
        tex = extract_textures(xml)
        # Mass properties MUST come from robosuite's LIVE model. get_xml() does
        # not preserve which geoms contribute mass, so recompiling it counts each
        # object's visual mesh (group 1, contype 0): LIBERO objects become 3-8x
        # too heavy (bowl 0.0459 vs 0.0056 kg, ramekin 0.038 vs 0.0097, table
        # 127 vs 60) with shifted COMs, and invweight0 is off by up to 1e5.
        live = getattr(ref_env.env.sim.model, "_model", None)
        inertial_src = live if live is not None else self._mjm
        import inspect
        _rkw = ({"mj_model": inertial_src}
                if "mj_model" in inspect.signature(hip_env.BatchedInstinctSimEnv.__init__).parameters else {})
        self._env = hip_env.BatchedInstinctSimEnv(
            xml_string=hip_env.inject_mujoco_inertial(xml, inertial_src),
            num_envs=self._nenv,
            camera_height=resolution, camera_width=resolution, gpu_id=gpu_id,
            body_invweight0=inertial_src.body_invweight0.copy(),
            dof_invweight0=inertial_src.dof_invweight0.copy(),
            success_metadata=success_metadata,
            texture_info=tex["texture_info"], tex_data=tex["tex_data"],
            geom_texids=tex["geom_texids"], geom_texrepeat=tex["geom_texrepeat"], **_rkw)
        # Box-mesh / mesh-mesh collision through the MuJoCo 3.9 mjc_Convex port, fed with
        # MuJoCo's own mesh data (the legacy GPU mesh collider gives different contact
        # counts on gripper-mesh vs object-box grasps). HP_MUJOCO_CONVEX=0 disables it.
        if os.environ.get("HP_MUJOCO_CONVEX", "1") == "1" and live is not None and                 hasattr(self._env.engine, "set_mujoco_convex"):
            from instinct_sim_env.mujoco_convex import mujoco_convex_data
            self._env.engine.set_mujoco_convex(mujoco_convex_data(self._env.model, live))
            print("[GPU-RESIDENT] MuJoCo convex collision (mjc_Convex port) enabled", flush=True)
        # Geom / joint / body parameters at MuJoCo's double precision (load_from_xml parses float).
        # HP_MODEL_DOUBLES=0 keeps the widened float parse.
        if os.environ.get("HP_MODEL_DOUBLES", "1") == "1" and live is not None and \
                hasattr(self._env.model, "model_array_d"):
            from instinct_sim_env.mujoco_convex import sync_model_doubles
            _rep = sync_model_doubles(self._env.model, live)
            print("[GPU-RESIDENT] model doubles from MjModel (replaced, kept):", _rep, flush=True)
        # robosuite-timed proprio (HP_OBS_LAG): computed on the GPU inside step_all
        # (BatchedInstinctSimEnv.obs_lag_steps). HP_OBS_LAG_IMPL=cpu keeps the original CPU
        # reference implementation (_robosuite_lagged_proprio) for validation only.
        _lag = int(os.environ.get("HP_OBS_LAG", "0"))
        if _lag > 0 and os.environ.get("HP_OBS_LAG_IMPL", "gpu") == "gpu":
            if not hasattr(self._env.engine, "lagged_kinematics"):
                raise RuntimeError("HP_OBS_LAG needs a build with engine.lagged_kinematics")
            self._env.obs_lag_steps = _lag
            print(f"[GPU-RESIDENT] robosuite-timed proprio on GPU (lag {_lag})", flush=True)
        if tex.get("tex_iscube") is not None:
            tw = [x["width"] for x in tex["texture_info"].get("textures", [])]
            th = [x["height"] for x in tex["texture_info"].get("textures", [])]
            try:
                self._env.model.set_textures(
                    tex["tex_data"], tw, th, tex["geom_texids"].tolist(),
                    tex["geom_texrepeat"], tex_iscube=tex["tex_iscube"])
            except TypeError:
                pass   # build predates the explicit tex_iscube flag
        # Raster renderer (HP_RENDERER=raster, the fallback): re-init at SSAA=4. hip_env calls
        # `CameraRenderer.init(model, w, h)` without the ssaa argument, and ssaa=4 is the
        # validated raster setting (ssaa=8 is measurably worse).
        _ssaa = int(os.environ.get("GPU_RESIDENT_SSAA", "4"))
        _rt = bool(getattr(self._env, "_rt", False))
        if _rt:
            # HP_RENDERER=rt: the RT snapshots the model's textures at init, so rebuild it after the
            # tex_iscube upload above. Its antialiasing is HP_RT_AA (default ssaa1 = none), not ssaa.
            self._env._renderer.init(self._env.model, resolution, resolution)
            print(f"[GPU-RESIDENT] RT re-init after the tex_iscube upload "
                  f"(renderer=rt aa={self._env.renderer_cfg['rt_aa']}; GPU_RESIDENT_SSAA not used)",
                  file=sys.stderr, flush=True)
        elif _ssaa > 1:
            try:
                self._env._renderer.init(self._env.model, resolution, resolution,
                                         ssaa=_ssaa)
                print(f"[GPU-RESIDENT] renderer re-init at ssaa={_ssaa}",
                      file=sys.stderr, flush=True)
            except TypeError:
                print("[GPU-RESIDENT] build predates ssaa arg; staying at ssaa=1",
                      file=sys.stderr, flush=True)
        # Metadata belongs to the final initialized renderer. This consumes the
        # already compiled setup model, never a new scene or CPU physics step.
        from .renderer_sort_metadata import install_sort_metadata
        install_sort_metadata(
            self._env._renderer, self._env.model, self._mjm,
            [mujoco.mj_id2name(self._mjm, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in range(self._mjm.ngeom)],
        )
        # Renderer config to match MuJoCo's images: set_geomgroup(0,False) /
        # set_gouraud_shading / set_shadow after init; without them frames are a
        # uniform ~4.4 grey levels darker.
        # skip_group0 matters most: robosuite hides geom group 0 unless
        # render_collision_mesh=True, and LIBERO defaults it False.
        _r = self._env._renderer
        for _call, _args in (("set_geomgroup", (0, False)),
                             ("set_gouraud_shading", (True,)),
                             ("set_shadow", (False,))):
            fn = getattr(_r, _call, None)
            if fn is not None:
                try:
                    fn(*_args)
                except Exception as _e:
                    print(f"[GPU-RESIDENT] {_call} failed: {_e}",
                          file=sys.stderr, flush=True)
        self._env.enable_gpu_osc(True)
        self._home = np.asarray(ref_env.env.robots[0].controller.initial_joint,
                                dtype=np.float64)
        # Keep the actual reference constants, not a serialized/reparsed copy:
        # MuJoCo's XML writer rounds values, which is not a reset-induced move.
        self._fixed_body_poses = self._reference_nonjoint_body_poses()
        self._obs: dict[str, Any] = {}
        # OWNER publishes the shared handle. The barrier is created even at N=1
        # so there is exactly one step path to reason about; at N=1 it fires
        # immediately on submit and costs one uncontended lock.
        from .barrier import StepBarrier
        self._barrier = StepBarrier(self._nenv, self._step_batch)
        from .site_visual_coordinator import SiteVisualCoordinator
        self._visual = SiteVisualCoordinator(self._env, self._barrier)
        self._barrier.online(self._slot)
        # At N>1 the shared snapshot stays the constructor (model-wide) pose that
        # peers attach with; each slot then publishes into its own world.
        import copy
        self.shared = {"env": self._env, "mjm": self._mjm,
                       "barrier": self._barrier, "home": self._home,
                       "site_visual": self._visual,
                       "fixed_body_poses": (copy.deepcopy(self._fixed_body_poses)
                                            if self._nenv > 1 else self._fixed_body_poses),
                       "submitters": [None] * self._nenv}
        if self._nenv > 1:
            print(f"[GPU-RESIDENT] engine widened to {self._nenv} envs, "
                  f"step barrier armed", file=sys.stderr, flush=True)

    def _step_batch(self, batch: np.ndarray):
        """One physical round, with each submitted facade's existing verdict.

        The barrier still owns the engine lock. A delayed waiter must receive
        the success value for its image's round, even if another forced round
        finishes before that waiter returns. Unsubmitted physical rows are not
        evaluated, and the original four-part step result is preserved.
        """
        with self._visual.transaction():
            slots = self._barrier.submitted_slots_locked()
            obs, rewards, dones, infos = self._visual.step_batch(
                batch, render_rows=slots if slots else None)
            if (len(infos) != self._nenv
                    or any(not isinstance(info, dict) or _STEP_SUCCESS_KEY in info
                           for info in infos)):
                raise ValueError("Malformed or already success-tagged round info")
            round_infos = [dict(info) for info in infos]
            if self._barrier.submitted_slots_locked() != slots:
                raise RuntimeError("submitted slot set changed inside a round")
            checker = getattr(self._env, "_success_checker", None)
            if os.environ.get("HP_SUCCESS_MODE") in (None, "", "robosuite"):
                checker = None     # step() asks robosuite instead (_robosuite_success)
            if checker is None:
                # Same verdict as BatchedInstinctSimEnv.check_success_env.
                for slot in slots:
                    round_infos[slot][_STEP_SUCCESS_KEY] = False
            elif slots:
                view = _RoundEngineView(self._env.engine)
                for slot in slots:
                    round_infos[slot][_STEP_SUCCESS_KEY] = bool(checker.check(view, slot))
            submitters = self.shared["submitters"]
            for slot in slots:
                facade = submitters[slot]
                if facade is not None and facade._slot == slot:
                    round_infos[slot][_PREMERGED_KEY] = (facade, facade._merge(obs[slot]))
            return obs, rewards, dones, round_infos

    def _seed_gripper_slot_only(self, k: int) -> None:
        """Gripper accumulator for slot k, leaving peers' rows untouched.

        ``set_osc_gripper_state`` takes the whole [N,2] batch, so the peer rows
        must be round-tripped rather than recomputed: read the LIVE accumulator
        back from the device and overwrite only row k. Recomputing peers from
        their finger qpos silently resets any peer that is mid-grasp (see
        set_init_state).

        At N=1 this is equivalent to the batch call, so the seeding that took arm
        tracking from 8.41x to 0.0070x of budget is preserved exactly.
        """
        env = self._env
        gr = getattr(env, "_gripper_ctrlrange", None)
        idx = getattr(env, "_gripper_actuator_indices", None)
        if gr is None or not idx:
            return
        try:
            acc = np.asarray(env.engine.read_osc_gripper_state(self._nenv),
                             dtype=np.float64).reshape(self._nenv, -1)[:, :2].copy()
        except Exception as _e:
            # Unallocated OSC state raises; zeros match the allocate-and-memset
            # path the C++ setter would take anyway.
            print(f"[GPU-RESIDENT] gripper readback failed ({_e}); zeroing",
                  file=sys.stderr, flush=True)
            acc = np.zeros((self._nenv, 2), dtype=np.float64)
        qpos = np.asarray(env.engine.data.read_qpos_d()).reshape(self._nenv, -1)
        adrs = getattr(env, "_gripper_qpos_adr", None)
        for j in range(min(2, len(idx))):
            lo, hi = float(gr[j, 0]), float(gr[j, 1])
            bias, weight = 0.5 * (hi + lo), 0.5 * (hi - lo)
            adr = adrs[j] if adrs else (7 + j)
            if adr < qpos.shape[1] and abs(weight) > 1e-12:
                acc[k, j] = float(np.clip((qpos[k, adr] - bias) / weight, -1.0, 1.0))
        env.engine.set_osc_gripper_state(acc)
        env._gripper_current_action[k] = acc[k, :env._gripper_current_action.shape[1]]

    # ---- robosuite surface ------------------------------------------------
    @property
    def robots(self):
        return self._ref.robots

    @property
    def env(self):
        return self._ref.env

    def seed(self, s):
        return self._ref.seed(s)

    def reset(self):
        self._barrier.raise_if_poisoned()
        if os.environ.get("HP_MJC_DIAG") == "1" and hasattr(self._env.engine, "mujoco_convex_fallbacks"):
            # cumulative count of queued convex pairs whose port result was withheld (no contacts)
            print(f"[GPU-RESIDENT] mjc_Convex fallbacks so far: {self._env.engine.mujoco_convex_fallbacks()}",
                  flush=True)
        try:
            # Every reset is LIBERO's own (hard) reset of the reference env, so fixture placements are
            # sampled exactly as on CPU MuJoCo and then mirrored into the GPU model below.
            # HP_SOFT_FIRST_RESET=1 (speed only, off by default) skips the hard reset on a slot's FIRST
            # reset; that keeps the constructor's fixture placement instead of a fresh sample.
            inner = self._ref.env
            soft = (os.environ.get("HP_SOFT_FIRST_RESET", "0") == "1"
                    and not getattr(self, "_ref_reset_done", False))
            hard = inner.hard_reset
            if soft:
                inner.hard_reset = False
            try:
                obs = self._ref.reset()
            finally:
                inner.hard_reset = hard
            self._ref_reset_done = True
            # Publish every world-fixed body's reset pose (not only the ones that differ from the
            # cached snapshot): the engine was built from get_xml(), whose writer rounds values, so an
            # unchanged cache does not prove the device holds LIBERO's exact pose. Any partial CPU/device
            # publication is fatal, never represented as a successful rollback or exposed as an observation.
            with self._visual.transaction():
                fixed = tuple(name for name, pose in self._reference_nonjoint_body_poses().items()
                              if pose["world_fixed"])
                self._sync_reference_fixed_body_poses(publish_names=fixed)
            return obs
        except BaseException as error:
            self._barrier.poison(error)
            raise

    def _reference_nonjoint_body_poses(self):
        """Snapshot local poses and topology for every non-jointed body by name."""
        model = self._ref.env.sim.model
        result = {}
        for bid in range(1, int(model.nbody)):
            if int(model.body_jntnum[bid]) != 0:
                continue
            name = model.body_id2name(bid)
            if not name or name in result:
                raise RuntimeError("Cannot synchronize an unnamed/duplicate non-jointed body")
            ancestor, fixed, depth = bid, True, 0
            while ancestor > 0:
                if ancestor >= model.nbody or depth >= model.nbody:
                    raise RuntimeError("Invalid reference body ancestry")
                fixed = fixed and int(model.body_jntnum[ancestor]) == 0
                ancestor = int(model.body_parentid[ancestor])
                depth += 1
            parent = int(model.body_parentid[bid])
            result[name] = {
                "pos": np.asarray(model.body_pos[bid], dtype=np.float64).copy(),
                "quat": np.asarray(model.body_quat[bid], dtype=np.float64).copy(),
                "parent": model.body_id2name(parent) if parent else "world",
                "world_fixed": fixed,
            }
        return result

    def _sync_reference_fixed_body_poses(self, *, publish_names=None):
        """Mirror reset-time fixture poses without rebuilding or stepping.

        LIBERO samples fixture positions into model.body_pos/body_quat during
        hard reset; those are NOT part of its saved qpos/qvel init states. The
        GPU engine was previously left at the constructor's fixture placement.
        Only fixed poses change here. Textures, visualization-site RGBA, solver
        settings, and all dynamic state retain their existing paths.
        """
        current = self._reference_nonjoint_body_poses()
        previous = self._fixed_body_poses
        if current.keys() != previous.keys():
            raise RuntimeError("Reference non-jointed body names changed; rebuild the GPU scene")
        changes = []
        for name, pose in current.items():
            old = previous[name]
            if pose["parent"] != old["parent"] or pose["world_fixed"] != old["world_fixed"]:
                raise RuntimeError(f"Reference body topology changed for {name}; rebuild the GPU scene")
            if (not np.array_equal(pose["pos"], old["pos"])
                    or not np.array_equal(pose["quat"], old["quat"])):
                if not pose["world_fixed"]:
                    raise RuntimeError(f"Non-fixed local body pose changed for {name}; rebuild the GPU scene")
                changes.append(name)
        if publish_names is not None:
            if (type(publish_names) is not tuple
                    or any(type(name) is not str for name in publish_names)
                    or len(publish_names) != len(set(publish_names))
                    or any(name not in current or not current[name]["world_fixed"] for name in publish_names)
                    or any(name not in publish_names for name in changes)):
                raise RuntimeError("Historical fixture publication has unexpected or non-fixed targets")
            # An equal reference cache does not establish native/XML equality:
            # the constructor's serialized pose may have been rounded.
            changes = list(publish_names)
        self._last_fixed_pose_updates = []
        if not changes:
            return
        if self._nenv != 1:
            self._sync_fixed_body_poses_world(current, changes)
            return
        setter = getattr(self._env.model, "set_fixed_body_poses", None)
        if setter is None:
            raise RuntimeError(
                "Fixed fixture poses changed but the GPU simulator lacks set_fixed_body_poses; "
                "rebuild the engine with the reset-parity patch. Set "
                "GPU_RESIDENT_SYNC_FIXED_POSES=0 only for an explicit baseline ablation.")
        ids = [int(self._env.model.body_name2id(name)) for name in changes]
        if any(bid <= 0 for bid in ids) or len(set(ids)) != len(ids):
            raise RuntimeError("Fixed fixture names do not map bijectively into the GPU model")
        positions = np.asarray([current[name]["pos"] for name in changes], dtype=np.float64)
        quaternions = np.asarray([current[name]["quat"] for name in changes], dtype=np.float64)

        # Prepare metadata before mutating the device, keeping the engine's XML
        # truthful for exact-model render/physics diagnostics. Preserve sites
        # and all other attributes instead of copying post-reset visual state.
        import xml.etree.ElementTree as ET
        xml_root = ET.fromstring(self._env.xml_string)
        bodies = {body.get("name"): body for body in xml_root.iter("body")}
        mj_ids = [int(self._mjm.body(name).id) for name in changes]
        for name in changes:
            if name not in bodies:
                raise RuntimeError(f"Fixed fixture {name} is absent from the engine XML")
            body = bodies[name]
            body.set("pos", " ".join(format(x, ".17g") for x in current[name]["pos"]))
            body.set("quat", " ".join(format(x, ".17g") for x in current[name]["quat"]))
            for alternate in ("axisangle", "euler", "xyaxes", "zaxis"):
                body.attrib.pop(alternate, None)
        updated_xml = ET.tostring(xml_root, encoding="unicode")
        with self._visual.transaction():
            self._env.engine.sync()
            setter(np.asarray(ids, dtype=np.int32), positions, quaternions)
            # No integration or CPU dynamics: refresh only derived GPU state.
            # The ordinary set_init_state/reset_env path clears contacts and
            # warmstarts before the subsequent settling steps.
            self._env.engine.forward()
            self._env.engine.sync()
            for row, mid in enumerate(mj_ids):
                self._mjm.body_pos[mid] = positions[row]
                self._mjm.body_quat[mid] = quaternions[row]
            self._env.xml_string = updated_xml
            self._fixed_body_poses = current
            self.shared["fixed_body_poses"] = current
            self._last_fixed_pose_updates = changes
        print(f"[GPU-RESIDENT] synchronized fixed fixture poses: {changes}",
              file=sys.stderr, flush=True)

    def _sync_fixed_body_poses_world(self, current, changes):
        """N>1: publish fixed fixture poses into this slot's world only.

        The XML, compiled CPU model and native host descriptor are shared by all
        worlds, so they keep the constructor pose; the per-world device storage
        is what FK reads. finish_pending_restore reads it back.
        """
        setter = getattr(self._env.model, "set_fixed_body_poses_env", None)
        if setter is None:
            raise RuntimeError(
                "Fixed fixture poses changed at num_envs>1 but the GPU simulator lacks "
                "set_fixed_body_poses_env; rebuild the engine with per-world poses.")
        ids = [int(self._env.model.body_name2id(name)) for name in changes]
        if any(bid <= 0 for bid in ids) or len(set(ids)) != len(ids):
            raise RuntimeError("Fixed fixture names do not map bijectively into the GPU model")
        positions = np.asarray([current[name]["pos"] for name in changes], dtype=np.float64)
        quaternions = np.asarray([current[name]["quat"] for name in changes], dtype=np.float64)
        with self._visual.transaction():
            self._env.engine.sync()
            setter(self._nenv, self._slot, np.asarray(ids, dtype=np.int32), positions, quaternions)
            self._env.engine.forward()
            self._env.engine.sync()
            self._fixed_body_poses = current
            self._last_fixed_pose_updates = changes
        if os.environ.get("GPU_RESIDENT_DIAG") == "1":
            print(f"[GPU-RESIDENT] slot{self._slot} synchronized fixed fixture poses: {changes}",
                  file=sys.stderr, flush=True)

    def close(self):
        # Deregister FIRST: a closing slot that is still a barrier participant
        # makes every peer wait out the full timeout and then raise.
        #
        # LIMITATION, multi-env + multi-task. At N>1 the engine is SHARED (one
        # scene XML), so every slot must run the SAME task; a new task needs a
        # new engine for all slots (HipLiberoVectorEnv is built per task).
        try:
            self._barrier.leave(self._slot)
        except Exception:
            pass
        try:
            self._ref.close()
        except Exception:
            pass
        # RELEASE THE HIP ENGINE. An eval that builds a new env per task would
        # otherwise leak a whole BatchedInstinctSimEnv (engine, GPU buffers and
        # ~195 MB of textures) per task, and success rates decay with task index.
        #
        # The GPU simulator exposes no teardown -- BatchedInstinctSimEnv.close() sets a
        # flag and InstinctSimEnv.close() is `pass` (hip_env.py:892, 2812) -- so
        # dropping the last reference and forcing a collection is the only lever
        # available from here. Guarded: at N>1 the engine is SHARED, and tearing
        # it down would destroy peers' physics mid-episode.
        if self._nenv == 1 and getattr(self, "shared", None) is None:
            try:
                env = getattr(self, "_env", None)
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
                    for attr in ("_renderer", "engine"):
                        try:
                            setattr(env, attr, None)
                        except Exception:
                            pass
                self._env = None
                import gc
                gc.collect()
            except Exception as _e:
                print(f"[GPU-RESIDENT] engine release failed: {_e}",
                      file=sys.stderr, flush=True)

    def render(self, *a, **k):
        return self._ref.render(*a, **k)

    def set_init_state(self, state):
        """Transfer episode/controller state without a reset-only physics step.

        The reference is used for setup only. Its controller cache intentionally
        need not describe the newly injected qpos: this is also the state that
        CPU robosuite consumes on its first warmup call. A native
        one-shot cache preserves that contract, then normal GPU updates resume.
        Leaving a barrier may complete an already-pending peer round; the reset
        itself submits no action and adds no extra latch/priming integration.
        """
        k = self._slot
        self._barrier.raise_if_poisoned()
        try:
            obs = self._ref.set_init_state(state)
            sim = self._ref.env.sim
            q0 = np.asarray(sim.data.qpos, dtype=np.float64).copy()
            v0 = np.asarray(sim.data.qvel, dtype=np.float64).copy()
            controller_state = self._reference_osc_reset_state()
        except BaseException as error:
            self._barrier.poison(error)
            raise
        # Leave the barrier for the duration: peers must not wait on a slot that
        # is resetting rather than stepping, and this slot must not be counted
        # as a participant until its next step() re-registers it.
        self._barrier.leave(k)
        with self._barrier._cv:
            self._visual.ensure_healthy()
            try:
                # reset_env already gathers a discarded observation: publish
                # before it, retaining both original gathers and no advance.
                self._visual.reset_from_reference(k, self._ref.env)
                self._env.reset_env(k)
                self._env.engine.data.write_qpos_d_env(q0, k)
                self._env.engine.data.write_qvel_d_env(v0, k)
                self._env.engine.initialize_osc_env_state(k, controller_state)
                self._env._gripper_current_action[k] = controller_state["gripper"]
                self._env.engine.reset_warmstart_env(k)
                self._env.engine.forward()
                self._env.engine.sync()
                self._obs = dict(obs)
                # No extra render, physical substep or callback during reset.
                return self._merge(self._visual.attach(self._env._get_obs(k), k))
            except BaseException as error:
                self._visual.fail(error)
                raise

    def _reference_osc_reset_state(self):
        """Copy setup-time controller state, without updating or evaluating it."""
        robot = self._ref.env.robots[0]
        ctl = robot.controller
        grip = np.asarray(robot.gripper.current_action, dtype=np.float64)
        state = {
            "home": np.array(ctl.initial_joint, dtype=np.float64, copy=True),
            "goal_pos": np.array(ctl.goal_pos, dtype=np.float64, copy=True),
            "goal_ori": np.array(ctl.goal_ori, dtype=np.float64, copy=True),
            "gripper": np.broadcast_to(grip, (2,)).copy(),
        }
        if not ctl.new_update:
            state["cache"] = {
                "ee_pos": np.array(ctl.ee_pos, dtype=np.float64, copy=True),
                "ee_ori": np.array(ctl.ee_ori_mat, dtype=np.float64, copy=True),
                "ee_vel": np.concatenate([ctl.ee_pos_vel, ctl.ee_ori_vel]).astype(np.float64),
                "joint_pos": np.array(ctl.joint_pos, dtype=np.float64, copy=True),
                "joint_vel": np.array(ctl.joint_vel, dtype=np.float64, copy=True),
                "J_full": np.array(ctl.J_full, dtype=np.float64, copy=True),
                "mass_matrix": np.array(ctl.mass_matrix, dtype=np.float64, copy=True),
            }
        return state

    def step(self, action):
        """GPU physics + GPU OSC. Returns robosuite's (obs, reward, done, info)."""
        k = self._slot
        a = np.asarray(action, dtype=np.float32).reshape(-1)[:7]
        # ONE step_all per round across all active slots; each reads its own row.
        submitters = self.shared.get("submitters")
        if submitters is not None:
            submitters[k] = self
        obs_list, _, _, round_infos = self._barrier.step(k, a)
        done = round_infos[k][_STEP_SUCCESS_KEY]
        if os.environ.get("HP_SUCCESS_MODE") in (None, "", "robosuite"):   # default; "native" = bundle checker
            # robosuite's own _check_success() on the GPU simulator's state, at robosuite's timing (see
            # _robosuite_success): the bundle checker (HP_SUCCESS_MODE=native) re-implements LIBERO's
            # predicates on the final state and disagrees on knife-edge On/In steps (goal ~ -2.75 / 500).
            done = self._robosuite_success()
        elif os.environ.get("HP_SUCCESS_XCHECK") == "1":
            self._success_xcheck(done)
        # 0/3 with clean 300-step episodes has several causes that look identical
        # from the outside: a checker that never fires, actions not moving the
        # arm, or a genuine miss. Report the evidence that separates them.
        self._n = getattr(self, "_n", 0) + 1
        # OBJECT-POSE TRACE. The 13 tasks that score exactly 0/10 all run to the
        # step cap, and settle physics is already exonerated (a bowl seeded at
        # the goal rests identically in hip and MuJoCo). What is NOT known is
        # whether the object is ever picked up at all. `success`/`steps` in the
        # result JSON cannot answer that, and GR-DIAG's 100-step sampling is too
        # coarse to attribute per task. Two scripted probes failed outright: a
        # hand-rolled reach never grasps (it also failed on a 9/10 control), so
        # the REAL policy has to be the driver. qpos is already read here, so the
        # object's own dofs cost nothing extra to log.
        # Separate env var: this is per-step and chatty, so it must not ride
        # along with the ordinary GPU_RESIDENT_DIAG eval runs.
        # `done` is computed above, EVERY step. The trace samples every N steps,
        # so pose and verdict have never been observed at the same instant: a
        # bowl that lands, is briefly in-slab, then slides out between samples
        # looks permanently in-slab in the trace and never in-slab to the
        # checker. Setting EVERY=1 puts both on one line for every step.
        if os.environ.get("GPU_RESIDENT_OBJTRACE") and self._n % int(
                os.environ.get("GPU_RESIDENT_OBJTRACE_EVERY", "10")) == 1 % max(
                    int(os.environ.get("GPU_RESIDENT_OBJTRACE_EVERY", "10")), 1):
            import sys as _s
            try:
                adr = self._objtrace_adr
            except AttributeError:
                # resolve the goal object's free-joint qpos address ONCE, by name
                # from the success bundle -- never by a name fragment (an earlier
                # diff keyed on `name.split('_')[1]` and read bowl-vs-plate
                # contacts as finger-vs-bowl for four iterations).
                # Read from the CHECKER, not from the bundle: `_obj_info` holds
                # addresses already re-resolved against hip's own model, whereas
                # the bundle's indices come from LIBERO's MuJoCo model.
                # (`_success_metadata` does not exist on the env -- the metadata
                # is consumed into the checker at construction.)
                adr, obj = -1, None
                chk = getattr(self._env, "_success_checker", None)
                gs = (getattr(chk, "goal_state", None) or [[]])[0] if chk else []
                obj = gs[1] if len(gs) > 1 else None
                if obj:
                    info = (getattr(chk, "_obj_info", None) or {}).get(obj, {})
                    qa = info.get("qpos_addrs") or []
                    adr = int(qa[0]) if qa else -1
                self._objtrace_adr = adr
                self._objtrace_name = obj
                print(f"[OBJTRACE] slot{k} object={obj} qpos_adr={adr}",
                      file=_s.stderr, flush=True)
            if adr >= 0:
                q = np.array(self._env.engine.data.read_qpos_d(),
                             dtype=np.float64, copy=True).reshape(self._nenv, -1)[k]
                if adr + 3 <= q.size:
                    print(f"[OBJTRACE] slot{k} step {self._n:4d} "
                          f"obj_xyz={np.round(q[adr:adr+3],4).tolist()} "
                          f"grip={np.round(q[7:9],4).tolist()} "
                          # COMMANDED gripper vs ACTUAL finger qpos. The trace
                          # showed the bowl carried to the target and HELD there
                          # with the fingers closed, so the release never
                          # happens -- but "policy never commands open" and
                          # "fingers ignore an open command" need opposite
                          # fixes. a[-1] is the gripper channel: <0 opens.
                          f"cmd_grip={float(a[-1]):+.3f} done={done} "
                          # site-frame dz/dxy computed HERE, from the same qpos
                          # read the pose above came from, so the geometry and
                          # the verdict describe one instant.
                          f"{self._objtrace_slab(q, adr)}",
                          file=_s.stderr, flush=True)
        if os.environ.get("GPU_RESIDENT_DIAG") == "1" and self._n % 100 == 1:
            import sys as _s
            q = np.array(self._env.engine.data.read_qpos_d(),
                         dtype=np.float64, copy=True).reshape(self._nenv, -1)[k]
            bs = self._barrier.stats()
            print(f"[GR-DIAG] slot{k} step {self._n:4d} |act|={np.abs(a).max():.3f} "
                  f"arm={np.abs(q[:7]).max():.4f} "
                  f"grip={np.round(q[7:9],4).tolist()} "
                  f"checker={'ON' if self._env._success_checker is not None else 'MISSING'} "
                  f"done={done} barrier(rounds={bs['rounds']} "
                  f"avg_part={bs['avg_participants']:.2f} active={bs['active']})",
                  file=_s.stderr, flush=True)
        premerged = round_infos[k].get(_PREMERGED_KEY)
        if premerged is not None and premerged[0] is self:
            merged = premerged[1]
        else:
            merged = self._merge(obs_list[k])
        lag = int(os.environ.get("HP_OBS_LAG", "0"))
        if lag > 0 and os.environ.get("HP_OBS_LAG_IMPL", "gpu") == "cpu":
            merged = self._robosuite_lagged_proprio(merged, lag)
        elif lag > 0 and os.environ.get("HP_OBS_LAG_CHECK") == "1":
            # GPU lag vs the CPU reference on the same state (diagnostic only)
            ref = self._robosuite_lagged_proprio(merged, lag)
            d = {kk: float(np.max(np.abs(np.asarray(merged[kk], np.float64) - np.asarray(ref[kk], np.float64))))
                 for kk in self._LAG_KEYS if kk in merged and kk in ref}
            ref0 = self._robosuite_lagged_proprio(merged, 0)   # same state, no lag: effect size
            e = {kk: float(np.max(np.abs(np.asarray(ref[kk], np.float64) - np.asarray(ref0[kk], np.float64))))
                 for kk in d}
            print(f"[OBS_LAG_CHECK] slot{k} step {self._n} max|gpu-cpu| " +
                  " ".join(f"{kk.replace('robot0_', '')}={v:.2e}" for kk, v in d.items()) +
                  " | lag effect " + " ".join(f"{kk.replace('robot0_', '')}={v:.2e}" for kk, v in e.items()),
                  flush=True)
        return merged, float(done), done, {}

    _LAG_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")

    def _round_state(self):
        """The GPU simulator's qpos / qvel of every world for the current physical round, read once and shared by
        all slots of the engine (keyed by the barrier's round counter; the next round cannot start until
        every slot, including this one, has submitted again)."""
        import threading
        sh = self.shared
        lock = sh.setdefault("_round_state_lock", threading.Lock())
        key = self._barrier.rounds
        with lock:
            cached = sh.get("_round_state")
            if cached is None or cached[0] != key:
                eng = self._env.engine
                eng.sync()
                qa = np.asarray(eng.data.read_qpos_d(), dtype=np.float64).reshape(self._nenv, -1)
                va = np.asarray(eng.data.read_qvel_d(), dtype=np.float64).reshape(self._nenv, -1)
                cached = (key, qa, va)
                sh["_round_state"] = cached
        return cached[1], cached[2]

    def _robosuite_success(self, lag=1):
        """robosuite's _check_success() evaluated on the GPU simulator's state. robosuite calls it after the last
        substep, but its body positions and contacts are those mj_step computed at the START of that
        substep, i.e. the state one integration before the final one. Step the GPU simulator's final qpos back
        along its qvel by `lag` substeps (mj_integratePos), mj_forward the (otherwise unstepped)
        reference env there, and ask robosuite."""
        import mujoco
        ref = self._ref.env
        m = ref.sim.model._model
        if not hasattr(self, "_lag_map"):
            self._robosuite_lagged_proprio({}, 0)      # builds the hip -> MuJoCo qpos/qvel index map
        qmap, vmap = self._lag_map
        qa, va = self._round_state()
        qh, vh = qa[self._slot], va[self._slot]
        q = np.ascontiguousarray(qh[qmap], dtype=np.float64)
        v = np.ascontiguousarray(vh[vmap], dtype=np.float64)
        if lag:
            mujoco.mj_integratePos(m, q, v, -lag * m.opt.timestep)
        d = ref.sim.data._data
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_forward(m, d)
        return bool(ref._check_success())

    def _success_xcheck(self, done):
        """HP_SUCCESS_XCHECK=1: load the GPU simulator's final state into the (otherwise unstepped) reference
        robosuite env and evaluate robosuite's own _check_success() on it. Logs every step where the
        GPU success checker and robosuite disagree."""
        import mujoco
        ref = self._ref.env
        m = ref.sim.model._model
        if not hasattr(self, "_lag_map"):
            self._robosuite_lagged_proprio({}, 0)      # builds the hip -> MuJoCo qpos/qvel index map
        qmap, vmap = self._lag_map
        eng = self._env.engine
        eng.sync()
        qh = np.asarray(eng.data.read_qpos_d()).reshape(self._nenv, -1)[self._slot]
        ref.sim.data.qpos[:] = qh[qmap]
        ref.sim.data.qvel[:] = 0.0
        mujoco.mj_forward(m, ref.sim.data._data)
        cpu = bool(ref._check_success())
        self._xc = getattr(self, "_xc", [0, 0, 0])
        self._xc[0] += 1
        if cpu != bool(done):
            self._xc[1 if cpu else 2] += 1
            print(f"[SUCCESS-XCHECK] step {getattr(self, '_n', 0) + 1}: gpu={bool(done)} robosuite={cpu} "
                  f"(steps {self._xc[0]}, robosuite-only {self._xc[1]}, gpu-only {self._xc[2]})", flush=True)

    def _robosuite_lagged_proprio(self, obs, lag):
        """CPU reference for HP_OBS_LAG (HP_OBS_LAG_IMPL=cpu); the default path is on the GPU.
        Report proprio the way robosuite does. robosuite samples its observables inside the
        substep loop, so the eef pose / gripper qpos it returns after a policy step are those of
        the state `lag` (measured: 2) substeps before the end, while the GPU simulator reports the final
        state. Step the final state back with MuJoCo's own mj_integratePos(-lag*h), load it into
        the (otherwise unstepped) reference env and read robosuite's observables there."""
        import mujoco
        ref = self._ref.env
        m = ref.sim.model._model
        if not hasattr(self, "_lag_map"):
            hp = self._env.model
            hq = {hp.joint_name(i): (hp.joint_qpos_adr(i), hp.joint_dof_adr(i)) for i in range(hp.njnt)}
            qmap, vmap = np.empty(m.nq, dtype=np.intp), np.empty(m.nv, dtype=np.intp)
            widths = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
            for j in range(m.njnt):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                wq, wv = widths[int(m.jnt_type[j])]
                ha, hv = hq[name]
                qmap[m.jnt_qposadr[j]:m.jnt_qposadr[j] + wq] = np.arange(ha, ha + wq)
                vmap[m.jnt_dofadr[j]:m.jnt_dofadr[j] + wv] = np.arange(hv, hv + wv)
            self._lag_map = (qmap, vmap)
        qmap, vmap = self._lag_map
        eng = self._env.engine
        eng.sync()
        qh = np.asarray(eng.data.read_qpos_d()).reshape(self._nenv, -1)[self._slot]
        vh = np.asarray(eng.data.read_qvel_d()).reshape(self._nenv, -1)[self._slot]
        q = np.ascontiguousarray(qh[qmap], dtype=np.float64)
        v = np.ascontiguousarray(vh[vmap], dtype=np.float64)
        mujoco.mj_integratePos(m, q, v, -lag * m.opt.timestep)
        ref.sim.data.qpos[:] = q
        ref.sim.data.qvel[:] = v
        ref.sim.forward()
        lagged = ref._get_observations(force_update=True)
        out = dict(obs)
        for key in self._LAG_KEYS:
            if key in lagged:
                out[key] = np.array(lagged[key], copy=True)
        return out

    def _objtrace_slab(self, q, adr):
        """Site-frame dz/dxy of the goal object vs the checker's own slab.

        Reads the SAME site pose the checker reads (f32 `read_site_xpos`) so a
        disagreement between this string and `done=` cannot be a units or frame
        artefact -- it would mean the predicate's other clause (parent contact)
        is what fails. Returns '' if anything is missing, so the trace degrades
        rather than throwing inside the step loop.
        """
        try:
            chk = getattr(self._env, "_success_checker", None)
            gs = (getattr(chk, "goal_state", None) or [[]])[0]
            if len(gs) < 3:
                return ""
            site = gs[2]
            info = (getattr(chk, "_obj_info", None) or {}).get(site, {})
            sid = info.get("site_id", -1)
            if sid < 0:
                return ""
            sx = np.array(self._env.engine.data.read_site_xpos())[self._slot][sid]
            sm = np.array(self._env.engine.data.read_site_xmat())[self._slot][sid].reshape(3, 3)
            size = np.asarray(info.get("site_size", [0.05] * 3), dtype=np.float64)
            d = sm.T @ (np.asarray(q[adr:adr + 3], dtype=np.float64) - sx)
            z_ok = (size[2] - 0.005) < d[2] < (size[2] + 0.10)
            xy_ok = bool(np.all(np.abs(d[:2]) < size[:2]))
            pc = "?"
            try:
                pbid = info.get("parent_body_id", -1)
                obids = chk._get_body_ids(gs[1]) if len(gs) > 1 else []
                if pbid >= 0 and obids:
                    cc = self._env.engine.data.read_contacts()
                    nn = int(np.asarray(cc["ncon"]).ravel()[self._slot])
                    b1 = np.asarray(cc["body1"])[self._slot][:nn]
                    b2 = np.asarray(cc["body2"])[self._slot][:nn]
                    prs = {(min(int(x), int(y)), max(int(x), int(y)))
                           for x, y in zip(b1, b2)}
                    fam = [pbid] + list(chk._collect_subtree(pbid))
                    pc = str(any((min(p, o), max(p, o)) in prs
                                 for p in fam for o in obids))
            except Exception:
                pass
            return (f"site={site} site_z={sx[2]:.4f} "
                    f"dx={d[0]:+.4f} dy={d[1]:+.4f} dz={d[2]:+.4f} "
                    f"under={z_ok and xy_ok} pcontact={pc}")
        except Exception as _e:
            return f"slab_err={type(_e).__name__}"

    def cleanup(self):
        """PERMANENT departure: this slot's work queue is drained.

        Must be permanent, not the transient episode-boundary leave -- otherwise
        the cohort still expects this slot and the remaining episodes time out.
        """
        try:
            self._barrier.leave(self._slot, permanent=True)
        except Exception:
            pass

    def _merge(self, hip_obs: dict) -> dict:
        """Overwrite the robosuite obs template with GPU state + frames.

        IMAGES NEED A SINGLE VERTICAL FLIP. hip_env renders in STANDARD
        convention; robosuite (and therefore the policies' preprocessing, which
        flips BOTH axes) expects robosuite's bottom-up convention, so exactly one
        flip, `img[::-1]`, is applied here. Any other orientation is off by
        ~57 grey levels MAE against MuJoCo's image.
        """
        # Keep retained observations paired with their original image receipt.
        # A later step/reset must not replace entries in an older dictionary.
        merged = dict(self._obs)
        for k, v in hip_obs.items():
            if v is None and k.endswith("_image"):
                raise RuntimeError(f"merge of unrendered image {k!r} (idle-world placeholder)")
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
                v = np.ascontiguousarray(v[::-1])
            elif isinstance(v, np.ndarray):
                v = v.copy()
            merged[k] = v
        self._obs = merged
        return self._obs
