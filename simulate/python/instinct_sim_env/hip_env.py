# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
InstinctSimEnv - GPU-accelerated environment matching LIBERO OffScreenRenderEnv API.

Replaces MuJoCo CPU physics + OSMesa rendering with GPU physics +
GPU rendering (the HIP ray tracer by default, or the
rasterizer with HP_RENDERER=raster / renderer='raster'; instinct_sim_env/rt_renderer.py).
Provides the same observation dict format that LIBERO evaluation scripts expect.

Usage:
    env = InstinctSimEnv(xml_string=scene_xml, camera_height=256, camera_width=256,
                        mj_model=live_mj_model)   # the ray tracer is configured from it
    # or, for the rasterizer:  InstinctSimEnv(..., renderer="raster")
    obs = env.reset()
    obs, reward, done, info = env.step(action)

    # Or from a compiled MuJoCo model:
    env = InstinctSimEnv.from_mujoco_xml(xml_path, ...)
"""

import sys
import os
import math
import numpy as np
import time as _hp_time
# HP_OBS_TIMERS=1: record a host-side wall split of step_all into env._obs_tlog
# (list of dicts, ms). Off by default; costs one bool test per step when off.
_HP_OBS_TIMERS = os.environ.get("HP_OBS_TIMERS", "0") == "1"
_HP_RENDER_SKIP_IDLE = os.environ.get("HP_RENDER_SKIP_IDLE", "1") != "0"
# world skip: HP_PHYS_SKIP=1 (default) lets step_all(step_rows=...) freeze every other
# world for that step (engine.set_world_active; see world_mask.h). =0 steps every
# world regardless.
_HP_PHYS_SKIP = os.environ.get("HP_PHYS_SKIP", "1") != "0"
from typing import Optional, Dict, List, Tuple, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))
import instinct_sim

# Relative-ish import with a flat fallback. Inside the package this resolves
# without sys.path; the fallback keeps the module usable if someone loads this
# file directly by path (several scratch scripts still do).
try:
    from instinct_sim_env.osc_controller import OSCController, quat_to_mat
except ImportError:  # loaded as a bare top-level module
    from osc_controller import OSCController, quat_to_mat
try:
    from instinct_sim_env.rt_renderer import renderer_settings, make_rt_renderer, mujoco_model_for
except ImportError:  # loaded as a bare top-level module
    from rt_renderer import renderer_settings, make_rt_renderer, mujoco_model_for


def inject_mujoco_inertial(xml_string: str, mj_model) -> str:
    """Inject <inertial> tags from a compiled MuJoCo model into MJCF XML.

    The GPU simulator computes body mass from geom volumes, which gives wrong results
    for mesh geoms (bounding-box approximation vs actual mesh volume). This
    function inserts <inertial> tags for every body that lacks one, using
    MuJoCo's pre-computed mass, COM, and inertia.
    """
    try:
        import mujoco as _mj
    except ImportError:
        return xml_string

    import re

    nb = mj_model.nbody
    # Build lookup: body name -> (mass, ipos, iquat, inertia)
    body_data = {}
    for bi in range(nb):
        name = _mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_BODY, bi)
        if not name or name == "world":
            continue
        mass = float(mj_model.body_mass[bi])
        ipos = mj_model.body_ipos[bi]
        inertia = mj_model.body_inertia[bi]
        iquat = mj_model.body_iquat[bi]
        body_data[name] = (mass, ipos, inertia, iquat)

    lines = xml_string.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)

        # Match <body name="..."> or <body name="..." ...>
        m = re.match(r'^(\s*)<body\s+name="([^"]+)"', line)
        if m:
            indent = m.group(1)
            bname = m.group(2)
            # Check if next non-empty line is already <inertial
            has_inertial = False
            for j in range(i+1, min(i+5, len(lines))):
                stripped = lines[j].strip()
                if stripped.startswith('<inertial'):
                    has_inertial = True
                    break
                if stripped and not stripped.startswith('<!--'):
                    break

            if not has_inertial and bname in body_data:
                mass, ipos, inertia, iquat = body_data[bname]
                if mass > 1e-12:
                    # Check if iquat is non-trivial
                    is_identity = (abs(iquat[0] - 1.0) < 1e-8 and
                                   abs(iquat[1]) < 1e-8 and
                                   abs(iquat[2]) < 1e-8 and
                                   abs(iquat[3]) < 1e-8)
                    tag = (f'{indent}  <inertial '
                           f'pos="{ipos[0]:.10g} {ipos[1]:.10g} {ipos[2]:.10g}" '
                           f'mass="{mass:.10g}" '
                           f'diaginertia="{inertia[0]:.10g} {inertia[1]:.10g} {inertia[2]:.10g}"')
                    if not is_identity:
                        tag += (f' quat="{iquat[0]:.10g} {iquat[1]:.10g} '
                                f'{iquat[2]:.10g} {iquat[3]:.10g}"')
                    tag += '/>'
                    out.append(tag)
        i += 1

    return '\n'.join(out)


def replace_decomposed_collision(xml_string: str) -> str:
    """Replace decomposed box collision geoms with native mesh collision.

    LIBERO models decompose mesh objects into many small box geoms for collision
    (e.g., bowl = 40 boxes). This causes noisy contact forces in the GPU simulator.
    Replace with the visual mesh geom (enabling collision on it) so the GPU simulator
    uses its convex hull collision instead.

    Only applies to free-floating bodies (with type="free" joint), not
    articulated parts like cabinet drawers where convex hulls would block
    joint motion.

    Detection: a body has decomposed collision if it has exactly 1 visual mesh
    geom (contype="0", group="1") and >=5 collision box geoms.
    """
    import xml.etree.ElementTree as ET
    import re as _re

    root = ET.fromstring(xml_string)
    replaced = 0

    for body in root.iter('body'):
        # Only replace for free-floating bodies (free joint)
        joint = body.find('joint')
        if joint is None or joint.get('type', 'hinge') != 'free':
            continue

        geoms = list(body.findall('geom'))
        if len(geoms) < 6:
            continue

        # Find visual mesh geom and collision box geoms
        visual_mesh = None
        collision_boxes = []
        for g in geoms:
            gtype = g.get('type', '')
            contype = g.get('contype', '1')  # default contype=1
            group = g.get('group', '0')
            if gtype == 'mesh' and contype == '0' and group == '1':
                visual_mesh = g
            elif gtype == 'box' and contype != '0':
                collision_boxes.append(g)

        if visual_mesh is None or len(collision_boxes) < 5:
            continue

        # Copy collision properties from first box to the mesh geom
        first_box = collision_boxes[0]
        visual_mesh.set('contype', '1')
        visual_mesh.set('conaffinity', '1')
        # Keep group="1" so it's not rendered as visual-only by the rasterizer
        # but ensure collision is enabled
        for attr in ('friction', 'solref', 'solimp', 'condim'):
            val = first_box.get(attr)
            if val is not None:
                visual_mesh.set(attr, val)

        # Remove decomposed box geoms
        for box in collision_boxes:
            body.remove(box)

        bname = body.get('name', '?')
        replaced += 1
        print(f"[MESH-COL] Body '{bname}': replaced {len(collision_boxes)} box "
              f"geoms with native mesh collision", flush=True)

    if replaced == 0:
        return xml_string

    return ET.tostring(root, encoding='unicode')


def resolve_tex_iscube(tex_iscube, texture_info, ntex: int, xml_string: str = None, mj_model=None):
    """Per-texture cube flags for PhysicsModel.set_textures, or None (legacy: no flag; every texture 2D).

    Order: the caller's tex_iscube, then texture_info['tex_iscube'], then MuJoCo's tex_type from
    mj_model (or xml_string compiled) with extract_textures' rule (tex_type == CUBE and a square or
    6-stacked layout). HP_TEX_ISCUBE=0 restores the legacy call without the flag. Without it the
    binding treats every cube map (LIBERO: the table, the skybox-shaped 6-stacks) as a 2D texture."""
    if os.environ.get("HP_TEX_ISCUBE", "1") == "0":
        return None
    if tex_iscube is None and texture_info:
        tex_iscube = texture_info.get("tex_iscube")
    if tex_iscube is None and (mj_model is not None or xml_string):
        try:
            m = mj_model
            if m is None:
                import mujoco
                m = mujoco.MjModel.from_xml_string(xml_string)
            if int(m.ntex) == ntex:
                tex_iscube = [int(int(m.tex_type[i]) == 1 and (int(m.tex_height[i]) == int(m.tex_width[i])
                                                             or int(m.tex_height[i]) == 6 * int(m.tex_width[i])))
                              for i in range(ntex)]
            else:
                print(f"[hip_env] tex_iscube: MuJoCo model has {int(m.ntex)} textures, texture_info {ntex}; "
                      f"not derived", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[hip_env] tex_iscube not derived ({type(e).__name__}: {e})", flush=True)
    if tex_iscube is None:
        return None
    tex_iscube = [int(v) for v in tex_iscube]
    if len(tex_iscube) != ntex:
        raise ValueError(f"tex_iscube has {len(tex_iscube)} entries for {ntex} textures")
    return tex_iscube


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert (w,x,y,z) to (x,y,z,w) for robosuite compatibility."""
    return np.array([q[1], q[2], q[3], q[0]])


def axisangle_to_quat_wxyz(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to (w,x,y,z) quaternion."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = aa / angle
    ha = angle * 0.5
    s = math.sin(ha)
    return np.array([math.cos(ha), axis[0]*s, axis[1]*s, axis[2]*s])


_ARTICULATED_THRESHOLDS = {
    "WoodenCabinet": {"open": ("lt", -0.14), "close": ("gt", 0.0)},
    "WhiteCabinet": {"open": ("lt", -0.14), "close": ("gt", 0.0)},
    "ShortCabinet": {"open": ("gt", 0.10), "close": ("lt", 0.0)},
    "ShortFridge": {"open": ("gt", 2.0), "close": ("lt", 0.0)},
    "Microwave": {"open": ("lt", -1.3), "close": ("gt", -0.005)},
    "FlatStove": {"turnon": ("ge", 0.5), "turnoff": ("lt", 0.0)},
}


class NativeSuccessChecker:
    """Bundle-driven BDDL success checker using GPU data only.

    Faithfully replicates LIBERO's predicate evaluation using body/site poses,
    joint qpos, and contact data from the GPU simulator.  Key predicates:

      On(obj1, obj2):
        - obj2 is ObjectState  → z-ordering + xy proximity + body contact
        - obj2 is SiteObjectState → under() slab check ± parent body contact

      In(obj1, obj2):
        - obj2 is SiteObjectState → in_box() (contact vacuously True)
        - obj2 is ObjectState → body contact + in_box()

      Open/Close/Turnon/Turnoff → joint qpos vs articulation thresholds
    """

    def __init__(self, success_source: Dict[str, Any], hip_model, seed: int = 42):
        if isinstance(success_source, dict) and "goal_state" in success_source:
            metadata = success_source
        else:
            raise ValueError("NativeSuccessChecker needs a success-metadata dict with 'goal_state' "
                             "(instinct_sim_libero.success_metadata.extract_success_metadata builds one)")

        self.goal_state = metadata["goal_state"]
        self._obj_info: Dict[str, Dict[str, Any]] = {}
        self._hip_model = hip_model

        self._body_children: Dict[int, List[int]] = {}
        nbody = getattr(hip_model, 'nbody', 0)
        if nbody > 0:
            # The native binding calls this body_parent; some older adapters
            # expose MuJoCo's body_parentid spelling. Silently assuming every
            # body is a root drops all fixture children from contact checks.
            parent_fn = getattr(hip_model, 'body_parent', None)
            if not callable(parent_fn):
                parent_fn = getattr(hip_model, 'body_parentid', None)
            if not callable(parent_fn):
                raise ValueError("Success checker requires body_parent or body_parentid")
            for bi in range(nbody):
                parent = int(parent_fn(bi))
                # MuJoCo's world parent is itself, unlike the GPU simulator's -1.
                if parent >= 0 and parent != bi:
                    self._body_children.setdefault(parent, []).append(bi)

        for name, raw_info in metadata["objects"].items():
            info = dict(raw_info)
            joint_names = list(info.get("joint_names", []))
            joint_ids = [hip_model.joint_name2id(j) for j in joint_names if hip_model.joint_name2id(j) >= 0]
            info["joint_ids"] = joint_ids
            info["qpos_addrs"] = [hip_model.joint_qpos_adr(j) for j in joint_ids]

            if info.get("state_type") == "SiteObjectState":
                site_name = info.get("site_name", name)
                info["site_id"] = hip_model.site_name2id(site_name)

                parent_bid = self._resolve_parent_body_id(info, hip_model)
                info["parent_body_id"] = parent_bid
            else:
                body_id = -1
                for candidate in info.get("body_name_candidates", [name]):
                    body_id = hip_model.body_name2id(candidate)
                    if body_id >= 0:
                        info["resolved_body_name"] = candidate
                        break
                info["body_id"] = body_id
                info["body_subtree"] = self._collect_subtree(body_id)

            self._obj_info[name] = info

    @staticmethod
    def _resolve_parent_body_id(info: dict, hip_model) -> int:
        """Resolve the parent body ID from metadata, trying multiple name strategies."""
        for key in ("parent_body_name", "parent_object_name"):
            pname = info.get(key)
            if not pname:
                continue
            bid = hip_model.body_name2id(pname)
            if bid >= 0:
                return bid
            bid = hip_model.body_name2id(pname + "_main")
            if bid >= 0:
                return bid
        return -1

    def _collect_subtree(self, root_bid: int) -> List[int]:
        """Return all body IDs in the subtree rooted at root_bid (BFS)."""
        if root_bid < 0:
            return []
        result = [root_bid]
        queue = [root_bid]
        while queue:
            parent = queue.pop(0)
            for child in self._body_children.get(parent, []):
                result.append(child)
                queue.append(child)
        return result

    def check(self, hip_engine, env_idx: int = 0) -> bool:
        xpos = np.array(hip_engine.data.read_xpos())[env_idx]
        xquat = np.array(hip_engine.data.read_xquat())[env_idx]
        qpos = np.array(hip_engine.data.read_qpos())[env_idx]
        site_xpos = np.array(hip_engine.data.read_site_xpos())[env_idx] if self._hip_model.nsite > 0 else None
        site_xmat = np.array(hip_engine.data.read_site_xmat())[env_idx] if self._hip_model.nsite > 0 else None

        contacts = hip_engine.data.read_contacts() if hasattr(hip_engine.data, "read_contacts") else None
        contact_pairs = set()
        if contacts:
            ncon = contacts.get("ncon", [0])
            nc = int(np.array(ncon).ravel()[env_idx]) if len(np.array(ncon).ravel()) > env_idx else 0
            body1 = np.array(contacts.get("body1", []))
            body2 = np.array(contacts.get("body2", []))
            if body1.ndim >= 2 and body2.ndim >= 2:
                for ci in range(min(nc, body1.shape[1], body2.shape[1])):
                    i1 = int(body1[env_idx, ci])
                    i2 = int(body2[env_idx, ci])
                    if i1 >= 0 and i2 >= 0:
                        contact_pairs.add((min(i1, i2), max(i1, i2)))

        for state in self.goal_state:
            if not self._eval_predicate(state, xpos, xquat, qpos, site_xpos, site_xmat, contact_pairs):
                return False
        return True

    def _get_pos(self, name, xpos, site_xpos):
        info = self._obj_info.get(name, {})
        if info.get("state_type") == "SiteObjectState":
            sid = info.get("site_id", -1)
            if sid >= 0 and site_xpos is not None:
                return site_xpos[sid]
        bid = info.get("body_id", -1)
        if bid >= 0:
            return xpos[bid]
        return np.zeros(3)

    def _get_mat(self, name, site_xmat):
        info = self._obj_info.get(name, {})
        if info.get("state_type") == "SiteObjectState":
            sid = info.get("site_id", -1)
            if sid >= 0 and site_xmat is not None:
                return site_xmat[sid].reshape(3, 3)
        return np.eye(3)

    def _get_body_ids(self, name) -> List[int]:
        """Return all body IDs associated with an object (subtree for objects,
        parent subtree for sites)."""
        info = self._obj_info.get(name, {})
        subtree = info.get("body_subtree")
        if subtree:
            return subtree
        bid = info.get("body_id", info.get("parent_body_id", -1))
        if bid >= 0:
            return [bid]
        return []

    def _check_contact(self, name1, name2, contact_pairs) -> bool:
        """Check whether any body in name1's subtree contacts any body in
        name2's subtree.  Matches robosuite's geom-level check_contact()
        semantics at the body-pair granularity the GPU simulator provides."""
        bids1 = self._get_body_ids(name1)
        bids2 = self._get_body_ids(name2)
        if not bids1 or not bids2:
            return False
        for b1 in bids1:
            for b2 in bids2:
                pair = (min(b1, b2), max(b1, b2))
                if pair in contact_pairs:
                    return True
        return False

    def _eval_predicate(self, state, xpos, xquat, qpos, site_xpos, site_xmat, contact_pairs):
        if len(state) == 3:
            pred, obj1, obj2 = state
            lp = pred.lower()
            if lp == "on":
                return self._check_on(obj1, obj2, xpos, site_xpos, site_xmat, contact_pairs)
            if lp == "in":
                return self._check_in(obj1, obj2, xpos, site_xpos, site_xmat, contact_pairs)
        elif len(state) == 2:
            pred, obj = state
            lp = pred.lower()
            if lp == "open":
                return self._check_open(obj, qpos)
            if lp == "close":
                return self._check_close(obj, qpos)
            if lp == "turnon":
                return self._check_turnon(obj, qpos)
            if lp == "turnoff":
                return self._check_turnoff(obj, qpos)
            if lp == "up":
                pos = self._get_pos(obj, xpos, site_xpos)
                return pos[2] >= 1.0
            if lp in ("true", "printjointstate"):
                return True
            if lp == "false":
                return False
        return False

    # ── On ────────────────────────────────────────────────────────────────────

    def _check_on(self, obj1, obj2, xpos, site_xpos, site_xmat, contact_pairs):
        """On(obj1, obj2) → obj2.check_ontop(obj1).

        Two paths mirror LIBERO's ObjectState vs SiteObjectState:
          - SiteObjectState: under() slab geometry ± parent body contact
          - ObjectState: support below, xy < 0.03, body contact
        """
        info2 = self._obj_info.get(obj2, {})

        if info2.get("state_type") == "SiteObjectState":
            return self._check_on_site(obj1, obj2, info2, xpos, site_xpos, site_xmat, contact_pairs)

        pos1 = self._get_pos(obj1, xpos, site_xpos)
        pos2 = self._get_pos(obj2, xpos, site_xpos)
        if pos2[2] > pos1[2]:
            return False
        if np.linalg.norm(pos1[:2] - pos2[:2]) >= 0.03:
            return False
        return self._check_contact(obj1, obj2, contact_pairs)

    def _check_on_site(self, obj1, site_name, site_info, xpos, site_xpos, site_xmat, contact_pairs):
        """SiteObjectState.check_ontop(other): under() ± parent contact."""
        site_pos = self._get_pos(site_name, xpos, site_xpos)
        site_mat = self._get_mat(site_name, site_xmat)
        obj_pos = self._get_pos(obj1, xpos, site_xpos)
        size = np.array(site_info.get("site_size", [0.05, 0.05, 0.05]), dtype=np.float64)

        if not self._under(site_pos, site_mat, obj_pos, size):
            return False

        # LIBERO gates contact on the site's semantic parent object, not the
        # MuJoCo body carrying the site. Arena TargetZones have parent_name=None
        # even though their sites belong to the table body: under() alone is
        # sufficient for those regions. A fixture's named parent covers the
        # whole object, including sibling bodies of the site-bearing body.
        # Older metadata omits parent_name; retain its physical-body fallback.
        if "parent_name" in site_info:
            parent_name = site_info["parent_name"]
            if parent_name is None:
                return True
            if parent_name in self._obj_info:
                return self._check_contact(obj1, parent_name, contact_pairs)

        parent_bid = site_info.get("parent_body_id", -1)
        if parent_bid < 0:
            return True

        obj1_bids = self._get_body_ids(obj1)
        if not obj1_bids:
            return False
        for b1 in obj1_bids:
            pair = (min(parent_bid, b1), max(parent_bid, b1))
            if pair in contact_pairs:
                return True
        parent_subtree = self._collect_subtree(parent_bid)
        for pb in parent_subtree:
            for b1 in obj1_bids:
                pair = (min(pb, b1), max(pb, b1))
                if pair in contact_pairs:
                    return True
        return False

    @staticmethod
    def _under(site_pos, site_mat, obj_pos, size, other_height=0.10):
        """Matches LIBERO SiteObject.under(): check if obj is in a slab above
        the site surface, expressed in the site's local frame."""
        delta = site_mat @ (obj_pos - site_pos)
        z_ok = (size[2] - 0.005) < delta[2] < (size[2] + other_height)
        xy_ok = np.all(np.abs(delta[:2]) < size[:2])
        return z_ok and xy_ok

    # ── In ────────────────────────────────────────────────────────────────────

    def _check_in(self, obj1, obj2, xpos, site_xpos, site_xmat, contact_pairs):
        """In(obj1, obj2) → obj2.check_contact(obj1) and obj2.check_contain(obj1).

        For SiteObjectState obj2, contact is vacuously True; for ObjectState
        obj2, actual body contact is required.
        """
        info2 = self._obj_info.get(obj2, {})
        is_site = info2.get("state_type") == "SiteObjectState"

        if not is_site and not self._check_contact(obj1, obj2, contact_pairs):
            return False

        pos1 = self._get_pos(obj1, xpos, site_xpos)
        pos2 = self._get_pos(obj2, xpos, site_xpos)
        mat2 = self._get_mat(obj2, site_xmat)
        size = np.array(info2.get("site_size", [0.05, 0.05, 0.05]), dtype=np.float64)
        total_size = np.abs(mat2 @ size)
        ub = pos2 + total_size
        lb = pos2 - total_size
        lb[2] -= 0.01
        return bool(np.all(pos1 > lb) and np.all(pos1 < ub))

    # ── Articulation ──────────────────────────────────────────────────────────

    def _check_articulation(self, obj, qpos, pred_key):
        info = self._obj_info.get(obj, {})
        art_type = info.get("articulation_type")
        addrs = info.get("qpos_addrs", [])
        if not art_type or not addrs:
            return pred_key in ("close", "turnoff")

        thresh = _ARTICULATED_THRESHOLDS.get(art_type, {}).get(pred_key)
        if thresh is None:
            return pred_key in ("close", "turnoff")

        op, val = thresh
        any_mode = pred_key in ("open", "turnon")
        for addr in addrs:
            if addr < 0 or addr >= len(qpos):
                continue
            q = float(qpos[addr])
            passes = (
                (op == "gt" and q > val)
                or (op == "lt" and q < val)
                or (op == "ge" and q >= val)
                or (op == "le" and q <= val)
            )
            if any_mode and passes:
                return True
            if not any_mode and not passes:
                return False
        return not any_mode

    def _check_open(self, obj, qpos):
        return self._check_articulation(obj, qpos, "open")

    def _check_close(self, obj, qpos):
        return self._check_articulation(obj, qpos, "close")

    def _check_turnon(self, obj, qpos):
        return self._check_articulation(obj, qpos, "turnon")

    def _check_turnoff(self, obj, qpos):
        return self._check_articulation(obj, qpos, "turnoff")


class InstinctSimEnv:
    """
    GPU-accelerated environment with LIBERO-compatible observation API.

    Provides: reset(), step(), set_init_state(), close(), seed(), check_success()
    Observations match LIBERO format:
        - agentview_image: (H, W, 3) uint8
        - robot0_eye_in_hand_image: (H, W, 3) uint8
        - robot0_eef_pos: (3,) float
        - robot0_eef_quat: (4,) float (x,y,z,w)
        - robot0_gripper_qpos: (2,) float
    """

    def __init__(
        self,
        xml_string: str,
        camera_height: int = 256,
        camera_width: int = 256,
        gpu_id: int = 0,
        num_envs: int = 1,
        env_idx: int = 0,
        control_freq: int = 20,
        horizon: int = 1000,
        camera_names: Optional[List[str]] = None,
        arm_joint_names: Optional[List[str]] = None,
        gripper_joint_names: Optional[List[str]] = None,
        eef_body_name: Optional[str] = None,
        arm_actuator_indices: Optional[List[int]] = None,
        gripper_actuator_indices: Optional[List[int]] = None,
        body_invweight0: Optional[np.ndarray] = None,
        dof_invweight0: Optional[np.ndarray] = None,
        success_metadata: Optional[Dict[str, Any]] = None,
        meshdir: str = "",
        texture_info: Optional[Dict[str, Any]] = None,
        tex_data: Optional[np.ndarray] = None,
        geom_texids: Optional[np.ndarray] = None,
        geom_texrepeat: Optional[np.ndarray] = None,
        mesh_texcoords: Optional[np.ndarray] = None,
        renderer: Optional[str] = None,
        rt_aa: Optional[str] = None,
        rt_sampler: Optional[str] = None,
        mj_model: Any = None,
        tex_iscube: Optional[List[int]] = None,
    ):
        """renderer / rt_aa / rt_sampler: the renderer switch (instinct_sim_env/rt_renderer.py; None ->
        HP_RENDERER / HP_RT_AA / HP_RT_SAMPLER, default rt; HP_RENDERER=raster opts out). mj_model:
        the env's MuJoCo model (preferably the live one), used by the RT and for tex_iscube. Without
        it the default RT raises (pass mj_model or set HP_RENDERER=raster); an explicit rt compiles
        xml_string (rt_renderer.mujoco_model_for).
        tex_iscube: per-texture cube flags (resolve_tex_iscube; None derives them)."""
        self.xml_string = xml_string
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.gpu_id = gpu_id
        self.num_envs = num_envs
        self.env_idx = env_idx
        self.control_freq = control_freq
        self.horizon = horizon
        self._seed = 0
        self._step_count = 0
        self._closed = False

        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]
        self.camera_names = camera_names

        self.model, self.engine = instinct_sim.create_engine(
            xml_string, num_envs=num_envs, gpu_id=gpu_id,
            body_invweight0=body_invweight0, dof_invweight0=dof_invweight0,
            meshdir=meshdir,
        )

        if texture_info and tex_data is not None and geom_texids is not None:
            tex_widths = [t["width"] for t in texture_info.get("textures", [])]
            tex_heights = [t["height"] for t in texture_info.get("textures", [])]
            _ic = resolve_tex_iscube(tex_iscube, texture_info, len(tex_widths), xml_string, mj_model)
            self.model.set_textures(tex_data, tex_widths, tex_heights,
                                    geom_texids.tolist(), geom_texrepeat,
                                    **({} if _ic is None else {"tex_iscube": _ic}))

        if mesh_texcoords is not None:
            self.model.set_mesh_texcoords(mesh_texcoords)

        self._camera_map = {}
        for cname in self.camera_names:
            cid = self.model.camera_name2id(cname)
            if cid >= 0:
                self._camera_map[cname] = cid

        self.renderer_cfg = renderer_settings(renderer, rt_aa, rt_sampler)
        if self.renderer_cfg["renderer"] == "rt":
            self._renderer = make_rt_renderer(self.model, camera_width, camera_height,
                                              mujoco_model_for(xml_string, mj_model, self.renderer_cfg),
                                              self.renderer_cfg)
        else:
            self._renderer = instinct_sim.CameraRenderer()
            self._renderer.init(self.model, camera_width, camera_height)

        self.set_bg_color = self._renderer.set_bg_color
        self.set_fog = self._renderer.set_fog

        self._setup_robot(
            arm_joint_names, gripper_joint_names, eef_body_name,
            arm_actuator_indices, gripper_actuator_indices,
        )

        desc = self.model
        physics_dt = desc.timestep
        self._steps_per_action = max(1, int(round(1.0 / (control_freq * physics_dt))))

        self.osc = None
        if self._arm_joint_indices is not None:
            self.osc = OSCController(
                model=self.model,
                engine=self.engine,
                arm_joint_indices=self._arm_joint_indices,
                gripper_actuator_indices=self._gripper_actuator_indices or [],
                eef_body_idx=self._eef_body_idx or 0,
                eef_site_idx=self._eef_site_idx,
                arm_actuator_indices=self._arm_actuator_indices,
                control_freq=control_freq,
            )
        self._success_checker = NativeSuccessChecker(success_metadata, self.model) if success_metadata else None

    def _setup_robot(self, arm_joint_names, gripper_joint_names, eef_body_name,
                     arm_actuator_indices, gripper_actuator_indices):
        """Resolve robot joint/body names to indices."""
        self._arm_joint_indices = None
        self._gripper_joint_indices = None
        self._eef_body_idx = None
        self._eef_site_idx = None
        self._arm_actuator_indices = arm_actuator_indices
        self._gripper_actuator_indices = gripper_actuator_indices

        if arm_joint_names is None:
            panda_names = [f"robot0_joint{i}" for i in range(1, 8)]
            found = [self.model.joint_name2id(n) for n in panda_names]
            if all(jid >= 0 for jid in found):
                arm_joint_names = panda_names
            else:
                panda_names2 = [f"panda_joint{i}" for i in range(1, 8)]
                found2 = [self.model.joint_name2id(n) for n in panda_names2]
                if all(jid >= 0 for jid in found2):
                    arm_joint_names = panda_names2

        if arm_joint_names is not None:
            self._arm_joint_indices = [self.model.joint_name2id(n) for n in arm_joint_names]
            if any(jid < 0 for jid in self._arm_joint_indices):
                missing = [n for n, jid in zip(arm_joint_names, self._arm_joint_indices) if jid < 0]
                raise ValueError(f"Arm joints not found: {missing}")

        if gripper_joint_names is None:
            for prefix in ["robot0_finger_joint", "gripper0_finger_joint", "panda_finger_joint"]:
                names = [f"{prefix}1", f"{prefix}2"]
                found = [self.model.joint_name2id(n) for n in names]
                if all(jid >= 0 for jid in found):
                    gripper_joint_names = names
                    break

        if gripper_joint_names is not None:
            self._gripper_joint_indices = [self.model.joint_name2id(n) for n in gripper_joint_names]

        if eef_body_name is None:
            for candidate in ["robot0_right_hand", "right_hand", "panda_hand"]:
                bid = self.model.body_name2id(candidate)
                if bid >= 0:
                    self._eef_body_idx = bid
                    break
        else:
            self._eef_body_idx = self.model.body_name2id(eef_body_name)

        for candidate in ["gripper0_grip_site", "grip_site", "robot0_grip_site"]:
            sid = self.model.site_name2id(candidate)
            if sid >= 0:
                self._eef_site_idx = sid
                break

    @classmethod
    def from_mujoco_xml(cls, xml_path: str, **kwargs) -> 'InstinctSimEnv':
        """Create from an MJCF XML file path."""
        with open(xml_path, 'r') as f:
            xml_string = f.read()
        return cls(xml_string=xml_string, **kwargs)

    @classmethod
    def from_mujoco_model(cls, mj_model, **kwargs) -> 'InstinctSimEnv':
        """Create from a compiled MuJoCo model by extracting its XML."""
        try:
            import mujoco
            xml_string = mujoco.mj_saveLastXML(mj_model)
        except Exception:
            raise RuntimeError("Cannot extract XML from MuJoCo model. "
                               "Pass xml_string or xml_path directly.")
        kwargs.setdefault("mj_model", mj_model)   # the ray tracer (default renderer) is configured from it
        return cls(xml_string=xml_string, **kwargs)

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self._seed = seed
        np.random.seed(seed)

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observation."""
        self.engine.reset_env(self.env_idx)
        self.engine.sync()
        self._step_count = 0
        if self.osc is not None:
            self.osc.reset(self.env_idx)
        return self._get_obs()

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Step environment with a 7D action [dx, dy, dz, dax, day, daz, gripper].

        If no OSC controller is configured, action is written directly as ctrl.
        """
        action = np.asarray(action, dtype=np.float32)

        if self.osc is not None:
            self.osc.control(action, env_idx=self.env_idx)
        else:
            if action.shape[0] == self.model.nu:
                self.engine.data.write_ctrl_env(action, self.env_idx)
            else:
                ctrl = np.zeros(self.model.nu, dtype=np.float32)
                ctrl[:min(len(action), self.model.nu)] = action[:self.model.nu]
                self.engine.data.write_ctrl_env(ctrl, self.env_idx)

        for _ in range(self._steps_per_action):
            self.engine.step()
        self.engine.site_transforms()
        self.engine.sync()

        self._step_count += 1
        obs = self._get_obs()
        reward = 0.0
        done = self._step_count >= self.horizon
        info = {"step": self._step_count}

        return obs, reward, done, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Build observation dict matching LIBERO format."""
        obs = {}

        for cname in self.camera_names:
            if cname in self._camera_map:
                img = self._renderer.render(
                    self.engine, camera_idx=self._camera_map[cname], env_idx=self.env_idx
                )
                key = f"{cname}_image"
                obs[key] = img

        if self._eef_site_idx is not None:
            site_xpos = self.engine.data.read_site_xpos()
            obs["robot0_eef_pos"] = site_xpos[self.env_idx, self._eef_site_idx].copy()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[self.env_idx, self._eef_body_idx].copy())
        elif self._eef_body_idx is not None:
            xpos = self.engine.data.read_xpos()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_pos"] = xpos[self.env_idx, self._eef_body_idx].copy()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[self.env_idx, self._eef_body_idx].copy())

        if self._gripper_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[self.env_idx]
            gripper_qpos = np.array([
                qpos[self.model.joint_qpos_adr(ji)]
                for ji in self._gripper_joint_indices
            ], dtype=np.float32)
            obs["robot0_gripper_qpos"] = gripper_qpos

        if self._arm_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[self.env_idx]
            obs["robot0_joint_pos"] = qpos.copy()

        return obs

    def _get_state_obs(self) -> Dict[str, np.ndarray]:
        """Return robot state observations only (no image rendering)."""
        obs = {}
        if self._eef_site_idx is not None:
            site_xpos = self.engine.data.read_site_xpos()
            obs["robot0_eef_pos"] = site_xpos[self.env_idx, self._eef_site_idx].copy()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[self.env_idx, self._eef_body_idx].copy())
        elif self._eef_body_idx is not None:
            xpos = self.engine.data.read_xpos()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_pos"] = xpos[self.env_idx, self._eef_body_idx].copy()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[self.env_idx, self._eef_body_idx].copy())

        if self._gripper_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[self.env_idx]
            gripper_qpos = np.array([
                qpos[self.model.joint_qpos_adr(ji)]
                for ji in self._gripper_joint_indices
            ], dtype=np.float32)
            obs["robot0_gripper_qpos"] = gripper_qpos

        if self._arm_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[self.env_idx]
            obs["robot0_joint_pos"] = qpos.copy()
        return obs

    def get_sim_state(self) -> np.ndarray:
        """Return flattened simulation state (qpos + qvel)."""
        qpos = self.engine.data.read_qpos()[self.env_idx]
        qvel = self.engine.data.read_qvel()[self.env_idx]
        return np.concatenate([qpos, qvel])

    def set_init_state(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Set simulation state and return observation.

        Writes qpos/qvel then runs one physics step to update forward
        kinematics (xpos, xquat). This matches LIBERO behavior.

        Accepts both MuJoCo flattened state [time, qpos, qvel, ...] and
        bare [qpos, qvel] formats.
        """
        nq = self.model.nq
        nv = self.model.nv
        off = 1 if len(state) >= 1 + nq + nv else 0
        qpos_f64 = state[off:off + nq].astype(np.float64)
        qvel_f64 = state[off + nq:off + nq + nv].astype(np.float64) if len(state) > off + nq else np.zeros(nv, dtype=np.float64)

        _has_d = hasattr(self.engine.data, 'write_qpos_d_env')
        def _write_state():
            if _has_d:
                self.engine.data.write_qpos_d_env(qpos_f64, self.env_idx)
                self.engine.data.write_qvel_d_env(qvel_f64, self.env_idx)
            else:
                self.engine.data.write_qpos_env(qpos_f64.astype(np.float32), self.env_idx)
                self.engine.data.write_qvel_env(qvel_f64.astype(np.float32), self.env_idx)

        _write_state()
        if hasattr(self.engine, 'reset_warmstart_env'):
            self.engine.reset_warmstart_env(self.env_idx)
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        if self.model.nu > 0:
            self.engine.data.write_ctrl_env(ctrl, self.env_idx)
        self.engine.step()
        _write_state()
        self.engine.forward()
        self.engine.sync()

        self._step_count = 0
        if self.osc is not None:
            self.osc.reset(self.env_idx)

        return self._get_obs()

    def check_success(self) -> bool:
        if self._success_checker is None:
            return False
        return self._success_checker.check(self.engine, self.env_idx)

    def render(self, camera_name: str = "agentview") -> np.ndarray:
        """Render a single camera view."""
        cid = self._camera_map.get(camera_name)
        if cid is None:
            cid = self.model.camera_name2id(camera_name)
            if cid < 0:
                raise ValueError(f"Camera '{camera_name}' not found")
        return self._renderer.render(self.engine, camera_idx=cid, env_idx=self.env_idx)

    def render_gpu(self, camera_name: str = "agentview",
                   flip_v: bool = False, flip_h: bool = False):
        """Render a camera view as a GPU-resident object with __cuda_array_interface__.

        Returns a GpuImage that PyTorch can wrap zero-copy via torch.as_tensor().
        The viewport transform already maps NDC to screen-space with row 0 at top,
        matching standard image convention. No flipping needed by default.
        The returned object is only valid until the next render_gpu() call.
        """
        cid = self._camera_map.get(camera_name)
        if cid is None:
            cid = self.model.camera_name2id(camera_name)
            if cid < 0:
                raise ValueError(f"Camera '{camera_name}' not found")
        return self._renderer.render_gpu(
            self.engine, camera_idx=cid, env_idx=self.env_idx,
            flip_v=flip_v, flip_h=flip_h
        )

    def close(self):
        """Release resources."""
        self._closed = True

    @property
    def data(self):
        """Access underlying physics data."""
        return self.engine.data


class BatchedInstinctSimEnv:
    """
    Batched GPU environment: N parallel environments sharing one physics engine.

    All environments use the same MJCF model but can have different states.
    Physics is stepped for all envs in a single GPU call; rendering and obs
    are gathered per-env but with amortised D2H copies.
    """

    def __init__(
        self,
        xml_string: str,
        num_envs: int,
        camera_height: int = 256,
        camera_width: int = 256,
        gpu_id: int = 0,
        control_freq: int = 20,
        horizon: int = 1000,
        camera_names: Optional[List[str]] = None,
        arm_joint_names: Optional[List[str]] = None,
        gripper_joint_names: Optional[List[str]] = None,
        eef_body_name: Optional[str] = None,
        arm_actuator_indices: Optional[List[int]] = None,
        gripper_actuator_indices: Optional[List[int]] = None,
        body_invweight0: Optional[np.ndarray] = None,
        dof_invweight0: Optional[np.ndarray] = None,
        arm_ctrl_limits: Optional[np.ndarray] = None,
        success_metadata: Optional[Dict[str, Any]] = None,
        meshdir: str = "",
        texture_info: Optional[Dict[str, Any]] = None,
        tex_data: Optional[np.ndarray] = None,
        geom_texids: Optional[np.ndarray] = None,
        geom_texrepeat: Optional[np.ndarray] = None,
        mesh_texcoords: Optional[np.ndarray] = None,
        robot_init_qpos: Optional[np.ndarray] = None,
        renderer: Optional[str] = None,
        rt_aa: Optional[str] = None,
        rt_sampler: Optional[str] = None,
        mj_model: Any = None,
        tex_iscube: Optional[List[int]] = None,
    ):
        """tex_iscube: per-texture cube flags for set_textures (resolve_tex_iscube: None derives them
        from texture_info / mj_model / xml_string; HP_TEX_ISCUBE=0 = legacy, no flag).
        renderer / rt_aa / rt_sampler: the renderer switch (instinct_sim_env/rt_renderer.py; None ->
        HP_RENDERER / HP_RT_AA / HP_RT_SAMPLER, default rt; 'raster' opts out to the
        CameraRenderer setup below, unchanged). With 'rt' the env owns ONE RtCameraRenderer
        (self._renderer) and the batched obs path renders every camera in one launch. mj_model: the
        env's MuJoCo model, preferably the LIVE one (robosuite env.sim.model._model), used only by
        the RT. Without it the DEFAULT rt raises (pass mj_model or set HP_RENDERER=raster); an
        explicit rt compiles xml_string (rt_renderer.mujoco_model_for)."""
        self.num_envs = num_envs
        self.xml_string = xml_string
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.gpu_id = gpu_id
        self.control_freq = control_freq
        self.horizon = horizon
        self._step_counts = np.zeros(num_envs, dtype=int)
        self.obs_lag_steps = 0
        # Lag of the eef pose keys (None: obs_lag_steps + 1, see _apply_lagged_proprio).
        _kin = os.environ.get("HP_OBS_KIN_LAG")
        self.obs_kin_lag_steps = int(_kin) if _kin else None
        self._seed = 0

        if camera_names is None:
            camera_names = ["agentview", "robot0_eye_in_hand"]
        self.camera_names = camera_names

        self.model, self.engine = instinct_sim.create_engine(
            xml_string, num_envs=num_envs, gpu_id=gpu_id,
            body_invweight0=body_invweight0, dof_invweight0=dof_invweight0,
            meshdir=meshdir,
        )

        if texture_info and tex_data is not None and geom_texids is not None:
            tex_widths = [t["width"] for t in texture_info.get("textures", [])]
            tex_heights = [t["height"] for t in texture_info.get("textures", [])]
            n_textured = int((geom_texids >= 0).sum()) if hasattr(geom_texids, '__ge__') else 0
            print(f"[TEX_DIAG] set_textures: ntex={len(tex_widths)}, "
                  f"tex_data={len(tex_data)} bytes, "
                  f"geom_texids={len(geom_texids)} ({n_textured} textured), "
                  f"geom_texrepeat={'present' if geom_texrepeat is not None else 'None'}",
                  flush=True)
            _ic = resolve_tex_iscube(tex_iscube, texture_info, len(tex_widths), xml_string, mj_model)
            print(f"[TEX_DIAG] tex_iscube: "
                  f"{'not passed (legacy, all 2D)' if _ic is None else [i for i, c in enumerate(_ic) if c]}",
                  flush=True)
            self.model.set_textures(tex_data, tex_widths, tex_heights,
                                    geom_texids.tolist(), geom_texrepeat,
                                    **({} if _ic is None else {"tex_iscube": _ic}))
        else:
            print(f"[TEX_DIAG] SKIPPED set_textures: "
                  f"texture_info={'present' if texture_info else 'None'}, "
                  f"tex_data={'present' if tex_data is not None else 'None'}, "
                  f"geom_texids={'present' if geom_texids is not None else 'None'}",
                  flush=True)

        if mesh_texcoords is not None:
            self.model.set_mesh_texcoords(mesh_texcoords)

        self._camera_map = {}
        for cname in camera_names:
            cid = self.model.camera_name2id(cname)
            if cid >= 0:
                self._camera_map[cname] = cid

        self.renderer_cfg = renderer_settings(renderer, rt_aa, rt_sampler)
        self._rt = self.renderer_cfg["renderer"] == "rt"
        if self._rt:
            # ONE RayTracer for every camera; _get_all_obs_batched issues one multi-camera launch.
            self._renderer = make_rt_renderer(self.model, camera_width, camera_height,
                                              mujoco_model_for(xml_string, mj_model, self.renderer_cfg),
                                              self.renderer_cfg)
            self._cam_renderers = {}
            self._all_renderers = [self._renderer]
            self._par_cams = False
            self._cam_pool = None
        else:
            self._init_raster_renderers(num_envs, camera_width, camera_height)

        physics_dt = self.model.timestep
        self._steps_per_action = max(1, int(round(1.0 / (control_freq * physics_dt))))

        self._setup_robot(
            arm_joint_names, gripper_joint_names, eef_body_name,
            arm_actuator_indices, gripper_actuator_indices,
        )

        self.osc = None
        if self._arm_joint_indices is not None:
            self.osc = OSCController(
                model=self.model,
                engine=self.engine,
                arm_joint_indices=self._arm_joint_indices,
                gripper_actuator_indices=self._gripper_actuator_indices or [],
                eef_body_idx=self._eef_body_idx or 0,
                eef_site_idx=self._eef_site_idx,
                arm_actuator_indices=self._arm_actuator_indices,
                control_freq=control_freq,
                arm_ctrl_limits=arm_ctrl_limits,
            )
        self._success_checker = NativeSuccessChecker(success_metadata, self.model) if success_metadata else None
        self._per_env_goal_pos = [None] * num_envs
        self._per_env_goal_ori = [None] * num_envs
        self._per_env_init_jpos = [None] * num_envs
        self._gripper_current_action = np.zeros((num_envs, 2), dtype=np.float32)
        self._gripper_speed = 0.01
        self._gripper_ctrlrange = self._parse_gripper_ctrlrange(xml_string)
        # The OSC is constructed ABOVE this line, so hand it the parsed gripper
        # ctrlrange now. Without this the controller writes the raw normalized
        # action (-1..+1) to POSITION actuators whose real range is [0,0.04] /
        # [-0.04,0] -- 25x out of range and both fingers driven the same sign.
        if self.osc is not None and self._gripper_ctrlrange is not None:
            self.osc.gripper_ctrlrange = self._gripper_ctrlrange
        if self.osc is not None and arm_ctrl_limits is None:
            try:
                import xml.etree.ElementTree as _ET
                _root = _ET.fromstring(xml_string)
                _acts = _root.findall(".//actuator/")
                _lims = []
                for _ai in self.osc.arm_actuator_indices:
                    _cr = _acts[_ai].get("ctrlrange") if _ai < len(_acts) else None
                    if _cr:
                        _lo, _hi = [float(x) for x in _cr.split()]
                        _lims.append(max(abs(_lo), abs(_hi)))
                    else:
                        _lims.append(80.0)
                if _lims:
                    self.osc.arm_ctrl_limits = np.asarray(_lims, dtype=np.float64)
                    print(f"[OSC] arm ctrl limits from MJCF: {_lims}", flush=True)
            except Exception as _e:
                print(f"[OSC] arm ctrl limit parse failed ({_e}); keeping defaults", flush=True)
        self._robot_init_qpos = np.asarray(robot_init_qpos, dtype=np.float64) if robot_init_qpos is not None else None
        self._cached_reset_M = None
        self._cached_reset_N = None
        self._freeze_object_dofs = False
        self._object_freeze_mask = None
        self._object_velocity_damping = 0.0
        self._init_object_qpos = None
        self._init_object_qvel = None
        self._warmup_steps_remaining = [0] * num_envs
        self._mj_sim = None
        self._mj_template_state = None
        self._mj_dyn = None
        self._use_mj_stepping = False
        self._use_gpu_osc = False
        self._gpu_osc_configured = False
        self._init_mj_dynamics(xml_string)
        self._bias_dof_indices = None

    def _init_raster_renderers(self, num_envs, camera_width, camera_height):
        """HP_RENDERER=raster (opt-out; the default is rt): the CameraRenderer setup, as it always was."""
        self._renderer = instinct_sim.CameraRenderer()
        self._renderer.init(self.model, camera_width, camera_height)
        self._renderer.set_num_streams(min(num_envs, 16))
        self._cam_renderers = {}
        self._all_renderers = [self._renderer]
        _first = True
        _percam = num_envs <= int(os.environ.get("HP_PERCAM_MAX_ENVS", "2"))
        for _cname in (self._camera_map if _percam else []):
            if _first:
                self._cam_renderers[_cname] = self._renderer
                _first = False
                continue
            _r = instinct_sim.CameraRenderer()
            _r.init(self.model, camera_width, camera_height)
            _r.set_num_streams(min(num_envs, 16))
            self._cam_renderers[_cname] = _r
            self._all_renderers.append(_r)
        # Two cameras can only overlap when each owns its own renderer (and so
        # its own private render stream); otherwise keep the serial loop.
        self._par_cams = (len(self._all_renderers) > 1
                          and os.environ.get("HP_PAR_CAMS", "1") != "0")
        self._cam_pool = None

    def enable_object_freeze(self, enable: bool = True,
                              gripper_unfreeze: bool = False,
                              gripper_threshold: float = 0.005,
                              velocity_damping: float = 0.0):
        """Freeze non-robot free-joint DOFs to prevent object contact divergence.

        The GPU simulator uses bounding-box collision while MuJoCo uses mesh collision,
        causing objects (bowls, plates etc.) to experience very different contact
        forces.  This manifests as 13+ m/s velocity spikes at step 0, which
        pollutes the constraint solver and corrupts the arm trajectory.

        When enabled, after each engine.step() the free-joint (type=0) DOFs for
        non-robot bodies are reset to their initial MuJoCo values.  The arm and
        gripper DOFs are untouched.

        If gripper_unfreeze=True, objects are unfrozen when the gripper aperture
        drops below gripper_threshold (indicating a grasp attempt).  After
        unfreezing, velocity_damping is applied: object velocities are multiplied
        by this factor each substep (0 = no damping, 0.1 = 90% reduction).
        """
        self._freeze_object_dofs = enable
        self._gripper_unfreeze = gripper_unfreeze
        self._gripper_unfreeze_threshold = gripper_threshold
        self._object_velocity_damping = velocity_damping
        self._objects_unfrozen = False
        self._objects_unfrozen_per_env = np.zeros(self.num_envs, dtype=bool)
        if not enable:
            self._object_freeze_mask = None
            return

        robot_joints = set()
        if self._arm_joint_indices:
            robot_joints.update(self._arm_joint_indices)
        if self._gripper_joint_indices:
            robot_joints.update(self._gripper_joint_indices)

        nq = self.model.nq
        nv = self.model.nv
        qpos_mask = np.zeros(nq, dtype=bool)
        qvel_mask = np.zeros(nv, dtype=bool)

        per_object_qpos_slices = []
        per_object_qvel_slices = []
        for ji in range(self.model.njnt):
            if ji in robot_joints:
                continue
            jtype = self.model.joint_type(ji)
            if jtype != 0:  # only freeze free joints (objects)
                continue
            qa = self.model.joint_qpos_adr(ji)
            da = self.model.joint_dof_adr(ji)
            qpos_mask[qa:qa + 7] = True   # free joint: 3 pos + 4 quat
            qvel_mask[da:da + 6] = True   # free joint: 6 dof
            per_object_qpos_slices.append((qa, qa + 3))  # xyz position only
            per_object_qvel_slices.append((da, da + 6))

        self._per_object_qpos_slices = per_object_qpos_slices
        self._per_object_qvel_slices = per_object_qvel_slices
        self._selective_displacement_threshold = 0.015
        self._grasped_object_per_env = np.full(self.num_envs, -1, dtype=int)

        n_frozen_q = int(qpos_mask.sum())
        n_frozen_v = int(qvel_mask.sum())
        n_objects = len(per_object_qpos_slices)
        if n_frozen_q > 0:
            print(f"[FREEZE] Freezing {n_frozen_q} qpos / {n_frozen_v} qvel "
                  f"DOFs from {n_objects} non-robot free joints", flush=True)
        self._object_freeze_mask = (qpos_mask, qvel_mask)

    def _snapshot_object_dofs(self):
        """Capture the current qpos/qvel for frozen object DOFs (per env)."""
        if self._object_freeze_mask is None:
            return
        qm, vm = self._object_freeze_mask
        self.engine.sync()
        if hasattr(self.engine.data, 'read_qpos_d'):
            all_qpos = self.engine.data.read_qpos_d()
            all_qvel = self.engine.data.read_qvel_d()
        else:
            all_qpos = self.engine.data.read_qpos()
            all_qvel = self.engine.data.read_qvel()
        self._init_object_qpos = all_qpos[:, qm].copy()
        self._init_object_qvel = all_qvel[:, vm].copy()

    def _restore_object_dofs(self, env_mask=None):
        """Reset frozen object DOFs to their snapshot values after stepping.

        If env_mask is provided, only restore DOFs for those environments.
        Uses np.ix_ to avoid the numpy fancy-indexing copy pitfall.
        """
        if (self._object_freeze_mask is None or
                self._init_object_qpos is None):
            return
        qm, vm = self._object_freeze_mask
        q_cols = np.where(qm)[0]
        v_cols = np.where(vm)[0]
        if hasattr(self.engine.data, 'read_qpos_d'):
            all_qpos = self.engine.data.read_qpos_d()
            all_qvel = self.engine.data.read_qvel_d()
            if env_mask is not None:
                rows = np.where(env_mask)[0]
                all_qpos[np.ix_(rows, q_cols)] = self._init_object_qpos[rows]
                all_qvel[np.ix_(rows, v_cols)] = self._init_object_qvel[rows]
            else:
                all_qpos[:, qm] = self._init_object_qpos
                all_qvel[:, vm] = self._init_object_qvel
            self.engine.data.write_qpos_d(all_qpos.ravel())
            self.engine.data.write_qvel_d(all_qvel.ravel())
        else:
            all_qpos = self.engine.data.read_qpos()
            all_qvel = self.engine.data.read_qvel()
            if env_mask is not None:
                rows = np.where(env_mask)[0]
                all_qpos[np.ix_(rows, q_cols)] = self._init_object_qpos[rows]
                all_qvel[np.ix_(rows, v_cols)] = self._init_object_qvel[rows]
            else:
                all_qpos[:, qm] = self._init_object_qpos
                all_qvel[:, vm] = self._init_object_qvel
            self.engine.data.write_qpos(all_qpos.ravel())
            self.engine.data.write_qvel(all_qvel.ravel())

    def _damp_object_velocities(self, env_mask=None):
        """Multiply object DOF velocities by the damping factor.

        Applied after gripper-triggered unfreeze to prevent solver-error-driven
        drift while still allowing the gripper to move grasped objects.
        If env_mask is provided, only damp those environments.
        """
        if (self._object_freeze_mask is None or
                self._object_velocity_damping <= 0):
            return
        _, vm = self._object_freeze_mask
        v_cols = np.where(vm)[0]
        d = self._object_velocity_damping
        if hasattr(self.engine.data, 'read_qvel_d'):
            all_qvel = self.engine.data.read_qvel_d()
            if env_mask is not None:
                rows = np.where(env_mask)[0]
                all_qvel[np.ix_(rows, v_cols)] *= d
            else:
                all_qvel[:, vm] *= d
            self.engine.data.write_qvel_d(all_qvel.ravel())
        else:
            all_qvel = self.engine.data.read_qvel()
            if env_mask is not None:
                rows = np.where(env_mask)[0]
                all_qvel[np.ix_(rows, v_cols)] *= d
            else:
                all_qvel[:, vm] *= d
            self.engine.data.write_qvel(all_qvel.ravel())

    def _detect_grasped_object(self, env_idx, qpos):
        """At unfreeze time, find which object is closest to the EEF and cache it."""
        if not self._per_object_qpos_slices:
            return

        eef_pos = None
        if self._eef_site_idx is not None:
            site_xpos = np.array(self.engine.data.read_site_xpos())
            eef_pos = site_xpos[env_idx, self._eef_site_idx]
        elif self._eef_body_idx is not None:
            xpos = np.array(self.engine.data.read_xpos())
            eef_pos = xpos[env_idx, self._eef_body_idx]

        if eef_pos is None:
            return

        min_dist = float('inf')
        best = -1
        for oi, (qa_s, qa_e) in enumerate(self._per_object_qpos_slices):
            obj_pos = qpos[env_idx, qa_s:qa_e]
            d = float(np.linalg.norm(eef_pos - obj_pos))
            if d < min_dist:
                min_dist = d
                best = oi
        self._grasped_object_per_env[env_idx] = best

    def _selective_restore_passive_objects(self, env_mask=None):
        """Restore non-manipulated objects to their snapshot positions.

        Uses the cached grasped-object ID (detected once at unfreeze) to
        exempt the grasped object. All other objects get hard-restored to
        their initial snapshot every substep, preventing drift without any
        per-substep EEF lookups.
        """
        if (self._object_freeze_mask is None or
                self._init_object_qpos is None or
                not hasattr(self, '_per_object_qpos_slices') or
                not self._per_object_qpos_slices):
            return

        qm, vm = self._object_freeze_mask
        q_cols = np.where(qm)[0]
        v_cols = np.where(vm)[0]

        use_double = hasattr(self.engine.data, 'read_qpos_d')
        if use_double:
            all_qpos = self.engine.data.read_qpos_d()
            all_qvel = self.engine.data.read_qvel_d()
        else:
            all_qpos = self.engine.data.read_qpos()
            all_qvel = self.engine.data.read_qvel()

        envs = np.where(env_mask)[0] if env_mask is not None else np.arange(self.num_envs)
        modified = False

        for ei in envs:
            grasped = self._grasped_object_per_env[ei]
            for obj_idx, (qa_start, qa_end) in enumerate(self._per_object_qpos_slices):
                if obj_idx == grasped:
                    continue
                da_start, da_end = self._per_object_qvel_slices[obj_idx]
                qpos_in_mask = np.searchsorted(q_cols, qa_start)
                qvel_in_mask = np.searchsorted(v_cols, da_start)
                v_span = da_end - da_start
                q_obj_cols = q_cols[qpos_in_mask:qpos_in_mask + 7]
                v_obj_cols = v_cols[qvel_in_mask:qvel_in_mask + v_span]

                all_qpos[ei, q_obj_cols] = self._init_object_qpos[
                    ei, qpos_in_mask:qpos_in_mask + 7]
                all_qvel[ei, v_obj_cols] = self._init_object_qvel[
                    ei, qvel_in_mask:qvel_in_mask + v_span]
                modified = True

        if modified:
            if use_double:
                self.engine.data.write_qpos_d(all_qpos.ravel())
                self.engine.data.write_qvel_d(all_qvel.ravel())
            else:
                self.engine.data.write_qpos(all_qpos.ravel())
                self.engine.data.write_qvel(all_qvel.ravel())

    def setup_mujoco_fog(self, extent: float, fogstart: float = 3.0,
                         fogend: float = 10.0,
                         fog_rgb: tuple = (0.0, 0.0, 0.0),
                         haze_rgb: tuple = (1.0, 1.0, 1.0),
                         znear_scale: float = 0.01, zfar_scale: float = 50.0):
        """Configure fog, gradient background, and clip planes to match MuJoCo defaults."""
        top_r = 0.5 * (fog_rgb[0] + haze_rgb[0])
        top_g = 0.5 * (fog_rgb[1] + haze_rgb[1])
        top_b = 0.5 * (fog_rgb[2] + haze_rgb[2])
        for _r in getattr(self, "_all_renderers", [self._renderer]):
            _r.set_fog(
                fogstart * extent, fogend * extent,
                fog_rgb[0], fog_rgb[1], fog_rgb[2]
            )
            _r.set_gradient_bg(
                fog_rgb[0], fog_rgb[1], fog_rgb[2],
                top_r, top_g, top_b
            )
            _r.set_clip_planes(znear_scale * extent, zfar_scale * extent)

    def _init_mj_dynamics(self, xml_string: str):
        """Create a lightweight MuJoCo model for dynamics and optional stepping.

        Provides MuJoCo-exact M, J, qfrc_bias for the native OSC. When
        _use_mj_stepping is True, the same model is also used for physics
        stepping (mj_step) so the native OSC torques are validated against
        MuJoCo's integrator independently of the GPU simulator's solver.
        """
        try:
            import mujoco
            mj_model = mujoco.MjModel.from_xml_string(xml_string)
            mj_data = mujoco.MjData(mj_model)
            self._mj_dyn = {
                'model': mj_model,
                'data': mj_data,
                'nq': mj_model.nq,
                'nv': mj_model.nv,
                'nu': mj_model.nu,
            }
            print(f"[MJ_DYN] Lightweight MuJoCo dynamics loaded "
                  f"(nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu})",
                  flush=True)
        except Exception as e:
            print(f"[MJ_DYN] WARNING: Could not load MuJoCo dynamics: {e}. "
                  f"Falling back to custom M/J.", flush=True)
            self._mj_dyn = None

    def enable_mj_stepping(self, enable: bool = True, mj_xml_string: str = None,
                           mj_model=None):
        """Enable MuJoCo-based physics stepping for the native OSC path.

        When enabled, _substep_native_osc computes torques with the native OSC
        but delegates physics stepping to MuJoCo (mj_step) instead of
        GPU engine.step(). One MuJoCo model per env is created to
        maintain independent state.

        Args:
            mj_xml_string: If provided, load the MuJoCo stepping model from
                this XML. WARNING: XML round-trips lose precision for body
                masses/inertias. Prefer mj_model instead.
            mj_model: Pre-compiled MjModel to use directly (avoids lossy
                XML round-trip). Takes priority over mj_xml_string.
        """
        if enable and self._mj_dyn is None:
            raise RuntimeError("Cannot enable MuJoCo stepping: _mj_dyn not initialised")
        self._use_mj_stepping = enable
        if enable:
            import mujoco
            if mj_model is not None:
                m = mj_model
                self._mj_dyn['model'] = m
                self._mj_dyn['data'] = mujoco.MjData(m)
                print(f"[MJ_STEP] Using pre-compiled MuJoCo model for stepping "
                      f"(nq={m.nq}, nv={m.nv}, nu={m.nu}) — no XML round-trip",
                      flush=True)
            elif mj_xml_string is not None:
                m = mujoco.MjModel.from_xml_string(mj_xml_string)
                self._mj_dyn['model'] = m
                self._mj_dyn['data'] = mujoco.MjData(m)
                print(f"[MJ_STEP] Loaded MuJoCo model from XML for stepping "
                      f"(nq={m.nq}, nv={m.nv}, nu={m.nu})", flush=True)
            else:
                m = self._mj_dyn['model']
            per_env = []
            for _ in range(self.num_envs):
                d = mujoco.MjData(m)
                per_env.append(d)
            self._mj_dyn['per_env_data'] = per_env
            self._mj_step_initialized = [False] * self.num_envs
            print(f"[MJ_STEP] MuJoCo stepping enabled for {self.num_envs} envs "
                  f"(native OSC torques → mj_step → sync back to the GPU simulator)",
                  flush=True)

    def enable_gpu_osc(self, enable: bool = True):
        """Enable GPU-side OSC controller (zero host sync during substeps).

        Requires the same arm/gripper topology as the Python OSC.
        Replaces _substep_native_osc with engine.step_with_osc().
        """
        self._use_gpu_osc = enable
        if enable and self.osc is not None and not self._gpu_osc_configured:
            osc = self.osc
            cfg = {
                'arm_joint_indices':    list(osc.arm_joint_indices),
                'arm_dof_indices':      list(osc.arm_dof_indices),
                'arm_qpos_indices':     list(osc.arm_qpos_indices),
                'arm_actuator_indices': list(osc.arm_actuator_indices),
                'gripper_actuator_indices': list(osc.gripper_actuator_indices) if osc.gripper_actuator_indices else [],
                'eef_site_idx':         osc.eef_site_idx if osc.eef_site_idx is not None else -1,
                'eef_body_idx':         osc.eef_body_idx,
                'kp_pos':  float(osc.kp_pos),
                'kp_ori':  float(osc.kp_ori),
                'kd_pos':  float(osc.kd_pos),
                'kd_ori':  float(osc.kd_ori),
                'kp_null': float(osc.kp_null),
                'kd_null': float(osc.kd_null),
                'action_scale_pos': float(osc.action_scale_pos),
                'action_scale_ori': float(osc.action_scale_ori),
                'arm_ctrl_limits': [float(x) for x in osc.arm_ctrl_limits],
                'gripper_speed': float(self._gripper_speed),
            }
            if self._gripper_ctrlrange is not None:
                flat = []
                for i in range(min(len(self._gripper_actuator_indices or []), 2)):
                    flat.append(float(self._gripper_ctrlrange[i, 0]))
                    flat.append(float(self._gripper_ctrlrange[i, 1]))
                cfg['gripper_ctrlrange'] = flat
            self.engine.set_osc_config(cfg)
            self._gpu_osc_configured = True
            print(f"[GPU_OSC] Configured GPU-side OSC: n_arm={osc.n_arm}, "
                  f"eef_site={osc.eef_site_idx}, gripper={len(osc.gripper_actuator_indices)} acts",
                  flush=True)

    def _substep_gpu_osc(self, actions: np.ndarray, step_rows=None):
        """Run physics with GPU-side OSC -- zero host sync during substeps.

        step_rows (world skip): world ids to step; every other world is frozen for
        this step and the mask is cleared again before returning.
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[1] < 7:
            padded = np.zeros((actions.shape[0], 7), dtype=np.float32)
            padded[:, :actions.shape[1]] = actions
            actions = padded
        self.engine.set_osc_actions(actions[:self.num_envs])
        mask = None
        if step_rows is not None and _HP_PHYS_SKIP:
            mask = np.zeros(self.num_envs, dtype=np.uint8)
            rows = np.asarray(sorted({int(r) for r in step_rows}), dtype=np.int64)
            if rows.size and (rows[0] < 0 or rows[-1] >= self.num_envs):
                raise ValueError(f"step_rows outside 0..{self.num_envs - 1}")
            mask[rows] = 1
            if mask.all():
                mask = None             # every world steps: the unmasked path exactly
        if mask is None:
            self.engine.step_with_osc(self._steps_per_action)
        else:
            self.engine.set_world_active(mask)
            try:
                self.engine.step_with_osc(self._steps_per_action)
            finally:
                self.engine.clear_world_active()
        self.engine.sync()

    def _mj_compute_dynamics(self, qpos: np.ndarray, qvel: np.ndarray):
        """Inject state into lightweight MuJoCo model and compute M, J, bias.

        Returns (mass_matrix_arm, J_pos, J_ori, qfrc_bias_arm) using
        MuJoCo's own mj_fullM, mj_jacSite, and qfrc_bias.
        """
        import mujoco
        mj = self._mj_dyn
        m, d = mj['model'], mj['data']
        osc = self.osc

        nq = min(len(qpos), m.nq)
        nv = min(len(qvel), m.nv)
        d.qpos[:nq] = qpos[:nq].astype(np.float64)
        d.qvel[:nv] = qvel[:nv].astype(np.float64)
        mujoco.mj_forward(m, d)

        full_M = np.zeros((m.nv, m.nv), dtype=np.float64)
        mujoco.mj_fullM(m, full_M, d.qM)
        arm_dofs = osc.arm_dof_indices
        M_arm = full_M[np.ix_(arm_dofs, arm_dofs)]

        eef_site_name = None
        if osc.eef_site_idx is not None:
            eef_site_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, osc.eef_site_idx)

        if eef_site_name is not None:
            jacp = np.zeros((3, m.nv), dtype=np.float64)
            jacr = np.zeros((3, m.nv), dtype=np.float64)
            site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, eef_site_name)
            mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
            J_pos = jacp[:, arm_dofs]
            J_ori = jacr[:, arm_dofs]
        else:
            jacp = np.zeros((3, m.nv), dtype=np.float64)
            jacr = np.zeros((3, m.nv), dtype=np.float64)
            mujoco.mj_jacBody(m, d, jacp, jacr, osc.eef_body_idx)
            J_pos = jacp[:, arm_dofs]
            J_ori = jacr[:, arm_dofs]

        bias_arm = d.qfrc_bias[arm_dofs].copy()

        eef_pos = d.site_xpos[osc.eef_site_idx].copy() if osc.eef_site_idx is not None else d.xpos[osc.eef_body_idx].copy()
        eef_mat = d.site_xmat[osc.eef_site_idx].reshape(3, 3).copy() if osc.eef_site_idx is not None else None

        return M_arm, J_pos, J_ori, bias_arm, eef_pos, eef_mat

    def _setup_robot(self, arm_joint_names, gripper_joint_names, eef_body_name,
                     arm_actuator_indices, gripper_actuator_indices):
        """Resolve robot joint/body names to indices (shared across envs)."""
        self._arm_joint_indices = None
        self._gripper_joint_indices = None
        self._eef_body_idx = None
        self._eef_site_idx = None
        self._arm_actuator_indices = arm_actuator_indices
        self._gripper_actuator_indices = gripper_actuator_indices

        if arm_joint_names is None:
            panda_names = [f"robot0_joint{i}" for i in range(1, 8)]
            found = [self.model.joint_name2id(n) for n in panda_names]
            if all(jid >= 0 for jid in found):
                arm_joint_names = panda_names
            else:
                panda_names2 = [f"panda_joint{i}" for i in range(1, 8)]
                found2 = [self.model.joint_name2id(n) for n in panda_names2]
                if all(jid >= 0 for jid in found2):
                    arm_joint_names = panda_names2

        if arm_joint_names is not None:
            self._arm_joint_indices = [self.model.joint_name2id(n) for n in arm_joint_names]
            if any(jid < 0 for jid in self._arm_joint_indices):
                missing = [n for n, jid in zip(arm_joint_names, self._arm_joint_indices) if jid < 0]
                raise ValueError(f"Arm joints not found: {missing}")

        if gripper_joint_names is None:
            for prefix in ["robot0_finger_joint", "gripper0_finger_joint", "panda_finger_joint"]:
                names = [f"{prefix}1", f"{prefix}2"]
                found = [self.model.joint_name2id(n) for n in names]
                if all(jid >= 0 for jid in found):
                    gripper_joint_names = names
                    break

        if gripper_joint_names is not None:
            self._gripper_joint_indices = [self.model.joint_name2id(n) for n in gripper_joint_names]

        # Auto-detect gripper actuator indices: last 2 actuators for Panda robots
        if self._gripper_actuator_indices is None and gripper_joint_names is not None:
            nu = self.model.nu
            if nu >= 9:
                self._gripper_actuator_indices = [nu - 2, nu - 1]
                print(f"[INFO] Auto-detected gripper actuator indices: {self._gripper_actuator_indices}", flush=True)

        if eef_body_name is None:
            for candidate in ["robot0_right_hand", "right_hand", "panda_hand"]:
                bid = self.model.body_name2id(candidate)
                if bid >= 0:
                    self._eef_body_idx = bid
                    break
        else:
            self._eef_body_idx = self.model.body_name2id(eef_body_name)

        for candidate in ["gripper0_grip_site", "grip_site", "robot0_grip_site"]:
            sid = self.model.site_name2id(candidate)
            if sid >= 0:
                self._eef_site_idx = sid
                break

    def _parse_gripper_ctrlrange(self, xml_string: str) -> Optional[np.ndarray]:
        """Extract ctrlrange for gripper actuators from the MJCF XML.

        Returns (2, 2) array [[low0, high0], [low1, high1]] or None.
        Used to rescale the [-1, 1] gripper accumulator to the actuator
        ctrlrange, matching robosuite's Manipulator.grip_action().
        """
        if not self._gripper_actuator_indices or len(self._gripper_actuator_indices) < 2:
            return None
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_string)
            actuators = root.findall(".//actuator/general")
            if not actuators:
                actuators = root.findall(".//actuator/*")
            ctrlranges = {}
            for act in actuators:
                cr = act.get("ctrlrange")
                if cr:
                    parts = cr.split()
                    if len(parts) == 2:
                        name = act.get("name", "")
                        joint = act.get("joint", "")
                        ctrlranges[name] = (float(parts[0]), float(parts[1]))
                        ctrlranges[joint] = (float(parts[0]), float(parts[1]))

            finger_names = [
                ("gripper0_gripper_finger_joint1", "gripper0_finger_joint1"),
                ("gripper0_gripper_finger_joint2", "gripper0_finger_joint2"),
            ]
            result = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
            found = 0
            for i, (act_name, jnt_name) in enumerate(finger_names):
                cr = ctrlranges.get(act_name) or ctrlranges.get(jnt_name)
                if cr:
                    result[i] = [cr[0], cr[1]]
                    found += 1
            if found == 0:
                return None
            print(f"[INFO] Gripper ctrlrange: finger1=[{result[0,0]:.4f}, {result[0,1]:.4f}], "
                  f"finger2=[{result[1,0]:.4f}, {result[1,1]:.4f}]", flush=True)
            return result
        except Exception:
            return None

    def _apply_gripper_ctrl(self, ctrl_batch, ei, grip_action):
        """Apply gripper control matching robosuite's full pipeline.

        Robosuite's gripper control has two stages:
        1. PandaGripper.format_action: accumulates in [-1, 1] with speed
        2. Manipulator.grip_action: rescales [-1, 1] → actuator ctrlrange

        The rescaling is: ctrl = bias + weight * accumulator
        where bias = (high + low) / 2, weight = (high - low) / 2.
        """
        if not self._gripper_actuator_indices:
            return
        direction = np.sign(grip_action)
        new_val = self._gripper_current_action[ei] + np.array([-1.0, 1.0]) * self._gripper_speed * direction
        self._gripper_current_action[ei] = np.clip(new_val, -1.0, 1.0)

        if self._gripper_ctrlrange is not None:
            for fi, ai in enumerate(self._gripper_actuator_indices):
                lo = self._gripper_ctrlrange[fi, 0]
                hi = self._gripper_ctrlrange[fi, 1]
                bias = 0.5 * (hi + lo)
                weight = 0.5 * (hi - lo)
                ctrl_batch[ei, ai] = bias + weight * self._gripper_current_action[ei, fi]
        else:
            ctrl_batch[ei, self._gripper_actuator_indices[0]] = self._gripper_current_action[ei, 0]
            if len(self._gripper_actuator_indices) > 1:
                ctrl_batch[ei, self._gripper_actuator_indices[1]] = self._gripper_current_action[ei, 1]

    def warmup_gripper(self, num_warmup_steps: int = 10, grip_action: float = -1.0):
        """Simulate warmup gripper accumulator updates without physics.

        Robosuite's gripper accumulator runs during warmup (dummy action)
        steps inside init_episode().  The native OSC skips those warmup
        steps (the checker does them), so the accumulator starts at zero.
        This causes massive closing force on the first real action.

        Call this after syncing the GPU simulator from the checker's post-warmup
        state to align the gripper accumulator with robosuite.
        """
        if not self._gripper_actuator_indices:
            return
        n_substeps = num_warmup_steps * self._steps_per_action
        nu = self.model.nu
        dummy_ctrl = np.zeros((self.num_envs, nu), dtype=np.float32)
        for _ in range(n_substeps):
            for ei in range(self.num_envs):
                self._apply_gripper_ctrl(dummy_ctrl, ei, grip_action)
        for ei in range(self.num_envs):
            print(f"[GRIPPER_WARMUP] env {ei}: accumulator={self._gripper_current_action[ei]}", flush=True)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def seed(self, seed: int):
        self._seed = seed
        np.random.seed(seed)

    def reset_all(self) -> List[Dict[str, np.ndarray]]:
        """Reset all environments."""
        self.engine.reset()
        self.engine.sync()
        self._step_counts[:] = 0
        self._gripper_current_action[:] = 0.0
        for i in range(self.num_envs):
            self._per_env_goal_pos[i] = None
            self._per_env_goal_ori[i] = None
            self._per_env_init_jpos[i] = None
        if hasattr(self, '_mj_step_initialized'):
            for i in range(self.num_envs):
                self._mj_step_initialized[i] = False
        if self._use_gpu_osc and self._gpu_osc_configured:
            self.engine.reset_osc_state()
        if self.osc is not None:
            if self._robot_init_qpos is not None and len(self._robot_init_qpos) >= self.osc.n_arm:
                null_ref = np.array(self._robot_init_qpos[:self.osc.n_arm], dtype=np.float64)
                for i in range(self.num_envs):
                    self._per_env_init_jpos[i] = null_ref.copy()
            else:
                all_qpos = self.engine.data.read_qpos()
                for i in range(self.num_envs):
                    self._per_env_init_jpos[i] = np.array([all_qpos[i, qi] for qi in self.osc.arm_qpos_indices])
        return [self._get_obs(i) for i in range(self.num_envs)]

    def seed_gpu_gripper_from_state(self) -> None:
        """Seed the GPU gripper accumulator from the PHYSICAL finger positions.

        robosuite's accumulator also starts at 0, but by the time an episode
        begins the eval loop has already run 30 dummy steps with act[6]=-1,
        so robosuite has ramped to a SATURATED [+1,-1] -- ctrl [0.04,-0.04],
        fingers physically fully open (qpos [0.0395,-0.0395]). hip builds its
        engine fresh and starts COLD at 0, whose affine map commands
        [0.02,-0.02] = HALF open, against fingers that are fully open. The solver
        fights that mismatch and the ARM feels it.

        Measured on libero_object/0, arm |gpu-cpu| peak vs the 1e-3 rad budget:
            unseeded  8.4084e-03  (8.41x)  -- steps 1,5 were 7x worse
            seeded    3.2135e-03  (3.21x)  -- step 1 = 3.97e-05, BETTER than
                                              robosuite's own 2.59e-04
        Inverts the affine map: acc = clip((qpos - bias)/weight, -1, 1).

        CALL ORDER MATTERS: invoke this AFTER writing the episode's initial qpos.
        Called from inside reset_all() it reads the post-reset default pose and
        makes tracking WORSE than no seeding at all (21.75x vs 8.41x, measured).
        """
        gr = self._gripper_ctrlrange
        idx = self._gripper_actuator_indices
        if gr is None or not idx:
            return
        try:
            qpos = np.asarray(self.engine.data.read_qpos_d())
            qpos = qpos.reshape(self.num_envs, -1) if qpos.size >= self.num_envs else None
            if qpos is None:
                return
            acc = np.zeros((self.num_envs, 2), dtype=np.float64)
            for ei in range(self.num_envs):
                for k in range(min(2, len(idx))):
                    lo, hi = float(gr[k, 0]), float(gr[k, 1])
                    bias, weight = 0.5 * (hi + lo), 0.5 * (hi - lo)
                    adr = self._gripper_qpos_adr[k] if getattr(self, '_gripper_qpos_adr', None) else (7 + k)
                    if adr < qpos.shape[1] and abs(weight) > 1e-12:
                        acc[ei, k] = float(np.clip((qpos[ei, adr] - bias) / weight, -1.0, 1.0))
            self.engine.set_osc_gripper_state(acc)
            self._gripper_current_action[:] = acc[:, :self._gripper_current_action.shape[1]]
        except Exception as _e:
            print(f"[GPU_OSC] gripper seeding skipped ({_e})", flush=True)

    def reset_env(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Reset a single environment."""
        self.engine.reset_env(env_idx)
        self.engine.sync()
        self._step_counts[env_idx] = 0
        self._gripper_current_action[env_idx] = 0.0
        self._per_env_goal_pos[env_idx] = None
        self._per_env_goal_ori[env_idx] = None
        self._per_env_init_jpos[env_idx] = None
        if hasattr(self, '_mj_step_initialized') and env_idx < len(self._mj_step_initialized):
            self._mj_step_initialized[env_idx] = False
        if self._use_gpu_osc and self._gpu_osc_configured:
            self.engine.reset_osc_env_state(env_idx)
        if self.osc is not None:
            if self._robot_init_qpos is not None and len(self._robot_init_qpos) >= self.osc.n_arm:
                self._per_env_init_jpos[env_idx] = np.array(self._robot_init_qpos[:self.osc.n_arm], dtype=np.float64)
            else:
                qpos = self.engine.data.read_qpos()[env_idx]
                self._per_env_init_jpos[env_idx] = np.array([qpos[qi] for qi in self.osc.arm_qpos_indices])
        return self._get_obs(env_idx)

    def set_mujoco_controller(self, ctrl_fn, bias_fn=None, step_fn=None,
                              run_substeps_fn=None):
        """Attach a MuJoCo-based controller function for computing ctrl.

        ctrl_fn(hip_qpos, hip_qvel, action) -> ctrl array of shape (nu,)
        step_fn(hip_qpos, hip_qvel, action) -> (ctrl, arm_qpos, arm_qvel)
            If provided, also steps MuJoCo physics and returns MuJoCo's predicted
            arm state for drift correction.
        run_substeps_fn(hip_qpos, hip_qvel, action, n_substeps) ->
            (ctrl, full_qpos, full_qvel)
            If provided, runs all substeps in MuJoCo with full float64 continuity.
            State is injected ONCE then MuJoCo runs natively for all substeps.
        """
        self._mj_ctrl_fn = ctrl_fn
        self._mj_bias_fn = None
        self._mj_step_fn = step_fn
        self._mj_run_substeps_fn = run_substeps_fn

    # ── Stepping ──────────────────────────────────────────────────────────────

    def step_all(self, actions: np.ndarray, render_rows=None, step_rows=None) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments with batched 7D actions [dx,dy,dz,dax,day,daz,gripper].

        render_rows : optional iterable of world ids whose camera images
        the caller will read from THIS round. Physics and state observations
        are produced for every world as always; images are rendered only for
        these rows and every other row's ``*_image`` entry is None -- a
        placeholder that must never be consumed. None (the default) or a set
        covering every world is the unchanged all-world path.

        step_rows (world skip): optional iterable of world ids to simulate this
        round; every other world is frozen (state unchanged, no contacts). Only
        the GPU-OSC path supports it. None = every world steps.

        Matches robosuite's internal loop: the OSC controller is called at every
        physics sub-step (500Hz), not just once per action (20Hz). This is critical
        because the OSC is a PD controller that requires high-frequency feedback.
        """
        import time as _time
        _t0 = _time.perf_counter()
        actions = np.asarray(actions, dtype=np.float32)

        use_substep_ctrl = (
            hasattr(self, '_mj_ctrl_fn') and self._mj_ctrl_fn is not None
        )

        if use_substep_ctrl:
            self._substep_control_loop(actions)
        elif self._use_gpu_osc and self._gpu_osc_configured:
            self._substep_gpu_osc(actions, step_rows)
        elif step_rows is not None:
            raise ValueError("step_rows (world skip) needs the GPU-OSC path")
        elif self.osc is not None:
            self._substep_native_osc(actions)
        else:
            nu = self.model.nu
            if nu > 0:
                ctrl = np.zeros((self.num_envs, nu), dtype=np.float32)
                for i in range(self.num_envs):
                    a = actions[i] if i < len(actions) else np.zeros(nu, dtype=np.float32)
                    ctrl[i, :min(len(a), nu)] = a[:nu]
                self.engine.data.write_ctrl(ctrl.ravel())
            self.engine.step_n(self._steps_per_action)
            self.engine.site_transforms()
            self.engine.sync()

        _t1 = _time.perf_counter()
        self._step_counts += 1

        img_lag = self._image_lag_steps()
        if img_lag > 0:
            self.engine.begin_lagged_render(img_lag * self.model.timestep)
        try:
            if self.num_envs >= 1 and self.camera_names and hasattr(self, '_camera_map') and self._camera_map:
                obs_list = self._get_all_obs_batched(render_rows)
            else:
                obs_list = [self._get_obs(i) for i in range(self.num_envs)]
        finally:
            if img_lag > 0:
                self.engine.end_lagged_render()
        if self.obs_lag_steps > 0:
            self._apply_lagged_proprio(obs_list)
        _t2 = _time.perf_counter()
        if _HP_OBS_TIMERS:
            _rec = getattr(self, "_obs_tcur", None) or {}
            _rec["phys"] = (_t1 - _t0) * 1e3
            _rec["obs"] = (_t2 - _t1) * 1e3
            self._obs_tcur = None
            if not hasattr(self, "_obs_tlog"):
                self._obs_tlog = []
            self._obs_tlog.append(_rec)
            self._obs_timer_summary()
        rewards = np.zeros(self.num_envs)
        dones = self._step_counts >= self.horizon
        info_list = [{"step": int(self._step_counts[i])} for i in range(self.num_envs)]

        step_ms = (_t1 - _t0) * 1000
        obs_ms = (_t2 - _t1) * 1000
        total_ms = (_t2 - _t0) * 1000
        if not hasattr(self, '_step_timing_count'):
            self._step_timing_count = 0
        self._step_timing_count += 1
        if self._step_timing_count <= 5 or self._step_timing_count % 20 == 0:
            print(f"[TIMING] step_all #{self._step_timing_count}: "
                  f"step={step_ms:.1f}ms, obs={obs_ms:.1f}ms, "
                  f"total={total_ms:.1f}ms", flush=True)

        return obs_list, rewards, dones, info_list

    def _image_lag_steps(self) -> int:
        """Substeps the camera images lag the final state: HP_IMG_LAG, else the pose lag when the
        robosuite-timed proprio is on (obs_lag_steps > 0), else 0 (images of the final state)."""
        env = os.environ.get("HP_IMG_LAG")
        if env is not None and env != "":
            return int(env)
        if self.obs_lag_steps <= 0 or not hasattr(self.engine, "begin_lagged_render"):
            return 0
        return self.obs_kin_lag_steps if self.obs_kin_lag_steps is not None else self.obs_lag_steps + 1

    def _apply_lagged_proprio(self, obs_list: List[Dict[str, np.ndarray]]) -> None:
        """Overwrite the proprio keys with robosuite-timed values (see obs_lag_steps).

        engine.lagged_kinematics steps qpos back along qvel with MuJoCo's mj_integratePos and
        reruns the regular FK + site kernels into separate device buffers; the live state is
        untouched. Only these three keys are lagged, matching robosuite.
        """
        eng = self.engine
        kin_lag = self.obs_kin_lag_steps if self.obs_kin_lag_steps is not None else self.obs_lag_steps + 1
        eng.lagged_kinematics(kin_lag * self.model.timestep)
        xquat = eng.read_lagged_xquat() if self._eef_body_idx is not None else None
        if self._eef_site_idx is not None:
            pos, pidx = eng.read_lagged_site_xpos(), self._eef_site_idx
        elif self._eef_body_idx is not None:
            pos, pidx = eng.read_lagged_xpos(), self._eef_body_idx
        else:
            pos = None
        if self._gripper_joint_indices is not None:
            if kin_lag != self.obs_lag_steps:
                eng.lagged_kinematics(self.obs_lag_steps * self.model.timestep)
            qpos = eng.read_lagged_qpos()
        else:
            qpos = None
        gadr = ([self.model.joint_qpos_adr(ji) for ji in self._gripper_joint_indices]
                if qpos is not None else None)
        for ei, obs in enumerate(obs_list):
            if pos is not None:
                obs["robot0_eef_pos"] = np.array(pos[ei, pidx], dtype=obs["robot0_eef_pos"].dtype
                                                 if "robot0_eef_pos" in obs else np.float32)
            if xquat is not None:
                obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[ei, self._eef_body_idx].copy())
            if gadr is not None:
                obs["robot0_gripper_qpos"] = np.array([qpos[ei, a] for a in gadr], dtype=np.float32)

    def _substep_control_loop(self, actions: np.ndarray):
        """Run physics with MuJoCo controller at every sub-step (500Hz).

        Two modes of operation:

        1. Pure-MuJoCo mode (when _mj_run_substeps_fn is available):
           Injects GPU state into MuJoCo ONCE, then runs ALL substeps
           in MuJoCo natively with full float64 precision and continuous
           solver state. Only writes the final result back to the GPU simulator.
           This matches the CPU baseline's physics exactly.

        2. Legacy per-substep mode (when only _mj_step_fn is available):
           Re-injects state at every substep through float32 round-trips.
           Kept for backwards compatibility but known to cause drift.
        """
        data = self.engine.data
        nu = self.model.nu
        nq = self.model.nq
        nv = self.model.nv

        _dbg_counter = getattr(self, '_dbg_ctrl_counter', 0)

        use_pure_mj = (hasattr(self, '_mj_run_substeps_fn') and
                       self._mj_run_substeps_fn is not None)

        if use_pure_mj:
            self._substep_pure_mujoco(actions, _dbg_counter)
        else:
            self._substep_legacy(actions, _dbg_counter)

        self._dbg_ctrl_counter = _dbg_counter + 1

    def _substep_pure_mujoco(self, actions: np.ndarray, _dbg_counter: int):
        """Run all substeps in MuJoCo with full float64 state continuity.

        Reads GPU state once, hands it to MuJoCo which runs all
        substeps natively (robot.control + sim.step per substep), then
        writes the final state back to the GPU simulator for rendering.
        No float32 round-trips between substeps.
        """
        data = self.engine.data
        nq = self.model.nq
        nv = self.model.nv
        nu = self.model.nu

        self.engine.sync()
        all_qpos = data.read_qpos()
        all_qvel = data.read_qvel()

        mj_final_qpos = [None] * self.num_envs
        mj_final_qvel = [None] * self.num_envs

        for ei in range(self.num_envs):
            qpos_ei = all_qpos[ei]
            qvel_ei = all_qvel[ei]
            if np.any(np.isnan(qpos_ei)) or np.any(np.isnan(qvel_ei)):
                np.nan_to_num(qpos_ei, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                np.nan_to_num(qvel_ei, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            ctrl, mj_q, mj_v = self._mj_run_substeps_fn(
                qpos_ei, qvel_ei, actions[ei], self._steps_per_action,
                env_idx=ei,
            )
            mj_final_qpos[ei] = mj_q
            mj_final_qvel[ei] = mj_v

            if _dbg_counter < 3:
                print(f"[PURE_MJ] step={_dbg_counter} env={ei} "
                      f"arm_ctrl={ctrl[:7] if ctrl is not None else 'None'} "
                      f"action={actions[ei][:3]}", flush=True)

        corrected_qpos = data.read_qpos().copy()
        corrected_qvel = data.read_qvel().copy()
        for ei in range(self.num_envs):
            if mj_final_qpos[ei] is not None:
                n_q = min(len(mj_final_qpos[ei]), nq)
                n_v = min(len(mj_final_qvel[ei]), nv)
                corrected_qpos[ei, :n_q] = mj_final_qpos[ei][:n_q]
                corrected_qvel[ei, :n_v] = mj_final_qvel[ei][:n_v]
        data.write_qpos(corrected_qpos.ravel())
        data.write_qvel(corrected_qvel.ravel())

        if _dbg_counter < 3:
            print(f"[PURE_MJ] step={_dbg_counter} wrote {nq} qpos, {nv} qvel "
                  f"from MuJoCo after {self._steps_per_action} substeps", flush=True)

        self.engine.forward()
        self.engine.sync()

    def _substep_legacy(self, actions: np.ndarray, _dbg_counter: int):
        """Legacy per-substep mode with float32 round-trips (kept for compat)."""
        data = self.engine.data
        nu = self.model.nu
        nq = self.model.nq
        nv = self.model.nv
        ctrl_batch = np.zeros((self.num_envs, nu), dtype=np.float32)
        last_valid_ctrl = np.zeros((self.num_envs, nu), dtype=np.float32)

        if not hasattr(self, '_per_env_ctrl_state'):
            self._per_env_ctrl_state = [None] * self.num_envs

        use_arm_correction = (hasattr(self, '_mj_step_fn') and
                              self._mj_step_fn is not None)

        arm_qpos_indices = None
        if self.osc is not None:
            arm_qpos_indices = self.osc.arm_qpos_indices

        _nan_counter = getattr(self, '_nan_counter', 0)

        mj_arm_qpos_per_env = [None] * self.num_envs
        mj_arm_qvel_per_env = [None] * self.num_envs

        for sub in range(self._steps_per_action):
            self.engine.sync()
            all_qpos = data.read_qpos()
            all_qvel = data.read_qvel()

            policy_step = (sub == 0)
            for ei in range(self.num_envs):
                qpos_ei = all_qpos[ei]
                qvel_ei = all_qvel[ei]
                if np.any(np.isnan(qpos_ei)) or np.any(np.isnan(qvel_ei)):
                    _nan_counter += 1
                    if _nan_counter <= 10:
                        nan_q = np.sum(np.isnan(qpos_ei))
                        nan_v = np.sum(np.isnan(qvel_ei))
                        print(f"[NaN_WARN] env={ei} sub={sub} step={_dbg_counter}: "
                              f"qpos NaN={nan_q}/{len(qpos_ei)}, qvel NaN={nan_v}/{len(qvel_ei)}", flush=True)
                    np.nan_to_num(qpos_ei, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                    np.nan_to_num(qvel_ei, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                self._restore_ctrl_state(ei)

                if use_arm_correction:
                    ctrl, mj_arm_q, mj_arm_v = self._mj_step_fn(
                        qpos_ei, qvel_ei, actions[ei],
                        policy_step=policy_step,
                    )
                    mj_arm_qpos_per_env[ei] = mj_arm_q
                    mj_arm_qvel_per_env[ei] = mj_arm_v
                else:
                    ctrl = self._mj_ctrl_fn(
                        qpos_ei, qvel_ei, actions[ei],
                        policy_step=policy_step,
                    )

                self._save_ctrl_state(ei)

                if np.any(np.isnan(ctrl)) or np.any(np.isinf(ctrl)):
                    ctrl = last_valid_ctrl[ei].copy()
                else:
                    last_valid_ctrl[ei] = ctrl[:nu].copy()

                ctrl_batch[ei, :min(nu, len(ctrl))] = ctrl[:nu]
                if self._gripper_actuator_indices and len(actions[ei]) > 6:
                    self._apply_gripper_ctrl(ctrl_batch, ei, actions[ei, -1])

            if _dbg_counter < 3 and sub == 0:
                c0 = ctrl_batch[0, :7]
                print(f"[CTRL_DBG] step={_dbg_counter} arm_ctrl={c0} action={actions[0][:3]}", flush=True)

            data.write_ctrl(ctrl_batch.ravel())
            self.engine.step()

            if use_arm_correction:
                self.engine.sync()
                corrected_qpos = data.read_qpos().copy()
                corrected_qvel = data.read_qvel().copy()
                applied = False
                for ei in range(self.num_envs):
                    mj_q = mj_arm_qpos_per_env[ei]
                    mj_v = mj_arm_qvel_per_env[ei]
                    if mj_q is not None:
                        n_copy_q = min(len(mj_q), nq)
                        n_copy_v = min(len(mj_v), nv)
                        corrected_qpos[ei, :n_copy_q] = mj_q[:n_copy_q]
                        corrected_qvel[ei, :n_copy_v] = mj_v[:n_copy_v]
                        applied = True
                if applied:
                    data.write_qpos(corrected_qpos.ravel())
                    data.write_qvel(corrected_qvel.ravel())
                    if _dbg_counter < 3 and sub == 0:
                        print(f"[FULL_CORR] step={_dbg_counter} sub={sub} "
                              f"copied {n_copy_q} qpos, {n_copy_v} qvel from MuJoCo", flush=True)

        self._nan_counter = _nan_counter

        if use_arm_correction and arm_qpos_indices is not None:
            self.engine.forward()
        else:
            self.engine.forward_kinematics()
            self.engine.geom_transforms()
        self.engine.sync()

    def _substep_native_osc(self, actions: np.ndarray):
        """Run physics with native OSC at 500Hz, matching robosuite exactly.

        Robosuite's OSC sets the goal ONCE per action (20Hz) but recomputes
        torques at every physics substep (500Hz) using updated Jacobians,
        mass matrices, and EEF state from sim.forward(). We match this:
        - set_goal equivalent (goal_pos, goal_ori) runs once
        - run_controller equivalent (torque computation) runs every substep

        When _use_mj_stepping is True, physics stepping is done by MuJoCo
        (mj_step) instead of the GPU simulator's engine.step(), isolating native OSC
        torque correctness from the GPU simulator's solver divergence.
        """
        from osc_controller import orientation_error
        import mujoco as _mj

        data = self.engine.data
        nu = self.model.nu
        osc = self.osc
        use_sites = osc.eef_site_idx is not None
        has_qfrc_bias = hasattr(data, 'read_qfrc_bias')
        use_mj_step = self._use_mj_stepping and self._mj_dyn is not None

        if use_mj_step:
            mj_m = self._mj_dyn['model']
            mj_per_env = self._mj_dyn['per_env_data']

        self.engine.sync()
        _has_double = hasattr(data, 'read_qpos_d')
        all_qpos = data.read_qpos_d() if _has_double else data.read_qpos()
        all_qvel = data.read_qvel_d() if _has_double else data.read_qvel()

        if use_mj_step:
            mj_init = getattr(self, '_mj_step_initialized', None)
            for ei in range(self.num_envs):
                d = mj_per_env[ei]
                if mj_init is None or not mj_init[ei]:
                    nq = min(len(all_qpos[ei]), mj_m.nq)
                    nv = min(len(all_qvel[ei]), mj_m.nv)
                    d.qpos[:nq] = all_qpos[ei, :nq].astype(np.float64)
                    d.qvel[:nv] = all_qvel[ei, :nv].astype(np.float64)
                    if mj_init is not None:
                        mj_init[ei] = True
                _mj.mj_forward(mj_m, d)

        if not use_mj_step:
            all_xpos = data.read_xpos()
            all_xquat = data.read_xquat()
            all_site_xpos = data.read_site_xpos() if use_sites else None
            all_site_xmat = data.read_site_xmat() if use_sites else None

        gripper_actions = np.zeros(self.num_envs, dtype=np.float32)
        for ei in range(self.num_envs):
            raw_action = actions[ei]
            action = np.clip(raw_action, -1.0, 1.0)

            if use_mj_step:
                d = mj_per_env[ei]
                if use_sites and osc.eef_site_idx is not None:
                    eef_pos = d.site_xpos[osc.eef_site_idx].copy()
                    eef_mat = d.site_xmat[osc.eef_site_idx].reshape(3, 3).copy()
                else:
                    eef_pos = d.xpos[osc.eef_body_idx].copy()
                    eef_mat = None
            else:
                xpos = all_xpos[ei]
                xquat = all_xquat[ei]
                if use_sites and all_site_xpos is not None:
                    eef_pos = all_site_xpos[ei, osc.eef_site_idx].copy()
                    eef_mat = all_site_xmat[ei, osc.eef_site_idx].reshape(3, 3).copy()
                else:
                    eef_pos = xpos[osc.eef_body_idx].copy()
                    eef_mat = quat_to_mat(xquat[osc.eef_body_idx])

            gripper_actions[ei] = raw_action[6] if len(raw_action) > 6 else 0.0

            if use_mj_step:
                arm_qpos = np.array([d.qpos[qi] for qi in osc.arm_qpos_indices])
            else:
                arm_qpos = np.array([all_qpos[ei, qi] for qi in osc.arm_qpos_indices])

            if self._per_env_goal_pos[ei] is None:
                self._per_env_goal_pos[ei] = eef_pos.copy()
                if self._per_env_goal_ori[ei] is None:
                    self._per_env_goal_ori[ei] = eef_mat.copy() if eef_mat is not None else eef_pos.copy()
                if self._robot_init_qpos is not None and len(self._robot_init_qpos) >= osc.n_arm:
                    self._per_env_init_jpos[ei] = np.array(self._robot_init_qpos[:osc.n_arm], dtype=np.float64)
                else:
                    self._per_env_init_jpos[ei] = arm_qpos.copy()

            pos_delta = action[:3] * osc.action_scale_pos
            ori_delta = action[3:6] * osc.action_scale_ori
            self._per_env_goal_pos[ei] = eef_pos + pos_delta
            angle = np.linalg.norm(ori_delta)
            if angle > 1e-6:
                axis = ori_delta / angle
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                self._per_env_goal_ori[ei] = dR @ eef_mat if eef_mat is not None else eef_pos.copy()
            else:
                if self._per_env_goal_ori[ei] is None:
                    if eef_mat is not None:
                        self._per_env_goal_ori[ei] = eef_mat.copy()

        # MuJoCo dynamics can be used in two scenarios:
        #  1. MuJoCo is stepping (always use MuJoCo dynamics)
        #  2. The GPU simulator is stepping but _use_mj_dyn_for_osc is set
        #     (diagnostic: accurate torques + GPU solver)
        use_mj_dyn = (self._mj_dyn is not None and
                      (use_mj_step or getattr(self, '_use_mj_dyn_for_osc', False)))

        _dbg = getattr(self, '_native_osc_dbg_counter', 0)

        for sub in range(self._steps_per_action):
            if use_mj_step:
                for ei in range(self.num_envs):
                    _mj.mj_forward(mj_m, mj_per_env[ei])
            else:
                self.engine.forward()
                self.engine.site_transforms()
                self.engine.sync()
                all_qpos = data.read_qpos_d() if _has_double else data.read_qpos()
                all_qvel = data.read_qvel_d() if _has_double else data.read_qvel()

            ctrl_batch = np.zeros((self.num_envs, nu), dtype=np.float64)
            for ei in range(self.num_envs):
                if use_mj_step:
                    d_ei = mj_per_env[ei]
                    qpos_ei = d_ei.qpos
                    qvel_ei = d_ei.qvel
                else:
                    qpos_ei = all_qpos[ei]
                    qvel_ei = all_qvel[ei]

                if use_mj_step:
                    arm_dofs = osc.arm_dof_indices
                    full_M = np.zeros((mj_m.nv, mj_m.nv), dtype=np.float64)
                    _mj.mj_fullM(mj_m, full_M, d_ei.qM)
                    M = full_M[np.ix_(arm_dofs, arm_dofs)]

                    if osc.eef_site_idx is not None:
                        jacp = np.zeros((3, mj_m.nv), dtype=np.float64)
                        jacr = np.zeros((3, mj_m.nv), dtype=np.float64)
                        _mj.mj_jacSite(mj_m, d_ei, jacp, jacr, osc.eef_site_idx)
                        J_pos = jacp[:, arm_dofs]
                        J_ori = jacr[:, arm_dofs]
                        eef_pos = d_ei.site_xpos[osc.eef_site_idx].copy()
                        eef_mat = d_ei.site_xmat[osc.eef_site_idx].reshape(3, 3).copy()
                    else:
                        jacp = np.zeros((3, mj_m.nv), dtype=np.float64)
                        jacr = np.zeros((3, mj_m.nv), dtype=np.float64)
                        _mj.mj_jacBody(mj_m, d_ei, jacp, jacr, osc.eef_body_idx)
                        J_pos = jacp[:, arm_dofs]
                        J_ori = jacr[:, arm_dofs]
                        eef_pos = d_ei.xpos[osc.eef_body_idx].copy()
                        eef_mat = None

                    bias_arm = d_ei.qfrc_bias[arm_dofs].copy()
                elif use_mj_dyn:
                    M, J_pos, J_ori, bias_arm, eef_pos, eef_mat = \
                        self._mj_compute_dynamics(qpos_ei, qvel_ei)
                else:
                    all_xpos = data.read_xpos()
                    all_xquat = data.read_xquat()
                    xpos = all_xpos[ei]
                    xquat = all_xquat[ei]
                    arm_qpos_only = np.array([qpos_ei[qi] for qi in osc.arm_qpos_indices])
                    if use_sites:
                        all_site_xpos = data.read_site_xpos()
                        all_site_xmat = data.read_site_xmat()
                        eef_pos = all_site_xpos[ei, osc.eef_site_idx].copy()
                        eef_mat = all_site_xmat[ei, osc.eef_site_idx].reshape(3, 3).copy()
                    else:
                        eef_pos = xpos[osc.eef_body_idx].copy()
                        eef_mat = quat_to_mat(xquat[osc.eef_body_idx])
                    J_pos, J_ori = osc._compute_jacobian(xpos, xquat, eef_pos)
                    M = osc._compute_mass_matrix(xpos, xquat)
                    bias_arm = None
                    if has_qfrc_bias:
                        all_qfrc_bias = data.read_qfrc_bias()
                        bias_arm = np.array([all_qfrc_bias[ei, di] for di in osc.arm_dof_indices])

                arm_qpos = np.array([qpos_ei[qi] for qi in osc.arm_qpos_indices])
                arm_qvel = np.array([qvel_ei[di] for di in osc.arm_dof_indices])

                J_full = np.vstack([J_pos, J_ori])
                vel_pos = J_pos @ arm_qvel
                vel_ori = J_ori @ arm_qvel

                pos_error = self._per_env_goal_pos[ei] - eef_pos
                ori_error = orientation_error(self._per_env_goal_ori[ei], eef_mat)

                F_desired = osc.kp_pos * pos_error - osc.kd_pos * vel_pos
                tau_desired = osc.kp_ori * ori_error - osc.kd_ori * vel_ori

                M_inv = np.linalg.inv(M)
                JMJ_pos = J_pos @ M_inv @ J_pos.T
                JMJ_ori = J_ori @ M_inv @ J_ori.T
                JMJ_full = J_full @ M_inv @ J_full.T

                Lambda_pos = np.linalg.pinv(JMJ_pos)
                Lambda_ori = np.linalg.pinv(JMJ_ori)

                decoupled_wrench = np.concatenate([
                    Lambda_pos @ F_desired,
                    Lambda_ori @ tau_desired
                ])
                torques = J_full.T @ decoupled_wrench

                if bias_arm is not None:
                    torques += bias_arm

                Lambda_full = np.linalg.pinv(JMJ_full)
                J_bar = M_inv @ J_full.T @ Lambda_full
                N = np.eye(osc.n_arm) - J_bar @ J_full
                if self._per_env_init_jpos[ei] is not None:
                    null_torques = N.T @ M @ (
                        osc.kp_null * (self._per_env_init_jpos[ei] - arm_qpos) -
                        osc.kd_null * arm_qvel
                    )
                    torques += null_torques

                ctrl = np.zeros(nu, dtype=np.float64)
                for i, ai in enumerate(osc.arm_actuator_indices):
                    if ai < nu:
                        lim = osc.arm_ctrl_limits[i]
                        ctrl[ai] = np.clip(torques[i], -lim, lim)

                ctrl_batch[ei] = ctrl
                if self._gripper_actuator_indices and len(actions[ei]) > 6:
                    self._apply_gripper_ctrl(ctrl_batch, ei, gripper_actions[ei])

            if use_mj_step:
                mj_nu = mj_m.nu
                for ei in range(self.num_envs):
                    d_ei = mj_per_env[ei]
                    d_ei.ctrl[:min(nu, mj_nu)] = ctrl_batch[ei, :min(nu, mj_nu)]
                    _mj.mj_step(mj_m, d_ei)
            else:
                data.write_ctrl(ctrl_batch.astype(np.float32).ravel())
                self.engine.step()
                if self._freeze_object_dofs:
                    if self._gripper_unfreeze:
                        qpos = self.engine.data.read_qpos()
                        for ei in range(self.num_envs):
                            if self._objects_unfrozen_per_env[ei]:
                                continue
                            if self._gripper_joint_indices:
                                g_vals = [abs(qpos[ei, gi])
                                          for gi in self._gripper_joint_indices]
                                aperture = sum(g_vals) / len(g_vals) if g_vals else 0.04
                                if aperture < self._gripper_unfreeze_threshold:
                                    self._objects_unfrozen_per_env[ei] = True
                                    self._detect_grasped_object(ei, qpos)
                                    print(f"[UNFREEZE] env {ei}: aperture {aperture:.4f} "
                                          f"< {self._gripper_unfreeze_threshold:.4f}, "
                                          f"grasped_obj={self._grasped_object_per_env[ei]}",
                                          flush=True)
                        self._objects_unfrozen = np.any(self._objects_unfrozen_per_env)

                    any_unfrozen = np.any(self._objects_unfrozen_per_env)
                    any_frozen = not np.all(self._objects_unfrozen_per_env)

                    if any_unfrozen:
                        self._selective_restore_passive_objects(
                            env_mask=self._objects_unfrozen_per_env)
                        if self._object_velocity_damping > 0:
                            self._damp_object_velocities(
                                env_mask=self._objects_unfrozen_per_env)
                    if any_frozen:
                        frozen_mask = ~self._objects_unfrozen_per_env
                        self._restore_object_dofs(env_mask=frozen_mask)

        if use_mj_step:
            final_qpos = np.zeros((self.num_envs, self.model.nq), dtype=np.float32)
            final_qvel = np.zeros((self.num_envs, self.model.nv), dtype=np.float32)
            for ei in range(self.num_envs):
                d_ei = mj_per_env[ei]
                nq = min(mj_m.nq, self.model.nq)
                nv = min(mj_m.nv, self.model.nv)
                final_qpos[ei, :nq] = d_ei.qpos[:nq].astype(np.float32)
                final_qvel[ei, :nv] = d_ei.qvel[:nv].astype(np.float32)
            data.write_qpos(final_qpos.ravel())
            data.write_qvel(final_qvel.ravel())
            if _dbg < 3:
                print(f"[MJ_STEP] Synced MuJoCo→GPU after "
                      f"{self._steps_per_action} substeps", flush=True)

        self._native_osc_dbg_counter = _dbg + 1
        self.engine.forward()
        self.engine.site_transforms()
        self.engine.sync()

    def _save_ctrl_state(self, env_idx):
        """Save the shared MuJoCo controller's goal state for this env."""
        try:
            checker = self._mj_ctrl_fn.__self__
            robot = checker._robot
            ctrl = robot.controller
            self._per_env_ctrl_state[env_idx] = {
                'goal_pos': ctrl.goal_pos.copy() if hasattr(ctrl, 'goal_pos') and ctrl.goal_pos is not None else None,
                'goal_ori': ctrl.goal_ori.copy() if hasattr(ctrl, 'goal_ori') and ctrl.goal_ori is not None else None,
            }
        except (AttributeError, IndexError):
            pass

    def _restore_ctrl_state(self, env_idx):
        """Restore the shared MuJoCo controller's goal state for this env."""
        state = self._per_env_ctrl_state[env_idx]
        if state is None:
            return
        try:
            checker = self._mj_ctrl_fn.__self__
            robot = checker._robot
            ctrl = robot.controller
            if state['goal_pos'] is not None and hasattr(ctrl, 'goal_pos'):
                ctrl.goal_pos = state['goal_pos'].copy()
            if state['goal_ori'] is not None and hasattr(ctrl, 'goal_ori'):
                ctrl.goal_ori = state['goal_ori'].copy()
        except (AttributeError, IndexError):
            pass

    def _control_batch(self, actions: np.ndarray):
        """Batched control: delegates to MuJoCo controller or custom OSC.
        Used only when _mj_ctrl_fn is not set (falls back to single-shot ctrl).
        """
        self.engine.sync()
        data = self.engine.data

        all_qpos = data.read_qpos()
        all_qvel = data.read_qvel()

        nu = self.model.nu
        ctrl_batch = np.zeros((self.num_envs, nu), dtype=np.float32)

        if hasattr(self, '_mj_ctrl_fn') and self._mj_ctrl_fn is not None:
            for ei in range(self.num_envs):
                ctrl = self._mj_ctrl_fn(all_qpos[ei], all_qvel[ei], actions[ei])
                ctrl_batch[ei, :min(nu, len(ctrl))] = ctrl[:nu]
                if self._gripper_actuator_indices and len(actions[ei]) > 6:
                    self._apply_gripper_ctrl(ctrl_batch, ei, actions[ei, -1])
        else:
            all_qfrc_bias = data.read_qfrc_bias() if hasattr(data, 'read_qfrc_bias') else None
            all_xpos = data.read_xpos()
            all_xquat = data.read_xquat()
            all_site_xpos2 = data.read_site_xpos() if self.osc.eef_site_idx is not None else None
            all_site_xmat2 = data.read_site_xmat() if self.osc.eef_site_idx is not None else None
            osc = self.osc

            for ei in range(self.num_envs):
                action = actions[ei]
                qpos_all = all_qpos[ei]
                qvel_all = all_qvel[ei]
                xpos = all_xpos[ei]
                xquat = all_xquat[ei]

                arm_qpos = np.array([qpos_all[qi] for qi in osc.arm_qpos_indices])
                arm_qvel = np.array([qvel_all[di] for di in osc.arm_dof_indices])

                if osc.eef_site_idx is not None and all_site_xpos2 is not None:
                    eef_pos = all_site_xpos2[ei, osc.eef_site_idx].copy()
                    eef_mat = all_site_xmat2[ei, osc.eef_site_idx].reshape(3, 3).copy()
                else:
                    eef_pos = xpos[osc.eef_body_idx].copy()
                    eef_mat = quat_to_mat(xquat[osc.eef_body_idx])

                pos_delta = action[:3] * osc.action_scale_pos
                ori_delta = action[3:6] * osc.action_scale_ori
                gripper_action = action[6] if len(action) > 6 else 0.0

                if self._per_env_goal_pos[ei] is None:
                    self._per_env_goal_pos[ei] = eef_pos.copy()
                    self._per_env_goal_ori[ei] = eef_mat.copy()
                    if self._robot_init_qpos is not None and len(self._robot_init_qpos) >= osc.n_arm:
                        self._per_env_init_jpos[ei] = np.array(self._robot_init_qpos[:osc.n_arm], dtype=np.float64)
                    else:
                        self._per_env_init_jpos[ei] = arm_qpos.copy()

                self._per_env_goal_pos[ei] = eef_pos + pos_delta

                angle = np.linalg.norm(ori_delta)
                if angle > 1e-6:
                    axis = ori_delta / angle
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                    self._per_env_goal_ori[ei] = dR @ eef_mat

                pos_error = self._per_env_goal_pos[ei] - eef_pos
                from osc_controller import orientation_error
                ori_error = orientation_error(self._per_env_goal_ori[ei], eef_mat)

                J_pos, J_ori = osc._compute_jacobian(xpos, xquat, eef_pos)
                J_full = np.vstack([J_pos, J_ori])

                vel_pos = J_pos @ arm_qvel
                vel_ori = J_ori @ arm_qvel

                F_desired = osc.kp_pos * pos_error - osc.kd_pos * vel_pos
                tau_desired = osc.kp_ori * ori_error - osc.kd_ori * vel_ori

                M = osc._compute_mass_matrix(xpos, xquat)
                M_inv = np.linalg.inv(M)

                JMJ_pos = J_pos @ M_inv @ J_pos.T
                JMJ_ori = J_ori @ M_inv @ J_ori.T
                JMJ_full = J_full @ M_inv @ J_full.T

                Lambda_pos = np.linalg.pinv(JMJ_pos)
                Lambda_ori = np.linalg.pinv(JMJ_ori)

                decoupled_wrench = np.concatenate([
                    Lambda_pos @ F_desired,
                    Lambda_ori @ tau_desired
                ])
                torques = J_full.T @ decoupled_wrench

                if all_qfrc_bias is not None:
                    arm_bias = np.array([all_qfrc_bias[ei, di] for di in osc.arm_dof_indices])
                    torques += arm_bias

                Lambda_full = np.linalg.pinv(JMJ_full)
                J_bar = M_inv @ J_full.T @ Lambda_full
                N = np.eye(osc.n_arm) - J_bar @ J_full
                if self._per_env_init_jpos[ei] is not None:
                    null_torques = N.T @ M @ (
                        osc.kp_null * (self._per_env_init_jpos[ei] - arm_qpos) -
                        osc.kd_null * arm_qvel
                    )
                    torques += null_torques

                lim = osc.arm_ctrl_limits
                ctrl = np.zeros(nu, dtype=np.float32)
                for i, ai in enumerate(osc.arm_actuator_indices):
                    if ai < nu:
                        l = lim[i] if i < len(lim) else 87.0
                        ctrl[ai] = np.clip(torques[i], -l, l)
                if self._gripper_actuator_indices and len(action) > 6:
                    ctrl_batch[ei] = ctrl
                    self._apply_gripper_ctrl(ctrl_batch, ei, gripper_action)
                    ctrl = ctrl_batch[ei]
                else:
                    for gi in osc.gripper_actuator_indices:
                        if gi < nu:
                            ctrl[gi] = gripper_action
                ctrl_batch[ei] = ctrl

        # Log ctrl for gripper, especially when gripper action changes
        if not hasattr(self, '_ctrl_log_count'):
            self._ctrl_log_count = 0
            self._last_grip_action = None
        self._ctrl_log_count += 1
        nu = self.model.nu
        grip_action = actions[0, -1] if len(actions) > 0 else 0
        grip_changed = (self._last_grip_action is not None and
                       np.sign(grip_action) != np.sign(self._last_grip_action))
        if self._ctrl_log_count <= 3 or grip_changed or self._ctrl_log_count % 40 == 0:
            qpos_env0 = self.engine.data.read_qpos()[0]
            grip_qpos = []
            if self._gripper_joint_indices:
                grip_qpos = [qpos_env0[self.model.joint_qpos_adr(ji)]
                            for ji in self._gripper_joint_indices]
            grip_ctrl = ctrl_batch[0, 7:nu]
            print(f"[CTRL] step {self._ctrl_log_count} env 0: "
                  f"action[-1]={grip_action:.3f}, gripper_ctrl={grip_ctrl}, "
                  f"gripper_qpos={grip_qpos}"
                  f"{' ** GRIP CHANGED **' if grip_changed else ''}",
                  flush=True)
        self._last_grip_action = grip_action

        data.write_ctrl(ctrl_batch.ravel())

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Build full LIBERO-format observation for a single environment."""
        obs = {}
        for cname in self.camera_names:
            if cname in self._camera_map:
                img = self._renderer.render(
                    self.engine, camera_idx=self._camera_map[cname], env_idx=env_idx
                )
                obs[f"{cname}_image"] = img

        if self._eef_site_idx is not None:
            site_xpos = self.engine.data.read_site_xpos()
            obs["robot0_eef_pos"] = site_xpos[env_idx, self._eef_site_idx].copy()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[env_idx, self._eef_body_idx].copy())
        elif self._eef_body_idx is not None:
            xpos = self.engine.data.read_xpos()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_pos"] = xpos[env_idx, self._eef_body_idx].copy()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[env_idx, self._eef_body_idx].copy())

        if self._gripper_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[env_idx]
            gripper_qpos = np.array([
                qpos[self.model.joint_qpos_adr(ji)]
                for ji in self._gripper_joint_indices
            ], dtype=np.float32)
            obs["robot0_gripper_qpos"] = gripper_qpos

        if self._arm_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[env_idx]
            obs["robot0_joint_pos"] = qpos.copy()

        return obs

    def _get_state_obs(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Return robot state observations only (no rendering)."""
        obs = {}
        if self._eef_site_idx is not None:
            site_xpos = self.engine.data.read_site_xpos()
            obs["robot0_eef_pos"] = site_xpos[env_idx, self._eef_site_idx].copy()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[env_idx, self._eef_body_idx].copy())
        elif self._eef_body_idx is not None:
            xpos = self.engine.data.read_xpos()
            xquat = self.engine.data.read_xquat()
            obs["robot0_eef_pos"] = xpos[env_idx, self._eef_body_idx].copy()
            obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(xquat[env_idx, self._eef_body_idx].copy())

        if self._gripper_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[env_idx]
            obs["robot0_gripper_qpos"] = np.array([
                qpos[self.model.joint_qpos_adr(ji)]
                for ji in self._gripper_joint_indices
            ], dtype=np.float32)

        if self._arm_joint_indices is not None:
            qpos = self.engine.data.read_qpos()[env_idx]
            obs["robot0_joint_pos"] = qpos.copy()
        return obs

    def _get_all_state_obs(self) -> List[Dict[str, np.ndarray]]:
        """Return robot state for all envs with a single D2H read."""
        all_qpos = self.engine.data.read_qpos()
        all_site_xpos = self.engine.data.read_site_xpos() if self._eef_site_idx is not None else None
        all_xpos = self.engine.data.read_xpos() if self._eef_body_idx is not None and self._eef_site_idx is None else None
        all_xquat = self.engine.data.read_xquat() if self._eef_body_idx is not None else None

        results = []
        for ei in range(self.num_envs):
            obs = {}
            if self._eef_site_idx is not None:
                obs["robot0_eef_pos"] = all_site_xpos[ei, self._eef_site_idx].copy()
                obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(all_xquat[ei, self._eef_body_idx].copy())
            elif self._eef_body_idx is not None:
                obs["robot0_eef_pos"] = all_xpos[ei, self._eef_body_idx].copy()
                obs["robot0_eef_quat"] = quat_wxyz_to_xyzw(all_xquat[ei, self._eef_body_idx].copy())
            if self._gripper_joint_indices is not None:
                obs["robot0_gripper_qpos"] = np.array([
                    all_qpos[ei, self.model.joint_qpos_adr(ji)]
                    for ji in self._gripper_joint_indices
                ], dtype=np.float32)
            if self._arm_joint_indices is not None:
                obs["robot0_joint_pos"] = all_qpos[ei].copy()
            results.append(obs)
        return results

    def _get_all_obs_batched(self, render_rows=None) -> List[Dict[str, np.ndarray]]:
        """Batched observation: 1 state read + 1 batch render per camera.

        Eliminates N redundant D2H transfers that the per-env _get_obs path
        incurs (the per-env renderer reads ALL envs' state for every call).

        render_rows restricts the render to the worlds that will
        consume this round's images (see step_all). Idle rows get None images.
        """
        _active = None
        if render_rows is not None and _HP_RENDER_SKIP_IDLE:
            _rows = sorted({int(r) for r in render_rows})
            if not _rows:
                raise ValueError("render_rows is empty: a round with no consumer")
            if _rows[0] < 0 or _rows[-1] >= self.num_envs:
                raise ValueError(f"render_rows {_rows} outside 0..{self.num_envs - 1}")
            if len(_rows) < self.num_envs:
                _active = _rows
        _tm = _HP_OBS_TIMERS
        if _tm:
            _pc = _hp_time.perf_counter
            _rec = {"n_active": self.num_envs if _active is None else len(_active)}
            _ta = _pc()
        all_qpos = self.engine.data.read_qpos()
        all_site_xpos = self.engine.data.read_site_xpos() if self._eef_site_idx is not None else None
        all_xpos = self.engine.data.read_xpos() if self._eef_body_idx is not None and self._eef_site_idx is None else None
        all_xquat = self.engine.data.read_xquat() if self._eef_body_idx is not None else None
        if _tm:
            _tb = _pc(); _rec["state_d2h"] = (_tb - _ta) * 1e3

        cam_images = {}
        _names = [c for c in self.camera_names if c in self._camera_map]
        if getattr(self, "_rt", False) and _names:
            # HP_RENDERER=rt: every camera in ONE launch, one D2H for all of them.
            _all = self._renderer.render_all_gpu_batch(
                self.engine, [self._camera_map[c] for c in _names], self.num_envs, active=_active)
            if _tm:
                _tc = _pc(); _rec["render"] = (_tc - _tb) * 1e3
            _host = _all.numpy()
            for _i, cname in enumerate(_names):
                cam_images[cname] = _host[_i]
            if _tm:
                _td = _pc(); _rec["img_d2h"] = (_td - _tc) * 1e3
        elif getattr(self, "_par_cams", False) and len(_names) > 1:
            import threading
            _out = [None] * len(_names)
            def _one(_i):
                _out[_i] = self.render_gpu_batch(_names[_i], active=_active)
            _ts = [threading.Thread(target=_one, args=(i,))
                   for i in range(1, len(_names))]
            for _t in _ts:
                _t.start()
            _one(0)
            for _t in _ts:
                _t.join()
            if _tm:
                _tc = _pc(); _rec["render"] = (_tc - _tb) * 1e3
            for _i, cname in enumerate(_names):
                cam_images[cname] = _out[_i].numpy()
            if _tm:
                _td = _pc(); _rec["img_d2h"] = (_td - _tc) * 1e3
        else:
            if _tm:
                _rec["render"] = 0.0; _rec["img_d2h"] = 0.0
            for cname in _names:
                if _tm:
                    _t0c = _pc()
                    _gb = self.render_gpu_batch(cname, active=_active)
                    _t1c = _pc()
                    cam_images[cname] = _gb.numpy()
                    _t2c = _pc()
                    _rec["render"] += (_t1c - _t0c) * 1e3
                    _rec["img_d2h"] += (_t2c - _t1c) * 1e3
                else:
                    cam_images[cname] = self.render_gpu_batch(cname, active=_active).numpy()
            if _tm:
                _td = _pc()

        n = self.num_envs
        img_keys = [(f"{cname}_image", images) for cname, images in cam_images.items()]
        eef_pos = eef_quat = grip = joint = None
        if self._eef_site_idx is not None:
            eef_pos = all_site_xpos[:n, self._eef_site_idx].copy()
        elif self._eef_body_idx is not None:
            eef_pos = all_xpos[:n, self._eef_body_idx].copy()
        if eef_pos is not None:
            # quat_wxyz_to_xyzw, batched: (w,x,y,z) -> (x,y,z,w)
            eef_quat = all_xquat[:n, self._eef_body_idx].take((1, 2, 3, 0), axis=1)
        if self._gripper_joint_indices is not None:
            adrs = getattr(self, "_obs_grip_adrs", None)
            if adrs is None:
                adrs = [self.model.joint_qpos_adr(ji) for ji in self._gripper_joint_indices]
                self._obs_grip_adrs = adrs
            grip = all_qpos[:n].take(adrs, axis=1).astype(np.float32, copy=False)
        if self._arm_joint_indices is not None:
            joint = all_qpos

        results = []
        _live = None if _active is None else set(_active)
        for ei in range(n):
            if _live is None or ei in _live:
                obs = {k: images[ei] for k, images in img_keys}
            else:
                obs = {k: None for k, _ in img_keys}
            if eef_pos is not None:
                obs["robot0_eef_pos"] = eef_pos[ei]
                obs["robot0_eef_quat"] = eef_quat[ei]
            if grip is not None:
                obs["robot0_gripper_qpos"] = grip[ei]
            if joint is not None:
                obs["robot0_joint_pos"] = joint[ei]
            results.append(obs)
        if _tm:
            _rec["dict"] = (_pc() - _td) * 1e3
            self._obs_tcur = _rec
        return results

    def _obs_timer_summary(self) -> None:
        """HP_OBS_TIMERS=1 -- every 200 rounds print mean render /
        img_d2h / obs ms of the last 200 rounds, bucketed by active count."""
        log = getattr(self, "_obs_tlog", None)
        if not log or len(log) % 200:
            return
        win = log[-200:]
        buckets = {}
        for r in win:
            buckets.setdefault(r.get("n_active", -1), []).append(r)
        def _m(rs, k):
            v = [r[k] for r in rs if k in r]
            return sum(v) / len(v) if v else float("nan")
        parts = " ".join(
            f"a{a}:n={len(rs)},render={_m(rs, 'render'):.2f},d2h={_m(rs, 'img_d2h'):.2f},obs={_m(rs, 'obs'):.2f}"
            for a, rs in sorted(buckets.items()))
        print(f"[TIMING] obs_tl #{len(log)} all:render={_m(win, 'render'):.2f},"
              f"obs={_m(win, 'obs'):.2f},phys={_m(win, 'phys'):.2f},"
              f"n_active={_m(win, 'n_active'):.2f} | {parts}", flush=True)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_gpu(self, camera_name: str, env_idx: int = 0,
                   flip_v: bool = False, flip_h: bool = False):
        """Render a single env camera as GPU-resident GpuImage."""
        cid = self._camera_map.get(camera_name)
        if cid is None:
            cid = self.model.camera_name2id(camera_name)
            if cid < 0:
                raise ValueError(f"Camera '{camera_name}' not found")
        return self._renderer.render_gpu(
            self.engine, camera_idx=cid, env_idx=env_idx,
            flip_v=flip_v, flip_h=flip_h
        )

    def render_cameras_gpu(self, camera_names=None, active=None):
        """Every requested camera (default: the env's cameras) for every world, left on the GPU.

        Returns (names, batches): with HP_RENDERER=rt ONE device array (C, N, H, W, 3) u8 from a
        single launch; with the raster a list of C device batches (N, H, W, 3) u8, available only
        when every camera owns its renderer (a raster batch is overwritten by that renderer's next
        call, so cameras sharing one renderer cannot be held at once: render_gpu_batch them one at
        a time instead). Both expose __cuda_array_interface__ (e.g. torch.as_tensor(x,
        device='cuda'), no host copy); row 0 = top, as render_gpu_batch. `active`: optional strict
        subset of worlds to draw (other rows are stale)."""
        names = [c for c in (camera_names or self.camera_names) if c in self._camera_map]
        if getattr(self, "_rt", False):
            return names, self._renderer.render_all_gpu_batch(
                self.engine, [self._camera_map[c] for c in names], self.num_envs, active=active)
        owners = [id(self._cam_renderers.get(c, self._renderer)) for c in names]
        if len(set(owners)) != len(owners):
            raise RuntimeError("render_cameras_gpu: raster cameras share one renderer (its next call "
                               "overwrites the previous batch); use render_gpu_batch per camera")
        return names, [self.render_gpu_batch(c, active=active) for c in names]

    def render_gpu_batch(self, camera_name: str,
                         flip_v: bool = False, flip_h: bool = False,
                         active=None):
        """Render all envs for one camera. Returns GpuImageBatch (N,H,W,3) on GPU.

        active : optional strict subset of world ids to draw; the other
        rows of the returned batch are stale and must not be read.
        """
        cid = self._camera_map.get(camera_name)
        if cid is None:
            cid = self.model.camera_name2id(camera_name)
            if cid < 0:
                raise ValueError(f"Camera '{camera_name}' not found")
        cr = self._cam_renderers.get(camera_name, self._renderer)
        if active is None:
            return cr.render_gpu_batch(
                self.engine, camera_idx=cid, num_envs=self.num_envs,
                flip_v=flip_v, flip_h=flip_h
            )
        return cr.render_gpu_batch(
            self.engine, camera_idx=cid, num_envs=self.num_envs,
            flip_v=flip_v, flip_h=flip_h, active=list(active)
        )

    # ── State management ──────────────────────────────────────────────────────

    def set_init_state(self, env_idx: int, state: np.ndarray) -> None:
        """Set state for a single environment.

        Accepts both MuJoCo flattened state [time, qpos, qvel, ...] and
        bare [qpos, qvel] formats.

        NOTE: Call finalize_init() after setting all environments to trigger
        a single physics step for FK computation without cross-env corruption.
        """
        nq = self.model.nq
        nv = self.model.nv
        off = 1 if len(state) >= 1 + nq + nv else 0
        qpos_f64 = state[off:off + nq].astype(np.float64)
        qvel_f64 = state[off + nq:off + nq + nv].astype(np.float64) if len(state) > off + nq else np.zeros(nv, dtype=np.float64)

        if hasattr(self.engine.data, 'write_qpos_d_env'):
            self.engine.data.write_qpos_d_env(qpos_f64, env_idx)
            self.engine.data.write_qvel_d_env(qvel_f64, env_idx)
        else:
            self.engine.data.write_qpos_env(qpos_f64.astype(np.float32), env_idx)
            self.engine.data.write_qvel_env(qvel_f64.astype(np.float32), env_idx)
        if hasattr(self.engine, 'reset_warmstart_env'):
            self.engine.reset_warmstart_env(env_idx)
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        if self.model.nu > 0:
            self.engine.data.write_ctrl_env(ctrl, env_idx)

        if not hasattr(self, '_pending_init'):
            self._pending_init = {}
        self._pending_init[env_idx] = (qpos_f64.copy(), qvel_f64.copy())

        self._step_counts[env_idx] = 0
        self._per_env_goal_pos[env_idx] = None
        self._per_env_goal_ori[env_idx] = None
        self._per_env_init_jpos[env_idx] = None
        self._gripper_current_action[env_idx] = 0.0
        if hasattr(self, '_mj_step_initialized') and env_idx < len(self._mj_step_initialized):
            self._mj_step_initialized[env_idx] = False

    def finalize_init(self):
        """Run one physics step to compute FK, then restore all pending init states."""
        if not hasattr(self, '_pending_init') or not self._pending_init:
            return
        self.engine.step()
        for env_idx, (qpos, qvel) in self._pending_init.items():
            if hasattr(self.engine.data, 'write_qpos_d_env'):
                self.engine.data.write_qpos_d_env(np.asarray(qpos, dtype=np.float64), env_idx)
                self.engine.data.write_qvel_d_env(np.asarray(qvel, dtype=np.float64), env_idx)
            else:
                self.engine.data.write_qpos_env(np.asarray(qpos, dtype=np.float32), env_idx)
                self.engine.data.write_qvel_env(np.asarray(qvel, dtype=np.float32), env_idx)
        self.engine.forward()
        self.engine.sync()
        if self._freeze_object_dofs:
            self._snapshot_object_dofs()
            self._objects_unfrozen = False
            self._objects_unfrozen_per_env = np.zeros(self.num_envs, dtype=bool)
            if hasattr(self, '_grasped_object_per_env'):
                self._grasped_object_per_env = np.full(self.num_envs, -1, dtype=int)
        if self.osc is not None:
            if self._robot_init_qpos is not None and len(self._robot_init_qpos) >= self.osc.n_arm:
                null_ref = np.array(self._robot_init_qpos[:self.osc.n_arm], dtype=np.float64)
                for env_idx in self._pending_init.keys():
                    self._per_env_init_jpos[env_idx] = null_ref.copy()
            else:
                all_qpos = self.engine.data.read_qpos()
                for env_idx in self._pending_init.keys():
                    self._per_env_init_jpos[env_idx] = np.array(
                        [all_qpos[env_idx, qi] for qi in self.osc.arm_qpos_indices]
                    )
        self._pending_init.clear()

    def cache_reset_dynamics(self, warmup_steps: int = 10):
        """Pre-compute M and N at the reset configuration (robot_init_qpos).

        Robosuite's OSC uses stale M/J/N from the reset config during warmup
        (policy_step=False skips controller.update()). This caches those values
        so the GPU simulator can replicate the same warmup behavior.
        """
        if self.osc is None or self._robot_init_qpos is None:
            return

        osc = self.osc

        # Save current state
        self.engine.sync()
        _has_d = hasattr(self.engine.data, 'read_qpos_d')
        saved_qpos = (self.engine.data.read_qpos_d() if _has_d else self.engine.data.read_qpos()).copy()
        saved_qvel = (self.engine.data.read_qvel_d() if _has_d else self.engine.data.read_qvel()).copy()

        # Temporarily set env 0 to the reset configuration
        reset_qpos = saved_qpos[0].copy()
        reset_qvel = np.zeros_like(saved_qvel[0])
        for i, qi in enumerate(osc.arm_qpos_indices):
            reset_qpos[qi] = self._robot_init_qpos[i]
        if _has_d:
            self.engine.data.write_qpos_d_env(reset_qpos.astype(np.float64), 0)
            self.engine.data.write_qvel_d_env(reset_qvel.astype(np.float64), 0)
        else:
            self.engine.data.write_qpos_env(reset_qpos.astype(np.float32), 0)
            self.engine.data.write_qvel_env(reset_qvel.astype(np.float32), 0)
        self.engine.forward()
        self.engine.sync()

        xpos = self.engine.data.read_xpos()[0]
        xquat = self.engine.data.read_xquat()[0]
        site_xpos = self.engine.data.read_site_xpos()[0] if osc.eef_site_idx is not None else None

        eef_pos = site_xpos[osc.eef_site_idx].copy() if site_xpos is not None else xpos[osc.eef_body_idx].copy()

        J_pos, J_ori = osc._compute_jacobian(xpos, xquat, eef_pos)
        J_full = np.vstack([J_pos, J_ori])
        M = osc._compute_mass_matrix(xpos, xquat)
        M_inv = np.linalg.inv(M)
        JMJ_full = J_full @ M_inv @ J_full.T
        Lambda_full = np.linalg.pinv(JMJ_full)
        J_bar = M_inv @ J_full.T @ Lambda_full
        N = np.eye(osc.n_arm) - J_bar @ J_full

        self._cached_reset_M = M
        self._cached_reset_N = N

        # Restore original state
        for ei in range(self.num_envs):
            if _has_d:
                self.engine.data.write_qpos_d_env(saved_qpos[ei].astype(np.float64), ei)
                self.engine.data.write_qvel_d_env(saved_qvel[ei].astype(np.float64), ei)
            else:
                self.engine.data.write_qpos_env(saved_qpos[ei].astype(np.float32), ei)
                self.engine.data.write_qvel_env(saved_qvel[ei].astype(np.float32), ei)
        self.engine.forward()
        self.engine.sync()

        self._warmup_steps_remaining = [warmup_steps] * self.num_envs

    def check_success(self) -> bool:
        return self.check_success_env(0)

    def check_success_env(self, env_idx: int) -> bool:
        if self._success_checker is None:
            return False
        return self._success_checker.check(self.engine, env_idx)

    def close(self):
        pass

    @property
    def data(self):
        return self.engine.data

    @property
    def success_checker(self):
        return self._success_checker
