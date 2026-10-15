# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""V3 preserves ordered visualization-name repetitions across hard resets.

Detached LIBERO site-visual bookkeeping; no simulator or renderer imports.

Construct after the reference domain's set_init_state has completed. Publish
``hp_rgba`` for the first image without calling ``advance``. For normal steps,
archive/render the current snapshot first, then call ``advance(native_qpos)``
and publish its returned snapshot for the NEXT image. This module deliberately
does not perform that integration, render, score, reset, or advance physics.

Supported callbacks are the installed LIBERO ObjectState update/turn_on chain
and the domain's update_state-then-set_visualization post-process structure.
The callbacks run against a newly constructed minimal proxy, never a copied
environment. Unsupported callback dependencies fail closed. Reset means
constructing a fresh adapter from the new reference state.
"""
from __future__ import annotations

import ast
import builtins
import copy
import dis
import inspect
from numbers import Integral
import textwrap
from types import FunctionType, MappingProxyType, MethodType, SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np


class VisualStateError(ValueError):
    """The inputs or reference callback contract are unsupported."""


def _require(ok, message):
    if not ok:
        raise VisualStateError(message)


def _integer(value, label):
    _require(isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)),
             f"{label} must be an integer")
    value = int(value)
    _require(value >= 0, f"{label} must be nonnegative")
    return value


def _array(value, shape, label):
    array = np.asarray(value)
    _require(array.shape == shape and array.dtype.kind == "f", f"Invalid {label} shape/dtype")
    _require(np.isfinite(array).all(), f"Nonfinite {label}")
    return np.array(array, copy=True, order="C")


def _names(model, count, kind, native):
    name_at = getattr(model, f"{kind}_name" if native else f"{kind}_id2name")
    name_to_id = getattr(model, f"{kind}_name2id")
    names = tuple(name_at(i) for i in range(count))
    _require(all(isinstance(name, str) and name for name in names), f"Unnamed {kind}")
    _require(len(set(names)) == count, f"Duplicate {kind} name")
    _require(all(_integer(name_to_id(name), f"{kind} id") == i
                 for i, name in enumerate(names)), f"Inconsistent {kind} name lookup")
    return names


def _function(cls, name, allowed_names, allowed_globals=()):
    """Conservative dependency guard for the small supported Python callbacks.

    This is a compatibility check for trusted installed source, not a sandbox
    for adversarial Python. No closures, imports, native dependencies, or
    environment-bound callables are accepted.
    """
    fn = inspect.getattr_static(cls, name)
    _require(isinstance(fn, FunctionType) and not fn.__closure__, f"Unsupported {name} callback")
    _require(set(fn.__code__.co_names) <= set(allowed_names), f"Unsupported dependencies in {name}")
    _require(all(value is None or type(value) in (bool, int, float, str)
                 for value in (fn.__defaults__ or ())), f"Captured default in {name}")
    for instruction in dis.get_instructions(fn):
        _require(instruction.opname not in {"IMPORT_NAME", "IMPORT_FROM", "STORE_GLOBAL", "DELETE_GLOBAL"},
                 f"Unsupported operation in {name}")
        if instruction.opname == "LOAD_GLOBAL":
            key = instruction.argval
            _require(key in allowed_globals and
                     fn.__globals__.get(key, getattr(builtins, key, None)) is getattr(builtins, key, None),
                     f"Unsupported global {key} in {name}")
    return fn


def _syntax(fn):
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
    except (OSError, TypeError, SyntaxError) as exc:
        raise VisualStateError("Callback source must be available for ordering validation") from exc
    _require(isinstance(node, ast.FunctionDef), "Unsupported post-process definition")
    node.decorator_list = []
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
        node.body = node.body[1:]
    return ast.dump(node, include_attributes=False)


_POST_VISUAL = ast.dump(ast.parse("""def _post_process(self):
    super()._post_process()
    self.set_visualization()
""").body[0], include_attributes=False)
_POST_TRACK = ast.dump(ast.parse("""def _post_process(self):
    for object_state in self.tracking_object_states_change:
        object_state.update_state()
""").body[0], include_attributes=False)


def _post_process_contract(cls):
    definitions = [base.__dict__["_post_process"] for base in cls.__mro__
                   if "_post_process" in base.__dict__]
    _require(len(definitions) >= 2 and _syntax(definitions[0]) == _POST_VISUAL and
             _syntax(definitions[1]) == _POST_TRACK, "Unsupported LIBERO post-process ordering")


def _check_definition(value, forbidden_ids, seen=None):
    """Reject simulator/resource references before deepcopying model definitions."""
    if seen is None:
        seen = set()
    _require(id(value) not in forbidden_ids, "Object definition retains a native/environment resource")
    if id(value) in seen:
        return
    seen.add(id(value))
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return
    if isinstance(value, np.ndarray):
        _require(value.dtype.kind != "O", "Object-valued definition array is unsupported")
        return
    if isinstance(value, ET.Element):
        return  # XML model definitions have no simulator handles.
    if isinstance(value, dict):
        children = (*value.keys(), *value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = value
    else:
        _require(hasattr(value, "__dict__") and not isinstance(value, (type, FunctionType, MethodType)),
                 f"Unsupported object-definition resource: {type(value).__name__}")
        children = vars(value).values()
    for child in children:
        _check_definition(child, forbidden_ids, seen)


class _VisualModel:
    __slots__ = ("site_rgba", "_site_ids", "_joint_addrs")

    def __init__(self, rgba, site_ids, joint_addrs):
        self.site_rgba = rgba
        self._site_ids = site_ids
        self._joint_addrs = joint_addrs

    def site_name2id(self, name):
        return self._site_ids[name]

    def get_joint_qpos_addr(self, name):
        return self._joint_addrs[name]


class _VisualProxy:
    __slots__ = ("sim", "objects_dict", "fixtures_dict", "object_sites_dict",
                 "visualization_sites_list", "tracking_object_states_change", "object_states_dict")

    def __init__(self, rgba, qpos, site_ids, joint_addrs, definitions, fixture_names, visual_names):
        qpos.setflags(write=False)
        self.sim = SimpleNamespace(model=_VisualModel(rgba, site_ids, joint_addrs),
                                   data=SimpleNamespace(qpos=qpos))
        self.objects_dict = {name: obj for name, obj in definitions.items() if name not in fixture_names}
        self.fixtures_dict = {name: obj for name, obj in definitions.items() if name in fixture_names}
        self.object_sites_dict = {}
        self.visualization_sites_list = visual_names
        self.tracking_object_states_change = []
        self.object_states_dict = {}

    def get_object(self, name):
        if name in self.fixtures_dict:
            return self.fixtures_dict[name]
        return self.objects_dict[name]


class SiteVisualState:
    """One world's detached state. All public array accessors return copies."""

    @classmethod
    def from_reference(cls, reference_domain_env, hp_model):
        ref = reference_domain_env
        cpu = ref.sim.model
        nsite, njnt, nq, nv = (_integer(getattr(cpu, key), f"CPU {key}")
                              for key in ("nsite", "njnt", "nq", "nv"))
        _require(all(_integer(getattr(hp_model, key), f"HP {key}") == value
                     for key, value in zip(("nsite", "njnt", "nq", "nv"), (nsite, njnt, nq, nv))),
                 "CPU/HP model dimensions differ")
        cpu_sites, hp_sites = _names(cpu, nsite, "site", False), _names(hp_model, nsite, "site", True)
        cpu_joints, hp_joints = _names(cpu, njnt, "joint", False), _names(hp_model, njnt, "joint", True)
        _require(set(cpu_sites) == set(hp_sites) and set(cpu_joints) == set(hp_joints),
                 "CPU/HP named model coverage differs")
        widths = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}
        # MuJoCo slide/hinge are 2/3; HP hinge/slide are 2/3.
        # This is the frozen diagnostic's explicit CPU->HP enum conversion.
        hp_types = {0: 0, 1: 1, 2: 3, 3: 2}
        cpu_fields = {}
        for field in ("jnt_type", "jnt_qposadr", "jnt_dofadr"):
            values = np.asarray(getattr(cpu, field))
            _require(values.shape == (njnt,) and values.dtype.kind in "iu", f"Invalid CPU {field}")
            cpu_fields[field] = values.copy()
        hp_for_cpu = np.empty(nq, dtype=np.intp)
        covered = {key: [] for key in ("cpu_q", "hp_q", "cpu_v", "hp_v")}
        joint_addrs = {}
        for cid, name in enumerate(cpu_joints):
            hid = hp_joints.index(name)
            kind = _integer(cpu_fields["jnt_type"][cid], "CPU joint type")
            _require(kind in widths and _integer(hp_model.joint_type(hid), "HP joint type") == hp_types[kind],
                     f"Joint type differs: {name}")
            cq = _integer(cpu_fields["jnt_qposadr"][cid], "CPU qpos address")
            cv = _integer(cpu_fields["jnt_dofadr"][cid], "CPU dof address")
            hq = _integer(hp_model.joint_qpos_adr(hid), "HP qpos address")
            hv = _integer(hp_model.joint_dof_adr(hid), "HP dof address")
            wq, wv = widths[kind]
            _require(cq + wq <= nq and hq + wq <= nq and cv + wv <= nv and hv + wv <= nv,
                     f"Joint address outside model: {name}")
            for key, address, width in (("cpu_q", cq, wq), ("hp_q", hq, wq),
                                         ("cpu_v", cv, wv), ("hp_v", hv, wv)):
                covered[key].extend(range(address, address + width))
            hp_for_cpu[cq:cq + wq] = np.arange(hq, hq + wq)
            joint_addrs[name] = cq if wq == 1 else (cq, cq + wq)
        _require(all(sorted(values) == list(range(nq if key.endswith("q") else nv))
                     for key, values in covered.items()), "Incomplete/overlapping joint address coverage")
        rgba = _array(cpu.site_rgba, (nsite, 4), "reference site RGBA")
        qpos = _array(ref.sim.data.qpos, (nq,), "reference qpos")
        _post_process_contract(type(ref))
        visualize = _function(type(ref), "set_visualization",
                              {"visualization_sites_list", "get_object", "object_properties", "items",
                               "sim", "model", "site_name2id", "site_rgba"})
        _require(getattr(ref.set_visualization, "__func__", None) is visualize,
                 "Instance visualization override is unsupported")
        visual_names = tuple(ref.visualization_sites_list)
        # LIBERO appends to this list on each hard reset. The installed callback
        # iterates every occurrence; preserve its exact order and multiplicity.
        # Tracked states and declared site ownership must still be unique below.
        _require(all(isinstance(name, str) and name for name in visual_names),
                 "Invalid visualization object names")
        tracked = tuple(ref.tracking_object_states_change)
        tracked_names = tuple(state.object_name for state in tracked)
        _require(all(isinstance(name, str) and name for name in tracked_names) and
                 len(set(tracked_names)) == len(tracked_names), "Invalid tracked object names")
        _require(set(visual_names) <= set(tracked_names), "Visual objects must be tracked before visualization")
        definitions, fixtures, specs = {}, set(), []
        forbidden = {id(ref), id(ref.sim), id(cpu), id(ref.sim.data), id(hp_model)}
        for state in tracked:
            name = state.object_name
            _require(state.env is ref and ref.object_states_dict[name] is state and
                     state.object_state_type == "object" and type(state.is_fixture) is bool,
                     f"Unsupported tracked state: {name}")
            owner = ref.fixtures_dict if state.is_fixture else ref.objects_dict
            _require(name in owner and ref.get_object(name) is owner[name], f"Ambiguous object: {name}")
            _require(not (name in ref.fixtures_dict and name in ref.objects_dict), f"Duplicate object: {name}")
            obj = owner[name]
            _check_definition(obj, forbidden)
            init = _function(type(state), "__init__", {"env", "object_name", "is_fixture", "query_dict",
                "fixtures_dict", "objects_dict", "object_state_type", "has_turnon_affordance", "hasattr", "get_object"}, {"hasattr"})
            update = _function(type(state), "update_state", {"has_turnon_affordance", "turn_on"})
            turn_on = _function(type(state), "turn_on", {"env", "get_object", "object_name", "joints", "sim", "model",
                "get_joint_qpos_addr", "data", "qpos", "turn_on"})
            object_turn_on = _function(type(obj), "turn_on", {"object_properties", "naming_prefix", "min", "max"}, {"min", "max"})
            _require(all(getattr(getattr(state, method), "__func__", None) is fn
                         for method, fn in (("update_state", update), ("turn_on", turn_on))) and
                     getattr(obj.turn_on, "__func__", None) is object_turn_on,
                     "Instance visual callback override is unsupported")
            joints = tuple(obj.joints)
            _require(joints and len(set(joints)) == len(joints) and
                     all(name in joint_addrs and isinstance(joint_addrs[name], int) for name in joints),
                     "Tracked visual callbacks require named scalar joints")
            definitions[name] = obj
            if state.is_fixture:
                fixtures.add(name)
            specs.append((name, state.is_fixture, type(state), init, update, turn_on, object_turn_on))
        definitions = copy.deepcopy(definitions)  # Only object model definitions, never ref/sim/state wrappers.
        _require(all(definitions[name] is not ref.get_object(name) and
                     definitions[name].object_properties is not ref.get_object(name).object_properties
                     for name in definitions), "Object definitions did not detach")
        allowed_sites = set()
        # Validate each distinct object's declarations once. This is NOT the
        # callback iteration list: _visual_names below retains all occurrences.
        for name in dict.fromkeys(visual_names):
            for item in definitions[name].object_properties["vis_site_names"].values():
                _require(isinstance(item, (tuple, list)) and len(item) == 2 and
                         item[0] in cpu_sites and type(item[1]) is bool, "Invalid visual site declaration")
                _require(item[0] not in allowed_sites, "Visual site declared by multiple objects")
                allowed_sites.add(item[0])
        result = object.__new__(cls)
        result._rgba, result._qpos = rgba, qpos
        result._definitions = definitions
        result._fixture_names, result._visual_names, result._specs = frozenset(fixtures), visual_names, tuple(specs)
        result._visualize = visualize
        result._site_ids = MappingProxyType({name: i for i, name in enumerate(cpu_sites)})
        result._joint_addrs = MappingProxyType(joint_addrs)
        result._cpu_for_hp_sites = np.array([cpu_sites.index(name) for name in hp_sites], dtype=np.intp)
        result._hp_for_cpu_qpos = hp_for_cpu
        result._untouched_sites = np.array([i for i, name in enumerate(cpu_sites) if name not in allowed_sites], dtype=np.intp)
        return result

    @property
    def cpu_rgba(self):
        """Copied CURRENT visual snapshot in reference/CPU site order."""
        return self._rgba.copy()

    @property
    def hp_rgba(self):
        """Copied CURRENT visual snapshot in HP site order, ready to publish."""
        return self._rgba[self._cpu_for_hp_sites].copy()

    @property
    def cpu_qpos(self):
        return self._qpos.copy()

    def advance(self, native_qpos):
        """Advance detached display bookkeeping for the NEXT image.

        The caller must already have captured the current image/snapshot. No
        reference environment, engine, controller, renderer, or scoring object
        is retained or consulted by this operation.

        The detached object definitions are advanced IN PLACE rather
        than deep-copied every round (the copy dragged each object's XML tree
        along and was most of this call). Atomicity is replaced by a
        fail-closed latch: any error leaves this adapter permanently unusable
        (the coordinator already poisons the whole engine on such an error, and
        reset builds a fresh adapter via from_reference). Callback inputs, the
        callbacks themselves, their order and every validation are unchanged,
        so the published RGBA is bit-identical.
        """
        broken = getattr(self, "_broken", None)
        _require(broken is None, "Visual state is unusable after an earlier advance failure")
        try:
            qpos = _array(native_qpos, self._qpos.shape, "native qpos")[self._hp_for_cpu_qpos]
            definitions = self._definitions
            proxy = _VisualProxy(self._rgba.copy(), qpos.copy(), self._site_ids, self._joint_addrs,
                                 definitions, self._fixture_names, self._visual_names)
            for name, fixture, state_type, init, update, turn_on, object_turn_on in self._specs:
                obj = proxy.get_object(name)
                obj.turn_on = MethodType(object_turn_on, obj)
                state = object.__new__(state_type)
                init(state, proxy, name, is_fixture=fixture)
                state.turn_on = MethodType(turn_on, state)
                state.update_state = MethodType(update, state)
                proxy.object_states_dict[name] = state
                proxy.tracking_object_states_change.append(state)
            for state in proxy.tracking_object_states_change:
                state.update_state()
            self._visualize(proxy)  # Unbound; never call reference._post_process().
            rgba = _array(proxy.sim.model.site_rgba, self._rgba.shape, "callback site RGBA")
            _require(np.array_equal(proxy.sim.data.qpos, qpos), "Visual callback changed qpos")
            _require(np.array_equal(rgba[:, :3], self._rgba[:, :3]) and
                     np.array_equal(rgba[self._untouched_sites], self._rgba[self._untouched_sites]),
                     "Visual callback changed RGB or an undeclared site")
            # Bound methods are rebuilt next time; keep definitions free of bound self cycles.
            for obj in definitions.values():
                del obj.turn_on
            result = rgba[self._cpu_for_hp_sites]
            self._rgba, self._qpos = rgba, qpos
            return result
        except BaseException as error:
            self._broken = error
            raise


def from_reference(reference_domain_env, hp_model):
    return SiteVisualState.from_reference(reference_domain_env, hp_model)
