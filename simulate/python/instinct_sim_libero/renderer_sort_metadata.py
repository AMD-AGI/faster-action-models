# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate indexed compiled geom metadata before installing it once; stdlib-only."""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def install_sort_metadata(renderer, native_model, compiled_model, compiled_geom_names):
    """No name lookup/remapping: unnamed geoms remain indexed empty strings.

    Caller passes the already compiled setup model after final renderer.init.
    Only the renderer metadata setter may mutate state; all mapping/payload
    validation finishes first. No physics/model writes, forwards, or rendering.
    """
    setter = getattr(renderer, 'set_transparency_sort_metadata', None)
    require(callable(setter), 'Candidate renderer lacks required transparency metadata setter')
    count = int(compiled_model.ngeom)
    require(type(native_model.ngeom) is int and native_model.ngeom == count and count > 0,
            'Compiled/native geom count differs')
    require(type(native_model.nbody) is int and native_model.nbody == int(compiled_model.nbody),
            'Compiled/native body count differs')
    require(type(compiled_geom_names) in (list, tuple) and len(compiled_geom_names) == count,
            'Need every indexed compiled geom name, including unnamed')
    require(callable(getattr(native_model, 'geom_name', None))
            and callable(getattr(native_model, 'geom_body_id', None)), 'Missing native geom mapping API')
    positions, radii, bodies = compiled_model.geom_pos, compiled_model.geom_rbound, compiled_model.geom_bodyid
    require(tuple(positions.shape) == (count, 3) and tuple(radii.shape) == (count,)
            and tuple(bodies.shape) == (count,), 'Malformed compiled geom metadata shapes')
    for i, raw_name in enumerate(compiled_geom_names):
        require(raw_name is None or type(raw_name) is str, 'Invalid compiled geom name')
        name = '' if raw_name is None else raw_name
        require(native_model.geom_name(i) == name, 'Compiled/native geom name/order differs at ' + str(i))
        body = int(bodies[i])
        native_body = native_model.geom_body_id(i)
        require(type(native_body) is int and 0 <= body < native_model.nbody and native_body == body,
                'Compiled/native geom body index differs at ' + str(i))
        require(all(math.isfinite(float(value)) for value in positions[i])
                and math.isfinite(float(radii[i])) and float(radii[i]) >= 0,
                'Invalid compiled sort metadata at ' + str(i))
    # Copy preserves original model storage. The required native API performs
    # its documented float conversion; no physics descriptor is modified.
    setter(positions.copy(), radii.copy())
