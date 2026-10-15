# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One GPU call per control step for robosuite's GR1 envs (HP_FAST_STEP=1, backend.attach).

robosuite's MujocoEnv.step runs, per substep: step1 (kinematics) -> _pre_action (the composite controller
writes ctrl from the current state) -> step2 -> _update_observables. Observables (cameras included) sample on
the first substep of a control step; the other substeps only advance their timers. The GR1 arm / torso parts
are JointPositionControllers (tau = M_jj (kp (q* - q) - kd qd) + qfrc_bias, clipped), the hands
SimpleGripControllers with a ctrl that is constant over the control step.

fast_step reproduces that with:
  substep 0      robosuite's own path (policy action, controller goals, observables sampled) on the GPU simulator;
  substeps 1..N-2 GPU step_n with the GPU joint PD (joint_pd.h) on the controllers' goals;
  substep N-1    stepped separately so mjData ends exactly as after robosuite's loop: kinematics (xpos, ...)
                 at the state before the last substep, qpos / qvel / time after it;
and the remaining _update_observables calls so every observable timer advances identically.
"""
import numpy as np


def pd_config(rs):
    """GPU joint PD config from robosuite's composite controller (JointPositionController parts)."""
    import mujoco
    from robosuite.controllers.parts.generic.joint_pos import JointPositionController
    m = rs.sim.model._model
    robot = rs.robots[0]
    parts = robot.composite_controller.part_controllers
    act, dof, qadr, gs, gl, kp, kd, gear, lo, hi, owners = [], [], [], [], [], [], [], [], [], [], []
    for name, c in parts.items():
        if not isinstance(c, JointPositionController):
            continue
        if c.interpolator is not None:
            raise NotImplementedError(f"{name}: interpolated JointPositionController not supported")
        acts = list(robot._ref_actuators_indexes_dict[name])
        dofs = list(c.qvel_index)
        if len(acts) != len(dofs):
            raise ValueError(f"{name}: {len(acts)} actuators vs {len(dofs)} dofs")
        start = len(act)
        for k, (a, v) in enumerate(zip(acts, dofs)):
            act.append(a); dof.append(v); qadr.append(int(c.qpos_index[k]))
            gs.append(start); gl.append(len(acts))
            kp.append(float(np.broadcast_to(c.kp, (len(acts),))[k]))
            kd.append(float(np.broadcast_to(c.kd, (len(acts),))[k]))
            gear.append(float(m.actuator_gear[a, 0]))
            lo.append(float(m.actuator_ctrlrange[a, 0])); hi.append(float(m.actuator_ctrlrange[a, 1]))
        owners.append((name, c, start, len(acts)))
    cfg = {k: np.array(v) for k, v in dict(act=act, dof=dof, qadr=qadr, grp_start=gs, grp_len=gl,
                                           kp=kp, kd=kd, gear=gear, lo=lo, hi=hi).items()}
    return cfg, owners


def install(rs, be):
    """Replace rs.step (robosuite MujocoEnv) with the fast path; be is the HipBackend."""
    orig_step = rs.step
    state = {"eng": None}

    def targets(owners, n):
        t = np.zeros(n)
        for name, c, start, cnt in owners:
            t[start:start + cnt] = np.asarray(c.goal_qpos, dtype=np.float64)
        return t

    def step(action):
        if rs.done:
            raise ValueError("executing action in terminated episode")
        nsub = int(round(rs.control_timestep / rs.model_timestep))
        if nsub < 3:
            return orig_step(action)
        if state["eng"] is not be.eng:   # new engine (new scene): configure its PD before any step_n
            cfg, owners = pd_config(rs)
            be.eng.set_joint_pd(cfg)
            # capture_step_graph runs one real warm-up step() before capturing: trigger it now, before the
            # state is pushed (every advance pushes the full state), so no substep is ever doubled
            be.eng.step_n(1)
            be.eng.sync()
            state.update(eng=be.eng, owners=owners, n=len(cfg["act"]))
        rs.timestep += 1
        sim = rs.sim
        d = sim.data._data
        # substep 0: robosuite's own path
        if rs.lite_physics:
            sim.step1()
        else:
            sim.forward()
        rs._pre_action(action, True)
        # the GPU PD also drives substep 0's PD actuators (same law as the ctrl robosuite just wrote): give it
        # this control step's goals before stepping
        be.eng.set_joint_pd_targets(targets(state["owners"], state["n"]))
        if rs.lite_physics:
            sim.step2()
        else:
            sim.step()
        rs._update_observables()
        # substeps 1 .. nsub-2 on the GPU (ctrl of the PD actuators from the GPU joint PD, the rest as set)
        be.advance_n(nsub - 2)
        # substep nsub-1: mjData kinematics at its pre-step state, qpos / qvel after it (as mj_step leaves)
        be.advance_last_kinematic()
        for _ in range(nsub - 1):
            rs._update_observables()
        rs.cur_time += rs.control_timestep
        reward, done, info = rs._post_action(action)
        if rs.viewer is not None and rs.renderer != "mujoco":
            rs.viewer.update()
        observations = rs.viewer._get_observations() if rs.viewer_get_obs else rs._get_observations()
        return observations, reward, done, info

    rs.step = step
