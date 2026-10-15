# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Operational Space Controller (OSC_POSE) for the GPU simulator.

Bridges the robosuite OSC_POSE controller to GPU physics by reading
qpos/qvel/xpos/xquat from the GPU, computing joint torques on CPU
using NumPy, and writing ctrl back.

The control law:
  F = kp_pos * pos_error + kd_pos * vel_error  (3D force)
  tau = kp_ori * ori_error + kd_ori * angvel_error  (3D torque)
  wrench = [Lambda_pos * F; Lambda_ori * tau]
  joint_torques = J^T * wrench + qfrc_bias + nullspace_torques
"""

import numpy as np
from typing import Optional, Tuple


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]
    ])


def orientation_error(desired_mat: np.ndarray, current_mat: np.ndarray) -> np.ndarray:
    """Robosuite-compatible orientation error between two rotation matrices."""
    rc1 = current_mat[0:3, 0]
    rc2 = current_mat[0:3, 1]
    rc3 = current_mat[0:3, 2]
    rd1 = desired_mat[0:3, 0]
    rd2 = desired_mat[0:3, 1]
    rd3 = desired_mat[0:3, 2]
    return 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))


class _JointInfo:
    """Lightweight container for joint properties read from Python API."""
    __slots__ = ('type', 'body', 'qpos_adr', 'dof_adr', 'axis', 'pos')


class _BodyInfo:
    """Lightweight container for body properties read from Python API."""
    __slots__ = ('parent', 'mass', 'inertia', 'ipos', 'iquat')


class OSCController:
    """
    Operational Space Controller matching robosuite's OSC_POSE.

    Works with the GPU simulator: reads state from GPU, computes torques on CPU.
    Accepts instinct_sim.PhysicsModel and instinct_sim.PhysicsEngine Python objects.
    """

    def __init__(
        self,
        model,
        engine,
        arm_joint_indices: list,
        gripper_actuator_indices: list,
        eef_body_idx: int,
        eef_site_idx: Optional[int] = None,
        arm_actuator_indices: Optional[list] = None,
        kp_pos: float = 150.0,
        kp_ori: float = 150.0,
        damping_ratio: float = 1.0,
        kp_null: float = 10.0,
        control_freq: int = 20,
        action_scale_pos: float = 0.05,
        action_scale_ori: float = 0.5,
        arm_ctrl_limits: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.engine = engine
        self.arm_joint_indices = list(arm_joint_indices)
        self.gripper_actuator_indices = list(gripper_actuator_indices)
        # Per-actuator ctrlrange for the gripper fingers, set by hip_env
        # (_parse_gripper_ctrlrange). None => fall back to the raw action.
        self.gripper_ctrlrange = None
        self.eef_body_idx = eef_body_idx
        self.eef_site_idx = eef_site_idx if eef_site_idx is not None and eef_site_idx >= 0 else None
        self.n_arm = len(arm_joint_indices)

        if arm_actuator_indices is None:
            self.arm_actuator_indices = list(range(len(arm_joint_indices)))
        else:
            self.arm_actuator_indices = list(arm_actuator_indices)

        self._joints = []
        self.arm_dof_indices = []
        self.arm_qpos_indices = []
        for ji in arm_joint_indices:
            j = _JointInfo()
            j.type = model.joint_type(ji)
            j.body = model.joint_body(ji)
            j.qpos_adr = model.joint_qpos_adr(ji)
            j.dof_adr = model.joint_dof_adr(ji)
            j.axis = np.array(model.joint_axis(ji))
            j.pos = np.array(model.joint_pos(ji))
            self._joints.append(j)
            self.arm_dof_indices.append(j.dof_adr)
            self.arm_qpos_indices.append(j.qpos_adr)

        self._bodies = []
        _has_iquat = hasattr(model, 'body_iquat')
        for bi in range(model.nbody):
            b = _BodyInfo()
            b.parent = model.body_parent(bi)
            b.mass = model.body_mass(bi)
            b.inertia = np.array(model.body_inertia(bi))
            b.ipos = np.array(model.body_ipos(bi))
            b.iquat = np.array(model.body_iquat(bi)) if _has_iquat else np.array([1.0, 0.0, 0.0, 0.0])
            self._bodies.append(b)

        self.kp_pos = kp_pos
        self.kp_ori = kp_ori
        self.kd_pos = 2.0 * np.sqrt(kp_pos) * damping_ratio
        self.kd_ori = 2.0 * np.sqrt(kp_ori) * damping_ratio
        self.kp_null = kp_null
        self.kd_null = 2.0 * np.sqrt(kp_null)
        self.action_scale_pos = action_scale_pos
        self.action_scale_ori = action_scale_ori

        grav = model.gravity
        self.gravity = np.array([grav[0], grav[1], grav[2]])

        if arm_ctrl_limits is not None:
            self.arm_ctrl_limits = np.asarray(arm_ctrl_limits, dtype=np.float64)
        else:
            derived = None
            try:
                rng = getattr(model, "actuator_ctrlrange", None)
                if rng is not None:
                    derived = np.array(
                        [max(abs(float(rng[ai][0])), abs(float(rng[ai][1])))
                         for ai in self.arm_actuator_indices], dtype=np.float64)
            except Exception:
                derived = None
            self.arm_ctrl_limits = (derived if derived is not None and len(derived)
                                    else np.full(len(arm_joint_indices), 80.0))

        self.goal_pos = None
        self.goal_ori = None
        self.initial_joint_pos = None

    def reset(self, env_idx: int = 0):
        """Reset controller state. Call after env.reset()."""
        self.engine.sync()
        data = self.engine.data
        qpos = data.read_qpos()[env_idx]
        xpos = data.read_xpos()[env_idx]
        xquat = data.read_xquat()[env_idx]
        site_xpos = data.read_site_xpos()[env_idx] if self.eef_site_idx is not None else None
        site_xmat = data.read_site_xmat()[env_idx] if self.eef_site_idx is not None else None

        if self.eef_site_idx is not None and site_xpos is not None and site_xmat is not None:
            self.goal_pos = site_xpos[self.eef_site_idx].copy()
            self.goal_ori = site_xmat[self.eef_site_idx].reshape(3, 3).copy()
        else:
            self.goal_pos = xpos[self.eef_body_idx].copy()
            self.goal_ori = quat_to_mat(xquat[self.eef_body_idx])
        self.initial_joint_pos = np.array([qpos[qi] for qi in self.arm_qpos_indices])

    def _compute_jacobian(self, xpos: np.ndarray, xquat: np.ndarray,
                          eef_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 3xN position and rotation Jacobians for arm joints."""
        J_pos = np.zeros((3, self.n_arm))
        J_ori = np.zeros((3, self.n_arm))

        for i, ji in enumerate(self.arm_joint_indices):
            j = self._joints[i]
            jbody = j.body

            # MuJoCo joint axes live in the child body frame, not the parent's.
            R_body = quat_to_mat(xquat[jbody])
            world_axis = R_body @ j.axis

            if j.type == 2:  # JOINT_HINGE
                pivot = xpos[jbody] + R_body @ j.pos
                r = eef_pos - pivot
                J_pos[:, i] = np.cross(world_axis, r)
                J_ori[:, i] = world_axis
            elif j.type == 3:  # JOINT_SLIDE
                J_pos[:, i] = world_axis
                J_ori[:, i] = 0.0

        return J_pos, J_ori

    def _compute_mass_matrix(self, xpos: np.ndarray, xquat: np.ndarray) -> np.ndarray:
        """Compute NxN mass matrix for arm joints using CRBA + armature."""
        N = self.n_arm
        M = np.zeros((N, N))

        for bid in range(1, self.model.nbody):
            b = self._bodies[bid]
            m = b.mass
            if m < 1e-12:
                continue

            R_body = quat_to_mat(xquat[bid])
            com = xpos[bid] + R_body @ b.ipos

            R_inertial = quat_to_mat(b.iquat)
            R_world_inertial = R_body @ R_inertial
            I_local = np.diag(b.inertia)
            I_world = R_world_inertial @ I_local @ R_world_inertial.T

            ancestors = []
            cur = bid
            while cur > 0:
                for idx, ji in enumerate(self.arm_joint_indices):
                    if self._joints[idx].body == cur:
                        ancestors.append(idx)
                cur = self._bodies[cur].parent

            for ci, i in enumerate(ancestors):
                j = self._joints[i]
                R_i = quat_to_mat(xquat[j.body])
                axis_i = R_i @ j.axis
                pivot_i = xpos[j.body] + R_i @ j.pos
                r_i = com - pivot_i

                for cj_off in range(ci, len(ancestors)):
                    jj_idx = ancestors[cj_off]
                    jd = self._joints[jj_idx]
                    R_j = quat_to_mat(xquat[jd.body])
                    axis_j = R_j @ jd.axis
                    pivot_j = xpos[jd.body] + R_j @ jd.pos
                    r_j = com - pivot_j

                    val = axis_i @ I_world @ axis_j
                    val += m * np.dot(np.cross(axis_i, r_i), np.cross(axis_j, r_j))
                    M[i, jj_idx] += val
                    if i != jj_idx:
                        M[jj_idx, i] += val

        for i, ji in enumerate(self.arm_joint_indices):
            M[i, i] += self.model.joint_armature(ji) + 1e-8
        return M

    def control(self, action: np.ndarray, env_idx: int = 0,
                policy_step: bool = True) -> np.ndarray:
        """
        Apply an OSC_POSE action: [dx,dy,dz,dax,day,daz,gripper].

        Reads state from GPU, computes torques, writes ctrl back.
        When policy_step=True (default), updates the goal from the action.
        When policy_step=False (substep), recomputes torques for current
        state without changing the goal.
        """
        self.engine.sync()
        data = self.engine.data

        qpos_all = data.read_qpos()[env_idx]
        qvel_all = data.read_qvel()[env_idx]
        xpos = data.read_xpos()[env_idx]
        xquat = data.read_xquat()[env_idx]
        qfrc_bias_all = data.read_qfrc_bias()[env_idx] if hasattr(data, "read_qfrc_bias") else None
        site_xpos = data.read_site_xpos()[env_idx] if self.eef_site_idx is not None else None
        site_xmat = data.read_site_xmat()[env_idx] if self.eef_site_idx is not None else None

        arm_qpos = np.array([qpos_all[qi] for qi in self.arm_qpos_indices])
        arm_qvel = np.array([qvel_all[di] for di in self.arm_dof_indices])

        if self.eef_site_idx is not None and site_xpos is not None and site_xmat is not None:
            eef_pos = site_xpos[self.eef_site_idx].copy()
            eef_mat = site_xmat[self.eef_site_idx].reshape(3, 3).copy()
        else:
            eef_pos = xpos[self.eef_body_idx].copy()
            eef_mat = quat_to_mat(xquat[self.eef_body_idx])

        gripper_action = action[6] if len(action) > 6 else 0.0

        if self.goal_pos is None:
            self.reset(env_idx)

        if policy_step:
            pos_delta = action[:3] * self.action_scale_pos
            ori_delta = action[3:6] * self.action_scale_ori
            self.goal_pos = eef_pos + pos_delta

            angle = np.linalg.norm(ori_delta)
            if angle > 1e-6:
                axis = ori_delta / angle
                K = np.array([[0, -axis[2], axis[1]],
                              [axis[2], 0, -axis[0]],
                              [-axis[1], axis[0], 0]])
                dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
                self.goal_ori = dR @ eef_mat

        pos_error = self.goal_pos - eef_pos
        ori_error = orientation_error(self.goal_ori, eef_mat)

        J_pos, J_ori = self._compute_jacobian(xpos, xquat, eef_pos)
        J_full = np.vstack([J_pos, J_ori])

        vel_pos = J_pos @ arm_qvel
        vel_ori = J_ori @ arm_qvel

        F_desired = self.kp_pos * pos_error - self.kd_pos * vel_pos
        tau_desired = self.kp_ori * ori_error - self.kd_ori * vel_ori

        M = self._compute_mass_matrix(xpos, xquat)
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

        if qfrc_bias_all is not None:
            arm_bias = np.array([qfrc_bias_all[di] for di in self.arm_dof_indices])
            torques += arm_bias

        Lambda_full = np.linalg.pinv(JMJ_full)
        J_bar = M_inv @ J_full.T @ Lambda_full
        N = np.eye(self.n_arm) - J_bar @ J_full
        if self.initial_joint_pos is not None:
            null_torques = N.T @ M @ (
                self.kp_null * (self.initial_joint_pos - arm_qpos) -
                self.kd_null * arm_qvel
            )
            torques += null_torques

        nu = self.model.nu
        ctrl = np.zeros(nu, dtype=np.float32)

        for i, ai in enumerate(self.arm_actuator_indices):
            if ai < nu:
                lim = self.arm_ctrl_limits[i] if i < len(self.arm_ctrl_limits) else 87.0
                ctrl[ai] = np.clip(torques[i], -lim, lim)

        for k, gi in enumerate(self.gripper_actuator_indices):
            if gi >= nu:
                continue
            rng = None
            if self.gripper_ctrlrange is not None and k < len(self.gripper_ctrlrange):
                rng = self.gripper_ctrlrange[k]
            if rng is None:
                ctrl[gi] = gripper_action
                continue
            lo, hi = float(rng[0]), float(rng[1])
            # gripper_action: -1 = open, +1 = close. The finger whose range is
            # positive closes toward hi; the mirrored finger closes toward lo.
            a = float(np.clip(gripper_action, -1.0, 1.0))
            if abs(hi) >= abs(lo):
                ctrl[gi] = np.clip(0.5 * (a + 1.0) * (hi - lo) + lo, min(lo, hi), max(lo, hi))
            else:
                ctrl[gi] = np.clip(0.5 * (1.0 - a) * (hi - lo) + lo, min(lo, hi), max(lo, hi))

        data.write_ctrl_env(ctrl, env_idx)
        return ctrl
