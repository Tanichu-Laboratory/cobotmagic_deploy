"""Piper URDF kinematics and numerical IK, usable without ROS or robot I/O."""
import logging
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation


def rpy_xyz_to_matrix(rpy):
    roll, pitch, yaw = [float(v) for v in rpy]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = np.cos(float(angle))
    s = np.sin(float(angle))
    one_c = 1.0 - c
    return np.asarray([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def transform_from_xyz_rpy(xyz, rpy):
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_xyz_to_matrix(rpy)
    t[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return t


def transform_from_pose_command(cmd):
    cmd = np.asarray(cmd, dtype=np.float64)
    return transform_from_xyz_rpy(cmd[:3], cmd[3:6])


def rotation_error_vector(target_rot, current_rot):
    # scipy handles both infinitesimal and pi rotations without a singular log map.
    return Rotation.from_matrix(target_rot @ current_rot.T).as_rotvec()


def parse_float_triplet(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


class PiperNumericalIK:
    CONDITION_REL_TOL = 1e-5  # Same float32 tolerance for solving and holding.

    def __init__(self, cfg, loginfo=None):
        self.loginfo = loginfo or logging.getLogger(__name__).info
        self.urdf_path = os.path.expanduser(str(cfg.get(
            'urdf_path',
            '/workspace/project/X-VLA/evaluation/SoftFold-Agilex/Piper_ros_private-ros-noetic/src/piper_description/urdf/piper_description.urdf',
        )))
        self.base_link = str(cfg.get('base_link', 'base_link'))
        self.tip_link = str(cfg.get('tip_link', 'link6'))
        self.max_iters = max(int(cfg.get('max_iters', 80)), 1)
        self.tolerance = max(float(cfg.get('tolerance', 1e-4)), 1e-8)
        self.damping = max(float(cfg.get('damping', 0.03)), 1e-8)
        self.fd_eps = max(float(cfg.get('finite_difference_eps', 1e-4)), 1e-7)
        self.max_step = max(float(cfg.get('max_step_rad', 0.18)), 1e-4)
        self.position_weight = max(float(cfg.get('position_weight', 1.0)), 0.0)
        self.orientation_weight = max(float(cfg.get('orientation_weight', 0.25)), 0.0)
        self.max_position_error_m = max(float(cfg.get('max_position_error_m', 0.04)), 0.0)
        self.max_orientation_error_rad = max(float(cfg.get('max_orientation_error_rad', 0.8)), 0.0)
        self.joint_regularization_weight = max(float(cfg.get('joint_regularization_weight', 0.08)), 0.0)
        max_joint_delta = cfg.get('max_joint_delta_rad', [0.12, 0.12, 0.12, 0.16, 0.16, 0.16])
        if isinstance(max_joint_delta, (int, float)):
            max_joint_delta = [float(max_joint_delta)] * 6
        self.max_joint_delta = np.asarray(max_joint_delta, dtype=np.float64)
        if self.max_joint_delta.shape != (6,) or not np.all(np.isfinite(self.max_joint_delta)):
            raise ValueError(f"eef_ik.max_joint_delta_rad must be scalar or 6D, got {self.max_joint_delta.shape}")
        self.max_joint_delta = np.maximum(self.max_joint_delta, 0.0)
        self.clip_to_max_joint_delta = bool(cfg.get('clip_to_max_joint_delta', True))
        self.publish_on_failure = bool(cfg.get('publish_on_failure', False))
        self.solver = str(cfg.get('solver', 'damped_least_squares'))
        if self.solver not in ('damped_least_squares', 'least_squares', 'differential'):
            raise ValueError("eef_ik.solver must be differential, damped_least_squares or least_squares")
        if self.solver == 'least_squares' and self.joint_regularization_weight != 0.0:
            raise ValueError("least_squares requires joint_regularization_weight=0: pose accuracy is the objective")
        self.solution_max_position_error_m = float(cfg.get('solution_max_position_error_m', self.max_position_error_m))
        self.solution_max_orientation_error_rad = float(cfg.get('solution_max_orientation_error_rad', self.max_orientation_error_rad))
        self.calibration_mode = str(cfg.get('calibration_mode', 'base'))
        if self.calibration_mode not in ('base', 'tool'):
            raise ValueError("eef_ik.calibration_mode must be base or tool")
        self.joint_signs = np.asarray(cfg.get('joint_signs', [1] * 6), dtype=np.float64)
        if self.joint_signs.shape != (6,) or not np.all(np.isin(self.joint_signs, [-1, 1])):
            raise ValueError("eef_ik.joint_signs must contain six values of +1 or -1")
        guard = cfg.get('wrist_singularity_avoidance', {})
        self.wrist_guard_enabled = bool(guard.get('enabled', False))
        self.wrist_guard = {
            'max_condition': float(guard.get('max_condition', 40.0)),
            'trigger_condition': float(guard.get('trigger_condition', 35.0)),
            'max_command_delta_rad': float(guard.get('max_command_delta_rad', 0.10)),
            'trigger_rad': float(guard.get('trigger_rad', np.deg2rad(15))),
            'min_bend_rad': float(guard.get('min_bend_rad', np.deg2rad(6))),
            'preferred_bend_rad': float(guard.get('preferred_bend_rad', np.deg2rad(12))),
            'max_position_error_m': float(guard.get('max_position_error_m', 0.003)),
            'max_orientation_error_rad': float(guard.get('max_orientation_error_rad', np.deg2rad(8))),
            'max_wrist_command_delta_rad': float(guard.get('max_wrist_command_delta_rad', 0.10)),
        }
        if self.wrist_guard_enabled:
            g = self.wrist_guard
            if not all(np.isfinite(v) and v > 0 for v in g.values()):
                raise ValueError('wrist singularity avoidance values must be finite and positive')
            if not 1 < g['trigger_condition'] < g['max_condition']:
                raise ValueError('require 1 < trigger_condition < max_condition')
            if not g['min_bend_rad'] <= g['preferred_bend_rad'] < g['trigger_rad'] < np.pi / 2:
                raise ValueError('require min_bend <= preferred_bend < trigger < pi/2')
            if self.solver != 'least_squares' or not self.clip_to_max_joint_delta or self.publish_on_failure:
                raise ValueError('wrist avoidance requires bounded least_squares and publish_on_failure=false')
        self.calibration = {}
        self.chain = self._load_chain()
        self.joint_indices = [idx for idx, joint in enumerate(self.chain) if joint['type'] in ('revolute', 'continuous')]
        if len(self.joint_indices) != 6:
            raise ValueError(
                f"Piper IK expects 6 revolute joints from {self.base_link} to {self.tip_link}, "
                f"got {len(self.joint_indices)} from {self.urdf_path}"
            )
        self.lower = np.asarray([self.chain[idx]['lower'] for idx in self.joint_indices], dtype=np.float64)
        self.upper = np.asarray([self.chain[idx]['upper'] for idx in self.joint_indices], dtype=np.float64)
        # Optimization, clipping and output all use ROS joint coordinates.
        # Map asymmetric URDF limits along with the sign of each axis.
        bounds_a = self.lower * self.joint_signs
        bounds_b = self.upper * self.joint_signs
        self.lower = np.minimum(bounds_a, bounds_b)
        self.upper = np.maximum(bounds_a, bounds_b)
        self.differential = None
        if self.solver == 'differential':
            from .piper_differential_ik import DifferentialIK
            self.differential = DifferentialIK(self, cfg.get('differential', {}))

    def _load_chain(self):
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"EEF IK URDF not found: {self.urdf_path}")
        root = ET.parse(self.urdf_path).getroot()
        joints_by_parent = {}
        for joint_elem in root.findall('joint'):
            origin = joint_elem.find('origin')
            axis = joint_elem.find('axis')
            limit = joint_elem.find('limit')
            parent_elem = joint_elem.find('parent')
            child_elem = joint_elem.find('child')
            if parent_elem is None or child_elem is None:
                continue
            joint = {
                'name': joint_elem.get('name', ''),
                'type': joint_elem.get('type', 'fixed'),
                'parent': parent_elem.get('link'),
                'child': child_elem.get('link'),
                'origin_xyz': parse_float_triplet(None if origin is None else origin.get('xyz'), [0.0, 0.0, 0.0]),
                'origin_rpy': parse_float_triplet(None if origin is None else origin.get('rpy'), [0.0, 0.0, 0.0]),
                'axis': parse_float_triplet(None if axis is None else axis.get('xyz'), [0.0, 0.0, 1.0]),
                'lower': -np.pi,
                'upper': np.pi,
            }
            if limit is not None and joint['type'] != 'continuous':
                joint['lower'] = float(limit.get('lower', joint['lower']))
                joint['upper'] = float(limit.get('upper', joint['upper']))
            joint['origin_transform'] = transform_from_xyz_rpy(joint['origin_xyz'], joint['origin_rpy'])
            joints_by_parent.setdefault(joint['parent'], []).append(joint)

        visited = set()

        def dfs(link, path):
            if link == self.tip_link:
                return path
            if link in visited:
                return None
            visited.add(link)
            for joint in joints_by_parent.get(link, []):
                result = dfs(joint['child'], path + [joint])
                if result is not None:
                    return result
            return None

        chain = dfs(self.base_link, [])
        if chain is None:
            raise ValueError(f"No URDF chain found from {self.base_link} to {self.tip_link} in {self.urdf_path}")
        return chain

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)[:6] * self.joint_signs
        t = np.eye(4, dtype=np.float64)
        q_idx = 0
        for joint in self.chain:
            t = t @ joint['origin_transform']
            if joint['type'] in ('revolute', 'continuous'):
                rot = np.eye(4, dtype=np.float64)
                rot[:3, :3] = axis_angle_to_matrix(joint['axis'], q[q_idx])
                t = t @ rot
                q_idx += 1
        return t

    def calibrate(self, side, joints, current_eef_cmd):
        joints = np.asarray(joints, dtype=np.float64)
        current_eef = np.asarray(current_eef_cmd, dtype=np.float64)
        if joints.shape[0] < 6 or current_eef.shape[0] < 6:
            return False
        model_fk = self.fk(joints[:6])
        measured = transform_from_pose_command(current_eef)
        if self.calibration_mode == 'tool':
            # ROS EEF and URDF share the arm base. The fixed frame mismatch is
            # attached to the moving tip, so it belongs on the RIGHT of FK.
            self.calibration[side] = np.linalg.inv(model_fk) @ measured
        else:
            self.calibration[side] = measured @ np.linalg.inv(model_fk)
        if self.differential is not None:
            self.differential.reset(side, joints)
        self.loginfo(
            f"EEF IK calibrated for {side}: "
            f"measured_xyz={current_eef[:3].tolist()} seed_joints={joints[:6].tolist()} "
            f"mode={self.calibration_mode} joint_signs={self.joint_signs.tolist()} "
            f"transform={self.calibration[side].tolist()}"
        )
        return True

    def _model_target(self, side, target_eef_cmd):
        target = transform_from_pose_command(target_eef_cmd)
        calib = self.calibration.get(side)
        if calib is None or self.calibration_mode == 'tool':
            return target
        return np.linalg.inv(calib) @ target

    def eef_fk(self, side, q):
        """FK in the measured EEF frame, with ROS joint coordinates as input."""
        correction = self.calibration.get(side, np.eye(4))
        if self.calibration_mode == 'tool':
            return self.fk(q) @ correction
        return correction @ self.fk(q)

    def _error(self, target, q, tool=None):
        current = self.fk(q)
        if tool is not None:
            current = current @ tool
        pos_error = target[:3, 3] - current[:3, 3]
        rot_error = rotation_error_vector(target[:3, :3], current[:3, :3])
        weighted = np.concatenate([
            pos_error * self.position_weight,
            rot_error * self.orientation_weight,
        ])
        return weighted, pos_error, rot_error

    def _jacobian(self, target, q, base_error, tool=None):
        jac = np.zeros((6, 6), dtype=np.float64)
        for idx in range(6):
            q_step = q.copy()
            q_step[idx] += self.fd_eps
            q_step = np.clip(q_step, self.lower, self.upper)
            denom = q_step[idx] - q[idx]
            if abs(denom) < 1e-12:
                q_step[idx] -= self.fd_eps
                q_step = np.clip(q_step, self.lower, self.upper)
                denom = q_step[idx] - q[idx]
            if abs(denom) < 1e-12:
                continue
            step_error, _, _ = self._error(target, q_step, tool)
            jac[:, idx] = (step_error - base_error) / denom
        return jac

    @staticmethod
    def _within_errors(pos_norm, rot_norm, max_pos, max_rot):
        return bool(np.isfinite(pos_norm) and np.isfinite(rot_norm)
                    and (max_pos <= 0.0 or pos_norm <= max_pos)
                    and (max_rot <= 0.0 or rot_norm <= max_rot))

    def _least_squares(self, target, initial, lower, upper, tool=None):
        # Zero delta limits lock a joint. scipy requires strict bounds, so solve
        # only for the remaining coordinates instead of relaxing locked limits.
        free = upper - lower > 1e-12
        fixed = np.clip(initial, lower, upper)
        if not np.any(free):
            return fixed, 0

        def expand(x):
            q = fixed.copy()
            q[free] = x
            return q

        result = least_squares(
            lambda x: self._error(target, expand(x), tool)[0], fixed[free],
            bounds=(lower[free], upper[free]), method='trf',
            max_nfev=self.max_iters, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        return expand(result.x), int(result.nfev)

    def _damped_least_squares(self, target, seed_q, tool=None):
        # Retain the legacy solver for deployments which have not opted in.
        q = seed_q.copy()
        for iteration in range(self.max_iters):
            error, _, _ = self._error(target, q, tool)
            if np.linalg.norm(error) <= self.tolerance:
                break
            jac = self._jacobian(target, q, error, tool)
            if self.joint_regularization_weight > 0.0:
                reg_jac = self.joint_regularization_weight * np.eye(6, dtype=np.float64)
                reg_error = self.joint_regularization_weight * (q - seed_q)
                jac = np.vstack([jac, reg_jac])
                error = np.concatenate([error, reg_error])
            lhs = jac.T @ jac + (self.damping ** 2) * np.eye(6, dtype=np.float64)
            dq = -np.linalg.solve(lhs, jac.T @ error)
            step_norm = np.linalg.norm(dq)
            if step_norm > self.max_step:
                dq *= self.max_step / step_norm
            q = np.clip(q + dq, self.lower, self.upper)
        return q, iteration + 1

    def geometric_jacobian(self, side, q):
        """Whole-arm geometric Jacobian, metres + 0.25 * radians.

        Includes the calibrated tool offset; base rotations do not affect SVD.
        """
        t = np.eye(4)
        origins, axes = [], []
        for joint in self.chain:
            t = t @ joint['origin_transform']
            if joint['type'] in ('revolute', 'continuous'):
                j = len(axes)
                axis = joint['axis'] / np.linalg.norm(joint['axis'])
                origins.append(t[:3, 3].copy())
                axes.append(t[:3, :3] @ axis * self.joint_signs[j])
                rotation = np.eye(4)
                rotation[:3, :3] = axis_angle_to_matrix(axis, q[j] * self.joint_signs[j])
                t = t @ rotation
        if self.calibration_mode == 'tool':
            t = t @ self.calibration.get(side, np.eye(4))
        jac = np.array([np.r_[np.cross(axis, t[:3, 3] - origin), axis]
                        for axis, origin in zip(axes, origins)]).T
        return jac

    def condition_number(self, side, q):
        jac = self.geometric_jacobian(side, q)
        jac[3:] *= 0.25
        sv = np.linalg.svd(jac, compute_uv=False)
        return float(sv[0] / max(sv[-1], 1e-12))

    def _avoid_wrist_singularity(self, side, target, seed, reference, initial, tool):
        """Piper wrist q[4]≈0 aligns the q[3]/q[5] axes.

        Keep the measured wrist branch, hard position/orientation tolerances,
        and both measured-step and previous-command bounds. No solver state is
        committed here: reference is supplied by the publisher.
        """
        g = self.wrist_guard
        sign = 1.0 if seed[4] >= 0 else -1.0
        lower = np.maximum(self.lower, seed - self.max_joint_delta)
        upper = np.minimum(self.upper, seed + self.max_joint_delta)
        lower = np.maximum(lower, reference - g['max_command_delta_rad'])
        upper = np.minimum(upper, reference + g['max_command_delta_rad'])
        for j in (3, 5):
            lower[j] = max(lower[j], reference[j] - g['max_wrist_command_delta_rad'])
            upper[j] = min(upper[j], reference[j] + g['max_wrist_command_delta_rad'])
        # When already singular, permit a bounded, monotonic escape. Never cross
        # through zero to change wrist branches in one command.
        floor = min(g['min_bend_rad'], abs(seed[4]) + 0.8 * self.max_joint_delta[4])
        if sign > 0:
            lower[4] = max(lower[4], floor)
        else:
            upper[4] = min(upper[4], -floor)
        info = {'active': True, 'branch_sign': sign, 'required_bend_rad': float(floor),
                'reference_joints': reference.tolist()}
        if np.any(lower > upper):
            return initial, dict(info, feasible=False, reason='incompatible_joint_bounds'), 0
        free = upper - lower > 1e-12
        fixed = np.clip(reference, lower, upper)
        scale = np.maximum(self.max_joint_delta, 0.01)

        def expand(x):
            q = fixed.copy()
            q[free] = x
            return q

        def errors(q):
            _, pos, rot = self._error(target, q, tool)
            return pos / g['max_position_error_m'], rot / g['max_orientation_error_rad']

        def objective(x):
            q = expand(x)
            pos, rot = errors(q)
            continuity = np.sum(((q - reference) / scale) ** 2)
            bend = ((q[4] - sign * g['preferred_bend_rad']) / 0.2) ** 2
            return 0.5 * continuity + 2.0 * bend + 10.0 * (pos @ pos) + 0.02 * (rot @ rot)

        # When starting inside a singular region, require improvement instead
        # of an impossible instantaneous jump to the safe region.
        initial_condition = self.condition_number(side, seed)
        condition_limit = max(g['max_condition'], 0.9 * initial_condition)
        info['condition_limit'] = condition_limit

        def constraints(x):
            q = expand(x)
            pos, rot = errors(q)
            return np.array([1.0 - pos @ pos, 1.0 - rot @ rot,
                             1.0 - self.condition_number(side, q) / condition_limit])

        if np.any(free):
            result = minimize(objective, fixed[free], method='SLSQP',
                              bounds=list(zip(lower[free], upper[free])),
                              constraints=[{'type': 'ineq', 'fun': constraints}],
                              options={'maxiter': min(self.max_iters, 40), 'ftol': 1e-9})
            candidate = expand(result.x)
            iterations = int(result.nit)
            info['optimizer_success'] = bool(result.success)
        else:
            candidate = fixed
            iterations = 0
        # Evaluate the exact float32 command. Strict bounds below remain the
        # final authority even if SLSQP claims success or fails to converge.
        candidate = candidate.astype(np.float32).astype(np.float64)
        pos, rot = errors(candidate)
        feasible = (np.all(np.isfinite(candidate))
                    and np.all(candidate >= lower - 1e-7)
                    and np.all(candidate <= upper + 1e-7)
                    and np.linalg.norm(pos) <= 1.00001
                    and np.linalg.norm(rot) <= 1.00001
                    and self.condition_number(side, candidate) <= condition_limit * (1.0 + self.CONDITION_REL_TOL))
        info['condition'] = self.condition_number(side, candidate)
        info.update(feasible=bool(feasible), bend_rad=float(abs(candidate[4])),
                    reason='accepted' if feasible else 'pose_or_joint_constraints')
        return candidate, info, iterations

    def solve(self, side, target_eef_cmd, seed_joints, reference_joints=None, dt=None):
        started = time.perf_counter()
        seed_joints = np.asarray(seed_joints, dtype=np.float64)
        target_eef_cmd = np.asarray(target_eef_cmd, dtype=np.float64)
        if (seed_joints.ndim != 1 or seed_joints.size < 6
                or target_eef_cmd.shape != (7,)
                or not np.all(np.isfinite(seed_joints))
                or not np.all(np.isfinite(target_eef_cmd))):
            raise ValueError("IK solve requires finite joint seed values and a finite 7D EEF target")
        seed_q = np.clip(seed_joints[:6].copy(), self.lower, self.upper)
        if self.calibration_mode == 'tool' and side not in self.calibration:
            raise ValueError("tool-frame IK requires calibration for this arm before solve")
        if self.differential is not None:
            return self.differential.solve(side, target_eef_cmd, seed_joints, reference_joints, dt)
        target = self._model_target(side, target_eef_cmd)
        tool = self.calibration.get(side) if self.calibration_mode == 'tool' else None
        if self.solver == 'least_squares':
            q, iterations = self._least_squares(target, seed_q, self.lower, self.upper, tool)
        else:
            q, iterations = self._damped_least_squares(target, seed_q, tool)
        solution_q = q.copy()
        _, solution_pos, solution_rot = self._error(target, solution_q, tool)
        solution_pos_norm = float(np.linalg.norm(solution_pos))
        solution_rot_norm = float(np.linalg.norm(solution_rot))
        solution_acceptable = self._within_errors(
            solution_pos_norm, solution_rot_norm,
            self.solution_max_position_error_m, self.solution_max_orientation_error_rad,
        )

        joint_delta_limited = False
        joint_delta_polished = False
        if self.clip_to_max_joint_delta and (self.solver == 'least_squares' or np.any(self.max_joint_delta > 0.0)):
            lower = np.maximum(self.lower, seed_q - self.max_joint_delta)
            upper = np.minimum(self.upper, seed_q + self.max_joint_delta)
            q = np.clip(q, lower, upper)
            joint_delta_limited = bool(np.linalg.norm(q - solution_q) > 1e-8)
            if joint_delta_limited and self.solver == 'least_squares':
                # Componentwise clipping destroys EEF accuracy. Refine the
                # command inside the SAME joint bounds, without changing target.
                refined, nfev = self._least_squares(target, q, lower, upper, tool)
                iterations += nfev
                if np.linalg.norm(self._error(target, refined, tool)[0]) < np.linalg.norm(self._error(target, q, tool)[0]):
                    q = refined
                    joint_delta_polished = True
        guard_info = {'active': False}
        reference = seed_q if reference_joints is None else np.asarray(reference_joints, dtype=np.float64)[:6]
        if reference.shape != (6,) or not np.all(np.isfinite(reference)):
            raise ValueError('reference_joints must contain six finite joint angles')
        if self.wrist_guard_enabled:
            trigger = self.wrist_guard['trigger_rad']
            # Also cover a branch crossing even if both endpoints are outside
            # the near-singular region, and avoid chatter while reference is near it.
            active = (min(abs(seed_q[4]), abs(solution_q[4]), abs(q[4]), abs(reference[4])) < trigger
                      or seed_q[4] * solution_q[4] <= 0
                      or max(self.condition_number(side, a) for a in (seed_q, solution_q, q, reference))
                         >= self.wrist_guard['trigger_condition'])
            if active:
                q, guard_info, nfev = self._avoid_wrist_singularity(side, target, seed_q, reference, q, tool)
                iterations += nfev
        joint_target = np.empty(7, dtype=np.float32)
        joint_target[:6] = q
        joint_target[6] = target_eef_cmd[6]
        # Check the actual transmitted float32 angles AFTER all limiting. A
        # converged unrestricted solve must never bypass this residual check.
        error, pos_error, rot_error = self._error(target, joint_target[:6], tool)
        pos_norm = float(np.linalg.norm(pos_error))
        rot_norm = float(np.linalg.norm(rot_error))
        acceptable = solution_acceptable and self._within_errors(
            pos_norm, rot_norm, self.max_position_error_m, self.max_orientation_error_rad,
        )
        if guard_info['active']:
            # The guarded solve intentionally relaxes orientation beyond the
            # ordinary full-IK tolerance, but has a stricter position bound.
            acceptable = bool(guard_info['feasible']) and self._within_errors(
                pos_norm, rot_norm,
                self.wrist_guard['max_position_error_m'] * 1.00001,
                self.wrist_guard['max_orientation_error_rad'] * 1.00001)
        target_reached = acceptable
        if guard_info['active'] and not acceptable:
            # Unreachable safe pose: keep only THIS arm at the last safe command.
            # Do not freeze the other arm or claim the model target was reached.
            hold = reference.astype(np.float32).astype(np.float64)
            hold_condition = self.condition_number(side, hold)
            guard_info['hold_condition'] = hold_condition
            hold_safe = (hold_condition <= self.wrist_guard['max_condition'] * (1.0 + self.CONDITION_REL_TOL)
                         and np.all(hold >= self.lower - 1e-7)
                         and np.all(hold <= self.upper + 1e-7)
                         and np.all(abs(hold - seed_q) <= self.max_joint_delta + 1e-7))
            if hold_safe:
                joint_target[:6] = hold
                source = seed_joints if reference_joints is None else np.asarray(reference_joints)
                if source.size >= 7:
                    joint_target[6] = source[6]
                error, pos_error, rot_error = self._error(target, joint_target[:6], tool)
                pos_norm, rot_norm = float(np.linalg.norm(pos_error)), float(np.linalg.norm(rot_error))
                acceptable = True
                guard_info['mode'] = 'hold_last_safe'
            else:
                guard_info['mode'] = 'reject_no_safe_hold'
        elif guard_info['active']:
            guard_info['mode'] = 'track_relaxed_pose'
        if self.wrist_guard_enabled:
            guard_info['command_condition'] = self.condition_number(side, joint_target[:6])
            guard_info['measured_condition'] = self.condition_number(side, seed_q)
        command_pose = self.eef_fk(side, joint_target[:6])
        return {
            'joints': joint_target,
            'wrist_singularity_avoidance': guard_info,
            'solver': self.solver,
            'calibration_mode': self.calibration_mode,
            'joint_signs': self.joint_signs.tolist(),
            'converged': bool(np.linalg.norm(error) <= self.tolerance),
            'acceptable': bool(acceptable),
            'target_reached': bool(target_reached),
            'solution_acceptable': bool(solution_acceptable),
            'solution_joints': solution_q.tolist(),
            'solution_position_error_m': solution_pos_norm,
            'solution_orientation_error_rad': solution_rot_norm,
            'position_error_m': pos_norm,
            'orientation_error_rad': rot_norm,
            'command_fk_xyz': command_pose[:3, 3].tolist(),
            'command_fk_rpy': Rotation.from_matrix(command_pose[:3, :3]).as_euler('xyz').tolist(),
            'joint_delta_limited': joint_delta_limited,
            'joint_delta_polished': joint_delta_polished,
            'joint_delta_norm': float(np.linalg.norm(joint_target[:6] - seed_q)),
            'unclipped_joint_delta_norm': float(np.linalg.norm(solution_q - seed_q)),
            'iterations': iterations,
            'solve_time_sec': time.perf_counter() - started,
        }

    def commit(self, side, result):
        if self.differential is not None:
            self.differential.commit(side, result)
