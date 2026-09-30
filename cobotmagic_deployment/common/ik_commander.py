"""EEF targets -> joint commands through Piper IK (ROS independent).

:class:`EefIkCommander` wraps :class:`~cobotmagic_deployment.common.piper_ik.PiperNumericalIK`
(including the differential solver) with the bridge-side bookkeeping:
per-arm calibration on first use, time step since the last published
command, rejection handling, bounded-tracking/stall warnings and the
wrist-guard hold of the last safe gripper command.

Typical use for one control step::

    results = commander.solve(target_left, target_right, seed_left, seed_right,
                              command_left, command_right)
    if results is None:              # solver error or rejected command
        return
    commander.apply_holds(results, target_left, target_right, gripper_candidate, transition)
    publish(results['left']['joints'], results['right']['joints'])
    commander.commit(results)
"""

import json
import time

import numpy as np

from cobotmagic_deployment.common.bridge_log import BridgeLog
from cobotmagic_deployment.common.piper_ik import PiperNumericalIK


def _jsonable(result):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.items()}


class EefIkCommander:
    def __init__(self, eef_ik_cfg, rate_hz, log=None, clock=time.monotonic):
        self.log = log or BridgeLog()
        self.rate_hz = rate_hz
        self.clock = clock
        if not bool(eef_ik_cfg.get('enabled', True)):
            raise ValueError("ros.action_mode='eef_absolute' now requires ros.eef_ik.enabled=true")
        self.ik = PiperNumericalIK(eef_ik_cfg, loginfo=self.log.info)
        if self.ik.differential is not None and abs(self.ik.differential.period - 1.0 / rate_hz) > 1e-6:
            raise ValueError("differential.control_period_sec must equal 1 / ros.rate_hz")
        self.last_published = {}
        self.last_publish_time = None
        self.rejection_log_path = None  # optional JSONL path for rejected commands

    def log_configuration(self):
        ik = self.ik
        if ik.differential is not None:
            self.log.info(
                "Differential IK: bounded incremental tracking; pose residuals indicate "
                "target accuracy, not automatic rejection; "
                f"vmax={ik.differential.vmax.tolist()} rad/s "
                f"amax={ik.differential.amax.tolist()} rad/s^2")
        self.log.info(
            "EEF IK configured: "
            f"solver={ik.solver} urdf={ik.urdf_path} base={ik.base_link} tip={ik.tip_link} "
            f"max_pos_err={ik.max_position_error_m:.4f}m "
            f"max_rot_err={ik.max_orientation_error_rad:.4f}rad"
        )

    def solve(self, target_left, target_right, seed_left, seed_right,
              command_left, command_right, context=None):
        """Solve both arms. Returns ``{'left': result, 'right': result}`` or ``None``.

        ``seed_*`` are measured joints, ``command_*`` the previous EEF command
        (used for first-use calibration). ``None`` means the solver failed or
        the command was rejected; nothing must be published. ``context`` is
        extra data recorded with a rejection.
        """
        ik = self.ik
        if 'left' not in ik.calibration:
            ik.calibrate('left', seed_left, command_left)
        if 'right' not in ik.calibration:
            ik.calibrate('right', seed_right, command_right)
        dt = (1.0 / self.rate_hz if self.last_publish_time is None
              else max(self.clock() - self.last_publish_time, 1e-6))
        try:
            left = ik.solve('left', target_left, seed_left, self.last_published.get('left'), dt=dt)
            right = ik.solve('right', target_right, seed_right, self.last_published.get('right'), dt=dt)
        except Exception as exc:  # noqa: BLE001
            self.log.warn_throttle('ik_solve_failed', 1.0, f"EEF IK solve failed; skipping command publish: {exc}")
            return None
        if (not left['acceptable'] or not right['acceptable']) and not ik.publish_on_failure:
            self._record_rejection(target_left, target_right, seed_left, seed_right, left, right, context)
            self.log.warn_throttle(
                'ik_rejected', 1.0,
                "EEF IK command rejected (pose/joint/singularity/hold constraints); skipping publish: "
                f"left_pos={left['position_error_m']:.4f}m "
                f"left_rot={left['orientation_error_rad']:.4f}rad "
                f"right_pos={right['position_error_m']:.4f}m "
                f"right_rot={right['orientation_error_rad']:.4f}rad "
                f"left_solution_pos={left['solution_position_error_m']:.4f}m "
                f"right_solution_pos={right['solution_position_error_m']:.4f}m "
                f"joint_delta_limited={left['joint_delta_limited']}/{right['joint_delta_limited']} "
                f"wrist_guard_left={left.get('wrist_singularity_avoidance', {})} "
                f"wrist_guard_right={right.get('wrist_singularity_avoidance', {})}"
            )
            return None
        self._report_limits(left, right)
        return {'left': left, 'right': right}

    def _record_rejection(self, target_left, target_right, seed_left, seed_right, left, right, context):
        if not self.rejection_log_path:
            return
        record = {'wall_time': time.time()}
        record.update(context or {})
        record.update({
            'target_left': target_left.tolist(), 'target_right': target_right.tolist(),
            'seed_left': seed_left.tolist(), 'seed_right': seed_right.tolist(),
            'left': _jsonable(left), 'right': _jsonable(right),
        })
        try:
            with open(self.rejection_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except OSError as exc:
            self.log.warn_throttle('ik_rejection_log', 1.0, f"Cannot record rejected IK command: {exc}")

    def _report_limits(self, left, right):
        if self.ik.differential is not None:
            for arm_name, result in (('left', left), ('right', right)):
                limits = result['differential_ik']['active_limits']
                if any(limits.values()):
                    message = (
                        f"EEF IK bounded tracking {arm_name}: active_limits={limits} "
                        f"target_reached={result['target_reached']} "
                        f"position_error={result['position_error_m']:.4f}m "
                        f"orientation_error={result['orientation_error_rad']:.4f}rad")
                    if result['target_reached']:
                        self.log.info_throttle(f'ik_bounded_{arm_name}_info', 1.0, message)
                    else:
                        self.log.warn_throttle(f'ik_bounded_{arm_name}_warn', 1.0, message)
        elif left.get('joint_delta_limited') or right.get('joint_delta_limited'):
            self.log.warn_throttle(
                'ik_joint_delta_limited', 1.0,
                "EEF IK joint delta limited: "
                f"left_delta={left.get('joint_delta_norm', 0.0):.4f} "
                f"left_unclipped={left.get('unclipped_joint_delta_norm', 0.0):.4f} "
                f"right_delta={right.get('joint_delta_norm', 0.0):.4f} "
                f"right_unclipped={right.get('unclipped_joint_delta_norm', 0.0):.4f}"
            )

    def apply_holds(self, results, target_left, target_right, gripper_candidate=None, gripper_transition=None):
        """Warn on stalls and apply wrist-guard holds.

        A held arm also keeps its previously published gripper: ``target_*[-1]``
        is overwritten in place and the gripper-hysteresis candidate re-synced.
        """
        for arm_index, (arm_name, target) in enumerate((('left', target_left), ('right', target_right))):
            result = results[arm_name]
            if result.get('differential_ik', {}).get('mode') == 'stalled':
                self.log.warn_throttle(
                    f'ik_stalled_{arm_name}', 1.0, f"Differential IK stalled on {arm_name}: "
                    f"position error={result['position_error_m']:.4f}m "
                    f"orientation error={result['orientation_error_rad']:.4f}rad")
            if result.get('wrist_singularity_avoidance', {}).get('mode') == 'hold_last_safe':
                self.log.warn_throttle(
                    f'ik_hold_{arm_name}', 1.0, f"Holding {arm_name} at last safe joint command: "
                    f"model target not reached; position error={result['position_error_m']:.4f}m "
                    f"orientation error={result['orientation_error_rad']:.4f}rad")
                target[-1] = result['joints'][-1]
                if gripper_candidate is not None:
                    # The guard holds the previously sent gripper too;
                    # do not commit a transition that was not published.
                    gripper_candidate.hold(arm_index, target[-1], gripper_transition)

    def commit(self, results):
        """Record a published solution (call right after publishing)."""
        self.last_publish_time = self.clock()
        for arm in ('left', 'right'):
            self.ik.commit(arm, results[arm])
            self.last_published[arm] = results[arm]['joints'].copy()
