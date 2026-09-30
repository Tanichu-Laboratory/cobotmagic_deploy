"""Per-step command shaping (ROS independent).

:class:`CommandShaper` turns the chunk action selected for the current step
into the command to publish, applying the ``ros`` options in this order:

``action_filter`` -> ``delta_clip`` -> ``gripper_delta_scale`` ->
``gripper_hysteresis`` -> ``gripper_threshold`` -> ``command_delta_deadband``
-> ``first_action_delta_scale`` -> ``initial_pose_delta_override``.

Also provides :class:`PolicyGripperInput` (``policy_gripper_input``: which
gripper value is reported to the policy) and :class:`VelocityIntegrator`
(``action_mode: velocity``).
"""

import numpy as np

from cobotmagic_deployment.common.action_processing import (
    apply_joint_delta_deadband,
    clip_joint_delta,
    override_arm_delta_from_initial_pose,
    parse_arm_selection,
    scale_action_delta_from_reference,
    scale_gripper_deltas,
    threshold_gripper_targets,
)
from cobotmagic_deployment.common.bridge_log import BridgeLog
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis

DEFAULT_ARM_STEPS_LENGTH = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]


class PolicyGripperInput:
    """``policy_gripper_input``: report measured, commanded or hybrid gripper state."""

    def __init__(self, ros_cfg, log=None):
        log = log or BridgeLog()
        cfg = ros_cfg.get('policy_gripper_input', {})
        mode = str(cfg.get('mode', 'measured')).lower()
        if mode not in ('measured', 'commanded', 'hybrid'):
            log.warn(
                "policy_gripper_input.mode must be one of measured/commanded/hybrid; "
                f"got {mode!r}. Using measured."
            )
            mode = 'measured'
        self.mode = mode
        self.hybrid_max_error = max(float(cfg.get('hybrid_max_error', 0.025)), 0.0)

    def _choose(self, measured, commanded):
        if commanded is None or self.mode == 'measured':
            return float(measured), 'measured'
        if self.mode == 'commanded':
            return float(commanded), 'commanded'
        if abs(float(measured) - float(commanded)) <= self.hybrid_max_error:
            return float(commanded), 'commanded'
        return float(measured), 'measured'

    def apply(self, measured_left, measured_right, commanded_left=None, commanded_right=None):
        """Return (policy_left, policy_right, info); ``commanded_*`` are gripper values or None."""
        measured_left = np.asarray(measured_left, dtype=np.float32)
        measured_right = np.asarray(measured_right, dtype=np.float32)
        policy_left = measured_left.copy()
        policy_right = measured_right.copy()
        left_value, left_source = self._choose(measured_left[-1], commanded_left)
        right_value, right_source = self._choose(measured_right[-1], commanded_right)
        policy_left[-1] = left_value
        policy_right[-1] = right_value
        info = {
            'mode': self.mode,
            'hybrid_max_error': self.hybrid_max_error,
            'left_source': left_source,
            'right_source': right_source,
            'left_measured': float(measured_left[-1]),
            'right_measured': float(measured_right[-1]),
            'left_policy': float(policy_left[-1]),
            'right_policy': float(policy_right[-1]),
            'left_commanded': commanded_left,
            'right_commanded': commanded_right,
        }
        return policy_left, policy_right, info


class VelocityIntegrator:
    """``action_mode: velocity``: integrate joint velocities; grippers stay absolute."""

    def __init__(self, dt):
        self.dt = dt
        self.left = None
        self.right = None

    def reset(self, left, right):
        self.left = np.asarray(left).copy()
        self.right = np.asarray(right).copy()

    def step(self, action_left, action_right):
        self.left = self.left + action_left * self.dt
        self.right = self.right + action_right * self.dt
        self.left[-1] = action_left[-1]
        self.right[-1] = action_right[-1]
        return self.left, self.right


class CommandShaper:
    """Per-step shaping configured from a bridge ``ros`` YAML section.

    ``shape`` is side-effect free except for the EMA filter state; the
    gripper-hysteresis state is only adopted by ``commit`` after the command
    was actually published.
    """

    def __init__(self, ros_cfg, joint_dim=7, command_publish_mode='direct', log=None):
        self.log = log or BridgeLog()
        step_lengths = ros_cfg.get('arm_steps_length', DEFAULT_ARM_STEPS_LENGTH)

        action_filter = ros_cfg.get('action_filter', {})
        self.filter_enabled = bool(action_filter.get('enabled', False))
        self.filter_alpha = min(max(float(action_filter.get('ema_alpha', 0.25)), 0.0), 1.0)
        self.filter_deadband = None
        if self.filter_enabled:
            deadband = np.asarray(action_filter.get('deadband', [0.0] * joint_dim), dtype=np.float32)
            if deadband.shape[0] != joint_dim:
                self.log.warn("action_filter.deadband size mismatch; disabling deadband.")
                deadband = np.zeros(joint_dim, dtype=np.float32)
            self.filter_deadband = deadband
        self.filtered_left = None
        self.filtered_right = None

        delta_clip = ros_cfg.get('delta_clip')
        self.delta_clip_enabled = bool(delta_clip and delta_clip.get('enabled', False))
        self.delta_clip_reference = 'command'
        self.delta_clip_max = None
        if self.delta_clip_enabled:
            reference = delta_clip.get('reference', 'command').lower()
            if reference not in ('command', 'current'):
                self.log.warn(f"Unsupported delta_clip.reference={reference!r}; using 'command'.")
                reference = 'command'
            self.delta_clip_reference = reference
            max_delta = np.asarray(delta_clip.get('max_delta', step_lengths), dtype=np.float32)
            if max_delta.shape[0] != joint_dim:
                self.log.warn("delta_clip.max_delta size mismatch; using arm_steps_length.")
                max_delta = np.asarray(step_lengths, dtype=np.float32)
            if max_delta.shape[0] != joint_dim:
                self.log.warn("delta clip disabled: max_delta and arm_steps_length sizes do not match joint_names.")
                self.delta_clip_enabled = False
            else:
                self.delta_clip_max = max_delta

        deadband_cfg = ros_cfg.get('command_delta_deadband', {})
        self.command_deadband_enabled = bool(deadband_cfg.get('enabled', False))
        self.command_deadband_left = None
        self.command_deadband_right = None
        if self.command_deadband_enabled:
            left = np.asarray(deadband_cfg.get('left', [0.0] * joint_dim), dtype=np.float32)
            right = np.asarray(deadband_cfg.get('right', [0.0] * joint_dim), dtype=np.float32)
            if left.shape[0] != joint_dim or right.shape[0] != joint_dim:
                self.log.warn("command_delta_deadband left/right size mismatch; disabling command delta deadband.")
                self.command_deadband_enabled = False
            elif np.any(left < 0.0) or np.any(right < 0.0):
                self.log.warn("command_delta_deadband thresholds must be non-negative; disabling command delta deadband.")
                self.command_deadband_enabled = False
            else:
                self.command_deadband_left = left
                self.command_deadband_right = right

        first = ros_cfg.get('first_action_delta_scale', {})
        self.first_scale_enabled = bool(first.get('enabled', False))
        raw_coefficient = float(first.get('coefficient', 1.0))
        self.first_scale_coefficient = float(np.clip(raw_coefficient, 0.0, 1.0))
        if self.first_scale_coefficient != raw_coefficient:
            self.log.warn(
                "first_action_delta_scale.coefficient must be in [0, 1]; "
                f"clamped {raw_coefficient} to {self.first_scale_coefficient}."
            )
        self.first_scale_include_gripper = bool(first.get('include_gripper', False))

        override = ros_cfg.get('initial_pose_delta_override', {})
        self.override_enabled = bool(override.get('enabled', False))
        self.override_arms, invalid = parse_arm_selection(override.get('arms', []))
        if invalid:
            self.log.warn(
                "initial_pose_delta_override.arms contains unsupported values "
                f"{sorted(invalid)}; only left/right/both are accepted."
            )
        if self.override_enabled and not self.override_arms:
            self.log.warn("initial_pose_delta_override is enabled but arms is empty; disabling the override.")
            self.override_enabled = False
        self.override_include_gripper = bool(override.get('include_gripper', False))

        threshold = ros_cfg.get('gripper_threshold', ros_cfg.get('gripper_binary', {}))
        self.threshold_enabled = bool(threshold.get('enabled', False))
        self.threshold_close = np.asarray(threshold.get('close_threshold', [0.004, 0.004]), dtype=np.float32)
        self.threshold_open = np.asarray(threshold.get('open_threshold', [0.020, 0.020]), dtype=np.float32)
        self.threshold_close_value = np.asarray(threshold.get('close_value', [-0.0037, -0.0033]), dtype=np.float32)
        self.threshold_open_value = np.asarray(threshold.get('open_value', [0.058, 0.059]), dtype=np.float32)

        hysteresis = ros_cfg.get('gripper_hysteresis', {})
        self.hysteresis = GripperHysteresis(hysteresis) if hysteresis.get('enabled', False) else None
        if self.hysteresis is not None and self.threshold_enabled:
            raise ValueError('gripper_hysteresis and gripper_threshold cannot both be enabled')
        if self.hysteresis is not None and command_publish_mode != 'direct':
            raise ValueError('gripper_hysteresis requires direct command publishing')
        if self.threshold_enabled:
            if any(v.shape[0] != 2 for v in (self.threshold_close, self.threshold_open,
                                             self.threshold_close_value, self.threshold_open_value)):
                self.log.warn(
                    "gripper_threshold close_threshold/open_threshold/close_value/open_value must each contain "
                    "two values [left, right]; disabling gripper thresholding."
                )
                self.threshold_enabled = False
            elif np.any(self.threshold_close >= self.threshold_open):
                self.log.warn(
                    "gripper_threshold close_threshold must be smaller than open_threshold; "
                    "disabling gripper thresholding."
                )
                self.threshold_enabled = False

        scale = ros_cfg.get('gripper_delta_scale', {})
        self.gripper_scale_enabled = bool(scale.get('enabled', False))
        self.gripper_scale_gains = np.asarray([
            max(float(scale.get('left_gain', 1.0)), 0.0),
            max(float(scale.get('right_gain', 1.0)), 0.0),
        ], dtype=np.float32)
        clip_min = scale.get('clip_min')
        clip_max = scale.get('clip_max')
        clip_min = None if clip_min is None else np.asarray(clip_min, dtype=np.float32)
        clip_max = None if clip_max is None else np.asarray(clip_max, dtype=np.float32)
        if (clip_min is not None and clip_min.shape != (2,)) or (clip_max is not None and clip_max.shape != (2,)):
            self.log.warn("gripper_delta_scale clip_min/clip_max must each contain [left, right]; disabling clipping.")
            clip_min = None
            clip_max = None
        self.gripper_scale_clip_min = clip_min
        self.gripper_scale_clip_max = clip_max

    @property
    def override_left(self):
        return self.override_enabled and 'left' in self.override_arms

    @property
    def override_right(self):
        return self.override_enabled and 'right' in self.override_arms

    def log_configuration(self):
        if self.delta_clip_enabled:
            self.log.info(
                f"Joint delta clip enabled: max_delta={self.delta_clip_max.tolist()} "
                f"reference={self.delta_clip_reference}"
            )
        if self.command_deadband_enabled:
            self.log.info(
                "Command delta deadband enabled: "
                f"left={self.command_deadband_left.tolist()} right={self.command_deadband_right.tolist()}"
            )
        if self.threshold_enabled:
            self.log.info(
                "Gripper threshold output enabled: "
                f"close_threshold={self.threshold_close.tolist()} "
                f"open_threshold={self.threshold_open.tolist()} "
                f"close_value={self.threshold_close_value.tolist()} "
                f"open_value={self.threshold_open_value.tolist()}"
            )
        if self.gripper_scale_enabled:
            clip_min = None if self.gripper_scale_clip_min is None else self.gripper_scale_clip_min.tolist()
            clip_max = None if self.gripper_scale_clip_max is None else self.gripper_scale_clip_max.tolist()
            self.log.info(
                "Gripper delta scaling enabled: "
                f"left_gain={self.gripper_scale_gains[0]:.3f} "
                f"right_gain={self.gripper_scale_gains[1]:.3f} "
                f"clip_min={clip_min} clip_max={clip_max}"
            )
        if self.filter_enabled:
            self.log.info(
                f"Action target EMA filter enabled: alpha={self.filter_alpha:.3f} "
                f"deadband={self.filter_deadband.tolist()}"
            )
        if self.first_scale_enabled:
            self.log.info(
                "First consumed action delta scaling enabled: "
                f"coefficient={self.first_scale_coefficient:.3f} "
                f"include_gripper={self.first_scale_include_gripper}"
            )
        if self.override_enabled:
            self.log.info(
                "Initial-pose delta override enabled: "
                f"arms={sorted(self.override_arms)} "
                f"include_gripper={self.override_include_gripper}"
            )

    def metadata(self):
        """Settings recorded with every chunk (action logs / npz metadata)."""
        return {
            'first_action_delta_scale_enabled': self.first_scale_enabled,
            'first_action_delta_scale_coefficient': self.first_scale_coefficient,
            'first_action_delta_scale_include_gripper': self.first_scale_include_gripper,
            'initial_pose_delta_override_enabled': self.override_enabled,
            'initial_pose_delta_override_arms': sorted(self.override_arms),
            'initial_pose_delta_override_include_gripper': self.override_include_gripper,
        }

    def clip_reference(self, command_left, command_right, current_left, current_right):
        if self.delta_clip_reference == 'current':
            return current_left, current_right
        return command_left, command_right

    def shape(self, target_left, target_right, command_left, command_right,
              current_left, current_right, measured_left, measured_right, chunk):
        """Shape one step.

        ``target_*``: chunk action (after temporal ensemble / integration).
        ``command_*``: previous published command. ``current_*``: delta-clip
        "current" reference. ``measured_*``: latest measured joints (gripper
        feedback for hysteresis). ``chunk`` supplies ``request_current_*``,
        ``request_measured_*`` and ``executed_steps``.
        """
        clip_left, clip_right = self.clip_reference(command_left, command_right, current_left, current_right)
        if self.filter_enabled:
            if self.filtered_left is None or self.filtered_right is None:
                self.filtered_left = command_left.copy()
                self.filtered_right = command_right.copy()
            self.filtered_left = self.filter_alpha * target_left + (1.0 - self.filter_alpha) * self.filtered_left
            self.filtered_right = self.filter_alpha * target_right + (1.0 - self.filter_alpha) * self.filtered_right
            filtered_left = self.filtered_left.copy()
            filtered_right = self.filtered_right.copy()
            small_left = np.abs(filtered_left - command_left) < self.filter_deadband
            small_right = np.abs(filtered_right - command_right) < self.filter_deadband
            filtered_left[small_left] = command_left[small_left]
            filtered_right[small_right] = command_right[small_right]
        else:
            filtered_left = target_left
            filtered_right = target_right
        if self.delta_clip_enabled:
            out_left = clip_joint_delta(clip_left, filtered_left, self.delta_clip_max)
            out_right = clip_joint_delta(clip_right, filtered_right, self.delta_clip_max)
        else:
            out_left = filtered_left
            out_right = filtered_right
        clipped_left = out_left.copy()
        clipped_right = out_right.copy()

        request_measured_left = chunk.get('request_measured_left', chunk['request_current_left'])
        request_measured_right = chunk.get('request_measured_right', chunk['request_current_right'])
        if self.gripper_scale_enabled:
            out_left, out_right = scale_gripper_deltas(
                out_left, out_right,
                request_measured_left, request_measured_right,
                self.gripper_scale_gains,
                clip_min=self.gripper_scale_clip_min,
                clip_max=self.gripper_scale_clip_max,
            )
        gripper_candidate = self.hysteresis
        gripper_transition = None
        if self.hysteresis is not None:
            values, gripper_candidate, gripper_transition = self.hysteresis.propose(
                [out_left[-1], out_right[-1]],
                [measured_left[-1], measured_right[-1]],
                request_opening=[chunk['request_current_left'][-1], chunk['request_current_right'][-1]],
            )
            out_left = out_left.copy()
            out_right = out_right.copy()
            out_left[-1], out_right[-1] = values
        if self.threshold_enabled:
            out_left, out_right = threshold_gripper_targets(
                out_left, out_right,
                self.threshold_close, self.threshold_open,
                self.threshold_close_value, self.threshold_open_value,
            )
        if self.command_deadband_enabled:
            out_left, residual_left = apply_joint_delta_deadband(command_left, out_left, self.command_deadband_left)
            out_right, residual_right = apply_joint_delta_deadband(command_right, out_right, self.command_deadband_right)
        else:
            residual_left = np.zeros_like(out_left, dtype=np.float32)
            residual_right = np.zeros_like(out_right, dtype=np.float32)

        first_scale_applied = self.first_scale_enabled and chunk['executed_steps'] == 0
        if first_scale_applied:
            out_left, out_right = scale_action_delta_from_reference(
                out_left, out_right,
                command_left, command_right,
                self.first_scale_coefficient,
                include_gripper=self.first_scale_include_gripper,
            )
        if self.override_enabled:
            out_left, out_right = override_arm_delta_from_initial_pose(
                out_left, out_right,
                request_measured_left, request_measured_right,
                self.override_left, self.override_right,
                include_gripper=self.override_include_gripper,
            )
        return {
            'filtered_left': filtered_left,
            'filtered_right': filtered_right,
            'clipped_left': clipped_left,
            'clipped_right': clipped_right,
            'clip_reference_left': clip_left,
            'clip_reference_right': clip_right,
            'target_left': out_left,
            'target_right': out_right,
            'deadband_residual_left': residual_left,
            'deadband_residual_right': residual_right,
            'first_action_delta_scale_applied': first_scale_applied,
            'gripper_candidate': gripper_candidate,
            'gripper_transition': gripper_transition,
        }

    def commit(self, shaped):
        """Adopt the gripper-hysteresis state after the command was published."""
        self.hysteresis = shaped['gripper_candidate']
