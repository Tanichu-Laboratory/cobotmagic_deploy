"""Response-time processing of dual-arm action chunks (ROS independent).

``parse_action_response`` decodes a policy server reply, and
:class:`ChunkPipeline` applies the chunk-level options from the ``ros``
section of a bridge YAML, in this order:

``action_chunk_lowpass`` -> ``action_chunk_monotonic`` ->
``action_chunk_smoothing`` -> ``chunk_interpolation`` ->
``chunk_terminal_displacement_filter``.

A new bridge can reuse the pipeline with::

    pipeline = ChunkPipeline(cfg['ros'], rate_hz)
    processed = pipeline.process(left, right, vel, command_left, command_right,
                                 measured_left, measured_right)
"""

import json

import numpy as np

from cobotmagic_deployment.common.action_processing import (
    adaptive_bridge_to_first_action,
    adaptive_delta_upsample_chunks,
    filter_chunk_by_terminal_displacement,
    interpolate_with_segment_factors,
    linear_upsample_chunk,
    lowpass_dual_action_chunks_zero_phase,
    parse_arm_selection,
    project_action_chunk_monotonic_to_endpoint,
    smooth_dual_action_chunks_savgol,
    validate_policy_action_mode,
)
from cobotmagic_deployment.common.bridge_log import BridgeLog

DEFAULT_ADAPTIVE_JOINT_THRESHOLD = [0.024, 0.069, 0.070, 0.039, 0.046, 0.046, 0.0034]


class InvalidPolicyResponse(Exception):
    """A policy reply that must be dropped. ``level`` is ``'warn'`` or ``'error'``."""

    def __init__(self, message, level='warn'):
        super().__init__(message)
        self.level = level


class ChunkRejected(Exception):
    """A chunk-processing stage failed; the whole response is dropped."""


def parse_action_response(frames, expected_action_mode, log=None):
    """Decode a policy reply into a dict of float32 matrices.

    Returns ``header``, ``left``, ``right`` (``(T, stride)``), ``vel``
    (``(T, 2)`` or ``None``) and ``model_raw_left``/``model_raw_right``.
    Raises :class:`InvalidPolicyResponse` for replies that must be dropped.
    """
    log = log or BridgeLog()
    try:
        header = json.loads(frames[0].decode('utf-8'))
    except (IndexError, json.JSONDecodeError, UnicodeDecodeError):
        raise InvalidPolicyResponse("Invalid policy response header.")
    try:
        validate_policy_action_mode(header, expected_action_mode)
    except ValueError as exc:
        raise InvalidPolicyResponse(str(exc), level='error')

    chunk_size = int(header.get('chunk_size', 0))
    if chunk_size == 0 or len(frames) < 3:
        raise InvalidPolicyResponse("Empty or incomplete policy response.")

    left_stride = int(header.get('left_stride', 7))
    right_stride = int(header.get('right_stride', 7))
    left_arr = np.frombuffer(frames[1], dtype=np.float32, count=chunk_size * left_stride)
    right_arr = np.frombuffer(frames[2], dtype=np.float32, count=chunk_size * right_stride)
    try:
        left = left_arr.reshape((chunk_size, left_stride))
        right = right_arr.reshape((chunk_size, right_stride))
    except ValueError:
        raise InvalidPolicyResponse("Invalid action matrix shape.")

    vel = None
    next_frame_idx = 3
    if header.get('has_vel', False) and len(frames) > next_frame_idx:
        vel_arr = np.frombuffer(frames[next_frame_idx], dtype=np.float32, count=chunk_size * 2)
        try:
            vel = vel_arr.reshape((chunk_size, 2))
        except ValueError:
            vel = None
        next_frame_idx += 1

    model_raw_left = None
    model_raw_right = None
    if header.get('has_model_raw_action', False):
        raw_left_stride = int(header.get('model_raw_left_stride', left_stride))
        raw_right_stride = int(header.get('model_raw_right_stride', right_stride))
        if len(frames) >= next_frame_idx + 2:
            raw_left_arr = np.frombuffer(
                frames[next_frame_idx], dtype=np.float32, count=chunk_size * raw_left_stride)
            raw_right_arr = np.frombuffer(
                frames[next_frame_idx + 1], dtype=np.float32, count=chunk_size * raw_right_stride)
            try:
                model_raw_left = raw_left_arr.reshape((chunk_size, raw_left_stride))
                model_raw_right = raw_right_arr.reshape((chunk_size, raw_right_stride))
            except ValueError:
                log.warn("Invalid model raw action matrix shape; skipping model_raw_action save.")
                model_raw_left = None
                model_raw_right = None
        else:
            log.warn("Policy response header has_model_raw_action=true but raw action frames are missing.")

    return {
        'header': header,
        'left': left,
        'right': right,
        'vel': vel,
        'model_raw_left': model_raw_left,
        'model_raw_right': model_raw_right,
    }


class ChunkPipeline:
    """Chunk-level processing configured from a bridge ``ros`` YAML section.

    ``default_overlap_steps`` is the temporal-ensemble overlap used when
    adaptive interpolation does not override it.
    """

    def __init__(self, ros_cfg, rate_hz, default_overlap_steps=None, log=None):
        self.log = log or BridgeLog()
        self.rate_hz = rate_hz
        self.default_overlap_steps = default_overlap_steps
        open_loop_steps = ros_cfg.get('open_loop_steps')
        self.open_loop_steps = None if open_loop_steps is None else max(int(open_loop_steps), 1)
        self.chunk_action_skip_steps = max(int(ros_cfg.get('chunk_action_skip_steps', 0)), 0)

        interp = ros_cfg.get('chunk_interpolation', {})
        self.interpolation_enabled = bool(interp.get('enabled', False))
        self.interpolation_mode = interp.get('mode', 'linear').lower()
        self.interpolation_factor = max(float(interp.get('factor', 1.0)), 1.0)
        self.adaptive_min_factor = max(int(interp.get('min_factor', 1)), 1)
        self.adaptive_max_factor = max(
            max(int(interp.get('max_factor', max(1, int(round(self.interpolation_factor))))), 1),
            self.adaptive_min_factor,
        )
        source_steps = interp.get('source_steps_to_execute')
        self.adaptive_source_steps_to_execute = None if source_steps is None else max(int(source_steps), 1)
        source_overlap = interp.get('source_overlap_steps')
        self.adaptive_source_overlap_steps = None if source_overlap is None else max(int(source_overlap), 0)
        self.adaptive_joint_threshold = interp.get('joint_threshold', DEFAULT_ADAPTIVE_JOINT_THRESHOLD)
        self.adaptive_bridge_enabled = bool(interp.get('initial_bridge_enabled', False))
        self.adaptive_bridge_max_factor = max(
            max(int(interp.get('initial_bridge_max_factor', self.adaptive_max_factor)), 1),
            self.adaptive_min_factor,
        )

        smoothing = ros_cfg.get('action_chunk_smoothing', {})
        self.smoothing_enabled = bool(smoothing.get('enabled', False))
        self.smoothing_upsample = max(int(smoothing.get('upsample_factor', 2)), 1)
        self.smoothing_window = max(int(smoothing.get('window_length', 21)), 3)
        self.smoothing_polyorder = max(int(smoothing.get('polyorder', 3)), 0)

        lowpass = ros_cfg.get('action_chunk_lowpass', {})
        self.lowpass_enabled = bool(lowpass.get('enabled', False))
        self.lowpass_cutoff_hz = float(lowpass.get('cutoff_hz', 1.2))
        self.lowpass_sample_rate_hz = float(lowpass.get('sample_rate_hz', rate_hz))
        self.lowpass_order = max(int(lowpass.get('order', 4)), 1)
        self.lowpass_preserve_endpoints = bool(lowpass.get('preserve_endpoints', True))
        self.lowpass_include_gripper = bool(lowpass.get('include_gripper', False))
        if self.lowpass_enabled and not (0.0 < self.lowpass_cutoff_hz < 0.5 * self.lowpass_sample_rate_hz):
            raise ValueError(
                "action_chunk_lowpass.cutoff_hz must be between 0 and "
                f"{0.5 * self.lowpass_sample_rate_hz:.6g} Hz"
            )

        monotonic = ros_cfg.get('action_chunk_monotonic', {})
        self.monotonic_enabled = bool(monotonic.get('enabled', False))
        self.monotonic_arms, invalid = parse_arm_selection(monotonic.get('arms', ['left', 'right']))
        if invalid:
            raise ValueError(
                "action_chunk_monotonic.arms only accepts left/right/both; got "
                f"{sorted(invalid)}"
            )
        self.monotonic_strength = float(monotonic.get('strength', 1.0))
        if not 0.0 <= self.monotonic_strength <= 1.0:
            raise ValueError("action_chunk_monotonic.strength must be in [0, 1]")
        self.monotonic_min_terminal_delta = max(float(monotonic.get('min_terminal_delta', 1e-4)), 0.0)
        self.monotonic_include_gripper = bool(monotonic.get('include_gripper', False))

        terminal = ros_cfg.get('chunk_terminal_displacement_filter', {})
        self.terminal_filter_enabled = bool(terminal.get('enabled', False))
        self.terminal_left_threshold = np.asarray(terminal.get('left_threshold', [0.0] * 7), dtype=np.float32)
        self.terminal_right_threshold = np.asarray(terminal.get('right_threshold', [0.0] * 7), dtype=np.float32)
        if self.terminal_left_threshold.shape != (7,) or self.terminal_right_threshold.shape != (7,):
            self.log.warn(
                "chunk_terminal_displacement_filter left_threshold/right_threshold must each contain 7 values; "
                "disabling the filter."
            )
            self.terminal_filter_enabled = False

    def log_configuration(self):
        if self.smoothing_enabled:
            self.log.info(
                "DreamZero action chunk smoothing enabled: "
                f"cubic_upsample={self.smoothing_upsample}x "
                f"savgol_window={self.smoothing_window} "
                f"polyorder={self.smoothing_polyorder}"
            )
        if self.terminal_filter_enabled:
            self.log.info(
                "Chunk terminal displacement filter enabled: "
                f"left_threshold={self.terminal_left_threshold.tolist()} "
                f"right_threshold={self.terminal_right_threshold.tolist()}"
            )
        if self.interpolation_enabled:
            if self.interpolation_mode == 'adaptive_delta':
                self.log.info(
                    "Adaptive delta chunk interpolation enabled: "
                    f"source_steps_to_execute={self.adaptive_source_steps_to_execute} "
                    f"source_overlap_steps={self.adaptive_source_overlap_steps} "
                    f"min_factor={self.adaptive_min_factor} max_factor={self.adaptive_max_factor} "
                    f"initial_bridge_enabled={self.adaptive_bridge_enabled} "
                    f"initial_bridge_max_factor={self.adaptive_bridge_max_factor} "
                    f"joint_threshold={self.adaptive_joint_threshold}"
                )
            else:
                self.log.info(f"Linear chunk interpolation enabled: factor={self.interpolation_factor:.3f}")

    def metadata(self):
        """Settings recorded with every chunk (action logs / npz metadata)."""
        return {
            'terminal_filter_enabled': self.terminal_filter_enabled,
            'chunk_lowpass_enabled': self.lowpass_enabled,
            'chunk_lowpass_cutoff_hz': self.lowpass_cutoff_hz,
            'chunk_lowpass_sample_rate_hz': self.lowpass_sample_rate_hz,
            'chunk_lowpass_order': self.lowpass_order,
            'chunk_lowpass_preserve_endpoints': self.lowpass_preserve_endpoints,
            'chunk_monotonic_enabled': self.monotonic_enabled,
            'chunk_monotonic_arms': sorted(self.monotonic_arms),
            'chunk_monotonic_strength': self.monotonic_strength,
            'chunk_monotonic_min_terminal_delta': self.monotonic_min_terminal_delta,
        }

    def _steps_to_execute(self, chunk_size):
        return chunk_size if self.open_loop_steps is None else min(self.open_loop_steps, chunk_size)

    def process(self, left, right, vel, command_left, command_right, initial_left, initial_right):
        """Process one received chunk.

        ``command_*`` is the bridge's current command (monotonic projection and
        initial bridge start from it); ``initial_*`` is the measured state at
        request time (terminal displacement filter). Raises
        :class:`ChunkRejected` when a stage fails.
        """
        received_left = left.copy()
        received_right = right.copy()
        chunk_size = int(left.shape[0])

        if self.lowpass_enabled:
            try:
                left, right = lowpass_dual_action_chunks_zero_phase(
                    left,
                    right,
                    sample_rate_hz=self.lowpass_sample_rate_hz,
                    cutoff_hz=self.lowpass_cutoff_hz,
                    order=self.lowpass_order,
                    preserve_endpoints=self.lowpass_preserve_endpoints,
                    include_gripper=self.lowpass_include_gripper,
                )
            except ValueError as exc:
                raise ChunkRejected(f"Chunk low-pass skipped: {exc}")
            lowpass_delta = np.concatenate(
                (left[:, :-1] - received_left[:, :-1], right[:, :-1] - received_right[:, :-1]),
                axis=1,
            )
            self.log.info(
                "Filtered chunk-wide high-frequency motion: "
                f"cutoff_hz={self.lowpass_cutoff_hz:.3f} "
                f"sample_rate_hz={self.lowpass_sample_rate_hz:.3f} "
                f"order={self.lowpass_order} "
                f"rms_delta={float(np.sqrt(np.mean(lowpass_delta ** 2))):.6f} "
                f"max_delta={float(np.max(np.abs(lowpass_delta))):.6f}"
            )
        lowpass_left = left.copy()
        lowpass_right = right.copy()

        if self.monotonic_enabled:
            before_left = left.copy()
            before_right = right.copy()
            try:
                if 'left' in self.monotonic_arms:
                    left = project_action_chunk_monotonic_to_endpoint(
                        left, command_left,
                        strength=self.monotonic_strength,
                        min_terminal_delta=self.monotonic_min_terminal_delta,
                        include_gripper=self.monotonic_include_gripper,
                    )
                if 'right' in self.monotonic_arms:
                    right = project_action_chunk_monotonic_to_endpoint(
                        right, command_right,
                        strength=self.monotonic_strength,
                        min_terminal_delta=self.monotonic_min_terminal_delta,
                        include_gripper=self.monotonic_include_gripper,
                    )
            except ValueError as exc:
                raise ChunkRejected(f"Chunk monotonic projection skipped: {exc}")
            monotonic_delta = np.concatenate(
                (left[:, :-1] - before_left[:, :-1], right[:, :-1] - before_right[:, :-1]),
                axis=1,
            )
            self.log.info(
                "Suppressed chunk-internal joint reversals: "
                f"arms={sorted(self.monotonic_arms)} "
                f"strength={self.monotonic_strength:.3f} "
                f"rms_delta={float(np.sqrt(np.mean(monotonic_delta ** 2))):.6f} "
                f"max_delta={float(np.max(np.abs(monotonic_delta))):.6f}"
            )
        monotonic_left = left.copy()
        monotonic_right = right.copy()

        if self.smoothing_enabled:
            left, right = smooth_dual_action_chunks_savgol(
                left, right,
                upsample_factor=self.smoothing_upsample,
                window_length=self.smoothing_window,
                polyorder=self.smoothing_polyorder,
            )

        overlap_steps = self.default_overlap_steps
        initial_bridge_factor = 1
        initial_bridge_added_steps = 0
        if self.interpolation_enabled and self.interpolation_mode == 'adaptive_delta':
            try:
                left, right, source_to_expanded, segment_factors = adaptive_delta_upsample_chunks(
                    left, right,
                    self.adaptive_min_factor,
                    self.adaptive_max_factor,
                    self.adaptive_joint_threshold,
                )
            except ValueError as exc:
                raise ChunkRejected(str(exc))
            if vel is not None:
                vel, _ = interpolate_with_segment_factors(vel, segment_factors)
            if self.adaptive_bridge_enabled:
                try:
                    before_bridge = left.shape[0]
                    left, right, source_to_expanded, initial_bridge_factor = adaptive_bridge_to_first_action(
                        left, right,
                        command_left, command_right,
                        self.adaptive_min_factor,
                        self.adaptive_bridge_max_factor,
                        self.adaptive_joint_threshold,
                        source_to_expanded,
                    )
                    if vel is not None and left.shape[0] > before_bridge:
                        vel = np.vstack([np.repeat(vel[:1], left.shape[0] - before_bridge, axis=0), vel])
                    initial_bridge_added_steps = left.shape[0] - before_bridge
                    if initial_bridge_added_steps > 0:
                        self.log.info(
                            "Inserted chunk-boundary bridge: "
                            f"factor={initial_bridge_factor} "
                            f"added_steps={initial_bridge_added_steps}"
                        )
                except ValueError as exc:
                    raise ChunkRejected(str(exc))
            chunk_size = int(left.shape[0])
            if self.adaptive_source_steps_to_execute is not None:
                source_idx = min(self.adaptive_source_steps_to_execute - 1, len(source_to_expanded) - 1)
                steps_to_execute = source_to_expanded[source_idx] + 1
            else:
                steps_to_execute = self._steps_to_execute(chunk_size)
            if self.adaptive_source_overlap_steps is not None:
                if self.adaptive_source_overlap_steps <= 0:
                    overlap_steps = 0
                else:
                    source_idx = min(self.adaptive_source_overlap_steps - 1, len(source_to_expanded) - 1)
                    overlap_steps = source_to_expanded[source_idx] + 1
        else:
            if self.interpolation_enabled and self.interpolation_factor > 1.0:
                left = linear_upsample_chunk(left, self.interpolation_factor)
                right = linear_upsample_chunk(right, self.interpolation_factor)
                if vel is not None:
                    vel = linear_upsample_chunk(vel, self.interpolation_factor)
                chunk_size = int(left.shape[0])
            steps_to_execute = self._steps_to_execute(chunk_size)

        terminal_delta_left = np.zeros(left.shape[1], dtype=np.float32)
        terminal_delta_right = np.zeros(right.shape[1], dtype=np.float32)
        terminal_cancel_left = np.zeros(left.shape[1], dtype=bool)
        terminal_cancel_right = np.zeros(right.shape[1], dtype=bool)
        terminal_endpoint_index = min(max(int(steps_to_execute), 1), chunk_size) - 1
        if self.terminal_filter_enabled:
            try:
                (
                    left, right,
                    terminal_delta_left, terminal_delta_right,
                    terminal_cancel_left, terminal_cancel_right,
                    terminal_endpoint_index,
                ) = filter_chunk_by_terminal_displacement(
                    left, right,
                    initial_left, initial_right,
                    steps_to_execute,
                    self.terminal_left_threshold,
                    self.terminal_right_threshold,
                )
            except ValueError as exc:
                self.log.warn(f"Chunk terminal displacement filter skipped: {exc}")
            else:
                self.log.info(
                    "Chunk terminal displacement filter: "
                    f"endpoint_index={terminal_endpoint_index} "
                    f"left_cancelled={np.flatnonzero(terminal_cancel_left).tolist()} "
                    f"right_cancelled={np.flatnonzero(terminal_cancel_right).tolist()}"
                )

        return {
            'left': left.copy(),
            'right': right.copy(),
            'vel': None if vel is None else vel.copy(),
            'received_left': received_left,
            'received_right': received_right,
            'lowpass_left': lowpass_left,
            'lowpass_right': lowpass_right,
            'monotonic_left': monotonic_left,
            'monotonic_right': monotonic_right,
            'steps_to_execute': steps_to_execute,
            'overlap_steps': overlap_steps,
            'action_skip_steps': min(self.chunk_action_skip_steps, max(chunk_size - 1, 0)),
            'initial_bridge_factor': initial_bridge_factor,
            'initial_bridge_added_steps': initial_bridge_added_steps,
            'terminal_endpoint_index': terminal_endpoint_index,
            'terminal_delta_left': terminal_delta_left.copy(),
            'terminal_delta_right': terminal_delta_right.copy(),
            'terminal_cancel_left': terminal_cancel_left.copy(),
            'terminal_cancel_right': terminal_cancel_right.copy(),
        }
