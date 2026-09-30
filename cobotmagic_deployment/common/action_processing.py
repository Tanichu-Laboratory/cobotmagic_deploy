"""ROS-independent action-chunk processing used by the CobotMagic bridge.

Everything here operates on NumPy arrays only, so it can be unit-tested
without a ROS installation. Rows are per-arm commands ``[j0..j5, gripper]``
(joint mode) or ``[x, y, z, roll, pitch, yaw, gripper]`` (EEF mode).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, savgol_filter, sosfiltfilt


# --- EEF pose conversions ---------------------------------------------------

def quat_xyzw_to_matrix(quat):
    x, y, z, w = [float(v) for v in quat]
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)



def quat_xyzw_to_rot6d(quat):
    mat = quat_xyzw_to_matrix(quat)
    return mat[:, :2].reshape(-1).astype(np.float32)



def quat_xyzw_to_euler_xyz(quat):
    mat = quat_xyzw_to_matrix(quat).astype(np.float64)
    # Match scipy.spatial.transform.Rotation.as_euler("xyz") / from_euler("xyz").
    sy = float(np.clip(-mat[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sy)
    if abs(sy) < 0.999999:
        roll = np.arctan2(mat[2, 1], mat[2, 2])
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        roll = np.arctan2(-mat[1, 2], mat[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float32)



def eef_pose_and_gripper_to_ee6d(pose, gripper):
    pose_arr = np.asarray(pose, dtype=np.float32)
    return np.concatenate([
        pose_arr[:3],
        quat_xyzw_to_rot6d(pose_arr[3:7]),
        np.asarray([float(gripper)], dtype=np.float32),
    ])



def eef_pose_and_gripper_to_command(pose, gripper):
    pose_arr = np.asarray(pose, dtype=np.float32)
    return np.concatenate([
        pose_arr[:3],
        quat_xyzw_to_euler_xyz(pose_arr[3:7]),
        np.asarray([float(gripper)], dtype=np.float32),
    ])



# --- Configuration helpers --------------------------------------------------

def parse_arm_selection(value):
    """Parse ``left``/``right``/``both`` (string or list) into (arms, invalid) sets."""
    if isinstance(value, str):
        value = [value]
    arms = {str(arm).strip().lower() for arm in value}
    if 'both' in arms:
        arms.update(('left', 'right'))
        arms.discard('both')
    invalid = arms - {'left', 'right'}
    return arms - invalid, invalid


# --- Joint command helpers --------------------------------------------------

def step_towards(current, target, step_lengths):
    next_pos = current.copy()
    for idx, step_len in enumerate(step_lengths):
        diff = target[idx] - current[idx]
        if abs(diff) <= step_len:
            next_pos[idx] = target[idx]
        else:
            next_pos[idx] = current[idx] + np.sign(diff) * step_len
    return next_pos



def clip_joint_delta(current, target, max_delta):
    delta = target - current
    return current + np.clip(delta, -max_delta, max_delta)



def apply_joint_delta_deadband(command_before, target, deadband):
    if deadband is None:
        return target, np.zeros_like(target, dtype=np.float32)
    out = target.copy()
    desired_delta = out - command_before
    small = np.abs(desired_delta) < deadband
    out[small] = command_before[small]
    return out, target - out



def validate_policy_action_mode(rep_header, expected_action_mode):
    server_action_mode = rep_header.get('action_mode')
    if server_action_mode is None:
        return
    server_action_mode = str(server_action_mode).lower()
    if server_action_mode != expected_action_mode:
        raise ValueError(
            f"Policy server action_mode={server_action_mode!r} does not match "
            f"ros.action_mode={expected_action_mode!r}"
        )



# --- Gripper and per-arm target shaping -------------------------------------

def threshold_gripper_targets(target_left, target_right, close_thresholds, open_thresholds, close_values, open_values):
    out_left = target_left.copy()
    out_right = target_right.copy()
    if out_left[-1] >= open_thresholds[0]:
        out_left[-1] = open_values[0]
    elif out_left[-1] <= close_thresholds[0]:
        out_left[-1] = close_values[0]
    if out_right[-1] >= open_thresholds[1]:
        out_right[-1] = open_values[1]
    elif out_right[-1] <= close_thresholds[1]:
        out_right[-1] = close_values[1]
    return out_left, out_right



def scale_gripper_deltas(
    target_left,
    target_right,
    reference_left,
    reference_right,
    gains,
    clip_min=None,
    clip_max=None,
):
    """Scale only the gripper displacement from a request-time reference."""
    out_left = np.asarray(target_left, dtype=np.float32).copy()
    out_right = np.asarray(target_right, dtype=np.float32).copy()
    ref_left = np.asarray(reference_left, dtype=np.float32)
    ref_right = np.asarray(reference_right, dtype=np.float32)
    gains = np.asarray(gains, dtype=np.float32)
    if out_left.size == 0 or out_right.size == 0 or ref_left.size == 0 or ref_right.size == 0:
        return out_left, out_right
    if gains.shape != (2,):
        raise ValueError(f"gripper delta gains must have shape (2,), got {gains.shape}")

    out_left[-1] = ref_left[-1] + gains[0] * (out_left[-1] - ref_left[-1])
    out_right[-1] = ref_right[-1] + gains[1] * (out_right[-1] - ref_right[-1])
    if clip_min is not None:
        clip_min = np.asarray(clip_min, dtype=np.float32)
        out_left[-1] = max(out_left[-1], clip_min[0])
        out_right[-1] = max(out_right[-1], clip_min[1])
    if clip_max is not None:
        clip_max = np.asarray(clip_max, dtype=np.float32)
        out_left[-1] = min(out_left[-1], clip_max[0])
        out_right[-1] = min(out_right[-1], clip_max[1])
    return out_left, out_right



def filter_chunk_by_terminal_displacement(
    left_mat,
    right_mat,
    initial_left,
    initial_right,
    steps_to_execute,
    left_threshold,
    right_threshold,
):
    """Cancel a joint's whole chunk when its consumed endpoint has no net displacement."""
    out_left = np.asarray(left_mat, dtype=np.float32).copy()
    out_right = np.asarray(right_mat, dtype=np.float32).copy()
    initial_left = np.asarray(initial_left, dtype=np.float32)
    initial_right = np.asarray(initial_right, dtype=np.float32)
    left_threshold = np.asarray(left_threshold, dtype=np.float32)
    right_threshold = np.asarray(right_threshold, dtype=np.float32)
    if out_left.ndim != 2 or out_right.ndim != 2 or out_left.shape[0] != out_right.shape[0]:
        raise ValueError("terminal displacement filter expects left/right 2D chunks with equal horizons")
    if initial_left.shape != (out_left.shape[1],) or initial_right.shape != (out_right.shape[1],):
        raise ValueError("terminal displacement filter initial state width does not match action width")
    if left_threshold.shape != initial_left.shape or right_threshold.shape != initial_right.shape:
        raise ValueError("terminal displacement filter thresholds must match the left/right action widths")
    if out_left.shape[0] == 0:
        return out_left, out_right, np.zeros_like(initial_left), np.zeros_like(initial_right), np.zeros_like(initial_left, dtype=bool), np.zeros_like(initial_right, dtype=bool), -1

    endpoint_index = min(max(int(steps_to_execute), 1), out_left.shape[0]) - 1
    left_delta = out_left[endpoint_index] - initial_left
    right_delta = out_right[endpoint_index] - initial_right
    cancel_left = np.abs(left_delta) < left_threshold
    cancel_right = np.abs(right_delta) < right_threshold
    out_left[:, cancel_left] = initial_left[cancel_left]
    out_right[:, cancel_right] = initial_right[cancel_right]
    return out_left, out_right, left_delta, right_delta, cancel_left, cancel_right, endpoint_index



def override_arm_delta_from_initial_pose(
    target_left,
    target_right,
    initial_left,
    initial_right,
    override_left,
    override_right,
    include_gripper=False,
):
    """Make the selected arm's command delta from its chunk-start pose exactly zero."""
    out_left = np.asarray(target_left, dtype=np.float32).copy()
    out_right = np.asarray(target_right, dtype=np.float32).copy()
    initial_left = np.asarray(initial_left, dtype=np.float32)
    initial_right = np.asarray(initial_right, dtype=np.float32)
    if out_left.shape != initial_left.shape or out_right.shape != initial_right.shape:
        raise ValueError("initial-pose delta override target and initial-state widths must match")

    stop = out_left.shape[0] if include_gripper else max(out_left.shape[0] - 1, 0)
    if override_left:
        out_left[:stop] = initial_left[:stop]
    if override_right:
        out_right[:stop] = initial_right[:stop]
    return out_left, out_right



def scale_action_delta_from_reference(
    target_left,
    target_right,
    reference_left,
    reference_right,
    coefficient,
    include_gripper=False,
):
    """Scale an action delta from a reference while optionally preserving grippers."""
    out_left = np.asarray(target_left, dtype=np.float32).copy()
    out_right = np.asarray(target_right, dtype=np.float32).copy()
    reference_left = np.asarray(reference_left, dtype=np.float32)
    reference_right = np.asarray(reference_right, dtype=np.float32)
    if out_left.shape != reference_left.shape or out_right.shape != reference_right.shape:
        raise ValueError("action-delta scale target and reference widths must match")

    coefficient = float(coefficient)
    left_stop = out_left.shape[0] if include_gripper else max(out_left.shape[0] - 1, 0)
    right_stop = out_right.shape[0] if include_gripper else max(out_right.shape[0] - 1, 0)
    out_left[:left_stop] = (
        reference_left[:left_stop]
        + coefficient * (out_left[:left_stop] - reference_left[:left_stop])
    )
    out_right[:right_stop] = (
        reference_right[:right_stop]
        + coefficient * (out_right[:right_stop] - reference_right[:right_stop])
    )
    return out_left, out_right



# --- Chunk smoothing / filtering --------------------------------------------

def smooth_action_chunk_savgol(mat, upsample_factor=2, window_length=21, polyorder=3):
    """Apply the DreamZero paper's cubic-upsample/Savitzky-Golay smoothing."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2 or mat.shape[0] < 2:
        return mat.copy()

    upsample_factor = max(int(upsample_factor), 1)
    polyorder = max(int(polyorder), 0)
    source_time = np.arange(mat.shape[0], dtype=np.float64)
    upsampled_time = np.linspace(
        0.0,
        float(mat.shape[0] - 1),
        mat.shape[0] * upsample_factor,
        dtype=np.float64,
    )
    upsampled = CubicSpline(source_time, mat, axis=0)(upsampled_time)

    max_window = upsampled.shape[0] if upsampled.shape[0] % 2 == 1 else upsampled.shape[0] - 1
    window_length = min(max(int(window_length), polyorder + 2), max_window)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length <= polyorder:
        return mat.copy()

    smoothed = savgol_filter(
        upsampled,
        window_length=window_length,
        polyorder=polyorder,
        axis=0,
        mode='interp',
    )
    return CubicSpline(upsampled_time, smoothed, axis=0)(source_time).astype(np.float32)



def smooth_dual_action_chunks_savgol(left_mat, right_mat, upsample_factor=2, window_length=21, polyorder=3):
    """Smooth both arms in one vectorized pass without coupling their columns."""
    left_mat = np.asarray(left_mat, dtype=np.float32)
    right_mat = np.asarray(right_mat, dtype=np.float32)
    if left_mat.ndim != 2 or right_mat.ndim != 2 or left_mat.shape[0] != right_mat.shape[0]:
        return left_mat.copy(), right_mat.copy()

    left_width = left_mat.shape[1]
    combined = np.concatenate((left_mat, right_mat), axis=1)
    smoothed = smooth_action_chunk_savgol(
        combined,
        upsample_factor=upsample_factor,
        window_length=window_length,
        polyorder=polyorder,
    )
    return smoothed[:, :left_width].copy(), smoothed[:, left_width:].copy()



def lowpass_action_chunk_zero_phase(
    mat,
    sample_rate_hz,
    cutoff_hz,
    order=4,
    preserve_endpoints=True,
    include_gripper=False,
):
    """Remove chunk-wide high-frequency motion without phase delay."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError("chunk low-pass input must be a 2D action matrix")
    if mat.shape[0] < 3 or mat.shape[1] == 0:
        return mat.copy()

    sample_rate_hz = float(sample_rate_hz)
    cutoff_hz = float(cutoff_hz)
    order = max(int(order), 1)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("chunk low-pass sample_rate_hz must be positive")
    nyquist_hz = 0.5 * sample_rate_hz
    if (
        not np.isfinite(cutoff_hz)
        or cutoff_hz <= 0.0
        or cutoff_hz >= nyquist_hz
    ):
        raise ValueError(
            "chunk low-pass cutoff_hz must be between 0 and Nyquist "
            f"({nyquist_hz:.6g} Hz), got {cutoff_hz}"
        )

    stop = mat.shape[1] if include_gripper else max(mat.shape[1] - 1, 0)
    if stop == 0:
        return mat.copy()
    source = mat[:, :stop].astype(np.float64, copy=False)
    sos = butter(order, cutoff_hz, btype='lowpass', fs=sample_rate_hz, output='sos')
    # Bound padding by the short action horizon instead of relying on scipy's
    # default, which can reject otherwise valid small chunks.
    padlen = min(3 * (2 * len(sos) + 1), mat.shape[0] - 1)
    filtered = sosfiltfilt(sos, source, axis=0, padlen=padlen)

    if preserve_endpoints:
        progress = np.linspace(0.0, 1.0, mat.shape[0], dtype=np.float64)[:, None]
        filtered += (
            (1.0 - progress) * (source[:1] - filtered[:1])
            + progress * (source[-1:] - filtered[-1:])
        )

    if not np.all(np.isfinite(filtered)):
        raise ValueError("chunk low-pass produced non-finite values")
    out = mat.copy()
    out[:, :stop] = filtered.astype(np.float32)
    return out



def lowpass_dual_action_chunks_zero_phase(
    left_mat,
    right_mat,
    sample_rate_hz,
    cutoff_hz,
    order=4,
    preserve_endpoints=True,
    include_gripper=False,
):
    """Apply the same zero-phase low-pass independently to both arms."""
    left_mat = np.asarray(left_mat, dtype=np.float32)
    right_mat = np.asarray(right_mat, dtype=np.float32)
    if (
        left_mat.ndim != 2
        or right_mat.ndim != 2
        or left_mat.shape[0] != right_mat.shape[0]
    ):
        raise ValueError("left/right low-pass chunks must be aligned 2D matrices")
    # Each arm owns a gripper column, so filter them separately to exclude both.
    left_out = lowpass_action_chunk_zero_phase(
        left_mat, sample_rate_hz, cutoff_hz, order,
        preserve_endpoints, include_gripper,
    )
    right_out = lowpass_action_chunk_zero_phase(
        right_mat, sample_rate_hz, cutoff_hz, order,
        preserve_endpoints, include_gripper,
    )
    return left_out, right_out



def _isotonic_nondecreasing_l2(values):
    """Least-squares projection onto a nondecreasing sequence (PAVA)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("isotonic projection input must be one-dimensional")
    block_values = []
    block_weights = []
    block_counts = []
    for value in values:
        block_values.append(float(value))
        block_weights.append(1.0)
        block_counts.append(1)
        while (
            len(block_values) >= 2
            and block_values[-2] > block_values[-1]
        ):
            weight = block_weights[-2] + block_weights[-1]
            block_values[-2] = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / weight
            block_weights[-2] = weight
            block_counts[-2] += block_counts[-1]
            block_values.pop()
            block_weights.pop()
            block_counts.pop()
    return np.concatenate([
        np.full(count, value, dtype=np.float64)
        for value, count in zip(block_values, block_counts)
    ])



def project_action_chunk_monotonic_to_endpoint(
    mat,
    start,
    strength=1.0,
    min_terminal_delta=1e-4,
    include_gripper=False,
):
    """Suppress per-joint reversals while preserving the chunk endpoint."""
    mat = np.asarray(mat, dtype=np.float32)
    start = np.asarray(start, dtype=np.float32)
    if mat.ndim != 2 or start.shape != (mat.shape[1],):
        raise ValueError("monotonic chunk input/start shape mismatch")
    if mat.shape[0] == 0:
        return mat.copy()
    strength = float(strength)
    min_terminal_delta = max(float(min_terminal_delta), 0.0)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("monotonic chunk strength must be in [0, 1]")
    stop = mat.shape[1] if include_gripper else max(mat.shape[1] - 1, 0)
    projected = mat.astype(np.float64, copy=True)
    source = mat.astype(np.float64, copy=False)
    start64 = start.astype(np.float64, copy=False)
    for axis in range(stop):
        endpoint = source[-1, axis]
        terminal_delta = endpoint - start64[axis]
        if abs(terminal_delta) <= min_terminal_delta:
            projected[:, axis] = endpoint
            continue
        direction = 1.0 if terminal_delta > 0.0 else -1.0
        terminal_progress = abs(terminal_delta)
        progress = direction * (
            np.concatenate(([start64[axis]], source[:, axis])) - start64[axis]
        )
        progress = np.clip(progress, 0.0, terminal_progress)
        monotonic_progress = _isotonic_nondecreasing_l2(progress)
        projected[:, axis] = (
            start64[axis] + direction * monotonic_progress[1:]
        )
        # Endpoint preservation is exact, independent of floating-point pooling.
        projected[-1, axis] = endpoint
    out = source + strength * (projected - source)
    out[-1, :stop] = source[-1, :stop]
    if not np.all(np.isfinite(out)):
        raise ValueError("monotonic chunk projection produced non-finite values")
    return out.astype(np.float32)



# --- Chunk interpolation ----------------------------------------------------

def linear_upsample_chunk(mat, factor):
    mat = np.asarray(mat, dtype=np.float32)
    if factor <= 1.0 or mat.shape[0] <= 1:
        return mat

    out_steps = max(int(round(mat.shape[0] * factor)), mat.shape[0])
    src_pos = np.arange(out_steps, dtype=np.float32) / float(factor)
    src_pos = np.clip(src_pos, 0.0, float(mat.shape[0] - 1))
    lo = np.floor(src_pos).astype(np.int64)
    hi = np.minimum(lo + 1, mat.shape[0] - 1)
    alpha = (src_pos - lo).astype(np.float32)[:, None]
    return (1.0 - alpha) * mat[lo] + alpha * mat[hi]



def interpolate_with_segment_factors(mat, segment_factors):
    mat = np.asarray(mat, dtype=np.float32)
    if mat.shape[0] <= 1:
        return mat.copy(), [0]

    out = [mat[0]]
    source_to_expanded = [0]
    for idx, factor in enumerate(segment_factors):
        factor = max(int(factor), 1)
        start = mat[idx]
        end = mat[idx + 1]
        for step in range(1, factor + 1):
            alpha = float(step) / float(factor)
            out.append((1.0 - alpha) * start + alpha * end)
        source_to_expanded.append(len(out) - 1)
    return np.asarray(out, dtype=np.float32), source_to_expanded



def split_joint_threshold(joint_threshold, left_dim, right_dim):
    threshold = np.asarray(joint_threshold, dtype=np.float32)
    if threshold.shape[0] == left_dim + right_dim:
        threshold_left = threshold[:left_dim]
        threshold_right = threshold[left_dim:]
    elif threshold.shape[0] == left_dim:
        threshold_left = threshold
        threshold_right = threshold
    else:
        raise ValueError(
            "chunk_interpolation.joint_threshold must have 7 values or 14 values "
            f"for the current action shape, got {threshold.shape[0]}"
        )
    return np.maximum(threshold_left, 1e-6), np.maximum(threshold_right, 1e-6)



def adaptive_delta_upsample_chunks(left_mat, right_mat, min_factor, max_factor, joint_threshold):
    left_mat = np.asarray(left_mat, dtype=np.float32)
    right_mat = np.asarray(right_mat, dtype=np.float32)
    if left_mat.shape[0] <= 1:
        return left_mat.copy(), right_mat.copy(), [0], []

    threshold_left, threshold_right = split_joint_threshold(
        joint_threshold, left_mat.shape[1], right_mat.shape[1]
    )

    segment_factors = []
    for idx in range(left_mat.shape[0] - 1):
        left_score = np.max(np.abs(left_mat[idx + 1] - left_mat[idx]) / threshold_left)
        right_score = np.max(np.abs(right_mat[idx + 1] - right_mat[idx]) / threshold_right)
        factor = int(np.ceil(max(float(left_score), float(right_score))))
        factor = min(max(factor, int(min_factor)), int(max_factor))
        segment_factors.append(max(factor, 1))

    left_out, source_to_expanded = interpolate_with_segment_factors(left_mat, segment_factors)
    right_out, _ = interpolate_with_segment_factors(right_mat, segment_factors)
    return left_out, right_out, source_to_expanded, segment_factors



def interpolate_arm_command_keep_gripper(start, target, alpha):
    """Linearly interpolate arm axes while applying the gripper target immediately."""
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if start.shape != target.shape:
        raise ValueError("command interpolation endpoints must have matching shapes")
    out = target.copy()
    if out.shape[0] > 1:
        alpha = float(np.clip(alpha, 0.0, 1.0))
        out[:-1] = start[:-1] + alpha * (target[:-1] - start[:-1])
    return out



def adaptive_bridge_to_first_action(
    left_mat,
    right_mat,
    command_left,
    command_right,
    min_factor,
    max_factor,
    joint_threshold,
    source_to_expanded,
):
    if left_mat.shape[0] == 0:
        return left_mat, right_mat, source_to_expanded, 1

    threshold_left, threshold_right = split_joint_threshold(
        joint_threshold,
        left_mat.shape[1],
        right_mat.shape[1],
    )
    left_score = np.max(np.abs(left_mat[0] - command_left) / threshold_left)
    right_score = np.max(np.abs(right_mat[0] - command_right) / threshold_right)
    factor = int(np.ceil(max(float(left_score), float(right_score))))
    factor = min(max(factor, int(min_factor)), int(max_factor))
    factor = max(factor, 1)
    if factor <= 1:
        return left_mat, right_mat, source_to_expanded, factor

    left_bridge = []
    right_bridge = []
    for step in range(1, factor):
        alpha = float(step) / float(factor)
        left_step = interpolate_arm_command_keep_gripper(
            command_left, left_mat[0], alpha
        )
        right_step = interpolate_arm_command_keep_gripper(
            command_right, right_mat[0], alpha
        )
        left_bridge.append(left_step)
        right_bridge.append(right_step)
    left_out = np.vstack([np.asarray(left_bridge, dtype=np.float32), left_mat])
    right_out = np.vstack([np.asarray(right_bridge, dtype=np.float32), right_mat])
    offset = factor - 1
    source_to_expanded = [idx + offset for idx in source_to_expanded]
    return left_out, right_out, source_to_expanded, factor



# --- Temporal ensemble ------------------------------------------------------

def exponential_temporal_ensemble(
    chunk_history,
    action_step,
    current_request_id,
    decay,
    max_candidate_age,
    min_candidate_action_index=0,
):
    candidates = []
    for chunk in chunk_history:
        rel_step = action_step - chunk['start_step']
        if rel_step < 0 or rel_step >= chunk['left'].shape[0]:
            continue
        if rel_step < min_candidate_action_index:
            continue
        age = current_request_id - chunk['request_id']
        if max_candidate_age is not None and age > max_candidate_age:
            continue
        weight = float(np.exp(-decay * max(age, 0)))
        candidates.append((weight, chunk['left'][rel_step], chunk['right'][rel_step]))

    if not candidates:
        return None, None, 0, []

    weights = np.asarray([item[0] for item in candidates], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-8)
    left = np.zeros_like(candidates[0][1], dtype=np.float32)
    right = np.zeros_like(candidates[0][2], dtype=np.float32)
    for weight, (_, left_candidate, right_candidate) in zip(weights, candidates):
        left += weight * left_candidate
        right += weight * right_candidate
    return left, right, len(candidates), weights.tolist()

