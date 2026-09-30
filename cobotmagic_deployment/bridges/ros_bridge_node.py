#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic side bridge (Python 3.8).
- Collects observations from ROS topics
- Sends compact multipart messages to the policy server via ZeroMQ (REQ)
- Receives actions (binary float32 arrays) and publishes to ROS
- Designed for minimal overhead between separate conda environments
"""

import argparse
import csv
import glob
import sys
import json
import os
import threading
import time

import cv2
import numpy as np
import yaml
import zmq
from cobotmagic_deployment.common.gripper_hysteresis import GripperHysteresis
from cobotmagic_deployment.common.piper_ik import (
    PiperNumericalIK, rpy_xyz_to_matrix, axis_angle_to_matrix,
    transform_from_xyz_rpy, transform_from_pose_command, rotation_error_vector,
    parse_float_triplet,
)
from scipy.interpolate import CubicSpline
from scipy.signal import butter, savgol_filter, sosfiltfilt

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
try:
    from piper_msgs.msg import PosCmd
except Exception:  # noqa: BLE001
    for _p in glob.glob("/workspace/ros_cobotmagic/Piper_ros_private-ros-noetic/devel/lib/python*/dist-packages"):
        if _p not in sys.path:
            sys.path.append(_p)
    try:
        from piper_msgs.msg import PosCmd
    except Exception:  # noqa: BLE001
        PosCmd = None
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool

bridge = CvBridge()

# shared buffers
buf = {
    'front': None,
    'left': None,
    'right': None,
    'jl': None,
    'jr': None,
    'odom': None,
    'eef_l': None,
    'eef_r': None,
}
buf_seq = {key: 0 for key in buf}
buf_time = {key: None for key in buf}
lock = threading.Lock()
enable_state = True  # /enable_flag があればこれに従う
published_first_command = False


def encode_jpeg(img, quality=80):
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return enc.tobytes()


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


def eef_pose_msg_to_list(msg):
    return [
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
        float(msg.pose.orientation.x),
        float(msg.pose.orientation.y),
        float(msg.pose.orientation.z),
        float(msg.pose.orientation.w),
    ]


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


def img_cb(which, mode='raw', quality=80):
    use_compressed = (mode == 'compressed')

    def f(msg):
        try:
            if use_compressed:
                data = bytes(msg.data)
                # best effort verification (optional)
                if not data:
                    return
                payload = data
            else:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                payload = encode_jpeg(img, quality=quality)
                if payload is None:
                    return
            with lock:
                buf[which] = payload
                buf_seq[which] += 1
                buf_time[which] = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"image cb error {which}: {exc}")

    return f


def jl_cb(msg: JointState):
    with lock:
        buf['jl'] = list(msg.position)
        buf_seq['jl'] += 1
        buf_time['jl'] = time.monotonic()


def jr_cb(msg: JointState):
    with lock:
        buf['jr'] = list(msg.position)
        buf_seq['jr'] += 1
        buf_time['jr'] = time.monotonic()


def eef_left_cb(msg: PoseStamped):
    with lock:
        buf['eef_l'] = eef_pose_msg_to_list(msg)
        buf_seq['eef_l'] += 1
        buf_time['eef_l'] = time.monotonic()


def eef_right_cb(msg: PoseStamped):
    with lock:
        buf['eef_r'] = eef_pose_msg_to_list(msg)
        buf_seq['eef_r'] += 1
        buf_time['eef_r'] = time.monotonic()


def odom_cb(msg: Odometry):
    with lock:
        buf['odom'] = [msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        buf_seq['odom'] += 1
        buf_time['odom'] = time.monotonic()


def enable_cb(msg: Bool):
    global enable_state
    enable_state = bool(msg.data)


def have_obs(use_base=False, require_eef=False):
    with lock:
        required = ['front', 'left', 'right', 'jl', 'jr']
        if require_eef:
            required.extend(['eef_l', 'eef_r'])
        ok = all(buf[key] is not None for key in required)
        if use_base:
            ok = ok and (buf['odom'] is not None)
        return ok


def snapshot(task_prompt: str, use_base=False, include_eef=False):
    """Capture a coherent observation while keeping callback lock time minimal."""
    with lock:
        required = ['front', 'left', 'right', 'jl', 'jr']
        if include_eef:
            required.extend(['eef_l', 'eef_r'])
        if not all(buf[key] is not None for key in required):
            return None
        front = bytes(buf['front'])
        left = bytes(buf['left'])
        right = bytes(buf['right'])
        jleft = list(buf['jl'])
        jright = list(buf['jr'])
        eef_left = list(buf['eef_l']) if include_eef else None
        eef_right = list(buf['eef_r']) if include_eef else None
        odom = list(buf['odom']) if use_base and buf['odom'] is not None else None
        obs_seq = dict(buf_seq)
        obs_time = dict(buf_time)

    pkt = {
        'task_prompt': task_prompt,
        'front': front,
        'left': left,
        'right': right,
        'jleft': jleft,
        'jright': jright,
        'obs_seq': obs_seq,
        'obs_time': obs_time,
    }
    if include_eef:
        left_gripper = float(jleft[-1])
        right_gripper = float(jright[-1])
        current_eef_left = eef_pose_and_gripper_to_command(eef_left, left_gripper)
        current_eef_right = eef_pose_and_gripper_to_command(eef_right, right_gripper)
        pkt['current_eef_left'] = current_eef_left.tolist()
        pkt['current_eef_right'] = current_eef_right.tolist()
        pkt['xvla_proprio'] = np.concatenate([
            eef_pose_and_gripper_to_ee6d(eef_left, left_gripper),
            eef_pose_and_gripper_to_ee6d(eef_right, right_gripper),
        ]).astype(np.float32).tolist()
    else:
        pkt['current_eef_left'] = None
        pkt['current_eef_right'] = None
        pkt['xvla_proprio'] = None
    pkt['odom'] = odom
    return pkt


def latest_joint_arrays():
    with lock:
        jl = None if buf['jl'] is None else np.array(buf['jl'], dtype=np.float32)
        jr = None if buf['jr'] is None else np.array(buf['jr'], dtype=np.float32)
    return jl, jr


def latest_joint_arrays_and_age():
    now = time.monotonic()
    with lock:
        jl = None if buf['jl'] is None else np.array(buf['jl'], dtype=np.float32)
        jr = None if buf['jr'] is None else np.array(buf['jr'], dtype=np.float32)
        jl_age = None if buf_time['jl'] is None else now - buf_time['jl']
        jr_age = None if buf_time['jr'] is None else now - buf_time['jr']
    return jl, jr, jl_age, jr_age


def wait_for_joint_arrays(timeout_sec=10.0):
    deadline = time.monotonic() + max(timeout_sec, 0.1)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        with lock:
            jl = buf['jl']
            jr = buf['jr']
        if jl is not None and jr is not None:
            return np.array(jl, dtype=np.float32), np.array(jr, dtype=np.float32)
        if time.monotonic() > deadline:
            break
        rate.sleep()
    return None, None


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


def adaptive_delta_upsample_chunks(left_mat, right_mat, min_factor, max_factor, joint_threshold):
    left_mat = np.asarray(left_mat, dtype=np.float32)
    right_mat = np.asarray(right_mat, dtype=np.float32)
    if left_mat.shape[0] <= 1:
        return left_mat.copy(), right_mat.copy(), [0], []

    threshold = np.asarray(joint_threshold, dtype=np.float32)
    if threshold.shape[0] == left_mat.shape[1] + right_mat.shape[1]:
        threshold_left = threshold[: left_mat.shape[1]]
        threshold_right = threshold[left_mat.shape[1] :]
    elif threshold.shape[0] == left_mat.shape[1]:
        threshold_left = threshold
        threshold_right = threshold
    else:
        raise ValueError(
            "chunk_interpolation.joint_threshold must have 7 values or 14 values "
            f"for the current action shape, got {threshold.shape[0]}"
        )
    threshold_left = np.maximum(threshold_left, 1e-6)
    threshold_right = np.maximum(threshold_right, 1e-6)

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


def vector_to_json(values):
    if values is None:
        return ''
    return json.dumps(np.asarray(values, dtype=np.float32).tolist(), separators=(',', ':'))


def norm_first_six(values):
    if values is None:
        return ''
    arr = np.asarray(values, dtype=np.float32)
    return f'{float(np.linalg.norm(arr[:6])):.6f}'


def make_action_logger(cfg, rate_hz):
    log_cfg = cfg['ros'].get('action_log', {})
    if not bool(log_cfg.get('enabled', False)):
        return None, None

    log_dir = os.path.expanduser(log_cfg.get('dir', 'logs/action_commands'))
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(log_dir, f'action_commands_{stamp}.csv')
    flush_every = max(int(log_cfg.get('flush_every_rows', 1)), 1)
    rich_cfg = log_cfg.get('rich', {})
    rich_enabled = bool(rich_cfg.get('enabled', False))
    save_request_images = rich_enabled and bool(rich_cfg.get('save_request_images', True))
    save_action_chunks = rich_enabled and bool(rich_cfg.get('save_action_chunks', True))
    request_snapshot_every_n = max(int(rich_cfg.get('request_snapshot_every_n', 1)), 1)
    request_snapshot_dir = None
    if save_request_images:
        request_snapshot_dir = os.path.join(log_dir, f'request_snapshots_{stamp}')
        os.makedirs(request_snapshot_dir, exist_ok=True)
    action_chunk_dir = None
    if save_action_chunks:
        action_chunk_dir = os.path.join(log_dir, f'action_chunks_{stamp}')
        os.makedirs(action_chunk_dir, exist_ok=True)

    columns = [
        'wall_time',
        'ros_time',
        'request_id',
        'tau',
        'global_action_step',
        'chunk_start_step',
        'action_index',
        'policy_latency_sec',
        'policy_latency_steps',
        'chunk_size',
        'received_chunk_size',
        'steps_to_execute',
        'discarded_steps',
        'rate_hz',
        'command_publish_rate_hz',
        'command_publish_substeps',
        'task_prompt',
        'action_mode',
        'chunk_interpolation_enabled',
        'chunk_interpolation_mode',
        'chunk_interpolation_factor',
        'request_obs_seq',
        'request_obs_age_sec',
        'request_snapshot_dir',
        'action_chunk_path',
        'rollout_sample_dir',
        'rollout_action_path',
        'request_fresh_camera_count',
        'request_fresh_camera_forced',
        'delta_clip_enabled',
        'delta_clip_reference',
        'command_delta_deadband_enabled',
        'first_action_delta_scale_enabled',
        'first_action_delta_scale_coefficient',
        'first_action_delta_scale_include_gripper',
        'first_action_delta_scale_applied',
        'initial_pose_delta_override_enabled',
        'initial_pose_delta_override_arms',
        'initial_pose_delta_override_include_gripper',
        'temporal_ensemble_count',
        'temporal_ensemble_weights',
        'current_left',
        'current_right',
        'request_measured_left',
        'request_measured_right',
        'publish_current_left',
        'publish_current_right',
        'clip_reference_left',
        'clip_reference_right',
        'command_left_before',
        'command_right_before',
        'raw_left',
        'raw_right',
        'ensembled_left',
        'ensembled_right',
        'filtered_left',
        'filtered_right',
        'target_left',
        'target_right',
        'ik_seed_joint_left',
        'ik_seed_joint_right',
        'ik_diagnostics_left',
        'ik_diagnostics_right',
        'ik_joint_left',
        'ik_joint_right',
        'ik_left_position_error_m',
        'ik_right_position_error_m',
        'ik_left_orientation_error_rad',
        'ik_right_orientation_error_rad',
        'raw_delta_left',
        'raw_delta_right',
        'applied_delta_left',
        'applied_delta_right',
        'reference_delta_left',
        'reference_delta_right',
        'tracking_error_before_left',
        'tracking_error_before_right',
        'tracking_error_after_left',
        'tracking_error_after_right',
        'clip_residual_left',
        'clip_residual_right',
        'command_delta_deadband_residual_left',
        'command_delta_deadband_residual_right',
        'raw_vs_publish_norm_left',
        'raw_vs_publish_norm_right',
        'ensembled_vs_publish_norm_left',
        'ensembled_vs_publish_norm_right',
        'target_vs_publish_norm_left',
        'target_vs_publish_norm_right',
        'reference_delta_norm_left',
        'reference_delta_norm_right',
        'applied_delta_norm_left',
        'applied_delta_norm_right',
        'tracking_error_before_norm_left',
        'tracking_error_before_norm_right',
        'tracking_error_after_norm_left',
        'tracking_error_after_norm_right',
        'clip_residual_norm_left',
        'clip_residual_norm_right',
        'command_delta_deadband_residual_norm_left',
        'command_delta_deadband_residual_norm_right',
        'publish_joint_age_left_sec',
        'publish_joint_age_right_sec',
        'policy_gripper_input_mode',
        'left_gripper_request_measured',
        'left_gripper_request_policy',
        'left_gripper_request_commanded',
        'right_gripper_request_measured',
        'right_gripper_request_policy',
        'right_gripper_request_commanded',
        'left_gripper_publish',
        'left_gripper_raw',
        'left_gripper_ensembled',
        'left_gripper_target',
        'right_gripper_publish',
        'right_gripper_raw',
        'right_gripper_ensembled',
        'right_gripper_target',
        'gripper_hysteresis',
        'base_vel',
    ]
    f = open(path, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    rospy.loginfo(f"Action command logging enabled: {path} (flush_every_rows={flush_every})")
    return {
        'file': f,
        'writer': writer,
        'path': path,
        'flush_every': flush_every,
        'rows_since_flush': 0,
        'rich_enabled': rich_enabled,
        'save_request_images': save_request_images,
        'save_action_chunks': save_action_chunks,
        'request_snapshot_every_n': request_snapshot_every_n,
        'request_snapshot_dir': request_snapshot_dir,
        'action_chunk_dir': action_chunk_dir,
        'request_snapshot_count': 0,
    }, path


def write_action_log(logger, row):
    if logger is None:
        return
    logger['writer'].writerow(row)
    logger['rows_since_flush'] += 1
    if logger['rows_since_flush'] >= logger['flush_every']:
        logger['file'].flush()
        logger['rows_since_flush'] = 0


def save_request_snapshot(logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback):
    if logger is None or not logger.get('save_request_images'):
        return ''
    logger['request_snapshot_count'] += 1
    if (logger['request_snapshot_count'] - 1) % logger['request_snapshot_every_n'] != 0:
        return ''

    root = logger.get('request_snapshot_dir')
    if not root:
        return ''
    name = f"step_{int(request_step):06d}_req_{logger['request_snapshot_count']:06d}"
    out_dir = os.path.join(root, name)
    os.makedirs(out_dir, exist_ok=True)
    for key in ('front', 'left', 'right'):
        with open(os.path.join(out_dir, f'{key}.jpg'), 'wb') as f:
            f.write(pkt[key])
    now = time.monotonic()
    meta = {
        'wall_time': time.time(),
        'request_step': int(request_step),
        'fresh_camera_count': int(fresh_camera_count),
        'forced_fresh_fallback': bool(forced_fresh_fallback),
        'header': header,
        'obs_seq': dict(pkt.get('obs_seq', {})),
        'obs_age_sec': {
            key: None if value is None else now - value
            for key, value in pkt.get('obs_time', {}).items()
        },
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return out_dir


def save_action_chunk(logger, request_id, chunk):
    if logger is None or not logger.get('save_action_chunks'):
        return ''
    root = logger.get('action_chunk_dir')
    if not root:
        return ''
    path = os.path.join(root, f"request_{int(request_id):06d}.npz")
    meta = {
        'request_id': int(request_id),
        'start_step': int(chunk['start_step']),
        'received_chunk_size': int(chunk['received_chunk_size']),
        'processed_chunk_size': int(chunk['left'].shape[0]),
        'steps_to_execute': int(chunk['steps_to_execute']),
        'overlap_steps': None if chunk['overlap_steps'] is None else int(chunk['overlap_steps']),
        'policy_latency_sec': float(chunk['policy_latency_sec']),
        'policy_latency_steps': int(chunk['policy_latency_steps']),
        'request_fresh_camera_count': int(chunk['request_fresh_camera_count']),
        'request_fresh_camera_forced': bool(chunk['request_fresh_camera_forced']),
        'request_obs_seq': dict(chunk['request_obs_seq']),
        'request_snapshot_dir': chunk.get('request_snapshot_dir', ''),
        'policy_gripper_input': dict(chunk.get('policy_gripper_input', {})),
        'terminal_filter_enabled': bool(chunk.get('terminal_filter_enabled', False)),
        'terminal_endpoint_index': int(chunk.get('terminal_endpoint_index', -1)),
        'terminal_delta_left': np.asarray(chunk.get('terminal_delta_left', []), dtype=np.float32).tolist(),
        'terminal_delta_right': np.asarray(chunk.get('terminal_delta_right', []), dtype=np.float32).tolist(),
        'terminal_cancel_left': np.asarray(chunk.get('terminal_cancel_left', []), dtype=bool).tolist(),
        'terminal_cancel_right': np.asarray(chunk.get('terminal_cancel_right', []), dtype=bool).tolist(),
        'initial_pose_delta_override_enabled': bool(
            chunk.get('initial_pose_delta_override_enabled', False)
        ),
        'initial_pose_delta_override_arms': list(
            chunk.get('initial_pose_delta_override_arms', [])
        ),
        'initial_pose_delta_override_include_gripper': bool(
            chunk.get('initial_pose_delta_override_include_gripper', False)
        ),
        'first_action_delta_scale_enabled': bool(
            chunk.get('first_action_delta_scale_enabled', False)
        ),
        'first_action_delta_scale_coefficient': float(
            chunk.get('first_action_delta_scale_coefficient', 1.0)
        ),
        'first_action_delta_scale_include_gripper': bool(
            chunk.get('first_action_delta_scale_include_gripper', False)
        ),
        'initial_bridge_factor': int(chunk.get('initial_bridge_factor', 1)),
        'initial_bridge_added_steps': int(
            chunk.get('initial_bridge_added_steps', 0)
        ),
        'chunk_lowpass_enabled': bool(chunk.get('chunk_lowpass_enabled', False)),
        'chunk_lowpass_cutoff_hz': float(
            chunk.get('chunk_lowpass_cutoff_hz', 0.0)
        ),
        'chunk_lowpass_sample_rate_hz': float(
            chunk.get('chunk_lowpass_sample_rate_hz', 0.0)
        ),
        'chunk_lowpass_order': int(chunk.get('chunk_lowpass_order', 0)),
        'chunk_lowpass_preserve_endpoints': bool(
            chunk.get('chunk_lowpass_preserve_endpoints', False)
        ),
        'chunk_monotonic_enabled': bool(
            chunk.get('chunk_monotonic_enabled', False)
        ),
        'chunk_monotonic_arms': list(chunk.get('chunk_monotonic_arms', [])),
        'chunk_monotonic_strength': float(
            chunk.get('chunk_monotonic_strength', 0.0)
        ),
        'chunk_monotonic_min_terminal_delta': float(
            chunk.get('chunk_monotonic_min_terminal_delta', 0.0)
        ),
    }
    np.savez_compressed(
        path,
        processed_left=chunk['left'],
        processed_right=chunk['right'],
        received_left=chunk.get('received_left', chunk['left']),
        received_right=chunk.get('received_right', chunk['right']),
        lowpass_left=chunk.get('lowpass_left', chunk['left']),
        lowpass_right=chunk.get('lowpass_right', chunk['right']),
        monotonic_left=chunk.get('monotonic_left', chunk['left']),
        monotonic_right=chunk.get('monotonic_right', chunk['right']),
        model_raw_left=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
        model_raw_right=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
        vel=np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
        request_current_left=chunk['request_current_left'],
        request_current_right=chunk['request_current_right'],
        request_measured_left=chunk.get('request_measured_left', chunk['request_current_left']),
        request_measured_right=chunk.get('request_measured_right', chunk['request_current_right']),
        terminal_delta_left=np.asarray(chunk.get('terminal_delta_left', []), dtype=np.float32),
        terminal_delta_right=np.asarray(chunk.get('terminal_delta_right', []), dtype=np.float32),
        terminal_cancel_left=np.asarray(chunk.get('terminal_cancel_left', []), dtype=bool),
        terminal_cancel_right=np.asarray(chunk.get('terminal_cancel_right', []), dtype=bool),
        meta=json.dumps(meta, separators=(',', ':')),
    )
    return path


def make_rollout_dataset_logger(cfg, rate_hz):
    rollout_cfg = cfg['ros'].get('rollout_dataset', {})
    if not bool(rollout_cfg.get('enabled', False)):
        return None

    root = os.path.expanduser(rollout_cfg.get('dir', '/workspace/project/cobotmagic_datasets'))
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    run_name = rollout_cfg.get('run_name') or f'rollout_{stamp}'
    episode_dir = os.path.join(root, run_name)
    suffix = 1
    while os.path.exists(episode_dir):
        episode_dir = os.path.join(root, f'{run_name}_{suffix:02d}')
        suffix += 1
    os.makedirs(episode_dir, exist_ok=False)
    dataset_format = str(rollout_cfg.get('format', 'hdf5')).lower()
    if dataset_format not in ('hdf5', 'directory'):
        rospy.logwarn(f"Unsupported rollout_dataset.format={dataset_format!r}; using 'hdf5'.")
        dataset_format = 'hdf5'
    samples_dir = os.path.join(episode_dir, 'samples')
    if dataset_format == 'directory':
        os.makedirs(samples_dir, exist_ok=False)

    meta = {
        'created_wall_time': time.time(),
        'created_stamp': stamp,
        'rate_hz': int(rate_hz),
        'task_prompt': cfg.get('task_prompt', ''),
        'policy_backend': cfg.get('policy_backend', ''),
        'openvla': cfg.get('openvla', {}),
        'ros': {
            'action_mode': cfg['ros'].get('action_mode'),
            'open_loop_steps': cfg['ros'].get('open_loop_steps'),
            'rate_hz': cfg['ros'].get('rate_hz'),
            'command_publish': cfg['ros'].get('command_publish', {}),
            'chunk_interpolation': cfg['ros'].get('chunk_interpolation', {}),
            'delta_clip': cfg['ros'].get('delta_clip', {}),
            'action_filter': cfg['ros'].get('action_filter', {}),
            'temporal_ensemble': cfg['ros'].get('temporal_ensemble', {}),
        },
        'format': {
            'type': dataset_format,
            'hdf5': 'episode.hdf5 with /samples/sample_XXXXXX groups',
            'directory': 'samples/sample_XXXXXX with jpg/json/npz files',
            'raw_action_definition': 'raw_left_chunk/raw_right_chunk are policy server outputs before bridge filters, clipping, thresholding, ensemble, and command blocking.',
        },
        'outcome': 'unknown',
    }
    with open(os.path.join(episode_dir, 'episode_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    h5_file = None
    h5_path = ''
    if dataset_format == 'hdf5':
        try:
            import h5py

            h5_path = os.path.join(episode_dir, 'episode.hdf5')
            h5_file = h5py.File(h5_path, 'w')
            h5_file.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
            h5_file.attrs['created_wall_time'] = meta['created_wall_time']
            h5_file.attrs['task_prompt'] = meta['task_prompt']
            h5_file.attrs['outcome'] = meta['outcome']
            h5_file.create_group('samples')
            h5_file.flush()
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"Failed to create rollout HDF5 file; falling back to directory format: {exc}")
            dataset_format = 'directory'
            samples_dir = os.path.join(episode_dir, 'samples')
            os.makedirs(samples_dir, exist_ok=True)

    rospy.loginfo(f"Rollout dataset logging enabled: {episode_dir} format={dataset_format}")
    return {
        'episode_dir': episode_dir,
        'samples_dir': samples_dir,
        'format': dataset_format,
        'h5_file': h5_file,
        'h5_path': h5_path,
        'sample_count': 0,
        'save_images': bool(rollout_cfg.get('save_images', True)),
    }


def h5_write_dataset(group, name, data, **kwargs):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, **kwargs)


def h5_write_json(group, name, payload):
    h5_write_dataset(group, name, np.bytes_(json.dumps(payload, separators=(',', ':'))))


def h5_write_jpeg(group, name, payload):
    h5_write_dataset(group, name, np.frombuffer(payload, dtype=np.uint8), compression='gzip')


def save_rollout_observation(logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback):
    if logger is None:
        return ''

    logger['sample_count'] += 1
    sample_name = f"sample_{logger['sample_count']:06d}"

    now = time.monotonic()
    obs_age_sec = {
        key: None if value is None else now - value
        for key, value in pkt.get('obs_time', {}).items()
    }
    meta = {
        'sample_id': logger['sample_count'],
        'wall_time': time.time(),
        'request_step': int(request_step),
        'task_prompt': pkt['task_prompt'],
        'fresh_camera_count': int(fresh_camera_count),
        'forced_fresh_fallback': bool(forced_fresh_fallback),
        'header': header,
        'obs_seq': dict(pkt.get('obs_seq', {})),
        'obs_age_sec': obs_age_sec,
    }

    if logger.get('format') == 'hdf5' and logger.get('h5_file') is not None:
        h5_file = logger['h5_file']
        group_path = f"samples/{sample_name}"
        sample_group = h5_file.create_group(group_path)
        sample_group.attrs['sample_id'] = logger['sample_count']
        sample_group.attrs['request_step'] = int(request_step)
        sample_group.attrs['wall_time'] = meta['wall_time']
        sample_group.attrs['task_prompt'] = pkt['task_prompt']
        sample_group.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
        obs_group = sample_group.create_group('observation')
        h5_write_dataset(obs_group, 'jleft', np.asarray(pkt['jleft'], dtype=np.float32))
        h5_write_dataset(obs_group, 'jright', np.asarray(pkt['jright'], dtype=np.float32))
        h5_write_dataset(
            obs_group,
            'odom',
            np.empty((0,), dtype=np.float32) if pkt.get('odom') is None else np.asarray(pkt['odom'], dtype=np.float32),
        )
        h5_write_json(obs_group, 'metadata_json', meta)
        h5_write_json(obs_group, 'obs_seq_json', pkt.get('obs_seq', {}))
        h5_write_json(obs_group, 'obs_age_sec_json', obs_age_sec)
        if logger.get('save_images', True):
            img_group = obs_group.create_group('images')
            for key in ('front', 'left', 'right'):
                h5_write_jpeg(img_group, f'{key}_jpg', pkt[key])
        h5_file.flush()
        return f"{logger['h5_path']}::/{group_path}"

    sample_dir = os.path.join(logger['samples_dir'], sample_name)
    os.makedirs(sample_dir, exist_ok=False)
    if logger.get('save_images', True):
        for key in ('front', 'left', 'right'):
            with open(os.path.join(sample_dir, f'{key}.jpg'), 'wb') as f:
                f.write(pkt[key])
    with open(os.path.join(sample_dir, 'observation.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(
        os.path.join(sample_dir, 'observation.npz'),
        jleft=np.asarray(pkt['jleft'], dtype=np.float32),
        jright=np.asarray(pkt['jright'], dtype=np.float32),
        odom=np.empty((0,), dtype=np.float32) if pkt.get('odom') is None else np.asarray(pkt['odom'], dtype=np.float32),
        obs_seq=json.dumps(pkt.get('obs_seq', {}), separators=(',', ':')),
        obs_age_sec=json.dumps(obs_age_sec, separators=(',', ':')),
    )
    return sample_dir


def save_rollout_action(logger, sample_dir, request_id, chunk):
    if logger is None or not sample_dir:
        return ''

    meta = {
        'request_id': int(request_id),
        'start_step': int(chunk['start_step']),
        'received_chunk_size': int(chunk['received_chunk_size']),
        'processed_chunk_size': int(chunk['left'].shape[0]),
        'steps_to_execute': int(chunk['steps_to_execute']),
        'overlap_steps': None if chunk['overlap_steps'] is None else int(chunk['overlap_steps']),
        'policy_latency_sec': float(chunk['policy_latency_sec']),
        'policy_latency_steps': int(chunk['policy_latency_steps']),
        'request_fresh_camera_count': int(chunk['request_fresh_camera_count']),
        'request_fresh_camera_forced': bool(chunk['request_fresh_camera_forced']),
        'raw_action_definition': (
            'raw_left_chunk/raw_right_chunk are the policy server outputs as received by the bridge, '
            'before bridge-side interpolation, filters, clipping, thresholding, temporal ensemble, '
            'right-arm blocking, and command publish logic.'
        ),
        'model_raw_action_definition': (
            'model_raw_left_chunk/model_raw_right_chunk are the direct OpenVLA action-head outputs '
            'before server-side action_delta_gripper_abs postprocessing when the server provides them; '
            'empty arrays mean the server did not provide model raw action frames.'
        ),
    }

    if logger.get('format') == 'hdf5' and logger.get('h5_file') is not None:
        marker = '::/'
        if marker not in sample_dir:
            return ''
        group_path = sample_dir.split(marker, 1)[1].lstrip('/')
        h5_file = logger['h5_file']
        sample_group = h5_file[group_path]
        action_group = sample_group.create_group('actions')
        action_group.attrs['request_id'] = int(request_id)
        action_group.attrs['metadata_json'] = json.dumps(meta, separators=(',', ':'))
        h5_write_dataset(action_group, 'raw_left_chunk', chunk.get('received_left', chunk['left']), compression='gzip')
        h5_write_dataset(action_group, 'raw_right_chunk', chunk.get('received_right', chunk['right']), compression='gzip')
        h5_write_dataset(
            action_group,
            'model_raw_left_chunk',
            np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
            compression='gzip',
        )
        h5_write_dataset(
            action_group,
            'model_raw_right_chunk',
            np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
            compression='gzip',
        )
        h5_write_dataset(action_group, 'processed_left_chunk', chunk['left'], compression='gzip')
        h5_write_dataset(action_group, 'processed_right_chunk', chunk['right'], compression='gzip')
        h5_write_dataset(
            action_group,
            'vel',
            np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
            compression='gzip',
        )
        h5_write_dataset(action_group, 'request_current_left', chunk['request_current_left'])
        h5_write_dataset(action_group, 'request_current_right', chunk['request_current_right'])
        h5_write_json(action_group, 'metadata_json', meta)
        h5_file.flush()
        return f"{logger['h5_path']}::/{group_path}/actions"

    action_path = os.path.join(sample_dir, 'actions.npz')
    with open(os.path.join(sample_dir, 'action_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    np.savez_compressed(
        action_path,
        raw_left_chunk=chunk.get('received_left', chunk['left']),
        raw_right_chunk=chunk.get('received_right', chunk['right']),
        model_raw_left_chunk=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_left') is None else chunk['model_raw_left'],
        model_raw_right_chunk=np.empty((0, 0), dtype=np.float32) if chunk.get('model_raw_right') is None else chunk['model_raw_right'],
        processed_left_chunk=chunk['left'],
        processed_right_chunk=chunk['right'],
        vel=np.empty((0, 0), dtype=np.float32) if chunk['vel'] is None else chunk['vel'],
        request_current_left=chunk['request_current_left'],
        request_current_right=chunk['request_current_right'],
        meta=json.dumps(meta, separators=(',', ':')),
    )
    return action_path


def policy_socket_kind(name):
    normalized = name.lower()
    if normalized == 'pair':
        return zmq.PAIR
    if normalized == 'req':
        return zmq.REQ
    raise ValueError(f"Unsupported zmq.socket_type={name!r}; expected 'pair' or 'req'.")


def make_policy_socket(ctx, connect_addr, timeout_ms, socket_type):
    sock = ctx.socket(policy_socket_kind(socket_type))
    sock.connect(connect_addr)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    return sock


def publish_joint_pair(pub_l, pub_r, name_list, left_pos, right_pos):
    stamp = rospy.Time.now()
    msg_left = JointState()
    msg_left.header.stamp = stamp
    msg_left.name = name_list
    msg_left.position = left_pos.tolist()
    pub_l.publish(msg_left)

    msg_right = JointState()
    msg_right.header.stamp = stamp
    msg_right.name = name_list
    msg_right.position = right_pos.tolist()
    pub_r.publish(msg_right)


def publish_eef_pair(pub_l, pub_r, left_cmd, right_cmd):
    if pub_l is None or pub_r is None:
        raise RuntimeError("EEF publishers are not initialized")
    msg_left = PosCmd()
    msg_left.x = float(left_cmd[0])
    msg_left.y = float(left_cmd[1])
    msg_left.z = float(left_cmd[2])
    msg_left.roll = float(left_cmd[3])
    msg_left.pitch = float(left_cmd[4])
    msg_left.yaw = float(left_cmd[5])
    msg_left.gripper = float(left_cmd[6])
    pub_l.publish(msg_left)

    msg_right = PosCmd()
    msg_right.x = float(right_cmd[0])
    msg_right.y = float(right_cmd[1])
    msg_right.z = float(right_cmd[2])
    msg_right.roll = float(right_cmd[3])
    msg_right.pitch = float(right_cmd[4])
    msg_right.yaw = float(right_cmd[5])
    msg_right.gripper = float(right_cmd[6])
    pub_r.publish(msg_right)


def wait_for_home_settle(
    pub_l,
    pub_r,
    name_list,
    target_left,
    target_right,
    rate_hz,
    tolerance,
    hold_sec,
    timeout_sec,
):
    rate = rospy.Rate(max(rate_hz, 1))
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    hold_start = None
    joint_dim = min(6, len(target_left), len(target_right))

    while not rospy.is_shutdown():
        publish_joint_pair(pub_l, pub_r, name_list, target_left, target_right)
        jl, jr, jl_age, jr_age = latest_joint_arrays_and_age()
        if jl is not None and jr is not None:
            left_err = float(np.max(np.abs(jl[:joint_dim] - target_left[:joint_dim])))
            right_err = float(np.max(np.abs(jr[:joint_dim] - target_right[:joint_dim])))
            joints_fresh = (
                (jl_age is None or jl_age <= 0.5)
                and (jr_age is None or jr_age <= 0.5)
            )
            if joints_fresh and left_err <= tolerance and right_err <= tolerance:
                if hold_start is None:
                    hold_start = time.monotonic()
                if time.monotonic() - hold_start >= hold_sec:
                    rospy.loginfo(
                        "Home position settled from measured joint states: "
                        f"left_max_err={left_err:.4f} right_max_err={right_err:.4f} "
                        f"tolerance={tolerance:.4f} hold_sec={hold_sec:.2f}."
                    )
                    return True
            else:
                hold_start = None

        if timeout_sec > 0.0 and time.monotonic() >= deadline:
            rospy.logwarn(
                "Timed out waiting for measured joints to settle at home; "
                "continuing with the latest observed state. "
                f"timeout_sec={timeout_sec:.2f} tolerance={tolerance:.4f}."
            )
            return False
        rate.sleep()

    return False


def move_to_home(
    pub_l,
    pub_r,
    name_list,
    target_left,
    target_right,
    step_lengths,
    rate_hz,
    mode="step",
    num_steps=None,
    settle=True,
    settle_tolerance=0.04,
    settle_hold_sec=0.5,
    settle_timeout_sec=8.0,
    settle_rate_hz=None,
    gripper_move_start_fraction=None,
):
    if gripper_move_start_fraction is not None:
        gripper_move_start_fraction = float(gripper_move_start_fraction)
        if mode != "linear" or not 0.0 <= gripper_move_start_fraction < 1.0:
            raise ValueError("gripper_move_start_fraction requires linear mode and 0 <= value < 1")
    jl, jr = wait_for_joint_arrays()
    if jl is None or jr is None:
        rospy.logwarn("home move skipped: joint state not available")
        return False

    rate = rospy.Rate(max(rate_hz, 1))
    cur_left = jl.copy()
    cur_right = jr.copy()

    if mode == "linear":
        steps = max(int(num_steps or 100), 1)
        left_traj = np.linspace(cur_left, target_left, steps, dtype=np.float32)
        right_traj = np.linspace(cur_right, target_right, steps, dtype=np.float32)
        if gripper_move_start_fraction is None:
            # Preserve the original immediate gripper target for other configs.
            left_traj[:, -1] = target_left[-1]
            right_traj[:, -1] = target_right[-1]
        else:
            progress = np.linspace(0.0, 1.0, steps) if steps > 1 else np.ones(1)
            grip_progress = np.clip(
                (progress - gripper_move_start_fraction) / (1.0 - gripper_move_start_fraction),
                0.0, 1.0,
            )
            left_traj[:, -1] = cur_left[-1] + grip_progress * (target_left[-1] - cur_left[-1])
            right_traj[:, -1] = cur_right[-1] + grip_progress * (target_right[-1] - cur_right[-1])
        for left_pos, right_pos in zip(left_traj, right_traj):
            publish_joint_pair(pub_l, pub_r, name_list, left_pos, right_pos)
            rate.sleep()
        if settle:
            return wait_for_home_settle(
                pub_l,
                pub_r,
                name_list,
                target_left,
                target_right,
                settle_rate_hz or rate_hz,
                settle_tolerance,
                settle_hold_sec,
                settle_timeout_sec,
            )
        return True

    done = False
    while not rospy.is_shutdown() and not done:
        cur_left = step_towards(cur_left, target_left, step_lengths)
        cur_right = step_towards(cur_right, target_right, step_lengths)

        publish_joint_pair(pub_l, pub_r, name_list, cur_left, cur_right)

        if np.allclose(cur_left, target_left, atol=1e-4) and np.allclose(cur_right, target_right, atol=1e-4):
            done = True
        rate.sleep()
    if settle:
        return wait_for_home_settle(
            pub_l,
            pub_r,
            name_list,
            target_left,
            target_right,
            settle_rate_hz or rate_hz,
            settle_tolerance,
            settle_hold_sec,
            settle_timeout_sec,
        )
    return True


def execute_home_phase(
    pub_l,
    pub_r,
    name_list,
    step_lengths,
    home_cfg,
    phase_cfg,
    phase_index=0,
):
    target_left = np.array(phase_cfg['left'], dtype=np.float32)
    target_right = np.array(phase_cfg['right'], dtype=np.float32)
    if target_left.shape[0] != len(name_list) or target_right.shape[0] != len(name_list):
        rospy.logwarn(f"home_position phase {phase_index} size mismatch; skipping phase.")
        return False

    phase_name = str(phase_cfg.get('name', f'phase_{phase_index}'))
    mode = str(phase_cfg.get('mode', home_cfg.get('mode', 'step'))).lower()
    if mode not in ('step', 'linear'):
        rospy.logwarn(f"Unsupported home_position phase mode={mode!r}; using 'step'.")
        mode = 'step'
    rate_hz = max(int(phase_cfg.get('rate_hz', home_cfg.get('rate_hz', 20))), 1)
    num_steps = phase_cfg.get('num_steps', home_cfg.get('num_steps'))
    settle = bool(phase_cfg.get('settle', home_cfg.get('settle', True)))
    settle_tolerance = float(phase_cfg.get('settle_tolerance', home_cfg.get('settle_tolerance', 0.04)))
    settle_hold_sec = float(phase_cfg.get('settle_hold_sec', home_cfg.get('settle_hold_sec', 0.5)))
    settle_timeout_sec = float(phase_cfg.get('settle_timeout_sec', home_cfg.get('settle_timeout_sec', 8.0)))
    settle_rate_hz = max(
        int(phase_cfg.get('settle_rate_hz', home_cfg.get('settle_rate_hz', min(rate_hz, 50)))),
        1,
    )

    rospy.loginfo(
        "Moving arms to home phase "
        f"{phase_index}:{phase_name} mode={mode} rate={rate_hz}Hz "
        f"num_steps={num_steps} settle={settle}."
    )
    return move_to_home(
        pub_l,
        pub_r,
        name_list,
        target_left,
        target_right,
        step_lengths,
        rate_hz,
        mode=mode,
        num_steps=num_steps,
        settle=settle,
        settle_tolerance=settle_tolerance,
        settle_hold_sec=settle_hold_sec,
        settle_timeout_sec=settle_timeout_sec,
        settle_rate_hz=settle_rate_hz,
        gripper_move_start_fraction=phase_cfg.get(
            'gripper_move_start_fraction', home_cfg.get('gripper_move_start_fraction')),
    )


def run_shutdown_safety(pub_l, pub_r, pub_v, enable_pub, name_list, cfg):
    safety_cfg = cfg['ros'].get('shutdown_safety', {})
    if not bool(safety_cfg.get('enabled', False)):
        return

    hold_sec = max(float(safety_cfg.get('hold_current_sec', 0.5)), 0.0)
    rate_hz = max(float(safety_cfg.get('hold_rate_hz', 50.0)), 1.0)
    publish_enable_false = bool(safety_cfg.get('publish_enable_false', False))
    zero_base_velocity = bool(safety_cfg.get('zero_base_velocity', True))

    left, right, left_age, right_age = latest_joint_arrays_and_age()
    if left is not None and right is not None:
        rospy.loginfo(
            "Shutdown safety: publishing measured current joint positions "
            f"for {hold_sec:.2f}s at {rate_hz:.1f}Hz "
            f"(joint_age_left={left_age} joint_age_right={right_age})."
        )
        deadline = time.monotonic() + hold_sec
        period = 1.0 / rate_hz
        while time.monotonic() < deadline:
            publish_joint_pair(pub_l, pub_r, name_list, left, right)
            if zero_base_velocity and pub_v is not None:
                pub_v.publish(Twist())
            time.sleep(period)
    else:
        rospy.logwarn("Shutdown safety: latest measured joints unavailable; cannot publish hold position.")

    if publish_enable_false and enable_pub is not None:
        rospy.loginfo("Shutdown safety: publishing enable flag False.")
        for _ in range(max(int(rate_hz * 0.2), 1)):
            enable_pub.publish(Bool(data=False))
            time.sleep(1.0 / rate_hz)


def main():
    global published_first_command

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to a backend-specific YAML config')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    rospy.init_node('cobotmagic_policy_bridge_node')

    image_type = cfg['ros'].get('image_type', 'raw').lower()
    jpeg_quality = int(cfg['ros'].get('jpeg_quality', 80))
    rate_hz = int(cfg['ros'].get('rate_hz', 20))
    rate_hz = max(rate_hz, 1)
    command_publish_cfg = cfg['ros'].get('command_publish', {})
    command_publish_mode = command_publish_cfg.get('mode', 'direct').lower()
    if command_publish_mode not in ('direct', 'interpolated'):
        rospy.logwarn(
            f"Unsupported command_publish.mode={command_publish_mode!r}; using ACT-style 'direct'."
        )
        command_publish_mode = 'direct'
    if command_publish_mode == 'direct':
        command_publish_rate_hz = float(rate_hz)
        command_publish_substeps = 1
    else:
        command_publish_rate_hz = float(command_publish_cfg.get('rate_hz', rate_hz))
        command_publish_rate_hz = max(command_publish_rate_hz, float(rate_hz))
        command_publish_substeps = max(int(round(command_publish_rate_hz / float(rate_hz))), 1)
        command_publish_rate_hz = float(command_publish_substeps * rate_hz)
    command_publish_interpolation = command_publish_cfg.get('interpolation', 'linear').lower()
    if command_publish_interpolation != 'linear':
        rospy.logwarn(
            f"Unsupported command_publish.interpolation={command_publish_interpolation!r}; using 'linear'."
        )
        command_publish_interpolation = 'linear'
    command_publish_period_sec = 1.0 / command_publish_rate_hz
    policy_response_timeout_sec = float(cfg['ros'].get('policy_response_timeout_sec', 5.0))
    max_joint_age_sec = float(cfg['ros'].get('max_joint_age_sec', 0.75))
    max_image_age_sec = float(cfg['ros'].get('max_image_age_sec', 2.0))
    action_mode = cfg['ros'].get('action_mode', 'velocity').lower()
    if action_mode not in ('velocity', 'absolute', 'eef_absolute'):
        raise ValueError(f"ros.action_mode must be 'velocity', 'absolute', or 'eef_absolute', got {action_mode!r}")
    eef_action_mode = action_mode == 'eef_absolute'
    if eef_action_mode and command_publish_mode != 'direct':
        rospy.logwarn("eef_absolute only supports command_publish.mode='direct'; using direct.")
        command_publish_mode = 'direct'
        command_publish_rate_hz = float(rate_hz)
        command_publish_substeps = 1
        command_publish_period_sec = 1.0 / command_publish_rate_hz
    open_loop_steps_cfg = cfg['ros'].get('open_loop_steps')
    open_loop_steps = None if open_loop_steps_cfg is None else max(int(open_loop_steps_cfg), 1)
    policy_request_cfg = cfg['ros'].get('policy_request', {})
    policy_request_when_live_chunk = str(policy_request_cfg.get('when_live_chunk', 'allow')).lower()
    if policy_request_when_live_chunk not in ('allow', 'wait'):
        rospy.logwarn(
            "policy_request.when_live_chunk must be 'allow' or 'wait'; using 'allow'."
        )
        policy_request_when_live_chunk = 'allow'
    interpolation_cfg = cfg['ros'].get('chunk_interpolation', {})
    interpolation_enabled = bool(interpolation_cfg.get('enabled', False))
    interpolation_mode = interpolation_cfg.get('mode', 'linear').lower()
    interpolation_factor = float(interpolation_cfg.get('factor', 1.0))
    interpolation_factor = max(interpolation_factor, 1.0)
    adaptive_min_factor = max(int(interpolation_cfg.get('min_factor', 1)), 1)
    adaptive_max_factor = max(int(interpolation_cfg.get('max_factor', max(1, int(round(interpolation_factor))))), 1)
    adaptive_max_factor = max(adaptive_max_factor, adaptive_min_factor)
    adaptive_source_steps_to_execute = interpolation_cfg.get('source_steps_to_execute')
    if adaptive_source_steps_to_execute is not None:
        adaptive_source_steps_to_execute = max(int(adaptive_source_steps_to_execute), 1)
    adaptive_source_overlap_steps = interpolation_cfg.get('source_overlap_steps')
    if adaptive_source_overlap_steps is not None:
        adaptive_source_overlap_steps = max(int(adaptive_source_overlap_steps), 0)
    adaptive_joint_threshold = interpolation_cfg.get('joint_threshold', [0.024, 0.069, 0.070, 0.039, 0.046, 0.046, 0.0034])
    adaptive_bridge_enabled = bool(interpolation_cfg.get('initial_bridge_enabled', False))
    adaptive_bridge_max_factor = max(int(interpolation_cfg.get('initial_bridge_max_factor', adaptive_max_factor)), 1)
    adaptive_bridge_max_factor = max(adaptive_bridge_max_factor, adaptive_min_factor)
    delta_clip_cfg = cfg['ros'].get('delta_clip')
    delta_clip_enabled = bool(delta_clip_cfg and delta_clip_cfg.get('enabled', False))
    command_delta_deadband_cfg = cfg['ros'].get('command_delta_deadband', {})
    command_delta_deadband_enabled = bool(command_delta_deadband_cfg.get('enabled', False))
    first_action_delta_scale_cfg = cfg['ros'].get('first_action_delta_scale', {})
    first_action_delta_scale_enabled = bool(first_action_delta_scale_cfg.get('enabled', False))
    first_action_delta_scale_raw = float(first_action_delta_scale_cfg.get('coefficient', 1.0))
    first_action_delta_scale_coefficient = float(np.clip(first_action_delta_scale_raw, 0.0, 1.0))
    if first_action_delta_scale_coefficient != first_action_delta_scale_raw:
        rospy.logwarn(
            "first_action_delta_scale.coefficient must be in [0, 1]; "
            f"clamped {first_action_delta_scale_raw} to {first_action_delta_scale_coefficient}."
        )
    first_action_delta_scale_include_gripper = bool(
        first_action_delta_scale_cfg.get('include_gripper', False)
    )
    initial_pose_override_cfg = cfg['ros'].get('initial_pose_delta_override', {})
    initial_pose_override_enabled = bool(initial_pose_override_cfg.get('enabled', False))
    initial_pose_override_arms_cfg = initial_pose_override_cfg.get('arms', [])
    if isinstance(initial_pose_override_arms_cfg, str):
        initial_pose_override_arms_cfg = [initial_pose_override_arms_cfg]
    initial_pose_override_arms = {
        str(arm).strip().lower() for arm in initial_pose_override_arms_cfg
    }
    if 'both' in initial_pose_override_arms:
        initial_pose_override_arms.update(('left', 'right'))
        initial_pose_override_arms.discard('both')
    invalid_override_arms = initial_pose_override_arms - {'left', 'right'}
    if invalid_override_arms:
        rospy.logwarn(
            "initial_pose_delta_override.arms contains unsupported values "
            f"{sorted(invalid_override_arms)}; only left/right/both are accepted."
        )
        initial_pose_override_arms -= invalid_override_arms
    if initial_pose_override_enabled and not initial_pose_override_arms:
        rospy.logwarn(
            "initial_pose_delta_override is enabled but arms is empty; disabling the override."
        )
        initial_pose_override_enabled = False
    initial_pose_override_left = initial_pose_override_enabled and 'left' in initial_pose_override_arms
    initial_pose_override_right = initial_pose_override_enabled and 'right' in initial_pose_override_arms
    initial_pose_override_include_gripper = bool(initial_pose_override_cfg.get('include_gripper', False))
    terminal_filter_cfg = cfg['ros'].get('chunk_terminal_displacement_filter', {})
    terminal_filter_enabled = bool(terminal_filter_cfg.get('enabled', False))
    terminal_filter_left_threshold = np.asarray(
        terminal_filter_cfg.get('left_threshold', [0.0] * 7), dtype=np.float32,
    )
    terminal_filter_right_threshold = np.asarray(
        terminal_filter_cfg.get('right_threshold', [0.0] * 7), dtype=np.float32,
    )
    if terminal_filter_left_threshold.shape != (7,) or terminal_filter_right_threshold.shape != (7,):
        rospy.logwarn(
            "chunk_terminal_displacement_filter left_threshold/right_threshold must each contain 7 values; "
            "disabling the filter."
        )
        terminal_filter_enabled = False
    gripper_threshold_cfg = cfg['ros'].get('gripper_threshold', cfg['ros'].get('gripper_binary', {}))
    gripper_threshold_enabled = bool(gripper_threshold_cfg.get('enabled', False))
    gripper_delta_scale_cfg = cfg['ros'].get('gripper_delta_scale', {})
    gripper_delta_scale_enabled = bool(gripper_delta_scale_cfg.get('enabled', False))
    gripper_delta_gains = np.asarray([
        max(float(gripper_delta_scale_cfg.get('left_gain', 1.0)), 0.0),
        max(float(gripper_delta_scale_cfg.get('right_gain', 1.0)), 0.0),
    ], dtype=np.float32)
    gripper_delta_clip_min = gripper_delta_scale_cfg.get('clip_min')
    gripper_delta_clip_max = gripper_delta_scale_cfg.get('clip_max')
    if gripper_delta_clip_min is not None:
        gripper_delta_clip_min = np.asarray(gripper_delta_clip_min, dtype=np.float32)
    if gripper_delta_clip_max is not None:
        gripper_delta_clip_max = np.asarray(gripper_delta_clip_max, dtype=np.float32)
    if (
        (gripper_delta_clip_min is not None and gripper_delta_clip_min.shape != (2,))
        or (gripper_delta_clip_max is not None and gripper_delta_clip_max.shape != (2,))
    ):
        rospy.logwarn("gripper_delta_scale clip_min/clip_max must each contain [left, right]; disabling clipping.")
        gripper_delta_clip_min = None
        gripper_delta_clip_max = None
    policy_gripper_input_cfg = cfg['ros'].get('policy_gripper_input', {})
    policy_gripper_input_mode = str(policy_gripper_input_cfg.get('mode', 'measured')).lower()
    if policy_gripper_input_mode not in ('measured', 'commanded', 'hybrid'):
        rospy.logwarn(
            "policy_gripper_input.mode must be one of measured/commanded/hybrid; "
            f"got {policy_gripper_input_mode!r}. Using measured."
        )
        policy_gripper_input_mode = 'measured'
    policy_gripper_input_hybrid_max_error = max(
        float(policy_gripper_input_cfg.get('hybrid_max_error', 0.025)),
        0.0,
    )
    action_filter_cfg = cfg['ros'].get('action_filter', {})
    action_filter_enabled = bool(action_filter_cfg.get('enabled', False))
    action_filter_alpha = float(action_filter_cfg.get('ema_alpha', 0.25))
    action_filter_alpha = min(max(action_filter_alpha, 0.0), 1.0)
    chunk_smoothing_cfg = cfg['ros'].get('action_chunk_smoothing', {})
    chunk_smoothing_enabled = bool(chunk_smoothing_cfg.get('enabled', False))
    chunk_smoothing_upsample = max(int(chunk_smoothing_cfg.get('upsample_factor', 2)), 1)
    chunk_smoothing_window = max(int(chunk_smoothing_cfg.get('window_length', 21)), 3)
    chunk_smoothing_polyorder = max(int(chunk_smoothing_cfg.get('polyorder', 3)), 0)
    chunk_lowpass_cfg = cfg['ros'].get('action_chunk_lowpass', {})
    chunk_lowpass_enabled = bool(chunk_lowpass_cfg.get('enabled', False))
    chunk_lowpass_cutoff_hz = float(chunk_lowpass_cfg.get('cutoff_hz', 1.2))
    chunk_lowpass_sample_rate_hz = float(
        chunk_lowpass_cfg.get('sample_rate_hz', rate_hz)
    )
    chunk_lowpass_order = max(int(chunk_lowpass_cfg.get('order', 4)), 1)
    chunk_lowpass_preserve_endpoints = bool(
        chunk_lowpass_cfg.get('preserve_endpoints', True)
    )
    chunk_lowpass_include_gripper = bool(
        chunk_lowpass_cfg.get('include_gripper', False)
    )
    if chunk_lowpass_enabled and not (
        0.0 < chunk_lowpass_cutoff_hz < 0.5 * chunk_lowpass_sample_rate_hz
    ):
        raise ValueError(
            "action_chunk_lowpass.cutoff_hz must be between 0 and "
            f"{0.5 * chunk_lowpass_sample_rate_hz:.6g} Hz"
        )
    chunk_monotonic_cfg = cfg['ros'].get('action_chunk_monotonic', {})
    chunk_monotonic_enabled = bool(chunk_monotonic_cfg.get('enabled', False))
    chunk_monotonic_arms_cfg = chunk_monotonic_cfg.get('arms', ['left', 'right'])
    if isinstance(chunk_monotonic_arms_cfg, str):
        chunk_monotonic_arms_cfg = [chunk_monotonic_arms_cfg]
    chunk_monotonic_arms = {
        str(arm).strip().lower() for arm in chunk_monotonic_arms_cfg
    }
    if 'both' in chunk_monotonic_arms:
        chunk_monotonic_arms.update(('left', 'right'))
        chunk_monotonic_arms.discard('both')
    invalid_monotonic_arms = chunk_monotonic_arms - {'left', 'right'}
    if invalid_monotonic_arms:
        raise ValueError(
            "action_chunk_monotonic.arms only accepts left/right/both; got "
            f"{sorted(invalid_monotonic_arms)}"
        )
    chunk_monotonic_strength = float(chunk_monotonic_cfg.get('strength', 1.0))
    if not 0.0 <= chunk_monotonic_strength <= 1.0:
        raise ValueError("action_chunk_monotonic.strength must be in [0, 1]")
    chunk_monotonic_min_terminal_delta = max(
        float(chunk_monotonic_cfg.get('min_terminal_delta', 1e-4)), 0.0
    )
    chunk_monotonic_include_gripper = bool(
        chunk_monotonic_cfg.get('include_gripper', False)
    )
    temporal_ensemble_cfg = cfg['ros'].get('temporal_ensemble', {})
    temporal_ensemble_enabled = bool(temporal_ensemble_cfg.get('enabled', False))
    temporal_exp_decay = float(temporal_ensemble_cfg.get('exp_decay', 0.7))
    temporal_exp_decay = max(temporal_exp_decay, 0.0)
    temporal_max_history_chunks = max(int(temporal_ensemble_cfg.get('max_history_chunks', 4)), 1)
    temporal_max_candidate_age = temporal_ensemble_cfg.get('max_candidate_age')
    if temporal_max_candidate_age is not None:
        temporal_max_candidate_age = max(int(temporal_max_candidate_age), 0)
    temporal_min_action_index = max(int(temporal_ensemble_cfg.get('min_action_index', 0)), 0)
    temporal_overlap_steps = temporal_ensemble_cfg.get('overlap_steps')
    if temporal_overlap_steps is not None:
        temporal_overlap_steps = max(int(temporal_overlap_steps), 0)
    latency_comp_cfg = temporal_ensemble_cfg.get('latency_compensation', {})
    latency_comp_enabled = bool(latency_comp_cfg.get('enabled', False))
    latency_comp_mode = latency_comp_cfg.get('mode', 'measured').lower()
    latency_fixed_steps = max(int(latency_comp_cfg.get('fixed_steps', 0)), 0)
    latency_max_steps = latency_comp_cfg.get('max_steps')
    if latency_max_steps is not None:
        latency_max_steps = max(int(latency_max_steps), 0)
    initial_action_skip_steps = max(int(cfg['ros'].get('initial_action_skip_steps', 0)), 0)
    chunk_action_skip_steps = max(int(cfg['ros'].get('chunk_action_skip_steps', 0)), 0)
    topics = cfg['ros']['topics']
    task_prompt = cfg.get('task_prompt', 'demo-task')

    use_base = bool(cfg['ros'].get('use_robot_base', False))
    clip = cfg['ros'].get('clip', {'v_max': 0.2, 'w_max': 0.6})
    v_max = float(clip.get('v_max', 0.2))
    w_max = float(clip.get('w_max', 0.6))
    rb_topics = cfg['ros'].get('robot_base_topics', {'odom': '/odom', 'cmd_vel': '/cmd_vel'})

    # Subscribers (images)
    if image_type == 'compressed':
        rospy.Subscriber(topics['img_front'] + '/compressed', CompressedImage, img_cb('front', 'compressed'), queue_size=10)
        rospy.Subscriber(topics['img_left'] + '/compressed', CompressedImage, img_cb('left', 'compressed'), queue_size=10)
        rospy.Subscriber(topics['img_right'] + '/compressed', CompressedImage, img_cb('right', 'compressed'), queue_size=10)
    else:
        rospy.Subscriber(topics['img_front'], Image, img_cb('front', 'raw', jpeg_quality), queue_size=10)
        rospy.Subscriber(topics['img_left'], Image, img_cb('left', 'raw', jpeg_quality), queue_size=10)
        rospy.Subscriber(topics['img_right'], Image, img_cb('right', 'raw', jpeg_quality), queue_size=10)

    # Subscribers (joints/base/enable)
    rospy.Subscriber(topics['puppet_arm_left'], JointState, jl_cb, queue_size=50)
    rospy.Subscriber(topics['puppet_arm_right'], JointState, jr_cb, queue_size=50)
    if eef_action_mode:
        rospy.Subscriber(topics.get('puppet_arm_left_pose', '/puppet/end_pose_left'), PoseStamped, eef_left_cb, queue_size=50)
        rospy.Subscriber(topics.get('puppet_arm_right_pose', '/puppet/end_pose_right'), PoseStamped, eef_right_cb, queue_size=50)
    if use_base and 'robot_base_topics' in cfg['ros']:
        rospy.Subscriber(rb_topics['odom'], Odometry, odom_cb, queue_size=50)

    if 'enable_flag' in topics and topics['enable_flag']:
        try:
            rospy.Subscriber(topics['enable_flag'], Bool, enable_cb, queue_size=10)
        except Exception:
            pass

    # Publishers
    pub_l = rospy.Publisher(topics['cmd_joint_left'], JointState, queue_size=10)
    pub_r = rospy.Publisher(topics['cmd_joint_right'], JointState, queue_size=10)
    if eef_action_mode:
        rospy.loginfo(
            "EEF action mode enabled with bridge-side IK; "
            f"publishing joint commands to {topics['cmd_joint_left']} and {topics['cmd_joint_right']}."
        )
    enable_pub = None
    if topics.get('enable_flag') and bool(cfg['ros'].get('publish_enable_flag', False)):
        enable_pub = rospy.Publisher(topics['enable_flag'], Bool, queue_size=1, latch=True)
        rospy.sleep(0.2)
        enable_pub.publish(Bool(data=True))
        rospy.loginfo(f"Published enable flag True on {topics['enable_flag']}.")
    pub_v = None
    if use_base:
        pub_v = rospy.Publisher(rb_topics['cmd_vel'], Twist, queue_size=10)
    rospy.loginfo(
        "Command publish configured: "
        f"mode={command_publish_mode} policy_rate={rate_hz}Hz "
        f"publish_rate={command_publish_rate_hz:.1f}Hz "
        f"substeps_per_action={command_publish_substeps} "
        f"interpolation={command_publish_interpolation}"
    )

    # Home positioning before policy loop
    name_list = cfg['ros'].get('joint_names', [f'joint{i}' for i in range(7)])
    eef_ik = None
    last_published_ik = {}
    last_ik_publish_time = None
    if eef_action_mode:
        eef_ik_cfg = cfg['ros'].get('eef_ik', {})
        if not bool(eef_ik_cfg.get('enabled', True)):
            raise ValueError("ros.action_mode='eef_absolute' now requires ros.eef_ik.enabled=true")
        eef_ik = PiperNumericalIK(eef_ik_cfg, loginfo=rospy.loginfo)
        if eef_ik.differential is not None:
            if abs(eef_ik.differential.period - 1.0 / rate_hz) > 1e-6:
                raise ValueError("differential.control_period_sec must equal 1 / ros.rate_hz")
            rospy.loginfo(
                "Differential IK: bounded incremental tracking; pose residuals indicate "
                "target accuracy, not automatic rejection; "
                f"vmax={eef_ik.differential.vmax.tolist()} rad/s "
                f"amax={eef_ik.differential.amax.tolist()} rad/s^2")
        rospy.loginfo(
            "EEF IK configured: "
            f"solver={eef_ik.solver} urdf={eef_ik.urdf_path} base={eef_ik.base_link} tip={eef_ik.tip_link} "
            f"max_pos_err={eef_ik.max_position_error_m:.4f}m "
            f"max_rot_err={eef_ik.max_orientation_error_rad:.4f}rad"
        )
    home_cfg = cfg['ros'].get('home_position')
    step_lengths = cfg['ros'].get('arm_steps_length', [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])
    command_publish_lock = threading.Lock()
    command_publish_state = {
        'start_time': None,
        'duration': 1.0 / float(rate_hz),
        'start_left': None,
        'start_right': None,
        'target_left': None,
        'target_right': None,
    }

    def command_publish_worker():
        global published_first_command

        next_publish_time = time.monotonic()
        last_report_time = next_publish_time
        report_count = 0
        active_report_count = 0
        while not rospy.is_shutdown():
            with command_publish_lock:
                start_time = command_publish_state['start_time']
                start_left = None if command_publish_state['start_left'] is None else command_publish_state['start_left'].copy()
                start_right = None if command_publish_state['start_right'] is None else command_publish_state['start_right'].copy()
                target_left = None if command_publish_state['target_left'] is None else command_publish_state['target_left'].copy()
                target_right = None if command_publish_state['target_right'] is None else command_publish_state['target_right'].copy()
                duration = float(command_publish_state['duration'])

            if start_time is not None and start_left is not None and target_left is not None:
                elapsed = time.monotonic() - start_time
                alpha = 1.0 if duration <= 0.0 else min(max(elapsed / duration, 0.0), 1.0)
                publish_left = interpolate_arm_command_keep_gripper(
                    start_left, target_left, alpha
                )
                publish_right = interpolate_arm_command_keep_gripper(
                    start_right, target_right, alpha
                )

                js = JointState()
                js.header.stamp = rospy.Time.now()
                js.name = name_list
                js.position = publish_left.tolist()
                pub_l.publish(js)
                js.position = publish_right.tolist()
                pub_r.publish(js)
                if not published_first_command:
                    rospy.loginfo(
                        f"Published first command to {topics['cmd_joint_left']} and {topics['cmd_joint_right']}."
                    )
                    published_first_command = True
                active_report_count += 1
            report_count += 1

            now_report = time.monotonic()
            if now_report - last_report_time >= 2.0:
                elapsed_report = now_report - last_report_time
                rospy.loginfo(
                    "Command publish worker rate: "
                    f"loop_hz={report_count / elapsed_report:.1f} "
                    f"active_publish_hz={active_report_count / elapsed_report:.1f} "
                    f"target_hz={command_publish_rate_hz:.1f}"
                )
                last_report_time = now_report
                report_count = 0
                active_report_count = 0

            next_publish_time += command_publish_period_sec
            sleep_sec = next_publish_time - time.monotonic()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
            else:
                next_publish_time = time.monotonic()

    if command_publish_mode == 'interpolated':
        command_publish_thread = threading.Thread(
            target=command_publish_worker,
            name='command_publish_worker',
            daemon=True,
        )
        command_publish_thread.start()

    delta_clip_left = None
    delta_clip_right = None
    command_delta_deadband_left = None
    command_delta_deadband_right = None
    action_deadband_left = None
    action_deadband_right = None
    if delta_clip_enabled:
        delta_clip_reference = delta_clip_cfg.get('reference', 'command').lower()
        if delta_clip_reference not in ('command', 'current'):
            rospy.logwarn(
                f"Unsupported delta_clip.reference={delta_clip_reference!r}; "
                "using 'command'."
            )
            delta_clip_reference = 'command'
        max_delta = delta_clip_cfg.get('max_delta', step_lengths)
        max_delta = np.asarray(max_delta, dtype=np.float32)
        if max_delta.shape[0] != len(name_list):
            rospy.logwarn("delta_clip.max_delta size mismatch; using arm_steps_length.")
            max_delta = np.asarray(step_lengths, dtype=np.float32)
        if max_delta.shape[0] != len(name_list):
            rospy.logwarn("delta clip disabled: max_delta and arm_steps_length sizes do not match joint_names.")
            delta_clip_enabled = False
        else:
            delta_clip_left = max_delta
            delta_clip_right = max_delta
            rospy.loginfo(
                f"Joint delta clip enabled: max_delta={max_delta.tolist()} "
                f"reference={delta_clip_reference}"
            )
    else:
        delta_clip_reference = 'command'
    if command_delta_deadband_enabled:
        deadband_left = np.asarray(
            command_delta_deadband_cfg.get('left', [0.0] * len(name_list)),
            dtype=np.float32,
        )
        deadband_right = np.asarray(
            command_delta_deadband_cfg.get('right', [0.0] * len(name_list)),
            dtype=np.float32,
        )
        if deadband_left.shape[0] != len(name_list) or deadband_right.shape[0] != len(name_list):
            rospy.logwarn(
                "command_delta_deadband left/right size mismatch; disabling command delta deadband."
            )
            command_delta_deadband_enabled = False
        elif np.any(deadband_left < 0.0) or np.any(deadband_right < 0.0):
            rospy.logwarn(
                "command_delta_deadband thresholds must be non-negative; disabling command delta deadband."
            )
            command_delta_deadband_enabled = False
        else:
            command_delta_deadband_left = deadband_left
            command_delta_deadband_right = deadband_right
            rospy.loginfo(
                "Command delta deadband enabled: "
                f"left={deadband_left.tolist()} right={deadband_right.tolist()}"
            )
    gripper_hysteresis_cfg = cfg['ros'].get('gripper_hysteresis', {})
    gripper_hysteresis = (GripperHysteresis(gripper_hysteresis_cfg)
                          if gripper_hysteresis_cfg.get('enabled', False) else None)
    if gripper_hysteresis is not None and gripper_threshold_enabled:
        raise ValueError('gripper_hysteresis and gripper_threshold cannot both be enabled')
    if gripper_hysteresis is not None and command_publish_mode != 'direct':
        raise ValueError('gripper_hysteresis requires direct command publishing')
    gripper_close_thresholds = np.asarray(gripper_threshold_cfg.get('close_threshold', [0.004, 0.004]), dtype=np.float32)
    gripper_open_thresholds = np.asarray(gripper_threshold_cfg.get('open_threshold', [0.020, 0.020]), dtype=np.float32)
    gripper_close_values = np.asarray(gripper_threshold_cfg.get('close_value', [-0.0037, -0.0033]), dtype=np.float32)
    gripper_open_values = np.asarray(gripper_threshold_cfg.get('open_value', [0.058, 0.059]), dtype=np.float32)
    if gripper_threshold_enabled:
        if (
            gripper_close_thresholds.shape[0] != 2
            or gripper_open_thresholds.shape[0] != 2
            or gripper_close_values.shape[0] != 2
            or gripper_open_values.shape[0] != 2
        ):
            rospy.logwarn(
                "gripper_threshold close_threshold/open_threshold/close_value/open_value must each contain "
                "two values [left, right]; disabling gripper thresholding."
            )
            gripper_threshold_enabled = False
        elif np.any(gripper_close_thresholds >= gripper_open_thresholds):
            rospy.logwarn(
                "gripper_threshold close_threshold must be smaller than open_threshold; "
                "disabling gripper thresholding."
            )
            gripper_threshold_enabled = False
        else:
            rospy.loginfo(
                "Gripper threshold output enabled: "
                f"close_threshold={gripper_close_thresholds.tolist()} "
                f"open_threshold={gripper_open_thresholds.tolist()} "
                f"close_value={gripper_close_values.tolist()} "
                f"open_value={gripper_open_values.tolist()}"
            )
    if gripper_delta_scale_enabled:
        rospy.loginfo(
            "Gripper delta scaling enabled: "
            f"left_gain={gripper_delta_gains[0]:.3f} "
            f"right_gain={gripper_delta_gains[1]:.3f} "
            f"clip_min={None if gripper_delta_clip_min is None else gripper_delta_clip_min.tolist()} "
            f"clip_max={None if gripper_delta_clip_max is None else gripper_delta_clip_max.tolist()}"
        )
    if action_filter_enabled:
        deadband = np.asarray(action_filter_cfg.get('deadband', [0.0] * len(name_list)), dtype=np.float32)
        if deadband.shape[0] != len(name_list):
            rospy.logwarn("action_filter.deadband size mismatch; disabling deadband.")
            deadband = np.zeros(len(name_list), dtype=np.float32)
        action_deadband_left = deadband
        action_deadband_right = deadband
        rospy.loginfo(
            f"Action target EMA filter enabled: alpha={action_filter_alpha:.3f} "
            f"deadband={deadband.tolist()}"
        )
    if chunk_smoothing_enabled:
        rospy.loginfo(
            "DreamZero action chunk smoothing enabled: "
            f"cubic_upsample={chunk_smoothing_upsample}x "
            f"savgol_window={chunk_smoothing_window} "
            f"polyorder={chunk_smoothing_polyorder}"
        )
    if terminal_filter_enabled:
        rospy.loginfo(
            "Chunk terminal displacement filter enabled: "
            f"left_threshold={terminal_filter_left_threshold.tolist()} "
            f"right_threshold={terminal_filter_right_threshold.tolist()}"
        )
    if first_action_delta_scale_enabled:
        rospy.loginfo(
            "First consumed action delta scaling enabled: "
            f"coefficient={first_action_delta_scale_coefficient:.3f} "
            f"include_gripper={first_action_delta_scale_include_gripper}"
        )
    if initial_pose_override_enabled:
        rospy.loginfo(
            "Initial-pose delta override enabled: "
            f"arms={sorted(initial_pose_override_arms)} "
            f"include_gripper={initial_pose_override_include_gripper}"
        )
    if temporal_ensemble_enabled:
        rospy.loginfo(
            "Exponential temporal ensemble enabled: "
            f"exp_decay={temporal_exp_decay:.3f} "
            f"max_history_chunks={temporal_max_history_chunks} "
            f"max_candidate_age={temporal_max_candidate_age} "
            f"min_action_index={temporal_min_action_index} "
            f"overlap_steps={temporal_overlap_steps} "
            f"latency_compensation={latency_comp_enabled} "
            f"latency_mode={latency_comp_mode} "
            f"fixed_steps={latency_fixed_steps} "
            f"max_steps={latency_max_steps}"
        )
    if interpolation_enabled:
        if interpolation_mode == 'adaptive_delta':
            rospy.loginfo(
                "Adaptive delta chunk interpolation enabled: "
                f"source_steps_to_execute={adaptive_source_steps_to_execute} "
                f"source_overlap_steps={adaptive_source_overlap_steps} "
                f"min_factor={adaptive_min_factor} max_factor={adaptive_max_factor} "
                f"initial_bridge_enabled={adaptive_bridge_enabled} "
                f"initial_bridge_max_factor={adaptive_bridge_max_factor} "
                f"joint_threshold={adaptive_joint_threshold}"
            )
        else:
            rospy.loginfo(f"Linear chunk interpolation enabled: factor={interpolation_factor:.3f}")
    if home_cfg:
        try:
            step_arr = np.asarray(step_lengths, dtype=np.float32)
            if step_arr.shape[0] != len(name_list):
                rospy.logwarn("arm_steps_length size mismatch; using uniform step length 0.01.")
                step_arr = np.full(len(name_list), 0.01, dtype=np.float32)

            require_home_settle = bool(home_cfg.get('require_settle', False))
            home_ok = True
            home_sequence = home_cfg.get('sequence')
            if home_sequence:
                for phase_index, phase_cfg in enumerate(home_sequence):
                    phase_ok = execute_home_phase(
                        pub_l,
                        pub_r,
                        name_list,
                        step_arr,
                        home_cfg,
                        phase_cfg,
                        phase_index,
                    )
                    home_ok = home_ok and bool(phase_ok)
                    if require_home_settle and not phase_ok:
                        break
            else:
                home_ok = move_to_home(
                    pub_l,
                    pub_r,
                    name_list,
                    np.array(home_cfg['left'], dtype=np.float32),
                    np.array(home_cfg['right'], dtype=np.float32),
                    step_arr,
                    max(int(home_cfg.get('rate_hz', rate_hz)), 1),
                    mode=str(home_cfg.get('mode', 'step')).lower(),
                    num_steps=home_cfg.get('num_steps'),
                    gripper_move_start_fraction=home_cfg.get('gripper_move_start_fraction'),
                    settle=bool(home_cfg.get('settle', True)),
                    settle_tolerance=float(home_cfg.get('settle_tolerance', 0.04)),
                    settle_hold_sec=float(home_cfg.get('settle_hold_sec', 0.5)),
                    settle_timeout_sec=float(home_cfg.get('settle_timeout_sec', 8.0)),
                    settle_rate_hz=max(int(home_cfg.get('settle_rate_hz', min(int(home_cfg.get('rate_hz', rate_hz)), 50))), 1),
                )
            if require_home_settle and not home_ok:
                rospy.logerr("Home position did not settle; stopping before policy inference.")
                return
            post_sleep_sec = max(float(home_cfg.get('post_sleep_sec', 0.0)), 0.0)
            if post_sleep_sec > 0.0:
                rospy.loginfo(f"Waiting {post_sleep_sec:.2f}s after home before inference.")
                rospy.sleep(post_sleep_sec)
        except (KeyError, ValueError, TypeError):
            rospy.logwarn("Invalid home_position configuration; skipping home move.")

    # ZeroMQ client
    ctx = zmq.Context.instance()
    connect_addr = cfg['zmq'].get('client_connect', 'tcp://127.0.0.1:5557')
    socket_type = cfg['zmq'].get('socket_type', 'req')
    recv_timeout_ms = max(int(1000 / rate_hz), 1)
    sock = make_policy_socket(ctx, connect_addr, recv_timeout_ms, socket_type)
    rospy.loginfo(f"Policy server endpoint set to {connect_addr} using ZMQ {socket_type.upper()}")

    rate = rospy.Rate(rate_hz)
    dt = 1.0 / rate_hz / 4.0
    integrated_left = None
    integrated_right = None
    last_response_wait_warn = 0.0
    last_obs_wait_warn = 0.0
    last_command_left = None
    last_command_right = None
    filtered_policy_left = None
    filtered_policy_right = None
    chunk_history = []
    global_action_step = 0
    action_logger, action_log_path = make_action_logger(cfg, rate_hz)
    rollout_logger = make_rollout_dataset_logger(cfg, rate_hz)
    request_id = 0
    command_left = None
    command_right = None
    pending_request = None
    last_request_camera_seq = None
    last_stale_obs_warn = 0.0
    last_stale_joint_warn = 0.0
    rospy.loginfo(
        "Async policy loop enabled: "
        f"request_check_period={1.0 / rate_hz:.3f}s; "
        "continuing to publish available chunk actions while waiting for policy responses; "
        "policy requests are checked on each global step and wait for fresh front/left/right images; "
        f"when_live_chunk={policy_request_when_live_chunk}; "
        f"initial_action_skip_steps={initial_action_skip_steps}; "
        f"chunk_action_skip_steps={chunk_action_skip_steps}; "
        f"policy_gripper_input_mode={policy_gripper_input_mode}."
    )

    def chunk_live_steps(chunk):
        steps = int(chunk.get('steps_to_execute', chunk['left'].shape[0]))
        skip = int(chunk.get('action_skip_steps', 0))
        return min(steps + skip, int(chunk['left'].shape[0]))

    def live_chunks_for_step(step):
        return [
            chunk for chunk in chunk_history
            if 0 <= step - chunk['start_step'] < chunk_live_steps(chunk)
        ]

    def select_chunk_for_step(step):
        live_chunks = live_chunks_for_step(step)
        if not live_chunks:
            return None
        if not temporal_ensemble_enabled or temporal_min_action_index <= 0:
            return live_chunks[-1]
        mature_chunks = [
            chunk for chunk in live_chunks
            if step - chunk['start_step'] >= temporal_min_action_index
        ]
        if mature_chunks:
            return mature_chunks[-1]
        return live_chunks[0]

    def make_policy_gripper_input(measured_left, measured_right):
        measured_left_arr = np.asarray(measured_left, dtype=np.float32)
        measured_right_arr = np.asarray(measured_right, dtype=np.float32)
        policy_left = measured_left_arr.copy()
        policy_right = measured_right_arr.copy()

        commanded_left_gripper = None
        commanded_right_gripper = None
        if last_command_left is not None and len(last_command_left) > 0:
            commanded_left_gripper = float(last_command_left[-1])
        elif command_left is not None and len(command_left) > 0:
            commanded_left_gripper = float(command_left[-1])
        if last_command_right is not None and len(last_command_right) > 0:
            commanded_right_gripper = float(last_command_right[-1])
        elif command_right is not None and len(command_right) > 0:
            commanded_right_gripper = float(command_right[-1])

        def choose(measured_value, commanded_value):
            if commanded_value is None or policy_gripper_input_mode == 'measured':
                return float(measured_value), 'measured'
            if policy_gripper_input_mode == 'commanded':
                return float(commanded_value), 'commanded'
            if abs(float(measured_value) - float(commanded_value)) <= policy_gripper_input_hybrid_max_error:
                return float(commanded_value), 'commanded'
            return float(measured_value), 'measured'

        left_value, left_source = choose(measured_left_arr[-1], commanded_left_gripper)
        right_value, right_source = choose(measured_right_arr[-1], commanded_right_gripper)
        policy_left[-1] = left_value
        policy_right[-1] = right_value
        info = {
            'mode': policy_gripper_input_mode,
            'hybrid_max_error': policy_gripper_input_hybrid_max_error,
            'left_source': left_source,
            'right_source': right_source,
            'left_measured': float(measured_left_arr[-1]),
            'right_measured': float(measured_right_arr[-1]),
            'left_policy': float(policy_left[-1]),
            'right_policy': float(policy_right[-1]),
            'left_commanded': commanded_left_gripper,
            'right_commanded': commanded_right_gripper,
        }
        return policy_left, policy_right, info

    def send_policy_request(pkt, fresh_camera_count, forced_fresh_fallback=False):
        nonlocal pending_request, last_request_camera_seq

        measured_left = np.asarray(pkt['jleft'], dtype=np.float32)
        measured_right = np.asarray(pkt['jright'], dtype=np.float32)
        policy_left, policy_right, gripper_input_info = make_policy_gripper_input(
            measured_left,
            measured_right,
        )

        header = {
            'task_prompt': pkt['task_prompt'],
            'episode_start': request_id == 0,
            'jleft': policy_left.tolist(),
            'jright': policy_right.tolist(),
            'control_hz': rate_hz,
            'action_mode': action_mode,
            'policy_gripper_input': gripper_input_info,
            'measured_jleft_gripper': float(measured_left[-1]),
            'measured_jright_gripper': float(measured_right[-1]),
        }
        if pkt.get('xvla_proprio') is not None:
            header['xvla_proprio'] = pkt['xvla_proprio']
        if pkt.get('current_eef_left') is not None:
            header['current_eef_left'] = pkt['current_eef_left']
        if pkt.get('current_eef_right') is not None:
            header['current_eef_right'] = pkt['current_eef_right']
        if use_base and pkt['odom'] is not None:
            header['odom'] = pkt['odom']

        frames = [
            json.dumps(header, separators=(',', ':')).encode('utf-8'),
            pkt['front'],
            pkt['left'],
            pkt['right'],
        ]
        policy_request_step = global_action_step
        request_snapshot_dir = save_request_snapshot(
            action_logger,
            policy_request_step,
            pkt,
            header,
            fresh_camera_count,
            forced_fresh_fallback,
        )
        try:
            sock.send_multipart(frames)
        except zmq.error.Again:
            rospy.logwarn("ZeroMQ send timeout; will retry next cycle.")
            return False

        rollout_sample_dir = save_rollout_observation(
            rollout_logger,
            policy_request_step,
            pkt,
            header,
            fresh_camera_count,
            forced_fresh_fallback,
        )

        if eef_action_mode and pkt.get('current_eef_left') is not None and pkt.get('current_eef_right') is not None:
            request_current_left = np.asarray(pkt['current_eef_left'], dtype=np.float32)
            request_current_right = np.asarray(pkt['current_eef_right'], dtype=np.float32)
            measured_current_left = request_current_left.copy()
            measured_current_right = request_current_right.copy()
            request_current_left[-1] = policy_left[-1]
            request_current_right[-1] = policy_right[-1]
        else:
            request_current_left = policy_left.copy()
            request_current_right = policy_right.copy()
            measured_current_left = measured_left.copy()
            measured_current_right = measured_right.copy()

        pending_request = {
            'step': policy_request_step,
            'time': time.monotonic(),
            'current_left': request_current_left.copy(),
            'current_right': request_current_right.copy(),
            'measured_current_left': measured_current_left.copy(),
            'measured_current_right': measured_current_right.copy(),
            'policy_gripper_input': dict(gripper_input_info),
            'obs_seq': dict(pkt['obs_seq']),
            'obs_time': dict(pkt['obs_time']),
            'fresh_camera_count': int(fresh_camera_count),
            'fresh_camera_forced': bool(forced_fresh_fallback),
            'request_snapshot_dir': request_snapshot_dir,
            'rollout_sample_dir': rollout_sample_dir,
        }
        last_request_camera_seq = tuple(pkt['obs_seq'][key] for key in ('front', 'left', 'right'))
        if forced_fresh_fallback:
            rospy.logwarn(
                "Sending policy request before all cameras refreshed to avoid action starvation: "
                f"fresh_camera_count={fresh_camera_count}/3 seq={last_request_camera_seq}"
            )
        rospy.loginfo_once(
            f"Sent first policy request to {connect_addr}; async action loop is running."
        )
        return True

    def obs_is_fresh_enough(pkt):
        now = time.monotonic()
        stale = []
        if max_image_age_sec > 0.0:
            for key in ('front', 'left', 'right'):
                age = None if pkt['obs_time'].get(key) is None else now - pkt['obs_time'][key]
                if age is None or age > max_image_age_sec:
                    stale.append((key, age))
        if max_joint_age_sec > 0.0:
            for key in ('jl', 'jr'):
                age = None if pkt['obs_time'].get(key) is None else now - pkt['obs_time'][key]
                if age is None or age > max_joint_age_sec:
                    stale.append((key, age))
        return len(stale) == 0, stale

    def ingest_policy_response(rep_frames, pending_snapshot):
        nonlocal chunk_history, global_action_step, request_id

        policy_request_step = pending_snapshot['step']
        policy_latency_sec = time.monotonic() - pending_snapshot['time']
        if latency_comp_enabled:
            if latency_comp_mode == 'fixed':
                policy_latency_steps = latency_fixed_steps
            else:
                policy_latency_steps = int(round(policy_latency_sec * rate_hz))
            policy_latency_steps = max(policy_latency_steps, 0)
            if latency_max_steps is not None:
                policy_latency_steps = min(policy_latency_steps, latency_max_steps)
            # Late responses contain actions for times that have already passed.
            # Skip those young indices and consume the chunk at the current global step.
            global_action_step = max(global_action_step, policy_request_step + policy_latency_steps)
        else:
            policy_latency_steps = 0
        if initial_action_skip_steps > 0 and request_id == 0:
            global_action_step = max(global_action_step, policy_request_step + initial_action_skip_steps)

        try:
            rep_header = json.loads(rep_frames[0].decode('utf-8'))
        except (IndexError, json.JSONDecodeError, UnicodeDecodeError):
            rospy.logwarn("Invalid policy response header.")
            return
        try:
            validate_policy_action_mode(rep_header, action_mode)
        except ValueError as exc:
            rospy.logerr(str(exc))
            return

        chunk_size = int(rep_header.get('chunk_size', 0))
        if chunk_size == 0 or len(rep_frames) < 3:
            rospy.logwarn("Empty or incomplete policy response.")
            return
        received_chunk_size = chunk_size

        left_stride = int(rep_header.get('left_stride', 7))
        right_stride = int(rep_header.get('right_stride', 7))
        left_arr = np.frombuffer(rep_frames[1], dtype=np.float32, count=chunk_size * left_stride)
        right_arr = np.frombuffer(rep_frames[2], dtype=np.float32, count=chunk_size * right_stride)
        try:
            left_mat = left_arr.reshape((chunk_size, left_stride))
            right_mat = right_arr.reshape((chunk_size, right_stride))
        except ValueError:
            rospy.logwarn("Invalid action matrix shape.")
            return
        received_left_mat = left_mat.copy()
        received_right_mat = right_mat.copy()
        if chunk_lowpass_enabled:
            try:
                left_mat, right_mat = lowpass_dual_action_chunks_zero_phase(
                    left_mat,
                    right_mat,
                    sample_rate_hz=chunk_lowpass_sample_rate_hz,
                    cutoff_hz=chunk_lowpass_cutoff_hz,
                    order=chunk_lowpass_order,
                    preserve_endpoints=chunk_lowpass_preserve_endpoints,
                    include_gripper=chunk_lowpass_include_gripper,
                )
            except ValueError as exc:
                rospy.logwarn(f"Chunk low-pass skipped: {exc}")
                return
            lowpass_delta = np.concatenate(
                (left_mat[:, :-1] - received_left_mat[:, :-1],
                 right_mat[:, :-1] - received_right_mat[:, :-1]),
                axis=1,
            )
            rospy.loginfo(
                "Filtered chunk-wide high-frequency motion: "
                f"cutoff_hz={chunk_lowpass_cutoff_hz:.3f} "
                f"sample_rate_hz={chunk_lowpass_sample_rate_hz:.3f} "
                f"order={chunk_lowpass_order} "
                f"rms_delta={float(np.sqrt(np.mean(lowpass_delta ** 2))):.6f} "
                f"max_delta={float(np.max(np.abs(lowpass_delta))):.6f}"
            )
        lowpass_left_mat = left_mat.copy()
        lowpass_right_mat = right_mat.copy()
        if chunk_monotonic_enabled:
            monotonic_before_left = left_mat.copy()
            monotonic_before_right = right_mat.copy()
            try:
                if 'left' in chunk_monotonic_arms:
                    left_mat = project_action_chunk_monotonic_to_endpoint(
                        left_mat,
                        command_left,
                        strength=chunk_monotonic_strength,
                        min_terminal_delta=chunk_monotonic_min_terminal_delta,
                        include_gripper=chunk_monotonic_include_gripper,
                    )
                if 'right' in chunk_monotonic_arms:
                    right_mat = project_action_chunk_monotonic_to_endpoint(
                        right_mat,
                        command_right,
                        strength=chunk_monotonic_strength,
                        min_terminal_delta=chunk_monotonic_min_terminal_delta,
                        include_gripper=chunk_monotonic_include_gripper,
                    )
            except ValueError as exc:
                rospy.logwarn(f"Chunk monotonic projection skipped: {exc}")
                return
            monotonic_delta = np.concatenate(
                (left_mat[:, :-1] - monotonic_before_left[:, :-1],
                 right_mat[:, :-1] - monotonic_before_right[:, :-1]),
                axis=1,
            )
            rospy.loginfo(
                "Suppressed chunk-internal joint reversals: "
                f"arms={sorted(chunk_monotonic_arms)} "
                f"strength={chunk_monotonic_strength:.3f} "
                f"rms_delta={float(np.sqrt(np.mean(monotonic_delta ** 2))):.6f} "
                f"max_delta={float(np.max(np.abs(monotonic_delta))):.6f}"
            )
        monotonic_left_mat = left_mat.copy()
        monotonic_right_mat = right_mat.copy()
        if chunk_smoothing_enabled:
            left_mat, right_mat = smooth_dual_action_chunks_savgol(
                left_mat,
                right_mat,
                upsample_factor=chunk_smoothing_upsample,
                window_length=chunk_smoothing_window,
                polyorder=chunk_smoothing_polyorder,
            )

        vel_mat = None
        next_frame_idx = 3
        if rep_header.get('has_vel', False) and len(rep_frames) > next_frame_idx:
            vel_arr = np.frombuffer(rep_frames[next_frame_idx], dtype=np.float32, count=chunk_size * 2)
            try:
                vel_mat = vel_arr.reshape((chunk_size, 2))
            except ValueError:
                vel_mat = None
            next_frame_idx += 1

        model_raw_left_mat = None
        model_raw_right_mat = None
        if rep_header.get('has_model_raw_action', False):
            raw_left_stride = int(rep_header.get('model_raw_left_stride', left_stride))
            raw_right_stride = int(rep_header.get('model_raw_right_stride', right_stride))
            if len(rep_frames) >= next_frame_idx + 2:
                raw_left_arr = np.frombuffer(
                    rep_frames[next_frame_idx],
                    dtype=np.float32,
                    count=chunk_size * raw_left_stride,
                )
                raw_right_arr = np.frombuffer(
                    rep_frames[next_frame_idx + 1],
                    dtype=np.float32,
                    count=chunk_size * raw_right_stride,
                )
                try:
                    model_raw_left_mat = raw_left_arr.reshape((chunk_size, raw_left_stride))
                    model_raw_right_mat = raw_right_arr.reshape((chunk_size, raw_right_stride))
                except ValueError:
                    rospy.logwarn("Invalid model raw action matrix shape; skipping model_raw_action save.")
                    model_raw_left_mat = None
                    model_raw_right_mat = None
            else:
                rospy.logwarn("Policy response header has_model_raw_action=true but raw action frames are missing.")

        request_temporal_overlap_steps = temporal_overlap_steps
        initial_bridge_factor = 1
        initial_bridge_added_steps = 0
        if interpolation_enabled and interpolation_mode == 'adaptive_delta':
            try:
                left_mat, right_mat, source_to_expanded, segment_factors = adaptive_delta_upsample_chunks(
                    left_mat,
                    right_mat,
                    adaptive_min_factor,
                    adaptive_max_factor,
                    adaptive_joint_threshold,
                )
            except ValueError as exc:
                rospy.logwarn(str(exc))
                return
            if vel_mat is not None:
                vel_mat, _ = interpolate_with_segment_factors(vel_mat, segment_factors)
            if adaptive_bridge_enabled:
                try:
                    left_before_bridge = left_mat.shape[0]
                    left_mat, right_mat, source_to_expanded, initial_bridge_factor = adaptive_bridge_to_first_action(
                        left_mat,
                        right_mat,
                        command_left,
                        command_right,
                        adaptive_min_factor,
                        adaptive_bridge_max_factor,
                        adaptive_joint_threshold,
                        source_to_expanded,
                    )
                    if vel_mat is not None and left_mat.shape[0] > left_before_bridge:
                        bridge_count = left_mat.shape[0] - left_before_bridge
                        vel_mat = np.vstack([
                            np.repeat(vel_mat[:1], bridge_count, axis=0),
                            vel_mat,
                        ])
                    initial_bridge_added_steps = (
                        left_mat.shape[0] - left_before_bridge
                    )
                    if initial_bridge_added_steps > 0:
                        rospy.loginfo(
                            "Inserted chunk-boundary bridge: "
                            f"factor={initial_bridge_factor} "
                            f"added_steps={initial_bridge_added_steps}"
                        )
                except ValueError as exc:
                    rospy.logwarn(str(exc))
                    return
            chunk_size = int(left_mat.shape[0])
            if adaptive_source_steps_to_execute is not None:
                source_idx = min(adaptive_source_steps_to_execute - 1, len(source_to_expanded) - 1)
                steps_to_execute = source_to_expanded[source_idx] + 1
            else:
                steps_to_execute = chunk_size if open_loop_steps is None else min(open_loop_steps, chunk_size)
            if adaptive_source_overlap_steps is not None:
                if adaptive_source_overlap_steps <= 0:
                    request_temporal_overlap_steps = 0
                else:
                    source_idx = min(adaptive_source_overlap_steps - 1, len(source_to_expanded) - 1)
                    request_temporal_overlap_steps = source_to_expanded[source_idx] + 1
        else:
            if interpolation_enabled and interpolation_factor > 1.0:
                left_mat = linear_upsample_chunk(left_mat, interpolation_factor)
                right_mat = linear_upsample_chunk(right_mat, interpolation_factor)
                if vel_mat is not None:
                    vel_mat = linear_upsample_chunk(vel_mat, interpolation_factor)
                chunk_size = int(left_mat.shape[0])
            steps_to_execute = chunk_size if open_loop_steps is None else min(open_loop_steps, chunk_size)

        terminal_delta_left = np.zeros(left_mat.shape[1], dtype=np.float32)
        terminal_delta_right = np.zeros(right_mat.shape[1], dtype=np.float32)
        terminal_cancel_left = np.zeros(left_mat.shape[1], dtype=bool)
        terminal_cancel_right = np.zeros(right_mat.shape[1], dtype=bool)
        terminal_endpoint_index = min(max(int(steps_to_execute), 1), chunk_size) - 1
        if terminal_filter_enabled:
            initial_left = pending_snapshot.get('measured_current_left', pending_snapshot['current_left'])
            initial_right = pending_snapshot.get('measured_current_right', pending_snapshot['current_right'])
            try:
                (
                    left_mat,
                    right_mat,
                    terminal_delta_left,
                    terminal_delta_right,
                    terminal_cancel_left,
                    terminal_cancel_right,
                    terminal_endpoint_index,
                ) = filter_chunk_by_terminal_displacement(
                    left_mat,
                    right_mat,
                    initial_left,
                    initial_right,
                    steps_to_execute,
                    terminal_filter_left_threshold,
                    terminal_filter_right_threshold,
                )
            except ValueError as exc:
                rospy.logwarn(f"Chunk terminal displacement filter skipped: {exc}")
            else:
                rospy.loginfo(
                    "Chunk terminal displacement filter: "
                    f"endpoint_index={terminal_endpoint_index} "
                    f"left_cancelled={np.flatnonzero(terminal_cancel_left).tolist()} "
                    f"right_cancelled={np.flatnonzero(terminal_cancel_right).tolist()}"
                )

        action_skip_steps = min(chunk_action_skip_steps, max(chunk_size - 1, 0))
        if action_skip_steps > 0:
            global_action_step = max(global_action_step, policy_request_step + action_skip_steps)
            rospy.loginfo(
                f"Skipping first {action_skip_steps} action step(s) of policy chunk "
                f"to avoid chunk-boundary transients."
            )

        request_id += 1
        new_chunk = {
            'request_id': request_id,
            'start_step': policy_request_step,
            'left': left_mat.copy(),
            'right': right_mat.copy(),
            'received_left': received_left_mat,
            'received_right': received_right_mat,
            'model_raw_left': None if model_raw_left_mat is None else model_raw_left_mat.copy(),
            'model_raw_right': None if model_raw_right_mat is None else model_raw_right_mat.copy(),
            'vel': None if vel_mat is None else vel_mat.copy(),
            'overlap_steps': request_temporal_overlap_steps,
            'executed_steps': 0,
            'received_chunk_size': received_chunk_size,
            'steps_to_execute': steps_to_execute,
            'terminal_filter_enabled': terminal_filter_enabled,
            'terminal_endpoint_index': terminal_endpoint_index,
            'terminal_delta_left': terminal_delta_left.copy(),
            'terminal_delta_right': terminal_delta_right.copy(),
            'terminal_cancel_left': terminal_cancel_left.copy(),
            'terminal_cancel_right': terminal_cancel_right.copy(),
            'first_action_delta_scale_enabled': first_action_delta_scale_enabled,
            'first_action_delta_scale_coefficient': first_action_delta_scale_coefficient,
            'first_action_delta_scale_include_gripper': first_action_delta_scale_include_gripper,
            'initial_bridge_factor': initial_bridge_factor,
            'initial_bridge_added_steps': initial_bridge_added_steps,
            'chunk_lowpass_enabled': chunk_lowpass_enabled,
            'chunk_lowpass_cutoff_hz': chunk_lowpass_cutoff_hz,
            'chunk_lowpass_sample_rate_hz': chunk_lowpass_sample_rate_hz,
            'chunk_lowpass_order': chunk_lowpass_order,
            'chunk_lowpass_preserve_endpoints': chunk_lowpass_preserve_endpoints,
            'lowpass_left': lowpass_left_mat.copy(),
            'lowpass_right': lowpass_right_mat.copy(),
            'chunk_monotonic_enabled': chunk_monotonic_enabled,
            'chunk_monotonic_arms': sorted(chunk_monotonic_arms),
            'chunk_monotonic_strength': chunk_monotonic_strength,
            'chunk_monotonic_min_terminal_delta': chunk_monotonic_min_terminal_delta,
            'monotonic_left': monotonic_left_mat.copy(),
            'monotonic_right': monotonic_right_mat.copy(),
            'initial_pose_delta_override_enabled': initial_pose_override_enabled,
            'initial_pose_delta_override_arms': sorted(initial_pose_override_arms),
            'initial_pose_delta_override_include_gripper': initial_pose_override_include_gripper,
            'action_skip_steps': action_skip_steps,
            'policy_latency_sec': policy_latency_sec,
            'policy_latency_steps': policy_latency_steps,
            'request_current_left': pending_snapshot['current_left'].copy(),
            'request_current_right': pending_snapshot['current_right'].copy(),
            'request_measured_left': pending_snapshot.get(
                'measured_current_left',
                pending_snapshot['current_left'],
            ).copy(),
            'request_measured_right': pending_snapshot.get(
                'measured_current_right',
                pending_snapshot['current_right'],
            ).copy(),
            'policy_gripper_input': dict(pending_snapshot.get('policy_gripper_input', {})),
            'request_obs_seq': dict(pending_snapshot['obs_seq']),
            'request_obs_time': dict(pending_snapshot['obs_time']),
            'request_fresh_camera_count': int(pending_snapshot['fresh_camera_count']),
            'request_fresh_camera_forced': bool(pending_snapshot['fresh_camera_forced']),
            'request_snapshot_dir': pending_snapshot.get('request_snapshot_dir', ''),
            'rollout_sample_dir': pending_snapshot.get('rollout_sample_dir', ''),
        }
        new_chunk['action_chunk_path'] = save_action_chunk(action_logger, request_id, new_chunk)
        new_chunk['rollout_action_path'] = save_rollout_action(
            rollout_logger,
            new_chunk.get('rollout_sample_dir', ''),
            request_id,
            new_chunk,
        )
        chunk_history.append(new_chunk)
        chunk_history = chunk_history[-temporal_max_history_chunks:]

    def publish_action_from_chunk(active_chunk):
        global published_first_command
        nonlocal command_left, command_right
        nonlocal filtered_policy_left, filtered_policy_right
        nonlocal integrated_left, integrated_right
        nonlocal last_command_left, last_command_right
        nonlocal global_action_step, chunk_history
        nonlocal last_stale_joint_warn, gripper_hysteresis, last_ik_publish_time

        current_chunk_start_step = active_chunk['start_step']
        chunk_size = int(active_chunk['left'].shape[0])
        action_index = global_action_step - current_chunk_start_step
        if action_index < 0 or action_index >= chunk_size:
            return False

        command_left_before = command_left.copy()
        command_right_before = command_right.copy()
        ik_seed_left, ik_seed_right, jl_age, jr_age = latest_joint_arrays_and_age()
        publish_current_left = ik_seed_left
        publish_current_right = ik_seed_right
        if eef_action_mode:
            publish_current_left = command_left.copy()
            publish_current_right = command_right.copy()
        if (
            max_joint_age_sec > 0.0
            and (
                publish_current_left is None
                or publish_current_right is None
                or jl_age is None
                or jr_age is None
                or jl_age > max_joint_age_sec
                or jr_age > max_joint_age_sec
            )
        ):
            now_warn = time.monotonic()
            if now_warn - last_stale_joint_warn >= 1.0:
                rospy.logwarn(
                    "Skipping action publish because joint feedback is stale: "
                    f"jl_age={jl_age} jr_age={jr_age} "
                    f"max_joint_age_sec={max_joint_age_sec:.3f}"
                )
                last_stale_joint_warn = now_warn
            return False
        if publish_current_left is None:
            publish_current_left = command_left.copy()
        if publish_current_right is None:
            publish_current_right = command_right.copy()
        if delta_clip_reference == 'current':
            clip_reference_left = publish_current_left
            clip_reference_right = publish_current_right
        else:
            clip_reference_left = command_left
            clip_reference_right = command_right
        temporal_ensemble_count = 0
        temporal_ensemble_weights = []

        if action_mode in ('absolute', 'eef_absolute'):
            raw_target_left = active_chunk['left'][action_index]
            raw_target_right = active_chunk['right'][action_index]
            ensembled_target_left = raw_target_left
            ensembled_target_right = raw_target_right
            if (
                temporal_ensemble_enabled
                and (
                    active_chunk['overlap_steps'] is None
                    or active_chunk['executed_steps'] < active_chunk['overlap_steps']
                )
            ):
                ensemble_left, ensemble_right, ensemble_count, ensemble_weights = exponential_temporal_ensemble(
                    chunk_history,
                    global_action_step,
                    active_chunk['request_id'],
                    temporal_exp_decay,
                    temporal_max_candidate_age,
                    temporal_min_action_index,
                )
                if ensemble_left is not None and ensemble_right is not None:
                    ensembled_target_left = ensemble_left
                    ensembled_target_right = ensemble_right
                    temporal_ensemble_count = ensemble_count
                    temporal_ensemble_weights = ensemble_weights
            if action_filter_enabled:
                if filtered_policy_left is None or filtered_policy_right is None:
                    filtered_policy_left = command_left.copy()
                    filtered_policy_right = command_right.copy()
                filtered_policy_left = (
                    action_filter_alpha * ensembled_target_left
                    + (1.0 - action_filter_alpha) * filtered_policy_left
                )
                filtered_policy_right = (
                    action_filter_alpha * ensembled_target_right
                    + (1.0 - action_filter_alpha) * filtered_policy_right
                )
                filtered_target_left = filtered_policy_left.copy()
                filtered_target_right = filtered_policy_right.copy()
                small_left = np.abs(filtered_target_left - command_left) < action_deadband_left
                small_right = np.abs(filtered_target_right - command_right) < action_deadband_right
                filtered_target_left[small_left] = command_left[small_left]
                filtered_target_right[small_right] = command_right[small_right]
            else:
                filtered_target_left = ensembled_target_left
                filtered_target_right = ensembled_target_right
            if delta_clip_enabled:
                target_left = clip_joint_delta(clip_reference_left, filtered_target_left, delta_clip_left)
                target_right = clip_joint_delta(clip_reference_right, filtered_target_right, delta_clip_right)
            else:
                target_left = filtered_target_left
                target_right = filtered_target_right
        else:
            integrated_left = integrated_left + active_chunk['left'][action_index] * dt
            integrated_right = integrated_right + active_chunk['right'][action_index] * dt
            integrated_left[-1] = active_chunk['left'][action_index, -1]
            integrated_right[-1] = active_chunk['right'][action_index, -1]
            raw_target_left = integrated_left
            raw_target_right = integrated_right
            ensembled_target_left = raw_target_left
            ensembled_target_right = raw_target_right
            if action_filter_enabled:
                if filtered_policy_left is None or filtered_policy_right is None:
                    filtered_policy_left = command_left.copy()
                    filtered_policy_right = command_right.copy()
                filtered_policy_left = (
                    action_filter_alpha * ensembled_target_left
                    + (1.0 - action_filter_alpha) * filtered_policy_left
                )
                filtered_policy_right = (
                    action_filter_alpha * ensembled_target_right
                    + (1.0 - action_filter_alpha) * filtered_policy_right
                )
                filtered_target_left = filtered_policy_left.copy()
                filtered_target_right = filtered_policy_right.copy()
                small_left = np.abs(filtered_target_left - command_left) < action_deadband_left
                small_right = np.abs(filtered_target_right - command_right) < action_deadband_right
                filtered_target_left[small_left] = command_left[small_left]
                filtered_target_right[small_right] = command_right[small_right]
            else:
                filtered_target_left = ensembled_target_left
                filtered_target_right = ensembled_target_right
            if delta_clip_enabled:
                target_left = clip_joint_delta(clip_reference_left, filtered_target_left, delta_clip_left)
                target_right = clip_joint_delta(clip_reference_right, filtered_target_right, delta_clip_right)
            else:
                target_left = filtered_target_left
                target_right = filtered_target_right

        clipped_target_left = target_left.copy()
        clipped_target_right = target_right.copy()

        if gripper_delta_scale_enabled:
            target_left, target_right = scale_gripper_deltas(
                target_left,
                target_right,
                active_chunk.get('request_measured_left', active_chunk['request_current_left']),
                active_chunk.get('request_measured_right', active_chunk['request_current_right']),
                gripper_delta_gains,
                clip_min=gripper_delta_clip_min,
                clip_max=gripper_delta_clip_max,
            )
        gripper_candidate = gripper_hysteresis
        gripper_transition = None
        if gripper_hysteresis is not None:
            values, gripper_candidate, gripper_transition = gripper_hysteresis.propose(
                [target_left[-1], target_right[-1]],
                [ik_seed_left[-1], ik_seed_right[-1]],
                request_opening=[active_chunk['request_current_left'][-1],
                                 active_chunk['request_current_right'][-1]],
            )
            target_left = target_left.copy()
            target_right = target_right.copy()
            target_left[-1], target_right[-1] = values
        if gripper_threshold_enabled:
            target_left, target_right = threshold_gripper_targets(
                target_left,
                target_right,
                gripper_close_thresholds,
                gripper_open_thresholds,
                gripper_close_values,
                gripper_open_values,
            )
        if command_delta_deadband_enabled:
            target_left, command_delta_deadband_residual_left = apply_joint_delta_deadband(
                command_left_before,
                target_left,
                command_delta_deadband_left,
            )
            target_right, command_delta_deadband_residual_right = apply_joint_delta_deadband(
                command_right_before,
                target_right,
                command_delta_deadband_right,
            )
        else:
            command_delta_deadband_residual_left = np.zeros_like(target_left, dtype=np.float32)
            command_delta_deadband_residual_right = np.zeros_like(target_right, dtype=np.float32)

        first_action_delta_scale_applied = (
            first_action_delta_scale_enabled
            and active_chunk['executed_steps'] == 0
        )
        if first_action_delta_scale_applied:
            target_left, target_right = scale_action_delta_from_reference(
                target_left,
                target_right,
                command_left_before,
                command_right_before,
                first_action_delta_scale_coefficient,
                include_gripper=first_action_delta_scale_include_gripper,
            )

        if initial_pose_override_enabled:
            target_left, target_right = override_arm_delta_from_initial_pose(
                target_left,
                target_right,
                active_chunk.get('request_measured_left', active_chunk['request_current_left']),
                active_chunk.get('request_measured_right', active_chunk['request_current_right']),
                initial_pose_override_left,
                initial_pose_override_right,
                include_gripper=initial_pose_override_include_gripper,
            )

        raw_delta_left = raw_target_left - command_left_before
        raw_delta_right = raw_target_right - command_right_before
        applied_delta_left = target_left - command_left_before
        applied_delta_right = target_right - command_right_before
        reference_delta_left = target_left - clip_reference_left
        reference_delta_right = target_right - clip_reference_right
        tracking_error_before_left = command_left_before - publish_current_left
        tracking_error_before_right = command_right_before - publish_current_right
        tracking_error_after_left = target_left - publish_current_left
        tracking_error_after_right = target_right - publish_current_right
        clip_residual_left = raw_target_left - clipped_target_left
        clip_residual_right = raw_target_right - clipped_target_right

        ik_joint_left = None
        ik_joint_right = None
        ik_left_result = None
        ik_right_result = None

        if command_publish_mode == 'direct':
            if eef_action_mode:
                if eef_ik is None:
                    rospy.logerr_throttle(1.0, "EEF action mode requested but IK solver is not initialized.")
                    return False
                if 'left' not in eef_ik.calibration:
                    eef_ik.calibrate('left', ik_seed_left, command_left_before)
                if 'right' not in eef_ik.calibration:
                    eef_ik.calibrate('right', ik_seed_right, command_right_before)
                ik_elapsed = (1.0 / rate_hz if last_ik_publish_time is None
                              else max(time.monotonic() - last_ik_publish_time, 1e-6))
                try:
                    ik_left_result = eef_ik.solve('left', target_left, ik_seed_left, last_published_ik.get('left'), dt=ik_elapsed)
                    ik_right_result = eef_ik.solve('right', target_right, ik_seed_right, last_published_ik.get('right'), dt=ik_elapsed)
                except Exception as exc:  # noqa: BLE001
                    rospy.logwarn_throttle(1.0, f"EEF IK solve failed; skipping command publish: {exc}")
                    return False
                if (not ik_left_result['acceptable'] or not ik_right_result['acceptable']) and not eef_ik.publish_on_failure:
                    if action_logger is not None:
                        rejection = {
                            'wall_time': time.time(), 'request_id': active_chunk['request_id'],
                            'action_index': action_index, 'global_action_step': global_action_step,
                            'target_left': target_left.tolist(), 'target_right': target_right.tolist(),
                            'seed_left': ik_seed_left.tolist(), 'seed_right': ik_seed_right.tolist(),
                            'left': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                     for k, v in ik_left_result.items()},
                            'right': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                      for k, v in ik_right_result.items()},
                        }
                        try:
                            with open(action_logger['path'] + '.rejected.jsonl', 'a', encoding='utf-8') as f:
                                f.write(json.dumps(rejection) + '\n')
                        except OSError as exc:
                            rospy.logwarn_throttle(1.0, f"Cannot record rejected IK command: {exc}")
                    rospy.logwarn_throttle(
                        1.0,
                        "EEF IK command rejected (pose/joint/singularity/hold constraints); skipping publish: "
                        f"left_pos={ik_left_result['position_error_m']:.4f}m "
                        f"left_rot={ik_left_result['orientation_error_rad']:.4f}rad "
                        f"right_pos={ik_right_result['position_error_m']:.4f}m "
                        f"right_rot={ik_right_result['orientation_error_rad']:.4f}rad "
                        f"left_solution_pos={ik_left_result['solution_position_error_m']:.4f}m "
                        f"right_solution_pos={ik_right_result['solution_position_error_m']:.4f}m "
                        f"joint_delta_limited={ik_left_result['joint_delta_limited']}/{ik_right_result['joint_delta_limited']} "
                        f"wrist_guard_left={ik_left_result.get('wrist_singularity_avoidance', {})} "
                        f"wrist_guard_right={ik_right_result.get('wrist_singularity_avoidance', {})}"
                    )
                    return False
                if eef_ik.differential is not None:
                    for arm_name, result in (('left', ik_left_result), ('right', ik_right_result)):
                        limits = result['differential_ik']['active_limits']
                        if any(limits.values()):
                            message = (
                                f"EEF IK bounded tracking {arm_name}: active_limits={limits} "
                                f"target_reached={result['target_reached']} "
                                f"position_error={result['position_error_m']:.4f}m "
                                f"orientation_error={result['orientation_error_rad']:.4f}rad")
                            if result['target_reached']:
                                rospy.loginfo_throttle(1.0, message)
                            else:
                                rospy.logwarn_throttle(1.0, message)
                elif ik_left_result.get('joint_delta_limited') or ik_right_result.get('joint_delta_limited'):
                    rospy.logwarn_throttle(
                        1.0,
                        "EEF IK joint delta limited: "
                        f"left_delta={ik_left_result.get('joint_delta_norm', 0.0):.4f} "
                        f"left_unclipped={ik_left_result.get('unclipped_joint_delta_norm', 0.0):.4f} "
                        f"right_delta={ik_right_result.get('joint_delta_norm', 0.0):.4f} "
                        f"right_unclipped={ik_right_result.get('unclipped_joint_delta_norm', 0.0):.4f}"
                    )
                ik_joint_left = ik_left_result['joints']
                ik_joint_right = ik_right_result['joints']
                for arm_index, (arm_name, result, target) in enumerate((
                    ('left', ik_left_result, target_left), ('right', ik_right_result, target_right),
                )):
                    if result.get('differential_ik', {}).get('mode') == 'stalled':
                        rospy.logwarn_throttle(
                            1.0, f"Differential IK stalled on {arm_name}: "
                            f"position error={result['position_error_m']:.4f}m "
                            f"orientation error={result['orientation_error_rad']:.4f}rad")
                    if result.get('wrist_singularity_avoidance', {}).get('mode') == 'hold_last_safe':
                        rospy.logwarn_throttle(
                            1.0, f"Holding {arm_name} at last safe joint command: "
                            f"model target not reached; position error={result['position_error_m']:.4f}m "
                            f"orientation error={result['orientation_error_rad']:.4f}rad")
                        target[-1] = result['joints'][-1]
                        if gripper_candidate is not None:
                            # The guard holds the previously sent gripper too;
                            # do not commit a transition that was not published.
                            fraction = ((target[-1] - gripper_candidate.closed[arm_index]) /
                                        (gripper_candidate.opened[arm_index] - gripper_candidate.closed[arm_index]))
                            gripper_candidate.is_open[arm_index] = fraction >= 0.5
                            gripper_candidate.count[arm_index] = 0
                            gripper_transition['output_open'][arm_index] = bool(fraction >= 0.5)
                            gripper_transition['switched'][arm_index] = False
                            gripper_transition['pending_count'][arm_index] = 0
                publish_joint_pair(pub_l, pub_r, name_list, ik_joint_left, ik_joint_right)
                last_ik_publish_time = time.monotonic()
                eef_ik.commit('left', ik_left_result)
                eef_ik.commit('right', ik_right_result)
                last_published_ik['left'] = ik_joint_left.copy()
                last_published_ik['right'] = ik_joint_right.copy()
                if not published_first_command:
                    rospy.loginfo(
                        f"Published first IK joint command to {topics['cmd_joint_left']} and {topics['cmd_joint_right']}."
                    )
                    published_first_command = True
            else:
                js = JointState()
                js.header.stamp = rospy.Time.now()
                js.name = name_list
                js.position = target_left.tolist()
                pub_l.publish(js)
                js.position = target_right.tolist()
                pub_r.publish(js)
                if not published_first_command:
                    rospy.loginfo(
                        f"Published first command to {topics['cmd_joint_left']} and {topics['cmd_joint_right']}."
                    )
                    published_first_command = True
        else:
            with command_publish_lock:
                command_publish_state['start_time'] = time.monotonic()
                command_publish_state['duration'] = 1.0 / float(rate_hz)
                command_publish_state['start_left'] = command_left_before.copy()
                command_publish_state['start_right'] = command_right_before.copy()
                command_publish_state['target_left'] = target_left.copy()
                command_publish_state['target_right'] = target_right.copy()

        gripper_hysteresis = gripper_candidate
        command_left = target_left.copy()
        command_right = target_right.copy()
        last_command_left = command_left.copy()
        last_command_right = command_right.copy()

        vel_mat = active_chunk['vel']
        if use_base and vel_mat is not None and pub_v is not None:
            v = Twist()
            v.linear.x = float(np.clip(vel_mat[action_index, 0], -v_max, v_max))
            v.angular.z = float(np.clip(vel_mat[action_index, 1], -w_max, w_max))
            pub_v.publish(v)

        base_vel = None
        if vel_mat is not None:
            base_vel = vel_mat[action_index]
        now = rospy.Time.now()
        now_mono = time.monotonic()
        request_obs_age = {
            key: None if value is None else now_mono - value
            for key, value in active_chunk['request_obs_time'].items()
        }
        write_action_log(action_logger, {
            'wall_time': f'{time.time():.6f}',
            'ros_time': f'{now.to_sec():.6f}',
            'request_id': active_chunk['request_id'],
            'tau': active_chunk['executed_steps'],
            'global_action_step': global_action_step,
            'chunk_start_step': current_chunk_start_step,
            'action_index': action_index,
            'policy_latency_sec': f"{active_chunk['policy_latency_sec']:.6f}",
            'policy_latency_steps': active_chunk['policy_latency_steps'],
            'chunk_size': chunk_size,
            'received_chunk_size': active_chunk['received_chunk_size'],
            'steps_to_execute': active_chunk['steps_to_execute'],
            'discarded_steps': action_index,
            'rate_hz': rate_hz,
            'command_publish_rate_hz': f'{command_publish_rate_hz:.6f}',
            'command_publish_substeps': command_publish_substeps,
            'task_prompt': task_prompt,
            'action_mode': action_mode,
            'chunk_interpolation_enabled': interpolation_enabled,
            'chunk_interpolation_mode': interpolation_mode,
            'chunk_interpolation_factor': f'{interpolation_factor:.6f}',
            'request_obs_seq': json.dumps(active_chunk['request_obs_seq'], separators=(',', ':')),
            'request_obs_age_sec': json.dumps(request_obs_age, separators=(',', ':')),
            'request_snapshot_dir': active_chunk.get('request_snapshot_dir', ''),
            'action_chunk_path': active_chunk.get('action_chunk_path', ''),
            'rollout_sample_dir': active_chunk.get('rollout_sample_dir', ''),
            'rollout_action_path': active_chunk.get('rollout_action_path', ''),
            'request_fresh_camera_count': active_chunk['request_fresh_camera_count'],
            'request_fresh_camera_forced': active_chunk['request_fresh_camera_forced'],
            'delta_clip_enabled': delta_clip_enabled,
            'delta_clip_reference': delta_clip_reference,
            'command_delta_deadband_enabled': command_delta_deadband_enabled,
            'first_action_delta_scale_enabled': first_action_delta_scale_enabled,
            'first_action_delta_scale_coefficient': f'{first_action_delta_scale_coefficient:.6f}',
            'first_action_delta_scale_include_gripper': first_action_delta_scale_include_gripper,
            'first_action_delta_scale_applied': first_action_delta_scale_applied,
            'initial_pose_delta_override_enabled': initial_pose_override_enabled,
            'initial_pose_delta_override_arms': json.dumps(sorted(initial_pose_override_arms), separators=(',', ':')),
            'initial_pose_delta_override_include_gripper': initial_pose_override_include_gripper,
            'temporal_ensemble_count': temporal_ensemble_count,
            'temporal_ensemble_weights': vector_to_json(temporal_ensemble_weights),
            'current_left': vector_to_json(active_chunk['request_current_left']),
            'current_right': vector_to_json(active_chunk['request_current_right']),
            'request_measured_left': vector_to_json(active_chunk.get('request_measured_left', active_chunk['request_current_left'])),
            'request_measured_right': vector_to_json(active_chunk.get('request_measured_right', active_chunk['request_current_right'])),
            'publish_current_left': vector_to_json(publish_current_left),
            'publish_current_right': vector_to_json(publish_current_right),
            'clip_reference_left': vector_to_json(clip_reference_left),
            'clip_reference_right': vector_to_json(clip_reference_right),
            'command_left_before': vector_to_json(command_left_before),
            'command_right_before': vector_to_json(command_right_before),
            'raw_left': vector_to_json(raw_target_left),
            'raw_right': vector_to_json(raw_target_right),
            'ensembled_left': vector_to_json(ensembled_target_left),
            'ensembled_right': vector_to_json(ensembled_target_right),
            'filtered_left': vector_to_json(filtered_target_left),
            'filtered_right': vector_to_json(filtered_target_right),
            'target_left': vector_to_json(target_left),
            'target_right': vector_to_json(target_right),
            'ik_seed_joint_left': vector_to_json(ik_seed_left) if eef_action_mode else '',
            'ik_seed_joint_right': vector_to_json(ik_seed_right) if eef_action_mode else '',
            'ik_diagnostics_left': '' if ik_left_result is None else json.dumps({k: v for k, v in ik_left_result.items() if k != 'joints'}, separators=(',', ':')),
            'ik_diagnostics_right': '' if ik_right_result is None else json.dumps({k: v for k, v in ik_right_result.items() if k != 'joints'}, separators=(',', ':')),
            'ik_joint_left': vector_to_json(ik_joint_left),
            'ik_joint_right': vector_to_json(ik_joint_right),
            'ik_left_position_error_m': '' if ik_left_result is None else f"{ik_left_result['position_error_m']:.6f}",
            'ik_right_position_error_m': '' if ik_right_result is None else f"{ik_right_result['position_error_m']:.6f}",
            'ik_left_orientation_error_rad': '' if ik_left_result is None else f"{ik_left_result['orientation_error_rad']:.6f}",
            'ik_right_orientation_error_rad': '' if ik_right_result is None else f"{ik_right_result['orientation_error_rad']:.6f}",
            'raw_delta_left': vector_to_json(raw_delta_left),
            'raw_delta_right': vector_to_json(raw_delta_right),
            'applied_delta_left': vector_to_json(applied_delta_left),
            'applied_delta_right': vector_to_json(applied_delta_right),
            'reference_delta_left': vector_to_json(reference_delta_left),
            'reference_delta_right': vector_to_json(reference_delta_right),
            'tracking_error_before_left': vector_to_json(tracking_error_before_left),
            'tracking_error_before_right': vector_to_json(tracking_error_before_right),
            'tracking_error_after_left': vector_to_json(tracking_error_after_left),
            'tracking_error_after_right': vector_to_json(tracking_error_after_right),
            'clip_residual_left': vector_to_json(clip_residual_left),
            'clip_residual_right': vector_to_json(clip_residual_right),
            'command_delta_deadband_residual_left': vector_to_json(command_delta_deadband_residual_left),
            'command_delta_deadband_residual_right': vector_to_json(command_delta_deadband_residual_right),
            'raw_vs_publish_norm_left': norm_first_six(raw_target_left - publish_current_left),
            'raw_vs_publish_norm_right': norm_first_six(raw_target_right - publish_current_right),
            'ensembled_vs_publish_norm_left': norm_first_six(ensembled_target_left - publish_current_left),
            'ensembled_vs_publish_norm_right': norm_first_six(ensembled_target_right - publish_current_right),
            'target_vs_publish_norm_left': norm_first_six(target_left - publish_current_left),
            'target_vs_publish_norm_right': norm_first_six(target_right - publish_current_right),
            'reference_delta_norm_left': norm_first_six(reference_delta_left),
            'reference_delta_norm_right': norm_first_six(reference_delta_right),
            'applied_delta_norm_left': norm_first_six(applied_delta_left),
            'applied_delta_norm_right': norm_first_six(applied_delta_right),
            'tracking_error_before_norm_left': norm_first_six(tracking_error_before_left),
            'tracking_error_before_norm_right': norm_first_six(tracking_error_before_right),
            'tracking_error_after_norm_left': norm_first_six(tracking_error_after_left),
            'tracking_error_after_norm_right': norm_first_six(tracking_error_after_right),
            'clip_residual_norm_left': norm_first_six(clip_residual_left),
            'clip_residual_norm_right': norm_first_six(clip_residual_right),
            'command_delta_deadband_residual_norm_left': norm_first_six(command_delta_deadband_residual_left),
            'command_delta_deadband_residual_norm_right': norm_first_six(command_delta_deadband_residual_right),
            'publish_joint_age_left_sec': '' if jl_age is None else f'{jl_age:.6f}',
            'publish_joint_age_right_sec': '' if jr_age is None else f'{jr_age:.6f}',
            'policy_gripper_input_mode': active_chunk.get('policy_gripper_input', {}).get('mode', 'measured'),
            'left_gripper_request_measured': f'{float(active_chunk.get("request_measured_left", active_chunk["request_current_left"])[-1]):.6f}',
            'left_gripper_request_policy': f'{float(active_chunk["request_current_left"][-1]):.6f}',
            'left_gripper_request_commanded': '' if active_chunk.get('policy_gripper_input', {}).get('left_commanded') is None else f'{float(active_chunk.get("policy_gripper_input", {}).get("left_commanded")):.6f}',
            'right_gripper_request_measured': f'{float(active_chunk.get("request_measured_right", active_chunk["request_current_right"])[-1]):.6f}',
            'right_gripper_request_policy': f'{float(active_chunk["request_current_right"][-1]):.6f}',
            'right_gripper_request_commanded': '' if active_chunk.get('policy_gripper_input', {}).get('right_commanded') is None else f'{float(active_chunk.get("policy_gripper_input", {}).get("right_commanded")):.6f}',
            'left_gripper_publish': f'{float(publish_current_left[-1]):.6f}',
            'left_gripper_raw': f'{float(raw_target_left[-1]):.6f}',
            'left_gripper_ensembled': f'{float(ensembled_target_left[-1]):.6f}',
            'left_gripper_target': f'{float(target_left[-1]):.6f}',
            'right_gripper_publish': f'{float(publish_current_right[-1]):.6f}',
            'right_gripper_raw': f'{float(raw_target_right[-1]):.6f}',
            'right_gripper_ensembled': f'{float(ensembled_target_right[-1]):.6f}',
            'right_gripper_target': f'{float(target_right[-1]):.6f}',
            'gripper_hysteresis': json.dumps(gripper_transition),
            'base_vel': vector_to_json(base_vel),
        })

        active_chunk['executed_steps'] += 1
        global_action_step += 1
        chunk_history = [
            chunk for chunk in chunk_history
            if global_action_step < chunk['start_step'] + chunk_live_steps(chunk)
        ][-temporal_max_history_chunks:]
        return True

    try:
        while not rospy.is_shutdown():
            if not enable_state:
                rate.sleep()
                continue

            has_obs = have_obs(use_base=use_base, require_eef=eef_action_mode)
            has_live_chunk = bool(live_chunks_for_step(global_action_step))
            if not has_obs and not has_live_chunk and pending_request is None:
                now = time.monotonic()
                if now - last_obs_wait_warn >= 2.0:
                    with lock:
                        missing = [
                            key for key in (('front', 'left', 'right', 'jl', 'jr') + (('eef_l', 'eef_r') if eef_action_mode else ()))
                            if buf[key] is None
                        ]
                        if use_base and buf['odom'] is None:
                            missing.append('odom')
                    rospy.logwarn(f"Waiting for observations: missing={missing}")
                    last_obs_wait_warn = now
                rate.sleep()
                continue

            if command_left is None or command_right is None:
                pkt = snapshot(task_prompt, use_base=use_base, include_eef=eef_action_mode) if has_obs else None
                if pkt is None:
                    rate.sleep()
                    continue
                if eef_action_mode:
                    command_left = np.array(pkt['current_eef_left'], dtype=np.float32)
                    command_right = np.array(pkt['current_eef_right'], dtype=np.float32)
                else:
                    command_left = np.array(pkt['jleft'], dtype=np.float32)
                    command_right = np.array(pkt['jright'], dtype=np.float32)
                integrated_left = command_left.copy()
                integrated_right = command_right.copy()

            if pending_request is not None:
                try:
                    rep_frames = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.error.Again:
                    rep_frames = None
                    now = time.monotonic()
                    elapsed = now - pending_request['time']
                    if elapsed >= 2.0 and now - last_response_wait_warn >= 2.0:
                        rospy.logwarn(
                            f"Waiting for policy server response from {connect_addr}. "
                            "Continuing with available chunk actions."
                        )
                        last_response_wait_warn = now
                    if elapsed >= max(policy_response_timeout_sec, 0.1):
                        rospy.logwarn(
                            f"Policy server response timed out after {policy_response_timeout_sec:.1f}s; "
                            "recreating ZeroMQ socket and keeping any available chunk actions."
                        )
                        sock.close(0)
                        sock = make_policy_socket(ctx, connect_addr, recv_timeout_ms, socket_type)
                        pending_request = None
                if rep_frames:
                    pending_snapshot = pending_request
                    pending_request = None
                    ingest_policy_response(rep_frames, pending_snapshot)

            if pending_request is None and has_obs:
                pkt = snapshot(task_prompt, use_base=use_base, include_eef=eef_action_mode)
                if pkt is not None:
                    fresh_enough, stale_obs = obs_is_fresh_enough(pkt)
                    if not fresh_enough:
                        now = time.monotonic()
                        if now - last_stale_obs_warn >= 1.0:
                            formatted = [
                                f"{key}={age:.3f}s" if age is not None else f"{key}=None"
                                for key, age in stale_obs
                            ]
                            rospy.logwarn(
                                "Delaying policy request because observations are stale: "
                                f"{formatted}"
                            )
                            last_stale_obs_warn = now
                    else:
                        camera_seq = tuple(pkt['obs_seq'][key] for key in ('front', 'left', 'right'))
                        if last_request_camera_seq is None:
                            fresh_flags = (True, True, True)
                        else:
                            fresh_flags = tuple(
                                seq > last
                                for seq, last in zip(camera_seq, last_request_camera_seq)
                            )
                        fresh_camera_count = sum(1 for is_fresh in fresh_flags if is_fresh)
                        fresh_camera_set = fresh_camera_count == 3
                        live_chunks = live_chunks_for_step(global_action_step)
                        remaining_steps = 0
                        if live_chunks:
                            active_for_budget = live_chunks[-1]
                            remaining_steps = (
                                active_for_budget['start_step']
                                + chunk_live_steps(active_for_budget)
                                - global_action_step
                            )
                        should_wait_for_chunk = (
                            policy_request_when_live_chunk == 'wait'
                            and bool(live_chunks)
                        )
                        force_fresh_fallback = (
                            not should_wait_for_chunk
                            and last_request_camera_seq is not None
                            and fresh_camera_count > 0
                            and (not live_chunks or remaining_steps <= max(6, latency_fixed_steps))
                        )
                        if should_wait_for_chunk:
                            now = time.monotonic()
                            if now - last_stale_obs_warn >= 2.0:
                                rospy.loginfo(
                                    "Delaying policy request until current chunk is fully consumed: "
                                    f"remaining_steps={remaining_steps}"
                                )
                                last_stale_obs_warn = now
                        elif fresh_camera_set:
                            send_policy_request(pkt, fresh_camera_count, forced_fresh_fallback=False)
                        elif force_fresh_fallback:
                            send_policy_request(pkt, fresh_camera_count, forced_fresh_fallback=True)
                        else:
                            now = time.monotonic()
                            if now - last_stale_obs_warn >= 2.0:
                                rospy.loginfo(
                                    "Delaying policy request until all camera images refresh: "
                                    f"last={last_request_camera_seq} current={camera_seq} "
                                    f"fresh={fresh_camera_count}/3 remaining_steps={remaining_steps}"
                                )
                                last_stale_obs_warn = now

            live_chunks = live_chunks_for_step(global_action_step)
            if live_chunks:
                active_chunk = select_chunk_for_step(global_action_step)
                if active_chunk is not None:
                    publish_action_from_chunk(active_chunk)

            rate.sleep()

    except (KeyboardInterrupt, rospy.ROSInterruptException):
        rospy.loginfo("cobotmagic_policy_bridge_node interrupted, shutting down.")
    finally:
        run_shutdown_safety(pub_l, pub_r, pub_v, enable_pub, name_list, cfg)
        if action_logger is not None:
            action_logger['file'].flush()
            action_logger['file'].close()
            rospy.loginfo(f"Action command log saved: {action_log_path}")
        if rollout_logger is not None and rollout_logger.get('h5_file') is not None:
            rollout_logger['h5_file'].flush()
            rollout_logger['h5_file'].close()
            rospy.loginfo(f"Rollout HDF5 saved: {rollout_logger.get('h5_path')}")
        sock.close(0)

if __name__ == '__main__':
    main()
