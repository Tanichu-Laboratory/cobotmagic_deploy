"""CobotMagic native EEF commands <-> Piper raw EEF20 (RoboDojo / GM100).

OpenWAM stores column 0 then column 1 (not X-VLA's interleaved columns).
ROS poses are per-arm base xyz in metres and extrinsic xyz Euler in radians.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def gripper_endpoints(cfg):
    closed = np.asarray(cfg["closed"], dtype=np.float64)
    opened = np.asarray(cfg["open"], dtype=np.float64)
    if (closed.shape != (2,) or opened.shape != (2,)
            or not np.isfinite([closed, opened]).all()
            or np.any(np.abs(opened - closed) < 1e-6)):
        raise ValueError("gripper open/closed must be two distinct finite values per arm")
    return closed, opened


def gripper_action_open_normalized(cfg):
    """Model gripper output mapped to physical full-open (output shaping only)."""
    value = float(cfg.get("action_open_normalized", 1.0))
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("gripper.action_open_normalized must be finite and in (0, 1]")
    return value


def request_state(header, gripper):
    closed, opened = gripper_endpoints(gripper)
    arms = []
    for i, side in enumerate(("left", "right")):
        # Never reinterpret joint angles as EEF coordinates.
        key = "current_eef_" + side
        if key not in header:
            raise ValueError(f"OpenWAM requires {key}; use ros.action_mode=eef_absolute")
        pose = np.asarray(header[key], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"{key} must be finite xyz+rpy+gripper (7D)")
        mat = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
        grip = (pose[6] - closed[i]) / (opened[i] - closed[i])
        tolerance = float(gripper.get("sensor_tolerance", 0.05 * abs(opened[i] - closed[i])))
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("gripper sensor_tolerance must be finite and non-negative")
        normalized_tolerance = tolerance / abs(opened[i] - closed[i])
        if not -normalized_tolerance <= grip <= 1 + normalized_tolerance:
            raise ValueError(f"{side} gripper outside calibrated range: {pose[6]}")
        arms.append(np.r_[pose[:3], mat[:, 0], mat[:, 1], np.clip(grip, 0, 1)])
    return np.concatenate(arms).astype(np.float32)


def actions_to_bridge(actions, gripper):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 20 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError(f"expected finite de-normalized raw actions (T,20), got {actions.shape}")
    closed, opened = gripper_endpoints(gripper)
    action_open = gripper_action_open_normalized(gripper)
    arms = []
    for i, offset in enumerate((0, 10)):
        a = actions[:, offset:offset + 10]
        first = a[:, 3:6]
        norm = np.linalg.norm(first, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("degenerate OpenWAM rotation: first column")
        first = first / norm
        second = a[:, 6:9] - (first * a[:, 6:9]).sum(-1, keepdims=True) * first
        norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("degenerate OpenWAM rotation: parallel columns")
        second = second / norm
        matrix = np.stack((first, second, np.cross(first, second)), axis=-1)
        euler = Rotation.from_matrix(matrix).as_euler("xyz")
        grip = closed[i] + np.clip(a[:, 9:10] / action_open, 0, 1) * (opened[i] - closed[i])
        arms.append(np.concatenate((a[:, :3], euler, grip), axis=-1))
    return np.concatenate(arms, axis=-1).astype(np.float32)


def validate_checkpoint(cfg):
    dl = cfg["dataloader"]
    # Dataset names may differ; retain the exact deployment representation
    # checks below instead of weakening validation for arbitrary checkpoints.
    if dl.get("type") not in ("robodojo", "gm100"):
        raise ValueError(f"incompatible checkpoint dataloader.type: {dl.get('type')!r}")
    expected = {"variant": "real", "embodiment": "piper",
                "action_mode": "eef", "unify_action": True}
    for key, value in expected.items():
        if dl.get(key) != value:
            raise ValueError(f"incompatible checkpoint dataloader.{key}: {dl.get(key)!r}")
    if list(dl["unify_action_map"]) != ["0-9", "34-43"]:
        raise ValueError("unexpected OpenWAM EEF20 mapping")
    if list(dl["camera_layout"]) != ["cam_head", "cam_left_wrist", "cam_right_wrist"]:
        raise ValueError("unexpected camera layout")

