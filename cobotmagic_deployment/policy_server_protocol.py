#!/usr/bin/env python3
"""Shared ZeroMQ wire protocol for CobotMagic policy servers."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import zmq


IMAGE_KEYS = ("front", "left", "right")


def decode_jpeg(payload: bytes) -> np.ndarray:
    """Decode a JPEG payload as an RGB image."""
    encoded = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode JPEG image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def recv_packet(sock: zmq.Socket) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Receive a request containing one JSON header and three JPEG images."""
    frames = sock.recv_multipart()
    expected_frames = 1 + len(IMAGE_KEYS)
    if len(frames) < expected_frames:
        raise ValueError(
            f"expected {expected_frames} frames (header + {len(IMAGE_KEYS)} images), got {len(frames)}"
        )
    header = json.loads(frames[0].decode("utf-8"))
    images = {key: decode_jpeg(payload) for key, payload in zip(IMAGE_KEYS, frames[1:])}
    return header, images


def server_socket_kind(name: str) -> int:
    """Map the client-facing socket protocol name to its server socket kind."""
    normalized = name.lower()
    if normalized == "pair":
        return zmq.PAIR
    if normalized == "req":
        return zmq.REP
    raise ValueError(f"Unsupported zmq.socket_type={name!r}; expected 'pair' or 'req'.")


def bind_server(endpoint: str, socket_type: str) -> tuple[zmq.Socket, str]:
    """Create and bind a policy server socket, returning its display name too."""
    sock = zmq.Context.instance().socket(server_socket_kind(socket_type))
    sock.bind(endpoint)
    kind_name = "REP" if socket_type.lower() == "req" else "PAIR"
    return sock, kind_name


def send_empty(sock: zmq.Socket, message: str, *, action_mode: str) -> None:
    """Send a protocol-compatible error response."""
    header = {"chunk_size": 0, "error": message, "action_mode": action_mode}
    sock.send_multipart([json.dumps(header, separators=(",", ":")).encode("utf-8")])


def send_actions(
    sock: zmq.Socket,
    actions: np.ndarray,
    control_hz: float,
    *,
    action_mode: str,
) -> None:
    """Send a dual-arm 14D action chunk using the shared binary protocol."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise ValueError(f"expected action chunk shape (T, >=14), got {actions.shape}")

    left = np.ascontiguousarray(actions[:, :7], dtype=np.float32)
    right = np.ascontiguousarray(actions[:, 7:14], dtype=np.float32)
    header = {
        "chunk_size": int(actions.shape[0]),
        "dtype": "float32",
        "left_stride": 7,
        "right_stride": 7,
        "has_vel": False,
        "has_model_raw_action": False,
        "action_mode": action_mode,
        "control_hz": control_hz,
    }
    sock.send_multipart(
        [
            json.dumps(header, separators=(",", ":")).encode("utf-8"),
            left.tobytes(order="C"),
            right.tobytes(order="C"),
        ]
    )
