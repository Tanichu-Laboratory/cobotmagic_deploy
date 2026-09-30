#!/usr/bin/env python3
"""Shared ZeroMQ wire protocol between the CobotMagic bridges and policy servers.

Request: one JSON header frame followed by front/left/right JPEG frames.
Response: one JSON header frame followed by float32 left/right action frames
(and an optional base-velocity frame when ``has_vel`` is true).

This module must stay importable from the ROS bridge environment (Python 3.8).
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np
import zmq


IMAGE_KEYS = ("front", "left", "right")


def encode_jpeg(image: np.ndarray, quality: int = 80) -> bytes | None:
    """Encode a BGR image as JPEG, returning ``None`` on failure."""
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return encoded.tobytes()


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


def _socket_kind(name: str, req_kind: int) -> int:
    normalized = str(name).lower()
    if normalized == "pair":
        return zmq.PAIR
    if normalized == "req":
        return req_kind
    raise ValueError(f"Unsupported zmq.socket_type={name!r}; expected 'pair' or 'req'.")


def client_socket_kind(name: str) -> int:
    """Map ``zmq.socket_type`` to the bridge (client) socket kind."""
    return _socket_kind(name, zmq.REQ)


def server_socket_kind(name: str) -> int:
    """Map the client-facing socket protocol name to its server socket kind."""
    return _socket_kind(name, zmq.REP)


def bind_server(endpoint: str, socket_type: str) -> tuple[zmq.Socket, str]:
    """Create and bind a policy server socket, returning its display name too."""
    sock = zmq.Context.instance().socket(server_socket_kind(socket_type))
    sock.bind(endpoint)
    kind_name = "REP" if socket_type.lower() == "req" else "PAIR"
    return sock, kind_name


def send_empty(sock: zmq.Socket, message: str, *, action_mode: str | None) -> None:
    """Send a protocol-compatible error response."""
    header = {"chunk_size": 0, "error": message}
    if action_mode is not None:
        header["action_mode"] = action_mode
    sock.send_multipart([json.dumps(header, separators=(",", ":")).encode("utf-8")])


def send_actions(
    sock: zmq.Socket,
    actions: np.ndarray,
    control_hz: float,
    *,
    action_mode: str | None,
    vel: np.ndarray | None = None,
) -> None:
    """Send a dual-arm 14D action chunk using the shared binary protocol.

    ``action_mode=None`` omits the field so the bridge skips its mode check.
    ``vel`` is an optional ``(T, 2)`` base-velocity chunk.
    """
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
        "has_vel": vel is not None,
        "has_model_raw_action": False,
        "control_hz": control_hz,
    }
    if action_mode is not None:
        header["action_mode"] = action_mode
    frames = [
        json.dumps(header, separators=(",", ":")).encode("utf-8"),
        left.tobytes(order="C"),
        right.tobytes(order="C"),
    ]
    if vel is not None:
        frames.append(np.ascontiguousarray(vel, dtype=np.float32).tobytes(order="C"))
    sock.send_multipart(frames)
