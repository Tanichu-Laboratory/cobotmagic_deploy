"""Reusable runtime for CobotMagic policy servers.

A new model only needs a backend object with::

    action_mode = 'absolute'            # or 'eef_absolute' / 'velocity' / None
    def predict(self, header, images):  # -> (T, 14) float array, or (actions, vel)
        ...
    def warmup(self, cfg):              # optional
        ...

``header`` is the bridge request header (``task_prompt``, ``jleft``/``jright``,
``current_eef_left``/``right`` in EEF mode, ``episode_start``...), ``images``
maps ``front``/``left``/``right`` to HxWx3 RGB uint8 arrays. Each action row is
``[left 7D, right 7D]`` in the bridge's ``ros.action_mode`` representation.

:func:`run_server` then provides config loading, ``--bind``, ``--mock``,
``--startup-test``, ``--inference-test``, warmup, the request loop and error
replies. See ``servers/policy_server_template.py``.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from cobotmagic_deployment.common.policy_server_protocol import bind_server, recv_packet, send_actions, send_empty

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_server_config(path: str | Path, section: str, backends: tuple[str, ...] = ()) -> dict[str, Any]:
    """Load a bridge/server YAML and flatten the model section for a server.

    Returns the ``section`` mapping plus ``backend``, ``task_prompt``,
    ``server_bind``, ``socket_type`` and ``_raw`` (the full YAML).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = str(raw.get("policy_backend", backends[0] if backends else section)).lower()
    if backends and backend not in backends:
        expected = " or ".join(backends)
        raise ValueError(f"Unsupported policy_backend={backend!r}; expected {expected}")
    if section not in raw:
        raise KeyError(f"{path} does not contain a {section!r} section")
    cfg = dict(raw[section])
    cfg["backend"] = backend
    cfg["task_prompt"] = raw.get("task_prompt", "demo-task")
    cfg["server_bind"] = raw.get("zmq", {}).get("server_bind", "tcp://127.0.0.1:5555")
    cfg["socket_type"] = raw.get("zmq", {}).get("socket_type", "req")
    cfg["_raw"] = raw
    cfg["_config_dir"] = str(path.parent)
    return cfg


class EchoStatePolicy:
    """Mock backend: repeat the current state (EEF pose if sent, else joints)."""

    def __init__(self, chunk_size: int = 30, action_mode: str | None = None) -> None:
        self.chunk_size = chunk_size
        self.action_mode = action_mode

    def predict(self, header: dict[str, Any], _images: dict[str, np.ndarray]) -> np.ndarray:
        if header.get("current_eef_left") is not None and header.get("current_eef_right") is not None:
            left = np.asarray(header["current_eef_left"], dtype=np.float32)
            right = np.asarray(header["current_eef_right"], dtype=np.float32)
        else:
            left = np.asarray(header["jleft"], dtype=np.float32)
            right = np.asarray(header["jright"], dtype=np.float32)
        current = np.concatenate([left[:7], right[:7]], axis=0)
        return np.repeat(current[None, :], self.chunk_size, axis=0).astype(np.float32)


def dummy_images(image_hw=(480, 640)) -> dict[str, np.ndarray]:
    height, width = int(image_hw[0]), int(image_hw[1])
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    return {"front": dummy, "left": dummy, "right": dummy}


def split_prediction(prediction: Any) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(prediction, tuple):
        actions, vel = prediction
        return np.asarray(actions, dtype=np.float32), None if vel is None else np.asarray(vel, dtype=np.float32)
    return np.asarray(prediction, dtype=np.float32), None


def print_action_stats(label: str, actions14: np.ndarray, per_arm: bool = True) -> None:
    actions14 = np.asarray(actions14, dtype=np.float32)
    print(
        f"[{label}] inference action_shape={actions14.shape} finite={bool(np.isfinite(actions14).all())} "
        f"min={float(np.nanmin(actions14)):.6f} max={float(np.nanmax(actions14)):.6f} "
        f"mean={float(np.nanmean(actions14)):.6f} std={float(np.nanstd(actions14)):.6f}",
        flush=True,
    )
    if per_arm:
        for arm, sl in zip(("left", "right"), (slice(0, 7), slice(7, 14))):
            arr = actions14[:, sl]
            print(
                f"[{label}] {arm} xyz_min={np.nanmin(arr[:, :3], axis=0).round(6).tolist()} "
                f"xyz_max={np.nanmax(arr[:, :3], axis=0).round(6).tolist()} "
                f"rpy_min={np.nanmin(arr[:, 3:6], axis=0).round(6).tolist()} "
                f"rpy_max={np.nanmax(arr[:, 3:6], axis=0).round(6).tolist()} "
                f"gripper_min={float(np.nanmin(arr[:, 6])):.6f} gripper_max={float(np.nanmax(arr[:, 6])):.6f}",
                flush=True,
            )
    print(f"[{label}] first={actions14[0].round(6).tolist()}", flush=True)
    print(f"[{label}] last={actions14[-1].round(6).tolist()}", flush=True)


def run_inference_test(policy: Any, header: dict[str, Any], images: dict[str, np.ndarray],
                       label: str, per_arm: bool = True) -> np.ndarray:
    started = time.perf_counter()
    actions14, _ = split_prediction(policy.predict(header, images))
    elapsed = time.perf_counter() - started
    print_action_stats(label, actions14, per_arm=per_arm)
    print(f"[{label}] inference elapsed={elapsed:.3f}s", flush=True)
    return actions14


def serve(
    policy: Any,
    sock: Any,
    *,
    name: str,
    action_mode: str | None,
    control_hz: float | None = None,
    verbose: bool = True,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Serve requests until ``KeyboardInterrupt``.

    ``control_hz=None`` echoes the request's ``control_hz`` (default 20 Hz).
    Any exception while handling a request is reported with ``on_error``
    (default: print) and answered with an empty, protocol-compatible reply.
    """
    while True:
        try:
            if verbose:
                print(f"[{name}] waiting for request", flush=True)
            header, images = recv_packet(sock)
            started = time.perf_counter()
            reply_hz = float(header.get("control_hz", 20.0)) if control_hz is None else float(control_hz)
            if verbose:
                print(
                    f"[{name}] request received task={header.get('task_prompt', 'demo-task')!r} "
                    f"control_hz={reply_hz}",
                    flush=True,
                )
            actions, vel = split_prediction(policy.predict(header, images))
            send_actions(sock, actions, control_hz=reply_hz, action_mode=action_mode, vel=vel)
            if verbose:
                print(
                    f"[{name}] response sent action_shape={actions.shape} "
                    f"elapsed={time.perf_counter() - started:.3f}s",
                    flush=True,
                )
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001
            if on_error is None:
                print(f"[{name}] request error: {exc}", flush=True)
            else:
                on_error(exc)
            send_empty(sock, str(exc), action_mode=action_mode)


def print_traceback(name: str) -> Callable[[Exception], None]:
    def report(exc: Exception) -> None:
        print(f"[{name}] request error: {exc}", flush=True)
        traceback.print_exc()
    return report


def add_server_arguments(parser: argparse.ArgumentParser, default_config: str | Path) -> argparse.ArgumentParser:
    """Standard server options: --config --bind --mock --startup-test --inference-test."""
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--bind", default="", help="Override zmq.server_bind from config YAML")
    parser.add_argument("--mock", action="store_true", help="Serve the current state without loading the model")
    parser.add_argument("--startup-test", action="store_true", help="Bind the socket, print readiness, and exit")
    parser.add_argument("--inference-test", action="store_true",
                        help="Run one dummy inference, print value ranges, and exit")
    return parser


def run_server(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    make_policy: Callable[[dict[str, Any]], Any],
    *,
    name: str,
    action_mode: str | None,
    test_header: Callable[[str], dict[str, Any]],
    mock_chunk_size: int = 30,
    per_arm_stats: bool = True,
) -> None:
    """Standard server lifecycle built from :func:`add_server_arguments` options.

    ``make_policy(cfg)`` loads the real backend (skipped with ``--mock``);
    ``cfg['warmup']`` calls ``policy.warmup(cfg)`` when the backend has it.
    ``test_header(task_prompt)`` builds the request used by ``--inference-test``.
    """
    if args.bind:
        cfg["server_bind"] = args.bind
    if args.mock:
        policy: Any = EchoStatePolicy(chunk_size=int(cfg.get("chunk_size", mock_chunk_size)), action_mode=action_mode)
        print(f"[{name}] mock policy enabled; model weights are not loaded", flush=True)
    else:
        policy = make_policy(cfg)
        if bool(cfg.get("warmup", False)) and hasattr(policy, "warmup"):
            policy.warmup(cfg)

    if args.inference_test:
        header = test_header(str(cfg["task_prompt"]))
        run_inference_test(policy, header, dummy_images(cfg.get("warmup_image_hw", [480, 640])), name, per_arm_stats)
        return

    sock, server_kind_name = bind_server(str(cfg["server_bind"]), str(cfg["socket_type"]))
    print(
        f"[{name}] CobotMagic server bind {cfg['server_bind']} "
        f"using ZMQ {server_kind_name} ({cfg['socket_type']} protocol)",
        flush=True,
    )
    if args.startup_test:
        sock.close(0)
        print(f"[{name}] startup test passed", flush=True)
        return
    try:
        serve(policy, sock, name=name, action_mode=action_mode)
    finally:
        sock.close(0)
