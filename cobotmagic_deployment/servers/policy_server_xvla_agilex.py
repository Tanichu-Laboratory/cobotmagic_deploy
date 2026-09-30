#!/usr/bin/env python3
"""Serve X-VLA EEF-action checkpoints through the CobotMagic ZMQ protocol."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import yaml
from cobotmagic_deployment.common.policy_server_protocol import bind_server, recv_packet, send_actions, send_empty


IDENTITY_ROT6D = np.asarray([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)



def normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def rot6d_to_euler_xyz(rot6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    rot6d = np.asarray(rot6d, dtype=np.float32)
    a1 = rot6d[..., 0:5:2]
    a2 = rot6d[..., 1:6:2]
    b1 = normalize(a1)
    b2 = normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2)
    mats = np.stack((b1, b2, b3), axis=-1)
    return Rotation.from_matrix(mats).as_euler("xyz").astype(np.float32)


def ee6d20_to_action14(actions20: np.ndarray) -> np.ndarray:
    actions20 = np.asarray(actions20, dtype=np.float32)
    if actions20.ndim != 2 or actions20.shape[1] < 20:
        raise ValueError(f"expected X-VLA EE6D action shape (T, >=20), got {actions20.shape}")

    left_euler = rot6d_to_euler_xyz(actions20[:, 3:9])
    right_euler = rot6d_to_euler_xyz(actions20[:, 13:19])
    left = np.concatenate([actions20[:, :3], left_euler, actions20[:, 9:10]], axis=-1)
    right = np.concatenate([actions20[:, 10:13], right_euler, actions20[:, 19:20]], axis=-1)
    return np.concatenate([left, right], axis=-1).astype(np.float32)


def request_proprio20(header: dict[str, Any]) -> np.ndarray:
    if "xvla_proprio" in header:
        proprio = np.asarray(header["xvla_proprio"], dtype=np.float32)
        if proprio.shape[-1] != 20:
            raise ValueError(f"header xvla_proprio must be 20D, got {proprio.shape}")
        return proprio

    jleft = np.asarray(header["jleft"], dtype=np.float32)
    jright = np.asarray(header["jright"], dtype=np.float32)
    if jleft.shape[-1] < 7 or jright.shape[-1] < 7:
        raise ValueError(f"expected jleft/jright to be 7D, got {jleft.shape} and {jright.shape}")

    proprio = np.zeros(20, dtype=np.float32)
    proprio[0:3] = jleft[:3]
    proprio[3:9] = IDENTITY_ROT6D
    proprio[9] = jleft[6]
    proprio[10:13] = jright[:3]
    proprio[13:19] = IDENTITY_ROT6D
    proprio[19] = jright[6]
    return proprio


class MockXVLAPolicy:
    def __init__(self, chunk_size: int = 30) -> None:
        self.chunk_size = chunk_size

    def predict(self, header: dict[str, Any], _imgs: dict[str, np.ndarray]) -> np.ndarray:
        if header.get("current_eef_left") is not None and header.get("current_eef_right") is not None:
            left = np.asarray(header["current_eef_left"], dtype=np.float32)
            right = np.asarray(header["current_eef_right"], dtype=np.float32)
            current = np.concatenate([left[:7], right[:7]], axis=0)
        else:
            jleft = np.asarray(header["jleft"], dtype=np.float32)
            jright = np.asarray(header["jright"], dtype=np.float32)
            current = np.concatenate([jleft[:7], jright[:7]], axis=0)
        return np.repeat(current[None, :], self.chunk_size, axis=0).astype(np.float32)


class XVLAAgilexPolicy:
    def __init__(self, cfg: dict[str, Any]) -> None:
        repo_path = Path(cfg.get("repo_path", "/workspace/project/X-VLA")).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        import torch
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        self.torch = torch
        self.device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
        dtype_name = str(cfg.get("torch_dtype", "float32"))
        self.dtype = getattr(torch, dtype_name)
        self.model_path = str(cfg["model_path"])
        self.processor_path = str(cfg.get("processor_path") or cfg["model_path"])
        self.domain_id = int(cfg.get("domain_id", 6))
        self.steps = int(cfg.get("steps", 10))
        self.local_files_only = bool(cfg.get("local_files_only", False))

        print(
            f"[xvla] loading processor={self.processor_path!r} "
            f"model={self.model_path!r} device={self.device} local_files_only={self.local_files_only}",
            flush=True,
        )
        self.processor = XVLAProcessor.from_pretrained(
            self.processor_path,
            local_files_only=self.local_files_only,
        )
        self.model = XVLA.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
        ).to(self.device).to(self.dtype)
        self.model.eval()
        torch.set_grad_enabled(False)
        print("[xvla] model loaded", flush=True)

    def predict(self, header: dict[str, Any], imgs: dict[str, np.ndarray]) -> np.ndarray:
        from PIL import Image

        task_prompt = str(header.get("task_prompt", "demo-task"))
        images = [
            Image.fromarray(imgs["front"]),
            Image.fromarray(imgs["left"]),
            Image.fromarray(imgs["right"]),
        ]
        inputs = self.processor(images, task_prompt)
        proprio = request_proprio20(header)

        def to_model(t: Any):
            if not isinstance(t, self.torch.Tensor):
                t = self.torch.as_tensor(t)
            if t.is_floating_point():
                return t.to(device=self.device, dtype=self.dtype)
            return t.to(device=self.device)

        model_inputs = {key: to_model(value) for key, value in inputs.items()}
        model_inputs.update(
            {
                "proprio": to_model(proprio[None, :]),
                "domain_id": self.torch.tensor([self.domain_id], dtype=self.torch.long, device=self.device),
            }
        )
        action20 = (
            self.model.generate_actions(**model_inputs, steps=self.steps)
            .squeeze(0)
            .float()
            .cpu()
            .numpy()
        )
        return ee6d20_to_action14(action20)

    def warmup(self, cfg: dict[str, Any], task_prompt: str) -> None:
        image_hw = cfg.get("warmup_image_hw", [480, 640])
        height, width = int(image_hw[0]), int(image_hw[1])
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        header = {
            "task_prompt": task_prompt,
            "jleft": [0.0] * 7,
            "jright": [0.0] * 7,
        }
        started = time.perf_counter()
        actions = self.predict(header, {"front": dummy, "left": dummy, "right": dummy})
        print(
            f"[xvla] warmup done action_shape={actions.shape} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = str(raw.get("policy_backend", "xvla")).lower()
    if backend != "xvla":
        raise ValueError(f"Unsupported policy_backend={backend!r}; expected xvla")
    section = "xvla"
    if section not in raw:
        raise KeyError(f"{path} does not contain a {section!r} section")
    cfg = dict(raw[section])
    cfg["backend"] = backend
    cfg["task_prompt"] = raw.get("task_prompt", "demo-task")
    cfg["server_bind"] = raw.get("zmq", {}).get("server_bind", "tcp://127.0.0.1:5555")
    cfg["socket_type"] = raw.get("zmq", {}).get("socket_type", "req")
    return cfg



def test_header(task_prompt: str) -> dict[str, Any]:
    left_cmd = np.asarray([0.1528, -0.1017, 0.2526, 0.0, 0.0, 0.0, 0.0250], dtype=np.float32)
    right_cmd = np.asarray([0.1847, 0.1075, 0.2526, 0.0, 0.0, 0.0, 0.0250], dtype=np.float32)
    return {
        "task_prompt": task_prompt,
        "control_hz": 5.0,
        "jleft": left_cmd.tolist(),
        "jright": right_cmd.tolist(),
        "current_eef_left": left_cmd.tolist(),
        "current_eef_right": right_cmd.tolist(),
        "xvla_proprio": np.concatenate([
            left_cmd[:3], IDENTITY_ROT6D, left_cmd[6:7],
            right_cmd[:3], IDENTITY_ROT6D, right_cmd[6:7],
        ]).astype(np.float32).tolist(),
    }


def print_action_stats(label: str, actions14: np.ndarray) -> None:
    actions14 = np.asarray(actions14, dtype=np.float32)
    print(
        f"[{label}] inference action_shape={actions14.shape} finite={bool(np.isfinite(actions14).all())} "
        f"min={float(np.nanmin(actions14)):.6f} max={float(np.nanmax(actions14)):.6f} "
        f"mean={float(np.nanmean(actions14)):.6f} std={float(np.nanstd(actions14)):.6f}",
        flush=True,
    )
    names = ["left", "right"]
    for arm, sl in zip(names, (slice(0, 7), slice(7, 14))):
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


def run_inference_test(policy: Any, cfg: dict[str, Any]) -> None:
    image_hw = cfg.get("warmup_image_hw", [480, 640])
    height, width = int(image_hw[0]), int(image_hw[1])
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    header = test_header(str(cfg["task_prompt"]))
    started = time.perf_counter()
    actions14 = policy.predict(header, {"front": dummy, "left": dummy, "right": dummy})
    elapsed = time.perf_counter() - started
    label = str(cfg.get("backend", "policy"))
    print_action_stats(label, actions14)
    print(f"[{label}] inference elapsed={elapsed:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "config_xvla_agilex.yaml")
    parser.add_argument("--bind", default="", help="Override zmq.server_bind from config YAML")
    parser.add_argument("--mock", action="store_true", help="Start protocol-compatible server without loading X-VLA")
    parser.add_argument("--startup-test", action="store_true", help="Bind the socket, print readiness, and exit")
    parser.add_argument("--inference-test", action="store_true", help="Run one dummy inference, print value ranges, and exit")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files from Hugging Face")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    if args.bind:
        cfg["server_bind"] = args.bind
    if args.local_files_only:
        cfg["local_files_only"] = True

    backend = str(cfg.get("backend", "xvla"))
    if args.mock:
        policy: Any = MockXVLAPolicy(chunk_size=int(cfg.get("chunk_size", 30)))
        print(f"[{backend}] mock policy enabled; model weights are not loaded", flush=True)
    else:
        policy = XVLAAgilexPolicy(cfg)
        if bool(cfg.get("warmup", False)):
            policy.warmup(cfg, str(cfg["task_prompt"]))

    if args.inference_test:
        run_inference_test(policy, cfg)
        return

    sock, server_kind_name = bind_server(str(cfg["server_bind"]), str(cfg["socket_type"]))
    print(
        f"[{cfg.get('backend', 'xvla')}] CobotMagic server bind {cfg['server_bind']} "
        f"using ZMQ {server_kind_name} ({cfg['socket_type']} protocol)",
        flush=True,
    )

    if args.startup_test:
        sock.close(0)
        print(f"[{cfg.get('backend', 'xvla')}] startup test passed", flush=True)
        return

    while True:
        try:
            print(f"[{cfg.get('backend', 'xvla')}] waiting for request", flush=True)
            header, imgs = recv_packet(sock)
            started = time.perf_counter()
            control_hz = float(header.get("control_hz", 20.0))
            print(
                f"[{cfg.get('backend', 'xvla')}] request received task={header.get('task_prompt', 'demo-task')!r} "
                f"control_hz={control_hz}",
                flush=True,
            )
            actions14 = policy.predict(header, imgs)
            send_actions(sock, actions14, control_hz=control_hz, action_mode="eef_absolute")
            print(
                f"[{cfg.get('backend', 'xvla')}] response sent action_shape={actions14.shape} "
                f"elapsed={time.perf_counter() - started:.3f}s",
                flush=True,
            )
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[{cfg.get('backend', 'xvla')}] request error: {exc}", flush=True)
            send_empty(sock, str(exc), action_mode="eef_absolute")

    sock.close(0)


if __name__ == "__main__":
    main()
