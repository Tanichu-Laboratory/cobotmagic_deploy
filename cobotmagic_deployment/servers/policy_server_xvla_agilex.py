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
from cobotmagic_deployment.common.policy_server_runtime import (
    CONFIG_DIR, add_server_arguments, load_server_config, run_server,
)


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

    def warmup(self, cfg: dict[str, Any]) -> None:
        task_prompt = str(cfg["task_prompt"])
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
    return load_server_config(path, "xvla", backends=("xvla",))


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


def parse_args() -> argparse.Namespace:
    parser = add_server_arguments(argparse.ArgumentParser(), CONFIG_DIR / "config_xvla_agilex.yaml")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download model files from Hugging Face")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    if args.local_files_only:
        cfg["local_files_only"] = True
    run_server(args, cfg, XVLAAgilexPolicy, name=str(cfg["backend"]), action_mode="eef_absolute",
               test_header=test_header)


if __name__ == "__main__":
    main()
