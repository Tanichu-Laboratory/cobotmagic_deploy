#!/usr/bin/env python3
"""Serve VLA EEF-action checkpoints through the CobotMagic ZMQ protocol."""

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
from policy_server_protocol import bind_server, recv_packet, send_actions, send_empty


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


def _to_chw_float(img: np.ndarray) -> np.ndarray:
    return img.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0


def _pad_last_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    if arr.shape[-1] == target_dim:
        return arr
    if arr.shape[-1] > target_dim:
        raise ValueError(f"cannot pad shape {arr.shape} to smaller dim {target_dim}")
    out = np.zeros((*arr.shape[:-1], target_dim), dtype=arr.dtype)
    out[..., : arr.shape[-1]] = arr
    return out


def euler_xyz_gripper_to_pose_wxyz(eef: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    eef = np.asarray(eef, dtype=np.float32)
    if eef.shape[-1] < 7:
        raise ValueError(f"expected EEF command/state as xyz+rpy+gripper, got {eef.shape}")
    q_xyzw = Rotation.from_euler("xyz", eef[3:6]).as_quat().astype(np.float32)
    return np.concatenate([eef[:3], q_xyzw[[3, 0, 1, 2]], eef[6:7]]).astype(np.float32)


def request_hy_state_wxyz(header: dict[str, Any]) -> np.ndarray:
    state = header.get("hy_eef_state_wxyz")
    if state is not None:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape[-1] != 16:
            raise ValueError(f"header hy_eef_state_wxyz must be 16D, got {arr.shape}")
        return arr

    left = header.get("current_eef_left")
    right = header.get("current_eef_right")
    if left is None or right is None:
        raise ValueError("Hy-VLA EEF mode requires hy_eef_state_wxyz or current_eef_left/right in the request header")
    return np.concatenate([
        euler_xyz_gripper_to_pose_wxyz(np.asarray(left, dtype=np.float32)),
        euler_xyz_gripper_to_pose_wxyz(np.asarray(right, dtype=np.float32)),
    ]).astype(np.float32)


def dual_pose_wxyz16_to_action14(actions16: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    actions16 = np.asarray(actions16, dtype=np.float32)
    if actions16.ndim != 2 or actions16.shape[1] < 16:
        raise ValueError(f"expected dual-arm pose action shape (T, >=16), got {actions16.shape}")
    left_q_xyzw = actions16[:, [4, 5, 6, 3]]
    right_q_xyzw = actions16[:, [12, 13, 14, 11]]
    left_euler = Rotation.from_quat(left_q_xyzw).as_euler("xyz").astype(np.float32)
    right_euler = Rotation.from_quat(right_q_xyzw).as_euler("xyz").astype(np.float32)
    left = np.concatenate([actions16[:, :3], left_euler, actions16[:, 7:8]], axis=-1)
    right = np.concatenate([actions16[:, 8:11], right_euler, actions16[:, 15:16]], axis=-1)
    return np.concatenate([left, right], axis=-1).astype(np.float32)


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


class HyVLAEEFPolicy:
    def __init__(self, cfg: dict[str, Any]) -> None:
        repo_path = Path(cfg.get("repo_path", "/workspace/project/Hy-Embodied-0.5-VLA")).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        import torch
        from robotwin_eval.policy_wrapper import HyVLAPolicyWrapper

        if not torch.cuda.is_available():
            raise RuntimeError("Hy-VLA deployment requires CUDA because HyVLAPolicyWrapper uses .cuda().")
        self.torch = torch
        dtype_name = str(cfg.get("torch_dtype", "bfloat16"))
        self.dtype = getattr(torch, dtype_name)
        requested_ckpt = Path(cfg["ckpt_path"]).expanduser().resolve()
        model_path = requested_ckpt / "model" if (requested_ckpt / "model").is_dir() else requested_ckpt
        norm_path = cfg.get("norm_path")
        if norm_path:
            norm_path = str(Path(norm_path).expanduser().resolve())
        else:
            candidates = [
                model_path / "norm_stats.pkl",
                requested_ckpt / "norm_stats.pkl",
                requested_ckpt.parent / "norm_stats.pkl",
            ]
            norm_path = next((str(c) for c in candidates if c.is_file()), None)
        if not norm_path:
            raise ValueError(f"norm_path is required for Hy-VLA; no norm_stats.pkl found near {requested_ckpt}")

        self.exc_action_size = int(cfg.get("exc_action_size", cfg.get("chunk_size", 50)))
        self.max_state_dim = int(cfg.get("max_state_dim", 32))
        self.task_prompt = str(cfg.get("task_prompt", "demo-task"))
        blend_mode = str(cfg.get("blend_mode", "rel_abs"))
        self.wrapper = HyVLAPolicyWrapper(
            ckpt_path=str(model_path),
            norm_path=norm_path,
            blend_mode=blend_mode,
            exc_action_size=self.exc_action_size,
            img_history_size=int(cfg.get("img_history_size", 6)),
            img_history_interval=int(cfg.get("img_history_interval", 1)),
            weight_dtype=self.dtype,
            vlm_model_path=cfg.get("vlm_model_path"),
        )
        if blend_mode == "rel_only" and bool(cfg.get("ignore_abs_stats_for_rel_only", True)):
            self.wrapper._has_abs_stats = False
            self.wrapper.norm_data["act_mean_abs"] = None
            self.wrapper.norm_data["act_std_abs"] = None
        torch.set_grad_enabled(False)
        print(
            f"[hy_vla] model loaded ckpt={model_path!s} norm={norm_path!s} "
            f"dtype={self.dtype} chunk={self.exc_action_size} blend={cfg.get('blend_mode', 'rel_abs')}",
            flush=True,
        )

    def _batch(self, header: dict[str, Any], imgs: dict[str, np.ndarray]) -> dict[str, Any]:
        state16 = request_hy_state_wxyz(header)
        state = _pad_last_dim(state16[None, :].astype(np.float32), self.max_state_dim)
        task_prompt = str(header.get("task_prompt", self.task_prompt))
        return {
            "observation.images.top_head": _to_chw_float(imgs["front"]),
            "observation.images.hand_left": _to_chw_float(imgs["left"]),
            "observation.images.hand_right": _to_chw_float(imgs["right"]),
            "observation.state": state,
            "task": [task_prompt],
            "raw_images.top_head": imgs["front"],
            "raw_images.hand_left": imgs["left"],
            "raw_images.hand_right": imgs["right"],
        }

    def predict(self, header: dict[str, Any], imgs: dict[str, np.ndarray]) -> np.ndarray:
        batch = self._batch(header, imgs)
        first = self.wrapper.get_action(batch)
        actions = [np.asarray(first, dtype=np.float32)]
        while len(actions) < self.exc_action_size and len(self.wrapper.action_cache) > 0:
            actions.append(np.asarray(self.wrapper.action_cache.popleft(), dtype=np.float32))
        self.wrapper.action_cache.clear()
        actions16 = np.stack(actions, axis=0)
        return dual_pose_wxyz16_to_action14(actions16)

    def warmup(self, cfg: dict[str, Any], task_prompt: str) -> None:
        image_hw = cfg.get("warmup_image_hw", [480, 640])
        height, width = int(image_hw[0]), int(image_hw[1])
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        header = test_header(task_prompt)
        started = time.perf_counter()
        actions = self.predict(header, {"front": dummy, "left": dummy, "right": dummy})
        self.wrapper.reset()
        print(
            f"[hy_vla] warmup done action_shape={actions.shape} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = str(raw.get("policy_backend", "xvla")).lower()
    if backend == "hy":
        backend = "hy_vla"
    if backend == "hy_vla":
        section = "hy_vla"
    elif backend == "xvla":
        section = "xvla"
    else:
        raise ValueError(f"Unsupported policy_backend={backend!r}; expected xvla or hy_vla")
    if section not in raw:
        raise KeyError(f"{path} does not contain a {section!r} section")
    cfg = dict(raw[section])
    cfg["backend"] = backend
    cfg["task_prompt"] = raw.get("task_prompt", "demo-task")
    cfg["server_bind"] = raw.get("zmq", {}).get("server_bind", "tcp://127.0.0.1:5555")
    cfg["socket_type"] = raw.get("zmq", {}).get("socket_type", "req")
    return cfg



def test_header(task_prompt: str) -> dict[str, Any]:
    left_pose_wxyz = np.asarray([0.1528, -0.1017, 0.2526, 1.0, 0.0, 0.0, 0.0, 0.0250], dtype=np.float32)
    right_pose_wxyz = np.asarray([0.1847, 0.1075, 0.2526, 1.0, 0.0, 0.0, 0.0, 0.0250], dtype=np.float32)
    left_cmd = np.asarray([left_pose_wxyz[0], left_pose_wxyz[1], left_pose_wxyz[2], 0.0, 0.0, 0.0, left_pose_wxyz[7]], dtype=np.float32)
    right_cmd = np.asarray([right_pose_wxyz[0], right_pose_wxyz[1], right_pose_wxyz[2], 0.0, 0.0, 0.0, right_pose_wxyz[7]], dtype=np.float32)
    state = np.concatenate([left_pose_wxyz, right_pose_wxyz]).astype(np.float32)
    return {
        "task_prompt": task_prompt,
        "control_hz": 5.0,
        "jleft": left_cmd.tolist(),
        "jright": right_cmd.tolist(),
        "current_eef_left": left_cmd.tolist(),
        "current_eef_right": right_cmd.tolist(),
        "hy_eef_state_wxyz": state.tolist(),
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
    parser.add_argument("--config", default=Path(__file__).with_name("config_xvla_agilex.yaml"))
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
        if backend == "hy_vla":
            policy = HyVLAEEFPolicy(cfg)
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
