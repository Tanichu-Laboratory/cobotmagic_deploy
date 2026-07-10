#!/usr/bin/env python3
"""Serve DreamZero AgileX checkpoints through the CobotMagic ZMQ protocol."""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pickle
import socket
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import torch.distributed as dist
import yaml
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from policy_server_protocol import bind_server, recv_packet, send_actions, send_empty


DREAMZERO_REPO = Path("/workspace/project/dreamzero")
if str(DREAMZERO_REPO) not in sys.path:
    sys.path.insert(0, str(DREAMZERO_REPO))

from groot.vla.data.schema import EmbodimentTag  # noqa: E402
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402


DEFAULT_CKPT = (
    "/workspace/project/dreamzero/checkpoints/"
    "robomind_agilex_3rgb_lora_3gpu_gbs6_pdbs2_chunk4_10k_save2k_wandb_v4_alloc/"
    "checkpoint-2000"
)


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = str(raw.get("policy_backend", "dreamzero")).lower()
    if backend != "dreamzero":
        raise ValueError(f"Unsupported policy_backend={backend!r}; expected dreamzero")
    cfg = dict(raw.get("dreamzero", {}))
    cfg["backend"] = backend
    cfg["task_prompt"] = raw.get("task_prompt", cfg.get("task_prompt", "demo-task"))
    cfg["server_bind"] = raw.get("zmq", {}).get("server_bind", "tcp://127.0.0.1:5555")
    cfg["socket_type"] = raw.get("zmq", {}).get("socket_type", "req")
    return cfg


def setup_dreamzero_flash(enabled: bool, num_dit_steps: int) -> None:
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    os.environ["ENABLE_DIT_CACHE"] = "true" if enabled else "false"
    if enabled:
        os.environ["NUM_DIT_STEPS"] = str(num_dit_steps)
    torch._dynamo.config.recompile_limit = 800


def setup_distributed(timeout_seconds: int) -> tuple[int, int, str, DeviceMesh | None, dist.ProcessGroup | None]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29537")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", os.environ["RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        timeout = _dt.timedelta(seconds=timeout_seconds)
        dist.init_process_group(backend=backend, timeout=timeout)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    device_mesh = None
    if torch.cuda.is_available():
        device_mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(world_size,),
            mesh_dim_names=("ip",),
        )

    signal_group = None
    if world_size > 1:
        signal_group = dist.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=_dt.timedelta(seconds=timeout_seconds),
        )

    return rank, world_size, device, device_mesh, signal_group


def broadcast_signal(value: int, signal_group: dist.ProcessGroup | None) -> None:
    if dist.get_world_size() == 1:
        return
    tensor = torch.tensor([value], dtype=torch.int32, device="cpu")
    dist.broadcast(tensor, src=0, group=signal_group)


def wait_signal(signal_group: dist.ProcessGroup | None) -> int:
    tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
    dist.broadcast(tensor, src=0, group=signal_group)
    return int(tensor.item())


def broadcast_obs_from_rank0(obs: dict[str, Any]) -> None:
    serialized = pickle.dumps(obs)
    size = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    payload = torch.frombuffer(serialized, dtype=torch.uint8).to("cuda")
    dist.broadcast(payload, src=0)


def receive_obs_on_worker() -> dict[str, Any]:
    size = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size, src=0)
    payload = torch.empty(int(size.item()), dtype=torch.uint8, device="cuda")
    dist.broadcast(payload, src=0)
    return pickle.loads(payload.cpu().numpy().tobytes())


def _stack_single_frame(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB image, got {img.shape}")
    return img[None, ...]


def make_observation(header: dict[str, Any], imgs: dict[str, np.ndarray], default_prompt: str) -> dict[str, Any]:
    left = np.asarray(header.get("jleft", [0.0] * 7), dtype=np.float64)
    right = np.asarray(header.get("jright", [0.0] * 7), dtype=np.float64)
    if left.shape[-1] < 7 or right.shape[-1] < 7:
        raise ValueError(f"jleft/jright must be at least 7D, got {left.shape} and {right.shape}")

    return {
        "video.front": _stack_single_frame(imgs["front"]),
        "video.left_wrist": _stack_single_frame(imgs["left"]),
        "video.right_wrist": _stack_single_frame(imgs["right"]),
        "state.left_arm": left[:7].reshape(1, 7),
        "state.right_arm": right[:7].reshape(1, 7),
        "annotation.human.task_description": str(header.get("task_prompt", default_prompt)),
    }


def batch_get(container: Any, key: str) -> Any:
    if isinstance(container, dict) and key in container:
        return container[key]
    try:
        return container[key]
    except Exception:
        pass
    if hasattr(container, key):
        return getattr(container, key)
    state = container.__getstate__() if hasattr(container, "__getstate__") else {}
    if isinstance(state, dict) and key in state:
        return state[key]
    raise KeyError(key)


def action_batch_to_numpy(action_batch: Any) -> np.ndarray:
    left = batch_get(action_batch, "action.left_arm")
    right = batch_get(action_batch, "action.right_arm")
    if isinstance(left, torch.Tensor):
        left = left.detach().float().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().float().cpu().numpy()
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.ndim == 1:
        left = left.reshape(1, -1)
    if right.ndim == 1:
        right = right.reshape(1, -1)
    if left.shape[-1] < 7 or right.shape[-1] < 7:
        raise ValueError(f"expected left/right action dims >=7, got {left.shape} and {right.shape}")
    horizon = min(left.shape[0], right.shape[0])
    return np.concatenate([left[:horizon, :7], right[:horizon, :7]], axis=-1).astype(np.float32)


class DreamZeroAgilexPolicy:
    def __init__(
        self,
        cfg: dict[str, Any],
        device: str,
        device_mesh: DeviceMesh | None,
        signal_group: dist.ProcessGroup | None,
    ) -> None:
        model_path = Path(cfg.get("model_path", DEFAULT_CKPT)).expanduser()
        if (model_path / "checkpoint-2000").is_dir():
            model_path = model_path / "checkpoint-2000"
        self.default_prompt = str(cfg.get("task_prompt", "demo-task"))
        self.signal_group = signal_group
        max_chunk_size = int(cfg.get("max_chunk_size", 1))
        model_overrides = [
            f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={max_chunk_size}",
        ]
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(str(cfg.get("embodiment", "xdof"))),
            model_path=str(model_path),
            device=device,
            device_mesh=device_mesh,
            model_config_overrides=model_overrides,
            skip_assert_delta_indices=True,
        )
        self._set_eval_transform_max_chunk_size(max_chunk_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[dreamzero] model loaded path={model_path} device={device} "
            f"flash_cache={os.environ.get('ENABLE_DIT_CACHE')} "
            f"num_dit_steps={os.environ.get('NUM_DIT_STEPS', 'default')} "
            f"max_chunk_size={max_chunk_size}",
            flush=True,
        )

    def _set_eval_transform_max_chunk_size(self, max_chunk_size: int) -> None:
        def visit(obj: Any) -> int:
            changed = 0
            if hasattr(obj, "max_chunk_size"):
                setattr(obj, "max_chunk_size", max_chunk_size)
                changed += 1
            for child in getattr(obj, "transforms", []) or []:
                changed += visit(child)
            return changed

        changed = visit(self.policy.eval_transform)
        print(f"[dreamzero] eval transform max_chunk_size override applied to {changed} transform(s)", flush=True)

    def predict(self, header: dict[str, Any], imgs: dict[str, np.ndarray]) -> np.ndarray:
        obs = make_observation(header, imgs, self.default_prompt)
        if dist.get_world_size() > 1:
            broadcast_signal(0, self.signal_group)
            broadcast_obs_from_rank0(obs)
            dist.barrier()
        batch, _video_pred = self.policy.lazy_joint_forward_causal(Batch(obs=obs))
        if dist.get_world_size() > 1:
            dist.barrier()
        return action_batch_to_numpy(batch.act)

    def warmup(self, cfg: dict[str, Any]) -> None:
        height, width = [int(v) for v in cfg.get("warmup_image_hw", [480, 640])]
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        header = test_header(str(cfg.get("task_prompt", self.default_prompt)))
        started = time.perf_counter()
        actions = self.predict(header, {"front": dummy, "left": dummy, "right": dummy})
        print(
            f"[dreamzero] warmup done action_shape={actions.shape} "
            f"elapsed={time.perf_counter() - started:.3f}s",
            flush=True,
        )


def worker_loop(policy: DreamZeroAgilexPolicy, signal_group: dist.ProcessGroup | None) -> None:
    print(f"[dreamzero] worker rank={dist.get_rank()} ready", flush=True)
    while True:
        signal = wait_signal(signal_group)
        if signal == 1:
            break
        if signal == 2:
            continue
        obs = receive_obs_on_worker()
        batch = Batch(obs=obs)
        dist.barrier()
        with torch.inference_mode():
            policy.policy.lazy_joint_forward_causal(batch)
        dist.barrier()


def test_header(task_prompt: str) -> dict[str, Any]:
    return {
        "task_prompt": task_prompt,
        "control_hz": 5.0,
        "jleft": [0.0] * 7,
        "jright": [0.0] * 7,
    }


def print_action_stats(label: str, actions14: np.ndarray) -> None:
    actions14 = np.asarray(actions14, dtype=np.float32)
    print(
        f"[{label}] inference action_shape={actions14.shape} finite={bool(np.isfinite(actions14).all())} "
        f"min={float(np.nanmin(actions14)):.6f} max={float(np.nanmax(actions14)):.6f} "
        f"mean={float(np.nanmean(actions14)):.6f} std={float(np.nanstd(actions14)):.6f}",
        flush=True,
    )
    print(f"[{label}] first={actions14[0].round(6).tolist()}", flush=True)
    print(f"[{label}] last={actions14[-1].round(6).tolist()}", flush=True)


def run_inference_test(policy: DreamZeroAgilexPolicy, cfg: dict[str, Any]) -> None:
    height, width = [int(v) for v in cfg.get("warmup_image_hw", [480, 640])]
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    header = test_header(str(cfg["task_prompt"]))
    started = time.perf_counter()
    actions14 = policy.predict(header, {"front": dummy, "left": dummy, "right": dummy})
    elapsed = time.perf_counter() - started
    print_action_stats("dreamzero", actions14)
    print(f"[dreamzero] inference elapsed={elapsed:.3f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).with_name("config_dreamzero_agilex.yaml"))
    parser.add_argument("--bind", default="", help="Override zmq.server_bind from config YAML")
    parser.add_argument("--flash", action="store_true", default=True, help="Enable DreamZero-Flash style DiT cache/mask")
    parser.add_argument("--no-flash", dest="flash", action="store_false", help="Disable Flash-style cache/mask")
    parser.add_argument("--num-dit-steps", type=int, default=5, help="Number of DiT compute steps for Flash mask")
    parser.add_argument("--startup-test", action="store_true", help="Bind the socket, print readiness, and exit")
    parser.add_argument("--inference-test", action="store_true", help="Run one dummy inference, print value ranges, and exit")
    parser.add_argument("--timeout-seconds", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    if args.bind:
        cfg["server_bind"] = args.bind

    setup_dreamzero_flash(args.flash, args.num_dit_steps)
    rank, world_size, device, device_mesh, signal_group = setup_distributed(args.timeout_seconds)
    if rank == 0:
        host = socket.gethostname()
        print(f"[dreamzero] rank {rank}/{world_size} host={host} device={device}", flush=True)
    else:
        print(f"[dreamzero] rank {rank}/{world_size} device={device}", flush=True)

    policy = DreamZeroAgilexPolicy(cfg, device=device, device_mesh=device_mesh, signal_group=signal_group)

    try:
        if rank != 0:
            worker_loop(policy, signal_group)
            return

        if bool(cfg.get("warmup", False)):
            policy.warmup(cfg)

        if args.inference_test:
            run_inference_test(policy, cfg)
            if world_size > 1:
                broadcast_signal(1, signal_group)
            return

        sock, server_kind_name = bind_server(str(cfg["server_bind"]), str(cfg["socket_type"]))
        print(
            f"[dreamzero] CobotMagic server bind {cfg['server_bind']} "
            f"using ZMQ {server_kind_name} ({cfg['socket_type']} protocol)",
            flush=True,
        )

        if args.startup_test:
            sock.close(0)
            print("[dreamzero] startup test passed", flush=True)
            if world_size > 1:
                broadcast_signal(1, signal_group)
            return

        while True:
            try:
                print("[dreamzero] waiting for request", flush=True)
                header, imgs = recv_packet(sock)
                started = time.perf_counter()
                control_hz = float(header.get("control_hz", 20.0))
                print(
                    f"[dreamzero] request received task={header.get('task_prompt', cfg['task_prompt'])!r} "
                    f"control_hz={control_hz}",
                    flush=True,
                )
                actions14 = policy.predict(header, imgs)
                send_actions(sock, actions14, control_hz=control_hz, action_mode="joint_absolute")
                print(
                    f"[dreamzero] response sent action_shape={actions14.shape} "
                    f"elapsed={time.perf_counter() - started:.3f}s",
                    flush=True,
                )
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[dreamzero] request error: {exc}", flush=True)
                send_empty(sock, str(exc), action_mode="joint_absolute")
    finally:
        if rank == 0 and world_size > 1:
            try:
                broadcast_signal(1, signal_group)
            except Exception:
                pass


if __name__ == "__main__":
    main()
