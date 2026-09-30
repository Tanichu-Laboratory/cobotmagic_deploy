#!/usr/bin/env python3
"""Serve DreamZero AgileX checkpoints through the CobotMagic ZMQ protocol."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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
import zmq
from tianshou.data import Batch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from cobotmagic_deployment.common.policy_server_protocol import bind_server, recv_packet, send_actions, send_empty
from cobotmagic_deployment.common.dreamzero_noise_adaptation import (
    NoiseAdaptationSettings,
    parse_noise_adaptation_settings,
)


DEFAULT_DREAMZERO_REPO = "/workspace/project/dreamzero"
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
    # Accept a root-level switch as well, while keeping the documented location
    # under ``dreamzero`` authoritative.
    if "noise_adaptation" not in cfg and "noise_adaptation" in raw:
        cfg["noise_adaptation"] = raw["noise_adaptation"]
    if "noise_adaptation_config" not in cfg and "noise_adaptation_config" in raw:
        cfg["noise_adaptation_config"] = raw["noise_adaptation_config"]
    cfg["_config_dir"] = str(path.parent)
    cfg["backend"] = backend
    cfg["task_prompt"] = raw.get("task_prompt", cfg.get("task_prompt", "demo-task"))
    cfg["server_bind"] = raw.get("zmq", {}).get("server_bind", "tcp://127.0.0.1:5555")
    cfg["socket_type"] = raw.get("zmq", {}).get("socket_type", "req")
    cfg["prompt_control_bind"] = raw.get("zmq", {}).get(
        "prompt_control_bind", "tcp://127.0.0.1:5559"
    )
    cfg["prompt_control_connect"] = raw.get("zmq", {}).get(
        "prompt_control_connect", cfg["prompt_control_bind"]
    )
    return cfg


def run_prompt_control_command(args: argparse.Namespace, cfg: dict[str, Any]) -> bool:
    """Send a prompt-control command without initializing the policy model."""
    if args.set_task_prompt is not None:
        prompt = args.set_task_prompt.strip()
        if not prompt:
            raise ValueError("--set-task-prompt must not be empty")
        request = {"command": "set", "task_prompt": prompt}
    elif args.clear_task_prompt:
        request = {"command": "clear"}
    elif args.get_task_prompt:
        request = {"command": "get"}
    else:
        return False

    endpoint = args.prompt_control_connect or str(cfg["prompt_control_connect"])
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, args.prompt_control_timeout_ms)
    sock.setsockopt(zmq.RCVTIMEO, args.prompt_control_timeout_ms)
    try:
        sock.connect(endpoint)
        sock.send_json(request)
        reply = sock.recv_json()
    except zmq.Again as exc:
        raise RuntimeError(
            f"prompt control server did not respond at {endpoint} within "
            f"{args.prompt_control_timeout_ms} ms"
        ) from exc
    finally:
        sock.close(0)

    if not reply.get("ok", False):
        raise RuntimeError(str(reply.get("error", "prompt control command failed")))
    print(json.dumps(reply, ensure_ascii=False), flush=True)
    return True


def handle_prompt_control_request(
    request: dict[str, Any],
    current_override: str | None,
    default_prompt: str,
) -> tuple[str | None, dict[str, Any]]:
    """Apply one prompt-control request and return the new state and reply."""
    command = str(request.get("command", "")).lower()
    if command == "set":
        prompt = str(request.get("task_prompt", "")).strip()
        if not prompt:
            raise ValueError("task_prompt must not be empty")
        current_override = prompt
    elif command == "clear":
        current_override = None
    elif command != "get":
        raise ValueError(f"unsupported prompt control command: {command!r}")

    return current_override, {
        "ok": True,
        "override_active": current_override is not None,
        "task_prompt": current_override if current_override is not None else default_prompt,
        "source": "server_override" if current_override is not None else "request_or_config",
    }


def setup_dreamzero_flash(enabled: bool, num_dit_steps: int) -> None:
    os.environ.setdefault("DREAMZERO_FAST_LOAD", "1")
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    os.environ.setdefault("DREAMZERO_SKIP_UNUSED_CROSSATTN_CACHE", "1")
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


def make_observation(
    header: dict[str, Any],
    imgs: dict[str, np.ndarray],
    default_prompt: str,
    camera_keys: dict[str, str],
) -> dict[str, Any]:
    left = np.asarray(header.get("jleft", [0.0] * 7), dtype=np.float64)
    right = np.asarray(header.get("jright", [0.0] * 7), dtype=np.float64)
    if left.shape[-1] < 7 or right.shape[-1] < 7:
        raise ValueError(f"jleft/jright must be at least 7D, got {left.shape} and {right.shape}")

    return {
        camera_keys["front"]: _stack_single_frame(imgs["front"]),
        camera_keys["left"]: _stack_single_frame(imgs["left"]),
        camera_keys["right"]: _stack_single_frame(imgs["right"]),
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


def import_dreamzero(repo_path: str | Path) -> tuple[Any, Any]:
    """Import DreamZero from its external repository (``dreamzero.repo_path``)."""
    repo = str(Path(repo_path).expanduser())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy

    return EmbodimentTag, GrootSimPolicy


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
        self.camera_keys = dict(cfg.get("camera_keys", {
            "front": "video.front",
            "left": "video.left_wrist",
            "right": "video.right_wrist",
        }))
        if set(self.camera_keys) != {"front", "left", "right"}:
            raise ValueError(f"camera_keys must define front/left/right, got {self.camera_keys}")
        max_chunk_size = int(cfg.get("max_chunk_size", 1))
        num_inference_steps = int(cfg.get("num_inference_steps", 1))
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        self.noise_adaptation: NoiseAdaptationSettings = (
            parse_noise_adaptation_settings(
                cfg,
                config_dir=Path(str(cfg.get("_config_dir", "."))),
            )
        )
        model_overrides = [
            f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={max_chunk_size}",
            f"action_head_cfg.config.num_inference_timesteps={num_inference_steps}",
        ]
        model_overrides.extend(
            self.noise_adaptation.model_overrides(
                # Every inference-parallel rank maintains the same adapter state,
                # but only rank 0 writes the shared JSONL diagnostics.
                include_log_path=dist.get_rank() == 0,
            )
        )
        EmbodimentTag, GrootSimPolicy = import_dreamzero(cfg.get("repo_path", DEFAULT_DREAMZERO_REPO))
        self.policy = GrootSimPolicy(
            embodiment_tag=EmbodimentTag(str(cfg.get("embodiment", "xdof"))),
            model_path=str(model_path),
            device=device,
            device_mesh=device_mesh,
            model_config_overrides=model_overrides,
            skip_assert_delta_indices=True,
        )
        action_head = self._action_head()
        adapter = getattr(action_head, "pb_adapter", None)
        if self.noise_adaptation.enabled and adapter is None:
            raise RuntimeError(
                "noise_adaptation=true was requested, but DreamZero did not "
                "construct its initial-noise adapter"
            )
        if not self.noise_adaptation.enabled and adapter is not None:
            raise RuntimeError(
                "noise_adaptation=false was requested, but DreamZero unexpectedly "
                "constructed an initial-noise adapter"
            )
        self._set_eval_transform_max_chunk_size(max_chunk_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[dreamzero] model loaded path={model_path} device={device} "
            f"flash_cache={os.environ.get('ENABLE_DIT_CACHE')} "
            f"num_dit_steps={os.environ.get('NUM_DIT_STEPS', 'default')} "
            f"solver_steps={action_head.num_inference_steps} "
            f"configured_inference_steps={num_inference_steps} "
            f"max_chunk_size={max_chunk_size} "
            f"noise_adaptation={self.noise_adaptation.enabled}",
            flush=True,
        )
        if self.noise_adaptation.enabled:
            print(
                "[dreamzero] noise adaptation configured "
                f"tau_v={self.noise_adaptation.tau_v:.6g} "
                f"M={self.noise_adaptation.window_size} "
                f"eta={self.noise_adaptation.eta:.6g} "
                f"c_clip={self.noise_adaptation.c_clip:.6g} "
                f"deadband={self.noise_adaptation.deadband:.6g} "
                f"max_delta_norm={self.noise_adaptation.max_delta_norm} "
                f"log_path={self.noise_adaptation.log_path if dist.get_rank() == 0 else None}",
                flush=True,
            )

    def _action_head(self) -> Any:
        trained_model = getattr(self.policy, "trained_model", None)
        action_head = getattr(trained_model, "action_head", None)
        if action_head is None:
            raise RuntimeError("loaded DreamZero policy has no action_head")
        return action_head

    def reset_sequence_state(self, reason: str) -> None:
        """Clear causal inference and noise-adaptation state at an episode boundary."""
        action_head = self._action_head()
        action_head.current_start_frame = 0
        if hasattr(action_head, "language"):
            action_head.language = None
        if hasattr(action_head, "prompt_embs"):
            action_head.prompt_embs = None
        if hasattr(action_head, "reset_pb_adapter"):
            action_head.reset_pb_adapter()
        if dist.get_rank() == 0:
            print(
                f"[dreamzero] sequence state reset reason={reason} "
                f"noise_adaptation={self.noise_adaptation.enabled}",
                flush=True,
            )

    def noise_adaptation_status(self) -> dict[str, Any]:
        action_head = self._action_head()
        adapter = getattr(action_head, "pb_adapter", None)
        if adapter is None:
            return {"enabled": False}
        last_result = adapter.last_result
        return {
            "enabled": True,
            "initialized": adapter.initialized,
            "frozen": adapter.frozen,
            "buffer_size": adapter.buffer_size,
            "window_size": adapter.config.M,
            "window_index": adapter.window_index,
            "last_updated": None if last_result is None else last_result.updated,
            "last_reason": None if last_result is None else last_result.reason,
        }

    def _print_noise_adaptation_status(self) -> None:
        if not self.noise_adaptation.enabled or dist.get_rank() != 0:
            return
        status = self.noise_adaptation_status()
        print(
            "[dreamzero] noise adaptation status "
            f"initialized={status['initialized']} "
            f"buffer={status['buffer_size']}/{status['window_size']} "
            f"window={status['window_index']} "
            f"updated={status['last_updated']} "
            f"reason={status['last_reason']} "
            f"frozen={status['frozen']}",
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
        reset_requested = bool(
            header.get("episode_start", False)
            or header.get("reset_noise_adaptation", False)
        )
        if reset_requested:
            self.reset_sequence_state("request")
            if dist.get_world_size() > 1:
                broadcast_signal(2, self.signal_group)
        obs = make_observation(header, imgs, self.default_prompt, self.camera_keys)
        if dist.get_world_size() > 1:
            broadcast_signal(0, self.signal_group)
            broadcast_obs_from_rank0(obs)
            dist.barrier()
        batch, _video_pred = self.policy.lazy_joint_forward_causal(Batch(obs=obs))
        if dist.get_world_size() > 1:
            dist.barrier()
        self._print_noise_adaptation_status()
        return action_batch_to_numpy(batch.act)

    def warmup(self, cfg: dict[str, Any]) -> None:
        runs = max(int(cfg.get("warmup_runs", 3)), 1)
        height, width = [int(v) for v in cfg.get("warmup_image_hw", [480, 640])]
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        header = test_header(str(cfg.get("task_prompt", self.default_prompt)))
        elapsed_values = []
        actions = None
        for run_index in range(runs):
            started = time.perf_counter()
            actions = self.predict(header, {"front": dummy, "left": dummy, "right": dummy})
            elapsed = time.perf_counter() - started
            elapsed_values.append(elapsed)
            print(f"[dreamzero] warmup run={run_index + 1}/{runs} elapsed={elapsed:.3f}s", flush=True)

        assert actions is not None
        values = np.asarray(elapsed_values, dtype=np.float64)
        print(
            f"[dreamzero] warmup done action_shape={actions.shape} runs={runs} "
            f"total={values.sum():.3f}s final={values[-1]:.3f}s",
            flush=True,
        )
        # Warmup observations must never become the first PB error window or
        # remain in the causal KV cache used for the real episode.
        self.reset_sequence_state("warmup_complete")
        if dist.get_world_size() > 1:
            broadcast_signal(2, self.signal_group)


def worker_loop(policy: DreamZeroAgilexPolicy, signal_group: dist.ProcessGroup | None) -> None:
    print(f"[dreamzero] worker rank={dist.get_rank()} ready", flush=True)
    while True:
        signal = wait_signal(signal_group)
        if signal == 1:
            break
        if signal == 2:
            policy.reset_sequence_state("rank0_request")
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


def run_inference_test(policy: DreamZeroAgilexPolicy, cfg: dict[str, Any], runs: int = 1) -> None:
    runs = max(int(runs), 1)
    height, width = [int(v) for v in cfg.get("warmup_image_hw", [480, 640])]
    dummy = np.zeros((height, width, 3), dtype=np.uint8)
    header = test_header(str(cfg["task_prompt"]))
    elapsed_values = []
    actions14 = None
    for run_index in range(runs):
        started = time.perf_counter()
        actions14 = policy.predict(header, {"front": dummy, "left": dummy, "right": dummy})
        elapsed = time.perf_counter() - started
        elapsed_values.append(elapsed)
        print(f"[dreamzero] benchmark run={run_index + 1}/{runs} elapsed={elapsed:.3f}s", flush=True)

    assert actions14 is not None
    print_action_stats("dreamzero", actions14)
    values = np.asarray(elapsed_values, dtype=np.float64)
    print(
        f"[dreamzero] benchmark summary runs={runs} min={values.min():.3f}s "
        f"median={np.median(values):.3f}s mean={values.mean():.3f}s max={values.max():.3f}s",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "config_dreamzero_agilex.yaml")
    parser.add_argument("--bind", default="", help="Override zmq.server_bind from config YAML")
    parser.add_argument(
        "--prompt-control-bind",
        default="",
        help="Override zmq.prompt_control_bind for the running server",
    )
    parser.add_argument(
        "--prompt-control-connect",
        default="",
        help="Override zmq.prompt_control_connect for a prompt-control command",
    )
    parser.add_argument("--prompt-control-timeout-ms", type=int, default=5000)
    prompt_command = parser.add_mutually_exclusive_group()
    prompt_command.add_argument(
        "--set-task-prompt",
        metavar="PROMPT",
        help="Update the prompt of an already-running server and exit",
    )
    prompt_command.add_argument(
        "--clear-task-prompt",
        action="store_true",
        help="Clear the live override and resume prompts supplied by the bridge",
    )
    prompt_command.add_argument(
        "--get-task-prompt",
        action="store_true",
        help="Print the prompt state of an already-running server and exit",
    )
    parser.add_argument("--flash", action="store_true", default=True, help="Enable DreamZero-Flash style DiT cache/mask")
    parser.add_argument("--no-flash", dest="flash", action="store_false", help="Disable Flash-style cache/mask")
    parser.add_argument("--num-dit-steps", type=int, default=1, help="DiT cache compute budget; use 1 for single-step Flash inference")
    parser.add_argument("--startup-test", action="store_true", help="Bind the socket, print readiness, and exit")
    parser.add_argument("--inference-test", action="store_true", help="Run one dummy inference, print value ranges, and exit")
    parser.add_argument("--benchmark-runs", type=int, default=1, help="Number of inference-test calls after warmup")
    parser.add_argument("--warmup", action="store_true", help="Run one dummy warmup before serving or inference testing")
    parser.add_argument("--timeout-seconds", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    if run_prompt_control_command(args, cfg):
        return
    if args.bind:
        cfg["server_bind"] = args.bind
    if args.prompt_control_bind:
        cfg["prompt_control_bind"] = args.prompt_control_bind

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

        if args.warmup or bool(cfg.get("warmup", False)):
            policy.warmup(cfg)

        if args.inference_test:
            run_inference_test(policy, cfg, runs=args.benchmark_runs)
            if world_size > 1:
                broadcast_signal(1, signal_group)
            return

        sock, server_kind_name = bind_server(str(cfg["server_bind"]), str(cfg["socket_type"]))
        prompt_control_sock = zmq.Context.instance().socket(zmq.REP)
        prompt_control_sock.setsockopt(zmq.LINGER, 0)
        prompt_control_sock.bind(str(cfg["prompt_control_bind"]))
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        poller.register(prompt_control_sock, zmq.POLLIN)
        prompt_override: str | None = None
        print(
            f"[dreamzero] CobotMagic server bind {cfg['server_bind']} "
            f"using ZMQ {server_kind_name} ({cfg['socket_type']} protocol)",
            flush=True,
        )
        print(
            f"[dreamzero] prompt control bind {cfg['prompt_control_bind']} using ZMQ REP",
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
                events = dict(poller.poll())
                if prompt_control_sock in events:
                    try:
                        request = prompt_control_sock.recv_json()
                        prompt_override, reply = handle_prompt_control_request(
                            request,
                            prompt_override,
                            str(cfg["task_prompt"]),
                        )
                    except Exception as exc:  # noqa: BLE001
                        reply = {"ok": False, "error": str(exc)}
                    prompt_control_sock.send_json(reply)
                    if reply.get("ok", False):
                        print(
                            f"[dreamzero] prompt control source={reply['source']} "
                            f"task={reply['task_prompt']!r}",
                            flush=True,
                        )
                    continue

                header, imgs = recv_packet(sock)
                started = time.perf_counter()
                control_hz = float(header.get("control_hz", 20.0))
                request_prompt = str(header.get("task_prompt", cfg["task_prompt"]))
                effective_prompt = prompt_override if prompt_override is not None else request_prompt
                header = dict(header)
                header["task_prompt"] = effective_prompt
                print(
                    f"[dreamzero] request received task={effective_prompt!r} "
                    f"prompt_source={'server_override' if prompt_override is not None else 'request'} "
                    f"control_hz={control_hz}",
                    flush=True,
                )
                actions14 = policy.predict(header, imgs)
                send_actions(sock, actions14, control_hz=control_hz, action_mode="absolute")
                print(
                    f"[dreamzero] response sent action_shape={actions14.shape} "
                    f"elapsed={time.perf_counter() - started:.3f}s",
                    flush=True,
                )
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[dreamzero] request error: {exc}", flush=True)
                send_empty(sock, str(exc), action_mode="absolute")
    finally:
        if rank == 0 and "prompt_control_sock" in locals():
            prompt_control_sock.close(0)
        if rank == 0 and world_size > 1:
            try:
                broadcast_signal(1, signal_group)
            except Exception:
                pass
        # torchrun owns multi-rank process-group teardown. Explicitly destroying all
        # groups here can hang while the inference-parallel P2P group is shutting down.
        if dist.is_initialized() and world_size == 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
