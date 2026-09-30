#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI side action server (Python 3.11+).
- Receives observations from ROS bridge via ZeroMQ (REP)
- Uses lightweight multipart messages (header + binary payloads)
- Runs the Pi0 policy and returns actions in binary form
- Settings loaded from a backend-specific YAML config
"""

# === must be at the very top ===
import os

# OpenMP/BLAS/Arrow の初期化衝突を抑制
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ARROW_NUM_THREADS", "1")


import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from cobotmagic_deployment.common.policy_server_protocol import bind_server
from cobotmagic_deployment.common.policy_server_runtime import serve
from cobotmagic_deployment.policies.policy_openpi import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_POLICY_CONFIG_NAME,
    Pi0Policy,
    preprocess_image,
)


def clip_base(vw, v_max=0.2, w_max=0.6):
    clipped = np.empty((vw.shape[0], 2), dtype=np.float32)
    clipped[:, 0] = np.clip(vw[:, 0], -v_max, v_max)
    clipped[:, 1] = np.clip(vw[:, 1], -w_max, w_max)
    return clipped


class OpenPIBackend:
    """Adapt ``Pi0Policy`` to the policy-server runtime (``predict(header, images)``)."""

    action_mode = None

    def __init__(self, policy, cfg):
        self.policy = policy
        self.rgb_hw = tuple(cfg.get("openpi", {}).get("rgb_hw", [256, 256]))
        self.use_base = bool(cfg["ros"].get("use_robot_base", False))
        clip_cfg = cfg["ros"].get("clip", {"v_max": 0.2, "w_max": 0.6})
        self.v_max = float(clip_cfg.get("v_max", 0.2))
        self.w_max = float(clip_cfg.get("w_max", 0.6))

    def predict(self, header, imgs):
        f_t, l_t, r_t = (
            torch.from_numpy(preprocess_image(imgs[key], self.rgb_hw)).float()
            for key in ("front", "left", "right")
        )
        jleft = np.asarray(header["jleft"], dtype=np.float32)
        jright = np.asarray(header["jright"], dtype=np.float32)
        qpos = torch.from_numpy(np.concatenate([jleft, jright], axis=0)).float()

        action_chunk = self.policy(f_t, l_t, r_t, qpos, header.get("task_prompt", "demo-task"))
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.detach().cpu().numpy()
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2 or action_chunk.shape[0] == 0:
            raise ValueError(f"unexpected action shape {action_chunk.shape}")
        vel = None
        if self.use_base and action_chunk.shape[1] >= 16:
            vel = clip_base(action_chunk[:, 14:16], v_max=self.v_max, w_max=self.w_max)
        return action_chunk[:, :14], vel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=Path(__file__).resolve().parents[1] / "configs" / "config_openpi.yaml",
        help="path to a backend-specific YAML config",
    )
    ap.add_argument("--bind", default="", help="Override zmq.server_bind from config YAML")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openpi_cfg = cfg.get("openpi", {})
    policy_config_name = openpi_cfg.get("policy_config_name", DEFAULT_POLICY_CONFIG_NAME)
    checkpoint_dir = openpi_cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)

    bind_addr = args.bind or cfg["zmq"].get("server_bind", "tcp://127.0.0.1:5557")
    sock, _ = bind_server(bind_addr, cfg["zmq"].get("socket_type", "req"))
    print(f"[openpi] server bind {bind_addr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[openpi] running on {device}")
    policy = Pi0Policy(config_name=policy_config_name, checkpoint_dir=checkpoint_dir)
    print("[openpi] model loaded")
    torch.set_grad_enabled(False)

    backend = OpenPIBackend(policy, cfg)
    # The bridge interprets OpenPI actions using ros.action_mode, so replies
    # do not declare an action_mode.
    serve(backend, sock, name="openpi", action_mode=None, verbose=False)


if __name__ == "__main__":
    main()
