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

# 「深い階層で import される前に」安定状態で先に読み込む


import argparse
import cv2
import numpy as np
import torch
import yaml

from cobotmagic_deployment.policies.policy_openpi import Pi0Policy


def decode_jpeg(buf: bytes):
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG image.")
    return img


def preprocess_rgb(img_bgr, out_hw=(256, 256)):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a backend-specific YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openpi_cfg = cfg.get("openpi", {})
    policy_config_name = openpi_cfg.get("policy_config_name", "pi0_mobile_aloha_lora_local")
    checkpoint_dir = openpi_cfg.get(
        "checkpoint_dir",
        "/workspace/project/openpi/checkpoints/pi0_mobile_aloha_lora_local/mobile_aloha_lora/10000",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[openpi] running on {device}")
    policy = Pi0Policy(config_name=policy_config_name, checkpoint_dir=checkpoint_dir)
    print("[openpi] model loaded")
    torch.set_grad_enabled(False)

    rgb_h, rgb_w = cfg["openpi"].get("rgb_hw", [256, 256])

    task_prompt = "dummy-task"
    imgs = {
        "front": np.random.randint(0, 256, size=(rgb_h, rgb_w, 3), dtype=np.uint8),
        "left": np.random.randint(0, 256, size=(rgb_h, rgb_w, 3), dtype=np.uint8),
        "right": np.random.randint(0, 256, size=(rgb_h, rgb_w, 3), dtype=np.uint8),
    }
    jleft = np.array([0.0, -0.5, 0.5, 0.0, 0.2, -0.2, 0.8], dtype=np.float32)
    jright = np.array([0.0, 0.5, -0.5, 0.0, -0.2, 0.2, 0.8], dtype=np.float32)

    f_t = torch.from_numpy(preprocess_rgb(imgs["front"], (rgb_h, rgb_w))).float()
    l_t = torch.from_numpy(preprocess_rgb(imgs["left"], (rgb_h, rgb_w))).float()
    r_t = torch.from_numpy(preprocess_rgb(imgs["right"], (rgb_h, rgb_w))).float()
    qpos = torch.from_numpy(np.concatenate([jleft, jright], axis=0)).float()

    action_chunk = policy(f_t, l_t, r_t, qpos, task_prompt)
    print(action_chunk.shape)


if __name__ == "__main__":
    main()
