#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one dummy OpenPI inference from a YAML config without ROS or ZeroMQ."""

# === must be at the very top ===
import os

# OpenMP/BLAS/Arrow の初期化衝突を抑制
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("ARROW_NUM_THREADS", "1")


import argparse

import numpy as np
import torch
import yaml

from cobotmagic_deployment.policies.policy_openpi import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_POLICY_CONFIG_NAME,
    Pi0Policy,
    preprocess_image,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a backend-specific YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    openpi_cfg = cfg.get("openpi", {})
    policy_config_name = openpi_cfg.get("policy_config_name", DEFAULT_POLICY_CONFIG_NAME)
    checkpoint_dir = openpi_cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)

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

    f_t = torch.from_numpy(preprocess_image(imgs["front"], (rgb_h, rgb_w))).float()
    l_t = torch.from_numpy(preprocess_image(imgs["left"], (rgb_h, rgb_w))).float()
    r_t = torch.from_numpy(preprocess_image(imgs["right"], (rgb_h, rgb_w))).float()
    qpos = torch.from_numpy(np.concatenate([jleft, jright], axis=0)).float()

    action_chunk = policy(f_t, l_t, r_t, qpos, task_prompt)
    print(action_chunk.shape)


if __name__ == "__main__":
    main()
