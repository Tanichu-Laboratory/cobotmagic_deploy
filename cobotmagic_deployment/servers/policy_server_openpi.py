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
import json
import time
import cv2
import numpy as np
import torch
import yaml
import zmq

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


def clip_base(vw, v_max=0.2, w_max=0.6):
    clipped = np.empty((vw.shape[0], 2), dtype=np.float32)
    clipped[:, 0] = np.clip(vw[:, 0], -v_max, v_max)
    clipped[:, 1] = np.clip(vw[:, 1], -w_max, w_max)
    return clipped


def recv_packet(sock):
    frames = sock.recv_multipart()
    if len(frames) < 4:
        raise ValueError(f"expected 4 frames (header + 3 images), got {len(frames)}")

    header = json.loads(frames[0].decode("utf-8"))
    task_prompt = header.get("task_prompt", "demo-task")

    imgs = {
        "front": decode_jpeg(frames[1]),
        "left": decode_jpeg(frames[2]),
        "right": decode_jpeg(frames[3]),
    }

    jleft = np.asarray(header["jleft"], dtype=np.float32)
    jright = np.asarray(header["jright"], dtype=np.float32)
    odom = header.get("odom")
    if odom is not None:
        odom = np.asarray(odom, dtype=np.float32)

    return task_prompt, imgs, jleft, jright, odom


def send_actions(sock, left, right, vel=None):
    header = {
        "chunk_size": left.shape[0],
        "dtype": "float32",
        "left_stride": left.shape[1],
        "right_stride": right.shape[1],
        "has_vel": vel is not None,
        "timestamp": time.time(),
    }

    frames = [
        json.dumps(header, separators=(",", ":")).encode("utf-8"),
        left.astype(np.float32, copy=False).tobytes(order="C"),
        right.astype(np.float32, copy=False).tobytes(order="C"),
    ]

    if vel is not None:
        frames.append(vel.astype(np.float32, copy=False).tobytes(order="C"))

    sock.send_multipart(frames)


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

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    bind_addr = cfg["zmq"].get("server_bind", "tcp://127.0.0.1:5557")
    sock.bind(bind_addr)
    print(f"[openpi] server bind {bind_addr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[openpi] running on {device}")
    policy = Pi0Policy(config_name=policy_config_name, checkpoint_dir=checkpoint_dir)
    print("[openpi] model loaded")
    torch.set_grad_enabled(False)

    rgb_h, rgb_w = cfg["openpi"].get("rgb_hw", [256, 256])
    use_base = bool(cfg["ros"].get("use_robot_base", False))
    clip_cfg = cfg["ros"].get("clip", {"v_max": 0.2, "w_max": 0.6})
    v_max = float(clip_cfg.get("v_max", 0.2))
    w_max = float(clip_cfg.get("w_max", 0.6))

    while True:
        try:
            task_prompt, imgs, jleft, jright, odom = recv_packet(sock)
        except Exception as exc:  # noqa: BLE001
            print(f"[openpi] recv error: {exc}")
            sock.send_multipart([b'{"chunk_size":0}'])
            continue

        f_t = torch.from_numpy(preprocess_rgb(imgs["front"], (rgb_h, rgb_w))).float()
        l_t = torch.from_numpy(preprocess_rgb(imgs["left"], (rgb_h, rgb_w))).float()
        r_t = torch.from_numpy(preprocess_rgb(imgs["right"], (rgb_h, rgb_w))).float()
        qpos = torch.from_numpy(np.concatenate([jleft, jright], axis=0)).float()

        action_chunk = policy(f_t, l_t, r_t, qpos, task_prompt)
        if isinstance(action_chunk, torch.Tensor):
            action_chunk_np = action_chunk.detach().cpu().numpy()
        else:
            action_chunk_np = np.asarray(action_chunk, dtype=np.float32)

        if action_chunk_np.ndim != 2 or action_chunk_np.shape[0] == 0:
            print("[openpi] unexpected action shape, reply empty")
            sock.send_multipart([b'{"chunk_size":0}'])
            continue

        left = action_chunk_np[:, :7]
        right = action_chunk_np[:, 7:14]
        vel = None
        if use_base and action_chunk_np.shape[1] >= 16:
            vel = clip_base(action_chunk_np[:, 14:16], v_max=v_max, w_max=w_max)

        send_actions(sock, left, right, vel)


if __name__ == "__main__":
    main()
