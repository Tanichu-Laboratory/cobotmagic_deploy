#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Right-arm-only CobotMagic ROS bridge for OpenVLA-OFT.

Collects front/right-wrist images plus right-arm joint state, requests a 7D
right-arm action chunk from the policy server, then publishes JointState
commands to /master/joint_right.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
import zmq
from cobotmagic_deployment.common.policy_server_protocol import client_socket_kind, encode_jpeg

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool

bridge = CvBridge()
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

buf: dict[str, Any] = {
    "front": None,
    "right": None,
    "jr": None,
}
buf_seq = {key: 0 for key in buf}
buf_time: dict[str, Optional[float]] = {key: None for key in buf}
enable_state = True


def img_cb(which: str, mode: str = "raw", quality: int = 80):
    use_compressed = mode == "compressed"

    def callback(msg):
        try:
            if use_compressed:
                payload = bytes(msg.data)
                if not payload:
                    return
            else:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                payload = encode_jpeg(img, quality=quality)
                if payload is None:
                    return
            buf[which] = payload
            buf_seq[which] += 1
            buf_time[which] = time.monotonic()
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"image cb error {which}: {exc}")

    return callback


def jr_cb(msg: JointState) -> None:
    if len(msg.position) < 7:
        rospy.logwarn_throttle(1.0, f"right joint state has <7 positions: {len(msg.position)}")
        return
    buf["jr"] = list(msg.position[:7])
    buf_seq["jr"] += 1
    buf_time["jr"] = time.monotonic()


def enable_cb(msg: Bool) -> None:
    global enable_state
    enable_state = bool(msg.data)


def latest_right_joint_and_age() -> tuple[Optional[np.ndarray], Optional[float]]:
    now = time.monotonic()
    jr = None if buf["jr"] is None else np.asarray(buf["jr"], dtype=np.float32)
    age = None if buf_time["jr"] is None else now - float(buf_time["jr"])
    return jr, age


def snapshot(task_prompt: str) -> Optional[dict[str, Any]]:
    if buf["front"] is None or buf["right"] is None or buf["jr"] is None:
        return None
    return {
        "task_prompt": task_prompt,
        "front": bytes(buf["front"]),
        "right": bytes(buf["right"]),
        "jright_measured": list(buf["jr"]),
        "obs_seq": dict(buf_seq),
        "obs_time": dict(buf_time),
    }


def wait_for_right_joint(timeout_sec: float = 10.0) -> Optional[np.ndarray]:
    deadline = time.monotonic() + max(timeout_sec, 0.1)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        jr, _ = latest_right_joint_and_age()
        if jr is not None:
            return jr
        if time.monotonic() > deadline:
            break
        rate.sleep()
    return None


def publish_right(pub: rospy.Publisher, position: np.ndarray) -> None:
    msg = JointState()
    msg.header.stamp = rospy.Time.now()
    msg.name = JOINT_NAMES
    msg.position = [float(x) for x in position]
    pub.publish(msg)


def move_home(pub: rospy.Publisher, cfg: dict[str, Any]) -> None:
    home_cfg = cfg["ros"].get("home_position", {})
    if not bool(home_cfg.get("enabled", True)):
        return
    target = np.asarray(home_cfg.get("right", []), dtype=np.float32)
    if target.shape[0] != 7:
        rospy.logwarn("home_position.right is not 7D; skipping home motion")
        return
    current = wait_for_right_joint(timeout_sec=float(home_cfg.get("wait_timeout_sec", 10.0)))
    if current is None:
        rospy.logwarn("right joint feedback unavailable; skipping home motion")
        return

    num_steps = max(int(home_cfg.get("num_steps", 200)), 1)
    rate_hz = float(home_cfg.get("rate_hz", 200.0))
    rate = rospy.Rate(rate_hz)
    rospy.loginfo(f"Moving right arm to single-arm home over {num_steps} steps at {rate_hz:.1f}Hz")
    for pos in np.linspace(current, target, num_steps, dtype=np.float32):
        if rospy.is_shutdown():
            return
        publish_right(pub, pos)
        rate.sleep()


def make_socket(cfg: dict[str, Any]) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(client_socket_kind(cfg["zmq"].get("socket_type", "req")))
    timeout_ms = int(float(cfg["ros"].get("policy_response_timeout_sec", 5.0)) * 1000.0)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.connect(cfg["zmq"].get("client_connect", "tcp://127.0.0.1:5556"))
    return sock


def apply_gripper_threshold(target: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    threshold_cfg = cfg["ros"].get("gripper_threshold", {})
    if not bool(threshold_cfg.get("enabled", False)):
        return target
    out = target.copy()
    close_threshold = float(threshold_cfg.get("close_threshold", -0.020))
    open_threshold = float(threshold_cfg.get("open_threshold", 0.0425))
    close_value = float(threshold_cfg.get("close_value", -0.0033))
    open_value = float(threshold_cfg.get("open_value", 0.059))
    if out[-1] >= open_threshold:
        out[-1] = open_value
    elif out[-1] <= close_threshold:
        out[-1] = close_value
    return out


def make_delta_scale(cfg: dict[str, Any], action_dim: int = 7) -> np.ndarray:
    openvla_cfg = cfg.get("openvla", {})
    scale = np.full(action_dim, float(openvla_cfg.get("action_delta_scale", 1.0)), dtype=np.float32)
    per_dim = openvla_cfg.get("action_delta_scale_per_dim")
    if per_dim is None:
        return scale
    if isinstance(per_dim, dict):
        for key, value in per_dim.items():
            idx = int(key)
            if 0 <= idx < action_dim:
                scale[idx] = float(value)
        return scale
    values = np.asarray(per_dim, dtype=np.float32)
    if values.shape[0] == action_dim:
        return values
    rospy.logwarn_throttle(2.0, f"ignoring action_delta_scale_per_dim with length {values.shape[0]}")
    return scale


def reconstruct_target_at_publish(
    *,
    current: np.ndarray,
    server_action: np.ndarray,
    raw_action: Optional[np.ndarray],
    reply_header: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    recon_cfg = cfg["ros"].get("joint_delta_reconstruction", {})
    if not bool(recon_cfg.get("enabled", True)):
        return server_action.copy()
    if raw_action is None:
        return server_action.copy()

    representation = reply_header.get("model_action_representation", {})
    representation_type = str(representation.get("type", "")).lower()
    if representation_type not in ("joint_delta", "joint_delta_gripper_abs"):
        return server_action.copy()

    scale = make_delta_scale(cfg, action_dim=server_action.shape[0])
    target = current + raw_action * scale
    if representation_type == "joint_delta_gripper_abs":
        for idx in representation.get("gripper_indices", []):
            idx = int(idx)
            if 0 <= idx < target.shape[0]:
                target[idx] = raw_action[idx]
    return target.astype(np.float32, copy=False)


def make_policy_state(measured: np.ndarray, last_command: Optional[np.ndarray], cfg: dict[str, Any]) -> np.ndarray:
    input_cfg = cfg["ros"].get("policy_gripper_input", {})
    mode = str(input_cfg.get("mode", "measured")).lower()
    state = measured.copy()
    if mode == "commanded" and last_command is not None:
        state[-1] = last_command[-1]
    elif mode == "hybrid" and last_command is not None:
        max_error = float(input_cfg.get("hybrid_max_error", 0.025))
        if abs(float(measured[-1] - last_command[-1])) <= max_error:
            state[-1] = last_command[-1]
    elif mode != "measured":
        rospy.logwarn_throttle(2.0, f"unsupported policy_gripper_input.mode={mode!r}; using measured")
    return state


def parse_action_reply(frames: list[bytes]) -> tuple[dict[str, Any], np.ndarray, Optional[np.ndarray]]:
    if not frames:
        raise ValueError("empty policy reply")
    header = json.loads(frames[0].decode("utf-8"))
    chunk_size = int(header.get("chunk_size", 0))
    if chunk_size <= 0:
        raise RuntimeError(header.get("error", "policy returned empty action chunk"))
    stride = int(header.get("action_stride", 7))
    if len(frames) < 2:
        raise ValueError("policy reply missing action payload")
    actions = np.frombuffer(frames[1], dtype=np.float32).reshape(chunk_size, stride)
    if stride != 7:
        raise ValueError(f"expected 7D right-arm actions, got stride={stride}")
    raw_actions = None
    if bool(header.get("has_model_raw_action", False)):
        if len(frames) < 3:
            raise ValueError("policy reply header says raw action exists but payload is missing")
        raw_stride = int(header.get("model_raw_action_stride", stride))
        raw_actions = np.frombuffer(frames[2], dtype=np.float32).reshape(chunk_size, raw_stride)
    return header, actions, raw_actions


def vector_json(vec: Optional[np.ndarray]) -> str:
    if vec is None:
        return ""
    return json.dumps([float(x) for x in vec], separators=(",", ":"))


def make_action_logger(cfg: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    log_cfg = cfg["ros"].get("action_log", {})
    if not bool(log_cfg.get("enabled", True)):
        return None, None
    log_dir = Path(log_cfg.get("dir", "logs/action_commands")).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"action_commands_single_right_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    file_obj = path.open("w", newline="", encoding="utf-8")
    fields = [
        "wall_time",
        "request_id",
        "action_index",
        "chunk_size",
        "policy_latency_sec",
        "model_action_representation",
        "joint_delta_reconstructed_at_publish",
        "measured_right",
        "policy_state_right",
        "raw_action_right",
        "server_action_right",
        "target_pre_threshold_right",
        "target_right",
        "target_vs_measured_norm6",
        "right_gripper_measured",
        "right_gripper_policy_state",
        "right_gripper_raw",
        "right_gripper_target",
        "joint_age_sec",
        "front_image_age_sec",
        "right_image_age_sec",
        "obs_seq",
    ]
    writer = csv.DictWriter(file_obj, fieldnames=fields)
    writer.writeheader()
    rospy.loginfo(f"Single-right action log: {path}")
    return {"file": file_obj, "writer": writer, "flush_every_rows": int(log_cfg.get("flush_every_rows", 1)), "rows": 0}, str(path)


def write_action_log(logger: Optional[dict[str, Any]], row: dict[str, Any]) -> None:
    if logger is None:
        return
    logger["writer"].writerow(row)
    logger["rows"] += 1
    if logger["rows"] % max(int(logger["flush_every_rows"]), 1) == 0:
        logger["file"].flush()


def check_snapshot_freshness(pkt: dict[str, Any], cfg: dict[str, Any]) -> bool:
    now = time.monotonic()
    max_joint_age = float(cfg["ros"].get("max_joint_age_sec", 0.75))
    max_image_age = float(cfg["ros"].get("max_image_age_sec", 2.0))
    joint_age = now - float(pkt["obs_time"]["jr"])
    front_age = now - float(pkt["obs_time"]["front"])
    right_age = now - float(pkt["obs_time"]["right"])
    if joint_age > max_joint_age:
        rospy.logwarn_throttle(1.0, f"stale right joint feedback: {joint_age:.3f}s")
        return False
    if front_age > max_image_age or right_age > max_image_age:
        rospy.logwarn_throttle(1.0, f"stale images: front={front_age:.3f}s right={right_age:.3f}s")
        return False
    return True


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "ros" not in cfg or "zmq" not in cfg:
        raise KeyError(f"{path} must contain 'ros' and 'zmq' sections")
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "config_single_right_openvla.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config).expanduser().resolve())
    ros_cfg = cfg["ros"]
    topics = ros_cfg["topics"]
    task_prompt = str(cfg.get("task_prompt", "stack the three cubes"))
    rate_hz = float(ros_cfg.get("rate_hz", 5.0))
    open_loop_steps = max(int(ros_cfg.get("open_loop_steps", 8)), 1)
    jpeg_quality = int(ros_cfg.get("jpeg_quality", 80))
    image_type = str(ros_cfg.get("image_type", "raw")).lower()
    image_msg_type = CompressedImage if image_type == "compressed" else Image

    rospy.init_node("cobotmagic_single_right_openvla_bridge", anonymous=True)
    rospy.Subscriber(topics["img_front"], image_msg_type, img_cb("front", image_type, jpeg_quality), queue_size=10)
    rospy.Subscriber(topics["img_right"], image_msg_type, img_cb("right", image_type, jpeg_quality), queue_size=10)
    rospy.Subscriber(topics["puppet_arm_right"], JointState, jr_cb, queue_size=50, tcp_nodelay=True)
    if topics.get("enable_flag"):
        rospy.Subscriber(topics["enable_flag"], Bool, enable_cb, queue_size=5)
    pub_right = rospy.Publisher(topics["cmd_joint_right"], JointState, queue_size=10)
    if topics.get("enable_flag") and bool(ros_cfg.get("publish_enable_flag", False)):
        enable_pub = rospy.Publisher(topics["enable_flag"], Bool, queue_size=1, latch=True)
        time.sleep(0.2)
        enable_pub.publish(Bool(data=True))

    rospy.loginfo(
        "Single-right OpenVLA bridge starting: "
        f"front={topics['img_front']} right={topics['img_right']} "
        f"joint={topics['puppet_arm_right']} cmd={topics['cmd_joint_right']} "
        f"rate={rate_hz:.1f}Hz open_loop_steps={open_loop_steps}"
    )
    move_home(pub_right, cfg)

    sock = make_socket(cfg)
    action_logger, action_log_path = make_action_logger(cfg)
    last_command: Optional[np.ndarray] = None
    request_id = 0
    idle_rate = rospy.Rate(rate_hz)
    publish_period_sec = 1.0 / max(rate_hz, 1e-6)

    try:
        while not rospy.is_shutdown():
            if not enable_state:
                idle_rate.sleep()
                continue
            pkt = snapshot(task_prompt)
            if pkt is None:
                rospy.logwarn_throttle(1.0, "waiting for front/right images and right joint state")
                idle_rate.sleep()
                continue
            if not check_snapshot_freshness(pkt, cfg):
                idle_rate.sleep()
                continue

            measured = np.asarray(pkt["jright_measured"], dtype=np.float32)
            policy_state = make_policy_state(measured, last_command, cfg)
            header = {
                "task_prompt": task_prompt,
                "control_hz": rate_hz,
                "jright": [float(x) for x in policy_state],
                "jright_measured": [float(x) for x in measured],
                "obs_seq": pkt["obs_seq"],
                "obs_time": pkt["obs_time"],
            }
            frames = [
                json.dumps(header, separators=(",", ":")).encode("utf-8"),
                pkt["front"],
                pkt["right"],
            ]
            started = time.monotonic()
            try:
                sock.send_multipart(frames)
                rep_frames = sock.recv_multipart()
            except zmq.error.Again:
                rospy.logwarn("policy request timed out")
                idle_rate.sleep()
                continue
            latency = time.monotonic() - started

            try:
                rep_header, actions, raw_actions = parse_action_reply(rep_frames)
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn(f"failed to parse policy reply: {exc}")
                idle_rate.sleep()
                continue

            request_id += 1
            steps = min(open_loop_steps, int(actions.shape[0]))
            rospy.loginfo(
                f"policy reply request={request_id} chunk={actions.shape} latency={latency:.3f}s "
                f"first={np.array2string(actions[0], precision=4)}"
            )

            next_publish_time = time.monotonic()
            for action_index in range(steps):
                if rospy.is_shutdown() or not enable_state:
                    break
                current, joint_age = latest_right_joint_and_age()
                if current is None:
                    current = measured.copy()
                raw = None if raw_actions is None else raw_actions[action_index].astype(np.float32, copy=True)
                server_action = actions[action_index].astype(np.float32, copy=True)
                target_pre_threshold = reconstruct_target_at_publish(
                    current=current,
                    server_action=server_action,
                    raw_action=raw,
                    reply_header=rep_header,
                    cfg=cfg,
                )
                reconstructed_at_publish = not np.allclose(target_pre_threshold, server_action, atol=1e-7, rtol=0.0)
                target = apply_gripper_threshold(target_pre_threshold, cfg)
                publish_right(pub_right, target)
                last_command = target.copy()

                now = time.monotonic()
                front_age = now - float(pkt["obs_time"]["front"])
                right_age = now - float(pkt["obs_time"]["right"])
                write_action_log(
                    action_logger,
                    {
                        "wall_time": f"{time.time():.6f}",
                        "request_id": request_id,
                        "action_index": action_index,
                        "chunk_size": int(rep_header.get("chunk_size", actions.shape[0])),
                        "policy_latency_sec": f"{latency:.6f}",
                        "model_action_representation": json.dumps(
                            rep_header.get("model_action_representation", {}),
                            separators=(",", ":"),
                        ),
                        "joint_delta_reconstructed_at_publish": int(reconstructed_at_publish),
                        "measured_right": vector_json(current),
                        "policy_state_right": vector_json(policy_state),
                        "raw_action_right": vector_json(raw),
                        "server_action_right": vector_json(server_action),
                        "target_pre_threshold_right": vector_json(target_pre_threshold),
                        "target_right": vector_json(target),
                        "target_vs_measured_norm6": f"{float(np.linalg.norm(target[:6] - current[:6])):.6f}",
                        "right_gripper_measured": f"{float(current[-1]):.6f}",
                        "right_gripper_policy_state": f"{float(policy_state[-1]):.6f}",
                        "right_gripper_raw": "" if raw is None else f"{float(raw[-1]):.6f}",
                        "right_gripper_target": f"{float(target[-1]):.6f}",
                        "joint_age_sec": "" if joint_age is None else f"{joint_age:.6f}",
                        "front_image_age_sec": f"{front_age:.6f}",
                        "right_image_age_sec": f"{right_age:.6f}",
                        "obs_seq": json.dumps(pkt["obs_seq"], separators=(",", ":")),
                    },
                )
                next_publish_time += publish_period_sec
                sleep_sec = next_publish_time - time.monotonic()
                if sleep_sec > 0.0:
                    time.sleep(sleep_sec)
                else:
                    # Drop accumulated delay instead of issuing catch-up bursts.
                    next_publish_time = time.monotonic()
    finally:
        sock.close(0)
        if action_logger is not None:
            action_logger["file"].flush()
            action_logger["file"].close()
            rospy.loginfo(f"Action command log saved: {action_log_path}")


if __name__ == "__main__":
    main()
