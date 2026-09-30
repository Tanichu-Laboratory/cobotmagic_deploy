#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Noetic side bridge (Python 3.8).
- Collects observations from ROS topics
- Sends compact multipart messages to the policy server via ZeroMQ (REQ)
- Receives actions (binary float32 arrays) and publishes to ROS
- Designed for minimal overhead between separate conda environments

The ROS-specific parts (topics, home motion, shutdown safety) live here; the
asynchronous inference, chunk processing, command shaping and IK are
ROS-independent modules under ``cobotmagic_deployment.common`` that
``DualArmPolicyBridge`` wires together.
"""

import argparse
import json
import threading
import time

import numpy as np
import yaml
from cobotmagic_deployment.bridges.action_logging import (
    make_action_logger, make_rollout_dataset_logger, norm_first_six,
    save_action_chunk, save_request_snapshot, save_rollout_action,
    save_rollout_observation, vector_to_json, write_action_log,
)
from cobotmagic_deployment.common.action_processing import (
    eef_pose_and_gripper_to_command, eef_pose_and_gripper_to_ee6d, step_towards,
)
from cobotmagic_deployment.common.bridge_log import BridgeLog
from cobotmagic_deployment.common.chunk_pipeline import (
    ChunkPipeline, ChunkRejected, InvalidPolicyResponse, parse_action_response,
)
from cobotmagic_deployment.common.chunk_scheduler import ChunkScheduler, PolicyRequestGate, stale_observations
from cobotmagic_deployment.common.command_publisher import InterpolatedCommandPublisher, command_publish_settings
from cobotmagic_deployment.common.command_shaper import CommandShaper, PolicyGripperInput, VelocityIntegrator
from cobotmagic_deployment.common.ik_commander import EefIkCommander
from cobotmagic_deployment.common.policy_client import AsyncPolicyClient
from cobotmagic_deployment.common.policy_server_protocol import encode_jpeg

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool

bridge = CvBridge()

# shared buffers
buf = {
    'front': None,
    'left': None,
    'right': None,
    'jl': None,
    'jr': None,
    'odom': None,
    'eef_l': None,
    'eef_r': None,
}
buf_seq = {key: 0 for key in buf}
buf_time = {key: None for key in buf}
lock = threading.Lock()
enable_state = True  # /enable_flag があればこれに従う
CAMERA_KEYS = ('front', 'left', 'right')


def store_observation(key, value):
    with lock:
        buf[key] = value
        buf_seq[key] += 1
        buf_time[key] = time.monotonic()


def eef_pose_msg_to_list(msg):
    return [
        float(msg.pose.position.x),
        float(msg.pose.position.y),
        float(msg.pose.position.z),
        float(msg.pose.orientation.x),
        float(msg.pose.orientation.y),
        float(msg.pose.orientation.z),
        float(msg.pose.orientation.w),
    ]


def img_cb(which, mode='raw', quality=80):
    use_compressed = (mode == 'compressed')

    def f(msg):
        try:
            if use_compressed:
                payload = bytes(msg.data)
            else:
                img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                payload = encode_jpeg(img, quality=quality)
            if not payload:
                return
            store_observation(which, payload)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn(f"image cb error {which}: {exc}")

    return f


def jl_cb(msg: JointState):
    store_observation('jl', list(msg.position))


def jr_cb(msg: JointState):
    store_observation('jr', list(msg.position))


def eef_left_cb(msg: PoseStamped):
    store_observation('eef_l', eef_pose_msg_to_list(msg))


def eef_right_cb(msg: PoseStamped):
    store_observation('eef_r', eef_pose_msg_to_list(msg))


def odom_cb(msg: Odometry):
    store_observation('odom', [msg.twist.twist.linear.x, msg.twist.twist.angular.z])


def enable_cb(msg: Bool):
    global enable_state
    enable_state = bool(msg.data)


def required_obs_keys(include_eef=False):
    keys = list(CAMERA_KEYS) + ['jl', 'jr']
    if include_eef:
        keys.extend(['eef_l', 'eef_r'])
    return keys


def missing_obs_keys(use_base=False, require_eef=False):
    with lock:
        missing = [key for key in required_obs_keys(require_eef) if buf[key] is None]
        if use_base and buf['odom'] is None:
            missing.append('odom')
    return missing


def have_obs(use_base=False, require_eef=False):
    return not missing_obs_keys(use_base, require_eef)


def snapshot(task_prompt: str, use_base=False, include_eef=False):
    """Capture a coherent observation while keeping callback lock time minimal."""
    with lock:
        if not all(buf[key] is not None for key in required_obs_keys(include_eef)):
            return None
        front = bytes(buf['front'])
        left = bytes(buf['left'])
        right = bytes(buf['right'])
        jleft = list(buf['jl'])
        jright = list(buf['jr'])
        eef_left = list(buf['eef_l']) if include_eef else None
        eef_right = list(buf['eef_r']) if include_eef else None
        odom = list(buf['odom']) if use_base and buf['odom'] is not None else None
        obs_seq = dict(buf_seq)
        obs_time = dict(buf_time)

    pkt = {
        'task_prompt': task_prompt,
        'front': front,
        'left': left,
        'right': right,
        'jleft': jleft,
        'jright': jright,
        'obs_seq': obs_seq,
        'obs_time': obs_time,
    }
    if include_eef:
        left_gripper = float(jleft[-1])
        right_gripper = float(jright[-1])
        current_eef_left = eef_pose_and_gripper_to_command(eef_left, left_gripper)
        current_eef_right = eef_pose_and_gripper_to_command(eef_right, right_gripper)
        pkt['current_eef_left'] = current_eef_left.tolist()
        pkt['current_eef_right'] = current_eef_right.tolist()
        pkt['xvla_proprio'] = np.concatenate([
            eef_pose_and_gripper_to_ee6d(eef_left, left_gripper),
            eef_pose_and_gripper_to_ee6d(eef_right, right_gripper),
        ]).astype(np.float32).tolist()
    else:
        pkt['current_eef_left'] = None
        pkt['current_eef_right'] = None
        pkt['xvla_proprio'] = None
    pkt['odom'] = odom
    return pkt


def latest_joint_arrays_and_age():
    now = time.monotonic()
    with lock:
        jl = None if buf['jl'] is None else np.array(buf['jl'], dtype=np.float32)
        jr = None if buf['jr'] is None else np.array(buf['jr'], dtype=np.float32)
        jl_age = None if buf_time['jl'] is None else now - buf_time['jl']
        jr_age = None if buf_time['jr'] is None else now - buf_time['jr']
    return jl, jr, jl_age, jr_age


def wait_for_joint_arrays(timeout_sec=10.0):
    deadline = time.monotonic() + max(timeout_sec, 0.1)
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        with lock:
            jl = buf['jl']
            jr = buf['jr']
        if jl is not None and jr is not None:
            return np.array(jl, dtype=np.float32), np.array(jr, dtype=np.float32)
        if time.monotonic() > deadline:
            break
        rate.sleep()
    return None, None


def publish_joint_pair(pub_l, pub_r, name_list, left_pos, right_pos):
    stamp = rospy.Time.now()
    msg_left = JointState()
    msg_left.header.stamp = stamp
    msg_left.name = name_list
    msg_left.position = left_pos.tolist()
    pub_l.publish(msg_left)

    msg_right = JointState()
    msg_right.header.stamp = stamp
    msg_right.name = name_list
    msg_right.position = right_pos.tolist()
    pub_r.publish(msg_right)


def wait_for_home_settle(
    pub_l,
    pub_r,
    name_list,
    target_left,
    target_right,
    rate_hz,
    tolerance,
    hold_sec,
    timeout_sec,
):
    rate = rospy.Rate(max(rate_hz, 1))
    deadline = time.monotonic() + max(timeout_sec, 0.0)
    hold_start = None
    joint_dim = min(6, len(target_left), len(target_right))

    while not rospy.is_shutdown():
        publish_joint_pair(pub_l, pub_r, name_list, target_left, target_right)
        jl, jr, jl_age, jr_age = latest_joint_arrays_and_age()
        if jl is not None and jr is not None:
            left_err = float(np.max(np.abs(jl[:joint_dim] - target_left[:joint_dim])))
            right_err = float(np.max(np.abs(jr[:joint_dim] - target_right[:joint_dim])))
            joints_fresh = (
                (jl_age is None or jl_age <= 0.5)
                and (jr_age is None or jr_age <= 0.5)
            )
            if joints_fresh and left_err <= tolerance and right_err <= tolerance:
                if hold_start is None:
                    hold_start = time.monotonic()
                if time.monotonic() - hold_start >= hold_sec:
                    rospy.loginfo(
                        "Home position settled from measured joint states: "
                        f"left_max_err={left_err:.4f} right_max_err={right_err:.4f} "
                        f"tolerance={tolerance:.4f} hold_sec={hold_sec:.2f}."
                    )
                    return True
            else:
                hold_start = None

        if timeout_sec > 0.0 and time.monotonic() >= deadline:
            rospy.logwarn(
                "Timed out waiting for measured joints to settle at home; "
                "continuing with the latest observed state. "
                f"timeout_sec={timeout_sec:.2f} tolerance={tolerance:.4f}."
            )
            return False
        rate.sleep()

    return False


def move_to_home(
    pub_l,
    pub_r,
    name_list,
    target_left,
    target_right,
    step_lengths,
    rate_hz,
    mode="step",
    num_steps=None,
    settle=True,
    settle_tolerance=0.04,
    settle_hold_sec=0.5,
    settle_timeout_sec=8.0,
    settle_rate_hz=None,
    gripper_move_start_fraction=None,
):
    if gripper_move_start_fraction is not None:
        gripper_move_start_fraction = float(gripper_move_start_fraction)
        if mode != "linear" or not 0.0 <= gripper_move_start_fraction < 1.0:
            raise ValueError("gripper_move_start_fraction requires linear mode and 0 <= value < 1")
    jl, jr = wait_for_joint_arrays()
    if jl is None or jr is None:
        rospy.logwarn("home move skipped: joint state not available")
        return False

    rate = rospy.Rate(max(rate_hz, 1))
    cur_left = jl.copy()
    cur_right = jr.copy()

    if mode == "linear":
        steps = max(int(num_steps or 100), 1)
        left_traj = np.linspace(cur_left, target_left, steps, dtype=np.float32)
        right_traj = np.linspace(cur_right, target_right, steps, dtype=np.float32)
        if gripper_move_start_fraction is None:
            # Preserve the original immediate gripper target for other configs.
            left_traj[:, -1] = target_left[-1]
            right_traj[:, -1] = target_right[-1]
        else:
            progress = np.linspace(0.0, 1.0, steps) if steps > 1 else np.ones(1)
            grip_progress = np.clip(
                (progress - gripper_move_start_fraction) / (1.0 - gripper_move_start_fraction),
                0.0, 1.0,
            )
            left_traj[:, -1] = cur_left[-1] + grip_progress * (target_left[-1] - cur_left[-1])
            right_traj[:, -1] = cur_right[-1] + grip_progress * (target_right[-1] - cur_right[-1])
        for left_pos, right_pos in zip(left_traj, right_traj):
            publish_joint_pair(pub_l, pub_r, name_list, left_pos, right_pos)
            rate.sleep()
    else:
        done = False
        while not rospy.is_shutdown() and not done:
            cur_left = step_towards(cur_left, target_left, step_lengths)
            cur_right = step_towards(cur_right, target_right, step_lengths)

            publish_joint_pair(pub_l, pub_r, name_list, cur_left, cur_right)

            if np.allclose(cur_left, target_left, atol=1e-4) and np.allclose(cur_right, target_right, atol=1e-4):
                done = True
            rate.sleep()
    if settle:
        return wait_for_home_settle(
            pub_l,
            pub_r,
            name_list,
            target_left,
            target_right,
            settle_rate_hz or rate_hz,
            settle_tolerance,
            settle_hold_sec,
            settle_timeout_sec,
        )
    return True


def home_motion_options(phase_cfg, home_cfg, default_rate_hz):
    """Resolve home-motion options: phase value, then home_position value, then default."""
    def option(key, default=None):
        return phase_cfg.get(key, home_cfg.get(key, default))

    mode = str(option('mode', 'step')).lower()
    if mode not in ('step', 'linear'):
        rospy.logwarn(f"Unsupported home_position mode={mode!r}; using 'step'.")
        mode = 'step'
    rate_hz = max(int(option('rate_hz', default_rate_hz)), 1)
    return {
        'rate_hz': rate_hz,
        'mode': mode,
        'num_steps': option('num_steps'),
        'settle': bool(option('settle', True)),
        'settle_tolerance': float(option('settle_tolerance', 0.04)),
        'settle_hold_sec': float(option('settle_hold_sec', 0.5)),
        'settle_timeout_sec': float(option('settle_timeout_sec', 8.0)),
        'settle_rate_hz': max(int(option('settle_rate_hz', min(rate_hz, 50))), 1),
        'gripper_move_start_fraction': option('gripper_move_start_fraction'),
    }


def execute_home_phase(
    pub_l,
    pub_r,
    name_list,
    step_lengths,
    home_cfg,
    phase_cfg,
    phase_index=None,
    default_rate_hz=20,
):
    """Move to one home target. ``phase_index=None`` is the single-target form."""
    target_left = np.array(phase_cfg['left'], dtype=np.float32)
    target_right = np.array(phase_cfg['right'], dtype=np.float32)
    if target_left.shape[0] != len(name_list) or target_right.shape[0] != len(name_list):
        if phase_index is None:
            raise ValueError("home_position left/right size does not match joint_names")
        rospy.logwarn(f"home_position phase {phase_index} size mismatch; skipping phase.")
        return False

    options = home_motion_options(phase_cfg, home_cfg, default_rate_hz)
    if phase_index is None:
        label = 'home'
    else:
        label = f"home phase {phase_index}:{phase_cfg.get('name', f'phase_{phase_index}')}"
    rospy.loginfo(
        f"Moving arms to {label} mode={options['mode']} rate={options['rate_hz']}Hz "
        f"num_steps={options['num_steps']} settle={options['settle']}."
    )
    return move_to_home(
        pub_l,
        pub_r,
        name_list,
        target_left,
        target_right,
        step_lengths,
        **options,
    )


def run_shutdown_safety(pub_l, pub_r, pub_v, enable_pub, name_list, cfg):
    safety_cfg = cfg['ros'].get('shutdown_safety', {})
    if not bool(safety_cfg.get('enabled', False)):
        return

    hold_sec = max(float(safety_cfg.get('hold_current_sec', 0.5)), 0.0)
    rate_hz = max(float(safety_cfg.get('hold_rate_hz', 50.0)), 1.0)
    publish_enable_false = bool(safety_cfg.get('publish_enable_false', False))
    zero_base_velocity = bool(safety_cfg.get('zero_base_velocity', True))

    left, right, left_age, right_age = latest_joint_arrays_and_age()
    if left is not None and right is not None:
        rospy.loginfo(
            "Shutdown safety: publishing measured current joint positions "
            f"for {hold_sec:.2f}s at {rate_hz:.1f}Hz "
            f"(joint_age_left={left_age} joint_age_right={right_age})."
        )
        deadline = time.monotonic() + hold_sec
        period = 1.0 / rate_hz
        while time.monotonic() < deadline:
            publish_joint_pair(pub_l, pub_r, name_list, left, right)
            if zero_base_velocity and pub_v is not None:
                pub_v.publish(Twist())
            time.sleep(period)
    else:
        rospy.logwarn("Shutdown safety: latest measured joints unavailable; cannot publish hold position.")

    if publish_enable_false and enable_pub is not None:
        rospy.loginfo("Shutdown safety: publishing enable flag False.")
        for _ in range(max(int(rate_hz * 0.2), 1)):
            enable_pub.publish(Bool(data=False))
            time.sleep(1.0 / rate_hz)


class DualArmPolicyBridge:
    """CobotMagic dual-arm bridge: ROS observations -> policy server -> ROS commands.

    The ROS-independent building blocks can be reused on their own:

    * :class:`~cobotmagic_deployment.common.policy_client.AsyncPolicyClient`: non-blocking requests
    * :class:`~cobotmagic_deployment.common.chunk_scheduler.ChunkScheduler` / ``PolicyRequestGate``:
      asynchronous chunk timeline, temporal ensemble, latency compensation, request timing
    * :class:`~cobotmagic_deployment.common.chunk_pipeline.ChunkPipeline`: filters and interpolation on receipt
    * :class:`~cobotmagic_deployment.common.command_shaper.CommandShaper`: per-step filters and gripper handling
    * :class:`~cobotmagic_deployment.common.ik_commander.EefIkCommander`: EEF targets -> joint commands
    * :class:`~cobotmagic_deployment.common.command_publisher.InterpolatedCommandPublisher`: high-rate publish
    """

    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or BridgeLog.for_rospy()
        ros_cfg = cfg['ros']
        self.image_type = ros_cfg.get('image_type', 'raw').lower()
        self.jpeg_quality = int(ros_cfg.get('jpeg_quality', 80))
        self.rate_hz = max(int(ros_cfg.get('rate_hz', 20)), 1)
        self.action_mode = ros_cfg.get('action_mode', 'velocity').lower()
        if self.action_mode not in ('velocity', 'absolute', 'eef_absolute'):
            raise ValueError(
                f"ros.action_mode must be 'velocity', 'absolute', or 'eef_absolute', got {self.action_mode!r}")
        self.eef_action_mode = self.action_mode == 'eef_absolute'
        (self.command_publish_mode, self.command_publish_rate_hz,
         self.command_publish_substeps, self.command_publish_interpolation) = command_publish_settings(
            ros_cfg, self.rate_hz, self.eef_action_mode, self.log)
        self.max_joint_age_sec = float(ros_cfg.get('max_joint_age_sec', 0.75))
        self.max_image_age_sec = float(ros_cfg.get('max_image_age_sec', 2.0))
        self.task_prompt = cfg.get('task_prompt', 'demo-task')
        self.topics = ros_cfg['topics']
        self.name_list = ros_cfg.get('joint_names', [f'joint{i}' for i in range(7)])
        self.use_base = bool(ros_cfg.get('use_robot_base', False))
        clip = ros_cfg.get('clip', {'v_max': 0.2, 'w_max': 0.6})
        self.v_max = float(clip.get('v_max', 0.2))
        self.w_max = float(clip.get('w_max', 0.6))

        self.scheduler = ChunkScheduler(ros_cfg, self.rate_hz, self.log)
        self.request_gate = PolicyRequestGate(ros_cfg, self.scheduler.latency_fixed_steps, self.log)
        self.pipeline = ChunkPipeline(ros_cfg, self.rate_hz, self.scheduler.overlap_steps, self.log)
        self.shaper = CommandShaper(ros_cfg, len(self.name_list), self.command_publish_mode, self.log)
        self.gripper_input = PolicyGripperInput(ros_cfg, self.log)
        self.integrator = VelocityIntegrator(1.0 / self.rate_hz / 4.0)

        self.command_left = None
        self.command_right = None
        self.published_first_command = False
        self.last_obs_wait_warn = 0.0
        self.last_stale_obs_warn = 0.0
        self.last_stale_joint_warn = 0.0
        self.ik = None
        self.publisher = None
        self.client = None
        self.action_logger = None
        self.action_log_path = None
        self.rollout_logger = None

    # --- ROS setup --------------------------------------------------------

    def setup_ros(self):
        topics = self.topics
        ros_cfg = self.cfg['ros']
        rb_topics = ros_cfg.get('robot_base_topics', {'odom': '/odom', 'cmd_vel': '/cmd_vel'})
        for key, topic_key in (('front', 'img_front'), ('left', 'img_left'), ('right', 'img_right')):
            if self.image_type == 'compressed':
                rospy.Subscriber(topics[topic_key] + '/compressed', CompressedImage,
                                 img_cb(key, 'compressed'), queue_size=10)
            else:
                rospy.Subscriber(topics[topic_key], Image, img_cb(key, 'raw', self.jpeg_quality), queue_size=10)
        rospy.Subscriber(topics['puppet_arm_left'], JointState, jl_cb, queue_size=50)
        rospy.Subscriber(topics['puppet_arm_right'], JointState, jr_cb, queue_size=50)
        if self.eef_action_mode:
            rospy.Subscriber(topics.get('puppet_arm_left_pose', '/puppet/end_pose_left'),
                             PoseStamped, eef_left_cb, queue_size=50)
            rospy.Subscriber(topics.get('puppet_arm_right_pose', '/puppet/end_pose_right'),
                             PoseStamped, eef_right_cb, queue_size=50)
        if self.use_base and 'robot_base_topics' in ros_cfg:
            rospy.Subscriber(rb_topics['odom'], Odometry, odom_cb, queue_size=50)
        if topics.get('enable_flag'):
            try:
                rospy.Subscriber(topics['enable_flag'], Bool, enable_cb, queue_size=10)
            except Exception:
                pass

        self.pub_l = rospy.Publisher(topics['cmd_joint_left'], JointState, queue_size=10)
        self.pub_r = rospy.Publisher(topics['cmd_joint_right'], JointState, queue_size=10)
        if self.eef_action_mode:
            self.log.info(
                "EEF action mode enabled with bridge-side IK; "
                f"publishing joint commands to {topics['cmd_joint_left']} and {topics['cmd_joint_right']}."
            )
        self.enable_pub = None
        if topics.get('enable_flag') and bool(ros_cfg.get('publish_enable_flag', False)):
            self.enable_pub = rospy.Publisher(topics['enable_flag'], Bool, queue_size=1, latch=True)
            rospy.sleep(0.2)
            self.enable_pub.publish(Bool(data=True))
            self.log.info(f"Published enable flag True on {topics['enable_flag']}.")
        self.pub_v = rospy.Publisher(rb_topics['cmd_vel'], Twist, queue_size=10) if self.use_base else None
        self.log.info(
            "Command publish configured: "
            f"mode={self.command_publish_mode} policy_rate={self.rate_hz}Hz "
            f"publish_rate={self.command_publish_rate_hz:.1f}Hz "
            f"substeps_per_action={self.command_publish_substeps} "
            f"interpolation={self.command_publish_interpolation}"
        )

        if self.eef_action_mode:
            self.ik = EefIkCommander(ros_cfg.get('eef_ik', {}), self.rate_hz, self.log)
            self.ik.log_configuration()
        if self.command_publish_mode == 'interpolated':
            self.publisher = InterpolatedCommandPublisher(
                lambda left, right: publish_joint_pair(self.pub_l, self.pub_r, self.name_list, left, right),
                self.command_publish_rate_hz,
                1.0 / float(self.rate_hz),
                rospy.is_shutdown,
                on_publish=self._note_published,
                log=self.log,
            ).start()
        self.shaper.log_configuration()
        self.pipeline.log_configuration()
        self.scheduler.log_configuration()

    def _note_published(self, ik=False):
        if not self.published_first_command:
            kind = 'IK joint command' if ik else 'command'
            self.log.info(
                f"Published first {kind} to {self.topics['cmd_joint_left']} and {self.topics['cmd_joint_right']}."
            )
            self.published_first_command = True

    def move_home(self):
        """Run ``ros.home_position``; returns ``False`` if inference must not start."""
        home_cfg = self.cfg['ros'].get('home_position')
        if not home_cfg:
            return True
        step_lengths = self.cfg['ros'].get('arm_steps_length', [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2])
        try:
            step_arr = np.asarray(step_lengths, dtype=np.float32)
            if step_arr.shape[0] != len(self.name_list):
                self.log.warn("arm_steps_length size mismatch; using uniform step length 0.01.")
                step_arr = np.full(len(self.name_list), 0.01, dtype=np.float32)
            require_home_settle = bool(home_cfg.get('require_settle', False))
            home_ok = True
            home_sequence = home_cfg.get('sequence')
            if home_sequence:
                for phase_index, phase_cfg in enumerate(home_sequence):
                    phase_ok = execute_home_phase(
                        self.pub_l, self.pub_r, self.name_list, step_arr, home_cfg, phase_cfg, phase_index)
                    home_ok = home_ok and bool(phase_ok)
                    if require_home_settle and not phase_ok:
                        break
            else:
                home_ok = execute_home_phase(
                    self.pub_l, self.pub_r, self.name_list, step_arr, home_cfg, home_cfg,
                    default_rate_hz=self.rate_hz)
            if require_home_settle and not home_ok:
                self.log.error("Home position did not settle; stopping before policy inference.")
                return False
            post_sleep_sec = max(float(home_cfg.get('post_sleep_sec', 0.0)), 0.0)
            if post_sleep_sec > 0.0:
                self.log.info(f"Waiting {post_sleep_sec:.2f}s after home before inference.")
                rospy.sleep(post_sleep_sec)
        except (KeyError, ValueError, TypeError):
            self.log.warn("Invalid home_position configuration; skipping home move.")
        return True

    # --- policy requests --------------------------------------------------

    def send_policy_request(self, pkt, fresh_camera_count, forced_fresh_fallback=False):
        measured_left = np.asarray(pkt['jleft'], dtype=np.float32)
        measured_right = np.asarray(pkt['jright'], dtype=np.float32)
        commanded_left = None
        commanded_right = None
        if self.command_left is not None and len(self.command_left) > 0:
            commanded_left = float(self.command_left[-1])
        if self.command_right is not None and len(self.command_right) > 0:
            commanded_right = float(self.command_right[-1])
        policy_left, policy_right, gripper_input_info = self.gripper_input.apply(
            measured_left, measured_right, commanded_left, commanded_right)

        header = {
            'task_prompt': pkt['task_prompt'],
            'episode_start': self.scheduler.request_id == 0,
            'jleft': policy_left.tolist(),
            'jright': policy_right.tolist(),
            'control_hz': self.rate_hz,
            'action_mode': self.action_mode,
            'policy_gripper_input': gripper_input_info,
            'measured_jleft_gripper': float(measured_left[-1]),
            'measured_jright_gripper': float(measured_right[-1]),
        }
        for key in ('xvla_proprio', 'current_eef_left', 'current_eef_right'):
            if pkt.get(key) is not None:
                header[key] = pkt[key]
        if self.use_base and pkt['odom'] is not None:
            header['odom'] = pkt['odom']

        frames = [json.dumps(header, separators=(',', ':')).encode('utf-8')] + [pkt[key] for key in CAMERA_KEYS]
        request_step = self.scheduler.step
        request_snapshot_dir = save_request_snapshot(
            self.action_logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback)
        if not self.client.send(frames):
            return False
        rollout_sample_dir = save_rollout_observation(
            self.rollout_logger, request_step, pkt, header, fresh_camera_count, forced_fresh_fallback)

        if self.eef_action_mode and pkt.get('current_eef_left') is not None and pkt.get('current_eef_right') is not None:
            request_current_left = np.asarray(pkt['current_eef_left'], dtype=np.float32)
            request_current_right = np.asarray(pkt['current_eef_right'], dtype=np.float32)
            measured_current_left = request_current_left.copy()
            measured_current_right = request_current_right.copy()
            request_current_left[-1] = policy_left[-1]
            request_current_right[-1] = policy_right[-1]
        else:
            request_current_left = policy_left.copy()
            request_current_right = policy_right.copy()
            measured_current_left = measured_left.copy()
            measured_current_right = measured_right.copy()

        self.client.mark_pending({
            'step': request_step,
            'time': time.monotonic(),
            'current_left': request_current_left.copy(),
            'current_right': request_current_right.copy(),
            'measured_current_left': measured_current_left.copy(),
            'measured_current_right': measured_current_right.copy(),
            'policy_gripper_input': dict(gripper_input_info),
            'obs_seq': dict(pkt['obs_seq']),
            'obs_time': dict(pkt['obs_time']),
            'fresh_camera_count': int(fresh_camera_count),
            'fresh_camera_forced': bool(forced_fresh_fallback),
            'request_snapshot_dir': request_snapshot_dir,
            'rollout_sample_dir': rollout_sample_dir,
        })
        self.request_gate.mark_sent(tuple(pkt['obs_seq'][key] for key in CAMERA_KEYS))
        if forced_fresh_fallback:
            self.log.warn(
                "Sending policy request before all cameras refreshed to avoid action starvation: "
                f"fresh_camera_count={fresh_camera_count}/3 seq={self.request_gate.last_camera_seq}"
            )
        self.log.info_once(
            'first_request', f"Sent first policy request to {self.client.connect_addr}; async action loop is running."
        )
        return True

    def maybe_send_request(self):
        pkt = snapshot(self.task_prompt, use_base=self.use_base, include_eef=self.eef_action_mode)
        if pkt is None:
            return
        stale_obs = stale_observations(
            pkt['obs_time'], time.monotonic(), self.max_image_age_sec, self.max_joint_age_sec, CAMERA_KEYS)
        if stale_obs:
            now = time.monotonic()
            if now - self.last_stale_obs_warn >= 1.0:
                formatted = [f"{key}={age:.3f}s" if age is not None else f"{key}=None" for key, age in stale_obs]
                self.log.warn(f"Delaying policy request because observations are stale: {formatted}")
                self.last_stale_obs_warn = now
            return
        camera_seq = tuple(pkt['obs_seq'][key] for key in CAMERA_KEYS)
        live = self.scheduler.live_chunks()
        remaining_steps = self.scheduler.remaining_steps(live)
        last_seq = self.request_gate.last_camera_seq
        decision, fresh = self.request_gate.decide(camera_seq, bool(live), remaining_steps)
        if decision in ('send', 'force'):
            self.send_policy_request(pkt, fresh, forced_fresh_fallback=decision == 'force')
            return
        now = time.monotonic()
        if now - self.last_stale_obs_warn >= 2.0:
            if decision == 'wait_chunk':
                self.log.info(
                    "Delaying policy request until current chunk is fully consumed: "
                    f"remaining_steps={remaining_steps}"
                )
            else:
                self.log.info(
                    "Delaying policy request until all camera images refresh: "
                    f"last={last_seq} current={camera_seq} "
                    f"fresh={fresh}/3 remaining_steps={remaining_steps}"
                )
            self.last_stale_obs_warn = now

    # --- policy responses ---------------------------------------------------

    def ingest_policy_response(self, rep_frames, request):
        request_step = request['step']
        policy_latency_sec = time.monotonic() - request['time']
        policy_latency_steps = self.scheduler.begin_response(request_step, policy_latency_sec)
        try:
            response = parse_action_response(rep_frames, self.action_mode, self.log)
            processed = self.pipeline.process(
                response['left'], response['right'], response['vel'],
                self.command_left, self.command_right,
                request.get('measured_current_left', request['current_left']),
                request.get('measured_current_right', request['current_right']),
            )
        except InvalidPolicyResponse as exc:
            (self.log.error if exc.level == 'error' else self.log.warn)(str(exc))
            return
        except ChunkRejected as exc:
            self.log.warn(str(exc))
            return

        action_skip_steps = processed['action_skip_steps']
        if action_skip_steps > 0:
            self.scheduler.skip_chunk_start(request_step, action_skip_steps)
            self.log.info(
                f"Skipping first {action_skip_steps} action step(s) of policy chunk "
                f"to avoid chunk-boundary transients."
            )

        request_id = self.scheduler.next_request_id()
        chunk = dict(processed)
        chunk.update(self.pipeline.metadata())
        chunk.update(self.shaper.metadata())
        model_raw_left = response['model_raw_left']
        model_raw_right = response['model_raw_right']
        chunk.update({
            'request_id': request_id,
            'start_step': request_step,
            'model_raw_left': None if model_raw_left is None else model_raw_left.copy(),
            'model_raw_right': None if model_raw_right is None else model_raw_right.copy(),
            'executed_steps': 0,
            'received_chunk_size': int(response['left'].shape[0]),
            'policy_latency_sec': policy_latency_sec,
            'policy_latency_steps': policy_latency_steps,
            'request_current_left': request['current_left'].copy(),
            'request_current_right': request['current_right'].copy(),
            'request_measured_left': request.get('measured_current_left', request['current_left']).copy(),
            'request_measured_right': request.get('measured_current_right', request['current_right']).copy(),
            'policy_gripper_input': dict(request.get('policy_gripper_input', {})),
            'request_obs_seq': dict(request['obs_seq']),
            'request_obs_time': dict(request['obs_time']),
            'request_fresh_camera_count': int(request['fresh_camera_count']),
            'request_fresh_camera_forced': bool(request['fresh_camera_forced']),
            'request_snapshot_dir': request.get('request_snapshot_dir', ''),
            'rollout_sample_dir': request.get('rollout_sample_dir', ''),
        })
        chunk['action_chunk_path'] = save_action_chunk(self.action_logger, request_id, chunk)
        chunk['rollout_action_path'] = save_rollout_action(
            self.rollout_logger, chunk.get('rollout_sample_dir', ''), request_id, chunk)
        self.scheduler.add_chunk(chunk)

    # --- per-step command -----------------------------------------------------

    def publish_step(self, chunk):
        """Publish the action of ``chunk`` for the current step; ``False`` if skipped."""
        scheduler = self.scheduler
        chunk_size = int(chunk['left'].shape[0])
        action_index = scheduler.step - chunk['start_step']
        if action_index < 0 or action_index >= chunk_size:
            return False

        command_left_before = self.command_left.copy()
        command_right_before = self.command_right.copy()
        seed_left, seed_right, jl_age, jr_age = latest_joint_arrays_and_age()
        publish_current_left = seed_left
        publish_current_right = seed_right
        if self.eef_action_mode:
            publish_current_left = self.command_left.copy()
            publish_current_right = self.command_right.copy()
        if self.max_joint_age_sec > 0.0 and (
            publish_current_left is None or publish_current_right is None
            or jl_age is None or jr_age is None
            or jl_age > self.max_joint_age_sec or jr_age > self.max_joint_age_sec
        ):
            now_warn = time.monotonic()
            if now_warn - self.last_stale_joint_warn >= 1.0:
                self.log.warn(
                    "Skipping action publish because joint feedback is stale: "
                    f"jl_age={jl_age} jr_age={jr_age} "
                    f"max_joint_age_sec={self.max_joint_age_sec:.3f}"
                )
                self.last_stale_joint_warn = now_warn
            return False
        if publish_current_left is None:
            publish_current_left = self.command_left.copy()
        if publish_current_right is None:
            publish_current_right = self.command_right.copy()

        ensemble_count = 0
        ensemble_weights = []
        if self.action_mode in ('absolute', 'eef_absolute'):
            raw_left = chunk['left'][action_index]
            raw_right = chunk['right'][action_index]
            ensembled_left, ensembled_right = raw_left, raw_right
            ensemble = scheduler.ensemble(chunk)
            if ensemble is not None:
                ensembled_left, ensembled_right, ensemble_count, ensemble_weights = ensemble
        else:
            raw_left, raw_right = self.integrator.step(chunk['left'][action_index], chunk['right'][action_index])
            ensembled_left, ensembled_right = raw_left, raw_right

        shaped = self.shaper.shape(
            ensembled_left, ensembled_right,
            command_left_before, command_right_before,
            publish_current_left, publish_current_right,
            seed_left, seed_right,
            chunk,
        )
        target_left = shaped['target_left']
        target_right = shaped['target_right']
        clip_reference_left = shaped['clip_reference_left']
        clip_reference_right = shaped['clip_reference_right']
        deltas = {
            'raw_delta': (raw_left - command_left_before, raw_right - command_right_before),
            'applied_delta': (target_left - command_left_before, target_right - command_right_before),
            'reference_delta': (target_left - clip_reference_left, target_right - clip_reference_right),
            'tracking_error_before': (command_left_before - publish_current_left,
                                      command_right_before - publish_current_right),
            'tracking_error_after': (target_left - publish_current_left, target_right - publish_current_right),
            'clip_residual': (raw_left - shaped['clipped_left'], raw_right - shaped['clipped_right']),
        }

        ik_results = None
        if self.command_publish_mode == 'direct':
            if self.eef_action_mode:
                ik_results = self.ik.solve(
                    target_left, target_right, seed_left, seed_right,
                    command_left_before, command_right_before,
                    context={'request_id': chunk['request_id'], 'action_index': action_index,
                             'global_action_step': scheduler.step},
                )
                if ik_results is None:
                    return False
                self.ik.apply_holds(ik_results, target_left, target_right,
                                    shaped['gripper_candidate'], shaped['gripper_transition'])
                publish_joint_pair(self.pub_l, self.pub_r, self.name_list,
                                   ik_results['left']['joints'], ik_results['right']['joints'])
                self.ik.commit(ik_results)
                self._note_published(ik=True)
            else:
                publish_joint_pair(self.pub_l, self.pub_r, self.name_list, target_left, target_right)
                self._note_published()
        else:
            self.publisher.set_target(command_left_before, command_right_before, target_left, target_right)

        self.shaper.commit(shaped)
        self.command_left = target_left.copy()
        self.command_right = target_right.copy()

        vel_mat = chunk['vel']
        if self.use_base and vel_mat is not None and self.pub_v is not None:
            v = Twist()
            v.linear.x = float(np.clip(vel_mat[action_index, 0], -self.v_max, self.v_max))
            v.angular.z = float(np.clip(vel_mat[action_index, 1], -self.w_max, self.w_max))
            self.pub_v.publish(v)

        self._log_step(
            chunk, action_index, chunk_size, shaped, deltas, ik_results,
            raw=(raw_left, raw_right), ensembled=(ensembled_left, ensembled_right),
            ensemble=(ensemble_count, ensemble_weights),
            command_before=(command_left_before, command_right_before),
            publish_current=(publish_current_left, publish_current_right),
            seed=(seed_left, seed_right), joint_age=(jl_age, jr_age),
        )
        scheduler.advance(chunk)
        return True

    def _log_step(self, chunk, action_index, chunk_size, shaped, deltas, ik_results,
                  raw, ensembled, ensemble, command_before, publish_current, seed, joint_age):
        if self.action_logger is None:
            return
        shaper = self.shaper
        pipeline = self.pipeline
        target = (shaped['target_left'], shaped['target_right'])
        filtered = (shaped['filtered_left'], shaped['filtered_right'])
        clip_reference = (shaped['clip_reference_left'], shaped['clip_reference_right'])
        residual = (shaped['deadband_residual_left'], shaped['deadband_residual_right'])
        ik_left = None if ik_results is None else ik_results['left']
        ik_right = None if ik_results is None else ik_results['right']
        request_measured = (
            chunk.get('request_measured_left', chunk['request_current_left']),
            chunk.get('request_measured_right', chunk['request_current_right']),
        )
        gripper_input = chunk.get('policy_gripper_input', {})
        vel_mat = chunk['vel']
        base_vel = None if vel_mat is None else vel_mat[action_index]
        now = rospy.Time.now()
        now_mono = time.monotonic()
        request_obs_age = {
            key: None if value is None else now_mono - value
            for key, value in chunk['request_obs_time'].items()
        }
        jl_age, jr_age = joint_age

        def fmt(value):
            return f'{float(value):.6f}'

        row = {
            'wall_time': f'{time.time():.6f}',
            'ros_time': f'{now.to_sec():.6f}',
            'request_id': chunk['request_id'],
            'tau': chunk['executed_steps'],
            'global_action_step': self.scheduler.step,
            'chunk_start_step': chunk['start_step'],
            'action_index': action_index,
            'policy_latency_sec': f"{chunk['policy_latency_sec']:.6f}",
            'policy_latency_steps': chunk['policy_latency_steps'],
            'chunk_size': chunk_size,
            'received_chunk_size': chunk['received_chunk_size'],
            'steps_to_execute': chunk['steps_to_execute'],
            'discarded_steps': action_index,
            'rate_hz': self.rate_hz,
            'command_publish_rate_hz': f'{self.command_publish_rate_hz:.6f}',
            'command_publish_substeps': self.command_publish_substeps,
            'task_prompt': self.task_prompt,
            'action_mode': self.action_mode,
            'chunk_interpolation_enabled': pipeline.interpolation_enabled,
            'chunk_interpolation_mode': pipeline.interpolation_mode,
            'chunk_interpolation_factor': f'{pipeline.interpolation_factor:.6f}',
            'request_obs_seq': json.dumps(chunk['request_obs_seq'], separators=(',', ':')),
            'request_obs_age_sec': json.dumps(request_obs_age, separators=(',', ':')),
            'request_snapshot_dir': chunk.get('request_snapshot_dir', ''),
            'action_chunk_path': chunk.get('action_chunk_path', ''),
            'rollout_sample_dir': chunk.get('rollout_sample_dir', ''),
            'rollout_action_path': chunk.get('rollout_action_path', ''),
            'request_fresh_camera_count': chunk['request_fresh_camera_count'],
            'request_fresh_camera_forced': chunk['request_fresh_camera_forced'],
            'delta_clip_enabled': shaper.delta_clip_enabled,
            'delta_clip_reference': shaper.delta_clip_reference,
            'command_delta_deadband_enabled': shaper.command_deadband_enabled,
            'first_action_delta_scale_enabled': shaper.first_scale_enabled,
            'first_action_delta_scale_coefficient': f'{shaper.first_scale_coefficient:.6f}',
            'first_action_delta_scale_include_gripper': shaper.first_scale_include_gripper,
            'first_action_delta_scale_applied': shaped['first_action_delta_scale_applied'],
            'initial_pose_delta_override_enabled': shaper.override_enabled,
            'initial_pose_delta_override_arms': json.dumps(sorted(shaper.override_arms), separators=(',', ':')),
            'initial_pose_delta_override_include_gripper': shaper.override_include_gripper,
            'temporal_ensemble_count': ensemble[0],
            'temporal_ensemble_weights': vector_to_json(ensemble[1]),
            'ik_seed_joint_left': vector_to_json(seed[0]) if self.eef_action_mode else '',
            'ik_seed_joint_right': vector_to_json(seed[1]) if self.eef_action_mode else '',
            'publish_joint_age_left_sec': '' if jl_age is None else f'{jl_age:.6f}',
            'publish_joint_age_right_sec': '' if jr_age is None else f'{jr_age:.6f}',
            'policy_gripper_input_mode': gripper_input.get('mode', 'measured'),
            'gripper_hysteresis': json.dumps(shaped['gripper_transition']),
            'base_vel': vector_to_json(base_vel),
        }
        for index, side in enumerate(('left', 'right')):
            ik = (ik_left, ik_right)[index]
            commanded = gripper_input.get(f'{side}_commanded')
            row.update({
                f'current_{side}': vector_to_json(chunk[f'request_current_{side}']),
                f'request_measured_{side}': vector_to_json(request_measured[index]),
                f'publish_current_{side}': vector_to_json(publish_current[index]),
                f'clip_reference_{side}': vector_to_json(clip_reference[index]),
                f'command_{side}_before': vector_to_json(command_before[index]),
                f'raw_{side}': vector_to_json(raw[index]),
                f'ensembled_{side}': vector_to_json(ensembled[index]),
                f'filtered_{side}': vector_to_json(filtered[index]),
                f'target_{side}': vector_to_json(target[index]),
                f'ik_diagnostics_{side}': '' if ik is None else json.dumps(
                    {k: v for k, v in ik.items() if k != 'joints'}, separators=(',', ':')),
                f'ik_joint_{side}': vector_to_json(None if ik is None else ik['joints']),
                f'ik_{side}_position_error_m': '' if ik is None else f"{ik['position_error_m']:.6f}",
                f'ik_{side}_orientation_error_rad': '' if ik is None else f"{ik['orientation_error_rad']:.6f}",
                f'command_delta_deadband_residual_{side}': vector_to_json(residual[index]),
                f'raw_vs_publish_norm_{side}': norm_first_six(raw[index] - publish_current[index]),
                f'ensembled_vs_publish_norm_{side}': norm_first_six(ensembled[index] - publish_current[index]),
                f'target_vs_publish_norm_{side}': norm_first_six(target[index] - publish_current[index]),
                f'command_delta_deadband_residual_norm_{side}': norm_first_six(residual[index]),
                f'{side}_gripper_request_measured': fmt(request_measured[index][-1]),
                f'{side}_gripper_request_policy': fmt(chunk[f'request_current_{side}'][-1]),
                f'{side}_gripper_request_commanded': '' if commanded is None else fmt(commanded),
                f'{side}_gripper_publish': fmt(publish_current[index][-1]),
                f'{side}_gripper_raw': fmt(raw[index][-1]),
                f'{side}_gripper_ensembled': fmt(ensembled[index][-1]),
                f'{side}_gripper_target': fmt(target[index][-1]),
            })
            for name, pair in deltas.items():
                row[f'{name}_{side}'] = vector_to_json(pair[index])
                row[f'{name}_norm_{side}'] = norm_first_six(pair[index])
        write_action_log(self.action_logger, row)

    # --- main loop ------------------------------------------------------------

    def run(self):
        self.setup_ros()
        if not self.move_home():
            return
        self.client = AsyncPolicyClient.from_config(self.cfg, self.rate_hz, self.log)
        self.log.info(
            f"Policy server endpoint set to {self.client.connect_addr} "
            f"using ZMQ {self.client.socket_type.upper()}")
        self.action_logger, self.action_log_path = make_action_logger(self.cfg, self.rate_hz)
        if self.action_logger is not None and self.ik is not None:
            self.ik.rejection_log_path = self.action_logger['path'] + '.rejected.jsonl'
        self.rollout_logger = make_rollout_dataset_logger(self.cfg, self.rate_hz)
        self.log.info(
            "Async policy loop enabled: "
            f"request_check_period={1.0 / self.rate_hz:.3f}s; "
            "continuing to publish available chunk actions while waiting for policy responses; "
            "policy requests are checked on each global step and wait for fresh front/left/right images; "
            f"when_live_chunk={self.request_gate.when_live_chunk}; "
            f"initial_action_skip_steps={self.scheduler.initial_action_skip_steps}; "
            f"chunk_action_skip_steps={self.pipeline.chunk_action_skip_steps}; "
            f"policy_gripper_input_mode={self.gripper_input.mode}."
        )
        rate = rospy.Rate(self.rate_hz)
        try:
            while not rospy.is_shutdown():
                self.spin_once()
                rate.sleep()
        except (KeyboardInterrupt, rospy.ROSInterruptException):
            self.log.info("cobotmagic_policy_bridge_node interrupted, shutting down.")
        finally:
            self.shutdown()

    def spin_once(self):
        """One control step: receive replies, maybe request, publish one action."""
        if not enable_state:
            return
        has_obs = have_obs(use_base=self.use_base, require_eef=self.eef_action_mode)
        if not has_obs and not self.scheduler.live_chunks() and not self.client.busy:
            now = time.monotonic()
            if now - self.last_obs_wait_warn >= 2.0:
                missing = missing_obs_keys(use_base=self.use_base, require_eef=self.eef_action_mode)
                self.log.warn(f"Waiting for observations: missing={missing}")
                self.last_obs_wait_warn = now
            return

        if self.command_left is None or self.command_right is None:
            pkt = snapshot(self.task_prompt, use_base=self.use_base, include_eef=self.eef_action_mode) if has_obs else None
            if pkt is None:
                return
            if self.eef_action_mode:
                self.command_left = np.array(pkt['current_eef_left'], dtype=np.float32)
                self.command_right = np.array(pkt['current_eef_right'], dtype=np.float32)
            else:
                self.command_left = np.array(pkt['jleft'], dtype=np.float32)
                self.command_right = np.array(pkt['jright'], dtype=np.float32)
            self.integrator.reset(self.command_left, self.command_right)

        reply = self.client.poll()
        if reply is not None:
            self.ingest_policy_response(*reply)
        if not self.client.busy and has_obs:
            self.maybe_send_request()

        chunk = self.scheduler.select_chunk()
        if chunk is not None:
            self.publish_step(chunk)

    def shutdown(self):
        run_shutdown_safety(self.pub_l, self.pub_r, self.pub_v, self.enable_pub, self.name_list, self.cfg)
        if self.action_logger is not None:
            self.action_logger['file'].flush()
            self.action_logger['file'].close()
            self.log.info(f"Action command log saved: {self.action_log_path}")
        if self.rollout_logger is not None and self.rollout_logger.get('h5_file') is not None:
            self.rollout_logger['h5_file'].flush()
            self.rollout_logger['h5_file'].close()
            self.log.info(f"Rollout HDF5 saved: {self.rollout_logger.get('h5_path')}")
        self.client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='path to a backend-specific YAML config')
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    rospy.init_node('cobotmagic_policy_bridge_node')
    DualArmPolicyBridge(cfg).run()


if __name__ == '__main__':
    main()
