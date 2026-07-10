#!/usr/bin/env python3
"""Small EEF PosCmd smoke test for Piper/CobotMagic.

Reads the current /puppet/end_pose_* pose, publishes a small interpolated
/control/end_pose_* command, then returns to the starting pose.
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from dataclasses import dataclass

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

try:
    from piper_msgs.msg import PosCmd
except Exception:  # noqa: BLE001
    for path in glob.glob('/workspace/ros_cobotmagic/Piper_ros_private-ros-noetic/devel/lib/python*/dist-packages'):
        if path not in sys.path:
            sys.path.append(path)
    from piper_msgs.msg import PosCmd


@dataclass
class EefState:
    xyz: list[float]
    rpy: list[float]
    gripper: float


def quat_xyzw_to_euler_xyz(quat: list[float]) -> list[float]:
    x, y, z, w = [float(v) for v in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        return [0.0, 0.0, 0.0]
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    m00 = 1.0 - 2.0 * (y * y + z * z)
    m01 = 2.0 * (x * y - w * z)
    m02 = 2.0 * (x * z + w * y)
    m11 = 1.0 - 2.0 * (x * x + z * z)
    m12 = 2.0 * (y * z - w * x)
    m21 = 2.0 * (y * z + w * x)
    m22 = 1.0 - 2.0 * (x * x + y * y)

    sy = max(-1.0, min(1.0, m02))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(-m12, m22)
        yaw = math.atan2(-m01, m00)
    else:
        roll = math.atan2(m21, m11)
        yaw = 0.0
    return [roll, pitch, yaw]


def wait_for_pose(topic: str, timeout: float) -> PoseStamped:
    rospy.loginfo('Waiting for %s', topic)
    return rospy.wait_for_message(topic, PoseStamped, timeout=timeout)


def wait_for_joint_gripper(topic: str, timeout: float, fallback: float) -> float:
    try:
        msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
        if msg.position:
            return float(msg.position[-1])
    except Exception as exc:  # noqa: BLE001
        rospy.logwarn('Could not read %s gripper, using fallback %.4f: %s', topic, fallback, exc)
    return float(fallback)


def pose_to_state(pose_msg: PoseStamped, gripper: float) -> EefState:
    p = pose_msg.pose.position
    q = pose_msg.pose.orientation
    return EefState(
        xyz=[float(p.x), float(p.y), float(p.z)],
        rpy=quat_xyzw_to_euler_xyz([float(q.x), float(q.y), float(q.z), float(q.w)]),
        gripper=float(gripper),
    )


def make_msg(state: EefState) -> PosCmd:
    msg = PosCmd()
    msg.x, msg.y, msg.z = state.xyz
    msg.roll, msg.pitch, msg.yaw = state.rpy
    msg.gripper = state.gripper
    return msg


def lerp_state(a: EefState, b: EefState, t: float) -> EefState:
    return EefState(
        xyz=[ai + (bi - ai) * t for ai, bi in zip(a.xyz, b.xyz)],
        rpy=[ai + (bi - ai) * t for ai, bi in zip(a.rpy, b.rpy)],
        gripper=a.gripper + (b.gripper - a.gripper) * t,
    )


def shifted(state: EefState, axis: str, distance: float) -> EefState:
    idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    out = EefState(xyz=list(state.xyz), rpy=list(state.rpy), gripper=state.gripper)
    out.xyz[idx] += float(distance)
    return out


def wait_for_subscribers(pubs: list[rospy.Publisher], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if all(pub.get_num_connections() > 0 for pub in pubs):
            return True
        rospy.sleep(0.1)
    return all(pub.get_num_connections() > 0 for pub in pubs)


def publish_segment(pairs: list[tuple[rospy.Publisher, EefState, EefState]], duration: float, rate_hz: float, dry_run: bool) -> None:
    steps = max(int(round(duration * rate_hz)), 1)
    rate = rospy.Rate(rate_hz)
    for i in range(steps + 1):
        t = i / float(steps)
        for pub, start, goal in pairs:
            msg = make_msg(lerp_state(start, goal, t))
            if not dry_run:
                pub.publish(msg)
        rate.sleep()


def format_state(label: str, state: EefState) -> str:
    vals = state.xyz + state.rpy + [state.gripper]
    return f"{label}: " + ', '.join(f'{v:.6f}' for v in vals)


def main() -> None:
    parser = argparse.ArgumentParser(description='Move EEF a small distance and return to the current pose.')
    parser.add_argument('--arm', choices=['left', 'right', 'both'], default='left')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default='x', help='Base-frame axis to move along. Default +x is treated as forward.')
    parser.add_argument('--distance', type=float, default=0.02, help='Move distance in meters. Use negative for the opposite direction.')
    parser.add_argument('--duration', type=float, default=2.0, help='Seconds for outbound and return motion respectively.')
    parser.add_argument('--hold', type=float, default=0.5, help='Seconds to hold at the shifted pose before returning.')
    parser.add_argument('--rate-hz', type=float, default=20.0)
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--fallback-gripper', type=float, default=0.03)
    parser.add_argument('--dry-run', action='store_true', help='Read and print poses but do not publish commands.')
    parser.add_argument('--allow-large', action='store_true', help='Allow abs(distance) > 0.05 m.')
    args = parser.parse_args()

    if abs(args.distance) > 0.05 and not args.allow_large:
        raise SystemExit('Refusing distance > 0.05 m without --allow-large')
    if args.duration < 0.5:
        raise SystemExit('Refusing duration < 0.5 s')
    if args.rate_hz <= 0.0:
        raise SystemExit('--rate-hz must be positive')

    rospy.init_node('eef_motion_smoke_test', anonymous=True)

    selected = ['left', 'right'] if args.arm == 'both' else [args.arm]
    topics = {
        'left': {
            'pose': '/puppet/end_pose_left',
            'joint': '/puppet/joint_left',
            'cmd': '/control/end_pose_left',
        },
        'right': {
            'pose': '/puppet/end_pose_right',
            'joint': '/puppet/joint_right',
            'cmd': '/control/end_pose_right',
        },
    }

    pubs: dict[str, rospy.Publisher] = {}
    starts: dict[str, EefState] = {}
    goals: dict[str, EefState] = {}
    for arm in selected:
        pose = wait_for_pose(topics[arm]['pose'], args.timeout)
        gripper = wait_for_joint_gripper(topics[arm]['joint'], min(args.timeout, 1.0), args.fallback_gripper)
        starts[arm] = pose_to_state(pose, gripper)
        goals[arm] = shifted(starts[arm], args.axis, args.distance)
        pubs[arm] = rospy.Publisher(topics[arm]['cmd'], PosCmd, queue_size=10)
        rospy.loginfo(format_state(f'{arm} start xyz/rpy/gripper', starts[arm]))
        rospy.loginfo(format_state(f'{arm} target xyz/rpy/gripper', goals[arm]))

    if not args.dry_run:
        if not wait_for_subscribers([pubs[arm] for arm in selected], args.timeout):
            raise SystemExit('No subscriber on one or more /control/end_pose_* topics; aborting')
        rospy.loginfo('Publishing start pose for 0.5s')
        start_pairs = [(pubs[arm], starts[arm], starts[arm]) for arm in selected]
        publish_segment(start_pairs, 0.5, args.rate_hz, dry_run=False)

    out_pairs = [(pubs[arm], starts[arm], goals[arm]) for arm in selected]
    back_pairs = [(pubs[arm], goals[arm], starts[arm]) for arm in selected]

    rospy.loginfo('Moving %s along %+0.4f m on %s axis', ','.join(selected), args.distance, args.axis)
    publish_segment(out_pairs, args.duration, args.rate_hz, args.dry_run)
    if args.hold > 0.0:
        rospy.sleep(args.hold)
    rospy.loginfo('Returning to captured start pose')
    publish_segment(back_pairs, args.duration, args.rate_hz, args.dry_run)
    rospy.loginfo('EEF smoke test complete')


if __name__ == '__main__':
    main()
