"""ROS-independent bridge components: scheduler, pipeline, shaper, IK, client, publisher."""
import json
import threading
import time

import numpy as np
import pytest
import zmq

from cobotmagic_deployment.common.bridge_log import BridgeLog
from cobotmagic_deployment.common.chunk_pipeline import (
    ChunkPipeline, ChunkRejected, InvalidPolicyResponse, parse_action_response,
)
from cobotmagic_deployment.common.chunk_scheduler import ChunkScheduler, PolicyRequestGate, stale_observations
from cobotmagic_deployment.common.command_publisher import InterpolatedCommandPublisher, command_publish_settings
from cobotmagic_deployment.common.command_shaper import CommandShaper, PolicyGripperInput, VelocityIntegrator
from cobotmagic_deployment.common.ik_commander import EefIkCommander
from cobotmagic_deployment.common.policy_client import AsyncPolicyClient
from test_piper_ik import SEED, URDF, pose, solver


class RecordingLog(BridgeLog):
    def __init__(self):
        self.messages = []
        super().__init__(lambda m: self.messages.append(('info', m)),
                         lambda m: self.messages.append(('warn', m)),
                         lambda m: self.messages.append(('error', m)))


def chunk(start, T=6, value=0.0, **extra):
    c = {'start_step': start, 'left': np.full((T, 7), value, np.float32), 'right': np.full((T, 7), value, np.float32),
         'steps_to_execute': T, 'overlap_steps': None, 'request_id': extra.pop('request_id', 1)}
    c.update(extra)
    return c


def reply(actions, action_mode='absolute', **header):
    actions = np.asarray(actions, np.float32)
    h = {'chunk_size': len(actions), 'left_stride': 7, 'right_stride': 7, 'action_mode': action_mode}
    h.update(header)
    return [json.dumps(h).encode(), actions[:, :7].tobytes(), actions[:, 7:14].tobytes()]


# --- scheduler / request gate ---------------------------------------------

def test_scheduler_timeline_and_pruning():
    s = ChunkScheduler({'temporal_ensemble': {'max_history_chunks': 2}}, rate_hz=5)
    c = chunk(0, T=3)
    c['request_id'] = s.next_request_id()
    s.add_chunk(c)
    assert s.select_chunk() is c and s.remaining_steps() == 3
    for _ in range(3):
        s.advance(c)
    assert s.select_chunk() is None and s.history == []


def test_latency_compensation_and_initial_skip():
    cfg = {'initial_action_skip_steps': 2,
           'temporal_ensemble': {'latency_compensation': {'enabled': True, 'mode': 'measured', 'max_steps': 3}}}
    s = ChunkScheduler(cfg, rate_hz=10)
    assert s.begin_response(request_step=0, latency_sec=0.1) == 1
    assert s.step == 2  # initial skip applies before the first chunk
    s.next_request_id()
    assert s.begin_response(request_step=4, latency_sec=1.0) == 3 and s.step == 7


def test_temporal_ensemble_respects_overlap():
    s = ChunkScheduler({'temporal_ensemble': {'enabled': True, 'exp_decay': 0.0}}, rate_hz=5)
    a, b = chunk(0, value=0.0, request_id=1), chunk(0, value=1.0, request_id=2)
    s.add_chunk(a)
    s.add_chunk(b)
    left, _, count, _ = s.ensemble(b)
    assert count == 2 and left[0] == pytest.approx(0.5)
    b['overlap_steps'] = 0
    assert s.ensemble(b) is None


def test_request_gate_decisions():
    gate = PolicyRequestGate({'policy_request': {'when_live_chunk': 'allow'}})
    assert gate.decide((1, 1, 1), False, 0) == ('send', 3)
    gate.mark_sent((1, 1, 1))
    assert gate.decide((2, 1, 1), True, 20) == ('wait_cameras', 1)
    assert gate.decide((2, 1, 1), True, 3) == ('force', 1)
    wait = PolicyRequestGate({'policy_request': {'when_live_chunk': 'wait'}})
    assert wait.decide((1, 1, 1), True, 5)[0] == 'wait_chunk'
    assert stale_observations({'front': 0.0, 'left': 9.0, 'right': 9.0, 'jl': 9.0, 'jr': None}, 10.0, 2.0, 0.75) == [
        ('front', 10.0), ('jl', 1.0), ('jr', None)]


# --- response parsing / pipeline --------------------------------------------

def test_parse_action_response_with_velocity_and_errors():
    acts = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    frames = reply(acts, has_vel=True) + [np.ones((3, 2), np.float32).tobytes()]
    parsed = parse_action_response(frames, 'absolute')
    np.testing.assert_array_equal(parsed['right'], acts[:, 7:])
    assert parsed['vel'].shape == (3, 2)
    with pytest.raises(InvalidPolicyResponse) as exc:
        parse_action_response(reply(acts, 'eef_absolute'), 'absolute')
    assert exc.value.level == 'error'
    with pytest.raises(InvalidPolicyResponse):
        parse_action_response([b'{"chunk_size":0}'], 'absolute')


def test_pipeline_interpolation_and_skip():
    pipe = ChunkPipeline({'open_loop_steps': 4, 'chunk_action_skip_steps': 1,
                          'chunk_interpolation': {'enabled': True, 'mode': 'linear', 'factor': 2.0}}, rate_hz=5)
    left = np.linspace(0, 1, 5, dtype=np.float32)[:, None].repeat(7, 1)
    out = pipe.process(left, left.copy(), None, left[0], left[0], left[0], left[0])
    assert out['left'].shape[0] == 10 and out['steps_to_execute'] == 4 and out['action_skip_steps'] == 1
    np.testing.assert_array_equal(out['received_left'], left)


def test_pipeline_rejects_failed_stage():
    pipe = ChunkPipeline({'chunk_interpolation': {'enabled': True, 'mode': 'adaptive_delta',
                                                  'joint_threshold': [0.1] * 5}}, rate_hz=5)
    left = np.zeros((4, 7), np.float32)
    with pytest.raises(ChunkRejected):
        pipe.process(left, left.copy(), None, left[0], left[0], left[0], left[0])
    with pytest.raises(ValueError):
        ChunkPipeline({'action_chunk_lowpass': {'enabled': True, 'cutoff_hz': 10}}, rate_hz=5)


# --- shaping ------------------------------------------------------------------

def test_command_shaper_clip_deadband_and_override():
    cfg = {'delta_clip': {'enabled': True, 'reference': 'command', 'max_delta': [0.1] * 7},
           'command_delta_deadband': {'enabled': True, 'left': [0.05] * 7, 'right': [0.0] * 7},
           'initial_pose_delta_override': {'enabled': True, 'arms': 'right'}}
    shaper = CommandShaper(cfg)
    zero = np.zeros(7, np.float32)
    target = np.array([0.5, 0.03, 0, 0, 0, 0, 0], np.float32)
    c = {'request_current_left': zero, 'request_current_right': zero, 'executed_steps': 0}
    out = shaper.shape(target, target, zero, zero, zero, zero, zero, zero, c)
    np.testing.assert_allclose(out['clipped_left'][:2], [0.1, 0.03])
    np.testing.assert_allclose(out['target_left'][:2], [0.1, 0.0])  # 0.03 < deadband
    np.testing.assert_allclose(out['target_right'][:6], 0.0)  # right arm held at request pose


def test_hysteresis_state_only_changes_on_commit():
    cfg = {'gripper_hysteresis': {'enabled': True, 'closed': [0, 0], 'open': [0.06, 0.06],
                                  'close_threshold_normalized': [0.4, 0.4],
                                  'open_threshold_normalized': [0.6, 0.6], 'confirm_steps': 1}}
    shaper = CommandShaper(cfg)
    zero = np.zeros(7, np.float32)
    opened = zero.copy()
    opened[-1] = 0.06
    c = {'request_current_left': zero, 'request_current_right': zero, 'executed_steps': 1}
    out = shaper.shape(opened, opened, zero, zero, zero, zero, zero, zero, c)
    assert out['target_left'][-1] == pytest.approx(0.06)
    assert shaper.hysteresis.is_open is None
    shaper.commit(out)
    assert shaper.hysteresis.is_open.tolist() == [True, True]


def test_policy_gripper_input_and_velocity_integrator():
    hybrid = PolicyGripperInput({'policy_gripper_input': {'mode': 'hybrid', 'hybrid_max_error': 0.01}})
    left, right, info = hybrid.apply([0] * 6 + [0.05], [0] * 6 + [0.05], 0.055, 0.0)
    assert left[-1] == pytest.approx(0.055) and right[-1] == pytest.approx(0.05)
    assert info['left_source'] == 'commanded' and info['right_source'] == 'measured'
    integ = VelocityIntegrator(dt=0.5)
    integ.reset(np.zeros(7), np.zeros(7))
    left, _ = integ.step(np.array([1, 0, 0, 0, 0, 0, 0.02]), np.zeros(7))
    assert left[0] == 0.5 and left[-1] == 0.02


# --- IK commander ---------------------------------------------------------------

def test_ik_commander_solves_and_rejects():
    ik = solver()
    commander = EefIkCommander(dict(urdf_path=URDF, solver='least_squares', joint_regularization_weight=0.,
                                    max_position_error_m=.005, max_orientation_error_rad=.05,
                                    max_joint_delta_rad=[.18, .18, .18, .24, .24, .24]), rate_hz=5)
    seed = np.r_[SEED, 0.03].astype(np.float32)
    start = pose(ik, SEED)
    target = pose(ik, SEED + [.02, .02, -.02, .02, -.02, .02])
    results = commander.solve(target, target, seed, seed, start, start)
    assert results is not None
    np.testing.assert_allclose(ik.fk(results['left']['joints'][:6])[:3, 3], target[:3], atol=5e-3)
    commander.commit(results)
    assert commander.last_publish_time is not None
    far = start.copy()
    far[:3] += 5.0  # unreachable
    assert commander.solve(far, far, seed, seed, start, start) is None


# --- transport / publisher --------------------------------------------------------

def test_async_client_poll_and_timeout():
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.REP)
    port = server.bind_to_random_port('tcp://127.0.0.1')
    log = RecordingLog()
    client = AsyncPolicyClient(f'tcp://127.0.0.1:{port}', 'req', io_timeout_ms=200, response_timeout_sec=0.2, log=log)
    assert client.send([b'ping'])
    client.mark_pending({'time': time.monotonic(), 'step': 3})
    assert client.poll() is None and client.busy
    server.send_multipart([server.recv(), b'reply'][1:])
    deadline = time.monotonic() + 2.0
    got = None
    while got is None and time.monotonic() < deadline:
        got = client.poll()
    assert got[0] == [b'reply'] and got[1]['step'] == 3 and not client.busy
    client.send([b'lost'])
    client.mark_pending({'time': time.monotonic() - 1.0, 'step': 4})
    assert client.poll() is None and not client.busy
    assert any('timed out' in m for _, m in log.messages)
    client.close()
    server.close(0)


def test_interpolated_publisher_and_settings():
    assert command_publish_settings({'command_publish': {'mode': 'interpolated', 'rate_hz': 70}}, 15)[:3] == (
        'interpolated', 75.0, 5)
    assert command_publish_settings({'command_publish': {'mode': 'interpolated'}}, 15, eef_action_mode=True)[0] == 'direct'
    published = []
    stop = threading.Event()
    pub = InterpolatedCommandPublisher(lambda l, r: published.append(l.copy()), 200.0, 0.1, stop.is_set,
                                       log=RecordingLog()).start()
    start = np.zeros(7)
    target = np.ones(7)
    pub.set_target(start, start, target, target)
    time.sleep(0.2)
    stop.set()
    assert published[0][-1] == 1.0  # gripper jumps immediately
    assert 0.0 <= published[0][0] < 0.5 and published[-1][0] == pytest.approx(1.0)
