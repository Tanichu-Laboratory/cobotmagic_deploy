import json
import threading

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import zmq

from cobotmagic_deployment.common.openwam_piper import actions_to_bridge, request_state, validate_checkpoint
from cobotmagic_deployment.common.policy_server_protocol import recv_packet, send_actions, send_empty

GRIP = {"closed": [0., 0.], "open": [0.06558, 0.06558]}


def header():
    return {"current_eef_left": [0.2, -0.1, 0.3, 0.3, -0.5, 0.7, 0.0],
            "current_eef_right": [0.4, 0.2, 0.1, -0.2, 0.6, -0.9, 0.06558]}


def test_rotations_match_openwam_and_roundtrip():
    # Non-identity rotations detect interleaved-XVLA and intrinsic-Euler errors.
    from openwam.dataloader.utils.poses import quat_wxyz_to_rot6d
    h = header()
    state = request_state(h, GRIP)
    for i, side in enumerate(("left", "right")):
        q = Rotation.from_euler("xyz", h["current_eef_" + side][3:6]).as_quat()
        np.testing.assert_allclose(state[i*10+3:i*10+9], quat_wxyz_to_rot6d(q[[3, 0, 1, 2]]), atol=1e-6)
    np.testing.assert_equal(state[[9, 19]], [0, 1])
    np.testing.assert_allclose(actions_to_bridge(state[None], GRIP)[0],
                               h["current_eef_left"] + h["current_eef_right"], atol=1e-6)


@pytest.mark.parametrize("key,value", [("current_eef_left", [0]*6), ("current_eef_right", [float("nan")]*7),
                                     ("current_eef_left", [0]*6 + [1.0])])
def test_invalid_state_rejected(key, value):
    h = header(); h[key] = value
    with pytest.raises(ValueError):
        request_state(h, GRIP)


def test_joint_only_rejected():
    with pytest.raises(ValueError):
        request_state({"jleft": [0]*7, "jright": [0]*7}, GRIP)


@pytest.mark.parametrize("bad", [np.zeros((8, 80)), np.zeros((8, 20)), np.full((8, 20), np.nan)])
def test_invalid_actions_rejected(bad):
    with pytest.raises(ValueError):
        actions_to_bridge(bad, GRIP)


def test_gripper_clamped_to_physical_limits():
    state = request_state(header(), GRIP)
    state[[9, 19]] = [-2, 3]
    a = actions_to_bridge(state[None], GRIP)
    np.testing.assert_allclose(a[0, [6, 13]], [0, 0.06558])


def test_wrong_embodiment_rejected():
    with pytest.raises(ValueError):
        validate_checkpoint({"dataloader": {"type": "robodojo", "variant": "sim"}})


def test_wire_roundtrip_and_error_recovery():
    ctx = zmq.Context()
    server = ctx.socket(zmq.REP); client = ctx.socket(zmq.REQ)
    server.bind("inproc://openwam-test"); client.connect("inproc://openwam-test")
    client.setsockopt(zmq.RCVTIMEO, 5000)
    def serve():
        for _ in range(2):
            try:
                h, images = recv_packet(server)
                assert images["front"].shape == (32, 32, 3)
                action = actions_to_bridge(request_state(h, GRIP)[None], GRIP)
                send_actions(server, action, 5, action_mode="eef_absolute")
            except ValueError as e:
                send_empty(server, str(e), action_mode="eef_absolute")
    worker = threading.Thread(target=serve); worker.start()
    jpeg = cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))[1].tobytes()
    client.send_multipart([b"{}", jpeg, jpeg, jpeg])
    assert json.loads(client.recv_multipart()[0])["chunk_size"] == 0
    client.send_multipart([json.dumps(header()).encode(), jpeg, jpeg, jpeg])
    frames = client.recv_multipart()
    assert json.loads(frames[0])["action_mode"] == "eef_absolute"
    np.testing.assert_allclose(np.frombuffer(frames[1], np.float32), header()["current_eef_left"], atol=1e-6)
    np.testing.assert_allclose(np.frombuffer(frames[2], np.float32), header()["current_eef_right"], atol=1e-6)
    worker.join(); server.close(); client.close(); ctx.term()


def test_requested_home_gripper_undershoot():
    h = header()
    h["current_eef_left"][-1] = -0.0037
    h["current_eef_right"][-1] = -0.0009
    calibration = dict(GRIP, sensor_tolerance=0.005)
    state = request_state(h, calibration)
    np.testing.assert_equal(state[[9, 19]], [0, 0])
    h["current_eef_left"][-1] = -0.006
    with pytest.raises(ValueError):
        request_state(h, calibration)


@pytest.mark.parametrize("execution_steps", [None, 8, 32, 64])
@pytest.mark.parametrize("generated_steps", [17, 32])
def test_full_generated_chunk_survives_policy_and_wire(tmp_path, execution_steps, generated_steps):
    from types import SimpleNamespace
    import yaml
    from cobotmagic_deployment.servers.policy_server_openwam_piper import OpenWAMPiperPolicy

    training = {"dataloader": {"type": "robodojo", "variant": "real", "embodiment": "piper",
                "action_mode": "eef", "unify_action": True, "unify_action_map": ["0-9", "34-43"],
                "camera_layout": ["cam_head", "cam_left_wrist", "cam_right_wrist"], "num_frames": 33}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(training))
    cfg = {"ros": {"action_mode": "eef_absolute", "open_loop_steps": execution_steps},
           "openwam": {"checkpoint_path": str(tmp_path), "gripper": GRIP}, "task_prompt": "test"}
    policy = OpenWAMPiperPolicy(cfg, mock=True)
    assert policy.predict(header(), {}).shape == (32, 14)

    # Exercise the real predict path with a deterministic engine, including a
    # returned length different from both the nominal window and execution cap.
    generated = np.repeat(request_state(header(), GRIP)[None], generated_steps, axis=0)
    generated[:, 0] += np.arange(generated_steps) * 0.001
    policy.mock = False
    policy.server = SimpleNamespace(engine=SimpleNamespace(generate=lambda conditions: {"actions": generated}))
    policy.preprocess = SimpleNamespace(preprocess=lambda obs: dict(obs, image=obs["images"]["head_camera"]))
    images = {key: np.zeros((32, 32, 3), np.uint8) for key in ("front", "left", "right")}
    actions = policy.predict(header(), images)
    assert actions.shape == (generated_steps, 14)
    np.testing.assert_allclose(actions[:, 0], generated[:, 0])

    ctx = zmq.Context()
    server, client = ctx.socket(zmq.PAIR), ctx.socket(zmq.PAIR)
    try:
        server.bind("inproc://full-model-chunk")
        client.connect("inproc://full-model-chunk")
        client.setsockopt(zmq.RCVTIMEO, 2000)
        send_actions(server, actions, 5, action_mode="eef_absolute")
        frames = client.recv_multipart()
        assert json.loads(frames[0])["chunk_size"] == generated_steps
        np.testing.assert_allclose(np.frombuffer(frames[1], np.float32).reshape(-1, 7), actions[:, :7])
        np.testing.assert_allclose(np.frombuffer(frames[2], np.float32).reshape(-1, 7), actions[:, 7:])
    finally:
        server.close(linger=0); client.close(linger=0); ctx.term()



def test_gripper_opening_boost_preserves_pose_limits_and_measured_state():
    boosted = dict(GRIP, action_open_normalized=0.75)
    state = request_state(header(), GRIP)
    actions = np.repeat(state[None], 6, axis=0)
    actions[:, 9] = [-0.2, 0., .375, .75, 1., 1.2]
    actions[:, 19] = [1.2, 1., .75, .375, 0., -0.2]
    output = actions_to_bridge(actions, boosted)
    np.testing.assert_allclose(output[:, 6], [0., 0., .03279, .06558, .06558, .06558], atol=1e-8)
    np.testing.assert_allclose(output[:, 13], [.06558, .06558, .06558, .03279, 0., 0.], atol=1e-8)
    baseline = actions_to_bridge(actions, GRIP)
    np.testing.assert_array_equal(output[:, [0,1,2,3,4,5,7,8,9,10,11,12]],
                                  baseline[:, [0,1,2,3,4,5,7,8,9,10,11,12]])
    # Do not misrepresent measured full-open as 0.75 to the model.
    np.testing.assert_array_equal(request_state(header(), boosted), state)


@pytest.mark.parametrize("bad", [0., -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_gripper_opening_threshold_rejected(bad):
    with pytest.raises(ValueError, match="action_open_normalized"):
        actions_to_bridge(request_state(header(), GRIP)[None], dict(GRIP, action_open_normalized=bad))


@pytest.mark.parametrize("dataset", ["robodojo", "gm100"])
def test_supported_piper_checkpoint_layout(dataset):
    validate_checkpoint({"dataloader": {
        "type": dataset, "variant": "real", "embodiment": "piper",
        "action_mode": "eef", "unify_action": True,
        "unify_action_map": ["0-9", "34-43"],
        "camera_layout": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
    }})


@pytest.mark.parametrize("change", [
    {"type": "unknown"}, {"variant": "sim"}, {"embodiment": "piper_x"},
    {"action_mode": "joint"}, {"unify_action": False},
    {"unify_action_map": ["34-43", "0-9"]},
    {"camera_layout": ["cam_head", "cam_right_wrist", "cam_left_wrist"]},
])
def test_gm100_incompatible_layout_still_rejected(change):
    dl = dict(type="gm100", variant="real", embodiment="piper",
              action_mode="eef", unify_action=True,
              unify_action_map=["0-9", "34-43"],
              camera_layout=["cam_head", "cam_left_wrist", "cam_right_wrist"])
    dl.update(change)
    with pytest.raises(ValueError):
        validate_checkpoint({"dataloader": dl})
