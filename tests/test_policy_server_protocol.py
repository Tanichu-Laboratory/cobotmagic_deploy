import json

import numpy as np
import pytest
import zmq

from cobotmagic_deployment.common.policy_server_protocol import (
    client_socket_kind, encode_jpeg, decode_jpeg, send_actions, send_empty, server_socket_kind,
)


@pytest.fixture
def pair():
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.PAIR)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    client = ctx.socket(zmq.PAIR)
    client.connect(f"tcp://127.0.0.1:{port}")
    client.setsockopt(zmq.RCVTIMEO, 2000)
    yield server, client
    server.close(0)
    client.close(0)


def test_socket_kinds():
    assert client_socket_kind("REQ") == zmq.REQ
    assert server_socket_kind("req") == zmq.REP
    assert client_socket_kind("pair") == server_socket_kind("pair") == zmq.PAIR
    with pytest.raises(ValueError):
        client_socket_kind("pub")


def test_jpeg_roundtrip_returns_rgb():
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # blue in OpenCV order
    rgb = decode_jpeg(encode_jpeg(bgr, quality=95))
    assert rgb[..., 2].mean() > 200 and rgb[..., 0].mean() < 50


def test_send_actions_with_velocity_and_without_mode(pair):
    server, client = pair
    actions = np.arange(3 * 16, dtype=np.float32).reshape(3, 16)
    send_actions(server, actions[:, :14], 5.0, action_mode=None, vel=actions[:, 14:16])
    frames = client.recv_multipart()
    header = json.loads(frames[0])
    assert "action_mode" not in header and header["has_vel"] is True
    assert len(frames) == 4
    np.testing.assert_array_equal(np.frombuffer(frames[3], np.float32).reshape(3, 2), actions[:, 14:16])

    send_empty(server, "boom", action_mode=None)
    assert json.loads(client.recv_multipart()[0]) == {"chunk_size": 0, "error": "boom"}
