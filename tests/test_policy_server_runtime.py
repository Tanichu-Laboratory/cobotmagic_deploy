import argparse
import json
import threading

import numpy as np
import pytest
import zmq

from cobotmagic_deployment.common.policy_server_protocol import encode_jpeg
from cobotmagic_deployment.common.policy_server_runtime import (
    EchoStatePolicy, add_server_arguments, load_server_config, serve,
)


def test_load_server_config(tmp_path):
    path = tmp_path / 'c.yaml'
    path.write_text('policy_backend: demo\ntask_prompt: hi\ndemo: {x: 1}\nzmq: {server_bind: "tcp://*:1"}\n')
    cfg = load_server_config(path, 'demo', backends=('demo',))
    assert cfg['x'] == 1 and cfg['task_prompt'] == 'hi' and cfg['server_bind'] == 'tcp://*:1'
    with pytest.raises(ValueError):
        load_server_config(path, 'demo', backends=('other',))


def test_add_server_arguments_defaults():
    args = add_server_arguments(argparse.ArgumentParser(), 'x.yaml').parse_args(['--mock'])
    assert args.mock and not args.startup_test and args.config == 'x.yaml' and args.bind == ''


class VelPolicy:
    def predict(self, header, images):
        if header.get('fail'):
            raise RuntimeError('boom')
        assert images['front'].shape == (8, 8, 3)
        return np.ones((2, 14), np.float32), np.full((2, 2), 0.5, np.float32)


def test_serve_replies_with_actions_and_errors():
    ctx = zmq.Context.instance()
    server = ctx.socket(zmq.REP)
    port = server.bind_to_random_port('tcp://127.0.0.1')
    thread = threading.Thread(target=serve, args=(VelPolicy(), server),
                              kwargs=dict(name='t', action_mode='absolute', verbose=False), daemon=True)
    thread.start()
    client = ctx.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 5000)
    client.connect(f'tcp://127.0.0.1:{port}')
    img = encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    client.send_multipart([json.dumps({'control_hz': 7}).encode(), img, img, img])
    frames = client.recv_multipart()
    header = json.loads(frames[0])
    assert header['chunk_size'] == 2 and header['has_vel'] and header['control_hz'] == 7 and len(frames) == 4
    client.send_multipart([json.dumps({'fail': True}).encode(), img, img, img])
    assert json.loads(client.recv_multipart()[0]) == {'chunk_size': 0, 'error': 'boom', 'action_mode': 'absolute'}
    client.close(0)


def test_echo_state_policy_prefers_eef():
    policy = EchoStatePolicy(chunk_size=3)
    out = policy.predict({'jleft': [0] * 7, 'jright': [0] * 7,
                          'current_eef_left': [1] * 7, 'current_eef_right': [2] * 7}, {})
    assert out.shape == (3, 14) and out[0, 0] == 1 and out[0, 7] == 2
