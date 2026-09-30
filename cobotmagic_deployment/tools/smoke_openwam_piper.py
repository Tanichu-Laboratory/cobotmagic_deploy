"""Send synthetic observations to the ZMQ policy server, without ROS."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml
import zmq

from cobotmagic_deployment.servers.policy_server_openwam_piper import test_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "configs" / "config_openwam_piper.yaml")
    parser.add_argument("--requests", type=int, default=2)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    checkpoint_cfg = yaml.safe_load((Path(cfg["openwam"]["checkpoint_path"]) / "config.yaml").read_text())
    expected_steps = int(checkpoint_cfg["dataloader"]["num_frames"]) - 1
    header, images = test_observation(cfg)
    frames = [json.dumps(header).encode()]
    for key in ("front", "left", "right"):
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(images[key], cv2.COLOR_RGB2BGR))
        assert ok
        frames.append(jpeg.tobytes())
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 180000)
    sock.setsockopt(zmq.SNDTIMEO, 10000)
    sock.connect(cfg["zmq"]["client_connect"])
    try:
        for index in range(args.requests):
            start = time.monotonic()
            sock.send_multipart(frames)
            reply = sock.recv_multipart()
            meta = json.loads(reply[0])
            if "error" in meta:
                raise RuntimeError(meta["error"])
            assert meta["action_mode"] == "eef_absolute" and len(reply) == 3, meta
            steps = meta["chunk_size"]
            assert steps == expected_steps, meta
            left = np.frombuffer(reply[1], dtype=np.float32).reshape(steps, 7)
            right = np.frombuffer(reply[2], dtype=np.float32).reshape(steps, 7)
            assert np.isfinite(left).all() and np.isfinite(right).all()
            for i, arm in enumerate((left, right)):
                endpoints = cfg["openwam"]["gripper"]
                lo, hi = sorted((endpoints["closed"][i], endpoints["open"][i]))
                assert ((arm[:, 6] >= lo - 1e-6) & (arm[:, 6] <= hi + 1e-6)).all()
            print(json.dumps({"request": index, "seconds": time.monotonic() - start,
                              "shape": [steps, 14], "finite": True,
                              "gripper_range": [float(min(left[:, 6].min(), right[:, 6].min())),
                                                float(max(left[:, 6].max(), right[:, 6].max()))]}), flush=True)
    finally:
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
