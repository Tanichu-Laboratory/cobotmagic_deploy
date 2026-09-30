"""OpenWAM Piper checkpoints served with the CobotMagic ZMQ protocol."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from cobotmagic_deployment.common.openwam_piper import actions_to_bridge, request_state, validate_checkpoint, gripper_action_open_normalized
from cobotmagic_deployment.common.policy_server_protocol import bind_server, recv_packet, send_actions, send_empty

LOG = logging.getLogger("openwam_piper")


class OpenWAMPiperPolicy:
    def __init__(self, cfg, mock=False):
        self.cfg = cfg
        self.options = cfg["openwam"]
        self.mock = mock
        gain = gripper_action_open_normalized(self.options["gripper"])
        hysteresis = cfg["ros"].get("gripper_hysteresis", {})
        if hysteresis.get("enabled", False):
            if gain != 1.0:
                raise ValueError("gripper_hysteresis requires action_open_normalized=1.0")
            for endpoint in ("closed", "open"):
                if not np.array_equal(hysteresis[endpoint], self.options["gripper"][endpoint]):
                    raise ValueError("server and gripper_hysteresis endpoints must match")
        if cfg["ros"]["action_mode"] != "eef_absolute":
            raise ValueError("OpenWAM Piper requires eef_absolute")
        checkpoint = Path(self.options["checkpoint_path"])
        training = yaml.safe_load((checkpoint / "config.yaml").read_text())
        validate_checkpoint(training)
        self.model_horizon = int(training["dataloader"]["num_frames"]) - 1
        if self.model_horizon < 1:
            raise ValueError("checkpoint must define a positive action horizon")
        if mock:
            return
        sys.path.insert(0, self.options["repo_path"])
        from omegaconf import OmegaConf
        from openwam.deploy.server import build_server_from_config
        from openwam.deploy.obs_preprocess import ObsPreprocessor

        deploy = OmegaConf.load(Path(self.options["repo_path"]) / "configs/deploy.yaml")
        deploy.optimization.compile.enabled = bool(self.options["compile"])
        deploy.optimization.decode_video = False
        deploy.inference.denoise_steps = int(self.options["denoise_steps"])
        deploy.inference.inference_mode = "sync"
        self.server = build_server_from_config(deploy, str(checkpoint),
                                              device=self.options["device"],
                                              ckpt_name=self.options["checkpoint_name"])
        self.preprocess = ObsPreprocessor.from_cfg(self.server.cfg, self.server.engine)

    def predict(self, header, images):
        state = request_state(header, self.options["gripper"])
        if self.mock:
            return actions_to_bridge(np.repeat(state[None], self.model_horizon, axis=0), self.options["gripper"])
        from PIL import Image
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference

        prompt = str(header.get("task_prompt", self.cfg["task_prompt"])).strip()
        if not prompt:
            raise ValueError("task_prompt must not be empty")
        obs = self.preprocess.preprocess({
            "images": {name: Image.fromarray(images[key]) for name, key in (
                ("head_camera", "front"), ("left_wrist_camera", "left"), ("right_wrist_camera", "right"))},
            "prompt": format_prompt_for_inference(prompt), "state": state,
        })
        # The ROS bridge owns receding-horizon execution. Generate exactly one
        # chunk from this observation; do not repeatedly pop a per-step WS buffer.
        result = self.server.engine.generate({"observation": obs, "first_frame_image": [obs["image"]],
                                              "prompt": obs["prompt"], "proprio": state})
        actions = result["actions"]
        if hasattr(actions, "detach"):
            actions = actions.detach().float().cpu().numpy()
        # Preserve every generated step. Only the ROS bridge selects how many
        # to execute with open_loop_steps; that setting is not a server input.
        return actions_to_bridge(actions, self.options["gripper"])


def test_observation(cfg):
    opened = cfg["openwam"]["gripper"]["open"]
    header = {"task_prompt": cfg["task_prompt"], "control_hz": cfg["ros"]["rate_hz"],
              "current_eef_left": [0.20, 0, 0.20, 0, 0.5, 0, opened[0]],
              "current_eef_right": [0.20, 0, 0.20, 0, 0.5, 0, opened[1]]}
    images = {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in ("front", "left", "right")}
    return header, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="cobotmagic_deployment/configs/config_openwam_piper.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--startup-test", action="store_true", help="infer once without ROS or a listening socket, then exit")
    parser.add_argument("--denoise-steps", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.denoise_steps is not None:
        cfg["openwam"]["denoise_steps"] = args.denoise_steps
    # Reserve the endpoint before loading 25GB of weights. Keeping the bound
    # socket avoids a check-then-bind race with another server starting up.
    sock = None
    if not args.startup_test:
        import zmq
        try:
            sock, kind = bind_server(cfg["zmq"]["server_bind"], cfg["zmq"]["socket_type"])
        except zmq.ZMQError as exc:
            if exc.errno == zmq.EADDRINUSE:
                parser.exit(2, f"Port already in use: {cfg['zmq']['server_bind']}. "
                            "Stop the existing policy server before restarting. "
                            "Model loading was skipped.\n")
            raise
    try:
        policy = OpenWAMPiperPolicy(cfg, mock=args.mock)
        if args.startup_test or cfg["openwam"].get("warmup", True):
            start = time.monotonic()
            actions = policy.predict(*test_observation(cfg))
            print(json.dumps({"mock": args.mock, "shape": list(actions.shape),
                              "finite": bool(np.isfinite(actions).all()), "seconds": time.monotonic() - start}), flush=True)
        if args.startup_test:
            return
        LOG.info("READY %s %s", kind, cfg["zmq"]["server_bind"])
        while True:
            try:
                header, images = recv_packet(sock)
                actions = policy.predict(header, images)
                send_actions(sock, actions, float(cfg["ros"]["rate_hz"]), action_mode="eef_absolute")
            except Exception as exc:
                LOG.exception("request failed")
                send_empty(sock, str(exc), action_mode="eef_absolute")
    except KeyboardInterrupt:
        pass
    finally:
        if sock is not None:
            sock.close(linger=0)


if __name__ == "__main__":
    main()
