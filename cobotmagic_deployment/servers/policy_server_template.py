#!/usr/bin/env python3
"""Template for adding a new policy model to the CobotMagic deployment.

Copy this file to ``policy_server_<model>.py`` and a config to
``configs/config_<model>.yaml`` (with a ``<model>:`` section and
``policy_backend: <model>``), then implement :class:`TemplatePolicy`.

Everything else is shared:

* Server side (``common/policy_server_runtime.py``): config loading, ``--bind``,
  ``--mock``, ``--startup-test``, ``--inference-test``, warmup, request loop and
  error replies.
* Bridge side (``bridges/ros_bridge_node.py``, no code changes needed): the
  YAML ``ros`` options select asynchronous inference, chunk filters and
  interpolation, per-step shaping, gripper handling and, for
  ``action_mode: eef_absolute``, differential IK. The ROS-independent
  components in ``cobotmagic_deployment.common`` (``ChunkPipeline``,
  ``ChunkScheduler``, ``CommandShaper``, ``EefIkCommander``,
  ``AsyncPolicyClient``...) can also be used directly by other bridges.

Try it without a model::

    python -m cobotmagic_deployment.servers.policy_server_template --mock --startup-test
    python -m cobotmagic_deployment.servers.policy_server_template --inference-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from cobotmagic_deployment.common.policy_server_runtime import (
    CONFIG_DIR, add_server_arguments, load_server_config, run_server,
)


class TemplatePolicy:
    """Replace ``predict`` with model inference.

    ``action_mode`` must match the bridge's ``ros.action_mode``:
    ``absolute`` (joint targets), ``eef_absolute`` (xyz + euler xyz + gripper
    per arm; the bridge runs IK) or ``velocity``.
    """

    action_mode = "absolute"

    def __init__(self, cfg: dict[str, Any]) -> None:
        # Load weights here, e.g. from cfg["checkpoint_path"].
        self.chunk_size = int(cfg.get("chunk_size", 16))

    def predict(self, header: dict[str, Any], images: dict[str, np.ndarray]) -> np.ndarray:
        """Return a ``(T, 14)`` chunk: left arm 7D then right arm 7D.

        ``images``: ``front``/``left``/``right`` HxWx3 RGB uint8.
        ``header``: ``task_prompt``, ``jleft``/``jright`` (7D joints incl.
        gripper), ``current_eef_left``/``right`` in EEF mode, ``episode_start``.
        Return ``(actions, vel)`` to also send a ``(T, 2)`` base velocity.
        """
        current = np.concatenate([header["jleft"][:7], header["jright"][:7]]).astype(np.float32)
        return np.repeat(current[None], self.chunk_size, axis=0)

    def warmup(self, cfg: dict[str, Any]) -> None:
        """Optional: called before serving when the config section has ``warmup: true``."""


def test_header(task_prompt: str) -> dict[str, Any]:
    """Request used by ``--inference-test``."""
    return {"task_prompt": task_prompt, "control_hz": 5.0, "jleft": [0.0] * 7, "jright": [0.0] * 7}


def main() -> None:
    parser = add_server_arguments(argparse.ArgumentParser(description=__doc__),
                                  CONFIG_DIR / "config_template.yaml")
    args = parser.parse_args()
    cfg = load_server_config(Path(args.config), "template", backends=("template",))
    run_server(args, cfg, TemplatePolicy, name="template", action_mode=TemplatePolicy.action_mode,
               test_header=test_header, per_arm_stats=False)


if __name__ == "__main__":
    main()
