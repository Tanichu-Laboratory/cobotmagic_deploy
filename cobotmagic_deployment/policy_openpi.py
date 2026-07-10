import os
import sys

import numpy as np
import torch.nn as nn

# Load OpenPI from the external repository configured by OPENPI_REPO_PATH.
OPENPI_ROOT = os.path.expanduser(os.environ.get("OPENPI_REPO_PATH", "/workspace/project/openpi"))
OPENPI_SOURCE_ROOT = os.path.join(OPENPI_ROOT, "src")
if not os.path.isdir(OPENPI_SOURCE_ROOT):
    raise ModuleNotFoundError(
        f"External OpenPI source not found at {OPENPI_SOURCE_ROOT}. "
        "Set OPENPI_REPO_PATH to the OpenPI repository root."
    )
if OPENPI_SOURCE_ROOT not in sys.path:
    sys.path.insert(0, OPENPI_SOURCE_ROOT)

from openpi.training import config as _config  # noqa: E402
from openpi.policies import policy_config  # noqa: E402
from openpi.shared import download  # noqa: E402

DEFAULT_POLICY_CONFIG_NAME = "pi0_mobile_aloha_lora_local"
DEFAULT_CHECKPOINT_DIR = "/workspace/project/openpi/checkpoints/pi0_mobile_aloha_lora_local/mobile_aloha_lora/10000"


class Pi0Policy(nn.Module):
    def __init__(
        self,
        config_name: str = DEFAULT_POLICY_CONFIG_NAME,
        checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    ):
        super().__init__()
        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(os.path.expanduser(checkpoint_dir))

        # Create a trained policy.
        self.model = policy_config.create_trained_policy(config, checkpoint_dir)

    def forward(self, center_image, left_image, right_image, robot_state, task_prompt):
        center_image = np.asarray(center_image)
        left_image = np.asarray(left_image)
        right_image = np.asarray(right_image)
        robot_state = np.asarray(robot_state)
        model_inputs = {
            "state": robot_state,
            "images": {
                "cam_high": center_image,
                "cam_low": np.zeros_like(center_image),
                "cam_left_wrist": left_image,
                "cam_right_wrist": right_image,
            },
            "prompt": task_prompt,
        }
        a_hat = self.model.infer(model_inputs)["actions"]
        return a_hat
