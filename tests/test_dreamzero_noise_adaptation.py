from pathlib import Path
import unittest

from cobotmagic_deployment.common.dreamzero_noise_adaptation import (
    parse_noise_adaptation_settings,
)


class NoiseAdaptationSettingsTest(unittest.TestCase):
    def test_disabled_setting_explicitly_disables_model_adapter(self) -> None:
        settings = parse_noise_adaptation_settings({"noise_adaptation": False})
        self.assertFalse(settings.enabled)
        self.assertEqual(
            settings.model_overrides(),
            [
                "action_head_cfg.config.pb_adapt_enabled=false",
                "action_head_cfg.config.action_adapt_enabled=false",
                "action_head_cfg.config.unconstrained_noise_adaptation=false",
            ],
        )

    def test_enabled_setting_builds_complete_model_overrides(self) -> None:
        settings = parse_noise_adaptation_settings(
            {
                "noise_adaptation": True,
                "num_inference_steps": 1,
                "noise_adaptation_config": {
                    "video_final_noise": 0.8,
                    "tau_v": 0.2,
                    "M": 3,
                    "eta": 0.002,
                    "c_clip": 20.0,
                    "deadband": 0.1,
                    "log_path": "../logs/pb.jsonl",
                },
            },
            config_dir=Path("/tmp/configs"),
        )
        overrides = settings.model_overrides()
        self.assertIn("action_head_cfg.config.pb_adapt_enabled=true", overrides)
        self.assertIn("action_head_cfg.config.decouple_inference_noise=true", overrides)
        self.assertIn("action_head_cfg.config.pb_adapt_config.M=3", overrides)
        self.assertIn("action_head_cfg.config.pb_adapt_config.eta=0.002", overrides)
        self.assertIn(
            'action_head_cfg.config.pb_adapt_log_path="/tmp/logs/pb.jsonl"',
            overrides,
        )

    def test_worker_can_omit_shared_log_path(self) -> None:
        settings = parse_noise_adaptation_settings(
            {
                "noise_adaptation": True,
                "noise_adaptation_config": {"log_path": "/tmp/pb.jsonl"},
            }
        )
        self.assertFalse(
            any("pb_adapt_log_path" in item for item in settings.model_overrides(
                include_log_path=False
            ))
        )

    def test_rejects_schedule_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "tau_v must equal"):
            parse_noise_adaptation_settings(
                {
                    "noise_adaptation": True,
                    "noise_adaptation_config": {
                        "video_final_noise": 0.8,
                        "tau_v": 0.3,
                    },
                }
            )

    def test_rejects_multistep_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_inference_steps=1"):
            parse_noise_adaptation_settings(
                {
                    "noise_adaptation": True,
                    "num_inference_steps": 4,
                }
            )


if __name__ == "__main__":
    unittest.main()
