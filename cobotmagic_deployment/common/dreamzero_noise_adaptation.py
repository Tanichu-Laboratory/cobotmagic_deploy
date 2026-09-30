"""Configuration bridge for DreamZero's online initial-noise adapter."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class NoiseAdaptationSettings:
    enabled: bool
    video_final_noise: float = 0.8
    tau_v: float = 0.2
    window_size: int = 4
    eta: float = 0.001
    lam: float = 0.1
    c_clip: float = 100.0
    deadband: float = 0.0
    coh_min: float = 0.0
    max_delta_norm: float | None = 0.25
    max_error_growth_windows: int = 2
    reset_on_episode: bool = True
    log_path: str | None = None
    action_noise_eval_seed: int | None = None

    def model_overrides(self, *, include_log_path: bool = True) -> list[str]:
        """Return the Hydra dot-list needed by ``GrootSimPolicy``."""
        if not self.enabled:
            # Keep experimental adapters completely out of the normal policy.
            return [
                "action_head_cfg.config.pb_adapt_enabled=false",
                "action_head_cfg.config.action_adapt_enabled=false",
                "action_head_cfg.config.unconstrained_noise_adaptation=false",
            ]

        prefix = "action_head_cfg.config"
        overrides = [
            f"{prefix}.pb_adapt_enabled=true",
            f"{prefix}.decouple_inference_noise=true",
            f"{prefix}.video_inference_final_noise={self.video_final_noise:.17g}",
            f"{prefix}.pb_adapt_config.tau_v={self.tau_v:.17g}",
            f"{prefix}.pb_adapt_config.M={self.window_size}",
            f"{prefix}.pb_adapt_config.eta={self.eta:.17g}",
            f"{prefix}.pb_adapt_config.lam={self.lam:.17g}",
            f"{prefix}.pb_adapt_config.c_clip={self.c_clip:.17g}",
            f"{prefix}.pb_adapt_config.deadband={self.deadband:.17g}",
            f"{prefix}.pb_adapt_config.coh_min={self.coh_min:.17g}",
            (
                f"{prefix}.pb_adapt_config.reset_on_episode="
                f"{str(self.reset_on_episode).lower()}"
            ),
            (
                f"{prefix}.pb_adapt_config.max_error_growth_windows="
                f"{self.max_error_growth_windows}"
            ),
        ]
        if self.max_delta_norm is not None:
            overrides.append(
                f"{prefix}.pb_adapt_config.max_delta_norm={self.max_delta_norm:.17g}"
            )
        if include_log_path and self.log_path is not None:
            overrides.append(
                f"{prefix}.pb_adapt_log_path={json.dumps(self.log_path)}"
            )
        if self.action_noise_eval_seed is not None:
            overrides.append(
                f"{prefix}.action_noise_eval_seed={self.action_noise_eval_seed}"
            )
        return overrides


def parse_noise_adaptation_settings(
    cfg: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
) -> NoiseAdaptationSettings:
    """Validate the deployment YAML and derive DreamZero adapter settings."""
    enabled = _as_bool(cfg.get("noise_adaptation", False), "noise_adaptation")
    raw_value = cfg.get("noise_adaptation_config", {})
    if raw_value is None:
        raw_value = {}
    if not isinstance(raw_value, Mapping):
        raise ValueError("noise_adaptation_config must be a mapping")
    raw = dict(raw_value)

    allowed = {
        "video_final_noise",
        "tau_v",
        "M",
        "eta",
        "lam",
        "c_clip",
        "deadband",
        "coh_min",
        "max_delta_norm",
        "max_error_growth_windows",
        "reset_on_episode",
        "log_path",
        "action_noise_eval_seed",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown noise_adaptation_config keys: {unknown}")

    video_final_noise = _finite_float(
        raw.get("video_final_noise", 0.8), "video_final_noise"
    )
    tau_v = _finite_float(raw.get("tau_v", 1.0 - video_final_noise), "tau_v")
    if not 0.0 < video_final_noise < 1.0:
        raise ValueError("video_final_noise must be in (0, 1)")
    if not 0.0 < tau_v < 1.0:
        raise ValueError("tau_v must be in (0, 1)")
    if abs(tau_v - (1.0 - video_final_noise)) > 1e-6:
        raise ValueError("tau_v must equal 1 - video_final_noise")

    window_size = int(raw.get("M", 4))
    eta = _finite_float(raw.get("eta", 0.001), "eta")
    lam = _finite_float(raw.get("lam", 0.1), "lam")
    c_clip = _finite_float(raw.get("c_clip", 100.0), "c_clip")
    deadband = _finite_float(raw.get("deadband", 0.0), "deadband")
    coh_min = _finite_float(raw.get("coh_min", 0.0), "coh_min")
    max_error_growth_windows = int(raw.get("max_error_growth_windows", 2))
    reset_on_episode = _as_bool(
        raw.get("reset_on_episode", True),
        "noise_adaptation_config.reset_on_episode",
    )

    max_delta_value = raw.get("max_delta_norm", 0.25)
    max_delta_norm = (
        None
        if max_delta_value is None
        else _finite_float(max_delta_value, "max_delta_norm")
    )
    action_seed_value = raw.get("action_noise_eval_seed")
    action_noise_eval_seed = (
        None if action_seed_value is None else int(action_seed_value)
    )

    if window_size <= 0:
        raise ValueError("M must be positive")
    if eta <= 0.0:
        raise ValueError("eta must be positive so adaptation can update the noise")
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    if c_clip <= 0.0:
        raise ValueError("c_clip must be positive")
    if deadband < 0.0:
        raise ValueError("deadband must be non-negative")
    if not -1.0 <= coh_min <= 1.0:
        raise ValueError("coh_min must be in [-1, 1]")
    if max_delta_norm is not None and max_delta_norm <= 0.0:
        raise ValueError("max_delta_norm must be positive or null")
    if max_error_growth_windows < 0:
        raise ValueError("max_error_growth_windows must be non-negative")

    log_path_value = raw.get("log_path")
    log_path = None
    if log_path_value not in (None, ""):
        path = Path(str(log_path_value)).expanduser()
        if not path.is_absolute() and config_dir is not None:
            path = config_dir / path
        log_path = str(path.resolve())

    num_inference_steps = int(cfg.get("num_inference_steps", 1))
    if enabled and num_inference_steps != 1:
        raise ValueError(
            "noise_adaptation=true requires dreamzero.num_inference_steps=1"
        )

    return NoiseAdaptationSettings(
        enabled=enabled,
        video_final_noise=video_final_noise,
        tau_v=tau_v,
        window_size=window_size,
        eta=eta,
        lam=lam,
        c_clip=c_clip,
        deadband=deadband,
        coh_min=coh_min,
        max_delta_norm=max_delta_norm,
        max_error_growth_windows=max_error_growth_windows,
        reset_on_episode=reset_on_episode,
        log_path=log_path,
        action_noise_eval_seed=action_noise_eval_seed,
    )
