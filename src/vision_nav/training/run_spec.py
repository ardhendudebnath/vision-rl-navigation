"""Reconstruct the env configuration a trained policy expects.

A policy can only be evaluated in an environment that produces the
observation it was trained on. Three separate analysis scripts used to
rebuild that configuration independently, and each time a new env field was
added — ``obs_mode``, then the RGB camera, then ``frame_stack`` — one or more
of them silently produced the wrong environment:

- an RGB policy printed as a lidar in the results table,
- a depth policy would have been scored against lidar readings,
- a 4-frame policy was handed a 1-frame observation and crashed.

The crash was the lucky case. The others were silent. This module is the
single place that knowledge lives, so adding an env field means updating one
function rather than remembering three call sites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

__all__ = [
    "env_overrides_for_run",
    "describe_run_sensor",
    "run_seed",
    "OBSERVATION_FIELDS",
    "CONDITION_FIELDS",
]

#: Every ``NavEnvConfig`` field that changes what the policy sees, and so must
#: travel with it. ``test_run_spec_covers_every_env_field`` fails if a field is
#: added to the config and classified in neither this set nor
#: :data:`CONDITION_FIELDS`, which is what turns the fourth recurrence of this
#: bug into the last one.
OBSERVATION_FIELDS = frozenset({
    "obs_mode", "lidar", "camera", "rgb_camera", "frame_stack",
    "obs_velocity", "max_goal_distance",
})

#: Fields belonging to the *evaluation condition* rather than the policy.
#: Carrying these would let a policy drag its training task along and score
#: itself on the wrong worlds.
CONDITION_FIELDS = frozenset({
    "world", "robot", "reward", "domain_randomization", "max_episode_steps",
    "freeze_dynamic", "world_seeds", "seed_range", "deterministic_seed_order",
    "world_cache_size",
})

#: Per-mode sensor geometry that must travel with the policy. Sensor
#: *corruption* (noise, dropout) deliberately does not: that belongs to the
#: evaluation condition, not the policy.
_GEOMETRY = {
    "depth": ("camera", ("fov", "width", "max_range")),
    "rgb": ("rgb_camera", ("fov", "width", "height", "max_range")),
    "privileged": ("lidar", ("n_beams", "fov", "max_range")),
}


def _load_env(run: Path | str):
    cfg_path = Path(run)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"no config.yaml for run {run}")
    return OmegaConf.load(cfg_path).env


def env_overrides_for_run(run: Path | str) -> dict:
    """Env overrides needed to evaluate the policy trained by ``run``.

    Carries the observation mode, the geometry of the sensor that mode
    actually reads, and the frame-stack depth. The idle sensor blocks of other
    modes are deliberately omitted, so a depth run's unused lidar settings
    cannot override its camera.
    """
    env = _load_env(run)
    mode = OmegaConf.select(env, "obs_mode") or "privileged"
    spec: dict = {"obs_mode": mode}

    section, fields = _GEOMETRY[mode]
    if section in env:
        block = OmegaConf.to_container(env[section], resolve=True)
        spec[section] = {k: block[k] for k in fields if k in block}

    stack = OmegaConf.select(env, "frame_stack")
    if stack and int(stack) > 1:
        spec["frame_stack"] = int(stack)

    if bool(OmegaConf.select(env, "obs_velocity")):
        spec["obs_velocity"] = True

    # Scales the goal-distance component, so a policy trained at one value
    # reads a different number at another. Every config to date uses 20.0, so
    # carrying it changes nothing today and stops it from mattering silently.
    max_goal = OmegaConf.select(env, "max_goal_distance")
    if max_goal is not None:
        spec["max_goal_distance"] = float(max_goal)
    return spec


def describe_run_sensor(run: Path | str) -> str:
    """Short human-readable description, for tables and logs."""
    spec = env_overrides_for_run(run)
    mode = spec["obs_mode"]
    stack = spec.get("frame_stack", 1)
    suffix = f" x{stack} frames" if stack > 1 else ""

    if mode == "depth":
        cam = spec.get("camera", {})
        base = f"depth {cam.get('width')}px @ {np.degrees(cam.get('fov', 0)):.0f}deg"
    elif mode == "rgb":
        cam = spec.get("rgb_camera", {})
        base = (
            f"rgb {cam.get('width')}x{cam.get('height')}px @ "
            f"{np.degrees(cam.get('fov', 0)):.0f}deg"
        )
    else:
        base = f"lidar {spec.get('lidar', {}).get('n_beams')} beams @ 360deg"
    return base + suffix


def run_seed(run: Path | str) -> int:
    cfg_path = Path(run)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.yaml"
    return int(OmegaConf.load(cfg_path).train.seed)
