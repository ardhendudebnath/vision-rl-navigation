"""Construct environments from configuration.

All three consumers — training, evaluation and the classical baseline — build
their environments through here, so a config change cannot silently apply to
one and not the others.

Stable-Baselines3 is imported lazily, inside the two functions that need it.
:func:`build_env_config` is pure config and is called from places where the RL
stack is absent — notably the ROS 2 bridge, whose conda env carries Nav2 but
not torch. An eager import would make the Nav2 baseline impossible to run
without installing a deep-learning framework it never uses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import gymnasium as gym
import numpy as np

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv, RewardConfig
from vision_nav.envs.randomization import DomainRandomization
from vision_nav.envs.rgb_camera import RGBCameraConfig
from vision_nav.envs.robot import RobotConfig
from vision_nav.envs.sensors import CameraConfig, LidarConfig
from vision_nav.envs.splits import shifted_config, split_seeds
from vision_nav.envs.world import WorldConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from stable_baselines3.common.vec_env import VecEnv

__all__ = ["build_env_config", "make_env", "make_vec_env"]

_TUPLE_FIELDS = {
    "n_circles",
    "circle_radius",
    "n_boxes",
    "box_size",
    "seed_range",
    "arena",
    "start_goal_fraction",
    "n_dynamic",
    "dynamic_radius",
    "dynamic_amplitude",
    "dynamic_speed",
    "wall_rgb",
    "circle_rgb",
    "box_rgb",
    "ceiling_rgb",
    "floor_rgb",
}


def _section(cls, values: Mapping[str, Any] | None):
    """Build a config dataclass, coercing YAML lists into tuples."""
    if not values:
        return cls()
    kwargs = {
        k: (tuple(v) if k in _TUPLE_FIELDS and isinstance(v, (list, tuple)) else v)
        for k, v in dict(values).items()
    }
    unknown = set(kwargs) - {f for f in cls.__dataclass_fields__}
    if unknown:
        raise TypeError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**kwargs)


def build_env_config(
    cfg: Mapping[str, Any],
    *,
    split: str | None = None,
    shift: str | None = None,
    n_worlds: int | None = None,
) -> NavEnvConfig:
    """Turn a config mapping into a :class:`NavEnvConfig`.

    Parameters
    ----------
    cfg:
        Mapping with optional ``world`` / ``robot`` / ``lidar`` / ``reward``
        sub-sections plus top-level :class:`NavEnvConfig` fields.
    split:
        Name from :data:`~vision_nav.envs.splits.SEED_BANDS`.  When given, it
        overrides ``world_seeds`` and (for non-train splits) forces
        deterministic episode ordering so every policy is scored on an
        identical sequence.
    shift:
        Named distribution shift applied to the world config.
    n_worlds:
        Truncate the split to this many seeds.
    """
    cfg = dict(cfg)
    world = _section(WorldConfig, cfg.pop("world", None))
    if shift:
        world = shifted_config(world, shift)

    env_kwargs: dict[str, Any] = {
        "world": world,
        "robot": _section(RobotConfig, cfg.pop("robot", None)),
        "lidar": _section(LidarConfig, cfg.pop("lidar", None)),
        "camera": _section(CameraConfig, cfg.pop("camera", None)),
        "rgb_camera": _section(RGBCameraConfig, cfg.pop("rgb_camera", None)),
        "reward": _section(RewardConfig, cfg.pop("reward", None)),
        "domain_randomization": _section(
            DomainRandomization, cfg.pop("domain_randomization", None)
        ),
    }

    if shift and env_kwargs["domain_randomization"].enabled:
        # A named shift defines an evaluation condition; randomisation would
        # overwrite the very fields the shift sets, silently evaluating
        # something other than the named condition.
        raise ValueError(
            f"cannot combine shift={shift!r} with domain randomisation: the "
            "randomiser would overwrite the shift's world parameters. Evaluate "
            "with domain_randomization.enabled=false."
        )

    for key in ("seed_range", "world_seeds"):
        if key in cfg and isinstance(cfg[key], (list, tuple)):
            cfg[key] = tuple(cfg[key]) if key == "seed_range" else list(cfg[key])

    unknown = set(cfg) - set(NavEnvConfig.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown NavEnvConfig field(s): {sorted(unknown)}")
    env_kwargs.update(cfg)

    if split is not None:
        env_kwargs["world_seeds"] = split_seeds(split, n_worlds)
        env_kwargs["deterministic_seed_order"] = split != "train"

    return NavEnvConfig(**env_kwargs)


def make_env(
    env_config: NavEnvConfig,
    seed: int | None = None,
    render_mode: str | None = None,
    monitor: bool = True,
):
    """Create a single, optionally Monitor-wrapped, environment."""
    from stable_baselines3.common.monitor import Monitor

    env: gym.Env = ProceduralNavEnv(env_config, render_mode=render_mode)
    if monitor:
        # ``info_keywords`` propagates per-episode outcomes into the Monitor
        # record, which is what the training-time metric callback reads.
        env = Monitor(env, info_keywords=("is_success", "collision"))
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_vec_env(
    env_config: NavEnvConfig,
    n_envs: int = 8,
    seed: int = 0,
    subprocess: bool = False,
    world_seed_shards: bool = True,
) -> VecEnv:
    """Create a vectorised environment for training.

    Parameters
    ----------
    world_seed_shards:
        Give each worker a disjoint slice of the world-seed pool.  Without
        this every worker draws from the same pool and, early in training when
        episodes are short and correlated, the batch is dominated by a handful
        of repeated scenes.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    seeds: Sequence[int] | None = env_config.world_seeds

    def _factory(rank: int):
        def _init():
            cfg = env_config
            if world_seed_shards and seeds is not None and len(seeds) >= n_envs:
                shard = list(np.array_split(np.asarray(seeds), n_envs)[rank])
                cfg = _replace_seeds(env_config, [int(s) for s in shard])
            return make_env(cfg, seed=seed + rank)

        return _init

    cls = SubprocVecEnv if subprocess and n_envs > 1 else DummyVecEnv
    return cls([_factory(i) for i in range(n_envs)])


def _replace_seeds(config: NavEnvConfig, seeds: list[int]) -> NavEnvConfig:
    from dataclasses import replace

    return replace(config, world_seeds=seeds)
