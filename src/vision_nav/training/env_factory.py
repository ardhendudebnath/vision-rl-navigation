"""Construct environments from configuration.

All three consumers — training, evaluation and the classical baseline — build
their environments through here, so a config change cannot silently apply to
one and not the others.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv, RewardConfig
from vision_nav.envs.robot import RobotConfig
from vision_nav.envs.sensors import LidarConfig
from vision_nav.envs.splits import shifted_config, split_seeds
from vision_nav.envs.world import WorldConfig

__all__ = ["build_env_config", "make_env", "make_vec_env"]

_TUPLE_FIELDS = {
    "n_circles",
    "circle_radius",
    "n_boxes",
    "box_size",
    "seed_range",
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
        "reward": _section(RewardConfig, cfg.pop("reward", None)),
    }

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
