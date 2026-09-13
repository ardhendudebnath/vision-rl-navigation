"""Uniform interface over the things being compared.

The whole project rests on scoring a learned policy and a classical planner
under *identical* conditions.  The cheapest way to guarantee that is to make
the evaluation loop unable to tell them apart: every candidate implements the
same two methods, and the loop never branches on which one it has.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from vision_nav.agents.classical import AStarPursuitAgent, PursuitConfig
from vision_nav.envs.nav_env import ProceduralNavEnv

__all__ = ["Actor", "RandomActor", "ClassicalActor", "SB3Actor", "build_actor"]


@runtime_checkable
class Actor(Protocol):
    """Anything that can drive the navigation environment."""

    name: str

    def reset(self, env: ProceduralNavEnv, obs: np.ndarray) -> bool:
        """Prepare for a new episode. Return ``False`` to declare failure."""
        ...

    def act(self, env: ProceduralNavEnv, obs: np.ndarray) -> np.ndarray:
        """Return an action in the env's normalised action space."""
        ...


class RandomActor:
    """Uniform random actions — the floor any result must clear."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, env: ProceduralNavEnv, obs: np.ndarray) -> bool:
        return True

    def act(self, env: ProceduralNavEnv, obs: np.ndarray) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


class ClassicalActor:
    """Adapter around :class:`AStarPursuitAgent`.

    Note that this actor reads ``env.world`` and ``env.robot.pose`` rather
    than ``obs``: the classical baseline is *defined* as having full map and
    pose access.  That privilege is the point of the comparison, and it is
    stated plainly in the report rather than hidden in the code.
    """

    name = "classical"

    def __init__(self, config: PursuitConfig | None = None, robot=None) -> None:
        self.agent = AStarPursuitAgent(config, robot)

    def reset(self, env: ProceduralNavEnv, obs: np.ndarray) -> bool:
        if self.agent.robot is not env.config.robot:
            self.agent.robot = env.config.robot
        return self.agent.start_episode(env.world, env.robot.pose)

    def act(self, env: ProceduralNavEnv, obs: np.ndarray) -> np.ndarray:
        return self.agent.act(env.robot.pose)


class SB3Actor:
    """Adapter around a Stable-Baselines3 policy."""

    def __init__(self, model, deterministic: bool = True, name: str = "rl") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name

    def reset(self, env: ProceduralNavEnv, obs: np.ndarray) -> bool:
        return True

    def act(self, env: ProceduralNavEnv, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(2)


def build_actor(kind: str, *, model_path: str | None = None, robot=None, **kwargs) -> Actor:
    """Construct an actor by name: ``random``, ``classical`` or ``rl``."""
    kind = kind.lower()
    if kind == "random":
        return RandomActor(**kwargs)
    if kind == "classical":
        return ClassicalActor(robot=robot, **kwargs)
    if kind == "rl":
        if model_path is None:
            raise ValueError("actor kind 'rl' requires model_path")
        from stable_baselines3 import PPO

        return SB3Actor(PPO.load(model_path, device="cpu"), **kwargs)
    raise ValueError(f"unknown actor kind {kind!r}; expected random|classical|rl")
