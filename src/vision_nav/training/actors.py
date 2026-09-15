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
    """Adapter around a Stable-Baselines3 policy.

    Carries the policy's recurrent state across steps and clears it between
    episodes. For a feed-forward policy this is inert -- ``predict`` returns
    ``None`` and nothing changes -- so it is done unconditionally rather than
    behind a branch on the model type.

    Doing it unconditionally matters. A recurrent policy evaluated without
    carrying state sees every step as the first, which destroys exactly the
    memory the policy was trained to use, and the result would read as
    "recurrence does not help" rather than as a broken evaluation path. This
    adapter is the single seam through which both training-time validation and
    every analysis script evaluate, so getting it wrong here would be silent
    and everywhere.
    """

    def __init__(self, model, deterministic: bool = True, name: str = "rl") -> None:
        self.model = model
        self.deterministic = deterministic
        self.name = name
        self._state = None
        self._episode_start = True

    def reset(self, env: ProceduralNavEnv, obs: np.ndarray) -> bool:
        self._state = None
        self._episode_start = True
        return True

    def act(self, env: ProceduralNavEnv, obs: np.ndarray) -> np.ndarray:
        action, self._state = self.model.predict(
            obs,
            state=self._state,
            episode_start=np.array([self._episode_start]),
            deterministic=self.deterministic,
        )
        self._episode_start = False
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

        # PPO.load does NOT reject a RecurrentPPO checkpoint: it happily
        # returns a PPO whose policy is a RecurrentActorCriticPolicy. That
        # mostly works, because predict() delegates to the policy, but it is
        # accidental rather than intended, so detect the recurrent policy and
        # reload under the class that owns it. sb3-contrib is an optional
        # extra, imported only once a checkpoint is known to need it.
        model = PPO.load(model_path, device="cpu")
        if type(model.policy).__name__.startswith("Recurrent"):
            from sb3_contrib import RecurrentPPO

            model = RecurrentPPO.load(model_path, device="cpu")
        return SB3Actor(model, **kwargs)
    raise ValueError(f"unknown actor kind {kind!r}; expected random|classical|rl")
