"""Gymnasium registration for the project's environments.

Registering under an id keeps the training scripts decoupled from the env
class, and means any Gymnasium-compatible RL library (Stable-Baselines3 now,
skrl / RLlib later) can construct the task by name.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

__all__ = ["register_envs", "ENV_IDS"]

ENV_IDS = {
    "privileged": "VisionNav-Privileged-v0",
}

_ENTRY_POINT = "vision_nav.envs.nav_env:ProceduralNavEnv"


def register_envs() -> None:
    """Register every project env id. Safe to call more than once."""
    for obs_mode, env_id in ENV_IDS.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point=_ENTRY_POINT,
            # The env enforces its own step limit and reports the terminal
            # info dict the metrics depend on, so no TimeLimit wrapper here.
            max_episode_steps=None,
            kwargs={},
        )
