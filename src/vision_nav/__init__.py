"""Vision + RL autonomous navigation — a comparative study of learned and
classical navigation policies.

The package is organised by role rather than by experiment, so each stage of
the project adds modules instead of editing the previous stage's:

- :mod:`vision_nav.envs`      — the shared navigation task and its worlds
- :mod:`vision_nav.planning`  — classical search (A*, geodesic fields)
- :mod:`vision_nav.agents`    — non-learned controllers (the baseline)
- :mod:`vision_nav.metrics`   — success rate, SPL, collisions, efficiency
- :mod:`vision_nav.training`  — RL training and the evaluation harness
- :mod:`vision_nav.viz`       — rendering for figures and demo videos
"""

from vision_nav.envs.registration import register_envs

__version__ = "0.1.0"

register_envs()

__all__ = ["__version__", "register_envs"]
