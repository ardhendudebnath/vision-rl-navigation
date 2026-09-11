"""Per-episode domain randomisation over world configuration.

Motivation: the nominal-trained policy collapsed under the `dense` and
`narrow` shifts (success 0.96 -> 0.64 / 0.60) while the classical planner
degraded only to 0.89 / 0.85.  The open question is whether that is
*out-of-distribution brittleness* or a real ceiling on reactive control.
Training across a world distribution wide enough to contain the shifts is the
direct test.

Design constraint that shapes everything here: **the world seed must remain
the sole determinant of the world.**  Evaluation replays worlds by seed, and
the env caches generated worlds by seed, so a randomisation that consulted
episode order or wall-clock state would silently break both.  Every parameter
below is therefore drawn from a generator keyed on the world seed alone.

The sampling RNG is decorrelated from the obstacle-placement RNG (which
``generate_world`` keys on the same seed) so that, say, "many obstacles" does
not end up permanently coupled to one particular layout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vision_nav.envs.world import WorldConfig

__all__ = ["DomainRandomization"]

#: Mixed into the seed so config sampling and obstacle placement do not share
#: a random stream.
_STREAM_OFFSET = 0x9E3779B9


@dataclass
class DomainRandomization:
    """Sampling ranges applied to :class:`WorldConfig` once per world.

    ``None`` leaves the corresponding base-config field untouched.  Robot
    radius, goal tolerance and grid resolution are deliberately *not*
    randomisable: they describe fixed hardware and the planner's
    discretisation, not the environment.
    """

    enabled: bool = False

    #: Arena side length, in metres. Width and height are sampled separately.
    arena: tuple[float, float] | None = None
    #: Start-goal separation as a fraction of the arena's smaller side.
    #: Expressed as a fraction rather than an absolute distance so a small
    #: arena can never be paired with an unsatisfiable separation.
    start_goal_fraction: tuple[float, float] | None = None

    #: Obstacle count and size ranges, passed straight through to the world
    #: generator (which samples within them per seed).
    n_circles: tuple[int, int] | None = None
    circle_radius: tuple[float, float] | None = None
    n_boxes: tuple[int, int] | None = None
    box_size: tuple[float, float] | None = None

    def sample(self, base: WorldConfig, seed: int) -> WorldConfig:
        """Return a :class:`WorldConfig` for ``seed``. Deterministic in seed."""
        if not self.enabled:
            return base

        rng = np.random.default_rng((int(seed) + _STREAM_OFFSET) & 0xFFFFFFFF)
        overrides: dict = {}

        if self.arena is not None:
            lo, hi = self.arena
            overrides["width"] = float(rng.uniform(lo, hi))
            overrides["height"] = float(rng.uniform(lo, hi))

        for field in ("n_circles", "n_boxes"):
            span = getattr(self, field)
            if span is not None:
                overrides[field] = (int(span[0]), int(span[1]))

        for field in ("circle_radius", "box_size"):
            span = getattr(self, field)
            if span is not None:
                overrides[field] = (float(span[0]), float(span[1]))

        if self.start_goal_fraction is not None:
            width = overrides.get("width", base.width)
            height = overrides.get("height", base.height)
            frac = float(rng.uniform(*self.start_goal_fraction))
            overrides["min_start_goal_dist"] = frac * min(width, height)

        return replace(base, **overrides)

    def describe(self) -> str:
        """One-line summary for logs and the report."""
        if not self.enabled:
            return "domain randomisation: off"
        parts = [
            f"{name}={getattr(self, name)}"
            for name in (
                "arena",
                "start_goal_fraction",
                "n_circles",
                "circle_radius",
                "n_boxes",
                "box_size",
            )
            if getattr(self, name) is not None
        ]
        return "domain randomisation: " + ", ".join(parts)
