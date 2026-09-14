"""Standard navigation evaluation metrics.

These are the metrics the embodied-navigation literature reports (and that
Habitat / the PointNav benchmarks standardised), so results computed here are
directly comparable to published numbers — which is the point of using them
rather than inventing project-specific scores.

Reference for SPL: Anderson et al., "On Evaluation of Embodied Navigation
Agents" (2018), arXiv:1807.06757.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["EpisodeResult", "NavigationMetrics", "aggregate"]


@dataclass
class EpisodeResult:
    """Outcome of one evaluation episode."""

    world_seed: int
    success: bool
    collision: bool
    steps: int
    path_length: float
    shortest_path_length: float
    final_goal_distance: float

    @property
    def timeout(self) -> bool:
        """Episode ended by hitting the step limit rather than resolving."""
        return not self.success and not self.collision

    @property
    def spl_term(self) -> float:
        """This episode's contribution to SPL, in ``[0, 1]``."""
        if not self.success:
            return 0.0
        l_star = self.shortest_path_length
        if l_star <= 0.0:
            return 1.0
        return float(l_star / max(self.path_length, l_star))

    @classmethod
    def from_info(cls, info: dict) -> EpisodeResult:
        """Build from a terminal ``info`` dict emitted by the environment."""
        return cls(
            world_seed=int(info["world_seed"]),
            success=bool(info.get("is_success", False)),
            collision=bool(info.get("collision", False)),
            steps=int(info["steps"]),
            path_length=float(info["path_length"]),
            shortest_path_length=float(info["shortest_path_length"]),
            final_goal_distance=float(info["goal_distance"]),
        )


@dataclass
class NavigationMetrics:
    """Aggregate results over an evaluation set."""

    n_episodes: int
    success_rate: float
    spl: float
    collision_rate: float
    timeout_rate: float
    #: Mean episode length **over successful episodes only** — averaging over
    #: failures too would let a policy look fast by crashing early.
    mean_steps_to_goal: float
    mean_path_length: float
    #: Mean of ``shortest_path / actual_path`` over successes; 1.0 is optimal.
    mean_path_efficiency: float
    mean_final_goal_distance: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def as_table(self, title: str = "Navigation metrics") -> str:
        """Render as a Markdown table, ready to paste into the report."""
        rows = [
            ("Episodes", f"{self.n_episodes}"),
            ("Success rate", f"{self.success_rate:.3f}"),
            ("SPL", f"{self.spl:.3f}"),
            ("Collision rate", f"{self.collision_rate:.3f}"),
            ("Timeout rate", f"{self.timeout_rate:.3f}"),
            ("Mean steps to goal", f"{self.mean_steps_to_goal:.1f}"),
            ("Mean path length (m)", f"{self.mean_path_length:.2f}"),
            ("Mean path efficiency", f"{self.mean_path_efficiency:.3f}"),
            ("Mean final goal dist (m)", f"{self.mean_final_goal_distance:.2f}"),
        ]
        width = max(len(k) for k, _ in rows)
        lines = [f"### {title}", "", f"| {'Metric':<{width}} | Value |", f"|{'-' * (width + 2)}|-------|"]
        lines += [f"| {k:<{width}} | {v} |" for k, v in rows]
        return "\n".join(lines)


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if len(values) else default


def aggregate(episodes: Iterable[EpisodeResult]) -> NavigationMetrics:
    """Reduce per-episode results to the headline metrics."""
    eps = list(episodes)
    if not eps:
        raise ValueError("cannot aggregate an empty set of episodes")

    n = len(eps)
    successes = [e for e in eps if e.success]

    return NavigationMetrics(
        n_episodes=n,
        success_rate=len(successes) / n,
        spl=float(np.mean([e.spl_term for e in eps])),
        collision_rate=sum(e.collision for e in eps) / n,
        timeout_rate=sum(e.timeout for e in eps) / n,
        mean_steps_to_goal=_mean([e.steps for e in successes]),
        mean_path_length=_mean([e.path_length for e in eps]),
        mean_path_efficiency=_mean([e.spl_term for e in successes]),
        mean_final_goal_distance=_mean([e.final_goal_distance for e in eps]),
    )
