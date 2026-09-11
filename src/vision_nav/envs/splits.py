"""Canonical train / validation / test world splits.

Every number in the final report should be traceable to one of these splits.
The seed bands are far apart and never overlap, so there is no way to
accidentally evaluate on a world the policy trained in — the single most
common way a navigation result gets silently invalidated.

``DENSE`` and ``SPARSE`` are *distribution shifts*, not just unseen seeds:
they change obstacle density relative to training, which is the Phase-5
generalization stress test.
"""

from __future__ import annotations

from dataclasses import replace

from vision_nav.envs.world import WorldConfig

__all__ = ["SEED_BANDS", "split_seeds", "shifted_config", "SHIFTS"]

#: Disjoint seed bands, ``(low, high)`` half-open.
SEED_BANDS: dict[str, tuple[int, int]] = {
    "train": (0, 1000),
    "val": (10_000, 10_100),
    "test": (20_000, 20_200),
    "test_ood": (30_000, 30_200),
}


def split_seeds(split: str, n: int | None = None) -> list[int]:
    """World seeds belonging to ``split``.

    Parameters
    ----------
    split:
        One of the keys of :data:`SEED_BANDS`.
    n:
        Truncate to the first ``n`` seeds.  Useful for quick smoke runs.
    """
    if split not in SEED_BANDS:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(SEED_BANDS)}")
    lo, hi = SEED_BANDS[split]
    seeds = list(range(lo, hi))
    return seeds[:n] if n is not None else seeds


#: Named distribution shifts applied on top of the training world config.
SHIFTS: dict[str, dict] = {
    "nominal": {},
    "dense": {"n_circles": (14, 22), "n_boxes": (6, 10)},
    "sparse": {"n_circles": (1, 4), "n_boxes": (0, 2)},
    "large": {"width": 16.0, "height": 16.0, "min_start_goal_dist": 8.0},
    "narrow": {"circle_radius": (0.5, 1.4), "n_circles": (10, 16)},
}


def shifted_config(base: WorldConfig, shift: str) -> WorldConfig:
    """Return ``base`` with the named distribution shift applied."""
    if shift not in SHIFTS:
        raise KeyError(f"unknown shift {shift!r}; expected one of {sorted(SHIFTS)}")
    return replace(base, **SHIFTS[shift])
