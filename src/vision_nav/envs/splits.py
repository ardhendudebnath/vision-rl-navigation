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

__all__ = [
    "SEED_BANDS",
    "split_seeds",
    "shifted_config",
    "SHIFTS",
    "BENCHMARK_CONDITIONS",
    "DYNAMIC_CONDITIONS",
    "FROZEN_CONDITIONS",
]

#: The six evaluation conditions every actor is scored on, as
#: ``(split, shift, lidar noise std)``. ``nominal`` is the in-distribution
#: held-out set; the rest are the robustness suite.
#:
#: This lives in the library rather than in ``scripts/run_benchmark.py``
#: because a second runner (``ros2_bridge/run_nav2_eval.py``, which cannot
#: import a script) has to reproduce these conditions exactly. Two copies of
#: this table drifting apart would show up as a Nav2 row that looks
#: comparable but was measured under different conditions.
BENCHMARK_CONDITIONS: dict[str, tuple[str, str | None, float]] = {
    "nominal": ("test", None, 0.0),
    "dense": ("test_ood", "dense", 0.0),
    "sparse": ("test_ood", "sparse", 0.0),
    "large": ("test_ood", "large", 0.0),
    "narrow": ("test_ood", "narrow", 0.0),
    "noisy_lidar": ("test", None, 0.10),
}

#: The moving-obstacle conditions, kept separate from BENCHMARK_CONDITIONS so
#: that adding them does not silently change what ``run_benchmark.py`` runs by
#: default for every other actor. Same ``(split, shift, noise)`` shape.
#:
#: These are scored by ``scripts/dynamic_experiment.py`` for the learned arms,
#: where the classical baseline replans every 10 steps — its measured best
#: setting here. Nav2 needs no equivalent knob: its default behaviour tree
#: re-plans on a 1 Hz rate controller, which at the 10 Hz sim step is the same
#: interval the baseline was tuned to.
DYNAMIC_CONDITIONS: dict[str, tuple[str, str | None, float]] = {
    "dynamic": ("test_ood", "dynamic", 0.0),
    "dynamic_dense": ("test_ood", "dynamic_dense", 0.0),
    # Same worlds, movers parked. Together these complete a 2x2 over
    # clutter and motion. See FROZEN_CONDITIONS.
    "dynamic_frozen": ("test_ood", "dynamic", 0.0),
    "dynamic_dense_frozen": ("test_ood", "dynamic_dense", 0.0),
}

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
    #: Moving obstacles absent from the map. The one condition in the suite
    #: where the classical planner's central privilege — a perfect, current
    #: map — stops being true, and therefore the one place a reactive policy
    #: has a structural reason to win.
    "dynamic": {"n_circles": (4, 8), "n_boxes": (1, 4), "n_dynamic": (3, 6)},
    #: Dynamic obstacles on top of clutter: both pressures at once.
    "dynamic_dense": {"n_circles": (10, 16), "n_boxes": (4, 8), "n_dynamic": (3, 6)},
}

#: Conditions whose movers are frozen in place after the world is generated.
#:
#: The controlled subtraction for report Section 9.3, which isolates whether
#: Nav2's advantage on `dynamic_dense` is the clutter or an interaction between
#: clutter and motion.
#:
#: Freezing is deliberately NOT a shift. Setting ``dynamic_amplitude`` to
#: ``(0, 0)`` in the world config looks equivalent and is not: mover placement
#: validates the swept path, so a zero sweep accepts positions the moving
#: config rejects, and the two worlds end up with different obstacles. That
#: would confound the subtraction with a geometry change. Instead the world is
#: generated exactly as `dynamic_dense` and the amplitudes are zeroed
#: afterwards, pinning each mover at the centre it already occupies at t = 0.
#: ``test_frozen_matches_dynamic_dense`` asserts the geometry is identical.
FROZEN_CONDITIONS: frozenset[str] = frozenset(
    {"dynamic_frozen", "dynamic_dense_frozen"}
)


def shifted_config(base: WorldConfig, shift: str) -> WorldConfig:
    """Return ``base`` with the named distribution shift applied."""
    if shift not in SHIFTS:
        raise KeyError(f"unknown shift {shift!r}; expected one of {sorted(SHIFTS)}")
    return replace(base, **SHIFTS[shift])
