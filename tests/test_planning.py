"""Tests for A*, the geodesic distance field and path smoothing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vision_nav.envs.world import generate_world
from vision_nav.planning.grid_astar import (
    astar_grid,
    geodesic_distance_field,
    plan_path,
    shortest_path_length,
)
from vision_nav.planning.smoothing import densify_path, path_length, simplify_path


def test_astar_finds_straight_line_in_empty_grid():
    occ = np.zeros((10, 10), dtype=bool)
    path = astar_grid(occ, (0, 0), (0, 9))
    assert path is not None
    assert len(path) == 10
    assert path[0] == (0, 0) and path[-1] == (0, 9)


def test_astar_diagonal_cost_is_optimal():
    occ = np.zeros((10, 10), dtype=bool)
    path = astar_grid(occ, (0, 0), (9, 9))
    assert path is not None
    # 9 diagonal steps is optimal for an 8-connected grid.
    assert len(path) == 10


def test_astar_returns_none_when_walled_off():
    occ = np.zeros((10, 10), dtype=bool)
    occ[:, 5] = True
    assert astar_grid(occ, (0, 0), (0, 9)) is None


def test_astar_rejects_blocked_endpoints():
    occ = np.zeros((5, 5), dtype=bool)
    occ[2, 2] = True
    assert astar_grid(occ, (2, 2), (0, 0)) is None
    assert astar_grid(occ, (0, 0), (2, 2)) is None


def test_astar_does_not_squeeze_between_diagonal_blockers():
    """A robot inflated to a full cell cannot pass a diagonal pinch point."""
    occ = np.zeros((3, 3), dtype=bool)
    occ[0, 1] = True
    occ[1, 0] = True
    path = astar_grid(occ, (0, 0), (1, 1))
    # (0,0) -> (1,1) is a diagonal move blocked on both orthogonal sides.
    assert path is None


def test_astar_path_is_collision_free():
    occ = np.zeros((20, 20), dtype=bool)
    occ[5:15, 10] = True
    path = astar_grid(occ, (0, 0), (19, 19))
    assert path is not None
    assert not any(occ[r, c] for r, c in path)


def test_distance_field_agrees_with_astar():
    rng = np.random.default_rng(0)
    occ = rng.random((24, 24)) < 0.2
    occ[0, 0] = occ[23, 23] = False

    field = geodesic_distance_field(occ, (23, 23))
    path = astar_grid(occ, (0, 0), (23, 23))

    if path is None:
        assert not math.isfinite(field[0, 0])
    else:
        cost = sum(
            1.0 if (a[0] == b[0] or a[1] == b[1]) else math.sqrt(2.0)
            for a, b in zip(path[:-1], path[1:], strict=True)
        )
        assert field[0, 0] == pytest.approx(cost, rel=1e-9)


def test_distance_field_marks_blocked_and_unreachable_as_inf():
    occ = np.zeros((10, 10), dtype=bool)
    occ[:, 5] = True
    field = geodesic_distance_field(occ, (0, 0))
    assert not np.isfinite(field[0, 5]), "blocked cell should be inf"
    assert not np.isfinite(field[0, 9]), "walled-off region should be inf"
    assert field[0, 0] == 0.0


def test_simplified_path_stays_clear_and_gets_shorter():
    w = generate_world(11)
    raw = plan_path(w, w.start[:2], w.goal)
    assert raw is not None
    simple = simplify_path(w, raw, w.config.robot_radius)

    assert len(simple) <= len(raw)
    assert path_length(simple) <= path_length(raw) + 1e-9

    dense = densify_path(simple, 0.02)
    assert np.all(w.clearance(dense) > w.config.robot_radius)


def test_shortest_path_length_is_at_least_euclidean():
    """l* can never be shorter than the straight line -- a sanity floor on SPL."""
    for seed in range(8):
        w = generate_world(seed)
        l_star = shortest_path_length(w, w.start[:2], w.goal)
        euclid = float(np.linalg.norm(w.goal - w.start[:2]))
        assert l_star is not None
        assert l_star >= euclid - 1e-6


def test_densify_preserves_endpoints_and_spacing():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    dense = densify_path(path, 0.1)
    assert np.allclose(dense[0], path[0])
    assert np.allclose(dense[-1], path[-1])
    steps = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    assert steps.max() <= 0.1 + 1e-9
    assert path_length(dense) == pytest.approx(path_length(path))
