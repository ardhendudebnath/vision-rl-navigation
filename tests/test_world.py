"""Tests for procedural world generation and geometry queries."""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.envs.world import World, WorldConfig, generate_world
from vision_nav.planning.grid_astar import shortest_path_length


def test_generation_is_deterministic():
    a = generate_world(7)
    b = generate_world(7)
    assert np.array_equal(a.circles, b.circles)
    assert np.array_equal(a.boxes, b.boxes)
    assert np.array_equal(a.start, b.start)
    assert np.array_equal(a.goal, b.goal)


def test_different_seeds_give_different_worlds():
    a = generate_world(1)
    b = generate_world(2)
    assert not np.array_equal(a.start, b.start) or not np.array_equal(a.circles, b.circles)


@pytest.mark.parametrize("seed", range(12))
def test_generated_worlds_satisfy_their_contract(seed):
    """Start and goal must be free, far enough apart, and connected.

    If this ever fails, every success-rate and SPL number in the project
    becomes uninterpretable: a failed episode could mean an unsolvable task
    rather than a bad policy.
    """
    w = generate_world(seed)
    cfg = w.config

    assert w.is_free(w.start[:2]), "start pose is in collision"
    assert w.is_free(w.goal), "goal is in collision"
    assert np.linalg.norm(w.goal - w.start[:2]) >= cfg.min_start_goal_dist
    assert shortest_path_length(w, w.start[:2], w.goal) is not None, "goal unreachable"


def test_clearance_matches_analytic_geometry():
    cfg = WorldConfig(width=10.0, height=10.0)
    w = World(
        config=cfg,
        circles=np.array([[5.0, 5.0, 1.0]]),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    # A point 3 m from a radius-1 circle centre has 2 m of clearance.
    assert w.clearance(np.array([8.0, 5.0])) == pytest.approx(2.0)
    # Inside the circle, clearance is negative.
    assert w.clearance(np.array([5.0, 5.0])) == pytest.approx(-1.0)
    # Near the wall, the wall dominates.
    assert w.clearance(np.array([0.25, 5.0])) == pytest.approx(0.25)


def test_clearance_inside_box_is_negative():
    cfg = WorldConfig(width=10.0, height=10.0)
    w = World(
        config=cfg,
        circles=np.zeros((0, 3)),
        boxes=np.array([[3.0, 3.0, 6.0, 6.0]]),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    assert w.clearance(np.array([4.5, 4.5])) < 0.0
    assert w.clearance(np.array([7.0, 4.5])) == pytest.approx(1.0)
    # Diagonal corner distance.
    assert w.clearance(np.array([7.0, 7.0])) == pytest.approx(np.sqrt(2.0))


def test_occupancy_inflation_grows_with_radius():
    w = generate_world(3)
    small = w.occupancy_at(w.config.robot_radius)
    large = w.occupancy_at(w.config.robot_radius + 0.3)
    assert large.sum() >= small.sum()
    # Inflation only ever adds blocked cells.
    assert np.all(large[small])


def test_grid_world_roundtrip_is_within_one_cell():
    w = generate_world(4)
    pts = np.array([[1.05, 2.35], [6.0, 6.0], [11.1, 0.4]])
    back = w.grid_to_world(w.world_to_grid(pts))
    assert np.all(np.abs(back - pts) <= w.config.grid_resolution)


def test_coordinate_conversions_preserve_query_shape():
    """A single point in must give a single point out.

    Returning a ``(1, 2)`` array for a ``(2,)`` query silently turns every
    downstream broadcast into an extra axis, which surfaces far from the
    cause.
    """
    w = generate_world(5)
    assert w.grid_to_world(np.array([3, 4])).shape == (2,)
    assert w.grid_to_world(np.array([[3, 4], [5, 6]])).shape == (2, 2)
    assert w.world_to_grid(np.array([1.0, 2.0])).shape == (2,)
    assert w.world_to_grid(np.array([[1.0, 2.0], [3.0, 4.0]])).shape == (2, 2)


def test_clearance_preserves_leading_dimensions():
    w = generate_world(5)
    assert np.ndim(w.clearance(np.zeros(2))) == 0
    assert w.clearance(np.zeros((7, 2))).shape == (7,)
    assert w.clearance(np.zeros((3, 5, 2))).shape == (3, 5)
    with pytest.raises(ValueError, match="trailing dim 2"):
        w.clearance(np.zeros((4, 3)))


def test_impossible_config_raises_rather_than_hanging():
    cfg = WorldConfig(width=6.0, height=6.0, min_start_goal_dist=3.0, n_circles=(60, 60))
    with pytest.raises(RuntimeError, match="failed to generate"):
        generate_world(0, cfg)
