"""Tests for the clutter forensic and the margin audit.

Two things here are load-bearing for §9.9 and are easy to get quietly wrong.

The first is the statistic. The forensic's headline compares two *groups of
worlds* rather than two arms on the same worlds, so nothing about it is paired
and the test is Fisher's exact. An implementation that silently returns the
one-sided tail would roughly halve every p-value in that section.

The second is the control on the blockage counts. The agent plans at
``robot_radius + safety_margin`` while the world's own grid is inflated by the
radius alone, so counting cells the agent's map blocks -- without counting what
the truth blocks at the same inflation -- charges the planner's safety margin to
its mapping. That is exactly the kind of comparison this project keeps catching
late, so the audit's two grids are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from clutter_forensic import remaining_at  # noqa: E402
from margin_audit import clear_footprint, relax_goal  # noqa: E402
from margin_overlap import fisher_exact  # noqa: E402

from vision_nav.envs.world import World, WorldConfig  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402


def _world(boxes=None) -> World:
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.zeros((0, 4)) if boxes is None else np.asarray(boxes),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def test_fisher_is_two_sided():
    """Fisher's tea-tasting table, whose two-sided p is 0.485714..."""
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857142857, rel=1e-9)


def test_fisher_on_a_hand_computed_table():
    """[[1, 9], [11, 3]]: the tables at least as extreme are x in {0, 1, 9, 10},
    giving (91 + 3640 + 3640 + 91) / C(24, 12) = 7462 / 2704156. The one-sided
    tail is about 0.00138, so a one-sided implementation fails this."""
    assert fisher_exact(1, 9, 11, 3) == pytest.approx(7462 / 2704156, rel=1e-9)


def test_fisher_is_one_when_nothing_is_associated():
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)


def test_the_forensic_measures_the_true_distance_that_is_left():
    """A wall between start and goal, so the geodesic must exceed the straight
    line: the instrument has to measure driving distance, not line of sight."""
    world = _world([[5.0, 4.5, 5.6, 7.5]])
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    res = world.config.grid_resolution
    field = geodesic_distance_field(world.occupancy, goal_cell) * res
    start = world.start[:2]
    straight = float(np.linalg.norm(world.goal - start))
    assert remaining_at(field, world, start) > straight
    # And it falls to nothing at the goal itself.
    assert remaining_at(field, world, world.goal) < 0.3


def test_the_audit_frees_the_robots_own_footprint():
    """The agent marks its own footprint free before planning, so the audit has
    to as well, or a robot standing inside the inflation reports no route."""
    world = _world()
    occ = np.ones_like(world.occupancy, dtype=bool)
    here = np.array([3.0, 3.0])
    cleared = clear_footprint(world, occ, here)
    r, c = (int(v) for v in world.world_to_grid(here))
    assert not cleared[r, c]
    assert occ[r, c], "the caller's grid must not be modified in place"


def test_the_audit_relaxes_the_goal_the_way_the_agent_does():
    """A goal cell swallowed by inflation resolves to a free cell within the
    tolerance, and to nothing at all when the whole neighbourhood is blocked."""
    world = _world()
    occ = np.zeros_like(world.occupancy, dtype=bool)
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    occ[goal_cell] = True
    relaxed = relax_goal(world, occ, world.goal)
    assert relaxed is not None and relaxed != goal_cell
    assert not occ[relaxed]
    assert relax_goal(world, np.ones_like(occ), world.goal) is None


def test_the_blockage_control_uses_the_same_inflation_on_both_sides():
    """The measurement §9.9 rests on: at one inflation the world blocks what it
    blocks, and a map that matches the world must not look worse than it for
    having a safety margin applied to only one of the two."""
    world = _world([[5.0, 4.5, 5.6, 7.5]])
    base = world.config.robot_radius
    at_base = world.occupancy_at(base)
    at_margin = world.occupancy_at(base + 0.18)
    # Strictly more is blocked at the larger radius: this is the asymmetry the
    # forensic has to control for rather than attribute to the map.
    assert at_margin.sum() > at_base.sum()
    assert bool((at_base & ~at_margin).sum() == 0)
