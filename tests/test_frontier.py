"""Tests for planning that does not pretend unexplored space is empty.

The property that matters is the one report §9.7 identified as the lever: a
route must never be *proposed* through space nobody has looked at. The planner
has to honour that, and so must the smoother afterwards, which is the easier
thing to get wrong -- it shortcuts wherever it has line of sight, and the map
answers "plenty of room" for everything it has never seen.
"""

from __future__ import annotations

import numpy as np

from vision_nav.agents.mapped import MappedPursuitAgent
from vision_nav.envs.world import World, WorldConfig
from vision_nav.mapping.occupancy import UNKNOWN
from vision_nav.planning.frontier import plan_through_known


def _half_known(rows: int = 20, cols: int = 20, edge: int = 10):
    """Left half known and free, right half unseen."""
    unknown = np.zeros((rows, cols), dtype=bool)
    unknown[:, edge:] = True
    blocked = unknown.copy()
    return blocked, unknown


def test_a_goal_in_known_space_is_planned_to_directly():
    blocked, unknown = _half_known()
    cells, kind = plan_through_known(blocked, unknown, (10, 1), (10, 8))
    assert kind == "goal"
    assert cells[0] == (10, 1) and cells[-1] == (10, 8)
    assert all(not unknown[r, c] for r, c in cells)


def test_a_goal_in_unseen_space_sends_the_robot_to_a_frontier():
    """The route runs through known space and takes exactly one step into the
    dark, which is what keeps it from being a route of length zero when the
    robot is already standing at the edge of what it has seen."""
    blocked, unknown = _half_known()
    cells, kind = plan_through_known(blocked, unknown, (10, 1), (10, 15))
    assert kind == "frontier"
    r, c = cells[-1]
    assert c == 10 and unknown[r, c], "the last step should enter the first unseen column"
    assert abs(r - 10) <= 1, "of the cells on offer, the one nearest the goal"
    assert all(not unknown[rr, cc] for rr, cc in cells[:-1]), "only the last step is unseen"


def test_where_an_opening_leads_counts_for_more_than_what_it_costs():
    """A three-cell corridor running towards the goal, with unseen floor on
    either side of the robot as well as at the end. The sides are a step away
    and the end is ten, but the end is where the goal is: picking the nearest
    opening instead would have the robot poking at the walls beside it."""
    rows, cols = 21, 21
    unknown = np.ones((rows, cols), dtype=bool)
    unknown[9:12, 0:13] = False         # the corridor the robot stands in
    blocked = unknown.copy()
    cells, kind = plan_through_known(blocked, unknown, (10, 2), (20, 20))
    assert kind == "frontier"
    assert cells[-1][1] == 13, cells[-1]
    assert len(cells) > 5, "a route that goes nowhere is not a route"


def test_nothing_reachable_reports_none():
    blocked = np.ones((10, 10), dtype=bool)
    blocked[5, 5] = False              # the robot's cell, walled in
    unknown = np.zeros((10, 10), dtype=bool)
    cells, kind = plan_through_known(blocked, unknown, (5, 5), (1, 1))
    assert cells is None and kind == "none"


def test_the_route_the_agent_drives_never_crosses_unseen_space():
    """End to end, including the smoother: every point of the committed track
    must lie in a cell the map has seen."""
    world = World(config=WorldConfig(), circles=np.zeros((0, 3)),
                  boxes=np.array([[5.0, 4.5, 5.6, 7.5]]),
                  start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))
    agent = MappedPursuitAgent(sensor="lidar32", frontier=True)
    assert agent.start_episode(world, world.start)
    assert agent.plan_kind == "frontier", "the goal is 8 m off and cannot be seen yet"
    for _ in range(60):
        agent.act(np.array([2.0, 6.0, 0.0]))
        if agent._track is None:
            continue
        cells = agent.map.world_to_grid(agent._track)
        seen = agent.map.grid[cells[:, 0], cells[:, 1]] != UNKNOWN
        # Known throughout, except a single step into one unseen cell at the
        # very end: nothing unseen may appear in the middle of the route, and
        # the route may not wander through more than one unseen cell.
        if seen.all():
            continue
        first_unseen = int(np.argmin(seen))
        assert not seen[first_unseen:].any(), "the route re-enters known space after the unseen"
        unseen_cells = {tuple(c) for c in cells[first_unseen:]}
        assert len(unseen_cells) == 1, f"the route crosses {len(unseen_cells)} unseen cells"


def test_the_optimistic_planner_is_untouched_by_default():
    """Every published result plans with unknown space counted as free."""
    world = World(config=WorldConfig(), circles=np.zeros((0, 3)),
                  boxes=np.array([[5.0, 4.5, 5.6, 7.5]]),
                  start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))
    agent = MappedPursuitAgent(sensor="lidar32")
    assert agent.start_episode(world, world.start)
    assert agent.frontier is False
    assert agent.plan_kind == "goal"
    cells = agent.map.world_to_grid(agent._track)
    unseen = agent.map.grid[cells[:, 0], cells[:, 1]] == UNKNOWN
    assert unseen.any(), "the optimistic route should run through unseen space here"
