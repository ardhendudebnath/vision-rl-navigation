"""Planning a buffer inside the margin's edge, and judging the plan at the edge."""

from __future__ import annotations

import numpy as np

from vision_nav.agents.mapped import MappedPursuitAgent
from vision_nav.envs.world import World, WorldConfig


def _world(boxes):
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def _plan(world, buffer: float):
    agent = MappedPursuitAgent(sensor="lidar360")
    agent.plan_buffer = buffer
    assert agent.start_episode(world, world.start.copy())
    return agent


def _closest(agent) -> float:
    """The route's least clearance to anything mapped, past the robot's footprint."""
    track = agent._track
    away = np.linalg.norm(track - agent._track[0], axis=1) > 0.5
    return float(agent.map.clearance(track[away]).min())


def test_without_a_buffer_the_route_runs_on_the_margins_edge():
    agent = _plan(_world([(5.0, 4.5, 5.6, 7.5)]), 0.0)
    margin = agent.map.robot_radius + agent.config.safety_margin
    assert agent._plan_radius == margin
    assert margin - 1e-9 <= _closest(agent) < margin + 0.10


def test_a_buffered_route_keeps_the_buffer_and_is_judged_at_the_margin():
    agent = _plan(_world([(5.0, 4.5, 5.6, 7.5)]), 0.10)
    margin = agent.map.robot_radius + agent.config.safety_margin
    assert agent._plan_radius == margin, "judged at the margin, not at the buffer"
    assert _closest(agent) >= margin + 0.10 - 1e-9


def test_where_the_buffer_does_not_fit_the_published_ladder_still_plans():
    """A gap wide enough for the margin and too narrow for the buffer: the
    buffered rung finds nothing and the margin rung plans through the gap."""
    wall = [(5.0, 0.0, 5.6, 5.45), (5.0, 6.55, 5.6, 12.0)]  # a 1.1 m gap at y = 6
    plain = _plan(_world(wall), 0.0)
    buffered = _plan(_world(wall), 0.10)
    margin = plain.map.robot_radius + plain.config.safety_margin
    assert plain._plan_radius == buffered._plan_radius == margin
    through = lambda a: bool((np.abs(a._track[:, 0] - 5.3) < 0.05).any() and  # noqa: E731
                             (np.abs(a._track[:, 1] - 6.0) < 0.6).all())
    assert through(plain) and through(buffered)
