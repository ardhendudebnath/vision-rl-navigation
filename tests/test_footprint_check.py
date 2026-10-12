"""The validity check that, like the planner, does not count the robot's footprint.

Report §9.22: the planner clears the robot's own footprint before it searches,
so a plan made inside a wall's margin begins inside it, and the published check
-- which does not clear it -- rebuilds that plan at the next change anywhere in
the map.
"""

from __future__ import annotations

import numpy as np

from vision_nav.agents.mapped import MappedPursuitAgent
from vision_nav.envs.world import World, WorldConfig


def _beside_a_wall(footprint_check: bool):
    """A robot beside a long wall, its goal along it: by its own map (which
    places surfaces conservatively) clear of its body, 0.22 m, and inside the
    planner's margin, 0.40 m."""
    world = World(config=WorldConfig(), circles=np.zeros((0, 3)),
                  boxes=np.array([[3.0, 5.0, 9.0, 5.7]]),
                  start=np.array([4.0, 6.1, 0.0]), goal=np.array([10.0, 6.0]))
    agent = MappedPursuitAgent(sensor="lidar360")
    agent.footprint_check = footprint_check
    pose = np.array([4.0, 6.1, 0.0])
    assert agent.start_episode(world, pose)
    full = agent.map.robot_radius + agent.config.safety_margin
    assert agent.map.robot_radius < agent.map.clearance(pose[:2]) < full
    return agent, pose


def test_the_published_check_rebuilds_a_plan_begun_inside_the_margin():
    """§9.22's mechanism: nothing on the route has changed, only the map
    somewhere, and the plan is thrown away at once."""
    agent, pose = _beside_a_wall(footprint_check=False)
    agent.map.version += 1  # a change anywhere in the map
    assert agent._path_ahead_blocked(pose[:2])
    assert agent._urgent, "the blockage is where the robot stands, so it cannot wait"


def test_the_footprint_consistent_check_keeps_it():
    agent, pose = _beside_a_wall(footprint_check=True)
    agent.map.version += 1
    assert not agent._path_ahead_blocked(pose[:2])
    # Only the footprint and the grid step out of it are excused: past them the
    # route is clear of the margin, and the published check agrees.
    ahead = agent._track[agent._cursor:]
    reach = agent.map.robot_radius + 2.0 * agent.map.half_diagonal
    outside = np.linalg.norm(ahead - pose[:2], axis=1) > reach
    assert (agent.map.clearance(ahead[outside]) >= agent._plan_radius - 1e-9).all()
    # And the step out is needed: excusing the footprint alone leaves points
    # just past it under the plan radius, so the plan would still be rebuilt.
    step = ~outside & (np.linalg.norm(ahead - pose[:2], axis=1) > agent.map.robot_radius)
    assert (agent.map.clearance(ahead[step]) < agent._plan_radius - 1e-9).any()


def test_the_footprint_consistent_check_still_sees_a_real_blockage():
    """A surface across the route just past the footprint is answered at once."""
    agent, pose = _beside_a_wall(footprint_check=True)
    ahead = agent._track[agent._cursor:]
    along = np.linalg.norm(ahead - pose[:2], axis=1)
    point = ahead[int(np.argmax(along > 0.6))]
    agent.map._occupied_pts = np.vstack([agent.map._occupied_pts, point])
    agent.map.version += 1
    assert agent._path_ahead_blocked(pose[:2])
    assert agent._urgent
