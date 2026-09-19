"""Tests for the self-built map and the planner that uses it.

The point of the experiment is that the planner no longer reads the true map,
so the tests that matter are the ones that would catch it reading the true map
anyway -- through the smoother, the blocked-path check, the controller's
slow-down, or the map's own construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.agents.mapped import MappedPursuitAgent, make_sensor
from vision_nav.envs.sensors import DepthCamera
from vision_nav.envs.world import World, WorldConfig
from vision_nav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN, OccupancyMap


class _NoGeometry:
    """A world that answers metadata and refuses geometry."""

    def __init__(self):
        self.config = WorldConfig()
        self.seed = 0
        self.goal = np.array([10.0, 6.0])

    def __getattr__(self, name):
        raise AssertionError(f"the map read world.{name}, which is geometry")


class _FakeSensor:
    """Returns whatever ranges it is told to, whatever the world holds."""

    def __init__(self, angles, ranges, max_range=6.0):
        self._angles = np.asarray(angles, dtype=float)
        self._ranges = np.asarray(ranges, dtype=float)
        self.config = type("C", (), {"max_range": max_range})()

    def scan(self, world, pose, rng=None):
        return self._ranges


def test_the_map_is_built_from_ranges_alone():
    """Fabricated returns, a world that raises on any geometry: the map must
    come out of the ranges and nothing else."""
    sensor = _FakeSensor(angles=[0.0], ranges=[2.0])
    m = OccupancyMap(_NoGeometry(), sensor)
    assert m.integrate(np.array([1.0, 1.05, 0.0]))
    r = int(1.05 / m.resolution)
    assert m.grid[r, int(1.5 / m.resolution)] == FREE, "short of the return must be free"
    assert m.grid[r, int(3.0 / m.resolution)] == OCCUPIED, "the return's cell must be occupied"
    assert m.grid[r, int(4.0 / m.resolution)] == UNKNOWN, "past the return is unseen"


def test_a_return_at_max_range_marks_nothing_occupied():
    m = OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [6.0]))
    assert not m.integrate(np.array([1.0, 1.05, 0.0]))
    assert not (m.grid == OCCUPIED).any()


def test_occupied_cells_are_not_cleared_by_later_beams():
    world = _NoGeometry()
    hit = _FakeSensor([0.0], [2.0])
    m = OccupancyMap(world, hit)
    m.integrate(np.array([1.0, 1.05, 0.0]))
    cell = (int(1.05 / m.resolution), int(3.0 / m.resolution))
    m.sensor = _FakeSensor([0.0], [6.0])  # a later beam passing straight through
    m.integrate(np.array([1.0, 1.05, 0.0]))
    assert m.grid[cell] == OCCUPIED


def test_unknown_space_is_free_to_plan_through():
    m = OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [6.0]))
    assert not m.occupancy_at(0.4).any()
    assert (m.clearance(np.array([[5.0, 5.0]])) > 100).all()


def test_inflation_reaches_the_radius_and_no_further():
    m = OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [2.0]))
    m.integrate(np.array([1.0, 6.05, 0.0]))
    occ = m.occupancy_at(0.4)
    hit = np.array([3.05, 6.05])
    centres = m.grid_to_world(np.argwhere(occ))
    far = np.linalg.norm(centres - hit, axis=1).max()
    assert far <= 0.4 + m.resolution * 1.5, far
    near = m.world_to_grid(hit + np.array([0.3, 0.0]))
    assert occ[tuple(near)], "a cell 0.3 m from the obstacle escaped a 0.4 m inflation"


def test_the_camera_sensor_samples_the_depth_cameras_columns():
    np.testing.assert_allclose(make_sensor("camera64")._angles, DepthCamera()._lidar._angles)


def _open_world(boxes=()):
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def test_the_mapped_planner_never_asks_the_world_for_its_map():
    """Planning, smoothing, blocked-path checks and the controller's slow-down
    all used to read the true map. Every one of them must now go to the robot's
    own; the sensor itself is the only thing allowed to touch geometry."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    # The env builds the world's own grid for collision checking; do it before
    # the spies go in. First written after them, which recorded the test's own
    # call and read as a leak in the agent.
    _ = world.occupancy
    calls = []
    for name in ("clearance", "occupancy_at"):
        real = getattr(world, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)
        object.__setattr__(world, name, spy)
    agent = MappedPursuitAgent(sensor="lidar32")
    pose = np.array([2.0, 6.0, 0.0])
    assert agent.start_episode(world, pose)
    for x in np.linspace(2.0, 4.5, 20):
        agent.act(np.array([x, 6.0, 0.0]))
    assert calls == [], f"the mapped planner read the true map: {set(calls)}"


def test_discovering_an_obstacle_on_the_path_triggers_a_replan():
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    agent = MappedPursuitAgent(sensor="camera64")
    # Start facing away, so the first scan does not see the box.
    assert agent.start_episode(world, np.array([2.0, 6.0, np.pi]))
    replans_before = agent.replans
    agent.act(np.array([2.0, 6.0, 0.0]))  # now facing it
    assert agent.replans > replans_before, "a newly seen obstacle on the plan was ignored"
    assert agent.path is not None
    # And the new plan goes round it rather than through.
    assert (agent.map.clearance(agent._track) >= agent.map.robot_radius - 1e-6).all()


def test_an_unknown_sensor_is_refused():
    with pytest.raises(ValueError, match="unknown sensor"):
        make_sensor("sonar")
