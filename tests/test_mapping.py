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


def test_one_contrary_beam_does_not_erase_a_surface_and_many_do():
    """Evidence, not stickiness. A single beam grazing through a cell once hit
    must not erase it -- that is how a surface would vanish from under a robot
    turning past it -- but a cell repeatedly seen through is free space, which
    is how noise stops accumulating as phantom obstacles."""
    m = OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [2.0]))
    pose = np.array([1.0, 1.05, 0.0])
    m.integrate(pose)
    cell = (int(1.05 / m.resolution), int(3.0 / m.resolution))
    assert m.grid[cell] == OCCUPIED
    m.sensor = _FakeSensor([0.0], [6.0])  # beams now pass straight through
    m.integrate(pose)
    assert m.grid[cell] == OCCUPIED, "one pass erased a surface"
    for _ in range(4):
        m.integrate(pose)
    assert m.grid[cell] == FREE, "repeated passes left a phantom in free space"


def test_every_mapped_cell_really_holds_a_surface_within_the_assumed_reach():
    """What conservative placement guarantees. It cannot promise the map never
    reports more room than there is -- unknown is free, and an unseen face of an
    obstacle is exactly such room. It does promise that every cell called
    occupied holds a real surface no further from its centre than the half
    diagonal the clearance subtracts, so clearance to what *has* been mapped is
    never overstated. First written as the stronger claim, which failed where
    32 beams had not yet landed on part of the box: correctly."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    m = OccupancyMap(world, make_sensor("lidar32"))
    for x, y in ((3.5, 6.0), (7.0, 6.0), (5.3, 3.0), (5.3, 9.0), (3.8, 3.8), (6.8, 8.2)):
        for heading in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            m.integrate(np.array([x, y, heading]))
    occupied = m.grid_to_world(np.argwhere(m.grid == OCCUPIED))
    assert len(occupied) > 20, "the scans mapped almost nothing; the check would be empty"
    true = world.clearance(occupied, include_dynamic=False)
    assert (true <= m.half_diagonal + 1e-9).all(), true.max()


def test_noise_does_not_accumulate_into_phantoms():
    """Phase 6d's first map filled open floor with obstacles under 0.10 m of
    range noise. With evidence, open floor far from any surface stays free."""
    world = _open_world(boxes=[(8.0, 4.0, 8.6, 8.0)])
    m = OccupancyMap(world, make_sensor("lidar32", noise_std=0.10),
                     rng=np.random.default_rng(0))
    for _ in range(200):
        m.integrate(np.array([3.0, 6.0, 0.0]))
    open_floor = m.world_to_grid(np.array([[4.5, 6.0], [5.5, 6.0], [3.0, 7.5]]))
    assert all(m.grid[tuple(c)] == FREE for c in open_floor)
    assert (m.clearance(np.array([[5.0, 6.0]])) > 2.0).all()


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
    """Three metres ahead is not urgent, so the answer comes at the replanning
    rate rather than on the next scan -- within a second either way."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    agent = MappedPursuitAgent(sensor="camera64")
    # Start facing away, so the first scan does not see the box.
    assert agent.start_episode(world, np.array([2.0, 6.0, np.pi]))
    replans_before = agent.replans
    for _ in range(agent.replan_period + 1):
        agent.act(np.array([2.0, 6.0, 0.0]))  # now facing it
    assert agent.replans > replans_before, "a newly seen obstacle on the plan was ignored"
    assert agent.path is not None
    # And the new plan goes round it rather than through.
    assert (agent.map.clearance(agent._track) >= agent.map.robot_radius - 1e-6).all()


def test_something_close_ahead_is_answered_at_once():
    """The rate limit must not delay the near horizon: a wall a metre away is
    replanned around on the scan that finds it, not a second later."""
    world = _open_world(boxes=[(3.2, 4.0, 3.8, 8.0)])
    agent = MappedPursuitAgent(sensor="camera64")
    assert agent.start_episode(world, np.array([2.0, 6.0, np.pi]))
    before = agent.replans
    agent.act(np.array([2.0, 6.0, 0.0]))
    assert agent.replans == before + 1, "a wall a metre ahead waited for the timer"


def test_the_dense_scanner_is_the_one_nav2_was_given():
    """Report §9.4 compared this stack with Nav2 at 360 beams. The bridge builds
    that scan by raising the env lidar's beam count and nothing else, and
    publishes it as angle_min = -pi in steps of 2*pi/360; the two must agree
    beam for beam or the comparison would hold the sensor only nominally."""
    from dataclasses import replace

    from vision_nav.envs.sensors import LidarConfig

    sensor = make_sensor("lidar360", noise_std=0.1)
    bridge = replace(LidarConfig(), n_beams=360).beam_angles()
    np.testing.assert_allclose(sensor._angles, bridge)
    np.testing.assert_allclose(sensor._angles, -np.pi + np.arange(360) * (2 * np.pi / 360))
    assert sensor.config.max_range == LidarConfig().max_range
    assert sensor.config.noise_std == 0.1


def test_an_unknown_sensor_is_refused():
    with pytest.raises(ValueError, match="unknown sensor"):
        make_sensor("sonar")
