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


def test_corroboration_is_inert_for_a_sparse_scan():
    """The rule weighs a cell's returns against the returns a surface there
    would have produced. At 32 beams fewer than one beam crosses a cell beyond
    half a metre, so every hit is already all it could earn and the map must
    come out identical -- which is what lets every published 32-beam result
    stand."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5), (7.5, 2.0, 8.1, 5.5)])
    grids = []
    for corroborate in (False, True):
        m = OccupancyMap(world, make_sensor("lidar32"), corroborate=corroborate)
        for x, y in ((3.0, 6.0), (4.0, 6.5), (6.5, 6.0), (6.5, 3.0)):
            for heading in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                m.integrate(np.array([x, y, heading]))
        grids.append(m.grid.copy())
    assert np.array_equal(grids[0], grids[1])
    assert (grids[0] == OCCUPIED).sum() > 20, "the scans mapped nothing; the check is empty"


def test_one_return_among_many_beams_no_longer_marks_a_cell():
    """What the rule is for. A 360-beam scan whose beams all miss except one,
    pointing at empty floor a metre away: that is the shape of a noisy return,
    and six beams cross the cell it lands in."""
    angles = -np.pi + np.arange(360) * (2 * np.pi / 360)
    ranges = np.full(360, 6.0)
    ranges[180] = 1.0  # one lone return, straight ahead in the robot's frame
    pose = np.array([3.0, 6.05, 0.0])

    marked = {}
    for corroborate in (False, True):
        m = OccupancyMap(_NoGeometry(), _FakeSensor(angles, ranges), corroborate=corroborate)
        m.integrate(pose)
        marked[corroborate] = (m.grid == OCCUPIED).sum()
    assert marked[False] == 1, "a single return should mark exactly its own cell"
    assert marked[True] == 0, "a lone return among six crossing beams was believed"

    # And it is not simply refusing everything: give the cell the returns a
    # real surface there would produce and it is marked.
    ranges[177:184] = 1.0
    m = OccupancyMap(_NoGeometry(), _FakeSensor(angles, ranges), corroborate=True)
    m.integrate(pose)
    assert (m.grid == OCCUPIED).any(), "a corroborated surface was not mapped"


def test_a_dense_scan_still_maps_real_walls():
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    m = OccupancyMap(world, make_sensor("lidar360"), corroborate=True)
    for heading in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        m.integrate(np.array([3.5, 6.0, heading]))
    occupied = m.grid_to_world(np.argwhere(m.grid == OCCUPIED))
    assert len(occupied) > 10, "a dense scan of a wall a metre away mapped almost nothing"
    assert (world.clearance(occupied, include_dynamic=False) <= m.half_diagonal + 1e-9).all()


def test_beam_votes_count_beams_and_not_samples():
    """A beam is sampled every half cell, so it lands two or three samples in
    each cell it crosses. Those must still be one vote; separate beams through
    the same cell must be separate votes."""
    m = OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [6.0]), beam_votes=True)
    pose = np.array([1.0, 1.05, 0.0])
    angles = np.asarray(m.sensor._angles) + pose[2]
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    step = m.resolution * 0.5
    ts = (np.arange(int(np.ceil(6.0 / step))) + 0.5) * step
    keep = ts[None, :] < 5.95
    pts = pose[None, None, :2] + dirs[:, None, :] * ts[None, :, None]
    rows, cols = m._cells(pts[keep])
    one_beam = m._beam_crossings(keep, rows, cols)
    assert one_beam.max() == 1.0, "one beam's samples voted more than once in a cell"

    # 360 beams: a cell close to the robot is crossed by many, one far away by few.
    angles = -np.pi + np.arange(360) * (2 * np.pi / 360)
    m = OccupancyMap(_NoGeometry(), _FakeSensor(angles, np.full(360, 6.0)),
                     beam_votes=True)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    keep = np.broadcast_to(ts[None, :] < 5.95, (360, len(ts)))
    pts = np.array([3.0, 6.05])[None, None, :] + dirs[:, None, :] * ts[None, :, None]
    rows, cols = m._cells(pts[keep])
    crossings = m._beam_crossings(keep, rows, cols)
    r = int(6.05 / m.resolution)
    near = crossings[r, int(3.45 / m.resolution)]    # 0.45 m ahead
    far = crossings[r, int(8.0 / m.resolution)]      # 5 m ahead
    assert near > 5 * far >= 5, (near, far)


def test_beam_votes_are_off_by_default():
    assert OccupancyMap(_NoGeometry(), _FakeSensor([0.0], [2.0])).beam_votes is False


def test_beam_votes_clear_a_phantom_that_one_vote_leaves_standing():
    """The mechanism the rule is for, in two scans. A lone return marks a cell a
    metre away (corroboration off, so it earns a full hit). Then a scan in which
    every beam reaches far past it: one vote per scan takes 0.9 to 0.6, still
    occupied, while the six beams that actually crossed it take it negative."""
    angles = -np.pi + np.arange(360) * (2 * np.pi / 360)
    pose = np.array([3.0, 6.05, 0.0])
    lone = np.full(360, 6.0)
    lone[180] = 1.0
    clear = np.full(360, 6.0)
    r, c = int(6.05 / 0.1), int(4.0 / 0.1)
    result = {}
    for votes in (False, True):
        m = OccupancyMap(_NoGeometry(), _FakeSensor(angles, lone), beam_votes=votes)
        m.integrate(pose)
        assert m.grid[r, c] == OCCUPIED, "the lone return should mark its cell first"
        m.sensor = _FakeSensor(angles, clear)
        m.integrate(pose)
        result[votes] = m.grid[r, c]
    assert result[False] == OCCUPIED, "one vote per scan should leave the phantom standing"
    assert result[True] == FREE, "six crossing beams should have cleared it"


def test_a_wall_survives_beam_votes_even_at_grazing_incidence():
    """The cost side. Stronger clearing could erase a real surface that beams
    graze on their way to somewhere else. Drive alongside a wall as well as
    facing it, and every mapped cell must still be a real surface while the
    wall is still mapped."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    m = OccupancyMap(world, make_sensor("lidar360"), corroborate=True, beam_votes=True)
    # Facing it, then sliding along it at a shallow angle.
    poses = [(3.5, 6.0, 0.0), (4.4, 4.0, 1.4), (4.4, 5.0, 1.5), (4.4, 6.0, 1.5),
             (4.4, 7.0, 1.6), (4.4, 8.0, 1.7), (3.5, 6.0, 0.3)]
    for x, y, h in poses:
        m.integrate(np.array([x, y, h]))
    occupied = m.grid_to_world(np.argwhere(m.grid == OCCUPIED))
    assert len(occupied) > 10, "the wall was erased"
    assert (world.clearance(occupied, include_dynamic=False) <= m.half_diagonal + 1e-9).all()
    # And the face the robot slid along is still there along its length.
    face = np.stack([np.full(20, 5.0), np.linspace(4.7, 7.3, 20)], axis=-1)
    rows, cols = m._cells(face)
    assert (m.grid[rows, cols] == OCCUPIED).mean() > 0.6


def test_beam_votes_at_32_beams_change_only_cells_beside_the_robot():
    """Measured rather than assumed. At 32 beams the beams are coarser than the
    cells beyond about half a metre, so per-beam votes can only differ from one
    vote per scan in the cells right beside the robot -- which are free either
    way, since the robot is standing in them. The occupied set must not move."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5), (7.5, 2.0, 8.1, 5.5)])
    grids = []
    for votes in (False, True):
        m = OccupancyMap(world, make_sensor("lidar32"), corroborate=True, beam_votes=votes)
        for x, y in ((3.0, 6.0), (4.0, 6.5), (6.5, 6.0), (6.5, 3.0)):
            for heading in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                m.integrate(np.array([x, y, heading]))
        grids.append(m.grid.copy())
    assert (grids[0] == OCCUPIED).sum() > 20, "the scans mapped nothing; the check is empty"
    assert np.array_equal(grids[0] == OCCUPIED, grids[1] == OCCUPIED)


class _NoisySensor(_FakeSensor):
    """A fake sensor whose configuration declares a range-noise figure."""

    def __init__(self, angles, ranges, noise_std, max_range=6.0):
        super().__init__(angles, ranges, max_range)
        self.config = type("C", (), {"max_range": max_range, "noise_std": noise_std})()


def test_the_noise_margin_is_off_by_default_and_inert_without_noise():
    m = OccupancyMap(_NoGeometry(), _NoisySensor([0.0], [2.0], noise_std=0.1))
    assert m.noise_margin == 0.0 and m.obstacle_range == 6.0
    # On, but the sensor has no noise: the line is the maximum itself.
    m = OccupancyMap(_NoGeometry(), _NoisySensor([0.0], [2.0], noise_std=0.0),
                     noise_margin=3.0)
    assert m.obstacle_range == 6.0


def test_a_max_range_beam_pushed_under_by_noise_is_not_a_surface():
    """The mechanism. A beam that hit nothing reads the maximum plus noise,
    clipped, so half the time it reads a little short. Without a margin that is
    a surface at the edge of the sensor's reach, in open floor; with one it is
    not -- and the space it crossed is still cleared."""
    pose = np.array([1.0, 1.05, 0.0])
    r = int(1.05 / 0.1)
    edge, middle = int((1.0 + 5.93) / 0.1), int(4.0 / 0.1)
    marked = {}
    for margin in (0.0, 3.0):
        m = OccupancyMap(_NoGeometry(), _NoisySensor([0.0], [5.93], noise_std=0.1),
                         noise_margin=margin)
        m.integrate(pose)
        marked[margin] = m.grid[r, edge]
        assert m.grid[r, middle] == FREE, "the beam must still clear what it crossed"
    assert marked[0.0] == OCCUPIED, "without a margin the noise is believed"
    assert marked[3.0] != OCCUPIED, "a reading inside the noise band was taken as a surface"


def test_a_real_surface_inside_the_obstacle_range_is_still_mapped():
    m = OccupancyMap(_NoGeometry(), _NoisySensor([0.0], [3.0], noise_std=0.1),
                     noise_margin=3.0)
    m.integrate(np.array([1.0, 1.05, 0.0]))
    assert m.grid[int(1.05 / 0.1), int(4.0 / 0.1)] == OCCUPIED


def test_the_noise_margin_leaves_a_noise_free_map_identical():
    """Every noise-free condition in the report must come out exactly as it
    was, which the property above guarantees and this checks on real scans."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5), (7.5, 2.0, 8.1, 5.5)])
    grids = []
    for margin in (0.0, 3.0):
        m = OccupancyMap(world, make_sensor("lidar360"), corroborate=True,
                         noise_margin=margin)
        for x, y in ((3.0, 6.0), (4.0, 6.5), (6.5, 6.0), (6.5, 3.0)):
            for heading in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                m.integrate(np.array([x, y, heading]))
        grids.append(m.grid.copy())
    assert (grids[0] == OCCUPIED).sum() > 20
    assert np.array_equal(grids[0], grids[1])


def test_a_blockage_far_ahead_no_longer_turns_the_robot_around():
    """Commitment. A box three metres along the route is beyond the two metres
    the robot stands by, and the way round it is no shorter, so the committed
    route is kept; without commitment the same discovery rebuilds it. §9.7's
    diagnostic: rebuilding on distant blockages is what makes 78% of the
    driving in a failed clutter episode fail to move the robot."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    out = {}
    for commit in (False, True):
        agent = MappedPursuitAgent(sensor="camera64", commit=commit)
        # Start facing away, so the first plan does not know about the box.
        assert agent.start_episode(world, np.array([2.0, 6.0, np.pi]))
        first = agent.path.copy()
        for _ in range(agent.replan_period + 1):
            agent.act(np.array([2.0, 6.0, 0.0]))  # now facing it, three metres off
        out[commit] = (first, agent.path.copy(), agent.plans_refused, agent.replans)

    assert out[False][3] >= 1, "the blockage was never noticed at all"
    assert not np.array_equal(out[False][0], out[False][1]), "the route should be rebuilt"
    assert np.array_equal(out[True][0], out[True][1]), "the committed route was abandoned"
    assert out[True][2] >= 1, "no replan was refused, so nothing was committed to"


def test_the_interval_gate_refuses_a_rebuild_that_comes_too_soon():
    """§9.7's diagnosis is that the argument in clutter is near-field and fast:
    a rebuild every two and a half steps. Gating on time is the only gate that
    can bind, and it must stop a second rebuild arriving immediately after the
    first while leaving one that waits."""
    world = _open_world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    agent = MappedPursuitAgent(sensor="lidar360", commit=True)
    agent.commit_distance, agent.commit_interval = 0.6, 10
    pose = np.array([2.0, 6.0, 0.0])
    assert agent.start_episode(world, pose)
    for _ in range(3):
        agent.act(pose)

    # Count the plans actually searched for, which is what the gate stops.
    # Refusals alone cannot tell the two cases apart: once the interval passes,
    # a route rebuilt from the same pose and the same map comes out identical
    # and is then refused by the hysteresis instead.
    searched = []
    real_reset = agent.reset

    def spy(*args, **kwargs):
        searched.append(1)
        return real_reset(*args, **kwargs)

    agent.reset = spy
    committed = agent.path.copy()
    agent._last_adopted = agent._steps  # it has just settled on this route
    refused = agent.plans_refused
    agent._replan_committed(pose)
    assert searched == [], "it searched for a new route straight away"
    assert np.array_equal(committed, agent.path), "it changed its mind immediately"
    assert agent.plans_refused == refused + 1

    # Once the interval has passed the rebuild is considered again.
    agent._last_adopted = agent._steps - agent.commit_interval
    agent._replan_committed(pose)
    assert len(searched) == 1, "the interval never released the gate"


def test_something_close_ahead_still_changes_the_route():
    """Commitment is not stubbornness: a wall inside the commit distance is
    acted on, the same as without it."""
    world = _open_world(boxes=[(3.2, 4.0, 3.8, 8.0)])
    agent = MappedPursuitAgent(sensor="camera64", commit=True)
    assert agent.start_episode(world, np.array([2.0, 6.0, np.pi]))
    first = agent.path.copy()
    agent.act(np.array([2.0, 6.0, 0.0]))  # the wall is 1.2 m ahead
    assert not np.array_equal(first, agent.path), "a wall a metre ahead was ignored"
    assert (agent.map.clearance(agent._track) >= agent.map.robot_radius - 1e-6).all()


def test_an_unknown_sensor_is_refused():
    with pytest.raises(ValueError, match="unknown sensor"):
        make_sensor("sonar")
