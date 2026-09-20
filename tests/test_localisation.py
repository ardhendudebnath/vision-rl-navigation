"""Tests for the pose the robot has to estimate.

The claim these have to protect is narrow and easy to break by accident: the
control path sees the estimate and never the truth, and with the noise off the
estimate *is* the truth, bit for bit, so the experiment's identity arm means
something. The rest measure what the two pieces are for -- odometry drifts, scan
matching drags it back -- against bars taken from measurements, not invented.
"""

from __future__ import annotations

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.agents.mapped import MappedPursuitAgent, make_sensor
from vision_nav.envs.robot import DiffDriveRobot, RobotConfig
from vision_nav.envs.world import World, WorldConfig
from vision_nav.mapping.localisation import (
    DeadReckoning,
    OdometryConfig,
    ScanMatcher,
)
from vision_nav.mapping.occupancy import OccupancyMap


def _world(boxes=(), start=(2.0, 6.0, 0.0), goal=(10.0, 6.0)):
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
                 start=np.array(start, dtype=float), goal=np.array(goal, dtype=float))


CLUTTER = [(5.0, 4.5, 5.6, 7.5), (7.5, 2.0, 8.1, 5.5), (3.0, 7.5, 6.5, 8.1)]


def _drive(agent, world, steps=120, robot_config=None):
    """Run an agent in closed loop on a real robot, returning the true poses."""
    robot = DiffDriveRobot(robot_config or RobotConfig())
    robot.reset(world.start)
    assert agent.start_episode(world, robot.pose.copy())
    poses = [robot.pose.copy()]
    for _ in range(steps):
        if isinstance(agent, LocalisedPursuitAgent):
            action = agent.act(robot.pose.copy(), robot.velocity.copy())
        else:
            action = agent.act(robot.pose.copy())
        robot.step(action)
        poses.append(robot.pose.copy())
    return np.asarray(poses)


# --- the integrator ----------------------------------------------------------
def test_perfect_encoders_reproduce_the_robots_own_pose_exactly():
    """Not "to a tolerance": the same arithmetic on the same numbers. Anything
    looser and the identity arm of the experiment would drift for reasons that
    have nothing to do with odometry."""
    robot = DiffDriveRobot(RobotConfig())
    robot.reset(np.array([1.5, 2.5, 0.3]))
    odom = DeadReckoning(RobotConfig())
    odom.reset(robot.pose.copy())
    rng = np.random.default_rng(0)
    for _ in range(200):
        robot.step(rng.uniform(-1.0, 1.0, size=2))
        odom.update(robot.velocity)
        assert np.array_equal(odom.pose, robot.pose), (odom.pose, robot.pose)


def test_noisy_encoders_drift_further_the_further_the_robot_drives():
    """No bar on the drift itself -- that is the experiment's business, and a
    number here would only pin one draw of a random variable. What has to hold
    is the shape: it grows with distance driven, and it repeats for a seed."""
    def run(seed, steps):
        robot = DiffDriveRobot(RobotConfig())
        robot.reset(np.zeros(3))
        odom = DeadReckoning(RobotConfig(), OdometryConfig(),
                             np.random.default_rng(seed))
        odom.reset(robot.pose.copy())
        for _ in range(steps):
            robot.step(np.array([1.0, 0.05]))  # a long, gentle arc
            odom.update(robot.velocity)
        return odom.pose, float(np.linalg.norm(odom.pose[:2] - robot.pose[:2]))

    first, near = run(1, 50)
    again, near_again = run(1, 50)
    _, far = run(1, 200)
    other, _ = run(2, 50)
    assert np.array_equal(first, again), "the same seed drifted two different ways"
    assert near_again == near
    assert not np.array_equal(first, other), "two seeds drifted identically"
    assert 0.0 < near < far, (near, far)


def test_the_systematic_part_is_a_property_of_the_robot_not_of_the_step():
    """A bias drawn once per episode: two runs of the same seed pull the same
    way, and turning it off leaves only the random walk."""
    def drift(config, seed):
        robot = DiffDriveRobot(RobotConfig())
        robot.reset(np.zeros(3))
        odom = DeadReckoning(RobotConfig(), config, np.random.default_rng(seed))
        odom.reset(robot.pose.copy())
        for _ in range(200):
            robot.step(np.array([1.0, 0.0]))  # straight, so a pull shows up
            odom.update(robot.velocity)
        return float(odom.pose[1] - robot.pose[1])

    unbiased = OdometryConfig(bias_trans=0.0, bias_rot_per_m=0.0)
    biased = [drift(OdometryConfig(), s) for s in range(8)]
    random_only = [drift(unbiased, s) for s in range(8)]
    assert np.std(biased) > 3 * np.std(random_only), (np.std(biased), np.std(random_only))


# --- the matcher -------------------------------------------------------------
def test_matching_walks_a_displaced_pose_back_onto_the_map():
    """The unit test of the search itself: build a map from a known pose, hand
    the matcher that pose displaced, and require the displacement back.

    Not in one pass. The prior deliberately makes each correction partial --
    a single match is worth about 0.03 m of accuracy and no more, so taking one
    whole is how the estimate ends up chasing noise. Ten passes stand for the
    ten control periods a real second of driving would give it."""
    world = _world(boxes=CLUTTER)
    omap = OccupancyMap(world, make_sensor("lidar32"))
    truth = np.array([4.0, 6.0, 0.2])
    for _ in range(3):
        omap.integrate(truth)
    ranges = omap.scan(truth)

    matcher = ScanMatcher()
    for offset in (np.array([0.12, -0.07, 0.0]), np.array([-0.1, 0.05, 0.04])):
        pose = truth + offset
        errors = [float(np.linalg.norm(pose[:2] - truth[:2]))]
        for _ in range(10):
            pose = matcher.correct(pose, ranges, omap)
            errors.append(float(np.linalg.norm(pose[:2] - truth[:2])))
        # It settles about half a cell out and stays there: the map quantises
        # surfaces to cell centres, so that is the accuracy on offer.
        assert errors[-1] < errors[0] / 2.0, errors
        assert errors[-1] <= omap.resolution, errors[-1]
        assert abs(float(pose[2] - truth[2])) < 0.02, pose[2] - truth[2]


def test_matching_refuses_when_there_is_nothing_to_match_against():
    """An empty map scores every candidate the same. Moving the pose on that
    evidence is how a scan matcher walks a robot into a wall."""
    world = _world()
    omap = OccupancyMap(world, make_sensor("lidar32"))
    omap.integrate(np.array([2.0, 6.0, 0.0]))
    matcher = ScanMatcher()
    pose = np.array([2.0, 6.0, 0.0])
    out = matcher.correct(pose, omap.scan(pose), omap)
    assert np.array_equal(out, pose)
    assert matcher.skipped == 1 and matcher.corrections == 0


def test_the_likelihood_field_is_rebuilt_only_when_the_map_changes():
    world = _world(boxes=CLUTTER)
    omap = OccupancyMap(world, make_sensor("lidar32"))
    omap.integrate(np.array([4.0, 6.0, 0.0]))
    matcher = ScanMatcher()
    first = matcher.likelihood_field(omap)
    assert matcher.likelihood_field(omap) is first, "rebuilt with the map unchanged"
    omap.integrate(np.array([6.0, 3.0, 1.0]))
    assert matcher.likelihood_field(omap) is not first, "kept a field for an older map"


def test_the_field_peaks_on_mapped_surfaces_and_decays_off_them():
    """What makes the search able to see which way to move: full score on a
    mapped cell, less the further off it a return lands, nothing at all out
    past 2 sigma -- so an unexplained return cannot drag the pose anywhere."""
    world = _world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    omap = OccupancyMap(world, make_sensor("lidar32"))
    omap.integrate(np.array([4.0, 6.0, 0.0]))
    matcher = ScanMatcher()
    field = matcher.likelihood_field(omap)
    occupied = np.argwhere(omap.grid == 1)
    assert len(occupied) > 5
    assert all(field[tuple(cell)] == 1.0 for cell in occupied)

    surface = omap.grid_to_world(occupied[len(occupied) // 2])
    off = [float(field[tuple(omap.world_to_grid(surface - np.array([d, 0.0])))])
           for d in (0.1, 0.2, 0.4)]
    assert 1.0 > off[0] > off[1] > off[2] == 0.0, off
    assert off[2] == 0.0, "a return 0.4 m from anything mapped still scored"


# --- the agent ---------------------------------------------------------------
def test_perfect_odometry_reproduces_the_mapped_agent_step_for_step():
    """The identity control. Same world, same sensor noise, same everything:
    the localised agent with perfect encoders must be the mapped agent, or a
    difference in the experiment could be the plumbing rather than the pose."""
    world = _world(boxes=CLUTTER)
    given = MappedPursuitAgent(sensor="lidar32", noise_std=0.05)
    estimated = LocalisedPursuitAgent(sensor="lidar32", noise_std=0.05, odometry=None)

    robot = DiffDriveRobot(RobotConfig())
    robot.reset(world.start)
    assert given.start_episode(world, robot.pose.copy())
    assert estimated.start_episode(world, robot.pose.copy())
    for step in range(150):
        a = given.act(robot.pose.copy())
        b = estimated.act(robot.pose.copy(), robot.velocity.copy())
        assert np.array_equal(a, b), f"diverged at step {step}: {a} vs {b}"
        robot.step(a)
    assert max(estimated.pose_errors) == 0.0


def test_the_planner_replans_from_where_the_robot_believes_it_is():
    """A leak test for the planning half. Displace the belief by half a metre,
    put the agent in the state it reaches after a failed plan -- which is where
    it replans from scratch -- and the new plan must start at the belief."""
    world = _world(boxes=CLUTTER)
    agent = LocalisedPursuitAgent(sensor="lidar32")
    truth = world.start.copy()
    assert agent.start_episode(world, truth)
    agent._odom.pose = truth + np.array([0.0, 0.5, 0.0])
    agent._track = None
    agent.act(truth, np.zeros(2))
    assert agent.path is not None
    np.testing.assert_allclose(agent.path[0], agent.pose[:2])
    assert abs(agent.path[0][1] - truth[1]) > 0.4, "the plan started at the true pose"


def test_the_controller_steers_from_the_belief_not_from_the_robot():
    """And a leak test for the tracking half. The plan runs due east along
    y = 6. A robot that believes it has drifted north of it must steer back
    south; one reading its true pose would drive straight on."""
    world = _world()
    agent = LocalisedPursuitAgent(sensor="lidar32")
    truth = world.start.copy()
    assert agent.start_episode(world, truth)
    straight = agent.act(truth, np.zeros(2))
    assert abs(straight[1]) < 1e-6, straight

    agent._odom.pose = truth + np.array([0.0, 0.5, 0.0])
    steered = agent.act(truth, np.zeros(2))
    assert steered[1] < -0.1, "a half-metre error in the belief did not reach the wheels"


def test_the_sensor_is_still_bolted_to_the_robot_not_to_the_belief():
    """The scan has to come from where the robot physically is. Believing it is
    somewhere else must move where the returns are *written*, not what they
    are: a wall a metre ahead is mapped a metre ahead of the belief."""
    world = _world(boxes=[(5.0, 4.5, 5.6, 7.5)])
    truth = np.array([4.0, 6.0, 0.0])
    agent = LocalisedPursuitAgent(sensor="lidar32")
    assert agent.start_episode(world, truth)
    agent._odom.pose = truth + np.array([0.0, 2.0, 0.0])  # believes it is 2 m north
    agent.act(truth, np.zeros(2))
    mapped = agent.map.grid_to_world(np.argwhere(agent.map.grid == 1))
    ahead_of_belief = np.abs(mapped - np.array([4.95, 8.0])).max(axis=1) < 0.2
    assert ahead_of_belief.any(), "the wall was not mapped where the robot believed it was"
    assert abs(agent.pose_errors[-1] - 2.0) < 1e-9


def test_drift_is_seeded_by_the_world_so_an_episode_repeats():
    world = _world(boxes=CLUTTER)

    def run():
        agent = LocalisedPursuitAgent(sensor="lidar32", odometry=OdometryConfig())
        _drive(agent, world, steps=60)
        return agent.pose_errors[-1]

    assert run() == run()


def test_scan_matching_holds_a_pose_that_odometry_loses():
    """Both arms of the experiment, on one cluttered world. The bar is a
    measurement, not a wish: over twelve `val`-band worlds per condition,
    median final pose error went 0.292 m -> 0.052 m in dense clutter and
    0.289 m -> 0.114 m in narrow corridors. A bar at 0.6 of the odometry error
    is loose against ratios of 0.18 and 0.39, and this world is cluttered.
    Open worlds are where matching has least to work with, and there the gain
    is smaller -- which is a finding, not a slack bar, and §9.3 reports it."""
    world = _world(boxes=CLUTTER)
    drifting = LocalisedPursuitAgent(sensor="lidar32", odometry=OdometryConfig())
    matched = LocalisedPursuitAgent(sensor="lidar32", odometry=OdometryConfig(),
                                    scan_matching=True)
    _drive(drifting, world, steps=150)
    _drive(matched, world, steps=150)
    assert drifting.pose_errors[-1] > 0.1, "the odometry did not drift enough to test"
    assert matched.pose_errors[-1] < 0.6 * drifting.pose_errors[-1], (
        matched.pose_errors[-1], drifting.pose_errors[-1])
    assert matched.matcher.corrections > 100
