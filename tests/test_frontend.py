"""Tests for the correlative front end.

Each test fixes its answer by construction -- real walls, a real sensor, and an
estimate deliberately put somewhere it is not -- because a matcher can converge
confidently to the wrong place and look fine doing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.agents.mapped import make_sensor
from vision_nav.envs.world import World, WorldConfig
from vision_nav.mapping.frontend import CorrelativeConfig, CorrelativeFrontEnd


def _world() -> World:
    # Arena walls plus three boxes: enough structure that a scan constrains
    # position and heading both.
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.array([[5.0, 4.5, 5.6, 7.5], [7.5, 2.0, 8.1, 3.4],
                                 [2.0, 8.0, 3.5, 8.6]]),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def _primed(world, sensor):
    """A front end holding one keyframe, taken where the robot really was."""
    fe = CorrelativeFrontEnd()
    a = np.array([3.0, 5.0, 0.0])
    fe.update(a, sensor.scan(world, a), sensor)
    return fe


def test_it_recovers_a_translation_the_old_window_could_not_reach():
    """The published matcher reaches +/-0.08 m. A drift of 0.2 m is the regime
    §9.18 found the failures in, and this window is +/-0.25 m."""
    world, sensor = _world(), make_sensor("lidar360")
    fe = _primed(world, sensor)
    truth = np.array([3.8, 5.3, 0.1])
    estimate = truth + np.array([0.18, -0.12, 0.0])
    out = fe.update(estimate, sensor.scan(world, truth), sensor)
    assert fe.matches == 1
    assert float(np.linalg.norm(out[:2] - truth[:2])) < 0.03, out


def test_it_recovers_a_heading_fifteen_times_the_old_reach():
    """The published matcher's heading window is +/-0.024 rad; this one is
    +/-0.349. A heading 0.2 rad wrong is well past the first and well inside
    the second."""
    world, sensor = _world(), make_sensor("lidar360")
    fe = _primed(world, sensor)
    truth = np.array([3.8, 5.3, 0.1])
    estimate = truth + np.array([0.0, 0.0, 0.2])
    out = fe.update(estimate, sensor.scan(world, truth), sensor)
    assert abs(float(out[2] - truth[2])) < 0.02, out
    assert float(np.linalg.norm(out[:2] - truth[:2])) < 0.05, out


def test_a_correct_estimate_stays_where_it_is():
    world, sensor = _world(), make_sensor("lidar360")
    fe = _primed(world, sensor)
    truth = np.array([3.8, 5.3, 0.1])
    out = fe.update(truth.copy(), sensor.scan(world, truth), sensor)
    assert float(np.linalg.norm(out[:2] - truth[:2])) < 0.02
    assert abs(float(out[2] - truth[2])) < 0.01


def test_between_keyframes_the_odometry_carries_the_pose():
    """Matching every scan gives match noise ten chances a metre to build up,
    which is what the published matcher's prior exists to suppress. Here a
    pose that has not moved far enough is returned untouched and not matched."""
    world, sensor = _world(), make_sensor("lidar360")
    fe = _primed(world, sensor)
    near = np.array([3.2, 5.0, 0.05])
    assert not fe.due(near)
    out = fe.update(near + np.array([0.1, 0.0, 0.0]), sensor.scan(world, near), sensor)
    assert np.allclose(out, near + np.array([0.1, 0.0, 0.0]))
    assert fe.matches == 0 and len(fe.buffer) == 1


def test_the_buffer_holds_only_the_last_ten_keyframes():
    world, sensor = _world(), make_sensor("lidar360")
    fe = CorrelativeFrontEnd()
    for k in range(15):
        pose = np.array([2.0 + 0.6 * (k % 6), 5.0 + 0.6 * (k // 6), 0.0])
        fe.update(pose, sensor.scan(world, pose), sensor)
    assert len(fe.buffer) == CorrelativeConfig().buffer_size


def test_too_little_to_match_against_is_refused_not_guessed():
    """A target with almost no points says nothing about where the robot is."""
    world, sensor = _world(), make_sensor("lidar360")
    fe = CorrelativeFrontEnd(CorrelativeConfig(min_points=10 ** 6))
    a = np.array([3.0, 5.0, 0.0])
    fe.update(a, sensor.scan(world, a), sensor)
    b = np.array([3.8, 5.3, 0.1])
    out = fe.update(b + np.array([0.1, 0.0, 0.0]), sensor.scan(world, b), sensor)
    assert np.allclose(out, b + np.array([0.1, 0.0, 0.0]))
    assert fe.matches == 0


def test_the_map_front_end_is_the_default_and_unchanged():
    """The identity control. Every published result localises with the map
    matcher, so the default must be that arm, step for step."""
    from vision_nav.agents.localised import LocalisedPursuitAgent
    from vision_nav.mapping.localisation import OdometryConfig

    world = _world()
    plain = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                  scan_matching=True)
    explicit = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                     scan_matching=True, front_end="map")
    assert plain.front_end == "map" and plain.correlative is None
    for agent in (plain, explicit):
        assert agent.start_episode(world, world.start)
    pose, velocity = world.start.copy(), np.array([0.4, 0.1])
    for _ in range(20):
        assert np.allclose(plain.act(pose, velocity), explicit.act(pose, velocity))


def test_the_correlative_front_end_replaces_the_map_matcher():
    """Two matchers correcting one pose would make neither attributable, so
    choosing the correlative front end removes the map matcher."""
    from vision_nav.agents.localised import LocalisedPursuitAgent
    from vision_nav.mapping.localisation import OdometryConfig

    world = _world()
    agent = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                  scan_matching=True, front_end="correlative")
    assert agent.matcher is None and agent.correlative is not None
    assert agent.start_episode(world, world.start)
    pose = world.start.copy()
    for _ in range(30):
        pose = pose.copy()
        pose[0] += 0.05
        agent.act(pose, np.array([0.5, 0.0]))
    assert len(agent.correlative.buffer) >= 2, "keyframes should accumulate"
    assert np.all(np.isfinite(agent.pose))


def test_an_unknown_front_end_is_refused():
    from vision_nav.agents.localised import LocalisedPursuitAgent

    with pytest.raises(ValueError):
        LocalisedPursuitAgent(sensor="lidar360", scan_matching=True, front_end="icp")


def test_the_penalty_is_karto_weak():
    """Karto's tie-break costs at most 2.5% of the response at the window's
    edge, so the response decides and the odometry only breaks ties."""
    cfg = CorrelativeConfig()
    worst = 1.0 - cfg.penalty_gain * cfg.window ** 2 / cfg.distance_variance_penalty
    assert worst == pytest.approx(0.975)
