"""Tests for the SLAM back end.

The solver is tested on graphs whose answer is known by construction rather than
on episodes, because a pose-graph bug does not announce itself: a wrong Jacobian
still converges to *something*, and the something looks like a trajectory. Each
test below fixes an answer independently of the code that computes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.mapping.posegraph import (
    PoseGraph,
    PoseGraphConfig,
    compose_pose,
    relative_pose,
    scan_points,
)


def test_relative_and_compose_are_inverses():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
        b = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
        back = compose_pose(a, relative_pose(a, b))
        assert back[:2] == pytest.approx(b[:2])
        assert np.cos(back[2] - b[2]) == pytest.approx(1.0)


def test_a_consistent_graph_is_already_optimal():
    """Odometry edges that agree with the poses leave nothing to solve. If this
    moves, the residual or its Jacobian has a sign error."""
    graph = PoseGraph()
    scan = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    for x in (0.0, 0.6, 1.2, 1.8):
        graph.add_keyframe(np.array([x, 0.0, 0.0]), scan)
    before = [p.copy() for p in graph.poses]
    graph.optimise()
    for old, new in zip(before, graph.poses, strict=True):
        assert new == pytest.approx(old, abs=1e-6)


def test_a_closure_pulls_a_drifted_loop_shut():
    """Four poses around a square whose odometry says it does not quite close.
    A closure saying node 0 and node 3 are the same place must distribute that
    error around the loop, and the gap must shrink."""
    graph = PoseGraph()
    scan = np.array([[1.0, 0.0], [0.0, 1.0]])
    poses = [np.array([0.0, 0.0, 0.0]),
             np.array([1.0, 0.0, np.pi / 2]),
             np.array([1.0, 1.0, np.pi]),
             np.array([0.25, 0.9, -np.pi / 2])]      # should have returned to (0, 1)
    for p in poses:
        graph.add_keyframe(p, scan)
    # The closure: from node 0, node 3 is 1 m ahead along y with a -90 deg turn.
    measured = np.array([0.0, 1.0, -np.pi / 2])
    graph.edges.append((0, 3, measured, 1.0, 2.0))

    def gap():
        return float(np.linalg.norm(
            compose_pose(graph.poses[0], measured)[:2] - graph.poses[3][:2]))

    before = gap()
    graph.optimise()
    assert before > 0.2
    assert gap() < before / 4.0
    # The gauge is node 0: the map frame is where the robot started.
    assert graph.poses[0] == pytest.approx(poses[0], abs=1e-4)


def test_the_solve_leaves_a_lone_odometry_chain_alone():
    """With no closure there is nothing to redistribute, so a drifting chain is
    the best estimate of itself. A back end that 'improves' this is inventing."""
    graph = PoseGraph()
    scan = np.array([[1.0, 0.0], [0.0, 1.0]])
    rng = np.random.default_rng(3)
    for _ in range(6):
        last = graph.poses[-1] if graph.poses else np.zeros(3)
        step = np.array([0.6, 0.0, 0.0]) + rng.normal(0, 0.02, 3)
        graph.add_keyframe(compose_pose(last, step), scan)
    before = [p.copy() for p in graph.poses]
    graph.optimise()
    for old, new in zip(before, graph.poses, strict=True):
        assert new == pytest.approx(old, abs=1e-5)


def test_keyframes_are_added_on_distance_and_on_rotation():
    graph = PoseGraph(PoseGraphConfig(keyframe_distance=0.5, keyframe_angle=0.5))
    scan = np.array([[1.0, 0.0]])
    assert graph.due(np.zeros(3)), "the first pose is always a keyframe"
    graph.add_keyframe(np.zeros(3), scan)
    assert not graph.due(np.array([0.2, 0.0, 0.0]))
    assert graph.due(np.array([0.6, 0.0, 0.0])), "far enough"
    assert graph.due(np.array([0.1, 0.0, 0.7])), "turned enough"


def test_a_closure_is_found_between_two_views_of_the_same_place():
    """Two keyframes looking at the same L-shaped corner from poses that the
    odometry has 0.25 m wrong. The closure must be accepted and must recover
    roughly the true offset."""
    cfg = PoseGraphConfig(loop_min_gap=1, loop_radius=3.0)
    graph = PoseGraph(cfg)
    # A corner of points, in the world.
    wall = np.concatenate([
        np.stack([np.linspace(1.0, 3.0, 40), np.full(40, 2.0)], axis=-1),
        np.stack([np.full(40, 3.0), np.linspace(0.0, 2.0, 40)], axis=-1)])

    def seen_from(pose):
        c, s = np.cos(-pose[2]), np.sin(-pose[2])
        d = wall - pose[:2]
        return np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], axis=-1)

    true_a = np.array([1.0, 0.5, 0.0])
    true_b = np.array([1.5, 0.7, 0.2])
    graph.add_keyframe(true_a, seen_from(true_a))
    # The second node's pose is wrong by 0.25 m: the scan is taken from where
    # the robot really is, the pose recorded is where it thinks it is.
    graph.add_keyframe(true_b + np.array([0.25, -0.1, 0.05]), seen_from(true_b))

    assert graph.find_closure(1), "the same corner seen twice should close"
    _, _, measured, _, _ = graph.edges[-1]
    truth = relative_pose(true_a, true_b)
    assert measured[:2] == pytest.approx(truth[:2], abs=0.08)
    assert measured[2] == pytest.approx(truth[2], abs=0.08)


def test_a_closure_is_refused_between_places_that_do_not_match():
    """The failure mode of loop closure is confident nonsense, so the score
    gate has to refuse a pair that does not fit."""
    cfg = PoseGraphConfig(loop_min_gap=1, loop_radius=10.0)
    graph = PoseGraph(cfg)
    rng = np.random.default_rng(7)
    graph.add_keyframe(np.zeros(3), rng.uniform(-2, 2, (60, 2)))
    graph.add_keyframe(np.array([0.5, 0.0, 0.0]), rng.uniform(-2, 2, (60, 2)))
    assert not graph.find_closure(1)
    assert graph.closures_rejected == 1


def _world():
    from vision_nav.envs.world import World, WorldConfig

    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.array([[5.0, 4.5, 5.6, 7.5]]),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def test_the_back_end_is_off_by_default_and_changes_nothing_when_off():
    """The identity control. Every published result runs the front end alone,
    so with ``pose_graph=False`` the agent must be the arm those numbers came
    from, step for step."""
    from vision_nav.agents.localised import LocalisedPursuitAgent
    from vision_nav.mapping.localisation import OdometryConfig

    world = _world()
    plain = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                  scan_matching=True)
    off = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                scan_matching=True, pose_graph=False)
    assert plain.graph is None and off.graph is None
    for agent in (plain, off):
        assert agent.start_episode(world, world.start)
    pose, velocity = world.start.copy(), np.array([0.4, 0.1])
    for _ in range(25):
        assert np.allclose(plain.act(pose, velocity), off.act(pose, velocity))
    assert off._history == [], "no history is kept without a back end"
    assert off.map_rebuilds == 0


def test_rebuilding_an_uncorrected_trajectory_reproduces_the_same_map():
    """The rebuild replays stored scans, and its docstring claims the replay is
    faithful. So a rebuild that corrects nothing must land on the same map --
    including the one scan taken in ``start_episode``, which is re-sampled from
    an identically seeded rng rather than replayed."""
    from vision_nav.agents.localised import LocalisedPursuitAgent
    from vision_nav.mapping.localisation import OdometryConfig
    from vision_nav.mapping.posegraph import PoseGraphConfig

    world = _world()
    # Keyframes often, optimisation never: the poses cannot move.
    cfg = PoseGraphConfig(keyframe_distance=0.2, keyframe_angle=0.2,
                          optimise_every=10 ** 6)
    agent = LocalisedPursuitAgent(sensor="lidar360", noise_std=0.02,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  pose_graph=True, graph_config=cfg)
    assert agent.start_episode(world, world.start)
    pose, velocity = world.start.copy(), np.array([0.4, 0.05])
    for _ in range(30):
        agent.act(pose, velocity)
    assert agent.graph.optimisations == 0
    before = agent.map.grid.copy()
    agent._rebuild_map()
    assert np.array_equal(before, agent.map.grid)
    assert agent.map_rebuilds == 1


def test_a_keyframed_episode_builds_a_graph_and_can_optimise():
    """End to end: the back end actually engages when driven, and leaves the
    estimate finite and near the pose it was carried from."""
    from vision_nav.agents.localised import LocalisedPursuitAgent
    from vision_nav.mapping.localisation import OdometryConfig
    from vision_nav.mapping.posegraph import PoseGraphConfig

    world = _world()
    cfg = PoseGraphConfig(keyframe_distance=0.25, keyframe_angle=0.3,
                          optimise_every=2, loop_min_gap=3, loop_radius=3.0)
    agent = LocalisedPursuitAgent(sensor="lidar360", odometry=OdometryConfig(),
                                  scan_matching=True, pose_graph=True,
                                  graph_config=cfg)
    assert agent.start_episode(world, world.start)
    pose = world.start.copy()
    for _ in range(40):
        pose = pose.copy()
        pose[0] += 0.06                      # drive the truth forward
        agent.act(pose, np.array([0.6, 0.0]))
    assert len(agent.graph.poses) >= 3, "keyframes should accumulate"
    assert agent.graph.optimisations >= 1
    assert np.all(np.isfinite(agent.pose))
    assert len(agent._history) == 40


def test_scan_points_drops_beams_that_hit_nothing():
    class _Config:
        max_range = 6.0

    class _Sensor:
        config = _Config()
        _angles = np.array([0.0, np.pi / 2, np.pi])

    ranges = np.array([1.0, 6.0, 2.0])          # the middle beam hit nothing
    pts = scan_points(ranges, _Sensor())
    assert len(pts) == 2
    assert pts[0] == pytest.approx([1.0, 0.0])
    assert pts[1] == pytest.approx([-2.0, 0.0], abs=1e-9)
