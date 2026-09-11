"""Tests for the navigation environment's API contract and dynamics."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv, RewardConfig
from vision_nav.envs.robot import DiffDriveRobot, RobotConfig
from vision_nav.envs.sensors import Lidar2D, LidarConfig
from vision_nav.envs.splits import SEED_BANDS, split_seeds
from vision_nav.envs.world import World, WorldConfig, generate_world


@pytest.fixture
def env():
    return ProceduralNavEnv(NavEnvConfig(world_seeds=split_seeds("train", 8)))


def test_passes_gymnasium_env_checker(env):
    check_env(env, skip_render_check=True)


def test_observation_is_within_declared_space(env):
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), "observation escaped its Box"
        if term or trunc:
            obs, _ = env.reset()


def test_reset_with_explicit_world_seed_is_reproducible(env):
    _, info_a = env.reset(seed=0, options={"world_seed": 42})
    pose_a = env.robot.pose.copy()
    _, info_b = env.reset(seed=999, options={"world_seed": 42})
    assert info_a["world_seed"] == info_b["world_seed"] == 42
    assert np.allclose(pose_a, env.robot.pose)


def test_deterministic_seed_order_repeats_the_same_sequence():
    cfg = NavEnvConfig(world_seeds=[1, 2, 3], deterministic_seed_order=True)
    env = ProceduralNavEnv(cfg)
    first = [env.reset()[1]["world_seed"] for _ in range(6)]
    assert first == [1, 2, 3, 1, 2, 3]


def test_terminal_info_carries_every_field_the_metrics_need(env):
    obs, _ = env.reset(seed=1)
    required = {
        "world_seed",
        "steps",
        "goal_distance",
        "path_length",
        "shortest_path_length",
        "is_success",
        "collision",
    }
    while True:
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            assert required <= set(info)
            break


def test_episode_truncates_at_the_step_limit():
    robot = RobotConfig()
    zero_speed = (
        2.0 * (0.0 - robot.min_linear_vel) / (robot.max_linear_vel - robot.min_linear_vel) - 1.0
    )
    cfg = NavEnvConfig(world_seeds=[5], max_episode_steps=25)
    env = ProceduralNavEnv(cfg)
    env.reset(seed=0)
    # Turning in place can neither reach the goal nor move into an obstacle,
    # so the episode can only end by truncation.
    for _ in range(25):
        _, _, term, trunc, info = env.step(np.array([zero_speed, 1.0]))
        assert not term
    assert trunc
    assert info["steps"] == 25


def test_success_terminates_and_is_reported():
    w = generate_world(0)
    cfg = NavEnvConfig(world_seeds=[0], max_episode_steps=1000)
    env = ProceduralNavEnv(cfg)
    env.reset(seed=0, options={"world_seed": 0})
    # Teleport next to the goal, then drive the last few centimetres.
    goal = env.world.goal
    env.robot.reset(np.array([goal[0] - 0.3, goal[1], 0.0]))
    _, reward, term, _, info = env.step(np.array([1.0, 0.0]))
    assert term and info["is_success"]
    assert reward > 0.0, "reaching the goal must be net positive"


def test_collision_terminates_with_a_penalty():
    cfg = NavEnvConfig(world_seeds=[0])
    env = ProceduralNavEnv(cfg)
    env.reset(seed=0, options={"world_seed": 0})
    world = env.world
    # Aim the robot at an obstacle from just outside it.
    cx, cy, r = world.circles[0]
    env.robot.reset(np.array([cx - (r + world.config.robot_radius + 0.05), cy, 0.0]))
    hit = False
    for _ in range(20):
        _, reward, term, _, info = env.step(np.array([1.0, 0.0]))
        if term:
            hit = info["collision"]
            break
    assert hit, "driving into an obstacle should register a collision"
    assert reward < 0.0, "collision must be net negative"


def test_path_length_accumulates_monotonically(env):
    env.reset(seed=3)
    last = 0.0
    for _ in range(60):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.1]))
        assert info["path_length"] >= last
        last = info["path_length"]
        if term or trunc:
            break


def test_splits_are_disjoint():
    """Overlapping splits would mean evaluating on trained-in worlds."""
    seen: set[int] = set()
    for name in SEED_BANDS:
        seeds = set(split_seeds(name))
        assert not (seeds & seen), f"split {name} overlaps an earlier split"
        seen |= seeds


def test_unimplemented_obs_mode_fails_loudly():
    with pytest.raises(NotImplementedError, match="Phase 4"):
        ProceduralNavEnv(NavEnvConfig(obs_mode="rgbd"))


# ----------------------------------------------------------------------
# Robot and sensor units
# ----------------------------------------------------------------------
def test_straight_drive_covers_the_expected_distance():
    robot = DiffDriveRobot(RobotConfig(max_linear_accel=1e6, dt=0.1))
    robot.reset(np.array([0.0, 0.0, 0.0]))
    for _ in range(10):
        robot.step(np.array([1.0, 0.0]))
    # 1 s at max_linear_vel = 0.6 m/s.
    assert robot.pose[0] == pytest.approx(0.6, rel=1e-6)
    assert robot.pose[1] == pytest.approx(0.0, abs=1e-9)


def test_turning_in_place_does_not_translate():
    cfg = RobotConfig()
    robot = DiffDriveRobot(cfg)
    robot.reset(np.array([1.0, 1.0, 0.0]))
    # action[0] mapping to exactly v = 0 in the asymmetric velocity range.
    zero_speed = 2.0 * (0.0 - cfg.min_linear_vel) / (cfg.max_linear_vel - cfg.min_linear_vel) - 1.0
    for _ in range(20):
        robot.step(np.array([zero_speed, 1.0]))
    assert robot.pose[:2] == pytest.approx([1.0, 1.0], abs=1e-6)
    assert robot.velocity[1] > 0.0, "the robot should still be rotating"


def test_heading_stays_wrapped():
    robot = DiffDriveRobot(RobotConfig())
    robot.reset(np.array([0.0, 0.0, 0.0]))
    for _ in range(200):
        robot.step(np.array([0.0, 1.0]))
        assert -np.pi < robot.pose[2] <= np.pi + 1e-12


def test_lidar_measures_a_known_circle():
    cfg = WorldConfig(width=10.0, height=10.0)
    w = World(
        config=cfg,
        circles=np.array([[7.0, 5.0, 1.0]]),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    lidar = Lidar2D(LidarConfig(n_beams=4, fov=2 * np.pi, max_range=10.0))
    ranges = lidar.scan(w, np.array([5.0, 5.0, 0.0]))
    # Beam angles for a 4-beam full scan are [-pi, -pi/2, 0, pi/2]; the
    # forward beam (index 2) hits the circle's near edge at 7 - 1 - 5 = 1 m.
    assert ranges[2] == pytest.approx(1.0, abs=1e-6)
    # The backward beam runs to the wall at x = 0.
    assert ranges[0] == pytest.approx(5.0, abs=1e-6)


def test_lidar_measures_a_known_box():
    cfg = WorldConfig(width=10.0, height=10.0)
    w = World(
        config=cfg,
        circles=np.zeros((0, 3)),
        boxes=np.array([[7.0, 4.0, 8.0, 6.0]]),
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    lidar = Lidar2D(LidarConfig(n_beams=4, fov=2 * np.pi, max_range=10.0))
    ranges = lidar.scan(w, np.array([5.0, 5.0, 0.0]))
    assert ranges[2] == pytest.approx(2.0, abs=1e-6)


def test_lidar_clips_to_max_range():
    w = generate_world(2)
    lidar = Lidar2D(LidarConfig(n_beams=16, max_range=1.5))
    ranges = lidar.scan(w, w.start)
    assert ranges.min() >= 0.0
    assert ranges.max() <= 1.5 + 1e-9


def test_geodesic_shaping_toggle_selects_the_right_distance():
    """``use_geodesic_progress`` must actually switch the shaping signal.

    The ablation in the report depends on this flag doing what it says; a
    silently-ignored toggle would make the two arms of the ablation identical.
    """
    geo = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    euc = ProceduralNavEnv(
        NavEnvConfig(world_seeds=[0], reward=RewardConfig(use_geodesic_progress=False))
    )
    geo.reset(seed=0, options={"world_seed": 0})
    euc.reset(seed=0, options={"world_seed": 0})

    pos = geo.robot.position
    euclid = float(np.linalg.norm(pos - geo.world.goal))

    assert euc._progress_distance(pos) == pytest.approx(euclid)
    # The geodesic distance must route around obstacles, never cut through.
    assert geo._progress_distance(pos) >= euclid - 1e-6
