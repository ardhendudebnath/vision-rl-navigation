"""The TurtleBot3 adapter must be the published stack, fed from outside."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.sensors import LidarConfig
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.localisation import OdometryConfig
from vision_nav.training.env_factory import build_env_config

GAZEBO = Path(__file__).resolve().parents[1] / "gazebo_tb3"


def _module():
    spec = importlib.util.spec_from_file_location("tb3_agent", GAZEBO / "tb3_agent.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tb3_agent"] = module
    spec.loader.exec_module(module)
    return module


def _env(condition: str, robot=None, steps: int | None = None):
    _, shift, noise = BENCHMARK_CONDITIONS[condition]
    overrides: dict = {"lidar": {"noise_std": noise}} if noise else {}
    cfg = build_env_config(overrides, split="val", shift=shift, n_worlds=3)
    if robot is not None:
        cfg.robot = robot
    if steps is not None:
        cfg.max_episode_steps = steps
    return cfg, ProceduralNavEnv(cfg)


def test_a_return_keeps_its_place_in_the_world_when_the_origin_moves():
    m = _module()
    bearings = np.array([0.0, np.pi / 2, np.pi, 1.0])
    ranges = np.array([2.0, 1.5, 3.5, 3.5])  # the last two: no return at 3.5 m
    r, b = m.to_centre(ranges, bearings, -0.064, 3.5)
    sensor = np.array([-0.064, 0.0])
    for i in (0, 1):
        world_pt = sensor + ranges[i] * np.array([np.cos(bearings[i]), np.sin(bearings[i])])
        assert np.allclose(r[i] * np.array([np.cos(b[i]), np.sin(b[i])]), world_pt)
    assert r[2] == 3.5 and r[3] == 3.5 and b[3] == pytest.approx(1.0)


def test_non_returns_read_exactly_the_maximum():
    m = _module()
    out = m.clean_ranges(np.array([np.inf, -np.inf, 0.05, 1.0, 3.5, 7.0, np.nan]), 3.5)
    assert list(out) == [3.5, 3.5, 3.5, 1.0, 3.5, 3.5, 3.5]


def test_the_map_is_never_built_from_the_world():
    """Without a scan handed to it, the adapter cannot start: nothing in it
    may fall back to ray-casting the world."""
    m = _module()
    cfg, env = _env("nominal", robot=m.TB3_ROBOT)
    env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    with pytest.raises(RuntimeError, match="no scan"):
        m.TB3Agent().start_episode(env.world, env.robot.pose)


def test_configured_as_published_and_fed_its_scans_it_is_the_published_stack():
    """Sensor at the centre, 6 m, a scan every step, the published odometry
    model, and the very scans the published agent casts for itself: every
    action must be the same, step for step."""
    m = _module()
    cfg, env = _env("nominal")
    env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    world = env.world
    published = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360", odometry=OdometryConfig(),
                                      scan_matching=True, corroborate=True)
    lidar = LidarConfig(n_beams=360, fov=2.0 * np.pi, max_range=6.0, noise_std=0.0)
    bearings = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    adapter = m.TB3Agent(robot=cfg.robot, lidar=lidar, bearings=bearings, lidar_x=0.0,
                         odometry=OdometryConfig())
    caster = m.analytic_lidar(lidar, bearings)
    published.start_episode(world, env.robot.pose)
    adapter.start_episode_with_scan(world, env.robot.pose, caster.scan(world, env.robot.pose))
    for _ in range(150):
        pose, vel = env.robot.pose.copy(), env.robot.velocity.copy()
        a = published.act(pose, vel)
        b = adapter.step(vel, caster.scan(world, pose), truth=pose)
        assert np.array_equal(a, b)
        _, _, done, truncated, _ = env.step(a)
        if done or truncated:
            break
    assert published.replans == adapter.replans


def test_it_drives_a_turtlebot3_through_a_world_in_2d():
    """The 2-D arm, end to end: TurtleBot3 limits, the LDS-01's bearings, a
    scan every other step from the lidar's own position 0.064 m behind the
    axle. With a 6 m range, because the LDS-01's 3.5 m genuinely costs this
    stack worlds -- on this one its pose drifts into an obstacle -- and that is
    what the transfer test measures, not something a unit test should pin."""
    m = _module()
    from dataclasses import replace

    lidar = replace(m.TB3_LIDAR, max_range=6.0)
    cfg, env = _env("nominal", robot=m.TB3_ROBOT, steps=1200)
    env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    world = env.world
    rng = np.random.default_rng(0)
    caster = m.analytic_lidar(lidar)

    def scan_at(pose):
        sensor = pose.copy()
        sensor[:2] += m.TB3_LIDAR_X * np.array([np.cos(pose[2]), np.sin(pose[2])])
        return caster.scan(world, sensor, rng)

    agent = m.TB3Agent(lidar=lidar, odometry=OdometryConfig())
    agent.start_episode_with_scan(world, env.robot.pose, scan_at(env.robot.pose))
    info: dict = {}
    for k in range(cfg.max_episode_steps):
        pose = env.robot.pose.copy()
        ranges = scan_at(pose) if k % 2 == 1 else None
        _, _, done, truncated, info = env.step(agent.step(env.robot.velocity.copy(), ranges, pose))
        if done or truncated:
            break
    assert info.get("is_success"), info
    assert agent.scans_used > 10
    assert max(agent.pose_errors) < 0.2
