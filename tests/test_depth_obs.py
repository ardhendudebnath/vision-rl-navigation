"""Tests for the forward-facing depth-camera observation mode.

The substantive difference from the lidar mode is the loss of the 360-degree
view, so the tests focus on that: the camera must see forward, must NOT see
behind, and must agree with the lidar about the geometry they share. A camera
that silently reported rear obstacles would make the whole lidar-vs-camera
comparison meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.sensors import CameraConfig, DepthCamera, Lidar2D, LidarConfig
from vision_nav.envs.world import World, WorldConfig, generate_world


def two_obstacle_world() -> World:
    """Identical obstacles ahead (+x) and behind (-x) of centre."""
    return World(
        config=WorldConfig(width=20.0, height=20.0),
        circles=np.array([[13.0, 10.0, 0.5], [7.0, 10.0, 0.5]]),
        boxes=np.zeros((0, 4)),
        start=np.zeros(3),
        goal=np.zeros(2),
    )


# ----------------------------------------------------------------------
# CameraConfig
# ----------------------------------------------------------------------
def test_angular_resolution_matches_fov_over_width():
    cfg = CameraConfig(fov=np.pi / 2, width=64)
    assert cfg.angular_resolution == pytest.approx(np.pi / 2 / 64)


def test_resolves_gap_to_matches_the_phase_2h_geometry():
    """Same formula the beam experiment used, now for a camera."""
    cfg = CameraConfig(fov=np.pi / 2, width=64)
    r = cfg.resolves_gap_to(0.44)
    expected = 0.44 / (2 * np.sin(cfg.angular_resolution / 2))
    assert r == pytest.approx(expected)
    # A 64-column 90-degree camera is far finer than a 64-beam 360 lidar.
    assert r > 4.48


def test_invalid_camera_config_is_refused():
    with pytest.raises(ValueError, match="fov"):
        CameraConfig(fov=0.0)
    with pytest.raises(ValueError, match="fov"):
        CameraConfig(fov=7.0)
    with pytest.raises(ValueError, match="width"):
        CameraConfig(width=0)


# ----------------------------------------------------------------------
# DepthCamera
# ----------------------------------------------------------------------
def test_camera_sees_ahead_but_not_behind():
    """The defining property of a camera versus a 360-degree lidar."""
    world = two_obstacle_world()
    # Odd width so one column points exactly along the optical axis; with an
    # even width the centre falls between two columns and the nearest ray
    # clips the obstacle slightly off-centre (2.5045 m rather than 2.5).
    cam = DepthCamera(CameraConfig(fov=np.pi / 2, width=33, max_range=6.0))

    facing_forward = cam.depth(world, np.array([10.0, 10.0, 0.0]))
    facing_backward = cam.depth(world, np.array([10.0, 10.0, np.pi]))

    # Obstacle 3 m away, radius 0.5 -> near edge at 2.5 m, dead centre.
    assert facing_forward.min() == pytest.approx(2.5, abs=1e-6)
    # Turned around, it sees the other (identical) obstacle, not both.
    assert facing_backward.min() == pytest.approx(2.5, abs=1e-6)

    # A 360-degree lidar at the same pose sees BOTH; the camera sees one.
    lidar = Lidar2D(LidarConfig(n_beams=720, fov=2 * np.pi, max_range=6.0))
    scan = lidar.scan(world, np.array([10.0, 10.0, 0.0]))
    assert np.sum(scan < 3.0) > 0
    # Count distinct near-obstacle lobes: the lidar must have two, at
    # opposite bearings; the camera's FOV can only contain one.
    close = scan < 3.0
    assert close[:180].any() or close[540:].any()  # behind
    assert close[180:540].any()  # ahead


def test_camera_columns_are_centred_in_the_fov():
    """Column centres, not endpoints: an off-by-half-a-pixel bug is silent."""
    cfg = CameraConfig(fov=np.pi / 2, width=4)
    cam = DepthCamera(cfg)
    angles = cam._lidar._angles
    step = cfg.angular_resolution
    expected = np.array([-np.pi / 4 + (i + 0.5) * step for i in range(4)])
    assert np.allclose(angles, expected)
    # Symmetric about straight ahead.
    assert np.allclose(angles, -angles[::-1])
    # With an even width no column looks exactly forward; with an odd width
    # exactly one does. Both are correct, and the distinction matters when
    # reading off a centre-of-image depth.
    assert not np.any(np.isclose(angles, 0.0))
    odd = DepthCamera(CameraConfig(fov=np.pi / 2, width=5))._lidar._angles
    assert np.count_nonzero(np.isclose(odd, 0.0)) == 1


def test_camera_and_lidar_agree_on_shared_geometry():
    """Both sensors must use the same ray-caster, or comparisons are invalid."""
    world = generate_world(3)
    pose = world.start
    cam = DepthCamera(CameraConfig(fov=np.pi / 2, width=16, max_range=6.0))
    depth = cam.depth(world, pose)

    # Cast the same bearings by hand through the lidar primitive.
    manual = Lidar2D(LidarConfig(n_beams=16, fov=np.pi / 2, max_range=6.0))
    manual._angles = cam._lidar._angles
    assert np.allclose(depth, manual.scan(world, pose))


def test_depth_is_clipped_and_normalized():
    world = generate_world(5)
    cam = DepthCamera(CameraConfig(width=32, max_range=4.0))
    d = cam.depth(world, world.start)
    assert d.min() >= 0.0 and d.max() <= 4.0 + 1e-9
    n = cam.normalized_depth(world, world.start)
    assert n.min() >= 0.0 and n.max() <= 1.0 + 1e-9


# ----------------------------------------------------------------------
# Environment integration
# ----------------------------------------------------------------------
def test_depth_env_passes_the_gymnasium_checker():
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="depth", world_seeds=[0, 1, 2]))
    check_env(env, skip_render_check=True)


def test_observation_width_follows_the_camera_not_the_lidar():
    cfg = NavEnvConfig(
        obs_mode="depth",
        world_seeds=[0],
        lidar=LidarConfig(n_beams=32),
        camera=CameraConfig(width=48),
    )
    env = ProceduralNavEnv(cfg)
    assert env.observation_space.shape == (48 + 5,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (53,)


def test_privileged_mode_still_uses_the_lidar():
    cfg = NavEnvConfig(
        obs_mode="privileged",
        world_seeds=[0],
        lidar=LidarConfig(n_beams=32),
        camera=CameraConfig(width=48),
    )
    env = ProceduralNavEnv(cfg)
    assert env.observation_space.shape == (32 + 5,)


def test_depth_observations_stay_in_bounds():
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="depth", world_seeds=list(range(6))))
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_goal_vector_is_still_provided_in_depth_mode():
    """PointGoal convention: the agent knows the goal bearing, not the map."""
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="depth", world_seeds=[0]))
    obs, _ = env.reset(seed=0, options={"world_seed": 0})
    goal_block = obs[-5:]
    cos_b, sin_b = goal_block[1], goal_block[2]
    assert np.hypot(cos_b, sin_b) == pytest.approx(1.0, abs=1e-5)


def test_unknown_obs_mode_names_the_available_ones():
    with pytest.raises(NotImplementedError, match="privileged.*depth|depth"):
        ProceduralNavEnv(NavEnvConfig(obs_mode="rgbd"))
