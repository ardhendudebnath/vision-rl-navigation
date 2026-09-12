"""Tests for the egocentric RGB observation mode.

The mode exists to isolate what a CNN encoder costs, which only works if the
render carries the *same* geometry the depth camera measures. So these tests
focus on agreement with the depth sensor and on the rendering conventions that
would otherwise silently distort it.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.rgb_camera import RGBCamera, RGBCameraConfig
from vision_nav.envs.sensors import CameraConfig, DepthCamera
from vision_nav.envs.world import World, WorldConfig, generate_world


def corridor_world() -> World:
    """A box straight ahead, a circle off to the right."""
    return World(
        config=WorldConfig(width=20.0, height=20.0),
        circles=np.array([[12.0, 11.5, 0.5]]),
        boxes=np.array([[12.0, 9.0, 13.0, 10.5]]),
        start=np.zeros(3),
        goal=np.zeros(2),
    )


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
def test_invalid_config_is_refused():
    with pytest.raises(ValueError, match="fov"):
        RGBCameraConfig(fov=0.0)
    with pytest.raises(ValueError, match="at least 1x1"):
        RGBCameraConfig(width=0)
    with pytest.raises(ValueError, match="at least 1x1"):
        RGBCameraConfig(height=0)


# ----------------------------------------------------------------------
# Agreement with the depth sensor
# ----------------------------------------------------------------------
def test_rgb_and_depth_sample_identical_bearings():
    """If they disagreed, the comparison would measure the ray-caster."""
    rgb = RGBCamera(RGBCameraConfig(fov=np.pi / 2, width=32))
    depth = DepthCamera(CameraConfig(fov=np.pi / 2, width=32))
    assert np.allclose(rgb._lidar._angles, depth._lidar._angles)


def test_nearer_surfaces_render_taller():
    """Slice height must be the distance cue it claims to be."""
    cfg = WorldConfig(width=20.0, height=20.0)
    cam = RGBCamera(RGBCameraConfig(fov=np.pi / 2, width=33, height=48, max_range=8.0))

    def centre_slice_height(obstacle_x):
        world = World(
            config=cfg,
            circles=np.array([[obstacle_x, 10.0, 0.5]]),
            boxes=np.zeros((0, 4)),
            start=np.zeros(3),
            goal=np.zeros(2),
        )
        img = cam.render(world, np.array([5.0, 10.0, 0.0]))
        col = img[:, 16]  # centre column, odd width so it is on-axis
        ceiling = np.asarray(cam.config.ceiling_rgb, dtype=np.uint8)
        return int(np.sum(~np.all(col == ceiling, axis=-1)))

    near = centre_slice_height(7.0)
    far = centre_slice_height(11.0)
    assert near > far, f"near slice {near} should exceed far slice {far}"


def test_no_fisheye_bow_on_a_flat_wall():
    """Perpendicular distance correction: a flat wall must render flat.

    Using radial distance would bow straight surfaces outward at the frame
    edges -- a rendering bug that would look like a perception difficulty.
    """
    cfg = WorldConfig(width=40.0, height=40.0)
    world = World(
        config=cfg,
        circles=np.zeros((0, 3)),
        boxes=np.array([[18.0, 5.0, 30.0, 35.0]]),  # broad flat face at x=18
        start=np.zeros(3),
        goal=np.zeros(2),
    )
    cam = RGBCamera(RGBCameraConfig(fov=np.pi / 3, width=41, height=48, max_range=20.0))
    img = cam.render(world, np.array([14.0, 20.0, 0.0]))
    ceiling = np.asarray(cam.config.ceiling_rgb, dtype=np.uint8)
    heights = [int(np.sum(~np.all(img[:, c] == ceiling, axis=-1))) for c in range(41)]
    # A flat wall should give a near-constant slice height across the frame.
    assert max(heights) - min(heights) <= 2, f"wall bows: {min(heights)}..{max(heights)}"


def test_surface_types_are_visually_distinct():
    """RGB can express what a depth vector cannot: what kind of surface."""
    world = corridor_world()
    cam = RGBCamera(RGBCameraConfig(fov=np.pi / 2, width=64, height=48, max_range=10.0))
    img = cam.render(world, np.array([8.0, 10.0, 0.0]))
    colours = {tuple(c) for c in img.reshape(-1, 3)}
    assert len(colours) > 3, "render should contain more than ceiling/floor/one surface"


def test_render_shape_and_dtype():
    cam = RGBCamera(RGBCameraConfig(width=48, height=32))
    img = cam.render(generate_world(1), generate_world(1).start)
    assert img.shape == (32, 48, 3) and img.dtype == np.uint8
    assert cam.render_chw(generate_world(1), generate_world(1).start).shape == (3, 32, 48)


# ----------------------------------------------------------------------
# Environment integration
# ----------------------------------------------------------------------
def test_rgb_env_passes_the_gymnasium_checker():
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="rgb", world_seeds=[0, 1, 2]))
    check_env(env, skip_render_check=True)


def test_rgb_observation_is_a_dict_with_image_and_vector():
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="rgb", world_seeds=[0]))
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"image", "vector"}
    assert obs["image"].shape == (3, 48, 64) and obs["image"].dtype == np.uint8
    assert obs["vector"].shape == (5,)
    assert env.observation_space.contains(obs)


def test_goal_vector_is_identical_across_modes():
    """The sensor must be the only thing that differs between arms."""
    seeds = [7]
    envs = {
        m: ProceduralNavEnv(NavEnvConfig(obs_mode=m, world_seeds=seeds))
        for m in ("depth", "rgb")
    }
    vectors = {}
    for m, env in envs.items():
        obs, _ = env.reset(seed=0, options={"world_seed": 7})
        vectors[m] = obs[-5:] if m == "depth" else obs["vector"]
    assert np.allclose(vectors["depth"], vectors["rgb"], atol=1e-6)


def test_rgb_observations_stay_in_bounds():
    env = ProceduralNavEnv(NavEnvConfig(obs_mode="rgb", world_seeds=list(range(5))))
    obs, _ = env.reset(seed=0)
    for _ in range(150):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_unknown_mode_lists_all_three():
    with pytest.raises(NotImplementedError) as e:
        ProceduralNavEnv(NavEnvConfig(obs_mode="lidar3d"))
    msg = str(e.value)
    assert all(m in msg for m in ("privileged", "depth", "rgb"))
