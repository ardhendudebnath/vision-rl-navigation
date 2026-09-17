"""Tests for the pixel stall audit.

The ambiguity measurement is only as good as its copy of the renderer, so the
copy is held to the real one image for image. The stall classifier decides
which of three explanations a result supports, so each class is built by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.analysis.pixel_audit import (
    column_signature,
    depth_ambiguity,
    stall_geometry,
    stall_steps,
    stalled_now,
)
from vision_nav.envs.rgb_camera import RGBCamera, RGBCameraConfig
from vision_nav.envs.world import World, WorldConfig, generate_world


def _image_from_signature(cfg: RGBCameraConfig, sig: np.ndarray) -> np.ndarray:
    h, w = cfg.height, cfg.width
    img = np.empty((h, w, 3), dtype=np.uint8)
    rows = np.arange(h)
    for c in range(w):
        top, bottom = sig[c, 0], sig[c, 1]
        img[:, c] = cfg.ceiling_rgb
        img[rows >= bottom, c] = cfg.floor_rgb
        img[(rows >= top) & (rows < bottom), c] = sig[c, 2:5]
    return img


@pytest.mark.parametrize("seed", [0, 5, 11])
def test_signature_rebuilds_the_rendered_image_exactly(seed):
    """If this fails, every ambiguity number describes a renderer that does not exist."""
    world = generate_world(seed)
    cam = RGBCamera()
    rng = np.random.default_rng(seed)
    for _ in range(5):
        pose = np.array([*rng.uniform(1.0, 11.0, 2), rng.uniform(-np.pi, np.pi)])
        ranges, kinds = cam._lidar.scan_with_hits(world, pose)
        sig = column_signature(cam.config, ranges * np.cos(cam._offsets), kinds)
        np.testing.assert_array_equal(_image_from_signature(cam.config, sig),
                                      cam.render(world, pose))


def test_without_shading_near_surfaces_are_indistinguishable():
    """Non-vacuity. With distance shading switched off, every surface nearer than
    wall_scale fills its column identically, so the whole near band must come
    back as one ambiguous run -- and with shading on, it must not."""
    grid = np.arange(0.2, 6.0, 0.001)
    flat = RGBCameraConfig(distance_falloff=1.0)
    near = grid < 1.0
    assert depth_ambiguity(flat, 0, grid)[near].min() >= 1.2 - 0.2 - 0.002
    shaded = depth_ambiguity(RGBCameraConfig(), 0, grid)
    assert shaded[near].max() < 0.5


def test_stall_steps_counts_going_nowhere_however_it_is_done():
    dt, n = 0.1, 80
    still = np.tile([[2.0, 2.0]], (n, 1))
    assert stall_steps(still, dt).all()
    t = np.arange(n)[:, None]
    straight = np.hstack([2.0 + 0.06 * t, np.full((n, 1), 2.0)])
    assert not stall_steps(straight, dt).any()
    dither = np.hstack([2.0 + 0.05 * np.sin(t), np.full((n, 1), 2.0)])
    assert stall_steps(dither, dt).all()


def test_the_causal_detector_agrees_with_the_labeller_once_a_window_has_passed():
    """stalled_now at step i must be exactly "the window ending at i is a stall"
    -- never true earlier, since it cannot see the future."""
    dt, k = 0.1, 30
    rng = np.random.default_rng(3)
    moving = np.cumsum(rng.normal(0.0, 0.03, size=(60, 2)), axis=0)
    parked = np.tile(moving[-1], (50, 1))
    path = np.vstack([moving, parked, parked[-1] + np.cumsum(np.full((40, 2), 0.05), axis=0)])
    labels = stall_steps(path, dt)
    for i in range(len(path)):
        now = stalled_now(path[:i + 1], dt)
        if i < k:
            assert not now
        window = path[i - k:i + 1] if i >= k else None
        if window is not None:
            expected = np.linalg.norm(window - window[0], axis=1).max() < 0.15
            assert now == expected, i
            if now:
                assert labels[i - k:i + 1].all()
    assert any(stalled_now(path[:i + 1], dt) for i in range(len(path)))


def _open_world(boxes=()):
    return World(config=WorldConfig(), circles=np.zeros((0, 3)),
                 boxes=np.asarray(boxes, dtype=float).reshape(-1, 4),
                 start=np.array([2.0, 6.0, 0.0]), goal=np.array([10.0, 6.0]))


def test_an_open_route_to_a_visible_goal_is_goal_open():
    assert stall_geometry(_open_world(), np.array([2.0, 6.0, 0.0]), np.pi / 2, 64) == "goal_open"


def test_a_goal_out_of_view_is_never_goal_open():
    # Facing away from the goal: the route is physically open but not visible.
    assert stall_geometry(_open_world(), np.array([2.0, 6.0, np.pi]), np.pi / 2, 64) == "detour"


def test_a_blocked_goal_with_a_visible_opening_is_detour():
    # A small block 1 m dead ahead. First built 0.6 m wide at 0.6 m, which the
    # classifier called boxed -- correctly: a 0.22 m disc clips it on every
    # heading inside +/-45 degrees. This one leaves the diagonals open.
    world = _open_world(boxes=[(3.0, 5.8, 3.2, 6.2)])
    assert stall_geometry(world, np.array([2.0, 6.0, 0.0]), np.pi / 2, 64) == "detour"


def test_a_wall_across_the_whole_view_is_boxed():
    world = _open_world(boxes=[(2.6, 2.0, 3.0, 10.0)])
    assert stall_geometry(world, np.array([2.0, 6.0, 0.0]), np.pi / 2, 64) == "boxed"
