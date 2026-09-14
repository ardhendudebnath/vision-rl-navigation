"""Dependency-light top-down renderer.

Pure NumPy rasterisation, so rendering a demo GIF never pulls a GUI toolkit
into the training environment.  Used for the README demo, for qualitative
side-by-side figures (classical vs. learned), and for eyeballing failure
cases during debugging.
"""

from __future__ import annotations

import numpy as np

__all__ = ["render_topdown", "RenderStyle"]


class RenderStyle:
    """Palette and sizing for :func:`render_topdown` (RGB, 0-255)."""

    px_per_metre: int = 48
    background = (247, 247, 245)
    obstacle = (74, 84, 96)
    robot = (30, 110, 200)
    heading = (250, 250, 250)
    goal = (44, 160, 90)
    goal_halo = (190, 226, 205)
    trajectory = (232, 120, 60)
    plan = (150, 150, 165)


def _disc_mask(xx: np.ndarray, yy: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def _draw_polyline(
    img: np.ndarray,
    points: np.ndarray,
    to_px,
    colour: tuple[int, int, int],
    width: int = 2,
) -> None:
    """Rasterise a polyline by dense sampling — adequate at these sizes."""
    if points is None or len(points) < 2:
        return
    h, w = img.shape[:2]
    seg = np.diff(points, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    for start, delta, length in zip(points[:-1], seg, lengths, strict=False):
        n = max(int(length * RenderStyle.px_per_metre), 1)
        ts = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = start[None, :] + ts * delta[None, :]
        cols, rows = to_px(pts)
        for dr in range(-width // 2, width // 2 + 1):
            for dc in range(-width // 2, width // 2 + 1):
                r = np.clip(rows + dr, 0, h - 1)
                c = np.clip(cols + dc, 0, w - 1)
                img[r, c] = colour


def render_topdown(
    world,
    pose: np.ndarray,
    trajectory: np.ndarray | None = None,
    plan: np.ndarray | None = None,
) -> np.ndarray:
    """Render a scene to an ``(H, W, 3)`` uint8 RGB image.

    Parameters
    ----------
    world:
        A :class:`~vision_nav.envs.world.World`.
    pose:
        ``(3,)`` robot pose ``(x, y, theta)``.
    trajectory:
        Optional ``(T, 2)`` array of positions already visited.
    plan:
        Optional ``(N, 2)`` reference path, e.g. the A* global plan.
    """
    s = RenderStyle
    cfg = world.config
    w_px = int(cfg.width * s.px_per_metre)
    h_px = int(cfg.height * s.px_per_metre)

    xs = (np.arange(w_px) + 0.5) / s.px_per_metre
    ys = (np.arange(h_px) + 0.5) / s.px_per_metre
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    img = np.empty((h_px, w_px, 3), dtype=np.uint8)
    img[:] = s.background

    occupied = np.zeros((h_px, w_px), dtype=bool)
    for cx, cy, r in world.circles:
        occupied |= _disc_mask(xx, yy, cx, cy, r)
    for x_lo, y_lo, x_hi, y_hi in world.boxes:
        occupied |= (xx >= x_lo) & (xx <= x_hi) & (yy >= y_lo) & (yy <= y_hi)
    img[occupied] = s.obstacle

    def to_px(pts: np.ndarray):
        pts = np.atleast_2d(pts)
        cols = np.clip((pts[:, 0] * s.px_per_metre).astype(int), 0, w_px - 1)
        rows = np.clip((pts[:, 1] * s.px_per_metre).astype(int), 0, h_px - 1)
        return cols, rows

    if plan is not None:
        _draw_polyline(img, np.asarray(plan), to_px, s.plan, width=2)
    if trajectory is not None:
        _draw_polyline(img, np.asarray(trajectory), to_px, s.trajectory, width=3)

    gx, gy = world.goal
    img[_disc_mask(xx, yy, gx, gy, cfg.goal_tolerance)] = s.goal_halo
    img[_disc_mask(xx, yy, gx, gy, cfg.goal_tolerance * 0.45)] = s.goal

    rx, ry, theta = float(pose[0]), float(pose[1]), float(pose[2])
    img[_disc_mask(xx, yy, rx, ry, cfg.robot_radius)] = s.robot
    nose = np.array(
        [[rx, ry], [rx + cfg.robot_radius * 0.95 * np.cos(theta), ry + cfg.robot_radius * 0.95 * np.sin(theta)]]
    )
    _draw_polyline(img, nose, to_px, s.heading, width=2)

    # World y points up; image rows go down.
    return np.flipud(img).copy()
