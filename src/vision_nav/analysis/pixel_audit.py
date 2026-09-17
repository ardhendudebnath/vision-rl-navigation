"""Why does the pixel policy stall? Two measurements that need no training.

Phase 5s found the cost of reading pixels through a CNN surviving the 1:1 reward
with its failure unmoved: the RGB policy still times out on 0.182 of `narrow`
episodes where the depth policy, reading the same geometry as a vector, times
out on 0.028. Every other perception deficit re-priced at 1:1 moved into
collisions. This module asks two things.

**Is the information really held constant?** Phase 3e's premise is that the
render carries the same geometry the depth vector does. It is not exact. A
surface closer than ``wall_scale`` metres fills its whole image column, so below
that distance slice height says nothing, and depth survives only as distance
shading quantised to eight bits. :func:`depth_ambiguity` measures, for each
surface kind and true distance, the span of distances that render to an
identical column -- how far apart two walls can be and still look the same.

**Where does it stall?** :func:`stall_steps` finds the steps of a trajectory
spent going nowhere, and :func:`stall_geometry` classifies each by what was
actually there, checked against the world rather than any sensor:

  ``goal_open``   the goal bearing is inside the field of view and the robot
                  could drive a probe distance along it -- nothing but the
                  policy stops it.
  ``detour``      the goal-ward route is blocked or out of view, but some
                  direction inside the field of view is open -- a way on is
                  visible and not taken.
  ``boxed``       nothing inside the field of view is open -- the robot must
                  turn to find a way, which a camera with the depth vector
                  has to do too.
"""

from __future__ import annotations

import numpy as np

from vision_nav.analysis.perception import traversable_mask
from vision_nav.envs.rgb_camera import RGBCameraConfig
from vision_nav.envs.robot import wrap_angle

__all__ = ["column_signature", "depth_ambiguity", "stall_steps", "stalled_now",
           "stall_geometry", "STALL_CLASSES"]

STALL_CLASSES = ("goal_open", "detour", "boxed")


def column_signature(cfg: RGBCameraConfig, perp: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """``(N, 5)`` integer description of rendered columns: top row, bottom row,
    and the surface's eight-bit colour.

    Mirrors :meth:`RGBCamera.render` column by column. Two columns with equal
    signatures are identical images; the test suite holds this function to the
    renderer on real poses so the two cannot drift apart.
    """
    h = cfg.height
    perp = np.maximum(np.asarray(perp, dtype=np.float64), 1e-3)
    kind = np.asarray(kind, dtype=int)
    slice_h = np.clip(cfg.wall_scale * h / perp, 1.0, h)
    top = np.clip((h - slice_h) * 0.5, 0, h).astype(np.int32)
    bottom = np.clip(h - top, 0, h).astype(np.int32)
    shade = 1.0 - (1.0 - cfg.distance_falloff) * np.clip(perp / cfg.max_range, 0.0, 1.0)
    surface = np.array([cfg.wall_rgb, cfg.circle_rgb, cfg.box_rgb], dtype=np.float64)
    colour = (surface[kind] * shade[:, None]).astype(np.uint8)
    return np.concatenate([top[:, None], bottom[:, None], colour.astype(np.int32)], axis=1)


def depth_ambiguity(cfg: RGBCameraConfig, kind: int, grid: np.ndarray) -> np.ndarray:
    """For each distance in ``grid`` (ascending, evenly spaced), the width in
    metres of the run of distances that render to the same column.

    A run of ``n`` grid points is reported as ``n * step``: the span over which
    the image cannot tell the distances apart, at the grid's resolution.
    """
    grid = np.asarray(grid, dtype=np.float64)
    step = float(grid[1] - grid[0])
    sig = column_signature(cfg, grid, np.full(len(grid), kind))
    change = np.any(sig[1:] != sig[:-1], axis=1)
    run_id = np.concatenate([[0], np.cumsum(change)])
    lengths = np.bincount(run_id)
    return lengths[run_id] * step


def stall_steps(positions: np.ndarray, dt: float, window_s: float = 3.0,
                min_displacement: float = 0.15) -> np.ndarray:
    """Steps inside any window of ``window_s`` seconds over which the robot's
    net displacement stays under ``min_displacement`` metres.

    Displacement rather than speed, so a robot spinning in place or dithering
    back and forth counts as going nowhere, which is what a timeout is made of.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = len(positions)
    k = max(1, int(round(window_s / dt)))
    stalled = np.zeros(n, dtype=bool)
    for i in range(n - k):
        segment = positions[i:i + k + 1]
        if np.linalg.norm(segment - segment[0], axis=1).max() < min_displacement:
            stalled[i:i + k + 1] = True
    return stalled


def stalled_now(positions: np.ndarray, dt: float, window_s: float = 3.0,
                min_displacement: float = 0.15) -> bool:
    """Whether the robot has gone nowhere over the last ``window_s`` seconds.

    The causal half of :func:`stall_steps`: it looks only backwards, so a
    controller can act on it mid-episode. Anything a controller does with it
    therefore starts ``window_s`` late, which :func:`stall_steps` -- labelling a
    whole window once it has passed -- does not.
    """
    positions = np.asarray(positions, dtype=np.float64)
    k = max(1, int(round(window_s / dt)))
    if len(positions) < k + 1:
        return False
    recent = positions[-(k + 1):]
    return bool(np.linalg.norm(recent - recent[0], axis=1).max() < min_displacement)


def stall_geometry(world, pose: np.ndarray, fov: float, columns: int,
                   probe: float = 1.0) -> str:
    """Classify one pose into :data:`STALL_CLASSES` by the world's geometry."""
    pose = np.asarray(pose, dtype=np.float64)
    position, heading = pose[:2], float(pose[2])
    to_goal = world.goal - position
    goal_bearing = float(np.arctan2(to_goal[1], to_goal[0]))
    in_view = abs(float(wrap_angle(goal_bearing - heading))) <= fov / 2.0
    if in_view and bool(traversable_mask(world, position, np.array([goal_bearing]), probe)[0]):
        return "goal_open"
    offsets = -fov / 2.0 + (np.arange(columns) + 0.5) * (fov / columns)
    if bool(traversable_mask(world, position, heading + offsets, probe).any()):
        return "detour"
    return "boxed"
