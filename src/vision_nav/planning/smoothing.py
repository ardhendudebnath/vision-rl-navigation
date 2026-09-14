"""Post-processing for grid-search paths.

A raw 8-connected A* path is a staircase whose length overestimates the true
geodesic distance by up to a few percent, and whose corners are not tractable
by a smooth controller.  Both issues matter here:

- Path **length** feeds ``l*`` in SPL.  An inflated ``l*`` makes
  ``l* / max(p, l*)`` saturate at 1.0 for any competent policy, which quietly
  destroys SPL's ability to discriminate between them.
- Path **shape** is what the local controller tracks.  Staircase corners
  produce steering jitter that shows up as extra measured path length.

String-pulling (line-of-sight shortcutting) fixes both.
"""

from __future__ import annotations

import numpy as np

__all__ = ["simplify_path", "densify_path", "path_length"]


def path_length(path: np.ndarray) -> float:
    """Total Euclidean length of a polyline."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def simplify_path(
    world,
    path: np.ndarray,
    radius: float,
    step: float = 0.05,
) -> np.ndarray:
    """Drop waypoints that have clear line of sight past them.

    Parameters
    ----------
    world:
        A :class:`~vision_nav.envs.world.World`.
    path:
        ``(N, 2)`` polyline in world coordinates.
    radius:
        Required clearance along each retained segment, in metres.
    step:
        Sampling interval for the line-of-sight check, in metres.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 3:
        return path

    def visible(a: np.ndarray, b: np.ndarray) -> bool:
        dist = float(np.linalg.norm(b - a))
        n = max(int(dist / step), 1)
        ts = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = a[None, :] + ts * (b - a)[None, :]
        return bool(np.all(world.clearance(pts) > radius))

    out = [path[0]]
    anchor = 0
    for i in range(2, len(path)):
        if not visible(path[anchor], path[i]):
            out.append(path[i - 1])
            anchor = i - 1
    out.append(path[-1])
    return np.asarray(out)


def densify_path(path: np.ndarray, spacing: float = 0.05) -> np.ndarray:
    """Resample a polyline to roughly uniform ``spacing``.

    Pure pursuit needs a point at a given *arc length* ahead of the robot.
    Searching for that on a string-pulled path whose waypoints are metres
    apart would snap the target to a distant corner, and the controller would
    drive straight at it — cutting the corner and through the obstacle the
    planner had carefully routed around.  Densifying first is what keeps the
    lookahead point actually on the planned path.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return path

    segments = [path[0][None, :]]
    for a, b in zip(path[:-1], path[1:], strict=False):
        dist = float(np.linalg.norm(b - a))
        n = max(int(np.ceil(dist / spacing)), 1)
        ts = np.linspace(0.0, 1.0, n + 1)[1:, None]
        segments.append(a[None, :] + ts * (b - a)[None, :])
    return np.concatenate(segments, axis=0)
