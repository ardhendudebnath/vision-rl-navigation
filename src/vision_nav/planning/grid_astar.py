"""Grid-based A* search over an inflated occupancy grid.

This module has two jobs in the project:

1. It supplies the **shortest-path length** that SPL (Success weighted by
   Path Length) is normalised by.  Without it, SPL cannot be computed.
2. It is the kernel of the **classical baseline** that the learned policies
   are compared against — the same role Nav2's global planner plays once the
   stack moves onto ROS 2.

The search is 8-connected with an octile heuristic, which is admissible and
consistent for 8-connected grids with unit/√2 step costs.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = [
    "astar_grid",
    "shortest_path_length",
    "plan_path",
    "geodesic_distance_field",
]

_SQRT2 = math.sqrt(2.0)

# (d_row, d_col, cost) for the 8 neighbours.
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)


def _octile(r0: int, c0: int, r1: int, c1: int) -> float:
    dr = abs(r0 - r1)
    dc = abs(c0 - c1)
    return (dr + dc) + (_SQRT2 - 2.0) * min(dr, dc)


def astar_grid(
    occupancy: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """Find a shortest 8-connected path through free cells.

    Parameters
    ----------
    occupancy:
        ``(n_rows, n_cols)`` boolean array; ``True`` means blocked.
    start, goal:
        ``(row, col)`` index pairs.

    Returns
    -------
    list of (row, col) or None
        The path including both endpoints, or ``None`` if ``goal`` is not
        reachable from ``start`` (or either endpoint is blocked).
    """
    occ = np.ascontiguousarray(occupancy, dtype=bool)
    n_rows, n_cols = occ.shape
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    for r, c in ((sr, sc), (gr, gc)):
        if not (0 <= r < n_rows and 0 <= c < n_cols) or occ[r, c]:
            return None

    if (sr, sc) == (gr, gc):
        return [(sr, sc)]

    blocked = occ.ravel()
    start_idx = sr * n_cols + sc
    goal_idx = gr * n_cols + gc

    g_score: dict[int, float] = {start_idx: 0.0}
    came_from: dict[int, int] = {}
    closed: set[int] = set()
    open_heap: list[tuple[float, float, int]] = [
        (_octile(sr, sc, gr, gc), 0.0, start_idx)
    ]

    while open_heap:
        _, g, idx = heapq.heappop(open_heap)
        if idx in closed:
            continue
        if idx == goal_idx:
            path = [idx]
            while idx in came_from:
                idx = came_from[idx]
                path.append(idx)
            path.reverse()
            return [(i // n_cols, i % n_cols) for i in path]
        closed.add(idx)

        r, c = divmod(idx, n_cols)
        for dr, dc, step in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            n_idx = nr * n_cols + nc
            if blocked[n_idx] or n_idx in closed:
                continue
            # Forbid cutting a diagonal between two blocked orthogonal cells.
            if dr and dc and (occ[r, nc] and occ[nr, c]):
                continue
            tentative = g + step
            if tentative < g_score.get(n_idx, math.inf):
                g_score[n_idx] = tentative
                came_from[n_idx] = idx
                heapq.heappush(
                    open_heap, (tentative + _octile(nr, nc, gr, gc), tentative, n_idx)
                )

    return None


def geodesic_distance_field(
    occupancy: np.ndarray,
    goal: tuple[int, int],
    max_iters: int = 4096,
) -> np.ndarray:
    """Geodesic distance (in cells) from every free cell to ``goal``.

    Computed by min-plus relaxation over the whole grid at once, which is far
    faster in NumPy than a per-cell Dijkstra loop for grids of this size.
    Blocked and unreachable cells hold ``inf``.

    The field is what the environment uses for **progress reward shaping**.
    Shaping on Euclidean distance creates a local optimum behind every
    obstacle — the agent is rewarded for pressing against a wall that happens
    to lie between it and the goal.  Shaping on geodesic distance removes
    that pathology.  This is privileged *training-time* information only; it
    never enters the observation, so the resulting policy is still honestly
    "vision-only" or "lidar-only" at evaluation time.
    """
    occ = np.ascontiguousarray(occupancy, dtype=bool)
    gr, gc = int(goal[0]), int(goal[1])
    dist = np.full(occ.shape, np.inf, dtype=np.float64)
    if occ[gr, gc]:
        return dist
    dist[gr, gc] = 0.0

    # (row shift, col shift, step cost)
    shifts = _NEIGHBOURS

    for _ in range(max_iters):
        prev = dist
        candidates = [dist]
        for dr, dc, cost in shifts:
            shifted = np.full_like(dist, np.inf)
            # Pull the value from the neighbour at (r+dr, c+dc).
            src_rows = slice(max(dr, 0), dist.shape[0] + min(dr, 0))
            dst_rows = slice(max(-dr, 0), dist.shape[0] + min(-dr, 0))
            src_cols = slice(max(dc, 0), dist.shape[1] + min(dc, 0))
            dst_cols = slice(max(-dc, 0), dist.shape[1] + min(-dc, 0))
            shifted[dst_rows, dst_cols] = dist[src_rows, src_cols] + cost
            candidates.append(shifted)
        dist = np.minimum.reduce(candidates)
        dist[occ] = np.inf
        if np.array_equal(dist, prev):
            break

    return dist


def plan_path(world, start_xy, goal_xy) -> np.ndarray | None:
    """Plan a path in **world coordinates** between two positions.

    Returns an ``(N, 2)`` array of waypoints, or ``None`` if unreachable.
    """
    start_cell = tuple(int(v) for v in world.world_to_grid(np.asarray(start_xy)))
    goal_cell = tuple(int(v) for v in world.world_to_grid(np.asarray(goal_xy)))
    cells = astar_grid(world.occupancy, start_cell, goal_cell)
    if cells is None:
        return None
    return world.grid_to_world(np.asarray(cells, dtype=int))


def shortest_path_length(world, start_xy, goal_xy, simplify: bool = True) -> float | None:
    """Length in metres of the shortest collision-free path, or ``None``.

    This is the ``l_i`` term in SPL.  It is measured on the robot-inflated
    grid, so it is the shortest path *the robot could actually drive*, not
    the straight-line distance — using Euclidean distance here would make SPL
    unfairly punish any policy in a cluttered scene.

    With ``simplify=True`` the grid path is string-pulled first.  That matters
    more than it looks: a raw 8-connected path overestimates the true geodesic
    distance, and an overestimated ``l*`` makes ``l* / max(p, l*)`` clamp to
    1.0 for every halfway-competent policy, rendering SPL useless as a
    discriminator.
    """
    from vision_nav.planning.smoothing import path_length, simplify_path

    path = plan_path(world, start_xy, goal_xy)
    if path is None:
        return None
    if len(path) < 2:
        return 0.0
    if simplify:
        # Endpoints come from grid-cell centres; snap them back to the true
        # query positions so l* measures the distance actually demanded.
        path = path.copy()
        path[0] = np.asarray(start_xy, dtype=np.float64)
        path[-1] = np.asarray(goal_xy, dtype=np.float64)
        path = simplify_path(world, path, world.config.robot_radius)
    return path_length(path)
