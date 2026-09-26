"""Planning that does not pretend unexplored space is empty.

Report §9.2 gave the mapping planner the usual convention: unknown cells count
as free, the planner is optimistic, and it replans when what it had not seen
turns out to be in the way. §9.7 measured what that costs in clutter. The
failures are not stuck and not exploring: they reverse 46 times against a
success's 1, rebuild the plan every three steps, and cover *less* ground than
the successes while driving twice the geodesic. Two rules that suppressed the
rebuilding were rejected on the val band, which left the optimism itself as the
thing to change -- the planner keeps proposing routes that end at something
nobody has looked at.

So this plans through **known free space only**, and when the goal cannot be
reached that way it plans to a frontier instead: a known-free cell that touches
the unknown, chosen to trade the cost of reaching it against how much closer to
the goal it is. Drive there, the map grows, decide again. That is
frontier-based exploration, and the difference from the optimistic planner is
what gets proposed rather than how often.

A robot that can see 6 m in every direction knows a good deal about its
surroundings at every step, so this is not as conservative as it sounds; the
price is that it cannot take a shortcut through the unknown even when one
exists, which is exactly the bet being measured.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["plan_through_known"]

_SQRT2 = math.sqrt(2.0)

#: How much the cost of reaching an opening counts against how much closer to
#: the goal it is, in the score that picks one. Below one on purpose: an unseen
#: cell a step away is cheap to reach and often useless -- a pocket behind a
#: corner the robot has just passed -- and weighting the two equally sends the
#: robot poking at whatever is nearest, which is the indecision §9.7 measured in
#: a different guise. Where an opening leads matters more than what it costs.
COST_WEIGHT = 0.25
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2),
)


def plan_through_known(
    blocked: np.ndarray,
    unknown: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int] | None,
    target: tuple[int, int] | None = None,
) -> tuple[list[tuple[int, int]] | None, str]:
    """A route through known free space, to the goal or to a frontier.

    Parameters
    ----------
    blocked:
        ``True`` where the robot may not go: inflated obstacles *and* unknown
        cells.
    unknown:
        ``True`` where the map has seen nothing. A frontier is a cell the robot
        may occupy that touches one of these.
    start:
        ``(row, col)`` of the robot.
    goal:
        ``(row, col)`` of the goal, whether or not the map has seen it. This is
        what makes the exploration goal-directed, and it is separate from
        ``target`` on purpose: the robot is told where the goal is at the start
        of the episode, and not having seen the floor there is no reason to
        stop steering towards it. Passing ``None`` here asks for undirected
        nearest-frontier exploration, which is a different algorithm.
    target:
        The cell to route to when the goal can be reached through known free
        space, or ``None`` when it cannot -- in which case this plans to a
        frontier. Defaults to ``goal``.

    Returns
    -------
    (cells, kind)
        ``kind`` is ``"goal"`` when the route ends at ``target``, ``"frontier"``
        when it ends at a frontier, and ``"none"`` when neither is reachable --
        which is when the caller should fall back to planning optimistically.
    """
    blocked = np.ascontiguousarray(blocked, dtype=bool)
    unknown = np.ascontiguousarray(unknown, dtype=bool)
    rows, cols = blocked.shape
    sr, sc = int(start[0]), int(start[1])
    if not (0 <= sr < rows and 0 <= sc < cols) or blocked[sr, sc]:
        return None, "none"

    # Dijkstra over the cells the robot may occupy. One sweep serves both
    # questions: whether the goal is reachable without entering the unknown,
    # and what every frontier costs to reach.
    flat_blocked = blocked.ravel()
    dist = np.full(rows * cols, np.inf)
    parent = np.full(rows * cols, -1, dtype=np.int64)
    start_idx = sr * cols + sc
    dist[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_idx)]
    if target is None:
        target_cell = goal
    else:
        target_cell = target
    goal_idx = (None if target_cell is None
                else int(target_cell[0]) * cols + int(target_cell[1]))
    if goal_idx is not None and flat_blocked[goal_idx]:
        goal_idx = None

    while heap:
        d, idx = heapq.heappop(heap)
        if d > dist[idx]:
            continue
        if goal_idx is not None and idx == goal_idx:
            break
        r, c = divmod(idx, cols)
        for dr, dc, step in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            n_idx = nr * cols + nc
            if flat_blocked[n_idx]:
                continue
            # No cutting a diagonal between two blocked orthogonal cells.
            if dr and dc and blocked[r, nc] and blocked[nr, c]:
                continue
            nd = d + step
            if nd < dist[n_idx]:
                dist[n_idx] = nd
                parent[n_idx] = idx
                heapq.heappush(heap, (nd, n_idx))

    def route(idx: int) -> list[tuple[int, int]]:
        out = [idx]
        while parent[out[-1]] >= 0:
            out.append(int(parent[out[-1]]))
        out.reverse()
        return [(i // cols, i % cols) for i in out]

    if goal_idx is not None and np.isfinite(dist[goal_idx]):
        return route(goal_idx), "goal"

    # No route to the goal through what is known: head for the unseen cell that
    # trades the cost of reaching it against how much closer to the goal it is.
    #
    # The target is the unknown cell itself rather than the known one beside it.
    # A route that stops short of the unknown is no route at all when the robot
    # is already standing at the edge of what it has seen -- which happens
    # whenever a corner hides the floor a step ahead -- and it would leave the
    # robot planning a path of length zero to where it already is. One step into
    # the dark is safe because the robot maps as it drives: the cell is seen
    # before it is reached.
    grid_dist = dist.reshape(rows, cols)
    reach_cost = np.full((rows, cols), np.inf)
    for dr, dc, step in _NEIGHBOURS:
        shifted = np.full((rows, cols), np.inf)
        src_rows = slice(max(dr, 0), rows + min(dr, 0))
        dst_rows = slice(max(-dr, 0), rows + min(-dr, 0))
        src_cols = slice(max(dc, 0), cols + min(dc, 0))
        dst_cols = slice(max(-dc, 0), cols + min(-dc, 0))
        shifted[dst_rows, dst_cols] = grid_dist[src_rows, src_cols] + step
        reach_cost = np.minimum(reach_cost, shifted)
    candidates = unknown & np.isfinite(reach_cost)
    if not candidates.any():
        return None, "none"

    cells = np.argwhere(candidates)
    score = COST_WEIGHT * reach_cost[candidates]
    if goal is not None:
        score = score + np.linalg.norm(cells - np.asarray(goal, dtype=float), axis=1)
    target = cells[int(np.argmin(score))]

    # Step in from whichever known neighbour reaches it most cheaply.
    tr, tc = int(target[0]), int(target[1])
    best_from, best_cost = None, np.inf
    for dr, dc, step in _NEIGHBOURS:
        nr, nc = tr + dr, tc + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        cost = grid_dist[nr, nc] + step
        if cost < best_cost:
            best_from, best_cost = (nr, nc), cost
    if best_from is None:
        return None, "none"
    return route(best_from[0] * cols + best_from[1]) + [(tr, tc)], "frontier"
