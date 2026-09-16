"""A* over (row, col, time): a planner that can decide to wait.

Phases 5m-5o established that the classical stack, given oracle knowledge of
every mover's future everywhere it reads movers, recovers half the motion cost
and no more, and that doubling the robot's agility does not move the rest. The
surviving explanation is that a planner reasoning only in space uses a perfect
input crudely: it can route around the region a mover sweeps, but it cannot
choose to wait for the mover to pass, or to go through before it arrives. This
planner can.

Time model. Every action -- the eight neighbour moves, or waiting in place --
advances time by exactly one step of ``dt``. ``dt`` is chosen by the caller so
that a *diagonal* cell move is feasible at the robot's top speed, which makes
every action in the graph physically executable; orthogonal moves then run
below top speed. Cost is elapsed time, so the search minimises arrival time.

Tractability. Movers are checked only inside a window of ``window`` steps.
Every state at or beyond the window is collapsed into one static layer, since
past the window time no longer matters to collision checking. That bounds the
state space at ``cells x (window + 1)`` instead of growing with the episode,
and the robot replans long before it reaches the window's edge.

Heuristic. Geodesic distance to the goal over the static grid, in octile cells,
divided by sqrt(2): one step moves at most one cell diagonally, reducing octile
distance by at most sqrt(2), so this never overestimates the remaining steps.
Each step changes it by at most 1 at a cost of exactly 1, so it is consistent
as well as admissible.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

__all__ = ["spacetime_astar", "WAIT"]

_SQRT2 = math.sqrt(2.0)

#: (d_row, d_col). The last entry is waiting in place.
WAIT = (0, 0)
_MOVES: tuple[tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
    WAIT,
)


def spacetime_astar(
    static_occ: np.ndarray,
    mover_occ: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    heuristic_cells: np.ndarray,
    max_steps: int = 5000,
) -> list[tuple[int, int, int]] | None:
    """Find a minimum-time path that avoids static obstacles and movers.

    Parameters
    ----------
    static_occ:
        ``(R, C)`` bool, the inflated static map. ``True`` is blocked.
    mover_occ:
        ``(W, R, C)`` bool, the inflated movers at time steps ``0 .. W-1``.
        ``W`` is the window; states at step ``W`` or later see only the map.
    start, goal:
        ``(row, col)`` cells.
    heuristic_cells:
        ``(R, C)`` geodesic distance to ``goal`` in octile cells over
        ``static_occ``, ``inf`` where unreachable.
    max_steps:
        Hard cap on elapsed steps, so an unreachable goal cannot run forever.

    Returns
    -------
    list of (row, col, step) or None
        One entry per time step from ``start`` at step 0 to ``goal``, waits
        included as repeated cells. ``None`` if no collision-free path exists
        within ``max_steps``.
    """
    static_occ = np.ascontiguousarray(static_occ, dtype=bool)
    mover_occ = np.ascontiguousarray(mover_occ, dtype=bool)
    n_rows, n_cols = static_occ.shape
    window = mover_occ.shape[0]
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])

    for r, c in ((sr, sc), (gr, gc)):
        if not (0 <= r < n_rows and 0 <= c < n_cols) or static_occ[r, c]:
            return None
    if not math.isfinite(heuristic_cells[sr, sc]):
        return None
    if window and mover_occ[0, sr, sc]:
        # Starting inside a mover's inflated disc. Planning out of it would
        # require passing through blocked cells, so report no plan rather than
        # invent one; the caller keeps its previous plan.
        return None

    def blocked(r: int, c: int, k: int) -> bool:
        if static_occ[r, c]:
            return True
        return k < window and bool(mover_occ[k, r, c])

    def h(r: int, c: int) -> float:
        return heuristic_cells[r, c] / _SQRT2

    # A state's layer index collapses to ``window`` once past it, which is
    # what bounds the search; the true elapsed step is carried separately.
    start_state = (sr, sc, 0)
    g_score: dict[tuple[int, int, int], int] = {start_state: 0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    closed: set[tuple[int, int, int]] = set()
    counter = 0
    open_heap = [(h(sr, sc), counter, 0, start_state)]

    while open_heap:
        _, _, g, state = heapq.heappop(open_heap)
        if state in closed:
            continue
        r, c, layer = state
        if (r, c) == (gr, gc):
            return _reconstruct(came_from, state, g)
        closed.add(state)
        if g >= max_steps:
            continue

        step = g + 1
        next_layer = min(step, window)
        for dr, dc in _MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if (dr, dc) == WAIT and layer >= window:
                # Waiting past the window changes nothing, so it can only
                # lengthen a path; pruning it keeps the static tail finite.
                continue
            # Destination must be free now and at the next step, so a robot
            # and a mover cannot swap cells between steps and pass through
            # each other unseen.
            if blocked(nr, nc, step) or blocked(nr, nc, g):
                continue
            if dr and dc and (blocked(r, nc, step) or blocked(nr, c, step)):
                continue
            nxt = (nr, nc, next_layer)
            if nxt in closed or step >= g_score.get(nxt, math.inf):
                continue
            hv = h(nr, nc)
            if not math.isfinite(hv):
                continue
            g_score[nxt] = step
            came_from[nxt] = state
            counter += 1
            heapq.heappush(open_heap, (step + hv, counter, step, nxt))

    return None


def _reconstruct(came_from, state, g_final):
    cells = [state[:2]]
    while state in came_from:
        state = came_from[state]
        cells.append(state[:2])
    cells.reverse()
    assert len(cells) == g_final + 1, "one entry per step"
    return [(r, c, k) for k, (r, c) in enumerate(cells)]
