"""Tests for the space-time planner.

A weak implementation here would produce the most misleading possible result:
"a planner that can wait does not help either", reported as a finding when it
was a bug. So the planner is held to optimality against brute force, not just
to finding a path, and the behaviours the experiment depends on -- waiting for
a mover, passing before one arrives, never swapping through one -- are each
built by hand and checked.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from vision_nav.planning.grid_astar import geodesic_distance_field
from vision_nav.planning.spacetime_astar import WAIT, spacetime_astar

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1), WAIT]


def _plan(static, movers, start, goal, **kw):
    h = geodesic_distance_field(static, goal)
    return spacetime_astar(static, movers, start, goal, h, **kw)


def _bfs_min_steps(static, movers, start, goal, max_steps):
    """Reference: exhaustive breadth-first search over (row, col, step).

    Uses exactly the planner's blocking rules and no collapse, so on a window
    at least as long as any path it is the ground-truth minimum time.
    """
    n_r, n_c = static.shape
    window = movers.shape[0]

    def blocked(r, c, k):
        return static[r, c] or (k < window and movers[k, r, c])

    if static[start] or (window and movers[0][start]):
        return None
    seen = {(start[0], start[1], 0)}
    q = deque([(start[0], start[1], 0)])
    while q:
        r, c, k = q.popleft()
        if (r, c) == goal:
            return k
        if k >= max_steps:
            continue
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_r and 0 <= nc < n_c):
                continue
            if blocked(nr, nc, k + 1) or blocked(nr, nc, k):
                continue
            if dr and dc and (blocked(r, nc, k + 1) or blocked(nr, c, k + 1)):
                continue
            if (nr, nc, k + 1) not in seen:
                seen.add((nr, nc, k + 1))
                q.append((nr, nc, k + 1))
    return None


def _assert_valid(path, static, movers, start, goal):
    window = movers.shape[0]
    assert path[0][:2] == start and path[-1][:2] == goal
    for i, (r, c, k) in enumerate(path):
        assert k == i, "one entry per step"
        assert not static[r, c], f"path enters the map at step {k}"
        if k < window:
            assert not movers[k, r, c], f"path enters a mover at step {k}"
    for (r0, c0, _), (r1, c1, _) in zip(path, path[1:], strict=False):
        assert max(abs(r1 - r0), abs(c1 - c0)) <= 1, "moved more than one cell a step"


def _corridor(length=12, width=1):
    """A 3-row grid whose middle row is the only free corridor."""
    static = np.ones((3, length), dtype=bool)
    static[1, :] = False
    return static


# ----------------------------------------------------------------------------
def test_no_movers_reaches_goal_without_waiting():
    static = np.zeros((10, 10), dtype=bool)
    static[3:7, 5] = True
    movers = np.zeros((40, 10, 10), dtype=bool)
    path = _plan(static, movers, (0, 0), (9, 9))
    _assert_valid(path, static, movers, (0, 0), (9, 9))
    assert all((b[0], b[1]) != (a[0], a[1]) for a, b in zip(path, path[1:], strict=False)), \
        "waited with nothing to wait for"


def test_waits_for_a_mover_blocking_the_only_corridor():
    """The capability the whole experiment is about.

    A mover sits in the corridor for steps 2-9 and then leaves. There is no way
    round, so a planner that cannot wait finds nothing; this one must wait and
    then pass.
    """
    static = _corridor(12)
    movers = np.zeros((30, 3, 12), dtype=bool)
    movers[2:10, 1, 6] = True
    path = _plan(static, movers, (1, 0), (1, 11))
    assert path is not None, "found no plan through a corridor that clears"
    _assert_valid(path, static, movers, (1, 0), (1, 11))
    waits = sum((a[0], a[1]) == (b[0], b[1]) for a, b in zip(path, path[1:], strict=False))
    assert waits > 0, "passed a blocked corridor without waiting"


def test_passes_just_before_a_mover_arrives_instead_of_waiting():
    """Timing cuts both ways: go through ahead of a mover, not behind it.

    The mover occupies the corridor cell at column 8 from step 9 to the end of
    the window. The robot reaches column 8 at step 8 if it does not dawdle, so
    passing just ahead takes 11 steps; missing that and waiting it out would
    take past step 30. A planner too conservative to thread the gap is caught.
    """
    static = _corridor(12)
    window = 30
    movers = np.zeros((window, 3, 12), dtype=bool)
    movers[9:window, 1, 8] = True
    path = _plan(static, movers, (1, 0), (1, 11))
    _assert_valid(path, static, movers, (1, 0), (1, 11))
    assert len(path) - 1 == 11, f"took {len(path) - 1} steps instead of passing ahead at 11"


def test_never_swaps_cells_with_a_mover():
    """A robot and a mover exchanging adjacent cells between steps would pass
    through each other, invisible to a check of the destination alone.

    Built so the swap is the *only* shortest route and a legal alternative
    exists: a two-row corridor, the mover walking left along the robot's row.
    The test then proves it is not vacuous -- that a planner without the swap
    guard would have swapped -- before trusting the real planner's answer.
    """
    static = np.ones((4, 10), dtype=bool)
    static[1:3, :] = False  # rows 1 and 2 free
    window = 30
    movers = np.zeros((window, 4, 10), dtype=bool)
    for k in range(window):
        col = 9 - k
        if 0 <= col < 10:
            movers[k, 1, col] = True

    # Non-vacuity: the straight run along row 1 swaps with the mover.
    straight = [(1, k) for k in range(10)]
    swaps = [
        k for k in range(9)
        if movers[k + 1, 1, straight[k][1]] and movers[k, 1, straight[k + 1][1]]
    ]
    assert swaps, "test setup does not actually present a swap"

    path = _plan(static, movers, (1, 0), (1, 9))
    assert path is not None, "a legal route exists via row 2"
    _assert_valid(path, static, movers, (1, 0), (1, 9))
    for (_r0, _c0, k0), (r1, c1, k1) in zip(path, path[1:], strict=False):
        if k1 < window:
            assert not movers[k0, r1, c1], "moved into a cell a mover held that step"


def test_refuses_to_cut_a_corner_between_two_blocked_cells():
    """A diagonal step squeezed between two blocked orthogonal cells clips both.

    Added after mutation testing: removing this rule from the planner left
    every other test passing, so the rule was unguarded. Here the diagonal is
    the only way to the goal, so a planner that allowed it returns a path and
    one that forbids it correctly returns none.
    """
    static = np.zeros((3, 3), dtype=bool)
    static[0, 1] = True
    static[1, 0] = True
    static[2, :] = True
    static[:, 2] = True
    movers = np.zeros((10, 3, 3), dtype=bool)
    assert _plan(static, movers, (0, 0), (1, 1)) is None


def test_refuses_to_cut_a_corner_past_a_mover():
    """The same rule must hold when one of the two cells is a mover, not map.

    Unlike the static case a path does exist -- the mover only lasts the
    window, after which the robot can go round -- so the assertion is that the
    diagonal is never taken while the mover is there, not that no path exists.
    """
    static = np.zeros((3, 3), dtype=bool)
    static[0, 1] = True
    static[2, :] = True
    static[:, 2] = True
    window = 10
    movers = np.zeros((window, 3, 3), dtype=bool)
    movers[:, 1, 0] = True
    path = _plan(static, movers, (0, 0), (1, 1))
    assert path is not None, "the route round opens once the window passes"
    _assert_valid(path, static, movers, (0, 0), (1, 1))
    for (r0, c0, _k0), (r1, c1, k1) in zip(path, path[1:], strict=False):
        cut = (r0, c0) == (0, 0) and (r1, c1) == (1, 1)
        assert not (cut and k1 < window), f"cut the corner past the mover at step {k1}"


def test_unreachable_goal_returns_none():
    static = np.zeros((5, 5), dtype=bool)
    static[:, 2] = True
    movers = np.zeros((10, 5, 5), dtype=bool)
    assert _plan(static, movers, (0, 0), (0, 4)) is None


def test_starting_inside_a_mover_returns_none():
    static = np.zeros((5, 5), dtype=bool)
    movers = np.zeros((10, 5, 5), dtype=bool)
    movers[0, 2, 2] = True
    assert _plan(static, movers, (2, 2), (4, 4)) is None


@pytest.mark.parametrize("seed", range(40))
def test_matches_brute_force_minimum_time(seed):
    """Optimality, not merely feasibility.

    Random static maps and random mover occupancy, on a window long enough that
    nothing collapses, compared against exhaustive search. A heuristic that
    overestimated, or a closed set that pruned a state still worth reaching,
    shows up here as a longer path than brute force finds.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 11))
    static = rng.random((n, n)) < 0.18
    movers = rng.random((3 * n, n, n)) < 0.10
    free = np.argwhere(~static)
    if len(free) < 2:
        return
    start = tuple(int(v) for v in free[rng.integers(len(free))])
    goal = tuple(int(v) for v in free[rng.integers(len(free))])
    movers[0][start] = False
    window = movers.shape[0]

    expected = _bfs_min_steps(static, movers, start, goal, max_steps=window - 1)
    path = _plan(static, movers, start, goal, max_steps=window - 1)
    if expected is None:
        # The planner may still find a path that uses the static tail past the
        # window, which brute force over the window alone cannot; that is the
        # collapse working, not an error. It must never find one strictly
        # inside the window that brute force missed.
        if path is not None:
            assert len(path) - 1 >= window - 1
        return
    assert path is not None, "brute force found a path the planner missed"
    _assert_valid(path, static, movers, start, goal)
    assert len(path) - 1 == expected, f"planner {len(path) - 1} steps, optimum {expected}"


def test_past_the_window_only_the_map_matters():
    """Collapse: a mover recorded beyond the window must not block anything."""
    static = _corridor(20)
    movers = np.zeros((5, 3, 20), dtype=bool)  # window of five steps only
    path = _plan(static, movers, (1, 0), (1, 19))
    _assert_valid(path, static, movers, (1, 0), (1, 19))
    assert len(path) - 1 == 19
