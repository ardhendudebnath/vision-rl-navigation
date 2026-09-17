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


def _bfs_min_steps(static, movers, start, goal, max_steps, tail=None):
    """Reference: exhaustive breadth-first search over (row, col, step).

    Uses exactly the planner's blocking rules and no collapse, so on a window
    at least as long as any path it is the ground-truth minimum time.
    """
    n_r, n_c = static.shape
    window = movers.shape[0]

    def blocked(r, c, k):
        if static[r, c]:
            return True
        if k < window:
            return bool(movers[k, r, c])
        return tail is not None and bool(tail[r, c])

    if blocked(start[0], start[1], 0):
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


def test_a_mover_that_never_moves_cannot_be_waited_out():
    """The exploit, and its fix, in one test.

    A mover sits in the only corridor for the whole window. If movers vanish
    past the window, the cheapest plan is to wait until it ends and drive
    through -- and since a replanning robot's window slides forward, it waits
    forever. The first assertion proves the exploit is real, so the second is
    not vacuous.
    """
    static = _corridor(12)
    window = 10
    movers = np.zeros((window, 3, 12), dtype=bool)
    movers[:, 1, 6] = True
    h = geodesic_distance_field(static, (1, 11))

    ghost = spacetime_astar(static, movers, (1, 0), (1, 11), h)
    assert ghost is not None, "without a tail the planner should exploit the window"
    assert any(c == 6 and k >= window for _r, c, k in ghost), "expected a pass after the window"

    fixed = spacetime_astar(static, movers, (1, 0), (1, 11), h, tail_occ=movers[-1])
    assert fixed is None, "a mover that never moves blocked the corridor and was driven through"


@pytest.mark.parametrize("seed", range(25))
def test_matches_brute_force_with_a_tail(seed):
    """Optimality still holds once obstacles persist past the window.

    The window is kept short so paths genuinely run into the tail, which is the
    part of the search this changes; brute force searches the same rule.
    """
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(6, 10))
    static = rng.random((n, n)) < 0.15
    window = int(rng.integers(3, 6))
    movers = rng.random((window, n, n)) < 0.12
    tail = rng.random((n, n)) < 0.08
    free = np.argwhere(~static)
    if len(free) < 2:
        return
    start = tuple(int(v) for v in free[rng.integers(len(free))])
    goal = tuple(int(v) for v in free[rng.integers(len(free))])
    movers[0][start] = False
    tail[goal] = False

    cap = 4 * n
    expected = _bfs_min_steps(static, movers, start, goal, max_steps=cap, tail=tail)
    path = _plan(static, movers, start, goal, max_steps=cap, tail_occ=tail)
    if expected is None:
        assert path is None, "planner found a path brute force says does not exist"
        return
    assert path is not None, "brute force found a path the planner missed"
    for _r, _c, k in path:
        r, c = path[k][0], path[k][1]
        assert not static[r, c]
        assert not (movers[k, r, c] if k < window else tail[r, c]), f"blocked cell at {k}"
    assert len(path) - 1 == expected, f"planner {len(path) - 1} steps, optimum {expected}"

# ============================================================================
# The agent: schedule tracking, and the controls the experiment rests on
# ============================================================================
def _episode(cond, seed, time_varying, margin=0, predictor="oracle"):
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig
    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, _ = DYNAMIC_CONDITIONS[cond]
    over = {"freeze_dynamic": True} if cond in FROZEN_CONDITIONS else {}
    cfg = build_env_config(over, split=split, shift=shift, n_worlds=seed + 1)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(cfg.world_seeds[seed])})
    agent = SpaceTimeAgent(SpaceTimeConfig(time_varying_movers=time_varying,
                                           temporal_margin_steps=margin,
                                           predictor=predictor),
                           robot=cfg.robot)
    assert agent.start_episode(env.world, env.robot.pose)
    traj, info, done = [env.robot.position.copy()], {}, False
    while not done:
        _, _, term, trunc, info = env.step(agent.act(env.robot.pose))
        done = term or trunc
        traj.append(env.robot.position.copy())
    return np.asarray(traj), info, agent


@pytest.mark.parametrize("seed", [0, 3])
def test_frozen_worlds_make_timing_knowledge_irrelevant(seed):
    """The identity control. A frozen mover occupies the same cells at every
    step, so the union over the window equals each step and the two variants
    must drive bit-identical trajectories."""
    full, _, _ = _episode("dynamic_dense_frozen", seed, True)
    swept, _, _ = _episode("dynamic_dense_frozen", seed, False)
    assert full.shape == swept.shape and np.array_equal(full, swept)


def test_timing_knowledge_is_not_silently_inert_on_moving_worlds():
    """An ablation arm identical to the treatment would report exactly the null
    the experiment could find, for no reason."""
    changed = 0
    for seed in range(6):
        full, _, _ = _episode("dynamic_dense", seed, True)
        swept, _, _ = _episode("dynamic_dense", seed, False)
        if full.shape != swept.shape or not np.array_equal(full, swept):
            changed += 1
    assert changed > 0


def test_frozen_worlds_plan_no_waits():
    """Guards the tail fix. Before it, a mover that vanished at the window's
    edge could be waited out, and the agent planned waits on worlds with
    nothing moving -- 1.7 an episode."""
    for seed in range(4):
        _, _, agent = _episode("dynamic_dense_frozen", seed, True)
        assert agent.planned_waits == 0, f"seed {seed} planned {agent.planned_waits} waits"


def test_controller_parameters_cannot_drift_from_the_spatial_baseline():
    """A difference between the agents must be a difference in planning."""
    from vision_nav.agents.classical import PursuitConfig
    from vision_nav.agents.spacetime import SpaceTimeConfig

    st, base = SpaceTimeConfig(), PursuitConfig()
    for name in ("safety_margin", "heading_gain", "turn_in_place_threshold",
                 "slowdown_radius", "caution_clearance", "min_speed_scale"):
        assert getattr(st, name) == getattr(base, name), name


def test_a_scheduled_wait_is_a_stop_on_the_ground():
    """The reason schedule tracking exists: pure pursuit would drive through a
    planned wait. Give the agent a schedule that holds position and check it
    commands zero forward speed."""
    from vision_nav.agents.spacetime import SpaceTimeAgent
    from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv

    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    env.reset(options={"world_seed": 0})
    agent = SpaceTimeAgent(robot=env.config.robot)
    agent.start_episode(env.world, env.robot.pose)
    here = env.robot.position.copy()
    agent._waypoints = np.stack([here, here, here])
    agent._times = env.world._t + np.array([0.0, 5.0, 10.0])
    agent._last_plan_t = env.world._t  # suppress the replan for this one step
    action = agent.act(env.robot.pose)
    assert action[0] == pytest.approx(agent._speed_to_action(0.0))
    assert action[1] == 0.0, "turned to face a point it already occupies"

# ============================================================================
# The experiment's verdict: it must be able to say "worse"
# ============================================================================
def _verdict():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from spacetime_experiment import verdict

    return verdict


def test_a_significant_harm_is_never_called_inconclusive():
    """The values the first version mislabelled, verbatim: full vs swept on
    `dynamic`, -0.050 at p = 0.002, ten episodes lost and none won."""
    assert _verdict()(-0.050, 0.0020, [-0.085, -0.020]) == "HARMS"


def test_the_registered_primary_still_reads_as_it_did():
    """Adding HARMS must not change the pre-registered call on the primary."""
    assert _verdict()(0.050, 0.0309, [0.010, 0.090]) == "MATTERS"


@pytest.mark.parametrize("gain, p, ci, expected", [
    (0.010, 0.60, [-0.020, 0.025], "INERT (bounded)"),
    (0.000, 1.00, [-0.40, 0.40], "inconclusive"),   # the 5o trap: wide, includes 0
    (0.020, 0.01, [0.005, 0.035], "inconclusive"),  # significant but under the bound
    (-0.020, 0.01, [-0.035, -0.005], "inconclusive"),
])
def test_bounded_null_and_boundaries(gain, p, ci, expected):
    """An interval that merely includes zero is not a null; the bound must hold."""
    assert _verdict()(gain, p, ci) == expected

# ============================================================================
# Temporal safety margin
# ============================================================================
def test_dilation_blocks_exactly_the_margin_either_side():
    from vision_nav.agents.spacetime import dilate_in_time

    occ = np.zeros((12, 3), dtype=bool)
    occ[5, 1] = True
    out = dilate_in_time(occ, 2)
    assert [k for k in range(12) if out[k, 1]] == [3, 4, 5, 6, 7]
    assert not out[:, 0].any() and not out[:, 2].any(), "spread in space, not only time"


def test_dilation_clips_at_the_window_ends_rather_than_wrapping():
    from vision_nav.agents.spacetime import dilate_in_time

    occ = np.zeros((6, 1), dtype=bool)
    occ[0, 0] = True
    occ[5, 0] = True
    out = dilate_in_time(occ, 2)
    assert out[:, 0].tolist() == [True, True, True, True, True, True]
    occ2 = np.zeros((6, 1), dtype=bool)
    occ2[0, 0] = True
    assert dilate_in_time(occ2, 2)[:, 0].tolist() == [True, True, True, False, False, False]


def test_zero_margin_is_exactly_no_dilation():
    from vision_nav.agents.spacetime import dilate_in_time

    occ = np.random.default_rng(0).random((10, 20)) < 0.3
    assert dilate_in_time(occ, 0) is occ


@pytest.mark.parametrize("seed", [0, 3])
def test_a_margin_leaves_frozen_worlds_unchanged(seed):
    """The identity control carries over: a frozen mover is the same at every
    step, so widening it in time changes nothing."""
    plain, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=0)
    padded, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=4)
    assert plain.shape == padded.shape and np.array_equal(plain, padded)


def test_a_margin_reaches_the_planner_and_only_adds_blocked_cells():
    """"Not silently ignored", asked at the level where it cannot be luck.

    First written as "some trajectory changes in episodes 0-5", which failed
    with no bug present: a 2-step margin changes 8 of 30 sparse trajectories,
    but the first is episode 8. Checking which episodes happened to be sampled
    is the wrong test. What must hold is that the planner's mover occupancy
    changes -- and, a real property of a margin, that it only ever grows.
    """
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig
    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, _ = DYNAMIC_CONDITIONS["dynamic"]
    cfg = build_env_config({}, split=split, shift=shift, n_worlds=1)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(cfg.world_seeds[0])})
    radius = env.world.config.robot_radius + SpaceTimeConfig().safety_margin
    occ = {}
    for margin in (0, 2):
        agent = SpaceTimeAgent(SpaceTimeConfig(temporal_margin_steps=margin), robot=cfg.robot)
        agent.start_episode(env.world, env.robot.pose)
        occ[margin] = agent._mover_occupancy(radius, 30)
    assert not np.array_equal(occ[0], occ[2]), "the margin did not reach the planner"
    assert (occ[2] >= occ[0]).all(), "a margin removed a blocked cell"

# ============================================================================
# Constant-velocity estimate in place of the oracle
# ============================================================================
def _cv_agent_on(cond="dynamic"):
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig
    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, _ = DYNAMIC_CONDITIONS[cond]
    over = {"freeze_dynamic": True} if cond in FROZEN_CONDITIONS else {}
    cfg = build_env_config(over, split=split, shift=shift, n_worlds=1)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(cfg.world_seeds[0])})
    agent = SpaceTimeAgent(SpaceTimeConfig(predictor="constant_velocity"), robot=cfg.robot)
    agent.start_episode(env.world, env.robot.pose)
    return agent, env


def test_constant_velocity_extrapolates_in_a_straight_line():
    agent, env = _cv_agent_on()
    p0 = np.array([[1.0, 1.0, 0.3]])
    p1 = np.array([[1.2, 0.9, 0.3]])
    agent._observations = [(2.0, p0), (2.1, p1)]
    got = agent._predicted_discs(3.1)  # one second past the last observation
    np.testing.assert_allclose(got[0, :2], [1.2 + 2.0 * 1.0, 0.9 - 1.0 * 1.0])
    assert got[0, 2] == 0.3, "radius must not be extrapolated"


def test_constant_velocity_assumes_stationary_without_two_observations():
    agent, env = _cv_agent_on()
    p = np.array([[4.0, 5.0, 0.3]])
    agent._observations = [(1.0, p)]
    np.testing.assert_array_equal(agent._predicted_discs(9.0), p)
    agent._observations = [(1.0, p), (1.0, p + 1.0)]  # zero elapsed time
    np.testing.assert_array_equal(agent._predicted_discs(9.0), p + 1.0)


def test_constant_velocity_never_reads_the_true_future():
    """An estimate must be *wrong* where the truth curves, or it is the oracle.

    An estimator that quietly read the oracle would reproduce the oracle's
    result and look like a triumph of estimation. The first version of this
    test advanced the world clock and checked the prediction held still -- but
    the oracle is a pure function of absolute time, so a leaking estimator
    passed it too, and it caught nothing. Movers here run on sinusoids, so a
    genuine straight-line extrapolation two seconds out must disagree with
    where they really are.
    """
    agent, env = _cv_agent_on("dynamic")
    for _ in range(12):
        env.step(agent.act(env.robot.pose))
    t_future = env.world._t + 2.0
    estimated = agent._predicted_discs(t_future)
    truth = env.world.dynamic_at(t_future)
    assert len(truth), "needs movers to mean anything"
    gaps = np.linalg.norm(estimated[:, :2] - truth[:, :2], axis=1)
    assert gaps.max() > 0.05, f"estimate matches the true future (gaps {gaps}); oracle leak"


def test_estimate_reaches_the_planner_on_moving_worlds():
    """Not silently the oracle: its planning grid must differ once movers move."""
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent, env = _cv_agent_on("dynamic")
    for _ in range(5):
        env.step(agent.act(env.robot.pose))
    radius = env.world.config.robot_radius + agent.config.safety_margin
    cv = agent._mover_occupancy(radius, 30)
    oracle = SpaceTimeAgent(SpaceTimeConfig(predictor="oracle"), robot=env.config.robot)
    oracle._world = env.world
    assert not np.array_equal(cv, oracle._mover_occupancy(radius, 30))


@pytest.mark.parametrize("seed", [0, 3])
def test_estimate_is_exact_on_frozen_worlds(seed):
    """The identity control: a frozen mover has zero velocity, so the estimate
    and the oracle must drive bit-identical trajectories."""
    oracle, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=2, predictor="oracle")
    cv, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=2,
                        predictor="constant_velocity")
    assert oracle.shape == cv.shape and np.array_equal(oracle, cv)


def test_unknown_predictor_fails_loudly():
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent, env = _cv_agent_on()
    bad = SpaceTimeAgent(SpaceTimeConfig(predictor="kalman"), robot=env.config.robot)
    bad._world = env.world
    with pytest.raises(ValueError, match="unknown predictor"):
        bad._predicted_discs(env.world._t)

# ============================================================================
# Mover margin floor: withholding zero-margin plans around movers
# ============================================================================
def _radii_tried_when_every_plan_fails(floor, monkeypatch):
    """Force every search to fail, and record the static and mover radii the
    agent tried at each fallback."""
    from vision_nav.agents import spacetime as st

    agent, env = _cv_agent_on("dynamic")
    agent.config.mover_margin_floor = floor
    static_radii, mover_radii = [], []
    real_static, real_movers = agent._static, agent._mover_occupancy

    def spy_static(radius):
        static_radii.append(radius)
        return real_static(radius)

    def spy_movers(radius, steps):
        mover_radii.append(radius)
        return real_movers(radius, steps)

    monkeypatch.setattr(agent, "_static", spy_static)
    monkeypatch.setattr(agent, "_mover_occupancy", spy_movers)
    monkeypatch.setattr(st, "spacetime_astar", lambda *a, **k: None)
    before = agent.bare_radius_attempts
    assert agent._plan(env.robot.pose) is False
    return static_radii, mover_radii, agent.bare_radius_attempts - before, env, agent


def test_zero_floor_falls_back_to_the_bare_radius_everywhere(monkeypatch):
    """The default must be what Phase 5r ran: all three fallbacks, movers too."""
    static, movers, bare, env, agent = _radii_tried_when_every_plan_fails(0.0, monkeypatch)
    r, m = env.world.config.robot_radius, agent.config.safety_margin
    np.testing.assert_allclose(static, [r + m, r + m / 2, r])
    np.testing.assert_allclose(movers, static)
    assert bare == 1


def test_a_floor_withholds_the_bare_radius_from_movers_only(monkeypatch):
    """The treatment: a tight static passage may still use the bare radius, a
    mover never does."""
    static, movers, bare, env, agent = _radii_tried_when_every_plan_fails(0.5, monkeypatch)
    r, m = env.world.config.robot_radius, agent.config.safety_margin
    np.testing.assert_allclose(static, [r + m, r + m / 2, r])
    np.testing.assert_allclose(movers, [r + m, r + m / 2, r + m / 2])
    assert bare == 1, "the counter must count attempts at the last fallback either way"


@pytest.mark.parametrize("episode", [25, 33])
def test_a_floor_cannot_act_before_the_last_fallback_is_reached(episode):
    """The identity the experiment's analysis rests on: an episode in which the
    planner never reached the bare radius drives the same trajectory with or
    without the floor.

    First written over dense episodes 0-5, which passed -- and passed a mutant
    that leaked the floor into the *middle* fallback too, because none of those
    episodes fell back at all. These two do reach the middle fallback and never
    the last, and the test says so rather than assuming it.
    """
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig
    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, _ = DYNAMIC_CONDITIONS["dynamic_dense"]
    cfg = build_env_config({}, split=split, shift=shift, n_worlds=episode + 1)
    env = ProceduralNavEnv(cfg)

    def drive(floor):
        env.reset(options={"world_seed": int(cfg.world_seeds[episode])})
        agent = SpaceTimeAgent(SpaceTimeConfig(predictor="constant_velocity",
                                               temporal_margin_steps=2,
                                               mover_margin_floor=floor), robot=cfg.robot)
        middle = env.world.config.robot_radius + agent.config.safety_margin / 2
        tried, real = [], agent._static
        agent._static = lambda r: (tried.append(r), real(r))[1]
        agent.start_episode(env.world, env.robot.pose)
        traj, done = [env.robot.position.copy()], False
        while not done:
            _, _, term, trunc, _ = env.step(agent.act(env.robot.pose))
            done = term or trunc
            traj.append(env.robot.position.copy())
        return np.asarray(traj), int(np.isclose(tried, middle).sum()), agent.bare_radius_attempts

    base, middle_attempts, bare_attempts = drive(0.0)
    assert middle_attempts > 0, "episode no longer reaches the middle fallback; pick another"
    assert bare_attempts == 0, "episode now reaches the last fallback; pick another"
    floored, _, _ = drive(0.5)
    assert base.shape == floored.shape and np.array_equal(base, floored)

# ============================================================================
# Replanning when an observation contradicts the estimate
# ============================================================================
def _drive(cond, episode, **config):
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig
    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, _ = DYNAMIC_CONDITIONS[cond]
    over = {"freeze_dynamic": True} if cond in FROZEN_CONDITIONS else {}
    cfg = build_env_config(over, split=split, shift=shift, n_worlds=episode + 1)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(cfg.world_seeds[episode])})
    agent = SpaceTimeAgent(SpaceTimeConfig(temporal_margin_steps=2, **config), robot=cfg.robot)
    agent.start_episode(env.world, env.robot.pose)
    traj, done = [env.robot.position.copy()], False
    while not done:
        _, _, term, trunc, _ = env.step(agent.act(env.robot.pose))
        done = term or trunc
        traj.append(env.robot.position.copy())
    return np.asarray(traj), agent


def test_innovation_is_the_distance_from_the_plans_estimate_to_the_observation():
    agent, env = _cv_agent_on("dynamic")
    for _ in range(3):
        env.step(agent.act(env.robot.pose))
    w = env.world
    basis = [(w._t - 0.2, w._dyn_now.copy()), (w._t - 0.1, w._dyn_now.copy())]
    basis[0][1][:, 0] -= 0.05  # the estimate thinks every mover moves +0.5 m/s in x
    agent._plan_basis = basis
    # Carried 0.1 s past the newer observation, the estimate is 0.05 m off in x.
    assert agent._innovation() == pytest.approx(0.05)


def test_the_oracle_is_never_contradicted_so_the_trigger_is_inert():
    """The free control, on a *moving* world: the oracle's estimate is the truth,
    so an oracle agent with the trigger must drive the same trajectory as one
    without, and never fire it."""
    base, _ = _drive("dynamic_dense", 1, predictor="oracle")
    trig, agent = _drive("dynamic_dense", 1, predictor="oracle", replan_innovation_m=0.02)
    assert agent.innovation_replans == 0
    assert base.shape == trig.shape and np.array_equal(base, trig)


def test_a_frozen_mover_never_contradicts_a_constant_velocity_estimate():
    base, _ = _drive("dynamic_dense_frozen", 1, predictor="constant_velocity")
    trig, agent = _drive("dynamic_dense_frozen", 1, predictor="constant_velocity",
                         replan_innovation_m=0.02)
    assert agent.innovation_replans == 0
    assert base.shape == trig.shape and np.array_equal(base, trig)


def test_the_trigger_fires_on_a_moving_world_under_an_estimate():
    """Not silently inert where it is meant to act."""
    _, base = _drive("dynamic_dense", 1, predictor="constant_velocity")
    _, trig = _drive("dynamic_dense", 1, predictor="constant_velocity", replan_innovation_m=0.02)
    assert base.innovation_replans == 0
    assert trig.innovation_replans > 0

# ============================================================================
# Capping how far a constant-velocity estimate is carried
# ============================================================================
def test_a_capped_estimate_holds_the_mover_where_the_line_put_it():
    agent, env = _cv_agent_on()
    agent.config.extrapolation_cap_s = 1.0
    p0 = np.array([[1.0, 1.0, 0.3]])
    p1 = np.array([[1.2, 0.9, 0.3]])  # 2 m/s in x, -1 m/s in y
    agent._observations = [(2.0, p0), (2.1, p1)]
    capped_at = [1.2 + 2.0 * 1.0, 0.9 - 1.0 * 1.0]
    np.testing.assert_allclose(agent._predicted_discs(3.1)[0, :2], capped_at)
    np.testing.assert_allclose(agent._predicted_discs(9.1)[0, :2], capped_at)
    # Inside the cap the line is untouched.
    np.testing.assert_allclose(agent._predicted_discs(2.6)[0, :2], [1.2 + 1.0, 0.9 - 0.5])


def test_no_cap_is_exactly_the_uncapped_estimate():
    """``min(x, inf)`` is ``x``, so the default must be bit-identical, not close."""
    agent, env = _cv_agent_on()
    for _ in range(4):
        env.step(agent.act(env.robot.pose))
    t = env.world._t + 6.5
    uncapped = agent._predicted_discs(t)
    agent.config.extrapolation_cap_s = float("inf")
    assert np.array_equal(uncapped, agent._predicted_discs(t))


def test_a_cap_changes_nothing_on_frozen_worlds():
    base, _ = _drive("dynamic_dense_frozen", 1, predictor="constant_velocity")
    capped, _ = _drive("dynamic_dense_frozen", 1, predictor="constant_velocity",
                       extrapolation_cap_s=1.0)
    assert base.shape == capped.shape and np.array_equal(base, capped)


def test_a_cap_reaches_the_planner_on_moving_worlds():
    agent, env = _cv_agent_on("dynamic")
    for _ in range(5):
        env.step(agent.act(env.robot.pose))
    radius = env.world.config.robot_radius + agent.config.safety_margin
    uncapped = agent._mover_occupancy(radius, 30)
    agent.config.extrapolation_cap_s = 1.0
    assert not np.array_equal(uncapped, agent._mover_occupancy(radius, 30))

# ============================================================================
# The harmonic estimator: fitting the oscillation instead of drawing a line
# ============================================================================
def _sine_history(centre, direction, amplitude, omega, times):
    """Observations of one mover on centre + dir * A sin(w t), as the agent sees them."""
    out = []
    for t in times:
        offset = amplitude * np.sin(omega * t)
        pos = np.asarray(centre, dtype=float) + np.asarray(direction, dtype=float) * offset
        out.append((float(t), np.array([[pos[0], pos[1], 0.3]])))
    return out


def test_the_harmonic_fit_recovers_an_oscillation_it_was_never_told():
    """The fit sees positions only: no amplitude, no frequency, no centre."""
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent = SpaceTimeAgent(SpaceTimeConfig(predictor="harmonic"))
    centre, direction, amplitude, omega = (4.0, 5.0), (0.6, 0.8), 2.0, 0.35
    times = np.arange(0.0, 3.01, 0.1)
    seen = _sine_history(centre, direction, amplitude, omega, times)
    for horizon in (1.0, 3.0, 7.0):
        t = times[-1] + horizon
        truth = np.asarray(centre) + np.asarray(direction) * amplitude * np.sin(omega * t)
        got = agent._harmonic_discs(seen, t)
        assert got is not None
        assert np.linalg.norm(got[0, :2] - truth) < 1e-3, horizon
        assert got[0, 2] == 0.3, "radius must not be fitted"


def test_the_harmonic_fit_follows_its_observations_not_the_world():
    """The causality control. Given a history that disagrees with the world, the
    estimate must track the history -- an estimator reading the truth would not."""
    agent, env = _cv_agent_on("dynamic")
    agent.config.predictor = "harmonic"
    times = np.arange(env.world._t, env.world._t + 3.01, 0.1)
    invented = _sine_history((1.0, 1.0), (1.0, 0.0), 1.5, 0.5, times)
    t = times[-1] + 2.0
    got = agent._predicted_discs(t, observations=invented)
    expected = 1.0 + 1.5 * np.sin(0.5 * t)
    assert abs(got[0, 0] - expected) < 1e-3
    assert not np.allclose(got[0, :2], env.world.dynamic_at(t)[0, :2])


def test_too_little_track_falls_back_to_the_straight_line():
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent = SpaceTimeAgent(SpaceTimeConfig(predictor="harmonic"))
    short = _sine_history((4.0, 5.0), (1.0, 0.0), 2.0, 0.35, np.arange(0.0, 0.31, 0.1))
    assert agent._harmonic_discs(short, 3.0) is None
    line = agent._constant_velocity_discs(short, 3.0)
    np.testing.assert_array_equal(agent._predicted_discs(3.0, observations=short), line)


def test_a_frozen_mover_has_no_oscillation_to_fit():
    """Degenerate by construction, so the fit declines and the frozen identity
    holds: harmonic drives frozen worlds exactly as the oracle does."""
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent = SpaceTimeAgent(SpaceTimeConfig(predictor="harmonic"))
    still = _sine_history((4.0, 5.0), (1.0, 0.0), 0.0, 0.35, np.arange(0.0, 3.01, 0.1))
    assert agent._harmonic_discs(still, 5.0) is None


@pytest.mark.parametrize("seed", [0, 3])
def test_the_harmonic_estimate_is_exact_on_frozen_worlds(seed):
    oracle, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=2, predictor="oracle")
    fitted, _, _ = _episode("dynamic_dense_frozen", seed, True, margin=2, predictor="harmonic")
    assert oracle.shape == fitted.shape and np.array_equal(oracle, fitted)


def test_the_harmonic_estimate_reaches_the_planner():
    from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig

    agent, env = _cv_agent_on("dynamic")
    for _ in range(40):
        env.step(agent.act(env.robot.pose))
    radius = env.world.config.robot_radius + agent.config.safety_margin
    line = agent._mover_occupancy(radius, 30)
    fitted = SpaceTimeAgent(SpaceTimeConfig(predictor="harmonic"), robot=env.config.robot)
    fitted._world = env.world
    fitted._observations = [(t, p.copy()) for t, p in agent._observations]
    assert len(fitted._observations) >= 2
    fitted._observations = []
    for _ in range(35):
        fitted._observe()
        env.world.set_time(env.world._t + 0.1)
    assert not np.array_equal(line, fitted._mover_occupancy(radius, 30))