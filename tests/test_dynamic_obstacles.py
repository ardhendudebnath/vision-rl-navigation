"""Tests for moving obstacles and the map/sensor asymmetry they depend on.

The whole point of this condition is that the movers are **visible to sensors
but absent from the map**. If that asymmetry leaked either way the experiment
would be meaningless: movers in the map would give the planner clairvoyance,
movers invisible to the lidar would give the policy a free pass.
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.sensors import Lidar2D, LidarConfig
from vision_nav.envs.splits import SHIFTS, shifted_config
from vision_nav.envs.world import WorldConfig, generate_world

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dynamic_experiment import sign_test  # noqa: E402
from fast_movers_experiment import tail_gain, verdict_lines  # noqa: E402

DYN = shifted_config(WorldConfig(), "dynamic")


# ----------------------------------------------------------------------
# Generation and motion
# ----------------------------------------------------------------------
def test_static_worlds_have_no_movers_by_default():
    """Every earlier experiment must be untouched."""
    w = generate_world(0)
    assert not w.has_dynamic
    assert len(w.dynamic) == 0


@pytest.mark.parametrize("seed", range(6))
def test_dynamic_worlds_generate_movers(seed):
    w = generate_world(seed, DYN)
    assert w.has_dynamic
    lo, hi = DYN.n_dynamic
    assert lo <= len(w.dynamic) <= hi


def test_movers_actually_move_and_are_periodic():
    w = generate_world(1, DYN)
    w.set_time(0.0)
    p0 = w._dyn_now[:, :2].copy()
    w.set_time(2.5)
    assert not np.allclose(p0, w._dyn_now[:, :2]), "movers did not move"
    # sin-driven, so returning to t=0 restores the original positions exactly.
    w.set_time(0.0)
    assert np.allclose(p0, w._dyn_now[:, :2])


def test_motion_is_deterministic_in_the_seed():
    a, b = generate_world(3, DYN), generate_world(3, DYN)
    assert np.array_equal(a.dynamic, b.dynamic)
    a.set_time(1.7)
    b.set_time(1.7)
    assert np.allclose(a._dyn_now, b._dyn_now)


def test_movers_never_sweep_through_static_geometry():
    """A mover embedded in a wall would be an unavoidable phantom collision."""
    for seed in range(8):
        w = generate_world(seed, DYN)
        for t in np.linspace(0, 20, 40):
            w.set_time(float(t))
            for x, y, r in w._dyn_now:
                clear = float(w.clearance(np.array([x, y]), include_dynamic=False))
                assert clear > r - 1e-6, f"mover clips static geometry at t={t}"


# ----------------------------------------------------------------------
# The map / sensor asymmetry
# ----------------------------------------------------------------------
def test_movers_are_absent_from_the_map():
    w = generate_world(2, DYN)
    w.set_time(0.0)
    static = w.occupancy_at(w.config.robot_radius, include_dynamic=False)
    with_dyn = w.occupancy_at(w.config.robot_radius, include_dynamic=True)
    assert with_dyn.sum() > static.sum(), "movers should block cells when included"
    # The default grid -- what the planner reads -- must be the static one.
    assert np.array_equal(w.occupancy, static)


def test_movers_are_visible_to_the_sensor():
    w = generate_world(2, DYN)
    w.set_time(0.0)
    assert len(w.sensed_circles) == len(w.circles) + len(w.dynamic)


def test_a_mover_placed_ahead_shortens_the_beam():
    """End-to-end: the range sensor must actually return the mover."""
    w = generate_world(4, DYN)
    w.set_time(0.0)
    lidar = Lidar2D(LidarConfig(n_beams=180, max_range=8.0))
    pose = w.start
    with_movers = lidar.scan(w, pose)
    saved = w._dyn_now
    w._dyn_now = np.zeros((0, 3))
    without = lidar.scan(w, pose)
    w._dyn_now = saved
    assert np.any(with_movers < without - 1e-6), "movers did not occlude any beam"
    assert np.all(with_movers <= without + 1e-6), "movers cannot lengthen a beam"


def test_clearance_includes_movers_by_default():
    w = generate_world(5, DYN)
    w.set_time(0.0)
    x, y, r = w._dyn_now[0]
    at_centre = np.array([x, y])
    assert float(w.clearance(at_centre, include_dynamic=True)) < 0.0
    assert float(w.clearance(at_centre, include_dynamic=False)) > 0.0


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
def test_env_advances_movers_each_step():
    cfg = NavEnvConfig(world_seeds=[0], world=DYN)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": 0})
    start = env.world._dyn_now[:, :2].copy()
    for _ in range(30):
        env.step(np.array([0.0, 0.0]))
    assert not np.allclose(start, env.world._dyn_now[:, :2])


def test_reset_rewinds_the_movers():
    """A cached world reused across episodes must not carry stale positions."""
    cfg = NavEnvConfig(world_seeds=[0], world=DYN)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": 0})
    first = env.world._dyn_now.copy()
    for _ in range(40):
        env.step(np.array([1.0, 0.2]))
    env.reset(options={"world_seed": 0})
    assert np.allclose(first, env.world._dyn_now)


def test_dynamic_shifts_are_registered():
    assert "dynamic" in SHIFTS and "dynamic_dense" in SHIFTS
    assert shifted_config(WorldConfig(), "dynamic").n_dynamic == (3, 6)


# ----------------------------------------------------------------------
# Sign test
# ----------------------------------------------------------------------
def test_all_seeds_above_reference_hits_the_floor():
    """Six seeds cannot produce a p below 2/64, whatever the margin."""
    k, p = sign_test(np.array([0.9] * 6), 0.5)
    assert k == 6
    assert p == pytest.approx(2 / 64)


def test_even_split_is_not_significant():
    k, p = sign_test(np.array([0.6, 0.6, 0.6, 0.4, 0.4, 0.4]), 0.5)
    assert k == 3 and p == pytest.approx(1.0)


def test_ties_are_dropped_conservatively():
    """A seed equal to the reference is evidence for neither side."""
    k, p = sign_test(np.array([0.5, 0.5, 0.9, 0.9]), 0.5)
    assert k == 2
    # Only two non-tied values remain, so the floor is 2/4.
    assert p == pytest.approx(2 * comb(2, 0) / 4)


def test_all_ties_gives_p_of_one():
    k, p = sign_test(np.array([0.5, 0.5, 0.5]), 0.5)
    assert k == 0 and p == 1.0


def test_sign_test_is_symmetric():
    above = sign_test(np.array([0.7, 0.8, 0.9, 0.75, 0.85, 0.95]), 0.5)
    below = sign_test(np.array([0.3, 0.2, 0.1, 0.25, 0.15, 0.05]), 0.5)
    assert above[1] == pytest.approx(below[1])


# ----------------------------------------------------------------------
# Frozen movers: the controlled subtraction of report Section 9.3
# ----------------------------------------------------------------------
def _frozen_and_moving(seed):
    dense = shifted_config(WorldConfig(), "dynamic_dense")
    moving = ProceduralNavEnv(NavEnvConfig(world_seeds=[seed], world=dense))
    frozen = ProceduralNavEnv(
        NavEnvConfig(world_seeds=[seed], world=dense, freeze_dynamic=True)
    )
    moving.reset(options={"world_seed": seed})
    frozen.reset(options={"world_seed": seed})
    return moving, frozen


@pytest.mark.parametrize("seed", [30000, 30001, 30017, 30099])
def test_frozen_matches_dynamic_dense(seed):
    """Freezing must remove motion and change nothing else.

    Zeroing ``dynamic_amplitude`` in the world config looks equivalent and is
    not: mover placement validates the swept path, so a zero sweep accepts
    positions the moving config rejects and the worlds end up with different
    obstacles. That would confound the subtraction with a geometry change, so
    the freeze happens after generation and this asserts the geometry survives.
    """
    moving, frozen = _frozen_and_moving(seed)
    m, f = moving.world, frozen.world

    assert np.allclose(m.circles, f.circles)
    assert np.allclose(m.boxes, f.boxes)
    assert np.allclose(m.start, f.start)
    assert np.allclose(m.goal, f.goal)
    # Same movers, same places, same radii -- differing only in amplitude.
    assert m.dynamic.shape == f.dynamic.shape
    assert np.allclose(m.dynamic[:, :5], f.dynamic[:, :5])
    assert np.allclose(f.dynamic[:, 5], 0.0)
    assert np.all(m.dynamic[:, 5] > 0.0)
    # The optimal path is measured on the static map, so l* must be unchanged
    # -- otherwise SPL would not be comparable between the two conditions.
    assert moving._shortest_path == pytest.approx(frozen._shortest_path)


def test_frozen_movers_do_not_move_but_are_still_sensed():
    moving, frozen = _frozen_and_moving(30000)

    at_zero = frozen.world.sensed_circles.copy()
    for _ in range(40):
        frozen.step(np.array([0.0, 0.0]))
    assert np.allclose(at_zero, frozen.world.sensed_circles)

    before = moving.world.sensed_circles.copy()
    for _ in range(40):
        moving.step(np.array([0.0, 0.0]))
    assert not np.allclose(before, moving.world.sensed_circles)

    # Frozen movers are obstacles, not ghosts: still sensed, still absent from
    # the map. The map has to stay wrong or the subtraction removes two things.
    w = frozen.world
    assert len(w.sensed_circles) == len(w.circles) + len(w.dynamic)
    assert np.array_equal(
        w.occupancy, w.occupancy_at(w.config.robot_radius, include_dynamic=False)
    )


def test_freeze_dynamic_is_off_by_default():
    """Every earlier result must be unaffected."""
    assert NavEnvConfig().freeze_dynamic is False


def test_frozen_conditions_pair_with_their_moving_twins():
    """A frozen condition must differ from its twin only by the freeze.

    The subtraction compares `dynamic_dense_frozen` against `dynamic_dense`,
    so if the two ever pointed at different splits or shifts the comparison
    would silently stop being a subtraction. Two runners consume this table
    (run_benchmark and the Nav2 bridge), which is exactly the drift this
    project has been bitten by before.
    """
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS

    assert FROZEN_CONDITIONS <= set(DYNAMIC_CONDITIONS)
    for name in FROZEN_CONDITIONS:
        twin = name.removesuffix("_frozen")
        assert twin in DYNAMIC_CONDITIONS, f"{name} has no moving twin"
        assert DYNAMIC_CONDITIONS[name] == DYNAMIC_CONDITIONS[twin], (
            f"{name} and {twin} must share split, shift and sensor noise"
        )
        assert twin not in FROZEN_CONDITIONS


def test_frozen_flag_reaches_the_env_through_config():
    """build_env_config must carry freeze_dynamic, not drop it silently."""
    from vision_nav.envs.splits import DYNAMIC_CONDITIONS
    from vision_nav.training.env_factory import build_env_config

    split, shift, noise = DYNAMIC_CONDITIONS["dynamic_dense_frozen"]
    frozen = build_env_config({"freeze_dynamic": True}, split=split, shift=shift,
                              n_worlds=4)
    moving = build_env_config({}, split=split, shift=shift, n_worlds=4)

    assert frozen.freeze_dynamic is True
    assert moving.freeze_dynamic is False
    # Identical in every other respect, or it is not a subtraction.
    assert frozen.world == moving.world
    assert list(frozen.world_seeds) == list(moving.world_seeds)


# ----------------------------------------------------------------------
# Block-triggered replanning (report Section 9.5)
# ----------------------------------------------------------------------
def _run_classical(condition_world, seed, config, steps=400):
    from vision_nav.agents.classical import AStarPursuitAgent

    cfg = NavEnvConfig(world_seeds=[seed], world=condition_world)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    agent = AStarPursuitAgent(config, robot=cfg.robot)
    assert agent.start_episode(env.world, env.robot.pose)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(agent.act(env.robot.pose))
        if terminated or truncated:
            break
    return agent


def test_block_triggered_never_fires_on_a_correct_static_map():
    """With a perfect map the committed path is never blocked.

    This is why block-triggered replanning is safe to adopt everywhere: on the
    six static benchmark conditions it reduces exactly to replan_every=0, so
    the headline results are untouched by the Section 9.5 change.
    """
    from vision_nav.agents.classical import PursuitConfig

    static = shifted_config(WorldConfig(), "dense")
    for seed in (30000, 30001, 30002):
        agent = _run_classical(static, seed, PursuitConfig(replan_on_block=True))
        assert agent.replans == 0
        assert agent.churn_total == 0.0


def test_block_triggered_fires_when_a_mover_obstructs():
    """It must actually replan somewhere, or it is just replan_every=0."""
    from vision_nav.agents.classical import PursuitConfig

    dense = shifted_config(WorldConfig(), "dynamic_dense")
    fired = sum(
        _run_classical(dense, seed, PursuitConfig(replan_on_block=True)).replans
        for seed in range(30000, 30012)
    )
    assert fired > 0


def test_churn_counters_survive_a_replan():
    """reset() performs replans, so it must not zero the counters it feeds."""
    from vision_nav.agents.classical import PursuitConfig

    dense = shifted_config(WorldConfig(), "dynamic_dense")
    agent = _run_classical(dense, 30000, PursuitConfig(replan_every=10))
    assert agent.replans > 1, "expected the timer to fire repeatedly"
    assert agent.churn_mean == pytest.approx(agent.churn_total / agent.replans)


def test_start_episode_zeroes_churn():
    from vision_nav.agents.classical import AStarPursuitAgent, PursuitConfig

    dense = shifted_config(WorldConfig(), "dynamic_dense")
    agent = _run_classical(dense, 30000, PursuitConfig(replan_every=10))
    assert agent.replans > 0
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[30001], world=dense))
    env.reset(options={"world_seed": 30001})
    agent.robot = env.config.robot
    agent.start_episode(env.world, env.robot.pose)
    assert agent.replans == 0 and agent.churn_total == 0.0
    assert isinstance(agent, AStarPursuitAgent)


# ----------------------------------------------------------------------
# Faster movers: a clean speed manipulation
# ----------------------------------------------------------------------
def test_fast_matches_dynamic_geometry():
    """`dynamic_fast` must differ from `dynamic` in speed and nothing else.

    Speed is drawn after the placement test and only feeds omega, so the
    accepted mover set should be identical seed for seed. If that ever stopped
    holding, the experiment would confound speed with a different world.
    """
    slow = shifted_config(WorldConfig(), "dynamic")
    fast = shifted_config(WorldConfig(), "dynamic_fast")

    for seed in (30000, 30001, 30042, 30199):
        a, b = generate_world(seed, slow), generate_world(seed, fast)
        assert np.allclose(a.circles, b.circles)
        assert np.allclose(a.boxes, b.boxes)
        assert np.allclose(a.start, b.start)
        assert np.allclose(a.goal, b.goal)
        assert a.dynamic.shape == b.dynamic.shape
        # centre, radius, direction and amplitude identical; only omega differs
        assert np.allclose(a.dynamic[:, :6], b.dynamic[:, :6])
        assert np.all(b.dynamic[:, 6] > a.dynamic[:, 6])


def test_fast_movers_actually_reach_their_target_speed():
    """Peak speed is amplitude x omega, and must land in the configured band."""
    fast = shifted_config(WorldConfig(), "dynamic_fast")
    peaks = []
    for seed in range(30000, 30020):
        w = generate_world(seed, fast)
        peaks.extend(w.dynamic[:, 5] * w.dynamic[:, 6])
    peaks = np.asarray(peaks)
    assert len(peaks) > 20
    assert peaks.min() >= 0.8 - 1e-9
    assert peaks.max() <= 1.5 + 1e-9
    # And they genuinely outrun the robot, which is the point of the condition.
    assert peaks.mean() > 0.6


# ----------------------------------------------------------------------
# Explicit velocity channel
# ----------------------------------------------------------------------
def test_obs_velocity_is_off_by_default():
    """Every earlier result must be unaffected."""
    assert NavEnvConfig().obs_velocity is False
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    obs, _ = env.reset(options={"world_seed": 0})
    assert obs.shape[0] == env.lidar.config.n_beams + 5


def test_obs_velocity_appends_one_delta_per_beam():
    cfg = NavEnvConfig(world_seeds=[0], obs_velocity=True)
    env = ProceduralNavEnv(cfg)
    obs, _ = env.reset(options={"world_seed": 0})
    n = env.lidar.config.n_beams
    assert obs.shape[0] == n + 5 + n
    assert env.observation_space.shape == obs.shape


def test_delta_is_zero_on_the_first_step_then_tracks_motion():
    """Opening frame must imply no motion, matching how stacking initialises."""
    dyn = shifted_config(WorldConfig(), "dynamic_fast")
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[30000], world=dyn,
                                        obs_velocity=True))
    obs, _ = env.reset(options={"world_seed": 30000})
    n = env.lidar.config.n_beams
    assert np.allclose(obs[n + 5:], 0.0), "first step must imply zero motion"

    # Standing still in a world with fast movers still produces range changes.
    moved = False
    for _ in range(12):
        obs, _, term, trunc, _ = env.step(np.array([0.0, 0.0]))
        if np.any(np.abs(obs[n + 5:]) > 1e-6):
            moved = True
        if term or trunc:
            break
    assert moved, "movers should change ranges even from a stationary robot"


def test_reset_clears_the_delta_history():
    dyn = shifted_config(WorldConfig(), "dynamic_fast")
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[30000], world=dyn,
                                        obs_velocity=True))
    env.reset(options={"world_seed": 30000})
    for _ in range(8):
        env.step(np.array([0.5, 0.1]))
    obs, _ = env.reset(options={"world_seed": 30000})
    n = env.lidar.config.n_beams
    assert np.allclose(obs[n + 5:], 0.0), "stale delta leaked across episodes"


def test_obs_velocity_and_frame_stack_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ProceduralNavEnv(NavEnvConfig(obs_velocity=True, frame_stack=2))


def _history_run(tmp_path, name, rates):
    run = tmp_path / name
    run.mkdir()
    (run / "validation_history.json").write_text(
        json.dumps([{"timesteps": 50_000 * (i + 1), "success_rate": r}
                    for i, r in enumerate(rates)]),
        encoding="utf-8",
    )
    return run


def test_tail_gain_separates_a_converged_arm_from_a_climbing_one(tmp_path):
    """A null result and an under-trained arm produce the same endpoint.

    This is the only thing standing between "recurrence does not help" and
    "the recurrent arm had not finished training", so it has to actually
    distinguish the two rather than merely return a number.
    """
    flat = _history_run(tmp_path, "flat", [0.1, 0.2, 0.3] + [0.40, 0.41, 0.39] * 4)
    climbing = _history_run(tmp_path, "climbing", [0.05 * i for i in range(15)])

    assert abs(np.mean(tail_gain([flat]))) <= 0.03
    assert np.mean(tail_gain([climbing])) > 0.03


def _report(fast_delta, fast_p, slow_delta, slow_p, coll=0.0, tmo=0.0):
    """Minimal report shaped like the one fast_movers_experiment writes."""
    def cell(delta, p):
        return {
            "base": [0.5] * 6,
            "treated": [0.5 + delta] * 6,
            "outcomes": {
                "base": {"success": [0.5] * 6, "collision": [0.3] * 6,
                         "timeout": [0.05] * 6},
                "treated": {"success": [0.5 + delta] * 6,
                            "collision": [0.3 + coll] * 6,
                            "timeout": [0.05 + tmo] * 6},
            },
            "delta": delta, "p": p, "seeds_higher": 6 if delta > 0 else 0,
        }
    return {"conditions": {"dynamic_fast": cell(fast_delta, fast_p),
                           "dynamic": cell(slow_delta, slow_p)}}


def test_verdict_never_calls_a_significant_harm_no_effect():
    """The bug this function was extracted for.

    The original could only test for *help*, so every other outcome fell to a
    branch announcing "NO EFFECT ON THE PRIMARY ENDPOINT" -- which it printed
    over the real recurrence result, a -0.068 at p = 0.017 carried by 5 of 6
    seeds. A verdict that cannot say "worse" mislabels a real effect as a null
    every time one appears, and it goes straight into the writeup.
    """
    text = " ".join(verdict_lines(_report(-0.068, 0.017, -0.078, 0.004),
                                  ["base", "treated"]))
    assert "NO EFFECT" not in text
    assert "HURTS" in text


def test_verdict_uses_the_control_cell_to_refuse_a_mechanism_claim():
    """A control that moves as much as the treatment leaves the question open."""
    both = " ".join(verdict_lines(_report(-0.068, 0.017, -0.078, 0.004),
                                  ["base", "treated"]))
    assert "NOT SPECIFIC TO MOTION" in both
    assert "unanswered" in both

    only_fast = " ".join(verdict_lines(_report(-0.068, 0.017, -0.005, 0.9),
                                       ["base", "treated"]))
    assert "SPECIFIC TO FAST MOVERS" in only_fast


def test_verdict_separates_reward_masking_from_a_worse_policy():
    """Collisions down with timeouts up is Phase 5h; collisions up is not."""
    masked = " ".join(verdict_lines(
        _report(-0.027, 0.44, -0.002, 1.0, coll=-0.018, tmo=0.045),
        ["base", "treated"]))
    assert "PHASE 5h SIGNATURE" in masked

    worse = " ".join(verdict_lines(
        _report(-0.068, 0.017, -0.078, 0.004, coll=0.023, tmo=0.045),
        ["base", "treated"]))
    assert "PHASE 5h SIGNATURE" not in worse
    assert "not reward masking" in worse


def test_tail_gain_reads_the_finished_arms_as_converged():
    """The threshold is calibrated against arms that ran the full budget.

    dynfast1 is the baseline of the recurrence comparison itself: a threshold
    that called it under-trained would invalidate the comparison rather than
    protect it. Skipped when the runs are absent, since runs/ is not in git.
    """
    runs = sorted(Path(__file__).resolve().parents[1].glob("runs/dynfast1_s*"))
    runs = [r for r in runs if (r / "validation_history.json").exists()]
    if len(runs) < 6:
        pytest.skip("training runs not present in this checkout")
    assert abs(float(np.mean(tail_gain(runs)))) <= 0.03
