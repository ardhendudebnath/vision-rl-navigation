"""Tests for the navigation metrics, especially SPL's definition."""

from __future__ import annotations

import pytest

from vision_nav.metrics.navigation import EpisodeResult, aggregate


def make(success=True, collision=False, steps=100, path=10.0, l_star=10.0, goal_dist=0.0):
    return EpisodeResult(
        world_seed=0,
        success=success,
        collision=collision,
        steps=steps,
        path_length=path,
        shortest_path_length=l_star,
        final_goal_distance=goal_dist,
    )


def test_spl_is_one_for_a_perfectly_optimal_episode():
    assert make(path=10.0, l_star=10.0).spl_term == pytest.approx(1.0)


def test_spl_penalises_a_detour_proportionally():
    assert make(path=20.0, l_star=10.0).spl_term == pytest.approx(0.5)


def test_spl_is_zero_for_any_failure():
    """A fast failure must never score better than a slow success."""
    assert make(success=False, path=1.0, l_star=10.0).spl_term == 0.0
    assert make(success=False, collision=True, path=0.5).spl_term == 0.0


def test_spl_never_exceeds_one_when_the_agent_beats_l_star():
    """A shorter-than-l* path caps at 1.0 rather than rewarding the shortfall."""
    assert make(path=5.0, l_star=10.0).spl_term == pytest.approx(1.0)


def test_timeout_is_the_residual_outcome():
    assert make(success=False, collision=False).timeout is True
    assert make(success=False, collision=True).timeout is False
    assert make(success=True).timeout is False


def test_aggregate_computes_each_rate():
    episodes = [
        make(success=True, steps=100, path=10.0, l_star=10.0),
        make(success=True, steps=200, path=20.0, l_star=10.0),
        make(success=False, collision=True, steps=30, goal_dist=5.0),
        make(success=False, collision=False, steps=500, goal_dist=3.0),
    ]
    m = aggregate(episodes)

    assert m.n_episodes == 4
    assert m.success_rate == pytest.approx(0.5)
    assert m.collision_rate == pytest.approx(0.25)
    assert m.timeout_rate == pytest.approx(0.25)
    assert m.spl == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)


def test_mean_steps_to_goal_ignores_failures():
    """Averaging failures in would let a policy look fast by crashing early."""
    episodes = [make(success=True, steps=200), make(success=False, collision=True, steps=5)]
    assert aggregate(episodes).mean_steps_to_goal == pytest.approx(200.0)


def test_mean_path_efficiency_ignores_failures():
    episodes = [make(success=True, path=20.0, l_star=10.0), make(success=False, path=1.0)]
    assert aggregate(episodes).mean_path_efficiency == pytest.approx(0.5)


def test_aggregate_of_all_failures_is_well_defined():
    m = aggregate([make(success=False, collision=True) for _ in range(3)])
    assert m.success_rate == 0.0
    assert m.spl == 0.0
    assert m.mean_steps_to_goal == 0.0


def test_aggregate_rejects_an_empty_set():
    with pytest.raises(ValueError, match="empty"):
        aggregate([])


def test_zero_length_task_does_not_divide_by_zero():
    assert make(path=0.0, l_star=0.0).spl_term == pytest.approx(1.0)


def test_from_info_reads_the_env_terminal_dict():
    info = {
        "world_seed": 17,
        "steps": 123,
        "goal_distance": 0.2,
        "path_length": 9.5,
        "shortest_path_length": 9.0,
        "is_success": True,
        "collision": False,
    }
    r = EpisodeResult.from_info(info)
    assert r.world_seed == 17 and r.success and not r.collision
    assert r.spl_term == pytest.approx(9.0 / 9.5)


def test_metrics_render_as_a_markdown_table():
    table = aggregate([make()]).as_table("Demo")
    assert "| Metric" in table and "SPL" in table and "Demo" in table
