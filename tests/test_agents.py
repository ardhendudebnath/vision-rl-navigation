"""Tests for the classical baseline and the evaluation harness.

The baseline is the control condition for every comparison in the project, so
these tests are really guarding the validity of the results: a silently
degraded planner would make the learned policy look good for the wrong
reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.splits import split_seeds
from vision_nav.training.actors import ClassicalActor, RandomActor, build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate, rollout


@pytest.fixture(scope="module")
def eval_config():
    return build_env_config({}, split="test", n_worlds=25)


def test_classical_baseline_solves_almost_every_world(eval_config):
    """Regression guard on baseline quality.

    Thresholds are deliberately below the measured values (SR 1.00, SPL ~0.99
    on this slice) so ordinary tuning does not break the suite, but any real
    regression in planning or tracking will.
    """
    actor = ClassicalActor(robot=eval_config.robot)
    metrics, _ = evaluate(actor, eval_config)

    assert metrics.success_rate >= 0.90, f"baseline degraded: {metrics.success_rate}"
    assert metrics.collision_rate <= 0.08, f"baseline collides: {metrics.collision_rate}"
    assert metrics.spl >= 0.85, f"baseline paths got worse: {metrics.spl}"


def test_baseline_clearly_beats_random(eval_config):
    classical, _ = evaluate(ClassicalActor(robot=eval_config.robot), eval_config)
    random_metrics, _ = evaluate(RandomActor(seed=0), eval_config, n_episodes=25)
    assert classical.success_rate > random_metrics.success_rate + 0.5


def test_evaluation_is_reproducible(eval_config):
    a, _ = evaluate(ClassicalActor(robot=eval_config.robot), eval_config, n_episodes=10)
    b, _ = evaluate(ClassicalActor(robot=eval_config.robot), eval_config, n_episodes=10)
    assert a.to_dict() == b.to_dict()


def test_evaluation_visits_the_requested_worlds(eval_config):
    seeds = split_seeds("test", 10)
    _, results = evaluate(ClassicalActor(robot=eval_config.robot), eval_config, world_seeds=seeds)
    assert [r.world_seed for r in results] == seeds


def test_planner_failure_is_recorded_not_skipped():
    """A refused plan must score as a failure, not vanish from the average."""

    class NeverPlans:
        name = "never-plans"

        def reset(self, env, obs):
            return False

        def act(self, env, obs):  # pragma: no cover - never called
            raise AssertionError("act() must not run after reset() failed")

    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    result, _ = rollout(env, NeverPlans(), world_seed=0)
    assert not result.success and result.steps == 0
    assert result.spl_term == 0.0


def test_rollout_can_collect_a_trajectory():
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    actor = ClassicalActor(robot=env.config.robot)
    result, path = rollout(env, actor, world_seed=0, collect_trajectory=True)
    assert path is not None
    assert len(path) == result.steps + 1
    assert np.allclose(path[0], env.world.start[:2])


def test_evaluation_without_any_worlds_is_an_error():
    with pytest.raises(ValueError, match="no evaluation worlds"):
        evaluate(RandomActor(), NavEnvConfig(world_seeds=[]))


def test_build_actor_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="unknown actor kind"):
        build_actor("magic")


def test_build_actor_requires_a_model_path_for_rl():
    with pytest.raises(ValueError, match="requires model_path"):
        build_actor("rl")


def test_env_factory_rejects_typos():
    """Silently ignoring an unknown key would mean a config that lies."""
    with pytest.raises(TypeError, match="unknown"):
        build_env_config({"world": {"widht": 10.0}})
    with pytest.raises(TypeError, match="unknown"):
        build_env_config({"max_epsiode_steps": 100})


def test_env_factory_applies_splits_and_shifts():
    cfg = build_env_config({}, split="test_ood", shift="dense", n_worlds=5)
    assert cfg.world_seeds == split_seeds("test_ood", 5)
    assert cfg.deterministic_seed_order is True
    assert cfg.world.n_circles == (14, 22)


def test_env_factory_leaves_train_split_sampled_randomly():
    cfg = build_env_config({}, split="train", n_worlds=50)
    assert cfg.deterministic_seed_order is False
