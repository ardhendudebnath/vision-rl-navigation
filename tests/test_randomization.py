"""Tests for per-episode domain randomisation."""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.randomization import DomainRandomization
from vision_nav.envs.splits import SHIFTS, shifted_config
from vision_nav.envs.world import WorldConfig
from vision_nav.training.env_factory import build_env_config

DR = DomainRandomization(
    enabled=True,
    arena=(10.0, 16.0),
    start_goal_fraction=(0.40, 0.58),
    n_circles=(1, 22),
    circle_radius=(0.25, 1.4),
    n_boxes=(0, 10),
    box_size=(0.4, 2.0),
)


def test_disabled_randomisation_is_the_identity():
    base = WorldConfig()
    assert DomainRandomization(enabled=False).sample(base, 7) == base


def test_sampling_is_deterministic_in_the_seed():
    """The whole evaluation protocol and the world cache depend on this."""
    base = WorldConfig()
    for seed in (0, 1, 99, 12345):
        assert DR.sample(base, seed) == DR.sample(base, seed)


def test_different_seeds_give_different_configs():
    base = WorldConfig()
    widths = {DR.sample(base, s).width for s in range(25)}
    assert len(widths) > 15, "arena size is barely varying"


def test_sampled_values_stay_inside_their_ranges():
    base = WorldConfig()
    for seed in range(200):
        cfg = DR.sample(base, seed)
        assert 10.0 <= cfg.width <= 16.0
        assert 10.0 <= cfg.height <= 16.0
        assert cfg.n_circles == (1, 22)
        assert cfg.circle_radius == (0.25, 1.4)
        smaller = min(cfg.width, cfg.height)
        assert 0.40 * smaller - 1e-9 <= cfg.min_start_goal_dist <= 0.58 * smaller + 1e-9


def test_start_goal_distance_is_always_satisfiable():
    """A fraction of the arena, never an absolute distance.

    Sampling an absolute separation independently of arena size would let a
    10 m arena draw an 8 m separation, which world generation can satisfy only
    rarely -- and would surface much later as an opaque RuntimeError.
    """
    base = WorldConfig()
    for seed in range(200):
        cfg = DR.sample(base, seed)
        diagonal = np.hypot(cfg.width, cfg.height)
        assert cfg.min_start_goal_dist < diagonal


def test_randomisation_leaves_robot_and_grid_alone():
    """Robot radius is hardware, not environment; it must not be randomised."""
    base = WorldConfig()
    for seed in range(20):
        cfg = DR.sample(base, seed)
        assert cfg.robot_radius == base.robot_radius
        assert cfg.goal_tolerance == base.goal_tolerance
        assert cfg.grid_resolution == base.grid_resolution


def test_dr_range_covers_every_evaluation_shift():
    """The experiment is only meaningful if the shifts are inside the range.

    If a shift fell outside, the result would answer a different question than
    the one being asked.
    """
    base = WorldConfig()
    for name in SHIFTS:
        shifted = shifted_config(base, name)
        assert DR.n_circles[0] <= shifted.n_circles[0]
        assert shifted.n_circles[1] <= DR.n_circles[1]
        assert DR.n_boxes[0] <= shifted.n_boxes[0]
        assert shifted.n_boxes[1] <= DR.n_boxes[1]
        assert DR.circle_radius[0] <= shifted.circle_radius[0]
        assert shifted.circle_radius[1] <= DR.circle_radius[1]
        assert DR.arena[0] <= shifted.width <= DR.arena[1]


@pytest.mark.parametrize("seed", range(12))
def test_randomised_worlds_still_generate_and_are_solvable(seed):
    """Wider ranges must not produce worlds the generator cannot satisfy."""
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[seed], domain_randomization=DR))
    _, info = env.reset(options={"world_seed": seed})
    assert info["shortest_path_length"] > 0.0
    assert env.world.is_free(env.world.start[:2])
    assert env.world.is_free(env.world.goal)


def test_env_observation_space_is_unchanged_by_randomisation():
    """Both policies must share an observation space to be comparable."""
    plain = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    randomised = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], domain_randomization=DR))
    assert plain.observation_space == randomised.observation_space
    assert plain.action_space == randomised.action_space


def test_randomised_env_observations_stay_in_bounds():
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=list(range(10)), domain_randomization=DR))
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            obs, _ = env.reset()


def test_shift_plus_randomisation_is_refused():
    """Randomising would overwrite the very fields the shift defines."""
    overrides = {"domain_randomization": {"enabled": True, "n_circles": [1, 22]}}
    with pytest.raises(ValueError, match="cannot combine shift"):
        build_env_config(overrides, split="test_ood", shift="dense")


def test_shift_with_randomisation_disabled_is_fine():
    cfg = build_env_config(
        {"domain_randomization": {"enabled": False}}, split="test_ood", shift="dense"
    )
    assert cfg.world.n_circles == (14, 22)


def test_factory_builds_randomisation_from_config():
    cfg = build_env_config(
        {"domain_randomization": {"enabled": True, "arena": [10.0, 16.0]}},
        split="train",
        n_worlds=10,
    )
    assert cfg.domain_randomization.enabled
    assert cfg.domain_randomization.arena == (10.0, 16.0)


def test_factory_rejects_unknown_randomisation_fields():
    with pytest.raises(TypeError, match="unknown"):
        build_env_config({"domain_randomization": {"enabld": True}})
