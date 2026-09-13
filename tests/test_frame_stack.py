"""Tests for observation frame stacking.

Frame stacking exists for one reason: a single range scan carries obstacle
positions but no velocities, so a one-frame policy cannot anticipate a moving
obstacle any better than a replanning planner can. Two frames encode motion.
These tests check that the stack really does carry that information, and that
it is assembled in a consistent order — a silently reversed or zero-padded
history would look fine and teach the policy nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.splits import shifted_config
from vision_nav.envs.world import WorldConfig

DYN = shifted_config(WorldConfig(), "dynamic")


# ----------------------------------------------------------------------
# Shape and validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_observation_width_scales_with_the_stack(k):
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=k))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (37 * k,)
    assert env.observation_space.contains(obs)


def test_default_is_unstacked():
    """Every earlier experiment must be bit-identical."""
    assert NavEnvConfig().frame_stack == 1
    plain = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    stacked1 = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=1))
    a, _ = plain.reset(seed=0, options={"world_seed": 0})
    b, _ = stacked1.reset(seed=0, options={"world_seed": 0})
    assert np.array_equal(a, b)


def test_invalid_stack_is_refused():
    with pytest.raises(ValueError, match="frame_stack must be >= 1"):
        ProceduralNavEnv(NavEnvConfig(frame_stack=0))


def test_stacking_rgb_is_refused():
    """Stacking images would change the CNN input and confound Phase 3e."""
    with pytest.raises(NotImplementedError, match="not supported for obs_mode='rgb'"):
        ProceduralNavEnv(NavEnvConfig(obs_mode="rgb", frame_stack=2))


def test_passes_the_gymnasium_checker():
    check_env(ProceduralNavEnv(NavEnvConfig(world_seeds=[0, 1], frame_stack=3)),
              skip_render_check=True)


# ----------------------------------------------------------------------
# Contents and ordering
# ----------------------------------------------------------------------
def test_first_observation_repeats_the_opening_frame():
    """Zero padding would imply motion that never happened."""
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=3))
    obs, _ = env.reset(seed=0)
    a, b, c = obs[:37], obs[37:74], obs[74:]
    assert np.array_equal(a, b) and np.array_equal(b, c)


def test_frames_shift_one_position_per_step():
    """Ordering must be stable, or the policy learns time running backwards.

    Checked by the shift property: this step's *oldest* block must equal the
    previous step's *newest*. That pins the direction without needing to
    reconstruct a reading independently.
    """
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=2))
    env.reset(seed=0, options={"world_seed": 0})
    obs1, *_ = env.step(np.array([1.0, 0.3]))
    obs2, *_ = env.step(np.array([1.0, 0.3]))

    assert np.array_equal(obs2[:37], obs1[37:]), "history did not shift by one frame"
    # And the robot is moving, so the two blocks within a step differ.
    assert not np.array_equal(obs2[:37], obs2[37:])


def test_shift_property_holds_for_deeper_stacks():
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=4))
    env.reset(seed=0, options={"world_seed": 0})
    prev, *_ = env.step(np.array([1.0, 0.2]))
    for _ in range(5):
        cur, *_ = env.step(np.array([1.0, 0.2]))
        # Blocks 0..2 of this step are blocks 1..3 of the last.
        assert np.array_equal(cur[: 37 * 3], prev[37:]), "stack did not shift cleanly"
        prev = cur


def test_history_is_cleared_between_episodes():
    """A carried-over frame would leak the previous episode's world."""
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0, 1], frame_stack=2))
    env.reset(seed=0, options={"world_seed": 0})
    for _ in range(10):
        env.step(np.array([1.0, 0.1]))
    obs, _ = env.reset(options={"world_seed": 1})
    assert np.array_equal(obs[:37], obs[37:]), "new episode must start self-identical"


def test_stack_encodes_motion_of_a_mover():
    """The whole point: consecutive frames must differ when the world moves.

    With the robot held still, any change between frames can only come from
    the moving obstacles — which is precisely the signal a one-frame policy
    cannot access.
    """
    cfg = NavEnvConfig(world_seeds=[0], world=DYN, frame_stack=2)
    env = ProceduralNavEnv(cfg)
    env.reset(seed=0, options={"world_seed": 0})
    pose = env.robot.pose.copy()

    changed = False
    for _ in range(40):
        obs, *_ = env.step(np.array([-1.0, 0.0]))
        env.robot.reset(pose)  # pin the robot; only the movers evolve
        prev, latest = obs[:37], obs[37:]
        if not np.allclose(prev[:32], latest[:32], atol=1e-6):
            changed = True
            break
    assert changed, "stacked frames never differed despite moving obstacles"


def test_static_world_frames_match_when_the_robot_is_still():
    """Control for the test above: no motion in, no motion out."""
    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0], frame_stack=2))
    env.reset(seed=0, options={"world_seed": 0})
    pose = env.robot.pose.copy()
    zero = 2.0 * (0.0 - env.config.robot.min_linear_vel) / (
        env.config.robot.max_linear_vel - env.config.robot.min_linear_vel
    ) - 1.0
    for _ in range(5):
        obs, *_ = env.step(np.array([zero, 0.0]))
        env.robot.reset(pose)
    assert np.allclose(obs[:32], obs[37:69], atol=1e-6)
