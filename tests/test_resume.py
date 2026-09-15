"""Tests for resume-from-checkpoint safety checks.

Resuming is only equivalent to continuous training when hyperparameter
schedules are constant. A non-constant schedule restarts from the top on
resume, which produces a wrong run rather than a failed one -- so the guard is
worth testing directly.
"""

from __future__ import annotations

import pytest

from vision_nav.training.train import _assert_constant_schedule, _load_for_resume


class _FakeModel:
    """Stands in for a PPO instance; only the schedule attributes matter."""

    def __init__(self, **schedules):
        for name, fn in schedules.items():
            setattr(self, name, fn)


def test_constant_schedules_are_accepted():
    model = _FakeModel(
        lr_schedule=lambda p: 3e-4,
        clip_range=lambda p: 0.2,
    )
    _assert_constant_schedule(model)  # must not raise


def test_linear_learning_rate_schedule_is_refused():
    model = _FakeModel(lr_schedule=lambda p: 3e-4 * p)
    with pytest.raises(ValueError, match="non-constant schedule"):
        _assert_constant_schedule(model)


def test_linear_clip_range_is_refused():
    model = _FakeModel(lr_schedule=lambda p: 3e-4, clip_range=lambda p: 0.2 * p)
    with pytest.raises(ValueError, match="clip_range"):
        _assert_constant_schedule(model)


def test_missing_optional_schedule_is_ignored():
    """clip_range_vf is commonly None; that must not be an error."""
    model = _FakeModel(lr_schedule=lambda p: 3e-4, clip_range_vf=None)
    _assert_constant_schedule(model)


def test_error_message_names_the_offending_values():
    model = _FakeModel(lr_schedule=lambda p: 1e-3 * p)
    with pytest.raises(ValueError) as excinfo:
        _assert_constant_schedule(model)
    assert "lr_schedule" in str(excinfo.value)
    assert "0.0" in str(excinfo.value) or "0.001" in str(excinfo.value)


def test_missing_checkpoint_fails_clearly():
    with pytest.raises(FileNotFoundError, match="resume_from checkpoint not found"):
        _load_for_resume("does/not/exist.zip", None, "cpu")


# ----------------------------------------------------------------------
# run_spec must keep up with NavEnvConfig
# ----------------------------------------------------------------------
def test_run_spec_covers_every_env_field():
    """Every env field must be classified as observation-shaping or not.

    Four separate times a field was added to NavEnvConfig and not carried by
    env_overrides_for_run: obs_mode, the RGB camera, frame_stack, and
    obs_velocity. Three failed silently -- a policy scored against the wrong
    sensor -- and only the fourth crashed. A docstring asking the next person
    to remember is not a mechanism; this is.
    """
    from dataclasses import fields

    from vision_nav.envs.nav_env import NavEnvConfig
    from vision_nav.training.run_spec import CONDITION_FIELDS, OBSERVATION_FIELDS

    declared = OBSERVATION_FIELDS | CONDITION_FIELDS
    actual = {f.name for f in fields(NavEnvConfig)}

    unclassified = actual - declared
    assert not unclassified, (
        f"NavEnvConfig field(s) {sorted(unclassified)} are classified in neither OBSERVATION_FIELDS "
        "nor CONDITION_FIELDS in run_spec.py. If the field changes what the "
        "policy sees it must be carried by env_overrides_for_run; if it "
        "belongs to the evaluation condition, say so explicitly."
        
    )
    stale = declared - actual
    assert not stale, f"run_spec classifies fields that no longer exist: {sorted(stale)}"


def test_run_spec_carries_every_observation_field(tmp_path):
    """A run using an observation-shaping field must get it back."""
    from omegaconf import OmegaConf

    from vision_nav.training.run_spec import env_overrides_for_run

    run = tmp_path / "r"
    run.mkdir()
    OmegaConf.save(
        OmegaConf.create({
            "env": {
                "obs_mode": "privileged",
                "obs_velocity": True,
                "max_goal_distance": 20.0,
                "lidar": {"n_beams": 64, "fov": 6.28, "max_range": 6.0},
            },
            "train": {"seed": 0},
        }),
        run / "config.yaml",
    )
    spec = env_overrides_for_run(run)
    assert spec["obs_velocity"] is True
    assert spec["lidar"]["n_beams"] == 64
    assert spec["max_goal_distance"] == 20.0


# ----------------------------------------------------------------------
# Recurrent policies: the evaluation path must carry LSTM state
# ----------------------------------------------------------------------
def test_sb3actor_carries_recurrent_state_across_a_rollout():
    """A recurrent policy evaluated without state sees every step as the first.

    That destroys exactly the memory the policy was trained to use, and the
    result reads as "recurrence does not help" rather than as a broken
    evaluation path. SB3Actor is the single seam both training-time validation
    and every analysis script evaluate through, so this failure would be
    silent and everywhere.
    """
    import numpy as np

    from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
    from vision_nav.training.actors import SB3Actor

    class FakeRecurrent:
        """Records what it is handed, and hands back an evolving state."""

        def __init__(self):
            self.seen = []

        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            self.seen.append((state, None if episode_start is None
                              else bool(np.asarray(episode_start).ravel()[0])))
            nxt = 1 if state is None else state + 1
            return np.zeros(2, dtype=np.float32), nxt

    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    obs, _ = env.reset(options={"world_seed": 0})
    model = FakeRecurrent()
    actor = SB3Actor(model)
    actor.reset(env, obs)
    for _ in range(4):
        actor.act(env, obs)

    states = [s for s, _ in model.seen]
    starts = [e for _, e in model.seen]
    assert states == [None, 1, 2, 3], "state was not carried between steps"
    assert starts == [True, False, False, False], "episode_start not handled"

    # A new episode must clear it, or memory leaks across episode boundaries.
    actor.reset(env, obs)
    actor.act(env, obs)
    assert model.seen[-1] == (None, True), "reset did not clear recurrent state"


def test_sb3actor_is_unchanged_for_feedforward_policies():
    """Passing state to a non-recurrent policy must stay inert."""
    import numpy as np

    from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
    from vision_nav.training.actors import SB3Actor

    class FakeFeedForward:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            return np.array([0.5, -0.25], dtype=np.float32), None

    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    obs, _ = env.reset(options={"world_seed": 0})
    actor = SB3Actor(FakeFeedForward())
    actor.reset(env, obs)
    a = actor.act(env, obs)
    assert a.shape == (2,)
    assert actor._state is None


def test_build_actor_loads_a_recurrent_checkpoint_as_recurrent(tmp_path):
    """``PPO.load`` does not reject a RecurrentPPO checkpoint.

    It returns a ``PPO`` whose policy is a ``RecurrentActorCriticPolicy``,
    which mostly behaves because ``predict`` delegates to the policy -- but it
    is accidental, and a checkpoint should come back under the class that owns
    it. The first assertion pins the surprising half of that so a future SB3
    version changing it is a test failure rather than a silent behaviour change.
    """
    pytest.importorskip("sb3_contrib", reason="optional extra: pip install -e .[recurrent]")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO

    from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
    from vision_nav.training.actors import build_actor

    env = ProceduralNavEnv(NavEnvConfig(world_seeds=[0]))
    model = RecurrentPPO(
        "MlpLstmPolicy", env, n_steps=8, batch_size=8, device="cpu",
        policy_kwargs={"lstm_hidden_size": 8, "net_arch": [8]},
    )
    path = tmp_path / "recurrent_model"
    model.save(path)

    assert type(PPO.load(path, device="cpu")).__name__ == "PPO", (
        "PPO.load now rejects recurrent checkpoints; build_actor's detection "
        "can be simplified to a try/except"
    )

    actor = build_actor("rl", model_path=str(path))
    assert type(actor.model).__name__ == "RecurrentPPO"
    assert actor.model.policy.lstm_actor.hidden_size == 8
