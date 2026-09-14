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
        "NavEnvConfig field(s) {} are classified in neither OBSERVATION_FIELDS "
        "nor CONDITION_FIELDS in run_spec.py. If the field changes what the "
        "policy sees it must be carried by env_overrides_for_run; if it "
        "belongs to the evaluation condition, say so explicitly."
        .format(sorted(unclassified))
    )
    stale = declared - actual
    assert not stale, "run_spec classifies fields that no longer exist: {}".format(
        sorted(stale))


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
