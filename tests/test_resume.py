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
