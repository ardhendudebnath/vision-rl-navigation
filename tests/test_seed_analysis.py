"""Tests for the across-seed permutation test.

At four seeds per arm the statistics are doing real work, so the test itself
needs to be right -- an exact permutation test that silently mis-enumerates
would hand back confident-looking p-values with nothing behind them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from omegaconf import OmegaConf  # noqa: E402

from seed_analysis import describe_sensor, load_arms, permutation_p, run_sensor  # noqa: E402


def test_identical_groups_give_p_of_one():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    assert permutation_p(a, a.copy()) == pytest.approx(1.0)


def test_maximally_separated_groups_hit_the_resolution_floor():
    """4-vs-4 cannot produce a p below 2/70, however large the gap."""
    a = np.array([0.10, 0.11, 0.12, 0.13])
    b = np.array([0.90, 0.91, 0.92, 0.93])
    assert permutation_p(a, b) == pytest.approx(2 / 70)


def test_p_value_is_symmetric_in_its_arguments():
    a = np.array([0.60, 0.62, 0.58, 0.64])
    b = np.array([0.70, 0.68, 0.72, 0.66])
    assert permutation_p(a, b) == pytest.approx(permutation_p(b, a))


def test_overlapping_groups_are_not_significant():
    a = np.array([0.60, 0.65, 0.70, 0.55])
    b = np.array([0.62, 0.68, 0.58, 0.66])
    assert permutation_p(a, b) > 0.05


def test_p_value_is_a_probability():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.random(4)
        b = rng.random(4)
        p = permutation_p(a, b)
        assert 0.0 < p <= 1.0


def test_enumeration_covers_every_split():
    """Every one of the 70 relabellings must be counted exactly once."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    p = permutation_p(a, b)
    # p is count/70, so 70*p must be a whole number.
    assert abs(70 * p - round(70 * p)) < 1e-9


def test_unequal_group_sizes_are_handled():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.8, 0.9, 0.85, 0.95, 0.87])
    p = permutation_p(a, b)
    assert 0.0 < p < 0.05


def test_malformed_arm_spec_is_refused():
    with pytest.raises(ValueError, match="malformed --arm"):
        load_arms(["justaname"])


def test_missing_model_is_refused(tmp_path):
    (tmp_path / "emptyrun").mkdir()
    with pytest.raises(FileNotFoundError, match="no best_model.zip"):
        load_arms([f"x={tmp_path / 'emptyrun'}"])


# ----------------------------------------------------------------------
# Observation mode must travel with the run
# ----------------------------------------------------------------------
def _write_run(tmp_path: Path, name: str, env: dict, seed: int = 0) -> Path:
    run = tmp_path / name
    run.mkdir()
    OmegaConf.save(OmegaConf.create({"env": env, "train": {"seed": seed}}), run / "config.yaml")
    (run / "best_model.zip").write_bytes(b"x")
    return run


def test_depth_run_yields_depth_mode_and_camera(tmp_path):
    """Regression: evaluating a depth policy in a lidar env is silently wrong.

    Returning only lidar geometry would leave obs_mode at its 'privileged'
    default, so the policy would be scored against a different sensor at a
    different dimensionality.
    """
    run = _write_run(
        tmp_path,
        "d",
        {
            "obs_mode": "depth",
            "lidar": {"n_beams": 32, "max_range": 6.0},
            "camera": {"fov": 1.5708, "width": 64, "max_range": 6.0},
        },
    )
    spec = run_sensor(run)
    assert spec["obs_mode"] == "depth"
    assert spec["camera"]["width"] == 64
    # The unused lidar block must NOT leak in and override the camera.
    assert "lidar" not in spec


def test_lidar_run_yields_privileged_mode_and_beams(tmp_path):
    run = _write_run(
        tmp_path, "l", {"obs_mode": "privileged", "lidar": {"n_beams": 64, "max_range": 6.0}}
    )
    spec = run_sensor(run)
    assert spec["obs_mode"] == "privileged"
    assert spec["lidar"]["n_beams"] == 64
    assert "camera" not in spec


def test_run_without_obs_mode_defaults_to_privileged(tmp_path):
    run = _write_run(tmp_path, "old", {"lidar": {"n_beams": 32, "max_range": 6.0}})
    assert run_sensor(run)["obs_mode"] == "privileged"


def test_rgb_run_yields_rgb_mode_and_camera(tmp_path):
    """An unhandled mode falls through to the lidar branch and mislabels."""
    run = _write_run(
        tmp_path,
        "r",
        {
            "obs_mode": "rgb",
            "lidar": {"n_beams": 32, "max_range": 6.0},
            "rgb_camera": {"fov": 1.5708, "width": 64, "height": 48, "max_range": 6.0},
        },
    )
    spec = run_sensor(run)
    assert spec["obs_mode"] == "rgb"
    assert spec["rgb_camera"]["width"] == 64
    assert "lidar" not in spec
    assert "rgb" in describe_sensor(run) and "64x48" in describe_sensor(run)
    assert "lidar" not in describe_sensor(run)


def test_sensor_description_distinguishes_the_arms(tmp_path):
    depth = _write_run(
        tmp_path, "d2",
        {"obs_mode": "depth", "camera": {"fov": 1.5708, "width": 64, "max_range": 6.0}},
    )
    lidar = _write_run(
        tmp_path, "l2", {"obs_mode": "privileged", "lidar": {"n_beams": 64, "max_range": 6.0}}
    )
    assert "depth" in describe_sensor(depth) and "64px" in describe_sensor(depth)
    assert "lidar" in describe_sensor(lidar) and "64 beams" in describe_sensor(lidar)
