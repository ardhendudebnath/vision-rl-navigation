"""Tests for the benchmark harness's per-actor sensor handling.

A policy trained with 64 lidar beams cannot be evaluated in a 32-beam env --
its observation is not even the right shape. So sensor *geometry* has to be
read from the run that produced the policy. Sensor *corruption* (noise,
dropout) is the opposite: it defines the evaluation condition and must never
be inherited from the training config, or the `noisy_lidar` row would quietly
evaluate a clean sensor and report it as robustness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_benchmark import (  # noqa: E402
    SENSOR_GEOMETRY_FIELDS,
    actor_env_spec,
    actor_sensor_geometry,
    apply_sensor_noise,
    build_actor_specs,
)


def _write_run(tmp_path: Path, **lidar) -> Path:
    """Create a fake run directory with a saved config and model."""
    run = tmp_path / "somerun"
    run.mkdir()
    cfg = {
        "env": {
            "lidar": {"n_beams": 32, "max_range": 6.0, "noise_std": 0.0, **lidar},
        }
    }
    OmegaConf.save(OmegaConf.create(cfg), run / "config.yaml")
    model = run / "best_model.zip"
    model.write_bytes(b"not a real model")
    return model


def test_geometry_is_read_from_the_run_config(tmp_path):
    model = _write_run(tmp_path, n_beams=128, max_range=8.0)
    geometry = actor_sensor_geometry(str(model))
    assert geometry["n_beams"] == 128
    assert geometry["max_range"] == 8.0


def test_noise_is_never_inherited_from_the_run_config(tmp_path):
    """The condition owns sensor corruption, not the training run."""
    model = _write_run(tmp_path, n_beams=64, noise_std=0.25, dropout_prob=0.3)
    geometry = actor_sensor_geometry(str(model))
    assert "noise_std" not in geometry
    assert "dropout_prob" not in geometry
    assert set(geometry) <= set(SENSOR_GEOMETRY_FIELDS)


def test_missing_config_yields_no_override(tmp_path):
    model = tmp_path / "bare" / "best_model.zip"
    model.parent.mkdir()
    model.write_bytes(b"x")
    assert actor_sensor_geometry(str(model)) == {}


def test_no_model_path_yields_no_override():
    assert actor_sensor_geometry(None) == {}


def test_actor_labels_default_to_the_run_directory(tmp_path):
    model = _write_run(tmp_path, n_beams=64)
    specs = build_actor_specs([str(model)])
    assert "somerun" in specs
    assert specs["somerun"]["kind"] == "rl"


def test_explicit_actor_label_wins(tmp_path):
    model = _write_run(tmp_path, n_beams=64)
    specs = build_actor_specs([f"mylabel={model}"])
    assert "mylabel" in specs


def test_duplicate_actor_labels_are_refused(tmp_path):
    model = _write_run(tmp_path, n_beams=64)
    with pytest.raises(ValueError, match="duplicate actor label"):
        build_actor_specs([f"a={model}", f"a={model}"])


def test_reserved_actor_labels_are_refused(tmp_path):
    model = _write_run(tmp_path, n_beams=64)
    with pytest.raises(ValueError, match="duplicate actor label"):
        build_actor_specs([f"classical={model}"])


def test_missing_model_file_is_refused():
    with pytest.raises(FileNotFoundError, match="model not found"):
        build_actor_specs(["ghost=no/such/model.zip"])


# ----------------------------------------------------------------------
# Observation mode and per-sensor noise
# ----------------------------------------------------------------------
def _write_depth_run(tmp_path: Path, width: int = 64) -> Path:
    run = tmp_path / "depthrun"
    run.mkdir()
    cfg = {
        "env": {
            "obs_mode": "depth",
            "lidar": {"n_beams": 32, "max_range": 6.0, "noise_std": 0.0},
            "camera": {"fov": 1.5708, "width": width, "max_range": 6.0, "noise_std": 0.0},
        }
    }
    OmegaConf.save(OmegaConf.create(cfg), run / "config.yaml")
    (run / "best_model.zip").write_bytes(b"x")
    return run / "best_model.zip"


def test_obs_mode_travels_with_the_policy(tmp_path):
    """A depth policy cannot be evaluated in a lidar env at all."""
    spec = actor_env_spec(str(_write_depth_run(tmp_path)))
    assert spec["obs_mode"] == "depth"
    assert spec["camera"]["width"] == 64


def test_lidar_runs_default_to_privileged_mode(tmp_path):
    spec = actor_env_spec(str(_write_run(tmp_path, n_beams=64)))
    assert spec["obs_mode"] == "privileged"
    assert spec["lidar"]["n_beams"] == 64


def test_noise_is_applied_to_the_camera_for_depth_policies(tmp_path):
    """The trap: a depth policy never reads lidar.noise_std.

    Applying the condition's noise to the lidar would leave the camera clean
    and report the result as robustness -- plausible-looking and wrong.
    """
    spec = apply_sensor_noise(actor_env_spec(str(_write_depth_run(tmp_path))), 0.10)
    assert spec["camera"]["noise_std"] == 0.10
    assert spec["lidar"].get("noise_std", 0.0) == 0.0


def test_noise_is_applied_to_the_lidar_for_privileged_policies(tmp_path):
    spec = apply_sensor_noise(actor_env_spec(str(_write_run(tmp_path, n_beams=64))), 0.10)
    assert spec["lidar"]["noise_std"] == 0.10
    assert spec.get("camera", {}).get("noise_std") is None


def test_zero_noise_leaves_the_spec_untouched(tmp_path):
    spec = actor_env_spec(str(_write_depth_run(tmp_path)))
    assert apply_sensor_noise(spec, 0.0) == spec


def test_apply_sensor_noise_does_not_mutate_its_input(tmp_path):
    """Specs are reused across conditions; in-place edits would leak noise."""
    spec = actor_env_spec(str(_write_depth_run(tmp_path)))
    apply_sensor_noise(spec, 0.10)
    assert spec["camera"].get("noise_std", 0.0) == 0.0
