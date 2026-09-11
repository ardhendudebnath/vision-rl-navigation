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
    actor_sensor_geometry,
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
