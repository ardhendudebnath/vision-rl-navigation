"""The virtual TurtleBot3 must carry the limits it is said to carry."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

GAZEBO = Path(__file__).resolve().parents[1] / "gazebo_tb3"
WAFFLE = (GAZEBO / "gz_waffle.sdf").read_text(encoding="utf-8")


def _module():
    spec = importlib.util.spec_from_file_location("robot_sdf", GAZEBO / "robot_sdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_real_profile_is_a_waffle_pi():
    m = _module()
    model = m.robot_model(WAFFLE, "real")
    drive, lidar = m.drive_spec(model), m.lidar_spec(model)
    assert drive["max_linear_velocity"] == pytest.approx(0.26)
    assert drive["max_angular_velocity"] == pytest.approx(1.82)
    assert lidar["range_min"] == pytest.approx(0.12)
    assert lidar["range_max"] == pytest.approx(3.5)
    # Reverse is limited the same way.
    plugin = next(p for p in model.findall("plugin") if "diff-drive" in p.get("filename"))
    assert float(plugin.find("min_linear_velocity").text) == pytest.approx(-0.26)


def test_the_nav2_profile_leaves_nav2s_values():
    m = _module()
    model = m.robot_model(WAFFLE, "nav2")
    assert m.drive_spec(model)["max_linear_velocity"] == pytest.approx(0.46)
    assert m.lidar_spec(model)["range_max"] == pytest.approx(20.0)


def test_the_lidar_is_read_from_the_model():
    """360 samples at 5 Hz, 0.01 m noise, 0.064 m behind the turning point."""
    m = _module()
    spec = m.lidar_spec(m.robot_model(WAFFLE))
    assert spec["samples"] == 360 and spec["update_rate"] == pytest.approx(5.0)
    assert spec["noise_std"] == pytest.approx(0.01)
    assert spec["x_offset"] == pytest.approx(-0.064)
    assert spec["topic"] == "/scan"


def test_the_model_is_made_loadable_and_placed():
    m = _module()
    model = m.robot_model(WAFFLE, pose=(1.5, 2.5, 0.7))
    assert model.get("name") == "turtlebot3"
    assert [float(v) for v in model.find("pose").text.split()] == pytest.approx(
        [1.5, 2.5, 0.01, 0, 0, 0.7])
    uris = [u.text for u in model.iter("uri")]
    assert uris and all(u.startswith("model://turtlebot3_model/") for u in uris)
    assert not [s for s in model.iter("sensor") if s.get("type") == "depth"]


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError):
        _module().robot_model(WAFFLE, "imaginary")


def test_the_drive_reads_its_topics():
    m = _module()
    d = m.drive_spec(m.robot_model(WAFFLE))
    assert (d["cmd_vel_topic"], d["odom_topic"]) == ("/cmd_vel", "/odom")
    assert d["wheel_separation"] == pytest.approx(0.287)
    assert ET.tostring(m.robot_model(WAFFLE))  # serialisable for embedding
