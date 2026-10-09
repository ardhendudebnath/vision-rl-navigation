"""The Gazebo export of a world must be the same world."""

from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.training.env_factory import build_env_config

GAZEBO = Path(__file__).resolve().parents[1] / "gazebo_tb3"


def _module():
    spec = importlib.util.spec_from_file_location("world_sdf", GAZEBO / "world_sdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world(condition: str = "dense"):
    split, shift, _ = BENCHMARK_CONDITIONS[condition]
    cfg = build_env_config({}, split="val", shift=shift, n_worlds=2)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    return env.world


def test_the_walls_bound_the_arena_from_outside():
    m, world = _module(), _world()
    walls = {s["name"]: s for s in m.obstacles(world) if s["name"].startswith("wall")}
    w, h = world.config.width, world.config.height
    assert walls["wall_south"]["y"] + walls["wall_south"]["sy"] / 2 == pytest.approx(0.0)
    assert walls["wall_north"]["y"] - walls["wall_north"]["sy"] / 2 == pytest.approx(h)
    assert walls["wall_west"]["x"] + walls["wall_west"]["sx"] / 2 == pytest.approx(0.0)
    assert walls["wall_east"]["x"] - walls["wall_east"]["sx"] / 2 == pytest.approx(w)


def test_every_circle_and_box_is_carried_over_exactly():
    m, world = _module(), _world()
    shapes = m.obstacles(world)
    cyl = [s for s in shapes if s["kind"] == "cylinder"]
    boxes = [s for s in shapes if s["name"].startswith("box_")]
    assert len(cyl) == len(world.circles) and len(boxes) == len(world.boxes)
    for s, (x, y, r) in zip(cyl, world.circles, strict=True):
        assert (s["x"], s["y"], s["r"]) == pytest.approx((x, y, r))
    for s, (x0, y0, x1, y1) in zip(boxes, world.boxes, strict=True):
        assert (s["x"] - s["sx"] / 2, s["y"] - s["sy"] / 2) == pytest.approx((x0, y0))
        assert (s["x"] + s["sx"] / 2, s["y"] + s["sy"] / 2) == pytest.approx((x1, y1))


def _robot_module():
    spec = importlib.util.spec_from_file_location("robot_sdf", GAZEBO / "robot_sdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sdf_parses_and_places_the_robot_at_the_start():
    m, r, world = _module(), _robot_module(), _world("narrow")
    waffle = (GAZEBO / "gz_waffle.sdf").read_text(encoding="utf-8")
    robot = r.robot_model(waffle, pose=tuple(float(v) for v in world.start))
    root = ET.fromstring(m.world_sdf(world, robot_xml=ET.tostring(robot, encoding="unicode")))
    w = root.find("world")
    names = [mod.get("name") for mod in w.findall("model")]
    # Ground plane, goal marker, every solid, and the robot.
    assert len(names) == 3 + len(m.obstacles(world))
    pose = [float(v) for v in w.find("model[@name='turtlebot3']/pose").text.split()]
    assert pose[0] == pytest.approx(world.start[0], abs=1e-6)
    assert pose[1] == pytest.approx(world.start[1], abs=1e-6)
    assert pose[5] == pytest.approx(world.start[2], abs=1e-6)
    # Tall enough for any TurtleBot3 lidar mount to see.
    assert m.OBSTACLE_HEIGHT > 0.2


def test_the_render_system_can_be_left_out():
    """Physics-only worlds (no lidar) must still be valid and carry everything
    else; the rendering system is the only difference."""
    m, world = _module(), _world()
    with_sensors = ET.fromstring(m.world_sdf(world)).find("world")
    without = ET.fromstring(m.world_sdf(world, sensors=False)).find("world")
    names = lambda w: [p.get("name") for p in w.findall("plugin")]  # noqa: E731
    assert "gz::sim::systems::Sensors" in names(with_sensors)
    assert "gz::sim::systems::Sensors" not in names(without)
    assert len(without.findall("model")) == len(with_sensors.findall("model"))


def test_the_control_period_is_a_whole_number_of_physics_steps():
    m = _module()
    steps = 0.1 / m.STEP_SIZE
    assert steps == pytest.approx(round(steps)) and np.isfinite(steps)
