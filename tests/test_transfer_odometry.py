"""The policy without its privilege: the goal vector from odometry, not truth."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
GAZEBO = ROOT / "gazebo_tb3"
MODEL = ROOT / "runs" / "ppo_privileged" / "best_model.zip"


def _module():
    sys.path.insert(0, str(GAZEBO))
    spec = importlib.util.spec_from_file_location("transfer", GAZEBO / "transfer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relative_then_compose_is_the_identity():
    m = _module()
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = rng.uniform([-5, -5, -np.pi], [5, 5, np.pi])
        b = rng.uniform([-5, -5, -np.pi], [5, 5, np.pi])
        assert np.allclose(m.compose(a, m.relative(a, b)), b)


def test_odometry_is_carried_into_the_world_frame_from_the_start():
    """A robot that starts facing +y and whose odometry says it drove 1 m
    straight ahead is 1 m further along +y, whatever odometry's own origin."""
    m = _module()
    start = np.array([2.0, 3.0, np.pi / 2])
    odom_start, odom_now = np.array([0.5, 0.0, 0.0]), np.array([1.5, 0.0, 0.0])
    assert np.allclose(m.compose(start, m.relative(odom_start, odom_now)), [2.0, 4.0, np.pi / 2])


def test_heading_is_read_from_the_quaternion():
    m = _module()
    h = 2.5
    msg = SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=-2.0, z=0.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=np.sin(h / 2), w=np.cos(h / 2))))
    assert np.allclose(m.odom_pose(msg), [1.0, -2.0, h])


def test_the_odometry_scale_runs_from_exact_to_the_published_model():
    from vision_nav.mapping.localisation import OdometryConfig

    m = _module()
    assert m.scaled_odometry(1.0) == OdometryConfig()
    assert m.scaled_odometry(0.5).bias_trans == OdometryConfig().bias_trans / 2
    assert all(v == 0.0 for v in vars(m.scaled_odometry(0.0)).values())


def test_at_scale_zero_the_stack_drives_on_exact_odometry():
    """The sweep's zero is the follow-up arm already published (§9.23)."""
    m = _module()
    exact = m.analytic_episode("tb3_2d_exact_odom", "nominal", 10000, "val", 100)
    zero = m.analytic_episode("tb3_2d", "nominal", 10000, "val", 100, odometry_scale=0.0)
    assert zero == exact


@pytest.mark.skipif(not MODEL.exists(), reason="trained policy not present")
def test_at_scale_zero_the_unprivileged_policy_is_the_privileged_one():
    m = _module()
    privileged = m.policy_episode_2d("policy_tb3_2d", "nominal", 10000, "val", 100)
    zero = m.policy_episode_2d("policy_tb3_2d_odom", "nominal", 10000, "val", 100, odometry_scale=0.0)
    assert zero.pop("odom_err_final") < 1e-9
    zero.pop("odom_goal_distance_final")
    assert zero == privileged


@pytest.mark.skipif(not MODEL.exists(), reason="trained policy not present")
@pytest.mark.parametrize("arm", ["policy_project", "policy_tb3_2d"])
def test_with_exact_odometry_the_unprivileged_arm_is_the_privileged_one(arm):
    """Dead reckoning on the true velocity is the true pose, so the episode
    must be the privileged arm's own; with the odometry model it drifts."""
    m = _module()
    seed = 10000  # a val world
    privileged = m.policy_episode_2d(arm, "nominal", seed, "val", 100)
    exact = m.policy_episode_2d(arm + "_odom", "nominal", seed, "val", 100, exact_odometry=True)
    assert exact.pop("odom_err_final") < 1e-9
    assert exact.pop("odom_goal_distance_final") == pytest.approx(exact["goal_distance"], abs=1e-9)
    assert exact == privileged
    drifting = m.policy_episode_2d(arm + "_odom", "nominal", seed, "val", 100)
    assert drifting["odom_err_final"] > 0.0
