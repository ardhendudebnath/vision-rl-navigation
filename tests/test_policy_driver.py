"""The policy adapter must hand the policy exactly what it was trained on."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.training.env_factory import build_env_config

ROOT = Path(__file__).resolve().parents[1]
GAZEBO = ROOT / "gazebo_tb3"
MODEL = ROOT / "runs" / "ppo_privileged" / "best_model.zip"


def _module():
    sys.path.insert(0, str(GAZEBO))
    spec = importlib.util.spec_from_file_location("policy_driver", GAZEBO / "policy_driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Echo:
    """Stands in for the policy: returns the first two observation values."""

    def predict(self, obs, deterministic=True):
        return obs[:2], None


def test_nearest_bearing_resampling():
    m = _module()
    bearings = np.deg2rad(np.arange(360.0))
    ranges = np.arange(360.0)
    out = m.resample(ranges, bearings, np.deg2rad(np.array([0.2, 90.4, -1.0, 179.6])))
    assert list(out) == [0.0, 90.0, 359.0, 180.0]


def test_an_empty_direction_reads_the_training_maximum_unless_told_otherwise():
    m = _module()
    bearings = np.linspace(0.0, 6.28, 360)
    ranges = np.full(360, np.inf)
    ranges[0] = 1.0  # one return straight ahead
    d = m.PolicyDriver(_Echo(), 3.5, bearings)
    d.feed(ranges)
    ahead = int(np.argmin(np.abs(m.POLICY_BEARINGS)))
    assert d.scan32[ahead] == pytest.approx(1.0)
    assert np.sum(d.scan32 == 6.0) == 31
    naive = m.PolicyDriver(_Echo(), 3.5, bearings, no_return=3.5)
    naive.feed(ranges)
    assert np.sum(naive.scan32 == 3.5) == 31


def test_given_the_policys_own_sensor_it_builds_the_envs_observation():
    """Policy bearings, 6 m, sensor at the centre, a scan every step: the
    observation must be the environment's, number for number."""
    m = _module()
    cfg = build_env_config({}, split="val", shift=None, n_worlds=2)
    env = ProceduralNavEnv(cfg)
    obs, _ = env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    d = m.PolicyDriver(_Echo(), 6.0, m.POLICY_BEARINGS, range_min=0.0)
    rng = np.random.default_rng(1)
    for _ in range(40):
        d.feed(env.lidar.scan(env.world, env.robot.pose))
        mine = d.observation(env.robot.pose, env.world.goal, env.robot.velocity)
        assert np.allclose(mine, obs, atol=1e-6)
        obs, _, done, truncated, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        if done or truncated:
            break


@pytest.mark.skipif(not MODEL.exists(), reason="trained policy not present")
def test_the_trained_policy_drives_through_the_adapter_as_it_does_natively():
    """With the policy's own sensor, adapter and environment agree on every
    action of an episode."""
    from stable_baselines3 import PPO

    m = _module()
    model = PPO.load(str(MODEL), device="cpu")
    cfg = build_env_config({}, split="val", shift=None, n_worlds=2)
    env = ProceduralNavEnv(cfg)
    obs, _ = env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    d = m.PolicyDriver(model, 6.0, m.POLICY_BEARINGS, range_min=0.0)
    for _ in range(60):
        native, _ = model.predict(obs, deterministic=True)
        d.feed(env.lidar.scan(env.world, env.robot.pose))
        assert np.allclose(d.act(env.robot.pose, env.world.goal, env.robot.velocity), native, atol=1e-5)
        obs, _, done, truncated, _ = env.step(native)
        if done or truncated:
            break
