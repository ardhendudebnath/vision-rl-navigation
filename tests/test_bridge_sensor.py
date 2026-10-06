"""The Nav2 bridge must deliver the evaluation condition's sensor noise.

Every Nav2 run on `noisy_lidar` before this test existed was scored on clean
scans. The bridge's scanner inherited ``noise_std`` from the condition -- which
the README's fairness section checked -- and was then called without a random
generator, and ``Lidar2D`` applies noise only when given one. A check on the
configured value passed for every run; a check on the delivered value would have
failed the first. These tests check what is delivered.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.sensors import Lidar2D
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.training.env_factory import build_env_config

BRIDGE = Path(__file__).resolve().parents[1] / "ros2_bridge"


def _bridge_sensor():
    spec = importlib.util.spec_from_file_location("bridge_sensor", BRIDGE / "bridge_sensor.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(condition: str):
    split, shift, noise = BENCHMARK_CONDITIONS[condition]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {}, split=split,
                           shift=shift, n_worlds=3)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": int(list(cfg.world_seeds)[0])})
    return cfg, env


def test_under_noisy_lidar_the_bridge_delivers_noise():
    cfg, env = _env("noisy_lidar")
    scanner = _bridge_sensor().BridgeScanner(cfg.lidar, 360)
    scanner.start_episode(int(env.world.seed))
    delivered = scanner.scan(env.world, env.robot.pose)
    clean = Lidar2D(scanner.config.__class__(**{**scanner.config.__dict__, "noise_std": 0.0})
                    ).scan(env.world, env.robot.pose)
    assert not np.array_equal(delivered, clean), "the bridge delivered a clean scan"
    # The difference is the configured noise, on beams that hit something
    # well inside the maximum (clipping distorts it near the limit).
    inside = clean < scanner.config.max_range - 0.5
    spread = float(np.std(delivered[inside] - clean[inside]))
    assert spread == pytest.approx(cfg.lidar.noise_std, rel=0.25), spread


def test_a_clean_condition_stays_clean():
    """No noise where none is configured: on `nominal` the delivered scan is
    exactly the sensor's own."""
    cfg, env = _env("nominal")
    scanner = _bridge_sensor().BridgeScanner(cfg.lidar, 360)
    scanner.start_episode(int(env.world.seed))
    delivered = scanner.scan(env.world, env.robot.pose)
    clean = Lidar2D(scanner.config).scan(env.world, env.robot.pose)
    assert np.array_equal(delivered, clean)


def test_the_noise_is_reproducible_per_world_and_fresh_per_scan():
    cfg, env = _env("noisy_lidar")
    module = _bridge_sensor()
    a, b = module.BridgeScanner(cfg.lidar, 360), module.BridgeScanner(cfg.lidar, 360)
    a.start_episode(20000)
    b.start_episode(20000)
    first = a.scan(env.world, env.robot.pose)
    assert np.array_equal(first, b.scan(env.world, env.robot.pose)), "same world, same noise"
    assert not np.array_equal(first, a.scan(env.world, env.robot.pose)), "a new scan, new noise"


def test_only_the_beam_count_is_replaced():
    cfg, _ = _env("noisy_lidar")
    scanner = _bridge_sensor().BridgeScanner(cfg.lidar, 32)
    assert scanner.config.n_beams == 32
    assert scanner.config.noise_std == cfg.lidar.noise_std
    assert scanner.config.max_range == cfg.lidar.max_range


def test_the_audit_measures_the_noise_delivered():
    """What the run reports is measured on the published scans: the configured
    size under noisy_lidar, exactly zero on a clean sensor, nothing before the
    first scan."""
    module = _bridge_sensor()
    for condition, expected in (("noisy_lidar", 0.10), ("nominal", 0.0)):
        cfg, env = _env(condition)
        scanner = module.BridgeScanner(cfg.lidar, 360)
        assert scanner.delivered_noise_std is None
        scanner.start_episode(int(env.world.seed))
        for _ in range(5):
            scanner.scan(env.world, env.robot.pose)
        assert scanner.delivered_noise_std == pytest.approx(expected, abs=0.01), condition
        assert module.delivered_matches(scanner.config, scanner.delivered_noise_std)


def test_a_mismatch_between_configured_and_delivered_noise_is_caught():
    """The check run_nav2_eval makes after the first episode. The old bridge
    delivered 0.0 against a configured 0.10, and must fail it."""
    module = _bridge_sensor()
    noisy, _ = _env("noisy_lidar")
    clean, _ = _env("nominal")
    assert not module.delivered_matches(noisy.lidar, 0.0), "the bug itself"
    assert not module.delivered_matches(noisy.lidar, None)
    assert not module.delivered_matches(noisy.lidar, 0.3)
    assert not module.delivered_matches(clean.lidar, 0.1)
    assert module.delivered_matches(noisy.lidar, 0.1)
    assert module.delivered_matches(clean.lidar, 0.0)


def test_the_runner_checks_and_records_the_delivered_noise():
    """run_nav2_eval.py needs ROS to import, so the check is pinned at the
    source: it must stop a run whose delivered noise disagrees with its
    configuration, and write what was delivered into the result file."""
    source = (BRIDGE / "run_nav2_eval.py").read_text(encoding="utf-8")
    assert "delivered_matches(bridge.scan_config, bridge.delivered_noise_std)" in source
    assert '"delivered_noise_std": bridge_noise' in source


def test_the_bridge_publishes_through_the_scanner_that_applies_noise():
    """nav2_bridge.py needs ROS to import, so its use of BridgeScanner is
    pinned at the source: the scanner it publishes from must be this one, and
    the bare Lidar2D call that dropped the noise must not come back."""
    source = (BRIDGE / "nav2_bridge.py").read_text(encoding="utf-8")
    assert "BridgeScanner(" in source
    assert "Lidar2D(" not in source
    assert "self._scan.start_episode(" in source, "the noise must be seeded per episode"
