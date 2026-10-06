"""The scan the Nav2 bridge publishes, with the evaluation condition's noise.

Kept free of ROS imports so the one property that matters can be tested on any
machine: under a noisy condition, Nav2 must receive a noisy scan.

It did not, for every Nav2 run in this project before this module existed.
``nav2_bridge.py`` built its scanner from the condition's ``LidarConfig`` -- so
``noise_std`` was inherited, which is what the README's fairness section checked
-- and then called ``Lidar2D.scan(world, pose)`` with no random generator. The
sensor applies noise and dropout only when it is handed one, so every scan
published under `noisy_lidar` was clean: Nav2, with and without SLAM, was scored
on a sensor the hand-written stack never had. The configured value was right and
the delivered value was not, and only the first was ever looked at.

The generator is seeded from the world exactly as the hand-written stack seeds
its own map's sensor (``np.random.default_rng(int(world.seed))``), so the two
stacks see noise of the same distribution, drawn from the same seed, per world.

Every scan is also audited: the same scan is cast once more without noise, and
the difference on beams that hit something well inside the maximum range is
accumulated, so a run can report the noise it *delivered* and refuse to continue
when that disagrees with the noise it was configured with.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from vision_nav.envs.sensors import Lidar2D, LidarConfig

#: Beams whose noise-free range is within this of the maximum are left out of
#: the audit, because the sensor clips after adding noise and the clipped
#: difference understates it.
AUDIT_MARGIN = 0.5


class BridgeScanner:
    """The bridge's lidar: the condition's sensor at a chosen beam count."""

    def __init__(self, lidar: LidarConfig, beams: int) -> None:
        # Only the beam count is replaced. Range, field of view and -- the
        # point of this module -- the noise and dropout of the condition are
        # inherited, and then actually applied, because ``scan`` passes a
        # generator.
        self.lidar = Lidar2D(replace(lidar, n_beams=int(beams)))
        self.rng: np.random.Generator = np.random.default_rng(0)
        self._clean = Lidar2D(replace(self.lidar.config, noise_std=0.0, dropout_prob=0.0))
        self._sq, self._n = 0.0, 0

    @property
    def config(self) -> LidarConfig:
        return self.lidar.config

    def start_episode(self, world_seed: int) -> None:
        """Seed this episode's noise from the world, as the hand-written stack
        seeds its own sensor, so an episode is reproducible and the two stacks
        draw from the same seed."""
        self.rng = np.random.default_rng(int(world_seed))

    def scan(self, world, pose: np.ndarray) -> np.ndarray:
        """The ranges to publish, with the condition's noise and dropout."""
        pose = np.asarray(pose, dtype=np.float64)
        ranges = self.lidar.scan(world, pose, self.rng)
        clean = self._clean.scan(world, pose)
        inside = clean < self.config.max_range - AUDIT_MARGIN
        d = ranges[inside] - clean[inside]
        self._sq += float(np.dot(d, d))
        self._n += int(inside.sum())
        return ranges

    @property
    def delivered_noise_std(self) -> float | None:
        """Root-mean-square difference between what was published and the
        noise-free scan, over every audited beam so far; None before any."""
        return math.sqrt(self._sq / self._n) if self._n else None


def delivered_matches(config: LidarConfig, delivered: float | None) -> bool:
    """Whether the noise a run delivered is the noise it was configured with:
    none on a clean sensor, and within half to one and a half times the
    configured size on a noisy one. Dropout reads as a large difference, so
    where it is configured only a floor is checked."""
    if delivered is None:
        return False
    if config.dropout_prob > 0.0:
        return delivered > 0.0 and delivered >= 0.5 * config.noise_std
    if config.noise_std == 0.0:
        return delivered == 0.0
    return 0.5 * config.noise_std <= delivered <= 1.5 * config.noise_std
