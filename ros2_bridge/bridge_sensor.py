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
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vision_nav.envs.sensors import Lidar2D, LidarConfig


class BridgeScanner:
    """The bridge's lidar: the condition's sensor at a chosen beam count."""

    def __init__(self, lidar: LidarConfig, beams: int) -> None:
        # Only the beam count is replaced. Range, field of view and -- the
        # point of this module -- the noise and dropout of the condition are
        # inherited, and then actually applied, because ``scan`` passes a
        # generator.
        self.lidar = Lidar2D(replace(lidar, n_beams=int(beams)))
        self.rng: np.random.Generator = np.random.default_rng(0)

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
        return self.lidar.scan(world, np.asarray(pose, dtype=np.float64), self.rng)
