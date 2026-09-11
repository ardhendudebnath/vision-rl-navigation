"""Vectorised 2D range sensing.

A single :class:`Lidar2D` scan is computed analytically against every
obstacle at once with NumPy, which keeps a full episode in the microsecond
range per step.  Ray-casting against exact primitives (rather than a
rasterised grid) means the sensor does not inherit the planner's
discretisation error, so the classical baseline and the learned policy are
not accidentally handed different world models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LidarConfig", "Lidar2D"]

_EPS = 1e-12


@dataclass
class LidarConfig:
    """Planar range-finder parameters (RPLIDAR-class defaults)."""

    n_beams: int = 32
    fov: float = 2.0 * np.pi  # radians, full 360° scan
    max_range: float = 6.0  # metres
    #: Std-dev of additive Gaussian range noise, in metres.  Left at zero for
    #: training; raised during the robustness suite as a domain shift.
    noise_std: float = 0.0
    #: Probability a beam returns max_range (a dropout / specular miss).
    dropout_prob: float = 0.0

    def beam_angles(self) -> np.ndarray:
        """Beam bearings relative to the robot's heading, in radians."""
        if self.fov >= 2.0 * np.pi - 1e-9:
            # Full circle: avoid duplicating the first and last beam.
            return np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False)
        return np.linspace(-self.fov / 2.0, self.fov / 2.0, self.n_beams)


class Lidar2D:
    """Analytic planar ray-caster against circles, boxes and arena walls."""

    def __init__(self, config: LidarConfig | None = None) -> None:
        self.config = config or LidarConfig()
        self._angles = self.config.beam_angles()

    @property
    def n_beams(self) -> int:
        return self.config.n_beams

    def scan(
        self,
        world,
        pose: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Cast all beams from ``pose`` and return ranges in metres.

        Parameters
        ----------
        world:
            A :class:`~vision_nav.envs.world.World`.
        pose:
            ``(3,)`` array of ``(x, y, theta)``.
        rng:
            Generator used for noise and dropout.  Required only when the
            sensor config enables either.

        Returns
        -------
        np.ndarray
            ``(n_beams,)`` ranges, clipped to ``[0, max_range]``.
        """
        cfg = self.config
        origin = np.asarray(pose[:2], dtype=np.float64)
        bearings = self._angles + float(pose[2])
        dirs = np.stack([np.cos(bearings), np.sin(bearings)], axis=-1)  # (N, 2)

        t = np.full(cfg.n_beams, np.inf)
        t = np.minimum(t, self._walls(world, origin, dirs))
        if len(world.circles):
            t = np.minimum(t, self._circles(world.circles, origin, dirs))
        if len(world.boxes):
            t = np.minimum(t, self._boxes(world.boxes, origin, dirs))

        ranges = np.clip(t, 0.0, cfg.max_range)

        if rng is not None:
            if cfg.noise_std > 0.0:
                ranges = ranges + rng.normal(0.0, cfg.noise_std, size=ranges.shape)
            if cfg.dropout_prob > 0.0:
                dropped = rng.random(ranges.shape) < cfg.dropout_prob
                ranges = np.where(dropped, cfg.max_range, ranges)

        return np.clip(ranges, 0.0, cfg.max_range)

    def normalized_scan(self, world, pose: np.ndarray, rng=None) -> np.ndarray:
        """:meth:`scan` mapped to ``[0, 1]`` for direct use as an observation."""
        return self.scan(world, pose, rng) / self.config.max_range

    # ------------------------------------------------------------------
    # Primitive intersections
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_inv(d: np.ndarray) -> np.ndarray:
        """Reciprocal that keeps the sign of near-zero components."""
        safe = np.where(np.abs(d) < _EPS, np.copysign(_EPS, np.where(d == 0.0, 1.0, d)), d)
        return 1.0 / safe

    def _walls(self, world, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        """Distance to the arena boundary (the ray always exits eventually)."""
        lo = np.zeros(2)
        hi = np.array([world.config.width, world.config.height])
        inv = self._safe_inv(dirs)  # (N, 2)
        t_lo = (lo - origin) * inv
        t_hi = (hi - origin) * inv
        t_exit = np.minimum(np.maximum(t_lo, t_hi), np.inf).min(axis=-1)
        return np.where(t_exit > 0.0, t_exit, np.inf)

    @staticmethod
    def _circles(circles: np.ndarray, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        oc = origin[None, :] - circles[:, :2]  # (M, 2)
        b = dirs @ oc.T  # (N, M)
        c = (oc**2).sum(axis=-1) - circles[:, 2] ** 2  # (M,)
        disc = b**2 - c[None, :]
        hit = disc >= 0.0
        sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
        t_near = -b - sqrt_disc
        t_far = -b + sqrt_disc
        # Prefer the near root; fall back to the far root if the origin is
        # inside the circle (which can happen at the instant of a collision).
        t = np.where(t_near > 0.0, t_near, t_far)
        t = np.where(hit & (t > 0.0), t, np.inf)
        return t.min(axis=1)

    def _boxes(self, boxes: np.ndarray, origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
        inv = self._safe_inv(dirs)[:, None, :]  # (N, 1, 2)
        lo = boxes[None, :, :2]  # (1, K, 2)
        hi = boxes[None, :, 2:]
        t_lo = (lo - origin[None, None, :]) * inv
        t_hi = (hi - origin[None, None, :]) * inv
        t_enter = np.minimum(t_lo, t_hi).max(axis=-1)  # (N, K)
        t_exit = np.maximum(t_lo, t_hi).min(axis=-1)
        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_exit > 0.0)
        t = np.where(t_enter > 0.0, t_enter, t_exit)
        t = np.where(hit & (t > 0.0), t, np.inf)
        return t.min(axis=1)
