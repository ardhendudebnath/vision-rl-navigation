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

__all__ = ["LidarConfig", "Lidar2D", "CameraConfig", "DepthCamera"]

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

    def scan_with_hits(self, world, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ranges plus which kind of surface each ray struck.

        Surface kind: ``0`` wall, ``1`` circle, ``2`` box. Needed to shade an
        egocentric render; exposed here rather than reimplemented elsewhere so
        there is exactly one copy of the intersection maths. A second
        implementation could drift from this one, and a renderer that disagreed
        with the range sensor about the world would quietly invalidate any
        comparison between them.
        """
        cfg = self.config
        origin = np.asarray(pose[:2], dtype=np.float64)
        bearings = self._angles + float(pose[2])
        dirs = np.stack([np.cos(bearings), np.sin(bearings)], axis=-1)

        candidates = [self._walls(world, origin, dirs)]
        kinds = [0]
        if len(world.circles):
            candidates.append(self._circles(world.circles, origin, dirs))
            kinds.append(1)
        if len(world.boxes):
            candidates.append(self._boxes(world.boxes, origin, dirs))
            kinds.append(2)

        stacked = np.stack(candidates, axis=0)  # (n_kinds, n_beams)
        which = np.argmin(stacked, axis=0)
        ranges = np.clip(stacked[which, np.arange(stacked.shape[1])], 0.0, cfg.max_range)
        return ranges, np.asarray(kinds, dtype=np.int8)[which]

    # ------------------------------------------------------------------
    # Primitive intersections
    # ------------------------------------------------------------------
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


@dataclass
class CameraConfig:
    """Forward-facing depth camera.

    The step from lidar to camera is not "fewer beams" — it is **losing the
    360-degree view**. A planar lidar tells the policy what is behind it; a
    camera does not. That is the constraint every real vision-based robot
    actually has, and it is the substantive difference this observation mode
    introduces.

    ``width`` is the number of image columns. Together with ``fov`` it fixes
    the angular resolution, which Phase 2i established is a real constraint on
    this task for lidar — so it is exposed here as a first-class knob rather
    than buried, to let the same question be asked of the camera.
    """

    #: Horizontal field of view in radians (90 degrees by default, typical of
    #: a RealSense-class depth camera).
    fov: float = np.pi / 2
    #: Image columns.
    width: int = 64
    max_range: float = 6.0
    noise_std: float = 0.0
    dropout_prob: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fov < 2.0 * np.pi:
            raise ValueError(f"camera fov must be in (0, 2pi), got {self.fov}")
        if self.width < 1:
            raise ValueError(f"camera width must be >= 1, got {self.width}")

    @property
    def angular_resolution(self) -> float:
        """Radians between adjacent columns."""
        return self.fov / self.width

    def resolves_gap_to(self, gap_width: float) -> float:
        """Range at which a ``gap_width`` opening spans one column.

        Beyond this range the camera can step over an opening the robot would
        fit through — the mechanism measured in Phase 2h.
        """
        return gap_width / (2.0 * np.sin(self.angular_resolution / 2.0))


class DepthCamera:
    """Egocentric depth strip, rendered by the same analytic ray-caster.

    Deliberately built on :class:`Lidar2D` rather than duplicating the
    intersection code: the two sensors must agree exactly about the world, or
    a comparison between them measures the ray-caster as much as the sensor.
    """

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        cfg = self.config
        # A camera samples its FOV at column centres, so both edges are inset
        # by half a column. Reusing Lidar2D's endpoint-inclusive spacing would
        # misplace every column by up to half a pixel.
        half = cfg.fov / 2.0
        offsets = -half + (np.arange(cfg.width) + 0.5) * cfg.angular_resolution
        self._lidar = Lidar2D(
            LidarConfig(
                n_beams=cfg.width,
                fov=cfg.fov,
                max_range=cfg.max_range,
                noise_std=cfg.noise_std,
                dropout_prob=cfg.dropout_prob,
            )
        )
        self._lidar._angles = offsets

    @property
    def width(self) -> int:
        return self.config.width

    def depth(self, world, pose: np.ndarray, rng=None) -> np.ndarray:
        """Depth per column, in metres, clipped to ``max_range``."""
        return self._lidar.scan(world, pose, rng)

    def normalized_depth(self, world, pose: np.ndarray, rng=None) -> np.ndarray:
        """:meth:`depth` mapped to ``[0, 1]`` for direct use as an observation."""
        return self.depth(world, pose, rng) / self.config.max_range
