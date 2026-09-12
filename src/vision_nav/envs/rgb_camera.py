"""Egocentric RGB rendering from the 2D world.

The point of this observation mode is *not* to add information. It renders the
same geometry the depth camera already measures, at the same field of view and
the same column count, so that any performance difference isolates **what the
learned encoder costs** — the same content, presented as pixels that a CNN
must interpret rather than as a vector an MLP reads directly.

That is the question worth asking in simulation, because it is the one that
transfers: real robots do not get a clean depth vector, they get images, and
the encoder is the part that has to be paid for.

Rendering is column-wise ray casting (the classic Wolfenstein construction):
one ray per image column, wall slice height inversely proportional to
perpendicular distance. Perpendicular rather than radial distance, or straight
surfaces bow outward at the edges of the frame — a fisheye artefact that would
be a rendering bug masquerading as a perception difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.sensors import Lidar2D, LidarConfig

__all__ = ["RGBCameraConfig", "RGBCamera"]


@dataclass
class RGBCameraConfig:
    """Egocentric colour camera."""

    fov: float = np.pi / 2
    #: Image columns; also the number of rays cast.
    width: int = 64
    height: int = 48
    max_range: float = 6.0

    #: Vertical extent of a surface at 1 m, as a fraction of image height.
    #: Sets how quickly things shrink with distance.
    wall_scale: float = 1.2

    #: Surface shading, indexed by hit kind (wall, circle, box). Distinct
    #: values so the encoder can tell surface types apart, which the depth
    #: vector cannot express.
    wall_rgb: tuple[int, int, int] = (150, 150, 160)
    circle_rgb: tuple[int, int, int] = (74, 84, 96)
    box_rgb: tuple[int, int, int] = (120, 92, 74)

    ceiling_rgb: tuple[int, int, int] = (247, 247, 245)
    floor_rgb: tuple[int, int, int] = (208, 208, 204)

    #: Fraction of brightness retained at max_range. Distance shading is the
    #: only monocular depth cue besides slice height, so it is not decorative.
    distance_falloff: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 < self.fov < 2.0 * np.pi:
            raise ValueError(f"camera fov must be in (0, 2pi), got {self.fov}")
        if self.width < 1 or self.height < 1:
            raise ValueError(f"image must be at least 1x1, got {self.width}x{self.height}")

    @property
    def angular_resolution(self) -> float:
        return self.fov / self.width


class RGBCamera:
    """Renders an egocentric RGB image using the shared ray-caster."""

    def __init__(self, config: RGBCameraConfig | None = None) -> None:
        self.config = config or RGBCameraConfig()
        cfg = self.config
        self._lidar = Lidar2D(
            LidarConfig(n_beams=cfg.width, fov=cfg.fov, max_range=cfg.max_range)
        )
        step = cfg.angular_resolution
        # Column centres, matching DepthCamera exactly so the two modes see
        # the same bearings and differ only in representation.
        self._offsets = -cfg.fov / 2.0 + (np.arange(cfg.width) + 0.5) * step
        self._lidar._angles = self._offsets
        self._surface = np.array(
            [cfg.wall_rgb, cfg.circle_rgb, cfg.box_rgb], dtype=np.float64
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(height, width, 3)``."""
        return (self.config.height, self.config.width, 3)

    def render(self, world, pose: np.ndarray) -> np.ndarray:
        """Render one frame as ``(H, W, 3)`` uint8."""
        cfg = self.config
        h, w = cfg.height, cfg.width

        ranges, kinds = self._lidar.scan_with_hits(world, pose)
        # Perpendicular distance removes the fisheye bow at frame edges.
        perp = np.maximum(ranges * np.cos(self._offsets), 1e-3)

        slice_h = np.clip(cfg.wall_scale * h / perp, 1.0, h)
        top = np.clip((h - slice_h) * 0.5, 0, h).astype(np.int32)
        bottom = np.clip(h - top, 0, h).astype(np.int32)

        shade = 1.0 - (1.0 - cfg.distance_falloff) * np.clip(
            perp / cfg.max_range, 0.0, 1.0
        )
        colours = self._surface[kinds] * shade[:, None]

        img = np.empty((h, w, 3), dtype=np.float64)
        img[:] = np.asarray(cfg.ceiling_rgb, dtype=np.float64)
        rows = np.arange(h)[:, None]
        floor_mask = rows >= bottom[None, :]
        img[floor_mask] = np.asarray(cfg.floor_rgb, dtype=np.float64)
        surface_mask = (rows >= top[None, :]) & (rows < bottom[None, :])
        img[surface_mask] = np.broadcast_to(colours[None, :, :], (h, w, 3))[surface_mask]

        return img.astype(np.uint8)

    def render_chw(self, world, pose: np.ndarray) -> np.ndarray:
        """:meth:`render` as ``(3, H, W)``, the layout SB3's CNN expects."""
        return np.ascontiguousarray(self.render(world, pose).transpose(2, 0, 1))
