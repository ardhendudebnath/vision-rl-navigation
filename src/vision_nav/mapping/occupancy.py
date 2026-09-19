"""An occupancy grid built from range returns, with the world's own interfaces.

Every classical result in this project so far gave the planner a perfect static
map: ``world.occupancy_at`` for planning, ``world.clearance`` for path
shortening and for the controller's slow-down near obstacles. This builds that
map instead, from the ranges a sensor returns as the robot drives, and exposes
the same two methods so the planner, the smoother and the controller can read it
without knowing the difference.

What it is allowed to read from the world is deliberately narrow: the range each
beam returns (``Lidar2D.scan``), and grid metadata -- resolution, extent, the
cell transforms -- that any map has. It never reads obstacle geometry. The unit
tests hold it to that.

Conventions a real mapper would share:

- Along each beam, cells short of the return are **free**; the cell the return
  lands in is **occupied**. A beam that reaches ``max_range`` saw nothing.
- **Unknown is free for planning.** The planner is optimistic and replans when
  what it has not yet seen turns out to be in the way -- the standard way to
  plan into unexplored space.
- **Occupied is sticky.** The world is static, so a cell once seen occupied is
  never cleared by a later beam grazing past it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OccupancyMap"]

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


class OccupancyMap:
    """A grid the robot fills in from its own range sensor."""

    def __init__(self, world, sensor, rng: np.random.Generator | None = None) -> None:
        #: Used only for grid metadata and to be handed to the sensor, which
        #: returns ranges. See the module docstring.
        self._world = world
        self.sensor = sensor
        self.rng = rng
        self.resolution = float(world.config.grid_resolution)
        # Extent from the arena's dimensions, not from ``world.occupancy``:
        # that property computes the true obstacle grid, and reading even its
        # shape would be touching geometry the map is supposed to learn.
        cfg = world.config
        self.shape = (int(np.ceil(cfg.height / self.resolution)),
                      int(np.ceil(cfg.width / self.resolution)))
        self.robot_radius = float(world.config.robot_radius)
        self.grid = np.full(self.shape, UNKNOWN, dtype=np.int8)
        self._occupied_pts = np.zeros((0, 2))
        self._inflated: dict[float, np.ndarray] = {}
        #: Bumped whenever a cell becomes occupied: everything derived from the
        #: occupied set (inflation, clearance) is cached against it.
        self.version = 0
        #: Scan index at which each cell was first seen occupied (-1: never).
        #: Diagnostic only -- says whether an obstacle was known in time.
        self.first_occupied = np.full(self.shape, -1, dtype=np.int32)
        self.scans = 0

    # --- building ---------------------------------------------------------
    def integrate(self, pose: np.ndarray) -> bool:
        """Add one scan taken at ``pose``. Returns whether any cell became
        occupied -- the only change that can invalidate a plan."""
        pose = np.asarray(pose, dtype=np.float64)
        self.scans += 1
        ranges = np.asarray(self.sensor.scan(self._world, pose, self.rng), dtype=np.float64)
        angles = np.asarray(self.sensor._angles, dtype=np.float64) + pose[2]
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        max_range = float(self.sensor.config.max_range)

        # Free space: samples every half cell, stopping half a cell short of
        # the return so the obstacle's own cell is never marked free.
        step = self.resolution * 0.5
        n = int(np.ceil(max_range / step))
        ts = (np.arange(n) + 0.5) * step
        reach = np.clip(ranges - step, 0.0, None)
        pts = pose[None, None, :2] + dirs[:, None, :] * ts[None, :, None]
        keep = ts[None, :] < reach[:, None]
        rows, cols = self._cells(pts[keep])
        free = self.grid[rows, cols] != OCCUPIED
        self.grid[rows[free], cols[free]] = FREE

        # Returns short of max range hit something.
        hit = ranges < max_range - 1e-6
        if not hit.any():
            return False
        ends = pose[None, :2] + dirs[hit] * ranges[hit, None]
        rows, cols = self._cells(ends)
        new = self.grid[rows, cols] != OCCUPIED
        if not new.any():
            return False
        self.grid[rows[new], cols[new]] = OCCUPIED
        self.first_occupied[rows[new], cols[new]] = self.scans
        self.version += 1
        self._inflated.clear()
        occ = np.argwhere(self.grid == OCCUPIED)
        res = self.resolution
        self._occupied_pts = np.stack([(occ[:, 1] + 0.5) * res, (occ[:, 0] + 0.5) * res], axis=-1)
        return True

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = self.resolution
        rows = np.clip((points[:, 1] / res).astype(int), 0, self.shape[0] - 1)
        cols = np.clip((points[:, 0] / res).astype(int), 0, self.shape[1] - 1)
        return rows, cols

    # --- the world's interfaces ----------------------------------------------
    def clearance(self, points: np.ndarray, **_ignored) -> np.ndarray:
        """Distance from each point to the nearest mapped obstacle surface.

        An occupied cell stands for a surface somewhere inside it, so the
        distance is taken to its centre less half a cell. Where nothing has been
        mapped the answer is large -- unknown is free.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if not len(self._occupied_pts):
            out = np.full(len(pts), 1e3)
        else:
            d = np.linalg.norm(pts[:, None, :] - self._occupied_pts[None, :, :], axis=-1)
            out = d.min(axis=1) - 0.5 * self.resolution
        return out if np.ndim(points) > 1 else out[0]

    def occupancy_at(self, radius: float, **_ignored) -> np.ndarray:
        """Cells within ``radius`` of a mapped obstacle, unknown counted free."""
        key = round(float(radius), 6)
        if key not in self._inflated:
            occupied = self.grid == OCCUPIED
            reach = radius + 0.5 * self.resolution
            k = int(np.ceil(reach / self.resolution))
            out = np.zeros(self.shape, dtype=bool)
            if occupied.any():
                padded = np.pad(occupied, k)
                for dr in range(-k, k + 1):
                    for dc in range(-k, k + 1):
                        if (dr * dr + dc * dc) * self.resolution ** 2 > reach ** 2:
                            continue
                        out |= padded[k + dr:k + dr + self.shape[0],
                                      k + dc:k + dc + self.shape[1]]
            self._inflated[key] = out
        return self._inflated[key]

    # Grid transforms: the same formulas as the world's, computed here so that
    # nothing on this object ever asks the world for its obstacle grid.
    def world_to_grid(self, points):
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        rows, cols = self._cells(pts)
        out = np.stack([rows, cols], axis=-1)
        return out[0] if np.ndim(points) == 1 else out

    def grid_to_world(self, cells):
        single = np.ndim(cells) == 1
        arr = np.atleast_2d(np.asarray(cells))
        res = self.resolution
        xy = np.stack([(arr[:, 1] + 0.5) * res, (arr[:, 0] + 0.5) * res], axis=-1)
        return xy[0] if single else xy

    @property
    def goal(self):
        return self._world.goal

    @property
    def config(self):
        return self._world.config

    @property
    def known_fraction(self) -> float:
        return float((self.grid != UNKNOWN).mean())
