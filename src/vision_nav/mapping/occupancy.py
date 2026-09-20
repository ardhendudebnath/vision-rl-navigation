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
- **Evidence accumulates.** Each cell keeps a log-odds score: a return landing
  in it adds evidence of occupancy, a beam passing through it subtracts some,
  and it counts as occupied only while the evidence says so. Phase 6d's first
  version made occupied cells sticky instead -- correct for a noise-free sensor,
  and on `noisy_lidar` every return landing short marked free space occupied
  for good.
- **Surfaces are placed conservatively.** A return says a surface lies
  somewhere in its cell, so clearance is measured to the cell's nearest possible
  point, half a cell diagonal short of its centre. The first version measured to
  half a cell short, and overestimated clearance by up to 0.07 m -- enough for
  plans to graze what the full-map planner keeps clear of.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OccupancyMap"]

UNKNOWN, FREE, OCCUPIED = -1, 0, 1

#: Log-odds evidence per return landing in a cell, and per beam passing through.
#: A hit outweighs a pass so one clean return marks a surface, and a surface
#: that keeps being hit cannot be erased by beams grazing its cell's corner.
HIT, MISS = 0.9, -0.3
#: A cell that has been seen is occupied at or above this evidence and free
#: below it; one never seen stays unknown. One hit (0.9) survives one beam
#: passing through (0.6) and not two (0.3). Clamped so neither state becomes
#: unrecoverable. Set at 0.5, between those, after 0.6 put 0.9 - 0.3 just under
#: the line in float32 and a single pass erased a surface.
OCCUPIED_AT, CLAMP = 0.5, (-2.0, 3.5)


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
        self.evidence = np.zeros(self.shape, dtype=np.float32)
        #: How far inside a cell a surface can lie from its centre.
        self.half_diagonal = self.resolution * np.sqrt(2.0) / 2.0
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
    def scan(self, pose: np.ndarray) -> np.ndarray:
        """The ranges the sensor returns from ``pose``.

        Separate from :meth:`integrate` for the one caller that needs the scan
        before it knows where to put it: a robot estimating its own pose matches
        the scan against the map first, and integrates it at the pose the match
        settles on (:mod:`vision_nav.mapping.localisation`).
        """
        return np.asarray(self.sensor.scan(self._world, np.asarray(pose, dtype=np.float64),
                                           self.rng), dtype=np.float64)

    def integrate(self, pose: np.ndarray, ranges: np.ndarray | None = None) -> bool:
        """Add one scan at ``pose``. Returns whether any cell became occupied --
        the only change that can invalidate a plan.

        ``ranges`` supplies a scan already taken; by default one is taken at
        ``pose``, which is the same thing whenever the robot knows where it is.
        """
        pose = np.asarray(pose, dtype=np.float64)
        self.scans += 1
        ranges = (self.scan(pose) if ranges is None
                  else np.asarray(ranges, dtype=np.float64))
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
        # One vote per cell per scan, however many samples of a beam fall in it.
        passed = np.zeros(self.shape, dtype=bool)
        rows, cols = self._cells(pts[keep])
        passed[rows, cols] = True

        # Returns short of max range hit something.
        landed = np.zeros(self.shape, dtype=bool)
        hit = ranges < max_range - 1e-6
        if hit.any():
            ends = pose[None, :2] + dirs[hit] * ranges[hit, None]
            rows, cols = self._cells(ends)
            landed[rows, cols] = True
        passed &= ~landed  # a cell a return landed in is evidence for, not against

        before = self.grid == OCCUPIED
        self.evidence[passed] += MISS
        self.evidence[landed] += HIT
        np.clip(self.evidence, *CLAMP, out=self.evidence)
        seen = passed | landed
        self.grid[seen & (self.evidence >= OCCUPIED_AT)] = OCCUPIED
        self.grid[seen & (self.evidence < OCCUPIED_AT)] = FREE
        after = self.grid == OCCUPIED
        newly = after & ~before
        self.first_occupied[newly & (self.first_occupied < 0)] = self.scans
        if np.array_equal(before, after):
            return False
        self.version += 1
        self._inflated.clear()
        occ = np.argwhere(after)
        res = self.resolution
        self._occupied_pts = (np.stack([(occ[:, 1] + 0.5) * res, (occ[:, 0] + 0.5) * res],
                                       axis=-1) if len(occ) else np.zeros((0, 2)))
        return bool(newly.any())

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        res = self.resolution
        rows = np.clip((points[:, 1] / res).astype(int), 0, self.shape[0] - 1)
        cols = np.clip((points[:, 0] / res).astype(int), 0, self.shape[1] - 1)
        return rows, cols

    # --- the world's interfaces ----------------------------------------------
    def clearance(self, points: np.ndarray, **_ignored) -> np.ndarray:
        """Distance from each point to the nearest mapped obstacle surface.

        An occupied cell stands for a surface somewhere inside it, so the
        distance is taken to the nearest point it could be: the centre less half
        a cell diagonal. Never more than the true clearance to what has been
        mapped. Where nothing has been mapped the answer is large -- unknown is
        free.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if not len(self._occupied_pts):
            out = np.full(len(pts), 1e3)
        else:
            d = np.linalg.norm(pts[:, None, :] - self._occupied_pts[None, :, :], axis=-1)
            out = d.min(axis=1) - self.half_diagonal
        return out if np.ndim(points) > 1 else out[0]

    def occupancy_at(self, radius: float, **_ignored) -> np.ndarray:
        """Cells within ``radius`` of a mapped obstacle, unknown counted free."""
        key = round(float(radius), 6)
        if key not in self._inflated:
            occupied = self.grid == OCCUPIED
            reach = radius + self.half_diagonal
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
