"""Procedural 2D navigation worlds.

A :class:`World` is a rectangular arena populated with circular and
axis-aligned box obstacles.  Worlds are generated deterministically from a
seed so that train / validation / test splits are reproducible and
held-out environments can be named by integer id (see
:func:`generate_world`).

The geometry here is intentionally analytic (no mesh, no physics engine) so
that the whole stack runs at tens of thousands of steps per second on CPU.
That is what makes it usable as the *debug substrate* for the heavier
Isaac Lab / Habitat runs later in the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "WorldConfig",
    "World",
    "generate_world",
]


@dataclass
class WorldConfig:
    """Parameters controlling procedural world generation."""

    width: float = 12.0
    height: float = 12.0

    n_circles: tuple[int, int] = (6, 12)
    circle_radius: tuple[float, float] = (0.3, 0.9)

    n_boxes: tuple[int, int] = (2, 6)
    box_size: tuple[float, float] = (0.5, 1.8)

    robot_radius: float = 0.22
    goal_tolerance: float = 0.35

    #: Minimum straight-line distance between start and goal, in metres.
    min_start_goal_dist: float = 5.0
    #: Clearance required around the sampled start and goal poses, in metres.
    spawn_clearance: float = 0.35
    #: Occupancy-grid resolution used for planning and reachability, in metres.
    grid_resolution: float = 0.1

    #: Attempts allowed when sampling a valid start/goal pair before the
    #: world is rejected and regenerated with a fresh obstacle layout.
    max_spawn_attempts: int = 200

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("world width and height must be positive")
        if self.grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")
        if self.min_start_goal_dist >= np.hypot(self.width, self.height):
            raise ValueError(
                "min_start_goal_dist exceeds the arena diagonal; no start/goal "
                "pair could ever satisfy it"
            )


@dataclass
class World:
    """A concrete, immutable navigation scene.

    Attributes
    ----------
    circles:
        ``(M, 3)`` array of ``(x, y, radius)``.
    boxes:
        ``(K, 4)`` array of ``(x_lo, y_lo, x_hi, y_hi)``.
    start:
        ``(3,)`` array of ``(x, y, theta)``.
    goal:
        ``(2,)`` array of ``(x, y)``.
    """

    config: WorldConfig
    circles: np.ndarray
    boxes: np.ndarray
    start: np.ndarray
    goal: np.ndarray
    seed: int = 0

    _occupancy: np.ndarray | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------
    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance from ``points`` to the nearest obstacle or wall.

        Parameters
        ----------
        points:
            ``(..., 2)`` array of query positions.

        Returns
        -------
        np.ndarray
            ``(...,)`` array of distances.  Negative inside an obstacle or
            outside the arena bounds.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 2:
            raise ValueError(f"expected points with trailing dim 2, got {pts.shape}")
        leading_shape = pts.shape[:-1]
        pts = pts.reshape(-1, 2)

        # Distance to the arena walls (positive inside the arena).
        d_wall = np.minimum.reduce(
            [
                pts[:, 0],
                pts[:, 1],
                self.config.width - pts[:, 0],
                self.config.height - pts[:, 1],
            ]
        )
        dist = d_wall

        if len(self.circles):
            delta = pts[:, None, :] - self.circles[None, :, :2]
            d_circ = np.linalg.norm(delta, axis=-1) - self.circles[None, :, 2]
            dist = np.minimum(dist, d_circ.min(axis=1))

        if len(self.boxes):
            lo = self.boxes[None, :, :2]
            hi = self.boxes[None, :, 2:]
            p = pts[:, None, :]
            # Outside distance is the norm of the per-axis overshoot; inside
            # distance is negative and equals the depth to the nearest face.
            outside = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=-1)
            inside = np.maximum(np.maximum(lo - p, p - hi), -np.inf).max(axis=-1)
            d_box = np.where(outside > 0.0, outside, inside)
            dist = np.minimum(dist, d_box.min(axis=1))

        return dist.reshape(leading_shape)

    def is_free(self, points: np.ndarray, radius: float | None = None) -> np.ndarray:
        """Whether a disc of ``radius`` centred at ``points`` is collision-free."""
        r = self.config.robot_radius if radius is None else radius
        return self.clearance(points) > r

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------
    @property
    def occupancy(self) -> np.ndarray:
        """Boolean grid, ``True`` where the robot centre may **not** go.

        The grid is inflated by the robot radius, which is exactly the
        configuration-space obstacle representation a classical planner
        (A*, and later Nav2's costmap) operates on.  Indexed ``[row, col]``
        with row along ``y`` and column along ``x``.
        """
        if self._occupancy is None:
            res = self.config.grid_resolution
            n_x = int(np.ceil(self.config.width / res))
            n_y = int(np.ceil(self.config.height / res))
            # Sample at cell centres.
            xs = (np.arange(n_x) + 0.5) * res
            ys = (np.arange(n_y) + 0.5) * res
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
            free = self.is_free(pts).reshape(n_y, n_x)
            self._occupancy = ~free
        return self._occupancy

    def occupancy_at(self, radius: float) -> np.ndarray:
        """Occupancy grid inflated by an arbitrary ``radius``.

        The classical baseline plans at ``robot_radius + safety_margin`` so
        its path has tracking margin, mirroring how Nav2's inflation layer
        keeps the global plan away from obstacle edges.
        """
        if np.isclose(radius, self.config.robot_radius):
            return self.occupancy
        res = self.config.grid_resolution
        n_x = int(np.ceil(self.config.width / res))
        n_y = int(np.ceil(self.config.height / res))
        xs = (np.arange(n_x) + 0.5) * res
        ys = (np.arange(n_y) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        return (self.clearance(pts) <= radius).reshape(n_y, n_x)

    def world_to_grid(self, points: np.ndarray) -> np.ndarray:
        """Convert world coordinates to ``(row, col)`` grid indices."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        res = self.config.grid_resolution
        n_y, n_x = self.occupancy.shape
        cols = np.clip((pts[:, 0] / res).astype(int), 0, n_x - 1)
        rows = np.clip((pts[:, 1] / res).astype(int), 0, n_y - 1)
        out = np.stack([rows, cols], axis=-1)
        return out[0] if np.ndim(points) == 1 else out

    def grid_to_world(self, cells: np.ndarray) -> np.ndarray:
        """Convert ``(row, col)`` grid indices to world coordinates."""
        single = np.ndim(cells) == 1
        arr = np.atleast_2d(np.asarray(cells))
        res = self.config.grid_resolution
        xy = np.stack([(arr[:, 1] + 0.5) * res, (arr[:, 0] + 0.5) * res], axis=-1)
        return xy[0] if single else xy


def _sample_obstacles(rng: np.random.Generator, cfg: WorldConfig) -> tuple[np.ndarray, np.ndarray]:
    n_circ = int(rng.integers(cfg.n_circles[0], cfg.n_circles[1] + 1))
    n_box = int(rng.integers(cfg.n_boxes[0], cfg.n_boxes[1] + 1))

    margin = 0.2
    if n_circ:
        radii = rng.uniform(cfg.circle_radius[0], cfg.circle_radius[1], size=n_circ)
        cx = rng.uniform(margin + radii, cfg.width - margin - radii)
        cy = rng.uniform(margin + radii, cfg.height - margin - radii)
        circles = np.stack([cx, cy, radii], axis=-1)
    else:
        circles = np.zeros((0, 3))

    if n_box:
        w = rng.uniform(cfg.box_size[0], cfg.box_size[1], size=n_box)
        h = rng.uniform(cfg.box_size[0], cfg.box_size[1], size=n_box)
        x_lo = rng.uniform(margin, cfg.width - margin - w)
        y_lo = rng.uniform(margin, cfg.height - margin - h)
        boxes = np.stack([x_lo, y_lo, x_lo + w, y_lo + h], axis=-1)
    else:
        boxes = np.zeros((0, 4))

    return circles, boxes


def generate_world(seed: int, config: WorldConfig | None = None) -> World:
    """Generate a reproducible world, guaranteed solvable.

    The start and goal are sampled to be collision-free, at least
    ``config.min_start_goal_dist`` apart, and **connected** on the inflated
    occupancy grid.  Guaranteeing reachability matters: if some episodes were
    unsolvable, success rate and SPL would silently conflate policy failure
    with task infeasibility, and the classical-vs-learned comparison would be
    meaningless.

    Parameters
    ----------
    seed:
        Integer identifying the world.  The same seed always yields the same
        scene, which is how held-out evaluation splits are defined.
    config:
        Generation parameters; defaults to :class:`WorldConfig`.
    """
    from vision_nav.planning.grid_astar import astar_grid  # local import: avoids a cycle

    cfg = config or WorldConfig()
    rng = np.random.default_rng(seed)

    for _ in range(64):  # outer retries: resample the whole obstacle layout
        circles, boxes = _sample_obstacles(rng, cfg)
        world = World(
            config=cfg,
            circles=circles,
            boxes=boxes,
            start=np.zeros(3),
            goal=np.zeros(2),
            seed=seed,
        )

        free = ~world.occupancy
        free_cells = np.argwhere(free)
        if len(free_cells) < 2:
            continue
        free_xy = world.grid_to_world(free_cells)

        clearance = world.clearance(free_xy)
        ok = clearance > cfg.robot_radius + cfg.spawn_clearance
        if ok.sum() < 2:
            continue
        cand_cells = free_cells[ok]
        cand_xy = free_xy[ok]

        for _ in range(cfg.max_spawn_attempts):
            i, j = rng.choice(len(cand_xy), size=2, replace=False)
            start_xy, goal_xy = cand_xy[i], cand_xy[j]
            if np.linalg.norm(goal_xy - start_xy) < cfg.min_start_goal_dist:
                continue
            path = astar_grid(world.occupancy, tuple(cand_cells[i]), tuple(cand_cells[j]))
            if path is None:
                continue
            theta = rng.uniform(-np.pi, np.pi)
            world.start = np.array([start_xy[0], start_xy[1], theta])
            world.goal = np.asarray(goal_xy, dtype=np.float64)
            return world

    raise RuntimeError(
        f"failed to generate a solvable world for seed={seed}; the obstacle "
        "density or min_start_goal_dist in WorldConfig is likely too aggressive"
    )
