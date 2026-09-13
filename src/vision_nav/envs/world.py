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

    #: Moving obstacles. Zero by default, so every earlier experiment is
    #: unaffected. These are deliberately **absent from the occupancy grid**:
    #: they are the hazard a map-based planner cannot know about in advance,
    #: which is the whole point of the condition.
    n_dynamic: tuple[int, int] = (0, 0)
    dynamic_radius: tuple[float, float] = (0.25, 0.45)
    #: Peak-to-centre travel of each mover, in metres.
    dynamic_amplitude: tuple[float, float] = (1.0, 2.5)
    #: Speed in m/s at the midpoint of the sweep.
    dynamic_speed: tuple[float, float] = (0.15, 0.45)

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

    #: ``(M, 7)`` array of moving obstacles:
    #: ``(cx, cy, radius, dir_x, dir_y, amplitude, omega)``. Position at time
    #: ``t`` is ``centre + dir * amplitude * sin(omega * t)`` — smooth,
    #: bounded, periodic and fully determined by the world seed.
    dynamic: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))

    _occupancy: np.ndarray | None = field(default=None, repr=False, compare=False)
    #: Current positions of the movers, ``(M, 3)`` as ``(x, y, radius)``.
    _dyn_now: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3)), repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def set_time(self, t: float) -> None:
        """Advance the moving obstacles to simulation time ``t`` seconds.

        Static worlds ignore this. The env calls it every step, so
        :meth:`clearance` and the range sensors automatically see the movers
        where they currently are without every call site having to thread a
        timestamp through.
        """
        if not len(self.dynamic):
            return
        d = self.dynamic
        offset = (d[:, 5] * np.sin(d[:, 6] * float(t)))[:, None]
        self._dyn_now = np.concatenate(
            [d[:, :2] + d[:, 3:5] * offset, d[:, 2:3]], axis=1
        )

    @property
    def has_dynamic(self) -> bool:
        return bool(len(self.dynamic))

    @property
    def sensed_circles(self) -> np.ndarray:
        """Discs a range sensor can see: static obstacles plus current movers.

        Distinct from :attr:`circles`, which is the static layout the map and
        the planner are built from. Keeping the two separate is what makes the
        dynamic condition meaningful.
        """
        if not len(self._dyn_now):
            return self.circles
        if not len(self.circles):
            return self._dyn_now
        return np.concatenate([self.circles, self._dyn_now], axis=0)

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------
    def clearance(self, points: np.ndarray, include_dynamic: bool = True) -> np.ndarray:
        """Signed distance from ``points`` to the nearest obstacle or wall.

        Parameters
        ----------
        points:
            ``(..., 2)`` array of query positions.
        include_dynamic:
            Whether moving obstacles count. ``True`` for physics and sensing —
            a mover you drive into is a collision like any other. ``False``
            for the occupancy grid, because the grid is the *map*, and the
            premise of the dynamic condition is that the map does not contain
            them.

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

        discs = self.circles
        if include_dynamic and len(self._dyn_now):
            discs = np.concatenate([discs, self._dyn_now], axis=0) if len(discs) else self._dyn_now

        if len(discs):
            delta = pts[:, None, :] - discs[None, :, :2]
            d_circ = np.linalg.norm(delta, axis=-1) - discs[None, :, 2]
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

    def is_free(
        self,
        points: np.ndarray,
        radius: float | None = None,
        include_dynamic: bool = True,
    ) -> np.ndarray:
        """Whether a disc of ``radius`` centred at ``points`` is collision-free."""
        r = self.config.robot_radius if radius is None else radius
        return self.clearance(points, include_dynamic=include_dynamic) > r

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
            # Static only: the grid is the map, and movers are not in it.
            free = self.is_free(pts, include_dynamic=False).reshape(n_y, n_x)
            self._occupancy = ~free
        return self._occupancy

    def occupancy_at(self, radius: float, include_dynamic: bool = False) -> np.ndarray:
        """Occupancy grid inflated by an arbitrary ``radius``.

        The classical baseline plans at ``robot_radius + safety_margin`` so
        its path has tracking margin, mirroring how Nav2's inflation layer
        keeps the global plan away from obstacle edges.
        """
        if np.isclose(radius, self.config.robot_radius) and not include_dynamic:
            return self.occupancy
        res = self.config.grid_resolution
        n_x = int(np.ceil(self.config.width / res))
        n_y = int(np.ceil(self.config.height / res))
        xs = (np.arange(n_x) + 0.5) * res
        ys = (np.arange(n_y) + 0.5) * res
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        occupied = self.clearance(pts, include_dynamic=include_dynamic) <= radius
        return occupied.reshape(n_y, n_x)

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


def _sample_dynamic(rng: np.random.Generator, cfg: WorldConfig, world: World) -> np.ndarray:
    """Place moving obstacles on free ground, sweeping along a clear line.

    Each mover's whole sweep is checked against the *static* layout, so it
    never oscillates through a wall. It may of course cross the robot's route
    — that is the entire point.
    """
    lo, hi = cfg.n_dynamic
    n = int(rng.integers(lo, hi + 1)) if hi >= lo else 0
    if n <= 0:
        return np.zeros((0, 7))

    movers = []
    for _ in range(n * 40):
        if len(movers) >= n:
            break
        radius = float(rng.uniform(*cfg.dynamic_radius))
        amplitude = float(rng.uniform(*cfg.dynamic_amplitude))
        speed = float(rng.uniform(*cfg.dynamic_speed))
        angle = float(rng.uniform(-np.pi, np.pi))
        direction = np.array([np.cos(angle), np.sin(angle)])
        centre = np.array(
            [rng.uniform(0.5, cfg.width - 0.5), rng.uniform(0.5, cfg.height - 0.5)]
        )

        # The mover must not clip static geometry anywhere along its sweep,
        # and must not start on top of the robot or the goal.
        ts = np.linspace(-1.0, 1.0, 9)[:, None]
        sweep = centre[None, :] + direction[None, :] * amplitude * ts
        if np.any(world.clearance(sweep, include_dynamic=False) <= radius + 0.05):
            continue
        endpoints = np.array([world.start[:2], world.goal])
        if np.min(np.linalg.norm(sweep[:, None, :] - endpoints[None, :, :], axis=-1)) < (
            radius + cfg.robot_radius + 0.6
        ):
            continue

        # omega chosen so peak speed (amplitude * omega) matches the target.
        omega = speed / max(amplitude, 1e-6)
        movers.append([centre[0], centre[1], radius, direction[0], direction[1], amplitude, omega])

    return np.asarray(movers, dtype=np.float64) if movers else np.zeros((0, 7))


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
            # Movers are placed last: their sweeps are validated against the
            # finished static layout and the chosen start/goal.
            world.dynamic = _sample_dynamic(rng, cfg, world)
            world.set_time(0.0)
            return world

    raise RuntimeError(
        f"failed to generate a solvable world for seed={seed}; the obstacle "
        "density or min_start_goal_dist in WorldConfig is likely too aggressive"
    )
