"""A SLAM back end: the pose graph this stack has been missing.

Everything up to here is a **front end**. The robot integrates its encoders,
matches each scan against the map it has built, and keeps the corrected pose
(:mod:`vision_nav.mapping.localisation`). §9.5 measured that at about 0.1 m on
noise-free worlds at 360 beams, with no pose graph at all, and §12 asked for a
back end on two separate grounds:

* on the 32-beam scanner, slam_toolbox's back end did better than this stack on
  `sparse` (0.690 against 0.570) and worse in tight corridors, so the sparse
  case is where a back end has visibly been worth having;
* §9.11 measured why tight gaps fail -- the robot holds its line to about
  0.02 m and is wrong about where that line is by 0.065 to 0.125 m, against a
  gap that leaves 0.017 to 0.025 m of room. The pose is the binding constraint.

A front end can only ever be as consistent as its last correction. It matches
against a map that its own earlier errors built, so a drift that has already
been written into the map is invisible to it: the scan agrees with the wrong
map, the score is high, and nothing objects. A back end keeps the *history* --
every keyframe's pose and the scan taken there -- notices when the robot is
somewhere it has been before, and redistributes the accumulated error over the
whole trajectory instead of absorbing it at the present pose.

What is implemented here:

**Keyframes.** A node every ``keyframe_distance`` metres or ``keyframe_angle``
radians, which is upstream slam_toolbox's rule and its default values -- chosen
there and kept here, after Phase 6g's lesson that lowering them to "scale for
small worlds" was what dragged that stack's pose backwards. Keyframing is also
what keeps the graph small enough for dense linear algebra: a 12 m world yields
a few tens of nodes, so a 120x120 solve, and this project has no scipy.

**Odometry edges** between consecutive keyframes, from the front end's own
estimates, weighted by ``odom_weight``.

**Loop closures.** For each new keyframe, the older keyframes within
``loop_radius`` and at least ``loop_min_gap`` keyframes back are candidates. The
new scan is matched against a likelihood field rasterised from the candidate's
own scan -- scan against scan, never against the map, because the map is what
the drift has already corrupted. A closure is accepted only if the best fit
scores at least ``loop_min_score``; a bad closure is far worse than none.

**Gauss-Newton on SE(2).** Standard pose-graph least squares: for an edge
``(i, j)`` measuring the transform ``z``, the error is
``(T_i^-1 T_j) - z`` in the tangent space, node 0 is held fixed as the gauge,
and ``H dx = -b`` is solved densely.

**The map is rebuilt, not patched.** An optimisation that moves the poses
invalidates the map those poses built, so when the correction exceeds
``rebuild_threshold`` the stored scans are re-integrated into a fresh grid at
the corrected poses. Moving the robot's estimate and leaving the map alone
would put the two in disagreement and the front end would simply pull the pose
back to the stale map on the next scan -- which is a way to make a back end
look inert while it is in fact fighting itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.robot import wrap_angle

__all__ = ["PoseGraph", "PoseGraphConfig", "compose_pose", "relative_pose",
           "scan_points"]


@dataclass
class PoseGraphConfig:
    """Everything the back end needs, and where each number comes from."""

    #: Distance and rotation between keyframes. Upstream slam_toolbox's
    #: ``minimum_travel_distance`` and ``minimum_travel_heading``, kept at their
    #: own defaults: Phase 6g lowered them to 0.2 "to scale for small worlds"
    #: and that alone dragged the pose 2.64 m backwards over 5.96 m driven.
    keyframe_distance: float = 0.5
    keyframe_angle: float = 0.5
    #: A closure is looked for only among keyframes this close in space and this
    #: far back in the graph. The gap is what stops a node closing against its
    #: own neighbour, which measures nothing and pins the graph to its drift.
    loop_radius: float = 2.5
    loop_min_gap: int = 6
    #: Coarse then fine, like the front end's search. The coarse window has to
    #: cover the drift a closure is meant to correct.
    loop_window: float = 0.6
    loop_step: float = 0.1
    loop_angular_window: float = 0.25
    loop_angular_step: float = 0.05
    #: The blur on a keyframe's scan when it is used as a target, and the grid
    #: that field is rasterised on.
    loop_sigma: float = 0.12
    loop_resolution: float = 0.05
    #: At most this many scan returns are used per closure attempt: the search
    #: is over thousands of candidate poses, and the fit is decided by the shape
    #: of the scan rather than by every last beam of it.
    loop_max_points: int = 120
    #: A closure is accepted only above this mean field score. Deliberately
    #: strict: a wrong closure drags the whole trajectory, and the failure mode
    #: of loop closure is confident nonsense in self-similar corridors.
    loop_min_score: float = 0.55
    #: Keyframes between optimisations, and Gauss-Newton iterations per solve.
    optimise_every: int = 3
    iterations: int = 12
    #: Information weights. Odometry between adjacent keyframes is trusted more
    #: in rotation than translation, which is how this robot's noise model
    #: behaves (:class:`OdometryConfig`); a closure is trusted less than
    #: odometry until it has earned it.
    odom_xy_weight: float = 1.0
    odom_theta_weight: float = 4.0
    loop_xy_weight: float = 0.6
    loop_theta_weight: float = 2.0
    #: How far the newest pose must move before the map is rebuilt from the
    #: corrected trajectory, and how many rebuilds an episode may pay for.
    rebuild_threshold: float = 0.05
    max_rebuilds: int = 6


def relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The pose of ``b`` in the frame of ``a``."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = b[:2] - a[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1],
                     wrap_angle(b[2] - a[2])])


def compose_pose(a: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """``a`` followed by the relative transform ``rel``.

    The pair is what lets the caller carry a pose across an optimisation: take
    the pose relative to a keyframe *before* the solve, and put it back on that
    keyframe's corrected pose afterwards. Every non-keyframe pose in an episode
    -- each scan the map was built from -- moves this way.
    """
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * rel[0] - s * rel[1],
                     a[1] + s * rel[0] + c * rel[1],
                     wrap_angle(a[2] + rel[2])])


def scan_points(ranges: np.ndarray, sensor) -> np.ndarray:
    """The returns of a scan as points in the robot's own frame.

    Only actual returns: a beam at maximum range hit nothing, and feeding its
    endpoint to a matcher would invent a surface out in free space.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    hit = ranges < float(sensor.config.max_range) - 1e-6
    bearings = np.asarray(sensor._angles, dtype=np.float64)[hit]
    r = ranges[hit]
    return np.stack([r * np.cos(bearings), r * np.sin(bearings)], axis=-1)


class PoseGraph:
    """Keyframes, closures and a least-squares solve over the trajectory."""

    def __init__(self, config: PoseGraphConfig | None = None) -> None:
        self.config = config or PoseGraphConfig()
        self.reset()

    def reset(self) -> None:
        #: Keyframe poses, and the scan taken at each in the robot's own frame.
        self.poses: list[np.ndarray] = []
        self.scans: list[np.ndarray] = []
        #: ``(i, j, measurement, xy_weight, theta_weight)``.
        self.edges: list[tuple[int, int, np.ndarray, float, float]] = []
        #: Diagnostics for the episode.
        self.closures = 0
        self.closures_rejected = 0
        self.optimisations = 0
        self.rebuilds = 0
        self.last_correction = 0.0
        self.total_correction = 0.0
        self._since_optimise = 0

    # ------------------------------------------------------------------
    def due(self, pose: np.ndarray) -> bool:
        """Whether ``pose`` is far enough from the last keyframe to add one."""
        if not self.poses:
            return True
        rel = relative_pose(self.poses[-1], np.asarray(pose, dtype=np.float64))
        return (float(np.hypot(rel[0], rel[1])) >= self.config.keyframe_distance
                or abs(float(rel[2])) >= self.config.keyframe_angle)

    def add_keyframe(self, pose: np.ndarray, local: np.ndarray) -> int:
        """Add a node at ``pose`` holding ``local`` scan points, and link it."""
        pose = np.asarray(pose, dtype=np.float64).copy()
        self.poses.append(pose)
        self.scans.append(np.asarray(local, dtype=np.float64))
        i = len(self.poses) - 1
        if i > 0:
            self.edges.append((i - 1, i, relative_pose(self.poses[i - 1], pose),
                               self.config.odom_xy_weight,
                               self.config.odom_theta_weight))
        self._since_optimise += 1
        return i

    # ------------------------------------------------------------------
    def find_closure(self, i: int) -> bool:
        """Look for one loop closure against keyframe ``i``. True if found."""
        cfg = self.config
        best: tuple[float, int, np.ndarray] | None = None
        for j in range(0, i - cfg.loop_min_gap + 1):
            if float(np.linalg.norm(self.poses[i][:2] - self.poses[j][:2])) > cfg.loop_radius:
                continue
            score, rel = self._match(i, j)
            if best is None or score > best[0]:
                best = (score, j, rel)
        if best is None:
            return False
        score, j, rel = best
        if score < cfg.loop_min_score:
            self.closures_rejected += 1
            return False
        self.edges.append((j, i, rel, cfg.loop_xy_weight, cfg.loop_theta_weight))
        self.closures += 1
        return True

    def _match(self, i: int, j: int) -> tuple[float, np.ndarray]:
        """Match keyframe ``i``'s scan against keyframe ``j``'s own scan.

        Against the scan, not the map: a drift already written into the map is
        invisible to a matcher that scores against the map.
        """
        cfg = self.config
        target = self._subsample(self.scans[j])
        source = self._subsample(self.scans[i])
        if len(target) < 8 or len(source) < 8:
            return -np.inf, np.zeros(3)

        # Rasterise the target scan, in keyframe j's own frame, and blur it.
        field, origin = self._field(target)
        # The source scan enters that frame through the current estimate of the
        # relative pose, which is the guess the search starts from.
        guess = relative_pose(self.poses[j], self.poses[i])

        best = self._search(guess, source, field, origin,
                            cfg.loop_window, cfg.loop_step,
                            cfg.loop_angular_window, cfg.loop_angular_step)
        best = self._search(best, source, field, origin,
                            cfg.loop_step, cfg.loop_step / 4.0,
                            cfg.loop_angular_step, cfg.loop_angular_step / 3.0)
        score = float(self._score(best, source, field, origin))
        return score, best

    def _subsample(self, local: np.ndarray) -> np.ndarray:
        n = self.config.loop_max_points
        if len(local) <= n:
            return local
        idx = np.linspace(0, len(local) - 1, n).astype(int)
        return local[idx]

    def _field(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A blurred occupancy image of ``points``, and its origin in metres."""
        cfg = self.config
        res = cfg.loop_resolution
        pad = 2.0 * cfg.loop_sigma + cfg.loop_window + res
        origin = points.min(axis=0) - pad
        extent = points.max(axis=0) + pad - origin
        w = int(np.ceil(extent[0] / res)) + 1
        h = int(np.ceil(extent[1] / res)) + 1
        grid = np.zeros((h, w), dtype=np.float32)
        cols = np.clip(((points[:, 0] - origin[0]) / res).astype(int), 0, w - 1)
        rows = np.clip(((points[:, 1] - origin[1]) / res).astype(int), 0, h - 1)
        grid[rows, cols] = 1.0

        k = int(np.ceil(2.0 * cfg.loop_sigma / res))
        padded = np.pad(grid, k)
        field = np.zeros_like(grid)
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                weight = float(np.exp(-((dr * dr + dc * dc) * res * res)
                                      / (2.0 * cfg.loop_sigma ** 2)))
                if weight < 0.01:
                    continue
                np.maximum(field, weight * padded[k + dr:k + dr + h, k + dc:k + dc + w],
                           out=field)
        return field, origin

    def _score(self, rel: np.ndarray, source: np.ndarray, field: np.ndarray,
               origin: np.ndarray) -> float:
        pts = _project(rel, source)
        return float(_sample(field, pts[None], self.config.loop_resolution, origin)[0].mean())

    def _search(self, centre: np.ndarray, source: np.ndarray, field: np.ndarray,
                origin: np.ndarray, window: float, step: float,
                angular_window: float, angular_step: float) -> np.ndarray:
        offsets = _grid(window, step)
        dthetas = _offsets(angular_window, angular_step)
        th = centre[2] + dthetas
        cos, sin = np.cos(th)[:, None], np.sin(th)[:, None]
        x = cos * source[None, :, 0] - sin * source[None, :, 1]
        y = sin * source[None, :, 0] + cos * source[None, :, 1]
        pts = np.stack([x, y], axis=-1) + centre[:2]
        cand = pts[:, None, :, :] + offsets[None, :, None, :]
        score = _sample(field, cand, self.config.loop_resolution, origin).mean(axis=-1)
        t, k = np.unravel_index(int(np.argmax(score)), score.shape)
        out = centre.copy()
        out[:2] = centre[:2] + offsets[k]
        out[2] = wrap_angle(centre[2] + dthetas[t])
        return out

    # ------------------------------------------------------------------
    def optimise(self) -> float:
        """Least-squares over the trajectory. Returns how far the last pose moved.

        Node 0 is the gauge: the map frame is defined by where the robot started,
        so holding it fixed is not an approximation, it is the convention the
        rest of the stack already uses.
        """
        n = len(self.poses)
        if n < 3 or not self.edges:
            return 0.0
        before = self.poses[-1].copy()
        for _ in range(self.config.iterations):
            H = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)
            for i, j, z, w_xy, w_th in self.edges:
                e, Ji, Jj = self._residual(i, j, z)
                omega = np.diag([w_xy, w_xy, w_th])
                si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                H[si, si] += Ji.T @ omega @ Ji
                H[si, sj] += Ji.T @ omega @ Jj
                H[sj, si] += Jj.T @ omega @ Ji
                H[sj, sj] += Jj.T @ omega @ Jj
                b[si] += Ji.T @ omega @ e
                b[sj] += Jj.T @ omega @ e
            # Gauge: pin node 0, and keep H invertible for nodes no edge reaches.
            H[:3, :3] += np.eye(3) * 1e6
            H += np.eye(3 * n) * 1e-6
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                break
            for k in range(n):
                self.poses[k][:2] += dx[3 * k:3 * k + 2]
                self.poses[k][2] = wrap_angle(self.poses[k][2] + dx[3 * k + 2])
            if float(np.max(np.abs(dx))) < 1e-6:
                break
        self.optimisations += 1
        self._since_optimise = 0
        moved = float(np.linalg.norm(self.poses[-1][:2] - before[:2]))
        self.last_correction = moved
        self.total_correction += moved
        return moved

    def _residual(self, i: int, j: int, z: np.ndarray):
        pi, pj = self.poses[i], self.poses[j]
        c, s = np.cos(pi[2]), np.sin(pi[2])
        R_t = np.array([[c, s], [-s, c]])
        d = pj[:2] - pi[:2]
        e = np.empty(3)
        e[:2] = R_t @ d - z[:2]
        e[2] = wrap_angle(pj[2] - pi[2] - z[2])
        dR_t = np.array([[-s, c], [-c, -s]])
        Ji = np.zeros((3, 3))
        Ji[:2, :2] = -R_t
        Ji[:2, 2] = dR_t @ d
        Ji[2, 2] = -1.0
        Jj = np.zeros((3, 3))
        Jj[:2, :2] = R_t
        Jj[2, 2] = 1.0
        return e, Ji, Jj

    @property
    def due_to_optimise(self) -> bool:
        return self._since_optimise >= self.config.optimise_every


def _project(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack([c * local[:, 0] - s * local[:, 1],
                     s * local[:, 0] + c * local[:, 1]], axis=-1) + pose[:2]


def _sample(field: np.ndarray, points: np.ndarray, res: float,
            origin: np.ndarray) -> np.ndarray:
    """Bilinear read of a local field whose corner sits at ``origin``."""
    h, w = field.shape
    gy = np.clip((points[..., 1] - origin[1]) / res, 0.0, h - 1.0000001)
    gx = np.clip((points[..., 0] - origin[0]) / res, 0.0, w - 1.0000001)
    i0, j0 = gy.astype(int), gx.astype(int)
    fy, fx = gy - i0, gx - j0
    i1, j1 = np.minimum(i0 + 1, h - 1), np.minimum(j0 + 1, w - 1)
    return ((1.0 - fy) * ((1.0 - fx) * field[i0, j0] + fx * field[i0, j1])
            + fy * ((1.0 - fx) * field[i1, j0] + fx * field[i1, j1]))


def _offsets(window: float, step: float) -> np.ndarray:
    n = int(np.floor(window / step + 1e-9))
    return np.arange(-n, n + 1) * step


def _grid(window: float, step: float) -> np.ndarray:
    line = _offsets(window, step)
    dx, dy = np.meshgrid(line, line, indexing="ij")
    return np.stack([dx.ravel(), dy.ravel()], axis=-1)
