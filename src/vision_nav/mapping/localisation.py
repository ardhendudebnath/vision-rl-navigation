"""Where the robot thinks it is, when nobody tells it.

Every classical number in this project has been measured with the planner
holding its exact pose. §9.2 took away the map and said so explicitly: the pose
was the last privilege left, and the one a real robot lacks most obviously. This
is the standard answer to not having it, in two pieces that are measured
separately:

- :class:`DeadReckoning` integrates the wheel encoders. It is the robot's own
  kinematics driven by a corrupted velocity reading, so with the noise switched
  off it reproduces the true pose exactly -- which is the identity control the
  experiment needs, and what the unit tests hold it to.
- :class:`ScanMatcher` drags the estimate back onto the map the robot is
  already building, by searching a small window of poses for the one that puts
  the current scan's returns on top of surfaces the map already holds. This is
  correlative scan matching against a likelihood field, the front end of every
  2-D SLAM stack.

What this is not: there is no back end. No pose graph, no loop closure, no
relinearising the past when the robot returns somewhere it has been. The map is
built from the corrected estimate and the estimate is corrected against that
map, so an error the matcher accepts is an error it will keep agreeing with.
That is exactly the behaviour of a front end on its own, and naming it is part
of the result -- the numbers bound what *this* stack pays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.mapping.occupancy import OCCUPIED

__all__ = ["OdometryConfig", "ScanMatchConfig", "DeadReckoning", "ScanMatcher"]


@dataclass
class OdometryConfig:
    """How badly the wheels lie, in the two ways wheels lie.

    The **random** part is the odometry motion model every probabilistic-robotics
    treatment uses: each control period's error scales with how far the robot
    drove and how far it turned, and is drawn fresh. On its own it is a random
    walk, and a random walk is optimistic -- it cancels itself out. What
    actually moves a real robot off its map is the **systematic** part: a wheel
    radius a percent off, a wheelbase measured slightly wrong, a base that pulls
    left. Those are constant for a robot, so their effect grows with the square
    of the distance driven rather than its square root, and calibrating them out
    is what UMBmark and its successors are for. A model with only the random
    part would be measuring a robot nobody owns.

    The defaults are middle-of-the-road figures for an uncalibrated wheeled
    base. Heading is the term worth reading: an error in it multiplies
    everything driven afterwards. Their consequence is not assumed -- the drift
    they produce is measured, and reported with the experiment.
    """

    #: Translation error std as a fraction of the distance driven.
    trans_per_m: float = 0.05
    #: Translation error std in metres per radian turned.
    trans_per_rad: float = 0.01
    #: Heading error std in radians per metre driven (0.02 ~ 1.1 deg/m).
    rot_per_m: float = 0.02
    #: Heading error std as a fraction of the rotation commanded.
    rot_per_rad: float = 0.05

    #: Std of the per-robot scale error on distance travelled, drawn once and
    #: then constant: a 2% error in the assumed wheel radius.
    bias_trans: float = 0.02
    #: Std of the per-robot heading bias, in radians per metre driven, drawn
    #: once and then constant (0.01 ~ 0.6 deg/m of unmodelled pull).
    bias_rot_per_m: float = 0.01


class DeadReckoning:
    """The robot's kinematics, driven by what the encoders report.

    The integration is :meth:`DiffDriveRobot.step`'s own arc formula, copied
    rather than shared because the two have different reasons to exist: the
    robot integrates what happened, this integrates what the robot was told
    happened. ``test_localisation`` drives both from the same actions and
    requires them to agree bit for bit when the noise is off.
    """

    def __init__(self, robot: RobotConfig | None = None,
                 noise: OdometryConfig | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.robot = robot or RobotConfig()
        self.noise = noise
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pose = np.zeros(3, dtype=np.float64)
        #: This robot's systematic error, drawn once per episode by
        #: :meth:`reset`: a scale error on distance and a heading pull per
        #: metre driven.
        self.bias = (0.0, 0.0)

    def reset(self, pose: np.ndarray) -> np.ndarray:
        """Start from a known pose. Real stacks define the map frame by the
        pose they start at, so this is not a privilege -- it is the origin."""
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        if self.noise is not None:
            self.bias = (float(self.rng.normal(0.0, self.noise.bias_trans)),
                         float(self.rng.normal(0.0, self.noise.bias_rot_per_m)))
        return self.pose.copy()

    def update(self, velocity: np.ndarray) -> np.ndarray:
        """Advance one control period on the reported ``(v, omega)``."""
        v = float(velocity[0])
        omega = float(velocity[1])
        if self.noise is not None:
            dt = self.robot.dt
            driven, turned = abs(v) * dt, abs(omega) * dt
            n = self.noise
            sigma_d = n.trans_per_m * driven + n.trans_per_rad * turned
            sigma_r = n.rot_per_m * driven + n.rot_per_rad * turned
            scale, pull = self.bias
            # Systematic first: a scale error on the wheels, and a heading pull
            # proportional to distance driven rather than to time, because it
            # comes from the wheels and not from the clock.
            omega += pull * v
            v *= 1.0 + scale
            # Then the random part: noise on the increment, converted back to a
            # velocity, since the integrator below is the robot's own and takes
            # velocities.
            if sigma_d > 0.0:
                v += float(self.rng.normal(0.0, sigma_d)) / dt
            if sigma_r > 0.0:
                omega += float(self.rng.normal(0.0, sigma_r)) / dt
        self._integrate(v, omega)
        return self.pose.copy()

    def _integrate(self, v: float, omega: float) -> None:
        dt = self.robot.dt
        theta = self.pose[2]
        if abs(omega) < 1e-6:
            self.pose[0] += v * np.cos(theta) * dt
            self.pose[1] += v * np.sin(theta) * dt
        else:
            theta_new = theta + omega * dt
            radius = v / omega
            self.pose[0] += radius * (np.sin(theta_new) - np.sin(theta))
            self.pose[1] -= radius * (np.cos(theta_new) - np.cos(theta))
            self.pose[2] = theta_new
        if abs(omega) < 1e-6:
            self.pose[2] = theta
        self.pose[2] = wrap_angle(self.pose[2])


@dataclass
class ScanMatchConfig:
    """The search, and what stops it wandering.

    Two settings here are not decoration, and the first version of this class
    lacked both. Measured on the ``val`` band with a *perfect* pose and a clean
    map, the raw score is nearly flat near the truth: the best pose in a 0.20 m
    window beat the true pose by 2% of the score, at a median 0.035 m away. Take
    that offer every control period, write the scan into the map at it, and the
    pose random-walks out of the building -- which is what it did, reaching
    2.9 m of error on a robot whose odometry was exact.

    - The score **interpolates** the field between cell centres, so a
      centimetre of movement changes it by a centimetre's worth. Looking the
      cells up directly makes the objective a staircase, and a staircase has
      plateaus wide enough to hide the true pose somewhere in the middle of one.
    - A **prior** charges for leaving the pose the odometry predicted, which is
      what a real front end does (Cartographer spells it
      ``translation_delta_cost_weight``). It has to be strong enough that a
      match's own noise -- about 0.03 m of it, measured -- cannot be taken
      seriously, and weak enough that a real error still gets corrected a
      little at a time.

    Both prior weights were chosen by a sweep on the ``val`` seed band, which
    no experiment scores, and the sweep is reported with the experiment. The
    heading weight is the one that mattered: at 0.04 a correction large enough
    to be worth making cost more score than it could win, so heading was never
    corrected at all and the drift that dominates a long run went unopposed.
    """

    #: Width of the likelihood field around a mapped surface, in metres. A
    #: return this far from a surface still scores something, which is what
    #: lets the search see which way to move.
    sigma: float = 0.15
    #: Half-width of the translation search, and its coarse resolution.
    window: float = 0.08
    step: float = 0.02
    #: Half-width of the heading search, and its coarse resolution.
    angular_window: float = 0.024
    angular_step: float = 0.008
    #: Deviation from the predicted pose that costs one unit of score -- the
    #: whole scan's worth. Translation in metres, heading in radians.
    prior_xy: float = 0.20
    prior_theta: float = 0.08
    #: A scan with fewer returns than this says too little to move a pose on.
    min_hits: int = 6
    #: So does a map holding fewer occupied cells than this.
    min_occupied: int = 20
    #: Nor does a scan whose returns mostly land where the map knows nothing:
    #: the fraction that must land on something already mapped.
    min_overlap: float = 0.4


class ScanMatcher:
    """Correlative scan matching against the robot's own occupancy map.

    The score of a candidate pose is the sum, over the scan's returns, of a
    likelihood field built from the mapped surfaces: high where the map holds a
    surface, decaying over :attr:`ScanMatchConfig.sigma`, zero where it holds
    nothing. The search is coarse then fine, and refuses to move the pose at all
    when either the scan or the map is too thin to constrain it.
    """

    def __init__(self, config: ScanMatchConfig | None = None) -> None:
        self.config = config or ScanMatchConfig()
        self._field: np.ndarray | None = None
        self._field_version = -1
        #: Diagnostics for the episode.
        self.corrections = 0
        self.skipped = 0
        self.total_shift = 0.0
        self.last_shift = 0.0

    def reset(self) -> None:
        self._field, self._field_version = None, -1
        self.corrections = self.skipped = 0
        self.total_shift = self.last_shift = 0.0

    def invalidate(self) -> None:
        """Drop the cached likelihood field because the map under it changed.

        The cache is keyed on the map's version counter, which is enough while
        there is one map that only grows. A back end that rebuilds the map from
        a corrected trajectory hands over a *different* map whose counter starts
        again, so the key can collide and the matcher would then score a new map
        with the old map's field. Rebuilding is rare and this is cheap.
        """
        self._field, self._field_version = None, -1

    # ------------------------------------------------------------------
    def likelihood_field(self, omap) -> np.ndarray:
        """Mapped surfaces, blurred. Cached against the map's version."""
        if self._field is not None and self._field_version == omap.version:
            return self._field
        res = float(omap.resolution)
        sigma = self.config.sigma
        occupied = (omap.grid == OCCUPIED).astype(np.float32)
        field = np.zeros(omap.shape, dtype=np.float32)
        k = int(np.ceil(2.0 * sigma / res))
        padded = np.pad(occupied, k)
        h, w = omap.shape
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                weight = float(np.exp(-((dr * dr + dc * dc) * res * res)
                                      / (2.0 * sigma * sigma)))
                if weight < 0.01:
                    continue
                np.maximum(field, weight * padded[k + dr:k + dr + h, k + dc:k + dc + w],
                           out=field)
        self._field, self._field_version = field, omap.version
        return field

    def correct(self, pose: np.ndarray, ranges: np.ndarray, omap) -> np.ndarray:
        """The pose that best explains this scan, near the one predicted."""
        cfg = self.config
        pose = np.asarray(pose, dtype=np.float64)
        ranges = np.asarray(ranges, dtype=np.float64)
        max_range = float(omap.sensor.config.max_range)
        hit = ranges < max_range - 1e-6
        if int(hit.sum()) < cfg.min_hits or int((omap.grid == OCCUPIED).sum()) < cfg.min_occupied:
            return self._skip(pose)

        bearings = np.asarray(omap.sensor._angles, dtype=np.float64)[hit]
        r = ranges[hit]
        # Returns in the robot's own frame: the scan is what it is, the pose is
        # what is being searched over.
        local = np.stack([r * np.cos(bearings), r * np.sin(bearings)], axis=-1)
        field = self.likelihood_field(omap)
        res = float(omap.resolution)

        here = self._sample(field, self._project(pose, local)[None], res)[0]
        if float((here > 0.0).mean()) < cfg.min_overlap:
            # Mostly looking at space the map knows nothing about. There is a
            # maximum in the window and it means nothing.
            return self._skip(pose)

        best = self._search(pose, pose, local, field, res,
                            cfg.window, cfg.step, cfg.angular_window, cfg.angular_step)
        # One refinement pass around it, at a quarter of the coarse spacing.
        # The prior is still measured from the predicted pose, not from here.
        best = self._search(pose, best, local, field, res,
                            cfg.step, cfg.step / 4.0,
                            cfg.angular_step, cfg.angular_step / 3.0)
        best[2] = wrap_angle(best[2])
        self.corrections += 1
        self.last_shift = float(np.linalg.norm(best[:2] - pose[:2]))
        self.total_shift += self.last_shift
        return best

    def _skip(self, pose: np.ndarray) -> np.ndarray:
        self.skipped += 1
        self.last_shift = 0.0
        return pose.copy()

    @staticmethod
    def _project(pose, local) -> np.ndarray:
        """Scan returns in the world, as seen from ``pose``."""
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return np.stack([c * local[:, 0] - s * local[:, 1],
                         s * local[:, 0] + c * local[:, 1]], axis=-1) + pose[:2]

    @staticmethod
    def _sample(field: np.ndarray, points: np.ndarray, res: float) -> np.ndarray:
        """The field at arbitrary points, interpolated between cell centres.

        Reading the containing cell instead makes the score a staircase, and
        the search then cannot tell a pose a centimetre out from the true one.
        """
        h, w = field.shape
        gy = np.clip(points[..., 1] / res - 0.5, 0.0, h - 1.0000001)
        gx = np.clip(points[..., 0] / res - 0.5, 0.0, w - 1.0000001)
        i0, j0 = gy.astype(int), gx.astype(int)
        fy, fx = gy - i0, gx - j0
        i1, j1 = np.minimum(i0 + 1, h - 1), np.minimum(j0 + 1, w - 1)
        return ((1.0 - fy) * ((1.0 - fx) * field[i0, j0] + fx * field[i0, j1])
                + fy * ((1.0 - fx) * field[i1, j0] + fx * field[i1, j1]))

    def _search(self, prior, centre, local, field, res, window, step,
                angular_window, angular_step) -> np.ndarray:
        cfg = self.config
        offsets = self._grid(window, step)
        dthetas = self._offsets(angular_window, angular_step)

        th = centre[2] + dthetas
        cos, sin = np.cos(th)[:, None], np.sin(th)[:, None]
        x = cos * local[None, :, 0] - sin * local[None, :, 1]
        y = sin * local[None, :, 0] + cos * local[None, :, 1]
        pts = np.stack([x, y], axis=-1) + centre[:2]               # (T, N, 2)
        cand = pts[:, None, :, :] + offsets[None, :, None, :]      # (T, K, N, 2)
        score = self._sample(field, cand, res).mean(axis=-1)

        # What it costs to disagree with the odometry, measured from the pose
        # the odometry predicted rather than from the centre of this pass.
        dxy = centre[:2] + offsets - prior[:2]                     # (K, 2)
        dth = wrap_angle(centre[2] + dthetas - prior[2])           # (T,)
        score = score - (np.sum(dxy * dxy, axis=-1)[None, :] / cfg.prior_xy ** 2
                         + (dth * dth)[:, None] / cfg.prior_theta ** 2)
        t, k = np.unravel_index(int(np.argmax(score)), score.shape)
        out = centre.copy()
        out[:2] = centre[:2] + offsets[k]
        out[2] = centre[2] + dthetas[t]
        return out

    @staticmethod
    def _offsets(window: float, step: float) -> np.ndarray:
        n = int(np.floor(window / step + 1e-9))
        return np.arange(-n, n + 1) * step

    def _grid(self, window: float, step: float) -> np.ndarray:
        line = self._offsets(window, step)
        dx, dy = np.meshgrid(line, line, indexing="ij")
        return np.stack([dx.ravel(), dy.ravel()], axis=-1)
