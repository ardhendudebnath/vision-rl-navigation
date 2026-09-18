"""A classical agent that plans in space-time and tracks a schedule, not a path.

The planner (``spacetime_astar``) can decide to wait for a mover or to pass ahead
of one. None of that survives a controller that follows a *path*: pure pursuit
steers at a point some distance along the route, so a planned "wait here for
two seconds" is driven straight through. This agent tracks a *schedule* instead
-- at each moment it steers for where the plan says the robot should be a short
time from now -- so a wait in the plan is a stop on the ground.

Everything else is deliberately the spatial baseline's. Controller gains,
thresholds, safety margin and reactive slow-down are read from
:class:`PursuitConfig`'s defaults rather than restated, so the two agents cannot
drift apart and a difference between them is a difference in planning.

One asymmetry is unavoidable and stated here. The plan's time step is chosen so a
*diagonal* cell move is feasible at top speed; orthogonal moves then run at about
70% of it. The alternative -- a step sized for straight moves -- makes diagonals
physically impossible to follow, which shows up as tracking error and confounds
everything downstream. The frozen-mover control measures what the slower
straight-line driving costs on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vision_nav.agents.classical import PursuitConfig
from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.planning.grid_astar import geodesic_distance_field
from vision_nav.planning.spacetime_astar import spacetime_astar

__all__ = ["SpaceTimeConfig", "SpaceTimeAgent", "dilate_in_time"]

_BASE = PursuitConfig()


def dilate_in_time(occ: np.ndarray, margin: int) -> np.ndarray:
    """A cell is blocked at step ``k`` if it is blocked anywhere in ``k +/- margin``.

    Symmetric on purpose. A robot running *late* reaches a cell after a mover
    has arrived, which the forward half covers; one running *early* reaches it
    before the mover has left, which the backward half covers. The window's
    ends are clipped rather than wrapped.
    """
    if margin <= 0:
        return occ
    steps = occ.shape[0]
    out = np.empty_like(occ)
    for k in range(steps):
        out[k] = occ[max(0, k - margin):min(steps, k + margin + 1)].any(axis=0)
    return out


@dataclass
class SpaceTimeConfig:
    #: Seconds of mover motion the planner reasons about. Past it, only the map.
    window_s: float = 7.0
    #: Replan at least this often, in seconds of simulation time.
    replan_every_s: float = 1.0
    #: Replan immediately if the robot is this far from its scheduled position.
    replan_deviation: float = 0.25
    #: How far ahead on the schedule the controller steers, in seconds.
    track_lead_s: float = 0.5
    #: Below this distance to the scheduled point the robot is on station:
    #: it holds still rather than turning to face a point it is already at.
    station_radius: float = 0.05
    #: ``True``: the planner knows *when* each cell holds a mover. ``False``: it
    #: sees every cell a mover occupies anywhere in the window as blocked at
    #: every step -- a swept region, as the spatial baseline sees it.
    #:
    #: This is the ablation that makes the experiment mean something. The agent
    #: differs from the spatial baseline in its window, replan timer, schedule
    #: tracking and step-minimising paths as well as in reasoning about time;
    #: switching only this isolates the last. With frozen movers the union over
    #: time equals every single step, so the two settings must be bit-identical
    #: there -- an identity control, checked.
    time_varying_movers: bool = True
    #: Plan steps either side of a mover's occupancy that its cells also count
    #: as blocked. ``0`` plans to the step, which is what Phase 5p ran.
    #:
    #: Phase 5p found timing worth +0.050 on dense clutter but -0.050 on sparse
    #: worlds, all collisions, and offered an untested cause: the planner
    #: threads gaps one 0.24 s step ahead of a mover, so any tracking lag puts
    #: the robot where the mover arrives. A margin is the test of that cause.
    #: A frozen mover occupies the same cells at every step, so widening its
    #: occupancy in time changes nothing and the frozen identity still holds.
    temporal_margin_steps: int = 0
    #: Where the planner's mover futures come from. ``"oracle"`` reads the exact
    #: analytic trajectories, which is what every result up to Phase 5q used.
    #: ``"constant_velocity"`` extrapolates each mover in a straight line from
    #: the last two positions the agent has itself observed, one control step
    #: apart, and assumes it stationary until it has two.
    #:
    #: Observations are noise-free on purpose. That isolates *model* error --
    #: movers here move on sinusoids, so a straight-line extrapolation drifts as
    #: they curve and reverse -- from sensing error, which is a second question.
    #: A frozen mover has zero velocity, so its estimate is exact and the
    #: estimated planner must be bit-identical to the oracle on frozen worlds.
    #: ``"harmonic"`` fits each mover's own oscillation to the track it has been
    #: seen to follow and carries that forward. Four changes to how the planner
    #: *uses* a straight-line estimate (Phases 5r to 5w) failed to recover its
    #: dense-clutter cost; none of them changed the line itself.
    #:
    #: It is matched to how these movers move, which is the easiest case an
    #: estimator can be given: noise-free observations of a world that really is
    #: harmonic. It bounds what a better model is worth here; it is not a claim
    #: about a real sensor.
    #:
    #: ``"orbit"`` fits the same oscillation to the observations directly, by
    #: searching the frequency and solving the rest by least squares -- no
    #: differencing, so noise is averaged rather than amplified. Phase 5z
    #: measured the differencing fit losing everything it had gained once
    #: observations carry a centimetre of error.
    predictor: str = "oracle"
    #: Seconds of mover track the harmonic fit reads. Other predictors keep two
    #: observations, whatever this says, so their results cannot move.
    history_s: float = 3.0
    #: Standard deviation, in metres, of the error on each observed mover
    #: position, drawn independently per mover per step. ``0.0`` observes
    #: exactly, which is what every phase up to 5y ran.
    #:
    #: Phase 5y's fitted estimator matched the oracle from *exact* observations
    #: of a world that really is harmonic. This is the first of those two gifts
    #: taken back. It corrupts what the agent sees, never the world, and the
    #: oracle predictor ignores observations entirely -- so an oracle agent must
    #: be bit-identical with noise and without, which is the experiment's
    #: control.
    observation_noise_m: float = 0.0
    #: Seed for that noise, so an episode's corruption is the same for every arm
    #: that shares it and a run can be reproduced.
    noise_seed: int = 0
    #: ``True``: observe only movers a sensor on the robot could see -- within
    #: ``sensor_range_m``, inside ``sensor_fov_deg`` of the robot's heading, and
    #: not hidden behind static geometry. ``False`` observes every mover in the
    #: world at every step, which is what every phase up to 6a ran, and is the
    #: last privilege the estimator holds.
    #:
    #: A mover that is not visible yields no new observation: the estimator
    #: coasts on the track it has. A mover never seen is not in the planner's
    #: grid at all -- the robot does not know it exists, and may drive into it.
    observe_visible_only: bool = False
    #: Sensor horizon for that test, in metres, and its angular coverage.
    #: 360 degrees is a planar lidar; 90 matches the depth camera the learned
    #: policy reads.
    sensor_range_m: float = 6.0
    sensor_fov_deg: float = 360.0
    #: Fraction of ``safety_margin`` that movers are always inflated by, however
    #: far the planner falls back. ``0.0`` lets the last fallback plan at the
    #: bare robot radius around movers too, which is what Phase 5r ran.
    #:
    #: Phase 5r found a constant-velocity estimate losing episodes the oracle won,
    #: all to contact with a mover, and in 15 of 18 the plan in force had been
    #: made at the bare radius -- no margin at all around a mover the estimate
    #: had a median 0.03 m wrong. A plan like that is safe only under exact
    #: prediction. This withholds it: the static map may still fall back to the
    #: bare radius for a tight passage, but movers never do. Until the planner
    #: first reaches that last fallback in an episode the setting cannot change
    #: anything, which is the identity the experiment checks.
    mover_margin_floor: float = 0.0
    #: Replan as soon as an observed mover is this far, in metres, from where
    #: the estimate behind the last plan attempt put it now. ``0.0`` never
    #: does, which is what Phase 5t ran.
    #:
    #: Phase 5t showed the estimate's zero-margin plans are a symptom: the robot
    #: is already too close to a mover by the time the planner falls back to
    #: one. Between scheduled replans it acts for up to a second on an estimate
    #: made at the last plan, and a straight line drifts from a sinusoid as the
    #: second goes on. This replans the moment the drift is observed. All
    #: movers count, not only nearby ones, which costs replans but decides
    #: nothing about which movers matter. The oracle is never contradicted, so
    #: an oracle agent with this set must be bit-identical to one without --
    #: on moving worlds, which makes the control free.
    replan_innovation_m: float = 0.0
    #: Seconds past the latest observation a constant-velocity estimate is
    #: carried; beyond it each mover is held where the line had put it.
    #: ``inf`` carries the line across the whole window, which is what Phase 5u
    #: ran. The oracle ignores it.
    #:
    #: Phase 5u left 0.050 of the estimate's dense-clutter cost unexplained
    #: after a wider temporal margin, a withheld fallback and fresh estimates.
    #: None of those touched the far end of the window, where a straight line
    #: drawn seven seconds out can be metres from a mover that has turned back,
    #: and where the last step is held as a permanent obstacle. A frozen mover
    #: has zero velocity, so capping its estimate changes nothing: the identity.
    extrapolation_cap_s: float = math.inf

    # --- matched to the spatial baseline, not restated --------------------
    safety_margin: float = field(default=_BASE.safety_margin)
    heading_gain: float = field(default=_BASE.heading_gain)
    turn_in_place_threshold: float = field(default=_BASE.turn_in_place_threshold)
    slowdown_radius: float = field(default=_BASE.slowdown_radius)
    caution_clearance: float = field(default=_BASE.caution_clearance)
    min_speed_scale: float = field(default=_BASE.min_speed_scale)


class SpaceTimeAgent:
    """Space-time A* planning with oracle mover trajectories, schedule tracking."""

    def __init__(self, config: SpaceTimeConfig | None = None,
                 robot: RobotConfig | None = None) -> None:
        self.config = config or SpaceTimeConfig()
        self.robot = robot or RobotConfig()
        self._world = None
        self._waypoints: np.ndarray | None = None
        self._times: np.ndarray | None = None
        self._last_plan_t = -math.inf
        self._static_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        self._grid_pts: np.ndarray | None = None
        self._observations: list[tuple[float, np.ndarray]] = []
        self.replans = 0
        self.plan_failures = 0
        self.planned_waits = 0
        #: Times the planner reached its last fallback, the bare robot radius.
        self.bare_radius_attempts = 0
        #: Replans triggered because an observation contradicted the estimate.
        self.innovation_replans = 0
        self._plan_basis: list[tuple[float, np.ndarray]] | None = None
        self._noise = np.random.default_rng(self.config.noise_seed)
        self._orbit_cache = None
        self._pose = np.zeros(3)

    # ------------------------------------------------------------------
    @property
    def dt_plan(self) -> float:
        """One plan step: long enough for a diagonal cell move at top speed."""
        res = self._world.config.grid_resolution
        return res * math.sqrt(2.0) / self.robot.max_linear_vel

    def start_episode(self, world, pose: np.ndarray) -> bool:
        if self.config.observe_visible_only and self.config.predictor == "harmonic":
            raise ValueError("the differencing fit cannot read a track with gaps; "
                             "use predictor='orbit' or 'constant_velocity' with "
                             "observe_visible_only")
        self._world = world
        self._pose = np.asarray(pose, dtype=np.float64)
        self._waypoints = self._times = None
        self._last_plan_t = -math.inf
        self._static_cache = {}
        self._grid_pts = None
        self.replans = 0
        self.plan_failures = 0
        self.planned_waits = 0
        self.bare_radius_attempts = 0
        self.innovation_replans = 0
        self._plan_basis = None
        self._observations = []
        self._noise = np.random.default_rng(self.config.noise_seed)
        self._orbit_cache = None
        self._observe(pose)
        return self._plan(np.asarray(pose, dtype=np.float64))

    # ------------------------------------------------------------------
    def _static(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Inflated map and goal heuristic, cached: neither changes in an episode."""
        if radius not in self._static_cache:
            occ = self._world.occupancy_at(radius, include_dynamic=False)
            goal = tuple(int(v) for v in self._world.world_to_grid(self._world.goal))
            self._static_cache[radius] = (occ, geodesic_distance_field(occ, goal))
        return self._static_cache[radius]

    def _mover_occupancy(self, radius: float, steps: int) -> np.ndarray:
        """``(steps, R, C)`` cells within ``radius`` of a mover at each plan step.

        Uses the same test as the map -- clearance to the disc at or below the
        inflation radius -- so a mover and a static obstacle are inflated alike.
        """
        w = self._world
        n_rows, n_cols = w.occupancy.shape
        if not len(w.dynamic):
            return np.zeros((steps, n_rows, n_cols), dtype=bool)
        if self._grid_pts is None:
            res = w.config.grid_resolution
            xs = (np.arange(n_cols) + 0.5) * res
            ys = (np.arange(n_rows) + 0.5) * res
            gx, gy = np.meshgrid(xs, ys, indexing="xy")
            self._grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
        pts = self._grid_pts
        out = np.zeros((steps, n_rows * n_cols), dtype=bool)
        for k in range(steps):
            discs = self._predicted_discs(w._t + k * self.dt_plan)
            known = ~np.isnan(discs[:, :2]).any(axis=1)
            if not known.all():
                # A mover the robot has never seen is not on its map at all.
                discs = discs[known]
                if not len(discs):
                    continue
            d = np.linalg.norm(pts[:, None, :] - discs[None, :, :2], axis=-1) - discs[None, :, 2]
            out[k] = (d <= radius).any(axis=1)
        if not self.config.time_varying_movers:
            out[:] = out.any(axis=0)
        elif self.config.temporal_margin_steps > 0:
            out = dilate_in_time(out, self.config.temporal_margin_steps)
        return out.reshape(steps, n_rows, n_cols)

    @property
    def _history_needed(self) -> int:
        """Observations kept. Two is all a straight line can use; the harmonic
        estimator needs a stretch of track to read curvature off."""
        if self.config.predictor not in ("harmonic", "orbit"):
            return 2
        return max(3, int(round(self.config.history_s / self.robot.dt)) + 1)

    def _visible(self, pose: np.ndarray, samples: int = 24) -> np.ndarray:
        """Which movers a sensor at ``pose`` could see: in range, in view, and not
        behind static geometry.

        Occlusion is checked by sampling the segment from the robot to the
        mover's near edge and asking the map whether any sample is inside an
        obstacle -- the same clearance test the planner inflates with, so a wall
        that blocks the robot also blocks its view.
        """
        w = self._world
        cfg = self.config
        discs = w._dyn_now
        if not len(discs):
            return np.zeros(0, dtype=bool)
        origin = np.asarray(pose[:2], dtype=np.float64)
        to_mover = discs[:, :2] - origin
        distance = np.linalg.norm(to_mover, axis=1)
        ok = distance - discs[:, 2] <= cfg.sensor_range_m
        if cfg.sensor_fov_deg < 360.0:
            bearing = np.arctan2(to_mover[:, 1], to_mover[:, 0]) - float(pose[2])
            wrapped = np.abs(np.arctan2(np.sin(bearing), np.cos(bearing)))
            ok &= wrapped <= np.radians(cfg.sensor_fov_deg) / 2.0
        if not ok.any():
            return ok
        # Sample from the robot to the near edge of each disc; a sample inside
        # the map means the sight line is blocked.
        reach = np.maximum(distance - discs[:, 2], 1e-6)
        unit = to_mover / np.maximum(distance, 1e-9)[:, None]
        steps = np.linspace(0.0, 1.0, samples)[None, :, None]
        pts = origin[None, None, :] + unit[:, None, :] * (reach[:, None, None] * steps)
        clear = w.clearance(pts.reshape(-1, 2), include_dynamic=False).reshape(len(discs), samples)
        return ok & (clear > 0.0).all(axis=1)

    def _observe(self, pose: np.ndarray | None = None) -> None:
        """Record where the movers are now -- what the robot's sensors see.

        Movers out of sight are recorded as NaN rather than dropped, so every
        observation keeps one row per mover and each predictor decides for
        itself how to handle a gap in a track.
        """
        w = self._world
        seen = w._dyn_now.copy()
        if self.config.observation_noise_m > 0.0 and len(seen):
            seen[:, :2] += self._noise.normal(0.0, self.config.observation_noise_m,
                                              size=(len(seen), 2))
        if self.config.observe_visible_only and len(seen):
            hidden = ~self._visible(self._pose if pose is None else pose)
            seen[hidden, :2] = np.nan
        self._observations.append((w._t, seen))
        self._orbit_cache = None  # the track moved; the fit is stale
        del self._observations[:-self._history_needed]

    def _harmonic_discs(self, seen, t: float) -> np.ndarray | None:
        """Where a fitted oscillation puts the movers at ``t``, or ``None`` if the
        history cannot support the fit.

        Every mover here runs ``centre + dir * A sin(w t)``, so each obeys
        ``a = -w^2 (p - c)``: linear in ``w^2`` and ``w^2 c``, which is a
        three-unknown least squares over the observed track and needs no
        knowledge of any mover's parameters. The prediction is then the exact
        solution of that equation from the current position and velocity,
        ``c + (p - c) cos(w tau) + (v / w) sin(w tau)``.

        A frozen mover has no curvature to read, so the fit is degenerate and
        this returns ``None`` -- which is what keeps the frozen identity.
        """
        times = np.array([s[0] for s in seen])
        if len(seen) < 5 or len({len(p) for _, p in seen}) != 1:
            return None
        dt = float(np.diff(times).mean())
        if dt <= 0 or not np.allclose(np.diff(times), dt, atol=1e-9):
            return None
        track = np.stack([p[:, :2] for _, p in seen])  # (N, M, 2)
        pos = track[1:-1]
        vel = (track[2:] - track[:-2]) / (2.0 * dt)
        acc = (track[2:] - 2.0 * track[1:-1] + track[:-2]) / (dt * dt)

        out = seen[-1][1].copy()
        # Propagate from the last sample where position *and* velocity are both
        # known at the same instant. A central difference gives the velocity at
        # the middle sample, and pairing it with the newest position instead
        # starts the prediction a step out of step -- worth 0.06 m at two
        # seconds when this was first written.
        tau = t - times[-2]
        for m in range(track.shape[1]):
            # Rows [-p_x, -p_y stacked]: a = -k p + u, unknowns (k, u_x, u_y).
            n = len(pos)
            design = np.zeros((2 * n, 3))
            design[0::2, 0] = -pos[:, m, 0]
            design[1::2, 0] = -pos[:, m, 1]
            design[0::2, 1] = 1.0
            design[1::2, 2] = 1.0
            target = acc[:, m, :].reshape(-1)
            (k, ux, uy), *_ = np.linalg.lstsq(design, target, rcond=None)
            omega = math.sqrt(k) if k > 1e-9 else 0.0
            if omega <= 0.0 or not np.isfinite(omega):
                return None
            centre = np.array([ux, uy]) / k
            p_ref, v_ref = track[-2, m], vel[-1, m]
            out[m, :2] = (centre + (p_ref - centre) * math.cos(omega * tau)
                          + v_ref / omega * math.sin(omega * tau))
        return out

    def _orbit_discs(self, seen, t: float) -> np.ndarray | None:
        """The same oscillation, fitted to the observations without differencing
        them.

        ``p(t) = c + u sin(w t) + v cos(w t)`` is linear in ``c``, ``u`` and
        ``v`` once ``w`` is fixed, so this searches ``w`` over a grid and solves
        the rest by least squares -- a separable fit, in which every observation
        gets an equal vote. The differencing fit reads curvature through a second
        difference instead, which divides observation error by ``dt^2``; Phase 5z
        measured what that costs once observations are noisy.

        The frequency grid is the one piece of prior knowledge here: nothing
        slower than a six-minute period or faster than 2 rad/s is treated as a
        moving obstacle. It is a weaker assumption than the model class itself,
        which both fits already share.

        The fit is cached per observation history: the planner asks for thirty
        horizons per plan, and they all rest on the same track.
        """
        times = np.array([s[0] for s in seen])
        if len(seen) < 8 or len({len(p) for _, p in seen}) != 1:
            return None
        if np.isnan(np.stack([p[:, :2] for _, p in seen])).any():
            return self._orbit_discs_per_mover(seen, t, times)
        # The positions go in the key, not just the timestamp and length: the
        # innovation check asks the same question of an older track, and two
        # tracks of equal length can end at the same instant.
        key = (float(times[-1]), len(seen), float(np.sum(seen[-1][1][:, :2])))
        if self._orbit_cache is None or self._orbit_cache[0] != key:
            track = np.stack([p[:, :2] for _, p in seen])  # (N, M, 2)
            self._orbit_cache = (key, self._search_frequency(times, track))
        omega, _, coef = self._orbit_cache[1]
        out = seen[-1][1].copy()
        basis = np.array([1.0, math.sin(omega * t), math.cos(omega * t)])
        out[:, :2] = (basis @ coef).reshape(-1, 2)
        return out

    @staticmethod
    def _search_frequency(times, track):
        """(frequency, residual, coefficients) minimising the fit's residual.

        Coarse geometric sweep, then one refinement around the winner. The
        coefficients are solved exactly for each candidate, so only the single
        nonlinear parameter is searched.
        """
        flat = track.reshape(len(times), -1)
        best = None
        for omegas in (np.geomspace(0.02, 2.0, 96), None):
            if omegas is None:
                step = best[0] * (2.0 / 0.02) ** (1.0 / 95) - best[0]
                omegas = np.linspace(max(best[0] - step, 1e-3), best[0] + step, 21)
            for omega in omegas:
                design = np.stack([np.ones_like(times), np.sin(omega * times),
                                   np.cos(omega * times)], axis=1)
                coef, *_ = np.linalg.lstsq(design, flat, rcond=None)
                resid = float(((design @ coef - flat) ** 2).sum())
                if best is None or resid < best[1]:
                    best = (float(omega), resid, coef)
        return best

    def _orbit_discs_per_mover(self, seen, t: float, times) -> np.ndarray:
        """The orbit fit when tracks have gaps: each mover fitted on the samples
        it was actually seen in, and left to the straight line if it has too few."""
        out = self._constant_velocity_discs(seen, t)
        for m in range(len(seen[-1][1])):
            visible = [(ts, pos[m]) for ts, pos in seen if not np.isnan(pos[m, :2]).any()]
            if len(visible) < 8:
                continue
            stamps = np.array([ts for ts, _ in visible])
            omega, _, coef = self._search_frequency(
                stamps, np.stack([pos[:2] for _, pos in visible])[:, None, :])
            basis = np.array([1.0, math.sin(omega * t), math.cos(omega * t)])
            out[m, :2] = basis @ coef
        return out

    def _constant_velocity_discs(self, seen, t: float) -> np.ndarray:
        t1, p1 = seen[-1]
        out = p1.copy()
        if len(seen) >= 2:
            t0, p0 = seen[-2]
            if t1 > t0 and len(p0) == len(p1):
                velocity = (p1[:, :2] - p0[:, :2]) / (t1 - t0)
                out[:, :2] = p1[:, :2] + velocity * min(t - t1, self.config.extrapolation_cap_s)
        if not np.isnan(out[:, :2]).any():
            return out
        # A mover out of sight now, or at the previous step, is carried from the
        # last two sightings it did have; one never seen stays NaN and drops out
        # of the planner's grid entirely.
        for m in np.unique(np.argwhere(np.isnan(out[:, :2]))[:, 0]):
            track = [(ts, pos[m]) for ts, pos in seen if not np.isnan(pos[m, :2]).any()]
            if not track:
                continue
            t1, p1 = track[-1]
            out[m] = p1
            if len(track) >= 2:
                t0, p0 = track[-2]
                if t1 > t0:
                    velocity = (p1[:2] - p0[:2]) / (t1 - t0)
                    out[m, :2] = p1[:2] + velocity * min(t - t1,
                                                         self.config.extrapolation_cap_s)
        return out

    def _predicted_discs(self, t: float,
                         observations: list[tuple[float, np.ndarray]] | None = None) -> np.ndarray:
        """Mover discs ``(M, 3)`` the planner believes occupy space at time ``t``.

        ``observations`` defaults to the latest; passing an older pair asks what
        an earlier estimate said. The oracle ignores it.
        """
        w = self._world
        if self.config.predictor == "oracle":
            return w.dynamic_at(t)
        if self.config.predictor not in ("constant_velocity", "harmonic", "orbit"):
            raise ValueError(f"unknown predictor {self.config.predictor!r}")
        seen = self._observations if observations is None else observations
        if self.config.predictor == "harmonic":
            fitted = self._harmonic_discs(seen, t)
            if fitted is not None:
                return fitted
        elif self.config.predictor == "orbit":
            fitted = self._orbit_discs(seen, t)
            if fitted is not None:
                return fitted
        return self._constant_velocity_discs(seen, t)

    def _innovation(self) -> float:
        """Largest distance between a mover as observed now and where the
        estimate behind the last plan attempt put it now. Zero before any plan."""
        w = self._world
        if self._plan_basis is None or not len(w._dyn_now):
            return 0.0
        expected = self._predicted_discs(w._t, observations=self._plan_basis)
        gaps = np.linalg.norm(expected[:, :2] - w._dyn_now[:, :2], axis=1)
        gaps = gaps[~np.isnan(gaps)]  # a mover never seen cannot contradict anything
        return float(gaps.max()) if len(gaps) else 0.0

    def _plan(self, pose: np.ndarray) -> bool:
        """Plan from ``pose`` at the current simulation time. Keeps the old plan
        on failure, as the spatial baseline does."""
        w, cfg = self._world, self.config
        steps = max(1, int(math.ceil(cfg.window_s / self.dt_plan)))
        start = tuple(int(v) for v in w.world_to_grid(pose[:2]))
        goal = tuple(int(v) for v in w.world_to_grid(w.goal))
        self._last_plan_t = w._t
        # The estimate this attempt plans on. Kept on failure too: the question
        # the trigger asks is whether the planner's belief is stale, and a
        # failed attempt refreshed it as much as a successful one did.
        self._plan_basis = list(self._observations)
        self.replans += 1
        floor = w.config.robot_radius + cfg.safety_margin * cfg.mover_margin_floor
        for radius in (w.config.robot_radius + cfg.safety_margin,
                       w.config.robot_radius + cfg.safety_margin * 0.5,
                       w.config.robot_radius):
            if radius <= w.config.robot_radius:
                self.bare_radius_attempts += 1
            static, heuristic = self._static(radius)
            movers = self._mover_occupancy(max(radius, floor), steps)
            # Movers persist past the window where they were at its last step,
            # rather than vanishing -- see spacetime_astar's tail_occ for the
            # exploit a vanishing mover invites.
            path = spacetime_astar(static, movers, start, goal, heuristic,
                                   tail_occ=movers[-1])
            if path is None:
                continue
            cells = np.asarray([(r, c) for r, c, _k in path], dtype=int)
            wp = w.grid_to_world(cells).astype(np.float64)
            wp[0] = pose[:2]
            wp[-1] = w.goal
            self._waypoints = wp
            self._times = w._t + self.dt_plan * np.arange(len(path))
            # Waits in the part of the plan that will actually run before the
            # next scheduled replan. An agent that never plans one is a spatial
            # planner with extra steps, and needs to be seen to be one.
            executed = int(math.ceil(cfg.replan_every_s / self.dt_plan))
            head = path[:executed + 1]
            self.planned_waits += sum(
                (a[0], a[1]) == (b[0], b[1]) for a, b in zip(head, head[1:], strict=False)
            )
            return True
        self.plan_failures += 1
        return False

    # ------------------------------------------------------------------
    def _scheduled(self, t: float) -> np.ndarray:
        """Where the plan puts the robot at time ``t``, interpolated."""
        times, wp = self._times, self._waypoints
        if t <= times[0]:
            return wp[0]
        if t >= times[-1]:
            return wp[-1]
        i = int(np.searchsorted(times, t) - 1)
        a = (t - times[i]) / (times[i + 1] - times[i])
        return wp[i] + a * (wp[i + 1] - wp[i])

    def act(self, pose: np.ndarray) -> np.ndarray:
        assert self._world is not None, "start_episode() must be called first"
        cfg, w = self.config, self._world
        pose = np.asarray(pose, dtype=np.float64)
        position = pose[:2]
        self._pose = pose
        self._observe(pose)

        if self._waypoints is not None:
            drift = float(np.linalg.norm(self._scheduled(w._t) - position))
            if w._t - self._last_plan_t >= cfg.replan_every_s or drift > cfg.replan_deviation:
                self._plan(pose)
            elif cfg.replan_innovation_m > 0 and self._innovation() > cfg.replan_innovation_m:
                self.innovation_replans += 1
                self._plan(pose)
        else:
            self._plan(pose)
        if self._waypoints is None:
            return np.array([self._speed_to_action(0.0), 0.0], dtype=np.float32)

        target = self._scheduled(w._t + cfg.track_lead_s)
        to_target = target - position
        dist = float(np.linalg.norm(to_target))
        if dist < cfg.station_radius:
            # On station: the schedule says be here. Hold still, do not spin
            # toward a point the robot already occupies.
            return np.array([self._speed_to_action(0.0), 0.0], dtype=np.float32)

        heading_error = float(wrap_angle(math.atan2(to_target[1], to_target[0]) - pose[2]))
        omega = float(np.clip(cfg.heading_gain * heading_error / self.robot.max_angular_vel,
                              -1.0, 1.0))
        if abs(heading_error) > cfg.turn_in_place_threshold:
            speed = 0.0
        else:
            # Speed that arrives at the scheduled point on time, never faster.
            speed = min(dist / cfg.track_lead_s / self.robot.max_linear_vel, 1.0)
            speed *= max(math.cos(heading_error), 0.0)
            goal_dist = float(np.linalg.norm(w.goal - position))
            speed *= min(goal_dist / cfg.slowdown_radius, 1.0)
            speed *= self._caution_scale(position)
        return np.array([self._speed_to_action(speed), omega], dtype=np.float32)

    # --- identical in form to AStarPursuitAgent -----------------------------
    def _caution_scale(self, position: np.ndarray) -> float:
        cfg = self.config
        margin = float(self._world.clearance(position)) - self._world.config.robot_radius
        if margin >= cfg.caution_clearance:
            return 1.0
        ratio = max(margin, 0.0) / cfg.caution_clearance
        return cfg.min_speed_scale + (1.0 - cfg.min_speed_scale) * ratio

    def _speed_to_action(self, speed_fraction: float) -> float:
        r = self.robot
        v = float(np.clip(speed_fraction, 0.0, 1.0)) * r.max_linear_vel
        span = r.max_linear_vel - r.min_linear_vel
        return float(np.clip(2.0 * (v - r.min_linear_vel) / span - 1.0, -1.0, 1.0))
