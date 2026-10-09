"""The hand-written stack, driving a TurtleBot3 from scans it is handed.

The published stack (:class:`LocalisedPursuitAgent`) senses by ray-casting the
2-D world itself. On a robot the scans arrive from a driver instead: at the
lidar's own rate, from the lidar's own position on the body. This adapter is
that driver interface and nothing more, so the same planner, controller, map
and scan matcher run unchanged on either source:

- the map's sensor is an :class:`ExternalScanner`, which returns the scan it
  was last handed and never touches the world;
- a scan is integrated only on the control steps a new one arrives (the
  LDS-01 runs at 5 Hz, the controller at 10 Hz); between scans the pose comes
  from odometry alone, as on the real robot;
- each scan is moved from the lidar's position to the robot's centre before
  use -- the static base-to-laser transform every ROS stack applies -- because
  the stack assumes its sensor sits where it steers from.

Both arms of the transfer test drive this class: the Gazebo TurtleBot3 hands it
Gazebo's scans and wheel odometry, the 2-D arm hands it the analytic lidar's
scans at the same bearings and rate. Kept free of ROS and Gazebo imports.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.envs.sensors import Lidar2D, LidarConfig
from vision_nav.mapping.localisation import OdometryConfig

#: A TurtleBot3 Waffle Pi's published limits (ROBOTIS), with the acceleration
#: limits of Nav2's Gazebo model, so the 2-D arm and Gazebo drive the same
#: envelope. Reverse is kept as the published stack has it.
TB3_ROBOT = RobotConfig(max_linear_vel=0.26, min_linear_vel=-0.1, max_angular_vel=1.82,
                        max_linear_accel=1.0, max_angular_accel=2.0, dt=0.1)
#: The LDS-01 as the Gazebo model has it: 360 samples from 0 to 6.28 rad.
TB3_BEARINGS = np.linspace(0.0, 6.28, 360)
TB3_LIDAR = LidarConfig(n_beams=360, fov=2.0 * np.pi, max_range=3.5, noise_std=0.01)
TB3_RANGE_MIN = 0.12
#: Where the lidar sits on the body: 0.064 m behind the axle.
TB3_LIDAR_X = -0.064


class ExternalScanner:
    """Stands in for :class:`Lidar2D` on the agent's map.

    The map and the scan matcher read ``config.max_range`` and ``_angles``
    from their sensor on every call, and take ranges from ``scan``; this hands
    them the last scan it was fed, at that scan's bearings.
    """

    def __init__(self, config: LidarConfig, bearings: np.ndarray) -> None:
        self.config = config
        self._angles = np.asarray(bearings, dtype=np.float64)
        self._ranges: np.ndarray | None = None

    def feed(self, ranges: np.ndarray, bearings: np.ndarray) -> None:
        self._ranges = np.asarray(ranges, dtype=np.float64)
        self._angles = np.asarray(bearings, dtype=np.float64)

    def scan(self, world, pose, rng=None) -> np.ndarray:  # noqa: ARG002 -- the world is never read
        if self._ranges is None:
            raise RuntimeError("no scan has been fed to the external scanner")
        return self._ranges


def clean_ranges(ranges: np.ndarray, max_range: float, range_min: float = TB3_RANGE_MIN) -> np.ndarray:
    """A driver's ranges in the stack's convention: anything that is not a
    return inside ``[range_min, max_range)`` reads exactly ``max_range``."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    bad = ~np.isfinite(r) | (r < range_min) | (r >= max_range)
    r[bad] = max_range
    return r


def to_centre(ranges: np.ndarray, bearings: np.ndarray, offset_x: float,
              max_range: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-express a scan taken ``offset_x`` metres ahead of the robot's centre
    (negative: behind) as seen from the centre.

    Each return's endpoint keeps its place in the world; only the origin moves.
    Beams with no return stay at exactly ``max_range`` -- they are free space
    out to the sensor's reach, not surfaces -- on their own bearing.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = np.asarray(bearings, dtype=np.float64)
    if offset_x == 0.0:
        return r.copy(), b.copy()
    hit = r < max_range - 1e-9
    ex = offset_x + r * np.cos(b)
    ey = r * np.sin(b)
    out_r = np.where(hit, np.hypot(ex, ey), max_range)
    out_b = np.where(hit, np.arctan2(ey, ex), b)
    # Moving the origin cannot be allowed to turn a return into "no return".
    out_r = np.where(hit, np.minimum(out_r, max_range - 1e-6), out_r)
    return out_r, wrap_angle(out_b)


class TB3Agent(LocalisedPursuitAgent):
    """The published localised stack on scans and odometry from outside.

    ``odometry`` is the noise model applied to the velocity reading: ``None``
    integrates the reading as given -- right for Gazebo, whose wheel odometry
    already carries physical slip -- and ``OdometryConfig()`` is the published
    stack's own model, right for the 2-D arm, whose velocity is the truth.
    """

    def __init__(self, robot: RobotConfig = TB3_ROBOT, lidar: LidarConfig = TB3_LIDAR,
                 bearings: np.ndarray = TB3_BEARINGS, lidar_x: float = TB3_LIDAR_X,
                 odometry: OdometryConfig | None = None) -> None:
        super().__init__(robot=robot, sensor="lidar360", noise_std=lidar.noise_std,
                         odometry=odometry, scan_matching=True, corroborate=True)
        self.lidar = lidar
        self.bearings = np.asarray(bearings, dtype=np.float64)
        self.lidar_x = float(lidar_x)
        self.scans_used = 0
        self.sensor_factory = lambda: ExternalScanner(self.lidar, self.bearings)

    def _prepare(self, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = clean_ranges(ranges, self.lidar.max_range)
        return to_centre(r, self.bearings, self.lidar_x, self.lidar.max_range)

    def start_episode_with_scan(self, world, pose: np.ndarray, ranges: np.ndarray) -> bool:
        """Begin at ``pose`` with the first scan. The world is handed over for
        its extent and goal only; the map is built from ``ranges``."""
        r, b = self._prepare(ranges)
        self._first = (r, b)
        factory = self.sensor_factory

        def primed():
            scanner = factory()
            scanner.feed(r, b)
            return scanner

        self.sensor_factory = primed
        try:
            started = self.start_episode(world, pose)
        finally:
            self.sensor_factory = factory
        self.scans_used = 1
        return started

    def step(self, velocity: np.ndarray, ranges: np.ndarray | None,
             truth: np.ndarray | None = None) -> np.ndarray:
        """One control period: odometry always, a scan when one has arrived.

        ``truth`` is for the pose-error diagnostic only; it reaches nothing
        that decides the action.
        """
        assert self.map is not None and self._odom is not None, "start an episode first"
        assert self.correlative is None and self.graph is None
        self._steps += 1
        estimate = self._odom.update(np.asarray(velocity, dtype=np.float64))
        if ranges is not None:
            r, b = self._prepare(ranges)
            self.map.sensor.feed(r, b)
            if self.matcher is not None:
                estimate = self.matcher.correct(estimate, r, self.map)
                self._odom.pose = estimate.copy()
            self.map.integrate(estimate, ranges=r)
            self.scans_used += 1
        if truth is not None:
            t = np.asarray(truth, dtype=np.float64)
            self.pose_errors.append(float(np.linalg.norm(estimate[:2] - t[:2])))
            self.heading_errors.append(abs(float(wrap_angle(estimate[2] - t[2]))))
        return self._drive(estimate)


def analytic_lidar(lidar: LidarConfig = TB3_LIDAR, bearings: np.ndarray = TB3_BEARINGS) -> Lidar2D:
    """The 2-D arm's sensor: the analytic lidar at the TurtleBot3's bearings,
    range and noise, for scans cast at the lidar's own position."""
    sensor = Lidar2D(replace(lidar, n_beams=len(bearings)))
    sensor._angles = np.asarray(bearings, dtype=np.float64)
    return sensor
