"""The classical baseline, without its map and without its pose.

:class:`~vision_nav.agents.mapped.MappedPursuitAgent` took away the map and said
what it kept: "It still reads its exact pose. Pose estimation is a separate
privilege, and taking both at once would leave a result nobody could attribute."
This is that second privilege, taken away on top of the first.

The robot's pose now comes from its own wheel encoders, integrated forward and
drifting, optionally dragged back onto the map by matching each scan against it
(:mod:`vision_nav.mapping.localisation`). Everything downstream reads the
estimate: the map is built at it, the plan is searched from it, the controller
steers from it, and the goal is a point in the frame the robot started in, not
in the world. The only thing the true pose is still used for is taking the scan
-- the sensor is bolted to the robot, not to its belief -- and scoring the
estimate afterwards, which is measurement and never reaches the control path.

Success is still judged by the environment on where the robot physically is. A
robot that believes it arrived and did not, fails.
"""

from __future__ import annotations

import numpy as np

from vision_nav.agents.classical import PursuitConfig
from vision_nav.agents.mapped import MappedPursuitAgent
from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.mapping.localisation import (
    DeadReckoning,
    OdometryConfig,
    ScanMatchConfig,
    ScanMatcher,
)

__all__ = ["LocalisedPursuitAgent"]


class LocalisedPursuitAgent(MappedPursuitAgent):
    """A* and pure pursuit on a map the robot builds, at a pose it estimates.

    With ``odometry=None`` the encoders are perfect and the estimate is the true
    pose, exactly: the integrator is the robot's own and nothing perturbs it.
    That is the identity control for this experiment -- the arm must reproduce
    :class:`MappedPursuitAgent` episode for episode, and ``test_localisation``
    requires it.
    """

    def __init__(self, config: PursuitConfig | None = None, robot: RobotConfig | None = None,
                 sensor: str = "lidar32", noise_std: float = 0.0,
                 odometry: OdometryConfig | None = None, scan_matching: bool = False,
                 match_config: ScanMatchConfig | None = None) -> None:
        super().__init__(config, robot, sensor, noise_std)
        self.odometry = odometry
        self.matcher = ScanMatcher(match_config) if scan_matching else None
        self._odom: DeadReckoning | None = None
        #: Per-step distance and heading between the estimate and the truth.
        #: Diagnostic: computed after the control action has been decided.
        self.pose_errors: list[float] = []
        self.heading_errors: list[float] = []

    # ------------------------------------------------------------------
    def start_episode(self, world, pose: np.ndarray) -> bool:
        pose = np.asarray(pose, dtype=np.float64)
        # Its own stream, seeded from the world, so an episode drifts the same
        # way every time it is run and never shares draws with the sensor.
        self._odom = DeadReckoning(self.robot, self.odometry,
                                   np.random.default_rng((int(world.seed), 7)))
        self._odom.reset(pose)
        if self.matcher is not None:
            self.matcher.reset()
        self.pose_errors, self.heading_errors = [], []
        # The robot starts where it starts: a real stack defines the map frame
        # by the pose it began at, so the first scan and the first plan are the
        # one place estimate and truth agree by construction.
        return super().start_episode(world, pose)

    def act(self, pose: np.ndarray, velocity: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """``pose`` is the true pose and ``velocity`` the encoder reading.

        The true pose reaches the sensor and the diagnostics. It does not reach
        the map, the planner or the controller.
        """
        assert self.map is not None and self._odom is not None, "start_episode() first"
        self._steps += 1
        truth = np.asarray(pose, dtype=np.float64)

        estimate = self._odom.update(velocity)
        ranges = self.map.scan(truth)
        if self.matcher is not None:
            estimate = self.matcher.correct(estimate, ranges, self.map)
            self._odom.pose = estimate.copy()
        self.map.integrate(estimate, ranges=ranges)

        self.pose_errors.append(float(np.linalg.norm(estimate[:2] - truth[:2])))
        self.heading_errors.append(abs(float(wrap_angle(estimate[2] - truth[2]))))
        return self._drive(estimate)

    # ------------------------------------------------------------------
    @property
    def pose(self) -> np.ndarray:
        """Where the robot believes it is."""
        assert self._odom is not None
        return self._odom.pose.copy()

    @property
    def pose_error(self) -> float:
        """How wrong that belief was when the episode ended, in metres."""
        return self.pose_errors[-1] if self.pose_errors else 0.0
