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
from vision_nav.agents.mapped import MappedPursuitAgent, make_sensor
from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.mapping.frontend import CorrelativeConfig, CorrelativeFrontEnd
from vision_nav.mapping.localisation import (
    DeadReckoning,
    OdometryConfig,
    ScanMatchConfig,
    ScanMatcher,
)
from vision_nav.mapping.occupancy import OccupancyMap
from vision_nav.mapping.posegraph import (
    PoseGraph,
    PoseGraphConfig,
    compose_pose,
    relative_pose,
    scan_points,
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
                 match_config: ScanMatchConfig | None = None,
                 corroborate: bool = False, commit: bool = False,
                 frontier: bool = False, relax_on_stall: int = 0,
                 pose_graph: bool = False,
                 graph_config: PoseGraphConfig | None = None,
                 noise_margin: float = 0.0, front_end: str = "map",
                 front_end_config: CorrelativeConfig | None = None) -> None:
        super().__init__(config, robot, sensor, noise_std, corroborate, commit,
                         frontier, relax_on_stall, noise_margin)
        if front_end not in ("map", "correlative"):
            raise ValueError(f"front_end must be 'map' or 'correlative', not {front_end!r}")
        self.odometry = odometry
        #: Which scan matcher localises the robot, when ``scan_matching`` is on.
        #: ``"map"`` is :class:`ScanMatcher`, every scan against the whole map
        #: in a narrow window -- every published result. ``"correlative"`` is
        #: :class:`CorrelativeFrontEnd`, keyframes against a buffer of recent
        #: keyframes in slam_toolbox's wide window, and it replaces the first
        #: rather than running beside it: two matchers correcting one pose
        #: would make neither's contribution attributable.
        self.front_end = front_end
        self.matcher = (ScanMatcher(match_config)
                        if scan_matching and front_end == "map" else None)
        self.correlative = (CorrelativeFrontEnd(front_end_config)
                            if scan_matching and front_end == "correlative" else None)
        #: The SLAM back end (:mod:`vision_nav.mapping.posegraph`). Off by
        #: default: every published result runs the front end alone, which is
        #: what §9.5's 0.1 m and §9.11's error budget were measured on.
        self.graph = PoseGraph(graph_config) if pose_graph else None
        #: Every scan of the episode, as ``(keyframe, pose relative to it,
        #: ranges)``. Kept only with a back end, and only so the map can be
        #: rebuilt from a corrected trajectory rather than patched.
        self._history: list[tuple[int, np.ndarray, np.ndarray]] = []
        self._start_pose: np.ndarray | None = None
        #: Diagnostic: how many times the map was rebuilt.
        self.map_rebuilds = 0
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
        if self.correlative is not None:
            self.correlative.reset()
        if self.graph is not None:
            self.graph.reset()
        self._history = []
        self._start_pose = pose.copy()
        self.map_rebuilds = 0
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
        elif self.correlative is not None:
            estimate = self.correlative.update(estimate, ranges, self.map.sensor)
            self._odom.pose = estimate.copy()
        if self.graph is not None:
            estimate = self._update_graph(estimate, ranges)
        self.map.integrate(estimate, ranges=ranges)
        if self.graph is not None and self.graph.poses:
            # Held against the keyframe it was taken near, so an optimisation
            # that moves that keyframe moves this scan with it.
            k = len(self.graph.poses) - 1
            self._history.append((k, relative_pose(self.graph.poses[k], estimate),
                                  np.asarray(ranges).copy()))

        self.pose_errors.append(float(np.linalg.norm(estimate[:2] - truth[:2])))
        self.heading_errors.append(abs(float(wrap_angle(estimate[2] - truth[2]))))
        return self._drive(estimate)

    # ------------------------------------------------------------------
    def _update_graph(self, estimate: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        """Keyframe, close a loop, optimise, and carry the estimate across it."""
        graph = self.graph
        assert graph is not None and self.map is not None
        if not graph.due(estimate):
            return estimate
        index = graph.add_keyframe(estimate, scan_points(ranges, self.map.sensor))
        if index >= graph.config.loop_min_gap:
            graph.find_closure(index)
        if not graph.due_to_optimise:
            return estimate
        # The current estimate is the newest keyframe's pose, so it rides on
        # that keyframe's correction: take it relative before, put it back after.
        before = graph.poses[-1].copy()
        rel = relative_pose(before, estimate)
        moved = graph.optimise()
        estimate = compose_pose(graph.poses[-1], rel)
        self._odom.pose = estimate.copy()
        if (moved > graph.config.rebuild_threshold
                and graph.rebuilds < graph.config.max_rebuilds):
            self._rebuild_map()
            graph.rebuilds += 1
        return estimate

    def _rebuild_map(self) -> None:
        """Re-integrate every stored scan at its corrected pose, into a fresh map.

        The alternative -- move the robot's estimate and keep the map the old
        poses built -- puts the two in disagreement, and the front end then pulls
        the pose straight back to the stale map on the next scan. That is how a
        back end ends up looking inert while fighting itself.

        The rng is seeded exactly as :meth:`MappedPursuitAgent.start_episode`
        seeds it, and the one scan taken there is the only one re-sampled rather
        than replayed, so it reproduces bit for bit.
        """
        assert self.graph is not None and self._world is not None
        fresh = OccupancyMap(self._world, make_sensor(self.sensor_kind, self.noise_std),
                             rng=np.random.default_rng(int(self._world.seed)),
                             corroborate=self.corroborate,
                             noise_margin=self.noise_margin)
        fresh.integrate(self._start_pose)
        for k, rel, ranges in self._history:
            fresh.integrate(compose_pose(self.graph.poses[k], rel), ranges=ranges)
        self.map = fresh
        if self.matcher is not None:
            self.matcher.invalidate()
        self._checked_version = self.map.version
        # The plan in force was searched on the map that has just been replaced.
        self._track = None
        self.map_rebuilds += 1

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
