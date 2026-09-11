"""Classical navigation baseline: A* global plan + pure-pursuit tracking.

This is the *control condition* for the whole study.  It is deliberately
given privileges the learned policies never get — the full obstacle map and
the exact robot pose — because the interesting question is not "can RL beat a
crippled planner" but "what does a learned policy buy you over a strong
classical stack, and where does each one break".

Structurally it mirrors Nav2: a global planner over an inflated costmap, then
a local controller tracking that plan.  When the project moves onto ROS 2,
this class is replaced by Nav2 itself and the comparison protocol is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_nav.envs.robot import RobotConfig, wrap_angle
from vision_nav.planning.grid_astar import astar_grid
from vision_nav.planning.smoothing import densify_path, simplify_path

__all__ = ["PursuitConfig", "AStarPursuitAgent"]


@dataclass
class PursuitConfig:
    """Tuning for the global planner and the pure-pursuit controller."""

    #: Extra inflation beyond the robot radius, in metres.  This is the
    #: tracking margin the controller is allowed to consume while cornering.
    safety_margin: float = 0.18
    #: Pure-pursuit lookahead, in metres.  Larger is smoother but cuts
    #: corners harder; this is the main collision/smoothness trade-off.
    lookahead: float = 0.45
    #: Spacing used to resample the plan for arc-length lookahead search.
    track_spacing: float = 0.05
    #: Proportional gain from heading error to angular velocity.
    heading_gain: float = 2.5
    #: Heading error (rad) beyond which the robot turns in place.
    turn_in_place_threshold: float = 0.9
    #: Distance to goal at which the controller starts decelerating, in m.
    slowdown_radius: float = 0.8
    #: Forward clearance (m) below which speed is scaled down reactively.
    caution_clearance: float = 0.45
    #: Floor on the reactive speed scale, so the robot never fully stalls.
    min_speed_scale: float = 0.25


class AStarPursuitAgent:
    """Map-based planner with a pure-pursuit local controller.

    Usage mirrors a policy: :meth:`reset` once per episode, then :meth:`act`
    per step.
    """

    def __init__(
        self,
        config: PursuitConfig | None = None,
        robot: RobotConfig | None = None,
    ) -> None:
        self.config = config or PursuitConfig()
        #: Must match the env's robot config: the controller emits normalised
        #: actions, so it has to know the scale the env will apply to them.
        self.robot = robot or RobotConfig()
        #: String-pulled plan, for plotting and path-length reporting.
        self.path: np.ndarray | None = None
        #: Uniformly resampled plan, for lookahead search.
        self._track: np.ndarray | None = None
        self._world = None
        self._cursor = 0

    # ------------------------------------------------------------------
    def reset(self, world, pose: np.ndarray) -> bool:
        """Plan for a new episode. Returns ``False`` if no plan was found."""
        self._world = world
        self._cursor = 0
        self.path = None
        self._track = None

        margin = self.config.safety_margin
        # Retry with progressively less inflation: in tight scenes the
        # preferred margin can make the goal unreachable, and refusing to
        # plan at all would score as a failure that the planner could in fact
        # have avoided.
        for radius in (
            world.config.robot_radius + margin,
            world.config.robot_radius + margin * 0.5,
            world.config.robot_radius,
        ):
            occ = world.occupancy_at(radius)
            start = tuple(int(v) for v in world.world_to_grid(pose[:2]))
            goal = tuple(int(v) for v in world.world_to_grid(world.goal))
            if occ[start] or occ[goal]:
                continue
            cells = astar_grid(occ, start, goal)
            if cells is None:
                continue

            raw = world.grid_to_world(np.asarray(cells, dtype=int))
            # Snap endpoints to the true start/goal, not their cell centres.
            raw[0] = pose[:2]
            raw[-1] = world.goal
            self.path = simplify_path(world, raw, radius)
            self._track = densify_path(self.path, self.config.track_spacing)
            return True

        return False

    # ------------------------------------------------------------------
    def act(self, pose: np.ndarray) -> np.ndarray:
        """Return a normalised ``[-1, 1]^2`` action for the current pose."""
        cfg = self.config
        assert self._world is not None, "reset() must be called before act()"
        if self._track is None:
            return np.zeros(2, dtype=np.float32)

        position = np.asarray(pose[:2], dtype=np.float64)
        target = self._lookahead_point(position)

        to_target = target - position
        heading_error = float(wrap_angle(np.arctan2(to_target[1], to_target[0]) - pose[2]))

        omega = np.clip(
            cfg.heading_gain * heading_error / self.robot.max_angular_vel, -1.0, 1.0
        )

        if abs(heading_error) > cfg.turn_in_place_threshold:
            # Too far off-axis for useful forward progress: rotate first.
            speed = 0.0
        else:
            speed = float(np.cos(heading_error))
            goal_dist = float(np.linalg.norm(self._world.goal - position))
            speed *= min(goal_dist / cfg.slowdown_radius, 1.0)
            speed *= self._caution_scale(position)

        # The env maps action[0] = -1 to min_linear_vel, not to zero, so a
        # commanded speed of 0 must be expressed in that asymmetric range.
        return np.array([self._speed_to_action(speed), omega], dtype=np.float32)

    # ------------------------------------------------------------------
    def _caution_scale(self, position: np.ndarray) -> float:
        """Scale speed down when the robot is close to an obstacle.

        The global plan already routes around obstacles, but the controller
        tracks it imperfectly.  Slowing near obstacles shortens the distance
        travelled per steering correction, which is what turns a near-miss
        into a clean pass.
        """
        cfg = self.config
        margin = float(self._world.clearance(position)) - self._world.config.robot_radius
        if margin >= cfg.caution_clearance:
            return 1.0
        ratio = max(margin, 0.0) / cfg.caution_clearance
        return cfg.min_speed_scale + (1.0 - cfg.min_speed_scale) * ratio

    def _lookahead_point(self, position: np.ndarray) -> np.ndarray:
        """Point on the plan roughly ``lookahead`` metres ahead of the robot.

        The cursor only ever moves forward, which prevents the controller from
        latching onto an earlier part of the path when the route doubles back
        on itself.
        """
        assert self._track is not None
        track = self._track

        # Re-anchor to the closest track point at or after the cursor.
        window = track[self._cursor :]
        nearest = int(np.argmin(np.linalg.norm(window - position, axis=1)))
        self._cursor += nearest

        # Walk forward until the lookahead distance is exceeded.
        idx = self._cursor
        while (
            idx < len(track) - 1
            and np.linalg.norm(track[idx] - position) < self.config.lookahead
        ):
            idx += 1
        return track[idx]

    def _speed_to_action(self, speed_fraction: float) -> float:
        """Map a desired forward-speed fraction in ``[0, 1]`` to ``action[0]``."""
        r = self.robot
        v = float(np.clip(speed_fraction, 0.0, 1.0)) * r.max_linear_vel
        span = r.max_linear_vel - r.min_linear_vel
        return float(np.clip(2.0 * (v - r.min_linear_vel) / span - 1.0, -1.0, 1.0))
