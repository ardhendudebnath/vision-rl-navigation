"""The classical baseline, without its map.

:class:`AStarPursuitAgent` reads the true static world in four places: to plan
(``occupancy_at``), to shorten the plan (``simplify_path`` checks line of sight
with ``clearance``), to decide the plan is blocked, and to slow near obstacles
(``clearance`` again). This subclass sends all four to an
:class:`~vision_nav.mapping.OccupancyMap` it builds from its own sensor as it
drives, and changes nothing else -- controller, gains, margins and fallbacks are
inherited, so a difference between the two agents is the map.

It still reads its exact pose. Pose estimation is a separate privilege, and
taking both at once would leave a result nobody could attribute.

Planning into unexplored space follows the usual convention: unknown cells are
free, and the agent replans whenever something it has just mapped cuts across
the rest of its plan.

Two behaviours a mapping planner needs and the full-map planner never did,
because on a static world it never replans. Both are standard costmap practice,
and both were added after Phase 6d's diagnostic found every collision following
a replan that had failed:

- **The robot's footprint is cleared** from the planning grid before searching.
  It is standing there, so those cells cannot hold an obstacle at its scale,
  and without clearing them a robot that has drifted inside its own map's
  inflation can find no route out at any radius.
- **A plan the map rules out is never driven.** The baseline keeps its previous
  plan when a replan fails, which is harmless when replans only ever answer
  moving obstacles. Here the previous plan is the one the map has just shown to
  run into something, so the robot stops and tries again on the next scan.
- **The goal is relaxed to the nearest reachable cell** when the map's inflation
  covers it. The goal is a point in free space that the map, inflating
  conservatively around a wall beside it, can swallow -- and refusing to plan at
  all would strand the robot short of a goal it could reach.
- **A failed plan is answered by turning on the spot.** Stopping alone leaves a
  robot that cannot plan waiting for a static world to change, which it never
  does: the first two repairs turned Phase 6d's collisions into timeouts.
  Rotating brings new bearings into the scan and lets mistaken evidence decay,
  which is the cheapest of the recovery behaviours a real stack carries.
- **Replanning is rate limited.** Something newly mapped close ahead is answered
  at once; anything further is answered at about 1 Hz, which is what a
  production stack does. Replanning at sensor rate had the robot changing its
  mind every third step -- a median of 137 replans an episode, 17 m driven on an
  8.6 m journey, and a timeout at the end of it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from vision_nav.agents.classical import AStarPursuitAgent, PursuitConfig
from vision_nav.envs.robot import RobotConfig
from vision_nav.envs.sensors import Lidar2D, LidarConfig
from vision_nav.mapping.occupancy import OccupancyMap
from vision_nav.planning.grid_astar import astar_grid
from vision_nav.planning.smoothing import densify_path, simplify_path

__all__ = ["MappedPursuitAgent", "make_sensor"]


def make_sensor(kind: str, noise_std: float = 0.0) -> Lidar2D:
    """The sensor the map is built from.

    ``"lidar32"`` is the 32-beam, 360-degree, 6 m scanner the privileged PPO
    policy reads, so the mapped planner and the policy see the same returns.
    ``"camera64"`` is the depth camera's geometry -- 64 columns over 90 degrees,
    sampled at column centres exactly as :class:`DepthCamera` does.
    """
    if kind == "lidar32":
        return Lidar2D(LidarConfig(n_beams=32, fov=2.0 * np.pi, max_range=6.0,
                                   noise_std=noise_std))
    if kind == "camera64":
        fov = np.pi / 2
        sensor = Lidar2D(LidarConfig(n_beams=64, fov=fov, max_range=6.0, noise_std=noise_std))
        sensor._angles = -fov / 2.0 + (np.arange(64) + 0.5) * (fov / 64)
        return sensor
    raise ValueError(f"unknown sensor {kind!r}; expected 'lidar32' or 'camera64'")


class MappedPursuitAgent(AStarPursuitAgent):
    """A* and pure pursuit on a map the robot builds itself."""

    def __init__(self, config: PursuitConfig | None = None, robot: RobotConfig | None = None,
                 sensor: str = "lidar32", noise_std: float = 0.0) -> None:
        # Replanning is done here, in :meth:`act`, rather than by the
        # baseline's block trigger, because a failed replan must stop the robot
        # instead of restoring the old plan. Everything else is the baseline's
        # own configuration.
        super().__init__(dataclasses.replace(config or PursuitConfig(), replan_on_block=False,
                                             replan_every=0), robot)
        self.sensor_kind = sensor
        self.noise_std = noise_std
        self.map: OccupancyMap | None = None
        self._plan_radius = 0.0
        self._checked_version = -1
        #: Steps between replans for anything that is not close ahead: 1 Hz at
        #: the env's 10 Hz control rate.
        self.replan_period = 10
        #: How far along the plan counts as close ahead, in metres.
        self.urgent_distance = 1.0
        self._last_replan = 0
        #: A replan the rate limit has deferred, and whether it cannot wait.
        self._pending, self._urgent = False, False
        #: Steps at which a replan found no route at any radius, and the old
        #: plan was kept. Diagnostic.
        self.failed_plan_steps: list[int] = []
        #: Steps spent turning on the spot because no route could be found.
        self.recovery_steps = 0
        self._steps = 0

    # ------------------------------------------------------------------
    def start_episode(self, world, pose: np.ndarray) -> bool:
        self.replans = 0
        self.churn_total = 0.0
        self.churn_max = 0.0
        # Seeded from the world, so a sensor with noise corrupts an episode the
        # same way every time it is run.
        self.map = OccupancyMap(world, make_sensor(self.sensor_kind, self.noise_std),
                                rng=np.random.default_rng(int(world.seed)))
        self.map.integrate(pose)
        self._checked_version = self.map.version
        self.failed_plan_steps = []
        self.recovery_steps = 0
        self._steps = 0
        self._last_replan = 0
        self._pending, self._urgent = False, False
        return self.reset(world, pose)

    def reset(self, world, pose: np.ndarray, include_dynamic: bool = False,
              predict: bool | None = None) -> bool:
        """Plan on the map. The same three-radius fallback as the baseline."""
        assert self.map is not None, "start_episode() must be called first"
        self._world = world
        self._cursor = 0
        self._since_replan = 0
        self.path = None
        self._track = None
        margin = self.config.safety_margin
        base = self.map.robot_radius
        start = tuple(int(v) for v in self.map.world_to_grid(pose[:2]))
        for radius in (base + margin, base + margin * 0.5, base):
            occ = self._clear_footprint(self.map.occupancy_at(radius), pose)
            goal = self._reachable_goal(occ)
            if goal is None or occ[start]:
                continue
            cells = astar_grid(occ, start, goal)
            if cells is None:
                continue
            raw = self.map.grid_to_world(np.asarray(cells, dtype=int))
            raw[0] = pose[:2]
            if goal == tuple(int(v) for v in self.map.world_to_grid(self.map.goal)):
                raw[-1] = self.map.goal
            # The smoother is handed the map, not the world: its line-of-sight
            # check only ever calls ``clearance``.
            self.path = simplify_path(self.map, raw, radius)
            self._track = densify_path(self.path, self.config.track_spacing)
            self._plan_radius = radius
            self._pending, self._urgent = False, False
            return True
        self.failed_plan_steps.append(self._steps)
        return False

    def _reachable_goal(self, occ: np.ndarray) -> tuple[int, int] | None:
        """The goal cell, or the nearest free cell to it within the goal
        tolerance if the map's inflation has swallowed the goal itself."""
        goal = tuple(int(v) for v in self.map.world_to_grid(self.map.goal))
        if not occ[goal]:
            return goal
        reach = float(self.map.config.goal_tolerance)
        k = int(np.ceil(reach / self.map.resolution))
        rows = np.arange(max(goal[0] - k, 0), min(goal[0] + k + 1, occ.shape[0]))
        cols = np.arange(max(goal[1] - k, 0), min(goal[1] + k + 1, occ.shape[1]))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        centres = self.map.grid_to_world(np.stack([rr.ravel(), cc.ravel()], axis=-1))
        d = np.linalg.norm(centres - self.map.goal, axis=1)
        free = (~occ[rr.ravel(), cc.ravel()]) & (d <= reach)
        if not free.any():
            return None
        best = int(np.argmin(np.where(free, d, np.inf)))
        return int(rr.ravel()[best]), int(cc.ravel()[best])

    def _clear_footprint(self, occ: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """The planning grid with the robot's own footprint marked free."""
        res = self.map.resolution
        r = self.map.robot_radius
        row, col = (int(v) for v in self.map.world_to_grid(pose[:2]))
        k = int(np.ceil(r / res)) + 1
        rows = np.arange(max(row - k, 0), min(row + k + 1, occ.shape[0]))
        cols = np.arange(max(col - k, 0), min(col + k + 1, occ.shape[1]))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        centres = np.stack([(cc + 0.5) * res, (rr + 0.5) * res], axis=-1)
        inside = np.linalg.norm(centres - np.asarray(pose[:2]), axis=-1) <= r
        if not occ[rr[inside], cc[inside]].any():
            return occ
        out = occ.copy()
        out[rr[inside], cc[inside]] = False
        return out

    def act(self, pose: np.ndarray) -> np.ndarray:
        assert self.map is not None, "start_episode() must be called first"
        self._steps += 1
        pose = np.asarray(pose, dtype=np.float64)
        self.map.integrate(pose)
        return self._drive(pose)

    def _drive(self, pose: np.ndarray) -> np.ndarray:
        """Plan if the map has invalidated the plan, then track it.

        Split from :meth:`act` for the subclass that has to sense and localise
        before it knows which pose to drive from; everything below is shared.
        """
        if self._track is None:
            # Stopped after a failed plan: try again on what the new scan adds.
            self.reset(self._world, pose)
        elif self._path_ahead_blocked(pose[:2]):
            self.replans += 1
            self._last_replan = self._steps
            self.reset(self._world, pose)  # leaves no plan if it finds none
        if self._track is None:
            # No route: turn on the spot rather than wait. New bearings enter the
            # scan and mistaken evidence decays, which is what lets the next
            # attempt succeed. The forward term is an explicit stop -- the
            # baseline's own no-plan action is all zeros, which this robot reads
            # as 0.25 m/s forward, its speed range being asymmetric.
            self.recovery_steps += 1
            return np.array([self._speed_to_action(0.0), 0.5], dtype=np.float32)
        return super().act(pose)

    # --- the three remaining reads of the world, sent to the map ----------
    def _path_ahead_blocked(self, position: np.ndarray) -> bool:
        """Is the plan worth rebuilding now?

        Anything newly mapped within :attr:`urgent_distance` along the plan is
        worth it at once. Anything further back is worth it at 1 Hz: rebuilding
        on every scan makes the robot change its mind faster than it can drive.
        """
        if self._track is None:
            return False
        if self.map.version != self._checked_version:
            self._checked_version = self.map.version
        elif not self._pending:
            return False  # nothing new, and nothing outstanding
        ahead = self._track[self._cursor:]
        if not len(ahead):
            return False
        invalid = self.map.clearance(ahead) < self._plan_radius - 1e-9
        near = max(1, int(self.urgent_distance / self.config.track_spacing))
        # Remembered, not consumed: deferring must not lose the fact that the
        # plan is broken, or anything further than the near horizon is never
        # answered at all.
        self._urgent = self._urgent or bool(invalid[:near].any())
        self._pending = self._pending or bool(invalid.any())
        if self._urgent:
            return True
        return self._pending and self._steps - self._last_replan >= self.replan_period

    def _caution_scale(self, position: np.ndarray) -> float:
        cfg = self.config
        margin = float(self.map.clearance(position)) - self.map.robot_radius
        if margin >= cfg.caution_clearance:
            return 1.0
        ratio = max(margin, 0.0) / cfg.caution_clearance
        return cfg.min_speed_scale + (1.0 - cfg.min_speed_scale) * ratio
