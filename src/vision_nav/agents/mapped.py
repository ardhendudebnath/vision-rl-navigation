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
        # Replanning is how a mapping planner lives; everything else is the
        # baseline's own configuration.
        super().__init__(dataclasses.replace(config or PursuitConfig(), replan_on_block=True),
                         robot)
        self.sensor_kind = sensor
        self.noise_std = noise_std
        self.map: OccupancyMap | None = None
        self._plan_radius = 0.0
        self._checked_version = -1

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
        goal = tuple(int(v) for v in self.map.world_to_grid(self.map.goal))
        for radius in (base + margin, base + margin * 0.5, base):
            occ = self.map.occupancy_at(radius)
            if occ[start] or occ[goal]:
                continue
            cells = astar_grid(occ, start, goal)
            if cells is None:
                continue
            raw = self.map.grid_to_world(np.asarray(cells, dtype=int))
            raw[0] = pose[:2]
            raw[-1] = self.map.goal
            # The smoother is handed the map, not the world: its line-of-sight
            # check only ever calls ``clearance``.
            self.path = simplify_path(self.map, raw, radius)
            self._track = densify_path(self.path, self.config.track_spacing)
            self._plan_radius = radius
            return True
        return False

    def act(self, pose: np.ndarray) -> np.ndarray:
        assert self.map is not None, "start_episode() must be called first"
        self.map.integrate(np.asarray(pose, dtype=np.float64))
        return super().act(pose)

    # --- the three remaining reads of the world, sent to the map ----------
    def _path_ahead_blocked(self, position: np.ndarray) -> bool:
        """Does anything mapped since the last check cut across the rest of the
        plan, closer than the clearance the plan was made with?"""
        if self._track is None or self.map.version == self._checked_version:
            return False
        self._checked_version = self.map.version
        ahead = self._track[self._cursor:]
        if not len(ahead):
            return False
        return bool(np.any(self.map.clearance(ahead) < self._plan_radius - 1e-9))

    def _caution_scale(self, position: np.ndarray) -> float:
        cfg = self.config
        margin = float(self.map.clearance(position)) - self.map.robot_radius
        if margin >= cfg.caution_clearance:
            return 1.0
        ratio = max(margin, 0.0) / cfg.caution_clearance
        return cfg.min_speed_scale + (1.0 - cfg.min_speed_scale) * ratio
