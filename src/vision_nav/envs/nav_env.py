"""Goal-conditioned procedural navigation environment (Gymnasium API).

This is the shared task definition that every stage of the project is
evaluated on:

===========================  ===========================================
Stage                        How it uses this env
===========================  ===========================================
1. Classical baseline        ``A*`` global plan + pure-pursuit controller
                             driving the same action space.
2. Privileged RL             ``obs_mode="privileged"`` — exact pose, goal
                             vector and ground-truth range readings.
3. Vision-conditioned RL      ``obs_mode="depth"`` (Phase 4) — rendered
                             egocentric observations only.
4. Robustness suite          Held-out ``world_seeds`` plus sensor noise,
                             dropout and obstacle-density shifts.
===========================  ===========================================

Keeping all four on one task definition is the point: it is what makes the
comparison in the final report an apples-to-apples one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from vision_nav.envs.randomization import DomainRandomization
from vision_nav.envs.rgb_camera import RGBCamera, RGBCameraConfig
from vision_nav.envs.robot import DiffDriveRobot, RobotConfig, wrap_angle
from vision_nav.envs.sensors import CameraConfig, DepthCamera, Lidar2D, LidarConfig
from vision_nav.envs.world import World, WorldConfig, generate_world
from vision_nav.planning.grid_astar import geodesic_distance_field, shortest_path_length

__all__ = ["RewardConfig", "NavEnvConfig", "ProceduralNavEnv"]


@dataclass
class RewardConfig:
    """Reward-term weights.

    Defaults are tuned so that a successful, reasonably direct episode earns
    roughly ``goal_bonus`` in total and a collision is clearly worse than a
    timeout — otherwise the agent learns to end episodes early by crashing.
    """

    #: Per-metre reward for reducing distance-to-goal.
    progress: float = 3.0
    #: One-off bonus on reaching the goal.
    goal_bonus: float = 20.0
    #: One-off penalty on collision (applied as a negative reward).
    collision_penalty: float = 20.0
    #: Constant per-step cost; drives the agent to finish quickly.
    step_penalty: float = 0.01
    #: Penalty ramped in as clearance drops below ``proximity_threshold``.
    proximity_penalty: float = 0.15
    proximity_threshold: float = 0.4
    #: Penalty on angular-velocity magnitude; discourages spin-in-place.
    spin_penalty: float = 0.005

    #: Shape progress on geodesic (A*) distance rather than Euclidean.
    use_geodesic_progress: bool = True


@dataclass
class NavEnvConfig:
    """Full environment specification."""

    world: WorldConfig = field(default_factory=WorldConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    #: Per-episode world-config sampling. Deterministic in the world seed, so
    #: evaluation replay and the world cache both stay valid.
    domain_randomization: DomainRandomization = field(default_factory=DomainRandomization)

    max_episode_steps: int = 500

    #: ``"privileged"`` — 360-degree lidar, exact goal vector, velocities.
    #: ``"depth"`` — forward-facing depth camera instead of the lidar. The
    #: substantive change is losing the rear view, which is the constraint a
    #: real camera-based robot has.
    #: ``"rgb"`` renders the same geometry as ``"depth"`` at the same FOV and
    #: column count, so the difference isolates the cost of the CNN encoder.
    obs_mode: str = "privileged"
    camera: CameraConfig = field(default_factory=CameraConfig)
    rgb_camera: RGBCameraConfig = field(default_factory=RGBCameraConfig)

    #: Explicit pool of world seeds to draw episodes from.  Passing an
    #: explicit list is how train / val / test splits are kept disjoint; see
    #: :mod:`vision_nav.envs.splits`.
    world_seeds: Sequence[int] | None = None
    #: Used only when ``world_seeds`` is ``None``.
    seed_range: tuple[int, int] = (0, 512)
    #: Iterate the seed pool in order instead of sampling it.  Set for
    #: evaluation so every policy sees an identical episode sequence.
    deterministic_seed_order: bool = False

    #: Distance beyond which goal-distance observations saturate, in metres.
    max_goal_distance: float = 20.0
    #: Worlds kept in memory; generation (A* + distance field) is not free.
    world_cache_size: int = 1024


class ProceduralNavEnv(gym.Env):
    """Drive a differential-drive base to a goal in a procedural scene."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        config: NavEnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NavEnvConfig()
        if self.config.obs_mode not in ("privileged", "depth", "rgb"):
            raise NotImplementedError(
                f"obs_mode={self.config.obs_mode!r} is not implemented. "
                "Available: 'privileged' (360-degree lidar), 'depth' "
                "(forward-facing depth camera), 'rgb' (egocentric colour)."
            )
        self.render_mode = render_mode

        self.robot = DiffDriveRobot(self.config.robot)
        self.lidar = Lidar2D(self.config.lidar)
        self.camera = DepthCamera(self.config.camera)
        self.rgb = RGBCamera(self.config.rgb_camera)

        #: Number of range/depth values in the observation, whichever sensor
        #: this mode uses.
        self._n_range = (
            self.lidar.n_beams if self.config.obs_mode == "privileged" else self.camera.width
        )

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        if self.config.obs_mode == "rgb":
            # Dict rather than a flattened image: the goal vector must reach
            # the policy without being pushed through a convolutional stack
            # that has no reason to preserve it.
            self.observation_space = spaces.Dict(
                {
                    "image": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3, *self.rgb.shape[:2]),  # (C, H, W)
                        dtype=np.uint8,
                    ),
                    "vector": spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self._n_range + 5,),
                dtype=np.float32,
            )

        self._world: World | None = None
        self._world_cache: dict[int, tuple[World, np.ndarray | None, float]] = {}
        self._seed_cursor = 0

        self._steps = 0
        self._path_length = 0.0
        self._prev_progress_dist = 0.0
        self._shortest_path = 0.0
        self._distance_field: np.ndarray | None = None
        self._last_action = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        world_seed = (options or {}).get("world_seed")
        if world_seed is None:
            world_seed = self._next_world_seed()
        world, dist_field, l_star = self._get_world(int(world_seed))

        self._world = world
        self._distance_field = dist_field
        self._shortest_path = l_star

        self.robot.reset(world.start)
        self._steps = 0
        self._path_length = 0.0
        self._last_action[:] = 0.0
        self._prev_progress_dist = self._progress_distance(self.robot.position)

        return self._observation(), self._info(terminated=False, truncated=False)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._world is not None, "reset() must be called before step()"
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)

        prev_pos = self.robot.position.copy()
        self.robot.step(action)
        new_pos = self.robot.position
        self._last_action = action
        self._steps += 1
        self._path_length += float(np.linalg.norm(new_pos - prev_pos))

        collided = self._check_collision(prev_pos, new_pos)
        goal_dist = float(np.linalg.norm(new_pos - self._world.goal))
        succeeded = (not collided) and goal_dist <= self._world.config.goal_tolerance

        reward = self._reward(collided=collided, succeeded=succeeded)

        terminated = bool(collided or succeeded)
        truncated = bool((not terminated) and self._steps >= self.config.max_episode_steps)

        info = self._info(terminated=terminated, truncated=truncated)
        info["collision"] = collided
        info["is_success"] = succeeded

        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        from vision_nav.viz.topdown import render_topdown

        assert self._world is not None, "reset() must be called before render()"
        return render_topdown(self._world, self.robot.pose)

    # ------------------------------------------------------------------
    # World management
    # ------------------------------------------------------------------
    def _next_world_seed(self) -> int:
        cfg = self.config
        pool = cfg.world_seeds
        if pool is None:
            if cfg.deterministic_seed_order:
                lo, hi = cfg.seed_range
                seed = lo + (self._seed_cursor % max(hi - lo, 1))
                self._seed_cursor += 1
                return seed
            return int(self.np_random.integers(*cfg.seed_range))
        pool = list(pool)
        if cfg.deterministic_seed_order:
            seed = pool[self._seed_cursor % len(pool)]
            self._seed_cursor += 1
            return int(seed)
        return int(pool[self.np_random.integers(len(pool))])

    def _get_world(self, seed: int) -> tuple[World, np.ndarray | None, float]:
        cached = self._world_cache.get(seed)
        if cached is not None:
            return cached

        world_config = self.config.domain_randomization.sample(self.config.world, seed)
        world = generate_world(seed, world_config)
        l_star = shortest_path_length(world, world.start[:2], world.goal)
        if l_star is None:  # generate_world guarantees reachability
            raise RuntimeError(f"world seed={seed} is unreachable after generation")

        dist_field = None
        if self.config.reward.use_geodesic_progress:
            goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
            dist_field = geodesic_distance_field(world.occupancy, goal_cell)
            dist_field = dist_field * world.config.grid_resolution

        entry = (world, dist_field, float(l_star))
        if len(self._world_cache) >= self.config.world_cache_size:
            self._world_cache.pop(next(iter(self._world_cache)))
        self._world_cache[seed] = entry
        return entry

    # ------------------------------------------------------------------
    # Observation / reward / termination
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        assert self._world is not None
        cfg = self.config
        pose = self.robot.pose

        to_goal_vec = self._goal_vector(pose)
        if cfg.obs_mode == "rgb":
            return {
                "image": self.rgb.render_chw(self._world, pose),
                "vector": to_goal_vec.astype(np.float32),
            }

        if cfg.obs_mode == "privileged":
            scan = self.lidar.normalized_scan(self._world, pose, self.np_random)
        else:
            scan = self.camera.normalized_depth(self._world, pose, self.np_random)

        obs = np.concatenate([scan, to_goal_vec])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _goal_vector(self, pose: np.ndarray) -> np.ndarray:
        """Goal distance, bearing as (cos, sin), and current velocities.

        Identical across every observation mode, so the sensor is the only
        thing that differs between them.
        """
        assert self._world is not None
        cfg = self.config
        to_goal = self._world.goal - pose[:2]
        goal_dist = float(np.linalg.norm(to_goal))
        bearing = wrap_angle(np.arctan2(to_goal[1], to_goal[0]) - pose[2])
        return np.clip(
            np.array(
                [
                    min(goal_dist / cfg.max_goal_distance, 1.0),
                    np.cos(bearing),
                    np.sin(bearing),
                    self.robot.velocity[0] / cfg.robot.max_linear_vel,
                    self.robot.velocity[1] / cfg.robot.max_angular_vel,
                ]
            ),
            -1.0,
            1.0,
        )

    def _progress_distance(self, position: np.ndarray) -> float:
        """Distance-to-goal used for progress shaping (geodesic or Euclidean)."""
        assert self._world is not None
        if self._distance_field is not None:
            row, col = self._world.world_to_grid(position)
            d = float(self._distance_field[row, col])
            if np.isfinite(d):
                return d
            # The robot is on an inflated cell (grazing an obstacle); fall
            # back to Euclidean so the reward stays finite.
        return float(np.linalg.norm(position - self._world.goal))

    def _reward(self, *, collided: bool, succeeded: bool) -> float:
        assert self._world is not None
        rc = self.config.reward

        dist = self._progress_distance(self.robot.position)
        progress = self._prev_progress_dist - dist
        self._prev_progress_dist = dist

        reward = rc.progress * progress - rc.step_penalty
        reward -= rc.spin_penalty * abs(self.robot.velocity[1])

        if rc.proximity_penalty > 0.0:
            clearance = float(self._world.clearance(self.robot.position))
            margin = clearance - self._world.config.robot_radius
            if margin < rc.proximity_threshold:
                deficit = (rc.proximity_threshold - max(margin, 0.0)) / rc.proximity_threshold
                reward -= rc.proximity_penalty * deficit

        if succeeded:
            reward += rc.goal_bonus
        if collided:
            reward -= rc.collision_penalty

        return float(reward)

    def _check_collision(self, prev_pos: np.ndarray, new_pos: np.ndarray) -> bool:
        """Collision test sampled along the step, not only at its end."""
        assert self._world is not None
        samples = np.stack([0.5 * (prev_pos + new_pos), new_pos])
        return bool(np.any(~self._world.is_free(samples)))

    def _info(self, *, terminated: bool, truncated: bool) -> dict[str, Any]:
        assert self._world is not None
        goal_dist = float(np.linalg.norm(self.robot.position - self._world.goal))
        return {
            "world_seed": self._world.seed,
            "steps": self._steps,
            "goal_distance": goal_dist,
            "path_length": self._path_length,
            "shortest_path_length": self._shortest_path,
            "is_success": False,
            "collision": False,
            "terminated": terminated,
            "truncated": truncated,
        }

    # ------------------------------------------------------------------
    @property
    def world(self) -> World | None:
        """The scene for the current episode (``None`` before first reset)."""
        return self._world
