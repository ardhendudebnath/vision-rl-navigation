"""The privileged PPO policy, driving a robot whose sensor it was not trained on.

The policy (``runs/ppo_privileged``) observes 37 numbers: 32 ranges at fixed
bearings around the full circle, divided by 6 m, then the goal's distance over
20 m, its bearing as cosine and sine, and its own speed and turn rate over the
training robot's limits (0.6 m/s, 1.6 rad/s). A TurtleBot3 offers 360 beams
reaching 3.5 m from 0.064 m behind the axle, at 5 Hz. This builds the policy's
observation from that, the way a deployment would:

- the scan is moved to the robot's centre (:func:`tb3_agent.to_centre`) and the
  return nearest each of the policy's 32 bearings is taken;
- a beam with no return reads the training sensor's maximum, 6 m, because that
  is what "nothing there" meant to the policy -- ``no_return`` overrides this,
  so reading it as the LDS-01's own 3.5 m can be measured too;
- between scans the policy sees the last one, as it would on the robot;
- speed and turn rate are normalised by the training robot's limits, and the
  policy's action is turned into a command by those same limits -- the drive
  then clips it to what the TurtleBot3 can do.

The goal vector is computed from the true pose, as in training: this is the
privileged policy, and the question is what the sensor and the body change.
Kept free of ROS and Gazebo imports.
"""

from __future__ import annotations

import numpy as np
from tb3_agent import clean_ranges, to_centre

from vision_nav.envs.robot import RobotConfig, wrap_angle

#: The training robot and sensor (runs/ppo_privileged/config.yaml).
POLICY_ROBOT = RobotConfig()
POLICY_BEAMS = 32
POLICY_MAX_RANGE = 6.0
POLICY_MAX_GOAL = 20.0
#: Lidar2D's full-circle bearings for 32 beams, relative to the heading.
POLICY_BEARINGS = np.linspace(-np.pi, np.pi, POLICY_BEAMS, endpoint=False)


def resample(ranges: np.ndarray, bearings: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """The range of the beam nearest each target bearing (all relative to the
    heading, radians)."""
    b = np.asarray(bearings, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    gap = np.abs(wrap_angle(t[:, None] - b[None, :]))
    return np.asarray(ranges, dtype=np.float64)[np.argmin(gap, axis=1)]


class PolicyDriver:
    """Builds the policy's observation from a robot's scan and runs the policy."""

    def __init__(self, model, sensor_max_range: float, bearings: np.ndarray,
                 lidar_x: float = 0.0, no_return: float = POLICY_MAX_RANGE,
                 range_min: float = 0.12) -> None:
        self.model = model
        self.sensor_max_range = float(sensor_max_range)
        self.bearings = np.asarray(bearings, dtype=np.float64)
        self.lidar_x = float(lidar_x)
        self.no_return = float(no_return)
        self.range_min = float(range_min)
        self.scan32: np.ndarray | None = None
        self.scans_used = 0

    def feed(self, ranges: np.ndarray) -> None:
        """A new scan from the robot's lidar."""
        r = clean_ranges(ranges, self.sensor_max_range, self.range_min)
        r, b = to_centre(r, self.bearings, self.lidar_x, self.sensor_max_range)
        s = resample(r, b, POLICY_BEARINGS)
        empty = s >= self.sensor_max_range - 1e-6
        s = np.where(empty, self.no_return, s)
        self.scan32 = np.clip(s, 0.0, POLICY_MAX_RANGE)
        self.scans_used += 1

    def observation(self, pose: np.ndarray, goal: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        assert self.scan32 is not None, "feed a scan first"
        to_goal = np.asarray(goal, dtype=np.float64) - np.asarray(pose[:2], dtype=np.float64)
        dist = float(np.linalg.norm(to_goal))
        bearing = float(wrap_angle(np.arctan2(to_goal[1], to_goal[0]) - pose[2]))
        vec = np.clip(np.array([min(dist / POLICY_MAX_GOAL, 1.0), np.cos(bearing), np.sin(bearing),
                                velocity[0] / POLICY_ROBOT.max_linear_vel,
                                velocity[1] / POLICY_ROBOT.max_angular_vel]), -1.0, 1.0)
        obs = np.concatenate([self.scan32 / POLICY_MAX_RANGE, vec])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def act(self, pose: np.ndarray, goal: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(self.observation(pose, goal, velocity), deterministic=True)
        return np.asarray(action, dtype=np.float32)
