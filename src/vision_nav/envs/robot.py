"""Differential-drive robot kinematics.

Deliberately kinematic rather than dynamic: the policy commands linear and
angular velocity, which is exactly the ``geometry_msgs/Twist`` interface
Nav2 and every ROS 2 mobile base expose.  Keeping the action space identical
to the real interface is what makes the eventual sim-to-real bridge a matter
of swapping the transport, not rewriting the policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RobotConfig", "DiffDriveRobot"]


@dataclass
class RobotConfig:
    """Kinematic limits of the simulated base (TurtleBot-class defaults)."""

    max_linear_vel: float = 0.6  # m/s
    min_linear_vel: float = -0.1  # m/s, slight reverse allowed
    max_angular_vel: float = 1.6  # rad/s

    max_linear_accel: float = 2.0  # m/s^2
    max_angular_accel: float = 6.0  # rad/s^2

    dt: float = 0.1  # s, control period (10 Hz, typical for Nav2)


class DiffDriveRobot:
    """Unicycle-model base with first-order velocity rate limits."""

    def __init__(self, config: RobotConfig | None = None) -> None:
        self.config = config or RobotConfig()
        self.pose = np.zeros(3, dtype=np.float64)  # x, y, theta
        self.velocity = np.zeros(2, dtype=np.float64)  # v, omega

    def reset(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float64).copy()
        self.velocity[:] = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.pose[:2]

    @property
    def heading(self) -> float:
        return float(self.pose[2])

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map a normalised ``[-1, 1]^2`` action to ``(v, omega)`` targets."""
        cfg = self.config
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Map [-1, 1] onto the asymmetric linear-velocity range.
        v = cfg.min_linear_vel + (a[0] + 1.0) * 0.5 * (cfg.max_linear_vel - cfg.min_linear_vel)
        omega = a[1] * cfg.max_angular_vel
        return np.array([v, omega])

    def unscale_action(self, velocity: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`scale_action`: ``(v, omega)`` back to ``[-1, 1]^2``.

        Needed by controllers that natively speak SI velocities — the Nav2
        bridge publishes ``geometry_msgs/Twist`` and has to enter the same
        normalised action space every other actor drives, rather than the env
        growing a second control interface for one caller. Kept next to
        ``scale_action`` so the two cannot drift; ``test_robot`` round-trips
        them.
        """
        cfg = self.config
        v, omega = np.asarray(velocity, dtype=np.float64)
        span = cfg.max_linear_vel - cfg.min_linear_vel
        v = np.clip(v, cfg.min_linear_vel, cfg.max_linear_vel)
        a_v = 2.0 * (v - cfg.min_linear_vel) / span - 1.0
        a_w = np.clip(omega / cfg.max_angular_vel, -1.0, 1.0)
        return np.array([a_v, a_w])

    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one control period. Returns the new pose ``(x, y, theta)``."""
        cfg = self.config
        target = self.scale_action(action)

        # Rate-limit towards the commanded velocity so the policy cannot
        # request physically impossible instantaneous changes.
        dv_max = cfg.max_linear_accel * cfg.dt
        dw_max = cfg.max_angular_accel * cfg.dt
        self.velocity[0] += np.clip(target[0] - self.velocity[0], -dv_max, dv_max)
        self.velocity[1] += np.clip(target[1] - self.velocity[1], -dw_max, dw_max)

        v, omega = self.velocity
        theta = self.pose[2]

        # Exact unicycle integration over the step (arc, not Euler chord).
        if abs(omega) < 1e-6:
            self.pose[0] += v * np.cos(theta) * cfg.dt
            self.pose[1] += v * np.sin(theta) * cfg.dt
        else:
            theta_new = theta + omega * cfg.dt
            radius = v / omega
            self.pose[0] += radius * (np.sin(theta_new) - np.sin(theta))
            self.pose[1] -= radius * (np.cos(theta_new) - np.cos(theta))
            self.pose[2] = theta_new

        if abs(omega) < 1e-6:
            self.pose[2] = theta

        self.pose[2] = wrap_angle(self.pose[2])
        return self.pose.copy()


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (or array of angles) to ``(-pi, pi]``."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
