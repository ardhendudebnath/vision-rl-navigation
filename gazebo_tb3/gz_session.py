"""A Gazebo Harmonic server for one world, stepped in lockstep from Python.

The server runs headless and paused. Every control period the caller sets a
velocity command and asks for exactly one period of simulation; this waits
until simulated time has reached the end of it, so a run is reproducible and
never limited by how fast Python is. Sensor and odometry messages are kept as
they arrive; the robot's true pose comes from the scene broadcaster and is for
scoring only.

Needs the ``ros_tb3`` RoboStack environment (``gz.transport13``, ``gz.msgs10``).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.odometry_pb2 import Odometry
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node


def yaw_of(q) -> float:
    """Heading from a gz quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _control_worker(service: str) -> None:
    """Issues world-control requests on behalf of the session, from its own
    process, reading ``pause steps timeout_ms`` lines on stdin and answering
    ``1`` or ``0``.

    gz-transport's Python ``request`` blocks while holding the interpreter
    lock. In a process that also subscribes to topics, the subscription
    callbacks then wait for that lock while the reply to the request waits
    behind them, and the request times out: a one-step request returns before
    any message arrives, a fifty-step one deadlocks. A process with no
    subscriptions has nothing to deadlock with.
    """
    import sys

    node = Node()
    for line in sys.stdin:
        pause, steps, timeout_ms = (int(v) for v in line.split())
        req = WorldControl()
        req.pause = bool(pause)
        if steps:
            req.multi_step = steps
        ok, _ = node.request(service, req, WorldControl, Boolean, timeout_ms)
        sys.stdout.write("1\n" if ok else "0\n")
        sys.stdout.flush()


class GzSession:
    """One Gazebo server running one world, driven step by step."""

    def __init__(self, sdf_text: str, world_name: str = "vision_nav", robot_name: str = "turtlebot3",
                 step_size: float = 0.002, resource_path: str | None = None,
                 headless: bool = True, verbose: int = 1, sync: bool = False) -> None:
        """``sync=True`` makes every :meth:`step` wait for the pose and the
        odometry stamped at the step's end, which the robot publishes only if
        it was built with ``report_hz`` at the physics rate
        (:func:`robot_sdf.robot_model`)."""
        self.world_name, self.robot_name, self.step_size = world_name, robot_name, step_size
        self.sync = sync
        self._dir = tempfile.mkdtemp(prefix="gz_world_")
        self.sdf_path = Path(self._dir) / "world.sdf"
        self.sdf_path.write_text(sdf_text, encoding="utf-8")
        env = dict(os.environ)
        if resource_path:
            env["GZ_SIM_RESOURCE_PATH"] = resource_path + os.pathsep + env.get("GZ_SIM_RESOURCE_PATH", "")
        cmd = ["gz", "sim", "-s", f"-v{verbose}", str(self.sdf_path)]
        if headless:
            cmd.insert(3, "--headless-rendering")
        self.log_path = Path(self._dir) / "gz.log"
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

        self.node = Node()
        self._lock = threading.Lock()
        self.scan: LaserScan | None = None
        self.scan_count = 0
        self.odom: Odometry | None = None
        self.true_pose: tuple[float, float, float] | None = None
        self.sim_time = 0.0
        #: Simulation time each latest message was stamped with, so a step can
        #: wait for the messages that belong to it rather than read whatever
        #: arrived last -- which is what makes a run repeatable.
        self.scan_stamp = self.odom_stamp = self.pose_stamp = -1.0
        self.node.subscribe(LaserScan, "/scan", self._on_scan)
        self.node.subscribe(Odometry, "/odom", self._on_odom)
        self.node.subscribe(Pose_V, f"/world/{world_name}/pose/info", self._on_poses)
        self.node.subscribe(Pose, f"/model/{robot_name}/pose", self._on_model_pose)
        self.node.subscribe(Clock, f"/world/{world_name}/clock", self._on_clock)
        self.cmd = self.node.advertise("/cmd_vel", Twist)
        self._control = f"/world/{world_name}/control"
        self._worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--control-worker", self._control],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def _request(self, pause: bool, steps: int, timeout_ms: int) -> bool:
        self._worker.stdin.write(f"{int(pause)} {int(steps)} {int(timeout_ms)}\n")
        self._worker.stdin.flush()
        return self._worker.stdout.readline().strip() == "1"

    # --- message callbacks ------------------------------------------------
    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9

    def _on_scan(self, msg: LaserScan) -> None:
        with self._lock:
            self.scan = msg
            self.scan_count += 1
            self.scan_stamp = self._stamp(msg)

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            self.odom = msg
            self.odom_stamp = self._stamp(msg)

    def _set_pose(self, p, stamp: float) -> None:
        # Two streams carry the pose; keep whichever reading is newest.
        with self._lock:
            if stamp >= self.pose_stamp:
                self.true_pose = (p.position.x, p.position.y, yaw_of(p.orientation))
                self.pose_stamp = stamp

    def _on_poses(self, msg: Pose_V) -> None:
        for p in msg.pose:
            if p.name == self.robot_name:
                self._set_pose(p, self._stamp(msg))
                return

    def _on_model_pose(self, msg: Pose) -> None:
        self._set_pose(msg, self._stamp(msg))

    def _on_clock(self, msg: Clock) -> None:
        with self._lock:
            self.sim_time = msg.sim.sec + msg.sim.nsec * 1e-9

    # --- control ----------------------------------------------------------
    def wait_ready(self, timeout: float = 60.0) -> None:
        """Until the server answers on its control service."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"gz sim exited ({self.proc.returncode}); see {self.log_path}")
            if self._request(True, 0, 1000):
                return
            time.sleep(0.5)
        raise TimeoutError(f"gz sim did not come up in {timeout} s; see {self.log_path}")

    def command(self, v: float, w: float, settle: float = 0.005) -> None:
        """Publish a velocity command and give it ``settle`` seconds of wall
        time to reach the drive before the next step is requested. Gazebo is
        paused in between, so the wait costs nothing in simulated time; it
        stops a command racing the step request through a different process."""
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd.publish(msg)
        if settle:
            time.sleep(settle)

    def wait_for(self, attribute: str, stamp: float, timeout: float = 10.0) -> bool:
        """Until the latest ``attribute`` (``scan``, ``odom`` or ``pose``)
        message is stamped at or after ``stamp``. False on timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with self._lock:
                if getattr(self, f"{attribute}_stamp") >= stamp - 1e-6:
                    return True
            time.sleep(0.0005)
        return False

    def step(self, seconds: float = 0.1, timeout: float = 180.0) -> None:
        """Advance exactly ``seconds`` of simulated time, then return.

        One request, never retried: a retried request that had in fact been
        accepted queues its steps on top of the first, and the run is no longer
        the one asked for. The wait is long because the first steps that need a
        lidar scan wait for the renderer to start, which takes tens of seconds
        the first time under WSL.
        """
        n = max(1, round(seconds / self.step_size))
        with self._lock:
            target = self.sim_time + n * self.step_size - 0.5 * self.step_size
        if not self._request(True, n, 10000):
            if self.proc.poll() is not None:
                raise RuntimeError(f"gz sim exited ({self.proc.returncode}); see {self.log_path}")
            raise RuntimeError(f"world control request failed; see {self.log_path}")
        start = time.monotonic()
        while True:
            with self._lock:
                if self.sim_time >= target:
                    break
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"simulation did not reach {target:.3f} s")
            time.sleep(0.001)
        if self.sync:
            # The pose and odometry of this step's end, not of the one before.
            for attribute in ("pose", "odom"):
                if not self.wait_for(attribute, target):
                    raise TimeoutError(f"no {attribute} message stamped at {target:.3f} s")

    def close(self) -> None:
        try:
            self._worker.stdin.close()
            self._worker.wait(timeout=5)
        except Exception:  # noqa: BLE001 -- the worker may already be gone
            self._worker.kill()
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001 -- best effort, then force
                os.killpg(self.proc.pid, 9)
        self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--control-worker":
        _control_worker(sys.argv[2])
