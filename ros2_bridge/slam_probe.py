"""slam_toolbox on its own, driven in a straight line, against the truth.

A development tool for the SLAM arm (report §9.4). The full evaluation takes
minutes per episode and has Nav2 in the loop, so when slam_toolbox's estimate
went wrong there was no quick way to tell whether the fault was in SLAM, in its
configuration, or in the messages the bridge feeds it. This removes Nav2: the
bridge publishes exactly what it publishes in the evaluation, the robot is
driven at a fixed velocity, and the only question is how far slam_toolbox
thinks it went.

    python ros2_bridge/slam_probe.py --condition large --seed 10000

Needs slam_toolbox running alone: nav2_launch.py with slam:=True and a
nav2_delay long enough that navigation never starts.
"""

from __future__ import annotations

import argparse
import threading

import numpy as np
import rclpy
from nav2_bridge import Nav2Bridge
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import SingleThreadedExecutor
from run_nav2_eval import LATCHED, Pace, believed_pose
from tf2_ros import Buffer, TransformListener

from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.training.env_factory import build_env_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="large")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--speed", type=float, default=0.5, help="m/s, straight ahead")
    p.add_argument("--turn", type=float, default=0.0, help="rad/s")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--beams", type=int, default=360)
    p.add_argument("--rtf", type=float, default=5.0)
    args = p.parse_args(argv)

    rclpy.init()
    _, shift, noise = BENCHMARK_CONDITIONS[args.condition]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split="val", shift=shift, n_worlds=1)
    bridge = Nav2Bridge(cfg, privileges="slam", beams=args.beams)
    bridge.pace = Pace(bridge.dt, args.rtf)
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    tf_buffer = Buffer()
    side = rclpy.create_node("slam_probe_observer")
    TransformListener(tf_buffer, side)
    maps = {"n": 0}
    side.create_subscription(OccupancyGrid, "/map",
                             lambda _m: maps.__setitem__("n", maps["n"] + 1), LATCHED)
    side_exec = SingleThreadedExecutor()
    side_exec.add_node(side)
    threading.Thread(target=side_exec.spin, daemon=True).start()

    bridge.start_episode(args.seed)
    start = bridge.env.robot.pose.copy()
    # Face into open space: pick the heading with the longest clear run, so a
    # straight drive does not end in a wall before it has told us anything.
    best, best_run = start[2], -1.0
    for heading in np.linspace(-np.pi, np.pi, 16, endpoint=False):
        pts = start[:2] + np.outer(np.linspace(0.2, 6.0, 30),
                                   [np.cos(heading), np.sin(heading)])
        clear = bridge.env.world.clearance(pts, include_dynamic=False)
        run = float(np.argmax(clear < 0.4) if (clear < 0.4).any() else 30)
        if run > best_run:
            best, best_run = heading, run
    bridge.env.robot.pose[2] = best
    bridge._odom.reset(bridge.env.robot.pose)

    for _ in range(400):  # until slam has a map and a correction
        bridge.pace.wait()
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.01)
        bridge._sim_time += bridge.dt
        if maps["n"] and tf_buffer.can_transform("map", "base_link", rclpy.time.Time()):
            break
    print(f"start {bridge.env.robot.pose[:2].round(2)} heading {best:.2f}; "
          f"slam ready after {maps['n']} maps", flush=True)

    bridge._cmd[:] = (args.speed, args.turn)
    origin = bridge.env.robot.pose[:2].copy()
    for step in range(1, args.steps + 1):
        bridge.pace.wait()
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.01)
        if not bridge.step_sim():
            print("episode ended (collision or goal)", flush=True)
            break
        if step % 20 == 0:
            truth = bridge.env.robot.pose[:2]
            slam = believed_pose(tf_buffer)
            moved = float(np.linalg.norm(truth - origin))
            slam_moved = float(np.linalg.norm(slam - origin)) if slam is not None else float("nan")
            print(f"  step {step:3d} true moved {moved:5.2f} m  slam moved {slam_moved:5.2f} m  "
                  f"error {float(np.linalg.norm(truth - slam)) if slam is not None else float('nan'):.2f} m",
                  flush=True)

    side_exec.shutdown()
    executor.shutdown()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
