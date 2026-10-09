"""Does Gazebo see the world the 2-D simulator sees?

One world, the TurtleBot3 at its start pose. The Gazebo lidar's scan is set
against the project's own ray-cast from the same true sensor position, at the
same bearings, with the same range limits, and noise-free on the 2-D side: if
the export, the frames and the conventions agree, the beams differ by the
lidar's 0.01 m of noise and little else. Then the robot drives forward for two
seconds and its wheel odometry is set against the true distance travelled.

Run in the ros_tb3 environment (see gazebo_tb3/README.md):

    python gazebo_tb3/sensor_check.py --condition dense
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from gz_session import GzSession, yaw_of  # noqa: E402
from robot_sdf import lidar_spec, robot_model  # noqa: E402
from world_sdf import world_sdf  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.sensors import Lidar2D, LidarConfig  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

MODELS = "share/nav2_minimal_tb3_sim/models"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="dense")
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--profile", default="real")
    p.add_argument("--no-headless", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    _, shift, _ = BENCHMARK_CONDITIONS[args.condition]
    cfg = build_env_config({}, split="val", shift=shift, n_worlds=args.seed_index + 1)
    env = ProceduralNavEnv(cfg)
    seed = int(list(cfg.world_seeds)[args.seed_index])
    env.reset(options={"world_seed": seed})
    world = env.world

    model = robot_model((HERE / "gz_waffle.sdf").read_text(encoding="utf-8"), args.profile,
                        pose=tuple(float(v) for v in world.start))
    spec = lidar_spec(model)
    prefix = Path(sys.prefix)
    sdf = world_sdf(world, robot_xml=ET.tostring(model, encoding="unicode"))
    report: dict = {"condition": args.condition, "seed": seed, "profile": args.profile,
                    "lidar": spec}
    with GzSession(sdf, resource_path=str(prefix / MODELS),
                   headless=not args.no_headless, verbose=3) as gz:
        gz.wait_ready()
        # Let it settle on its wheels and deliver a scan or two (5 Hz).
        for _ in range(10):
            gz.command(0.0, 0.0)
            gz.step(0.1)
        if gz.scan is None or gz.true_pose is None:
            raise RuntimeError(f"no scan or pose after 1 s; see {gz.log_path}")
        scan, (x, y, th) = gz.scan, gz.true_pose
        ranges = np.asarray(scan.ranges, dtype=float)
        bearings = scan.angle_min + scan.angle_step * np.arange(len(ranges))
        # The 2-D sensor, at the lidar's true position and the scan's bearings.
        sensor_xy = np.array([x, y]) + spec["x_offset"] * np.array([math.cos(th), math.sin(th)])
        lidar = Lidar2D(replace(LidarConfig(), n_beams=len(ranges), max_range=spec["range_max"],
                                noise_std=0.0, dropout_prob=0.0))
        lidar._angles = bearings
        ref = lidar.scan(world, np.array([sensor_xy[0], sensor_xy[1], th]))
        hit_gz = np.isfinite(ranges) & (ranges < spec["range_max"] - 1e-6)
        hit_ref = ref < spec["range_max"] - 1e-6
        both = hit_gz & hit_ref
        diff = np.abs(ranges[both] - ref[both])
        report["scan"] = {
            "beams": int(len(ranges)), "angle_min": scan.angle_min, "angle_step": scan.angle_step,
            "start_pose_error": float(np.hypot(x - world.start[0], y - world.start[1])),
            "hit_agreement": float(np.mean(hit_gz == hit_ref)),
            "both_hit": int(both.sum()),
            "median_abs_diff": float(np.median(diff)) if len(diff) else float("nan"),
            "p95_abs_diff": float(np.percentile(diff, 95)) if len(diff) else float("nan"),
            "within_3_sigma": float(np.mean(diff <= 3 * max(spec["noise_std"], 1e-3))) if len(diff) else float("nan"),
        }
        # Two seconds forward at 0.2 m/s: odometry against the truth.
        x0, y0, _ = gz.true_pose
        o = gz.odom.pose.position
        ox0, oy0 = o.x, o.y
        for _ in range(20):
            gz.command(0.2, 0.0)
            gz.step(0.1)
        gz.command(0.0, 0.0)
        gz.step(0.1)
        x1, y1, th1 = gz.true_pose
        o = gz.odom.pose.position
        report["drive"] = {
            "true_distance": float(np.hypot(x1 - x0, y1 - y0)),
            "odom_distance": float(np.hypot(o.x - ox0, o.y - oy0)),
            "true_heading_change": float(th1 - th),
            "odom_yaw": float(yaw_of(gz.odom.pose.orientation)),
            "sim_time": gz.sim_time,
        }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
