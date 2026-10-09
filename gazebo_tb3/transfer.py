"""The transfer test: the same worlds, driven in the 2-D simulator and in Gazebo.

Every arm runs the same agent (:class:`tb3_agent.TB3Agent`, shown by its tests
to be the published stack when configured as published) on the same worlds,
with the same step budget and the same rules for arriving and colliding, both
judged from the true pose. What differs is named by the arm:

  project     the published configuration in 2-D: the project's robot limits,
              a 6 m lidar at the centre, a scan every control step
  tb3_2d      a TurtleBot3 Waffle Pi in 2-D: its limits, the LDS-01's
              bearings, 3.5 m range and 5 Hz, 0.064 m behind the axle
  tb3_2d_6m   the same with a 6 m lidar, to separate the range from the rest
  gazebo      the TurtleBot3 in Gazebo: everything tb3_2d is, plus a physics
              engine, wheels that slip, a rendered lidar and the real body

so ``tb3_2d - project`` is what the robot's specification costs, and
``gazebo - tb3_2d`` is what a physics simulator adds on top -- the sim-to-sim
gap, which is the stand-in for sim-to-real.

Lidar noise is the same in every arm: 0.01 m Gaussian on returns, from a
generator seeded by the world, added by this script (Gazebo's own is switched
off, because Gazebo cannot be seeded).

    python gazebo_tb3/transfer.py --arm tb3_2d --conditions nominal dense narrow --episodes 50
    python gazebo_tb3/transfer.py --arm gazebo ...      # in the ros_tb3 environment
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
from tb3_agent import (  # noqa: E402
    TB3_BEARINGS,
    TB3_LIDAR,
    TB3_LIDAR_X,
    TB3_ROBOT,
    TB3Agent,
    analytic_lidar,
)

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.robot import DiffDriveRobot, RobotConfig  # noqa: E402
from vision_nav.envs.sensors import LidarConfig  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

#: ``tb3_2d_exact_odom`` was added after the registered run, to test the one
#: difference between ``tb3_2d`` and ``gazebo`` named before it: the 2-D arms
#: drift with the published odometry model, Gazebo only by wheel slip. It is
#: ``tb3_2d`` with that model switched off -- odometry integrates the true
#: velocity, no drift at all -- so it brackets Gazebo from the other side.
ARMS = ("project", "tb3_2d", "tb3_2d_6m", "gazebo", "tb3_2d_exact_odom")
BUDGET = 1200
NOISE = 0.01
#: The published configuration's sensor: 360 beams over the full circle, 6 m.
PROJECT_LIDAR = LidarConfig(n_beams=360, fov=2.0 * np.pi, max_range=6.0, noise_std=NOISE)
PROJECT_BEARINGS = np.linspace(-np.pi, np.pi, 360, endpoint=False)


def arm_spec(arm: str) -> dict:
    """Robot, sensor and timing of an arm."""
    if arm == "project":
        return {"robot": RobotConfig(), "lidar": PROJECT_LIDAR, "bearings": PROJECT_BEARINGS,
                "lidar_x": 0.0, "scan_every": 1, "odometry": OdometryConfig()}
    lidar = replace(TB3_LIDAR, max_range=6.0) if arm == "tb3_2d_6m" else TB3_LIDAR
    odometry = None if arm == "tb3_2d_exact_odom" else OdometryConfig()
    return {"robot": TB3_ROBOT, "lidar": lidar, "bearings": TB3_BEARINGS,
            "lidar_x": TB3_LIDAR_X, "scan_every": 2, "odometry": odometry}


def add_noise(ranges: np.ndarray, max_range: float, rng: np.random.Generator) -> np.ndarray:
    """Every arm's lidar noise: Gaussian on returns only, then clipped."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    hit = np.isfinite(r) & (r < max_range - 1e-6)
    r[hit] = np.clip(r[hit] + rng.normal(0.0, NOISE, int(hit.sum())), 0.0, max_range)
    return r


def world_for(cond: str, seed: int, split: str, n: int, robot: RobotConfig):
    _, shift, _ = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({}, split=split, shift=shift, n_worlds=n)
    cfg.robot = robot
    cfg.max_episode_steps = BUDGET
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    return cfg, env


def summary(agent: TB3Agent, success: bool, collision: bool, steps: int, goal_distance: float,
            path: float, extra: dict | None = None) -> dict:
    errs = agent.pose_errors or [0.0]
    return {"success": bool(success), "collision": bool(collision), "steps": int(steps),
            "goal_distance": float(goal_distance), "path_length": float(path),
            "pose_err_median": float(np.median(errs)), "pose_err_final": float(errs[-1]),
            "pose_err_max": float(np.max(errs)), "replans": int(agent.replans),
            "scans_used": int(agent.scans_used), **(extra or {})}


def analytic_episode(arm: str, cond: str, seed: int, split: str, n: int) -> dict:
    spec = arm_spec(arm)
    cfg, env = world_for(cond, seed, split, n, spec["robot"])
    world = env.world
    rng = np.random.default_rng((seed, 11))
    caster = analytic_lidar(replace(spec["lidar"], noise_std=0.0), spec["bearings"])

    def scan_at(pose: np.ndarray) -> np.ndarray:
        s = np.asarray(pose, dtype=np.float64).copy()
        s[:2] += spec["lidar_x"] * np.array([np.cos(s[2]), np.sin(s[2])])
        return add_noise(caster.scan(world, s), spec["lidar"].max_range, rng)

    agent = TB3Agent(robot=spec["robot"], lidar=spec["lidar"], bearings=spec["bearings"],
                     lidar_x=spec["lidar_x"], odometry=spec["odometry"])
    agent.start_episode_with_scan(world, env.robot.pose, scan_at(env.robot.pose))
    info: dict = {}
    pending = None
    for k in range(BUDGET):
        pose = env.robot.pose.copy()
        action = agent.step(env.robot.velocity.copy(), pending, truth=pose)
        _, _, done, truncated, info = env.step(action)
        # The scan taken at the end of this step, used at the next decision:
        # Gazebo's arrives the same way.
        pending = scan_at(env.robot.pose) if (k + 1) % spec["scan_every"] == 0 else None
        if done or truncated:
            break
    return summary(agent, info.get("is_success"), info.get("collision"), info.get("steps", k + 1),
                   info.get("goal_distance", np.nan), info.get("path_length", np.nan))


def gazebo_episode(cond: str, seed: int, split: str, n: int, profile: str = "real") -> dict:
    from gz_session import GzSession  # needs the ros_tb3 environment
    from robot_sdf import robot_model
    from world_sdf import world_sdf

    spec = arm_spec("tb3_2d")
    cfg, env = world_for(cond, seed, split, n, spec["robot"])
    world = env.world
    model = robot_model((HERE / "gz_waffle.sdf").read_text(encoding="utf-8"), profile,
                        pose=tuple(float(v) for v in world.start), lidar_noise=False,
                        report_hz=500)
    sdf = world_sdf(world, robot_xml=ET.tostring(model, encoding="unicode"))
    rng = np.random.default_rng((seed, 11))
    max_range = spec["lidar"].max_range
    scale = DiffDriveRobot(spec["robot"])
    tol, radius = float(world.config.goal_tolerance), float(world.config.robot_radius)
    started = time.monotonic()
    with GzSession(sdf, resource_path=str(Path(sys.prefix) / "share/nav2_minimal_tb3_sim/models"),
                   sync=True) as gz:
        gz.wait_ready()
        # At rest until the lidar has delivered (renderer start-up), then the
        # episode begins on a scan boundary.
        for _ in range(20):
            gz.command(0.0, 0.0)
            gz.step(0.1)
            if gz.scan is not None and abs(gz.scan_stamp - gz.sim_time) < 1e-3:
                break
        if gz.scan is None:
            raise RuntimeError(f"no lidar scan after 2 s; see {gz.log_path}")
        x0, y0, th0 = gz.true_pose
        agent = TB3Agent(robot=spec["robot"], lidar=spec["lidar"], bearings=spec["bearings"],
                         lidar_x=spec["lidar_x"], odometry=None)
        ranges = add_noise(np.asarray(gz.scan.ranges), max_range, rng)
        agent.start_episode_with_scan(world, np.array([x0, y0, th0]), ranges)
        pending, missed, path = None, 0, 0.0
        last = np.array([x0, y0])
        success = collision = False
        k = 0
        for k in range(BUDGET):
            truth = np.array(gz.true_pose)
            vel = np.array([gz.odom.twist.linear.x, gz.odom.twist.angular.z])
            action = agent.step(vel, pending, truth=truth)
            v, w = scale.scale_action(action)
            gz.command(v, w)
            gz.step(0.1)
            now = gz.sim_time
            pending = None
            if (k + 1) % spec["scan_every"] == 0:
                if gz.wait_for("scan", now - 1e-4, timeout=30.0) and abs(gz.scan_stamp - now) < 1e-3:
                    pending = add_noise(np.asarray(gz.scan.ranges), max_range, rng)
                else:
                    missed += 1
            here = np.array(gz.true_pose[:2])
            path += float(np.linalg.norm(here - last))
            last = here
            if np.linalg.norm(here - world.goal) <= tol:
                success = True
                break
            if float(world.clearance(here, include_dynamic=False)) < radius:
                collision = True
                break
        goal_distance = float(np.linalg.norm(last - world.goal))
    return summary(agent, success, collision, k + 1, goal_distance, path,
                   {"missed_scans": missed, "wall_seconds": time.monotonic() - started})


_STAGGER = None


def _stagger_start(counter, seconds: float) -> None:
    """Pool initialiser: each worker waits its turn before its first server,
    so the GPU renderer is never started by every worker at once -- six
    simultaneous start-ups hung where two did not."""
    global _STAGGER
    with counter.get_lock():
        turn = counter.value
        counter.value += 1
    _STAGGER = turn * seconds
    time.sleep(_STAGGER)


def _run(job: tuple) -> dict:
    arm, cond, seed, split, n, partition = job
    if arm != "gazebo":
        return {"arm": arm, "cond": cond, "seed": seed, **analytic_episode(arm, cond, seed, split, n)}
    errors = []
    # A run is deterministic, so an episode whose server failed to start is
    # simply run again on a fresh one; the result is the same episode.
    for attempt in range(2):
        # Each server on its own transport partition, so parallel servers
        # never hear each other.
        os.environ["GZ_PARTITION"] = f"transfer_{partition}_{attempt}"
        try:
            out = gazebo_episode(cond, seed, split, n)
            return {"arm": arm, "cond": cond, "seed": seed, "attempts": attempt + 1, **out}
        except Exception as exc:  # noqa: BLE001 -- recorded, then retried once
            errors.append(f"{type(exc).__name__}: {exc}")
    return {"arm": arm, "cond": cond, "seed": seed, "error": errors}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--conditions", nargs="+", default=["nominal", "dense", "narrow"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--split", default="val")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    jobs = []
    for cond in args.conditions:
        _, shift, _ = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({}, split=args.split, shift=shift, n_worlds=args.episodes)
        for s in list(cfg.world_seeds)[:args.episodes]:
            jobs.append((args.arm, cond, int(s), args.split, args.episodes, len(jobs)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Resumable: episodes already written are kept and not rerun.
    if out.exists():
        rows = json.loads(out.read_text(encoding="utf-8")).get("episodes", [])
    done = {(r["cond"], r["seed"]) for r in rows}
    todo = [j for j in jobs if (j[1], j[2]) not in done]
    failed: list[dict] = []
    counter = mp.Value("i", 0)
    stagger = 20.0 if args.arm == "gazebo" else 0.0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_stagger_start,
                             initargs=(counter, stagger)) as pool:
        futures = [pool.submit(_run, j) for j in todo]
        for fut in as_completed(futures):
            row = fut.result()
            if "error" in row:
                # Never written as an episode, never silently dropped.
                failed.append(row)
                print(f"{row['cond']:8s} {row['seed']} FAILED: {row['error']}", flush=True)
                continue
            rows.append(row)
            out.write_text(json.dumps({"arm": args.arm, "split": args.split, "budget": BUDGET,
                                       "episodes": rows, "failed": failed}, indent=1),
                           encoding="utf-8")
            print(f"{row['cond']:8s} {row['seed']} success {row['success']} collision "
                  f"{row['collision']} steps {row['steps']} pose {row['pose_err_median']:.3f}",
                  flush=True)
    out.write_text(json.dumps({"arm": args.arm, "split": args.split, "budget": BUDGET,
                               "episodes": rows, "failed": failed}, indent=1), encoding="utf-8")
    for cond in args.conditions:
        sub = [r for r in rows if r["cond"] == cond]
        if sub:
            print(f"{args.arm} {cond}: success {np.mean([r['success'] for r in sub]):.3f} "
                  f"collision {np.mean([r['collision'] for r in sub]):.3f} over {len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
