"""Evaluate Nav2 on the project's held-out worlds, as one more actor.

Runs the identical protocol every other actor is scored under — same worlds,
same seed order, same success criterion, same metrics — with the real Nav2
stack in place of the hand-written A* + pure-pursuit baseline. The point is
not a new finding but a credibility check: is the hand-written baseline as
strong as the report claims?

    python ros2_bridge/run_nav2_eval.py --condition narrow --episodes 100

Must run inside the ROS 2 env with Nav2 already launched against
nav2_params.yaml. See ros2_bridge/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_bridge import CMD_VEL_TYPE, Nav2Bridge, yaw_to_quaternion
from nav2_msgs.srv import ClearEntireCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

try:  # only the SLAM arm needs it, and only that arm requires it installed
    from slam_toolbox.srv import Reset as SlamReset
except ImportError:  # pragma: no cover - depends on the ROS environment
    SlamReset = None

from vision_nav.envs.splits import (
    BENCHMARK_CONDITIONS,
    DYNAMIC_CONDITIONS,
    FROZEN_CONDITIONS,
)
from vision_nav.metrics import EpisodeResult, aggregate
from vision_nav.training.env_factory import build_env_config

#: Every condition Nav2 can be scored on: the six static benchmark rows plus
#: the two moving-obstacle ones.
CONDITIONS = {**BENCHMARK_CONDITIONS, **DYNAMIC_CONDITIONS}

#: Sim steps to run before sending the goal, letting Nav2's costmaps populate
#: from the first scan. Without this the first plan is made against an empty
#: local costmap and the robot lurches before correcting.
WARMUP_STEPS = 10

#: Nodes that must reach the ACTIVE lifecycle state before goals are sent.
#: Matches LIFECYCLE_NODES in nav2_launch.py.
REQUIRED_NODES = ("controller_server", "planner_server",
                  "behavior_server", "bt_navigator")

#: Costmap-clearing services, called at every episode reset. Both costmaps:
#: the static layer re-applies the freshly published map, so the walls survive
#: the clear while the accumulated obstacle marks do not.
CLEAR_SERVICES = (
    "/global_costmap/clear_entirely_global_costmap",
    "/local_costmap/clear_entirely_local_costmap",
)

#: Sim steps between publishing the new world's map and clearing the costmaps,
#: so the static layer has the new layout to reset *to* rather than re-applying
#: the previous episode's.
MAP_SETTLE_STEPS = 5

#: SLAM arm only. After the pose graph is reset the robot has no map at all, so
#: the goal cannot be sent until slam_toolbox has published one and the
#: map -> odom correction exists. Bounded rather than infinite: a run that
#: silently waited forever would look like a hang, and one that gave up
#: silently would score an unplannable episode as Nav2's performance.
SLAM_READY_TIMEOUT_STEPS = 600
#: Extra sim steps after the first map arrives, so the global costmap has
#: consumed it before the first plan is asked for.
SLAM_SETTLE_STEPS = 10

LATCHED = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

#: SLAM arm only: the fastest the simulator may run, as a multiple of real time.
#:
#: The loop that advances sim time is otherwise unthrottled — ``spin_once``
#: returns at once whenever a message is waiting, and one always is. With full
#: privileges that is harmless: startup lasts about a second and Nav2 was
#: verified to keep up during episodes (commands per sim step near 1.0). The
#: SLAM arm broke it. Sim time ran at about 190x real time while navigation
#: waited to start, publishing a 360-beam scan every half millisecond;
#: slam_toolbox's queue overflowed and dropped them, and the machine starved
#: badly enough that the lifecycle manager's call to configure controller_server
#: timed out and the bringup hung for good.
#:
#: Capping the rate only ever gives the stack *more* wall-clock time per sim
#: step, so it cannot help the planner against the full-privilege row, only stop
#: SLAM being scored on a flood of dropped scans. The full arm keeps its
#: original, unthrottled protocol so the published row stays comparable.
SLAM_REALTIME_FACTOR = 5.0


class Pace:
    """Hold the loop to at most ``factor`` times real time; 0 means no limit."""

    def __init__(self, dt: float, factor: float) -> None:
        self.min_wall = dt / factor if factor else 0.0
        self._last = time.monotonic()

    def wait(self) -> None:
        if self.min_wall:
            remaining = self._last + self.min_wall - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self._last = time.monotonic()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="nominal", choices=list(CONDITIONS),
                   help="Evaluation condition, from the shared benchmark table")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--privileges", default="full", choices=("full", "slam"),
                   help="'full': the static map and exact pose, as published. "
                        "'slam': neither — slam_toolbox builds the map and "
                        "localises against it, on drifting odometry (§9.4).")
    p.add_argument("--beams", type=int, default=360,
                   help="Scan beams handed to Nav2. 360 as published; 32 to "
                        "match the hand-written mapping stack's sensor (§9.2).")
    p.add_argument("--split", default=None,
                   help="Override the condition's split, e.g. 'val' while "
                        "developing. The registered run leaves this alone.")
    return p.parse_args(argv)


def pose_msg(x: float, y: float, yaw: float, stamp) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp.sec, p.header.stamp.nanosec = stamp
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.orientation = yaw_to_quaternion(float(yaw))
    return p


def pump(bridge: Nav2Bridge, executor) -> None:
    """Advance sim time and republish, without stepping the physics.

    Nav2 runs with ``use_sim_time``, so its timers, TF tolerances and
    lifecycle bonds only tick when ``/clock`` does. During startup and while
    waiting on a service the robot must stay still but the clock must not, or
    the whole stack stalls waiting for a time that never arrives.
    """
    bridge.pace.wait()
    bridge.publish_all()
    bridge.publish_map()
    executor.spin_once(timeout_sec=0.01)
    bridge._sim_time += bridge.dt


def wait_until_active(bridge: Nav2Bridge, executor, names, timeout: float = 180.0) -> bool:
    """Block until every named lifecycle node reports ACTIVE.

    ``BasicNavigator.waitUntilNav2Active`` cannot be used: it blocks, and this
    process is the only publisher of ``/clock``, so blocking stops the clock
    the nodes are waiting on. Polling ``wait_for_server`` is not enough either
    — bt_navigator creates its action server during *configure*, so the server
    exists while the node is still inactive and rejects every goal.
    """
    clients = {n: bridge.create_client(GetState, f"/{n}/get_state") for n in names}
    pending = list(names)
    deadline = time.monotonic() + timeout

    while pending and time.monotonic() < deadline:
        name = pending[0]
        client = clients[name]
        if not client.service_is_ready():
            pump(bridge, executor)
            continue
        future = client.call_async(GetState.Request())
        call_deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < call_deadline:
            pump(bridge, executor)
        result = future.result() if future.done() else None
        if result is not None and result.current_state.id == State.PRIMARY_STATE_ACTIVE:
            pending.pop(0)
            print(f"  {name}: active", flush=True)

    for client in clients.values():
        bridge.destroy_client(client)
    if pending:
        print(f"never became active: {', '.join(pending)}", file=sys.stderr)
    return not pending


def clear_costmaps(bridge: Nav2Bridge, executor, clients, timeout: float = 10.0) -> bool:
    """Wipe both costmaps so the next episode starts from a clean map.

    Every other actor in this study is evaluated on independent episodes.
    Nav2's costmaps are stateful, so without an explicit reset its episodes
    are not independent: obstacle marks are cleared only by ray-tracing from
    the robot's current position, and the robot teleports between worlds.
    Scoring Nav2 on dependent episodes would understate it, which is the
    direction that would falsely flatter the hand-written baseline.
    """
    futures = [c.call_async(ClearEntireCostmap.Request()) for c in clients]
    deadline = time.monotonic() + timeout
    while not all(f.done() for f in futures) and time.monotonic() < deadline:
        pump(bridge, executor)
    return all(f.done() for f in futures)


def reset_slam(bridge: Nav2Bridge, executor, client, timeout: float = 20.0) -> bool:
    """Throw away the pose graph and the map between episodes.

    The same independence requirement the costmap clear exists for, one level
    deeper. slam_toolbox accumulates a pose graph over its whole lifetime, and
    the robot teleports into an unrelated world every episode; without this the
    second episode would be matching its scans against the first world's map.
    """
    future = client.call_async(SlamReset.Request())
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        pump(bridge, executor)
    return future.done()


def wait_for_slam(bridge: Nav2Bridge, executor, tf_buffer, seen_map) -> bool:
    """Pump until slam_toolbox has a map and a map -> odom correction."""
    for _ in range(SLAM_READY_TIMEOUT_STEPS):
        if seen_map["count"] and tf_buffer.can_transform("map", "base_link", Time()):
            for _ in range(SLAM_SETTLE_STEPS):
                pump(bridge, executor)
            return True
        pump(bridge, executor)
    return False


def believed_pose(tf_buffer) -> np.ndarray | None:
    """Where the robot thinks it is, in the map frame: map -> base_link."""
    try:
        t = tf_buffer.lookup_transform("map", "base_link", Time()).transform
    except Exception:
        return None
    return np.array([t.translation.x, t.translation.y])


def correction_lag(bridge: Nav2Bridge, tf_buffer) -> float | None:
    """Sim seconds between now and the newest map -> odom correction.

    slam_toolbox stamps that correction with the time of the last scan it has
    finished processing. If it cannot keep pace with the simulator the stamp
    falls behind, and everything reading map -> base_link — Nav2 included —
    sees where the robot was rather than where it is. The SLAM arm is only
    measuring SLAM if this stays small, the way the controller is only being
    measured if commands per sim step stays near 1.0.
    """
    try:
        stamp = tf_buffer.lookup_transform("map", "odom", Time()).header.stamp
    except Exception:
        return None
    return bridge._sim_time - (stamp.sec + stamp.nanosec * 1e-9)


def run_episode(bridge: Nav2Bridge, nav: BasicNavigator, executor, seed: int,
                clear_clients, slam_client=None, tf_buffer=None,
                seen_map=None) -> tuple[EpisodeResult, float]:
    """One episode: reset the world, hand Nav2 the goal, step until resolved."""
    bridge.start_episode(seed)
    world = bridge.env.world

    if bridge.slam:
        if not reset_slam(bridge, executor, slam_client):
            print("  warning: slam reset timed out; this episode may be "
                  "matching against the previous world", file=sys.stderr)
        seen_map["count"] = 0

    # Publish the new layout before clearing, so the static layer resets to
    # this world rather than re-applying the previous one.
    for _ in range(MAP_SETTLE_STEPS):
        pump(bridge, executor)
    if not clear_costmaps(bridge, executor, clear_clients):
        print("  warning: costmap clear timed out; episode may inherit "
              "obstacles from the previous world", file=sys.stderr)

    # No setInitialPose. With full privileges the bridge broadcasts ground-truth
    # map->odom->base_link, the same privilege the hand-written baseline gets;
    # in the SLAM arm the localiser is slam_toolbox and seeding it with the true
    # pose would hand back the privilege the arm exists to remove.
    # Warm up so the costmaps hold a scan before the first plan is made.
    for _ in range(WARMUP_STEPS):
        bridge.pace.wait()
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.02)

    # Recorded per episode rather than only warned about, so a failure that
    # began before the robot moved can be told apart from one SLAM caused
    # while driving.
    bridge.slam_ready = True
    bridge.nav2_aborted = False
    if bridge.slam and not wait_for_slam(bridge, executor, tf_buffer, seen_map):
        bridge.slam_ready = False
        print("  warning: slam produced no map in time; goal sent blind",
              file=sys.stderr)

    nav.goToPose(pose_msg(world.goal[0], world.goal[1], 0.0, bridge.now_msg()))

    # The *env* owns termination, not Nav2. Nav2 declaring success while the
    # robot sits outside goal_tolerance would otherwise be scored as a win.
    running = True
    before = bridge.commands_received
    # Development aid: NAV2_TRACE=1 prints, every 25 steps, the true pose, what
    # odometry alone believes, and what slam_toolbox believes — enough to see
    # whether the map frame starts aligned with the world and when it leaves it.
    trace = bool(os.environ.get("NAV2_TRACE")) and bridge.slam
    # Commands counted only while Nav2 is still navigating. After it abandons a
    # goal the robot is stepped to the timeout with no commands coming, which
    # is an honest failure, not a starved controller; counting those steps
    # made the starvation check read one as the other.
    active_steps, active_commands, completed = 0, 0, False
    while running:
        bridge.pace.wait()
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.02)
        if not completed:
            active_steps += 1
            active_commands = bridge.commands_received - before
        running = bridge.step_sim()
        completed = completed or nav.isTaskComplete()
        if trace and bridge.steps % 25 == 1:
            truth = bridge.env.robot.pose
            odom = bridge.believed_pose
            slam = believed_pose(tf_buffer)
            lag = correction_lag(bridge, tf_buffer)
            print(f"    step {bridge.steps:3d} true ({truth[0]:.2f},{truth[1]:.2f}) "
                  f"odom ({odom[0]:.2f},{odom[1]:.2f}) slam "
                  + (f"({slam[0]:.2f},{slam[1]:.2f})" if slam is not None else "none")
                  + (f" lag {lag:.2f}s" if lag is not None else " lag n/a")
                  + f" cmds {bridge.commands_received - before}", flush=True)
        if nav.isTaskComplete():
            result = nav.getResult()
            if result != TaskResult.SUCCEEDED and running:
                # Nav2 gave up (planning failure, recovery exhausted). Keep
                # stepping so the episode ends on the env's own terms.
                bridge.nav2_aborted = True
                bridge.env._steps = bridge.env.config.max_episode_steps
                running = bridge.step_sim()

    if trace and seen_map.get("last") is not None:
        # Where slam_toolbox put the surfaces it mapped, against where the
        # world's own obstacles are: a scale or frame error shows up here.
        m = seen_map["last"]
        grid = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        rows, cols = np.nonzero(grid > 50)
        if len(rows):
            xs = m.info.origin.position.x + (cols + 0.5) * m.info.resolution
            ys = m.info.origin.position.y + (rows + 0.5) * m.info.resolution
            print(f"    slam map: res {m.info.resolution:.2f} occupied {len(rows)} "
                  f"x [{xs.min():.2f},{xs.max():.2f}] y [{ys.min():.2f},{ys.max():.2f}]"
                  f"  arena {world.config.width:.0f} x {world.config.height:.0f}",
                  flush=True)
            pts = np.stack([xs, ys], axis=-1)
            true_clear = world.clearance(pts, include_dynamic=False)
            print(f"    mapped cells' true distance to a surface: median "
                  f"{np.median(true_clear):.2f} m, p90 {np.percentile(true_clear, 90):.2f} m",
                  flush=True)

    # How wrong the robot's belief about its own position was when the episode
    # ended, in metres — the same quantity §9.3 reports for the hand-written
    # stack, so the two localisers can be compared and not just their outcomes.
    pose_error = 0.0
    if bridge.slam:
        believed = believed_pose(tf_buffer)
        if believed is not None:
            pose_error = float(np.linalg.norm(believed - bridge.env.robot.pose[:2]))

    # Commands per sim step. The simulator runs faster than real time, so a
    # ratio well below 1.0 would mean Nav2's 10 Hz control loop was starved of
    # wall-clock time and the robot coasted on stale commands — a handicap,
    # not a result. Reported alongside the metrics so the claim is checkable.
    bridge.command_ratio = (bridge.commands_received - before) / max(bridge.steps, 1)
    bridge.active_command_ratio = active_commands / max(active_steps, 1)
    return EpisodeResult.from_info(bridge.last_info), pose_error


def main(argv=None) -> int:
    args = parse_args(argv)
    rclpy.init()

    split, shift, noise = CONDITIONS[args.condition]
    split = args.split or split
    overrides: dict = {"lidar": {"noise_std": noise}} if noise else {}
    if args.condition in FROZEN_CONDITIONS:
        overrides["freeze_dynamic"] = True
    env_config = build_env_config(overrides, split=split, shift=shift,
                                  n_worlds=args.episodes)
    seeds = list(env_config.world_seeds)

    bridge = Nav2Bridge(env_config, privileges=args.privileges, beams=args.beams)
    # The full arm stays unthrottled unless NAV2_FULL_RTF says otherwise: that
    # is the protocol its published passes were recorded under. Throttling it
    # is a development option for runs made alongside others, where the
    # unthrottled loop can outrun the global costmap's switch to a new world's
    # map and ask for a plan on the previous one.
    factor = float(os.environ.get("NAV2_SLAM_RTF", SLAM_REALTIME_FACTOR)) if bridge.slam \
        else float(os.environ.get("NAV2_FULL_RTF", 0.0))
    bridge.pace = Pace(bridge.dt, factor)
    nav = BasicNavigator()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(nav)

    # SLAM arm: a TF listener to read back what the robot believes, a /map
    # subscription to know when there is one, and the reset service.
    #
    # The listener and the subscription live on their own node, spun by their
    # own executor on their own thread. The first version put them on the
    # bridge, in the one executor that also delivers Nav2's velocity commands.
    # spin_once handles a single callback per call and slam_toolbox broadcasts
    # map -> odom several times per sim step, so commands queued behind TF
    # traffic: 0.44 commands per step instead of about 1.0, stale commands
    # applied to the robot, and a pose-error readout taken from a buffer that
    # was behind. That was a defect in this runner, not a property of SLAM.
    tf_buffer = seen_map = slam_client = tf_side = None
    if bridge.slam:
        if SlamReset is None:
            print("privileges=slam needs slam_toolbox on the ROS path",
                  file=sys.stderr)
            return 1
        tf_buffer = Buffer()
        tf_node = rclpy.create_node("nav2_eval_observer")
        TransformListener(tf_buffer, tf_node)
        seen_map = {"count": 0, "last": None}

        def on_map(msg: OccupancyGrid) -> None:
            seen_map["count"] += 1
            seen_map["last"] = msg

        tf_node.create_subscription(OccupancyGrid, "/map", on_map, LATCHED)
        tf_executor = SingleThreadedExecutor()
        tf_executor.add_node(tf_node)
        threading.Thread(target=tf_executor.spin, daemon=True).start()
        tf_side = (tf_executor, tf_node)
        slam_client = bridge.create_client(SlamReset, "/slam_toolbox/reset")

    print(f"waiting for Nav2 to activate ({len(seeds)} episodes queued)...", flush=True)
    bridge.start_episode(seeds[0])
    # Tell run_nav2.sh that /clock, /scan and TF are now being published, so it
    # can start Nav2. Nav2 started first loses a race: its costmaps refuse to
    # activate without the odom frame and the lifecycle manager aborts the
    # whole bringup — which is what happened when six runs started at once and
    # one runner sat behind micromamba's lock for longer than Nav2 would wait.
    ready_file = os.environ.get("NAV2_READY_FILE")
    if ready_file:
        pump(bridge, executor)
        Path(ready_file).write_text("publishing\n", encoding="utf-8")

    started = time.monotonic()
    if not wait_until_active(bridge, executor, REQUIRED_NODES):
        return 1
    while not nav.nav_to_pose_client.wait_for_server(timeout_sec=0.0):
        pump(bridge, executor)
    print(f"Nav2 active after {time.monotonic() - started:.0f}s wall", flush=True)

    clear_clients = [bridge.create_client(ClearEntireCostmap, s)
                     for s in CLEAR_SERVICES]
    for client, name in zip(clear_clients, CLEAR_SERVICES, strict=True):
        while not client.service_is_ready():
            pump(bridge, executor)
            if time.monotonic() - started > 240:
                print(f"costmap-clear service never appeared: {name}",
                      file=sys.stderr)
                return 1
    if bridge.slam:
        while not slam_client.service_is_ready():
            pump(bridge, executor)
            if time.monotonic() - started > 300:
                print("slam_toolbox reset service never appeared; is "
                      "slam:=True set on the launch?", file=sys.stderr)
                return 1
        print("slam_toolbox ready", flush=True)

    results = []
    ratios = []
    pose_errors = []
    ready_flags, aborted_flags, active_ratios = [], [], []
    try:
        for i, seed in enumerate(seeds, 1):
            r, pose_error = run_episode(bridge, nav, executor, seed, clear_clients,
                                        slam_client, tf_buffer, seen_map)
            ratios.append(bridge.command_ratio)
            active_ratios.append(bridge.active_command_ratio)
            pose_errors.append(pose_error)
            ready_flags.append(bool(bridge.slam_ready))
            aborted_flags.append(bool(bridge.nav2_aborted))
            if i == 1 and bridge.commands_received == 0:
                # A /cmd_vel type mismatch produces a robot that never moves,
                # which otherwise scores as a clean sweep of timeouts and
                # would be published as Nav2's performance. Refuse to continue.
                raise RuntimeError(
                    "no /cmd_vel command was received during the first "
                    "episode. Nav2's publisher type almost certainly "
                    "disagrees with CMD_VEL_TYPE in nav2_bridge.py "
                    f"({CMD_VEL_TYPE.__name__}); check enable_stamped_cmd_vel "
                    "in nav2_params.yaml."
                )
            results.append(r)
            err = f" pose err {pose_error:.3f}m" if bridge.slam else ""
            print(
                f"  [{i:>3}/{len(seeds)}] world {seed}: "
                f"{'success' if r.success else ('collision' if r.collision else 'timeout')} "
                f"in {r.steps} steps (cmd/step {bridge.command_ratio:.2f}){err}",
                flush=True,
            )
    finally:
        metrics = aggregate(results) if results else None
        if tf_side is not None:
            tf_side[0].shutdown()
            tf_side[1].destroy_node()
        executor.shutdown()
        bridge.destroy_node()
        nav.destroy_node()
        rclpy.shutdown()

    if metrics is None:
        print("no episodes completed", file=sys.stderr)
        return 1

    mean_ratio = float(sum(ratios) / len(ratios)) if ratios else 0.0
    # Episodes where Nav2 never issued a single command: it could not plan at
    # all. This was the signature of the costmap-accumulation bug, so it is
    # reported every run rather than left to be rediscovered.
    zero_cmd = sum(1 for r in ratios if r == 0.0)
    print()
    label = f"Nav2 | condition={args.condition}"
    if bridge.slam:
        label += " | SLAM"
    print(metrics.as_table(label))
    print(f"\nmean commands per sim step: {mean_ratio:.3f}")
    print(f"episodes with no command at all: {zero_cmd}/{len(ratios)}"
          + ("  <-- Nav2 could not plan; check the costmaps" if zero_cmd else ""))
    extra: dict = {"nav2_aborted_episodes": int(sum(aborted_flags)),
                   "nav2_aborted_per_episode": aborted_flags,
                   "mean_active_commands_per_step": float(np.mean(active_ratios)),
                   "active_commands_per_step_per_episode": [float(a) for a in active_ratios],
                   "commands_per_step_per_episode": [float(r) for r in ratios]}
    print(f"mean commands per sim step while Nav2 was navigating: "
          f"{extra['mean_active_commands_per_step']:.3f}")
    print(f"episodes Nav2 abandoned before the env ended them: {sum(aborted_flags)}/{len(ratios)}")
    if bridge.slam:
        errors = np.asarray(pose_errors, dtype=float)
        extra.update({"pose_error_median": float(np.median(errors)),
                      "pose_error_mean": float(errors.mean()),
                      "pose_error_max": float(errors.max()),
                      "pose_error_per_episode": [float(e) for e in errors],
                      "slam_not_ready_episodes": int(len(ready_flags) - sum(ready_flags)),
                      "slam_ready_per_episode": ready_flags})
        print(f"final pose error: median {extra['pose_error_median']:.3f} m, "
              f"mean {extra['pose_error_mean']:.3f} m, "
              f"max {extra['pose_error_max']:.3f} m")
        print(f"episodes whose goal went out before SLAM was ready: "
              f"{extra['slam_not_ready_episodes']}/{len(ready_flags)}")
    # Same filename convention as run_benchmark.py, so the Nav2 rows join the
    # existing results without a second loader. The SLAM arm gets its own stem
    # rather than overwriting the published privileged row.
    stem = ("nav2" if args.privileges == "full" else "nav2_slam") + (
        "" if args.beams == 360 else f"_b{args.beams}")
    if args.split:
        stem += f"_{args.split}"  # a development run never lands on a scored name
    out = Path(args.out_dir) / f"{args.condition}__{stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "actor": stem,
                "condition": args.condition,
                "privileges": args.privileges,
                "scan_beams": args.beams,
                "realtime_factor_cap": factor or None,
                "split": split,
                "shift": shift or "none",
                "lidar_noise_std": noise,
                "mean_commands_per_step": mean_ratio,
                "zero_command_episodes": zero_cmd,
                **metrics.to_dict(),
                **extra,
                "per_episode": [r.__dict__ for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
