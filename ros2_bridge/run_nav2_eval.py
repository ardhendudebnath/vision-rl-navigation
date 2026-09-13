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
import sys
from pathlib import Path

import time

import rclpy
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from nav2_msgs.srv import ClearEntireCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.executors import SingleThreadedExecutor

from nav2_bridge import CMD_VEL_TYPE, Nav2Bridge, yaw_to_quaternion
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition", default="nominal", choices=list(CONDITIONS),
                   help="Evaluation condition, from the shared benchmark table")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out-dir", default="results")
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


def run_episode(bridge: Nav2Bridge, nav: BasicNavigator, executor, seed: int,
                clear_clients) -> EpisodeResult:
    """One episode: reset the world, hand Nav2 the goal, step until resolved."""
    bridge.start_episode(seed)
    world = bridge.env.world

    # Publish the new layout before clearing, so the static layer resets to
    # this world rather than re-applying the previous one.
    for _ in range(MAP_SETTLE_STEPS):
        pump(bridge, executor)
    if not clear_costmaps(bridge, executor, clear_clients):
        print("  warning: costmap clear timed out; episode may inherit "
              "obstacles from the previous world", file=sys.stderr)

    # No setInitialPose: there is no localiser to seed. The bridge broadcasts
    # ground-truth map->odom->base_link, the same privilege the hand-written
    # baseline gets, so localisation error is not part of the comparison.
    # Warm up so the costmaps hold a scan before the first plan is made.
    for _ in range(WARMUP_STEPS):
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.02)

    nav.goToPose(pose_msg(world.goal[0], world.goal[1], 0.0, bridge.now_msg()))

    # The *env* owns termination, not Nav2. Nav2 declaring success while the
    # robot sits outside goal_tolerance would otherwise be scored as a win.
    running = True
    before = bridge.commands_received
    while running:
        bridge.publish_all()
        executor.spin_once(timeout_sec=0.02)
        running = bridge.step_sim()
        if nav.isTaskComplete():
            result = nav.getResult()
            if result != TaskResult.SUCCEEDED and running:
                # Nav2 gave up (planning failure, recovery exhausted). Keep
                # stepping so the episode ends on the env's own terms.
                bridge.env._steps = bridge.env.config.max_episode_steps
                running = bridge.step_sim()

    # Commands per sim step. The simulator runs faster than real time, so a
    # ratio well below 1.0 would mean Nav2's 10 Hz control loop was starved of
    # wall-clock time and the robot coasted on stale commands — a handicap,
    # not a result. Reported alongside the metrics so the claim is checkable.
    bridge.command_ratio = (bridge.commands_received - before) / max(bridge.steps, 1)
    return EpisodeResult.from_info(bridge.last_info)


def main(argv=None) -> int:
    args = parse_args(argv)
    rclpy.init()

    split, shift, noise = CONDITIONS[args.condition]
    overrides: dict = {"lidar": {"noise_std": noise}} if noise else {}
    if args.condition in FROZEN_CONDITIONS:
        overrides["freeze_dynamic"] = True
    env_config = build_env_config(overrides, split=split, shift=shift,
                                  n_worlds=args.episodes)
    seeds = list(env_config.world_seeds)

    bridge = Nav2Bridge(env_config)
    nav = BasicNavigator()
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(nav)

    print(f"waiting for Nav2 to activate ({len(seeds)} episodes queued)...", flush=True)
    bridge.start_episode(seeds[0])

    started = time.monotonic()
    if not wait_until_active(bridge, executor, REQUIRED_NODES):
        return 1
    while not nav.nav_to_pose_client.wait_for_server(timeout_sec=0.0):
        pump(bridge, executor)
    print(f"Nav2 active after {time.monotonic() - started:.0f}s wall", flush=True)

    clear_clients = [bridge.create_client(ClearEntireCostmap, s)
                     for s in CLEAR_SERVICES]
    for client, name in zip(clear_clients, CLEAR_SERVICES):
        while not client.service_is_ready():
            pump(bridge, executor)
            if time.monotonic() - started > 240:
                print(f"costmap-clear service never appeared: {name}",
                      file=sys.stderr)
                return 1

    results = []
    ratios = []
    try:
        for i, seed in enumerate(seeds, 1):
            r = run_episode(bridge, nav, executor, seed, clear_clients)
            ratios.append(bridge.command_ratio)
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
            print(
                f"  [{i:>3}/{len(seeds)}] world {seed}: "
                f"{'success' if r.success else ('collision' if r.collision else 'timeout')} "
                f"in {r.steps} steps (cmd/step {bridge.command_ratio:.2f})",
                flush=True,
            )
    finally:
        metrics = aggregate(results) if results else None
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
    print(metrics.as_table(f"Nav2 | condition={args.condition}"))
    print(f"\nmean commands per sim step: {mean_ratio:.3f}")
    print(f"episodes with no command at all: {zero_cmd}/{len(ratios)}"
          + ("  <-- Nav2 could not plan; check the costmaps" if zero_cmd else ""))
    # Same filename convention as run_benchmark.py, so the Nav2 rows join the
    # existing results without a second loader.
    out = Path(args.out_dir) / f"{args.condition}__nav2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "actor": "nav2",
                "condition": args.condition,
                "split": split,
                "shift": shift or "none",
                "lidar_noise_std": noise,
                "mean_commands_per_step": mean_ratio,
                "zero_command_episodes": zero_cmd,
                **metrics.to_dict(),
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
