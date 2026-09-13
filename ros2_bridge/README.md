# Nav2 baseline

Runs the real [Nav2](https://docs.nav2.org/) stack over the same task, the same
worlds and the same metrics as every other actor in the study.

## Why this exists

The report's `classical` baseline is a hand-written A\* planner followed by a
pure-pursuit controller. It is *structurally* what Nav2 is — a global plan over
an inflated costmap, a local controller tracking it — but it is not Nav2. That
distinction matters, because the report's headline claim is that a classical
planner beats the learned policies on every condition. If the hand-written
baseline were secretly weak, that claim would shrink to "the learned policies
lose to one particular script", which is worth much less.

So Nav2 runs as one more actor and the two are compared directly.

## What Nav2 is given

Deliberately generous. A baseline that loses through handicap proves nothing,
so where there was a choice, Nav2 got the better side of it:

| | |
|---|---|
| Pose | Ground truth, via TF `map → odom → base_link`. No localisation error. |
| Map | The static occupancy grid, published uninflated so Nav2's own inflation layer is not double-counted. |
| Scan | **360 beams** — between 3× and 22× denser than anything the learned policies see. |
| Kinematics | `nav2_params.yaml` copies the limits out of `RobotConfig`, so the planner never assumes a more agile base than the simulator provides. |
| Goal tolerance | `xy_goal_tolerance: 0.35`, identical to the env's success criterion. |
| Compute | Verified, not assumed — see *Fairness checks* below. |

Movers are absent from `/map` and present in `/scan`, exactly as in the
dynamic-obstacle experiments.

## Fairness checks

Two silent failure modes would have produced a plausible-looking but wrong
Nav2 row, so both are checked at runtime rather than trusted:

- **Sensor corruption reaching Nav2.** The bridge rebuilds the scan at 360
  beams but inherits `noise_std` and `dropout_prob` from the evaluation
  condition. Constructing a fresh `LidarConfig` here would have handed Nav2 a
  clean sensor under `noisy_lidar` and scored the wrong experiment.
- **Control-loop starvation.** The simulator runs faster than real time. If
  Nav2's 10 Hz controller could not keep up, the robot would coast on stale
  commands and Nav2 would lose to a scheduling artefact. Every run reports
  `mean commands per sim step`; 1.0 means the loop was never starved.

A third mode is made loud rather than checked: if Nav2's `/cmd_vel` message
type disagrees with `CMD_VEL_TYPE`, the robot never moves and the run would
otherwise score as a clean sweep of timeouts. The runner aborts after the first
episode with no command received.

## Design notes

**Sim time.** The simulator is a 10 Hz stepped process, not a wall-clock one.
The bridge publishes `/clock` and every Nav2 node runs with `use_sim_time`.
Two consequences:

- The eval process is the only publisher of `/clock`, so it can never block
  on Nav2 — a blocking `waitUntilNav2Active()` would stop the clock the nodes
  are waiting for. Startup polls `<node>/get_state` while pumping the clock.
- Sim time runs continuously across episodes and is never reset to zero. Nav2
  treats a backwards clock as a time jump and purges its TF buffers.

**Readiness.** Waiting on the `navigate_to_pose` action server is not enough:
bt_navigator creates it during *configure*, so the server exists while the node
is still inactive and rejects every goal. The runner waits for
`PRIMARY_STATE_ACTIVE` on all four nodes.

**Termination.** The environment owns termination, not Nav2. Nav2 declaring
success while the robot sits outside the goal tolerance is scored as a failure,
the same as for every other actor.

**Custom launch.** `nav2_launch.py` starts four nodes rather than the eleven in
`nav2_bringup/navigation_launch.py`. That launch requires every node to
configure or the lifecycle manager aborts the whole bringup, and its command
chain is `controller → cmd_vel_nav → velocity_smoother → cmd_vel_smoothed →
collision_monitor → /cmd_vel`, so a missing collision_monitor means no velocity
command ever reaches the simulator. The omitted nodes (velocity smoothing,
collision monitoring) only ever slow the robot down, so leaving them out can
only help Nav2.

## Running it

Nav2 lives in a userspace conda environment (RoboStack) inside WSL, because
this machine has no passwordless sudo and the apt route needs root. Nothing is
installed system-wide; deleting `~/mamba` removes the whole stack.

```bash
bash ros2_bridge/run_nav2.sh --condition nominal --episodes 100
```

Conditions come from `BENCHMARK_CONDITIONS` in `src/vision_nav/envs/splits.py` —
the same table `scripts/run_benchmark.py` scores every other actor against, so
the two cannot drift apart. Results are written as
`results/<condition>__nav2.json`, matching the benchmark's filename
convention.

Then:

```bash
python scripts/nav2_comparison.py
```

## Files

| File | |
|---|---|
| `nav2_bridge.py` | ROS 2 node wrapping `ProceduralNavEnv`: publishes `/clock`, `/scan`, `/map`, `/odom` and TF; applies `/cmd_vel`. |
| `nav2_launch.py` | Minimal four-node bringup. |
| `nav2_params.yaml` | DWB controller + NavFn planner, limits copied from `RobotConfig`. |
| `run_nav2_eval.py` | Episodic runner: same worlds, same seed order, same metrics. |
| `run_nav2.sh` | Launches Nav2 and the runner together, and cleans up after both. |
