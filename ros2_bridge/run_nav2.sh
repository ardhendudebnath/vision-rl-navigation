#!/usr/bin/env bash
# Launch Nav2 against the simulator bridge and run an evaluation.
#
#   bash ros2_bridge/run_nav2.sh [--condition nominal] [--episodes 100]
#
# Conditions come from BENCHMARK_CONDITIONS, the same table run_benchmark.py
# scores every other actor against.
#
# Several conditions can run at once: give each a distinct ROS_DOMAIN_ID and
# NAV2_LOG_DIR. The simulator is rate-limited rather than CPU-bound, so
# parallel runs cost almost nothing — but check that the reported
# "commands per sim step" stays near 1.0, which is what would fall if the
# parallel runs ever did starve Nav2's control loop.
#
# ROS 2 Jazzy + Nav2 live in a userspace conda env (RoboStack), because this
# WSL image has no passwordless sudo and the apt route needs root. Nothing
# here touches the system; deleting ~/mamba removes the whole stack.
set -uo pipefail

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
MM="${MM:-$HOME/bin/micromamba}"
REPO="${REPO:-/mnt/c/Users/ardhendudebnath/OneDrive/Documents/isaac-vision-nav}"
BRIDGE="$REPO/ros2_bridge"

# Keep DDS traffic on loopback: WSL's bridged interfaces otherwise cause
# multicast discovery to hang or leak onto the host network.
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export PYTHONPATH="$REPO/src:$BRIDGE${PYTHONPATH:+:$PYTHONPATH}"

LOG="${NAV2_LOG_DIR:-$HOME/nav2_logs}"
mkdir -p "$LOG"
echo "logs: $LOG"

# DWB_SIM_TIME overrides the controller's rollout horizon, for the sweep in
# report Section 9.2. Written into a copy of the params rather than passed as
# a launch argument, because the value lives inside the FollowPath plugin
# block and ros2 launch cannot reach a nested key.
PARAMS="$BRIDGE/nav2_params.yaml"
if [ -n "${DWB_SIM_TIME:-}" ]; then
  PARAMS="$LOG/nav2_params_sim${DWB_SIM_TIME}.yaml"
  sed "s/^\( *sim_time:\) *[0-9.]*/\1 ${DWB_SIM_TIME}/" \
    "$BRIDGE/nav2_params.yaml" > "$PARAMS"
  echo "DWB sim_time -> ${DWB_SIM_TIME}s  ($PARAMS)"
  grep -n "sim_time:" "$PARAMS"
fi

# Killing the micromamba wrapper leaves the ros2 launch children running, and
# a stale controller_server poisons the next run by answering on the same
# domain. `setsid` puts the launch in its own process group so the whole tree
# dies together — killing by process name instead would also take out any
# sibling run evaluating a different condition in parallel.
cleanup() {
  [ -n "${NAV2_PID:-}" ] && kill -- -"$NAV2_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

# The SLAM arm (report §9.4). `--privileges slam` has to reach both the launch,
# which then starts slam_toolbox, and the evaluation, which stops publishing the
# map and the true pose. Passing it to only one of them would silently produce
# a robot with no map and no localiser, or two publishers of map -> odom.
SLAM=False
for arg in "$@"; do
  [ "$arg" = "slam" ] && SLAM=True
done
# Under SLAM, navigation starts after slam_toolbox has had time to publish the
# map frame; see the comment above `navigation` in nav2_launch.py. And the
# global costmap becomes a rolling window over the whole arena, derived from
# the published params in code; see make_slam_params.py for why.
DELAY=0.0
if [ "$SLAM" = True ]; then
  DELAY="${NAV2_SLAM_DELAY:-15.0}"
  SLAM_PARAMS="$LOG/nav2_params_slam.yaml"
  "$MM" run -n ros_nav2 python "$BRIDGE/make_slam_params.py" "$PARAMS" "$SLAM_PARAMS" || exit 1
  PARAMS="$SLAM_PARAMS"
fi

# The evaluation starts first and Nav2 only once it is publishing. Started the
# other way round, Nav2's costmaps look for the odom frame, give up after half a
# second, and the lifecycle manager aborts the whole bringup; that happened when
# six runs started together and one runner waited on micromamba's lock for
# longer than Nav2 would. The runner touches NAV2_READY_FILE when /clock, /scan
# and TF are flowing.
export NAV2_READY_FILE="$LOG/runner_publishing"
rm -f "$NAV2_READY_FILE"
echo "=== running evaluation ==="
"$MM" run -n ros_nav2 python "$BRIDGE/run_nav2_eval.py" "$@" &
EVAL_PID=$!
for _ in $(seq 1 600); do
  [ -f "$NAV2_READY_FILE" ] && break
  kill -0 "$EVAL_PID" 2>/dev/null || break
  sleep 0.5
done

echo "=== launching Nav2 (slam=$SLAM, navigation delay ${DELAY}s) ==="
setsid "$MM" run -n ros_nav2 ros2 launch "$BRIDGE/nav2_launch.py" \
  use_sim_time:=True \
  params_file:="$PARAMS" \
  slam:="$SLAM" \
  slam_params_file:="$BRIDGE/slam_params.yaml" \
  nav2_delay:="$DELAY" \
  > "$LOG/nav2.log" 2>&1 &
NAV2_PID=$!

wait "$EVAL_PID"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "--- last 40 lines of Nav2 log ---"
  tail -40 "$LOG/nav2.log"
fi
exit $STATUS
