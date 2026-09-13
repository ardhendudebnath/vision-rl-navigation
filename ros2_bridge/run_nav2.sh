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

echo "=== launching Nav2 ==="
setsid "$MM" run -n ros_nav2 ros2 launch "$BRIDGE/nav2_launch.py" \
  use_sim_time:=True \
  params_file:="$PARAMS" \
  > "$LOG/nav2.log" 2>&1 &
NAV2_PID=$!

echo "=== running evaluation ==="
"$MM" run -n ros_nav2 python "$BRIDGE/run_nav2_eval.py" "$@"
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo "--- last 40 lines of Nav2 log ---"
  tail -40 "$LOG/nav2.log"
fi
exit $STATUS
