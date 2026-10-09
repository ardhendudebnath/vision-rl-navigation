#!/usr/bin/env bash
# Regenerate gazebo_tb3/gz_waffle.sdf from Nav2's TurtleBot3 Waffle xacro.
#
#   bash gazebo_tb3/expand_model.sh
#
# Runs in the ros_tb3 RoboStack env (see gazebo_tb3/README.md).
set -euo pipefail
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
MM="${MM:-$HOME/bin/micromamba}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SHARE="$MAMBA_ROOT_PREFIX/envs/ros_tb3/share/nav2_minimal_tb3_sim"
"$MM" run -n ros_tb3 xacro "$SHARE/urdf/gz_waffle.sdf.xacro" namespace:= > "$HERE/gz_waffle.sdf.new"
echo "wrote $HERE/gz_waffle.sdf.new -- add the provenance comment and replace gz_waffle.sdf"
