#!/usr/bin/env bash
# One arm of gazebo_tb3/transfer.py in the ros_tb3 environment, after clearing
# any Gazebo server a dead run left behind. Arguments go to transfer.py.
#
#   wsl -d Ubuntu-24.04 -- bash tools/wsl/detach.sh gz_odom -- bash tools/wsl/transfer_arm.sh \
#       --arm policy_gazebo_odom --split val --episodes 100 --workers 4 \
#       --out results/gazebo_tb3/transfer_policy_gazebo_odom.json
#
# One Gazebo run at a time: the clean-up stops every Gazebo server on the
# machine, and Gazebo paces itself against the wall clock besides, so a second
# CPU-heavy job alongside slows it. transfer.py is resumable, so a run stopped
# partway is continued by running the same command again.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
# "[g]" keeps the pattern from matching a command line that merely contains the
# text, such as a `bash -c` wrapper's, which once killed a run before it started.
pkill -f "[g]z sim" 2>/dev/null
sleep 2
exec "${MM:-$HOME/bin/micromamba}" run -n ros_tb3 python gazebo_tb3/transfer.py "$@"
