#!/usr/bin/env bash
# Run a long job in WSL detached from whatever started it, so it outlives the
# PowerShell call -- and the session -- that launched it.
#
#   wsl -d Ubuntu-24.04 -- bash tools/wsl/detach.sh TAG -- COMMAND [ARGS...]
#   wsl -d Ubuntu-24.04 -- env ROS_DOMAIN_ID=63 bash tools/wsl/detach.sh TAG -- bash ros2_bridge/run_nav2.sh ...
#
# COMMAND runs from the repository root. Its output goes to $RUNS_DIR/TAG.log
# (default ~/runs), and when it ends its exit status goes to TAG.done; status.sh
# and wait.sh read both. The environment is passed through.
#
# Two traps this exists to avoid, both met in this project:
#  - returning straight after backgrounding lets WSL tear the job down before
#    setsid has detached it, and the job silently never starts; so this waits
#    (DETACH_WAIT seconds, default 20) and reports whether it is running;
#  - the command arrives as an argument list and is quoted here, so PowerShell's
#    quoting of $, quotes and pipes never reaches it.
set -euo pipefail
if [ $# -lt 3 ] || [ "$2" != "--" ]; then
  echo "usage: detach.sh TAG -- COMMAND [ARGS...]" >&2
  exit 2
fi
tag="$1"
shift 2
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runs="${RUNS_DIR:-$HOME/runs}"
mkdir -p "$runs"
rm -f "$runs/$tag.done"
cmd=$(printf '%q ' "$@")
setsid nohup bash -c "cd $(printf '%q' "$repo") && { $cmd; } > $(printf '%q' "$runs/$tag.log") 2>&1; \
echo \"exit \$?\" > $(printf '%q' "$runs/$tag.done")" > /dev/null 2>&1 < /dev/null &
disown
sleep "${DETACH_WAIT:-20}"
if [ -f "$runs/$tag.done" ]; then
  echo "$tag already finished: $(cat "$runs/$tag.done") (log $runs/$tag.log)"
else
  echo "$tag running (log $runs/$tag.log)"
fi
