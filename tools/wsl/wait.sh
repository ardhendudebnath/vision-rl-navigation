#!/usr/bin/env bash
# Block until a detached job finishes, then print its status. Meant for a
# backgrounded call on the Windows side that should return when the job does.
#
#   wsl -d Ubuntu-24.04 -- bash tools/wsl/wait.sh TAG [LINES]
runs="${RUNS_DIR:-$HOME/runs}"
until [ -f "$runs/$1.done" ]; do sleep 30; done
bash "$(dirname "${BASH_SOURCE[0]}")/status.sh" "$@"
