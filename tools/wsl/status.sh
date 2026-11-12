#!/usr/bin/env bash
# Where a detached job is: finished or not, the end of its log, the machine.
#
#   wsl -d Ubuntu-24.04 -- bash tools/wsl/status.sh TAG [LINES]
runs="${RUNS_DIR:-$HOME/runs}"
tag="$1"
if [ -f "$runs/$tag.done" ]; then echo "$tag finished: $(cat "$runs/$tag.done")"; else echo "$tag running"; fi
tail -n "${2:-12}" "$runs/$tag.log" 2>/dev/null
# The bracketed first letter keeps each pattern from matching this line itself.
echo "gazebo servers $(pgrep -fc '[g]z sim'); nav2 evaluators $(pgrep -fc '[r]un_nav2_eval'); load $(cut -d' ' -f1-3 /proc/loadavg); free $(free -m | awk '/^Mem:/ {print $7}') MB"
