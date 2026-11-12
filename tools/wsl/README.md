# Long runs in WSL

The Nav2 runs ([`ros2_bridge/`](../../ros2_bridge/)) and the Gazebo TurtleBot3
runs ([`gazebo_tb3/`](../../gazebo_tb3/)) live in WSL Ubuntu-24.04 and take
from half an hour to several. These start them so they survive the call that
launched them, and let a Windows-side shell check on them or wait for them.

| script | what it does |
|---|---|
| [`detach.sh`](detach.sh) | `detach.sh TAG -- COMMAND...`: run COMMAND from the repository root, detached; log to `~/runs/TAG.log`, exit status to `~/runs/TAG.done` |
| [`status.sh`](status.sh) | `status.sh TAG [LINES]`: finished or not, the end of the log, Gazebo and Nav2 processes, load and free memory |
| [`wait.sh`](wait.sh) | `wait.sh TAG`: block until the job finishes, then print its status |
| [`transfer_arm.sh`](transfer_arm.sh) | one arm of `gazebo_tb3/transfer.py` in the `ros_tb3` environment, after clearing stale Gazebo servers |

From PowerShell:

```powershell
wsl -d Ubuntu-24.04 -- bash tools/wsl/detach.sh gz_odom -- bash tools/wsl/transfer_arm.sh --arm policy_gazebo_odom --split val --episodes 100 --workers 4 --out results/gazebo_tb3/transfer_policy_gazebo_odom.json
wsl -d Ubuntu-24.04 -- env ROS_DOMAIN_ID=63 bash tools/wsl/detach.sh more_worlds -- bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 200 --first 100 --privileges slam --out-dir results/nav2_more_worlds
wsl -d Ubuntu-24.04 -- bash tools/wsl/status.sh gz_odom
```

What these encode, each learned the hard way:

- **Wait before returning.** A job backgrounded by a call that exits at once can
  be torn down with it before `setsid` has detached it, and silently never
  start; `detach.sh` waits 20 s and says whether it is running.
- **Commands are arguments, not strings.** PowerShell's handling of `$`, quotes
  and `|` mangles a command passed as one string to `bash -c`; `detach.sh`
  quotes the argument list itself.
- **One wall-clock-paced job at a time.** Nav2 and Gazebo pace themselves
  against the wall clock. Beside a CPU-heavy job, Nav2 has failed to come up
  and Gazebo has slowed; give them the machine.
- **Separate ROS domains** (`ROS_DOMAIN_ID`, `NAV2_LOG_DIR`) for Nav2 runs that
  must overlap, and separate `GZ_PARTITION`s for Gazebo servers (which
  `transfer.py` sets per worker).
- **Resume, don't restart.** `transfer.py` keeps every finished episode in its
  output and the registered scripts keep theirs in a `*.checkpoint.jsonl` file
  ([`scripts/resumable.py`](../../scripts/resumable.py)), so a stopped run is
  continued by running the same command again.
