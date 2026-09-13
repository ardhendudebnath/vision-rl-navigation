#!/usr/bin/env bash
# Install ROS 2 Jazzy + Nav2 into a userspace conda environment, for the Nav2
# baseline in report Section 4.1.
#
#   bash scripts/install_ros2_nav2.sh
#   bash ros2_bridge/run_nav2.sh --condition narrow --episodes 100
#
# Why not apt: the documented ROS 2 route needs root, and the machine this was
# built on has no passwordless sudo inside WSL. RoboStack ships ROS 2 as
# conda-forge packages, so the whole stack installs under $HOME. Nothing here
# touches the system and `rm -rf ~/mamba ~/bin/micromamba` removes all of it.
#
# Verified on WSL2 Ubuntu 24.04 with the versions pinned below (937 packages).
set -euo pipefail

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
MM_BIN="${MM_BIN:-$HOME/bin/micromamba}"
ENV_NAME="${ENV_NAME:-ros_nav2}"
export MAMBA_ROOT_PREFIX

# Pinned rather than latest: RoboStack rebuilds move, and an unpinned install
# is not a reproduction of the reported numbers.
ROS_DISTRO_PKGS=(
  "ros-jazzy-ros-base=0.11.0"
  "ros-jazzy-navigation2=1.3.12"
  "ros-jazzy-nav2-bringup=1.3.12"
  "ros-jazzy-nav2-simple-commander=1.3.12"
)

echo "=== 1/3  micromamba ==="
if [ -x "$MM_BIN" ]; then
  echo "already present: $("$MM_BIN" --version)"
else
  mkdir -p "$(dirname "$MM_BIN")"
  # The official installer pipes through bzip2, which this image lacks; Python
  # is guaranteed present here and extracts the same archive.
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/mm.tar.bz2
  python3 - "$MM_BIN" <<'PY'
import sys, tarfile
with tarfile.open("/tmp/mm.tar.bz2", "r:bz2") as t:
    member = t.extractfile("bin/micromamba")
    open(sys.argv[1], "wb").write(member.read())
PY
  chmod +x "$MM_BIN"
  echo "installed: $("$MM_BIN" --version)"
fi

echo "=== 2/3  ROS 2 Jazzy + Nav2 (~937 packages, several minutes) ==="
if "$MM_BIN" env list | grep -qE "^ *$ENV_NAME "; then
  echo "environment '$ENV_NAME' already exists; skipping"
else
  "$MM_BIN" create -y -n "$ENV_NAME" \
    -c https://repo.prefix.dev/robostack-jazzy -c conda-forge \
    python=3.12 "${ROS_DISTRO_PKGS[@]}"
fi

echo "=== 3/3  verify ==="
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# vision_nav is imported from source, not installed: the ROS environment needs
# the simulator and the metrics, but not torch or Stable-Baselines3, which
# env_factory imports lazily for exactly this reason.
PYTHONPATH="$REPO/src" "$MM_BIN" run -n "$ENV_NAME" python - <<'PY'
import rclpy                                        # noqa: F401
from nav2_simple_commander.robot_navigator import BasicNavigator  # noqa: F401
from vision_nav.envs import NavEnvConfig, ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS, DYNAMIC_CONDITIONS

env = ProceduralNavEnv(NavEnvConfig())
env.reset(options={"world_seed": 20000})
print("ok: rclpy, nav2_simple_commander and vision_nav all import")
print("conditions available:",
      ", ".join(list(BENCHMARK_CONDITIONS) + list(DYNAMIC_CONDITIONS)))
PY

cat <<EOF

Done. Run the Nav2 baseline with:

  bash ros2_bridge/run_nav2.sh --condition narrow --episodes 100

Results land in results/<condition>__nav2.json. See ros2_bridge/README.md for
what Nav2 is given and the two failure modes the harness guards against.
EOF
