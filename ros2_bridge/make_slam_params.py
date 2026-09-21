"""Derive the SLAM arm's Nav2 parameters from nav2_params.yaml.

Written as code rather than kept as a second YAML file so that the two arms
cannot drift apart: every setting is the published arm's except the ones
changed below, and each change says why.

    python ros2_bridge/make_slam_params.py nav2_params.yaml out.yaml

One change, to the global costmap. With the map handed over, Nav2 sizes the
global costmap to that map, which covers the whole arena. Under SLAM the map is
whatever slam_toolbox has seen so far, and a costmap sized to it does not
contain a goal the robot has not yet looked at: the first smoke test failed
exactly so, "Goal Coordinates of (5.75, 1.05) was outside bounds", and the goal
was aborted before the robot moved. A rolling window large enough to hold the
whole arena from anywhere inside it restores what the published arm had —
somewhere to plan to — and leaves unknown space free, which is how both that
arm and the hand-written mapping stack of report §9.2 plan into the unexplored.
"""

from __future__ import annotations

import sys

import yaml

#: The largest arena is 16 m square (``SHIFTS["large"]``), so every point in it
#: is within 16 m of every other along each axis. A window of 34 m centred on
#: the robot therefore always contains the goal, with a metre to spare.
WINDOW_M = 34


def derive(params: dict) -> dict:
    g = params["global_costmap"]["global_costmap"]["ros__parameters"]
    g["rolling_window"] = True
    g["width"] = WINDOW_M
    g["height"] = WINDOW_M
    return params


def main(argv: list[str]) -> int:
    src, dst = argv[1], argv[2]
    with open(src, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    with open(dst, "w", encoding="utf-8") as fh:
        yaml.safe_dump(derive(params), fh, sort_keys=False)
    print(f"SLAM params -> {dst} (global costmap: rolling {WINDOW_M} m window)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
