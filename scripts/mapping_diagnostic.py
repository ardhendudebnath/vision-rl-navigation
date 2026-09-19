"""Post hoc: when the mapped planner fails in clutter, is it the map or the planner?

``mapping_experiment.py`` found the classical planner losing 0.28-0.31 of its
success on `dense` and `narrow` once it builds its own map. Before that is read as
a property of mapping, it has to be ruled out as a property of this
implementation. The discriminating fact is what the robot hit:

  - an obstacle it had **already mapped** -- a planning or control fault that
    the map made visible, and a bug to fix before reporting anything;
  - an obstacle it had **not yet mapped** -- the sensor's limit: it cannot plan
    around what its returns never landed on.

For every collision this records whether the contact point lies in, or next to,
a cell the map held as occupied at the moment of contact; and for every episode
the failure mode and how much of the world the map had seen. Nothing here was
predicted, and it is labelled post hoc wherever it is quoted.

    python scripts/mapping_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.mapped import MappedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.occupancy import OCCUPIED
from vision_nav.training.env_factory import build_env_config


def contact_point(world, position: np.ndarray) -> np.ndarray:
    """Where the robot's disc touches the nearest true surface."""
    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    ring = position[None, :] + world.config.robot_radius * np.stack(
        [np.cos(angles), np.sin(angles)], axis=-1)
    return ring[int(np.argmin(world.clearance(ring, include_dynamic=False)))]


def diagnose(cond: str, sensor: str, episodes: int) -> dict:
    split, shift, noise = BENCHMARK_CONDITIONS[cond]
    overrides = {"lidar": {"noise_std": noise}} if noise else {}
    cfg = build_env_config(overrides, split=split, shift=shift, n_worlds=episodes)
    env = ProceduralNavEnv(cfg)
    rows = []
    for seed in cfg.world_seeds:
        env.reset(options={"world_seed": int(seed)})
        agent = MappedPursuitAgent(robot=cfg.robot, sensor=sensor, noise_std=noise)
        if not agent.start_episode(env.world, env.robot.pose):
            rows.append({"outcome": "no_plan"})
            continue
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(agent.act(env.robot.pose))
            done = term or trunc
        outcome = ("success" if info.get("is_success") else
                   "collision" if info.get("collision") else "timeout")
        row = {"outcome": outcome, "known": agent.map.known_fraction, "replans": agent.replans}
        if outcome == "collision":
            touch = contact_point(env.world, env.robot.position)
            r, c = (int(v) for v in agent.map.world_to_grid(touch))
            patch = agent.map.grid[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            row["contact_mapped"] = bool((patch == OCCUPIED).any())
            # How long before contact the obstacle was first mapped: known in
            # time points at the planner, known too late at the sensor.
            first = agent.map.first_occupied[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            first = first[first >= 0]
            row["scans_mapped_before_contact"] = (int(agent.map.scans - first.min())
                                                  if len(first) else None)
            row["mapped_clearance_at_contact"] = float(agent.map.clearance(env.robot.position))
        rows.append(row)
    coll = [r for r in rows if r["outcome"] == "collision"]
    return {
        "episodes": len(rows),
        "success": sum(r["outcome"] == "success" for r in rows) / len(rows),
        "collision": len(coll) / len(rows),
        "timeout": sum(r["outcome"] == "timeout" for r in rows) / len(rows),
        "collisions": len(coll),
        "collisions_into_mapped": sum(r["contact_mapped"] for r in coll),
        "collisions_into_unmapped": sum(not r["contact_mapped"] for r in coll),
        "median_known": float(np.median([r["known"] for r in rows if "known" in r])),
        # Mapped at least a second (10 scans) before contact, versus in the last
        # half second -- too late to stop at 0.6 m/s.
        "collisions_mapped_1s_before": sum(
            (r.get("scans_mapped_before_contact") or 0) >= 10 for r in coll),
        "collisions_mapped_last_0p5s": sum(
            (r.get("scans_mapped_before_contact") or 0) < 5 for r in coll),
        "median_scans_mapped_before_contact": float(np.median(
            [r["scans_mapped_before_contact"] for r in coll
             if r.get("scans_mapped_before_contact") is not None])) if coll else None,
        "median_replans": float(np.median([r["replans"] for r in rows if "replans" in r])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/mapping_diagnostic.json")
    args = p.parse_args(argv)
    report = {}
    for cond in ("dense", "narrow"):
        for sensor in ("lidar32", "camera64"):
            d = diagnose(cond, sensor, args.episodes)
            report[f"{cond}/{sensor}"] = d
            print(f"{cond:7s} {sensor:9s} success {d['success']:.3f} collision {d['collision']:.3f} "
                  f"timeout {d['timeout']:.3f} | collisions into mapped {d['collisions_into_mapped']}"
                  f" / unmapped {d['collisions_into_unmapped']} | map seen {d['median_known']:.2f}"
                  f" | replans {d['median_replans']:.0f} | mapped >=1 s before contact "
                  f"{d['collisions_mapped_1s_before']}, in the last 0.5 s "
                  f"{d['collisions_mapped_last_0p5s']}, median "
                  f"{d['median_scans_mapped_before_contact']} scans", flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
