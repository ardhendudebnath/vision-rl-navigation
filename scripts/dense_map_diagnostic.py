"""What does a dense scan do to this stack's map?

Report §9.5 left the mapper as the whole of what this stack pays at 360 beams:
its map gets *worse* with ten times the returns under sensor noise (0.940 to
0.830 on `noisy_lidar`), and in clutter it stays about 0.15 behind Nav2 with
slam_toolbox. The obvious explanation -- dense scans pile up evidence faster --
is wrong on reading the code: hits and misses are one vote per cell per scan,
so the evidence *rate* does not depend on the beam count. What a dense scan
changes is which cells get marked.

This measures the map itself rather than the navigating, against the world it
was built from, at the end of every episode:

  phantoms      occupied cells with no real surface inside them, and how far
                they sit from the nearest one
  thickening    occupied cells that do hold a surface, and the spread of the
                true distance from the cell centre to it
  blocked       free floor lost to inflation around everything mapped, against
                the same figure for the true map -- the quantity a planner
                actually feels
  churn         cells marked and unmarked per scan

Run on the ``val`` seed band, which no experiment scores.

    python scripts/dense_map_diagnostic.py --episodes 12
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

CONDITIONS = ("noisy_lidar", "dense", "narrow", "nominal")
SENSORS = ("lidar32", "lidar360")


def measure(agent: MappedPursuitAgent, world) -> dict:
    """The map it ended with, against the world it was built from."""
    omap = agent.map
    occupied = np.argwhere(omap.grid == OCCUPIED)
    out: dict = {"occupied": int(len(occupied)), "known": float(omap.known_fraction),
                 "replans": int(agent.replans), "scans": int(omap.scans)}
    if len(occupied):
        centres = omap.grid_to_world(occupied)
        # Distance from each mapped cell's centre to the nearest real surface.
        d = world.clearance(centres, include_dynamic=False)
        phantom = d > omap.half_diagonal
        out.update({
            "phantom_cells": int(phantom.sum()),
            "phantom_fraction": float(phantom.mean()),
            "phantom_distance_median": float(np.median(d[phantom])) if phantom.any() else 0.0,
            "real_cell_distance_median": float(np.median(d[~phantom])) if (~phantom).any() else 0.0,
        })
    # Free floor the planner cannot use, against the same figure for the truth.
    radius = omap.robot_radius + agent.config.safety_margin
    mine = omap.occupancy_at(radius)
    truth = world.occupancy_at(radius, include_dynamic=False)
    free_truth = ~truth
    out["blocked_fraction_mapped"] = float(mine.mean())
    out["blocked_fraction_true"] = float(truth.mean())
    # Floor that is really free and really seen, but blocked on this map.
    seen = omap.grid != -1
    lost = mine & free_truth & seen
    out["free_floor_lost"] = float(lost.sum() / max(free_truth.sum(), 1))
    return out


def run(cond: str, sensor: str, episodes: int, corroborate: bool = False) -> list[dict]:
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split="val", shift=shift, n_worlds=episodes)
    env = ProceduralNavEnv(cfg)
    rows = []
    for seed in list(cfg.world_seeds)[:episodes]:
        env.reset(options={"world_seed": int(seed)})
        agent = MappedPursuitAgent(robot=cfg.robot, sensor=sensor, noise_std=noise,
                                   corroborate=corroborate)
        agent.start_episode(env.world, env.robot.pose)
        info: dict = {}
        for _ in range(cfg.max_episode_steps):
            _, _, terminated, truncated, info = env.step(agent.act(env.robot.pose))
            if terminated or truncated:
                break
        row = measure(agent, env.world)
        row["success"] = bool(info.get("is_success"))
        rows.append(row)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--out", default="results/dense_map_diagnostic.json")
    args = p.parse_args(argv)

    report: dict = {"episodes": args.episodes, "split": "val", "conditions": {}}
    keys = ("occupied", "phantom_cells", "phantom_fraction", "phantom_distance_median",
            "real_cell_distance_median", "blocked_fraction_mapped", "blocked_fraction_true",
            "free_floor_lost", "known", "replans")
    # The dense scanner with the corroboration rule on as well, so the repair
    # can be read against what it was meant to repair.
    arms = [(s, False) for s in SENSORS] + [("lidar360", True)]
    for cond in CONDITIONS:
        report["conditions"][cond] = {}
        for sensor, corroborate in arms:
            rows = run(cond, sensor, args.episodes, corroborate)
            summary = {k: float(np.mean([r.get(k, 0.0) for r in rows])) for k in keys}
            summary["success"] = float(np.mean([r["success"] for r in rows]))
            name = sensor + ("_corroborated" if corroborate else "")
            report["conditions"][cond][name] = summary
            print(f"{cond:12s} {name:22s} SR {summary['success']:.2f}  "
                  f"occupied {summary['occupied']:6.0f}  "
                  f"phantoms {summary['phantom_cells']:6.0f} "
                  f"({summary['phantom_fraction']:.2f})  "
                  f"phantom dist {summary['phantom_distance_median']:.2f} m  "
                  f"blocked {summary['blocked_fraction_mapped']:.3f} "
                  f"(true {summary['blocked_fraction_true']:.3f})  "
                  f"free floor lost {summary['free_floor_lost']:.3f}  "
                  f"replans {summary['replans']:.0f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
