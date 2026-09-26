"""Is the route this benchmark scores against drivable at the planner's margin?

A question about the *worlds*, with no agent in it. SPL's denominator is the
shortest path on a grid inflated by the robot radius alone -- the shortest route
the robot could physically drive, touching walls if it must. The classical stack
plans at ``robot_radius + safety_margin`` and only falls back to smaller radii
when that fails (`MappedPursuitAgent.reset` tries three).

The forensic raised this: on one `dense` world the truth blocked nine cells of
its *own* shortest route once the planner's margin was applied. If that is
typical, then in clutter the distance the benchmark scores against is not a
distance the planner can drive while keeping the clearance it asks for, and part
of what §9.7 read as indecision is a robot alternating between radii.

Measured per world, mirroring what the agent actually does: the robot's own
footprint counts as free (`_clear_footprint`), and the goal may be relaxed to any
free cell inside the goal tolerance (`_reachable_goal`).

    python scripts/margin_audit.py --episodes 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.classical import PursuitConfig
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.planning.grid_astar import astar_grid
from vision_nav.training.env_factory import build_env_config

CONDITIONS = ("nominal", "dense", "narrow")


def route_length(world, occ: np.ndarray, start_xy, goal_xy) -> float | None:
    """Metres along the 8-connected route through ``occ``, or ``None``."""
    start = tuple(int(v) for v in world.world_to_grid(np.asarray(start_xy)))
    goal = tuple(int(v) for v in world.world_to_grid(np.asarray(goal_xy)))
    occ = clear_footprint(world, occ, start_xy)
    goal = relax_goal(world, occ, goal_xy) or goal
    if occ[start] or occ[goal]:
        return None
    cells = astar_grid(occ, start, goal)
    if cells is None:
        return None
    pts = world.grid_to_world(np.asarray(cells, dtype=int))
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def clear_footprint(world, occ: np.ndarray, position) -> np.ndarray:
    """The robot's own cell and footprint count as free, as the agent does."""
    occ = occ.copy()
    r, c = (int(v) for v in world.world_to_grid(np.asarray(position)))
    k = int(np.ceil(world.config.robot_radius / world.config.grid_resolution))
    occ[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1] = False
    return occ


def relax_goal(world, occ: np.ndarray, goal_xy) -> tuple[int, int] | None:
    """The nearest free cell to the goal inside the goal tolerance."""
    goal = np.asarray(goal_xy, dtype=float)
    gr, gc = (int(v) for v in world.world_to_grid(goal))
    if not occ[gr, gc]:
        return gr, gc
    reach = float(world.config.goal_tolerance)
    k = int(np.ceil(reach / world.config.grid_resolution))
    rows = np.arange(max(gr - k, 0), min(gr + k + 1, occ.shape[0]))
    cols = np.arange(max(gc - k, 0), min(gc + k + 1, occ.shape[1]))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    centres = world.grid_to_world(np.stack([rr.ravel(), cc.ravel()], axis=-1))
    d = np.linalg.norm(centres - goal, axis=1)
    free = (~occ[rr.ravel(), cc.ravel()]) & (d <= reach)
    if not free.any():
        return None
    best = int(np.argmin(np.where(free, d, np.inf)))
    return int(rr.ravel()[best]), int(cc.ravel()[best])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--out", default="results/margin_audit.json")
    args = p.parse_args(argv)

    margin = PursuitConfig().safety_margin
    report: dict = {"split": "val", "episodes": args.episodes,
                    "safety_margin": margin, "conditions": {}}
    print(f"val band, {args.episodes} worlds per condition, "
          f"safety margin {margin:.2f} m\n")
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        rows = []
        for seed in [int(s) for s in list(cfg.world_seeds)[:args.episodes]]:
            env.reset(options={"world_seed": seed})
            world = env.world
            base = world.config.robot_radius
            start, goal = world.start[:2], world.goal
            radii = {"base": base, "half": base + margin * 0.5, "full": base + margin}
            lengths = {name: route_length(world, world.occupancy_at(r), start, goal)
                       for name, r in radii.items()}
            rows.append({"seed": seed,
                         **{f"l_{k}": v for k, v in lengths.items()}})
        report["conditions"][cond] = {"worlds": rows}

        have = {k: [r[f"l_{k}"] for r in rows if r[f"l_{k}"] is not None]
                for k in ("base", "half", "full")}
        n = len(rows)
        entry = {
            "n": n,
            "no_route_at_full": n - len(have["full"]),
            "no_route_at_half": n - len(have["half"]),
            "no_route_at_base": n - len(have["base"]),
        }
        # Where both exist, how much longer is the margin-safe route?
        pairs = [(r["l_base"], r["l_full"]) for r in rows
                 if r["l_base"] and r["l_full"]]
        entry["detour_ratio"] = (float(np.mean([f / b for b, f in pairs]))
                                 if pairs else float("nan"))
        entry["detour_max"] = (float(np.max([f / b for b, f in pairs]))
                               if pairs else float("nan"))
        report["conditions"][cond].update(entry)
        print(f"{cond:8s} of {n} worlds: no margin-safe route in "
              f"{entry['no_route_at_full']:2d}, none at half margin in "
              f"{entry['no_route_at_half']:2d}, none at the bare radius in "
              f"{entry['no_route_at_base']:2d}")
        print(f"         where both exist, the margin-safe route is "
              f"{entry['detour_ratio']:.2f}x the scored one "
              f"(worst {entry['detour_max']:.2f}x)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
