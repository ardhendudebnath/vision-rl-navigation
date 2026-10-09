"""What is a "stale" rebuild? A follow-up to stall_diagnostic.py, added afterwards.

``stall_diagnostic.py`` classed each plan invalidation by the cells blocking the
route that were *not occupied when the plan was made*, and 77% to 85% of
rebuilds came out "stale". Its classifier gave that label to two different
things, which is why it explained nothing:

  pre-existing  the route ahead is blocked, but only by cells that were already
                occupied when the plan was made. The planner clears the robot's
                own footprint before searching (``_clear_footprint``), so a plan
                made with the robot inside a wall's margin starts inside it; the
                validity check does not clear the footprint, so at the next
                change anywhere in the map it finds the route's first points
                inside the margin and rebuilds at once
  deferred      nothing blocks the route when the rebuild fires: a blockage
                beyond the one-metre urgent horizon was remembered by the rate
                limit and had gone by the time the deferral ran out

This separates them, and for the first measures how deep inside the margin the
blocked point is and where along the plan it lies; for the second it catches
the blockage at its onset and asks what became of its cells (cleared by the map,
or driven past). It also records, every step, whether the point the robot steers
from is inside the margin of something mapped -- the condition the first kind
needs.

Not registered: written after seeing the stall diagnostic's result. It runs the
published arm only, on the same 200 val worlds, and its identity control is that
every episode's replan count and outcome match ``results/stall_diagnostic.json``.

    python scripts/stale_rebuild_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clutter_forensic import remaining_at  # noqa: E402
from stall_diagnostic import STALL_DISTANCE, make_agent  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.occupancy import OCCUPIED  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

KINDS = ("new cells", "pre-existing", "deferred")


class Watcher:
    """Wraps one agent's replanning; records what each rebuild was answering."""

    def __init__(self, agent) -> None:
        self.agent = agent
        self.events: list[dict] = []
        self.grid_at_plan: np.ndarray | None = None
        self.onset: dict | None = None
        reset, blocked = agent.reset, agent._path_ahead_blocked

        def watched_reset(world_, pose, *args, **kwargs):
            ok = reset(world_, pose, *args, **kwargs)
            if ok:
                self.grid_at_plan = agent.map.grid.copy()
                self.onset = None
            return ok

        def watched_blocked(position):
            was = agent._pending or agent._urgent
            fired = blocked(position)
            state = self.state()
            if (agent._pending or agent._urgent or fired) and not was and self.onset is None:
                self.onset = {"step": agent._steps, "cells": state["blocking"]}
            if fired:
                self.events.append(self.describe(state))
            return fired

        agent.reset = watched_reset
        agent._path_ahead_blocked = watched_blocked

    def state(self) -> dict:
        """The route ahead against the map, now."""
        agent, omap = self.agent, self.agent.map
        if agent._track is None:
            return {"invalid": np.zeros(0, dtype=int), "blocking": np.zeros((0, 2), dtype=int)}
        ahead = agent._track[agent._cursor:]
        radius = float(agent._plan_radius)
        clearance = omap.clearance(ahead) if len(ahead) else np.zeros(0)
        invalid = np.flatnonzero(clearance < radius - 1e-9)
        blocking = np.zeros((0, 2), dtype=int)
        if len(invalid):
            occ = np.argwhere(omap.grid == OCCUPIED)
            centres = omap.grid_to_world(occ)
            near = (np.linalg.norm(centres - ahead[invalid[0]], axis=1)
                    <= radius + omap.half_diagonal + 1e-9)
            blocking = occ[near]
        return {"invalid": invalid, "blocking": blocking,
                "depth": float(radius - clearance[invalid[0]]) if len(invalid) else 0.0}

    def describe(self, state: dict) -> dict:
        agent, omap = self.agent, self.agent.map
        spacing = agent.config.track_spacing
        event = {"step": agent._steps}
        if len(state["invalid"]):
            cells = state["blocking"]
            new = (self.grid_at_plan[cells[:, 0], cells[:, 1]] != OCCUPIED
                   if self.grid_at_plan is not None and len(cells) else np.ones(len(cells), bool))
            event["kind"] = "new cells" if new.any() else "pre-existing"
            # Where the blocked point lies along the plan, from the point the
            # plan began, and how far the robot has come along it.
            event["from_plan_start"] = float((agent._cursor + state["invalid"][0]) * spacing)
            event["robot_along"] = float(agent._cursor * spacing)
            event["depth"] = state["depth"]
            return event
        event["kind"] = "deferred"
        if self.onset is None or not len(self.onset["cells"]):
            event["resolution"] = "no onset"
            return event
        cells = self.onset["cells"]
        still = omap.grid[cells[:, 0], cells[:, 1]] == OCCUPIED
        event["resolution"] = "cleared" if not still.any() else "passed or moved"
        event["delay"] = agent._steps - self.onset["step"]
        return event


def episode(args: tuple) -> dict:
    cond, seed, n = args
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split="val", shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution
    agent = make_agent("own_pose", cfg)
    watch = Watcher(agent)
    agent.start_episode(world, env.robot.pose)
    info: dict = {}
    remaining, inside = [], []
    full = agent.map.robot_radius + agent.config.safety_margin
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        remaining.append(remaining_at(field, world, env.robot.pose[:2]))
        # Inside the full margin of something mapped, measured where the robot
        # believes it is -- the point every plan and every check starts from.
        # Scan matching writes its corrected estimate back to the odometry.
        here = np.asarray(agent._odom.pose[:2], dtype=float)
        inside.append(bool(agent.map.clearance(here) < full))
        if terminated or truncated:
            break
    rem = np.asarray(remaining)
    ev = watch.events
    pre = [e for e in ev if e["kind"] == "pre-existing"]
    deferred = [e for e in ev if e["kind"] == "deferred"]
    return {
        "cond": cond, "seed": seed, "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")), "replans": int(agent.replans),
        "steps": len(rem),
        "final_remaining": float(rem[-1]) if np.isfinite(rem[-1]) else float("nan"),
        "kinds": dict(Counter(e["kind"] for e in ev)),
        "pre_from_start": [e["from_plan_start"] for e in pre],
        "pre_robot_along": [e["robot_along"] for e in pre],
        "pre_depth": [e["depth"] for e in pre],
        "deferred_resolution": dict(Counter(e["resolution"] for e in deferred)),
        "inside_margin_share": float(np.mean(inside)) if inside else float("nan"),
    }


def summarise(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    kinds = Counter()
    res = Counter()
    for r in rows:
        kinds.update(r["kinds"])
        res.update(r["deferred_resolution"])
    total = sum(kinds.values())
    pre_start = [x for r in rows for x in r["pre_from_start"]]
    pre_along = [x for r in rows for x in r["pre_robot_along"]]
    pre_depth = [x for r in rows for x in r["pre_depth"]]
    deferred = sum(res.values())
    steps = sum(r["steps"] for r in rows)
    return {
        "n": len(rows),
        "rebuilds": total,
        "per_100_steps": 100.0 * total / steps,
        "shares": {k: kinds[k] / total if total else float("nan") for k in KINDS},
        "pre_existing_per_100_steps": 100.0 * kinds["pre-existing"] / steps,
        "pre_from_plan_start_median": float(np.median(pre_start)) if pre_start else float("nan"),
        "pre_within_half_metre_of_start": (float(np.mean(np.asarray(pre_start) <= 0.5))
                                          if pre_start else float("nan")),
        "pre_robot_along_median": float(np.median(pre_along)) if pre_along else float("nan"),
        "pre_depth_median": float(np.median(pre_depth)) if pre_depth else float("nan"),
        "deferred_resolution": {k: res[k] / deferred if deferred else float("nan")
                                for k in ("cleared", "passed or moved", "no onset")},
        "inside_margin_share": float(np.mean([r["inside_margin_share"] for r in rows])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="results/stall_diagnostic.json")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="results/stale_rebuild_diagnostic.json")
    args = p.parse_args(argv)
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    published = {(e["cond"], e["seed"]): e for e in source["episodes_detail"]
                 if e["arm"] == "own_pose"}
    n = source["episodes"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(episode, [(c, s, n) for (c, s) in sorted(published)], chunksize=2))
    for r in rows:
        r["class"] = published[(r["cond"], r["seed"])]["class"]
    checks = {"reproduces_stall_diagnostic": all(
        r["replans"] == published[(r["cond"], r["seed"])]["replans"]
        and r["success"] == published[(r["cond"], r["seed"])]["success"] for r in rows),
        # Every rebuild the stall diagnostic called stale must be one of the
        # two kinds here, and every one it classed by its cells must be "new".
        "stale_is_pre_existing_or_deferred": all(
            r["kinds"].get("new cells", 0)
            == sum(v for k, v in published[(r["cond"], r["seed"])]["categories"].items()
                   if k != "stale") for r in rows)}

    safe = [r for r in rows if r["class"] != "no route"]
    groups = {
        "stall": summarise([r for r in safe if not r["success"] and not r["collision"]
                            and r["final_remaining"] >= STALL_DISTANCE]),
        "arrive": summarise([r for r in safe if r["success"]]),
        "no_route_stall": summarise([r for r in rows if r["class"] == "no route"
                                     and not r["success"] and not r["collision"]
                                     and r["final_remaining"] >= STALL_DISTANCE]),
    }
    report = {"split": "val", "episodes": n, "registered": False, "checks": checks,
              "groups": groups, "episodes_detail": rows}
    for k, g in groups.items():
        if not g["n"]:
            continue
        print(f"{k:15s} n={g['n']:3d}  {g['per_100_steps']:5.1f} rebuilds/100 steps  "
              + "  ".join(f"{a} {b:.0%}" for a, b in g["shares"].items())
              + f"  | inside margin {g['inside_margin_share']:.0%} of steps")
        print(f"{'':15s} pre-existing: {g['pre_existing_per_100_steps']:.1f}/100 steps, blocked point "
              f"{g['pre_from_plan_start_median']:.2f} m from where the plan began "
              f"({g['pre_within_half_metre_of_start']:.0%} within 0.5 m), robot "
              f"{g['pre_robot_along_median']:.2f} m along, {g['pre_depth_median']:.3f} m inside")
        print(f"{'':15s} deferred: " + "  ".join(
            f"{a} {b:.0%}" for a, b in g["deferred_resolution"].items()))
    print(f"checks: {checks}")
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
