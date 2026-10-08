"""Why does the robot stall on clutter worlds that have a margin-safe route?

Report §9.21: on held-out clutter worlds 72 of 81 failures end at least a metre
from the goal with the pose known to within a tenth of a metre, and 33 of those
stalls are on worlds where a route clears the planner's full margin. §9.7 called
the clutter failure indecision -- a rebuild every two and a half steps, 46
reversals in a failing episode -- and two commitment rules built on that
reading failed, so indecision is the symptom and its cause is still open.

This measures the cause. A plan is only ever rebuilt because the map has
changed under it: the agent's ``_path_ahead_blocked`` finds a point of the route
ahead closer to a mapped obstacle than the radius it was planned at. Every such
invalidation is caught before the old plan is discarded, and the cells that
caused it -- occupied now, within the plan radius of the first blocked point,
and not occupied when the plan was made -- are classed by their history:

  discovery   unknown when the plan was made: the map had not seen there, and
              the planner, which counts unknown space as free, routed through it
  flicker     seen occupied before the plan was made, free when it was made,
              occupied again now: the map changed its mind about a cell
  newly seen  seen free when the plan was made, and occupied for the first time
              now: a surface the scans had missed from earlier viewpoints

and by the truth: each trigger cell's distance from its centre to the nearest
real surface, so a phantom can be told from a wall. The world is read for that
measurement only; it never reaches the agent.

Each adopted plan also records how far its first metre turns from the one it
replaced (more than 90 degrees is a reversal) and how much of it runs through
cells the map has never seen.

Two arms, so the pose can be ruled in or out: the stack as published (own map,
own pose, 360 beams, scan matching, corroboration -- the stack §9.21's outcomes
come from) and the same map built at the exact pose (§9.3 found the pose costs
nothing measurable in clutter, so a stall mechanism that needs the pose would
be surprising).

Run on the ``val`` band, 100 worlds per clutter condition.

    python scripts/stall_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clutter_forensic import remaining_at  # noqa: E402
from route_class_test import route_classes  # noqa: E402

from vision_nav.agents.localised import LocalisedPursuitAgent  # noqa: E402
from vision_nav.agents.mapped import MappedPursuitAgent  # noqa: E402
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("dense", "narrow")
ARMS = ("own_pose", "exact_pose")
CATEGORIES = ("discovery", "flicker", "newly seen", "stale")
#: A stall ends at least this far from the goal by the true geodesic, in metres.
STALL_DISTANCE = 1.0
#: A replan that turns the route's first metre by more than this is a reversal.
REVERSAL = np.pi / 2

#: Recorded and committed before this script had run on any val world.
#:
#: DISCLOSED: the code was smoke-tested on one ``train``-band world per arm,
#: which no analysis here reads. The first 25 val worlds of each condition were driven by
#: clutter_forensic.py and their outcomes, replan counts and stall fractions are
#: published (§9.9); no per-invalidation measurement of any kind has been made
#: on any world. The reasoning: §9.3 found the exact pose worth nothing in
#: clutter, which argues against a pose mechanism; §9.7 found the rebuilds fast
#: and near the robot, inside two metres, where a 360-degree scanner has
#: already seen everything -- which argues against discovery and for the map
#: changing its mind about cells it has already seen, at the edges of real
#: surfaces where some beams hit and some graze past.
PREDICTION = (
    "the stall is the map changing its mind about real surfaces it has already "
    "seen. In the published arm, pooled over val dense and narrow, in episodes "
    "that fail at least 1 m from the goal on worlds with a margin-safe route: "
    "flicker is the largest category of plan invalidations and at least 40% of "
    "them, and discovery under 30%. At least 60% of the cells that trigger them "
    "lie within half a cell diagonal of a real surface (at least 80% in the "
    "exact-pose arm). Those stalls invalidate their plans at least three times "
    "as often per step as arrivals on margin-safe worlds, and at least 25% of "
    "their adopted replans are reversals, against at most 10% of the arrivals'. "
    "The exact-pose arm's stalls have the same leading category. "
    "Decision on the published arm -- FLICKER: flicker leads with at least 40% "
    "and the real-surface share is at least 60%. DISCOVERY: discovery at least "
    "50%. NEWLY SEEN: newly seen at least 50%. Otherwise MIXED. Precondition: "
    "at least 10 such stalls; fewer and the decision is UNDERPOWERED."
)


def heading_change(old: np.ndarray | None, new: np.ndarray, position: np.ndarray,
                   spacing: float) -> float:
    """Angle between where the old and the new route lead in their first metre."""
    if old is None or len(old) < 2 or len(new) < 2:
        return float("nan")
    k = max(1, int(round(1.0 / spacing)))
    a = old[min(k, len(old) - 1)] - position
    b = new[min(k, len(new) - 1)] - position
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return float("nan")
    return float(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))


class Recorder:
    """Wraps one agent's replanning and records what drove each rebuild."""

    def __init__(self, agent, world) -> None:
        self.agent, self.world = agent, world
        self.events: list[dict] = []
        self.plans: list[dict] = []
        self.grid_at_plan: np.ndarray | None = None
        self.scan_at_plan = 0
        reset, blocked = agent.reset, agent._path_ahead_blocked

        def recorded_reset(world_, pose, *args, **kwargs):
            old = (None if agent._track is None
                   else agent._track[agent._cursor:].copy())
            ok = reset(world_, pose, *args, **kwargs)
            if ok:
                new = agent._track
                cells = agent.map.world_to_grid(new)
                unknown = agent.map.grid[cells[:, 0], cells[:, 1]] == UNKNOWN
                self.plans.append({
                    "step": agent._steps,
                    "radius": float(agent._plan_radius),
                    "turn": heading_change(old, new, np.asarray(pose[:2], dtype=float),
                                           agent.config.track_spacing),
                    "unknown_share": float(unknown.mean()),
                })
                self.grid_at_plan = agent.map.grid.copy()
                self.scan_at_plan = agent.map.scans
            return ok

        def recorded_blocked(position):
            fired = blocked(position)
            if fired:
                self.events.append(self.classify(np.asarray(position, dtype=float)))
            return fired

        agent.reset = recorded_reset
        agent._path_ahead_blocked = recorded_blocked

    def classify(self, position: np.ndarray) -> dict:
        """What made the plan in force invalid, read before it is discarded."""
        agent, omap = self.agent, self.agent.map
        ahead = agent._track[agent._cursor:]
        radius = float(agent._plan_radius)
        invalid = np.flatnonzero(omap.clearance(ahead) < radius - 1e-9)
        event = {"step": agent._steps, "category": "stale", "along": float("nan"),
                 "truth_distance": float("nan"), "triggers": 0}
        if not len(invalid) or self.grid_at_plan is None:
            # Fired on a blockage the rate limit deferred and the map has since
            # cleared: the rebuild happens, but nothing on the route blocks it.
            return event
        first = ahead[invalid[0]]
        event["along"] = float(invalid[0] * agent.config.track_spacing)
        occ = np.argwhere(omap.grid == OCCUPIED)
        centres = omap.grid_to_world(occ)
        near = np.linalg.norm(centres - first, axis=1) <= radius + omap.half_diagonal + 1e-9
        new = near & (self.grid_at_plan[occ[:, 0], occ[:, 1]] != OCCUPIED)
        if not new.any():
            return event
        cells, pts = occ[new], centres[new]
        before = self.grid_at_plan[cells[:, 0], cells[:, 1]]
        seen_occupied = ((omap.first_occupied[cells[:, 0], cells[:, 1]] >= 0)
                         & (omap.first_occupied[cells[:, 0], cells[:, 1]] <= self.scan_at_plan))
        kinds = np.where(before == UNKNOWN, "discovery",
                         np.where((before == FREE) & seen_occupied, "flicker", "newly seen"))
        truth = np.atleast_1d(self.world.clearance(pts, include_dynamic=False))
        nearest = int(np.argmin(np.linalg.norm(pts - first, axis=1)))
        event.update(category=str(kinds[nearest]), truth_distance=float(truth[nearest]),
                     triggers=int(len(cells)),
                     real=bool(truth[nearest] <= omap.half_diagonal))
        return event


def make_agent(arm: str, cfg):
    if arm == "own_pose":
        return LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                     noise_std=cfg.lidar.noise_std,
                                     odometry=OdometryConfig(), scan_matching=True,
                                     corroborate=True)
    return MappedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                              noise_std=cfg.lidar.noise_std, corroborate=True)


def episode(args: tuple) -> dict:
    cond, arm, seed, n, split = args
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split, shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution

    agent = make_agent(arm, cfg)
    rec = Recorder(agent, world)
    agent.start_episode(world, env.robot.pose)
    remaining, speeds = [], []
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        if arm == "own_pose":
            action = agent.act(env.robot.pose, env.robot.velocity)
        else:
            action = agent.act(env.robot.pose)
        _, _, terminated, truncated, info = env.step(action)
        remaining.append(remaining_at(field, world, env.robot.pose[:2]))
        speeds.append(abs(float(env.robot.velocity[0])))
        if terminated or truncated:
            break

    rem = np.asarray(remaining)
    finite = np.isfinite(rem)
    steps = len(rem)
    ev = rec.events
    counts = {c: sum(1 for e in ev if e["category"] == c) for c in CATEGORIES}
    real = [e["real"] for e in ev if "real" in e]
    turns = np.asarray([p["turn"] for p in rec.plans[1:]], dtype=float)
    turns = turns[np.isfinite(turns)]
    return {
        "cond": cond, "arm": arm, "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": steps,
        "final_remaining": float(rem[-1]) if finite[-1] else float("nan"),
        "best_remaining": float(rem[finite].min()) if finite.any() else float("nan"),
        "replans": int(agent.replans),
        "invalidations": len(ev),
        "categories": counts,
        "real_share": float(np.mean(real)) if real else float("nan"),
        "triggers_real": int(sum(real)),
        "triggers_classified": len(real),
        "along_median": float(np.nanmedian([e["along"] for e in ev])) if ev else float("nan"),
        "plans": len(rec.plans),
        "reversals": int((turns > REVERSAL).sum()),
        "turns": int(len(turns)),
        "unknown_share_median": float(np.median([p["unknown_share"] for p in rec.plans]))
        if rec.plans else float("nan"),
        "standing_share": float(np.mean(np.asarray(speeds) < 0.05)) if speeds else float("nan"),
        "recovery_steps": int(agent.recovery_steps),
        "pose_err_median": (float(np.median(agent.pose_errors))
                            if arm == "own_pose" and agent.pose_errors else 0.0),
    }


def group(rows: list[dict]) -> dict:
    """Pooled shares and rates over a set of episodes."""
    if not rows:
        return {"n": 0}
    total = sum(r["invalidations"] for r in rows)
    counts = {c: sum(r["categories"][c] for r in rows) for c in CATEGORIES}
    classified = sum(r["triggers_classified"] for r in rows)
    turns = sum(r["turns"] for r in rows)
    return {
        "n": len(rows),
        "invalidations": total,
        "per_100_steps": 100.0 * total / sum(r["steps"] for r in rows),
        "shares": {c: counts[c] / total if total else float("nan") for c in CATEGORIES},
        "real_share": sum(r["triggers_real"] for r in rows) / classified if classified else float("nan"),
        "reversal_share": sum(r["reversals"] for r in rows) / turns if turns else float("nan"),
        "along_median": float(np.nanmedian([r["along_median"] for r in rows])),
        "unknown_share_median": float(np.nanmedian([r["unknown_share_median"] for r in rows])),
        "standing_share": float(np.mean([r["standing_share"] for r in rows])),
        "recovery_steps": float(np.mean([r["recovery_steps"] for r in rows])),
        "pose_err_median": float(np.median([r["pose_err_median"] for r in rows])),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default="results/stall_diagnostic.json")
    args = p.parse_args(argv)

    classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
               for w in route_classes("val", c, args.episodes)}
    jobs = [(c, a, s, args.episodes, "val") for c in CONDITIONS for a in ARMS
            for s in sorted(seed for cond, seed in classes if cond == c)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(episode, jobs, chunksize=2))
    for r in rows:
        r["class"] = classes[(r["cond"], r["seed"])]

    # Identity control: the wrappers only watch. On the worlds clutter_forensic
    # drove with the same stack, every outcome and every replan count must match.
    forensic = json.loads(Path("results/clutter_forensic.json").read_text(encoding="utf-8"))
    seen = {(c, e["seed"]): e for c in CONDITIONS for e in forensic["conditions"][c]["episodes"]}
    mine = [r for r in rows if r["arm"] == "own_pose" and (r["cond"], r["seed"]) in seen]
    checks = {"own_pose_reproduces_forensic": bool(mine) and all(
        r["success"] == seen[(r["cond"], r["seed"])]["success"]
        and r["replans"] == seen[(r["cond"], r["seed"])]["replans"] for r in mine),
        "forensic_worlds_compared": len(mine)}

    def pick(arm: str, kind: str) -> list[dict]:
        safe = [r for r in rows if r["arm"] == arm and r["class"] != "no route"]
        if kind == "stall":
            return [r for r in safe if not r["success"] and not r["collision"]
                    and r["final_remaining"] >= STALL_DISTANCE]
        if kind == "arrive":
            return [r for r in safe if r["success"]]
        return [r for r in rows if r["arm"] == arm and r["class"] == "no route"
                and not r["success"] and not r["collision"]
                and r["final_remaining"] >= STALL_DISTANCE]

    groups = {arm: {k: group(pick(arm, k)) for k in ("stall", "arrive", "no_route_stall")}
              for arm in ARMS}
    s, a = groups["own_pose"]["stall"], groups["own_pose"]["arrive"]
    x = groups["exact_pose"]["stall"]
    lead = (max(CATEGORIES[:3], key=lambda c: s["shares"][c]) if s["n"] else None)
    x_lead = (max(CATEGORIES[:3], key=lambda c: x["shares"][c]) if x["n"] else None)
    clauses = {}
    if s["n"]:
        clauses = {
            "flicker_leads_40": lead == "flicker" and s["shares"]["flicker"] >= 0.40,
            "discovery_under_30": s["shares"]["discovery"] < 0.30,
            "real_60": s["real_share"] >= 0.60,
            "exact_real_80": bool(x["n"]) and x["real_share"] >= 0.80,
            "rate_3x": s["per_100_steps"] >= 3 * a["per_100_steps"],
            "reversals": s["reversal_share"] >= 0.25 and a["reversal_share"] <= 0.10,
            "exact_same_lead": x_lead == lead,
        }
    if s["n"] < 10:
        decision = "UNDERPOWERED"
    elif clauses["flicker_leads_40"] and clauses["real_60"]:
        decision = "FLICKER"
    elif s["shares"]["discovery"] >= 0.50:
        decision = "DISCOVERY"
    elif s["shares"]["newly seen"] >= 0.50:
        decision = "NEWLY SEEN"
    else:
        decision = "MIXED"

    report = {"split": "val", "episodes": args.episodes, "prediction": PREDICTION,
              "checks": checks, "groups": groups, "lead": lead, "exact_lead": x_lead,
              "clauses": clauses, "decision": decision, "episodes_detail": rows}
    print(f"val, {args.episodes} worlds per clutter condition\n")
    for arm in ARMS:
        for k, g in groups[arm].items():
            if not g["n"]:
                print(f"  {arm:10s} {k:15s} none")
                continue
            sh = "  ".join(f"{c} {g['shares'][c]:.0%}" for c in CATEGORIES)
            print(f"  {arm:10s} {k:15s} n={g['n']:3d}  {g['per_100_steps']:5.1f}/100 steps  {sh}  "
                  f"real {g['real_share']:.0%}  reversals {g['reversal_share']:.0%}  "
                  f"ahead {g['along_median']:.2f} m  standing {g['standing_share']:.0%}")
    print(f"\n  lead: {lead} (exact pose: {x_lead})")
    print(f"  clauses: {clauses}")
    print(f"  decision: {decision}")
    print(f"  checks: {checks}")
    print("pre-registered: " + PREDICTION)
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if checks["own_pose_reproduces_forensic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
