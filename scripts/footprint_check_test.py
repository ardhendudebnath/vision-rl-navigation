"""The repair §9.22 named: a validity check that clears the footprint, as the planner does.

Report §9.22 found the stalls on clutter worlds that have a margin-safe route
rebuilding their plan on about two steps in three while the robot is inside a
wall's margin. The planner clears the robot's own footprint before it searches,
so a plan made there begins inside the margin; the check that decides whether a
plan still holds did not clear it, so at the next change anywhere in the map it
found the plan's first points inside the margin and rebuilt from scratch. That
was an association with a mechanism read off the code, not a cause. This is the
test: the published stack against the same stack with
``MappedPursuitAgent.footprint_check`` on, which excuses the route's points
inside the footprint and on the one grid step out of it -- what the planner
accepted cell by cell -- and nothing else (``tests/test_footprint_check.py``).

Both arms are the stack §9.21 and §9.22 measured (own map, own pose, 360 beams,
scan matching, corroboration) on the same val worlds, 100 each of `dense` and
`narrow`, with `nominal` as a guard, at the published 500-step budget. Each
episode records its outcome, how often it rebuilt its plan, how far it ended
from the goal by the true geodesic, and on what share of its steps the point it
steers from was inside the full margin of something mapped (§9.22's measure).
Worlds carry their route class (``route_class_test.py``).

    python scripts/footprint_check_test.py
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
from stall_diagnostic import STALL_DISTANCE, make_agent  # noqa: E402
from transfer_test import BOUND, bootstrap, mcnemar  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("nominal", "dense", "narrow")
CLUTTER = ("dense", "narrow")
ARMS = ("published", "repaired")

#: Recorded and committed before the repaired arm had run on any val world.
#:
#: DISCLOSED: both arms were smoke-tested on three train-band worlds per
#: condition. Their outcomes were identical (nominal 3/3, dense 2/3, narrow
#: 2/3, no collision); on the five margin-safe clutter worlds rebuilds fell from
#: 20.3 to 4.0 per 100 steps, the one margin-safe stall (narrow) still stalled,
#: with 125 rebuilds against 18, and the share of steps inside the margin moved
#: from 0.156 to 0.144. The published arm's val outcomes are §9.22's (0.72 on
#: dense, 0.58 on narrow, 32 stalls on margin-safe worlds). The reasoning: the
#: stale rebuilds lead the same way as the plans they replace -- §9.22 found
#: their first metre reversing on 4% of them -- so removing them should change
#: how often the robot plans, not where it drives.
PREDICTION = (
    "the repair removes the rebuilds and not the stalls: rebuilding was the "
    "symptom of being inside the margin, not why the robot stays there. On the "
    "margin-safe val clutter worlds, the repaired arm rebuilds at most 40% as "
    "often per step as the published arm; at most a third of the published "
    "arm's stalls there arrive when repaired; and the share of steps spent "
    "inside the margin moves by at most 0.05 either way. The repaired arm "
    "collides on at most 0.02 of episodes on every condition, nominal moves by "
    "at most 0.03 either way, and at most 2 of the worlds with no route at the "
    "full margin arrive. Decision on the pooled repaired - published difference "
    "over the 200 paired val clutter worlds -- REPAIRS: at least +0.05 with its "
    "95% interval above zero. HARMS: at most -0.05 with the interval below "
    "zero. NO EFFECT: within 0.05 either way with the interval inside 0.10 "
    "either way. Otherwise UNRESOLVED. Prediction: NO EFFECT."
)


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
    agent = make_agent("own_pose", cfg)
    agent.footprint_check = arm == "repaired"
    agent.start_episode(world, env.robot.pose)
    full = agent.map.robot_radius + agent.config.safety_margin
    info: dict = {}
    remaining, inside = [], []
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        remaining.append(remaining_at(field, world, env.robot.pose[:2]))
        # Inside the full margin of something mapped, where the robot believes
        # it is -- §9.22's measure, the condition the stale rebuild needs.
        here = np.asarray(agent._odom.pose[:2], dtype=float)
        inside.append(bool(agent.map.clearance(here) < full))
        if terminated or truncated:
            break
    rem = np.asarray(remaining)
    return {"cond": cond, "arm": arm, "seed": seed,
            "success": bool(info.get("is_success")), "collision": bool(info.get("collision")),
            "steps": len(rem), "replans": int(agent.replans),
            "final_remaining": float(rem[-1]) if np.isfinite(rem[-1]) else float("nan"),
            "inside_margin_share": float(np.mean(inside)),
            "recovery_steps": int(agent.recovery_steps),
            "pose_err_median": float(np.median(agent.pose_errors)) if agent.pose_errors else 0.0}


def stalled(r: dict) -> bool:
    return (not r["success"] and not r["collision"]
            and np.isfinite(r["final_remaining"]) and r["final_remaining"] >= STALL_DISTANCE)


def analyse(rows: list[dict], rng: np.random.Generator) -> dict:
    by = {(r["arm"], r["cond"], r["seed"]): r for r in rows}
    keys = sorted({(r["cond"], r["seed"]) for r in rows},
                  key=lambda k: (CONDITIONS.index(k[0]), k[1]))

    def col(arm, field, conds=CONDITIONS, cls=None):
        return np.array([by[(arm, c, s)][field] for c, s in keys
                         if c in conds and (cls is None or by[(arm, c, s)]["class"] in cls)])

    report: dict = {"worlds": len(keys), "conditions": {}, "pairs": {}}
    for cond in CONDITIONS:
        report["conditions"][cond] = {
            arm: {"success": float(col(arm, "success", (cond,)).mean()),
                  "collision": float(col(arm, "collision", (cond,)).mean()),
                  "replans_median": float(np.median(col(arm, "replans", (cond,)))),
                  "replans_per_100_steps": float(100 * col(arm, "replans", (cond,)).sum()
                                                 / col(arm, "steps", (cond,)).sum()),
                  "stalls": int(sum(stalled(by[(arm, c, s)]) for c, s in keys if c == cond))}
            for arm in ARMS}
    for name, conds in (("nominal", ("nominal",)), ("dense", ("dense",)), ("narrow", ("narrow",)),
                        ("clutter", CLUTTER)):
        sa = col("published", "success", conds).astype(bool)
        sb = col("repaired", "success", conds).astype(bool)
        pv, won, lost = mcnemar(sa, sb)
        groups = np.array([c for c, _ in keys if c in conds])
        report["pairs"][name] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won,
                                 "lost": lost, "ci95": bootstrap(sa, sb, groups, rng)}
    # Where the repair should act: worlds with a margin-safe route.
    safe = ("short", "long detour")
    clutter_keys = [k for k in keys if k[0] in CLUTTER]
    pub = {k: by[("published",) + k] for k in clutter_keys}
    rep = {k: by[("repaired",) + k] for k in clutter_keys}
    safe_keys = [k for k in clutter_keys if pub[k]["class"] in safe]
    pub_stalls = [k for k in safe_keys if stalled(pub[k])]
    report["margin_safe"] = {
        "worlds": len(safe_keys),
        "published_stalls": len(pub_stalls),
        "published_stalls_arrived": int(sum(rep[k]["success"] for k in pub_stalls)),
        "published_arrivals_lost": int(sum(pub[k]["success"] and not rep[k]["success"]
                                           for k in safe_keys)),
        "inside_margin_share": {arm: float(np.mean([by[(arm,) + k]["inside_margin_share"]
                                                    for k in safe_keys])) for arm in ARMS},
        "replans_per_100_steps": {arm: float(100 * sum(by[(arm,) + k]["replans"] for k in safe_keys)
                                             / sum(by[(arm,) + k]["steps"] for k in safe_keys))
                                  for arm in ARMS}}
    no_route = [k for k in clutter_keys if pub[k]["class"] == "no route"]
    report["no_route"] = {"worlds": len(no_route),
                          "arrived": {arm: int(sum(by[(arm,) + k]["success"] for k in no_route))
                                      for arm in ARMS}}
    g = report["pairs"]["clutter"]
    if g["delta"] >= BOUND and g["ci95"][0] > 0:
        decision = "REPAIRS"
    elif g["delta"] <= -BOUND and g["ci95"][1] < 0:
        decision = "HARMS"
    elif abs(g["delta"]) <= BOUND and -2 * BOUND < g["ci95"][0] and g["ci95"][1] < 2 * BOUND:
        decision = "NO EFFECT"
    else:
        decision = "UNRESOLVED"
    report["decision"] = decision
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="results/footprint_check_test.json")
    args = p.parse_args(argv)

    classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
               for w in route_classes(args.split, c, args.episodes)}
    jobs = [(c, a, s, args.episodes, args.split) for c in CONDITIONS for a in ARMS
            for s in sorted(seed for cond, seed in classes if cond == c)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(episode, jobs, chunksize=1))
    for r in rows:
        r["class"] = classes[(r["cond"], r["seed"])]
    report = analyse(rows, np.random.default_rng(0))
    report.update(split=args.split, prediction=PREDICTION)
    stall = Path("results/stall_diagnostic.json")
    if args.split == "val" and args.episodes == 100 and stall.exists():
        # Identity control: the published arm is the stack §9.22 drove on the
        # same worlds, so it must reproduce every outcome and replan count.
        before = {(r["cond"], r["seed"]): r for r in json.loads(stall.read_text(encoding="utf-8"))
                  ["episodes_detail"] if r["arm"] == "own_pose"}
        mine = [r for r in rows if r["arm"] == "published" and r["cond"] in CLUTTER]
        report["reproduces_stall_diagnostic"] = len(mine) == len(before) and all(
            all(r[f] == before[(r["cond"], r["seed"])][f] for f in ("success", "collision", "replans", "steps"))
            for r in mine)
        print(f"published arm reproduces §9.22: {report['reproduces_stall_diagnostic']}")
    report["episodes_detail"] = rows

    for cond, c in report["conditions"].items():
        print(cond.ljust(8) + "  ".join(
            f"{arm} {v['success']:.2f}/{v['collision']:.2f} replans {v['replans_median']:.0f} "
            f"({v['replans_per_100_steps']:.1f}/100) stalls {v['stalls']}" for arm, v in c.items()))
    for name, q in report["pairs"].items():
        print(f"repaired - published {name:8s} {q['delta']:+.3f} (p {q['p']:.3f}, {q['won']}/{q['lost']}) "
              f"CI {np.round(q['ci95'], 3).tolist()}")
    ms = report["margin_safe"]
    print(f"margin-safe: {ms['worlds']} worlds, published stalls {ms['published_stalls']}, of which "
          f"{ms['published_stalls_arrived']} arrive repaired; arrivals lost {ms['published_arrivals_lost']}; "
          f"inside margin {ms['inside_margin_share']}; replans/100 {ms['replans_per_100_steps']}")
    print(f"no route: {report['no_route']}")
    print(f"decision {report['decision']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
