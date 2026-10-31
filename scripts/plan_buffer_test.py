"""The repair §9.31 points to: plan a buffer inside the margin's edge.

Report §9.31 found the hovering stall's flips forced by the map at the margin's
very edge. The planner treats inflation as blocked-or-free, so its routes run
along that edge, and a cell the map marks occupied there -- most often one it
had marked before and freed -- knocks a route off where the world had room, on
three flips in four; the robot takes the other way round, and comes back once
the first side reopens. ``MappedPursuitAgent.plan_buffer`` plans a first rung at
the margin plus a buffer and still judges the plan at the margin, so a cell's
worth of new evidence at the edge no longer rules the route out. The rung is
taken only when it costs at most 1.25 times the distance of the route at the
margin (``BUFFER_DETOUR``); otherwise the published ladder plans as before.

Two arms on the same 100 val worlds each of `nominal`, `dense` and `narrow`, at
the 500-step budget, both with §9.26's footprint-consistent check -- without
it, the footprint fault forces most rebuilds and the edge never gets a say:

  footprint  §9.26's repaired arm, which must reproduce its episodes exactly
  buffered   the same with ``plan_buffer`` = 0.10 m, one map cell

Each episode also counts its flips (a rebuild whose route sets off more than
90 degrees from the old one, ``flip_diagnostic.py``) and how many of its plans
the buffered rung supplied.

    python scripts/plan_buffer_test.py
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
from flip_diagnostic import Watcher  # noqa: E402
from route_class_test import route_classes  # noqa: E402
from stall_diagnostic import STALL_DISTANCE, make_agent  # noqa: E402
from transfer_test import BOUND, bootstrap, mcnemar  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("nominal", "dense", "narrow")
CLUTTER = ("dense", "narrow")
ARMS = ("footprint", "buffered")
BUFFER = 0.10

#: Recorded and committed before the buffered arm had run on any val world.
#:
#: DISCLOSED: smoke-tested on the first twelve train-band worlds per condition.
#: Outcomes moved on two worlds (nominal one won and one lost; narrow one won);
#: on the twenty margin-safe clutter worlds the buffered rung supplied 73% of
#: plans (83% in the footprint arm's five stalls, one of which then arrived),
#: rebuilds fell from 5.9 to 3.3 per 100 steps and flips from 0.36 to 0.33.
#: The reasoning: §9.31 says most flips are the edge's doing, so the buffer
#: should cut them; but three repairs before this one removed what they aimed
#: at and left the stalls standing, and the stall may hold for a reason the
#: flips only express.
PREDICTION = (
    "the buffer takes the routes off the edge and cuts the flips, and the "
    "stalls mostly stay. On the margin-safe val clutter worlds: the buffered "
    "rung supplies at least half of the buffered arm's plans, both overall and "
    "on the worlds where the footprint arm stalls; rebuilds and flips per 100 "
    "steps each fall by at least a third; at most a third of the footprint "
    "arm's stalls there arrive when buffered. The buffered arm collides on at "
    "most 0.02 of episodes on every condition and moves nominal by at most "
    "0.03. The footprint arm reproduces §9.26's repaired arm exactly. "
    "Decision on the pooled buffered - footprint difference over the 200 "
    "paired val clutter worlds -- REPAIRS: at least +0.05 with its 95% "
    "interval above zero. HARMS: at most -0.05 with the interval below zero. "
    "NO EFFECT: within 0.05 either way with the interval inside 0.10 either "
    "way. Otherwise UNRESOLVED. Prediction: NO EFFECT."
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
    agent.footprint_check = True
    agent.plan_buffer = BUFFER if arm == "buffered" else 0.0
    watch = Watcher(agent, world)
    agent.start_episode(world, env.robot.pose)
    info: dict = {}
    remaining = []
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        remaining.append(remaining_at(field, world, env.robot.pose[:2]))
        if terminated or truncated:
            break
    rem = np.asarray(remaining)
    return {"cond": cond, "arm": arm, "seed": seed,
            "success": bool(info.get("is_success")), "collision": bool(info.get("collision")),
            "steps": len(rem), "replans": int(agent.replans),
            "final_remaining": float(rem[-1]) if np.isfinite(rem[-1]) else float("nan"),
            "flips": len(watch.flips), "plans": int(agent.goal_plans + agent.frontier_plans),
            "buffered_plans": int(agent.buffered_plans),
            "pose_err_median": float(np.median(agent.pose_errors)) if agent.pose_errors else 0.0}


def stalled(r: dict) -> bool:
    return (not r["success"] and not r["collision"]
            and np.isfinite(r["final_remaining"]) and r["final_remaining"] >= STALL_DISTANCE)


def analyse(rows: list[dict], rng: np.random.Generator) -> dict:
    by = {(r["arm"], r["cond"], r["seed"]): r for r in rows}
    keys = sorted({(r["cond"], r["seed"]) for r in rows}, key=lambda k: (CONDITIONS.index(k[0]), k[1]))

    def col(arm, field, conds):
        return np.array([by[(arm, c, s)][field] for c, s in keys if c in conds])

    report: dict = {"conditions": {}, "pairs": {}}
    for cond in CONDITIONS:
        report["conditions"][cond] = {
            arm: {"success": float(col(arm, "success", (cond,)).mean()),
                  "collision": float(col(arm, "collision", (cond,)).mean()),
                  "stalls": int(sum(stalled(by[(arm, c, s)]) for c, s in keys if c == cond))}
            for arm in ARMS}
    for name, conds in (("nominal", ("nominal",)), ("dense", ("dense",)), ("narrow", ("narrow",)),
                        ("clutter", CLUTTER)):
        sa = col("footprint", "success", conds).astype(bool)
        sb = col("buffered", "success", conds).astype(bool)
        pv, won, lost = mcnemar(sa, sb)
        groups = np.array([c for c, _ in keys if c in conds])
        report["pairs"][name] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won,
                                 "lost": lost, "ci95": bootstrap(sa, sb, groups, rng)}
    safe = [k for k in keys if k[0] in CLUTTER and by[("footprint",) + k]["class"] in ("short", "long detour")]
    mech = {}
    for arm in ARMS:
        steps = sum(by[(arm,) + k]["steps"] for k in safe)
        mech[arm] = {"flips_per_100_steps": 100.0 * sum(by[(arm,) + k]["flips"] for k in safe) / steps,
                     "rebuilds_per_100_steps": 100.0 * sum(by[(arm,) + k]["replans"] for k in safe) / steps}
    plans = sum(by[("buffered",) + k]["plans"] for k in safe)
    mech["buffered"]["buffered_plan_share"] = sum(by[("buffered",) + k]["buffered_plans"] for k in safe) / plans
    stalls = [k for k in safe if stalled(by[("footprint",) + k])]
    sp = sum(by[("buffered",) + k]["plans"] for k in stalls)
    mech["buffered"]["buffered_plan_share_in_footprint_stalls"] = (
        sum(by[("buffered",) + k]["buffered_plans"] for k in stalls) / sp if sp else None)
    report["margin_safe"] = {
        "worlds": len(safe), "footprint_stalls": len(stalls),
        "footprint_stalls_arrived": int(sum(by[("buffered",) + k]["success"] for k in stalls)),
        "footprint_arrivals_lost": int(sum(by[("footprint",) + k]["success"] and not by[("buffered",) + k]["success"]
                                           for k in safe)),
        **mech}
    g = report["pairs"]["clutter"]
    # "At least +0.05" as registered. The first run compared floats exactly, so
    # 14 worlds won and 4 lost of 200 -- +10/200, +0.050 -- came out as
    # 0.0499999... and was printed NO EFFECT; corrected here and disclosed in
    # report §9.32. The outcomes were re-scored, not re-run (``--rescore``).
    eps = 1e-9
    if g["delta"] >= BOUND - eps and g["ci95"][0] > 0:
        decision = "REPAIRS"
    elif g["delta"] <= -BOUND + eps and g["ci95"][1] < 0:
        decision = "HARMS"
    elif abs(g["delta"]) <= BOUND + eps and -2 * BOUND < g["ci95"][0] and g["ci95"][1] < 2 * BOUND:
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
    p.add_argument("--out", default="results/plan_buffer_test.json")
    p.add_argument("--rescore", action="store_true",
                   help="Score the episodes already in --out instead of driving them again.")
    args = p.parse_args(argv)
    if args.rescore:
        rows = json.loads(Path(args.out).read_text(encoding="utf-8"))["episodes_detail"]
    else:
        classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
                   for w in route_classes(args.split, c, args.episodes)}
        jobs = [(c, a, s, args.episodes, args.split) for c in CONDITIONS for a in ARMS
                for s in sorted(seed for cond, seed in classes if cond == c)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(episode, jobs, chunksize=1))
        for r in rows:
            r["class"] = classes[(r["cond"], r["seed"])]
    report = analyse(rows, np.random.default_rng(0))
    report.update(split=args.split, buffer=BUFFER, prediction=PREDICTION)
    published = Path("results/footprint_check_test.json")
    if args.split == "val" and args.episodes == 100 and published.exists():
        # Identity control: the footprint arm is §9.26's repaired arm.
        before = {(r["cond"], r["seed"]): r for r in json.loads(published.read_text(encoding="utf-8"))
                  ["episodes_detail"] if r["arm"] == "repaired"}
        mine = [r for r in rows if r["arm"] == "footprint"]
        report["reproduces_footprint_test"] = len(mine) == len(before) and all(
            all(r[f] == before[(r["cond"], r["seed"])][f] for f in ("success", "collision", "replans", "steps"))
            for r in mine)
        print(f"footprint arm reproduces §9.26: {report['reproduces_footprint_test']}")
    report["episodes_detail"] = rows
    for cond, c in report["conditions"].items():
        print(cond.ljust(8) + "  ".join(f"{arm} {v['success']:.2f}/{v['collision']:.2f} stalls {v['stalls']}"
                                        for arm, v in c.items()))
    for name, q in report["pairs"].items():
        print(f"buffered - footprint {name:8s} {q['delta']:+.3f} (p {q['p']:.3f}, {q['won']}/{q['lost']}) "
              f"CI {np.round(q['ci95'], 3).tolist()}")
    print(f"margin-safe: {json.dumps(report['margin_safe'])}")
    print(f"decision {report['decision']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
