"""The repair §9.31 pointed to next: a map that stops changing its mind about walls.

Report §9.31 found the clutter stalls' route flips forced at the margin's edge,
and most of them (56%) by flicker: a cell the map had marked occupied, freed,
and now marks again. The map scores each cell by evidence -- +0.9 for a return
landing in it, -0.3 for a scan whose beams pass through it -- and one line,
0.5, both marks a cell and frees it. A cell on a wall's edge that some beams
hit and others graze hovers about that line: one hit marks it, two grazes free
it, the next hit marks it again. Planning off the edge (§9.32, confirmed on
held-out worlds in §9.33) cut the flips by a third; the flicker itself stands.

``OccupancyMap.release_at`` gives the map hysteresis: a cell is marked occupied
at the same line as before, but once occupied it is freed only when its
evidence falls below -0.15 -- when the passes through it outweigh the hit that
marked it. The line sits between the 0.0 three grazes leave a once-hit cell and
the -0.3 four leave it, so float32 rounding cannot decide it. Nothing else
changes; a free cell needs the same evidence to be marked.

Two arms on the same 100 val worlds each of `nominal`, `dense` and `narrow`,
and `noisy_lidar` as a guard -- holding cells longer could hold noise phantoms
longer -- at the 500-step budget:

  buffered    the stack as it now stands: the footprint-consistent check and
              the 0.10 m plan buffer; on the three noise-free conditions it
              must reproduce plan_buffer_test's buffered arm exactly
  hysteresis  the same, with the map's release line at -0.15

Each episode also counts its route flips, the cells its map freed, and the cells
it marked occupied again after freeing them.

    python scripts/map_release_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clutter_forensic import remaining_at  # noqa: E402
from flip_diagnostic import Watcher  # noqa: E402
from plan_buffer_test import BUFFER  # noqa: E402
from resumable import checkpoint_for, run_resumable  # noqa: E402
from route_class_test import route_classes  # noqa: E402
from stall_diagnostic import STALL_DISTANCE, make_agent  # noqa: E402
from transfer_test import BOUND, EPS, bootstrap, mcnemar  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("nominal", "dense", "narrow", "noisy_lidar")
CLUTTER = ("dense", "narrow")
ARMS = ("buffered", "hysteresis")
RELEASE = -0.15

#: Recorded and committed before the hysteresis arm had run on any val world.
#:
#: DISCLOSED: smoke-tested on the first twelve train-band worlds per condition.
#: Outcomes were identical on nominal, dense and narrow; on noisy_lidar the
#: hysteresis arm lost one world (11/12 to 10/12). On the twenty margin-safe
#: clutter worlds, cells re-occupied after being freed fell from 68.8 to 26.3
#: per 100 steps and cells freed from 120.4 to 71.7, while route flips fell
#: only from 0.33 to 0.27 per 100 steps and rebuilds rose from 3.27 to 3.65.
#: The reasoning: the map flickers far more often than routes flip, so most
#: flicker never touches a plan; cutting it should cut the flips it does cause,
#: but the smoke says that is a small share of what is left of them after the
#: buffer.
PREDICTION = (
    "the hysteresis stops most of the map's flicker and moves arrival by "
    "nothing measurable. On the margin-safe val clutter worlds: cells "
    "re-occupied after being freed fall by at least half per step, while route "
    "flips per step fall by less than a third, and rebuilds per step do not "
    "fall. The hysteresis arm collides on at most 0.02 of episodes on every "
    "condition, moves nominal by at most 0.03 either way, and moves "
    "noisy_lidar by at most 0.05 either way. The buffered arm reproduces "
    "plan_buffer_test's buffered arm exactly on the noise-free conditions. "
    "Decision on the pooled hysteresis - buffered difference over the 200 "
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
    agent.plan_buffer = BUFFER
    agent.map_release_at = RELEASE if arm == "hysteresis" else None
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
            "flips": len(watch.flips), "released": int(agent.map.released),
            "reoccupied": int(agent.map.reoccupied),
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
                  "stalls": int(sum(stalled(by[(arm, c, s)]) for c, s in keys if c == cond)),
                  "pose_err_median": float(np.median(col(arm, "pose_err_median", (cond,))))}
            for arm in ARMS}
    for name, conds in [(c, (c,)) for c in CONDITIONS] + [("clutter", CLUTTER)]:
        sa = col("buffered", "success", conds).astype(bool)
        sb = col("hysteresis", "success", conds).astype(bool)
        pv, won, lost = mcnemar(sa, sb)
        groups = np.array([c for c, _ in keys if c in conds])
        report["pairs"][name] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won,
                                 "lost": lost, "ci95": bootstrap(sa, sb, groups, rng)}
    safe = [k for k in keys if k[0] in CLUTTER and by[("buffered",) + k]["class"] in ("short", "long detour")]
    mech = {}
    for arm in ARMS:
        steps = sum(by[(arm,) + k]["steps"] for k in safe)
        mech[arm] = {f"{f}_per_100_steps": 100.0 * sum(by[(arm,) + k][f] for k in safe) / steps
                     for f in ("reoccupied", "released", "flips", "replans")}
    stalls = [k for k in safe if stalled(by[("buffered",) + k])]
    report["margin_safe"] = {
        "worlds": len(safe), "buffered_stalls": len(stalls),
        "buffered_stalls_arrived": int(sum(by[("hysteresis",) + k]["success"] for k in stalls)),
        "buffered_arrivals_lost": int(sum(by[("buffered",) + k]["success"] and not by[("hysteresis",) + k]["success"]
                                          for k in safe)),
        **mech}
    g = report["pairs"]["clutter"]
    # The same rule, and the same exact-arithmetic reading of it, as §9.32-§9.33.
    if g["delta"] >= BOUND - EPS and g["ci95"][0] > 0:
        decision = "REPAIRS"
    elif g["delta"] <= -BOUND + EPS and g["ci95"][1] < 0:
        decision = "HARMS"
    elif (abs(g["delta"]) <= BOUND + EPS and -2 * BOUND + EPS < g["ci95"][0]
          and g["ci95"][1] < 2 * BOUND - EPS):
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
    p.add_argument("--out", default="results/map_release_test.json")
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
        # Kept episode by episode, so an interrupted run resumes (resumable.py).
        rows = run_resumable(episode, jobs, args.workers, checkpoint_for(args.out))
        for r in rows:
            r["class"] = classes[(r["cond"], r["seed"])]
    report = analyse(rows, np.random.default_rng(0))
    report.update(split=args.split, release_at=RELEASE, buffer=BUFFER, prediction=PREDICTION)
    published = Path("results/plan_buffer_test.json")
    if args.split == "val" and args.episodes == 100 and published.exists():
        # Identity control: on the noise-free conditions the buffered arm is
        # plan_buffer_test's buffered arm.
        before = {(r["cond"], r["seed"]): r for r in json.loads(published.read_text(encoding="utf-8"))
                  ["episodes_detail"] if r["arm"] == "buffered"}
        mine = [r for r in rows if r["arm"] == "buffered" and r["cond"] != "noisy_lidar"]
        report["reproduces_plan_buffer_test"] = len(mine) == len(before) and all(
            all(r[f] == before[(r["cond"], r["seed"])][f] for f in ("success", "collision", "replans", "steps"))
            for r in mine)
        print(f"buffered arm reproduces §9.32: {report['reproduces_plan_buffer_test']}")
    report["episodes_detail"] = rows
    for cond, c in report["conditions"].items():
        print(cond.ljust(12) + "  ".join(f"{arm} {v['success']:.2f}/{v['collision']:.2f} stalls {v['stalls']}"
                                         for arm, v in c.items()))
    for name, q in report["pairs"].items():
        print(f"hysteresis - buffered {name:12s} {q['delta']:+.3f} (p {q['p']:.3f}, {q['won']}/{q['lost']}) "
              f"CI {np.round(q['ci95'], 3).tolist()}")
    print(f"margin-safe: {json.dumps(report['margin_safe'])}")
    print(f"decision {report['decision']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint_for(args.out).unlink(missing_ok=True)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
