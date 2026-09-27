"""What does a SLAM back end buy this stack, and is it enough for a tight gap?

Everything published in this report runs a **front end** alone: encoders
integrated forward, each scan matched against the map the robot has built, the
corrected pose kept (§9.2-§9.6). §12 asked for a back end
(:mod:`vision_nav.mapping.posegraph`) on two separate grounds, and this tests
both at once.

**The sparse case, §12 item 3.** At 32 beams slam_toolbox's back end reached
0.690 on `sparse` where this stack reached 0.570 (§9.4). That is the one place a
production back end visibly beat this one, so it is where a back end of my own
should show first.

**The tight-gap case, §9.11.** The robot holds its line to about 0.02 m and is
wrong about where that line is by 0.065-0.125 m, against gaps that leave 0.017
to 0.025 m of room. The pose is the binding constraint, so the question is how
far a back end moves it -- and, separately, whether that is anywhere near
enough.

Run on the ``val`` seed band, which no experiment scores.

    python scripts/backend_experiment.py --episodes 25
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.localisation import OdometryConfig
from vision_nav.training.env_factory import build_env_config

#: ``(condition, sensor)``. The sparse cell is at 32 beams because that is where
#: §12's question lives; the rest are at 360, which is where §9.11 measured the
#: error budget.
CELLS = (("sparse", "lidar32"), ("dense", "lidar360"),
         ("narrow", "lidar360"), ("nominal", "lidar360"))
CLUTTER = ("dense", "narrow")

#: Registered before the arm had been run on any val seed. Four claims, chosen
#: to be falsifiable in different directions rather than to be flattering.
#:
#:   1. **`sparse` at 32 beams improves by at least +0.05.** This is the cell
#:      where a production back end beat this stack by 0.12, so a back end that
#:      cannot move it has not earned the name. The endpoint.
#:   2. **Pose error p95 on clutter at 360 beams falls by at least 25%.** The
#:      quantity §9.11 identified as binding. Disclosed rather than presented as
#:      a blind forecast: a three-world check on the **train** band (scored by
#:      nothing, and deliberately not val) gave 0.091 -> 0.072, about 21%, so
#:      25% is set just above what that check suggested and can fail.
#:   3. **The eleven tight worlds are still not recovered -- at most 2 of 11.**
#:      A negative prediction, and the honest one: §9.11's arithmetic wants
#:      about 0.02 m of pose accuracy to pass gaps that leave 0.017-0.025 m, the
#:      front end sits at 0.065-0.125 m, and nothing about a pose graph over a
#:      2 cm scan-match grid closes that. If this is wrong, the back end is
#:      better than its own components suggest.
#:   4. **`nominal` stays inside +/-0.03.** The open worlds where the front end
#:      already localises well: a back end should neither help nor hurt there,
#:      and this project's null band is 0.03.
#:
#: The obvious way for this to fail as a whole: loop closure needs a place to
#: have been before, and a robot that drives 8 m to a goal may never revisit
#: anything. Then the graph is a chain, the solve has nothing to redistribute,
#: and every number above is inert -- which the chain test in
#: ``test_posegraph.py`` pins as the correct behaviour rather than a bug.
PREDICTION = {
    "sparse_gain_at_least": 0.05,
    "clutter_pose_p95_reduction_at_least": 0.25,
    "tight_recovered_at_most": 2,
    "nominal_band": 0.03,
}


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar on paired binary outcomes."""
    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), a_only, b_only


def paired_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = 10000) -> list[float]:
    n = len(a)
    diffs = [b[i].mean() - a[i].mean()
             for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def episode(env, seed: int, cfg, sensor: str, back_end: bool) -> dict:
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor=sensor,
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, pose_graph=back_end)
    agent.start_episode(env.world, env.robot.pose)
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    errors = np.asarray(agent.pose_errors) if agent.pose_errors else np.zeros(1)
    graph = agent.graph
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": len(errors),
        "goal_distance": float(info.get("goal_distance", np.nan)),
        "pose_err_mean": float(errors.mean()),
        "pose_err_p95": float(np.percentile(errors, 95)),
        "pose_err_final": float(errors[-1]),
        "keyframes": len(graph.poses) if graph else 0,
        "closures": graph.closures if graph else 0,
        "closures_rejected": graph.closures_rejected if graph else 0,
        "optimisations": graph.optimisations if graph else 0,
        "correction": float(graph.total_correction) if graph else 0.0,
        "map_rebuilds": int(agent.map_rebuilds),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--cells", nargs="+", default=[c for c, _ in CELLS])
    p.add_argument("--audit", default="results/margin_audit.json")
    p.add_argument("--out", default="results/backend_experiment.json")
    p.add_argument("--from-results", default=None,
                   help="Re-score a finished run instead of driving it again.")
    args = p.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))["conditions"]
    tight = {c: {r["seed"] for r in audit[c]["worlds"] if r["l_full"] is None}
             for c in audit}
    finished = (json.loads(Path(args.from_results).read_text(encoding="utf-8"))
                if args.from_results else None)
    rng = np.random.default_rng(20260927)
    report: dict = {"split": "val", "episodes": args.episodes,
                    "registered": PREDICTION, "cells": {}}

    for cond, sensor in CELLS:
        if cond not in args.cells:
            continue
        if finished is not None:
            arms = {k: finished["cells"][cond][k]["episodes"]
                    for k in ("front_end", "back_end")}
        else:
            _, shift, noise = BENCHMARK_CONDITIONS[cond]
            cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                                   split="val", shift=shift, n_worlds=args.episodes)
            env = ProceduralNavEnv(cfg)
            seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
            arms = {"front_end": [episode(env, s, cfg, sensor, False) for s in seeds],
                    "back_end": [episode(env, s, cfg, sensor, True) for s in seeds]}

        a = np.array([r["success"] for r in arms["front_end"]], dtype=bool)
        b = np.array([r["success"] for r in arms["back_end"]], dtype=bool)
        pv, lost, won = mcnemar_p(a, b)
        ci = paired_ci(a.astype(float), b.astype(float), rng)
        p95 = [float(np.mean([r["pose_err_p95"] for r in arms[k]]))
               for k in ("front_end", "back_end")]
        entry = {
            "sensor": sensor,
            **{k: {"episodes": v} for k, v in arms.items()},
            "success": [float(a.mean()), float(b.mean())],
            "gain": float(b.mean() - a.mean()), "mcnemar_p": pv,
            "won": won, "lost": lost, "ci": ci,
            "pose_err_p95": p95,
            "pose_p95_reduction": (1.0 - p95[1] / p95[0]) if p95[0] else float("nan"),
            "keyframes": float(np.mean([r["keyframes"] for r in arms["back_end"]])),
            "closures": float(np.mean([r["closures"] for r in arms["back_end"]])),
            "rejected": float(np.mean([r["closures_rejected"] for r in arms["back_end"]])),
            "rebuilds": float(np.mean([r["map_rebuilds"] for r in arms["back_end"]])),
        }
        marked = tight.get(cond, set())
        if marked:
            rows = {r["seed"]: r for r in arms["back_end"]}
            entry["tight"] = {
                "seeds": sorted(marked),
                "recovered": sorted(s for s in marked if rows[s]["success"]),
                "pose_err_p95": float(np.mean([rows[s]["pose_err_p95"] for s in marked])),
            }
        report["cells"][cond] = entry

        print(f"{cond:8s} ({sensor})  SR {a.mean():.2f} -> {b.mean():.2f} "
              f"({b.mean() - a.mean():+.3f})  McNemar p={pv:.3f} "
              f"({won} won / {lost} lost)  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]", flush=True)
        print(f"         pose p95 {p95[0]:.3f} -> {p95[1]:.3f} m "
              f"({100 * entry['pose_p95_reduction']:+.0f}%)   "
              f"keyframes {entry['keyframes']:.0f}  closures {entry['closures']:.1f} "
              f"(rejected {entry['rejected']:.1f})  rebuilds {entry['rebuilds']:.1f}",
              flush=True)

    # ---- the registered endpoints ----
    ep: dict = {}
    sp = report["cells"].get("sparse")
    if sp:
        ep["sparse_gain"] = sp["gain"]
        ep["sparse_held"] = bool(sp["gain"] >= PREDICTION["sparse_gain_at_least"])
    clutter = [report["cells"][c] for c in CLUTTER if c in report["cells"]]
    if clutter:
        before = float(np.mean([c["pose_err_p95"][0] for c in clutter]))
        after = float(np.mean([c["pose_err_p95"][1] for c in clutter]))
        red = 1.0 - after / before if before else float("nan")
        ep["clutter_pose_p95"] = [before, after]
        ep["clutter_pose_reduction"] = red
        ep["pose_held"] = bool(red >= PREDICTION["clutter_pose_p95_reduction_at_least"])
        rec = sum(len(c["tight"]["recovered"]) for c in clutter if "tight" in c)
        total = sum(len(c["tight"]["seeds"]) for c in clutter if "tight" in c)
        ep["tight_recovered"] = rec
        ep["tight_total"] = total
        ep["tight_held"] = bool(rec <= PREDICTION["tight_recovered_at_most"])
    nom = report["cells"].get("nominal")
    if nom:
        ep["nominal_gain"] = nom["gain"]
        ep["nominal_held"] = bool(abs(nom["gain"]) <= PREDICTION["nominal_band"])
    report["endpoints"] = ep

    print()
    if "sparse_held" in ep:
        print(f"endpoint 1 -- sparse at 32 beams {ep['sparse_gain']:+.3f}, needed "
              f"{PREDICTION['sparse_gain_at_least']:+.2f}: "
              f"{'HELD' if ep['sparse_held'] else 'FAILED'}")
    if "pose_held" in ep:
        print(f"endpoint 2 -- clutter pose p95 {ep['clutter_pose_p95'][0]:.3f} -> "
              f"{ep['clutter_pose_p95'][1]:.3f} m "
              f"({100 * ep['clutter_pose_reduction']:+.0f}%), needed "
              f"{100 * PREDICTION['clutter_pose_p95_reduction_at_least']:.0f}%: "
              f"{'HELD' if ep['pose_held'] else 'FAILED'}")
        print(f"endpoint 3 -- tight worlds recovered {ep['tight_recovered']} of "
              f"{ep['tight_total']}, predicted at most "
              f"{PREDICTION['tight_recovered_at_most']}: "
              f"{'HELD' if ep['tight_held'] else 'FAILED'}")
    if "nominal_held" in ep:
        print(f"endpoint 4 -- nominal {ep['nominal_gain']:+.3f} inside "
              f"+/-{PREDICTION['nominal_band']}: "
              f"{'HELD' if ep['nominal_held'] else 'FAILED'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
