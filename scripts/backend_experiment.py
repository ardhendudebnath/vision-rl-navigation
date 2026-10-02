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
#: error budget. ``large`` and ``noisy_lidar`` are driven only on the test band,
#: where the comparison is against the published Result 1 row rather than
#: against a val diagnostic.
CELLS = (("sparse", "lidar32"), ("dense", "lidar360"),
         ("narrow", "lidar360"), ("nominal", "lidar360"),
         ("large", "lidar360"), ("noisy_lidar", "lidar360"))
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

#: Registered before the back end was run on a single **test** seed, after the
#: val result above and at the user's instruction to carry it across. This moves
#: the headline numbers of the report, so it is registered rather than decided
#: afterwards.
#:
#: Every forecast here is derived from the val measurement of the same
#: conditions with the same sensor, which is the one kind of extrapolation §10
#: found reliable: the four measurement-derived predictions in this project held,
#: and the ones that failed carried a measurement across a regime boundary.
#: Nothing here crosses one except the seed band itself.
#:
#:   1. **Pooled success over the six cells changes by between 0.00 and +0.05.**
#:      Val won 3 episodes of 100 and lost none, so the claim is that it helps a
#:      little and harms nothing. The endpoint, and the first clause matters as
#:      much as the second: a *fall* fails this.
#:   2. **Clutter pose error p95 falls by 20% to 32%.** Val gave 26% on the same
#:      two conditions at the same beam count.
#:   3. **At most 2 collisions across all cells, with the back end.** Val had
#:      zero in 200 episodes, and the front end has never collided in any
#:      measurement in this report.
#:   4. **`sparse` at 32 beams stays inside ±0.03.** Val gave +0.000, and §9.12's
#:      mechanism says why -- a front end 0.472 m lost gives a graph nothing to
#:      work with -- so this should replicate rather than merely repeat.
#:
#: The way this fails: 100 worlds per condition is four times val's 25, so an
#: effect that was three episodes out of a hundred may simply not be there. A
#: pooled gain of exactly 0.000 would satisfy claim 1's letter while saying the
#: val result was noise, and the report is to say so if that is what happens.
TEST_PREDICTION = {
    "pooled_gain_between": [0.00, 0.05],
    "clutter_pose_p95_reduction_between": [0.20, 0.32],
    "collisions_at_most": 2,
    "sparse_band": 0.03,
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


def reproduces_published(report: dict,
                         path: str = "results/corroboration_experiment.json") -> dict:
    """The control: the front-end arm must be the published arm, per episode.

    Phase 6i's ``matched_corroborated`` is exactly this stack at 360 beams --
    scan matching, corroboration, the same odometry -- on the same held-out
    bands, and it saved every episode's outcome. If the front end here does not
    reproduce it episode for episode, the back end is being compared against
    something other than the numbers the report publishes, and nothing below
    means what it says. ``sparse`` is excluded: it runs at 32 beams here, and
    Phase 6i ran it at 360.
    """
    published = json.loads(Path(path).read_text(encoding="utf-8"))["conditions"]
    out = {}
    for cond, cell in report["cells"].items():
        if cell["sensor"] != "lidar360" or cond not in published:
            continue
        theirs = published[cond]["success_per_episode"]["matched_corroborated"]
        mine = [int(r["success"]) for r in cell["front_end"]["episodes"]]
        n = min(len(mine), len(theirs))
        out[cond] = {"matched": int(sum(a == b for a, b in zip(mine[:n], theirs[:n],
                                                                  strict=True))),
                     "of": n}
    return out


def score_test(report: dict, rng) -> dict:
    """The four claims registered in ``TEST_PREDICTION``, scored."""
    reg = TEST_PREDICTION
    cells = report["cells"]
    ep: dict = {}
    ep["reproduction"] = reproduces_published(report)
    ep["reproduces"] = bool(ep["reproduction"]) and all(
        v["matched"] == v["of"] for v in ep["reproduction"].values())

    # 1. Pooled over every cell, paired episode by episode.
    a = np.concatenate([[r["success"] for r in c["front_end"]["episodes"]]
                        for c in cells.values()]).astype(bool)
    b = np.concatenate([[r["success"] for r in c["back_end"]["episodes"]]
                        for c in cells.values()]).astype(bool)
    pv, lost, won = mcnemar_p(a, b)
    gain = float(b.mean() - a.mean())
    lo, hi = reg["pooled_gain_between"]
    ep.update({"pooled_success": [float(a.mean()), float(b.mean())],
               "pooled_gain": gain, "pooled_mcnemar_p": pv,
               "pooled_won": won, "pooled_lost": lost, "pooled_n": int(len(a)),
               "pooled_ci": paired_ci(a.astype(float), b.astype(float), rng),
               "pooled_held": bool(lo <= gain <= hi)})

    # 2. Pose error p95 on the clutter pair.
    clutter = [cells[c] for c in CLUTTER if c in cells]
    if clutter:
        before = float(np.mean([c["pose_err_p95"][0] for c in clutter]))
        after = float(np.mean([c["pose_err_p95"][1] for c in clutter]))
        red = 1.0 - after / before if before else float("nan")
        lo, hi = reg["clutter_pose_p95_reduction_between"]
        ep.update({"clutter_pose_p95": [before, after],
                   "clutter_pose_reduction": red,
                   "pose_held": bool(lo <= red <= hi)})

    # 3. Collisions with the back end, anywhere.
    colls = sum(int(r["collision"]) for c in cells.values()
                for r in c["back_end"]["episodes"])
    colls_before = sum(int(r["collision"]) for c in cells.values()
                       for r in c["front_end"]["episodes"])
    ep.update({"collisions": [colls_before, colls],
               "collisions_held": bool(colls <= reg["collisions_at_most"])})

    # 4. Sparse at 32 beams, inert.
    if "sparse" in cells:
        g = cells["sparse"]["gain"]
        ep.update({"sparse_gain": g,
                   "sparse_held": bool(abs(g) <= reg["sparse_band"])})

    print()
    for cond, v in ep["reproduction"].items():
        print(f"control   -- {cond:11s} front end reproduces Phase 6i "
              f"{v['matched']}/{v['of']} episodes")
    print(f"control   -- {'REPRODUCES' if ep['reproduces'] else 'DOES NOT REPRODUCE'}"
          f" the published front end")
    lo, hi = reg["pooled_gain_between"]
    print(f"endpoint 1 -- pooled over {ep['pooled_n']} paired episodes "
          f"{ep['pooled_success'][0]:.3f} -> {ep['pooled_success'][1]:.3f} "
          f"({gain:+.3f}), McNemar p={pv:.3f} ({won} won / {lost} lost), "
          f"registered [{lo:+.2f}, {hi:+.2f}]: "
          f"{'HELD' if ep['pooled_held'] else 'FAILED'}")
    if "pose_held" in ep:
        lo, hi = reg["clutter_pose_p95_reduction_between"]
        print(f"endpoint 2 -- clutter pose p95 {ep['clutter_pose_p95'][0]:.3f} -> "
              f"{ep['clutter_pose_p95'][1]:.3f} m "
              f"({100 * ep['clutter_pose_reduction']:.0f}%), registered "
              f"{100 * lo:.0f}-{100 * hi:.0f}%: "
              f"{'HELD' if ep['pose_held'] else 'FAILED'}")
    print(f"endpoint 3 -- collisions {colls_before} -> {colls}, bound "
          f"{reg['collisions_at_most']}: "
          f"{'HELD' if ep['collisions_held'] else 'FAILED'}")
    if "sparse_held" in ep:
        print(f"endpoint 4 -- sparse at 32 beams {ep['sparse_gain']:+.3f} inside "
              f"+/-{reg['sparse_band']}: "
              f"{'HELD' if ep['sparse_held'] else 'FAILED'}")
    return ep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--cells", nargs="+", default=None,
                   help="Defaults to the four cells registered for val, or all six "
                        "on test, so the val run stays exactly the run registered.")
    p.add_argument("--audit", default="results/margin_audit.json",
                   help="Which val worlds have no margin-safe route. Val only: "
                        "the audit was never run on test, so on test there is "
                        "no tight-world endpoint to score.")
    p.add_argument("--out", default=None)
    p.add_argument("--from-results", default=None,
                   help="Re-score a finished run instead of driving it again.")
    p.add_argument("--resume", action="store_true",
                   help="Keep cells already present in --out and drive only the "
                        "rest. Episodes are deterministic given a seed, so a "
                        "resumed run is the same run; this exists because a "
                        "forty-minute job that writes only at the end loses "
                        "everything to one interruption.")
    args = p.parse_args(argv)
    test = args.split == "test"
    if args.cells is None:
        args.cells = ([c for c, _ in CELLS] if test
                      else ["sparse", "dense", "narrow", "nominal"])
    if args.out is None:
        args.out = ("results/backend_test.json" if test
                    else "results/backend_experiment.json")

    tight: dict = {}
    if not test:
        audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))["conditions"]
        tight = {c: {r["seed"] for r in audit[c]["worlds"] if r["l_full"] is None}
                 for c in audit}
    finished = (json.loads(Path(args.from_results).read_text(encoding="utf-8"))
                if args.from_results else None)
    rng = np.random.default_rng(20260927)
    out = Path(args.out)
    report: dict = {"split": args.split, "episodes": args.episodes,
                    "registered": TEST_PREDICTION if test else PREDICTION,
                    "cells": {}}
    done: dict = {}
    if args.resume and out.exists():
        done = json.loads(out.read_text(encoding="utf-8")).get("cells", {})

    for cond, sensor in CELLS:
        if cond not in args.cells:
            continue
        if cond in done:
            report["cells"][cond] = done[cond]
            e = done[cond]
            print(f"{cond:8s} ({e['sensor']})  kept from a previous run: "
                  f"SR {e['success'][0]:.2f} -> {e['success'][1]:.2f}, "
                  f"pose p95 {e['pose_err_p95'][0]:.3f} -> "
                  f"{e['pose_err_p95'][1]:.3f} m", flush=True)
            continue
        if finished is not None:
            arms = {k: finished["cells"][cond][k]["episodes"]
                    for k in ("front_end", "back_end")}
        else:
            own_split, shift, noise = BENCHMARK_CONDITIONS[cond]
            # On the held-out side each condition has its own band: `nominal`
            # and `noisy_lidar` are scored on `test`, the four shifted ones on
            # `test_ood`. A blanket "test" would drive four of the six cells on
            # worlds no published number was ever measured on.
            band = own_split if test else args.split
            cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                                   split=band, shift=shift,
                                   n_worlds=args.episodes)
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
        # Written after every cell, so an interruption costs one cell and not
        # the whole run.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"{cond:8s} ({sensor})  SR {a.mean():.2f} -> {b.mean():.2f} "
              f"({b.mean() - a.mean():+.3f})  McNemar p={pv:.3f} "
              f"({won} won / {lost} lost)  CI [{ci[0]:+.2f}, {ci[1]:+.2f}]", flush=True)
        print(f"         pose p95 {p95[0]:.3f} -> {p95[1]:.3f} m "
              f"({100 * entry['pose_p95_reduction']:+.0f}%)   "
              f"keyframes {entry['keyframes']:.0f}  closures {entry['closures']:.1f} "
              f"(rejected {entry['rejected']:.1f})  rebuilds {entry['rebuilds']:.1f}",
              flush=True)

    if test:
        report["endpoints"] = score_test(report, rng)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")
        return 0

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

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
