"""Does giving up the safety margin when stalled recover the worlds it rules out?

§9.9's forensic found that 11 of 50 val clutter worlds admit no route at all at
``robot_radius + safety_margin`` (0.22 + 0.18 = 0.40 m), that every one of them
admits a route at the bare 0.22 m, and that **not one of the eleven is ever
solved** -- 61% of the clutter failures on 22% of the worlds, Fisher
*p* = 8.5e-07. The agent's existing three-radius ladder does not rescue them: it
drops a rung only when planning *fails*, and an optimistic map keeps producing
routes that succeed on paper and fail on contact, so it plans at the full margin
for 79% of its steps and averages 0.35 m at its lowest.

The treatment drops the clearance on a different signal -- the robot has stopped
getting closer to the goal -- and then keeps it down. The signal is the robot's
own: the distance from the pose it steers by to the goal in its own map frame,
never the true geodesic the forensic measured with.

``STALL_STEPS`` is 50, derived from the forensic rather than tuned here: episodes
that arrive spend 3% of their steps not improving (about 6 steps of a 190-step
episode), and episodes that fail spend 52-58% of 500. Fifty sits an order of
magnitude above the first and well below the second.

    python scripts/margin_experiment.py --episodes 25

Run on the ``val`` seed band, which no experiment scores.
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

CONDITIONS = ("dense", "narrow", "nominal")
#: Steps without closing on the goal before the margin is given up.
STALL_STEPS = 50

#: Registered before the arm had been run on any seed. The endpoint is the
#: eleven worlds §9.9 identified, named in ``margin_overlap.json`` before this
#: file existed, so which worlds count was fixed in advance and cannot drift.
#:
#:   1. **At least 4 of those 11 worlds are recovered.** Every one admits a
#:      route at the bare radius and the detour is only 1.09-1.17x, so the
#:      route exists and is not much longer; against that, threading a gap
#:      narrower than 0.40 m with a 0.22 m robot leaves no clearance for the
#:      0.07-0.22 m of pose error §9.5 measured, and some will be lost to
#:      contact or to oscillation at the mouth of the gap. Four of eleven is a
#:      little over a third and is the endpoint; nothing else can rescue a miss.
#:   2. **No more than 0.03 of success given up on the 39 margin-safe worlds.**
#:      A world that never stalls never triggers the rule, so this is close to
#:      an identity claim. Registered as this project's null band.
#:   3. **Collisions across the 50 clutter worlds stay at or below 2.** The
#:      forensic saw zero collisions in 39 failures, and driving at zero
#:      clearance invites contact. This is a bound on the cost, and exceeding
#:      it is a real cost to be reported as one rather than a failed guess.
#:   4. **`nominal` is unchanged at 0.96**, with the rule firing on at most one
#:      of its 25 worlds -- the one failure §9.8 found, which is a believed-
#:      arrival and not a clearance problem, so firing there should change
#:      nothing either.
#:
#: The obvious way for this to fail: the eleven are tight worlds, and tight is
#: not only a planning property. If the robot now reaches the gap and cannot
#: thread it, recoveries will be near zero and collisions will rise -- which
#: would say the margin was not what ruled those worlds out, and that §9.9's
#: association was the confound it warned about.
PREDICTION = {
    "recovered_at_least": 4,
    "margin_safe_loss_at_most": 0.03,
    "clutter_collisions_at_most": 2,
    "nominal_success": 0.96,
    "nominal_triggers_at_most": 1,
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


def episode(env, seed: int, cfg, relax: int) -> dict:
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, relax_on_stall=relax)
    agent.start_episode(env.world, env.robot.pose)
    start = env.robot.pose[:2].copy()
    previous, driven, steps = start.copy(), 0.0, 0
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        here = env.robot.pose[:2].copy()
        driven += float(np.linalg.norm(here - previous))
        previous = here
        steps += 1
        if terminated or truncated:
            break
    shortest = float(info.get("shortest_path_length", np.nan))
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": steps,
        "driven": driven,
        "driven_over_shortest": driven / shortest if shortest else float("nan"),
        "goal_distance": float(info.get("goal_distance", np.nan)),
        "relaxed_at": int(agent.relaxed_at),
        "relaxed": bool(agent.relaxed_at >= 0),
        "replans": int(agent.replans),
        "recovery_steps": int(agent.recovery_steps),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--overlap", default="results/margin_overlap.json",
                   help="Which worlds have no margin-safe route, fixed in advance.")
    p.add_argument("--audit", default="results/margin_audit.json")
    p.add_argument("--out", default="results/margin_experiment.json")
    args = p.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))["conditions"]
    tight = {cond: {r["seed"] for r in audit[cond]["worlds"] if r["l_full"] is None}
             for cond in audit}

    report: dict = {"split": "val", "episodes": args.episodes,
                    "stall_steps": STALL_STEPS, "registered": PREDICTION,
                    "conditions": {}}
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
        arms = {"as_published": [episode(env, s, cfg, 0) for s in seeds],
                "relaxed": [episode(env, s, cfg, STALL_STEPS) for s in seeds]}
        report["conditions"][cond] = {k: {"episodes": v} for k, v in arms.items()}

        a = np.array([r["success"] for r in arms["as_published"]], dtype=bool)
        b = np.array([r["success"] for r in arms["relaxed"]], dtype=bool)
        pv, lost, won = mcnemar_p(a, b)
        fired = sum(r["relaxed"] for r in arms["relaxed"])
        colls = sum(r["collision"] for r in arms["relaxed"])
        report["conditions"][cond].update({
            "success": [float(a.mean()), float(b.mean())],
            "gain": float(b.mean() - a.mean()), "mcnemar_p": pv,
            "won": won, "lost": lost, "fired": int(fired),
            "collisions": [int(sum(r["collision"] for r in arms["as_published"])),
                           int(colls)],
        })
        print(f"{cond:8s} SR {a.mean():.2f} -> {b.mean():.2f} "
              f"({b.mean() - a.mean():+.3f})  McNemar p={pv:.3f} "
              f"({won} won / {lost} lost)  rule fired in {fired}/{len(seeds)}  "
              f"collisions {sum(r['collision'] for r in arms['relaxed'])}", flush=True)

        # The registered endpoint: the worlds §9.9 named, before this ran.
        marked = tight.get(cond, set())
        if marked:
            rows = {r["seed"]: r for r in arms["relaxed"]}
            base = {r["seed"]: r for r in arms["as_published"]}
            recovered = sorted(s for s in marked if rows[s]["success"])
            print(f"         of the {len(marked)} with no margin-safe route: "
                  f"{len(recovered)} recovered {recovered}, "
                  f"rule fired in {sum(rows[s]['relaxed'] for s in marked)}, "
                  f"collisions {sum(rows[s]['collision'] for s in marked)}, "
                  f"was {sum(base[s]['success'] for s in marked)} before")
            report["conditions"][cond]["tight"] = {
                "seeds": sorted(marked), "recovered": recovered,
                "fired": int(sum(rows[s]["relaxed"] for s in marked)),
                "collisions": int(sum(rows[s]["collision"] for s in marked)),
            }

    # ---- the registered endpoints, scored ----
    tight_all = [s for cond in report["conditions"]
                 for s in report["conditions"][cond].get("tight", {}).get("seeds", [])]
    recovered = [s for cond in report["conditions"]
                 for s in report["conditions"][cond].get("tight", {}).get("recovered", [])]
    safe_a, safe_b, clutter_colls = [], [], 0
    for cond in ("dense", "narrow"):
        c = report["conditions"].get(cond)
        if not c:
            continue
        marked = set(c.get("tight", {}).get("seeds", []))
        safe_a += [r["success"] for r in c["as_published"]["episodes"]
                   if r["seed"] not in marked]
        safe_b += [r["success"] for r in c["relaxed"]["episodes"]
                   if r["seed"] not in marked]
        clutter_colls += c["collisions"][1]

    held1 = len(recovered) >= PREDICTION["recovered_at_least"]
    loss = float(np.mean(safe_a) - np.mean(safe_b)) if safe_a else float("nan")
    held2 = loss <= PREDICTION["margin_safe_loss_at_most"]
    held3 = clutter_colls <= PREDICTION["clutter_collisions_at_most"]
    nom = report["conditions"].get("nominal")
    held4 = (nom is not None
             and round(nom["success"][1], 2) == PREDICTION["nominal_success"]
             and nom["fired"] <= PREDICTION["nominal_triggers_at_most"])
    report["endpoints"] = {
        "recovered": len(recovered), "of": len(tight_all), "held": bool(held1),
        "margin_safe_loss": loss, "margin_safe_held": bool(held2),
        "clutter_collisions": clutter_colls, "collisions_held": bool(held3),
        "nominal_held": bool(held4),
        "margin_safe_success": [float(np.mean(safe_a)) if safe_a else float("nan"),
                                float(np.mean(safe_b)) if safe_b else float("nan")],
    }
    print(f"\nendpoint 1 -- recovered {len(recovered)} of {len(tight_all)} worlds "
          f"with no margin-safe route, needed "
          f"{PREDICTION['recovered_at_least']}: {'HELD' if held1 else 'FAILED'}")
    print(f"endpoint 2 -- margin-safe worlds "
          f"{np.mean(safe_a):.3f} -> {np.mean(safe_b):.3f} "
          f"(loss {loss:+.3f}, bound {PREDICTION['margin_safe_loss_at_most']}): "
          f"{'HELD' if held2 else 'FAILED'}")
    print(f"endpoint 3 -- clutter collisions {clutter_colls}, bound "
          f"{PREDICTION['clutter_collisions_at_most']}: "
          f"{'HELD' if held3 else 'FAILED'}")
    if nom is not None:
        print(f"endpoint 4 -- nominal {nom['success'][1]:.2f}, rule fired in "
              f"{nom['fired']}: {'HELD' if held4 else 'FAILED'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
