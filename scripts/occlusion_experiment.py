"""What does it cost to see only what a sensor could see?

Phase 6a closed the estimator chain: fitting each mover's oscillation to the
robot's own noisy observations plans as well as an oracle. Those observations
still cover every mover in the world at every step -- through walls, behind the
robot, at any range. That is the last privilege, and a larger one than
exactness was.

Here the estimator observes a mover only when it is within range, within the
sensor's field of view, and not hidden behind static geometry. A mover it cannot
see yields no observation and the estimate coasts; a mover it has never seen is
not in the planner's grid at all, and the robot can drive into it.

Arms, the Phase 6a estimator at its centimetre of noise, 200 episodes each:

  all_round      sees everything, as Phase 6a did: the reproduction check
  lidar          360 degrees, 6 m, occluded by walls -- a planar scanner
  camera         90 degrees, 6 m, occluded -- the depth camera's geometry

The oracle reads no observations, so it is unaffected by any of this; Phase 5r's
oracle episodes are the reference. Frozen worlds are not run: a parked mover is
as visible or as hidden as a moving one, and the identity that mattered there
(the fit declining) is unit-tested.

    python scripts/occlusion_experiment.py --episodes 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spacetime_experiment import SPATIAL, paired_ci, succ, verdict  # noqa: E402
from velocity_experiment import mcnemar_p, rate, run_cell  # noqa: E402

from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig  # noqa: E402

MOVING = ("dynamic", "dynamic_dense")
SIGMA = 0.01

#: Recorded before the run and committed before any result exists.
#:
#: Confidence LOW on the size, MODERATE on the order. Occlusion should cost
#: something: `dynamic_dense` is built of boxes and circles a mover can hide
#: behind, and a mover first seen at three metres gives the fit no track to
#: work from. A 90 degree camera should cost much more than a 360 degree
#: scanner, because a robot driving forward has most of the world behind it and
#: these movers cross its path from the side. What nothing here measures is how
#: often a mover is hidden at the moment it matters, which is the quantity the
#: cost actually depends on -- so the numbers below are guesses in a way the
#: Phase 6a prediction was not.
PREDICTION = (
    "seeing less costs something, and the camera costs much more: lidar minus "
    "all_round on dynamic_dense is negative and outside a bounded null (not "
    "INERT), between -0.02 and -0.10; camera minus all_round is worse still, "
    "at least -0.10, and HARMS. Both remain far better than Phase 5z's "
    "straight line at the same noise (0.815 on dense). "
    "Decision on lidar minus all_round, dynamic_dense -- CHEAP: INERT (bounded). "
    "COSTLY: HARMS. Otherwise UNRESOLVED."
)


def factory(visible_only, fov=360.0):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor="orbit", temporal_margin_steps=2,
                                              observation_noise_m=SIGMA,
                                              observe_visible_only=visible_only,
                                              sensor_fov_deg=fov), robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/occlusion_experiment.json")
    args = p.parse_args(argv)

    arms = {"all_round": factory(False), "lidar": factory(True, 360.0),
            "camera": factory(True, 90.0)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    for arm in arms:
        for cond in MOVING:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
            runs[arm][cond] = rows
            print(f"  {arm:10s} {cond:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}",
                  flush=True)

    report = {"episodes": args.episodes, "sigma": SIGMA, "prediction": PREDICTION,
              "checks": {}, "cells": {}, "contrasts": {}}

    prior = json.loads(Path("results/orbit_experiment.json").read_text(encoding="utf-8"))["cells"]
    repro = all([int(r["success"]) for r in runs["all_round"][c]]
                == prior["orbit@0.01"][c]["success_per_episode"][:args.episodes] for c in MOVING)
    report["checks"]["reproduces_phase_6a"] = repro
    print("\nall_round reproduces Phase 6a on every episode:", repro)

    est = json.loads(
        Path("results/estimate_experiment.json").read_text(encoding="utf-8"))["cells"]
    noise = json.loads(
        Path("results/noise_experiment.json").read_text(encoding="utf-8"))["cells"]
    refs = {
        "all_round": {c: runs["all_round"][c] for c in MOVING},
        "oracle": {c: [{"success": bool(s)} for s in
                       est["oracle_m2"][c]["success_per_episode"][:args.episodes]]
                   for c in MOVING},
        "line@0.01": {c: [{"success": bool(s)} for s in
                          noise["constant_velocity@0.01"][c]["success_per_episode"][:args.episodes]]
                      for c in MOVING},
    }

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated in ("lidar", "camera"):
        for ref_name, ref in refs.items():
            for cond in MOVING:
                a, b = ref[cond], runs[treated][cond]
                gain = rate(b, "success") - rate(a, "success")
                p_val, lost, won = mcnemar_p(succ(a), succ(b))
                ci = paired_ci(a, b, rng, args.bootstrap)
                entry = {"success_gain": gain, "p": p_val, "episodes_won": won,
                         "episodes_lost": lost, "ci95": ci, "verdict": verdict(gain, p_val, ci)}
                if "collision" in a[0]:
                    entry["collision_delta"] = rate(b, "collision") - rate(a, "collision")
                    entry["timeout_delta"] = rate(b, "timeout") - rate(a, "timeout")
                report["contrasts"].setdefault(f"{treated}_vs_{ref_name}", {})[cond] = entry
                print(f"  {treated:7s} vs {ref_name:10s} {cond:14s} {gain:+.3f}  "
                      f"p {p_val:.4f} (+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  "
                      f"{entry['verdict']}", flush=True)

    label = report["contrasts"]["lidar_vs_all_round"]["dynamic_dense"]["verdict"]
    report["decision"] = ("CHEAP" if label == "INERT (bounded)"
                          else "COSTLY" if label == "HARMS" else "UNRESOLVED")
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if repro else 1


if __name__ == "__main__":
    raise SystemExit(main())
