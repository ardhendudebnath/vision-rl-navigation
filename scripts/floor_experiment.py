"""Is the estimate's zero-margin fallback the cause of its collisions, or a symptom?

Phase 5r replaced the oracle with a constant-velocity estimate and lost 0.065 on
dense clutter, all of it contact with movers, none recovered by a wider temporal
margin. Replaying the 18 lost episodes found that in 15 the plan in force had
been made at the bare robot radius -- the planner's last fallback, no spatial
margin -- around a mover the estimate had a median 0.03 m wrong.

That is either the cause (a plan with no margin is safe only under exact
prediction, and the fallback hands one to an inexact planner) or a symptom (the
planner falls back when a mover is already close, and the robot got close for
some other reason). This withholds the bare radius from movers -- the static map
may still use it for a tight passage -- and asks whether collisions fall.

Arms, space-time agent, two-step temporal margin, 200 episodes each:

  cv_m2         -- Phase 5r's estimating agent; must reproduce it episode for episode
  cv_floor      -- the same, movers never inflated below half the safety margin
  oracle_m2     -- Phase 5r's oracle agent, re-run to count its fallbacks
  oracle_floor  -- the control: what withholding the fallback does when the
                   prediction is exact, and a zero-margin plan is safe

Identity: until the planner first reaches the bare radius in an episode the floor
cannot change anything, so every episode in which the base arm never attempted
it must be bit-identical in the floored arm. Unit-tested on episodes that reach
the middle fallback, where a leaking floor would show; checked here on all 800.

    python scripts/floor_experiment.py --episodes 200
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
FLOOR = 0.5

#: Recorded before the run and committed to the repository before any result
#: exists.
#:
#: Confidence LOW, and the reasoning runs both ways. For CAUSE: a bare-radius
#: plan is only adopted when the full- and half-margin searches both fail, and
#: when it is withheld the agent either finds a route with half the margin
#: around movers or keeps its previous plan, which was made with a margin and
#: whose estimate is at most a second old -- either way a route with room for a
#: 0.03 m error instead of one with none.
#: For SYMPTOM: the fallback is reached most often when a mover's dilated
#: occupancy already covers the robot's own cell, and a previous plan made
#: before that mover arrived may lead nowhere safer. The one observation behind
#: the lean -- 15 of 18 plans in force at the bare radius -- is a post hoc
#: description of eighteen episodes, not a measurement of this regime.
PREDICTION = (
    "the fallback is the cause: cv_floor minus cv_m2 on dynamic_dense is MATTERS "
    "(gain >= +0.03, p < 0.05), between +0.03 and +0.065 (no more than the whole "
    "5r loss), carried by collisions falling, with timeouts rising by less than "
    "0.02. Control: oracle_floor minus oracle_m2 bounded-INERT (95% CI inside "
    "+/-0.03) on both conditions. Identity holds. "
    "Decision -- CAUSE: MATTERS on dense. SYMPTOM: INERT (bounded) or HARMS on "
    "dense. Otherwise UNRESOLVED."
)


def factory(predictor, floor):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor=predictor, temporal_margin_steps=2,
                                              mover_margin_floor=floor), robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/floor_experiment.json")
    args = p.parse_args(argv)

    arms = {"cv_m2": factory("constant_velocity", 0.0),
            "cv_floor": factory("constant_velocity", FLOOR),
            "oracle_m2": factory("oracle", 0.0),
            "oracle_floor": factory("oracle", FLOOR)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    for arm in arms:
        for cond in MOVING:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
            runs[arm][cond] = rows
            print(f"  {arm:13s} {cond:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}  "
                  f"episodes reaching bare radius "
                  f"{sum(r['bare_radius_attempts'] > 0 for r in rows)}", flush=True)

    report = {"episodes": args.episodes, "floor": FLOOR, "prediction": PREDICTION,
              "checks": {}, "cells": {}, "contrasts": {}}

    identity_ok = True
    for base, floored in (("cv_m2", "cv_floor"), ("oracle_m2", "oracle_floor")):
        for cond in MOVING:
            pairs = list(zip(runs[base][cond], runs[floored][cond], strict=True))
            untouched = [(a, b) for a, b in pairs if a["bare_radius_attempts"] == 0]
            broken = sum(a != b for a, b in untouched)
            changed = sum(a["success"] != b["success"] for a, b in pairs)
            report["checks"][f"identity_{floored}_{cond}"] = {
                "episodes_never_reaching_bare_radius": len(untouched),
                "of_those_not_identical": broken,
                "outcomes_changed": changed,
            }
            identity_ok &= broken == 0
            print(f"identity {floored:12s} {cond:14s}: {len(untouched)} episodes never reach "
                  f"the bare radius, {broken} of them differ; {changed} outcomes changed")

    prior = Path("results/estimate_experiment.json")
    if prior.exists():
        est = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        repro = all([int(r["success"]) for r in runs[arm][c]] == est[arm][c]["success_per_episode"]
                    for arm in ("cv_m2", "oracle_m2") for c in MOVING)
        report["checks"]["reproduces_phase_5r"] = repro
        print("cv_m2 and oracle_m2 reproduce Phase 5r on every episode:", repro)

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "episodes_reaching_bare_radius": sum(r["bare_radius_attempts"] > 0 for r in rows),
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated, ref in (("cv_floor", "cv_m2"), ("oracle_floor", "oracle_m2"),
                         ("cv_floor", "oracle_m2")):
        for cond in MOVING:
            a, b = runs[ref][cond], runs[treated][cond]
            gain = rate(b, "success") - rate(a, "success")
            p_val, lost, won = mcnemar_p(succ(a), succ(b))
            ci = paired_ci(a, b, rng, args.bootstrap)
            label = verdict(gain, p_val, ci)
            report["contrasts"].setdefault(f"{treated}_vs_{ref}", {})[cond] = {
                "success_gain": gain, "p": p_val, "episodes_won": won, "episodes_lost": lost,
                "ci95": ci, "verdict": label,
                "collision_delta": rate(b, "collision") - rate(a, "collision"),
                "timeout_delta": rate(b, "timeout") - rate(a, "timeout"),
            }
            print(f"  {treated:12s} vs {ref:9s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  {label}", flush=True)

    dense = report["contrasts"]["cv_floor_vs_cv_m2"]["dynamic_dense"]["verdict"]
    report["decision"] = ("CAUSE" if dense == "MATTERS"
                          else "SYMPTOM" if dense in ("INERT (bounded)", "HARMS")
                          else "UNRESOLVED")
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
