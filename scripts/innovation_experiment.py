"""Does acting on a stale estimate let movers close in?

Phase 5r: a constant-velocity estimate loses 0.065 on dense clutter against the
oracle, all of it contact with movers. Phase 5t: withholding zero-margin plans
recovers nothing -- the robot is already too close by the time the planner falls
back to one. The candidate left is how it gets close. Between scheduled replans
the agent acts for up to a second on the estimate made at the last plan, and a
straight line drifts from a sinusoid as that second goes on.

This replans the moment an observed mover is more than a threshold from where
that estimate put it. Arms, space-time agent, two-step temporal margin, 200
episodes each:

  cv_m2       -- Phase 5r's estimating agent; must reproduce it episode for episode
  cv_i05      -- the same, replanning on a 0.05 m contradiction
  cv_i02      -- the same, replanning on a 0.02 m contradiction (primary)
  oracle_m2   -- Phase 5r's oracle agent; must reproduce it
  oracle_i02  -- the control: the oracle is never contradicted, so this arm must be
                 bit-identical to oracle_m2 on moving worlds, episode for episode

    python scripts/innovation_experiment.py --episodes 200
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

#: Recorded before the run and committed to the repository before any result
#: exists.
#:
#: Confidence LOW. For: at contact, the estimate behind the plan in force was a
#: median 0.03 m wrong (Phase 5r), so a 0.02 m trigger would have replanned on
#: fresh observations during the approach, and a fresh straight line is nearly
#: exact over the next tenth of a second. Against: replanning more often is
#: exactly the churn Phase 5d measured costing the spatial planner in clutter,
#: which would show as timeouts; and the approach may be set by choices made
#: seconds earlier on the far end of the window, which no near-term trigger
#: touches. The 0.03 m is itself a post hoc description, the kind Phase 5t just
#: showed can be right about the episodes and wrong about the cause.
PREDICTION = (
    "staleness is the cause: cv_i02 minus cv_m2 on dynamic_dense is MATTERS "
    "(gain >= +0.03, p < 0.05), carried by collisions falling, with timeouts rising "
    "by less than 0.02. Dose: cv_i05 gains less than cv_i02 on dense. Control: "
    "oracle_i02 bit-identical to oracle_m2 on every episode. "
    "Decision -- STALENESS: MATTERS on dense. NOT STALENESS: INERT (bounded) or "
    "HARMS on dense. Otherwise UNRESOLVED."
)


def factory(predictor, threshold):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor=predictor, temporal_margin_steps=2,
                                              replan_innovation_m=threshold), robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/innovation_experiment.json")
    args = p.parse_args(argv)

    arms = {"cv_m2": factory("constant_velocity", 0.0),
            "cv_i05": factory("constant_velocity", 0.05),
            "cv_i02": factory("constant_velocity", 0.02),
            "oracle_m2": factory("oracle", 0.0),
            "oracle_i02": factory("oracle", 0.02)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    for arm in arms:
        for cond in MOVING:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
            runs[arm][cond] = rows
            print(f"  {arm:11s} {cond:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}  "
                  f"replans/episode {np.mean([r['replans'] for r in rows]):.1f} "
                  f"(triggered {np.mean([r['innovation_replans'] for r in rows]):.1f})",
                  flush=True)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "checks": {},
              "cells": {}, "contrasts": {}}

    identity = all(runs["oracle_m2"][c] == runs["oracle_i02"][c] for c in MOVING)
    report["checks"]["oracle_identity_on_moving_worlds"] = identity
    print("\noracle with the trigger bit-identical to without, on moving worlds:", identity)

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
                "replans_per_episode": float(np.mean([r["replans"] for r in rows])),
                "triggered_per_episode": float(np.mean([r["innovation_replans"] for r in rows])),
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated, ref in (("cv_i02", "cv_m2"), ("cv_i05", "cv_m2"), ("cv_i02", "oracle_m2")):
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
            print(f"  {treated:7s} vs {ref:9s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  {label}", flush=True)

    dense = report["contrasts"]["cv_i02_vs_cv_m2"]["dynamic_dense"]["verdict"]
    report["decision"] = ("STALENESS" if dense == "MATTERS"
                          else "NOT STALENESS" if dense in ("INERT (bounded)", "HARMS")
                          else "UNRESOLVED")
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
