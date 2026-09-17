"""Does the far end of the estimate's window cost the dense-clutter episodes?

Phase 5r: a constant-velocity estimate loses 0.065 on dense clutter against the
oracle, all contact with movers. Three explanations have since failed to account
for it -- a wider temporal margin (5r), the zero-margin fallback (5t), a stale
estimate (5u, unresolved; 0.050 remains with fresh estimates). The one named
first, in 5r's own prediction, and never tested is the far end of the window:
the planner carries each straight line seven seconds out, and holds its last
step past the window as a permanent obstacle.

This caps how far the line is carried. Arms, space-time agent, two-step
temporal margin, 200 episodes each:

  cv_m2    -- Phase 5r's estimating agent; must reproduce it episode for episode
  cv_cap2  -- the line carried at most 2 s past the latest observation (primary)
  cv_cap1  -- at most 1 s: the second executed before the next scheduled replan

Identity: a frozen mover has zero velocity, so capping changes nothing, and
cv_cap1 must be bit-identical to cv_m2 on every frozen-world episode. The oracle
agent is not re-run; its per-episode successes come from Phase 5r's result file,
checked there to reproduce Phase 5q.

    python scripts/cap_experiment.py --episodes 200
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
FROZEN = {"dynamic": "dynamic_frozen", "dynamic_dense": "dynamic_dense_frozen"}

#: Recorded before the run and committed to the repository before any result
#: exists.
#:
#: Confidence LOW -- this is the fourth explanation offered for the same 0.050,
#: and the three before it failed or came back unresolved. For: Phase 5t found
#: the estimate driving the planner into its last fallback in 31 dense episodes
#: against the oracle's 6, and a straight line held seven seconds out blocks
#: cells no mover will reach, so the planner is squeezed towards the movers that
#: are really there; a cap removes the phantoms and should let it keep its
#: margin. Against: 5t also showed that withholding the fallback changes
#: nothing, so being squeezed may not be what loses the episodes; and a capped
#: mover is believed to stop, which a plan can route behind.
PREDICTION = (
    "the far end is the cause: cv_cap2 minus cv_m2 on dynamic_dense is MATTERS "
    "(gain >= +0.03, p < 0.05), carried by collisions falling. Secondary: fewer "
    "dense episodes reach the bare-radius fallback under cv_cap2 than under cv_m2. "
    "Identity: cv_cap1 bit-identical to cv_m2 on every frozen episode. "
    "Decision -- REACH: MATTERS on dense. NOT REACH: INERT (bounded) or HARMS on "
    "dense. Otherwise UNRESOLVED."
)


def factory(cap):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor="constant_velocity",
                                              temporal_margin_steps=2,
                                              extrapolation_cap_s=cap), robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/cap_experiment.json")
    args = p.parse_args(argv)

    arms = {"cv_m2": factory(float("inf")), "cv_cap2": factory(2.0), "cv_cap1": factory(1.0)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    plan = [(a, c) for a in arms for c in MOVING]
    plan += [(a, FROZEN[c]) for a in ("cv_m2", "cv_cap1") for c in MOVING]
    for arm, cond in plan:
        rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
        runs[arm][cond] = rows
        print(f"  {arm:8s} {cond:22s} success {rate(rows, 'success'):.3f}  "
              f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}  "
              f"reach bare radius {sum(r['bare_radius_attempts'] > 0 for r in rows)}", flush=True)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "checks": {},
              "cells": {}, "contrasts": {}}
    identity = all(runs["cv_m2"][c] == runs["cv_cap1"][c] for c in FROZEN.values())
    report["checks"]["frozen_identity_cap1_vs_uncapped"] = identity
    print("\nfrozen identity, capped vs uncapped:", identity)

    est = json.loads(Path("results/estimate_experiment.json").read_text(encoding="utf-8"))["cells"]
    repro = all([int(r["success"]) for r in runs["cv_m2"][c]] == est["cv_m2"][c]["success_per_episode"]
                for c in MOVING)
    report["checks"]["reproduces_phase_5r"] = repro
    print("cv_m2 reproduces Phase 5r on every episode:", repro)
    # Episode i is the same world at any episode count, so a shorter run pairs
    # with the head of the stored list.
    oracle = {c: [{"success": bool(s)}
                  for s in est["oracle_m2"][c]["success_per_episode"][:args.episodes]]
              for c in MOVING}

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "episodes_reaching_bare_radius": sum(r["bare_radius_attempts"] > 0 for r in rows),
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated, ref in (("cv_cap2", "cv_m2"), ("cv_cap1", "cv_m2"), ("cv_cap2", "oracle_m2")):
        for cond in MOVING:
            a = oracle[cond] if ref == "oracle_m2" else runs[ref][cond]
            b = runs[treated][cond]
            gain = rate(b, "success") - rate(a, "success")
            p_val, lost, won = mcnemar_p(succ(a), succ(b))
            ci = paired_ci(a, b, rng, args.bootstrap)
            label = verdict(gain, p_val, ci)
            entry = {"success_gain": gain, "p": p_val, "episodes_won": won,
                     "episodes_lost": lost, "ci95": ci, "verdict": label}
            if ref != "oracle_m2":
                entry["collision_delta"] = rate(b, "collision") - rate(a, "collision")
                entry["timeout_delta"] = rate(b, "timeout") - rate(a, "timeout")
            report["contrasts"].setdefault(f"{treated}_vs_{ref}", {})[cond] = entry
            print(f"  {treated:7s} vs {ref:9s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  {label}", flush=True)

    dense = report["contrasts"]["cv_cap2_vs_cv_m2"]["dynamic_dense"]["verdict"]
    report["decision"] = ("REACH" if dense == "MATTERS"
                          else "NOT REACH" if dense in ("INERT (bounded)", "HARMS")
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
