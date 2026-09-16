"""How much of the oracle's motion-cost removal survives an estimate?

Phase 5q: with exact mover trajectories, a space-time planner with a two-step
temporal margin leaves a motion cost of 0.010 on dense clutter and 0.000 on
sparse worlds. No real robot has exact trajectories. This replaces them with the
simplest real estimator -- constant-velocity extrapolation from the last two
positions the agent itself observed -- and measures what that costs.

Observations are noise-free, so this isolates *model* error: movers run on
sinusoids and a straight-line extrapolation drifts as they curve. Sensing noise
is a separate question.

Arms, all the space-time agent at 200 episodes:

  oracle_m2  -- Phase 5q's best; must reproduce it episode for episode
  cv_m2      -- constant velocity, same two-step margin
  cv_m4      -- constant velocity, four steps: does a wider margin absorb the
                estimation error, if there is any?

**cv_m2 against oracle_m2 is the test**: the same agent and margin on the same
worlds, differing only in where the mover futures come from.

Controls: a frozen mover has zero velocity, so its estimate is exact, and cv_m2
and oracle_m2 must be bit-identical on frozen worlds.

    python scripts/estimate_experiment.py --episodes 200
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
#: exists. Null bounded at +/-0.03.
#:
#: Derived from a calculation, not a measurement. A constant-velocity
#: extrapolation of A sin(wt) drifts by about (1/2) A w^2 tau^2. Peak speed is
#: A w, so A w^2 = speed^2 / A <= 0.45^2 / 1.0 = 0.2 m/s^2 for the fastest,
#: shortest-stroke mover, and over the one second executed before each replan the
#: drift is at most ~0.1 m -- under the planner's 0.18 m spatial safety margin
#: when it plans at full margin, before the temporal margin adds more.
#:
#: The same arithmetic flags where it could fail. The planner reads the whole
#: 7 s window, not just the executed second, and holds the last slice as a
#: permanent obstacle (tail_occ). Seven seconds out a straight line can be 3 m
#: from a mover that has turned back, and a phantom that blocks a corridor can
#: steer the executed head of the plan or make the search fail. That would show
#: as timeouts on dense clutter, not collisions, and a wider temporal margin
#: would not help it. A near-term error would show as collisions that it would.
#:
#: Nothing has measured this stack with estimated trajectories, which by this
#: project's calibration rule makes the prediction a crossing into unmeasured
#: territory, however careful the arithmetic.
PREDICTION = (
    "estimation is cheap: cv_m2 minus oracle_m2 is bounded-INERT (paired 95% CI "
    "inside +/-0.03) on both dynamic and dynamic_dense. Frozen identical. "
    "Decision -- CHEAP: INERT on both. COSTLY: HARMS on either. Otherwise PARTIAL. "
    "Secondary, if not CHEAP: the loss is on dynamic_dense, carried by timeouts "
    "rather than collisions, and cv_m4 does not recover it."
)


def factory(predictor, margin):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor=predictor,
                                              temporal_margin_steps=margin), robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/estimate_experiment.json")
    args = p.parse_args(argv)

    arms = {"oracle_m2": factory("oracle", 2), "cv_m2": factory("constant_velocity", 2),
            "cv_m4": factory("constant_velocity", 4)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    plan = [(a, c) for a in arms for c in MOVING]
    plan += [(a, FROZEN[c]) for a in ("oracle_m2", "cv_m2") for c in MOVING]
    for arm, cond in plan:
        rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
        runs[arm][cond] = rows
        print(f"  {arm:10s} {cond:22s} success {rate(rows, 'success'):.3f}  "
              f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}",
              flush=True)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "checks": {},
              "cells": {}, "contrasts": {}}

    identity = all(runs["oracle_m2"][c] == runs["cv_m2"][c] for c in FROZEN.values())
    report["checks"]["frozen_identity_cv_vs_oracle"] = identity
    print("\nfrozen identity, estimate vs oracle:", identity)

    prior = Path("results/margin_experiment.json")
    if prior.exists():
        mg = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        repro = all([int(r["success"]) for r in runs["oracle_m2"][c]]
                    == mg["m2"][c]["success_per_episode"] for c in MOVING)
        report["checks"]["oracle_reproduces_phase_5q"] = repro
        print("oracle_m2 reproduces Phase 5q on every episode:", repro)

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated, ref in (("cv_m2", "oracle_m2"), ("cv_m4", "oracle_m2"), ("cv_m4", "cv_m2")):
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
            print(f"  {treated:6s} vs {ref:9s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  {label}", flush=True)

    for cond in MOVING:
        cost = {arm: rate(runs["oracle_m2"][FROZEN[cond]], "success") - rate(runs[arm][cond], "success")
                for arm in runs}
        report["checks"][f"motion_cost_{cond}"] = cost
        print(f"motion cost on {cond}: " + ", ".join(f"{k} {v:+.3f}" for k, v in cost.items()))

    primary = report["contrasts"]["cv_m2_vs_oracle_m2"]
    labels = [primary[c]["verdict"] for c in MOVING]
    report["decision"] = ("CHEAP" if all(v == "INERT (bounded)" for v in labels)
                          else "COSTLY" if any(v == "HARMS" for v in labels) else "PARTIAL")
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
