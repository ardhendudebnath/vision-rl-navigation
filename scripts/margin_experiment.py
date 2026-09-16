"""Does a temporal safety margin remove the harm timing did on sparse worlds?

Phase 5p: knowing *when* movers occupy cells was worth +0.050 on dense clutter
and cost -0.050 on sparse worlds, all of it collisions, with ten episodes lost
and none won. The cause offered, and not tested: the planner threads gaps one
0.24 s plan step ahead of a mover, so any lag in tracking the schedule puts the
robot where the mover arrives. A margin -- a mover's cells count as blocked for
``m`` steps either side of its occupancy -- is the test of that cause.

Arms, all the full space-time agent on moving worlds unless noted:

  m0      -- no margin; must reproduce Phase 5p's full arm exactly
  m1, m2, m4
  swept   -- the when-blind ablation; must reproduce Phase 5p's swept arm

Controls: on frozen worlds a margin widens nothing, so m0 and m4 must be
bit-identical there; and m0 and swept must match Phase 5p episode for episode,
which shows nothing but the margin changed.

    python scripts/margin_experiment.py --episodes 200
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

MARGINS = (0, 1, 2, 4)
PRIMARY = 2
MOVING = ("dynamic", "dynamic_dense")
FROZEN = {"dynamic": "dynamic_frozen", "dynamic_dense": "dynamic_dense_frozen"}

#: Recorded before the run and committed to the repository before any result
#: exists. Null bounded at +/-0.03.
#:
#: The primary margin is 2 steps (0.47 s), chosen to match the tracker's 0.5 s
#: lead rather than picked from the sweep afterwards. The magnitude is derived
#: from Phase 5p's measured harm: ten episodes lost on sparse, so a margin that
#: removes the cause should win most of them back. That is a measurement in the
#: regime being tested, but the mechanism behind it is a hypothesis nothing has
#: yet tested -- which is exactly the crossing this project's calibration record
#: says to distrust.
PREDICTION = (
    "H1: m2 minus m0 on dynamic is +0.03 to +0.05, p < 0.05, through fewer "
    "collisions -- the margin removes the sparse harm. H2: m2 minus swept on "
    "dynamic_dense still MATTERS (>= +0.03, p < 0.05) -- the dense gain survives. "
    "FIXED if both hold. CAUSE REFUTED if m2 minus m0 on dynamic is bounded-INERT "
    "and sparse collisions do not fall. m4 erodes the dense gain towards swept."
)


def factory(margin=None, time_varying=True):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(time_varying_movers=time_varying,
                                              temporal_margin_steps=margin or 0),
                              robot=robot)
    return make


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/margin_experiment.json")
    args = p.parse_args(argv)

    arms = {f"m{m}": factory(m) for m in MARGINS}
    arms["swept"] = factory(time_varying=False)
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}

    for arm, make in arms.items():
        for cond in MOVING:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=make)
            runs[arm][cond] = rows
            print(f"  {arm:6s} {cond:22s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}  "
                  f"waits {np.mean([r['planned_waits'] for r in rows]):4.1f}", flush=True)
    for arm in ("m0", "m4"):
        for cond in FROZEN.values():
            runs[arm][cond] = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
            print(f"  {arm:6s} {cond:22s} success {rate(runs[arm][cond], 'success'):.3f}",
                  flush=True)

    report = {"episodes": args.episodes, "primary_margin": PRIMARY, "prediction": PREDICTION,
              "checks": {}, "cells": {}, "contrasts": {}}

    identity = all(runs["m0"][c] == runs["m4"][c] for c in FROZEN.values())
    report["checks"]["frozen_identity_m0_vs_m4"] = identity
    print("\nfrozen identity, margin 0 vs 4:", identity)

    prior = Path("results/spacetime_experiment.json")
    if prior.exists():
        st = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        repro = all(
            [int(r["success"]) for r in runs[mine][c]] == st[c][theirs]["success_per_episode"]
            for mine, theirs in (("m0", "full"), ("swept", "swept"))
            for c in MOVING
        )
        report["checks"]["reproduces_phase_5p"] = repro
        print("m0 and swept reproduce Phase 5p on every episode:", repro)

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "mean_planned_waits": float(np.mean([r["planned_waits"] for r in rows])),
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    pairs = [(f"m{m}", "m0") for m in MARGINS if m] + [(f"m{m}", "swept") for m in MARGINS]
    for treated, ref in pairs:
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
            print(f"  {treated:>3s} vs {ref:5s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  "
                  f"coll {rate(b, 'collision') - rate(a, 'collision'):+.3f}  {label}", flush=True)

    h1 = report["contrasts"][f"m{PRIMARY}_vs_m0"]["dynamic"]
    h2 = report["contrasts"][f"m{PRIMARY}_vs_swept"]["dynamic_dense"]
    fixed = h1["verdict"] == "MATTERS" and h2["verdict"] == "MATTERS"
    refuted = h1["verdict"] == "INERT (bounded)" and h1["collision_delta"] >= 0
    report["decision"] = "FIXED" if fixed else "CAUSE REFUTED" if refuted else "neither"
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
