"""The other half of the motion cost: what else still acts on snapshots?

Phase 5m gave the classical planner oracle knowledge of where movers are going,
for replanning and the replan trigger, and recovered about half the motion
cost on `dynamic_dense`. Because that prediction is an oracle, the remaining
half survives perfect motion knowledge *at the planner*. Report Section 12
named the two places that still read movers as snapshots:

- **the initial plan**, left static-only in 5m so its arms differed in
  replanning alone;
- **the controller's reactive slow-down**, which reads clearance to where a
  mover currently is.

Both are tested here against 5m's best configuration (2 s horizon).

"The initial plan sees movers" is two interventions, and only one has a clean
control. Seeing movers *at all* changes frozen worlds too -- a frozen mover is
an obstacle the map lacks -- so it has no identity control. Seeing *where they
are going* changes only moving worlds, so ``predicted`` against ``current`` is
bit-identical on frozen cells, as is prediction in the slow-down. Those two
identities are checked before anything is concluded.

    python scripts/snapshot_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from velocity_experiment import CELLS, mcnemar_p, rate, run_cell  # noqa: E402

from vision_nav.agents.classical import PursuitConfig  # noqa: E402

H = 2.0

#: name -> (config, description). base2 is Phase 5m's best and must reproduce it.
ARMS = {
    "base0": (PursuitConfig(replan_on_block=True), "no prediction"),
    "base2": (PursuitConfig(replan_on_block=True, predict_horizon=H),
              "5m best: prediction in replanning"),
    "initS": (PursuitConfig(replan_on_block=True, predict_horizon=H,
                            initial_plan_movers="current"),
              "+ initial plan sees movers where they are"),
    "initP": (PursuitConfig(replan_on_block=True, predict_horizon=H,
                            initial_plan_movers="predicted"),
              "+ initial plan sees where movers are going"),
    "cauP": (PursuitConfig(replan_on_block=True, predict_horizon=H,
                           predict_caution=True),
             "+ slow-down reads where movers are going"),
}

#: (label, treated, reference, has identity control on frozen cells)
CONTRASTS = [
    ("initial plan sees movers at all", "initS", "base2", False),
    ("initial plan predicts", "initP", "initS", True),
    ("slow-down predicts", "cauP", "base2", True),
]

#: Recorded before the run.
#:
#: The first line is measurement-derived: in 8 of 8 diagnostic episodes no
#: mover's disc lay on the static route at t = 0, so sensing movers there
#: should change nothing. The other two are intuition. The initial plan is
#: replaced at the first replan, and 5m's arm replans about four times an
#: episode, so its influence should be brief; slowing for an approaching mover
#: trades collisions for timeouts. Two significance tests of interest, so
#: alpha is corrected to 0.05/2.
PREDICTION = (
    "initS identical to base2 on every episode (movers never start on the route); "
    "initP vs initS |delta| < 0.03, not significant; cauP vs base2 +0.00 to +0.04, "
    "not significant. Neither closes the remaining half: it is not an information "
    "problem at planning or at reactive control"
)
ALPHA = 0.05 / 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/snapshot_experiment.json")
    args = p.parse_args(argv)

    runs: dict[str, dict[str, list[dict]]] = {}
    for name, (cfg, desc) in ARMS.items():
        print(f"\n=== {name}: {desc} ===")
        runs[name] = {}
        for label, cond in CELLS:
            rows = run_cell(cond, args.episodes, cfg)
            runs[name][cond] = rows
            print(f"  {label:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  "
                  f"timeout {rate(rows, 'timeout'):.3f}  "
                  f"replans {np.mean([r['replans'] for r in rows]):4.1f}")

    report = {"episodes": args.episodes, "horizon": H, "prediction": PREDICTION,
              "alpha_corrected": ALPHA, "checks": {}, "cells": {}, "contrasts": {}}

    # --- reproduction of Phase 5m -----------------------------------------
    print("\n=== does base2 reproduce Phase 5m exactly? ===")
    prior = Path("results/velocity_experiment.json")
    reproduces = None
    if prior.exists():
        vm = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        reproduces = True
        for arm, h in (("base0", "0.0"), ("base2", "2.0")):
            for _label, cond in CELLS:
                for k in ("success", "collision", "timeout"):
                    if rate(runs[arm][cond], k) != vm[cond][h][k]:
                        reproduces = False
                        print(f"  DIFFERS: {arm}/{cond}/{k}")
        print("  yes, every rate" if reproduces else "  NO -- investigate first")
    report["checks"]["reproduces_phase_5m"] = reproduces

    # --- identity controls --------------------------------------------------
    print("\n=== frozen identities (the controls) ===")
    identity_ok = True
    for label, treated, ref, has_identity in CONTRASTS:
        if not has_identity:
            continue
        for cond in ("dynamic_frozen", "dynamic_dense_frozen"):
            same = all(a == b for a, b in zip(runs[ref][cond], runs[treated][cond],
                                              strict=True))
            if not same:
                identity_ok = False
                print(f"  VIOLATED: {label} on {cond}")
    print("  hold" if identity_ok else "  CONTROL FAILED -- results void")
    report["checks"]["frozen_identities_hold"] = identity_ok

    for _label, cond in CELLS:
        report["cells"][cond] = {
            name: {**{k: rate(runs[name][cond], k)
                      for k in ("success", "collision", "timeout")},
                   "replans_mean": float(np.mean([r["replans"] for r in runs[name][cond]])),
                   "per_episode": runs[name][cond]}
            for name in ARMS
        }

    # --- contrasts ------------------------------------------------------------
    print("\n=== contrasts (paired, McNemar) ===")
    for label, treated, ref, has_identity in CONTRASTS:
        report["contrasts"][label] = {"treated": treated, "reference": ref,
                                      "has_identity_control": has_identity}
        for cond in ("dynamic", "dynamic_frozen", "dynamic_dense", "dynamic_dense_frozen"):
            a = np.array([r["success"] for r in runs[ref][cond]], dtype=bool)
            b = np.array([r["success"] for r in runs[treated][cond]], dtype=bool)
            identical = all(x == y for x, y in zip(runs[ref][cond], runs[treated][cond],
                                                   strict=True))
            p_val, lost, won = mcnemar_p(a, b)
            gain = float(b.mean() - a.mean())
            d_coll = rate(runs[treated][cond], "collision") - rate(runs[ref][cond], "collision")
            d_tmo = rate(runs[treated][cond], "timeout") - rate(runs[ref][cond], "timeout")
            report["contrasts"][label][cond] = {
                "success_gain": gain, "p": p_val, "episodes_won": won,
                "episodes_lost": lost, "collision_delta": d_coll,
                "timeout_delta": d_tmo, "identical_every_episode": identical,
            }
            tag = "identical" if identical else f"p {p_val:.4f} (+{won}/-{lost})"
            sig = "  <-- significant at corrected alpha" if p_val < ALPHA else ""
            print(f"  {label:32s} {cond:22s} {gain:+.3f}  coll {d_coll:+.3f}  "
                  f"tmo {d_tmo:+.3f}  {tag}{sig}")

    base = report["cells"]
    for moving, frozen in (("dynamic", "dynamic_frozen"),
                           ("dynamic_dense", "dynamic_dense_frozen")):
        remaining = base[frozen]["base2"]["success"] - base[moving]["base2"]["success"]
        report["checks"][f"{moving}_cost_remaining_after_5m"] = remaining
        print(f"\n{moving}: motion cost remaining after Phase 5m = {remaining:+.3f}")

    print("\npre-registered: " + PREDICTION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if (identity_ok and reproduces is not False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
