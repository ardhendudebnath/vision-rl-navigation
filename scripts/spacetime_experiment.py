"""Does reasoning about *when* close the rest of the motion cost?

Phases 5m-5o left one explanation standing for the half of the classical stack's
motion cost that oracle prediction cannot recover: a planner that reasons only in
space uses a perfect input crudely. It can route around where a mover goes, but
not wait for it to pass or go through before it arrives. A robot twice as agile
did not help (5o), which is what that explanation predicts.

Three arms, all with oracle mover trajectories:

  spatial  -- Phase 5m's best: spatial A*, 2 s swept prediction, pure pursuit
  swept    -- the space-time agent, but every mover blocks the union of the
              cells it occupies anywhere in the window, at every step
  full     -- the space-time agent, knowing when each cell holds a mover

**full against swept is the test.** The space-time agent differs from the
spatial one in its window, its replan timer, schedule tracking and
step-minimising paths as well as in reasoning about time, so full against
spatial cannot be credited to timing. full and swept share all of that and
differ in exactly one thing: whether the planner knows *when*.

Controls, checked before anything is concluded:

- **Identity.** On frozen worlds a mover's union over the window equals every
  step, so full and swept must be bit-identical there on every episode.
- **Reproduction.** The spatial arm must match Phase 5m on its first 100 episodes.

    python scripts/spacetime_experiment.py --episodes 200
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
from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig  # noqa: E402

SPATIAL = PursuitConfig(replan_on_block=True, predict_horizon=2.0)

#: Recorded before the run, with the null BOUNDED. Phase 5o registered its null
#: as "the interval includes zero"; intervals near +/-0.4 satisfied it whatever
#: was true, and the conclusion had to rest on a test added afterwards. Here
#: "inert" means the interval is *inside* +/-0.03, which a design too weak to
#: see an effect cannot satisfy.
#:
#: The magnitude comes from a 20-episode smoke test (+0.05 on dense). That is a
#: measurement in the right regime, but of 20 episodes -- which Phase 5n showed
#: cannot pin a rate finely -- so the band is wide.
PREDICTION = (
    "timing matters: full minus swept on dynamic_dense success is +0.03 to +0.08, "
    "McNemar p < 0.05, carried by fewer collisions. Frozen cells identical. "
    "Decision rule -- MATTERS: gain >= +0.03 and p < 0.05. INERT: paired bootstrap "
    "95% CI entirely inside [-0.03, +0.03]. Otherwise inconclusive."
)
BOUND = 0.03


def succ(rows):
    return np.array([r["success"] for r in rows], dtype=bool)


def paired_ci(a_rows, b_rows, rng, n_boot):
    """95% CI of mean(b) - mean(a), resampling episodes jointly."""
    a, b = succ(a_rows), succ(b_rows)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(b[i].mean() - a[i].mean())
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/spacetime_experiment.json")
    args = p.parse_args(argv)

    factories = {
        "spatial": None,
        "swept": lambda robot: SpaceTimeAgent(SpaceTimeConfig(time_varying_movers=False),
                                              robot=robot),
        "full": lambda robot: SpaceTimeAgent(SpaceTimeConfig(time_varying_movers=True),
                                             robot=robot),
    }
    runs: dict[str, dict[str, list[dict]]] = {}
    for arm, factory in factories.items():
        runs[arm] = {}
        print(f"\n=== {arm} ===", flush=True)
        for label, cond in CELLS:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=factory)
            runs[arm][cond] = rows
            waits = (f"  waits {np.mean([r['planned_waits'] for r in rows]):4.1f}"
                     if "planned_waits" in rows[0] else "")
            print(f"  {label:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  "
                  f"timeout {rate(rows, 'timeout'):.3f}  "
                  f"steps {np.mean([r['steps'] for r in rows]):5.1f}{waits}", flush=True)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "bound": BOUND,
              "checks": {}, "cells": {}, "contrasts": {}}

    prior = Path("results/velocity_experiment.json")
    if prior.exists():
        vm = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        ok = all(
            [e["success"] for e in vm[c]["2.0"]["per_episode"]]
            == [e["success"] for e in runs["spatial"][c][:len(vm[c]["2.0"]["per_episode"])]]
            for _l, c in CELLS
        )
        report["checks"]["spatial_reproduces_phase_5m"] = ok
        print("\nspatial arm reproduces Phase 5m on its first 100:", ok)

    identity = all(runs["full"][c] == runs["swept"][c]
                   for c in ("dynamic_frozen", "dynamic_dense_frozen"))
    report["checks"]["frozen_identity_full_vs_swept"] = identity
    print("frozen identity, full vs swept:", identity)

    for _l, c in CELLS:
        report["cells"][c] = {
            arm: {**{k: rate(runs[arm][c], k) for k in ("success", "collision", "timeout")},
                  "mean_steps": float(np.mean([r["steps"] for r in runs[arm][c]])),
                  "success_per_episode": [int(r["success"]) for r in runs[arm][c]],
                  **({"mean_planned_waits": float(np.mean([r["planned_waits"]
                                                           for r in runs[arm][c]]))}
                     if "planned_waits" in runs[arm][c][0] else {})}
            for arm in runs
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts on moving worlds ===")
    for treated, ref in (("full", "swept"), ("full", "spatial"), ("swept", "spatial")):
        for moving, frozen in (("dynamic", "dynamic_frozen"),
                               ("dynamic_dense", "dynamic_dense_frozen")):
            a, b = runs[ref][moving], runs[treated][moving]
            gain = rate(b, "success") - rate(a, "success")
            p_val, lost, won = mcnemar_p(succ(a), succ(b))
            ci = paired_ci(a, b, rng, args.bootstrap)
            cost_t = rate(runs[treated][frozen], "success") - rate(b, "success")
            cost_r = rate(runs[ref][frozen], "success") - rate(a, "success")
            if gain >= BOUND and p_val < 0.05:
                verdict = "MATTERS"
            elif ci[0] > -BOUND and ci[1] < BOUND:
                verdict = "INERT (bounded)"
            else:
                verdict = "inconclusive"
            key = f"{treated}_vs_{ref}"
            report["contrasts"].setdefault(key, {})[moving] = {
                "success_gain": gain, "p": p_val, "episodes_won": won,
                "episodes_lost": lost, "ci95": ci,
                "collision_delta": rate(b, "collision") - rate(a, "collision"),
                "timeout_delta": rate(b, "timeout") - rate(a, "timeout"),
                "motion_cost_treated": cost_t, "motion_cost_reference": cost_r,
                "verdict": verdict,
            }
            print(f"  {key:18s} {moving:14s} {gain:+.3f}  p {p_val:.4f} (+{won}/-{lost})  "
                  f"CI [{ci[0]:+.3f},{ci[1]:+.3f}]  cost {cost_r:+.3f} -> {cost_t:+.3f}  "
                  f"{verdict}", flush=True)

    print("\npre-registered: " + PREDICTION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
