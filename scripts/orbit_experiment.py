"""Does fitting without differencing survive the noise that killed the fit?

Phase 5y: fitting each mover's oscillation matched the oracle -- from exact
observations. Phase 5z: a centimetre of observation error cost the fitted
planner 0.160 of dense-clutter success, more than the model had ever bought,
and left the two estimators indistinguishable. The diagnosis was in how the fit
reads curvature: a second difference divides the error by dt squared.

This fits the same model without differencing anything. ``p(t) = c + u sin(w t)
+ v cos(w t)`` is linear in ``c``, ``u`` and ``v`` once ``w`` is fixed, so the
estimator searches ``w`` and solves the rest by least squares over the whole
observed track, giving every observation an equal vote.

Arms, space-time agent, two-step temporal margin, 200 episodes each, all at
Phase 5z's centimetre of noise except where stated:

  orbit@0.01       the fit above on 3 s of track (primary)
  orbit6@0.01      the same on 6 s
  orbit@0.0        noise-free, as a sanity check against Phase 5y and the oracle

The straight line and the differencing fit at the same noise are Phase 5z's
arms, and the oracle is Phase 5r's; their per-episode outcomes are the
references, not re-run. Frozen worlds hold no oscillation to find, so the fit
declines and falls back to the line, which is exact when nothing moves: the
frozen cells must be bit-identical to the oracle's.

    python scripts/orbit_experiment.py --episodes 200
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
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import DYNAMIC_CONDITIONS  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

MOVING = ("dynamic", "dynamic_dense")
FROZEN = {"dynamic": "dynamic_frozen", "dynamic_dense": "dynamic_dense_frozen"}
HORIZONS = (0.5, 1.0, 2.0, 7.0)
SIGMA = 0.01

#: Recorded before the run and committed before any result exists -- and, as in
#: Phase 5y, derived from a measurement taken first and disclosed in the commit.
#: At a centimetre of noise on `dynamic_dense`, with no planner, the median
#: error two seconds out is 0.295 m for the straight line, 0.438 m for the
#: differencing fit, and 0.057 m for this one on 3 s of track (0.039 m on 6 s).
#: At seven seconds: 1.281, 1.341 and 0.521. So at a centimetre of noise this
#: estimator is better than the *noise-free* straight line was at every horizon,
#: and Phase 5r measured what that line was worth to the planner: 0.910 on dense
#: clutter against the oracle's 0.975.
#:
#: That is the basis for the prediction. What it cannot pin down is how much of
#: the remaining gap to the oracle closes, since 0.057 m of error is not zero
#: and Phase 5r's chain showed small estimate errors mattering in ways the error
#: alone did not predict.
PREDICTION = (
    "filtering is the repair: orbit@0.01 minus constant_velocity@0.01 on "
    "dynamic_dense is MATTERS (gain >= +0.03, p < 0.05) and at least +0.05, and "
    "orbit@0.01 beats the differencing fit at the same noise by at least as much. "
    "Secondary: orbit@0.01 is still significantly below the oracle (its estimate "
    "is good, not exact); the 6 s history is no worse than the 3 s one on dense "
    "clutter; orbit@0.0 is bounded-INERT against the oracle. Frozen cells "
    "bit-identical to the oracle's. "
    "Decision -- FILTER: MATTERS on dense. NO FILTER: INERT (bounded) or HARMS. "
    "Otherwise UNRESOLVED."
)


def factory(sigma, history=3.0):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor="orbit", temporal_margin_steps=2,
                                              observation_noise_m=sigma, history_s=history),
                              robot=robot)
    return make


def estimator_error(worlds: int) -> dict:
    """This estimator's own error at the experiment's noise, no planner."""
    out: dict[str, dict] = {}
    for cond in MOVING:
        split, shift, _ = DYNAMIC_CONDITIONS[cond]
        cfg = build_env_config({}, split=split, shift=shift, n_worlds=worlds)
        env = ProceduralNavEnv(cfg)
        arms = {"orbit3": 3.0, "orbit6": 6.0}
        errs = {k: {h: [] for h in HORIZONS} for k in arms}
        for seed in cfg.world_seeds:
            env.reset(options={"world_seed": int(seed)})
            agents = {k: SpaceTimeAgent(
                SpaceTimeConfig(predictor="orbit", history_s=h, observation_noise_m=SIGMA),
                robot=cfg.robot) for k, h in arms.items()}
            for a in agents.values():
                a._world, a._observations = env.world, []
            for _ in range(61):
                for a in agents.values():
                    a._observe()
                env.world.set_time(env.world._t + cfg.robot.dt)
            now = env.world._t
            for h in HORIZONS:
                truth = env.world.dynamic_at(now + h)[:, :2]
                for k, a in agents.items():
                    got = a._predicted_discs(now + h)[:, :2]
                    errs[k][h].extend(np.linalg.norm(got - truth, axis=1).tolist())
        out[cond] = {k: {f"{h}s": {"median": float(np.median(v[h])),
                                   "p95": float(np.percentile(v[h], 95))} for h in HORIZONS}
                     for k, v in errs.items()}
        for k in arms:
            line = "  ".join(f"{h}s {out[cond][k][f'{h}s']['median']:.4f}" for h in HORIZONS)
            print(f"  {cond:14s} {k:8s} {line}", flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--error-worlds", type=int, default=30)
    p.add_argument("--out", default="results/orbit_experiment.json")
    args = p.parse_args(argv)

    print("estimator error at sigma = 0.01, no planner:")
    report = {"episodes": args.episodes, "sigma": SIGMA, "prediction": PREDICTION,
              "estimator_error": estimator_error(args.error_worlds),
              "checks": {}, "cells": {}, "contrasts": {}}

    arms = {"orbit@0.01": factory(SIGMA), "orbit6@0.01": factory(SIGMA, 6.0),
            "orbit@0.0": factory(0.0)}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    plan = [(a, c) for a in arms for c in MOVING] + [("orbit@0.0", FROZEN[c]) for c in MOVING]
    for arm, cond in plan:
        rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
        runs[arm][cond] = rows
        print(f"  {arm:12s} {cond:22s} success {rate(rows, 'success'):.3f}  "
              f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}",
              flush=True)

    def stored(path, arm, key="success_per_episode"):
        cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
        return {c: [{"success": bool(s)} for s in cells[arm][c][key][:args.episodes]]
                for c in MOVING}

    line = stored("results/noise_experiment.json", "constant_velocity@0.01")
    differenced = stored("results/noise_experiment.json", "harmonic@0.01")
    oracle = stored("results/estimate_experiment.json", "oracle_m2")
    frozen_oracle = json.loads(
        Path("results/estimate_experiment.json").read_text(encoding="utf-8"))["cells"]["oracle_m2"]

    identity = all([int(r["success"]) for r in runs["orbit@0.0"][FROZEN[c]]]
                   == frozen_oracle[FROZEN[c]]["success_per_episode"][:args.episodes]
                   for c in MOVING)
    report["checks"]["frozen_matches_oracle"] = identity
    print("\nfrozen cells match the oracle's episode for episode:", identity)

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    refs = {"line@0.01": line, "differenced@0.01": differenced, "oracle": oracle}
    print("\n=== contrasts ===")
    for treated in ("orbit@0.01", "orbit6@0.01", "orbit@0.0"):
        for ref_name, ref in refs.items():
            for cond in MOVING:
                a, b = ref[cond], runs[treated][cond]
                gain = rate(b, "success") - rate(a, "success")
                p_val, lost, won = mcnemar_p(succ(a), succ(b))
                ci = paired_ci(a, b, rng, args.bootstrap)
                report["contrasts"].setdefault(f"{treated}_vs_{ref_name}", {})[cond] = {
                    "success_gain": gain, "p": p_val, "episodes_won": won,
                    "episodes_lost": lost, "ci95": ci, "verdict": verdict(gain, p_val, ci)}
                print(f"  {treated:12s} vs {ref_name:16s} {cond:14s} {gain:+.3f}  "
                      f"p {p_val:.4f} (+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]",
                      flush=True)
    # And the two histories against each other.
    for cond in MOVING:
        a, b = runs["orbit@0.01"][cond], runs["orbit6@0.01"][cond]
        gain = rate(b, "success") - rate(a, "success")
        p_val, lost, won = mcnemar_p(succ(a), succ(b))
        ci = paired_ci(a, b, rng, args.bootstrap)
        report["contrasts"].setdefault("orbit6@0.01_vs_orbit@0.01", {})[cond] = {
            "success_gain": gain, "p": p_val, "episodes_won": won, "episodes_lost": lost,
            "ci95": ci, "verdict": verdict(gain, p_val, ci)}
        print(f"  orbit6@0.01  vs orbit@0.01       {cond:14s} {gain:+.3f}  p {p_val:.4f} "
              f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)

    label = report["contrasts"]["orbit@0.01_vs_line@0.01"]["dynamic_dense"]["verdict"]
    report["decision"] = ("FILTER" if label == "MATTERS"
                          else "NO FILTER" if label in ("INERT (bounded)", "HARMS")
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
