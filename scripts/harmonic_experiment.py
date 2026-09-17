"""What is a better motion model worth, when the model is right?

Phases 5r to 5w changed how the planner *uses* a constant-velocity estimate --
a wider temporal margin, withholding zero-margin plans, replanning on every
contradiction, capping the extrapolation -- and none recovered its 0.065
dense-clutter cost against the oracle. None of them changed the line itself.

This changes the model: each mover's own oscillation, fitted to the track the
agent has watched it follow, carried forward in closed form. It is matched to
how these movers move and reads noise-free observations, so it is the most
favourable case a better model can be given, and the result is an upper bound
on what better prediction is worth to this stack -- not a claim about a real
sensor, which is the step after.

Arms, space-time agent, two-step temporal margin, 200 episodes each:

  cv_m2    -- Phase 5r's straight-line agent; must reproduce it episode for episode
  harm_m2  -- the same, with the fitted oscillation

The oracle's episodes come from Phase 5r's result file. A frozen mover has no
oscillation to fit, so the fit declines and the agent falls back to the line,
which on a frozen world is exact: harm_m2 must be bit-identical to cv_m2 on
every frozen episode.

    python scripts/harmonic_experiment.py --episodes 200
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

#: Recorded before the run and committed to the repository before any result
#: exists -- but derived from a measurement taken first, and said so in the
#: commit: how far each estimator is from the truth on these worlds, with no
#: planner involved (``estimator_error`` below re-runs it). Over 30 worlds per
#: condition and 3 s of observation, the straight line is a median 0.021 m out
#: at 1 s and 1.15 m out at 7 s; the fitted oscillation is 0.0000 m at the
#: median and 0.0001 m at the 95th percentile, at every horizon. It is, for
#: practical purposes, the oracle with a three-second delay at the start of an
#: episode, which is the basis for the prediction below.
#:
#: This is the project's one category of forecast that has held: derived from a
#: measurement, in the regime it was measured in. What it cannot say is anything
#: about a noisy sensor.
PREDICTION = (
    "the model was the problem: harm_m2 minus cv_m2 on dynamic_dense is MATTERS "
    "(gain >= +0.03, p < 0.05), and harm_m2 minus oracle_m2 is bounded-INERT "
    "(95% CI inside +/-0.03) on both conditions -- the fitted estimate should "
    "plan like the oracle. Frozen identity holds. "
    "Decision -- MODEL: MATTERS on dense. NOT MODEL: INERT (bounded) or HARMS on "
    "dense. Otherwise UNRESOLVED."
)


def factory(predictor):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor=predictor, temporal_margin_steps=2),
                              robot=robot)
    return make


def estimator_error(episodes: int = 30) -> dict:
    """Each estimator's distance from the truth, with no planner in the loop."""
    out: dict[str, dict] = {}
    for cond in MOVING:
        split, shift, _ = DYNAMIC_CONDITIONS[cond]
        cfg = build_env_config({}, split=split, shift=shift, n_worlds=episodes)
        env = ProceduralNavEnv(cfg)
        errs = {kind: {h: [] for h in HORIZONS} for kind in ("constant_velocity", "harmonic")}
        for seed in cfg.world_seeds:
            env.reset(options={"world_seed": int(seed)})
            agents = {k: SpaceTimeAgent(SpaceTimeConfig(predictor=k), robot=cfg.robot)
                      for k in errs}
            for a in agents.values():
                a._world, a._observations = env.world, []
            for _ in range(31):
                for a in agents.values():
                    a._observe()
                env.world.set_time(env.world._t + cfg.robot.dt)
            now = env.world._t
            for h in HORIZONS:
                truth = env.world.dynamic_at(now + h)[:, :2]
                for kind, a in agents.items():
                    got = a._predicted_discs(now + h)[:, :2]
                    errs[kind][h].extend(np.linalg.norm(got - truth, axis=1).tolist())
        out[cond] = {kind: {f"{h}s": {"median": float(np.median(v[h])),
                                      "p95": float(np.percentile(v[h], 95))}
                            for h in HORIZONS}
                     for kind, v in errs.items()}
        for kind in errs:
            line = "  ".join(f"{h}s med {out[cond][kind][f'{h}s']['median']:.4f} "
                             f"p95 {out[cond][kind][f'{h}s']['p95']:.4f}" for h in HORIZONS)
            print(f"  {cond:14s} {kind:18s} {line}", flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--error-worlds", type=int, default=30)
    p.add_argument("--out", default="results/harmonic_experiment.json")
    args = p.parse_args(argv)

    print("estimator error, no planner:")
    report = {"episodes": args.episodes, "prediction": PREDICTION,
              "estimator_error": estimator_error(args.error_worlds),
              "checks": {}, "cells": {}, "contrasts": {}}

    arms = {"cv_m2": factory("constant_velocity"), "harm_m2": factory("harmonic")}
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    plan = [(a, c) for a in arms for c in MOVING] + [(a, FROZEN[c]) for a in arms for c in MOVING]
    for arm, cond in plan:
        rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
        runs[arm][cond] = rows
        print(f"  {arm:8s} {cond:22s} success {rate(rows, 'success'):.3f}  "
              f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}",
              flush=True)

    identity = all(runs["cv_m2"][c] == runs["harm_m2"][c] for c in FROZEN.values())
    report["checks"]["frozen_identity_harmonic_vs_line"] = identity
    print("\nfrozen identity, fitted vs line:", identity)

    est = json.loads(Path("results/estimate_experiment.json").read_text(encoding="utf-8"))["cells"]
    repro = all([int(r["success"]) for r in runs["cv_m2"][c]] == est["cv_m2"][c]["success_per_episode"]
                for c in MOVING)
    report["checks"]["reproduces_phase_5r"] = repro
    print("cv_m2 reproduces Phase 5r on every episode:", repro)
    oracle = {c: [{"success": bool(s)}
                  for s in est["oracle_m2"][c]["success_per_episode"][:args.episodes]]
              for c in MOVING}

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    for treated, ref in (("harm_m2", "cv_m2"), ("harm_m2", "oracle_m2"), ("cv_m2", "oracle_m2")):
        for cond in MOVING:
            a = oracle[cond] if ref == "oracle_m2" else runs[ref][cond]
            b = runs[treated][cond]
            gain = rate(b, "success") - rate(a, "success")
            p_val, lost, won = mcnemar_p(succ(a), succ(b))
            ci = paired_ci(a, b, rng, args.bootstrap)
            entry = {"success_gain": gain, "p": p_val, "episodes_won": won,
                     "episodes_lost": lost, "ci95": ci, "verdict": verdict(gain, p_val, ci)}
            if ref != "oracle_m2":
                entry["collision_delta"] = rate(b, "collision") - rate(a, "collision")
                entry["timeout_delta"] = rate(b, "timeout") - rate(a, "timeout")
            report["contrasts"].setdefault(f"{treated}_vs_{ref}", {})[cond] = entry
            print(f"  {treated:7s} vs {ref:9s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]  {entry['verdict']}",
                  flush=True)

    dense = report["contrasts"]["harm_m2_vs_cv_m2"]["dynamic_dense"]["verdict"]
    report["decision"] = ("MODEL" if dense == "MATTERS"
                          else "NOT MODEL" if dense in ("INERT (bounded)", "HARMS")
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
