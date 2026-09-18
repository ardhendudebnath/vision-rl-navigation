"""How much sensing accuracy does a fitted motion model need?

Phase 5y's fitted oscillation matched the oracle and reduced the motion cost to
0.005 -- reading observations that were exact. This takes that gift back and
sweeps the error on each observed mover position.

Two stages, the second's arms chosen by a rule fixed here before either ran.

**Stage 1, no planner.** Each estimator's distance from the truth, over a grid
of noise levels, at horizons the planner actually uses.

**Stage 2, the planner**, 200 episodes an arm. Both estimators at
``PLANNER_SIGMAS``: 0.01 m, a centimetre, about what a good scan-based tracker
manages on a disc this size, and the largest grid level at which the fit is
still the better estimator by stage 1 -- its 2 s median error below the straight
line's at the same noise. If there is no such level, the smallest grid level
above zero is used instead, and the run says so.

Control: the oracle reads no observations, so an oracle arm must be
bit-identical with noise and without. The zero-noise arms are not re-run --
Phase 5y's are exactly this configuration, and its per-episode outcomes are the
reference each noisy arm is paired against.

    python scripts/noise_experiment.py --episodes 200
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
SIGMAS = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)
HORIZONS = (0.5, 1.0, 2.0, 7.0)
ESTIMATORS = ("constant_velocity", "harmonic")
#: The level the second stage always runs, whatever stage 1 says.
FIXED_SIGMA = 0.01

#: Recorded before the run and committed to the repository before any result
#: exists.
#:
#: Confidence MODERATE on the shape, LOW on where it breaks. The fit reads
#: curvature through a second difference, which divides the observation error by
#: dt squared: a centimetre of noise becomes 2.4 m/s^2 of apparent acceleration
#: where the real thing is at most 0.2. Least squares over three seconds averages
#: some of that away, but errors-in-variables biases the frequency low, so the
#: fit should degrade fast and, unlike the straight line, degrade at every
#: horizon rather than only the far ones. The straight line's own error is
#: dominated by curvature it ignores, so at short horizons noise should hurt it
#: too, and at 7 s barely matter.
#:
#: What this cannot settle is whether a *filtered* fit survives. Nothing here
#: smooths the observations, and the obvious repair -- a filter that tracks
#: position, velocity and frequency together -- is the next phase, not this one.
PREDICTION = (
    "the fit is fragile: by stage 1 the fitted estimator's 2 s median error "
    "exceeds the straight line's at every noise level of 0.005 m and above, and "
    "at 0.01 m exceeds it by more than tenfold. In stage 2 at 0.01 m the fitted "
    "arm is no better than the line on dynamic_dense -- not MATTERS -- and both "
    "are worse than Phase 5y's noise-free fit. Controls: the oracle arm "
    "bit-identical with noise and without; each noisy arm paired against Phase "
    "5y's noise-free arms. "
    "Decision on the stage-2 primary (fitted minus line at 0.01 m, dynamic_dense) "
    "-- ROBUST: MATTERS. FRAGILE: INERT (bounded) or HARMS. Otherwise UNRESOLVED."
)


def factory(predictor, sigma):
    def make(robot):
        return SpaceTimeAgent(SpaceTimeConfig(predictor=predictor, temporal_margin_steps=2,
                                              observation_noise_m=sigma), robot=robot)
    return make


def stage_one(worlds: int) -> dict:
    """Estimator error against the truth, over the noise grid. No planner."""
    out: dict[str, dict] = {}
    for cond in MOVING:
        split, shift, _ = DYNAMIC_CONDITIONS[cond]
        cfg = build_env_config({}, split=split, shift=shift, n_worlds=worlds)
        env = ProceduralNavEnv(cfg)
        errs = {(k, s): {h: [] for h in HORIZONS} for k in ESTIMATORS for s in SIGMAS}
        for seed in cfg.world_seeds:
            env.reset(options={"world_seed": int(seed)})
            agents = {key: SpaceTimeAgent(
                SpaceTimeConfig(predictor=key[0], observation_noise_m=key[1]), robot=cfg.robot)
                for key in errs}
            for a in agents.values():
                a._world, a._observations = env.world, []
            for _ in range(31):
                for a in agents.values():
                    a._observe()
                env.world.set_time(env.world._t + cfg.robot.dt)
            now = env.world._t
            for h in HORIZONS:
                truth = env.world.dynamic_at(now + h)[:, :2]
                for key, a in agents.items():
                    got = a._predicted_discs(now + h)[:, :2]
                    errs[key][h].extend(np.linalg.norm(got - truth, axis=1).tolist())
        out[cond] = {f"{k}@{s}": {f"{h}s": {"median": float(np.median(v[h])),
                                            "p95": float(np.percentile(v[h], 95))}
                                  for h in HORIZONS}
                     for (k, s), v in errs.items()}
        for key in errs:
            row = out[cond][f"{key[0]}@{key[1]}"]
            line = "  ".join(f"{h}s {row[f'{h}s']['median']:.4f}" for h in HORIZONS)
            print(f"  {cond:14s} {key[0]:18s} sigma {key[1]:<6} {line}", flush=True)
    return out


def chosen_sigmas(stage1: dict) -> tuple[list[float], float | None]:
    """The registered rule: 0.01 m, plus the largest level where the fit still
    beats the line at 2 s on dense clutter."""
    dense = stage1["dynamic_dense"]
    better = [s for s in SIGMAS if s > 0.0
              and dense[f"harmonic@{s}"]["2.0s"]["median"]
              < dense[f"constant_velocity@{s}"]["2.0s"]["median"]]
    crossing = max(better) if better else None
    second = crossing if crossing is not None else min(s for s in SIGMAS if s > 0.0)
    return sorted({FIXED_SIGMA, second}), crossing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--error-worlds", type=int, default=30)
    p.add_argument("--out", default="results/noise_experiment.json")
    args = p.parse_args(argv)

    print("stage 1: estimator error against noise, no planner")
    stage1 = stage_one(args.error_worlds)
    sigmas, crossing = chosen_sigmas(stage1)
    print(f"\nstage 2 runs sigma {sigmas} (last level where the fit still wins at 2 s: "
          f"{crossing})", flush=True)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "sigmas": SIGMAS,
              "stage1": stage1, "stage2_sigmas": sigmas, "fit_still_wins_up_to": crossing,
              "checks": {}, "cells": {}, "contrasts": {}}

    arms = {f"{k}@{s}": factory(k, s) for s in sigmas for k in ESTIMATORS}
    arms["oracle@0.0"] = factory("oracle", 0.0)
    arms[f"oracle@{max(sigmas)}"] = factory("oracle", max(sigmas))
    runs: dict[str, dict[str, list[dict]]] = {a: {} for a in arms}
    for arm in arms:
        for cond in MOVING:
            rows = run_cell(cond, args.episodes, SPATIAL, agent_factory=arms[arm])
            runs[arm][cond] = rows
            print(f"  {arm:24s} {cond:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  timeout {rate(rows, 'timeout'):.3f}",
                  flush=True)

    deaf = all(runs["oracle@0.0"][c] == runs[f"oracle@{max(sigmas)}"][c] for c in MOVING)
    report["checks"]["oracle_is_deaf_to_noise"] = deaf
    print("\noracle bit-identical with noise and without:", deaf)

    cells5y = json.loads(
        Path("results/harmonic_experiment.json").read_text(encoding="utf-8"))["cells"]
    oracle5r = json.loads(
        Path("results/estimate_experiment.json").read_text(encoding="utf-8"))["cells"]
    report["checks"]["oracle_reproduces_phase_5r"] = all(
        [int(r["success"]) for r in runs["oracle@0.0"][c]]
        == oracle5r["oracle_m2"][c]["success_per_episode"] for c in MOVING)
    print("oracle reproduces Phase 5r:", report["checks"]["oracle_reproduces_phase_5r"])
    # Phase 5y's noise-free arms, as rows, to pair the noisy ones against.
    quiet = {f"{kind}@0.0(5y)": {c: [{"success": bool(s)} for s in
                                     cells5y[arm][c]["success_per_episode"][:args.episodes]]
                                 for c in MOVING}
             for kind, arm in (("harmonic", "harm_m2"), ("constant_velocity", "cv_m2"))}

    for arm in runs:
        report["cells"][arm] = {
            c: {**{k: rate(rows, k) for k in ("success", "collision", "timeout")},
                "success_per_episode": [int(r["success"]) for r in rows]}
            for c, rows in runs[arm].items()
        }

    rng = np.random.default_rng(0)
    print("\n=== contrasts ===")
    pairs = [(f"harmonic@{s}", f"constant_velocity@{s}") for s in sigmas]
    pairs += [(f"harmonic@{s}", "oracle@0.0") for s in sigmas]
    pairs += [(f"{kind}@{s}", f"{kind}@0.0(5y)") for s in sigmas for kind in ESTIMATORS]
    for treated, ref in pairs:
        for cond in MOVING:
            a = quiet[ref][cond] if ref.endswith("(5y)") else runs[ref][cond]
            b = runs[treated][cond]
            gain = rate(b, "success") - rate(a, "success")
            p_val, lost, won = mcnemar_p(succ(a), succ(b))
            ci = paired_ci(a, b, rng, args.bootstrap)
            entry = {"success_gain": gain, "p": p_val, "episodes_won": won,
                     "episodes_lost": lost, "ci95": ci, "verdict": verdict(gain, p_val, ci)}
            # A stored reference carries success only; a re-run arm carries all three.
            if "collision" in a[0]:
                entry["collision_delta"] = rate(b, "collision") - rate(a, "collision")
                entry["timeout_delta"] = rate(b, "timeout") - rate(a, "timeout")
            report["contrasts"].setdefault(f"{treated}_vs_{ref}", {})[cond] = entry
            print(f"  {treated:22s} vs {ref:22s} {cond:14s} {gain:+.3f}  p {p_val:.4f} "
                  f"(+{won}/-{lost})  CI [{ci[0]:+.3f},{ci[1]:+.3f}]", flush=True)

    primary = report["contrasts"][f"harmonic@{FIXED_SIGMA}_vs_constant_velocity@{FIXED_SIGMA}"]
    label = primary["dynamic_dense"]["verdict"]
    report["decision"] = ("ROBUST" if label == "MATTERS"
                          else "FRAGILE" if label in ("INERT (bounded)", "HARMS")
                          else "UNRESOLVED")
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if deaf else 1


if __name__ == "__main__":
    raise SystemExit(main())
