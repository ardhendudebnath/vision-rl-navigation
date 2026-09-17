"""Would knowing where the movers are going remove the motion cost?

Report Section 12 ranked this first among what remains. Four mechanisms for the
classical stack's motion cost are eliminated -- sensing, planning failure,
replanning churn, commitment length -- and every actor in the project still
treats each mover as a static snapshot at its current position. A planner that
cannot tell a mover approaching its path from one leaving it replans too late
for the first and needlessly for the second.

This gives the adopted baseline (block-triggered replanning) **oracle**
knowledge of motion: the trajectories are analytic, so it plans around, and
triggers replans on, each mover's exact swept region over the next H seconds.
That is an upper bound on what any real velocity layer could supply, which is
what makes the result decisive in either direction:

- if oracle prediction recovers the motion cost, velocity-blindness explains it
  and a velocity costmap layer is worth building;
- if it does not, no velocity estimate can, and a fifth mechanism is gone.

Scope, added after Phase 5n and left outside the pre-registration below, which
is unchanged: "no velocity estimate can" holds for *this* planner, which
consumes the oracle as swept regions and plans in space. It does not bound a
planner that reasons in space-time with the same information. Reading it as a
bound on velocity in general was an overclaim in the first writeup.

The frozen cells are the control, and here the control is an **identity**
rather than a tolerance. A frozen mover has zero amplitude, so its swept region
at any horizon is its current disc; every frozen episode must be bit-identical
across horizons, and ``test_frozen_movers_make_any_horizon_identical_to_none``
enforces it. Any frozen difference voids the run.

    python scripts/velocity_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from vision_nav.agents.classical import AStarPursuitAgent, PursuitConfig
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS
from vision_nav.training.env_factory import build_env_config

CELLS = [
    ("sparse moving", "dynamic"),
    ("sparse frozen", "dynamic_frozen"),
    ("dense moving", "dynamic_dense"),
    ("dense frozen", "dynamic_dense_frozen"),
]

#: Recorded before any horizon above zero was run.
#:
#: Horizons are a sweep, and Phase 5e read a tidy optimum across a sweep as
#: confirmation of a mechanism its control cell then refuted. Two guards follow
#: from that. The significance threshold is Bonferroni-corrected for the three
#: non-zero horizons, since reporting the best of three inflates the chance of
#: a spurious gain. And the claim is conditional on the frozen identity holding;
#: without it no horizon's number means anything.
PREDICTION = (
    "best horizon recovers dense moving by +0.03 to +0.08 success; long horizons "
    "hurt (over-conservative swept regions block corridors); frozen cells "
    "bit-identical at every horizon. >= +0.06 at alpha 0.05/3 = velocity-blindness "
    "explains a substantial part of the motion cost; < +0.03 at every horizon = "
    "fifth mechanism eliminated, decisively, since the prediction is an oracle"
)
ALPHA = 0.05 / 3


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar on paired binary outcomes.

    The actor is deterministic and every horizon runs on identical worlds, so
    the unit is the episode and only discordant pairs carry information: an
    episode both arms solve, or both fail, says nothing about the difference.
    """
    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), a_only, b_only


def run_cell(condition: str, episodes: int, config: PursuitConfig,
             env_overrides: dict | None = None,
             agent_factory=None) -> list[dict]:
    """One condition, per-episode outcomes.

    ``env_overrides`` is merged into the env config -- Phase 5o uses it to
    change the robot's kinematic limits. ``agent_factory(robot)`` substitutes a
    different agent -- Phase 5p's space-time planner -- through the same loop.
    ``None`` for both reproduces every earlier run.
    """
    split, shift, _ = DYNAMIC_CONDITIONS[condition]
    overrides = {"freeze_dynamic": True} if condition in FROZEN_CONDITIONS else {}
    if env_overrides:
        overrides.update(env_overrides)
    env_config = build_env_config(dict(overrides), split=split, shift=shift,
                                  n_worlds=episodes)
    env = ProceduralNavEnv(env_config)
    agent = (agent_factory(env_config.robot) if agent_factory is not None
             else AStarPursuitAgent(config, robot=env_config.robot))

    rows = []
    for seed in env_config.world_seeds:
        env.reset(options={"world_seed": int(seed)})
        agent.robot = env.config.robot
        if not agent.start_episode(env.world, env.robot.pose):
            rows.append({"seed": int(seed), "success": False, "collision": False,
                         "timeout": True, "replans": 0, "steps": 0,
                         "planning_failure": True})
            continue
        done, info, steps = False, {}, 0
        while not done:
            _, _, term, trunc, info = env.step(agent.act(env.robot.pose))
            done = term or trunc
            steps += 1
        succ, coll = bool(info.get("is_success")), bool(info.get("collision"))
        row = {"seed": int(seed), "success": succ, "collision": coll,
               "timeout": not succ and not coll, "replans": agent.replans,
               "steps": steps, "planning_failure": False}
        # Only agents that plan waits report them, so every earlier arm's rows
        # -- compared elsewhere by dict equality -- stay exactly as they were.
        if hasattr(agent, "planned_waits"):
            row["planned_waits"] = agent.planned_waits
            row["bare_radius_attempts"] = agent.bare_radius_attempts
        rows.append(row)
    return rows


def rate(rows, key):
    return float(np.mean([r[key] for r in rows]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--horizons", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0])
    p.add_argument("--out", default="results/velocity_experiment.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.horizons[0] != 0.0:
        raise ValueError("the first horizon must be 0.0: it is the baseline")

    arms: dict[float, dict[str, list[dict]]] = {}
    for h in args.horizons:
        print(f"\n=== horizon {h:.1f} s ===")
        cfg = PursuitConfig(replan_on_block=True, predict_horizon=h)
        arms[h] = {}
        for label, cond in CELLS:
            rows = run_cell(cond, args.episodes, cfg)
            arms[h][cond] = rows
            print(f"  {label:14s} success {rate(rows, 'success'):.3f}  "
                  f"collision {rate(rows, 'collision'):.3f}  "
                  f"timeout {rate(rows, 'timeout'):.3f}  "
                  f"replans {np.mean([r['replans'] for r in rows]):5.1f}")

    base = arms[0.0]

    # --- the control, checked before anything is concluded ---------------
    print("\n=== frozen identity (the control) ===")
    identity_ok = True
    for h in args.horizons[1:]:
        for cond in ("dynamic_frozen", "dynamic_dense_frozen"):
            same = [a == b for a, b in zip(base[cond], arms[h][cond], strict=True)]
            if not all(same):
                identity_ok = False
                print(f"  VIOLATED at horizon {h}: {cond}, "
                      f"{same.count(False)} episode(s) differ")
    print("  holds at every horizon" if identity_ok else
          "  CONTROL FAILED -- implementation has a side effect; results void")

    # --- the motion cost, and how much of it each horizon recovers --------
    report = {"episodes": args.episodes, "prediction": PREDICTION,
              "alpha_corrected": ALPHA, "frozen_identity_holds": identity_ok,
              "cells": {}, "effects": {}}
    for _label, cond in CELLS:
        report["cells"][cond] = {
            str(h): {
                **{k: rate(arms[h][cond], k)
                   for k in ("success", "collision", "timeout")},
                # Stored because the writeup makes a claim about it: prediction
                # replans far more often, which bears on Phase 5d's churn result.
                "replans_mean": float(np.mean([r["replans"] for r in arms[h][cond]])),
                "planning_failures": int(sum(r["planning_failure"]
                                             for r in arms[h][cond])),
                "per_episode": arms[h][cond],
            }
            for h in args.horizons
        }

    print("\n=== effect of prediction on moving worlds (paired vs horizon 0) ===")
    for moving, frozen in (("dynamic", "dynamic_frozen"),
                           ("dynamic_dense", "dynamic_dense_frozen")):
        cost = rate(base[frozen], "success") - rate(base[moving], "success")
        print(f"  {moving}: motion cost at horizon 0 = {cost:+.3f}")
        report["effects"][moving] = {"motion_cost": cost, "by_horizon": {}}
        a = np.array([r["success"] for r in base[moving]], dtype=bool)
        for h in args.horizons[1:]:
            b = np.array([r["success"] for r in arms[h][moving]], dtype=bool)
            gain = float(b.mean() - a.mean())
            p, lost, won = mcnemar_p(a, b)
            recovered = gain / cost if cost > 0 else float("nan")
            d_coll = rate(arms[h][moving], "collision") - rate(base[moving], "collision")
            d_tmo = rate(arms[h][moving], "timeout") - rate(base[moving], "timeout")
            sig = "  <-- significant at corrected alpha" if p < ALPHA else ""
            print(f"    h={h:.1f}s  success {gain:+.3f} (p {p:.4f}, +{won}/-{lost})  "
                  f"recovers {recovered:+.0%} of cost  "
                  f"collision {d_coll:+.3f}  timeout {d_tmo:+.3f}{sig}")
            report["effects"][moving]["by_horizon"][str(h)] = {
                "success_gain": gain, "p": p, "episodes_won": won,
                "episodes_lost": lost, "fraction_of_cost_recovered": recovered,
                "collision_delta": d_coll, "timeout_delta": d_tmo,
            }

    print("\npre-registered: " + PREDICTION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
