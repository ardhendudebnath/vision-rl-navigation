"""Is the rest of the motion cost physical -- a robot that cannot get out of the way?

Phases 5m and 5n gave the classical stack oracle knowledge of every mover's
future everywhere it reads movers, and recovered half the dense motion cost and
no more. Two explanations for the remaining half survive: the planner uses a
perfect input crudely (swept regions in a plan that exists only in space), or
the robot physically cannot evade a mover it has correctly anticipated.

This tests the second by changing the robot's agility and nothing else about
the task. Linear and angular velocity *and* acceleration limits scale by the
same factor, so the time to reach top speed is unchanged and only how fast the
robot can act on what it knows differs.

Worth stating before the result, because it shapes the prediction: on
`dynamic_dense` the movers peak at 0.15-0.45 m/s, and only momentarily, since
their motion is sinusoidal. The robot at 1x already does 0.6 m/s. It outruns
every mover. A robot that fast with oracle knowledge of where they are going
should be able to evade, which makes a physical limit the *less* likely of the
two explanations before anything is run.

Three confounds, each handled rather than assumed away:

1. **Exposure.** A faster robot finishes sooner and meets fewer movers, which
   shrinks the motion cost whether or not it evades better. The discriminator
   is the *interaction*: a physical limit means agility helps the arm that
   anticipates movers disproportionately, so the fraction of the cost that
   prediction recovers rises with speed. Exposure alone shrinks both arms
   alike and leaves that fraction flat. Mean episode length is logged too.
2. **Controller tuning.** Lookahead, slow-down and caution distances are in
   metres and tuned at 0.6 m/s. Frozen-mover success at each speed is the
   check: a speed whose frozen success falls more than 0.03 below 1x has a
   mistuned controller and is reported as uninterpretable, not as a result.
3. **Anticipation window.** The replan trigger looks 2 m along the path, which
   is less *time* at higher speed and would bias the test against agility.
   ``block_check_distance`` scales with speed so anticipation time is held
   constant; nothing else about the controller changes.

Identity control at every speed: on frozen worlds the predicting and
non-predicting arms must be bit-identical, as in Phase 5m.

    python scripts/speed_experiment.py --episodes 200
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
from vision_nav.envs.robot import RobotConfig  # noqa: E402

SPEEDS = (0.75, 1.0, 1.5, 2.0)
HORIZON = 2.0
BASE_BLOCK_CHECK = PursuitConfig().block_check_distance

#: Recorded before the run. Reasoned from the configured mover and robot
#: speeds, not from any experiment: nothing has measured the stack at another
#: robot speed. Low confidence on the exact shape, higher on the direction.
PREDICTION = (
    "exposure, not a physical limit. The robot already outruns every dense mover "
    "at 1x (0.6 vs 0.15-0.45 m/s peak). Raising speed shrinks the motion cost in "
    "both arms roughly alike; the fraction prediction recovers stays flat, with "
    "the 2x-minus-1x difference's bootstrap CI including zero. At 0.75x, where the "
    "robot is only as fast as the fastest mover, remaining cost rises. A physical "
    "limit would instead show that fraction rising by >= 0.25 with a CI excluding 0"
)


def robot_overrides(factor: float) -> dict:
    base = RobotConfig()
    return {"robot": {
        "max_linear_vel": base.max_linear_vel * factor,
        "min_linear_vel": base.min_linear_vel * factor,
        "max_angular_vel": base.max_angular_vel * factor,
        "max_linear_accel": base.max_linear_accel * factor,
        "max_angular_accel": base.max_angular_accel * factor,
    }}


def config(factor: float, horizon: float) -> PursuitConfig:
    return PursuitConfig(replan_on_block=True, predict_horizon=horizon,
                         block_check_distance=BASE_BLOCK_CHECK * factor)


def succ(rows):
    return np.array([r["success"] for r in rows], dtype=bool)


def recovery_fraction(frozen, moving0, moving2, idx):
    """Share of the motion cost prediction recovers, on a resample of episodes."""
    cost = succ(frozen)[idx].mean() - succ(moving0)[idx].mean()
    if cost <= 0:
        return None
    gain = succ(moving2)[idx].mean() - succ(moving0)[idx].mean()
    return gain / cost


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--speeds", type=float, nargs="+", default=list(SPEEDS))
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--out", default="results/speed_experiment.json")
    args = p.parse_args(argv)
    if 1.0 not in args.speeds:
        raise ValueError("1.0 must be among the speeds: it is the reference")

    runs: dict[float, dict[float, dict[str, list[dict]]]] = {}
    for s in args.speeds:
        runs[s] = {}
        for h in (0.0, HORIZON):
            print(f"\n=== speed {s:.2f}x, horizon {h:.0f} s ===", flush=True)
            runs[s][h] = {}
            for label, cond in CELLS:
                rows = run_cell(cond, args.episodes, config(s, h), robot_overrides(s))
                runs[s][h][cond] = rows
                print(f"  {label:14s} success {rate(rows, 'success'):.3f}  "
                      f"collision {rate(rows, 'collision'):.3f}  "
                      f"timeout {rate(rows, 'timeout'):.3f}  "
                      f"steps {np.mean([r['steps'] for r in rows]):5.1f}", flush=True)

    report = {"episodes": args.episodes, "horizon": HORIZON, "speeds": args.speeds,
              "prediction": PREDICTION, "checks": {}, "by_speed": {}}

    # --- reproduction of Phase 5m on its own 100 episodes -----------------
    prior = Path("results/velocity_experiment.json")
    if prior.exists():
        vm = json.loads(prior.read_text(encoding="utf-8"))["cells"]
        ok = True
        for h, key in ((0.0, "0.0"), (HORIZON, "2.0")):
            for _label, cond in CELLS:
                old = vm[cond][key]["per_episode"]
                new = runs[1.0][h][cond][:len(old)]
                if [e["success"] for e in old] != [e["success"] for e in new]:
                    ok = False
                    print(f"  REPRODUCTION FAILS: 1x horizon {h} {cond}")
        report["checks"]["reproduces_phase_5m_first_100"] = ok
        print("\n1x reproduces Phase 5m on its first 100 episodes:", ok)

    # --- identity: prediction changes nothing on frozen worlds -------------
    identity = all(
        runs[s][0.0][c] == runs[s][HORIZON][c]
        for s in args.speeds for c in ("dynamic_frozen", "dynamic_dense_frozen")
    )
    report["checks"]["frozen_identity_every_speed"] = identity
    print("frozen identity at every speed:", identity)

    # Compact per-episode record, so every interval below can be recomputed
    # from the result file. The first version kept only summaries, which left
    # the result's most direct test impossible to run without re-running it.
    report["per_episode"] = {
        str(s): {str(h): {c: {"success": [int(r["success"]) for r in runs[s][h][c]],
                              "steps": [int(r["steps"]) for r in runs[s][h][c]]}
                          for _l, c in CELLS}
                 for h in (0.0, HORIZON)}
        for s in args.speeds
    }

    rng = np.random.default_rng(0)
    for moving, frozen in (("dynamic", "dynamic_frozen"),
                           ("dynamic_dense", "dynamic_dense_frozen")):
        print(f"\n=== {moving} ===")
        print(f"  {'speed':>6s} {'frozen':>7s} {'valid':>6s} {'cost':>7s} "
              f"{'remain':>7s} {'gain':>7s} {'p':>7s} {'recov':>6s} {'steps':>6s}")
        ref_frozen = rate(runs[1.0][0.0][frozen], "success")
        n = args.episodes
        boot_idx = [rng.integers(0, n, n) for _ in range(args.bootstrap)]
        ref_boot = [recovery_fraction(runs[1.0][0.0][frozen], runs[1.0][0.0][moving],
                                      runs[1.0][HORIZON][moving], i) for i in boot_idx]

        # The direct test, added after the first run and so NOT pre-registered.
        # The registered criterion -- "the 2x-1x recovery interval includes
        # zero" -- is an unbounded null, the weak form Phase 3d warned against:
        # its intervals came out near +/-0.4 and would have included zero
        # whatever the truth. A physical limit makes a sharper prediction:
        # given twice the agility, the cost *remaining with prediction* should
        # fall towards zero. So bound that difference directly, paired over
        # the same resampled episodes at every speed.
        arr = {s: (succ(runs[s][0.0][frozen]), succ(runs[s][0.0][moving]),
                   succ(runs[s][HORIZON][moving])) for s in args.speeds}

        def costs(s, idx, arr=arr):
            f0, m0, m2 = (a[idx].mean() for a in arr[s])
            return f0 - m0, f0 - m2

        report["by_speed"][moving] = {}
        for s in args.speeds:
            f0 = rate(runs[s][0.0][frozen], "success")
            m0 = rate(runs[s][0.0][moving], "success")
            m2 = rate(runs[s][HORIZON][moving], "success")
            cost, remain, gain = f0 - m0, f0 - m2, m2 - m0
            valid = f0 >= ref_frozen - 0.03
            p_val, lost, won = mcnemar_p(succ(runs[s][0.0][moving]),
                                         succ(runs[s][HORIZON][moving]))
            recov = gain / cost if cost > 0 else None
            steps = float(np.mean([r["steps"] for r in runs[s][0.0][moving]]))

            diffs = []
            for i, r_ref in zip(boot_idx, ref_boot, strict=True):
                r_s = recovery_fraction(runs[s][0.0][frozen], runs[s][0.0][moving],
                                        runs[s][HORIZON][moving], i)
                if r_s is not None and r_ref is not None:
                    diffs.append(r_s - r_ref)
            ci = ([float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
                  if len(diffs) > 100 else None)

            d_cost, d_remain = [], []
            for i in boot_idx:
                c_s, r_s = costs(s, i)
                c_1, r_1 = costs(1.0, i)
                d_cost.append(c_s - c_1)
                d_remain.append(r_s - r_1)

            def pct(v):
                return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

            report["by_speed"][moving][str(s)] = {
                "frozen_success": f0, "controller_valid": bool(valid),
                "motion_cost": cost, "cost_remaining_with_prediction": remain,
                "prediction_gain": gain, "p": p_val, "episodes_won": won,
                "episodes_lost": lost, "fraction_recovered": recov,
                "recovery_minus_1x_ci95": ci, "bootstrap_samples_used": len(diffs),
                "mean_steps_no_prediction": steps,
                # post hoc, see above
                "remaining_minus_1x": remain - (rate(runs[1.0][0.0][frozen], "success")
                                                - rate(runs[1.0][HORIZON][moving], "success")),
                "remaining_minus_1x_ci95": pct(d_remain),
                "cost_minus_1x_ci95": pct(d_cost),
            }
            ci_s = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "   n/a"
            rec_s = f"{recov:+.2f}" if recov is not None else "  n/a"
            rr = report["by_speed"][moving][str(s)]["remaining_minus_1x_ci95"]
            print(f"  {s:5.2f}x {f0:7.3f} {str(valid):>6s} {cost:+7.3f} "
                  f"{remain:+7.3f} {gain:+7.3f} {p_val:7.4f} {rec_s:>6s} "
                  f"{steps:6.1f}  recov vs 1x {ci_s}  remain vs 1x "
                  f"[{rr[0]:+.3f},{rr[1]:+.3f}]")

    print("\npre-registered: " + PREDICTION)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if identity else 1


if __name__ == "__main__":
    raise SystemExit(main())
