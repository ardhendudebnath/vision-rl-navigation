"""What the policy's privilege is worth: its goal vector from odometry, not truth.

The policy transfer test (``policy_transfer_test.py``, report §9.24) found the
privileged PPO policy moving to a TurtleBot3 at no measurable cost, and named
why: its goal vector comes from the true pose, so it uses the lidar only for
what is near. A real robot without localisation has no true pose. The same
trained policy can be deployed without it, computing the goal vector from the
robot's own dead-reckoned odometry -- no retraining, only a different pose
passed to the adapter. Three arms of ``gazebo_tb3/transfer.py``, on the same
100 val worlds each of `nominal`, `dense` and `narrow`, same rules, same
1200-step budget, arrival judged where the robot really is:

  policy_project_odom  the policy on its training robot in 2-D, odometry from
                       the published odometry model (``OdometryConfig``), the
                       one the classical stack's own localisation corrects
  policy_tb3_2d_odom   the TurtleBot3 in 2-D, the same odometry model
  policy_gazebo_odom   the TurtleBot3 in Gazebo, odometry from the drive's own
                       wheel odometry (/odom): slip is its only error

Each is read against its privileged arm, already run in §9.24 on the same
worlds, and the Gazebo arm against the 2-D one and against the classical stack
in Gazebo (§9.23), which localises and so holds no privilege either:

  policy_project_odom - policy_project   the privilege, published odometry
  policy_tb3_2d_odom - policy_tb3_2d     the same, on the TurtleBot3
  policy_gazebo_odom - policy_gazebo     the privilege, with Gazebo's odometry
  policy_gazebo_odom - policy_tb3_2d_odom  the odometry model's difference
  gazebo - policy_gazebo_odom            the stack against the unprivileged policy

With odometry switched off, each ``_odom`` arm reproduces its privileged arm
episode for episode (``tests/test_transfer_odometry.py``).

    python scripts/policy_privilege_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_test import BOUND, CONDITIONS, bootstrap, mcnemar  # noqa: E402

ARMS = ("policy_project", "policy_tb3_2d", "policy_gazebo",
        "policy_project_odom", "policy_tb3_2d_odom", "policy_gazebo_odom", "gazebo")
ODOM_ARMS = ("policy_project_odom", "policy_tb3_2d_odom", "policy_gazebo_odom")
PAIRS = (("policy_project", "policy_project_odom"), ("policy_tb3_2d", "policy_tb3_2d_odom"),
         ("policy_gazebo", "policy_gazebo_odom"), ("policy_tb3_2d_odom", "policy_gazebo_odom"),
         ("policy_gazebo_odom", "gazebo"))
#: The goal tolerance the worlds judge arrival by (``WorldConfig``).
TOLERANCE = 0.35

#: Recorded and committed before any ``_odom`` arm had run on these val worlds.
#:
#: DISCLOSED: each arm was smoke-tested on train-band worlds. In 2-D, three per
#: condition of nominal and dense: policy_project_odom 2/3 and 2/3 (one
#: timeout, one collision), policy_tb3_2d_odom 2/3 and 2/3 (two timeouts),
#: final odometry errors 0.09 to 1.11 m. In Gazebo, two per condition: 2/2,
#: 2/2 and 1/2 on narrow (one collision), the privileged arm's own smoke
#: tallies on the same worlds, final odometry errors 0.013 to 0.162 m. The
#: privileged arms' val results are §9.24, the stack's in Gazebo §9.23, and the
#: classical stack's own cost of dead reckoning without correction is §9.3.
PREDICTION = (
    "the privilege is worth a lot under the published odometry model and "
    "little under Gazebo's wheel odometry, and the policy does not collide "
    "more for the want of it. policy_project_odom against policy_project: "
    "lower on every condition, and lower pooled by between 0.15 and 0.40. "
    "policy_tb3_2d_odom against policy_tb3_2d: lower pooled by between 0.15 "
    "and 0.40, and its pooled drop within 0.10 of policy_project_odom's. "
    "Every _odom arm's collision rate within 0.05 of its privileged arm's on "
    "every condition, and in each 2-D pair at least two thirds of the worlds "
    "lost end in a timeout. policy_gazebo_odom against policy_tb3_2d_odom: at "
    "least +0.10 pooled, and its median final odometry error at most half the "
    "2-D arm's. gazebo, the classical stack, against policy_gazebo_odom: "
    "pooled success within 0.10 either way, and the stack collides less on "
    "every condition. Decision on the pooled policy_gazebo_odom - "
    "policy_gazebo difference over the 300 paired worlds -- FREE: within 0.05 "
    "either way with its 95% interval inside 0.10 either way. COSTLY: at most "
    "-0.05 with the interval below zero. HELPS: at least +0.05 with the "
    "interval above zero. Otherwise UNRESOLVED. Prediction: FREE."
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/gazebo_tb3")
    p.add_argument("--out", default="results/policy_privilege_test.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    runs = {}
    for arm in ARMS:
        rows = json.loads((Path(args.dir) / f"transfer_{arm}.json").read_text(encoding="utf-8"))["episodes"]
        runs[arm] = {(r["cond"], r["seed"]): r for r in rows}
    keys = sorted(set.intersection(*(set(v) for v in runs.values())),
                  key=lambda k: (CONDITIONS.index(k[0]), k[1]))
    assert all(len(v) == len(keys) for v in runs.values()), "arms ran different worlds"

    def col(arm, field, cond=None):
        return np.array([runs[arm][k][field] for k in keys if cond is None or k[0] == cond])

    report: dict = {"prediction": PREDICTION, "worlds": len(keys), "conditions": {}, "pairs": {},
                    "odometry": {}}
    for cond in CONDITIONS:
        report["conditions"][cond] = {
            arm: {"success": float(col(arm, "success", cond).mean()),
                  "collision": float(col(arm, "collision", cond).mean()),
                  "timeout": float(1 - col(arm, "success", cond).mean() - col(arm, "collision", cond).mean()),
                  "steps_median": float(np.median(col(arm, "steps", cond)))}
            for arm in ARMS}
    groups = np.array([k[0] for k in keys])
    for a, b in PAIRS:
        pair = {}
        for cond in CONDITIONS:
            sa, sb = col(a, "success", cond).astype(bool), col(b, "success", cond).astype(bool)
            pv, won, lost = mcnemar(sa, sb)
            pair[cond] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won, "lost": lost}
        sa, sb = col(a, "success").astype(bool), col(b, "success").astype(bool)
        pv, won, lost = mcnemar(sa, sb)
        # How the worlds b lost ended: out of time, or against something.
        gone = sa & ~sb
        pair["pooled"] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won, "lost": lost,
                          "ci95": bootstrap(sa, sb, groups, rng),
                          "lost_to_timeout": int((gone & ~col(b, "collision").astype(bool)).sum()),
                          "lost_to_collision": int((gone & col(b, "collision").astype(bool)).sum())}
        report["pairs"][f"{b} - {a}"] = pair
    for arm in ODOM_ARMS:
        err, ok = col(arm, "odom_err_final"), col(arm, "success").astype(bool)
        clear = ~col(arm, "collision").astype(bool)
        within = err < TOLERANCE
        report["odometry"][arm] = {
            "final_error_median": float(np.median(err)),
            "final_error_median_by_condition": {c: float(np.median(col(arm, "odom_err_final", c)))
                                                for c in CONDITIONS},
            "final_error_median_successes": float(np.median(err[ok])) if ok.any() else None,
            # Among episodes that did not collide: success when the odometry
            # ended within the goal tolerance of the truth, and when it did not.
            "success_if_within_tolerance": float(ok[clear & within].mean()) if (clear & within).any() else None,
            "success_if_beyond_tolerance": float(ok[clear & ~within].mean()) if (clear & ~within).any() else None,
            "beyond_tolerance": int((clear & ~within).sum())}
    # Added after the run, not registered: which way the odometry was wrong.
    # Arrival is judged where the robot is and ends the episode, so odometry
    # lagging behind the robot should be harmless and odometry running ahead
    # of it -- believing itself nearer the goal than it is -- should strand it.
    report["direction_unregistered"] = {}
    for arm in ODOM_ARMS:
        if any("odom_goal_distance_final" not in runs[arm][k] for k in keys):
            continue
        ok = col(arm, "success").astype(bool)
        clear = ~col(arm, "collision").astype(bool)
        beyond = clear & (col(arm, "odom_err_final") >= TOLERANCE)
        ahead = col(arm, "odom_goal_distance_final") < col(arm, "goal_distance")
        timeout = clear & ~ok
        report["direction_unregistered"][arm] = {
            "beyond_ahead": int((beyond & ahead).sum()),
            "beyond_ahead_success": float(ok[beyond & ahead].mean()) if (beyond & ahead).any() else None,
            "beyond_behind": int((beyond & ~ahead).sum()),
            "beyond_behind_success": float(ok[beyond & ~ahead].mean()) if (beyond & ~ahead).any() else None,
            "timeouts": int(timeout.sum()),
            "timeouts_ahead": int((timeout & ahead).sum()),
            "timeout_goal_distance_median": float(np.median(col(arm, "goal_distance")[timeout])),
            "timeout_believed_distance_median": float(np.median(col(arm, "odom_goal_distance_final")[timeout]))}
    g = report["pairs"]["policy_gazebo_odom - policy_gazebo"]["pooled"]
    if abs(g["delta"]) <= BOUND and -2 * BOUND < g["ci95"][0] and g["ci95"][1] < 2 * BOUND:
        decision = "FREE"
    elif g["delta"] <= -BOUND and g["ci95"][1] < 0:
        decision = "COSTLY"
    elif g["delta"] >= BOUND and g["ci95"][0] > 0:
        decision = "HELPS"
    else:
        decision = "UNRESOLVED"
    report["decision"] = decision
    report["missed_scans"] = int(col("policy_gazebo_odom", "missed_scans").sum())

    for cond in CONDITIONS:
        c = report["conditions"][cond]
        print(cond.ljust(8) + "  ".join(f"{arm} {c[arm]['success']:.2f}/{c[arm]['collision']:.2f}"
                                        for arm in ARMS))
    for name, pair in report["pairs"].items():
        print(name.ljust(42) + "  ".join(
            f"{k} {v['delta']:+.3f} (p {v['p']:.3f}, {v['won']}/{v['lost']})" for k, v in pair.items()))
        print(" " * 42 + f"CI {pair['pooled']['ci95']}; lost to timeout "
              f"{pair['pooled']['lost_to_timeout']}, to collision {pair['pooled']['lost_to_collision']}")
    for arm, o in report["odometry"].items():
        print(f"{arm}: final odometry error median {o['final_error_median']:.3f} m "
              f"{o['final_error_median_by_condition']}; success within/beyond tolerance "
              f"{o['success_if_within_tolerance']}/{o['success_if_beyond_tolerance']} "
              f"({o['beyond_tolerance']} beyond)")
    for arm, d in report["direction_unregistered"].items():
        print(f"{arm} (unregistered): beyond tolerance, ahead {d['beyond_ahead']} arrive "
              f"{d['beyond_ahead_success']}, behind {d['beyond_behind']} arrive {d['beyond_behind_success']}; "
              f"timeouts {d['timeouts']}, ahead {d['timeouts_ahead']}, ending "
              f"{d['timeout_goal_distance_median']:.2f} m out believing {d['timeout_believed_distance_median']:.2f} m")
    print(f"decision {decision}; missed scans {report['missed_scans']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
