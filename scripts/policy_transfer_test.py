"""The policy transfer test: the privileged PPO policy on a TurtleBot3.

The classical stack's transfer test (``transfer_test.py``, report §9.23) found
that with odometry of matching quality the 2-D simulator predicts a TurtleBot3
in Gazebo to within two points. The learned policy is the harder case: it was
trained on one robot and one sensor, and the TurtleBot3 changes both. Four arms
of ``gazebo_tb3/transfer.py`` on the same 100 val worlds each of `nominal`,
`dense` and `narrow`, same rules, same 1200-step budget:

  policy_project       the policy in 2-D on its training robot and own lidar
  policy_tb3_2d        a TurtleBot3's limits and LDS-01 in 2-D, through the
                       adapter (``policy_driver.py``), empty directions read
                       as the training sensor's 6 m
  policy_tb3_2d_naive  the same, empty directions read as the LDS-01's 3.5 m
  policy_gazebo        the TurtleBot3 in Gazebo, through the adapter

Read in pairs, world by world:

  policy_tb3_2d - policy_project        what the robot and sensor cost the policy
  policy_tb3_2d_naive - policy_tb3_2d   what the adapter's one choice is worth
  policy_gazebo - policy_tb3_2d         what a physics simulator adds

The policy's goal vector comes from the true pose, as in training; odometry
reaches it only as its speed and turn-rate inputs. So the odometry model that
separated the stack's arms in §9.23 should matter little here.

    python scripts/policy_transfer_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_test import BOUND, CONDITIONS, bootstrap, mcnemar  # noqa: E402

ARMS = ("policy_project", "policy_tb3_2d", "policy_tb3_2d_naive", "policy_gazebo")

#: Recorded and committed before any policy arm had run on these val worlds.
#:
#: DISCLOSED: each arm was smoke-tested on train-band worlds -- three per
#: condition of nominal and dense in 2-D (policy_project 3/3 and 2/3, with one
#: collision; policy_tb3_2d 3/3 and 2/3; the naive arm 3/3 and 2/3, with one
#: collision), and two per condition in Gazebo (2/2, 2/2, and 1/2 on narrow,
#: one collision). The policy's published val and test results are known from
#: the report, at a 500-step budget. Nothing of the stack's transfer test was
#: hidden: its result is report §9.23.
PREDICTION = (
    "the TurtleBot3 costs the policy in clutter, not in the open, the adapter's "
    "choice for empty directions matters, and the physics simulator adds "
    "nothing. policy_tb3_2d against policy_project: on nominal within 0.05 "
    "either way; on dense and narrow each lower by between 0.05 and 0.25. "
    "policy_tb3_2d_naive against policy_tb3_2d: on nominal at least 0.10 "
    "lower. policy_gazebo against policy_tb3_2d: on each condition within 0.10 "
    "either way, and policy_gazebo collides no more than 0.05 more often than "
    "policy_tb3_2d on any condition. The Gazebo arm misses no lidar scan. "
    "Decision on the pooled policy_gazebo - policy_tb3_2d difference over the "
    "300 paired worlds -- TRANSFERS: within 0.05 either way with its 95% "
    "interval inside 0.10 either way. WORSE IN GAZEBO: at most -0.05 with the "
    "interval below zero. BETTER IN GAZEBO: at least +0.05 with the interval "
    "above zero. Otherwise UNRESOLVED. Prediction: TRANSFERS."
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/gazebo_tb3")
    p.add_argument("--out", default="results/policy_transfer_test.json")
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

    report: dict = {"prediction": PREDICTION, "worlds": len(keys), "conditions": {}, "pairs": {}}
    for cond in CONDITIONS:
        report["conditions"][cond] = {
            arm: {"success": float(col(arm, "success", cond).mean()),
                  "collision": float(col(arm, "collision", cond).mean()),
                  "timeout": float(1 - col(arm, "success", cond).mean() - col(arm, "collision", cond).mean()),
                  "steps_median": float(np.median(col(arm, "steps", cond)))}
            for arm in ARMS}
    groups = np.array([k[0] for k in keys])
    for a, b in (("policy_project", "policy_tb3_2d"), ("policy_tb3_2d", "policy_tb3_2d_naive"),
                 ("policy_tb3_2d", "policy_gazebo")):
        pair = {}
        for cond in CONDITIONS:
            sa, sb = col(a, "success", cond).astype(bool), col(b, "success", cond).astype(bool)
            pv, won, lost = mcnemar(sa, sb)
            pair[cond] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won, "lost": lost}
        sa, sb = col(a, "success").astype(bool), col(b, "success").astype(bool)
        pv, won, lost = mcnemar(sa, sb)
        pair["pooled"] = {"delta": float(sb.mean() - sa.mean()), "p": pv, "won": won, "lost": lost,
                          "ci95": bootstrap(sa, sb, groups, rng)}
        report["pairs"][f"{b} - {a}"] = pair
    g = report["pairs"]["policy_gazebo - policy_tb3_2d"]["pooled"]
    if abs(g["delta"]) <= BOUND and -2 * BOUND < g["ci95"][0] and g["ci95"][1] < 2 * BOUND:
        decision = "TRANSFERS"
    elif g["delta"] <= -BOUND and g["ci95"][1] < 0:
        decision = "WORSE IN GAZEBO"
    elif g["delta"] >= BOUND and g["ci95"][0] > 0:
        decision = "BETTER IN GAZEBO"
    else:
        decision = "UNRESOLVED"
    report["decision"] = decision
    report["missed_scans"] = int(col("policy_gazebo", "missed_scans").sum())

    for cond in CONDITIONS:
        c = report["conditions"][cond]
        print(cond.ljust(8) + "  ".join(f"{arm} {c[arm]['success']:.2f}/{c[arm]['collision']:.2f}"
                                        for arm in ARMS))
    for name, pair in report["pairs"].items():
        print(name.ljust(38) + "  ".join(
            f"{k} {v['delta']:+.3f} (p {v['p']:.3f}, {v['won']}/{v['lost']})" for k, v in pair.items()))
    print(f"pooled gazebo CI {g['ci95']}; decision {decision}; missed scans {report['missed_scans']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
