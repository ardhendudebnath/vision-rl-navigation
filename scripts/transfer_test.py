"""The transfer test, scored: what survives the move to a TurtleBot3 in Gazebo?

Four arms of ``gazebo_tb3/transfer.py`` on the same val worlds -- 100 each of
`nominal`, `dense` and `narrow` -- every one driven by the same agent, judged
by the same rules from the true pose, within the same 1200-step budget:

  project     the published configuration in 2-D
  tb3_2d      a TurtleBot3 Waffle Pi's specification in 2-D (its limits, the
              LDS-01's bearings, 3.5 m and 5 Hz, 0.064 m behind the axle)
  tb3_2d_6m   the same with a 6 m lidar
  gazebo      the TurtleBot3 in Gazebo Harmonic: a physics engine, wheels that
              slip, a rendered lidar, the real body

Read in pairs, world by world (exact McNemar, bootstrap intervals):

  tb3_2d - project       what the robot's specification costs
  tb3_2d_6m - tb3_2d     how much of that is the lidar's range
  gazebo - tb3_2d        what a physics simulator adds: the sim-to-sim gap,
                         this project's stand-in for sim-to-real

One difference between the last pair is known before any data, and is part of
what the gap measures rather than something to remove: the 2-D arms drift with
the published stack's odometry model (a random walk plus a per-robot scale and
heading bias), while Gazebo's odometry errs only by its wheels' physical slip.

    python scripts/transfer_test.py
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ARMS = ("project", "tb3_2d", "tb3_2d_6m", "gazebo")
CONDITIONS = ("nominal", "dense", "narrow")
#: The bounded-null half-width for a pooled sim-to-sim difference.
BOUND = 0.05

#: Recorded and committed before any arm had been run on these val worlds.
#:
#: DISCLOSED: while finding why the 2-D TurtleBot3 arm failed a unit test, the
#: TurtleBot3 configurations were run on the first three `nominal` val worlds
#: (10000-10002): the 3.5 m lidar arm arrived on two, every 6 m arm on all
#: three, with pose error three to five times higher at 3.5 m. The published
#: configuration's val outcomes on `dense` and `narrow` are known from report
#: §9.9 and §9.22, at a 500-step budget, not this one's 1200. The Gazebo arm has
#: been run only on train-band worlds (four episodes, twice: identical, every
#: one arrived, median pose error 0.03 to 0.05 m).
PREDICTION = (
    "a TurtleBot3's specification costs this stack open worlds, not clutter, "
    "and the physics simulator costs it nothing more. tb3_2d against project: "
    "on nominal at least 0.10 lower with McNemar p < 0.05; on dense and narrow "
    "each within 0.08 either way. tb3_2d_6m against tb3_2d: on nominal at least "
    "0.08 higher; on dense and narrow each within 0.05 either way; and tb3_2d's "
    "median pose error on nominal is at least twice tb3_2d_6m's. gazebo against "
    "tb3_2d: on each condition within 0.10 either way, and gazebo's median pose "
    "error is lower than tb3_2d's on all three conditions, because physical "
    "wheel slip is a gentler odometry error than the published model's bias. "
    "The gazebo arm misses no lidar scan in any episode. "
    "Decision on the pooled gazebo - tb3_2d difference over the 300 paired "
    "worlds -- TRANSFERS: within 0.05 either way with its 95% interval inside "
    "0.10 either way. WORSE IN GAZEBO: at most -0.05 with the interval below "
    "zero. BETTER IN GAZEBO: at least +0.05 with the interval above zero. "
    "Otherwise UNRESOLVED. Prediction: TRANSFERS."
)


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar on paired outcomes: (p, b won, b lost)."""
    won = int(np.sum(~a & b))
    lost = int(np.sum(a & ~b))
    n = won + lost
    if n == 0:
        return 1.0, 0, 0
    k = min(won, lost)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)), won, lost


def bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, rng, n_boot: int = 10000):
    """95% interval of mean(b - a), resampling worlds within each condition."""
    d = b.astype(float) - a.astype(float)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = np.mean(np.concatenate([d[rng.choice(j, len(j))] for j in idx]))
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/gazebo_tb3")
    p.add_argument("--out", default="results/transfer_test.json")
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
                  "pose_err_median": float(np.median(col(arm, "pose_err_median", cond))),
                  "steps_median": float(np.median(col(arm, "steps", cond)))}
            for arm in ARMS}
    groups = np.array([k[0] for k in keys])
    for a, b in (("project", "tb3_2d"), ("tb3_2d", "tb3_2d_6m"), ("tb3_2d", "gazebo")):
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
    g = report["pairs"]["gazebo - tb3_2d"]["pooled"]
    if abs(g["delta"]) <= BOUND and -2 * BOUND < g["ci95"][0] and g["ci95"][1] < 2 * BOUND:
        decision = "TRANSFERS"
    elif g["delta"] <= -BOUND and g["ci95"][1] < 0:
        decision = "WORSE IN GAZEBO"
    elif g["delta"] >= BOUND and g["ci95"][0] > 0:
        decision = "BETTER IN GAZEBO"
    else:
        decision = "UNRESOLVED"
    report["decision"] = decision
    report["missed_scans"] = int(col("gazebo", "missed_scans").sum())

    for cond in CONDITIONS:
        c = report["conditions"][cond]
        print(cond.ljust(8) + "  ".join(f"{arm} {c[arm]['success']:.2f} (pose {c[arm]['pose_err_median']:.3f})"
                                        for arm in ARMS))
    for name, pair in report["pairs"].items():
        print(name.ljust(22) + "  ".join(
            f"{k} {v['delta']:+.3f} (p {v['p']:.3f}, {v['won']}/{v['lost']})" for k, v in pair.items()))
    print(f"pooled gazebo - tb3_2d CI {g['ci95']}; decision {decision}; missed scans {report['missed_scans']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
