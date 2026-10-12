"""How much odometry drift each controller can stand: a sweep on the 2-D TurtleBot3.

§9.23 and §9.25 left both controllers' results on a TurtleBot3 turning on one
unmeasured quantity -- how far a real one's odometry drifts. The classical stack
lost 0.13 to 0.16 between exact odometry and the published odometry model; the
policy without its privilege lost 0.11. Two points do not make a curve, and a
curve is what a measurement on hardware could be read against. This sweeps the
published model's error terms -- random walk and per-robot bias together -- by
a scale: 0 is exact odometry, 1 the model as published (its scale bias a 2%
wheel-radius error, one standard deviation), 0.5 a robot calibrated to half
that. On the same 100 val worlds each of `nominal`, `dense` and `narrow`, at
the 1200-step budget, both controllers on the 2-D TurtleBot3:

  stack    the published stack through ``tb3_agent.py`` (``tb3_2d`` arm),
           localising with odometry and scan matching
  policy   the PPO policy with its goal vector from odometry
           (``policy_tb3_2d_odom`` arm)

Scales 0 and 1 are the arms already run (scale 0: ``tb3_2d_exact_odom`` and
``policy_tb3_2d``, which ``tests/test_transfer_odometry.py`` shows are what
scale 0 reproduces); 0.25, 0.5 and 2 are new:

    python gazebo_tb3/transfer.py --arm tb3_2d --odometry-scale 0.5 \\
        --out results/gazebo_tb3/drift_stack_0.5.json
    python scripts/drift_sweep_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_test import BOUND, CONDITIONS, bootstrap, mcnemar  # noqa: E402

SCALES = (0.0, 0.25, 0.5, 1.0, 2.0)
#: Where each controller's run at each scale is.
FILES = {
    "stack": {0.0: "transfer_tb3_2d_exact_odom", 1.0: "transfer_tb3_2d",
              **{k: f"drift_stack_{k:g}" for k in (0.25, 0.5, 2.0)}},
    "policy": {0.0: "transfer_policy_tb3_2d", 1.0: "transfer_policy_tb3_2d_odom",
               **{k: f"drift_policy_{k:g}" for k in (0.25, 0.5, 2.0)}},
}
DECIDING = 0.5

#: Recorded and committed before any new scale had run on these val worlds.
#:
#: DISCLOSED: scales 0 and 1 are published for both controllers (§9.23-§9.25):
#: pooled 0.757 and 0.617 for the stack, 0.737 and 0.623 for the policy. Each
#: new arm was smoke-tested at scales 0.5 and 2 on two train-band worlds per
#: condition: at 0.5 the stack 2/2, 2/2, 1/2 and the policy 2/2, 2/2, 1/2 (one
#: collision); at 2 the stack 1/2, 2/2, 1/2 and the policy 1/2, 1/2, 1/2 (one
#: collision). The reasoning: §9.25 found the policy failing only when its
#: odometry ends past the goal tolerance and ahead of the robot, a threshold
#: that drift scaled down crosses rarely; the stack corrects its drift by scan
#: matching, which should leave small drift nearly free as well.
PREDICTION = (
    "loss grows faster than the drift: small drift is nearly free, and the "
    "published model already sits past the knee. Against exact odometry, "
    "pooled over the 300 paired worlds, each controller loses at most 0.03 at "
    "scale 0.25 and at most 0.05 at scale 0.5, and between 0.15 and 0.40 at "
    "scale 2. Neither controller's pooled success rises by more than 0.02 from "
    "one scale to the next. At scale 2 the policy loses less than the stack, as "
    "at scale 1. At every new scale each controller's collision rate is within "
    "0.05 of its scale-0 arm's on every condition. The stack's median pose "
    "error and the policy's median final odometry error rise with every step "
    "in scale. Decision at scale 0.5 -- a controller TOLERATES it if its pooled "
    "success is at most 0.05 below exact odometry's with its 95% interval's "
    "lower end above -0.10: BOTH TOLERATE, STACK ONLY, POLICY ONLY or NEITHER. "
    "Prediction: BOTH TOLERATE."
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/gazebo_tb3")
    p.add_argument("--out", default="results/drift_sweep_test.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    runs = {}
    for ctl, files in FILES.items():
        for k, name in files.items():
            data = json.loads((Path(args.dir) / f"{name}.json").read_text(encoding="utf-8"))
            if k not in (0.0, 1.0):
                assert data["odometry_scale"] == k, name
            runs[(ctl, k)] = {(r["cond"], r["seed"]): r for r in data["episodes"]}
    keys = sorted(set.intersection(*(set(v) for v in runs.values())),
                  key=lambda x: (CONDITIONS.index(x[0]), x[1]))
    assert all(len(v) == len(keys) for v in runs.values()), "runs cover different worlds"
    groups = np.array([c for c, _ in keys])

    def col(ctl, k, field, cond=None):
        return np.array([runs[(ctl, k)][key][field] for key in keys if cond is None or key[0] == cond])

    report: dict = {"prediction": PREDICTION, "worlds": len(keys), "controllers": {}}
    for ctl in FILES:
        out = {}
        exact = col(ctl, 0.0, "success").astype(bool)
        for k in SCALES:
            s = col(ctl, k, "success").astype(bool)
            row = {"success": float(s.mean()),
                   "by_condition": {c: {"success": float(col(ctl, k, "success", c).mean()),
                                        "collision": float(col(ctl, k, "collision", c).mean())}
                                    for c in CONDITIONS}}
            if k > 0:
                pv, won, lost = mcnemar(exact, s)
                row.update(delta=float(s.mean() - exact.mean()), p=pv, won=won, lost=lost,
                           ci95=bootstrap(exact, s, groups, rng))
            field = "pose_err_median" if ctl == "stack" else "odom_err_final"
            if k > 0 or ctl == "stack":
                row["error_median"] = float(np.median(col(ctl, k, field)))
            out[f"{k:g}"] = row
        d = out[f"{DECIDING:g}"]
        out["tolerates_0.5"] = bool(d["delta"] >= -BOUND and d["ci95"][0] > -2 * BOUND)
        # The largest scale swept at which the controller still tolerates drift.
        tolerated = [k for k in SCALES[1:] if out[f"{k:g}"]["delta"] >= -BOUND
                     and out[f"{k:g}"]["ci95"][0] > -2 * BOUND]
        out["largest_tolerated"] = max(tolerated) if tolerated else 0.0
        report["controllers"][ctl] = out
    st, po = report["controllers"]["stack"]["tolerates_0.5"], report["controllers"]["policy"]["tolerates_0.5"]
    report["decision"] = ("BOTH TOLERATE" if st and po else "STACK ONLY" if st
                          else "POLICY ONLY" if po else "NEITHER")

    for ctl, out in report["controllers"].items():
        for k in SCALES:
            r = out[f"{k:g}"]
            extra = (f"  {r['delta']:+.3f} (p {r['p']:.3f}, {r['won']}/{r['lost']}) "
                     f"CI {np.round(r['ci95'], 3).tolist()}" if k > 0 else "")
            err = f"  error {r['error_median']:.3f} m" if "error_median" in r else ""
            print(f"{ctl:6s} x{k:<4g} " + "  ".join(
                f"{c} {v['success']:.2f}/{v['collision']:.2f}" for c, v in r["by_condition"].items())
                + f"  pooled {r['success']:.3f}{extra}{err}")
        print(f"{ctl}: tolerates 0.5 {out['tolerates_0.5']}, largest tolerated {out['largest_tolerated']:g}")
    print(f"decision {report['decision']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
