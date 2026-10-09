"""Is Gazebo's edge over the 2-D TurtleBot3 the odometry model? A follow-up.

The registered transfer test (``transfer_test.py``) found the TurtleBot3 in
Gazebo arriving more often than its 2-D twin on every condition, with lower
pose error -- where it predicted no difference. Its registration had named the
one difference between the two arms before any data: the 2-D arm drifts with
the published stack's odometry model (a random walk plus a per-robot scale and
heading bias), Gazebo only by its wheels' physical slip.

This tests that reading with one more 2-D arm, added after the registered run
and not registered: ``tb3_2d_exact_odom`` is ``tb3_2d`` with the odometry model
switched off, its odometry integrating the true velocity. If the gap is the
odometry model, this arm lands at Gazebo or above it -- it has no drift at all,
where Gazebo still slips -- and the physics simulator itself costs nothing.

    python scripts/transfer_odometry_followup.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_test import CONDITIONS, bootstrap, mcnemar  # noqa: E402

ARMS = ("tb3_2d", "gazebo", "tb3_2d_exact_odom")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="results/gazebo_tb3")
    p.add_argument("--out", default="results/transfer_odometry_followup.json")
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

    report: dict = {"registered": False, "worlds": len(keys), "conditions": {}, "pairs": {}}
    for cond in CONDITIONS:
        report["conditions"][cond] = {
            arm: {"success": float(col(arm, "success", cond).mean()),
                  "pose_err_median": float(np.median(col(arm, "pose_err_median", cond))),
                  "pose_err_final_median": float(np.median(col(arm, "pose_err_final", cond)))}
            for arm in ARMS}
    groups = np.array([k[0] for k in keys])
    for a, b in (("tb3_2d", "tb3_2d_exact_odom"), ("gazebo", "tb3_2d_exact_odom")):
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

    for cond in CONDITIONS:
        c = report["conditions"][cond]
        print(cond.ljust(8) + "  ".join(f"{arm} {c[arm]['success']:.2f} (pose {c[arm]['pose_err_median']:.3f})"
                                        for arm in ARMS))
    for name, pair in report["pairs"].items():
        print(name.ljust(30) + "  ".join(
            f"{k} {v['delta']:+.3f} (p {v['p']:.3f}, {v['won']}/{v['lost']})" for k, v in pair.items()),
            pair["pooled"]["ci95"])
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
