"""Does the length of the controller's commitment explain the cost of motion?

Report Section 9.2 ruled out replanning churn, planning failure and sensing as
explanations for moving obstacles costing the classical stack 0.120-0.160
success where they cost the learned policy 0.048-0.068. What survived is that
**committing to a plan is itself the cost**: a trajectory is computed against a
snapshot and goes stale the moment the world moves, so the longer the
commitment, the worse it should be.

Nav2's DWB re-scores trajectories every control cycle over a `sim_time`
horizon, which is exactly the length of that commitment. Sweeping it turns the
single data point Section 9.2 had -- DWB paying 0.075 under clutter against
the hand-written stack's 0.160 -- into a curve.

The frozen arm is the control. Reading the moving row alone cannot separate
"long horizons are bad under motion" from "long horizons are bad", and that
distinction is the whole hypothesis. The churn experiment failed in exactly
that way and only its control revealed it.

    python scripts/horizon_analysis.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

#: A gap this size or smaller is noise: 100 episodes gives a binomial standard
#: error near 0.03 at these rates, and Nav2's own run-to-run spread reaches
#: 0.030.
NOISE_BAND = 0.05


def load(root: str) -> dict[float, dict[str, dict]]:
    out: dict[float, dict[str, dict]] = {}
    for d in sorted(glob.glob(os.path.join(root, "sim*"))):
        m = re.search(r"sim([0-9.]+)$", d)
        if not m:
            continue
        hz = float(m.group(1))
        for path in glob.glob(os.path.join(d, "*__nav2.json")):
            cond = os.path.basename(path).split("__")[0]
            out.setdefault(hz, {})[cond] = json.loads(
                open(path, encoding="utf-8").read())
    return out


def spearman(xs, ys) -> float:
    """Rank correlation, computed directly to avoid a scipy dependency."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/nav2_horizon")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    data = load(args.results)
    if not data:
        print("no results in {}/".format(args.results))
        return 1

    hzs = sorted(data)
    lines = [
        "| DWB horizon | Moving | Frozen (control) | Cost of motion |",
        "|---|---|---|---|",
    ]
    moving, frozen, cost = [], [], []
    for hz in hzs:
        cells = data[hz]
        if "dynamic_dense" not in cells or "dynamic_dense_frozen" not in cells:
            continue
        m = cells["dynamic_dense"]["success_rate"]
        f = cells["dynamic_dense_frozen"]["success_rate"]
        moving.append(m)
        frozen.append(f)
        cost.append(m - f)
        lines.append("| {:.1f} s | {:.3f} | {:.3f} | **{:+.3f}** |".format(
            hz, m, f, m - f))

    table = "\n".join(lines)
    print(table)
    if len(cost) < 3:
        print("\nneed at least three horizons for a trend")
        return 1

    used = hzs[:len(cost)]
    rho_cost = spearman(used, cost)
    rho_frozen = spearman(used, frozen)
    spread_cost = max(cost) - min(cost)
    spread_frozen = max(frozen) - min(frozen)

    print()
    print("cost of motion vs horizon : rho = {:+.3f}, range {:.3f}".format(
        rho_cost, spread_cost))
    print("frozen success vs horizon : rho = {:+.3f}, range {:.3f}".format(
        rho_frozen, spread_frozen))

    # Pre-registered: cost of motion rises with horizon (rho <= -0.8 on the
    # signed cost, which is negative and grows more negative), by at least
    # 0.04 across the range, while the frozen control stays within +/-0.04.
    print()
    grew = rho_cost <= -0.8 and spread_cost >= 0.04
    control_flat = spread_frozen <= 0.04
    if grew and control_flat:
        print("PREDICTION HELD. Commitment length is causal: the cost of "
              "motion grows with the horizon while the frozen control does "
              "not. The Section 9.2 hypothesis is supported.")
    elif grew and not control_flat:
        print("PARTIAL, AND THE CONTROL SAYS NO. The cost grows with horizon, "
              "but the frozen arm moves by {:.3f} too, so longer horizons are "
              "simply worse here and motion is incidental. Same shape of "
              "failure as the churn experiment.".format(spread_frozen))
    else:
        print("PREDICTION FAILED. The cost of motion is flat in the horizon "
              "(range {:.3f}), so commitment length is not the mechanism and "
              "the motion cost stays unexplained.".format(spread_cost))

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(table + "\n")
        print("\nWrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
