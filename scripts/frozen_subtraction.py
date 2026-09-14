"""The frozen-mover subtraction: what does motion actually cost each actor?

Report Section 9 found the learned policy statistically indistinguishable from
the classical planner once the map stops being correct, and called that a
regime change. Section 9.3 then found the same claim fails against Nav2 on
`dynamic_dense`. Neither result says *why*, because both compare actors within
a single condition.

This compares the same actor across conditions instead. Each dynamic condition
is re-run with the movers parked at the positions they already occupy at
t = 0 — identical worlds, same seeds, movers still absent from the map, so the
map is exactly as wrong as before and only the motion is gone. The difference
is the cost of motion, per actor, on the same geometry.

    python scripts/frozen_subtraction.py

Reads results/dynamic_experiment.json (moving, learned + classical),
results/dynamic_frozen_experiment.json and results/dynamic_frozen_sparse.json
(frozen, learned + classical), and results/nav2_dynamic/run*/ (both).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

#: (label, moving condition, frozen condition).
PAIRS = [
    ("sparse", "dynamic", "dynamic_frozen"),
    ("dense", "dynamic_dense", "dynamic_dense_frozen"),
]

#: Success-rate gap below which two actors are not worth separating. At 100
#: episodes the binomial standard error near 0.9 is about 0.03, and Nav2's own
#: run-to-run spread reaches 0.030, so anything inside this is noise.
NOISE_BAND = 0.05


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def _nav2(condition):
    out = []
    for run in sorted(glob.glob("results/nav2_dynamic/run*")):
        p = os.path.join(run, condition + "__nav2.json")
        if os.path.exists(p):
            out.append(_load(p))
    return out


def _best_arm(entry):
    """The learned arm with the highest mean success, and its per-seed list."""
    means = {k: st.mean(v["success"]) for k, v in entry["arms"].items()}
    name = max(means, key=means.get)
    return name, entry["arms"][name]["success"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=None, help="Write the markdown table here")
    args = p.parse_args(argv)

    conditions = _load("results/dynamic_experiment.json")["conditions"]
    lines = [
        "| Clutter | Actor | Frozen | Moving | Cost of motion |",
        "|---|---|---|---|---|",
    ]
    notes = []

    for label, mov_cond, frz_cond in PAIRS:
        if mov_cond not in conditions or frz_cond not in conditions:
            notes.append(f"no paired results for {label}")
            continue
        mov, frz = conditions[mov_cond], conditions[frz_cond]

        rows = [("classical",
                 frz["classical"]["success_rate"], mov["classical"]["success_rate"])]

        arm, frz_seeds = _best_arm(frz)
        rows.append((f"learned ({arm})",
                     st.mean(frz_seeds), st.mean(mov["arms"][arm]["success"])))

        nf, nm = _nav2(frz_cond), _nav2(mov_cond)
        if nf and nm:
            rows.append(("Nav2",
                         st.mean(n["success_rate"] for n in nf),
                         st.mean(n["success_rate"] for n in nm)))
        else:
            notes.append(f"no Nav2 results for {label}")

        for i, (actor, f, m) in enumerate(rows):
            lines.append("| {} | {} | {:.3f} | {:.3f} | **{:+.3f}** |".format(
                label if i == 0 else "", actor, f, m, m - f))

        # The decomposition the subtraction exists to produce.
        by = {a: (f, m) for a, f, m in rows}
        if "Nav2" in by:
            fz_gap = by["Nav2"][0] - by["classical"][0]
            mv_gap = by["Nav2"][1] - by["classical"][1]
            swing = mv_gap - fz_gap
            notes.append(
                "{}: Nav2 stands {:+.3f} against the hand-written planner with "
                "the movers frozen and {:+.3f} with them moving — a swing of "
                "{:+.3f} attributable to motion alone, which is {} the noise "
                "band.".format(label, fz_gap, mv_gap, swing,
                               "outside" if abs(swing) > NOISE_BAND else "inside"))

    table = "\n".join(lines)
    print(table)
    print()
    for n in notes:
        print("- " + n)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(table + "\n\n" + "\n".join("- " + n for n in notes) + "\n")
        print("\nWrote " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
