"""Verify the numbers in the documents against the result files.

Every figure in the report, README and one-page summary was typed by hand from
a result file. Dozens of them, across four documents, re-typed whenever an
experiment changed a conclusion. A transcription error would be invisible to
every other check in this repo -- the tests pass, the links resolve, the prose
reads fine, and the number is simply wrong.

This does not parse prose. It checks a curated list of load-bearing claims:
the ones a reader would quote, and the ones that changed most often.

    python scripts/check_numbers.py

Exits non-zero if any claim disagrees with its source.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics as st
import sys

TOL = 0.0005  # printed to three decimals


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def doc(name):
    with open(name, encoding="utf-8") as fh:
        return fh.read()


def classical(cond, field="success_rate"):
    return load(f"results/{cond}__classical.json")[field]


def nav2_range(cond, field="success_rate"):
    vals = []
    for run in sorted(glob.glob("results/nav2_runs/run*")):
        p = os.path.join(run, f"{cond}__nav2.json")
        if os.path.exists(p):
            vals.append(load(p)[field])
    return (min(vals), max(vals)) if vals else None


def dyn(cond, key):
    return load("results/dynamic_experiment.json")["conditions"][cond][key]


def arm_mean(path, cond, arm):
    return st.mean(load(path)["conditions"][cond][arm])


#: benchmark.md column -> field in the corresponding results JSON. Unlike the
#: curated list below, every row of that table is checkable mechanically,
#: because each one maps to exactly one result file.
BENCHMARK_COLUMNS = [
    (2, "success_rate", 3),
    (3, "spl", 3),
    (4, "collision_rate", 3),
    (5, "timeout_rate", 3),
    (6, "mean_steps_to_goal", 0),
]


def benchmark_claims():
    """Every data row of results/benchmark.md against its result file."""
    out = []
    if not os.path.exists("results/benchmark.md"):
        return out
    for line in doc("results/benchmark.md").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("Condition", "---"):
            continue
        cond, actor = cells[0], cells[1]
        path = f"results/{cond}__{actor}.json"
        if not os.path.exists(path):
            out.append((f"benchmark.md {cond}/{actor}: no result file", 1.0, 0.0))
            continue
        data = load(path)
        for idx, field, places in BENCHMARK_COLUMNS:
            try:
                printed = float(cells[idx])
            except ValueError:
                continue
            out.append((f"benchmark.md {cond}/{actor} {field}",
                        printed, round(data[field], places)))
    return out


def claims():
    """(label, value found in the docs, value from the result file)."""
    out = benchmark_claims()

    # --- README results table: classical column -----------------------
    readme = doc("README.md")
    for cond in ("nominal", "sparse", "large", "dense", "narrow"):
        m = re.search(rf"^\| {cond} \| \*\*([0-9.]+)\*\*", readme, re.M)
        if m:
            out.append((f"README classical {cond}",
                        float(m.group(1)), classical(cond)))

    # --- dynamic rows, which were restated three times ----------------
    for cond in ("dynamic", "dynamic_dense"):
        src = dyn(cond, "classical")["success_rate"]
        m = re.search(rf"\| {cond} \| \*\*([0-9.]+)\*\*", readme)
        if m:
            out.append((f"README classical {cond}", float(m.group(1)), src))

    # --- Nav2 ranges in the README ------------------------------------
    for cond in ("nominal", "sparse", "large", "noisy_lidar", "dense", "narrow"):
        rng = nav2_range(cond)
        if not rng:
            continue
        m = re.search(rf"^\| {cond} \| [0-9.]+ \| ([0-9.]+)(?:–([0-9.]+))? \|",
                      readme, re.M)
        if m:
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            out.append((f"README nav2 {cond} low", lo, rng[0]))
            out.append((f"README nav2 {cond} high", hi, rng[1]))

    # --- frame-stacking arms, report section 9.1 ----------------------
    fm, vc, rm = ("results/fast_movers.json", "results/velocity_channel.json",
                  "results/reward_masking.json")
    if all(os.path.exists(p) for p in (fm, vc, rm)):
        out += [
            ("stack1 fast", 0.652, arm_mean(fm, "dynamic_fast", "stack1")),
            ("stack4 fast", 0.625, arm_mean(fm, "dynamic_fast", "stack4")),
            ("stack2 fast", 0.637, arm_mean(vc, "dynamic_fast", "stack2")),
            ("velocity fast", 0.678, arm_mean(vc, "dynamic_fast", "velocity")),
            ("indiff1 fast", 0.660, arm_mean(rm, "dynamic_fast", "indiff1")),
            ("indiff4 fast", 0.693, arm_mean(rm, "dynamic_fast", "indiff4")),
            ("indiff delta fast", 0.033,
             load(rm)["conditions"]["dynamic_fast"]["delta"]),
            ("indiff p fast", 0.019,
             load(rm)["conditions"]["dynamic_fast"]["p"]),
        ]

    # --- recurrence, report section 9.1 and dynamic_obstacles.md ------
    rc, ab = "results/recurrence.json", "results/recurrence_state_ablation.json"
    if os.path.exists(rc) and os.path.exists(ab):
        cond = load(rc)["conditions"]

        def outc(shift, arm, key):
            return st.mean(cond[shift]["outcomes"][arm][key])

        out += [
            ("recurrence memoryless fast", 0.652,
             arm_mean(rc, "dynamic_fast", "memoryless")),
            ("recurrence recurrent fast", 0.583,
             arm_mean(rc, "dynamic_fast", "recurrent")),
            ("recurrence delta fast", -0.068, cond["dynamic_fast"]["delta"]),
            ("recurrence p fast", 0.017, cond["dynamic_fast"]["p"]),
            ("recurrence memoryless slow", 0.802,
             arm_mean(rc, "dynamic", "memoryless")),
            ("recurrence recurrent slow", 0.723,
             arm_mean(rc, "dynamic", "recurrent")),
            ("recurrence delta slow", -0.078, cond["dynamic"]["delta"]),
            ("recurrence p slow", 0.004, cond["dynamic"]["p"]),
            # The breakdown is what separates "worse policy" from the Phase 5h
            # masking pattern, so it is checked rather than trusted.
            ("recurrence collisions memoryless", 0.300,
             outc("dynamic_fast", "memoryless", "collision")),
            ("recurrence collisions recurrent", 0.323,
             outc("dynamic_fast", "recurrent", "collision")),
            ("recurrence timeouts memoryless", 0.048,
             outc("dynamic_fast", "memoryless", "timeout")),
            ("recurrence timeouts recurrent", 0.093,
             outc("dynamic_fast", "recurrent", "timeout")),
            # Both artefact checks are load-bearing for the negative result.
            ("recurrence tail gain baseline", 0.026,
             st.mean(load(rc)["tail_gain"]["memoryless"])),
            ("recurrence tail gain recurrent", -0.008,
             st.mean(load(rc)["tail_gain"]["recurrent"])),
            ("recurrence state ablation", 0.315, load(ab)["mean_delta"]),
            ("recurrence without memory", 0.268,
             st.mean([r["without_memory"] for r in load(ab)["per_seed"]])),
        ]

    # --- horizon sweep ------------------------------------------------
    hz_rows = {}
    for d in sorted(glob.glob("results/nav2_horizon/sim*")):
        hz = float(re.search(r"sim([0-9.]+)$", d).group(1))
        cells = {}
        for p in glob.glob(os.path.join(d, "*__nav2.json")):
            cells[os.path.basename(p).split("__")[0]] = load(p)["success_rate"]
        hz_rows[hz] = cells
    for hz, expect_moving, expect_frozen in [
        (0.5, 0.780, 0.830), (1.0, 0.910, 0.960),
        (1.5, 0.870, 0.960), (3.0, 0.690, 0.740),
    ]:
        if hz in hz_rows and "dynamic_dense" in hz_rows[hz]:
            out.append((f"horizon {hz}s moving", expect_moving,
                        hz_rows[hz]["dynamic_dense"]))
            out.append((f"horizon {hz}s frozen", expect_frozen,
                        hz_rows[hz]["dynamic_dense_frozen"]))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    bad = []
    rows = claims()
    for label, printed, actual in rows:
        if actual is None or abs(printed - actual) > TOL:
            bad.append((label, printed, actual))

    if not args.quiet:
        print(f"checked {len(rows)} numeric claims against results/")
    for label, printed, actual in bad:
        print(f"  MISMATCH {label}: docs say {printed}, "
              f"results say {actual}", file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} mismatch(es)", file=sys.stderr)
        return 1
    if not args.quiet:
        print("every checked number matches its result file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
