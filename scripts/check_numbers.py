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
import subprocess
import sys

TOL = 0.0005  # printed to three decimals


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def doc(name):
    with open(name, encoding="utf-8") as fh:
        return fh.read()


def collected_tests():
    """How many tests pytest actually collects, or None if it cannot be asked.

    The suite size is quoted in three documents and drifts every time a test is
    added -- it was wrong by five before this check existed, and wrong again by
    fourteen within the same day. A number no one can be bothered to re-derive
    by hand is exactly the kind that should not be maintained by hand.

    CHECK_NUMBERS_NO_PYTEST exists so that a test which invokes this module
    cannot recurse into pytest invoking this module.
    """
    if os.environ.get("CHECK_NUMBERS_NO_PYTEST"):
        return None
    env = dict(os.environ, CHECK_NUMBERS_NO_PYTEST="1")
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only"],
                           capture_output=True, text=True, timeout=300, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+) tests? collected", r.stdout)
    return int(m.group(1)) if m else None


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

    # --- report section 8.1, the perception table ---------------------
    # None of these five rows was checked until now. The row for "samples at
    # fixed FOV" is the one that matters most: its success deltas are the
    # evidence for a claim the prose then generalised past, so the collision
    # and timeout deltas are pinned alongside them.
    def cmp_cell(path, cond, metric, field):
        return load(path)["conditions"][cond]["comparisons"][metric][field]

    perception = [
        ("8.1 depth vs lidar narrow", -0.097,
         "results/seed_analysis_depth.json", "narrow", "success", "delta"),
        ("8.1 depth vs lidar p", 0.019,
         "results/seed_analysis_depth.json", "narrow", "success", "p"),
        ("8.1 samples at 90 narrow", 0.002,
         "results/seed_3d_samples_at_90.json", "narrow", "success", "delta"),
        ("8.1 samples at 90 p", 1.000,
         "results/seed_3d_samples_at_90.json", "narrow", "success", "p"),
        ("8.1 samples at 360 narrow", -0.003,
         "results/seed_3d_samples_at_360.json", "narrow", "success", "delta"),
        ("8.1 samples at 360 p", 0.955,
         "results/seed_3d_samples_at_360.json", "narrow", "success", "p"),
        # The delta the success-only reading missed.
        ("8.1 samples at 360 collision", 0.083,
         "results/seed_3d_samples_at_360.json", "narrow", "collision", "delta"),
        ("8.1 samples at 360 collision p", 0.006,
         "results/seed_3d_samples_at_360.json", "narrow", "collision", "p"),
        # The 90-degree contrast is the clean one: nothing moves on any metric
        # or condition, which is what makes the 360-degree result specific
        # rather than a general property of adding samples.
        ("8.1 samples at 90 timeout", 0.038,
         "results/seed_3d_samples_at_90.json", "narrow", "timeout", "delta"),
        ("8.1 samples at 90 timeout p", 0.294,
         "results/seed_3d_samples_at_90.json", "narrow", "timeout", "p"),
        ("8.1 samples at 360 timeout", -0.080,
         "results/seed_3d_samples_at_360.json", "narrow", "timeout", "delta"),
        ("8.1 coverage narrow", 0.095,
         "results/seed_3d_coverage_isores.json", "narrow", "success", "delta"),
        ("8.1 coverage p", 0.024,
         "results/seed_3d_coverage_isores.json", "narrow", "success", "p"),
        # How coverage pays: timeouts become successes, collisions unmoved.
        ("8.1 coverage narrow timeout", -0.127,
         "results/seed_3d_coverage_isores.json", "narrow", "timeout", "delta"),
        ("8.1 coverage narrow timeout p", 0.004,
         "results/seed_3d_coverage_isores.json", "narrow", "timeout", "p"),
        ("8.1 coverage dense timeout", -0.108,
         "results/seed_3d_coverage_isores.json", "dense", "timeout", "delta"),
        ("8.1 coverage dense timeout p", 0.013,
         "results/seed_3d_coverage_isores.json", "dense", "timeout", "p"),
        ("8.1 coverage narrow collision", 0.032,
         "results/seed_3d_coverage_isores.json", "narrow", "collision", "delta"),
        ("8.1 representation narrow", -0.218,
         "results/seed_3e_rgb_vs_depth.json", "narrow", "success", "delta"),
        ("8.1 representation p", 0.002,
         "results/seed_3e_rgb_vs_depth.json", "narrow", "success", "p"),
    ]
    for label, expect, path, cond, metric, field in perception:
        if os.path.exists(path):
            out.append((label, expect, cmp_cell(path, cond, metric, field)))

    if os.path.exists("results/trend_fov_narrow.json"):
        fov = load("results/trend_fov_narrow.json")
        out.append(("8.1 FOV sweep rho", 0.508, fov["spearman_rho"]))
        out.append(("8.1 FOV sweep p", 0.013, fov["p_value"]))

    # --- report 8.4, the re-pricing interaction -----------------------
    # The interaction is the claim: neither single-reward comparison states
    # it, so neither can be checked in place of it.
    rp, rpi = ("results/repricing_resolution.json",
               "results/repricing_interaction.json")
    if os.path.exists(rp) and os.path.exists(rpi):
        narrow = load(rpi)["conditions"]["narrow"]
        for metric, expect_i, expect_p in (("success", 0.035, 0.262),
                                           ("collision", -0.142, 0.0043),
                                           ("timeout", 0.107, 0.0087)):
            out.append((f"8.4 {metric} interaction", expect_i,
                        narrow[metric]["interaction"]))
            out.append((f"8.4 {metric} interaction p", expect_p,
                        narrow[metric]["p"]))
        out += [
            ("8.4 1:1 success delta", 0.032,
             cmp_cell(rp, "narrow", "success", "delta")),
            ("8.4 1:1 collision delta", -0.058,
             cmp_cell(rp, "narrow", "collision", "delta")),
            # The control cell: nominal must not move.
            ("8.4 nominal success delta", 0.000,
             cmp_cell(rp, "nominal", "success", "delta")),
            ("8.4 nominal collision delta", -0.002,
             cmp_cell(rp, "nominal", "collision", "delta")),
            # The absolute cost of indifference on static clutter, which the
            # interaction alone would hide.
            ("8.4 b64 narrow success at 1:1", 0.598,
             st.mean(load(rp)["conditions"]["narrow"]["per_arm"]["b64i"]["success"])),
            ("8.4 b128 narrow success at 1:1", 0.630,
             st.mean(load(rp)["conditions"]["narrow"]["per_arm"]["b128i"]["success"])),
            ("8.4 nominal success b64i", 0.927,
             st.mean(load(rp)["conditions"]["nominal"]["per_arm"]["b64i"]["success"])),
        ]

    # --- 8.5, the recovered timeout column ----------------------------
    # The claim is a contrast *between channels within a comparison*:
    # timeouts significant where collisions are not, on the same six seeds.
    # Both halves are pinned, since the finding is the pair.
    audit = [
        ("8.5 coverage narrow timeout", -0.127,
         "results/seed_3d_coverage_isores.json", "narrow", "timeout", "delta"),
        ("8.5 coverage narrow collision", 0.032,
         "results/seed_3d_coverage_isores.json", "narrow", "collision", "delta"),
        ("8.5 FOV loss dense timeout", 0.122,
         "results/seed_analysis_depth.json", "dense", "timeout", "delta"),
        ("8.5 FOV loss dense timeout p", 0.004,
         "results/seed_analysis_depth.json", "dense", "timeout", "p"),
        ("8.5 FOV loss dense collision", 0.008,
         "results/seed_analysis_depth.json", "dense", "collision", "delta"),
        ("8.5 FOV loss dense success", -0.130,
         "results/seed_analysis_depth.json", "dense", "success", "delta"),
        ("8.5 16v64 dense collision", -0.087,
         "results/seed_analysis_16v64.json", "dense", "collision", "delta"),
        ("8.5 16v64 dense collision p", 0.039,
         "results/seed_analysis_16v64.json", "dense", "collision", "p"),
        ("8.5 16v64 dense timeout", -0.028,
         "results/seed_analysis_16v64.json", "dense", "timeout", "delta"),
        ("8.5 16v64 dense success", 0.115,
         "results/seed_analysis_16v64.json", "dense", "success", "delta"),
        ("8.3 rgb nominal timeout", 0.115,
         "results/seed_3e_rgb_vs_depth.json", "nominal", "timeout", "delta"),
        ("8.3 rgb nominal timeout p", 0.011,
         "results/seed_3e_rgb_vs_depth.json", "nominal", "timeout", "p"),
        ("8.3 rgb nominal collision", 0.047,
         "results/seed_3e_rgb_vs_depth.json", "nominal", "collision", "delta"),
        # 3f's rejection, now null on all three channels rather than one.
        ("8.3 rgb compute narrow timeout", 0.015,
         "results/seed_3f_rgb_compute.json", "narrow", "timeout", "delta"),
        ("8.3 rgb compute narrow collision", -0.072,
         "results/seed_3f_rgb_compute.json", "narrow", "collision", "delta"),
    ]
    for label, expect, path, cond, metric, field in audit:
        if os.path.exists(path):
            out.append((label, expect, cmp_cell(path, cond, metric, field)))

    # --- coverage re-priced at 1:1 ------------------------------------
    # The report's strongest perception claim, re-measured under the reward
    # that closes the channel it was paying through. Success and mechanism are
    # pinned separately because the finding is that one held and one moved.
    cov1, covx = ("results/repricing_coverage.json",
                  "results/repricing_coverage_interaction.json")
    if os.path.exists(cov1) and os.path.exists(covx):
        cx = load(covx)["conditions"]
        out += [
            ("coverage 1:1 narrow success", 0.080,
             cmp_cell(cov1, "narrow", "success", "delta")),
            ("coverage 1:1 narrow success p", 0.032,
             cmp_cell(cov1, "narrow", "success", "p")),
            ("coverage 1:1 dense success", 0.107,
             cmp_cell(cov1, "dense", "success", "delta")),
            ("coverage 1:1 dense success p", 0.004,
             cmp_cell(cov1, "dense", "success", "p")),
            ("coverage 1:1 narrow collision", -0.102,
             cmp_cell(cov1, "narrow", "collision", "delta")),
            ("coverage 1:1 narrow timeout", 0.022,
             cmp_cell(cov1, "narrow", "timeout", "delta")),
            # The effect is invariant: this interaction must stay null.
            ("coverage interaction narrow success", -0.015,
             cx["narrow"]["success"]["interaction"]),
            ("coverage interaction narrow success p", 0.771,
             cx["narrow"]["success"]["p"]),
            # The mechanism is not: these must not.
            ("coverage interaction narrow collision", -0.133,
             cx["narrow"]["collision"]["interaction"]),
            ("coverage interaction narrow collision p", 0.024,
             cx["narrow"]["collision"]["p"]),
            ("coverage interaction narrow timeout", 0.148,
             cx["narrow"]["timeout"]["interaction"]),
            ("coverage interaction narrow timeout p", 0.0022,
             cx["narrow"]["timeout"]["p"]),
            ("coverage interaction dense collision", -0.133,
             cx["dense"]["collision"]["interaction"]),
            ("coverage interaction dense timeout", 0.115,
             cx["dense"]["timeout"]["interaction"]),
        ]

    # --- device benchmark, quoted in resolve_device's docstring -------
    # That docstring decides what every run in this project trains on, so its
    # table should not be able to drift from the measurement behind it.
    db = "results/device_benchmark.json"
    if os.path.exists(db):
        s = load(db)["summary"]
        # Rounded, because the docstring quotes whole steps per second and a
        # 0.0005 tolerance on a four-digit rate would fail on the decimals.
        for cell, cpu, cuda in (("mlp", 1986, 1598), ("mlp128", 1956, 1702),
                                ("lstm", 129, 163)):
            if cell in s:
                out.append((f"device {cell} cpu", cpu, round(s[cell]["cpu_fps"])))
                out.append((f"device {cell} cuda", cuda, round(s[cell]["cuda_fps"])))

    # --- suite size, quoted in three documents ------------------------
    n_tests = collected_tests()
    if n_tests is not None:
        for name, pattern in (
            ("docs/report.md", r"pytest\s+#\s*([0-9]+) tests"),
            ("README.md", r"([0-9]+)-test suite"),
            ("docs/one_page_summary.md", r"([0-9]+)-test suite"),
        ):
            if os.path.exists(name):
                m = re.search(pattern, doc(name))
                if m:
                    out.append((f"{name} suite size", float(m.group(1)),
                                float(n_tests)))

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
