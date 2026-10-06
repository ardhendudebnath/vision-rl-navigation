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
    if r.returncode != 0:
        # Collection errors still print "N tests collected" for the modules
        # that imported -- under an interpreter without the package installed,
        # 11 -- which would read as the suite having shrunk.
        print("  suite size not checked: pytest collection failed "
              f"(exit {r.returncode}; is {sys.executable} the project interpreter?)")
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
        # A † marks a cell measured on the wrong sensor; it is still the
        # recorded number, so it is still checked.
        m = re.search(rf"^\| {cond} \| [0-9.]+ \| ([0-9.]+)(?:–([0-9.]+))?(?: †)? \|",
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

    # --- oracle motion prediction, report 9.1 and dynamic_obstacles.md --
    # The frozen identity is a precondition for every other number here, so
    # it is checked as a claim in its own right rather than assumed.
    ve = "results/velocity_experiment.json"
    if os.path.exists(ve):
        v = load(ve)
        dd = v["effects"]["dynamic_dense"]
        dy = v["effects"]["dynamic"]
        out += [
            ("velocity frozen identity", 1.0, float(v["frozen_identity_holds"])),
            ("velocity dense motion cost", 0.160, dd["motion_cost"]),
            ("velocity sparse motion cost", 0.120, dy["motion_cost"]),
            ("velocity dense 2s gain", 0.070, dd["by_horizon"]["2.0"]["success_gain"]),
            ("velocity dense 2s p", 0.0156, dd["by_horizon"]["2.0"]["p"]),
            ("velocity dense 2s won", 7, dd["by_horizon"]["2.0"]["episodes_won"]),
            ("velocity dense 2s lost", 0, dd["by_horizon"]["2.0"]["episodes_lost"]),
            ("velocity dense 2s recovered", 0.4375,
             dd["by_horizon"]["2.0"]["fraction_of_cost_recovered"]),
            ("velocity dense 2s collision", -0.100,
             dd["by_horizon"]["2.0"]["collision_delta"]),
            ("velocity dense 1s gain", 0.020, dd["by_horizon"]["1.0"]["success_gain"]),
            ("velocity dense 4s gain", 0.040, dd["by_horizon"]["4.0"]["success_gain"]),
            ("velocity sparse 4s gain", 0.060, dy["by_horizon"]["4.0"]["success_gain"]),
            ("velocity sparse 4s p", 0.0703, dy["by_horizon"]["4.0"]["p"]),
            # Quoted in dynamic_obstacles.md, and once hardcoded there from
            # printed output before the script stored them.
            ("velocity dense replans h0", 0.9,
             v["cells"]["dynamic_dense"]["0.0"]["replans_mean"]),
            ("velocity dense replans h2", 3.8,
             v["cells"]["dynamic_dense"]["2.0"]["replans_mean"]),
        ]

    # --- snapshot experiment, Phase 5n ---------------------------------
    # Both checks are preconditions for every contrast, so they are claims
    # in their own right rather than assumptions.
    se = "results/snapshot_experiment.json"
    if os.path.exists(se):
        sn = load(se)
        ch, ct = sn["checks"], sn["contrasts"]
        out += [
            ("snapshot reproduces 5m", 1.0, float(ch["reproduces_phase_5m"])),
            ("snapshot frozen identities", 1.0, float(ch["frozen_identities_hold"])),
            ("snapshot dense cost remaining", 0.090,
             ch["dynamic_dense_cost_remaining_after_5m"]),
            ("snapshot sparse cost remaining", 0.090,
             ch["dynamic_cost_remaining_after_5m"]),
            ("snapshot initial sees dense", 0.010,
             ct["initial plan sees movers at all"]["dynamic_dense"]["success_gain"]),
            ("snapshot initial sees sparse", -0.010,
             ct["initial plan sees movers at all"]["dynamic"]["success_gain"]),
            ("snapshot initial predicts dense", 0.000,
             ct["initial plan predicts"]["dynamic_dense"]["success_gain"]),
            ("snapshot initial predicts sparse", 0.000,
             ct["initial plan predicts"]["dynamic"]["success_gain"]),
            ("snapshot caution sparse", -0.040,
             ct["slow-down predicts"]["dynamic"]["success_gain"]),
            ("snapshot caution sparse p", 0.2188,
             ct["slow-down predicts"]["dynamic"]["p"]),
            ("snapshot caution sparse collision", 0.040,
             ct["slow-down predicts"]["dynamic"]["collision_delta"]),
            ("snapshot caution dense", 0.000,
             ct["slow-down predicts"]["dynamic_dense"]["success_gain"]),
        ]

    # --- robot agility, Phase 5o ---------------------------------------
    # The conclusion rests on the post hoc interval on remaining cost, so both
    # of its ends are pinned, alongside the exposure interval the writeup
    # declines to draw a conclusion from.
    sp = "results/speed_experiment.json"
    if os.path.exists(sp):
        spd = load(sp)
        spc, bd, bs = spd["checks"], spd["by_speed"]["dynamic_dense"], spd["by_speed"]["dynamic"]
        out += [
            ("speed reproduces 5m", 1.0, float(spc["reproduces_phase_5m_first_100"])),
            ("speed frozen identity", 1.0, float(spc["frozen_identity_every_speed"])),
            ("speed dense 0.75x invalid", 0.0, float(bd["0.75"]["controller_valid"])),
            ("speed dense 1x remaining", 0.085, bd["1.0"]["cost_remaining_with_prediction"]),
            ("speed dense 2x remaining diff", 0.005, bd["2.0"]["remaining_minus_1x"]),
            ("speed dense 2x remaining ci lo", -0.045, bd["2.0"]["remaining_minus_1x_ci95"][0]),
            ("speed dense 2x remaining ci hi", 0.055, bd["2.0"]["remaining_minus_1x_ci95"][1]),
            ("speed sparse 2x remaining diff", -0.030, bs["2.0"]["remaining_minus_1x"]),
            ("speed sparse 2x remaining ci lo", -0.085, bs["2.0"]["remaining_minus_1x_ci95"][0]),
            ("speed sparse 2x remaining ci hi", 0.025, bs["2.0"]["remaining_minus_1x_ci95"][1]),
            ("speed dense 2x cost ci lo", -0.070, bd["2.0"]["cost_minus_1x_ci95"][0]),
            ("speed dense 2x cost ci hi", 0.050, bd["2.0"]["cost_minus_1x_ci95"][1]),
            ("speed dense 1x steps", 188, round(bd["1.0"]["mean_steps_no_prediction"])),
            ("speed dense 2x steps", 98, round(bd["2.0"]["mean_steps_no_prediction"])),
            ("speed dense 1x gain p", 0.004, bd["1.0"]["p"]),
        ]

    # --- space-time planning, Phase 5p ---------------------------------
    stp = "results/spacetime_experiment.json"
    if os.path.exists(stp):
        stj = load(stp)
        stc, stx = stj["checks"], stj["contrasts"]
        fsd, fss = stx["full_vs_swept"]["dynamic_dense"], stx["full_vs_swept"]["dynamic"]
        fpd = stx["full_vs_spatial"]["dynamic_dense"]
        sps = stx["swept_vs_spatial"]["dynamic"]
        spd = stx["swept_vs_spatial"]["dynamic_dense"]
        out += [
            ("spacetime spatial reproduces 5m", 1.0, float(stc["spatial_reproduces_phase_5m"])),
            ("spacetime frozen identity", 1.0, float(stc["frozen_identity_full_vs_swept"])),
            ("spacetime dense timing gain", 0.050, fsd["success_gain"]),
            ("spacetime dense timing p", 0.031, fsd["p"]),
            ("spacetime dense timing won", 14, fsd["episodes_won"]),
            ("spacetime dense timing lost", 4, fsd["episodes_lost"]),
            ("spacetime dense timing ci lo", 0.010, fsd["ci95"][0]),
            ("spacetime dense timing ci hi", 0.090, fsd["ci95"][1]),
            ("spacetime dense timing timeout delta", -0.060, fsd["timeout_delta"]),
            ("spacetime dense timing collision delta", 0.010, fsd["collision_delta"]),
            ("spacetime sparse timing gain", -0.050, fss["success_gain"]),
            ("spacetime sparse timing p", 0.002, fss["p"]),
            ("spacetime sparse timing won", 0, fss["episodes_won"]),
            ("spacetime sparse timing lost", 10, fss["episodes_lost"]),
            ("spacetime sparse timing collision delta", 0.050, fss["collision_delta"]),
            ("spacetime dense cost spatial", 0.085, fpd["motion_cost_reference"]),
            ("spacetime dense cost full", 0.025, fpd["motion_cost_treated"]),
            ("spacetime sparse cost spatial", 0.095, sps["motion_cost_reference"]),
            ("spacetime sparse cost swept", 0.000, sps["motion_cost_treated"]),
            ("spacetime sparse swept won", 18, sps["episodes_won"]),
            ("spacetime sparse swept lost", 0, sps["episodes_lost"]),
            ("spacetime dense swept gain", 0.020, spd["success_gain"]),
            # Quoted to two decimals in the documents, so checked at two.
            ("spacetime dense swept p", 0.58, round(spd["p"], 2)),
        ]
        # The registered labels are part of the record: the documents say the
        # sparse contrast was *registered* as inconclusive and corrected to
        # HARMS, so both halves of that sentence are checked.
        registered = {"dense": fsd["verdict"], "sparse": fss["verdict"]}
        for key, want in (("dense", "MATTERS"), ("sparse", "inconclusive")):
            out.append((f"spacetime registered label {key}", 1.0,
                        float(registered[key] == want)))

    # --- temporal safety margin, Phase 5q ------------------------------
    mg = "results/margin_experiment.json"
    if os.path.exists(mg):
        mj = load(mg)
        mc, mcells, mx = mj["checks"], mj["cells"], mj["contrasts"]
        mh1, mh2 = mx["m2_vs_m0"]["dynamic"], mx["m2_vs_swept"]["dynamic_dense"]
        out += [
            ("margin frozen identity", 1.0, float(mc["frozen_identity_m0_vs_m4"])),
            ("margin reproduces 5p", 1.0, float(mc["reproduces_phase_5p"])),
            ("margin decision FIXED", 1.0, float(mj["decision"] == "FIXED")),
            ("margin H1 gain", 0.050, mh1["success_gain"]),
            ("margin H1 p", 0.002, mh1["p"]),
            ("margin H1 won", 10, mh1["episodes_won"]),
            ("margin H1 lost", 0, mh1["episodes_lost"]),
            ("margin H1 collision delta", -0.050, mh1["collision_delta"]),
            ("margin H2 gain", 0.065, mh2["success_gain"]),
            ("margin H2 p", 0.004, mh2["p"]),
            ("margin sparse m0", 0.945, mcells["m0"]["dynamic"]["success"]),
            ("margin sparse m1", 0.980, mcells["m1"]["dynamic"]["success"]),
            ("margin sparse m2", 0.995, mcells["m2"]["dynamic"]["success"]),
            ("margin dense m4", 0.970, mcells["m4"]["dynamic_dense"]["success"]),
            ("margin dense swept", 0.910, mcells["swept"]["dynamic_dense"]["success"]),
            # Derived: margin-2 frozen is margin-0 frozen, by the identity.
            ("margin motion cost dense",  0.010,
             mcells["m0"]["dynamic_dense_frozen"]["success"]
             - mcells["m2"]["dynamic_dense"]["success"]),
            ("margin motion cost sparse", 0.000,
             mcells["m0"]["dynamic_frozen"]["success"] - mcells["m2"]["dynamic"]["success"]),
        ]

    # --- constant-velocity estimate, Phase 5r ---------------------------
    es = "results/estimate_experiment.json"
    if os.path.exists(es):
        ej = load(es)
        ec, ecells = ej["checks"], ej["cells"]
        primary, wide = ej["contrasts"]["cv_m2_vs_oracle_m2"], ej["contrasts"]["cv_m4_vs_cv_m2"]
        ed, esp, ew = primary["dynamic_dense"], primary["dynamic"], wide["dynamic_dense"]
        out += [
            ("estimate frozen identity", 1.0, float(ec["frozen_identity_cv_vs_oracle"])),
            ("estimate reproduces 5q", 1.0, float(ec["oracle_reproduces_phase_5q"])),
            ("estimate decision COSTLY", 1.0, float(ej["decision"] == "COSTLY")),
            ("estimate dense gain", -0.065, ed["success_gain"]),
            ("estimate dense p", 0.001, ed["p"]),
            ("estimate dense won", 1, ed["episodes_won"]),
            ("estimate dense lost", 14, ed["episodes_lost"]),
            ("estimate dense ci lo", -0.105, ed["ci95"][0]),
            ("estimate dense ci hi", -0.030, ed["ci95"][1]),
            ("estimate dense collision delta", 0.065, ed["collision_delta"]),
            ("estimate dense timeout delta", 0.000, ed["timeout_delta"]),
            ("estimate sparse gain", -0.020, esp["success_gain"]),
            ("estimate sparse p", 0.125, esp["p"]),
            ("estimate sparse won", 0, esp["episodes_won"]),
            ("estimate sparse lost", 4, esp["episodes_lost"]),
            ("estimate sparse ci lo", -0.040, esp["ci95"][0]),
            ("estimate sparse ci hi", -0.005, esp["ci95"][1]),
            ("estimate m4 vs m2 dense", 0.000, ew["success_gain"]),
            ("estimate m4 vs m2 ci lo", -0.025, ew["ci95"][0]),
            ("estimate m4 vs m2 ci hi", 0.030, ew["ci95"][1]),
            ("estimate motion cost dense", 0.075, ec["motion_cost_dynamic_dense"]["cv_m2"]),
            ("estimate motion cost sparse", 0.020, ec["motion_cost_dynamic"]["cv_m2"]),
        ]
        # The companion's table, cell by cell.
        for arm, cond, key, want in (
            ("oracle_m2", "dynamic", "success", 0.995), ("oracle_m2", "dynamic", "collision", 0.000),
            ("cv_m2", "dynamic", "success", 0.975), ("cv_m2", "dynamic", "collision", 0.020),
            ("cv_m4", "dynamic", "success", 0.970), ("cv_m4", "dynamic", "collision", 0.025),
            ("oracle_m2", "dynamic_dense", "success", 0.975),
            ("oracle_m2", "dynamic_dense", "collision", 0.010),
            ("oracle_m2", "dynamic_dense", "timeout", 0.015),
            ("cv_m2", "dynamic_dense", "success", 0.910),
            ("cv_m2", "dynamic_dense", "collision", 0.075),
            ("cv_m2", "dynamic_dense", "timeout", 0.015),
            ("cv_m4", "dynamic_dense", "success", 0.910),
            ("cv_m4", "dynamic_dense", "collision", 0.070),
            ("cv_m4", "dynamic_dense", "timeout", 0.020),
        ):
            out.append((f"estimate {arm} {cond} {key}", want, ecells[arm][cond][key]))

    # --- zero-margin fallback withheld, Phase 5t -------------------------
    fl = "results/floor_experiment.json"
    if os.path.exists(fl):
        fj = load(fl)
        fc, fcells, fx = fj["checks"], fj["cells"], fj["contrasts"]
        treated = fx["cv_floor_vs_cv_m2"]
        control = fx["oracle_floor_vs_oracle_m2"]["dynamic_dense"]
        remaining = fx["cv_floor_vs_oracle_m2"]["dynamic_dense"]
        identity = [fc[f"identity_{arm}_{cond}"] for arm in ("cv_floor", "oracle_floor")
                    for cond in ("dynamic", "dynamic_dense")]
        out += [
            ("floor decision SYMPTOM", 1.0, float(fj["decision"] == "SYMPTOM")),
            ("floor reproduces 5r", 1.0, float(fc["reproduces_phase_5r"])),
            ("floor identity episodes", 745,
             sum(c["episodes_never_reaching_bare_radius"] for c in identity)),
            ("floor identity broken", 0, sum(c["of_those_not_identical"] for c in identity)),
            ("floor dense gain", 0.010, treated["dynamic_dense"]["success_gain"]),
            ("floor dense p", 0.5, treated["dynamic_dense"]["p"]),
            ("floor dense won", 2, treated["dynamic_dense"]["episodes_won"]),
            ("floor dense lost", 0, treated["dynamic_dense"]["episodes_lost"]),
            ("floor dense ci lo", 0.000, treated["dynamic_dense"]["ci95"][0]),
            ("floor dense ci hi", 0.025, treated["dynamic_dense"]["ci95"][1]),
            ("floor sparse outcomes changed", 0,
             fc["identity_cv_floor_dynamic"]["outcomes_changed"]),
            ("floor control dense gain", -0.005, control["success_gain"]),
            ("floor control ci lo", -0.015, control["ci95"][0]),
            ("floor control ci hi", 0.000, control["ci95"][1]),
            ("floor remaining gap", -0.055, remaining["success_gain"]),
            ("floor remaining p", 0.003, remaining["p"]),
            ("floor remaining won", 1, remaining["episodes_won"]),
            ("floor remaining lost", 12, remaining["episodes_lost"]),
        ]
        for arm, sparse_n, dense_n, dense_s, dense_c in (
            ("oracle_m2", 2, 6, 0.975, 0.010), ("oracle_floor", 2, 6, 0.970, 0.015),
            ("cv_m2", 16, 31, 0.910, 0.075), ("cv_floor", 16, 31, 0.920, 0.065),
        ):
            out += [
                (f"floor {arm} sparse reach bare", sparse_n,
                 fcells[arm]["dynamic"]["episodes_reaching_bare_radius"]),
                (f"floor {arm} dense reach bare", dense_n,
                 fcells[arm]["dynamic_dense"]["episodes_reaching_bare_radius"]),
                (f"floor {arm} dense success", dense_s, fcells[arm]["dynamic_dense"]["success"]),
                (f"floor {arm} dense collision", dense_c,
                 fcells[arm]["dynamic_dense"]["collision"]),
            ]

    # --- replanning on a contradicted estimate, Phase 5u ------------------
    ie = "results/innovation_experiment.json"
    if os.path.exists(ie):
        ij = load(ie)
        ic, icells, ix = ij["checks"], ij["cells"], ij["contrasts"]
        d02, s02 = ix["cv_i02_vs_cv_m2"]["dynamic_dense"], ix["cv_i02_vs_cv_m2"]["dynamic"]
        d05 = ix["cv_i05_vs_cv_m2"]["dynamic_dense"]
        gap, sparse_gap = ix["cv_i02_vs_oracle_m2"]["dynamic_dense"], ix["cv_i02_vs_oracle_m2"]["dynamic"]
        out += [
            ("innovation decision UNRESOLVED", 1.0, float(ij["decision"] == "UNRESOLVED")),
            ("innovation oracle identity", 1.0, float(ic["oracle_identity_on_moving_worlds"])),
            ("innovation reproduces 5r", 1.0, float(ic["reproduces_phase_5r"])),
            ("innovation dense gain", 0.015, d02["success_gain"]),
            ("innovation dense p", 0.549, d02["p"]),
            ("innovation dense won", 7, d02["episodes_won"]),
            ("innovation dense lost", 4, d02["episodes_lost"]),
            ("innovation dense ci lo", -0.015, d02["ci95"][0]),
            ("innovation dense ci hi", 0.050, d02["ci95"][1]),
            ("innovation dense collision delta", -0.020, d02["collision_delta"]),
            ("innovation dense timeout delta", 0.005, d02["timeout_delta"]),
            ("innovation 0.05 dense gain", 0.015, d05["success_gain"]),
            ("innovation 0.05 dense p", 0.453, d05["p"]),
            ("innovation sparse won", 4, s02["episodes_won"]),
            ("innovation sparse lost", 0, s02["episodes_lost"]),
            ("innovation sparse p", 0.125, s02["p"]),
            ("innovation sparse matches oracle, discordant", 0,
             sparse_gap["episodes_won"] + sparse_gap["episodes_lost"]),
            ("innovation remaining dense gap", -0.050, gap["success_gain"]),
            ("innovation remaining dense p", 0.002, gap["p"]),
            ("innovation remaining dense lost", 10, gap["episodes_lost"]),
            ("innovation remaining dense won", 0, gap["episodes_won"]),
            ("innovation dense replans base", 24.9, round(icells["cv_m2"]["dynamic_dense"]["replans_per_episode"], 1)),
            ("innovation dense replans 0.02", 35.7, round(icells["cv_i02"]["dynamic_dense"]["replans_per_episode"], 1)),
            ("innovation dense replans 0.05", 26.5, round(icells["cv_i05"]["dynamic_dense"]["replans_per_episode"], 1)),
            ("innovation dense triggered 0.02", 29.2,
             round(icells["cv_i02"]["dynamic_dense"]["triggered_per_episode"], 1)),
            ("innovation dense triggered 0.05", 6.9,
             round(icells["cv_i05"]["dynamic_dense"]["triggered_per_episode"], 1)),
        ]
        for arm, sparse_s, dense_s, dense_c in (
            ("oracle_m2", 0.995, 0.975, 0.010), ("cv_m2", 0.975, 0.910, 0.075),
            ("cv_i05", 0.960, 0.925, 0.055), ("cv_i02", 0.995, 0.925, 0.055),
        ):
            out += [(f"innovation {arm} sparse success", sparse_s, icells[arm]["dynamic"]["success"]),
                    (f"innovation {arm} dense success", dense_s,
                     icells[arm]["dynamic_dense"]["success"]),
                    (f"innovation {arm} dense collision", dense_c,
                     icells[arm]["dynamic_dense"]["collision"])]

    # --- the repaired map test, Phase 6e -----------------------------------
    rp = "results/mapping_experiment_repaired.json"
    if os.path.exists(rp):
        rj = load(rp)
        out += [("repaired map decision COSTLY", 1.0, float(rj["decision"] == "COSTLY")),
                ("repaired map full reproduces", 1.0,
                 float(rj["checks"]["full_map_reproduces_all"]))]
        # success, scanner / camera / full, per condition, as both documents print.
        table = {
            "sparse": (1.000, 1.000, 1.000), "large": (1.000, 0.990, 0.990),
            "nominal": (1.000, 0.970, 0.980), "noisy_lidar": (1.000, 0.940, 0.930),
            "dense": (0.890, 0.650, 0.690), "narrow": (0.850, 0.590, 0.610),
        }
        for cond, (full, lidar, camera) in table.items():
            cells = rj["conditions"][cond]["cells"]
            out += [(f"repaired {cond} full", full, cells["full_map"]["success"]),
                    (f"repaired {cond} lidar", lidar, cells["mapped_lidar32"]["success"]),
                    (f"repaired {cond} camera", camera, cells["mapped_camera64"]["success"])]
        for cond, gain, coll, tmo in (("dense", -0.240, 0.020, 0.330),
                                      ("narrow", -0.260, 0.020, 0.390)):
            c = rj["conditions"][cond]
            out += [(f"repaired {cond} gain", gain, c["contrasts"]["mapped_lidar32"]["success_gain"]),
                    (f"repaired {cond} collision", coll, c["cells"]["mapped_lidar32"]["collision"]),
                    (f"repaired {cond} timeout", tmo, c["cells"]["mapped_lidar32"]["timeout"]),
                    (f"repaired {cond} harms", 1.0,
                     float(c["contrasts"]["mapped_lidar32"]["verdict"] == "HARMS"))]
        dense_ci = rj["conditions"]["dense"]["contrasts"]["mapped_lidar32"]["ci95"]
        narrow_ci = rj["conditions"]["narrow"]["contrasts"]["mapped_lidar32"]["ci95"]
        noisy = rj["conditions"]["noisy_lidar"]["contrasts"]["mapped_lidar32"]
        out += [
            ("repaired dense ci lo", -0.330, dense_ci[0]),
            ("repaired dense ci hi", -0.150, dense_ci[1]),
            ("repaired narrow ci lo", -0.350, narrow_ci[0]),
            ("repaired narrow ci hi", -0.180, narrow_ci[1]),
            ("repaired noisy gain", -0.060, noisy["success_gain"]),
            ("repaired noisy p", 0.031, noisy["p"]),
            ("repaired nominal worst open loss", -0.030,
             min(rj["conditions"][c]["contrasts"]["mapped_lidar32"]["success_gain"]
                 for c in ("nominal", "sparse", "large"))),
            # The learned columns the documents compare against.
            ("repaired dense ppo", 0.640, rj["conditions"]["dense"]["learned"]["ppo_privileged"]),
            ("repaired narrow ppo", 0.600, rj["conditions"]["narrow"]["learned"]["ppo_privileged"]),
            # SPL, quoted in the Result 1 table.
            ("repaired narrow lidar spl", 0.544,
             rj["conditions"]["narrow"]["cells"]["mapped_lidar32"]["spl"]),
            ("repaired dense lidar spl", 0.602,
             rj["conditions"]["dense"]["cells"]["mapped_lidar32"]["spl"]),
            ("repaired nominal lidar spl", 0.951,
             rj["conditions"]["nominal"]["cells"]["mapped_lidar32"]["spl"]),
            ("repaired sparse lidar spl", 1.000,
             rj["conditions"]["sparse"]["cells"]["mapped_lidar32"]["spl"]),
            ("repaired large lidar spl", 0.977,
             rj["conditions"]["large"]["cells"]["mapped_lidar32"]["spl"]),
        ]

    # --- the pose test, Phase 6f -------------------------------------------
    lp = "results/localisation_experiment.json"
    if os.path.exists(lp):
        lj = load(lp)
        lc = lj["conditions"]
        out += [
            ("pose decision COSTLY", 1.0, float(lj["decision"] == "COSTLY")),
            ("pose matching PAYS", 1.0, float(lj["decision_matching"] == "PAYS")),
            ("pose full reproduces", 1.0, float(lj["checks"]["full_map_reproduces_all"])),
            ("pose reproduces 6e", 1.0, float(lj["checks"]["mapped_reproduces_phase_6e"])),
            ("pose identity control", 1.0,
             float(lj["checks"]["perfect_odometry_is_the_mapped_agent"])),
        ]
        # The table both the report and the plan print.
        table = {
            "sparse": (1.000, 1.000, 0.670, 0.570), "large": (1.000, 0.990, 0.300, 0.530),
            "nominal": (1.000, 0.970, 0.700, 0.850), "noisy_lidar": (1.000, 0.940, 0.640, 0.780),
            "dense": (0.890, 0.650, 0.410, 0.640), "narrow": (0.850, 0.590, 0.420, 0.630),
        }
        for cond, (full, mapped, odom, matched) in table.items():
            cells = lc[cond]["cells"]
            out += [(f"pose {cond} full", full, cells["full_map"]["success"]),
                    (f"pose {cond} mapped", mapped, cells["mapped"]["success"]),
                    (f"pose {cond} odometry", odom, cells["odometry"]["success"]),
                    (f"pose {cond} matched", matched, cells["matched"]["success"])]
        # Odometry alone: HARMS everywhere, and the spread quoted in the prose.
        for cond in table:
            out.append((f"pose {cond} odometry harms", 1.0,
                        float(lc[cond]["contrasts"]["odometry_vs_mapped"]["verdict"] == "HARMS")))
        gains = {c: lc[c]["contrasts"]["odometry_vs_mapped"]["success_gain"] for c in table}
        out += [
            ("pose odometry best case", -0.170, max(gains.values())),
            ("pose odometry worst case", -0.690, min(gains.values())),
            ("pose odometry worst is large", 1.0,
             float(min(gains, key=lambda c: gains[c]) == "large")),
            # "every one at p < 0.001" — pinned as the claim, not as a value.
            ("pose odometry all significant", 1.0,
             float(max(lc[c]["contrasts"]["odometry_vs_mapped"]["p"] for c in table) < 0.001)),
            ("pose odometry collision ceiling", 0.030,
             max(lc[c]["cells"]["odometry"]["collision"] for c in table)),
            ("pose odometry timeout ceiling", 0.680,
             max(lc[c]["cells"]["odometry"]["timeout"] for c in table)),
        ]
        # Matching: MATTERS on five of six, and the residual against an exact pose.
        matters = sum(lc[c]["contrasts"]["matched_vs_odometry"]["verdict"] == "MATTERS"
                      for c in table)
        out.append(("pose matching matters on five", 5, matters))
        for cond, gain in (("dense", 0.230), ("large", 0.230), ("narrow", 0.210),
                           ("nominal", 0.150), ("noisy_lidar", 0.140)):
            out.append((f"pose {cond} matching gain", gain,
                        lc[cond]["contrasts"]["matched_vs_odometry"]["success_gain"]))
        for cond, gain in (("dense", -0.010), ("narrow", 0.040),
                           ("large", -0.460), ("sparse", -0.430)):
            out.append((f"pose {cond} residual", gain,
                        lc[cond]["contrasts"]["matched_vs_mapped"]["success_gain"]))
        # The one clause of prediction 30 that failed.
        sparse = lc["sparse"]["contrasts"]["matched_vs_odometry"]
        out += [("pose sparse matching gain", -0.100, sparse["success_gain"]),
                ("pose sparse matching p", 0.184, sparse["p"])]
        # The mechanism: the pose error itself.
        for cond, odom, matched in (("dense", 0.329, 0.084), ("narrow", 0.393, 0.083)):
            cells = lc[cond]["cells"]
            out += [(f"pose {cond} odometry error", odom,
                     cells["odometry"]["pose_error_median"]),
                    (f"pose {cond} matched error", matched,
                     cells["matched"]["pose_error_median"])]
        # The sensor ordering reversing.
        for cond, gain, pv in (("narrow", -0.100, 0.0063), ("large", 0.140, 0.038)):
            cam = lc[cond]["contrasts"]["matched_cam_vs_matched"]
            out += [(f"pose {cond} camera gain", gain, cam["success_gain"]),
                    (f"pose {cond} camera p", pv, cam["p"])]
        out += [("pose narrow camera", 0.530, lc["narrow"]["cells"]["matched_cam"]["success"]),
                ("pose large camera", 0.670, lc["large"]["cells"]["matched_cam"]["success"]),
                ("pose sparse ppo", 0.980, lc["sparse"]["learned"]["ppo_privileged"]),
                ("pose large ppo", 0.970, lc["large"]["learned"]["ppo_privileged"])]
        # SPL, quoted in the Result 1 table.
        for cond, spl in (("nominal", 0.823), ("sparse", 0.569), ("large", 0.524),
                          ("dense", 0.587), ("narrow", 0.584)):
            out.append((f"pose {cond} matched spl", spl, lc[cond]["cells"]["matched"]["spl"]))

    # --- clutter is indecision, the Phase 6j diagnostic --------------------
    kp = "results/clutter_diagnostic.json"
    if os.path.exists(kp):
        kj = load(kp)["conditions"]
        # The table §9.7 and the plan both print: wandering, driven/shortest,
        # replans, reversals and spread, split by outcome.
        table = {
            ("dense", "successes"): (0.19, 1.15, 38, 1, 2.11),
            ("dense", "failures"): (0.78, 1.91, 151, 46, 1.62),
            ("narrow", "successes"): (0.05, 0.99, 13, 2, 2.38),
            ("narrow", "failures"): (0.79, 1.68, 105, 17, 2.76),
            ("nominal", "successes"): (0.05, 0.99, 16, 3, 2.35),
        }
        for (cond, outcome), (wander, driven, replans, reversals, spread) in table.items():
            e = kj[cond]["as_published"][outcome]
            out += [
                (f"clutter {cond} {outcome} wandering", wander, round(e["wandering"], 2)),
                (f"clutter {cond} {outcome} driven", driven, round(e["driven_over_shortest"], 2)),
                (f"clutter {cond} {outcome} replans", replans, round(e["replans"])),
                (f"clutter {cond} {outcome} reversals", reversals, round(e["reversals"])),
                (f"clutter {cond} {outcome} spread", spread, round(e["gyration"], 2)),
            ]
        # Not stuck: no recoveries and no failed plans anywhere.
        out.append(("clutter no recoveries or failed plans", 0.0,
                    max(kj[c]["as_published"]["all"][k] for c in kj
                        for k in ("recovery_steps", "failed_plans"))))
        # The commitment rule, rejected on val.
        out += [
            ("clutter dense as published", 0.750, kj["dense"]["as_published"]["success"]),
            ("clutter dense committed", 0.667, round(kj["dense"]["committed"]["success"], 3)),
            ("clutter narrow as published", 0.667, round(kj["narrow"]["as_published"]["success"], 3)),
            ("clutter narrow committed", 0.667, round(kj["narrow"]["committed"]["success"], 3)),
            ("clutter refusals low", 4, round(min(kj[c]["committed"]["all"]["plans_refused"]
                                                  for c in ("dense", "narrow")))),
            ("clutter refusals high", 6, round(max(kj[c]["committed"]["all"]["plans_refused"]
                                                   for c in ("dense", "narrow")))),
            ("clutter replans an episode", 50, round(st.mean(
                [kj[c]["committed"]["all"]["replans"] for c in ("dense", "narrow")]) / 10) * 10),
        ]

    # --- frontier planning, the Phase 6k test of §9.7's diagnosis ----------
    fp = "results/frontier_analysis.json"
    if os.path.exists(fp):
        fj = load(fp)
        fc = fj["conditions"]
        # The §9.8 headline table: success before and after, and the contrast.
        for cond, before, after, gain, won, lost in (
                ("dense", 0.68, 0.64, -0.040, 0, 1),
                ("narrow", 0.60, 0.60, 0.000, 1, 1),
                ("nominal", 0.96, 0.96, 0.000, 0, 0)):
            e = fc[cond]
            out += [
                (f"frontier {cond} as published", before, round(e["success"][0], 2)),
                (f"frontier {cond} arm", after, round(e["success"][1], 2)),
                (f"frontier {cond} gain", gain, round(e["gain"], 3)),
                (f"frontier {cond} mcnemar", 1.000, round(e["mcnemar_p"], 3)),
                (f"frontier {cond} won", won, e["won"]),
                (f"frontier {cond} lost", lost, e["lost"]),
            ]
        pooled = fj["clutter_pooled"]
        out += [
            ("frontier clutter as published", 0.64, round(pooled["success"][0], 2)),
            ("frontier clutter arm", 0.62, round(pooled["success"][1], 2)),
            ("frontier clutter gain", -0.020, round(pooled["gain"], 3)),
            ("frontier clutter mcnemar", 1.000, round(pooled["mcnemar_p"], 3)),
            ("frontier clutter won", 1, pooled["won"]),
            ("frontier clutter lost", 2, pooled["lost"]),
            ("frontier clutter ci low", -0.08, round(pooled["ci"][0], 2)),
            ("frontier clutter ci high", 0.04, round(pooled["ci"][1], 2)),
            # The endpoint failed, and the report says so in those words.
            ("frontier endpoint held", 0, int(pooled["registered_held"])),
            ("frontier nominal bound held", 1, int(fc["nominal"]["registered_held"])),
        ]
        # Mechanism, on the episodes both arms solve.
        for cond, both, ra, rb, da, db, va, vb, aa, ab in (
                ("dense", 16, 25, 21, 1.09, 1.07, 2.4, 2.2, 4.34, 3.21),
                ("narrow", 14, 20, 14, 0.98, 0.99, 2.0, 1.9, 4.36, 3.45),
                ("nominal", 24, 22, 20, 1.03, 1.04, 1.8, 1.9, 4.44, 3.38)):
            e = fc[cond]
            out += [
                (f"frontier {cond} both solve", both, e["both"]),
                (f"frontier {cond} replans before", ra, round(e["replans"][0])),
                (f"frontier {cond} replans after", rb, round(e["replans"][1])),
                (f"frontier {cond} driven before", da,
                 round(e["driven_over_shortest"][0], 2)),
                (f"frontier {cond} driven after", db,
                 round(e["driven_over_shortest"][1], 2)),
                (f"frontier {cond} reversals before", va, round(e["reversals"][0], 1)),
                (f"frontier {cond} reversals after", vb, round(e["reversals"][1], 1)),
                (f"frontier {cond} ahead before", aa, round(e["plan_ahead"][0], 2)),
                (f"frontier {cond} ahead after", ab, round(e["plan_ahead"][1], 2)),
            ]
        rep = fj["clutter_replans"]
        out += [
            ("frontier replans pooled before", 23, round(rep["mean"][0])),
            ("frontier replans pooled after", 18, round(rep["mean"][1])),
            ("frontier replan cut", 21.9, round(100 * rep["reduction"], 1)),
            ("frontier replan cut ci low", -1.8, round(100 * rep["ci"][0], 1)),
            ("frontier replan cut ci high", 40.2, round(100 * rep["ci"][1], 1)),
            ("frontier both solve pooled", 30, rep["n"]),
            # In force: the treatment changes what nearly every plan targets.
            ("frontier dense frontier plans", 64, round(fc["dense"]["frontier_plans"])),
            ("frontier dense plans total", 72,
             round(fc["dense"]["frontier_plans"] + fc["dense"]["goal_plans"])),
        ]

    # --- the forensic, Phase 6l: what the clutter failures are -------------
    cp = "results/clutter_forensic.json"
    if os.path.exists(cp):
        cj = load(cp)["conditions"]
        # §9.9's first table: how close it got, where it ended, what it gave
        # back, and how much of the episode it spent not improving.
        for cond, group, best, final, back, stall in (
                ("dense", "arrive", 0.09, 0.09, 0.00, 0.03),
                ("dense", "fail", 5.89, 7.68, 1.79, 0.58),
                ("narrow", "arrive", 0.10, 0.10, 0.00, 0.03),
                ("narrow", "fail", 5.22, 8.87, 3.65, 0.52)):
            e = cj[cond][group]
            out += [
                (f"forensic {cond} {group} best", best, round(e["best_remaining"], 2)),
                (f"forensic {cond} {group} final", final, round(e["final_remaining"], 2)),
                (f"forensic {cond} {group} given back", back, round(e["given_back"], 2)),
                (f"forensic {cond} {group} stall", stall, round(e["stall_fraction"], 2)),
            ]
        # A complete route to the goal, held in every failing episode.
        out += [(f"forensic {c} fail route held", 1.0,
                 round(cj[c]["fail"]["goal_plans_after"], 3)) for c in ("dense", "narrow")]
        # The blockage counts, each with the control that makes it readable.
        for cond, blocked, truth, phantom in (("dense", 23.8, 29.9, 8.8),
                                              ("narrow", 26.3, 42.1, 8.9)):
            e = cj[cond]["fail"]
            out += [
                (f"forensic {cond} map blocks", blocked, round(e["blocked_margin"], 1)),
                (f"forensic {cond} truth blocks", truth,
                 round(e["truth_blocked_margin"], 1)),
                (f"forensic {cond} phantoms", phantom, round(e["phantom_route"], 1)),
            ]

    # --- the worlds themselves: is the scored route drivable at the margin?
    mp = "results/margin_audit.json"
    if os.path.exists(mp):
        mj = load(mp)
        out.append(("audit safety margin", 0.18, round(mj["safety_margin"], 2)))
        for cond, full, half, base, detour, worst in (
                ("nominal", 0, 0, 0, 1.03, 1.24),
                ("dense", 6, 1, 0, 1.09, 1.79),
                ("narrow", 5, 3, 0, 1.17, 1.95)):
            e = mj["conditions"][cond]
            out += [
                (f"audit {cond} no margin route", full, e["no_route_at_full"]),
                (f"audit {cond} no half route", half, e["no_route_at_half"]),
                (f"audit {cond} no base route", base, e["no_route_at_base"]),
                (f"audit {cond} detour", detour, round(e["detour_ratio"], 2)),
                (f"audit {cond} detour worst", worst, round(e["detour_max"], 2)),
            ]

    # --- and the join: those are the worlds that fail ----------------------
    op = "results/margin_overlap.json"
    if os.path.exists(op):
        oj = load(op)
        for cond, safe_n, safe_sr, tight_n, pv, dp in (
                ("dense", 19, 0.89, 6, 0.00016, 5),
                ("narrow", 20, 0.75, 5, 0.0047, 4)):
            e = oj["conditions"][cond]
            out += [
                (f"overlap {cond} safe worlds", safe_n, e["margin_safe"]["n"]),
                (f"overlap {cond} safe success", safe_sr,
                 round(e["margin_safe"]["success"], 2)),
                (f"overlap {cond} tight worlds", tight_n, e["no_margin_safe"]["n"]),
                (f"overlap {cond} tight success", 0.0,
                 round(e["no_margin_safe"]["success"], 2)),
                (f"overlap {cond} fisher", pv, round(e["fisher_p"], dp)),
            ]
        po = oj["pooled"]
        out += [
            ("overlap pooled safe arrive", 32, po["table"][0]),
            ("overlap pooled safe worlds", 39, po["margin_safe"]["n"]),
            ("overlap pooled safe success", 0.82, round(po["margin_safe"]["success"], 2)),
            ("overlap pooled tight worlds", 11, po["no_margin_safe"]["n"]),
            ("overlap pooled tight arrive", 0, po["table"][2]),
            # Quoted in §9.9 as 8.5e-07, so pinned at that resolution.
            ("overlap pooled fisher", 8.5, round(po["fisher_p"] * 1e7, 1)),
            ("overlap share of failures", 0.61, round(oj["share_of_failures_tight"], 2)),
            # The two kinds of failure differ in rate, not in mode.
            ("overlap tight stall", 0.55,
             round(oj["failure_groups"]["tight"]["stall_fraction"], 2)),
            ("overlap roomy stall", 0.54,
             round(oj["failure_groups"]["roomy"]["stall_fraction"], 2)),
            ("overlap tight given back", 2.43,
             round(oj["failure_groups"]["tight"]["given_back"], 2)),
            ("overlap roomy given back", 3.45,
             round(oj["failure_groups"]["roomy"]["given_back"], 2)),
            ("overlap tight failures", 11, oj["failure_groups"]["tight"]["n"]),
            ("overlap roomy failures", 7, oj["failure_groups"]["roomy"]["n"]),
            # The fallback ladder does not reach the radius that would work.
            ("overlap tight full margin", 0.79,
             round(oj["failure_groups"]["tight"]["full_margin_steps"], 2)),
            ("overlap tight lowest radius", 0.35,
             round(oj["failure_groups"]["tight"]["min_plan_radius"], 2)),
            ("overlap tight replans", 145,
             round(oj["failure_groups"]["tight"]["replans"])),
        ]

    # --- the clearance rule, Phase 6m: §9.9's confound, separated ---------
    # Not to be confused with `margin_experiment.json` above, which is Phase
    # 5q's *temporal* margin. These are different experiments.
    xp = "results/clearance_experiment.json"
    if os.path.exists(xp):
        xj = load(xp)
        out.append(("margin stall steps", 50, xj["stall_steps"]))
        for cond, before, after, gain, fired, colls in (
                ("dense", 0.68, 0.64, -0.040, 10, 8),
                ("narrow", 0.60, 0.64, 0.040, 12, 3),
                ("nominal", 0.96, 0.92, -0.040, 3, 1)):
            e = xj["conditions"][cond]
            out += [
                (f"margin {cond} as published", before, round(e["success"][0], 2)),
                (f"margin {cond} relaxed", after, round(e["success"][1], 2)),
                (f"margin {cond} gain", gain, round(e["gain"], 3)),
                (f"margin {cond} mcnemar", 1.000, round(e["mcnemar_p"], 3)),
                (f"margin {cond} fired", fired, e["fired"]),
                (f"margin {cond} collisions before", 0, e["collisions"][0]),
                (f"margin {cond} collisions after", colls, e["collisions"][1]),
            ]
        ep = xj["endpoints"]
        out += [
            # The endpoint, and the nuance that keeps it from being "no effect".
            ("margin recovered", 0, ep["recovered"]),
            ("margin tight worlds", 11, ep["of"]),
            ("margin endpoint held", 0, int(ep["held"])),
            ("margin tight gap before", 6.29, round(ep["tight_gap"][0], 2)),
            ("margin tight gap after", 4.80, round(ep["tight_gap"][1], 2)),
            ("margin tight gap closed", 8, ep["tight_gap_closed"]),
            # The one that held, and the two costs that did not.
            ("margin safe success before", 0.821,
             round(ep["margin_safe_success"][0], 3)),
            ("margin safe success after", 0.821,
             round(ep["margin_safe_success"][1], 3)),
            ("margin safe loss", 0.000, round(ep["margin_safe_loss"], 3)),
            ("margin safe held", 1, int(ep["margin_safe_held"])),
            ("margin clutter collisions", 11, ep["clutter_collisions"]),
            ("margin collisions held", 0, int(ep["collisions_held"])),
            ("margin nominal held", 0, int(ep["nominal_held"])),
            # 12 collisions across all three conditions, against 0 as published.
            ("margin all collisions", 12,
             sum(c["collisions"][1] for c in xj["conditions"].values())),
            ("margin all collisions before", 0,
             sum(c["collisions"][0] for c in xj["conditions"].values())),
        ]

    # --- the error budget, Phase 6n: pose against controller in a gap ------
    gp = "results/gap_diagnostic.json"
    if os.path.exists(gp):
        gj = load(gp)["conditions"]
        # §9.11's table, every row: room at the tightest moment of the episode
        # and the two errors measured against it.
        budget = {
            ("dense", "as_published", "tight"): (6, 0.120, 0.065, 0.039, 5.1, 1.0),
            ("dense", "relaxed", "tight"): (6, 0.017, 0.125, 0.021, 28.3, 3.9),
            ("dense", "relaxed", "collide"): (8, 0.002, 0.118, 0.011, 20.9, 3.9),
            ("dense", "relaxed", "arrive"): (16, 0.256, 0.061, 0.026, 0.9, 0.1),
            ("narrow", "as_published", "tight"): (5, 0.170, 0.067, 0.003, 0.0, 0.0),
            ("narrow", "relaxed", "tight"): (5, 0.025, 0.065, 0.003, 36.0, 2.0),
            ("narrow", "relaxed", "collide"): (3, 0.003, 0.065, 0.036, 18.8, 3.7),
            ("narrow", "relaxed", "arrive"): (16, 0.240, 0.052, 0.016, 3.4, 0.6),
        }
        for (cond, arm, group), (n, room, pe, te, pex, tex) in budget.items():
            e = gj[cond][arm][group]
            tag = f"budget {cond} {arm} {group}"
            out += [
                (f"{tag} n", n, e["n"]),
                (f"{tag} room", room, round(e["room_at_tightest"], 3)),
                (f"{tag} pose err", pe, round(e["pose_err_at_tightest"], 3)),
                (f"{tag} tracking err", te, round(e["track_err_at_tightest"], 3)),
                (f"{tag} pose exceeds", pex, round(100 * e["pose_exceeds_room"], 1)),
                (f"{tag} tracking exceeds", tex,
                 round(100 * e["track_exceeds_room"], 1)),
            ]
        # The claim §9.11 turns on, checked mechanically rather than asserted:
        # at the tightest moment of every group measured -- all sixteen, not
        # just the eight printed -- the controller's own error is the smaller
        # of the two.
        rows = [gj[c][a][g] for c in ("dense", "narrow")
                for a in ("as_published", "relaxed")
                for g in ("tight", "roomy", "arrive", "collide")
                if gj[c][a].get(g)]
        out.append(("budget tracking always under pose", 1.0,
                    float(all(r["track_err_at_tightest"] < r["pose_err_at_tightest"]
                              for r in rows))))
        # The exceedance ordering holds on every row §9.11 prints, and there is
        # exactly one row in the full set where it does not: `narrow` as
        # published on the margin-safe worlds, where the pose error never
        # exceeds the room and the controller does on 0.03% of steps. Pinned as
        # one exception rather than asserted away, because a blanket version of
        # this claim was written first and this check is what refused it.
        printed = [gj[c][a][g] for (c, a, g) in budget]
        out += [
            ("budget pose exceeds more often in the table", 1.0,
             float(all(r["track_exceeds_room"] <= r["pose_exceeds_room"]
                       for r in printed))),
            ("budget exceedance exceptions", 1,
             sum(1 for r in rows if r["track_exceeds_room"] > r["pose_exceeds_room"])),
            ("budget exception tracking share", 0.03,
             round(100 * gj["narrow"]["as_published"]["roomy"]["track_exceeds_room"], 2)),
        ]

    # --- the SLAM back end, Phase 6o -------------------------------------
    bp = "results/backend_experiment.json"
    if os.path.exists(bp):
        bj = load(bp)
        bc = bj["cells"]
        for cond, sensor, before, after, gain, p95a, p95b, closures in (
                ("sparse", "lidar32", 0.44, 0.44, 0.000, 0.637, 0.636, 20.2),
                ("dense", "lidar360", 0.68, 0.68, 0.000, 0.133, 0.089, 13.5),
                ("narrow", "lidar360", 0.60, 0.68, 0.080, 0.106, 0.089, 12.2),
                ("nominal", "lidar360", 0.96, 1.00, 0.040, 0.119, 0.101, 1.6)):
            e = bc[cond]
            out += [
                (f"backend {cond} sensor", 1.0, float(e["sensor"] == sensor)),
                (f"backend {cond} front end", before, round(e["success"][0], 2)),
                (f"backend {cond} back end", after, round(e["success"][1], 2)),
                (f"backend {cond} gain", gain, round(e["gain"], 3)),
                (f"backend {cond} p95 before", p95a, round(e["pose_err_p95"][0], 3)),
                (f"backend {cond} p95 after", p95b, round(e["pose_err_p95"][1], 3)),
                (f"backend {cond} closures", closures, round(e["closures"], 1)),
            ]
        # McNemar and the won/lost counts §9.12 prints.
        out += [
            ("backend narrow mcnemar", 0.500, round(bc["narrow"]["mcnemar_p"], 3)),
            ("backend narrow won", 2, bc["narrow"]["won"]),
            ("backend narrow lost", 0, bc["narrow"]["lost"]),
            ("backend nominal won", 1, bc["nominal"]["won"]),
            ("backend nominal lost", 0, bc["nominal"]["lost"]),
        ]
        ep = bj["endpoints"]
        out += [
            ("backend clutter p95 before", 0.119, round(ep["clutter_pose_p95"][0], 3)),
            ("backend clutter p95 after", 0.089, round(ep["clutter_pose_p95"][1], 3)),
            ("backend clutter reduction", 26,
             round(100 * ep["clutter_pose_reduction"])),
            ("backend tight recovered", 0, ep["tight_recovered"]),
            ("backend tight of", 11, ep["tight_total"]),
            # Two held, two failed, and which.
            ("backend sparse endpoint held", 0, int(ep["sparse_held"])),
            ("backend pose endpoint held", 1, int(ep["pose_held"])),
            ("backend tight endpoint held", 1, int(ep["tight_held"])),
            ("backend nominal endpoint held", 0, int(ep["nominal_held"])),
        ]
        # The claim that it never costs anything: no arm of any cell collides,
        # and no episode is lost anywhere.
        cells = list(bc.values())
        out += [
            ("backend collisions", 0,
             sum(int(r["collision"]) for c in cells for k in ("front_end", "back_end")
                 for r in c[k]["episodes"])),
            ("backend episodes lost", 0, sum(c["lost"] for c in cells)),
            ("backend episodes won", 3, sum(c["won"] for c in cells)),
            ("backend arrived front end", 67,
             sum(int(r["success"]) for c in cells for r in c["front_end"]["episodes"])),
            ("backend arrived back end", 70,
             sum(int(r["success"]) for c in cells for r in c["back_end"]["episodes"])),
            # Sparse is not idle, it is misinformed: solves and total correction.
            ("backend sparse correction", 0.252,
             round(st.mean(r["correction"] for r in
                           bc["sparse"]["back_end"]["episodes"]), 3)),
            ("backend sparse front end error", 0.472,
             round(st.mean(r["pose_err_mean"] for r in
                           bc["sparse"]["front_end"]["episodes"]), 3)),
            ("backend tight p95 dense", 0.128,
             round(bc["dense"]["tight"]["pose_err_p95"], 3)),
            ("backend tight p95 narrow", 0.082,
             round(bc["narrow"]["tight"]["pose_err_p95"], 3)),
        ]

    # --- the back end on the held-out worlds, Phase 6p ----------------------
    tp = "results/backend_test.json"
    if os.path.exists(tp):
        tj = load(tp)
        tc, te = tj["cells"], tj["endpoints"]
        for cond, before, after, gain, won, lost, p95a, p95b in (
                ("sparse", 0.58, 0.60, 0.020, 2, 0, 0.564, 0.582),
                ("dense", 0.59, 0.61, 0.020, 5, 3, 0.127, 0.097),
                ("narrow", 0.60, 0.59, -0.010, 3, 4, 0.139, 0.115),
                ("nominal", 0.96, 0.97, 0.010, 1, 0, 0.097, 0.089),
                ("large", 0.89, 0.91, 0.020, 2, 0, 0.173, 0.140),
                ("noisy_lidar", 0.89, 0.89, 0.000, 1, 1, 0.254, 0.249)):
            e = tc[cond]
            out += [
                (f"held-out back end {cond} front end", before, round(e["success"][0], 2)),
                (f"held-out back end {cond} back end", after, round(e["success"][1], 2)),
                (f"held-out back end {cond} gain", gain, round(e["gain"], 3)),
                (f"held-out back end {cond} won", won, e["won"]),
                (f"held-out back end {cond} lost", lost, e["lost"]),
                (f"held-out back end {cond} p95 before", p95a,
                 round(e["pose_err_p95"][0], 3)),
                (f"held-out back end {cond} p95 after", p95b,
                 round(e["pose_err_p95"][1], 3)),
            ]
        out += [
            ("held-out back end dense mcnemar", 0.727, round(tc["dense"]["mcnemar_p"], 3)),
            # The control: the front end is the published arm, 100/100 in each
            # of the five 360-beam cells.
            ("held-out back end reproduces", 1.0, float(te["reproduces"])),
            ("held-out back end reproduction cells", 5, len(te["reproduction"])),
            ("held-out back end reproduction episodes", 500,
             sum(v["matched"] for v in te["reproduction"].values())),
            # The endpoints.
            ("held-out back end pooled before", 0.752, round(te["pooled_success"][0], 3)),
            ("held-out back end pooled after", 0.762, round(te["pooled_success"][1], 3)),
            ("held-out back end pooled gain", 0.010, round(te["pooled_gain"], 3)),
            ("held-out back end pooled mcnemar", 0.286, round(te["pooled_mcnemar_p"], 3)),
            ("held-out back end pooled won", 14, te["pooled_won"]),
            ("held-out back end pooled lost", 8, te["pooled_lost"]),
            ("held-out back end pooled n", 600, te["pooled_n"]),
            ("held-out back end pooled ci low", -0.005, round(te["pooled_ci"][0], 3)),
            ("held-out back end pooled ci high", 0.025, round(te["pooled_ci"][1], 3)),
            ("held-out back end clutter p95 before", 0.133,
             round(te["clutter_pose_p95"][0], 3)),
            ("held-out back end clutter p95 after", 0.106,
             round(te["clutter_pose_p95"][1], 3)),
            ("held-out back end clutter reduction", 20.2,
             round(100 * te["clutter_pose_reduction"], 1)),
            ("held-out back end collisions front end", 4, te["collisions"][0]),
            ("held-out back end collisions back end", 4, te["collisions"][1]),
            ("held-out back end pooled held", 1, int(te["pooled_held"])),
            ("held-out back end pose held", 1, int(te["pose_held"])),
            ("held-out back end collisions held", 0, int(te["collisions_held"])),
            ("held-out back end sparse held", 1, int(te["sparse_held"])),
            # Inert under noise: closures attempted and refused.
            ("held-out back end noisy closures", 3.0, round(tc["noisy_lidar"]["closures"], 1)),
            ("held-out back end noisy rejected", 2.5, round(tc["noisy_lidar"]["rejected"], 1)),
        ]
        # Rejection rates, quoted in §9.13 and §12 from the unrounded counts.
        for cond, share in (("noisy_lidar", 46), ("dense", 10), ("narrow", 9),
                            ("sparse", 19), ("nominal", 20), ("large", 20)):
            c = tc[cond]
            out.append((f"held-out back end {cond} closures refused %", share,
                        round(100 * c["rejected"] / (c["closures"] + c["rejected"]))))
        # Collisions in different worlds on dense and narrow, the same on noisy.
        for cond, same in (("dense", False), ("narrow", False), ("noisy_lidar", True)):
            fe = {r["seed"] for r in tc[cond]["front_end"]["episodes"] if r["collision"]}
            be = {r["seed"] for r in tc[cond]["back_end"]["episodes"] if r["collision"]}
            out.append((f"held-out back end {cond} collides in the same worlds",
                        float(same), float(fe == be)))

    # --- the noisy map, repaired at its source, Phase 6q --------------------
    dp = "results/clearing_diagnostic.json"
    if os.path.exists(dp):
        dj = load(dp)["conditions"]
        nl = dj["noisy_lidar"]
        for arm, ph, far, recall, lost in (("published", 309, 190, 0.806, 0.270),
                                           ("beam_votes", 288, 186, 0.779, 0.264),
                                           ("noise_margin", 21, 0, 0.754, 0.051)):
            e = nl[arm]
            out += [
                (f"clearing {arm} phantoms", ph, round(e["phantoms"])),
                (f"clearing {arm} beyond half a metre", far,
                 round(e["phantoms_beyond_0.5m"])),
                (f"clearing {arm} recall", recall, round(e["surface_recall"], 3)),
                (f"clearing {arm} floor lost", lost, round(e["floor_lost"], 3)),
            ]
        out += [
            # Every one beyond 0.2 m is gone; the 21 left hug real walls.
            ("clearing margin phantoms within 0.2 m", 21,
             round(nl["noise_margin"]["phantoms_within_0.2m"])),
            ("clearing margin phantoms 0.2 to 0.5 m", 0,
             round(nl["noise_margin"]["phantoms_0.2_to_0.5m"])),
            ("clearing nominal floor lost", 0.047,
             round(dj["nominal"]["published"]["floor_lost"], 3)),
            # Inert without noise, open loop: the occupied set does not move.
            ("clearing nominal occupied", 456, round(dj["nominal"]["published"]["occupied"])),
            ("clearing nominal occupied with margin", 456,
             round(dj["nominal"]["noise_margin"]["occupied"])),
            ("clearing dense occupied", 514, round(dj["dense"]["published"]["occupied"])),
            ("clearing dense occupied with margin", 514,
             round(dj["dense"]["noise_margin"]["occupied"])),
            # The null shadow reproduced the robot's own map on every episode.
            ("clearing null shadow identical", 36,
             sum(int(r["null_shadow_identical"]) for c in dj.values()
                 for r in c["episodes"])),
        ]
    vp = "results/obstacle_range_val100.json"
    if os.path.exists(vp):
        v = load(vp)["cells"]["noisy_lidar"]
        out += [
            ("obstacle range front end", 0.87, round(v["success"][0], 2)),
            ("obstacle range with rule", 0.89, round(v["success"][1], 2)),
            ("obstacle range gain", 0.020, round(v["gain"], 3)),
            ("obstacle range mcnemar", 0.754, round(v["mcnemar_p"], 3)),
            ("obstacle range won", 6, v["won"]),
            ("obstacle range lost", 4, v["lost"]),
            ("obstacle range ci low", -0.04, round(v["ci"][0], 2)),
            ("obstacle range ci high", 0.08, round(v["ci"][1], 2)),
            ("obstacle range collisions before", 2, v["collisions"][0]),
            ("obstacle range collisions after", 0, v["collisions"][1]),
            ("obstacle range replans before", 61, round(v["replans"][0])),
            ("obstacle range replans after", 59, round(v["replans"][1])),
            ("obstacle range pose before", 0.141, round(v["pose_err_median"][0], 3)),
            ("obstacle range pose after", 0.099, round(v["pose_err_median"][1], 3)),
        ]
    vs = "results/obstacle_range_val.json"
    if os.path.exists(vs):
        vn = load(vs)["cells"]
        out += [
            # Closed loop, the noise-free control is identical in every field.
            ("obstacle range nominal identical", 1.0, float(vn["nominal"]["identical"])),
            ("obstacle range nominal pose", 0.049,
             round(vn["nominal"]["pose_err_median"][0], 3)),
            ("obstacle range 25 worlds front end", 0.92,
             round(vn["noisy_lidar"]["success"][0], 2)),
            ("obstacle range 25 worlds with rule", 0.92,
             round(vn["noisy_lidar"]["success"][1], 2)),
        ]

    # --- both repairs together on noisy_lidar, Phase 6r ---------------------
    kp2 = "results/combined_noisy_val.json"
    if os.path.exists(kp2):
        kj2 = load(kp2)
        ka, kc = kj2["arms"], kj2["contrasts"]
        for arm, sr, coll, med, p95 in (("front_end", 0.87, 2, 0.141, 0.277),
                                        ("obstacle_range", 0.89, 0, 0.099, 0.206),
                                        ("back_end", 0.89, 0, 0.138, 0.263),
                                        ("both", 0.89, 0, 0.092, 0.193)):
            e = ka[arm]
            out += [
                (f"combined {arm} success", sr, round(e["success"], 2)),
                (f"combined {arm} collisions", coll, e["collisions"]),
                (f"combined {arm} pose median", med, round(e["pose_err_median"], 3)),
                (f"combined {arm} pose p95", p95, round(e["pose_err_p95"], 3)),
            ]
        out += [
            ("combined each rule alone", 0.020,
             round(kc["obstacle_range_vs_front_end"]["gain"], 3)),
            ("combined back end alone", 0.020, round(kc["back_end_vs_front_end"]["gain"], 3)),
            ("combined both", 0.020, round(kc["both_vs_front_end"]["gain"], 3)),
            ("combined both vs obstacle range", 0.000,
             round(kc["both_vs_obstacle_range"]["gain"], 3)),
            ("combined both vs obstacle range won", 2, kc["both_vs_obstacle_range"]["won"]),
            ("combined both vs obstacle range lost", 2, kc["both_vs_obstacle_range"]["lost"]),
            ("combined interaction", -0.020, round(kj2["interaction"], 3)),
        ]
        # Closures by outcome, back end alone, all 100 worlds.
        be_eps = ka["back_end"]["episodes"]
        failed = [e for e in be_eps if not e["success"]]
        arrived = [e for e in be_eps if e["success"]]
        out += [
            ("combined closures per failure", 27.5,
             round(st.mean(e["closures"] for e in failed), 1)),
            ("combined closures per arrival", 0.8,
             round(st.mean(e["closures"] for e in arrived), 1)),
            ("combined episodes with no closure", 69,
             sum(1 for e in be_eps if e["closures"] == 0)),
        ]
        # The two arms already measured in Phase 6q reproduce it exactly.
        vj = load("results/obstacle_range_val100.json")["cells"]["noisy_lidar"]
        keys = ("success", "collision", "steps", "goal_distance", "pose_err_median", "replans")
        for arm in ("front_end", "obstacle_range"):
            old = {r["seed"]: r for r in vj[arm]["episodes"]}
            new = {r["seed"]: r for r in ka[arm]["episodes"]}
            out.append((f"combined {arm} reproduces phase 6q", 100,
                        sum(all(new[s][k] == old[s][k] for k in keys) for s in old)))
        # What fails, with the obstacle range on.
        orx = vj["obstacle_range"]["episodes"]
        stuck = [e for e in orx if not e["success"] and not e["collision"]
                 and e["steps"] >= 500 and e["goal_distance"] <= 0.8]
        out += [
            ("noisy failures with obstacle range", 11, sum(1 for e in orx if not e["success"])),
            ("noisy stopped just outside", 5, len(stuck)),
            ("noisy stopped just outside pose", 0.47,
             round(st.median(e["pose_err_median"] for e in stuck), 2)),
            ("noisy arrivals pose", 0.09,
             round(st.median(e["pose_err_median"] for e in orx if e["success"]), 2)),
        ]
    np2 = "results/noisy_localisation_diagnostic.json"
    if os.path.exists(np2):
        nj = load(np2)
        fe2, be2, tk = nj["front_end"], nj["back_end"], nj["trimmed_keyframes"]
        out += [
            ("noisy largest rise stuck", 0.011,
             round(max(r["max_step_rise"] for r in fe2["stuck"]), 3)),
            ("noisy largest rise arrived", 0.009,
             round(max(r["max_step_rise"] for r in fe2["arrived"]), 3)),
            ("noisy final error stuck", 0.529,
             round(st.median(r["final_error"] for r in fe2["stuck"]), 3)),
            ("noisy final error arrived", 0.162,
             round(st.median(r["final_error"] for r in fe2["arrived"]), 3)),
            ("noisy zero corrections stuck", 85,
             round(100 * st.median(r["zero_corrections"] for r in fe2["stuck"]))),
            ("noisy zero corrections arrived", 79,
             round(100 * st.median(r["zero_corrections"] for r in fe2["arrived"]))),
            ("closure attempts noisy", 373, be2["noisy_lidar"]["attempts"]),
            ("closures accepted published", 302, be2["noisy_lidar"]["published"]["accepted"]),
            ("closures right published", 299, be2["noisy_lidar"]["published"]["right"]),
            ("closures wrong published", 0, be2["noisy_lidar"]["published"]["wrong"]),
            ("closures accepted noise aware", 348, be2["noisy_lidar"]["noise_aware"]["accepted"]),
            ("closures wrong noise aware", 1, be2["noisy_lidar"]["noise_aware"]["wrong"]),
            # Inert on clean scans by construction.
            ("closures nominal identical", 1.0,
             float(be2["nominal"]["published"] == be2["nominal"]["noise_aware"])),
            ("closures gate full scans", 29,
             round(100 * tk["full_scans"]["share_clearing_gate"])),
            ("closures gate trimmed scans", 14,
             round(100 * tk["trimmed_scans"]["share_clearing_gate"])),
            ("closures gate twelve worlds", 81,
             round(100 * be2["noisy_lidar"]["published"]["accepted"]
                   / be2["noisy_lidar"]["attempts"])),
        ]

    # --- the closures reach far enough, Phase 6s ----------------------------
    rp2 = "results/closure_reach_diagnostic.json"
    if os.path.exists(rp2):
        rj = load(rp2)
        rf = rj["failed"]
        out += [
            ("reach failing episodes", 10, rf["episodes"]),
            ("reach failing closures", 303, rf["closures"]),
            ("reach anchor error", 0.235, round(rf["anchor_error_median"], 3)),
            ("reach newest error", 0.233, round(rf["newest_error_median"], 3)),
            ("reach well localised anchors", 37, round(100 * rf["well_localised_anchor_share"])),
            ("reach drifted anchors", 40, round(100 * rf["drifted_anchor_share"])),
            ("reach solves", 117, rf["solves"]),
            ("reach solve change", -0.012, round(rf["solve_change_median"], 3)),
            ("reach solve change abs", 0.017, round(rf["solve_change_abs_median"], 3)),
            ("reach worlds", 31, len(rj["episodes"])),
            # One of three registered claims held, and which.
            ("reach claim 1 held", 0, int(rj["held"][0])),
            ("reach claim 2 held", 0, int(rj["held"][1])),
            ("reach claim 3 held", 1, int(rj["held"][2])),
        ]
        # It reproduces Phase 6r's back-end arm on every world it drove.
        kj3 = {e["seed"]: e for e in
               load("results/combined_noisy_val.json")["arms"]["back_end"]["episodes"]}
        out.append(("reach reproduces phase 6r", 31,
                    sum(1 for r in rj["episodes"]
                        if r["success"] == kj3[r["seed"]]["success"]
                        and len(r["closures"]) == kj3[r["seed"]]["closures"])))

    # --- the closures agree with the drift, Phase 6t -------------------------
    sp2 = "results/closure_resolve_diagnostic.json"
    if os.path.exists(sp2):
        sj = load(sp2)
        sm, sc, sr = sj["median"], sj["cut"], sj["residuals"]
        out += [
            ("resolve episodes", 10, len(sj["episodes"])),
            ("resolve robot final", 0.388, round(sm["robot_final_error"], 3)),
            ("resolve unsolved final", 0.388, round(sm["unsolved_final_error"], 3)),
            ("resolve published final", 0.388, round(sm["published_final_error"], 3)),
            ("resolve weights up final", 0.388, round(sm["weights_up_final_error"], 3)),
            ("resolve good anchors final", 0.408, round(sm["good_anchors_final_error"], 3)),
            ("resolve weights up cut", 0, round(100 * sc["weights_up"])),
            ("resolve good anchors cut", -5, round(100 * sc["good_anchors"])),
            ("resolve control", 0.000, round(sj["control"], 3)),
            ("resolve claim 1 held", 0, int(sj["held"][0])),
            ("resolve claim 2 held", 1, int(sj["held"][1])),
            # The exploratory part, from the same saved graphs.
            ("resolve closures", 303, sr["closures"]),
            ("resolve against truth", 0.048, round(sr["against_truth_median"], 3)),
            ("resolve against chain", 0.054, round(sr["residual_median"], 3)),
            ("resolve disagreeing closures", 10, round(100 * sr["residual_above_0.10"])),
            ("resolve pair drift", 0.028, round(sr["pair_drift_median"], 3)),
            ("resolve anchor absolute", 0.235, round(sr["anchor_error_median"], 3)),
            ("resolve newest absolute", 0.233, round(sr["newest_error_median"], 3)),
            ("resolve good anchor closures", 113, sr["good_anchor_closures"]),
            ("resolve good anchor newest", 0.073, round(sr["good_anchor_newest_median"], 3)),
            ("resolve largest move median", 0.067, round(sj["largest_move_median"], 3)),
            ("resolve largest move max", 0.115, round(sj["largest_move_max"], 3)),
        ]

    # --- the motion prior is not the lever, Phase 6u ------------------------
    pp2 = "results/prior_binding_diagnostic.json"
    if os.path.exists(pp2):
        pj = load(pp2)
        for key, binds, better, pred, pub, nop in (
                ("nominal", 46, 55, 0.039, 0.038, 0.036),
                ("noisy", 87, 49, 0.210, 0.209, 0.213),
                ("noisy_failed", 86, 58, 0.318, 0.318, 0.331),
                ("noisy_arrived", 91, 15, 0.123, 0.125, 0.147)):
            e = pj[key]
            out += [
                (f"prior {key} binds", binds, round(100 * e["binding_share"])),
                (f"prior {key} overridden better", better,
                 round(100 * e["no_prior_better_when_binding"])),
                (f"prior {key} predicted error", pred, round(e["error_predicted"], 3)),
                (f"prior {key} published error", pub, round(e["error_published"], 3)),
                (f"prior {key} no-prior error", nop, round(e["error_no_prior"], 3)),
            ]
        out += [
            ("prior binding ratio", 1.9, round(pj["binding_ratio"], 1)),
            ("prior claim 1 held", 0, int(pj["held"][0])),
            ("prior claim 2 held", 0, int(pj["held"][1])),
            ("prior claim 3 held", 1, int(pj["held"][2])),
            ("prior failed overall better", 57,
             round(100 * pj["noisy_failed"]["no_prior_better_overall"])),
            ("prior arrived overall better", 16,
             round(100 * pj["noisy_arrived"]["no_prior_better_overall"])),
        ]
    # The scan matcher's window against slam_toolbox's, from the configuration
    # Nav2 was actually run with and the class this stack actually uses.
    sy = "ros2_bridge/slam_params.yaml"
    if os.path.exists(sy):
        from vision_nav.mapping.localisation import ScanMatchConfig

        with open(sy, encoding="utf-8") as f:
            text = f.read()

        def yaml_value(name):
            return float(re.search(rf"^\s*{name}:\s*([0-9.]+)", text, re.M).group(1))

        smc = ScanMatchConfig()
        out += [
            ("slam correlation window", 0.5, yaml_value("correlation_search_space_dimension")),
            ("slam heading search", 0.349, yaml_value("coarse_search_angle_offset")),
            ("slam scan buffer", 10, yaml_value("scan_buffer_size")),
            ("slam loop search", 8.0, yaml_value("loop_search_space_dimension")),
            ("stack window", 0.08, smc.window),
            ("stack heading window", 0.024, smc.angular_window),
            ("heading search ratio", 15,
             round(yaml_value("coarse_search_angle_offset") / smc.angular_window)),
        ]

    # --- slam_toolbox's front end rebuilt, Phase 6v --------------------------
    fv = "results/frontend_val.json"
    if os.path.exists(fv):
        fvj = load(fv)["cells"]
        for cell, sr0, sr1, won, lost, med0, med1, fin0, fin1 in (
                ("noisy_lidar", 0.87, 0.83, 9, 13, 0.141, 0.113, 0.251, 0.252),
                ("nominal", 0.95, 0.99, 4, 0, 0.047, 0.060, 0.068, 0.120),
                ("dense", 0.72, 0.73, 4, 3, 0.052, 0.079, 0.078, 0.163)):
            e = fvj[cell]
            out += [
                (f"frontend {cell} map success", sr0, round(e["success"][0], 2)),
                (f"frontend {cell} correlative success", sr1, round(e["success"][1], 2)),
                (f"frontend {cell} won", won, e["won"]),
                (f"frontend {cell} lost", lost, e["lost"]),
                (f"frontend {cell} pose median map", med0, round(e["pose_err_median"][0], 3)),
                (f"frontend {cell} pose median correlative", med1,
                 round(e["pose_err_median"][1], 3)),
                (f"frontend {cell} pose final map", fin0, round(e["pose_err_final"][0], 3)),
                (f"frontend {cell} pose final correlative", fin1,
                 round(e["pose_err_final"][1], 3)),
            ]
        out += [
            ("frontend noisy mcnemar", 0.52, round(fvj["noisy_lidar"]["mcnemar_p"], 2)),
            ("frontend nominal mcnemar", 0.125, round(fvj["nominal"]["mcnemar_p"], 3)),
        ]
    # Is slam_toolbox's pose better? From the committed held-out results.
    nl = "results/nav2_slam_runs/noisy_lidar__nav2_slam.json"
    bt = "results/backend_test.json"
    if os.path.exists(nl) and os.path.exists(bt):
        nlj = load(nl)
        mine = load(bt)["cells"]["noisy_lidar"]["front_end"]["episodes"]
        by_seed = {e["seed"]: e for e in mine}
        per = nlj["per_episode"]
        both = [i for i, p in enumerate(per)
                if p["success"] and by_seed.get(p["world_seed"], {}).get("success")]
        a = [nlj["pose_error_per_episode"][i] for i in both]
        b = [by_seed[per[i]["world_seed"]]["pose_err_final"] for i in both]
        out += [
            ("nav2 slam noisy success", 1.00, round(nlj["success_rate"], 2)),
            ("nav2 slam noisy pose median", 0.075, round(nlj["pose_error_median"], 3)),
            ("nav2 slam noisy pose max", 0.294, round(nlj["pose_error_max"], 3)),
            ("stack noisy pose median", 0.238,
             round(st.median(e["pose_err_final"] for e in mine), 3)),
            ("stack noisy pose max", 1.043, round(max(e["pose_err_final"] for e in mine), 3)),
            ("both arrived noisy", 89, len(both)),
            ("both arrived nav2 pose", 0.074, round(st.median(a), 3)),
            ("both arrived stack pose", 0.228, round(st.median(b), 3)),
            ("both arrived stack worse", 90,
             round(100 * sum(1 for x, y in zip(a, b, strict=True) if y > x) / len(both))),
            ("both arrived nav2 steps", 140,
             round(st.median(per[i]["steps"] for i in both))),
            ("both arrived stack steps", 178,
             round(st.median(by_seed[per[i]["world_seed"]]["steps"] for i in both))),
        ]
    # Graph density: slam_toolbox's linking against this back end's closures.
    if os.path.exists(sy):
        from vision_nav.mapping.posegraph import PoseGraphConfig

        pgc = PoseGraphConfig()
        out += [
            ("slam link distance", 1.5, yaml_value("link_scan_maximum_distance")),
            ("slam link gate", 0.1, yaml_value("link_match_minimum_response_fine")),
            ("stack closure gap", 6, pgc.loop_min_gap),
            ("stack closure gate", 0.55, pgc.loop_min_score),
        ]

    # Every failure in that run is a timeout, not a crash -- §9.8 and §12.
    fd = "results/frontier_diagnostic.json"
    if os.path.exists(fd):
        eps = [e for cond in load(fd)["conditions"].values() for arm in cond.values()
               for e in arm["episodes"] if not e["success"]]
        clutter = [e for cond, c in load(fd)["conditions"].items() if cond != "nominal"
                   for arm in c.values() for e in arm["episodes"] if not e["success"]]
        out += [
            ("frontier failures", 39, len(eps)),
            ("frontier failure collisions", 0, sum(int(e["collision"]) for e in eps)),
            ("frontier failures all timeout", 500, min(e["steps"] for e in eps)),
            ("frontier failure reached best", 0.96,
             round(max(e["reached"] for e in clutter), 2)),
        ]

    # --- commitment gated on time, rejected on forty val worlds ------------
    c40 = "results/clutter_commitment_val40.json"
    if os.path.exists(c40):
        fj = load(c40)
        f = fj["conditions"]
        out += [
            ("commitment40 episodes", 40, fj["episodes"]),
            ("commitment40 dense as published", 0.725, f["dense"]["as_published"]["success"]),
            ("commitment40 dense committed", 0.650, f["dense"]["committed_interval"]["success"]),
            ("commitment40 narrow as published", 0.600, f["narrow"]["as_published"]["success"]),
            ("commitment40 narrow committed", 0.600, f["narrow"]["committed_interval"]["success"]),
            # The gate binds: refusals an episode, and the symptom that does not move.
            ("commitment40 refusals low", 34,
             round(min(f[c]["committed_interval"]["all"]["plans_refused"] for c in f))),
            ("commitment40 refusals high", 49,
             round(max(f[c]["committed_interval"]["all"]["plans_refused"] for c in f))),
            ("commitment40 dense wandering before", 0.27,
             round(f["dense"]["as_published"]["all"]["wandering"], 2)),
            ("commitment40 dense wandering after", 0.28,
             round(f["dense"]["committed_interval"]["all"]["wandering"], 2)),
            ("commitment40 narrow wandering before", 0.34,
             round(f["narrow"]["as_published"]["all"]["wandering"], 2)),
            ("commitment40 narrow wandering after", 0.33,
             round(f["narrow"]["committed_interval"]["all"]["wandering"], 2)),
        ]

    # --- the corroboration rule, Phase 6i ----------------------------------
    cp = "results/corroboration_experiment.json"
    if os.path.exists(cp):
        cj = load(cp)
        cc, cpool = cj["conditions"], cj["pooled"]
        out += [("corroboration decision UNRESOLVED", 1.0,
                 float(cj["decision"] == "UNRESOLVED"))]
        out += [(f"corroboration control {k}", 1.0, float(v)) for k, v in cj["checks"].items()]
        table = {
            "sparse": (1.000, 1.000, 0.970, 0.990), "large": (0.990, 0.990, 0.900, 0.890),
            "nominal": (0.970, 0.960, 0.960, 0.960), "noisy_lidar": (0.830, 0.910, 0.750, 0.890),
            "dense": (0.640, 0.620, 0.620, 0.590), "narrow": (0.590, 0.590, 0.590, 0.600),
        }
        arms = ("mapped", "mapped_corroborated", "matched", "matched_corroborated")
        for cond, row in table.items():
            s = cc[cond]["success"]
            out += [(f"corroboration {cond} {a}", v, s[a]) for a, v in zip(arms, row, strict=True)]
        noisy = cc["noisy_lidar"]["contrasts"]
        out += [
            ("corroboration noisy own-map gain", 0.080, noisy["mapped"]["gain"]),
            ("corroboration noisy own-map p", 0.0574, noisy["mapped"]["p"]),
            ("corroboration noisy own-map won", 11, noisy["mapped"]["won"]),
            ("corroboration noisy own-map lost", 3, noisy["mapped"]["lost"]),
            ("corroboration noisy own-pose gain", 0.140, noisy["matched"]["gain"]),
            ("corroboration noisy own-pose p", 0.0005, round(noisy["matched"]["p"], 4)),
            ("corroboration noisy own-pose won", 15, noisy["matched"]["won"]),
            ("corroboration noisy own-pose lost", 1, noisy["matched"]["lost"]),
            ("corroboration noisy own-pose MATTERS", 1.0,
             float(noisy["matched"]["verdict"] == "MATTERS")),
            ("corroboration pooled own-map", 0.008, cpool["mapped"]["gain"]),
            ("corroboration pooled own-map ci lo", -0.007, cpool["mapped"]["ci95"][0]),
            ("corroboration pooled own-map ci hi", 0.023, cpool["mapped"]["ci95"][1]),
            ("corroboration pooled own-pose", 0.022, cpool["matched"]["gain"]),
            ("corroboration pooled own-pose ci lo", 0.002, cpool["matched"]["ci95"][0]),
            ("corroboration pooled own-pose ci hi", 0.042, cpool["matched"]["ci95"][1]),
            ("corroboration collision ceiling", 0.02,
             max(cc[c]["collision"][a] for c in table for a in arms)),
            ("corroboration noisy pose error off", 0.217,
             cc["noisy_lidar"]["pose_error_median"]["off"]),
            ("corroboration noisy pose error on", 0.238,
             cc["noisy_lidar"]["pose_error_median"]["on"]),
        ]
        # The difference in costs against Nav2, before and after the repair.
        passes = list(cpool["did_vs_nav2_matched"])
        for i, (before, after, lo, hi) in enumerate(((0.112, 0.090, 0.058, 0.123),
                                                     (0.120, 0.098, 0.065, 0.133))):
            b = cpool["did_vs_nav2_matched"][passes[i]]
            a = cpool["did_vs_nav2_matched_corroborated"][passes[i]]
            out += [(f"corroboration did before pass{i + 1}", before, b["did"]),
                    (f"corroboration did after pass{i + 1}", after, a["did"]),
                    (f"corroboration did after pass{i + 1} ci lo", lo, a["ci95"][0]),
                    (f"corroboration did after pass{i + 1} ci hi", hi, a["ci95"][1]),
                    (f"corroboration did after pass{i + 1} unresolved", 1.0,
                     float(a["class"] == "UNRESOLVED"))]

    # --- what a dense scan does to the map, the Phase 6i diagnostic -------
    dm = "results/dense_map_diagnostic.json"
    if os.path.exists(dm):
        dj = load(dm)["conditions"]
        noisy = dj["noisy_lidar"]
        out += [
            ("dense map noisy phantoms 32", 104, round(noisy["lidar32"]["phantom_cells"])),
            ("dense map noisy phantoms 360", 370, round(noisy["lidar360"]["phantom_cells"])),
            ("dense map noisy blocked 360", 0.46, round(noisy["lidar360"]["blocked_fraction_mapped"], 2)),
            ("dense map noisy blocked true", 0.38, round(noisy["lidar360"]["blocked_fraction_true"], 2)),
            ("dense map noisy floor lost 360", 0.28, round(noisy["lidar360"]["free_floor_lost"], 2)),
            # No phantoms in clutter, at either beam count, and a map that
            # blocks less floor than the truth: the clutter gap is not the map.
            ("dense map clutter has no phantoms", 0.0,
             max(dj[c][s]["phantom_cells"] for c in ("dense", "narrow") for s in ("lidar32", "lidar360"))),
            ("dense map clutter blocks less than the truth", 1.0,
             float(all(dj[c][s]["blocked_fraction_mapped"] < dj[c][s]["blocked_fraction_true"]
                       for c in ("dense", "narrow") for s in ("lidar32", "lidar360")))),
            ("dense map dense replans 360", 49, round(dj["dense"]["lidar360"]["replans"])),
            # The repair, as the plan reports it from val.
            ("dense map noisy repaired success", 0.92,
             round(noisy["lidar360_corroborated"]["success"], 2)),
            ("dense map noisy repaired phantoms", 309,
             round(noisy["lidar360_corroborated"]["phantom_cells"])),
            ("dense map noisy repaired replans", 91,
             round(noisy["lidar360_corroborated"]["replans"])),
        ]

    # --- the missing cell: this stack at 360 beams, Phase 6h ---------------
    hp = "results/sensor_experiment.json"
    if os.path.exists(hp):
        hj = load(hp)
        hc, hpool = hj["conditions"], hj["pooled"]
        hpasses = list(hpool["did_vs_nav2_360"])
        out += [("sensor decision IMPLEMENTATION", 1.0, float(hj["decision"] == "IMPLEMENTATION"))]
        out += [(f"sensor control {k}", 1.0, float(v)) for k, v in hj["checks"].items()]
        table = {
            "sparse": (1.000, 1.000, 0.570, 1.000, 0.970, 0.960),
            "large": (1.000, 0.990, 0.530, 0.990, 0.900, 0.960),
            "nominal": (1.000, 0.970, 0.850, 0.970, 0.960, 0.970),
            "noisy_lidar": (1.000, 0.940, 0.780, 0.830, 0.750, 1.000),
            "dense": (0.890, 0.650, 0.640, 0.640, 0.620, 0.820),
            "narrow": (0.850, 0.590, 0.630, 0.590, 0.590, 0.810),
        }
        arms = ("full_map", "mapped32", "matched32", "mapped360", "matched360")
        for cond, row in table.items():
            s = hc[cond]["success"]
            out += [(f"sensor {cond} {arm}", v, s[arm]) for arm, v in zip(arms, row[:5], strict=True)]
            out.append((f"sensor {cond} nav2 slam360", row[5], hc[cond]["nav2"]["slam360"]))
        for i, (did, lo, hi) in enumerate(((0.112, 0.077, 0.147), (0.120, 0.085, 0.155))):
            v = hpool["did_vs_nav2_360"][hpasses[i]]
            out += [(f"sensor did pass{i + 1}", did, v["did"]),
                    (f"sensor did pass{i + 1} ci lo", lo, v["ci95"][0]),
                    (f"sensor did pass{i + 1} ci hi", hi, v["ci95"][1])]
        for key, val in (("cost32", -0.290), ("cost360", -0.158), ("sensor_effect", 0.132),
                         ("map_cost32", -0.100), ("map_cost360", -0.120),
                         ("pose_cost32", -0.190), ("pose_cost360", -0.038)):
            out.append((f"sensor pooled {key}", val, hpool[key]["gain"]))
        out += [("sensor effect ci lo", 0.097, hpool["sensor_effect"]["ci95"][0]),
                ("sensor effect ci hi", 0.167, hpool["sensor_effect"]["ci95"][1])]
        # Pose error, as §9.5 quotes it.
        for cond, e32, e360 in (("sparse", 0.382, 0.084), ("large", 0.439, 0.113)):
            pe = hc[cond]["pose_error_median"]
            out += [(f"sensor {cond} pose error 32", e32, pe["32"]),
                    (f"sensor {cond} pose error 360", e360, pe["360"])]
        pe360 = [hc[c]["pose_error_median"]["360"] for c in hc]
        out += [("sensor pose error 360 low", 0.07, round(min(pe360), 2)),
                ("sensor pose error 360 high", 0.22, round(max(pe360), 2))]
        if os.path.exists("results/nav2_slam_comparison.json"):
            nc = load("results/nav2_slam_comparison.json")["conditions"]
            npe = [nc[c]["nav2_without"]["360"]["pose_error_median"] for c in nc]
            out += [("sensor nav2 pose error low", 0.07, round(min(npe), 2)),
                    ("sensor nav2 pose error high", 0.17, round(max(npe), 2))]
        # Where the residual is: per-condition differences in costs.
        per = [hpool["did_vs_nav2_360"][p]["per_condition"] for p in hpasses]
        out += [("sensor sparse residual", 0.000, max(abs(x["sparse"]) for x in per)),
                ("sensor nominal residual low", 0.03, min(x["nominal"] for x in per)),
                ("sensor nominal residual high", 0.04, max(x["nominal"] for x in per)),
                ("sensor large residual", 0.07, per[0]["large"]),
                ("sensor noise residual low", 0.27, min(x["noisy_lidar"] for x in per)),
                ("sensor noise residual high", 0.28, max(x["noisy_lidar"] for x in per)),
                ("sensor clutter residual low", 0.14, min(min(x["dense"], x["narrow"]) for x in per)),
                ("sensor clutter residual high", 0.18, max(max(x["dense"], x["narrow"]) for x in per))]
        shares = [(x["noisy_lidar"] + x["dense"] + x["narrow"]) / sum(x.values()) for x in per]
        # Quoted as whole percentages, so pinned at two decimals.
        out.append(("sensor noise and clutter share of residual", 0.85, round(st.mean(shares), 2)))
        noise_share = [x["noisy_lidar"] / sum(x.values()) for x in per]
        clutter_share = [(x["dense"] + x["narrow"]) / sum(x.values()) for x in per]
        out += [("sensor noise share low", 0.38, round(min(noise_share), 2)),
                ("sensor noise share high", 0.42, round(max(noise_share), 2)),
                ("sensor clutter share low", 0.43, round(min(clutter_share), 2)),
                ("sensor clutter share high", 0.47, round(max(clutter_share), 2))]

    # --- a production stack asked the same question, Phase 6g --------------
    sp = "results/nav2_slam_comparison.json"
    if os.path.exists(sp):
        sj = load(sp)
        sc = sj["conditions"]
        passes = list(sc["sparse"]["nav2_with"])
        out += [("slam decision UNRESOLVED", 1.0, float(sj["decision"] == "UNRESOLVED")),
                ("slam decision 360 IMPLEMENTATION", 1.0,
                 float(sj["decision_360"] == "IMPLEMENTATION"))]
        # The table the report and the plan print: hand-written with / without,
        # Nav2 with (two passes), Nav2 + SLAM at 360 and at 32.
        table = {
            "sparse": (1.000, 0.570, 0.990, 0.990, 0.960, 0.690),
            "large": (1.000, 0.530, 0.990, 0.990, 0.960, 0.400),
            "nominal": (1.000, 0.850, 0.980, 0.970, 0.970, 0.800),
            "noisy_lidar": (1.000, 0.780, 0.970, 0.980, 1.000, 0.750),
            "dense": (0.890, 0.640, 0.940, 0.910, 0.820, 0.620),
            "narrow": (0.850, 0.630, 0.930, 0.910, 0.810, 0.450),
        }
        for cond, (hw_w, hw_wo, n1, n2, s360, s32) in table.items():
            e = sc[cond]
            out += [(f"slam {cond} hw with", hw_w, e["handwritten"]["with"]),
                    (f"slam {cond} hw without", hw_wo, e["handwritten"]["without"]),
                    (f"slam {cond} nav2 with pass1", n1, e["nav2_with"][passes[0]]),
                    (f"slam {cond} nav2 with pass2", n2, e["nav2_with"][passes[1]]),
                    (f"slam {cond} 360", s360, e["nav2_without"]["360"]["success"]),
                    (f"slam {cond} 32", s32, e["nav2_without"]["32"]["success"])]
        p360, p32 = sj["pooled"]["360"], sj["pooled"]["32"]
        for label, pool, rows in (
            ("360", p360, ((-0.047, 0.243, 0.203, 0.285), (-0.038, 0.252, 0.212, 0.292))),
            ("32", p32, ((-0.348, -0.058, -0.108, -0.008), (-0.340, -0.050, -0.100, 0.000))),
        ):
            for i, (cost, did, lo, hi) in enumerate(rows):
                v = pool[passes[i]]
                out += [(f"slam {label} pass{i + 1} nav2 cost", cost, v["nav2_cost"]),
                        (f"slam {label} pass{i + 1} did", did, v["did"]),
                        (f"slam {label} pass{i + 1} ci lo", lo, v["ci95"][0]),
                        (f"slam {label} pass{i + 1} ci hi", hi, v["ci95"][1])]
        out.append(("slam hand-written pooled cost", -0.290, p360[passes[0]]["handwritten_cost"]))
        # "a seventh of it": Nav2's pooled cost at 360 over the hand-written one.
        ratio = st.mean([p360[p]["nav2_cost"] for p in passes]) / p360[passes[0]]["handwritten_cost"]
        out.append(("slam 360 cost is about a seventh", 1.0, float(1 / 8 < ratio < 1 / 6)))
        # Clutter at 360: timeouts, not collisions, with pose error under 8 cm.
        for cond, to, coll in (("dense", 0.140, 0.040), ("narrow", 0.180, 0.010)):
            w = sc[cond]["nav2_without"]["360"]
            out += [(f"slam {cond} 360 timeout", to, w["timeout"]),
                    (f"slam {cond} 360 collision", coll, w["collision"]),
                    (f"slam {cond} 360 pose error under 8 cm", 1.0,
                     float(w["pose_error_median"] < 0.08))]
        gaps = [sc[c]["nav2_with"][p] - sc[c]["nav2_without"]["360"]["success"]
                for c in ("dense", "narrow") for p in passes]
        out += [("slam clutter cost low", 0.09, min(gaps)), ("slam clutter cost high", 0.12, max(gaps))]
        # Tight corridors at 32 beams, and sparse.
        d = sc["narrow"]["direct"]["32"]
        w = sc["narrow"]["nav2_without"]["32"]
        out += [("slam narrow 32 direct", -0.180, d["gain"]), ("slam narrow 32 p", 0.0039, d["p"]),
                ("slam narrow 32 collision", 0.180, w["collision"]),
                ("slam narrow 32 timeout", 0.370, w["timeout"])]
        # Prediction 31's failed clauses, as the plan states them.
        out += [("slam dense 360 gap to pass 2", 0.09,
                 sc["dense"]["nav2_with"][passes[1]] - sc["dense"]["nav2_without"]["360"]["success"])]

    # --- why matching hurts where it hurts, Phase 6f -----------------------
    ld = "results/localisation_diagnostic.json"
    if os.path.exists(ld):
        dj = load(ld)["conditions"]
        out += [
            ("pose diag sparse score", 0.925, dj["sparse"]["score_median"]),
            ("pose diag sparse error", 0.570, dj["sparse"]["pose_error_median"]),
            ("pose diag sparse error when confident", 0.441,
             dj["sparse"]["pose_error_median_when_confident"]),
            ("pose diag dense score", 0.965, dj["dense"]["score_median"]),
            ("pose diag dense error", 0.058, dj["dense"]["pose_error_median"]),
        ]

    # --- the map test as registered, Phase 6d ------------------------------
    mp, md = "results/mapping_experiment.json", "results/mapping_diagnostic.json"
    if os.path.exists(mp):
        mj = load(mp)
        out += [("map decision COSTLY", 1.0, float(mj["decision"] == "COSTLY")),
                ("map full reproduces", 1.0, float(mj["checks"]["full_map_reproduces_all"]))]
        table = {
            "sparse": (1.000, 1.000, 1.000), "nominal": (1.000, 0.990, 0.960),
            "large": (1.000, 0.980, 0.990), "dense": (0.890, 0.610, 0.660),
            "narrow": (0.850, 0.540, 0.590), "noisy_lidar": (1.000, 0.540, 0.520),
        }
        for cond, (full, lidar, camera) in table.items():
            cells = mj["conditions"][cond]["cells"]
            out += [(f"map {cond} full", full, cells["full_map"]["success"]),
                    (f"map {cond} lidar", lidar, cells["mapped_lidar32"]["success"]),
                    (f"map {cond} camera", camera, cells["mapped_camera64"]["success"])]
    if os.path.exists(md):
        dj = load(md)
        keys = ("dense/lidar32", "dense/camera64", "narrow/lidar32", "narrow/camera64")
        out += [
            ("map diag collisions", 79, sum(dj[k]["collisions"] for k in keys)),
            ("map diag mapped a second before", 79,
             sum(dj[k]["collisions_mapped_1s_before"] for k in keys)),
            ("map diag mapped in the last half second", 0,
             sum(dj[k]["collisions_mapped_last_0p5s"] for k in keys)),
            ("map diag median scans low", 63, min(dj[k]["median_scans_mapped_before_contact"]
                                                  for k in keys)),
            ("map diag median scans high", 121, max(dj[k]["median_scans_mapped_before_contact"]
                                                    for k in keys)),
        ]

    # --- observing only what a sensor could see, Phase 6c -----------------
    oc = "results/occlusion_experiment.json"
    if os.path.exists(oc):
        vj = load(oc)
        vcells, vx = vj["cells"], vj["contrasts"]
        lid, cam = vx["lidar_vs_all_round"], vx["camera_vs_all_round"]
        out += [
            ("occlusion decision UNRESOLVED", 1.0, float(vj["decision"] == "UNRESOLVED")),
            ("occlusion reproduces 6a", 1.0, float(vj["checks"]["reproduces_phase_6a"])),
            ("occlusion lidar dense", -0.035, lid["dynamic_dense"]["success_gain"]),
            ("occlusion lidar dense p", 0.092, lid["dynamic_dense"]["p"]),
            ("occlusion lidar dense won", 3, lid["dynamic_dense"]["episodes_won"]),
            ("occlusion lidar dense lost", 10, lid["dynamic_dense"]["episodes_lost"]),
            ("occlusion lidar dense ci lo", -0.070, lid["dynamic_dense"]["ci95"][0]),
            ("occlusion lidar dense ci hi", 0.000, lid["dynamic_dense"]["ci95"][1]),
            ("occlusion lidar sparse", -0.015, lid["dynamic"]["success_gain"]),
            ("occlusion camera dense", -0.125, cam["dynamic_dense"]["success_gain"]),
            ("occlusion camera dense won", 1, cam["dynamic_dense"]["episodes_won"]),
            ("occlusion camera dense lost", 26, cam["dynamic_dense"]["episodes_lost"]),
            ("occlusion camera sparse", -0.085, cam["dynamic"]["success_gain"]),
            ("occlusion camera sparse p", 0.0002, cam["dynamic"]["p"]),
            ("occlusion lidar vs line dense", 0.110,
             vx["lidar_vs_line@0.01"]["dynamic_dense"]["success_gain"]),
            ("occlusion lidar vs line dense p", 0.0001, vx["lidar_vs_line@0.01"]["dynamic_dense"]["p"]),
            ("occlusion camera vs line dense", 0.020,
             vx["camera_vs_line@0.01"]["dynamic_dense"]["success_gain"]),
            ("occlusion camera vs line dense p", 0.61,
             round(vx["camera_vs_line@0.01"]["dynamic_dense"]["p"], 2)),
            ("occlusion lidar vs oracle dense", -0.050,
             vx["lidar_vs_oracle"]["dynamic_dense"]["success_gain"]),
            ("occlusion lidar vs oracle dense p", 0.002, vx["lidar_vs_oracle"]["dynamic_dense"]["p"]),
            ("occlusion camera vs oracle dense", -0.140,
             vx["camera_vs_oracle"]["dynamic_dense"]["success_gain"]),
        ]
        for arm, s_s, d_s, d_c in (("all_round", 0.975, 0.960, 0.020),
                                   ("lidar", 0.960, 0.925, 0.050),
                                   ("camera", 0.890, 0.835, 0.155)):
            out += [(f"occlusion {arm} sparse success", s_s, vcells[arm]["dynamic"]["success"]),
                    (f"occlusion {arm} dense success", d_s,
                     vcells[arm]["dynamic_dense"]["success"]),
                    (f"occlusion {arm} dense collision", d_c,
                     vcells[arm]["dynamic_dense"]["collision"])]

    # --- released crashes and the gap probe, Phase 6b ---------------------
    cr = "results/stall_crash_audit.json"
    if os.path.exists(cr):
        cj = load(cr)
        added, probe, per = cj["added_collisions"], cj["probe"], cj["per_seed"]
        width, clear = probe["gap_width_rad"], probe["clearance_ahead_m"]
        # The documents' post hoc split, re-derived here rather than trusted.
        near = total = 0
        for s in range(6):
            quiet = {e["episode"]: e for e in per[f"rgbi_s{s}"]}
            loud = {e["episode"]: e for e in per[f"rgbi_s{s}_released"]}
            for ep, q in quiet.items():
                r = loud[ep]
                if q["timeout"] and r["collision"] and "metres_from_stall" in r:
                    total += 1
                    near += r["metres_from_stall"] <= 1.0
        out += [
            ("crash decision MIXED", 1.0, float(cj["decision"] == "MIXED")),
            ("crash added collisions", 19, added["n"]),
            ("crash at the opening", 9, added["at_opening"]),
            ("crash elsewhere", 10, added["elsewhere"]),
            ("crash median metres", 0.51, added["median_metres_from_stall"]),
            ("crash median seconds", 3.1, added["median_seconds_since_release"]),
            ("crash within a metre at any delay", 13, near),
            ("crash added total re-derived", 19, total),
            ("probe gap width rgbi", -0.293, width["rgbi"]["mean"]),
            ("probe gap width depthi", 0.220, width["depthi"]["mean"]),
            ("probe gap width gap", 0.513, width["gap"]),
            ("probe gap width p", 0.0022, width["p"]),
            ("probe clearance rgbi", 0.205, clear["rgbi"]["mean"]),
            ("probe clearance depthi", 0.264, clear["depthi"]["mean"]),
            ("probe clearance gap", 0.059, clear["gap"]),
            ("probe clearance p", 0.132, clear["p"]),
            ("probe gap width complete separation", 1.0,
             float(max(width["rgbi"]["per_seed"]) < min(width["depthi"]["per_seed"]))),
        ]

    # --- fitting without differencing, Phase 6a ---------------------------
    ob = "results/orbit_experiment.json"
    if os.path.exists(ob):
        oj = load(ob)
        ocells, ox, oerr = oj["cells"], oj["contrasts"], oj["estimator_error"]
        v_line, v_diff = ox["orbit@0.01_vs_line@0.01"], ox["orbit@0.01_vs_differenced@0.01"]
        v_oracle, quiet = ox["orbit@0.01_vs_oracle"], ox["orbit@0.0_vs_oracle"]
        six = ox["orbit6@0.01_vs_orbit@0.01"]
        out += [
            ("orbit decision FILTER", 1.0, float(oj["decision"] == "FILTER")),
            ("orbit frozen matches oracle", 1.0, float(oj["checks"]["frozen_matches_oracle"])),
            ("orbit vs line dense", 0.145, v_line["dynamic_dense"]["success_gain"]),
            ("orbit vs line dense won", 32, v_line["dynamic_dense"]["episodes_won"]),
            ("orbit vs line dense lost", 3, v_line["dynamic_dense"]["episodes_lost"]),
            ("orbit vs line sparse", 0.065, v_line["dynamic"]["success_gain"]),
            ("orbit vs line sparse p", 0.004, v_line["dynamic"]["p"]),
            ("orbit vs differenced dense", 0.140, v_diff["dynamic_dense"]["success_gain"]),
            ("orbit vs differenced sparse", 0.100, v_diff["dynamic"]["success_gain"]),
            ("orbit vs oracle dense", -0.015, v_oracle["dynamic_dense"]["success_gain"]),
            ("orbit vs oracle dense p", 0.453, v_oracle["dynamic_dense"]["p"]),
            ("orbit vs oracle sparse", -0.020, v_oracle["dynamic"]["success_gain"]),
            ("orbit vs oracle sparse p", 0.125, v_oracle["dynamic"]["p"]),
            ("orbit noise-free vs oracle dense", -0.005, quiet["dynamic_dense"]["success_gain"]),
            ("orbit noise-free vs oracle bounded", 1.0,
             float(quiet["dynamic_dense"]["verdict"] == "INERT (bounded)")),
            ("orbit six seconds vs three", -0.010, six["dynamic_dense"]["success_gain"]),
            ("orbit six seconds p", 0.727, six["dynamic_dense"]["p"]),
            ("orbit dense success", 0.960, ocells["orbit@0.01"]["dynamic_dense"]["success"]),
            ("orbit dense collision", 0.020, ocells["orbit@0.01"]["dynamic_dense"]["collision"]),
            ("orbit sparse success", 0.975, ocells["orbit@0.01"]["dynamic"]["success"]),
            ("orbit6 dense success", 0.950, ocells["orbit6@0.01"]["dynamic_dense"]["success"]),
            ("orbit noise-free dense success", 0.970,
             ocells["orbit@0.0"]["dynamic_dense"]["success"]),
            ("orbit error 1s", 0.024,
             round(oerr["dynamic_dense"]["orbit3"]["1.0s"]["median"], 3)),
            ("orbit error 2s", 0.050,
             round(oerr["dynamic_dense"]["orbit3"]["2.0s"]["median"], 3)),
            ("orbit error 7s", 0.303,
             round(oerr["dynamic_dense"]["orbit3"]["7.0s"]["median"], 3)),
        ]

    # --- noise on the observations, Phase 5z ------------------------------
    nz = "results/noise_experiment.json"
    if os.path.exists(nz):
        nj = load(nz)
        ncells, nx, s1 = nj["cells"], nj["contrasts"], nj["stage1"]["dynamic_dense"]
        at01 = nx["harmonic@0.01_vs_constant_velocity@0.01"]["dynamic_dense"]
        at05 = nx["harmonic@0.05_vs_constant_velocity@0.05"]["dynamic_dense"]
        line_cost = nx["constant_velocity@0.01_vs_constant_velocity@0.0(5y)"]["dynamic_dense"]
        fit_cost = nx["harmonic@0.01_vs_harmonic@0.0(5y)"]["dynamic_dense"]
        v_oracle = nx["harmonic@0.01_vs_oracle@0.0"]["dynamic_dense"]
        out += [
            ("noise decision UNRESOLVED", 1.0, float(nj["decision"] == "UNRESOLVED")),
            ("noise oracle deaf", 1.0, float(nj["checks"]["oracle_is_deaf_to_noise"])),
            ("noise oracle reproduces 5r", 1.0, float(nj["checks"]["oracle_reproduces_phase_5r"])),
            ("noise crossing sigma", 0.05, nj["fit_still_wins_up_to"]),
            ("noise 1cm dense gain", 0.005, at01["success_gain"]),
            ("noise 1cm dense won", 15, at01["episodes_won"]),
            ("noise 1cm dense lost", 14, at01["episodes_lost"]),
            ("noise 1cm dense ci lo", -0.050, at01["ci95"][0]),
            ("noise 1cm dense ci hi", 0.055, at01["ci95"][1]),
            ("noise 5cm dense gain", 0.115, at05["success_gain"]),
            ("noise 5cm dense p", 0.0002, at05["p"]),
            ("noise 5cm dense won", 30, at05["episodes_won"]),
            ("noise 5cm dense lost", 7, at05["episodes_lost"]),
            ("noise line cost at 1cm", -0.095, line_cost["success_gain"]),
            ("noise line cost p", 0.0013, line_cost["p"]),
            ("noise fit cost at 1cm", -0.160, fit_cost["success_gain"]),
            ("noise fit vs oracle at 1cm", -0.155, v_oracle["success_gain"]),
            # Stage 1, dense clutter, the two horizons the documents quote.
            ("noise s1 line 2s at 0", 0.086, s1["constant_velocity@0.0"]["2.0s"]["median"]),
            ("noise s1 line 2s at 0.005", 0.245, s1["constant_velocity@0.005"]["2.0s"]["median"]),
            ("noise s1 line 2s at 0.01", 0.443, s1["constant_velocity@0.01"]["2.0s"]["median"]),
            ("noise s1 line 2s at 0.02", 0.860, s1["constant_velocity@0.02"]["2.0s"]["median"]),
            ("noise s1 line 2s at 0.05", 2.093, s1["constant_velocity@0.05"]["2.0s"]["median"]),
            ("noise s1 fit 2s at 0.005", 0.304, s1["harmonic@0.005"]["2.0s"]["median"]),
            ("noise s1 fit 2s at 0.01", 0.688, s1["harmonic@0.01"]["2.0s"]["median"]),
            ("noise s1 fit 2s at 0.02", 0.840, s1["harmonic@0.02"]["2.0s"]["median"]),
            ("noise s1 fit 2s at 0.05", 0.715, s1["harmonic@0.05"]["2.0s"]["median"]),
            ("noise s1 line 7s at 0.05", 7.253, s1["constant_velocity@0.05"]["7.0s"]["median"]),
            ("noise s1 fit 7s at 0.05", 0.932, s1["harmonic@0.05"]["7.0s"]["median"]),
        ]
        for arm, dense_s, dense_c in (
            ("constant_velocity@0.01", 0.815, 0.160), ("harmonic@0.01", 0.820, 0.160),
            ("constant_velocity@0.05", 0.680, 0.310), ("harmonic@0.05", 0.795, 0.195),
        ):
            out += [(f"noise {arm} dense success", dense_s,
                     ncells[arm]["dynamic_dense"]["success"]),
                    (f"noise {arm} dense collision", dense_c,
                     ncells[arm]["dynamic_dense"]["collision"])]

    # --- fitting the oscillation, Phase 5y --------------------------------
    hm = "results/harmonic_experiment.json"
    if os.path.exists(hm):
        hj = load(hm)
        hc, hcells, hx, herr = hj["checks"], hj["cells"], hj["contrasts"], hj["estimator_error"]
        line_v_fit = hx["harm_m2_vs_cv_m2"]
        v_oracle = hx["harm_m2_vs_oracle_m2"]
        out += [
            ("harmonic decision MODEL", 1.0, float(hj["decision"] == "MODEL")),
            ("harmonic frozen identity", 1.0, float(hc["frozen_identity_harmonic_vs_line"])),
            ("harmonic reproduces 5r", 1.0, float(hc["reproduces_phase_5r"])),
            ("harmonic dense gain", 0.070, line_v_fit["dynamic_dense"]["success_gain"]),
            ("harmonic dense p", 0.0001, line_v_fit["dynamic_dense"]["p"]),
            ("harmonic dense won", 14, line_v_fit["dynamic_dense"]["episodes_won"]),
            ("harmonic dense lost", 0, line_v_fit["dynamic_dense"]["episodes_lost"]),
            ("harmonic dense ci lo", 0.035, line_v_fit["dynamic_dense"]["ci95"][0]),
            ("harmonic dense ci hi", 0.105, line_v_fit["dynamic_dense"]["ci95"][1]),
            ("harmonic dense collision delta", -0.070,
             line_v_fit["dynamic_dense"]["collision_delta"]),
            ("harmonic sparse gain", 0.020, line_v_fit["dynamic"]["success_gain"]),
            ("harmonic sparse won", 4, line_v_fit["dynamic"]["episodes_won"]),
            ("harmonic vs oracle dense", 0.005, v_oracle["dynamic_dense"]["success_gain"]),
            ("harmonic vs oracle dense ci lo", 0.000, v_oracle["dynamic_dense"]["ci95"][0]),
            ("harmonic vs oracle dense ci hi", 0.015, v_oracle["dynamic_dense"]["ci95"][1]),
            ("harmonic vs oracle dense bounded", 1.0,
             float(v_oracle["dynamic_dense"]["verdict"] == "INERT (bounded)")),
            ("harmonic vs oracle sparse discordant", 0,
             v_oracle["dynamic"]["episodes_won"] + v_oracle["dynamic"]["episodes_lost"]),
            ("harmonic dense success", 0.980, hcells["harm_m2"]["dynamic_dense"]["success"]),
            ("harmonic dense collision", 0.005, hcells["harm_m2"]["dynamic_dense"]["collision"]),
            ("harmonic sparse success", 0.995, hcells["harm_m2"]["dynamic"]["success"]),
            # Derived, as elsewhere: frozen success minus moving success.
            ("harmonic motion cost dense", 0.005,
             hcells["harm_m2"]["dynamic_dense_frozen"]["success"]
             - hcells["harm_m2"]["dynamic_dense"]["success"]),
            ("harmonic motion cost sparse", 0.000,
             hcells["harm_m2"]["dynamic_frozen"]["success"]
             - hcells["harm_m2"]["dynamic"]["success"]),
            # The estimator's own error, which is what the prediction rested on.
            ("estimator line 1s dense", 0.023,
             round(herr["dynamic_dense"]["constant_velocity"]["1.0s"]["median"], 3)),
            ("estimator line 7s dense", 1.192,
             round(herr["dynamic_dense"]["constant_velocity"]["7.0s"]["median"], 3)),
            ("estimator line 2s dense", 0.086,
             round(herr["dynamic_dense"]["constant_velocity"]["2.0s"]["median"], 3)),
            ("estimator fit 7s dense median", 0.000,
             round(herr["dynamic_dense"]["harmonic"]["7.0s"]["median"], 4)),
            ("estimator fit 7s dense p95", 0.0001,
             round(herr["dynamic_dense"]["harmonic"]["7.0s"]["p95"], 4)),
        ]

    # --- capping the estimate's reach, Phase 5w ---------------------------
    ce = "results/cap_experiment.json"
    if os.path.exists(ce):
        kj = load(ce)
        kc, kcells, kx = kj["checks"], kj["cells"], kj["contrasts"]
        c2d, c2s = kx["cv_cap2_vs_cv_m2"]["dynamic_dense"], kx["cv_cap2_vs_cv_m2"]["dynamic"]
        c1s, c1d = kx["cv_cap1_vs_cv_m2"]["dynamic"], kx["cv_cap1_vs_cv_m2"]["dynamic_dense"]
        od, osp = kx["cv_cap2_vs_oracle_m2"]["dynamic_dense"], kx["cv_cap2_vs_oracle_m2"]["dynamic"]
        out += [
            ("cap decision UNRESOLVED", 1.0, float(kj["decision"] == "UNRESOLVED")),
            ("cap frozen identity", 1.0, float(kc["frozen_identity_cap1_vs_uncapped"])),
            ("cap reproduces 5r", 1.0, float(kc["reproduces_phase_5r"])),
            ("cap2 dense gain", 0.010, c2d["success_gain"]),
            ("cap2 dense p", 0.727, c2d["p"]),
            ("cap2 dense won", 5, c2d["episodes_won"]),
            ("cap2 dense lost", 3, c2d["episodes_lost"]),
            ("cap2 dense ci lo", -0.015, c2d["ci95"][0]),
            ("cap2 dense ci hi", 0.040, c2d["ci95"][1]),
            ("cap2 sparse gain", -0.020, c2s["success_gain"]),
            ("cap2 sparse p", 0.289, c2s["p"]),
            ("cap1 sparse gain", -0.085, c1s["success_gain"]),
            ("cap1 sparse p", 0.0005, c1s["p"]),
            ("cap1 sparse won", 3, c1s["episodes_won"]),
            ("cap1 sparse lost", 20, c1s["episodes_lost"]),
            ("cap1 sparse collision delta", 0.085, c1s["collision_delta"]),
            ("cap1 dense gain", -0.035, c1d["success_gain"]),
            ("cap1 dense p", 0.143, c1d["p"]),
            ("cap2 vs oracle dense", -0.055, od["success_gain"]),
            ("cap2 vs oracle dense p", 0.003, od["p"]),
            ("cap2 vs oracle sparse", -0.040, osp["success_gain"]),
            ("cap2 vs oracle sparse p", 0.008, osp["p"]),
        ]
        for arm, s_s, s_c, d_s, d_c, d_bare in (
            ("cv_m2", 0.975, 0.020, 0.910, 0.075, 31), ("cv_cap2", 0.955, 0.040, 0.920, 0.065, 21),
            ("cv_cap1", 0.890, 0.105, 0.875, 0.110, 34),
        ):
            out += [(f"cap {arm} sparse success", s_s, kcells[arm]["dynamic"]["success"]),
                    (f"cap {arm} sparse collision", s_c, kcells[arm]["dynamic"]["collision"]),
                    (f"cap {arm} dense success", d_s, kcells[arm]["dynamic_dense"]["success"]),
                    (f"cap {arm} dense collision", d_c, kcells[arm]["dynamic_dense"]["collision"]),
                    (f"cap {arm} dense reach bare", d_bare,
                     kcells[arm]["dynamic_dense"]["episodes_reaching_bare_radius"])]

    # --- pixel stall audit, Phase 5v --------------------------------------
    pa = "results/pixel_stall_audit.json"
    if os.path.exists(pa):
        aj = load(pa)
        p1, ps = aj["part1"], aj["summary"]
        rg, dp = ps["rgbi"], ps["depthi"]
        out += [
            ("pixel wall max", 0.069, p1["wall"]["max_m"]),
            ("pixel box max", 0.091, p1["box"]["max_m"]),
            ("pixel circle max", 0.114, p1["circle"]["max_m"]),
            ("pixel wall median", 0.046, p1["wall"]["median_m"]),
            ("pixel box median", 0.048, p1["box"]["median_m"]),
            ("pixel circle median", 0.060, p1["circle"]["median_m"]),
            ("pixel circle max below 1.2", 0.114, p1["circle"]["max_below_1p2_m"]),
            ("pixel decision BLIND DETOUR", 1.0, float(aj["part2_decision"] == "BLIND DETOUR")),
            ("pixel rgbi stall share", 0.350, rg["stall_share_mean"]),
            ("pixel depthi stall share", 0.076, dp["stall_share_mean"]),
            ("pixel stall share p", 0.022, ps["stall_share_p"]),
            ("pixel rgbi stalls goal_open", 0.187, rg["stall_class_shares"]["goal_open"]),
            ("pixel rgbi stalls detour", 0.812, rg["stall_class_shares"]["detour"]),
            ("pixel depthi stalls goal_open", 0.000, dp["stall_class_shares"]["goal_open"]),
            ("pixel depthi stalls detour", 0.964, dp["stall_class_shares"]["detour"]),
            ("pixel rgbi detour out of view", 0.856,
             rg["detour_stalls_goal_out_of_view"] / rg["detour_stalls"]),
            ("pixel depthi detour out of view", 1.000,
             dp["detour_stalls_goal_out_of_view"] / dp["detour_stalls"]),
            ("pixel rgbi probe", 0.813, rg["probe_balanced_accuracy_mean"]),
            ("pixel depthi probe", 0.902, dp["probe_balanced_accuracy_mean"]),
            ("pixel probe p", 0.004, ps["probe_p"]),
            ("pixel rgbi open stalls read blocked", 2281, rg["goal_open_stalls_probe_says_blocked"]),
            ("pixel rgbi open stalls", 4466, rg["goal_open_stalls"]),
            ("pixel rgbi open stalls blocked share", 0.511,
             rg["goal_open_stalls_probe_says_blocked"] / rg["goal_open_stalls"]),
            ("pixel rgbi open moving blocked share", 0.150,
             rg["goal_open_moving_probe_says_blocked"] / rg["goal_open_moving"]),
            ("pixel rgbi sensor open stalls blocked share", 0.244,
             rg["goal_open_stalls_sensor_probe_says_blocked"] / rg["goal_open_stalls"]),
            ("pixel rgbi sensor open moving blocked share", 0.350,
             rg["goal_open_moving_sensor_probe_says_blocked"] / rg["goal_open_moving"]),
        ]
        # "Concentrated in seeds 3-5".
        seeds = {e["seed"]: e["stall_classes"][0] for e in aj["per_seed"]["rgbi"]}
        out.append(("pixel rgbi open stalls in seeds 3-5 share", 0.998,
                    sum(seeds[s] for s in (3, 4, 5)) / sum(seeds.values())))

    # --- velocity latch replay, Phase 5x ----------------------------------
    sc = "results/stall_counterfactual.json"
    if os.path.exists(sc):
        cj = load(sc)
        rg, dp = cj["arms"]["rgbi"], cj["arms"]["depthi"]
        seeds_down = sum(e["timeout_override"] < e["timeout"] for e in rg["per_seed"])
        seeds_up = sum(e["collision_override"] > e["collision"] for e in rg["per_seed"])
        out += [
            ("latch decision PARTIAL", 1.0, float(cj["decision"] == "PARTIAL")),
            ("latch rgbi open stall go real", 0.013, rg["open_stall_go_real"]),
            ("latch rgbi open stall go told moving", 0.380, rg["open_stall_go_move"]),
            ("latch rgbi detour stall go real", 0.015, rg["detour_stall_go_real"]),
            ("latch rgbi detour stall go told moving", 0.441, rg["detour_stall_go_move"]),
            ("latch rgbi open moving go real", 0.992, rg["open_moving_go_real"]),
            ("latch rgbi open moving go told stopped", 0.984, rg["open_moving_go_zero"]),
            ("latch depthi stall go real", 0.007, dp["detour_stall_go_real"]),
            ("latch depthi stall go told moving", 0.006, dp["detour_stall_go_move"]),
            ("latch depthi moving go real", 0.999, dp["open_moving_go_real"]),
            ("latch depthi moving go told stopped", 1.000, dp["open_moving_go_zero"]),
            ("latch exit shift", 0.367, cj["rgbi_exit_shift"]),
            ("latch entry shift", 0.008, cj["rgbi_entry_shift"]),
            ("latch rgbi timeout change", -0.073, rg["timeout_change"]),
            ("latch rgbi timeout p", 0.031, rg["timeout_change_p"]),
            ("latch rgbi collision change", 0.063, rg["collision_change"]),
            ("latch rgbi collision p", 0.031, rg["collision_change_p"]),
            ("latch rgbi success change", 0.010, rg["success_change"]),
            ("latch rgbi success p", 0.25, rg["success_change_p"]),
            ("latch rgbi seeds with fewer timeouts", 6, seeds_down),
            ("latch rgbi seeds with more collisions", 6, seeds_up),
            ("latch depthi timeout change", 0.010, dp["timeout_change"]),
            ("latch depthi collision change", -0.010, dp["collision_change"]),
            ("latch depthi success change", 0.000, dp["success_change"]),
        ]

    # --- encoder cost re-priced at 1:1, Phase 5s -------------------------
    ri, rl = "results/repricing_encoder_interaction.json", "results/repricing_encoder.json"
    if os.path.exists(ri) and os.path.exists(rl):
        ij, lj = load(ri), load(rl)
        ic, lc = ij["conditions"], lj["conditions"]
        out.append(("encoder decision SURVIVES", 1.0, float(ij["decision"] == "SURVIVES")))
        for cond, metric, hi, lo, inter, p in (
            ("narrow", "success", -0.218, -0.180, 0.038, 0.4697),
            ("narrow", "collision", 0.105, 0.027, -0.078, 0.3896),
            ("narrow", "timeout", 0.113, 0.153, 0.040, 0.7251),
            ("dense", "success", -0.238, -0.193, 0.045, 0.4697),
            ("nominal", "success", -0.162, -0.088, 0.073, 0.0195),
        ):
            e = ic[cond][metric]
            out += [(f"encoder {cond} {metric} 4:1", hi, e["delta_4to1"]),
                    (f"encoder {cond} {metric} 1:1", lo, e["delta_1to1"]),
                    (f"encoder {cond} {metric} interaction", inter, e["interaction"]),
                    (f"encoder {cond} {metric} interaction p", p, e["p"])]
        narrow_arms = lc["narrow"]["per_arm"]
        out += [
            ("encoder narrow 1:1 success p", 0.002, lc["narrow"]["comparisons"]["success"]["p"]),
            ("encoder dense 1:1 success p", 0.002, lc["dense"]["comparisons"]["success"]["p"]),
            ("encoder narrow 1:1 timeout p", 0.017, lc["narrow"]["comparisons"]["timeout"]["p"]),
            ("encoder narrow 1:1 collision p", 0.662,
             lc["narrow"]["comparisons"]["collision"]["p"]),
            ("encoder depthi narrow timeout", 0.028,
             sum(narrow_arms["depthi"]["timeout"]) / 6),
            ("encoder rgbi narrow timeout", 0.182, sum(narrow_arms["rgbi"]["timeout"]) / 6),
            ("encoder narrow timeout seeds positive", 6,
             sum(d > 0 for d in ic["narrow"]["timeout"]["per_seed_1to1"])),
        ]
        # "Every depth seed above every RGB seed" on both registered conditions.
        for cond in ("narrow", "dense"):
            arms = lc[cond]["per_arm"]
            out.append((f"encoder {cond} 1:1 complete separation", 1.0,
                        float(min(arms["depthi"]["success"]) > max(arms["rgbi"]["success"]))))

    # Post hoc description of the episodes the estimate lost.
    edg = "results/estimate_diagnostic.json"
    if os.path.exists(edg):
        dj = load(edg)
        ds, dc = dj["summary"], dj["conditions"]
        out += [
            ("estimate diag lost", 18, ds["lost"]),
            ("estimate diag mover contacts", 18, ds["mover_contacts"]),
            ("estimate diag median error", 0.030, ds["median_estimate_error_m"]),
            ("estimate diag max error", 0.159, ds["max_estimate_error_m"]),
            ("estimate diag bare radius", 15, ds["plans_at_bare_radius"]),
            ("estimate diag dense early share", 0.116,
             dc["dynamic_dense"]["median_reduced_share_early_estimate"]),
            ("estimate diag dense oracle share", 0.000,
             dc["dynamic_dense"]["median_reduced_share_oracle"]),
            ("estimate diag sparse early share", 0.033,
             dc["dynamic"]["median_reduced_share_early_estimate"]),
            ("estimate diag sparse oracle share", 0.077,
             dc["dynamic"]["median_reduced_share_oracle"]),
            ("estimate diag no early fallback", 6,
             dc["dynamic"]["no_early_fallback"] + dc["dynamic_dense"]["no_early_fallback"]),
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
                # A document whose count has gone missing is a failure, not a
                # skip: a scripted edit once blanked all three to "-test suite"
                # and "# tests", and the check passed by finding nothing to check.
                quoted = float(m.group(1)) if m else -1.0
                out.append((f"{name} suite size", quoted, float(n_tests)))

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
