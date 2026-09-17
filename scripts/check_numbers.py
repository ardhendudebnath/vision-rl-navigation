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
