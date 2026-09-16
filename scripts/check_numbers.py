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
