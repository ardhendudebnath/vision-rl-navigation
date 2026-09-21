"""How much of what the map and the pose were worth was this implementation?

Report §9.2 and §9.3 took the map and then the exact pose away from the
hand-written stack and measured each. Both sections ended on the same caution:
what was measured is *this* stack, and a production stack would say how much of
it is the implementation and how much is the problem. This asks one: Nav2 with
slam_toolbox, no map, no pose, on the same worlds.

The quantity that answers the question is a **difference in costs**. Each stack
is scored with its privileges and without them, on the same worlds, and what is
compared is what each one *loses*:

    d = (Nav2 without - Nav2 with) - (hand-written without - hand-written with)

computed world by world and pooled over all six conditions. Comparing the two
stacks' success rates directly would not do: with full privileges Nav2 already
beats the hand-written controller in tight corridors (§4.1), and that would be
counted as an answer to a question about privileges.

"Without" is the same on both sides where it can be: the odometry model, its
parameters and its per-world seeding are §9.3's, imported by the bridge, and
the like-for-like arm gives Nav2 the same 32-beam scanner the hand-written
stack mapped from. The 360-beam arm is Nav2 as every published Nav2 row has run
it, and says what a production sensor adds on top of a production stack.

Nav2 is not deterministic, so its with-privileges side has two published passes
(``results/nav2_runs/run1``, ``run2``) and every verdict here is required to
hold against both — the rule §4.1 already applies to Nav2 comparisons.

    python scripts/nav2_slam_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spacetime_experiment import verdict  # noqa: E402
from velocity_experiment import mcnemar_p  # noqa: E402

CONDITIONS = ("sparse", "large", "nominal", "noisy_lidar", "dense", "narrow")
#: Half-width, in success rate, inside which a pooled difference in costs counts
#: as "the same", and the size a difference must reach to count as material.
#:
#: Set from the val band before any test world was run, and changed once there:
#: the first draft used 0.05. The hand-written stack's pooled cost for losing
#: both privileges is about 0.33, so 0.10 is under a third of it -- "paid about
#: the same". And val's variance says 600 paired worlds resolve about +-0.047:
#: at 0.05 a null could be declared only if the point estimate landed within
#: 0.003 of zero, a bound no real result could meet.
BOUND = 0.10
#: A Nav2 run whose controller issued fewer commands than this per sim step was
#: starved of wall-clock time and is not scored.
#:
#: Measured over the episodes that *succeeded*. A starved controller is slow in
#: every episode, the ones that reach the goal included; a failing one is not
#: starved just because it sat in a recovery behaviour -- Nav2's "wait" recovery
#: publishes nothing -- or stopped after abandoning the goal. Averaged over all
#: episodes, the 32-beam val runs read 0.68 to 0.88 with no goal abandoned,
#: which would have refused to score an arm for navigating badly. Runs recorded
#: before per-episode ratios were kept (the published full-privilege passes)
#: carry only the whole-run mean, which is then used.
MIN_COMMANDS_PER_STEP = 0.9

#: Recorded before the run and committed before any SLAM result on the test
#: worlds exists.
#:
#: DISCLOSED: derived from the val band, twenty worlds per condition, which no
#: experiment scores -- a forecast from measurement, one band away. The Nav2
#: SLAM arm was debugged there, and every val number below was collected after
#: the last fix (report §9.4 lists them; the one that mattered was this
#: project's own slam_toolbox thresholds, which made SLAM drag the pose
#: backwards). The val full-privilege arm was run paced at 5x real time, not
#: under the published passes' unthrottled protocol: run alongside other
#: stacks, the unthrottled loop asked for plans on the previous world's map.
#: The test comparison uses the published passes, unchanged.
#:
#:     val, success      hand-written      Nav2       Nav2+SLAM
#:                       with / without    with       360    32
#:     sparse            1.000 / 0.350     1.000      1.000  0.700
#:     large             1.000 / 0.600     1.000      0.950  0.400
#:     nominal           1.000 / 0.700     1.000      1.000  0.750
#:     noisy_lidar       1.000 / 0.750     1.000      1.000  0.750
#:     dense             0.850 / 0.650     0.950      0.700  0.550
#:     narrow            0.850 / 0.650     0.800      0.650  0.550
#:
#: Pooled difference in costs, val: +0.258 [+0.167, +0.350] at 360 beams,
#: -0.008 [-0.117, +0.092] at 32. Collisions in clutter, 360 beams: 0.15 dense,
#: 0.20 narrow, against 0.00 and 0.05 with privileges. Timeouts in open worlds,
#: 32 beams: 0.20 to 0.60. Median final pose error: 0.058 to 0.145 m at 360
#: beams, 0.171 to 0.637 m at 32.
PREDICTION = (
    "a production stack pays for losing the map and the pose what this "
    "project's stack pays when it has the same sensor, and a quarter of it "
    "when it has a dense one: the sensor, not the implementation, is what "
    "the privileges were standing in for. At 32 beams the pooled difference "
    "in costs is PROBLEM -- between -0.07 and +0.07, 95% CI inside +-0.10 -- "
    "against both published full-privilege passes. At 360 beams it is "
    "IMPLEMENTATION, between +0.15 and +0.35 against both. At 360 beams "
    "Nav2+SLAM scores at least 0.93 on sparse, large, nominal and "
    "noisy_lidar; on dense and narrow it scores at least 0.10 below Nav2's "
    "own full-privilege row in both passes, and collides at a rate of at "
    "least 0.08 on each, so what it pays in clutter is collisions. At 32 "
    "beams it scores below the 360-beam arm on every condition, lowest on "
    "large, with a timeout rate of at least 0.15 on each of sparse, large, "
    "nominal and noisy_lidar. Median final pose error is at most 0.20 m on "
    "every condition at 360 beams and larger at 32 than at 360 on every "
    "condition. All twelve runs pass the starvation gate, and no run sends "
    "more than 5% of its goals before SLAM is ready. "
    "Decision on the 32-beam arm, like for like -- IMPLEMENTATION: pooled "
    "difference at least +0.10 with CI above 0. PROBLEM: CI inside +-0.10. "
    "BACKWARDS: at most -0.10 with CI below 0. Otherwise UNRESOLVED. A label "
    "stands only if it holds against both passes. Same rule, secondary, at "
    "360 beams."
)


def load_nav2(path: Path) -> dict:
    j = json.loads(path.read_text(encoding="utf-8"))
    per = j.get("commands_per_step_per_episode")
    if per is not None:
        wins = [r for r, e in zip(per, j["per_episode"], strict=True) if e["success"]]
        ratio = float(np.mean(wins)) if wins else float(np.mean(per))
    else:
        ratio = float(j["mean_commands_per_step"])
    if ratio < MIN_COMMANDS_PER_STEP:
        raise SystemExit(f"{path}: {ratio:.3f} commands per sim step in successful episodes; "
                         "Nav2 was starved and this run cannot be scored")
    return {"seeds": [e["world_seed"] for e in j["per_episode"]],
            "success": np.array([int(e["success"]) for e in j["per_episode"]]),
            "collision": float(j["collision_rate"]), "timeout": float(j["timeout_rate"]),
            "commands": float(ratio),
            "aborted": j.get("nav2_aborted_episodes"),
            "not_ready": j.get("slam_not_ready_episodes"),
            "pose_error_median": j.get("pose_error_median")}


def pooled_did(nav_with, nav_without, hw_with, hw_without, rng, n_boot):
    """Pooled difference in costs and its 95% CI, resampling worlds within
    each condition so every bootstrap draw keeps the six-condition design."""
    per = {c: (nav_without[c] - nav_with[c]) - (hw_without[c] - hw_with[c]) for c in CONDITIONS}
    point = float(np.mean(np.concatenate([per[c] for c in CONDITIONS])))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.mean(np.concatenate(
            [per[c][rng.integers(0, len(per[c]), len(per[c]))] for c in CONDITIONS]))
    return point, [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], per


def classify(point: float, ci) -> str:
    """The same four-way rule as spacetime_experiment.verdict, on a pooled
    difference in costs: significant *and* material either way, bounded, or
    neither. Positive means Nav2 pays less for losing its privileges."""
    if ci[0] > 0 and point >= BOUND:
        return "IMPLEMENTATION"
    if ci[1] < 0 and point <= -BOUND:
        return "BACKWARDS"
    if ci[0] > -BOUND and ci[1] < BOUND:
        return "PROBLEM"
    return "UNRESOLVED"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--slam-dir", default="results/nav2_slam_runs")
    p.add_argument("--full-dirs", nargs="+",
                   default=["results/nav2_runs/run1", "results/nav2_runs/run2"])
    p.add_argument("--handwritten", default="results/localisation_experiment.json",
                   help="Phase 6f's result file: full_map and matched, per episode")
    p.add_argument("--suffix", default="", help="'_val' to read development runs")
    p.add_argument("--out", default="results/nav2_slam_comparison.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)

    hw_json = json.loads(Path(args.handwritten).read_text(encoding="utf-8"))
    hw_with, hw_without, seeds = {}, {}, {}
    for c in CONDITIONS:
        if "conditions" in hw_json:  # the registered Phase 6f result file
            cells = hw_json["conditions"][c]["cells"]
            hw_with[c] = np.array(cells["full_map"]["success_per_episode"])
            hw_without[c] = np.array(cells["matched"]["success_per_episode"])
            seeds[c] = None
        else:  # a development file: {"seeds", "full", "matched"} per condition
            hw_with[c] = np.array(hw_json[c]["full"])
            hw_without[c] = np.array(hw_json[c]["matched"])
            seeds[c] = hw_json[c]["seeds"]

    full = {d: {c: load_nav2(Path(d) / f"{c}__nav2{args.suffix}.json") for c in CONDITIONS}
            for d in args.full_dirs}
    slam = {b: {c: load_nav2(Path(args.slam_dir) / f"{c}__{stem}{args.suffix}.json")
                for c in CONDITIONS}
            for b, stem in ((360, "nav2_slam"), (32, "nav2_slam_b32"))}

    # Every arm must be on the same worlds, in the same order.
    for c in CONDITIONS:
        ref = full[args.full_dirs[0]][c]["seeds"]
        n = len(hw_with[c])
        assert len(ref) == n, (c, "episode counts differ")
        if seeds[c] is not None:
            assert list(seeds[c]) == ref, (c, "hand-written worlds differ")
        for d in args.full_dirs:
            assert full[d][c]["seeds"] == ref, (c, d, "world order differs")
        for b in slam:
            assert slam[b][c]["seeds"] == ref, (c, b, "world order differs")

    report = {"prediction": PREDICTION, "bound": BOUND, "conditions": {}, "pooled": {}}
    for c in CONDITIONS:
        e = {"handwritten": {"with": float(hw_with[c].mean()), "without": float(hw_without[c].mean())},
             "nav2_with": {d: float(full[d][c]["success"].mean()) for d in args.full_dirs},
             "nav2_without": {}, "direct": {}}
        for b in slam:
            s = slam[b][c]
            e["nav2_without"][str(b)] = {k: (float(v.mean()) if k == "success" else v)
                                         for k, v in s.items() if k != "seeds"}
            # Direct, like-for-like where b = 32: Nav2 with neither privilege
            # against the hand-written stack with neither, world by world.
            a, bb = hw_without[c].astype(bool), s["success"].astype(bool)
            pv, lost, won = mcnemar_p(a, bb)
            draws = [(bb[i].mean() - a[i].mean()) for i in
                     (rng.integers(0, len(a), len(a)) for _ in range(args.bootstrap))]
            ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
            gain = float(bb.mean() - a.mean())
            e["direct"][str(b)] = {"gain": gain, "p": pv, "won": won, "lost": lost, "ci95": ci,
                                   "verdict": verdict(gain, pv, ci)}
        report["conditions"][c] = e

    decisions = {}
    for b in slam:
        per_pass = {}
        for d in args.full_dirs:
            nav_with = {c: full[d][c]["success"] for c in CONDITIONS}
            nav_without = {c: slam[b][c]["success"] for c in CONDITIONS}
            point, ci, per = pooled_did(nav_with, nav_without, hw_with, hw_without,
                                        rng, args.bootstrap)
            hw_cost = float(np.mean(np.concatenate([hw_without[c] - hw_with[c] for c in CONDITIONS])))
            nav_cost = float(np.mean(np.concatenate([nav_without[c] - nav_with[c] for c in CONDITIONS])))
            per_pass[d] = {"did": point, "ci95": ci, "class": classify(point, ci),
                           "nav2_cost": nav_cost, "handwritten_cost": hw_cost,
                           "per_condition": {c: float(per[c].mean()) for c in CONDITIONS}}
        labels = {v["class"] for v in per_pass.values()}
        decisions[str(b)] = labels.pop() if len(labels) == 1 else "UNRESOLVED"
        report["pooled"][str(b)] = per_pass
    report["decision"] = decisions["32"]
    report["decision_360"] = decisions["360"]

    for c in CONDITIONS:
        e = report["conditions"][c]
        print(f"{c:12s} hw {e['handwritten']['with']:.3f}->{e['handwritten']['without']:.3f}  "
              f"nav2 " + "/".join(f"{v:.3f}" for v in e["nav2_with"].values())
              + f" -> 360 {e['nav2_without']['360']['success']:.3f}"
              f" | 32 {e['nav2_without']['32']['success']:.3f}")
    for b, per_pass in report["pooled"].items():
        for d, v in per_pass.items():
            print(f"beams {b} vs {Path(d).name}: nav2 cost {v['nav2_cost']:+.3f}  "
                  f"hand-written cost {v['handwritten_cost']:+.3f}  "
                  f"difference {v['did']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['class']}")
    print(f"decision (32 beams, like for like): {report['decision']}")
    print(f"decision (360 beams): {report['decision_360']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
