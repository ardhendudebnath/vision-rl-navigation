"""Nav2 on `noisy_lidar` with the noise delivered: the four runs the bridge voided.

Until commit 233faae the ROS bridge built its scanner from the condition's
``LidarConfig`` and called it without a random generator, and ``Lidar2D`` adds
noise only when it is handed one. So every Nav2 run on `noisy_lidar` was scored
on clean scans, while the hand-written stack read the noisy sensor. Those are
`nominal`'s worlds, so each was in effect one more pass of `nominal`. Four runs
are affected and are repeated here with the fixed bridge, on the same held-out
worlds in the same order, to the same protocol:

  run1, run2   Nav2 with ground-truth pose and the static map, 360 beams --
               report §4.1's two passes, unthrottled, each run alone
  slam         Nav2 + slam_toolbox, neither privilege, at 360 and at 32 beams --
               report §9.4's two arms, at 5x real time

Every run now records the range noise its published scans *carried*, measured
against the same scans cast without noise, and stops after the first episode
if that disagrees with the configuration (``ros2_bridge/bridge_sensor.py``).
This script refuses any run whose recorded delivery fails the same check.

What it then re-scores, with only the `noisy_lidar` cells replaced:

  - §9.19's comparison of final pose error against this stack on the worlds
    both completed -- the claim the correction withdrew (registered primary);
  - §9.4's pooled difference in costs at 32 and 360 beams, Phase 6g's rule;
  - §9.5's at 360 beams, Phase 6h's rule, and which condition carries the most;
  - §9.6's against the repaired mapper.

Identity control: re-scored with the original, clean-sensor files in place of
the reruns, every pooled point estimate must equal the one published. The
intervals are bootstrap draws from a different stream and are not compared.

    bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 100 --privileges slam --out-dir results/nav2_noise_rerun/slam
    bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 100 --privileges slam --beams 32 --out-dir results/nav2_noise_rerun/slam
    bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 100 --out-dir results/nav2_noise_rerun/run1
    bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 100 --out-dir results/nav2_noise_rerun/run2
    python scripts/nav2_noise_rerun.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "ros2_bridge"))
from bridge_sensor import delivered_matches  # noqa: E402
from nav2_slam_comparison import classify, load_nav2, pooled_did  # noqa: E402

from vision_nav.envs.sensors import LidarConfig  # noqa: E402

CONDITIONS = ("sparse", "large", "nominal", "noisy_lidar", "dense", "narrow")
NOISE = 0.10
PASSES = ("run1", "run2")
ORIGINAL_FULL = {p: Path(f"results/nav2_runs/{p}") for p in PASSES}
ORIGINAL_SLAM = Path("results/nav2_slam_runs")
PHASE_6F = Path("results/localisation_experiment.json")
PHASE_6H = Path("results/sensor_experiment.json")
PHASE_6I = Path("results/corroboration_experiment.json")
#: §9.19's comparison read this stack's held-out `noisy_lidar` episodes from here.
THIS_STACK = Path("results/backend_test.json")

#: The published pooled points, which the identity control must reproduce.
PUBLISHED = {
    "6g_32": {"run1": -0.058, "run2": -0.050},
    "6g_360": {"run1": 0.243, "run2": 0.252},
    "6h_360": {"run1": 0.112, "run2": 0.120},
    "6i_corroborated": {"run1": 0.090, "run2": 0.098},
}

#: Recorded and committed before any Nav2 run with the noise delivered had been
#: seen on any band.
#:
#: DISCLOSED: made knowing every clean-sensor Nav2 result these runs repeat, and
#: every hand-written result on the same worlds. A three-episode val smoke test
#: of the fixed bridge was started before this was written; its output had not
#: been read when this was committed. Also read before writing: two
#: configuration values, Nav2's costmap obstacle_max_range of 5.5 m (which drops
#: the near-maximum readings §9.14 found planting phantoms) and slam_toolbox's
#: max_laser_range of 6.0 m (which does not).
PREDICTION = (
    "with the noise delivered, slam_toolbox still localises better than this "
    "stack on the same sensor, by less than the withdrawn factor of three, and "
    "the full-privilege arm does not notice the noise. "
    "Primary, on the held-out worlds both stacks complete at 360 beams: Nav2 + "
    "slam_toolbox ends closer to the truth on more than half of them, two-sided "
    "sign test p < 0.05, with a median final pose error between 0.40 and 0.67 "
    "of this stack's. Decision -- BETTER: Nav2 closer with sign-test p < 0.05 "
    "and median ratio at most 0.67. WORSE: this stack closer with p < 0.05. "
    "Otherwise UNRESOLVED. Prediction: BETTER. "
    "Nav2 + SLAM at 360 beams: success at least 0.90, median final pose error "
    "between 0.08 and 0.18 m. At 32 beams: success between 0.50 and 0.78, "
    "median final pose error above the clean run's 0.226 m. Full privileges: "
    "both passes succeed between 0.94 and 1.00 and collide at most 0.04. "
    "Re-scored: Phase 6g's decisions stand, UNRESOLVED at 32 beams and "
    "IMPLEMENTATION at 360; Phase 6h's stands at IMPLEMENTATION, every pass at "
    "least +0.100, and noisy_lidar is still the condition carrying the largest "
    "difference in both passes; against the repaired mapper (Phase 6i) the "
    "difference stays UNRESOLVED. Every pooled difference falls, because Nav2 "
    "now pays something under noise where it paid nothing. All four runs "
    "deliver noise within the bridge's check, pass the starvation gate, and the "
    "SLAM arms send no more than 5% of goals before SLAM is ready."
)


def load_rerun(path: Path) -> dict:
    """A rerun file, refused unless it records noise delivered as configured."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    delivered = raw.get("delivered_noise_std")
    if not delivered_matches(LidarConfig(noise_std=NOISE), delivered):
        raise SystemExit(f"{path}: delivered noise {delivered} m against {NOISE} m configured")
    return {**load_nav2(path), "raw": raw}


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test, ties dropped."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def pose_comparison(slam_raw: dict) -> dict:
    """§9.19's comparison: final pose error, Nav2 + SLAM against this stack, on
    the worlds both completed."""
    mine = json.loads(THIS_STACK.read_text(encoding="utf-8"))
    by_seed = {e["seed"]: e for e in mine["cells"]["noisy_lidar"]["front_end"]["episodes"]}
    per = slam_raw["per_episode"]
    errs = slam_raw["pose_error_per_episode"]
    both = [i for i, p in enumerate(per)
            if p["success"] and by_seed.get(p["world_seed"], {}).get("success")]
    nav = np.array([errs[i] for i in both])
    hw = np.array([by_seed[per[i]["world_seed"]]["pose_err_final"] for i in both])
    wins, losses = int(np.sum(nav < hw)), int(np.sum(nav > hw))
    p = sign_test(wins, losses)
    ratio = float(np.median(nav) / np.median(hw))
    if wins > losses and p < 0.05 and ratio <= 0.67:
        decision = "BETTER"
    elif losses > wins and p < 0.05:
        decision = "WORSE"
    else:
        decision = "UNRESOLVED"
    return {"both_completed": len(both), "nav2_median": float(np.median(nav)),
            "this_stack_median": float(np.median(hw)), "ratio": ratio,
            "nav2_closer": wins, "this_stack_closer": losses, "sign_p": p,
            "decision": decision}


def rescore(noisy_full: dict, noisy_slam: dict, rng, n_boot: int) -> dict:
    """The pooled differences in costs of Phases 6g, 6h and 6i, with the
    `noisy_lidar` cells taken from ``noisy_full`` (per pass) and
    ``noisy_slam`` (per beam count) and every other cell as published."""
    full = {p: {c: (noisy_full[p] if c == "noisy_lidar"
                    else load_nav2(ORIGINAL_FULL[p] / f"{c}__nav2.json"))
                for c in CONDITIONS} for p in PASSES}
    slam = {b: {c: (noisy_slam[b] if c == "noisy_lidar"
                    else load_nav2(ORIGINAL_SLAM / f"{c}__{stem}.json"))
                for c in CONDITIONS}
            for b, stem in ((360, "nav2_slam"), (32, "nav2_slam_b32"))}
    for c in CONDITIONS:
        ref = full["run1"][c]["seeds"]
        assert all(full[p][c]["seeds"] == ref for p in PASSES), (c, "world order differs")
        assert all(slam[b][c]["seeds"] == ref for b in slam), (c, "world order differs")

    f6 = json.loads(PHASE_6F.read_text(encoding="utf-8"))["conditions"]
    h6 = json.loads(PHASE_6H.read_text(encoding="utf-8"))["conditions"]
    i6 = json.loads(PHASE_6I.read_text(encoding="utf-8"))["conditions"]
    hands = {
        "6g_32": ({c: np.array(f6[c]["cells"]["full_map"]["success_per_episode"]) for c in CONDITIONS},
                  {c: np.array(f6[c]["cells"]["matched"]["success_per_episode"]) for c in CONDITIONS},
                  32),
        "6g_360": ({c: np.array(f6[c]["cells"]["full_map"]["success_per_episode"]) for c in CONDITIONS},
                   {c: np.array(f6[c]["cells"]["matched"]["success_per_episode"]) for c in CONDITIONS},
                   360),
        "6h_360": ({c: np.array(h6[c]["success_per_episode"]["full_map"]) for c in CONDITIONS},
                   {c: np.array(h6[c]["success_per_episode"]["matched360"]) for c in CONDITIONS},
                   360),
        "6i_corroborated": ({c: np.array(h6[c]["success_per_episode"]["full_map"]) for c in CONDITIONS},
                            {c: np.array(i6[c]["success_per_episode"]["matched_corroborated"])
                             for c in CONDITIONS},
                            360),
    }
    out = {}
    for name, (hw_with, hw_without, beams) in hands.items():
        per_pass, labels = {}, set()
        for p in PASSES:
            point, ci, per = pooled_did({c: full[p][c]["success"] for c in CONDITIONS},
                                        {c: slam[beams][c]["success"] for c in CONDITIONS},
                                        hw_with, hw_without, rng, n_boot)
            label = classify(point, ci)
            labels.add(label)
            per_cond = {c: float(per[c].mean()) for c in CONDITIONS}
            per_pass[p] = {"did": point, "ci95": ci, "class": label, "per_condition": per_cond,
                           "largest": max(per_cond, key=per_cond.get)}
        out[name] = {"passes": per_pass,
                     "decision": labels.pop() if len(labels) == 1 else "UNRESOLVED"}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rerun-dir", default="results/nav2_noise_rerun")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/nav2_noise_rerun.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    rd = Path(args.rerun_dir)

    report: dict = {"prediction": PREDICTION, "checks": {}}

    # --- identity control: the original files reproduce the published points
    orig_full = {q: load_nav2(ORIGINAL_FULL[q] / "noisy_lidar__nav2.json") for q in PASSES}
    orig_slam = {360: load_nav2(ORIGINAL_SLAM / "noisy_lidar__nav2_slam.json"),
                 32: load_nav2(ORIGINAL_SLAM / "noisy_lidar__nav2_slam_b32.json")}
    identity = rescore(orig_full, orig_slam, rng, args.bootstrap)
    report["checks"]["identity_reproduces_published_points"] = all(
        round(identity[k]["passes"][q]["did"], 3) == v
        for k, pts in PUBLISHED.items() for q, v in pts.items())
    report["identity"] = {k: {q: round(v["passes"][q]["did"], 3) for q in PASSES}
                          for k, v in identity.items()}

    # --- the reruns
    full = {q: load_rerun(rd / q / "noisy_lidar__nav2.json") for q in PASSES}
    slam = {360: load_rerun(rd / "slam" / "noisy_lidar__nav2_slam.json"),
            32: load_rerun(rd / "slam" / "noisy_lidar__nav2_slam_b32.json")}
    for q in PASSES:
        assert full[q]["seeds"] == orig_full[q]["seeds"], (q, "worlds differ from the original")
    for b in slam:
        assert slam[b]["seeds"] == orig_slam[b]["seeds"], (b, "worlds differ from the original")
    report["checks"]["slam_ready_within_5pct"] = all(
        (slam[b]["not_ready"] or 0) <= 5 for b in slam)

    report["runs"] = {
        **{q: {"success": float(full[q]["success"].mean()), "collision": full[q]["collision"],
               "timeout": full[q]["timeout"], "commands": full[q]["commands"],
               "delivered_noise_std": full[q]["raw"]["delivered_noise_std"],
               "clean_success": float(orig_full[q]["success"].mean())} for q in PASSES},
        **{f"slam{b}": {"success": float(slam[b]["success"].mean()),
                        "collision": slam[b]["collision"], "timeout": slam[b]["timeout"],
                        "pose_error_median": slam[b]["pose_error_median"],
                        "commands": slam[b]["commands"],
                        "delivered_noise_std": slam[b]["raw"]["delivered_noise_std"],
                        "clean_success": float(orig_slam[b]["success"].mean()),
                        "clean_pose_error_median": orig_slam[b]["pose_error_median"]}
           for b in slam},
    }
    report["pose_vs_this_stack"] = pose_comparison(slam[360]["raw"])
    report["rescored"] = rescore(full, slam, rng, args.bootstrap)

    for name, r in report["runs"].items():
        print(f"{name:8s} success {r['success']:.3f} (clean {r['clean_success']:.3f})  "
              f"collision {r['collision']:.3f}  timeout {r['timeout']:.3f}  "
              f"noise delivered {r['delivered_noise_std']:.4f} m"
              + (f"  pose median {r['pose_error_median']:.3f} m "
                 f"(clean {r['clean_pose_error_median']:.3f})" if "pose_error_median" in r else ""))
    pc = report["pose_vs_this_stack"]
    print(f"pose, {pc['both_completed']} worlds both completed: Nav2 {pc['nav2_median']:.3f} m, "
          f"this stack {pc['this_stack_median']:.3f} m, ratio {pc['ratio']:.2f}, "
          f"Nav2 closer {pc['nav2_closer']} / {pc['this_stack_closer']}, "
          f"sign p {pc['sign_p']:.2g} -> {pc['decision']}")
    for name, r in report["rescored"].items():
        for q, v in r["passes"].items():
            print(f"{name:16s} {q}: {v['did']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                  f"{v['class']}  (published {PUBLISHED[name][q]:+.3f}; largest {v['largest']})")
        print(f"{name:16s} decision: {r['decision']}")
    print(f"checks: {report['checks']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
