"""Is it the sensor for this stack too?

Report §9.4 asked Nav2 with slam_toolbox what the map and the pose were worth,
and found that with the hand-written stack's 32-beam scanner it pays at least
as much as the hand-written stack, while with a 360-beam scanner it pays about
a seventh. It concluded that what the privileges stood in for was mostly the
sensor -- with one cell inferred rather than measured: this project's own stack
had never been run at 360 beams. This runs it.

The stack is unchanged: same planner, same mapper, same odometry model, same
scan matcher with the same prior weights, which were tuned on the val band at
32 beams and are deliberately not re-tuned. Nav2's slam_toolbox kept its
parameters between 32 and 360 beams as well; changing only the sensor is the
comparison §9.4 made, and re-tuning here would make it a different one.

Arms, on the six benchmark conditions, 100 held-out worlds each:

  full_map     the published classical baseline, given the map and the pose
  mapped32     own map, exact pose, 32 beams      -- must reproduce Phase 6e/6f
  matched32    own map, own pose, 32 beams        -- must reproduce Phase 6f
  mapped360    own map, exact pose, 360 beams
  matched360   own map, own pose, 360 beams       -- the missing cell

The primary statistic is §9.4's difference in costs, now with the sensor held
the same on both sides: what Nav2 at 360 beams loses without its privileges,
minus what this stack at 360 beams loses. Against both published Nav2
full-privilege passes, as before.

    python scripts/sensor_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localisation_experiment import LocalisedActor, MappedActor  # noqa: E402
from nav2_slam_comparison import BOUND, classify, load_nav2, pooled_did  # noqa: E402
from spacetime_experiment import verdict  # noqa: E402
from velocity_experiment import mcnemar_p  # noqa: E402

from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.training.actors import build_actor  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402
from vision_nav.training.evaluate import evaluate  # noqa: E402

CONDITIONS = ("sparse", "large", "nominal", "noisy_lidar", "dense", "narrow")
PHASE_6F = Path("results/localisation_experiment.json")

#: Recorded before the run and committed before any 360-beam result on the test
#: worlds exists.
#:
#: DISCLOSED: derived from the val band, twenty worlds per condition, run through
#: this script with --split val against the val-band Nav2 runs made for §9.4.
#:
#:     val, success   full   own map 32  +pose 32   own map 360  +pose 360   Nav2 360
#:     sparse         1.00   1.00        0.35       1.00         1.00         1.00
#:     large          1.00   1.00        0.60       1.00         0.80         0.95
#:     nominal        1.00   1.00        0.70       1.00         0.95         1.00
#:     noisy_lidar    1.00   0.95        0.75       0.80         0.70         1.00
#:     dense          0.85   0.70        0.65       0.65         0.65         0.70
#:     narrow         0.85   0.65        0.65       0.70         0.60         0.65
#:
#: Pooled, val: this stack's cost 0.333 at 32 beams and 0.167 at 360; the
#: dense scanner's effect +0.167 [+0.092, +0.242]; pose cost 0.267 -> 0.075,
#: map cost 0.067 -> 0.092. Difference in costs against Nav2 at 360: +0.092
#: [+0.017, +0.167], per condition sparse 0.00, large +0.15, nominal +0.05,
#: noisy_lidar +0.30, dense -0.05, narrow +0.10.
#:
#: The primary decision is close to a coin flip and the prediction says so
#: rather than hiding it: val's point sits just under the 0.10 materiality line,
#: and scaling val to the test band's known costs (this stack 0.290 at 32 beams
#: against val's 0.333; Nav2 0.043 at 360 against val's 0.075) puts it at about
#: +0.10. What the prediction commits to is the band, and what it excludes.
PREDICTION = (
    "a dense scanner halves this stack's cost of losing the map and the pose, "
    "and all of the gain is the pose; it does not close the gap to Nav2, and "
    "what is left is mostly this stack's map under sensor noise. The difference "
    "in costs against Nav2 at 360 beams is between +0.05 and +0.15 against both "
    "passes, so the decision is IMPLEMENTATION or UNRESOLVED and not SENSOR or "
    "BACKWARDS. Of the six conditions, noisy_lidar contributes the largest "
    "difference. This stack's pooled cost is between 0.08 and 0.22 at 360 "
    "beams; the scanner's pooled effect on it, matched360 minus matched32, is "
    "between +0.08 and +0.25 with a CI above zero. The pose cost at 360 beams, "
    "matched360 minus mapped360, is at most 0.12 and below the 32-beam pose "
    "cost; the map cost at 360, full_map minus mapped360, is no smaller than "
    "the 32-beam map cost less 0.03. On noisy_lidar the 360-beam map does worse "
    "than the 32-beam one: mapped360 below mapped32. matched360 scores at least "
    "0.90 on sparse. Median final pose error at 360 beams is at most 0.15 m on "
    "every condition except noisy_lidar, and below the 32-beam median on "
    "sparse, large, nominal and noisy_lidar. full_map, mapped32 and matched32 "
    "reproduce Phase 6f per episode on all six conditions. "
    "Decision on the difference in costs, sensor held at 360 on both sides -- "
    "IMPLEMENTATION: at least +0.10 with CI above 0. SENSOR: CI inside +-0.10. "
    "BACKWARDS: at most -0.10 with CI below 0. Otherwise UNRESOLVED. A label "
    "stands only if it holds against both published Nav2 passes."
)


def run_condition(cond: str, split: str | None, episodes: int) -> dict:
    """Every hand-written arm on one condition. One process per condition:
    each episode is seeded by its world, so the result does not depend on how
    the conditions are spread over processes."""
    default_split, shift, noise = BENCHMARK_CONDITIONS[cond]
    overrides = {"lidar": {"noise_std": noise}} if noise else {}
    cfg = build_env_config(overrides, split=split or default_split, shift=shift,
                           n_worlds=episodes)
    out: dict = {}

    _, results = evaluate(build_actor("classical", robot=cfg.robot), cfg)
    out["seeds"] = [int(r.world_seed) for r in results]
    out["full_map"] = {"success": [int(r.success) for r in results],
                       "collision": [int(r.collision) for r in results]}
    for beams in (32, 360):
        sensor = f"lidar{beams}"
        _, results = evaluate(MappedActor(sensor, noise, cfg.robot), cfg)
        out[f"mapped{beams}"] = {"success": [int(r.success) for r in results],
                                 "collision": [int(r.collision) for r in results]}
        actor = LocalisedActor(f"matched{beams}", sensor, noise, cfg.robot,
                               OdometryConfig(), True)
        _, results = evaluate(actor, cfg)
        actor.finish()
        out[f"matched{beams}"] = {"success": [int(r.success) for r in results],
                                  "collision": [int(r.collision) for r in results],
                                  "pose_error": [float(e) for e in actor.errors],
                                  **actor.diagnostics()}
    return out


def paired(a, b, rng, n_boot):
    """b minus a on the same worlds: McNemar, bootstrap CI, verdict."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    p, lost, won = mcnemar_p(a, b)
    draws = [b[i].mean() - a[i].mean()
             for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    gain = float(b.mean() - a.mean())
    return {"gain": gain, "p": p, "won": won, "lost": lost, "ci95": ci,
            "verdict": verdict(gain, p, ci)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--split", default=None, help="'val' while developing")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nav2-full-dirs", nargs="+",
                   default=["results/nav2_runs/run1", "results/nav2_runs/run2"])
    p.add_argument("--nav2-slam-dir", default="results/nav2_slam_runs")
    p.add_argument("--nav2-suffix", default="")
    p.add_argument("--out", default="results/sensor_experiment.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {c: pool.submit(run_condition, c, args.split, args.episodes)
                   for c in CONDITIONS}
        runs = {c: f.result() for c, f in futures.items()}

    arr = {c: {arm: np.array(runs[c][arm]["success"])
               for arm in ("full_map", "mapped32", "matched32", "mapped360", "matched360")}
           for c in CONDITIONS}
    report: dict = {"prediction": PREDICTION, "episodes": args.episodes,
                    "split": args.split or "test", "bound": BOUND,
                    "conditions": {}, "checks": {}, "pooled": {}}

    # --- controls: the 32-beam arms are Phase 6e/6f's, and must reproduce them
    if args.split is None and PHASE_6F.exists():
        before = json.loads(PHASE_6F.read_text(encoding="utf-8"))["conditions"]
        for arm, old in (("full_map", "full_map"), ("mapped32", "mapped"), ("matched32", "matched")):
            report["checks"][f"{arm}_reproduces_phase_6f"] = all(
                runs[c][arm]["success"] == before[c]["cells"][old]["success_per_episode"]
                for c in CONDITIONS)

    # --- Nav2, from the files §9.4 was scored on
    nav_with = {d: {c: load_nav2(Path(d) / f"{c}__nav2{args.nav2_suffix}.json")
                    for c in CONDITIONS} for d in args.nav2_full_dirs}
    nav_without = {c: load_nav2(Path(args.nav2_slam_dir) / f"{c}__nav2_slam{args.nav2_suffix}.json")
                   for c in CONDITIONS}
    for c in CONDITIONS:
        for d in args.nav2_full_dirs:
            assert nav_with[d][c]["seeds"] == runs[c]["seeds"], (c, d, "world order differs")
        assert nav_without[c]["seeds"] == runs[c]["seeds"], (c, "world order differs")

    for c in CONDITIONS:
        a = arr[c]
        m360 = runs[c]["matched360"]
        report["conditions"][c] = {
            "success": {arm: float(v.mean()) for arm, v in a.items()},
            "collision": {arm: float(np.mean(runs[c][arm]["collision"])) for arm in a},
            "pose_error_median": {"32": runs[c]["matched32"]["pose_error_median"],
                                  "360": m360["pose_error_median"]},
            "nav2": {"with": {d: float(nav_with[d][c]["success"].mean()) for d in args.nav2_full_dirs},
                     "slam360": float(nav_without[c]["success"].mean())},
            "contrasts": {
                # What the dense sensor does for this stack, privileges gone.
                "matched360_vs_matched32": paired(a["matched32"], a["matched360"], rng, args.bootstrap),
                # The map and the pose at 360 beams, as §9.2 and §9.3 at 32.
                "mapped360_vs_full": paired(a["full_map"], a["mapped360"], rng, args.bootstrap),
                "matched360_vs_mapped360": paired(a["mapped360"], a["matched360"], rng, args.bootstrap),
                # The stacks directly, both without privileges, both at 360.
                "nav2slam360_vs_matched360": paired(a["matched360"], nav_without[c]["success"],
                                                    rng, args.bootstrap),
            },
            "success_per_episode": {arm: [int(x) for x in v] for arm, v in a.items()},
        }

    # --- the primary: difference in costs, sensor held at 360 on both sides
    hw_with = {c: arr[c]["full_map"] for c in CONDITIONS}
    hw_without = {c: arr[c]["matched360"] for c in CONDITIONS}
    per_pass, labels = {}, set()
    for d in args.nav2_full_dirs:
        point, ci, per = pooled_did({c: nav_with[d][c]["success"] for c in CONDITIONS},
                                    {c: nav_without[c]["success"] for c in CONDITIONS},
                                    hw_with, hw_without, rng, args.bootstrap)
        label = classify(point, ci)
        labels.add(label)
        per_pass[d] = {"did": point, "ci95": ci, "class": label,
                       "per_condition": {c: float(per[c].mean()) for c in CONDITIONS}}
    report["pooled"]["did_vs_nav2_360"] = per_pass
    label = labels.pop() if len(labels) == 1 else "UNRESOLVED"
    report["decision"] = {"IMPLEMENTATION": "IMPLEMENTATION", "PROBLEM": "SENSOR",
                          "BACKWARDS": "BACKWARDS"}.get(label, "UNRESOLVED")

    # --- secondary: the sensor's effect on this stack, pooled
    def pooled_gain(a_arm, b_arm):
        d = {c: arr[c][b_arm] - arr[c][a_arm] for c in CONDITIONS}
        point = float(np.mean(np.concatenate(list(d.values()))))
        draws = [np.mean(np.concatenate([d[c][rng.integers(0, len(d[c]), len(d[c]))]
                                         for c in CONDITIONS])) for _ in range(args.bootstrap)]
        return {"gain": point, "ci95": [float(np.percentile(draws, 2.5)),
                                        float(np.percentile(draws, 97.5))]}

    report["pooled"]["cost32"] = pooled_gain("full_map", "matched32")
    report["pooled"]["cost360"] = pooled_gain("full_map", "matched360")
    report["pooled"]["sensor_effect"] = pooled_gain("matched32", "matched360")
    report["pooled"]["map_cost360"] = pooled_gain("full_map", "mapped360")
    report["pooled"]["pose_cost360"] = pooled_gain("mapped360", "matched360")
    report["pooled"]["map_cost32"] = pooled_gain("full_map", "mapped32")
    report["pooled"]["pose_cost32"] = pooled_gain("mapped32", "matched32")

    for c in CONDITIONS:
        s = report["conditions"][c]["success"]
        n = report["conditions"][c]["nav2"]
        print(f"{c:12s} full {s['full_map']:.3f}  mapped32 {s['mapped32']:.3f}  "
              f"matched32 {s['matched32']:.3f}  mapped360 {s['mapped360']:.3f}  "
              f"matched360 {s['matched360']:.3f}  | nav2 slam360 {n['slam360']:.3f}", flush=True)
    for k, v in report["pooled"].items():
        if k == "did_vs_nav2_360":
            for d, x in v.items():
                print(f"difference in costs vs {Path(d).name}: {x['did']:+.3f} "
                      f"[{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] {x['class']}")
        else:
            print(f"{k:14s} {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print(f"checks: {report['checks']}")
    print(f"decision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
