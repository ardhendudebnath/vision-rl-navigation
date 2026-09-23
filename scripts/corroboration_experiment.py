"""Does weighing a cell's returns against the returns it should have had
repair the map under a dense, noisy scan?

Report §9.5 left this stack's mapper as the whole of what it pays at 360 beams,
and `dense_map_diagnostic.py` measured why, on the val band: under sensor noise
ten times the returns tripled the phantom cells -- occupied cells with no real
surface inside them -- from 104 to 370, blocked 46% of the arena where the truth
is 38%, and lost 29% of the free floor to inflation around things that are not
there. In clutter it found no phantoms at all at either beam count, so this is a
noise repair and not a clutter one; the clutter gap belongs to what the stack
does with an incomplete map, not to the map's fidelity.

The repair is one rule, in ``OccupancyMap``: a cell's hit is weighted by the
returns it received over the returns a surface there would have produced, which
is its width over the arc a beam separation subtends at its range. A lone return
among six crossing beams earns a sixth of a hit; a corroborated surface earns a
whole one. It is inert wherever beams are coarser than cells, so at 32 beams
nothing changes, and it is off by default so every published result stands.

Arms, on the six benchmark conditions, 100 held-out worlds each, all at 360
beams:

  mapped / matched                 own map with exact pose, and with its own
                                   pose -- Phase 6h's arms, which must reproduce
  mapped_corroborated / matched_   the same two with the rule on
  corroborated

    python scripts/corroboration_experiment.py --episodes 100
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
from nav2_slam_comparison import classify, load_nav2, pooled_did  # noqa: E402
from sensor_experiment import paired  # noqa: E402

from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402
from vision_nav.training.evaluate import evaluate  # noqa: E402

CONDITIONS = ("sparse", "large", "nominal", "noisy_lidar", "dense", "narrow")
PHASE_6H = Path("results/sensor_experiment.json")
#: The bounded-null half-width this project uses for a single contrast.
BOUND = 0.03

#: Recorded before the run and committed before any corroborated result on the
#: test worlds exists.
#:
#: DISCLOSED: derived from the val band. Twenty worlds per condition through
#: this script (--split val), and twelve through dense_map_diagnostic.py for the
#: mechanism.
#:
#:     val, 360 beams      own map          own map + own pose
#:     sparse              1.000 -> 1.000   1.000 -> 1.000
#:     large               1.000 -> 1.000   0.800 -> 0.850
#:     nominal             1.000 -> 1.000   0.950 -> 0.950
#:     noisy_lidar         0.800 -> 0.950   0.700 -> 0.900
#:     dense               0.650 -> 0.650   0.650 -> 0.700
#:     narrow              0.700 -> 0.650   0.600 -> 0.600
#:     pooled              +0.017           +0.050
#:
#: The mechanism is *not* the one the rule was designed around, and the
#: prediction says so. On val the phantom cells barely move -- 370 to 309 of
#: about 1100 occupied, and the free floor lost to inflation 0.285 to 0.270 --
#: while success goes 0.83 to 0.92 and replans 132 to 91. Making a phantom take
#: several scans to appear stops most of them from ever blocking a plan; it does
#: not stop them existing by the end of the episode.
PREDICTION = (
    "weighing a cell's returns against the returns a surface there would have "
    "produced repairs the dense-scan map under noise, and leaves every other "
    "condition alone. On noisy_lidar the own-map arm gains between +0.05 and "
    "+0.20 and the contrast is MATTERS, so the decision is REPAIRED; the "
    "own-pose arm gains between +0.05 and +0.25 there. On each of the other "
    "five conditions both arms move by at most 0.06 in absolute value. Pooled "
    "over the six, the own-map arm gains between 0.00 and +0.06 and the "
    "own-pose arm between +0.01 and +0.09. Against Nav2 at 360 beams the "
    "difference in costs with the repaired stack is smaller than Phase 6h's "
    "+0.112 and +0.120 by between 0.01 and 0.09, against both passes. "
    "Collisions stay at or below 0.05 on every condition and arm. The "
    "uncorroborated arms reproduce Phase 6h per episode on all six conditions. "
    "Decision on the own-map arm, noisy_lidar -- REPAIRED: MATTERS. "
    "INERT: INERT (bounded). HARMS: HARMS. Otherwise UNRESOLVED."
)


def run_condition(cond: str, split: str | None, episodes: int) -> dict:
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    overrides = {"lidar": {"noise_std": noise}} if noise else {}
    cfg = build_env_config(overrides, split=split or default_split(cond), shift=shift,
                           n_worlds=episodes)
    out: dict = {}
    for corroborate in (False, True):
        tag = "_corroborated" if corroborate else ""
        actor = MappedActor("lidar360", noise, cfg.robot)
        actor.agent.corroborate = corroborate
        _, results = evaluate(actor, cfg)
        out[f"mapped{tag}"] = {"success": [int(r.success) for r in results],
                               "collision": [int(r.collision) for r in results]}
        out.setdefault("seeds", [int(r.world_seed) for r in results])
        actor = LocalisedActor(f"matched{tag}", "lidar360", noise, cfg.robot,
                               OdometryConfig(), True)
        actor.agent.corroborate = corroborate
        _, results = evaluate(actor, cfg)
        actor.finish()
        out[f"matched{tag}"] = {"success": [int(r.success) for r in results],
                                "collision": [int(r.collision) for r in results],
                                **actor.diagnostics()}
    return out


def default_split(cond: str) -> str:
    return BENCHMARK_CONDITIONS[cond][0]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--split", default=None)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--nav2-full-dirs", nargs="+",
                   default=["results/nav2_runs/run1", "results/nav2_runs/run2"])
    p.add_argument("--nav2-slam-dir", default="results/nav2_slam_runs")
    p.add_argument("--nav2-suffix", default="")
    p.add_argument("--out", default="results/corroboration_experiment.json")
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {c: pool.submit(run_condition, c, args.split, args.episodes)
                   for c in CONDITIONS}
        runs = {c: f.result() for c, f in futures.items()}

    arms = ("mapped", "matched", "mapped_corroborated", "matched_corroborated")
    arr = {c: {a: np.array(runs[c][a]["success"]) for a in arms} for c in CONDITIONS}
    report: dict = {"prediction": PREDICTION, "episodes": args.episodes,
                    "split": args.split or "test", "bound": BOUND,
                    "conditions": {}, "checks": {}, "pooled": {}}

    if args.split is None and PHASE_6H.exists():
        before = json.loads(PHASE_6H.read_text(encoding="utf-8"))["conditions"]
        for arm, old in (("mapped", "mapped360"), ("matched", "matched360")):
            report["checks"][f"{arm}_reproduces_phase_6h"] = all(
                runs[c][arm]["success"] == before[c]["success_per_episode"][old]
                for c in CONDITIONS)

    for c in CONDITIONS:
        a = arr[c]
        report["conditions"][c] = {
            "success": {arm: float(v.mean()) for arm, v in a.items()},
            "collision": {arm: float(np.mean(runs[c][arm]["collision"])) for arm in arms},
            "pose_error_median": {
                "off": runs[c]["matched"]["pose_error_median"],
                "on": runs[c]["matched_corroborated"]["pose_error_median"]},
            "contrasts": {
                "mapped": paired(a["mapped"], a["mapped_corroborated"], rng, args.bootstrap),
                "matched": paired(a["matched"], a["matched_corroborated"], rng, args.bootstrap)},
            "success_per_episode": {arm: [int(x) for x in v] for arm, v in a.items()},
        }

    def pooled(a_arm, b_arm):
        d = {c: arr[c][b_arm] - arr[c][a_arm] for c in CONDITIONS}
        point = float(np.mean(np.concatenate(list(d.values()))))
        draws = [np.mean(np.concatenate([d[c][rng.integers(0, len(d[c]), len(d[c]))]
                                         for c in CONDITIONS])) for _ in range(args.bootstrap)]
        return {"gain": point, "ci95": [float(np.percentile(draws, 2.5)),
                                        float(np.percentile(draws, 97.5))]}

    report["pooled"]["mapped"] = pooled("mapped", "mapped_corroborated")
    report["pooled"]["matched"] = pooled("matched", "matched_corroborated")

    # Does the repair narrow what §9.5 measured against Nav2?
    nav_with = {d: {c: load_nav2(Path(d) / f"{c}__nav2{args.nav2_suffix}.json")
                    for c in CONDITIONS} for d in args.nav2_full_dirs}
    nav_without = {c: load_nav2(Path(args.nav2_slam_dir) / f"{c}__nav2_slam{args.nav2_suffix}.json")
                   for c in CONDITIONS}
    hw_with = {}
    if PHASE_6H.exists() and args.split is None:
        before = json.loads(PHASE_6H.read_text(encoding="utf-8"))["conditions"]
        hw_with = {c: np.array(before[c]["success_per_episode"]["full_map"]) for c in CONDITIONS}
    if hw_with:
        for arm in ("matched", "matched_corroborated"):
            per_pass = {}
            for d in args.nav2_full_dirs:
                point, ci, _ = pooled_did({c: nav_with[d][c]["success"] for c in CONDITIONS},
                                          {c: nav_without[c]["success"] for c in CONDITIONS},
                                          hw_with, {c: arr[c][arm] for c in CONDITIONS},
                                          rng, args.bootstrap)
                per_pass[d] = {"did": point, "ci95": ci, "class": classify(point, ci)}
            report["pooled"][f"did_vs_nav2_{arm}"] = per_pass

    noisy = report["conditions"]["noisy_lidar"]["contrasts"]["mapped"]
    report["decision"] = ("REPAIRED" if noisy["verdict"] == "MATTERS"
                          else "HARMS" if noisy["verdict"] == "HARMS"
                          else "INERT" if noisy["verdict"] == "INERT (bounded)"
                          else "UNRESOLVED")

    for c in CONDITIONS:
        e = report["conditions"][c]
        print(f"{c:12s} mapped {e['success']['mapped']:.3f} -> "
              f"{e['success']['mapped_corroborated']:.3f} "
              f"({e['contrasts']['mapped']['verdict']})   matched "
              f"{e['success']['matched']:.3f} -> {e['success']['matched_corroborated']:.3f} "
              f"({e['contrasts']['matched']['verdict']})", flush=True)
    for k in ("mapped", "matched"):
        v = report["pooled"][k]
        print(f"pooled {k:8s} {v['gain']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for k in ("did_vs_nav2_matched", "did_vs_nav2_matched_corroborated"):
        if k in report["pooled"]:
            for d, v in report["pooled"][k].items():
                print(f"{k} vs {Path(d).name}: {v['did']:+.3f} "
                      f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] {v['class']}")
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
