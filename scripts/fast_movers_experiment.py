"""Does frame stacking earn its keep once the movers outrun the robot?

Phase 3h found frame stacking inert on moving obstacles: -0.008 (p = 0.784)
and +0.013 (p = 0.703). The leading explanation offered was that the movers
were simply too slow -- 0.15-0.45 m/s against a 0.6 m/s robot -- so there was
nothing worth anticipating. `dynamic_fast` raises them to 0.8-1.5 m/s, which
outruns the robot, on worlds that are geometrically identical seed for seed.

Both arms are also scored on the slow condition as a **control**. Frame
stacking should stay inert there. Without that cell, "stacking helps" cannot
be told apart from "these particular runs came out better" -- the failure mode
that the churn and horizon experiments each walked into and each of their
controls caught.

    python scripts/fast_movers_experiment.py \\
        --arm "stack1=runs/dynfast1_s0,..." --arm "stack4=runs/dynfast4_s0,..."
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from vision_nav.training.actors import build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate
from vision_nav.training.run_spec import env_overrides_for_run

#: (label, split, shift). The slow row is the control.
CONDITIONS = [
    ("fast (primary)", "test_ood", "dynamic_fast"),
    ("slow (control)", "test_ood", "dynamic"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True,
                   metavar="NAME=DIR[,DIR...]")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/fast_movers.json")
    return p.parse_args(argv)


def permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation test on the difference of means.

    Seed is the unit of analysis, which is the whole point: Section 6 of the
    report documents what happens when episodes are used instead.
    """
    pooled = np.concatenate([a, b])
    n = len(a)
    observed = abs(a.mean() - b.mean())
    hits = total = 0
    for idx in combinations(range(len(pooled)), n):
        left = pooled[list(idx)]
        right = np.delete(pooled, list(idx))
        total += 1
        if abs(left.mean() - right.mean()) >= observed - 1e-12:
            hits += 1
    return hits / total


def main(argv=None) -> int:
    args = parse_args(argv)
    arms: dict[str, list[Path]] = {}
    for spec in args.arm:
        name, dirs = spec.split("=", 1)
        runs = [Path(d) for d in dirs.split(",")]
        for r in runs:
            if not (r / "best_model.zip").exists():
                raise FileNotFoundError("no best_model.zip in {}".format(r))
        arms[name] = runs

    report = {"episodes": args.episodes, "conditions": {}}
    for label, split, shift in CONDITIONS:
        print("\n=== {} : {} ===".format(label, shift))
        per_arm: dict[str, list[float]] = {}
        for name, runs in arms.items():
            succ = []
            for run in runs:
                cfg = build_env_config(env_overrides_for_run(run), split=split,
                                       shift=shift, n_worlds=args.episodes)
                actor = build_actor("rl", model_path=str(run / "best_model.zip"),
                                    robot=cfg.robot)
                m, _ = evaluate(actor, cfg)
                succ.append(m.success_rate)
            per_arm[name] = succ
            print("  {:8s} per-seed {}".format(
                name, [round(v, 3) for v in succ]))
            print("           mean {:.3f} +/- {:.3f}".format(
                float(np.mean(succ)), float(np.std(succ, ddof=1))))

        names = list(per_arm)
        entry = {n: per_arm[n] for n in names}
        if len(names) == 2:
            a, b = np.array(per_arm[names[0]]), np.array(per_arm[names[1]])
            delta = float(b.mean() - a.mean())
            p = permutation_p(a, b)
            above = int((b[:, None] > a[None, :]).mean() * len(a))
            entry.update({"delta": delta, "p": p})
            print("  {} - {} = {:+.3f}   p = {:.3f} (exact, seed as unit)".format(
                names[1], names[0], delta, p))
        report["conditions"][shift] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fast = report["conditions"].get("dynamic_fast", {})
    slow = report["conditions"].get("dynamic", {})
    if "delta" in fast and "delta" in slow:
        print("\n=== verdict ===")
        helped = fast["delta"] >= 0.05 and fast["p"] < 0.05
        control_flat = abs(slow["delta"]) <= 0.03
        if helped and control_flat:
            print("PREDICTION HELD. Stacking pays once the movers outrun the "
                  "robot and stays inert when they do not, so mover speed was "
                  "why Phase 3h saw nothing.")
        elif helped and not control_flat:
            print("AMBIGUOUS. Stacking helps on fast, but the slow control "
                  "moved by {:+.3f} too, so this may be the arms rather than "
                  "the speed.".format(slow["delta"]))
        else:
            print("PREDICTION FAILED. Stacking is inert at {:+.3f} "
                  "(p = {:.3f}) even against movers that outrun the robot. "
                  "The speed explanation for Phase 3h is dead.".format(
                      fast["delta"], fast["p"]))
    print("\nWrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
