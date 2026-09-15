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


def tail_gain(runs: list[Path]) -> list[float]:
    """Per-seed change in validation success over the last third of training.

    A null result and an under-trained arm predict the same number, and the
    two are not distinguishable from the endpoint alone. This is the cheapest
    way to tell them apart: if an arm is still climbing when its budget runs
    out, "it did not help" has not yet been shown -- what has been shown is
    that it had not finished. Near zero means the comparison is between two
    converged arms and the endpoint means what it says.

    Compares the mean of the final third of validation evaluations against the
    third before it, so it is robust to the noise in any single evaluation.

    What counts as "near zero" is measured, not guessed. The four arms that
    have run this budget to completion read +0.026 (dynfast1), +0.001
    (dynfast4), +0.016 (dyn1) and +0.010 (dynvel), on 50-episode validations
    whose per-seed noise is itself around +/-0.1. So a mean tail gain inside
    +/-0.03 is indistinguishable from an arm everyone already treats as
    converged, and a threshold tighter than that would flag dynfast1 -- the
    baseline of this very comparison -- as under-trained.
    """
    gains = []
    for run in runs:
        history = json.loads((run / "validation_history.json").read_text(encoding="utf-8"))
        rates = [h["success_rate"] for h in history]
        third = max(len(rates) // 3, 1)
        gains.append(float(np.mean(rates[-third:]) - np.mean(rates[-2 * third : -third])))
    return gains


def main(argv=None) -> int:
    args = parse_args(argv)
    arms: dict[str, list[Path]] = {}
    for spec in args.arm:
        name, dirs = spec.split("=", 1)
        runs = [Path(d) for d in dirs.split(",")]
        for r in runs:
            if not (r / "best_model.zip").exists():
                raise FileNotFoundError(f"no best_model.zip in {r}")
        arms[name] = runs

    report = {"episodes": args.episodes, "conditions": {}}

    # Reported before the endpoints, so a rising tail is seen while reading
    # the result rather than after having believed it.
    print("=== still improving at the end of the budget? ===")
    report["tail_gain"] = {}
    for name, runs in arms.items():
        gains = tail_gain(runs)
        report["tail_gain"][name] = gains
        mean = float(np.mean(gains))
        verdict = "converged" if abs(mean) <= 0.03 else "STILL MOVING"
        print(f"  {name:11s} final-third minus previous-third success: "
              f"{mean:+.3f}  ({verdict})")
    for label, split, shift in CONDITIONS:
        print(f"\n=== {label} : {shift} ===")
        per_arm: dict[str, list[float]] = {}
        outcomes: dict[str, dict[str, list[float]]] = {}
        for name, runs in arms.items():
            succ, coll, tmo = [], [], []
            for run in runs:
                cfg = build_env_config(env_overrides_for_run(run), split=split,
                                       shift=shift, n_worlds=args.episodes)
                actor = build_actor("rl", model_path=str(run / "best_model.zip"),
                                    robot=cfg.robot)
                m, _ = evaluate(actor, cfg)
                succ.append(m.success_rate)
                coll.append(m.collision_rate)
                tmo.append(m.timeout_rate)
            per_arm[name] = succ
            outcomes[name] = {"success": succ, "collision": coll, "timeout": tmo}
            print(f"  {name:11s} per-seed {[round(v, 3) for v in succ]}")
            print(f"              mean {float(np.mean(succ)):.3f} +/- {float(np.std(succ, ddof=1)):.3f}")

        # Success alone cannot tell "the treatment supplies nothing" from "the
        # treatment supplies something the reward will not let the policy
        # spend". Phase 5h is the precedent and the warning: at the 4:1
        # collision-to-timeout ratio, frame stacking moved success by -0.027 --
        # readable as inert -- while cutting collisions 0.018 and raising
        # timeouts 0.045. The information was being used the whole time. Every
        # arm here trains under that same 4:1 reward, so the breakdown is not
        # extra colour; it is what makes the endpoint interpretable.
        print(f"  {'arm':11s} {'success':>8s} {'collision':>10s} {'timeout':>8s}")
        for name in arms:
            o = outcomes[name]
            print(f"  {name:11s} {np.mean(o['success']):8.3f} "
                  f"{np.mean(o['collision']):10.3f} {np.mean(o['timeout']):8.3f}")

        names = list(per_arm)
        entry = {n: per_arm[n] for n in names}
        entry["outcomes"] = outcomes
        if len(names) == 2:
            a, b = np.array(per_arm[names[0]]), np.array(per_arm[names[1]])
            delta = float(b.mean() - a.mean())
            p = permutation_p(a, b)
            # Both arms use training seeds 0-5, so index-pairing compares the
            # same seed under the two treatments. A clean 6/6 is worth seeing
            # next to the mean, since a small delta carried by every seed says
            # something different from one carried by two outliers.
            paired = sum(1 for x, y in zip(a, b, strict=True) if y > x)
            entry.update({"delta": delta, "p": p, "seeds_higher": paired})
            print(f"  {names[1]} - {names[0]} = {delta:+.3f}   p = {p:.3f} "
                  f"(exact, seed as unit)   {paired}/{len(a)} seeds higher")
        report["conditions"][shift] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fast = report["conditions"].get("dynamic_fast", {})
    slow = report["conditions"].get("dynamic", {})
    if "delta" in fast and "delta" in slow:
        # Deliberately generic: this script runs more than one comparison, and
        # a verdict hard-coded to the first one printed a conclusion about
        # mover speed underneath a table about observation encoding.
        #
        # Taken from the arm list rather than by excluding the statistic keys.
        # Excluding them was already wrong -- "seeds_higher" was never in the
        # exclusion list, so the header silently fell back to the placeholder
        # the moment that key was added, and every later key would have done
        # the same.
        names = list(arms)
        pair = " - ".join(reversed(names)) if len(names) == 2 else "treated - control"
        print(f"\n=== verdict ({pair}) ===")
        helped = fast["delta"] >= 0.05 and fast["p"] < 0.05
        control_moved = abs(slow["delta"]) > 0.03
        if helped and not control_moved:
            print("EFFECT IS SPECIFIC TO FAST MOVERS: {:+.3f} (p = {:.3f}) on "
                  "fast, {:+.3f} on the slow control. Whatever the treatment "
                  "supplies, it is used for motion.".format(
                      fast["delta"], fast["p"], slow["delta"]))
        elif helped and control_moved:
            print("EFFECT IS NOT SPECIFIC: {:+.3f} on fast but {:+.3f} on the "
                  "slow control too, so it is a general property of the arm "
                  "rather than anything to do with mover speed.".format(
                      fast["delta"], slow["delta"]))
        else:
            print("NO EFFECT ON THE PRIMARY ENDPOINT: {:+.3f} (p = {:.3f}) on "
                  "fast, {:+.3f} (p = {:.3f}) on the slow control.".format(
                      fast["delta"], fast["p"], slow["delta"], slow["p"]))

            # A flat endpoint has two very different causes, and Phase 5h shows
            # this project cannot tell them apart by eye: stacking read as
            # inert at 4:1 (-0.027) while cutting collisions 0.018 and raising
            # timeouts 0.045. Anything of that size here means the treatment
            # supplies usable information that the reward converts to stalling,
            # and the clean test is a retrain at 1:1 rather than a null.
            outs = fast.get("outcomes", {})
            if len(names) == 2 and all(n in outs for n in names):
                base, treated = (outs[n] for n in names)
                d_coll = float(np.mean(treated["collision"]) - np.mean(base["collision"]))
                d_tmo = float(np.mean(treated["timeout"]) - np.mean(base["timeout"]))
                print(f"  outcome shift: collisions {d_coll:+.3f}, "
                      f"timeouts {d_tmo:+.3f}")
                if d_coll <= -0.015 and d_tmo >= 0.015:
                    print("  BUT THIS IS THE PHASE 5h SIGNATURE, NOT AN ABSENT "
                          "EFFECT: collisions fall and timeouts rise by as much "
                          "or more, so the information is being used and the 4:1 "
                          "reward is spending it on stalling. Do not report this "
                          "as 'no effect'; retrain both arms at 1:1 to price it.")
                else:
                    print("  and the breakdown agrees: no collision saving "
                          "hidden behind the flat success rate.")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
