"""Do the clutter failures land on the worlds with no margin-safe route?

`margin_audit.py` found that a fifth to a quarter of val clutter worlds admit no
route at all at ``robot_radius + safety_margin``, while every one of them admits a
route at the bare radius. `clutter_forensic.py` recorded which worlds the
published stack actually fails on. This joins the two by seed.

The comparison is between worlds, not between arms, so the test is a two-sided
Fisher exact test on the 2x2 table -- margin-safe route or not, against arrived
or not. Nothing here is paired and nothing is bootstrapped.

**This is exploratory.** It was found by looking at val data rather than
registered in advance, and it is reported as a lead rather than a result: the
question it raises -- whether relaxing the margin where no margin-safe route
exists recovers those episodes -- is a separate experiment that has to be
registered before it is run.

    python scripts/margin_overlap.py
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the same margins
    whose probability is no greater than the observed one.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c
    total = comb(n, col1)

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / total

    observed = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", default="results/margin_audit.json")
    p.add_argument("--forensic", default="results/clutter_forensic.json")
    p.add_argument("--out", default="results/margin_overlap.json")
    args = p.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))["conditions"]
    forensic = json.loads(Path(args.forensic).read_text(encoding="utf-8"))["conditions"]
    report: dict = {"split": "val", "conditions": {}}

    print("Worlds with no route at the planner's full margin, against arriving\n")
    pooled = [0, 0, 0, 0]
    for cond, fc in forensic.items():
        if cond not in audit:
            continue
        safe = {r["seed"]: r["l_full"] is not None for r in audit[cond]["worlds"]}
        rows = [(safe[e["seed"]], e["success"]) for e in fc["episodes"]
                if e["seed"] in safe]
        # a: margin-safe and arrived, b: margin-safe and failed,
        # c: no margin-safe route and arrived, d: no margin-safe route and failed
        a = sum(1 for s, ok in rows if s and ok)
        b = sum(1 for s, ok in rows if s and not ok)
        c = sum(1 for s, ok in rows if not s and ok)
        d = sum(1 for s, ok in rows if not s and not ok)
        pv = fisher_exact(a, b, c, d)
        sr_safe = a / (a + b) if a + b else float("nan")
        sr_tight = c / (c + d) if c + d else float("nan")
        report["conditions"][cond] = {
            "margin_safe": {"n": a + b, "success": sr_safe},
            "no_margin_safe": {"n": c + d, "success": sr_tight},
            "table": [a, b, c, d], "fisher_p": pv,
        }
        for i, v in enumerate((a, b, c, d)):
            pooled[i] += v
        print(f"{cond:8s} margin-safe: {a}/{a + b} arrive ({sr_safe:.2f})   "
              f"no margin-safe route: {c}/{c + d} arrive ({sr_tight:.2f})   "
              f"Fisher p={pv:.3f}")

    a, b, c, d = pooled
    pv = fisher_exact(a, b, c, d)
    sr_safe = a / (a + b) if a + b else float("nan")
    sr_tight = c / (c + d) if c + d else float("nan")
    report["pooled"] = {
        "margin_safe": {"n": a + b, "success": sr_safe},
        "no_margin_safe": {"n": c + d, "success": sr_tight},
        "table": [a, b, c, d], "fisher_p": pv,
    }
    print(f"\npooled   margin-safe: {a}/{a + b} arrive ({sr_safe:.2f})   "
          f"no margin-safe route: {c}/{c + d} arrive ({sr_tight:.2f})   "
          f"Fisher p={pv:.3f}")
    share = d / (b + d) if b + d else float("nan")
    report["share_of_failures_tight"] = share
    print(f"         {d} of {b + d} failures ({share:.0%}) are on worlds with no "
          f"route at the margin, which are {c + d} of {a + b + c + d} worlds")

    # What the two kinds of failure look like. The point of splitting them: a
    # robot still closing on the goal when the clock stops is failing at the
    # step budget, and one that reached its closest point early and then gave
    # ground back is failing at something else.
    groups: dict[str, list[dict]] = {"tight": [], "roomy": []}
    for cond, fc in forensic.items():
        if cond not in audit:
            continue
        safe = {r["seed"]: r["l_full"] is not None for r in audit[cond]["worlds"]}
        for e in fc["episodes"]:
            if e["seed"] in safe and not e["success"]:
                groups["tight" if not safe[e["seed"]] else "roomy"].append(e)
    keys = ("stall_fraction", "given_back", "best_remaining", "final_remaining",
            "progress_steps", "backward_steps", "replans", "phantom_route",
            "goal_plans_after", "full_margin_steps", "min_plan_radius")
    report["failure_groups"] = {}
    print("\nthe two kinds of failure, pooled over dense and narrow:")
    for name, rows in groups.items():
        if not rows:
            continue
        e = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
        # Still closing on the goal when the clock stopped.
        e["still_closing"] = sum(1 for r in rows if r["stall_fraction"] < 0.2) / len(rows)
        e["n"] = len(rows)
        report["failure_groups"][name] = e
        label = ("no margin-safe route" if name == "tight" else "margin-safe world")
        print(f"  {label:22s} n={e['n']:2d}  stall {e['stall_fraction']:.2f}  "
              f"gave back {e['given_back']:.2f} m  best {e['best_remaining']:.2f} m  "
              f"fwd/back {e['progress_steps']:.0f}/{e['backward_steps']:.0f}  "
              f"replans {e['replans']:.0f}  still closing at the buzzer "
              f"{e['still_closing']:.0%}  full route held {e['goal_plans_after']:.0%}")
        print(f"  {'':22s}    planned at the full margin "
              f"{e['full_margin_steps']:.0%} of steps, and the ladder never went "
              f"below {e['min_plan_radius']:.2f} m")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
