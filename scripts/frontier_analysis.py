"""Does planning through known free space only pay, on the val band?

Reads ``results/frontier_diagnostic.json`` -- the paired run of ``as_published``
against ``frontier`` written by ``clutter_diagnostic.py`` -- and tests the three
claims registered in ``clutter_diagnostic.PREDICTION`` before that run existed.

The arms are deterministic given a world seed and both arms run the same seeds,
so the unit is the episode and the test is exact McNemar on the discordant
pairs, as everywhere else in this project. An episode both arms solve, or both
fail, carries no information about the difference between them.

    python scripts/frontier_analysis.py

One caveat is stated here rather than buried, because it weakens the second
registered endpoint before any number is read. ``replans`` does not mean the
same thing in the two arms. A frontier route ends one cell into the unseen, so
the agent rebuilds it each time it consumes one -- that is the cadence of
incremental exploration, not the indecision §9.7 measured. Registering a
quarter cut in a metric that the treatment redefines was a mistake in the
pre-registration, and the honest reading is the one restricted to episodes that
arrive, where both arms are doing the same job.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

#: The null band used for every success contrast in this project (§9.1).
BOUND = 0.03
#: Registered thresholds, repeated here so this script can be read alone. The
#: authority is ``clutter_diagnostic.PREDICTION``, committed before the run.
REGISTERED = {
    "clutter_success_gain_at_least": 0.05,
    "clutter_replan_reduction_at_least": 0.25,
    "nominal_success_loss_at_most": 0.03,
}
CLUTTER = ("dense", "narrow")


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar on paired binary outcomes."""
    b_only = int(np.sum(~a & b))
    a_only = int(np.sum(a & ~b))
    n = a_only + b_only
    if n == 0:
        return 1.0, a_only, b_only
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), a_only, b_only


def verdict(gain: float, p: float, ci: list[float], bound: float = BOUND) -> str:
    """The four-outcome rule from ``spacetime_experiment.verdict``."""
    if p < 0.05 and gain >= bound:
        return "MATTERS"
    if p < 0.05 and gain <= -bound:
        return "HARMS"
    if ci[0] > -bound and ci[1] < bound:
        return "INERT (bounded)"
    return "inconclusive"


def paired_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int) -> list[float]:
    """95% CI of mean(b) - mean(a), resampling episodes jointly."""
    n = len(a)
    diffs = [b[i].mean() - a[i].mean()
             for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def ratio_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int) -> list[float]:
    """95% CI of the fractional reduction 1 - mean(b)/mean(a), paired."""
    n = len(a)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        base = a[i].mean()
        out.append(1.0 - b[i].mean() / base if base else np.nan)
    out = np.asarray(out, dtype=float)
    out = out[np.isfinite(out)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def field(entry: dict, key: str) -> np.ndarray:
    return np.array([e[key] for e in entry["episodes"]], dtype=float)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/frontier_diagnostic.json")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/frontier_analysis.json")
    args = p.parse_args(argv)

    data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260926)
    conditions = data["conditions"]
    report: dict = {"split": data["split"], "episodes": data["episodes"],
                    "registered": REGISTERED, "conditions": {}}

    print(f"frontier vs as_published -- {data['split']} band, "
          f"{data['episodes']} worlds per condition\n")
    pooled: dict[str, list[np.ndarray]] = {"a": [], "b": [], "ra": [], "rb": []}
    for cond, arms in conditions.items():
        base, arm = arms["as_published"], arms["frontier"]
        assert arms["as_published"]["seeds"] == arm["seeds"], "arms must share seeds"
        a = field(base, "success").astype(bool)
        b = field(arm, "success").astype(bool)
        gain = float(b.mean() - a.mean())
        pv, a_only, b_only = mcnemar_p(a, b)
        ci = paired_ci(a.astype(float), b.astype(float), rng, args.bootstrap)
        entry = {"success": [float(a.mean()), float(b.mean())], "gain": gain,
                 "mcnemar_p": pv, "won": b_only, "lost": a_only, "ci": ci,
                 "verdict": verdict(gain, pv, ci)}
        # Mechanism, on the episodes that arrive in both arms: there the two
        # are doing the same job and the driving is comparable.
        both = a & b
        entry["both"] = int(both.sum())
        for key in ("replans", "driven_over_shortest", "wandering", "reversals",
                    "plan_ahead", "known", "steps"):
            entry[key] = [float(field(base, key)[both].mean()),
                          float(field(arm, key)[both].mean())] if both.any() else None
        entry["frontier_plans"] = float(field(arm, "frontier_plans").mean())
        entry["goal_plans"] = float(field(arm, "goal_plans").mean())
        report["conditions"][cond] = entry

        print(f"{cond:8s} SR {a.mean():.2f} -> {b.mean():.2f}  {gain:+.3f}  "
              f"McNemar p={pv:.3f} ({b_only} won / {a_only} lost)  "
              f"CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  {entry['verdict']}")
        if both.any():
            print(f"         on the {int(both.sum())} both solve: "
                  f"replans {entry['replans'][0]:.0f} -> {entry['replans'][1]:.0f}  "
                  f"driven/shortest {entry['driven_over_shortest'][0]:.2f} -> "
                  f"{entry['driven_over_shortest'][1]:.2f}  "
                  f"reversals {entry['reversals'][0]:.1f} -> "
                  f"{entry['reversals'][1]:.1f}  "
                  f"ahead {entry['plan_ahead'][0]:.2f} -> "
                  f"{entry['plan_ahead'][1]:.2f} m")
        print(f"         frontier plans {entry['frontier_plans']:.0f} of "
              f"{entry['frontier_plans'] + entry['goal_plans']:.0f}")
        if cond in CLUTTER:
            pooled["a"].append(a)
            pooled["b"].append(b)
            pooled["ra"].append(field(base, "replans")[both])
            pooled["rb"].append(field(arm, "replans")[both])

    # ---- endpoint 1: success on the clutter conditions, pooled ----
    a = np.concatenate(pooled["a"])
    b = np.concatenate(pooled["b"])
    gain = float(b.mean() - a.mean())
    pv, a_only, b_only = mcnemar_p(a, b)
    ci = paired_ci(a.astype(float), b.astype(float), rng, args.bootstrap)
    need = REGISTERED["clutter_success_gain_at_least"]
    held = gain >= need and pv < 0.05
    report["clutter_pooled"] = {"success": [float(a.mean()), float(b.mean())],
                               "gain": gain, "mcnemar_p": pv, "won": b_only,
                               "lost": a_only, "ci": ci,
                               "verdict": verdict(gain, pv, ci),
                               "registered_held": bool(held)}
    print(f"\nclutter pooled ({len(a)} worlds)  SR {a.mean():.2f} -> {b.mean():.2f}  "
          f"{gain:+.3f}  McNemar p={pv:.3f} ({b_only} won / {a_only} lost)  "
          f"CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  {report['clutter_pooled']['verdict']}")
    print(f"  endpoint 1 -- clutter success >= {need:+.2f} with p < 0.05: "
          f"{'HELD' if held else 'FAILED'}")

    # ---- endpoint 2: replans, on the episodes both arms solve ----
    ra, rb = np.concatenate(pooled["ra"]), np.concatenate(pooled["rb"])
    if len(ra):
        reduction = float(1.0 - rb.mean() / ra.mean()) if ra.mean() else float("nan")
        rci = ratio_ci(ra, rb, rng, args.bootstrap)
        need2 = REGISTERED["clutter_replan_reduction_at_least"]
        held2 = reduction >= need2 and rci[0] > 0
        report["clutter_replans"] = {"mean": [float(ra.mean()), float(rb.mean())],
                                     "reduction": reduction, "ci": rci,
                                     "registered_held": bool(held2),
                                     "n": int(len(ra))}
        print(f"  endpoint 2 -- replans {ra.mean():.0f} -> {rb.mean():.0f} on the "
              f"{len(ra)} both solve, cut {reduction:+.1%} "
              f"CI [{rci[0]:+.1%}, {rci[1]:+.1%}], needed {need2:.0%}: "
              f"{'HELD' if held2 else 'FAILED'}")

    # ---- endpoint 3: the registered cost on nominal, a bound ----
    if "nominal" in conditions:
        n = report["conditions"]["nominal"]
        loss = -n["gain"]
        need3 = REGISTERED["nominal_success_loss_at_most"]
        held3 = loss <= need3
        n["registered_held"] = bool(held3)
        print(f"  endpoint 3 -- nominal success loss {loss:+.3f}, bound "
              f"{need3:.2f}: {'HELD' if held3 else 'FAILED'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
