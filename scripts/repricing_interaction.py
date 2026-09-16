"""Does the reward change what an observation channel is worth?

Phase 3d compared 64 against 128 lidar beams under the 4:1 reward and reported
the success delta: -0.003, a null. Phase 5h later established that a flat
success rate can hide a real behavioural change, and re-reading 3d's own
result file showed one -- collisions +0.083 at p = 0.007, with timeouts
falling by the same amount. Same net success, different failures.

This scores the same contrast under indifference (collision 5 == a full
500-step timeout at 0.01/step) and tests the **interaction**: not "does
resolution help", but "does the reward decide what resolution does". Neither
single-reward comparison can answer that, because the quantity of interest is
the difference of two differences.

Seeds are index-paired across all four cells -- every arm trains seeds 0-5 --
so the per-seed delta is well defined within each reward, and the interaction
is an exact permutation test over those twelve numbers.

    python scripts/repricing_interaction.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

#: (label, 4:1 result file, 1:1 result file, baseline arm, treated arm)
DEFAULT_CELLS = (
    "results/seed_3d_samples_at_360.json", "results/repricing_resolution.json",
)


def permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation test on the difference of means."""
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


def per_seed_delta(report: dict, cond: str, base: str, treated: str, metric: str):
    cell = report["conditions"][cond]["per_arm"]
    a = np.asarray(cell[base][metric], dtype=float)
    b = np.asarray(cell[treated][metric], dtype=float)
    return b - a


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--four-to-one", default=DEFAULT_CELLS[0])
    p.add_argument("--one-to-one", default=DEFAULT_CELLS[1])
    p.add_argument("--arms-4to1", nargs=2, default=["b64", "b128"])
    p.add_argument("--arms-1to1", nargs=2, default=["b64i", "b128i"])
    p.add_argument("--conditions", nargs="+", default=["narrow"])
    p.add_argument("--out", default="results/repricing_interaction.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    hi = json.loads(Path(args.four_to_one).read_text(encoding="utf-8"))
    lo = json.loads(Path(args.one_to_one).read_text(encoding="utf-8"))
    report = {"conditions": {}}

    for cond in args.conditions:
        if cond not in hi["conditions"] or cond not in lo["conditions"]:
            print(f"{cond}: not scored under both rewards, skipping")
            continue
        print(f"\n=== {cond} ===")
        entry = {}
        for metric in ("success", "collision", "timeout"):
            d_hi = per_seed_delta(hi, cond, *args.arms_4to1, metric)
            d_lo = per_seed_delta(lo, cond, *args.arms_1to1, metric)
            interaction = float(d_lo.mean() - d_hi.mean())
            p = permutation_p(d_hi, d_lo)
            # A sign flip carried by every seed says something a mean does not.
            flipped = int(np.sum(np.sign(d_lo) != np.sign(d_hi)))
            entry[metric] = {
                "delta_4to1": float(d_hi.mean()),
                "delta_1to1": float(d_lo.mean()),
                "interaction": interaction,
                "p": p,
                "per_seed_4to1": d_hi.tolist(),
                "per_seed_1to1": d_lo.tolist(),
                "seeds_changing_sign": flipped,
            }
            mark = "  <-- SIGNIFICANT" if p < 0.05 else ""
            print(f"  {metric:10s} 4:1 {d_hi.mean():+.3f}   1:1 {d_lo.mean():+.3f}"
                  f"   interaction {interaction:+.3f}   p = {p:.4f}{mark}")
            print(f"             per-seed 4:1 {np.round(d_hi, 3).tolist()}")
            print(f"             per-seed 1:1 {np.round(d_lo, 3).tolist()}")
        report["conditions"][cond] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
