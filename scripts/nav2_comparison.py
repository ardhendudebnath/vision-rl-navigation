"""Compare the real Nav2 stack against the hand-written classical baseline.

The report's largest standing caveat was that the baseline it calls "classical"
is *structurally analogous* to Nav2 (global plan over an inflated costmap,
local controller tracking it) rather than being Nav2. If the hand-written
baseline were secretly weak, every claim of the form "the learned policy loses
to classical planning" would collapse into "the learned policy loses to one
particular script".

This reads both sets of results and answers one question: is the hand-written
baseline as strong as the report treats it?

    python scripts/nav2_comparison.py --nav2-runs results/nav2_runs/*

Nav2 rows come from ros2_bridge/run_nav2_eval.py, which scores the same worlds
in the same order under the same BENCHMARK_CONDITIONS table.

Why repeated runs
-----------------
Every other actor in this study is either deterministic (the hand-written
planner) or replicated across training seeds. Nav2 is neither: it is a set of
asynchronous processes whose timing jitter changes which trajectory DWB picks,
so two runs over identical worlds in identical order do not agree. Measured
spread on `nominal` was 0.950 vs 0.990 — larger than several of the gaps being
interpreted. Quoting one Nav2 run against a deterministic baseline would repeat
exactly the single-measurement error this project documents in Section 6, so
the table reports the range across independent runs and the verdict is taken
against the range, not the mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vision_nav.envs.splits import BENCHMARK_CONDITIONS

#: A gap this size or smaller is not worth interpreting: at 100 episodes the
#: binomial standard error on a success rate near 0.9 is about 0.03.
NOISE_BAND = 0.05


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def verdict(runs: list[float], hand: float) -> str:
    """Read the comparison against the *whole range* of Nav2 runs.

    Only claim a winner when every Nav2 run falls on the same side by more
    than the noise band; otherwise the run-to-run spread already covers the
    difference and there is nothing to conclude.
    """
    deltas = [r - hand for r in runs]
    if all(d > NOISE_BAND for d in deltas):
        return "Nav2 better"
    if all(d < -NOISE_BAND for d in deltas):
        return "hand-written better"
    return "parity"


def fmt_range(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{min(values):.3f}-{max(values):.3f}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results",
                   help="Directory holding <condition>__classical.json")
    p.add_argument("--nav2-runs", nargs="+", required=True,
                   help="One directory per independent Nav2 run")
    p.add_argument("--out", default=None, help="Write the markdown table here")
    args = p.parse_args(argv)

    root = Path(args.results)
    run_dirs = [Path(d) for d in args.nav2_runs]

    rows, missing = [], []
    for cond in BENCHMARK_CONDITIONS:
        hand = load(root / f"{cond}__classical.json")
        nav2 = [load(d / f"{cond}__nav2.json") for d in run_dirs]
        nav2 = [n for n in nav2 if n is not None]
        if hand is None or not nav2:
            missing.append(cond)
            continue
        rows.append((cond, hand, nav2))

    if not rows:
        print(f"no paired results (need <condition>__classical.json in {root}/ "
              f"and <condition>__nav2.json in the run directories)")
        return 1

    n_runs = max(len(n) for _, _, n in rows)
    lines = [
        f"| Condition | Hand-written | Nav2 ({n_runs} runs) | Δ success | Reading |",
        "|---|---|---|---|---|",
    ]
    for cond, hand, nav2 in rows:
        succ = [n["success_rate"] for n in nav2]
        spl = [n["spl"] for n in nav2]
        deltas = [s - hand["success_rate"] for s in succ]
        d = f"{min(deltas):+.3f}" if len(deltas) == 1 else \
            f"{min(deltas):+.3f} to {max(deltas):+.3f}"
        lines.append(
            f"| {cond} | {hand['success_rate']:.3f} / {hand['spl']:.3f} "
            f"| {fmt_range(succ)} / {fmt_range(spl)} "
            f"| {d} | {verdict(succ, hand['success_rate'])} |"
        )

    table = "\n".join(lines)
    print(table)

    spreads = [max(n["success_rate"] for n in nav2) - min(n["success_rate"] for n in nav2)
               for _, _, nav2 in rows if len(nav2) > 1]
    print()
    if spreads:
        print(f"Nav2 run-to-run spread on identical worlds: "
              f"{min(spreads):.3f}-{max(spreads):.3f} success. "
              "Nav2 is asynchronous, so this is irreducible without more runs.")
    else:
        print("Only one Nav2 run per condition: the table cannot separate a "
              "real gap from Nav2's run-to-run jitter. Pass more --nav2-runs.")

    beaten = [c for c, h, n in rows
              if verdict([x["success_rate"] for x in n], h["success_rate"]) == "Nav2 better"]
    print()
    if beaten:
        print("Nav2 beats the hand-written baseline on: " + ", ".join(beaten)
              + ". The baseline understates classical planning there and the "
                "report must say so.")
    else:
        print("Nav2 beats the hand-written baseline on no condition, so the "
              "baseline was not the weak link and the report's "
              "classical-vs-learned comparisons stand as written.")

    if missing:
        print(f"\nno paired results for: {', '.join(missing)}")

    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
