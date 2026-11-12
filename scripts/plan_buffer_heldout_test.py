"""The plan buffer on held-out worlds: does §9.32's val gain carry?

Report §9.32 planned a cell off the margin's edge and moved arrival for the
first time: +0.050 pooled over val clutter, 14 worlds won and 4 lost -- exactly
the registered threshold, and one world from NO EFFECT. A val result that sits
on its threshold is a lead, not a finding. This runs the same two arms, unchanged,
on the held-out worlds §9.13 and §9.21 report: the first 100 `test` worlds for
`nominal` and the first 100 `test_ood` worlds each for `dense` and `narrow`, at
the 500-step budget. A third arm is the control:

  published  the stack as §9.13 ran it, which must reproduce that section's
             held-out front-end episodes exactly
  footprint  §9.26's footprint-consistent check, never run on these worlds
  buffered   the same with ``plan_buffer`` = 0.10 m, as in §9.32

The decision, the clutter pairs and the margin-safe mechanism are scored by
``plan_buffer_test.analyse``, the function that scored val.

    python scripts/plan_buffer_heldout_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_buffer_test import BUFFER, CLUTTER, CONDITIONS, analyse, episode, stalled  # noqa: E402
from resumable import checkpoint_for, run_resumable  # noqa: E402
from route_class_test import route_classes  # noqa: E402
from transfer_test import bootstrap, mcnemar  # noqa: E402

from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402

ARMS = ("published", "footprint", "buffered")

#: Recorded and committed before any arm had run on these held-out worlds.
#:
#: DISCLOSED: what is already known of them. The published stack's per-world
#: outcomes are §9.13's (``backend_test.json``, front end) and their route
#: classes §9.21's: clutter success 0.595, no margin-safe route on 42 worlds (1
#: arrives), and 33 stalls on worlds that have one. Neither repaired arm has run
#: on any of them; each was timed on one train-band `dense` world, its outcome
#: not read. The reasoning: on val the buffer won 14 worlds and lost 4, and 11
#: of the 14 were margin-safe stalls converted -- 11 of 31, about a third. At
#: that rate the held-out stalls give about +0.05 again. But a third of 31 is
#: known only to between about a fifth and a half, the buffer and its 1.25x cut
#: were set on train worlds and checked on val, and the five repairs before it
#: moved nothing; so the forecast is shrunk towards zero, and lands under the
#: threshold. The mechanism clauses are val's measurements with room either side.
PREDICTION = (
    "the buffer's gain carries to held-out worlds, smaller than on val and under "
    "the threshold. Pooled over the 200 held-out clutter worlds the buffered - "
    "footprint difference is between +0.01 and +0.045. On the margin-safe "
    "held-out clutter worlds: the buffered rung supplies at least half of the "
    "buffered arm's plans; rebuilds and flips per 100 steps each fall by between "
    "20% and 45%; between a fifth and a half of the footprint arm's stalls there "
    "arrive when buffered. The buffered arm collides on at most 0.02 of episodes "
    "on every condition and moves nominal by at most 0.03 either way. The "
    "footprint arm is within 0.03 of the published stack pooled over clutter, as "
    "on val. The published arm reproduces §9.13's held-out front-end episodes "
    "exactly. Decision on the pooled buffered - footprint difference over the "
    "200 paired held-out clutter worlds -- REPAIRS: at least +0.05 with its 95% "
    "interval above zero. HARMS: at most -0.05 with the interval below zero. NO "
    "EFFECT: within 0.05 either way with the interval inside 0.10 either way. "
    "Otherwise UNRESOLVED. Prediction: NO EFFECT."
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="results/plan_buffer_heldout_test.json")
    p.add_argument("--rescore", action="store_true",
                   help="Score the episodes already in --out instead of driving them again.")
    args = p.parse_args(argv)
    if args.rescore:
        rows = json.loads(Path(args.out).read_text(encoding="utf-8"))["episodes_detail"]
    else:
        classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
                   for w in route_classes(None, c, args.episodes)}
        jobs = [(c, a, s, args.episodes, BENCHMARK_CONDITIONS[c][0]) for c in CONDITIONS for a in ARMS
                for s in sorted(seed for cond, seed in classes if cond == c)]
        # Kept episode by episode, so an interrupted run resumes (resumable.py).
        rows = run_resumable(episode, jobs, args.workers, checkpoint_for(args.out))
        for r in rows:
            r["class"] = classes[(r["cond"], r["seed"])]

    rng = np.random.default_rng(0)
    report = analyse([r for r in rows if r["arm"] != "published"], rng)

    # §9.26's check on its own, out of sample: footprint against published.
    by = {(r["arm"], r["cond"], r["seed"]): r for r in rows}
    keys = sorted({(r["cond"], r["seed"]) for r in rows}, key=lambda k: (CONDITIONS.index(k[0]), k[1]))
    for cond in CONDITIONS:
        mine = [by[("published", c, s)] for c, s in keys if c == cond]
        report["conditions"][cond]["published"] = {
            "success": float(np.mean([r["success"] for r in mine])),
            "collision": float(np.mean([r["collision"] for r in mine])),
            "stalls": int(sum(stalled(r) for r in mine))}
    report["footprint_vs_published"] = {}
    for name, conds in (("nominal", ("nominal",)), ("clutter", CLUTTER)):
        sel = [k for k in keys if k[0] in conds]
        sa = np.array([by[("published",) + k]["success"] for k in sel], dtype=bool)
        sb = np.array([by[("footprint",) + k]["success"] for k in sel], dtype=bool)
        pv, won, lost = mcnemar(sa, sb)
        report["footprint_vs_published"][name] = {
            "delta": float(sb.mean() - sa.mean()), "p": pv, "won": won, "lost": lost,
            "ci95": bootstrap(sa, sb, np.array([c for c, _ in sel]), rng)}
    no_route = [k for k in keys if k[0] in CLUTTER and by[("footprint",) + k]["class"] == "no route"]
    report["no_route"] = {"worlds": len(no_route),
                          "arrived": {arm: int(sum(by[(arm,) + k]["success"] for k in no_route))
                                      for arm in ARMS}}

    published = Path("results/backend_test.json")
    if args.episodes == 100 and published.exists():
        cells = json.loads(published.read_text(encoding="utf-8"))["cells"]
        before = {(c, e["seed"]): e for c in CONDITIONS for e in cells[c]["front_end"]["episodes"]}
        mine = [r for r in rows if r["arm"] == "published"]
        report["reproduces_backend_test"] = len(mine) == len(before) and all(
            all(r[f] == before[(r["cond"], r["seed"])][f] for f in ("success", "collision", "steps"))
            for r in mine)
        print(f"published arm reproduces §9.13: {report['reproduces_backend_test']}")

    report.update(split="held-out", buffer=BUFFER, prediction=PREDICTION)
    report["episodes_detail"] = rows
    for cond, c in report["conditions"].items():
        print(cond.ljust(8) + "  ".join(f"{arm} {v['success']:.2f}/{v['collision']:.2f} stalls {v['stalls']}"
                                        for arm, v in c.items()))
    for name, q in report["pairs"].items():
        print(f"buffered - footprint {name:8s} {q['delta']:+.3f} (p {q['p']:.3f}, {q['won']}/{q['lost']}) "
              f"CI {np.round(q['ci95'], 3).tolist()}")
    for name, q in report["footprint_vs_published"].items():
        print(f"footprint - published {name:8s} {q['delta']:+.3f} (p {q['p']:.3f}, {q['won']}/{q['lost']}) "
              f"CI {np.round(q['ci95'], 3).tolist()}")
    print(f"margin-safe: {json.dumps(report['margin_safe'])}")
    print(f"no route: {json.dumps(report['no_route'])}")
    print(f"decision {report['decision']}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint_for(args.out).unlink(missing_ok=True)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
