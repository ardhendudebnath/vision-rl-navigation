"""Is the clutter gap a property of the worlds? A held-out test of a val lead.

Report §9.9 found that every val clutter world with no route at the planner's
full margin (``robot_radius + safety_margin``) fails, and left "the seven clutter
failures on worlds that had a margin-safe route all along" unexplained. Joining
the same two val files again -- ``margin_audit.json``'s route lengths and
``clutter_forensic.json``'s outcomes -- six of those seven are worlds whose
margin-safe route is a long detour: 1.35x to 1.95x the shortest route, where 31
of the 32 arrivals' routes are at most 1.17x (one arrival, 1.38x). Classed by
what the margin does to the shortest route:

  no route      no route at all at the full margin                    0 of 11 arrive
  long detour   the margin-safe route is at least DETOUR x the shortest  1 of 7
  short         it is shorter than that                               31 of 32

The detours fit the step budget easily at the arrivals' own pace (216 to 430
steps of 500), so the robot does not fail them by running out of road; it fails
them stalled, replanning 94 to 203 times. The cut between the classes was chosen
by looking at those numbers, in the gap between 1.17 and 1.35, and the
association is between worlds, so on val it is a lead and nothing more.

This tests it out of sample. The route classes of the held-out clutter worlds
(``test_ood``, seeds 30000-30099 of `dense` and `narrow`) are computed here,
with ``margin_audit``'s own route measure, and joined by seed to the held-out
outcomes ``backend_experiment.py`` already recorded for the same stack -- 360
beams, scan matching, corroboration -- with and without the back end
(``results/backend_test.json``). No agent is run.

Identity control: on val the classes and route lengths must reproduce
``margin_audit.json`` exactly, and the tallies above.

    python scripts/route_class_test.py --split val     # the control
    python scripts/route_class_test.py                 # the registered test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin_audit import route_length  # noqa: E402
from margin_overlap import fisher_exact  # noqa: E402

from vision_nav.agents.classical import PursuitConfig  # noqa: E402
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("dense", "narrow")
#: The cut between a short and a long margin-safe route, chosen on val.
DETOUR = 1.2
CLASSES = ("no route", "long detour", "short")
#: The first two together: worlds where the margin rules out the shortest route.
BLOCKED = ("no route", "long detour")

#: Recorded and committed before the route classes of any held-out world had
#: been computed.
#:
#: DISCLOSED: derived from val, from the two files named above, by looking --
#: the detour cut included. The held-out outcomes being joined to were already
#: committed and reported in §9.13 as pooled and per-condition rates, and their
#: per-world records were read for that section's analysis; they have never been
#: set against any measure of the worlds' geometry. The val back end gained on
#: the long-detour class (1 of 7 to 3 of 7) and nowhere else, which is where the
#: back-end clause comes from.
PREDICTION = (
    "the clutter gap is a property of the worlds: held-out clutter failures "
    "concentrate on worlds where the planner's margin rules out the shortest "
    "route. Pooled over held-out dense and narrow (200 worlds), with the stack "
    "as published (no back end): worlds with no margin-safe route or a "
    "margin-safe route at least 1.2x the shortest make up between 25% and 50% "
    "of the worlds; the stack succeeds on at most 0.25 of them and on at least "
    "0.80 of the rest; they hold at least 70% of the failures; and the two-sided "
    "Fisher exact test between the two groups gives p < 0.001. On worlds with "
    "no margin-safe route at all it succeeds on at most 0.10. With the back "
    "end, its net gain over the published stack (worlds won minus worlds lost) "
    "on the long-detour class is positive and at least as large as on the "
    "other two classes together. "
    "Decision on the first four pooled clauses -- STRUCTURAL: all four hold. "
    "NOT STRUCTURAL: the Fisher p is at least 0.05, or the blocked worlds hold "
    "under half the failures. Otherwise PARTIAL."
)


def route_classes(split: str, cond: str, n: int) -> list[dict]:
    """Route lengths at the bare radius and the full margin, and the class."""
    split_name, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split or split_name, shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    margin = PursuitConfig().safety_margin
    rows = []
    for seed in [int(s) for s in list(cfg.world_seeds)[:n]]:
        env.reset(options={"world_seed": seed})
        world = env.world
        base = world.config.robot_radius
        start, goal = world.start[:2], world.goal
        l_base = route_length(world, world.occupancy_at(base), start, goal)
        l_full = route_length(world, world.occupancy_at(base + margin), start, goal)
        if l_full is None:
            kind = "no route"
        elif l_base and l_full / l_base >= DETOUR:
            kind = "long detour"
        else:
            kind = "short"
        rows.append({"seed": seed, "l_base": l_base, "l_full": l_full, "class": kind})
    return rows


def outcomes(split: str) -> dict:
    """{(cond, seed): {"front_end": bool, "back_end": bool}} for the same stack."""
    out: dict = {}
    if split == "val":
        forensic = json.loads(Path("results/clutter_forensic.json").read_text(encoding="utf-8"))
        backend = json.loads(Path("results/backend_experiment.json").read_text(encoding="utf-8"))
        for cond in CONDITIONS:
            for e in forensic["conditions"][cond]["episodes"]:
                out.setdefault((cond, e["seed"]), {})["front_end"] = bool(e["success"])
            for e in backend["cells"][cond]["back_end"]["episodes"]:
                out.setdefault((cond, e["seed"]), {})["back_end"] = bool(e["success"])
    else:
        backend = json.loads(Path("results/backend_test.json").read_text(encoding="utf-8"))
        for cond in CONDITIONS:
            for arm in ("front_end", "back_end"):
                for e in backend["cells"][cond][arm]["episodes"]:
                    out.setdefault((cond, e["seed"]), {})[arm] = bool(e["success"])
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default=None, help="'val' for the identity control")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    val = args.split == "val"
    n = args.episodes or (25 if val else 100)
    out = Path(args.out or ("results/route_class_val.json" if val else "results/route_class_test.json"))

    results = outcomes("val" if val else "test")
    report: dict = {"split": "val" if val else "test_ood", "episodes": n, "detour": DETOUR,
                    "prediction": PREDICTION, "checks": {}, "conditions": {}}
    rows = []
    for cond in CONDITIONS:
        worlds = route_classes("val" if val else None, cond, n)
        for w in worlds:
            r = results[(cond, w["seed"])]
            w.update(cond=cond, front_end=r["front_end"], back_end=r.get("back_end"))
        report["conditions"][cond] = {"worlds": worlds}
        rows += worlds

    if val:
        audit = json.loads(Path("results/margin_audit.json").read_text(encoding="utf-8"))
        report["checks"]["lengths_reproduce_margin_audit"] = all(
            {k: w[k] for k in ("seed", "l_base", "l_full")}
            == {k: a[k] for k in ("seed", "l_base", "l_full")}
            for cond in CONDITIONS
            for w, a in zip(report["conditions"][cond]["worlds"],
                            audit["conditions"][cond]["worlds"], strict=True))

    def tally(arm: str, kinds) -> tuple[int, int]:
        sub = [w for w in rows if w["class"] in kinds]
        return sum(w[arm] for w in sub), len(sub)

    per_class = {k: dict(zip(("arrived", "n"), tally("front_end", (k,)), strict=True))
                 for k in CLASSES}
    if val:
        report["checks"]["val_tallies_reproduce"] = (
            [(per_class[k]["arrived"], per_class[k]["n"]) for k in CLASSES]
            == [(0, 11), (1, 7), (31, 32)])
    ab, nb = tally("front_end", BLOCKED)
    ar, nr = tally("front_end", ("short",))
    failures = (nb - ab) + (nr - ar)
    fisher = fisher_exact(ab, nb - ab, ar, nr - ar)
    pooled = {"blocked_share_of_worlds": nb / len(rows),
              "blocked_success": ab / nb if nb else float("nan"),
              "short_success": ar / nr if nr else float("nan"),
              "blocked_share_of_failures": (nb - ab) / failures if failures else float("nan"),
              "fisher_p": fisher,
              "no_route_success": (per_class["no route"]["arrived"] / per_class["no route"]["n"]
                                   if per_class["no route"]["n"] else float("nan"))}
    held = [0.25 <= pooled["blocked_share_of_worlds"] <= 0.50,
            pooled["blocked_success"] <= 0.25,
            pooled["short_success"] >= 0.80,
            pooled["blocked_share_of_failures"] >= 0.70,
            fisher < 0.001]
    if all(held[1:]):
        decision = "STRUCTURAL"
    elif fisher >= 0.05 or pooled["blocked_share_of_failures"] < 0.5:
        decision = "NOT STRUCTURAL"
    else:
        decision = "PARTIAL"

    # The back end's net gain over the published stack, per class.
    net = {}
    for k in CLASSES:
        sub = [w for w in rows if w["class"] == k and w["back_end"] is not None]
        won = sum(1 for w in sub if w["back_end"] and not w["front_end"])
        lost = sum(1 for w in sub if w["front_end"] and not w["back_end"])
        net[k] = {"won": won, "lost": lost, "net": won - lost}
    back_end_clause = (net["long detour"]["net"] > 0 and net["long detour"]["net"]
                       >= net["no route"]["net"] + net["short"]["net"])

    report.update(per_class=per_class, pooled=pooled, back_end_net=net,
                  clauses={"blocked_share_of_worlds": held[0], "blocked_success": held[1],
                           "short_success": held[2], "blocked_share_of_failures": held[3],
                           "fisher": held[4],
                           "no_route_success": pooled["no_route_success"] <= 0.10,
                           "back_end": back_end_clause},
                  decision=decision)

    print(f"{report['split']}, {n} worlds per condition, detour cut {DETOUR}x\n")
    for k in CLASSES:
        c = per_class[k]
        print(f"  {k:12s} {c['arrived']:3d} of {c['n']:3d} arrive"
              f"   back end net {net[k]['net']:+d} ({net[k]['won']} won, {net[k]['lost']} lost)")
    print(f"\n  blocked worlds: {pooled['blocked_share_of_worlds']:.0%} of worlds, success "
          f"{pooled['blocked_success']:.3f} against {pooled['short_success']:.3f}, "
          f"{pooled['blocked_share_of_failures']:.0%} of failures, Fisher p {fisher:.2g}")
    print(f"  clauses: {report['clauses']}")
    print(f"  decision: {decision}")
    print(f"  checks: {report['checks']}")
    print("pre-registered: " + PREDICTION)

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
