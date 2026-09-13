"""Compare learned policies against the classical planner where the map is wrong.

Every other condition in this study hands the classical planner a perfect,
current, static map — its single largest privilege, and the one real
deployments do not have. This condition adds obstacles that **move and are
absent from the map**, which is the first place a reactive policy has a
structural reason to win.

The baseline is kept strong on purpose. Nav2 does not plan once and drive
blind: it maintains a local costmap from live sensor data and replans
continuously. So the classical arm replans every N steps against a costmap
containing the movers *where they currently are*. Beating a planner that
drove blind into them would prove nothing.

    python scripts/dynamic_experiment.py \\
        --arm b64=runs/beams64,runs/b64_s1,... --condition dynamic

Statistics: the learned arm has one value per training seed, but the classical
planner is deterministic and has exactly one. A permutation test with the seed
as the unit therefore does not apply. Instead we use an **exact binomial sign
test** of the seeds against the classical scalar: under the null that a seed is
equally likely to fall either side of it, the count of seeds above it is
Binomial(n, 0.5). With six seeds the smallest attainable two-sided p is
2/2^6 = 0.031, which is stated rather than left implicit.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from vision_nav.agents.classical import PursuitConfig
from vision_nav.envs.splits import FROZEN_CONDITIONS
from vision_nav.training.actors import ClassicalActor, build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate
from vision_nav.training.run_spec import env_overrides_for_run

#: (split, shift, best classical replan interval).
#:
#: The replan interval is chosen PER CONDITION, from a measured sweep, because
#: no single setting is best everywhere: replanning is worth +0.06 to +0.07
#: where movers exist and costs -0.04 to -0.16 where they do not (path churn
#: in tight corridors). Handing the baseline one global setting would
#: handicap it on half the suite -- and using replan_every=10 on `narrow`
#: drops it from 0.850 to 0.690, which would have manufactured a false parity
#: with the learned policy. The baseline gets its best configuration on every
#: condition; anything else is a strawman by omission.
CONDITIONS = {
    "dynamic": ("test_ood", "dynamic", 10),
    "dynamic_dense": ("test_ood", "dynamic_dense", 10),
    #: The same worlds as `dynamic_dense` with the movers parked. Nav2 beats
    #: the learned policy by 0.150-0.170 on `dynamic_dense` but only ties on
    #: `dynamic`; this separates the clutter from the motion. The map stays
    #: wrong -- frozen movers are still absent from it -- so the subtraction
    #: removes motion and nothing else. See FROZEN_CONDITIONS in envs.splits.
    "dynamic_frozen": ("test_ood", "dynamic", 10),
    "dynamic_dense_frozen": ("test_ood", "dynamic_dense", 10),
    "narrow": ("test_ood", "narrow", 0),
    "nominal": ("test", None, 0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=DIR[,DIR...]")
    p.add_argument("--condition", nargs="+", default=["dynamic", "dynamic_dense"],
                   choices=list(CONDITIONS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument(
        "--classical-mode", choices=("on_block", "timer"), default="on_block",
        help=(
            "How the classical baseline decides to replan. 'on_block' rebuilds "
            "the plan only when a mover actually obstructs it and is the "
            "measured best on every dynamic condition (+0.010 to +0.070 over "
            "the timer, and identical to it on static worlds, where a correct "
            "map means the path is never blocked). 'timer' reproduces the "
            "earlier published numbers."
        ),
    )
    p.add_argument("--replan-every", type=int, default=None,
                   help="Override the per-condition best classical replan interval")
    p.add_argument("--out", default="results/dynamic_experiment.json")
    return p.parse_args(argv)


def sign_test(values: np.ndarray, reference: float) -> tuple[int, float]:
    """Exact two-sided binomial sign test of ``values`` against a scalar.

    Ties are dropped, which is the conservative convention: a seed exactly
    equal to the reference is evidence for neither side.
    """
    diffs = values - reference
    nonzero = diffs[np.abs(diffs) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 0, 1.0
    k = int(np.sum(nonzero > 0))
    # Two-sided: probability of a split at least this lopsided.
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
    return k, min(1.0, 2.0 * tail / (2**n))


def main(argv=None) -> int:
    args = parse_args(argv)

    arms: dict[str, list[Path]] = {}
    for spec in args.arm:
        name, _, dirs = spec.partition("=")
        runs = [Path(d) for d in dirs.split(",")]
        for r in runs:
            if not (r / "best_model.zip").exists():
                raise FileNotFoundError(f"no best_model.zip in {r}")
        arms[name] = runs

    report: dict = {"episodes": args.episodes, "conditions": {}}

    for cond in args.condition:
        split, shift, best_replan = CONDITIONS[cond]
        replan = args.replan_every if args.replan_every is not None else best_replan
        print(f"\n=== {cond} ({args.episodes} worlds) ===")

        # Freezing is an env-config flag, not a world shift, so it applies
        # identically to every actor and leaves the world config comparable.
        frozen = {"freeze_dynamic": True} if cond in FROZEN_CONDITIONS else {}
        base = build_env_config(dict(frozen), split=split, shift=shift,
                                n_worlds=args.episodes)
        if args.classical_mode == "on_block":
            cls_config = PursuitConfig(replan_on_block=True)
            mode_label = "replan on block"
        else:
            cls_config = PursuitConfig(replan_every=replan)
            mode_label = f"replan every {replan}"
        cls_actor = ClassicalActor(cls_config, robot=base.robot)
        cls_metrics, _ = evaluate(cls_actor, base)
        print(
            f"  classical ({mode_label}): "
            f"SR={cls_metrics.success_rate:.3f} SPL={cls_metrics.spl:.3f} "
            f"coll={cls_metrics.collision_rate:.3f}"
        )

        entry = {"classical": cls_metrics.to_dict(),
                 "classical_mode": args.classical_mode,
                 "classical_replan_every": replan if args.classical_mode == "timer" else None,
                 "arms": {}}
        for name, runs in arms.items():
            succ, spl, coll = [], [], []
            for run in runs:
                cfg = build_env_config(
                    {**env_overrides_for_run(run), **frozen},
                    split=split, shift=shift, n_worlds=args.episodes,
                )
                assert cfg.world == base.world, "arms must share the same worlds"
                assert cfg.freeze_dynamic == base.freeze_dynamic, (
                    "arms must see the same movers as the baseline"
                )
                actor = build_actor(
                    "rl", model_path=str(run / "best_model.zip"), robot=cfg.robot
                )
                m, _ = evaluate(actor, cfg)
                succ.append(m.success_rate)
                spl.append(m.spl)
                coll.append(m.collision_rate)

            succ_a = np.array(succ)
            k, p = sign_test(succ_a, cls_metrics.success_rate)
            print(f"  {name}: per-seed {[round(v, 3) for v in succ]}")
            print(
                f"    mean {succ_a.mean():.3f} +/- {succ_a.std(ddof=1):.3f}  "
                f"vs classical {cls_metrics.success_rate:.3f}  "
                f"delta {succ_a.mean() - cls_metrics.success_rate:+.3f}"
            )
            print(f"    sign test: {k}/{len(succ_a)} seeds above classical, p = {p:.3f}")
            entry["arms"][name] = {
                "success": succ, "spl": spl, "collision": coll,
                "seeds_above_classical": k, "sign_p": p,
                "delta_mean": float(succ_a.mean() - cls_metrics.success_rate),
            }
        report["conditions"][cond] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
