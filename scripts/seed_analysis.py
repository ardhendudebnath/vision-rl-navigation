"""Compare experiment arms across training seeds.

Phases 2c-2f each compared single training runs. Those comparisons were paired
over *episodes*, which controls world difficulty but says nothing about
*training-seed* variance — and RL results are notoriously seed-sensitive.
This script does the comparison that actually licenses a claim: independent
training seeds per arm, with the seed as the unit of analysis.

    python scripts/seed_analysis.py \\
        --arm b32=runs/ppo_dr,runs/b32_s1,runs/b32_s2,runs/b32_s3 \\
        --arm b64=runs/beams64,runs/b64_s1,runs/b64_s2,runs/b64_s3 \\
        --conditions narrow dense nominal

Significance is assessed with an **exact permutation test**, not a t-test. At
four seeds per arm a t-test leans entirely on a normality assumption nobody
can check from four points, whereas enumerating all C(8,4)=70 relabellings is
assumption-free and exact. The cost is a resolution floor: the smallest
two-sided p-value obtainable from 4-vs-4 is 2/70 = 0.029, so "p < 0.05" here
means "the observed split is one of the two most extreme of seventy" and
nothing finer.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from vision_nav.training.actors import build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate

CONDITIONS: dict[str, tuple[str, str | None]] = {
    "nominal": ("test", None),
    "dense": ("test_ood", "dense"),
    "sparse": ("test_ood", "sparse"),
    "large": ("test_ood", "large"),
    "narrow": ("test_ood", "narrow"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=DIR[,DIR...]",
        help="An arm and its per-seed run directories. Repeatable.",
    )
    p.add_argument("--conditions", nargs="+", default=["narrow", "dense", "nominal"],
                   choices=list(CONDITIONS))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/seed_analysis.json")
    return p.parse_args(argv)


def load_arms(specs: list[str]) -> dict[str, list[Path]]:
    arms: dict[str, list[Path]] = {}
    for spec in specs:
        name, _, dirs = spec.partition("=")
        if not name or not dirs:
            raise ValueError(f"malformed --arm {spec!r}; expected NAME=DIR[,DIR...]")
        runs = []
        for d in dirs.split(","):
            run = Path(d)
            model = run / "best_model.zip"
            if not model.exists():
                raise FileNotFoundError(f"no best_model.zip in {run}")
            runs.append(run)
        arms[name] = runs
    return arms


def run_sensor(run: Path) -> dict:
    """Sensor geometry for this run; a policy cannot be evaluated without it."""
    cfg = OmegaConf.load(run / "config.yaml")
    lidar = OmegaConf.to_container(cfg.env.lidar, resolve=True)
    return {k: lidar[k] for k in ("n_beams", "fov", "max_range") if k in lidar}


def run_seed(run: Path) -> int:
    return int(OmegaConf.load(run / "config.yaml").train.seed)


def permutation_p(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sided permutation p-value on the difference of means.

    Enumerates every way of splitting the pooled values into groups of the
    original sizes, and asks how often the split is at least as extreme as the
    observed one. Exact and assumption-free, which matters at these tiny n.
    """
    pooled = np.concatenate([a, b])
    n_a = len(a)
    observed = abs(a.mean() - b.mean())
    idx = range(len(pooled))
    count = total = 0
    for combo in combinations(idx, n_a):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combo)] = True
        if abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12:
            count += 1
        total += 1
    return count / total


def main(argv=None) -> int:
    args = parse_args(argv)
    arms = load_arms(args.arm)

    report: dict = {"episodes": args.episodes, "arms": {}, "conditions": {}}
    for name, runs in arms.items():
        report["arms"][name] = [
            {"run": str(r), "seed": run_seed(r), **run_sensor(r)} for r in runs
        ]

    for cond in args.conditions:
        split, shift = CONDITIONS[cond]
        print(f"\n=== {cond} ({args.episodes} worlds per seed) ===")
        per_arm: dict[str, dict[str, list[float]]] = {}

        for name, runs in arms.items():
            succ, spl, coll = [], [], []
            for run in runs:
                cfg = build_env_config(
                    {"lidar": run_sensor(run)}, split=split, shift=shift,
                    n_worlds=args.episodes,
                )
                actor = build_actor(
                    "rl", model_path=str(run / "best_model.zip"), robot=cfg.robot
                )
                metrics, _ = evaluate(actor, cfg)
                succ.append(metrics.success_rate)
                spl.append(metrics.spl)
                coll.append(metrics.collision_rate)
            per_arm[name] = {"success": succ, "spl": spl, "collision": coll}

            seeds = [run_seed(r) for r in runs]
            print(f"  {name}: seeds {seeds}")
            print(
                f"    success   per-seed {[round(v, 3) for v in succ]}"
                f"  mean {np.mean(succ):.3f} +/- {np.std(succ, ddof=1):.3f}"
            )
            print(
                f"    SPL       per-seed {[round(v, 3) for v in spl]}"
                f"  mean {np.mean(spl):.3f} +/- {np.std(spl, ddof=1):.3f}"
            )
            print(
                f"    collision per-seed {[round(v, 3) for v in coll]}"
                f"  mean {np.mean(coll):.3f} +/- {np.std(coll, ddof=1):.3f}"
            )

        names = list(per_arm)
        comparisons = {}
        if len(names) == 2:
            a_name, b_name = names
            print(f"  {b_name} vs {a_name} (exact permutation, seed as unit):")
            for metric in ("success", "spl", "collision"):
                a = np.array(per_arm[a_name][metric])
                b = np.array(per_arm[b_name][metric])
                delta = float(b.mean() - a.mean())
                p = permutation_p(a, b)
                floor = 2 / len(list(combinations(range(len(a) + len(b)), len(a))))
                note = "  (at the resolution floor)" if abs(p - floor) < 1e-9 else ""
                print(f"    {metric:>9}: {delta:+.3f}  p = {p:.3f}{note}")
                comparisons[metric] = {"delta": delta, "p": p}

        report["conditions"][cond] = {"per_arm": per_arm, "comparisons": comparisons}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
