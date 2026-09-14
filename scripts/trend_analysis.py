"""Test for a monotone trend across an ordered sweep of sensor configurations.

Pairwise tests are the wrong tool for a sweep whose adjacent levels are
predicted to differ by less than the seed spread: each comparison is
underpowered by construction, and running all of them invites picking whichever
one happens to clear 0.05. A single trend statistic over every seed uses the
ordering as information rather than throwing it away.

    python scripts/trend_analysis.py \\
        --level 90=runs/depth_s0,runs/depth_s1 \\
        --level 360=runs/beams64,runs/b64_s1 \\
        --condition narrow

Significance is a **Monte Carlo** permutation test, not the exact enumeration
used for two-arm comparisons: with 24 seeds in four groups the number of
distinct relabellings is about 2.3e12, so exhaustive enumeration is
impossible. The reported p is therefore resolved only to ~1/n_permutations,
and is itself a random variable — reproducible here because the shuffling RNG
is seeded.
"""

from __future__ import annotations

import argparse
import json
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
        "--level",
        action="append",
        required=True,
        metavar="VALUE=DIR[,DIR...]",
        help="An ordered level and its per-seed run directories. Repeatable.",
    )
    p.add_argument("--condition", default="narrow", choices=list(CONDITIONS))
    p.add_argument("--metric", default="success", choices=["success", "spl", "collision"])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--permutations", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the shuffling")
    p.add_argument("--predicted", default=None,
                   help="Comma-separated predicted means, in level order, to score against")
    p.add_argument("--out", default="results/trend_analysis.json")
    return p.parse_args(argv)


def rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — required, since levels are heavily tied."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # Average the ranks within each group of equal values.
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of tied-average ranks)."""
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def permutation_trend_p(
    levels: np.ndarray, values: np.ndarray, n_perm: int, seed: int
) -> tuple[float, float]:
    """Two-sided Monte Carlo p for the observed Spearman correlation."""
    observed = spearman(levels, values)
    rng = np.random.default_rng(seed)
    shuffled = values.copy()
    count = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if abs(spearman(levels, shuffled)) >= abs(observed) - 1e-12:
            count += 1
    # Add-one correction: a Monte Carlo p of exactly 0 is not attainable and
    # reporting it would overstate the evidence.
    return observed, (count + 1) / (n_perm + 1)


def run_sensor(run: Path) -> dict:
    """Env overrides needed to evaluate this run's policy."""
    env = OmegaConf.load(run / "config.yaml").env
    mode = OmegaConf.select(env, "obs_mode") or "privileged"
    spec: dict = {"obs_mode": mode}
    if mode == "depth":
        cam = OmegaConf.to_container(env.camera, resolve=True)
        spec["camera"] = {k: cam[k] for k in ("fov", "width", "max_range") if k in cam}
    else:
        lid = OmegaConf.to_container(env.lidar, resolve=True)
        spec["lidar"] = {k: lid[k] for k in ("n_beams", "fov", "max_range") if k in lid}
    return spec


def main(argv=None) -> int:
    args = parse_args(argv)
    split, shift = CONDITIONS[args.condition]

    levels: list[tuple[float, list[Path]]] = []
    for spec in args.level:
        value, _, dirs = spec.partition("=")
        runs = [Path(d) for d in dirs.split(",")]
        for r in runs:
            if not (r / "best_model.zip").exists():
                raise FileNotFoundError(f"no best_model.zip in {r}")
        levels.append((float(value), runs))
    levels.sort(key=lambda t: t[0])

    print(f"\n=== {args.condition}: {args.metric} vs level "
          f"({args.episodes} worlds per seed) ===")

    xs, ys = [], []
    per_level = {}
    for value, runs in levels:
        scores = []
        for run in runs:
            cfg = build_env_config(
                run_sensor(run), split=split, shift=shift, n_worlds=args.episodes
            )
            actor = build_actor("rl", model_path=str(run / "best_model.zip"), robot=cfg.robot)
            metrics, _ = evaluate(actor, cfg)
            scores.append(
                {"success": metrics.success_rate, "spl": metrics.spl,
                 "collision": metrics.collision_rate}[args.metric]
            )
        per_level[value] = scores
        xs += [value] * len(scores)
        ys += scores
        print(f"  level {value:>6.0f}: {[round(s, 3) for s in scores]}  "
              f"mean {np.mean(scores):.3f} +/- {np.std(scores, ddof=1):.3f}")

    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    rho, p = permutation_trend_p(x, y, args.permutations, args.seed)
    slope = float(np.polyfit(x, y, 1)[0])

    print(f"\n  Spearman rho = {rho:+.3f}")
    print(f"  Monte Carlo p = {p:.5f}  ({args.permutations:,} permutations, "
          f"resolution ~{1 / args.permutations:.0e})")
    print(f"  linear slope  = {slope:+.5f} per degree "
          f"({slope * 270:+.3f} across 90->360)")
    print(f"  n = {len(y)} seeds across {len(levels)} levels")

    if args.predicted:
        pred = [float(v) for v in args.predicted.split(",")]
        if len(pred) != len(levels):
            print(f"  (predicted has {len(pred)} values for {len(levels)} levels; skipped)")
        else:
            print("\n  Predicted vs observed level means:")
            errs = []
            for (value, _), pv in zip(levels, pred, strict=False):
                obs = float(np.mean(per_level[value]))
                errs.append(obs - pv)
                print(f"    {value:>6.0f}: predicted {pv:.3f}  observed {obs:.3f}  "
                      f"error {obs - pv:+.3f}")
            print(f"    mean |error| = {np.mean(np.abs(errs)):.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "condition": args.condition,
                "metric": args.metric,
                "episodes": args.episodes,
                "per_level": {str(k): v for k, v in per_level.items()},
                "spearman_rho": rho,
                "p_value": p,
                "permutations": args.permutations,
                "rng_seed": args.seed,
                "linear_slope_per_degree": slope,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
