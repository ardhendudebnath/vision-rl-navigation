"""Run the comparison matrix and write the results table.

This is the script that produces the numbers in the report.  It evaluates
every actor under every condition on the *same* worlds, so rows are directly
comparable.

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --rl runs/ppo_privileged/best_model.zip
    python scripts/run_benchmark.py --conditions nominal dense --episodes 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vision_nav.envs.splits import SHIFTS
from vision_nav.metrics.navigation import NavigationMetrics
from vision_nav.training.actors import build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate

#: Each condition is (label, split, shift, lidar noise std).  ``test`` is the
#: in-distribution held-out set; the rest are the Phase-5 robustness suite.
CONDITIONS: dict[str, tuple[str, str | None, float]] = {
    "nominal": ("test", None, 0.0),
    "dense": ("test_ood", "dense", 0.0),
    "sparse": ("test_ood", "sparse", 0.0),
    "large": ("test_ood", "large", 0.0),
    "narrow": ("test_ood", "narrow", 0.0),
    "noisy_lidar": ("test", None, 0.10),
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rl", default=None, help="Path to a trained SB3 model (.zip)")
    p.add_argument("--episodes", type=int, default=100, help="Episodes per condition")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="Subset of conditions to run",
    )
    p.add_argument("--actors", nargs="+", default=None, help="Subset of actors to run")
    p.add_argument("--out-dir", default="results", help="Where to write results")
    return p.parse_args(argv)


def build_actor_specs(rl_path: str | None) -> dict[str, dict]:
    specs: dict[str, dict] = {
        "random": {"kind": "random"},
        "classical": {"kind": "classical"},
    }
    if rl_path:
        specs["ppo_privileged"] = {"kind": "rl", "model_path": rl_path}
    return specs


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_actor_specs(args.rl)
    if args.actors:
        missing = set(args.actors) - set(specs)
        if missing:
            print(f"error: unknown actor(s) {sorted(missing)}", file=sys.stderr)
            print(f"       available: {sorted(specs)}", file=sys.stderr)
            return 2
        specs = {k: v for k, v in specs.items() if k in args.actors}

    if not args.rl:
        print(
            "note: no --rl model given; running baselines only. Train one with\n"
            "      python -m vision_nav.training.train\n",
            flush=True,
        )

    rows: list[dict] = []
    for cond in args.conditions:
        split, shift, noise = CONDITIONS[cond]
        overrides = {"lidar": {"noise_std": noise}} if noise else {}
        env_config = build_env_config(
            overrides, split=split, shift=shift, n_worlds=args.episodes
        )

        for actor_name, spec in specs.items():
            print(f"[{cond:>12}] {actor_name} ...", end=" ", flush=True)
            actor = build_actor(robot=env_config.robot, **spec)
            metrics, results = evaluate(actor, env_config)
            print(
                f"SR={metrics.success_rate:.3f} SPL={metrics.spl:.3f} "
                f"coll={metrics.collision_rate:.3f}",
                flush=True,
            )
            rows.append(
                {
                    "condition": cond,
                    "split": split,
                    "shift": shift or "none",
                    "lidar_noise_std": noise,
                    "actor": actor_name,
                    **metrics.to_dict(),
                }
            )
            metrics.save(out_dir / f"{cond}__{actor_name}.json")

    (out_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = render_table(rows)
    (out_dir / "benchmark.md").write_text(table, encoding="utf-8")

    print()
    print(table)
    print(f"\nWrote {out_dir / 'benchmark.json'} and {out_dir / 'benchmark.md'}")
    return 0


def render_table(rows: list[dict]) -> str:
    """Render the matrix as a Markdown table, ready for the report."""
    header = (
        "| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Benchmark results",
        "",
        "Every row is evaluated on the same worlds in the same order.",
        "`nominal` is the held-out in-distribution test split; the remaining",
        "conditions are distribution shifts never seen during training.",
        "",
        header,
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['actor']} | {r['success_rate']:.3f} | "
            f"{r['spl']:.3f} | {r['collision_rate']:.3f} | "
            f"{r['timeout_rate']:.3f} | {r['mean_steps_to_goal']:.0f} |"
        )
    lines += [
        "",
        "## Reading this table",
        "",
        "- `random` is the floor. Any result that does not clearly clear it is noise.",
        "- **`noisy_lidar` is a no-op for `classical` by construction.** The classical",
        "  baseline navigates from the map and never reads the lidar, so its row is",
        "  identical to `nominal`. That is not robustness — it is non-exposure, and it",
        "  is exactly the axis on which a sensor-driven policy should be expected to",
        "  differ.",
        "- SPL can reach 1.000 when the agent stops inside the goal tolerance and so",
        "  travels slightly less than `l*`; the `max(p, l*)` term caps it there.",
        "",
        f"_Shift definitions: {', '.join(sorted(SHIFTS))}._",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
