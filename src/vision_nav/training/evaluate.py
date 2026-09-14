"""Evaluation harness — the single place results come from.

Every number in the report is produced by :func:`evaluate`, on an env built
with ``deterministic_seed_order=True``.  That means all actors see the same
worlds in the same order, so differences between them are attributable to the
policy and not to episode sampling.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vision_nav.envs.nav_env import NavEnvConfig, ProceduralNavEnv
from vision_nav.metrics.navigation import EpisodeResult, NavigationMetrics, aggregate
from vision_nav.training.actors import Actor, build_actor
from vision_nav.training.env_factory import build_env_config

__all__ = ["rollout", "evaluate", "main"]


def rollout(
    env: ProceduralNavEnv,
    actor: Actor,
    world_seed: int | None = None,
    collect_trajectory: bool = False,
) -> tuple[EpisodeResult, np.ndarray | None]:
    """Run one episode and return its result (and optionally its path)."""
    options = {"world_seed": world_seed} if world_seed is not None else None
    obs, info = env.reset(options=options)

    if not actor.reset(env, obs):
        # The actor could not produce a plan.  Recorded as an honest failure
        # rather than skipped — dropping it would inflate the success rate.
        return (
            EpisodeResult(
                world_seed=int(info["world_seed"]),
                success=False,
                collision=False,
                steps=0,
                path_length=0.0,
                shortest_path_length=float(info["shortest_path_length"]),
                final_goal_distance=float(info["goal_distance"]),
            ),
            None,
        )

    trajectory = [env.robot.position.copy()] if collect_trajectory else None

    while True:
        action = actor.act(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        if trajectory is not None:
            trajectory.append(env.robot.position.copy())
        if terminated or truncated:
            break

    path = np.asarray(trajectory) if trajectory is not None else None
    return EpisodeResult.from_info(info), path


def evaluate(
    actor: Actor,
    env_config: NavEnvConfig,
    n_episodes: int | None = None,
    world_seeds: Sequence[int] | None = None,
    progress: bool = False,
) -> tuple[NavigationMetrics, list[EpisodeResult]]:
    """Score an actor over an evaluation set.

    Parameters
    ----------
    world_seeds:
        Explicit worlds to evaluate on.  Defaults to the env config's pool,
        which is what keeps different actors on identical episodes.
    """
    env = ProceduralNavEnv(env_config)
    seeds = list(world_seeds) if world_seeds is not None else list(env_config.world_seeds or [])
    if not seeds:
        raise ValueError(
            "no evaluation worlds: pass world_seeds, or set world_seeds on the "
            "env config (see vision_nav.envs.splits.split_seeds)"
        )
    if n_episodes is not None:
        seeds = seeds[:n_episodes]

    results: list[EpisodeResult] = []
    for i, seed in enumerate(seeds, 1):
        result, _ = rollout(env, actor, world_seed=seed)
        results.append(result)
        if progress and (i % 25 == 0 or i == len(seeds)):
            running = aggregate(results)
            print(
                f"  [{i:>4}/{len(seeds)}] SR={running.success_rate:.3f} "
                f"SPL={running.spl:.3f} coll={running.collision_rate:.3f}",
                flush=True,
            )

    return aggregate(results), results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a navigation actor on a held-out world split."
    )
    p.add_argument("--actor", default="classical", choices=["classical", "random", "rl"])
    p.add_argument("--model", default=None, help="Path to a saved SB3 model (actor=rl)")
    p.add_argument("--split", default="test", help="World split name")
    p.add_argument("--shift", default=None, help="Named world distribution shift")
    p.add_argument("--episodes", type=int, default=None, help="Cap on episode count")
    p.add_argument("--lidar-noise", type=float, default=None, help="Range noise std (m)")
    p.add_argument("--out", default=None, help="Write metrics JSON here")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    overrides: dict = {}
    if args.lidar_noise is not None:
        overrides["lidar"] = {"noise_std": args.lidar_noise}

    env_config = build_env_config(
        overrides, split=args.split, shift=args.shift, n_worlds=args.episodes
    )
    actor = build_actor(args.actor, model_path=args.model, robot=env_config.robot)

    label = f"{actor.name} | split={args.split}"
    if args.shift:
        label += f" | shift={args.shift}"
    if args.lidar_noise:
        label += f" | lidar noise={args.lidar_noise}m"

    if not args.quiet:
        print(f"Evaluating {label}", flush=True)
    metrics, results = evaluate(actor, env_config, progress=not args.quiet)
    print()
    print(metrics.as_table(label))

    if args.out:
        out = Path(args.out)
        metrics.save(out)
        episodes_path = out.with_name(out.stem + "_episodes.json")
        episodes_path.write_text(
            json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
        )
        print(f"\nWrote {out} and {episodes_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
