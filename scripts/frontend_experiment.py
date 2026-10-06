"""Does slam_toolbox's kind of front end close `noisy_lidar` without costing the rest?

:class:`CorrelativeFrontEnd` (:mod:`vision_nav.mapping.frontend`) copies the
three ways slam_toolbox's scan matcher differs from this stack's: keyframes
rather than every scan, a buffer of recent keyframe scans rather than the whole
map, and a search fifteen times wider in heading. §12 named it as the one lever
left on `noisy_lidar` after five others were set aside.

A train-band check on six worlds already showed it is not a clear win: under
noise the final pose error fell from 0.259 m to 0.218 m, and on clean worlds
the median pose error rose from 0.044 m to 0.071 m, because matching only every
half metre lets the odometry drift in between where the published matcher
tracks every step. Six worlds decide nothing, so this measures it on all 100 val
worlds, on the target condition and two controls:

  noisy_lidar   the condition it is for
  nominal       open and clean, where the published matcher is at its best
  dense         clutter, where §9.11 found the pose is what limits the tight gaps

Development on val; nothing here is registered and nothing is scored. Each
(cell, arm) runs in its own process and checkpoints after every episode.

    python scripts/frontend_experiment.py --cell noisy_lidar --arm correlative
    python scripts/frontend_experiment.py --score
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.localisation import OdometryConfig
from vision_nav.training.env_factory import build_env_config

CELLS = ("noisy_lidar", "nominal", "dense")
ARMS = ("map", "correlative")


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


def config(cell: str, episodes: int):
    _, shift, noise = BENCHMARK_CONDITIONS[cell]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {}, split="val",
                           shift=shift, n_worlds=episodes)
    return cfg, noise, [int(s) for s in list(cfg.world_seeds)[:episodes]]


def episode(env, seed: int, cfg, noise: float, arm: str) -> dict:
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360", noise_std=noise,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, front_end=arm)
    agent.start_episode(env.world, env.robot.pose)
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        _, _, term, trunc, info = env.step(agent.act(env.robot.pose, env.robot.velocity))
        if term or trunc:
            break
    err = np.asarray(agent.pose_errors) if agent.pose_errors else np.zeros(1)
    fe = agent.correlative
    return {"seed": seed, "success": bool(info.get("is_success")),
            "collision": bool(info.get("collision")), "steps": len(err),
            "goal_distance": float(info.get("goal_distance", np.nan)),
            "pose_err_median": float(np.median(err)),
            "pose_err_p95": float(np.percentile(err, 95)),
            "pose_err_final": float(err[-1]),
            "matches": int(fe.matches) if fe else 0,
            "refused": int(fe.refused) if fe else 0}


def checkpoint(stem: Path, cell: str, arm: str) -> Path:
    return stem.with_name(f"{stem.name}.{cell}.{arm}.jsonl")


def load(stem: Path, cell: str, arm: str) -> dict:
    p = checkpoint(stem, cell, arm)
    if not p.exists():
        return {}
    return {r["seed"]: r for r in (json.loads(x) for x in
                                   p.read_text(encoding="utf-8").splitlines())}


def run(args, stem: Path) -> int:
    cfg, noise, seeds = config(args.cell, args.episodes)
    env = ProceduralNavEnv(cfg)
    done = load(stem, args.cell, args.arm)
    print(f"{args.cell}/{args.arm}: {len(done)} of {len(seeds)} checkpointed", flush=True)
    for s in seeds:
        if s in done:
            continue
        row = episode(env, s, cfg, noise, args.arm)
        with checkpoint(stem, args.cell, args.arm).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    print(f"{args.cell}/{args.arm}: done", flush=True)
    return 0


def score(args, stem: Path) -> int:
    report: dict = {"split": "val", "episodes": args.episodes, "cells": {}}
    for cell in CELLS:
        _, _, seeds = config(cell, args.episodes)
        arms = {}
        for arm in ARMS:
            rows = load(stem, cell, arm)
            if any(s not in rows for s in seeds):
                print(f"{cell}/{arm}: incomplete; not scoring")
                return 1
            arms[arm] = [rows[s] for s in seeds]
        a = np.array([r["success"] for r in arms["map"]], dtype=bool)
        b = np.array([r["success"] for r in arms["correlative"]], dtype=bool)
        pv, lost, won = mcnemar_p(a, b)

        def med(arm, key, arms=arms):
            return float(np.median([r[key] for r in arms[arm]]))

        entry = {"arms": {k: {"episodes": v} for k, v in arms.items()},
                 "success": [float(a.mean()), float(b.mean())],
                 "gain": float(b.mean() - a.mean()), "mcnemar_p": pv,
                 "won": won, "lost": lost,
                 "collisions": [sum(int(r["collision"]) for r in arms[k]) for k in ARMS],
                 "pose_err_median": [med(k, "pose_err_median") for k in ARMS],
                 "pose_err_final": [med(k, "pose_err_final") for k in ARMS],
                 "matches": float(np.mean([r["matches"] for r in arms["correlative"]])),
                 "refused": float(np.mean([r["refused"] for r in arms["correlative"]]))}
        report["cells"][cell] = entry
        print(f"{cell:12s} success {a.mean():.2f} -> {b.mean():.2f} ({b.mean() - a.mean():+.3f})  "
              f"McNemar p={pv:.4f} ({won} won / {lost} lost)  collisions "
              f"{entry['collisions'][0]} -> {entry['collisions'][1]}")
        print(f"{'':12s} pose median {entry['pose_err_median'][0]:.3f} -> "
              f"{entry['pose_err_median'][1]:.3f} m  final {entry['pose_err_final'][0]:.3f} -> "
              f"{entry['pose_err_final'][1]:.3f} m  matches {entry['matches']:.0f}  refused "
              f"{entry['refused']:.1f}")
    out = stem.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", choices=CELLS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--score", action="store_true")
    p.add_argument("--stem", default="results/frontend_val")
    args = p.parse_args(argv)
    stem = Path(args.stem)
    if args.score:
        return score(args, stem)
    if not (args.cell and args.arm):
        p.error("--cell and --arm are required unless --score")
    return run(args, stem)


if __name__ == "__main__":
    raise SystemExit(main())
