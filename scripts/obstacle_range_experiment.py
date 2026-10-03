"""An obstacle range inside the sensor's maximum: the rest of the noise regression?

§9.6 repaired the dense-scan map under noise with a corroboration rule and
found its own mechanism wanting -- the phantom cells barely moved, 370 to 309.
`clearing_diagnostic.py` found where the 309 come from. 190 of them sit more
than half a metre from any real surface, which Gaussian range noise of 0.1 m
cannot do, and the sensor explains it: :meth:`Lidar2D.scan` adds its noise
*after* clipping to maximum range and clips again, so a beam that hit nothing
reads the maximum plus noise and half of those come back just under it. The
mapper took every reading below the maximum as a surface, and planted a ring of
phantoms at the edge of the sensor's reach in open floor.

The repair is Nav2's ``obstacle_max_range``: readings within three standard
deviations of the sensor's range noise of its maximum never mark a surface,
though they still clear the space they cross. On one val world, open loop, it
took the phantoms from 211 to 15 and the free floor lost from 0.319 to 0.051.
With a noise-free sensor the line sits at the maximum, so the rule is inert on
five of the six conditions by construction, and ``nominal`` is driven here as
the control that checks it, episode for episode.

    python scripts/obstacle_range_experiment.py --split val --episodes 25
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

CELLS = ("noisy_lidar", "nominal")
#: Standard deviations of the sensor's own range noise.
MARGIN = 3.0
ARMS = {"front_end": 0.0, "obstacle_range": MARGIN}


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


def paired_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = 10000) -> list[float]:
    n = len(a)
    diffs = [b[i].mean() - a[i].mean()
             for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def episode(env, seed: int, cfg, margin: float) -> dict:
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, noise_margin=margin)
    agent.start_episode(env.world, env.robot.pose)
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    errors = np.asarray(agent.pose_errors) if agent.pose_errors else np.zeros(1)
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": len(errors),
        "goal_distance": float(info.get("goal_distance", np.nan)),
        "pose_err_median": float(np.median(errors)),
        "replans": int(agent.replans),
    }


def run_cell(cond: str, split: str, episodes: int, ckpt: Path, kept: dict) -> dict:
    own_split, shift, noise = BENCHMARK_CONDITIONS[cond]
    band = own_split if split == "test" else split
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=band, shift=shift, n_worlds=episodes)
    env = ProceduralNavEnv(cfg)
    seeds = [int(s) for s in list(cfg.world_seeds)[:episodes]]
    arms = {}
    for arm, margin in ARMS.items():
        rows = []
        for s in seeds:
            key = (cond, arm, s)
            if key not in kept:
                kept[key] = episode(env, s, cfg, margin)
                with ckpt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"cell": cond, "arm": arm, "row": kept[key]}) + "\n")
            rows.append(kept[key])
        arms[arm] = rows
    return {"band": band, "seeds": seeds, **{k: {"episodes": v} for k, v in arms.items()}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--cells", nargs="+", default=list(CELLS))
    p.add_argument("--out", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out or f"results/obstacle_range_{args.split}.json")
    ckpt = out.with_suffix(".episodes.jsonl")
    kept: dict = {}
    if args.resume and ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            kept[(rec["cell"], rec["arm"], rec["row"]["seed"])] = rec["row"]
        print(f"resuming with {len(kept)} episodes already checkpointed", flush=True)

    rng = np.random.default_rng(20261003)
    report: dict = {"split": args.split, "episodes": args.episodes,
                    "margin_sigmas": MARGIN, "cells": {}}
    for cond in args.cells:
        cell = run_cell(cond, args.split, args.episodes, ckpt, kept)
        a = np.array([r["success"] for r in cell["front_end"]["episodes"]], dtype=bool)
        b = np.array([r["success"] for r in cell["obstacle_range"]["episodes"]], dtype=bool)
        pv, lost, won = mcnemar_p(a, b)
        cell.update({
            "success": [float(a.mean()), float(b.mean())],
            "gain": float(b.mean() - a.mean()), "mcnemar_p": pv,
            "won": won, "lost": lost,
            "ci": paired_ci(a.astype(float), b.astype(float), rng),
            "collisions": [sum(int(r["collision"]) for r in cell[k]["episodes"])
                           for k in ARMS],
            "pose_err_median": [float(np.median([r["pose_err_median"]
                                                 for r in cell[k]["episodes"]]))
                                for k in ARMS],
            "replans": [float(np.mean([r["replans"] for r in cell[k]["episodes"]]))
                        for k in ARMS],
            # The identity check: a noise-free cell must come out the same
            # episode by episode, not merely at the same rate.
            "identical": all(x == y for x, y in zip(cell["front_end"]["episodes"],
                                                     cell["obstacle_range"]["episodes"],
                                                     strict=True)),
        })
        report["cells"][cond] = cell
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{cond:12s} ({cell['band']})  SR {a.mean():.2f} -> {b.mean():.2f} "
              f"({b.mean() - a.mean():+.3f})  McNemar p={pv:.4f} ({won} won / {lost} "
              f"lost)  CI [{cell['ci'][0]:+.2f}, {cell['ci'][1]:+.2f}]  collisions "
              f"{cell['collisions'][0]} -> {cell['collisions'][1]}  replans "
              f"{cell['replans'][0]:.0f} -> {cell['replans'][1]:.0f}  pose median "
              f"{cell['pose_err_median'][0]:.3f} -> {cell['pose_err_median'][1]:.3f}  "
              f"identical {cell['identical']}", flush=True)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
