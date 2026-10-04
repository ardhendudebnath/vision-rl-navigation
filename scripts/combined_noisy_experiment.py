"""Do the back end and the obstacle range together close `noisy_lidar`?

`noisy_lidar` is the one condition where this stack still trails Nav2 in the
open: 0.89 on the held-out worlds against Nav2's 1.000. Two repairs have each
been measured against it and each left it where it was.

* The **back end** (§9.13) is inert under noise: 0.89 -> 0.89 on the held-out
  worlds, pose error p95 down 2%. It refuses 46% of its loop closures there,
  against 9-10% in clutter, because a noisy scan matched against a noisy map
  rarely clears the score gate.
* The **obstacle range** (§9.14) removes 93% of the phantom cells and brings the
  floor lost down to the noise-free figure, and moves arrivals by two in a
  hundred on val, 0.87 -> 0.89.

The mechanism for why the two might do together what neither does alone is in
those two sentences: the closures the back end refuses are scored against the
map, and the obstacle range is what cleans the map. So this is a 2x2, which is
the only design that separates "the pair helps" from "one of them helps":

  front_end       neither rule -- the published stack
  obstacle_range  the map repaired, the pose left alone
  back_end        the pose graph, on the unrepaired map
  both            the pose graph on the repaired map

Each arm runs in its own process and checkpoints after every episode, because
the runs in this project have been killed partway by the tool's background
limit, and a long run that writes only at the end loses everything to one stop.

    python scripts/combined_noisy_experiment.py --split val --episodes 100 --arms both
    python scripts/combined_noisy_experiment.py --split val --episodes 100 --score
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

COND = "noisy_lidar"
#: ``arm: (noise_margin in sigmas, pose graph on)``.
ARMS = {
    "front_end": (0.0, False),
    "obstacle_range": (3.0, False),
    "back_end": (0.0, True),
    "both": (3.0, True),
}


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


def episode(env, seed: int, cfg, margin: float, graph: bool) -> dict:
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, noise_margin=margin,
                                  pose_graph=graph)
    agent.start_episode(env.world, env.robot.pose)
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    errors = np.asarray(agent.pose_errors) if agent.pose_errors else np.zeros(1)
    g = agent.graph
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": len(errors),
        "goal_distance": float(info.get("goal_distance", np.nan)),
        "pose_err_median": float(np.median(errors)),
        "pose_err_p95": float(np.percentile(errors, 95)),
        "replans": int(agent.replans),
        "closures": int(g.closures) if g else 0,
        "closures_rejected": int(g.closures_rejected) if g else 0,
    }


def config(split: str, episodes: int):
    own_split, shift, noise = BENCHMARK_CONDITIONS[COND]
    band = own_split if split == "test" else split
    cfg = build_env_config({"lidar": {"noise_std": noise}}, split=band, shift=shift,
                           n_worlds=episodes)
    return cfg, band, [int(s) for s in list(cfg.world_seeds)[:episodes]]


def checkpoint(stem: Path, arm: str) -> Path:
    return stem.with_name(f"{stem.name}.{arm}.jsonl")


def load_arm(stem: Path, arm: str) -> dict:
    path = checkpoint(stem, arm)
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {r["seed"]: r for r in rows}


def run(args, stem: Path) -> int:
    cfg, band, seeds = config(args.split, args.episodes)
    env = ProceduralNavEnv(cfg)
    for arm in args.arms:
        margin, graph = ARMS[arm]
        done = load_arm(stem, arm)
        print(f"{arm}: {len(done)} of {len(seeds)} already checkpointed ({band})", flush=True)
        for s in seeds:
            if s in done:
                continue
            row = episode(env, s, cfg, margin, graph)
            with checkpoint(stem, arm).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        print(f"{arm}: done", flush=True)
    return 0


def score(args, stem: Path) -> int:
    _, band, seeds = config(args.split, args.episodes)
    arms = {}
    for arm in ARMS:
        rows = load_arm(stem, arm)
        missing = [s for s in seeds if s not in rows]
        if missing:
            print(f"{arm}: {len(missing)} episodes still missing; not scoring")
            return 1
        arms[arm] = [rows[s] for s in seeds]
    rng = np.random.default_rng(20261004)
    succ = {k: np.array([r["success"] for r in v], dtype=bool) for k, v in arms.items()}
    report: dict = {"split": args.split, "band": band, "episodes": len(seeds),
                    "condition": COND, "arms": {}, "contrasts": {}}
    for arm, rows in arms.items():
        report["arms"][arm] = {
            "episodes": rows,
            "success": float(succ[arm].mean()),
            "collisions": int(sum(r["collision"] for r in rows)),
            "pose_err_median": float(np.median([r["pose_err_median"] for r in rows])),
            "pose_err_p95": float(np.mean([r["pose_err_p95"] for r in rows])),
            "replans": float(np.mean([r["replans"] for r in rows])),
            "closures": float(np.mean([r["closures"] for r in rows])),
            "rejected": float(np.mean([r["closures_rejected"] for r in rows])),
        }
    for a, b in (("front_end", "both"), ("front_end", "obstacle_range"),
                 ("front_end", "back_end"), ("obstacle_range", "both"),
                 ("back_end", "both")):
        pv, lost, won = mcnemar_p(succ[a], succ[b])
        report["contrasts"][f"{b}_vs_{a}"] = {
            "gain": float(succ[b].mean() - succ[a].mean()), "mcnemar_p": pv,
            "won": won, "lost": lost,
            "ci": paired_ci(succ[a].astype(float), succ[b].astype(float), rng)}
    # The interaction: is the pair worth more than the sum of its parts?
    f, o, g, both = (succ[k].mean() for k in ("front_end", "obstacle_range",
                                              "back_end", "both"))
    report["interaction"] = float((both - g) - (o - f))

    print(f"noisy_lidar, {args.split} band ({band}), {len(seeds)} worlds\n")
    for arm, e in report["arms"].items():
        tried = e["closures"] + e["rejected"]
        refused = f"{e['rejected'] / tried:.0%}" if tried else "--"
        print(f"  {arm:15s} success {e['success']:.2f}  collisions {e['collisions']}  "
              f"pose median {e['pose_err_median']:.3f} m  p95 {e['pose_err_p95']:.3f} m  "
              f"replans {e['replans']:.0f}  closures refused {refused}")
    print()
    for name, c in report["contrasts"].items():
        print(f"  {name:26s} {c['gain']:+.3f}  McNemar p={c['mcnemar_p']:.4f} "
              f"({c['won']} won / {c['lost']} lost)  CI [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]")
    print(f"\n  interaction (both - back end) - (obstacle range - front end): "
          f"{report['interaction']:+.3f}")
    out = stem.with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--score", action="store_true",
                   help="Score the four checkpointed arms instead of driving.")
    p.add_argument("--stem", default=None,
                   help="Results path without extension; defaults by split.")
    args = p.parse_args(argv)
    stem = Path(args.stem or f"results/combined_noisy_{args.split}")
    return score(args, stem) if args.score else run(args, stem)


if __name__ == "__main__":
    raise SystemExit(main())
