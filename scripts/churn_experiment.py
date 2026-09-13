"""Is the hand-written baseline's motion cost caused by replanning churn?

Report Section 9.4 measured that moving obstacles cost the hand-written
classical stack 0.110-0.160 success where they cost the learned policy only
0.048-0.068, and attributed the difference to *path churn*: the baseline
re-commits to a fresh global plan every second, and Section 9.1 separately
measured that pathology costing `narrow` 0.160 in a static world. That
explanation fit every number available but was never isolated. This isolates
it.

Churn is operationalised as the shift in the lookahead point the controller
is steering at, measured across a replan from an unchanged pose. It is
literally "how far the commitment moved when the plan was rebuilt".

Two things are tested, in order, because the first is cheap and can kill the
hypothesis outright:

1. **The differential prediction.** If churn drives the motion cost, churn
   must be elevated specifically where that cost is largest -- dense worlds
   with movers -- and not merely wherever replanning happens. A 2x2 over
   clutter and motion measures that. Churn that is flat across the four cells,
   or that tracks replan count rather than condition, falsifies the story.

2. **The intervention.** ``replan_on_block`` keeps the benefit of replanning
   (re-route when a mover actually blocks the committed path) and removes the
   gratuitous part (rebuilding a working plan on a timer). If churn is causal,
   this improves `dynamic_dense` markedly and leaves the frozen conditions
   roughly alone. If it improves everything equally, or nothing, the
   explanation was wrong.

    python scripts/churn_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.classical import AStarPursuitAgent, PursuitConfig
from vision_nav.envs.splits import DYNAMIC_CONDITIONS, FROZEN_CONDITIONS
from vision_nav.training.env_factory import build_env_config

#: The 2x2 over clutter and motion, as (label, condition name).
CELLS = [
    ("sparse moving", "dynamic"),
    ("sparse frozen", "dynamic_frozen"),
    ("dense moving", "dynamic_dense"),
    ("dense frozen", "dynamic_dense_frozen"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--replan-every", type=int, default=10)
    p.add_argument("--out", default="results/churn_experiment.json")
    return p.parse_args(argv)


def run_cell(condition: str, episodes: int, config: PursuitConfig) -> dict:
    """One condition, per-episode churn alongside the outcome."""
    split, shift, noise = DYNAMIC_CONDITIONS[condition]
    overrides = {"freeze_dynamic": True} if condition in FROZEN_CONDITIONS else {}
    env_config = build_env_config(dict(overrides), split=split, shift=shift,
                                  n_worlds=episodes)
    env_config = env_config.__class__(**{**env_config.__dict__})

    from vision_nav.envs.nav_env import ProceduralNavEnv

    env = ProceduralNavEnv(env_config)
    agent = AStarPursuitAgent(config, robot=env_config.robot)

    rows = []
    for seed in env_config.world_seeds:
        env.reset(options={"world_seed": int(seed)})
        agent.robot = env.config.robot
        if not agent.start_episode(env.world, env.robot.pose):
            rows.append({"seed": int(seed), "success": False, "collision": False,
                         "timeout": True, "churn_mean": 0.0, "churn_max": 0.0,
                         "replans": 0, "planning_failure": True})
            continue

        done = False
        info: dict = {}
        while not done:
            action = agent.act(env.robot.pose)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rows.append({
            "seed": int(seed),
            "success": bool(info.get("is_success")),
            "collision": bool(info.get("collision")),
            "timeout": not info.get("is_success") and not info.get("collision"),
            "churn_mean": agent.churn_mean,
            "churn_max": agent.churn_max,
            "replans": agent.replans,
            "planning_failure": False,
        })

    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    churn = np.array([r["churn_mean"] for r in rows])
    coll = np.array([r["collision"] for r in rows], dtype=bool)
    succ = np.array([r["success"] for r in rows], dtype=bool)
    out = {
        "n": len(rows),
        "success_rate": float(succ.mean()),
        "collision_rate": float(coll.mean()),
        "churn_mean": float(churn.mean()),
        "churn_p90": float(np.percentile(churn, 90)),
        "replans_mean": float(np.mean([r["replans"] for r in rows])),
        "per_episode": rows,
    }
    # Within-condition: do the episodes that collided churn more than those
    # that did not? Confounded by difficulty -- harder worlds churn more AND
    # collide more -- so this is supporting evidence, not the causal test.
    if coll.any() and (~coll).any():
        out["churn_when_collided"] = float(churn[coll].mean())
        out["churn_when_not"] = float(churn[~coll].mean())
        out["churn_gap"] = out["churn_when_collided"] - out["churn_when_not"]
    return out


#: Recorded before the intervention was run.
#:
#: If churn causes the motion cost, replacing the timer with block-triggered
#: replanning should recover `dynamic_dense` by at least +0.05 while leaving
#: the frozen cells within noise, and churn should fall where success rises.
#: A uniform improvement across all four cells would instead mean the timer
#: was simply a bad setting everywhere, which is a different claim. No change
#: anywhere falsifies the explanation outright.
PREDICTION = (
    "dense moving improves by >= +0.05; frozen cells within +/-0.03; "
    "churn falls wherever success rises"
)


def arm(name: str, config: PursuitConfig, episodes: int) -> dict:
    print("\n=== {} ===".format(name))
    print("{:16s} {:>8s} {:>7s} {:>9s} {:>9s} {:>10s}".format(
        "cell", "success", "coll", "churn(m)", "replans", "coll-gap"))
    cells = {}
    for label, cond in CELLS:
        r = run_cell(cond, episodes, config)
        cells[cond] = r
        print("{:16s} {:>8.3f} {:>7.3f} {:>9.3f} {:>9.1f} {:>10s}".format(
            label, r["success_rate"], r["collision_rate"], r["churn_mean"],
            r["replans_mean"],
            "{:+.3f}".format(r["churn_gap"]) if "churn_gap" in r else "-"))
    return cells


def main(argv=None) -> int:
    args = parse_args(argv)

    report = {"replan_every": args.replan_every, "episodes": args.episodes,
              "prediction": PREDICTION, "arms": {}}

    report["arms"]["timer"] = arm(
        "timed replanning (replan_every={})".format(args.replan_every),
        PursuitConfig(replan_every=args.replan_every), args.episodes)
    report["arms"]["on_block"] = arm(
        "block-triggered replanning",
        PursuitConfig(replan_on_block=True), args.episodes)

    print("\n=== intervention effect ===")
    print("{:16s} {:>10s} {:>10s} {:>9s} {:>9s}".format(
        "cell", "timer", "on-block", "d success", "d churn"))
    for label, cond in CELLS:
        t, b = report["arms"]["timer"][cond], report["arms"]["on_block"][cond]
        print("{:16s} {:>10.3f} {:>10.3f} {:>+9.3f} {:>+9.3f}".format(
            label, t["success_rate"], b["success_rate"],
            b["success_rate"] - t["success_rate"],
            b["churn_mean"] - t["churn_mean"]))
    print("\npre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
