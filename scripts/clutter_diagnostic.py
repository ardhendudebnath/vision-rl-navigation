"""Where does the time go in clutter, when the map is right but unfinished?

Report §9.6 left this as the larger half of what this stack still pays at 360
beams, and `dense_map_diagnostic.py` ruled out the obvious explanation: in
clutter the map holds no phantom cells at either beam count and blocks *less*
floor than the truth, because unexplored space counts as free. The map is
accurate and unfinished. What the stack does with it is the question.

This measures the driving rather than the mapping, per episode:

  driven / shortest   how far the robot travelled against the geodesic it could
                      have taken -- the cost of committing to routes through
                      space nobody has looked at yet
  wandering           1 - displacement / driven, over the whole episode: what
                      share of the driving did not end up moving the robot
  replans, recovery   how often the plan was rebuilt, and how many steps were
                      spent turning on the spot after a plan failed
  slow                share of steps below half speed, which is the reactive
                      slow-down near obstacles
  reached             how much of the journey was completed before time ran out

Both with the stack as §9.6 left it and with a commitment rule that stands by
the near part of the route; report §9.7 reports what that was worth. The
``frontier`` arm added afterwards tests the remaining candidate -- planning
through known free space only -- against the prediction registered in
``PREDICTION`` below.

What it is measured against is the published Nav2 row: with SLAM at 360 beams
Nav2 reaches 0.820 on `dense` and 0.810 on `narrow`, where this stack reaches
0.620 and 0.590 (§9.5, §9.6). Those are test-band numbers and this is a val-band
tool, so the comparison is made in the prose rather than here.

Run on the ``val`` seed band, which no experiment scores.

    python scripts/clutter_diagnostic.py --episodes 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.localisation import OdometryConfig
from vision_nav.training.env_factory import build_env_config

CONDITIONS = ("dense", "narrow", "nominal")


#: The arms: the stack as §9.6 left it, commitment gated on distance,
#: commitment gated on time with a small imminent-blockage override, and
#: planning through known free space only (:mod:`vision_nav.planning.frontier`).
ARMS = {
    "as_published": {},
    "committed": {"commit": True},
    "committed_interval": {"commit": True, "commit_distance": 0.6, "commit_interval": 10},
    "frontier": {"frontier": True},
}

#: Registered before the ``frontier`` arm had been run on any seed, val or
#: otherwise. §9.7 rejected two commitment rules and left the optimism itself as
#: the thing to change: the planner proposes routes that end at something nobody
#: has looked at, and the next scan invalidates them. If that diagnosis is
#: right, refusing to propose such a route should
#:
#:   1. raise success on the clutter conditions -- ``dense`` and ``narrow``
#:      pooled -- by at least ``+0.05``. This is the endpoint; the rest is
#:      mechanism and cannot rescue a miss here.
#:   2. cut ``replans`` per episode by at least a quarter on those conditions,
#:      because a route that crosses only seen floor is not invalidated by what
#:      the next scan reveals.
#:   3. cost something on ``nominal``, where the unknown mostly *is* free and
#:      the optimistic shortcut is mostly right. Registered as a bound rather
#:      than a direction: success no more than ``0.03`` below ``as_published``,
#:      this project's null band. A larger drop is a real cost and is to be
#:      reported as one.
#:
#: The obvious way for this to fail: a frontier route advances one cell into the
#: dark at a time, so the robot may explore tidily and run out of steps --
#: ``driven_over_shortest`` up, ``wandering`` down, success flat. That pattern
#: would say the optimism was buying more than §9.7 credited it with.
PREDICTION = {
    "clutter_success_gain_at_least": 0.05,
    "clutter_replan_reduction_at_least": 0.25,
    "nominal_success_loss_at_most": 0.03,
}


def episode(env, seed: int, cfg, arm: str = "as_published") -> dict:
    env.reset(options={"world_seed": seed})
    settings = ARMS[arm]
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, commit=settings.get("commit", False),
                                  frontier=settings.get("frontier", False))
    for key in ("commit_distance", "commit_interval"):
        if key in settings:
            setattr(agent, key, settings[key])
    agent.start_episode(env.world, env.robot.pose)
    start = env.robot.pose[:2].copy()
    previous = start.copy()
    driven, slow, steps = 0.0, 0, 0
    visited, headings, goal_distance = [], [], []
    ahead: list[float] = []
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        # How far the route the robot is steering by extends past it. Pure
        # pursuit needs something to aim at: a route that stops a cell ahead
        # gives it nothing, which is one way frontier planning could hurt
        # without the planner itself being wrong.
        ahead.append(agent._track_length(agent._track, agent._cursor))
        _, _, terminated, truncated, info = env.step(action)
        here = env.robot.pose[:2].copy()
        driven += float(np.linalg.norm(here - previous))
        previous = here
        steps += 1
        visited.append(here)
        headings.append(float(env.robot.pose[2]))
        goal_distance.append(float(np.linalg.norm(env.world.goal - here)))
        if abs(float(env.robot.velocity[0])) < 0.5 * cfg.robot.max_linear_vel:
            slow += 1
        if terminated or truncated:
            break
    shortest = float(info.get("shortest_path_length", np.nan))
    displacement = float(np.linalg.norm(previous - start))
    # How far the episode spread out, and how often it turned back on itself:
    # a small gyration with many reversals is a robot arguing with itself in
    # one place, which is a different failure from wandering and not arriving.
    p = np.asarray(visited)
    gyration = float(np.sqrt(((p - p.mean(axis=0)) ** 2).sum(axis=1).mean()))
    h = np.asarray(headings)
    reversals = int(np.sum(np.abs(np.angle(np.exp(1j * (h[10:] - h[:-10])))) > np.pi / 2)) \
        if len(h) > 10 else 0
    return {
        "gyration": gyration, "reversals": reversals,
        "closest_to_goal": float(min(goal_distance)),
        "pose_error": float(agent.pose_errors[-1]) if agent.pose_errors else 0.0,
        "success": bool(info.get("is_success")), "collision": bool(info.get("collision")),
        "steps": steps, "driven": driven, "displacement": displacement,
        "driven_over_shortest": driven / shortest if shortest else np.nan,
        "wandering": 1.0 - displacement / driven if driven > 0 else 0.0,
        "reached": 1.0 - float(info.get("goal_distance", 0.0)) / shortest if shortest else np.nan,
        "replans": int(agent.replans), "recovery_steps": int(agent.recovery_steps),
        "failed_plans": len(agent.failed_plan_steps),
        "plans_refused": int(agent.plans_refused),
        "frontier_plans": int(agent.frontier_plans),
        "goal_plans": int(agent.goal_plans),
        "plan_ahead": float(np.mean(ahead)) if ahead else 0.0,
        "slow_fraction": slow / max(steps, 1),
        "known": float(agent.map.known_fraction),
    }


def summarise(rows: list[dict], keys) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                   help="Which arms to run; a narrower comparison on more worlds "
                        "is how a candidate that looks like a wash gets decided.")
    p.add_argument("--out", default="results/clutter_diagnostic.json")
    args = p.parse_args(argv)

    keys = ("steps", "driven", "driven_over_shortest", "wandering", "reached",
            "replans", "recovery_steps", "failed_plans", "plans_refused",
            "frontier_plans", "goal_plans", "plan_ahead",
            "slow_fraction", "known", "gyration", "reversals", "closest_to_goal",
            "pose_error")
    report: dict = {"episodes": args.episodes, "split": "val", "arms": args.arms,
                    "conditions": {}}
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
        report["conditions"][cond] = {}
        for name in args.arms:
            rows = [episode(env, s, cfg, name) for s in seeds]
            won = [r for r in rows if r["success"]]
            lost = [r for r in rows if not r["success"]]
            entry = {"success": float(np.mean([r["success"] for r in rows])),
                     "all": summarise(rows, keys),
                     "successes": summarise(won, keys) if won else None,
                     "failures": summarise(lost, keys) if lost else None}
            report["conditions"][cond][name] = entry
            a = entry["all"]
            print(f"{cond:8s} {name:19s} SR {entry['success']:.2f}  "
                  f"driven/shortest {a['driven_over_shortest']:.2f}  "
                  f"wandering {a['wandering']:.2f}  replans {a['replans']:.0f}  "
                  f"refused {a['plans_refused']:.0f}  "
                  f"frontier {a['frontier_plans']:.0f}/{a['goal_plans']:.0f}  "
                  f"ahead {a['plan_ahead']:.2f}m  "
                  f"slow {a['slow_fraction']:.2f}  known {a['known']:.2f}", flush=True)
            for label in ("successes", "failures"):
                e = entry[label]
                if e:
                    print(f"         {label:10s} steps {e['steps']:3.0f}  "
                          f"driven/shortest {e['driven_over_shortest']:.2f}  "
                          f"wandering {e['wandering']:.2f}  reached {e['reached']:.2f}  "
                          f"replans {e['replans']:.0f}  gyration {e['gyration']:.2f}  "
                          f"reversals {e['reversals']:.0f}  "
                          f"closest {e['closest_to_goal']:.2f}  "
                          f"slow {e['slow_fraction']:.2f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
