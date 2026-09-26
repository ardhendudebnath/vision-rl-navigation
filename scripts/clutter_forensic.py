"""Where does a clutter failure actually stop making progress?

§9.7 diagnosed the clutter failures as indecision and §9.8 tested the lever that
diagnosis named -- the planner's optimism about unseen space -- and found it
changed three episodes in fifty. Three repairs have now been rejected on val, so
this stops proposing fixes and measures the failure instead.

The instrument is the **true geodesic** from the robot's cell to the goal,
computed once per world on the inflated grid the environment scores against
(:func:`geodesic_distance_field`) and read off every step. It is privileged
information used for measurement only: it never touches the agent. Against it:

  best, t_best       the closest the robot ever gets, and when
  stall fraction     the share of the episode spent after that moment
  belief vs truth    the length of the route the agent is steering by, against
                     how far there really is left to go
  blocked on truth   cells of the true remaining route that the agent's own map
                     calls occupied, and cells it has never seen

Those four separate the explanations that are still open. A robot that never
approaches the goal is failing differently from one that arrives within a metre
and then spends 300 steps not closing it. A robot whose map blocks the true way
is failing at mapping; one whose map is clear and whose plan is right, that
still does not drive it, is failing at control.

Run on the ``val`` seed band, which no experiment scores.

    python scripts/clutter_forensic.py --episodes 25
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
from vision_nav.mapping.occupancy import UNKNOWN
from vision_nav.planning.grid_astar import astar_grid, geodesic_distance_field
from vision_nav.training.env_factory import build_env_config

CONDITIONS = ("dense", "narrow")
#: A step counts as progress when the true remaining distance drops by more than
#: this, in metres. Below it the robot is holding station within sensor noise.
PROGRESS_EPS = 0.05


def remaining_at(field: np.ndarray, world, position: np.ndarray) -> float:
    """True geodesic from ``position`` to the goal, in metres.

    The field is computed on the robot-inflated grid, so a robot driving legally
    close to a wall can sit in a cell marked blocked. Taking the best value in a
    small neighbourhood asks "how far is there left to go from about here",
    which is the question, rather than punishing a legal pose for the inflation.
    """
    r, c = (int(v) for v in world.world_to_grid(np.asarray(position)))
    rows, cols = field.shape
    window = field[max(r - 2, 0):min(r + 3, rows), max(c - 2, 0):min(c + 3, cols)]
    return float(window.min()) if window.size else float("inf")


def episode(env, seed: int, cfg) -> dict:
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution

    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True)
    agent.start_episode(world, env.robot.pose)

    full_margin = agent.map.robot_radius + agent.config.safety_margin
    remaining, belief, kinds, positions, radii = [], [], [], [], []
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        here = env.robot.pose[:2].copy()
        positions.append(here)
        remaining.append(remaining_at(field, world, here))
        belief.append(agent._track_length(agent._track, agent._cursor))
        kinds.append(agent.plan_kind)
        # Which of the three radii the plan in force was found at. The agent
        # drops to a smaller one only when planning *fails* at a larger one, so
        # this says whether the fallback ladder ever engages.
        radii.append(float(agent._plan_radius))
        if terminated or truncated:
            break

    rem = np.asarray(remaining)
    pos = np.asarray(positions)
    steps = len(rem)
    finite = np.isfinite(rem)
    # The closest it ever got, and the first step that got within a cell of it.
    best = float(rem[finite].min()) if finite.any() else float("inf")
    t_best = int(np.argmax(finite & (rem <= best + world.config.grid_resolution)))
    after = slice(t_best, steps)

    tail = pos[after]
    centre = tail.mean(axis=0) if len(tail) else pos[-1]
    spread = float(np.sqrt(((tail - centre) ** 2).sum(axis=1).mean())) if len(tail) else 0.0
    gains = -np.diff(rem[finite]) if finite.sum() > 1 else np.zeros(1)

    # What the agent's own map says about the way home, from where it ended up.
    #
    # Read with a control, because the naive version of this measurement is
    # wrong in the agent's favour and then against it. The agent plans at
    # radius + margin while the world's grid is inflated by the radius alone,
    # so counting cells the agent blocks would charge its safety margin to its
    # mapping. Every count below is therefore paired with the same count taken
    # on the *truth* at the same inflation, and the difference is the only part
    # that is a map error.
    start_cell = tuple(int(v) for v in world.world_to_grid(pos[-1]))
    route = astar_grid(world.occupancy, start_cell, goal_cell)
    base = agent.map.robot_radius
    margin = base + agent.config.safety_margin
    blocked_margin = blocked_base = -1
    truth_blocked_margin = phantom_route = unseen_true_route = -1
    if route is not None and agent.map is not None:
        pts = world.grid_to_world(np.asarray(route, dtype=int))
        cells = agent.map.world_to_grid(pts)
        rows, cols = cells[:, 0], cells[:, 1]
        believed_margin = agent.map.occupancy_at(margin)[rows, cols]
        believed_base = agent.map.occupancy_at(base)[rows, cols]
        # The same cells, judged by the world at the same two inflations.
        w_cells = world.world_to_grid(pts)
        w_rows, w_cols = w_cells[:, 0], w_cells[:, 1]
        truth_margin = world.occupancy_at(margin)[w_rows, w_cols]
        truth_base = world.occupancy_at(base)[w_rows, w_cols]
        blocked_margin = int(believed_margin.sum())
        blocked_base = int(believed_base.sum())
        truth_blocked_margin = int(truth_margin.sum())
        # A phantom on the way home: the agent's most permissive planning grid
        # blocks a cell the world leaves open at the same radius.
        phantom_route = int((believed_base & ~truth_base).sum())
        unseen_true_route = int((agent.map.grid[rows, cols] == UNKNOWN).sum())

    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": steps,
        "start_remaining": float(rem[0]) if finite[0] else float("nan"),
        "best_remaining": best,
        "final_remaining": float(rem[-1]) if finite[-1] else float("nan"),
        "t_best": t_best,
        "stall_fraction": 1.0 - t_best / steps,
        # After its closest approach: does it hold station, or fall back?
        "given_back": float(rem[-1] - best) if finite[-1] else float("nan"),
        "stall_spread": spread,
        "progress_steps": int(np.sum(gains > PROGRESS_EPS)),
        "backward_steps": int(np.sum(gains < -PROGRESS_EPS)),
        # Belief against truth, over the stalled part of the episode.
        "belief_after": float(np.mean(np.asarray(belief)[after])),
        "truth_after": float(np.mean(rem[after][np.isfinite(rem[after])]))
        if np.isfinite(rem[after]).any() else float("nan"),
        "goal_plans_after": float(np.mean([k == "goal" for k in kinds[t_best:]])),
        "full_margin_steps": float(np.mean(np.isclose(radii, full_margin))),
        "min_plan_radius": float(np.min(radii)),
        "unreachable_steps": int((~finite).sum()),
        "replans": int(agent.replans),
        "recovery_steps": int(agent.recovery_steps),
        "blocked_margin": blocked_margin,
        "blocked_base": blocked_base,
        "truth_blocked_margin": truth_blocked_margin,
        "phantom_route": phantom_route,
        "unseen_true_route": unseen_true_route,
        "route_cells": 0 if route is None else len(route),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--out", default="results/clutter_forensic.json")
    args = p.parse_args(argv)

    report: dict = {"split": "val", "episodes": args.episodes, "conditions": {}}
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
        rows = [episode(env, s, cfg) for s in seeds]
        report["conditions"][cond] = {"episodes": rows}

        lost = [r for r in rows if not r["success"]]
        won = [r for r in rows if r["success"]]
        print(f"\n{cond}: {len(won)} arrive, {len(lost)} do not", flush=True)
        print(f"  {'seed':>6} {'steps':>5} {'start':>6} {'best':>6} {'final':>6} "
              f"{'t_best':>6} {'stall':>5} {'spread':>6} {'fwd':>4} {'back':>4} "
              f"{'belief':>6} {'truth':>6} {'goal%':>5} "
              f"{'blkM':>4} {'trM':>4} {'phan':>4} {'uns':>4}")
        for r in lost:
            print(f"  {r['seed']:>6} {r['steps']:>5} {r['start_remaining']:>6.2f} "
                  f"{r['best_remaining']:>6.2f} {r['final_remaining']:>6.2f} "
                  f"{r['t_best']:>6} {r['stall_fraction']:>5.2f} "
                  f"{r['stall_spread']:>6.2f} {r['progress_steps']:>4} "
                  f"{r['backward_steps']:>4} {r['belief_after']:>6.2f} "
                  f"{r['truth_after']:>6.2f} {r['goal_plans_after']:>5.2f} "
                  f"{r['blocked_margin']:>4} {r['truth_blocked_margin']:>4} "
                  f"{r['phantom_route']:>4} {r['unseen_true_route']:>4}")

        def mean(rs, k):
            v = [r[k] for r in rs if np.isfinite(r[k])]
            return float(np.mean(v)) if v else float("nan")

        for label, rs in (("arrive", won), ("fail", lost)):
            if rs:
                report["conditions"][cond][label] = {
                    k: mean(rs, k) for k in
                    ("steps", "start_remaining", "best_remaining", "final_remaining",
                     "t_best", "stall_fraction", "given_back", "stall_spread",
                     "progress_steps", "backward_steps", "belief_after",
                     "truth_after", "goal_plans_after", "full_margin_steps",
                     "min_plan_radius", "replans",
                     "recovery_steps", "blocked_margin",
                     "blocked_base", "truth_blocked_margin", "phantom_route",
                     "unseen_true_route", "unreachable_steps")}
                e = report["conditions"][cond][label]
                print(f"  {label:6s} mean: best {e['best_remaining']:.2f} m  "
                      f"final {e['final_remaining']:.2f} m  "
                      f"stall {e['stall_fraction']:.2f}  "
                      f"given back {e['given_back']:.2f} m  "
                      f"belief {e['belief_after']:.2f} vs truth "
                      f"{e['truth_after']:.2f}  "
                      f"blocked {e['blocked_margin']:.1f} (truth "
                      f"{e['truth_blocked_margin']:.1f}, phantom "
                      f"{e['phantom_route']:.1f})  "
                      f"unseen {e['unseen_true_route']:.1f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
