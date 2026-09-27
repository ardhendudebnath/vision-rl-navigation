"""In a tight gap, does the robot not know where it is, or not hold its line?

§9.10 relaxed the planner's clearance on the eleven val clutter worlds that
admit no route at ``robot_radius + safety_margin``. It carried the robot from
6.29 m short to 4.80 m short, recovered none of them, and turned a stack that
had never collided in 75 episodes into one that collided twelve times. The
reading offered there: a 0.22 m robot in a gap under 0.40 m has no room for the
0.07-0.22 m of pose error §9.5 measured. That is a claim about an error budget,
and this measures it.

At every step, three quantities in the same units:

  room            ``clearance(true position) - robot_radius``: how much lateral
                  error the robot could absorb before touching something.
  pose error      ``|estimate - truth|``. The robot steers by the estimate, so
                  wherever this goes, the robot goes.
  tracking error  distance from the *estimate* to the route it is following.
                  This is the controller's own error, measured in the frame the
                  controller works in, so it is free of localisation error.

The two errors add against the room. If pose error alone exceeds the room, no
controller could have saved the episode -- the robot is somewhere other than it
believes, by more than the gap allows. If tracking error dominates instead, the
robot knows where it is and cannot hold the line, which is a different repair.

Run on the ``val`` seed band, which no experiment scores.

    python scripts/gap_diagnostic.py --episodes 25
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

CONDITIONS = ("dense", "narrow")
#: Steps without closing before the clearance is given up, as registered in
#: `clearance_experiment.py`. The relaxed arm is the one that enters the gaps.
STALL_STEPS = 50


def to_track(point: np.ndarray, track: np.ndarray | None) -> float:
    """Distance from a point to the densified route, or ``nan`` without one.

    Nearest vertex rather than nearest segment: the route is densified to
    ``track_spacing``, so the two differ by at most half that, which is well
    below the quantities being compared here.
    """
    if track is None or len(track) == 0:
        return float("nan")
    return float(np.min(np.linalg.norm(np.asarray(track) - point, axis=1)))


def episode(env, seed: int, cfg, relax: int) -> dict:
    env.reset(options={"world_seed": seed})
    world = env.world
    radius = world.config.robot_radius
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True,
                                  corroborate=True, relax_on_stall=relax)
    agent.start_episode(world, env.robot.pose)

    room, pose_err, track_err = [], [], []
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        truth = env.robot.pose[:2].copy()
        action = agent.act(env.robot.pose, env.robot.velocity)
        estimate = agent.pose[:2]
        # Static clearance only: these conditions hold no movers, and the
        # question is about walls and clutter the map is supposed to know.
        room.append(float(world.clearance(truth, include_dynamic=False)) - radius)
        pose_err.append(float(np.linalg.norm(estimate - truth)))
        track_err.append(to_track(estimate, agent._track))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    r = np.asarray(room)
    pe = np.asarray(pose_err)
    te = np.asarray(track_err)
    tightest = int(np.argmin(r))
    finite_te = te[np.isfinite(te)]
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "collision": bool(info.get("collision")),
        "steps": len(r),
        "min_room": float(r.min()),
        "room_p5": float(np.percentile(r, 5)),
        # At the tightest moment of the episode, the whole budget at once.
        "room_at_tightest": float(r[tightest]),
        "pose_err_at_tightest": float(pe[tightest]),
        "track_err_at_tightest": float(te[tightest]) if np.isfinite(te[tightest])
        else float("nan"),
        "pose_err_mean": float(pe.mean()),
        "pose_err_p95": float(np.percentile(pe, 95)),
        "track_err_mean": float(finite_te.mean()) if finite_te.size else float("nan"),
        "track_err_p95": float(np.percentile(finite_te, 95)) if finite_te.size
        else float("nan"),
        # The decisive share: steps where the robot's belief is wrong by more
        # than the room it has, so no controller could have kept it clear.
        "pose_exceeds_room": float(np.mean(pe > r)),
        "track_exceeds_room": float(np.mean(finite_te > r[np.isfinite(te)]))
        if finite_te.size else float("nan"),
        "relaxed": bool(agent.relaxed_at >= 0),
    }


def summarise(rows: list[dict], keys) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


KEYS = ("min_room", "room_p5", "room_at_tightest", "pose_err_at_tightest",
        "track_err_at_tightest", "pose_err_mean", "pose_err_p95",
        "track_err_mean", "track_err_p95", "pose_exceeds_room",
        "track_exceeds_room")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--audit", default="results/margin_audit.json")
    p.add_argument("--out", default="results/gap_diagnostic.json")
    args = p.parse_args(argv)

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))["conditions"]
    tight = {c: {r["seed"] for r in audit[c]["worlds"] if r["l_full"] is None}
             for c in audit}

    report: dict = {"split": "val", "episodes": args.episodes, "conditions": {}}
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
        marked = tight.get(cond, set())
        report["conditions"][cond] = {}
        for arm, relax in (("as_published", 0), ("relaxed", STALL_STEPS)):
            rows = [episode(env, s, cfg, relax) for s in seeds]
            groups = {
                "tight": [r for r in rows if r["seed"] in marked],
                "roomy": [r for r in rows if r["seed"] not in marked],
                "arrive": [r for r in rows if r["success"]],
                "collide": [r for r in rows if r["collision"]],
            }
            report["conditions"][cond][arm] = {
                "episodes": rows,
                **{g: (summarise(rs, KEYS) | {"n": len(rs)}) if rs else None
                   for g, rs in groups.items()},
            }
            print(f"\n{cond} / {arm}", flush=True)
            for g, rs in groups.items():
                if not rs:
                    continue
                e = report["conditions"][cond][arm][g]
                print(f"  {g:7s} n={e['n']:2d}  room at tightest "
                      f"{e['room_at_tightest']:+.3f} m   pose err there "
                      f"{e['pose_err_at_tightest']:.3f}   tracking err there "
                      f"{e['track_err_at_tightest']:.3f}", flush=True)
                print(f"  {'':7s}       pose p95 {e['pose_err_p95']:.3f}  "
                      f"tracking p95 {e['track_err_p95']:.3f}  "
                      f"steps where pose error exceeds the room "
                      f"{e['pose_exceeds_room']:.1%}  tracking "
                      f"{e['track_exceeds_room']:.1%}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
