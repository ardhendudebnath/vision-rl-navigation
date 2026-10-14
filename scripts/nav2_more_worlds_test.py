"""Does Nav2 arrive more often than this stack under noise? A hundred more worlds.

Report §9.20 left one gap open on held-out `noisy_lidar` at 360 beams, neither
stack holding the map or the pose: Nav2 with slam_toolbox arrived on 0.950 of
100 worlds and this stack on 0.890, 8 worlds won and 2 lost, McNemar p = 0.11,
and five of this stack's eight losses were stalls with the pose known. The
held-out band has 200 worlds and only the first 100 have ever been driven, by
anything (``results/`` holds no episode on seeds 20100-20199). This drives the
second hundred with both stacks, exactly as §9.20 did, and reads all 200:

  Nav2       slam_toolbox's map and pose, 360 beams, 0.10 m noise delivered,
             5x real time (``run_nav2_eval.py --first 100``)
  this stack own map and pose, 360 beams, odometry and scan matching,
             corroboration -- the arm §9.20 compared (``backend_experiment``'s
             front end)

The stack's arm also re-drives the first hundred worlds, which must reproduce
§9.20's episodes exactly.

    bash ros2_bridge/run_nav2.sh --condition noisy_lidar --episodes 200 --first 100 \\
        --privileges slam --out-dir results/nav2_more_worlds
    python scripts/nav2_more_worlds_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transfer_test import mcnemar  # noqa: E402

from vision_nav.agents.localised import LocalisedPursuitAgent  # noqa: E402
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

COND = "noisy_lidar"
NAV2_FIRST = Path("results/nav2_noise_rerun/slam/noisy_lidar__nav2_slam.json")
NAV2_MORE = Path("results/nav2_more_worlds/noisy_lidar__nav2_slam_from100.json")
STACK_FIRST = Path("results/backend_test.json")
#: A stall: no arrival, no collision, ending at least this far from the goal.
STALL = 1.0
#: The parity band for the decision, on the pooled difference's interval.
PARITY = 0.05

#: Recorded and committed before either stack had driven any of the new worlds.
#:
#: DISCLOSED: the first hundred worlds' outcomes are §9.20's (Nav2 0.950, this
#: stack 0.890, 8 won and 2 lost, five of the eight stalls). This stack's
#: episode function was checked on three of those published worlds, which it
#: reproduced exactly. ``run_nav2_eval.py --first`` was smoke-tested on val
#: worlds; a first attempt never brought Nav2 up while every core was busy with
#: another run, and the smoke was repeated on an idle machine, where it drove
#: val worlds 10001 and 10002 (2 of 2 arrived, 0.99 commands per step, 0.0999 m
#: noise delivered). The reasoning: a
#: gap read off one sample shrinks on the next, but Nav2 with slam_toolbox has
#: arrived more often than this stack on open worlds before (§9.4, §9.20), so
#: the gap should be smaller and real.
PREDICTION = (
    "the gap is real and smaller than the first hundred showed. On the new 100 "
    "worlds Nav2 arrives more often than this stack; pooled over all 200, "
    "Nav2 - stack is between +0.02 and +0.08. The two localise alike: their "
    "median final pose errors over the 200 are within 25% of each other. At "
    "least half of this stack's losses on the new worlds are stalls -- no "
    "collision, at least 1 m from the goal -- and each stack collides at most "
    "twice on them. Nav2's harness holds on the new run (0.10 m noise "
    "delivered, at least 0.95 commands per step, no episode without a command, "
    "at most 3 abandoned); if it does not, that run is repeated, not scored. "
    "Decision on the pooled 200 -- NAV2 ARRIVES MORE: Nav2 ahead with McNemar "
    "p < 0.05. STACK ARRIVES MORE: the stack ahead with p < 0.05. PARITY: "
    "neither, with the 95% interval inside 0.05 either way. Otherwise "
    "UNRESOLVED. Prediction: NAV2 ARRIVES MORE."
)


def episode(seed: int) -> dict:
    """§9.20's arm of this stack, as ``backend_experiment.episode`` drives it."""
    split, shift, noise = BENCHMARK_CONDITIONS[COND]
    cfg = build_env_config({"lidar": {"noise_std": noise}}, split=split, shift=shift, n_worlds=200)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360", noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True, corroborate=True,
                                  pose_graph=False)
    agent.start_episode(env.world, env.robot.pose)
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    errors = np.asarray(agent.pose_errors) if agent.pose_errors else np.zeros(1)
    return {"seed": seed, "success": bool(info.get("is_success")),
            "collision": bool(info.get("collision")), "steps": len(errors),
            "goal_distance": float(info.get("goal_distance", np.nan)),
            "pose_err_final": float(errors[-1])}


def paired_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = 10000) -> list[float]:
    n = len(a)
    diffs = [b[i].mean() - a[i].mean() for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def nav2_rows(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    errs = data.get("pose_error_per_episode", [np.nan] * len(data["per_episode"]))
    return {r["world_seed"]: {**r, "pose_err_final": e} for r, e in zip(data["per_episode"], errs, strict=True)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--stack-out", default="results/nav2_more_worlds/noisy_lidar__stack.json")
    p.add_argument("--out", default="results/nav2_more_worlds_test.json")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Drive only these worlds and stop (smoke runs).")
    args = p.parse_args(argv)

    if args.seeds:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(episode, args.seeds):
                print(r)
        return 0

    stack_out = Path(args.stack_out)
    if stack_out.exists():
        stack = {r["seed"]: r for r in json.loads(stack_out.read_text(encoding="utf-8"))["episodes"]}
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            stack = {r["seed"]: r for r in pool.map(episode, range(20000, 20200), chunksize=1)}
        stack_out.parent.mkdir(parents=True, exist_ok=True)
        stack_out.write_text(json.dumps({"condition": COND, "episodes": list(stack.values())}, indent=1),
                             encoding="utf-8")
    published = {e["seed"]: e for e in json.loads(STACK_FIRST.read_text(encoding="utf-8"))
                 ["cells"][COND]["front_end"]["episodes"]}
    reproduces = all(stack[s]["success"] == e["success"] and stack[s]["collision"] == e["collision"]
                     and stack[s]["steps"] == e["steps"] for s, e in published.items())

    nav2 = {**nav2_rows(NAV2_FIRST), **nav2_rows(NAV2_MORE)}
    more = json.loads(NAV2_MORE.read_text(encoding="utf-8"))
    harness = {"delivered_noise_std": more["delivered_noise_std"],
               "mean_commands_per_step": more["mean_commands_per_step"],
               "zero_command_episodes": more["zero_command_episodes"],
               "aborted": more["nav2_aborted_episodes"],
               "slam_not_ready": more["slam_not_ready_episodes"],
               "realtime_factor_cap": more["realtime_factor_cap"]}
    seeds = list(range(20000, 20200))
    assert set(seeds) <= set(nav2) and set(seeds) <= set(stack), "a world is missing"
    rng = np.random.default_rng(20261012)
    report: dict = {"prediction": PREDICTION, "reproduces_published": reproduces,
                    "harness": harness, "sets": {}}
    for name, part in (("first", seeds[:100]), ("new", seeds[100:]), ("pooled", seeds)):
        a = np.array([stack[s]["success"] for s in part], dtype=bool)
        b = np.array([nav2[s]["success"] for s in part], dtype=bool)
        pv, won, lost = mcnemar(a, b)
        lost_by_stack = [s for s in part if nav2[s]["success"] and not stack[s]["success"]]
        stalls = [s for s in lost_by_stack if not stack[s]["collision"] and stack[s]["goal_distance"] >= STALL]
        report["sets"][name] = {
            "worlds": len(part), "nav2": float(b.mean()), "stack": float(a.mean()),
            "delta": float(b.mean() - a.mean()), "p": pv, "nav2_won": won, "nav2_lost": lost,
            "ci95": paired_ci(a, b, rng),
            "stack_collisions": int(sum(stack[s]["collision"] for s in part)),
            "nav2_collisions": int(sum(nav2[s]["collision"] for s in part)),
            "pose_err_median": {"nav2": float(np.nanmedian([nav2[s]["pose_err_final"] for s in part])),
                                "stack": float(np.median([stack[s]["pose_err_final"] for s in part]))},
            "stack_losses_stalled": len(stalls),
            "stack_losses_stalled_pose_err_median": float(np.median(
                [stack[s]["pose_err_final"] for s in stalls])) if stalls else None}
    g = report["sets"]["pooled"]
    if g["delta"] > 0 and g["p"] < 0.05:
        decision = "NAV2 ARRIVES MORE"
    elif g["delta"] < 0 and g["p"] < 0.05:
        decision = "STACK ARRIVES MORE"
    elif -PARITY < g["ci95"][0] and g["ci95"][1] < PARITY:
        decision = "PARITY"
    else:
        decision = "UNRESOLVED"
    report["decision"] = decision

    print(f"stack reproduces §9.20 on the first 100: {reproduces}; Nav2 harness {harness}")
    for name, s in report["sets"].items():
        print(f"{name:7s} Nav2 {s['nav2']:.3f} stack {s['stack']:.3f}  {s['delta']:+.3f} "
              f"(p {s['p']:.3f}, {s['nav2_won']}/{s['nav2_lost']}) CI {np.round(s['ci95'], 3).tolist()}  "
              f"pose {s['pose_err_median']}  stack losses stalled {s['stack_losses_stalled']}")
    print(f"decision {decision}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
