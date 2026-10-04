"""What is left wrong with `noisy_lidar`, once the map is clean?

§9.14 removed 93% of the phantom cells under noise and moved arrivals by two in
a hundred, which says the map is no longer what limits `noisy_lidar`. This
measures what does, in two parts, on the ``val`` band only.

**Part 1, the front end.** With the obstacle range on, re-drive every val world
where the robot stopped just outside the goal believing it had arrived, and
trace the pose error. Two questions separate the explanations. Does the error
*jump* -- the matcher locking onto a wrong alignment -- or *drift*? And how
often does the matcher return no correction at all, against episodes that
arrive? A matcher that agrees with a drifted pose step after step is matching
against a map that the same drift built.

**Part 2, the back end.** With the published back end driving, every closure
attempt is scored twice on the same keyframe pair -- with the published blur
and with a blur widened by the sensor's noise on both scans,
``sqrt(sigma^2 + 2 sigma_noise^2)`` -- and each best match is checked against
the *true* relative pose, recorded when the keyframe was added. That says
whether a noise-aware match would admit true closures without false ones; and
the per-episode closure counts say whether closures are even missing where the
robot fails.

    python scripts/noisy_localisation_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping import posegraph
from vision_nav.mapping.localisation import OdometryConfig
from vision_nav.mapping.posegraph import relative_pose
from vision_nav.training.env_factory import build_env_config

#: Goal tolerance is 0.35 m; a robot that stops inside this distance of the goal
#: without arriving stopped "just outside" it.
JUST_OUTSIDE = 0.8
#: A closure within this distance of the true relative pose is right, and one
#: beyond the larger one is wrong. Between them it is merely imprecise.
RIGHT, WRONG = 0.15, 0.30


def make_env(cond: str, n: int):
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split="val", shift=shift, n_worlds=n)
    return cfg, ProceduralNavEnv(cfg), noise


def front_end(val100: Path) -> dict:
    """Part 1: trace the episodes that stopped just outside, and some arrivals."""
    cell = json.loads(val100.read_text(encoding="utf-8"))["cells"]["noisy_lidar"]
    eps = cell["obstacle_range"]["episodes"]
    stuck = [e["seed"] for e in eps if not e["success"] and not e["collision"]
             and e["steps"] >= 500 and e["goal_distance"] <= JUST_OUTSIDE]
    arrived = [e["seed"] for e in eps if e["success"]][:len(stuck)]
    cfg, env, _ = make_env("noisy_lidar", 100)

    def trace(seed):
        env.reset(options={"world_seed": seed})
        agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                      noise_std=cfg.lidar.noise_std,
                                      odometry=OdometryConfig(), scan_matching=True,
                                      corroborate=True, noise_margin=3.0)
        agent.start_episode(env.world, env.robot.pose)
        shifts = []
        for _ in range(cfg.max_episode_steps):
            _, _, term, trunc, _ = env.step(agent.act(env.robot.pose, env.robot.velocity))
            shifts.append(agent.matcher.last_shift)
            if term or trunc:
                break
        err = np.asarray(agent.pose_errors)
        return {"seed": seed, "final_error": float(err[-1]),
                "max_step_rise": float(np.diff(err).max()) if len(err) > 1 else 0.0,
                "steps_over_0.3m": int((err > 0.3).sum()),
                "zero_corrections": float(np.mean(np.asarray(shifts) == 0.0)),
                "steps": len(err)}

    out = {"stuck": [trace(s) for s in stuck], "arrived": [trace(s) for s in arrived]}
    for group in ("stuck", "arrived"):
        rows = out[group]
        print(f"  {group:8s} n={len(rows)}  largest one-step rise "
              f"{max(r['max_step_rise'] for r in rows):.3f} m  final error median "
              f"{np.median([r['final_error'] for r in rows]):.3f} m  zero corrections "
              f"{np.median([r['zero_corrections'] for r in rows]):.0%} of steps", flush=True)
    return out


def back_end(n: int) -> dict:
    """Part 2: shadow-score every closure attempt against the truth."""
    truth: list[np.ndarray] = []
    current = [np.zeros(3)]
    noise_std = [0.0]
    records: list[dict] = []
    match, add = posegraph.PoseGraph._match, posegraph.PoseGraph.add_keyframe

    def recording_add(self, pose, local):
        truth.append(current[0].copy())
        return add(self, pose, local)

    def shadowed(self, i, j):
        out = match(self, i, j)
        true_rel = relative_pose(truth[j], truth[i])
        row = {}
        widened = float(np.sqrt(self.config.loop_sigma ** 2 + 2 * noise_std[0] ** 2))
        for name, sigma in (("published", self.config.loop_sigma),
                            ("noise_aware", widened)):
            saved = self.config.loop_sigma
            self.config.loop_sigma = sigma
            try:
                score, rel = match(self, i, j)
            finally:
                self.config.loop_sigma = saved
            row[name] = [float(score) if np.isfinite(score) else None,
                         float(np.linalg.norm(rel[:2] - true_rel[:2]))]
        records.append(row)
        return out

    posegraph.PoseGraph._match = shadowed
    posegraph.PoseGraph.add_keyframe = recording_add
    report: dict = {}
    try:
        for cond in ("noisy_lidar", "nominal"):
            cfg, env, noise = make_env(cond, n)
            noise_std[0] = noise
            records.clear()
            for seed in [int(s) for s in list(cfg.world_seeds)[:n]]:
                env.reset(options={"world_seed": seed})
                agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                              noise_std=noise, odometry=OdometryConfig(),
                                              scan_matching=True, corroborate=True,
                                              pose_graph=True)
                truth.clear()
                agent.start_episode(env.world, env.robot.pose)
                for _ in range(cfg.max_episode_steps):
                    current[0] = env.robot.pose.copy()
                    _, _, term, trunc, _ = env.step(agent.act(env.robot.pose,
                                                              env.robot.velocity))
                    if term or trunc:
                        break
            gate = posegraph.PoseGraphConfig().loop_min_score
            entry = {"attempts": len(records)}
            for name in ("published", "noise_aware"):
                scores = np.array([r[name][0] if r[name][0] is not None else -np.inf
                                   for r in records])
                errs = np.array([r[name][1] for r in records])
                ok = scores >= gate
                entry[name] = {"accepted": int(ok.sum()),
                               "right": int((errs[ok] <= RIGHT).sum()),
                               "wrong": int((errs[ok] > WRONG).sum()),
                               "median_error": float(np.median(errs[ok])) if ok.any() else None}
            report[cond] = entry
            print(f"  {cond:12s} {entry['attempts']} attempts", flush=True)
            for name in ("published", "noise_aware"):
                e = entry[name]
                print(f"    {name:12s} accepted {e['accepted']}  right {e['right']}  "
                      f"wrong {e['wrong']}  median error {e['median_error']}", flush=True)
    finally:
        posegraph.PoseGraph._match, posegraph.PoseGraph.add_keyframe = match, add
    return report


def trimmed_keyframes(n: int) -> dict:
    """Part 3: would trimming the keyframe scans at the obstacle range help?

    The mechanism first proposed for combining the two repairs was that a
    cleaner map would help the closures. Closures never read the map, so the
    nearest real version is to drop, from the keyframe scans themselves, the
    readings the map refuses to believe -- the noise-pushed max-range returns.
    This measures it the way it was first measured: the share of closure
    attempts whose best score clears the gate, untrimmed against trimmed. The
    trimming was reverted from the code after this measurement, so it is
    reproduced here by substituting the keyframe scan function.
    """
    from vision_nav.agents import localised

    gate = posegraph.PoseGraphConfig().loop_min_score
    scores: list[float] = []
    match, points = posegraph.PoseGraph._match, localised.scan_points

    def recording(self, i, j):
        s, rel = match(self, i, j)
        if np.isfinite(s):
            scores.append(float(s))
        return s, rel

    out: dict = {}
    posegraph.PoseGraph._match = recording
    try:
        for name, margin in (("full_scans", 0.0), ("trimmed_scans", 3.0)):
            if margin:
                def trimmed(ranges, sensor, _m=margin):
                    # A point's distance from the robot is its range, so
                    # dropping points beyond the line is dropping those readings.
                    line = sensor.config.max_range - _m * sensor.config.noise_std
                    local = points(ranges, sensor)
                    return local[np.linalg.norm(local, axis=1) < line - 1e-6]
                localised.scan_points = trimmed
            else:
                localised.scan_points = points
            cfg, env, noise = make_env("noisy_lidar", n)
            scores.clear()
            for seed in [int(s) for s in list(cfg.world_seeds)[:n]]:
                env.reset(options={"world_seed": seed})
                agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                              noise_std=noise, odometry=OdometryConfig(),
                                              scan_matching=True, corroborate=True,
                                              pose_graph=True, noise_margin=margin)
                agent.start_episode(env.world, env.robot.pose)
                for _ in range(cfg.max_episode_steps):
                    _, _, term, trunc, _ = env.step(agent.act(env.robot.pose,
                                                              env.robot.velocity))
                    if term or trunc:
                        break
            s = np.asarray(scores)
            out[name] = {"attempts": len(s), "median_score": float(np.median(s)),
                         "share_clearing_gate": float(np.mean(s >= gate))}
            print(f"  {name:14s} {len(s)} attempts  median score {np.median(s):.3f}  "
                  f"clearing the gate {np.mean(s >= gate):.0%}", flush=True)
    finally:
        posegraph.PoseGraph._match, localised.scan_points = match, points
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val100", default="results/obstacle_range_val100.json")
    p.add_argument("--worlds", type=int, default=12)
    p.add_argument("--trim-worlds", type=int, default=6)
    p.add_argument("--parts", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--out", default="results/noisy_localisation_diagnostic.json")
    args = p.parse_args(argv)
    out = Path(args.out)
    report = (json.loads(out.read_text(encoding="utf-8")) if out.exists()
              else {"split": "val"})
    if 1 in args.parts:
        print("Part 1: the front end, obstacle range on", flush=True)
        report["front_end"] = front_end(Path(args.val100))
    if 2 in args.parts:
        print("Part 2: the back end, shadow-scored against the truth", flush=True)
        report["back_end"] = back_end(args.worlds)
    if 3 in args.parts:
        print("Part 3: keyframe scans trimmed at the obstacle range", flush=True)
        report["trimmed_keyframes"] = trimmed_keyframes(args.trim_worlds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
