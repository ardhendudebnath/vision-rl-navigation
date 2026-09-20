"""Why does scan matching make `sparse` worse instead of better?

Phase 6f's registered prediction said it would, and named the mechanism: the map
is built at the estimate and the estimate is matched against the map, so the two
can agree with each other while both drift away from the world. A front end with
no back end cannot tell that apart from being right.

This measures it. For every control period of a matched run it records the
matcher's own score -- the mean likelihood-field value under the scan's returns
at the pose it accepted, which is exactly what it maximised -- against the true
pose error at that moment. If the mechanism is real, the score stays high on
`sparse` while the error is large: the matcher is confident and wrong. `dense`
is the contrast, where structure pins the pose and the two agree.

Run on the ``val`` seed band, which no experiment scores, and quoted in report
§9.3 as val.

    python scripts/localisation_diagnostic.py --episodes 20
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

CONDITIONS = ("sparse", "dense", "large", "narrow")


def episode(env, seed: int, cfg) -> list[tuple[float, float]]:
    """One matched episode, as (match score, true pose error) per step."""
    env.reset(options={"world_seed": seed})
    agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar32",
                                  noise_std=cfg.lidar.noise_std,
                                  odometry=OdometryConfig(), scan_matching=True)
    agent.start_episode(env.world, env.robot.pose)
    matcher, omap = agent.matcher, agent.map

    # Record the scan the agent actually used, rather than taking another one:
    # a second scan would consume the sensor's noise draws and put this
    # diagnostic on a different trajectory from the experiment it explains.
    taken: dict[str, np.ndarray] = {}
    real_scan = omap.scan

    def spy(pose):
        taken["ranges"] = real_scan(pose)
        return taken["ranges"]

    omap.scan = spy

    rows = []
    for _ in range(cfg.max_episode_steps):
        truth = env.robot.pose.copy()
        action = agent.act(truth, env.robot.velocity)
        accepted = agent.pose
        # The score under the pose the matcher settled on, which is the
        # quantity it was maximising.
        ranges = taken["ranges"]
        hit = ranges < omap.sensor.config.max_range - 1e-6
        if int(hit.sum()) >= matcher.config.min_hits:
            bearings = omap.sensor._angles[hit] + accepted[2]
            ends = accepted[:2] + np.stack([ranges[hit] * np.cos(bearings),
                                            ranges[hit] * np.sin(bearings)], axis=-1)
            score = float(matcher._sample(matcher.likelihood_field(omap), ends[None],
                                          omap.resolution)[0].mean())
            rows.append((score, agent.pose_errors[-1]))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="results/localisation_diagnostic.json")
    args = p.parse_args(argv)

    report: dict = {"episodes": args.episodes, "split": "val", "conditions": {}}
    for cond in CONDITIONS:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        rows = []
        for seed in list(cfg.world_seeds)[:args.episodes]:
            rows.extend(episode(env, int(seed), cfg))
        a = np.asarray(rows, dtype=float)
        entry = {"steps": int(len(a)),
                 "score_median": float(np.median(a[:, 0])),
                 "score_p10": float(np.percentile(a[:, 0], 10)),
                 "pose_error_median": float(np.median(a[:, 1])),
                 "pose_error_p90": float(np.percentile(a[:, 1], 90))}
        # The mechanism in one number: the pose error at the steps the matcher
        # was most confident about.
        confident = a[a[:, 0] >= np.median(a[:, 0])]
        entry["pose_error_median_when_confident"] = float(np.median(confident[:, 1]))
        report["conditions"][cond] = entry
        print(f"{cond:8s} steps={entry['steps']:5d} "
              f"score med={entry['score_median']:.3f} "
              f"pose error med={entry['pose_error_median']:.3f} "
              f"(when confident {entry['pose_error_median_when_confident']:.3f})",
              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
