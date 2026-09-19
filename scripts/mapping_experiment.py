"""Does the classical planner still win once it has to map the world itself?

Result 1 of the report -- the planner wins on every condition -- was measured
with the planner handed a perfect static map and its exact pose, a privilege the
report called deliberate: a baseline that loses through handicap proves nothing.
This takes the map away and keeps the pose. The planner builds an occupancy grid
from its own range sensor as it drives, plans on it with unknown space treated
as free, and replans when something it has just mapped cuts across its plan.
Controller, margins and fallbacks are the baseline's own
(``vision_nav.agents.mapped``).

Arms, on the report's six benchmark conditions, 100 held-out worlds each -- the
same worlds, in the same order, that ``run_benchmark.py`` scores:

  full_map         the published classical baseline; must reproduce it
  mapped_lidar32   mapping from the 32-beam 360-degree scanner the privileged
                   PPO policy reads -- the same returns the policy sees
  mapped_camera64  mapping from the depth camera's geometry: 64 columns over 90
                   degrees, the depth policy's sensor

On ``noisy_lidar`` both mapped arms read the condition's 0.10 m range noise --
the first time in this project the classical stack has been exposed to it.

    python scripts/mapping_experiment.py --episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spacetime_experiment import paired_ci, verdict  # noqa: E402
from velocity_experiment import mcnemar_p  # noqa: E402

from vision_nav.agents.mapped import MappedPursuitAgent  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.training.actors import build_actor  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402
from vision_nav.training.evaluate import evaluate  # noqa: E402

CONDITIONS = tuple(BENCHMARK_CONDITIONS)
#: The report's learned columns, read from their result files for comparison.
LEARNED = {"ppo_privileged": "nominal_trained", "depth_policy": "depth64",
           "best_lidar_policy": "lidar64"}

#: Recorded before the run and committed to the repository before any result
#: exists.
#:
#: Confidence LOW on every number, and none rests on a measurement. The reasoning:
#: a 32-beam scanner samples a surface every 0.2 m at a metre and every 0.6 m
#: at three, against a 0.40 m planning inflation, so what the robot is about to
#: pass is well mapped and what is far away is not. Open conditions should
#: barely notice. In `narrow` and `dense`, planning optimistically through
#: space nobody has looked at means committing to gaps that turn out closed,
#: then replanning -- and Phase 5d measured replanning in clutter costing the
#: pure-pursuit controller. The margin to the learned policy there is 0.25, which
#: is why the planner is predicted to keep its lead even so.
PREDICTION = (
    "mapping costs the planner in clutter and little elsewhere, and it still "
    "wins: mapped_lidar32 minus full_map on narrow is HARMS, between -0.05 and "
    "-0.15; on sparse, large and nominal it loses at most 0.03. mapped_lidar32 "
    "still beats the privileged PPO policy's published success on all five "
    "static conditions. mapped_camera64 loses more than mapped_lidar32 on narrow "
    "and on dense. full_map reproduces the published classical row on all six. "
    "Decision on mapped_lidar32 minus full_map, narrow -- CHEAP: INERT (bounded). "
    "COSTLY: HARMS. Otherwise UNRESOLVED."
)


class MappedActor:
    """The actor interface around :class:`MappedPursuitAgent`. Like the
    classical actor it reads ``env.world`` and the pose -- but only to hand the
    world to its own sensor."""

    def __init__(self, sensor: str, noise_std: float, robot) -> None:
        self.name = f"mapped_{sensor}"
        self.agent = MappedPursuitAgent(robot=robot, sensor=sensor, noise_std=noise_std)
        self.known, self.replans = [], []

    def reset(self, env, obs) -> bool:
        if self.agent.map is not None:
            self.known.append(self.agent.map.known_fraction)
            self.replans.append(self.agent.replans)
        self.agent.robot = env.config.robot
        return self.agent.start_episode(env.world, env.robot.pose)

    def act(self, env, obs):
        return self.agent.act(env.robot.pose)

    def finish(self) -> None:
        if self.agent.map is not None:
            self.known.append(self.agent.map.known_fraction)
            self.replans.append(self.agent.replans)


def rows_of(results):
    return [{"success": bool(r.success), "collision": bool(r.collision)} for r in results]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/mapping_experiment.json")
    args = p.parse_args(argv)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "conditions": {},
              "checks": {}}
    rng = np.random.default_rng(0)
    reproduced = []
    for cond in CONDITIONS:
        split, shift, noise = BENCHMARK_CONDITIONS[cond]
        overrides = {"lidar": {"noise_std": noise}} if noise else {}
        cfg = build_env_config(overrides, split=split, shift=shift, n_worlds=args.episodes)

        arms = {}
        metrics, results = evaluate(build_actor("classical", robot=cfg.robot), cfg)
        arms["full_map"] = {"metrics": metrics, "rows": rows_of(results)}
        for sensor in ("lidar32", "camera64"):
            actor = MappedActor(sensor, noise, cfg.robot)
            metrics, results = evaluate(actor, cfg)
            actor.finish()
            arms[f"mapped_{sensor}"] = {"metrics": metrics, "rows": rows_of(results),
                                        "known": float(np.mean(actor.known)),
                                        "replans": float(np.mean(actor.replans))}

        published = json.loads(Path(f"results/{cond}__classical.json").read_text(encoding="utf-8"))
        same = abs(arms["full_map"]["metrics"].success_rate - published["success_rate"]) < 1e-9
        reproduced.append(same)

        entry = {"cells": {}, "contrasts": {}, "learned": {}, "reproduces_published": same}
        for arm, data in arms.items():
            m = data["metrics"]
            entry["cells"][arm] = {"success": m.success_rate, "collision": m.collision_rate,
                                   "timeout": m.timeout_rate, "spl": m.spl,
                                   "success_per_episode": [int(r["success"]) for r in data["rows"]]}
            for extra in ("known", "replans"):
                if extra in data:
                    entry["cells"][arm][extra] = data[extra]
        for arm in ("mapped_lidar32", "mapped_camera64"):
            a, b = arms["full_map"]["rows"], arms[arm]["rows"]
            sa = np.array([r["success"] for r in a])
            sb = np.array([r["success"] for r in b])
            p_val, lost, won = mcnemar_p(sa, sb)
            ci = paired_ci(a, b, rng, args.bootstrap)
            gain = float(sb.mean() - sa.mean())
            entry["contrasts"][arm] = {"success_gain": gain, "p": p_val, "episodes_won": won,
                                       "episodes_lost": lost, "ci95": ci,
                                       "verdict": verdict(gain, p_val, ci)}
        for label, stem in LEARNED.items():
            path = Path(f"results/{cond}__{stem}.json")
            if path.exists():
                entry["learned"][label] = json.loads(path.read_text(encoding="utf-8"))["success_rate"]
        report["conditions"][cond] = entry

        c = entry["cells"]
        print(f"{cond:12s} full {c['full_map']['success']:.3f}  "
              f"lidar {c['mapped_lidar32']['success']:.3f} "
              f"({entry['contrasts']['mapped_lidar32']['verdict']})  "
              f"camera {c['mapped_camera64']['success']:.3f} "
              f"({entry['contrasts']['mapped_camera64']['verdict']})  "
              f"ppo {entry['learned'].get('ppo_privileged')}  reproduces {same}", flush=True)

    report["checks"]["full_map_reproduces_all"] = all(reproduced)
    label = report["conditions"]["narrow"]["contrasts"]["mapped_lidar32"]["verdict"]
    report["decision"] = ("CHEAP" if label == "INERT (bounded)"
                          else "COSTLY" if label == "HARMS" else "UNRESOLVED")
    print(f"\nfull map reproduces the published row everywhere: {all(reproduced)}")
    print(f"decision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(reproduced) else 1


if __name__ == "__main__":
    raise SystemExit(main())
