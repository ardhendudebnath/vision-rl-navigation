"""What was the exact pose worth?

Report §9.2 took the map away from the classical planner and kept its pose,
saying why: taking both privileges at once would leave a result nobody could
attribute. This takes the second one. The robot now drives on what its wheels
report -- drifting -- and, in the arm that matters, on that estimate dragged
back onto the map it is already building by matching each scan against it
(``vision_nav.mapping.localisation``). Nothing else changes: same planner, same
controller, same margins, same map.

Arms, on the report's six benchmark conditions, 100 held-out worlds each -- the
same worlds, in the same order, that every other actor in this project is
scored on:

  full_map      the published classical baseline; must reproduce it
  mapped        §9.2's arm: own map, exact pose; must reproduce Phase 6e
  odometry      own map, wheel odometry alone, no correction
  matched       own map, odometry corrected by scan matching -- the real stack
  matched_cam   the same, mapping and matching from the 90-degree depth camera
                instead of the 360-degree scanner

Two controls. ``mapped`` must reproduce Phase 6e's per-episode outcomes exactly,
which is a cross-phase check that nothing in this phase's refactoring moved.
And on ``nominal`` the localised agent is also run with its odometry noise
switched off, where it must be bit-identical to ``mapped``: the estimator is
then the robot's own integrator, so any difference would be plumbing rather
than pose.

    python scripts/localisation_experiment.py --episodes 100
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

from vision_nav.agents.localised import LocalisedPursuitAgent  # noqa: E402
from vision_nav.agents.mapped import MappedPursuitAgent  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.localisation import OdometryConfig  # noqa: E402
from vision_nav.training.actors import build_actor  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402
from vision_nav.training.evaluate import evaluate  # noqa: E402

CONDITIONS = tuple(BENCHMARK_CONDITIONS)
#: The report's learned columns, read from their result files for comparison.
LEARNED = {"ppo_privileged": "nominal_trained"}
#: Phase 6e's results, for the cross-phase reproduction check.
PHASE_6E = Path("results/mapping_experiment_repaired.json")

#: Arms: (sensor, odometry, scan matching).
ARMS: dict[str, tuple[str, OdometryConfig | None, bool]] = {
    "odometry": ("lidar32", OdometryConfig(), False),
    "matched": ("lidar32", OdometryConfig(), True),
    "matched_cam": ("camera64", OdometryConfig(), True),
}

#: Recorded before the run and committed before any result on the test worlds
#: exists.
#:
#: DISCLOSED: the odometry model, the matcher and both of its prior weights were
#: developed and tuned on the ``val`` seed band, which this experiment never
#: scores, and the numbers below are read off those val runs -- a forecast
#: derived from measurement, one band away from where it is applied. Twenty val
#: worlds per condition, success rate:
#:
#:     condition     exact   odometry   matched   matched_cam
#:     nominal       1.000   0.700      0.700     0.850
#:     sparse        1.000   0.750      0.350     0.700
#:     large         1.000   0.300      0.600     0.550
#:     dense         0.700   0.400      0.650     0.700
#:     narrow        0.650   0.400      0.650     0.600
#:     noisy_lidar   0.950   0.600      0.750     0.650
#:
#: Median final pose error, odometry / matched, in metres: 0.318/0.255 nominal,
#: 0.267/0.512 sparse, 0.590/0.319 large, 0.389/0.058 dense, 0.396/0.113 narrow,
#: 0.310/0.374 noisy_lidar. At twenty worlds each of these has a standard error
#: near 0.10, so the bands below are wide on purpose.
#:
#: The bands are the val deltas applied to Phase 6e's published test row, not
#: the val rates themselves: the two bands are different world samples, and the
#: exact-pose arm already differs between them (0.650 against 0.590 on narrow).
#:
#: ``sparse`` is predicted to be the one condition where matching makes things
#: worse, and the mechanism is measured rather than guessed: in sparse val
#: worlds the matcher's own score stays high (median 0.92) while the pose is
#: 0.46 m out. The map is built from the estimate and the estimate is matched
#: against the map, so the two agree with each other while both drift away from
#: the world. That is what a front end without a back end does, and it does it
#: worst where there is least structure to pin it.
PREDICTION = (
    "taking the exact pose away costs the mapped planner about a quarter of its "
    "success in clutter, and scan matching against its own map pays most of that "
    "back where there is structure to match against and nothing where there is "
    "not. odometry minus mapped on narrow is HARMS, between -0.15 and -0.35, and "
    "on dense between -0.10 and -0.30; on large odometry loses at least 0.30, the "
    "worst of the six. matched minus odometry on narrow is MATTERS, between +0.10 "
    "and +0.35, and on dense between +0.05 and +0.35. matched minus mapped is "
    "between -0.12 and +0.05 on both narrow and dense. On sparse, matched scores "
    "below odometry by between 0.15 and 0.50 -- the one condition where matching "
    "makes things worse. Median final pose error: odometry between 0.25 and "
    "0.65 m on every condition; matched at most 0.20 m on dense and narrow, and "
    "at least 0.25 m on sparse. Collisions stay at or below 0.10 in every arm on "
    "every condition, so the cost is timeouts. full_map reproduces the published "
    "classical row on all six, mapped reproduces Phase 6e's per-episode outcomes "
    "on all six, and the perfect-odometry arm is bit-identical to mapped. "
    "Decision on odometry minus mapped, narrow -- CHEAP: INERT (bounded). "
    "COSTLY: HARMS. Otherwise UNRESOLVED. "
    "Second decision on matched minus odometry, narrow -- PAYS: MATTERS. "
    "BACKFIRES: HARMS. INERT: INERT (bounded). Otherwise UNRESOLVED."
)


class LocalisedActor:
    """The actor interface around :class:`LocalisedPursuitAgent`.

    It reads the true pose and hands it to the agent, which uses it for the
    scan and for scoring its own estimate and for nothing else; and it reads
    the robot's velocity, which is what an encoder measures.
    """

    def __init__(self, name: str, sensor: str, noise_std: float, robot,
                 odometry: OdometryConfig | None, matching: bool) -> None:
        self.name = name
        self.agent = LocalisedPursuitAgent(robot=robot, sensor=sensor, noise_std=noise_std,
                                           odometry=odometry, scan_matching=matching)
        self.errors: list[float] = []
        self.headings: list[float] = []
        self.known: list[float] = []
        self.replans: list[int] = []

    def _collect(self) -> None:
        a = self.agent
        if a.pose_errors:
            self.errors.append(a.pose_errors[-1])
            self.headings.append(float(np.degrees(a.heading_errors[-1])))
            self.known.append(a.map.known_fraction)
            self.replans.append(a.replans)

    def reset(self, env, obs) -> bool:
        self._collect()
        self.agent.robot = env.config.robot
        return self.agent.start_episode(env.world, env.robot.pose)

    def act(self, env, obs):
        return self.agent.act(env.robot.pose, env.robot.velocity)

    def finish(self) -> None:
        self._collect()

    def diagnostics(self) -> dict:
        err = np.asarray(self.errors) if self.errors else np.zeros(1)
        hdg = np.asarray(self.headings) if self.headings else np.zeros(1)
        return {"pose_error_median": float(np.median(err)),
                "pose_error_mean": float(err.mean()),
                "pose_error_max": float(err.max()),
                "heading_error_median_deg": float(np.median(hdg)),
                "heading_error_max_deg": float(hdg.max()),
                "known": float(np.mean(self.known)) if self.known else 0.0,
                "replans": float(np.mean(self.replans)) if self.replans else 0.0}


class MappedActor:
    """Phase 6e's arm, unchanged: own map, exact pose."""

    def __init__(self, sensor: str, noise_std: float, robot) -> None:
        self.name = f"mapped_{sensor}"
        self.agent = MappedPursuitAgent(robot=robot, sensor=sensor, noise_std=noise_std)

    def reset(self, env, obs) -> bool:
        self.agent.robot = env.config.robot
        return self.agent.start_episode(env.world, env.robot.pose)

    def act(self, env, obs):
        return self.agent.act(env.robot.pose)


def successes(results) -> np.ndarray:
    return np.array([int(bool(r.success)) for r in results])


def rows_of(results):
    return [{"success": bool(r.success), "collision": bool(r.collision)} for r in results]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="results/localisation_experiment.json")
    args = p.parse_args(argv)

    report = {"episodes": args.episodes, "prediction": PREDICTION, "conditions": {},
              "checks": {}}
    rng = np.random.default_rng(0)
    phase_6e = json.loads(PHASE_6E.read_text(encoding="utf-8")) if PHASE_6E.exists() else None
    reproduced_published, reproduced_6e = [], []

    for cond in CONDITIONS:
        split, shift, noise = BENCHMARK_CONDITIONS[cond]
        overrides = {"lidar": {"noise_std": noise}} if noise else {}
        cfg = build_env_config(overrides, split=split, shift=shift, n_worlds=args.episodes)

        arms, diagnostics = {}, {}
        metrics, results = evaluate(build_actor("classical", robot=cfg.robot), cfg)
        arms["full_map"] = {"metrics": metrics, "rows": rows_of(results)}

        metrics, results = evaluate(MappedActor("lidar32", noise, cfg.robot), cfg)
        arms["mapped"] = {"metrics": metrics, "rows": rows_of(results)}

        for arm, (sensor, odometry, matching) in ARMS.items():
            actor = LocalisedActor(arm, sensor, noise, cfg.robot, odometry, matching)
            metrics, results = evaluate(actor, cfg)
            actor.finish()
            arms[arm] = {"metrics": metrics, "rows": rows_of(results)}
            diagnostics[arm] = actor.diagnostics()

        # --- controls
        published = json.loads(
            Path(f"results/{cond}__classical.json").read_text(encoding="utf-8"))
        same = abs(arms["full_map"]["metrics"].success_rate - published["success_rate"]) < 1e-9
        reproduced_published.append(same)

        matches_6e = None
        if phase_6e is not None:
            before = phase_6e["conditions"][cond]["cells"]["mapped_lidar32"]
            matches_6e = ([int(r["success"]) for r in arms["mapped"]["rows"]]
                          == before["success_per_episode"])
            reproduced_6e.append(matches_6e)

        entry = {"cells": {}, "contrasts": {}, "learned": {},
                 "reproduces_published": same, "reproduces_phase_6e": matches_6e}
        for arm, data in arms.items():
            m = data["metrics"]
            entry["cells"][arm] = {
                "success": m.success_rate, "collision": m.collision_rate,
                "timeout": m.timeout_rate, "spl": m.spl,
                "success_per_episode": [int(r["success"]) for r in data["rows"]],
                **diagnostics.get(arm, {})}
        for arm, against in (("odometry", "mapped"), ("matched", "mapped"),
                             ("matched", "odometry"), ("matched_cam", "matched")):
            a, b = arms[against]["rows"], arms[arm]["rows"]
            sa, sb = successes_of(a), successes_of(b)
            p_val, lost, won = mcnemar_p(sa, sb)
            ci = paired_ci(a, b, rng, args.bootstrap)
            gain = float(sb.mean() - sa.mean())
            entry["contrasts"][f"{arm}_vs_{against}"] = {
                "success_gain": gain, "p": p_val, "episodes_won": won,
                "episodes_lost": lost, "ci95": ci, "verdict": verdict(gain, p_val, ci)}
        for label, stem in LEARNED.items():
            path = Path(f"results/{cond}__{stem}.json")
            if path.exists():
                entry["learned"][label] = json.loads(
                    path.read_text(encoding="utf-8"))["success_rate"]
        report["conditions"][cond] = entry

        c = entry["cells"]
        print(f"{cond:12s} full {c['full_map']['success']:.3f}  "
              f"mapped {c['mapped']['success']:.3f}  "
              f"odom {c['odometry']['success']:.3f} "
              f"(err {c['odometry']['pose_error_median']:.3f})  "
              f"matched {c['matched']['success']:.3f} "
              f"(err {c['matched']['pose_error_median']:.3f})  "
              f"cam {c['matched_cam']['success']:.3f}  "
              f"[published {same}, 6e {matches_6e}]", flush=True)

    # --- the identity control, on one condition
    split, shift, noise = BENCHMARK_CONDITIONS["nominal"]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split, shift=shift, n_worlds=args.episodes)
    _, exact = evaluate(LocalisedActor("exact", "lidar32", noise, cfg.robot, None, False), cfg)
    identity = (successes(exact).tolist()
                == report["conditions"]["nominal"]["cells"]["mapped"]["success_per_episode"])
    report["checks"]["perfect_odometry_is_the_mapped_agent"] = identity

    report["checks"]["full_map_reproduces_all"] = all(reproduced_published)
    report["checks"]["mapped_reproduces_phase_6e"] = (all(reproduced_6e)
                                                      if reproduced_6e else None)
    narrow = report["conditions"]["narrow"]["contrasts"]
    label = narrow["odometry_vs_mapped"]["verdict"]
    report["decision"] = ("CHEAP" if label == "INERT (bounded)"
                          else "COSTLY" if label == "HARMS" else "UNRESOLVED")
    label = narrow["matched_vs_odometry"]["verdict"]
    report["decision_matching"] = ("PAYS" if label == "MATTERS"
                                   else "BACKFIRES" if label == "HARMS"
                                   else "INERT" if label == "INERT (bounded)"
                                   else "UNRESOLVED")
    print(f"\nfull map reproduces the published row everywhere: {all(reproduced_published)}")
    print(f"mapped reproduces Phase 6e everywhere: {all(reproduced_6e) if reproduced_6e else None}")
    print(f"perfect odometry is bit-identical to the mapped agent: {identity}")
    print(f"decision, what the pose was worth (registered rule): {report['decision']}")
    print(f"decision, whether matching pays it back: {report['decision_matching']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0 if all(reproduced_published) else 1


def successes_of(rows) -> np.ndarray:
    return np.array([int(r["success"]) for r in rows])


if __name__ == "__main__":
    raise SystemExit(main())
