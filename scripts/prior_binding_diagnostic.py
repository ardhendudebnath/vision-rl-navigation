"""Does the scan matcher's motion prior suppress real corrections under noise?

§9.17 showed the `noisy_lidar` drift is invisible to the back end: every closure
links keyframes that drifted together. The drift is made in the front end, and
the first candidate is the scan matcher's motion prior. §9.5 chose it on the val
band so that about 0.03 m of match noise could not be taken seriously -- a
0.20 m departure from the odometry costs the whole scan's score -- and under
range noise the score surface is flatter, so the same prior may now override the
scan when the scan is right.

"Binding" alone would prove nothing, because a prior that overrides a noisy
argmax is doing exactly its job. So at every scan match this computes three
poses and scores each against the ground truth:

  predicted   what the odometry said
  published   what the matcher returned, with its prior
  no_prior    what the matcher's own score prefers, with the prior removed

The prior *binds* on a scan when ``published`` and ``no_prior`` differ by more
than half a coarse search step. It *suppresses a real correction* when, on such
a scan, ``no_prior`` is closer to the truth than ``published``.

Registered in ``PREDICTION`` before any val world was driven. Val only.

    python scripts/prior_binding_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.localised import LocalisedPursuitAgent
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.robot import wrap_angle
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping import localisation
from vision_nav.mapping.localisation import OCCUPIED, OdometryConfig
from vision_nav.training.env_factory import build_env_config

#: The prior binds when the two answers differ by more than half a coarse step.
BINDS = localisation.ScanMatchConfig().step / 2.0

#: Written before the data. If the prior suppresses real corrections under
#: noise, then on `noisy_lidar`:
#:
#:   1. it binds on at least twice the share of scans it binds on `nominal`.
#:      *Why this measures it:* binding is the prior changing the matcher's
#:      decision, and a prior tuned for clean scans should change it rarely on
#:      clean ones; a jump under noise is the score surface flattening under it.
#:   2. on the scans where it binds, the no-prior answer is closer to the truth
#:      than the published one more than half the time.
#:      *Why this measures it:* "suppressing a real correction" means the
#:      overridden answer was the better one, which only the ground truth can
#:      say; a prior that overrides worse answers is filtering noise, not
#:      suppressing signal.
#:   3. in the episodes that fail, the no-prior answer is closer to the truth
#:      than the published one on more scans than in the episodes that arrive.
#:      *Why this measures it:* if the prior is what lets the drift build, the
#:      suppression should be heavier exactly where the drift becomes a failure.
#:
#: The alternative: the prior binds under noise but the overridden answers are
#: worse than the published ones (claim 2 fails). Then the prior is doing its
#: job, and the drift comes from somewhere else.
PREDICTION = {
    "binding_ratio_at_least": 2.0,
    "no_prior_better_share_above": 0.5,
    "failures_suppressed_more": True,
}


def instrument(records: list, truth: list):
    """Wrap ScanMatcher.correct to also compute the no-prior answer."""
    original = localisation.ScanMatcher.correct

    def correct(self, pose, ranges, omap):
        published = original(self, pose, ranges, omap)
        cfg = self.config
        pose = np.asarray(pose, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
        max_range = float(omap.sensor.config.max_range)
        hit = ranges < max_range - 1e-6
        if int(hit.sum()) < cfg.min_hits or int((omap.grid == OCCUPIED).sum()) < cfg.min_occupied:
            return published
        bearings = np.asarray(omap.sensor._angles, dtype=float)[hit]
        r = ranges[hit]
        local = np.stack([r * np.cos(bearings), r * np.sin(bearings)], axis=-1)
        field = self.likelihood_field(omap)
        res = float(omap.resolution)
        here = self._sample(field, self._project(pose, local)[None], res)[0]
        if float((here > 0.0).mean()) < cfg.min_overlap:
            return published
        saved = (cfg.prior_xy, cfg.prior_theta)
        cfg.prior_xy, cfg.prior_theta = 1e9, 1e9
        try:
            best = self._search(pose, pose, local, field, res, cfg.window, cfg.step,
                                cfg.angular_window, cfg.angular_step)
            best = self._search(pose, best, local, field, res, cfg.step, cfg.step / 4.0,
                                cfg.angular_step, cfg.angular_step / 3.0)
        finally:
            cfg.prior_xy, cfg.prior_theta = saved
        t = truth[0]

        def err(p):
            return float(np.linalg.norm(np.asarray(p)[:2] - t[:2]))

        records.append({
            "predicted": err(pose), "published": err(published), "no_prior": err(best),
            "gap": float(np.linalg.norm(np.asarray(best)[:2] - np.asarray(published)[:2])),
            "gap_heading": abs(float(wrap_angle(best[2] - published[2]))),
        })
        return published

    return original, correct


def drive(cond: str, seeds: list[int], split: str = "val") -> list[dict]:
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {}, split=split,
                           shift=shift, n_worlds=100)
    env = ProceduralNavEnv(cfg)
    records: list = []
    truth = [np.zeros(3)]
    original, correct = instrument(records, truth)
    localisation.ScanMatcher.correct = correct
    rows = []
    try:
        for seed in seeds:
            env.reset(options={"world_seed": seed})
            agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                          noise_std=noise, odometry=OdometryConfig(),
                                          scan_matching=True, corroborate=True)
            agent.start_episode(env.world, env.robot.pose)
            records.clear()
            info: dict = {}
            for _ in range(cfg.max_episode_steps):
                truth[0] = env.robot.pose.copy()
                _, _, term, trunc, info = env.step(agent.act(env.robot.pose,
                                                             env.robot.velocity))
                if term or trunc:
                    break
            rows.append({"seed": seed, "success": bool(info.get("is_success")),
                         "scans": [dict(r) for r in records]})
            print(f"  {cond:11s} {seed}  {'arrived' if rows[-1]['success'] else 'failed '}  "
                  f"scans {len(records)}", flush=True)
    finally:
        localisation.ScanMatcher.correct = original
    return rows


def summarise(rows: list[dict]) -> dict:
    s = [x for r in rows for x in r["scans"]]
    if not s:
        return {"scans": 0}
    binds = np.array([x["gap"] > BINDS for x in s])
    better = np.array([x["no_prior"] < x["published"] for x in s])
    out = {
        "episodes": len(rows), "scans": len(s),
        "binding_share": float(binds.mean()),
        "no_prior_better_when_binding": float(better[binds].mean()) if binds.any() else None,
        "no_prior_better_overall": float(better.mean()),
        "error_predicted": float(np.median([x["predicted"] for x in s])),
        "error_published": float(np.median([x["published"] for x in s])),
        "error_no_prior": float(np.median([x["no_prior"] for x in s])),
        "gap_median_when_binding": float(np.median([x["gap"] for x, b in zip(s, binds, strict=True)
                                                    if b])) if binds.any() else None,
    }
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val100", default="results/obstacle_range_val100.json",
                   help="Which noisy val worlds failed with the published front end, "
                        "fixed by Phase 6q before this ran.")
    p.add_argument("--arrivals", type=int, default=7)
    p.add_argument("--out", default="results/prior_binding_diagnostic.json")
    args = p.parse_args(argv)

    eps = json.loads(Path(args.val100).read_text(encoding="utf-8"))
    eps = eps["cells"]["noisy_lidar"]["front_end"]["episodes"]
    failed = [e["seed"] for e in eps if not e["success"]]
    arrived = [e["seed"] for e in eps if e["success"]][:args.arrivals]
    print(f"noisy_lidar: {len(failed)} failing worlds and {len(arrived)} arrivals; "
          f"nominal: {args.arrivals} worlds", flush=True)
    noisy = drive("noisy_lidar", failed + arrived)
    nominal = drive("nominal", list(range(10000, 10000 + args.arrivals)))

    report = {"split": "val", "registered": PREDICTION, "binds_above": BINDS,
              "noisy": summarise(noisy),
              "noisy_failed": summarise([r for r in noisy if not r["success"]]),
              "noisy_arrived": summarise([r for r in noisy if r["success"]]),
              "nominal": summarise(nominal)}
    for k in ("nominal", "noisy", "noisy_failed", "noisy_arrived"):
        e = report[k]
        print(f"\n{k:14s} {e['scans']} scans: prior binds on {e['binding_share']:.0%}; "
              f"no-prior closer to truth when binding "
              f"{e['no_prior_better_when_binding'] if e['no_prior_better_when_binding'] is None else format(e['no_prior_better_when_binding'], '.0%')}"
              f", overall {e['no_prior_better_overall']:.0%}")
        print(f"{'':14s} error vs truth, median: predicted {e['error_predicted']:.3f} m, "
              f"published {e['error_published']:.3f} m, no prior {e['error_no_prior']:.3f} m")
    ratio = report["noisy"]["binding_share"] / max(report["nominal"]["binding_share"], 1e-9)
    held = [ratio >= PREDICTION["binding_ratio_at_least"],
            (report["noisy"]["no_prior_better_when_binding"] or 0.0)
            > PREDICTION["no_prior_better_share_above"],
            report["noisy_failed"]["no_prior_better_overall"]
            > report["noisy_arrived"]["no_prior_better_overall"]]
    report["binding_ratio"] = float(ratio)
    report["held"] = held
    print(f"\nclaim 1 -- binds twice as often under noise ({ratio:.1f}x): "
          f"{'HELD' if held[0] else 'FAILED'}")
    print(f"claim 2 -- overridden answers are better more than half the time: "
          f"{'HELD' if held[1] else 'FAILED'}")
    print(f"claim 3 -- suppression heavier in failures: {'HELD' if held[2] else 'FAILED'}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
