"""Why do correct loop closures not fix the drift under noise?

§9.15 found the back end engaged exactly where `noisy_lidar` fails: failing
episodes accept 27.5 closures each against 0.8 for arrivals, and 299 of 302
accepted closures sit within 0.15 m of the true relative pose. The robots still
drift to about half a metre and stop short of the goal believing they have
arrived. A closure that is right and changes nothing has to be wrong about
something else, and there are two candidates.

**Where it reaches.** A closure measures the relative pose of two keyframes. If
both drifted together -- late in an episode spent circling where the robot
believes the goal is -- the measurement agrees with the drifted chain and the
solve has nothing to correct. Only a closure back to a keyframe that was still
well localised carries information about the absolute error.

**How hard it pulls.** If closures *do* reach well-localised keyframes, the
solve should pull the drift out. It may not: each closure is weighted below
each odometry edge (0.6 against 1.0 in translation, 2.0 against 4.0 in
rotation, :class:`PoseGraphConfig`), and twenty odometry edges in between,
each confident, can outvote one closure.

So for every accepted closure this records the true error of both keyframes at
the moment each was added, and for every solve that has closures to work with,
the newest keyframe's true error before and after.

Registered in ``PREDICTION`` before this ran. Val only; nothing here is scored.

    python scripts/closure_reach_diagnostic.py
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
from vision_nav.training.env_factory import build_env_config

#: A keyframe within this distance of the truth when it was added was still
#: well localised; one beyond the larger figure had already drifted.
WELL, DRIFTED = 0.15, 0.30

#: Written before the data, for the failing episodes. If the closures link
#: keyframes that drifted together:
#:
#:   1. the median *anchor* -- the older keyframe a closure reaches back to --
#:      was already more than 0.30 m off when it was added;
#:   2. fewer than 20% of closures reach a keyframe that was within 0.15 m;
#:   3. a solve moves the newest keyframe's true error by less than 0.05 m at
#:      the median, because there is little disagreement to resolve.
#:
#: The alternative leaves a different signature: anchors that were well
#: localised (claim 1 and 2 fail) and solves that still barely move the error
#: (claim 3 holds). That would put the blame on how hard a closure pulls -- the
#: information weights -- rather than on where it reaches.
PREDICTION = {
    "anchor_error_median_above": DRIFTED,
    "well_localised_anchor_share_below": 0.20,
    "solve_change_median_below": 0.05,
}


def run(seeds: list[int], cfg, env) -> list[dict]:
    created: dict[int, list] = {}
    closures: dict[int, list] = {}
    solves: dict[int, list] = {}
    current = [np.zeros(3)]
    add, find, optimise = (posegraph.PoseGraph.add_keyframe,
                           posegraph.PoseGraph.find_closure,
                           posegraph.PoseGraph.optimise)

    def err(estimate, truth):
        return float(np.linalg.norm(np.asarray(estimate)[:2] - np.asarray(truth)[:2]))

    def recording_add(self, pose, local):
        created.setdefault(id(self), []).append(
            (np.asarray(pose, dtype=float).copy(), current[0].copy()))
        return add(self, pose, local)

    def recording_find(self, i):
        before = len(self.edges)
        ok = find(self, i)
        if ok and len(self.edges) > before:
            j, k = self.edges[-1][0], self.edges[-1][1]
            meta = created[id(self)]
            closures.setdefault(id(self), []).append({
                "anchor": int(j), "newest": int(k), "gap": int(k - j),
                "anchor_error": err(*meta[j]), "newest_error": err(*meta[k])})
        return ok

    def recording_optimise(self):
        meta = created.get(id(self), [])
        has_loops = len(closures.get(id(self), [])) > 0
        before = err(self.poses[-1], meta[-1][1]) if (meta and has_loops) else None
        moved = optimise(self)
        if before is not None:
            solves.setdefault(id(self), []).append(
                {"before": before, "after": err(self.poses[-1], meta[-1][1])})
        return moved

    posegraph.PoseGraph.add_keyframe = recording_add
    posegraph.PoseGraph.find_closure = recording_find
    posegraph.PoseGraph.optimise = recording_optimise
    rows = []
    try:
        for seed in seeds:
            env.reset(options={"world_seed": seed})
            agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                          noise_std=cfg.lidar.noise_std,
                                          odometry=OdometryConfig(), scan_matching=True,
                                          corroborate=True, pose_graph=True)
            agent.start_episode(env.world, env.robot.pose)
            info: dict = {}
            for _ in range(cfg.max_episode_steps):
                current[0] = env.robot.pose.copy()
                _, _, term, trunc, info = env.step(agent.act(env.robot.pose,
                                                             env.robot.velocity))
                if term or trunc:
                    break
            gid = id(agent.graph)
            rows.append({"seed": seed, "success": bool(info.get("is_success")),
                         "keyframes": len(agent.graph.poses),
                         "closures": closures.pop(gid, []),
                         "solves": solves.pop(gid, [])})
            created.pop(gid, None)
            print(f"  {seed}  {'arrived' if rows[-1]['success'] else 'failed '}  "
                  f"keyframes {rows[-1]['keyframes']:3d}  closures "
                  f"{len(rows[-1]['closures']):3d}", flush=True)
    finally:
        posegraph.PoseGraph.add_keyframe = add
        posegraph.PoseGraph.find_closure = find
        posegraph.PoseGraph.optimise = optimise
    return rows


def summarise(rows: list[dict]) -> dict:
    cl = [c for r in rows for c in r["closures"]]
    sv = [s for r in rows for s in r["solves"]]
    if not cl:
        return {"episodes": len(rows), "closures": 0}
    anchor = np.array([c["anchor_error"] for c in cl])
    change = np.array([s["before"] - s["after"] for s in sv]) if sv else np.zeros(0)
    return {
        "episodes": len(rows), "closures": len(cl),
        "anchor_error_median": float(np.median(anchor)),
        "newest_error_median": float(np.median([c["newest_error"] for c in cl])),
        "well_localised_anchor_share": float(np.mean(anchor <= WELL)),
        "drifted_anchor_share": float(np.mean(anchor > DRIFTED)),
        "gap_median": float(np.median([c["gap"] for c in cl])),
        "anchor_index_median": float(np.median([c["anchor"] for c in cl])),
        "solves": len(sv),
        "solve_change_median": float(np.median(change)) if len(change) else None,
        "solve_change_abs_median": float(np.median(np.abs(change))) if len(change) else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined", default="results/combined_noisy_val.json",
                   help="Which val worlds failed, and which formed closures, with the "
                        "back end on -- fixed by §9.15 before this ran.")
    p.add_argument("--out", default="results/closure_reach_diagnostic.json")
    args = p.parse_args(argv)
    eps = json.loads(Path(args.combined).read_text(encoding="utf-8"))["arms"]["back_end"]["episodes"]
    seeds = [e["seed"] for e in eps if e["closures"] > 0]
    _, shift, noise = BENCHMARK_CONDITIONS["noisy_lidar"]
    cfg = build_env_config({"lidar": {"noise_std": noise}}, split="val", shift=shift,
                           n_worlds=100)
    env = ProceduralNavEnv(cfg)
    print(f"{len(seeds)} val worlds that formed at least one closure with the back end on",
          flush=True)
    rows = run(seeds, cfg, env)
    report = {"split": "val", "registered": PREDICTION,
              "failed": summarise([r for r in rows if not r["success"]]),
              "arrived": summarise([r for r in rows if r["success"]]),
              "episodes": rows}
    f = report["failed"]
    print(f"\nfailed episodes: {f['closures']} closures over {f['episodes']} episodes")
    print(f"  anchor error median {f['anchor_error_median']:.3f} m  (newest keyframe "
          f"{f['newest_error_median']:.3f} m)  index gap median {f['gap_median']:.0f}  "
          f"anchor index median {f['anchor_index_median']:.0f}")
    print(f"  anchors within {WELL} m: {f['well_localised_anchor_share']:.0%}   "
          f"beyond {DRIFTED} m: {f['drifted_anchor_share']:.0%}")
    print(f"  {f['solves']} solves with closures: newest error changed by "
          f"{f['solve_change_median']:+.3f} m at the median (|change| "
          f"{f['solve_change_abs_median']:.3f} m)")
    held = [f["anchor_error_median"] > PREDICTION["anchor_error_median_above"],
            f["well_localised_anchor_share"] < PREDICTION["well_localised_anchor_share_below"],
            (f["solve_change_abs_median"] or 0.0) < PREDICTION["solve_change_median_below"]]
    report["held"] = held
    for n, (name, ok) in enumerate(zip(("anchors already drifted",
                                        "few well-localised anchors",
                                        "solves barely move the error"), held,
                                       strict=True), 1):
        print(f"  claim {n} -- {name}: {'HELD' if ok else 'FAILED'}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
