"""Which cause keeps the solve from using correct closures? Re-solve the graphs.

§9.16 found that in the failing `noisy_lidar` episodes over a third of the
back end's closures reach keyframes that were still well localised, and a solve
still moves the newest pose by 0.017 m at the median, slightly the wrong way.
Two causes were left and that measurement could not separate them:

* **The weights.** Each closure is weighted below each odometry edge -- 0.6
  against 1.0 in translation, 2.0 against 4.0 in rotation -- so a chain of
  confident odometry edges can outvote it.
* **The drifted anchors.** 40% of the closures reach keyframes that had already
  drifted beyond 0.30 m. Such a closure tells the solve the newest keyframe sits
  at the true relative pose *from a drifted anchor*, which pulls it towards the
  drift; the well-anchored ones pull it towards the truth; and the two may
  cancel.

Both are properties of the solve, so they can be separated without driving an
episode differently. The failing episodes are driven once with the published
back end and their graphs saved -- every edge, its measurement and weight, and
each keyframe's estimate and true pose when it was added. Each graph is then
re-solved offline from those estimates, four ways:

  published        as the robot solved it
  weights_up       closures weighted as the odometry is
  good_anchors     only closures whose anchor was within 0.15 m of the truth
  both             both changes

``good_anchors`` uses the ground truth to choose closures. It is a diagnostic,
not a rule a robot could run: it asks whether the drifted-anchor closures are
what blocks the correction, and a deployable version would need a proxy for
which keyframes are still well localised.

The control comes first: the offline ``published`` re-solve has to reproduce the
pose the robot actually ended with, or the offline answers do not speak for it.

Registered in ``PREDICTION`` before any val graph was saved. Val only.

    python scripts/closure_resolve_diagnostic.py
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

WELL = 0.15
#: The decision rule, and my guess, both written before the data.
#:
#: A variant counts as removing the obstruction if it cuts the median true
#: error of the final keyframe by at least 25% against the ``published``
#: re-solve. Whichever does names the cause; if both do, both contribute; if
#: neither does, it is something about the solve itself.
#:
#: My directional guess, scored as two predictions:
#:   1. ``good_anchors`` cuts the median final-keyframe error by at least 25%:
#:      the drifted-anchor closures pull towards the drift and cancel the good
#:      ones, so removing them should let the solve correct.
#:   2. ``weights_up`` cuts it by less than 25%: raising every closure's weight
#:      amplifies the drifted ones as much as the good ones.
#:
#: And the control, which is not a prediction: the ``published`` re-solve
#: reproduces the robot's final keyframe pose to within 0.05 m at the median.
PREDICTION = {
    "decisive_cut": 0.25,
    "good_anchors_cut_at_least": 0.25,
    "weights_up_cut_below": 0.25,
    "control_reproduction_within": 0.05,
}
VARIANTS = ("published", "weights_up", "good_anchors", "both")


def drive(seeds: list[int], split: str = "val") -> list[dict]:
    """Drive each world with the published back end and save its graph."""
    _, shift, noise = BENCHMARK_CONDITIONS["noisy_lidar"]
    cfg = build_env_config({"lidar": {"noise_std": noise}}, split=split, shift=shift,
                           n_worlds=100)
    env = ProceduralNavEnv(cfg)
    created: list = []
    current = [np.zeros(3)]
    add = posegraph.PoseGraph.add_keyframe

    def recording_add(self, pose, local):
        created.append({"estimate": np.asarray(pose, dtype=float).tolist(),
                        "truth": current[0].tolist()})
        return add(self, pose, local)

    posegraph.PoseGraph.add_keyframe = recording_add
    dumps = []
    try:
        for seed in seeds:
            env.reset(options={"world_seed": seed})
            agent = LocalisedPursuitAgent(robot=cfg.robot, sensor="lidar360",
                                          noise_std=noise, odometry=OdometryConfig(),
                                          scan_matching=True, corroborate=True,
                                          pose_graph=True)
            created.clear()
            agent.start_episode(env.world, env.robot.pose)
            info: dict = {}
            for _ in range(cfg.max_episode_steps):
                current[0] = env.robot.pose.copy()
                _, _, term, trunc, info = env.step(agent.act(env.robot.pose,
                                                             env.robot.velocity))
                if term or trunc:
                    break
            g = agent.graph
            dumps.append({
                "seed": seed, "success": bool(info.get("is_success")),
                "keyframes": [dict(k) for k in created],
                "final_poses": [p.tolist() for p in g.poses],
                "edges": [{"i": int(i), "j": int(j), "z": np.asarray(z).tolist(),
                           "w_xy": float(wx), "w_th": float(wt)}
                          for i, j, z, wx, wt in g.edges],
                "closures": int(g.closures),
            })
            print(f"  {seed}  {'arrived' if dumps[-1]['success'] else 'failed '}  "
                  f"keyframes {len(created):3d}  closures {g.closures:3d}", flush=True)
    finally:
        posegraph.PoseGraph.add_keyframe = add
    return dumps


def resolve(dump: dict, variant: str) -> np.ndarray:
    """Re-solve one saved graph from its keyframes' estimates at creation."""
    cfg = posegraph.PoseGraphConfig()
    graph = posegraph.PoseGraph(cfg)
    graph.poses = [np.asarray(k["estimate"], dtype=float) for k in dump["keyframes"]]
    graph.scans = [np.zeros((0, 2)) for _ in graph.poses]
    truth = [np.asarray(k["truth"], dtype=float) for k in dump["keyframes"]]
    anchor_error = [float(np.linalg.norm(np.asarray(k["estimate"])[:2]
                                         - np.asarray(k["truth"])[:2]))
                    for k in dump["keyframes"]]
    edges = []
    for e in dump["edges"]:
        loop = e["j"] - e["i"] > 1
        if loop and variant in ("good_anchors", "both") and anchor_error[e["i"]] > WELL:
            continue
        w_xy, w_th = e["w_xy"], e["w_th"]
        if loop and variant in ("weights_up", "both"):
            w_xy, w_th = cfg.odom_xy_weight, cfg.odom_theta_weight
        edges.append((e["i"], e["j"], np.asarray(e["z"], dtype=float), w_xy, w_th))
    graph.edges = edges
    graph.optimise()
    return np.array([float(np.linalg.norm(p[:2] - t[:2]))
                     for p, t in zip(graph.poses, truth, strict=True)])


def closure_residuals(dumps: list[dict]) -> dict:
    """Exploratory, added after the registered re-solve. Do the closures
    disagree with the drifted chain at all?

    A solve moves poses only where a measurement disagrees with what the poses
    imply. For a closure that disagreement is the residual between the relative
    pose the chain implies -- from the keyframes' estimates -- and the relative
    pose the closure measured. §9.16 tested the co-drift reading through each
    anchor's *absolute* error, which does not measure it: two keyframes can both
    be well off and agree perfectly with each other if they drifted together.
    What decides it is the drift *between* the linked pair.
    """
    from vision_nav.mapping.posegraph import relative_pose

    resid, truth_err, pair_drift, anchor, newest = [], [], [], [], []
    for d in dumps:
        if d["success"]:
            continue
        est = [np.asarray(k["estimate"], dtype=float) for k in d["keyframes"]]
        tru = [np.asarray(k["truth"], dtype=float) for k in d["keyframes"]]
        for e in d["edges"]:
            if e["j"] - e["i"] <= 1:
                continue
            i, j, z = e["i"], e["j"], np.asarray(e["z"], dtype=float)
            chain, true = relative_pose(est[i], est[j]), relative_pose(tru[i], tru[j])
            resid.append(float(np.linalg.norm(chain[:2] - z[:2])))
            truth_err.append(float(np.linalg.norm(true[:2] - z[:2])))
            pair_drift.append(float(np.linalg.norm(chain[:2] - true[:2])))
            anchor.append(float(np.linalg.norm(est[i][:2] - tru[i][:2])))
            newest.append(float(np.linalg.norm(est[j][:2] - tru[j][:2])))
    r, t, pd_, a, n = (np.asarray(x) for x in (resid, truth_err, pair_drift, anchor, newest))
    good = a <= WELL
    return {
        "closures": len(r),
        "against_truth_median": float(np.median(t)),
        "residual_median": float(np.median(r)),
        "residual_above_0.10": float(np.mean(r > 0.10)),
        "pair_drift_median": float(np.median(pd_)),
        "anchor_error_median": float(np.median(a)),
        "newest_error_median": float(np.median(n)),
        "good_anchor_closures": int(good.sum()),
        "good_anchor_newest_median": float(np.median(n[good])) if good.any() else None,
        "good_anchor_residual_median": float(np.median(r[good])) if good.any() else None,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined", default="results/combined_noisy_val.json",
                   help="Which val worlds failed and formed closures, fixed by §9.15.")
    p.add_argument("--graphs", default="results/closure_resolve_graphs.json")
    p.add_argument("--out", default="results/closure_resolve_diagnostic.json")
    args = p.parse_args(argv)

    graphs = Path(args.graphs)
    if graphs.exists():
        dumps = json.loads(graphs.read_text(encoding="utf-8"))
        print(f"using {len(dumps)} saved graphs", flush=True)
    else:
        eps = json.loads(Path(args.combined).read_text(encoding="utf-8"))
        eps = eps["arms"]["back_end"]["episodes"]
        seeds = [e["seed"] for e in eps if not e["success"] and e["closures"] > 0]
        print(f"driving {len(seeds)} failing val worlds that formed closures", flush=True)
        dumps = drive(seeds)
        graphs.parent.mkdir(parents=True, exist_ok=True)
        graphs.write_text(json.dumps(dumps), encoding="utf-8")

    rows = []
    for d in dumps:
        if d["success"]:
            continue
        robot_final = np.asarray(d["final_poses"][-1])
        truth_final = np.asarray(d["keyframes"][-1]["truth"])
        unsolved = np.array([float(np.linalg.norm(np.asarray(k["estimate"])[:2]
                                                  - np.asarray(k["truth"])[:2]))
                             for k in d["keyframes"]])
        row = {"seed": d["seed"], "keyframes": len(d["keyframes"]),
               "closures": d["closures"],
               "robot_final_error": float(np.linalg.norm(robot_final[:2] - truth_final[:2])),
               "unsolved_final_error": float(unsolved[-1]),
               "unsolved_mean_error": float(unsolved.mean())}
        for v in VARIANTS:
            err = resolve(d, v)
            row[f"{v}_final_error"] = float(err[-1])
            row[f"{v}_mean_error"] = float(err.mean())
        rows.append(row)

    def med(key):
        return float(np.median([r[key] for r in rows]))

    base = med("published_final_error")
    report: dict = {"split": "val", "registered": PREDICTION, "episodes": rows,
                    "median": {k: med(k) for k in rows[0] if k not in ("seed",)}}
    report["cut"] = {v: 1.0 - med(f"{v}_final_error") / base for v in VARIANTS[1:]}
    report["control"] = float(np.median([abs(r["published_final_error"]
                                              - r["robot_final_error"]) for r in rows]))

    print(f"\n{len(rows)} failing episodes, final keyframe's true error, median:")
    print(f"  robot, as it ended        {med('robot_final_error'):.3f} m")
    print(f"  unsolved front end        {med('unsolved_final_error'):.3f} m")
    for v in VARIANTS:
        extra = "" if v == "published" else f"   cut {report['cut'][v]:+.0%}"
        print(f"  {v:24s}  {med(f'{v}_final_error'):.3f} m  (mean over keyframes "
              f"{med(f'{v}_mean_error'):.3f} m){extra}")
    print(f"\ncontrol -- published re-solve against the robot's own final pose: "
          f"{report['control']:.3f} m at the median, registered within "
          f"{PREDICTION['control_reproduction_within']} m: "
          f"{'REPRODUCES' if report['control'] <= PREDICTION['control_reproduction_within'] else 'DOES NOT REPRODUCE'}")
    held = [report["cut"]["good_anchors"] >= PREDICTION["good_anchors_cut_at_least"],
            report["cut"]["weights_up"] < PREDICTION["weights_up_cut_below"]]
    report["held"] = held
    print(f"claim 1 -- good anchors cut it by at least 25%: {'HELD' if held[0] else 'FAILED'}")
    print(f"claim 2 -- weights alone cut it by less than 25%: {'HELD' if held[1] else 'FAILED'}")
    cause = [v for v in ("weights_up", "good_anchors")
             if report["cut"][v] >= PREDICTION["decisive_cut"]]
    report["cause"] = cause or ["the solve itself"]
    print(f"decision rule -- the cause: {', '.join(report['cause'])}")
    # How far the published re-solve moves any keyframe at all.
    moved = []
    for d in dumps:
        if d["success"]:
            continue
        unsolved = np.array([float(np.linalg.norm(np.asarray(k["estimate"])[:2]
                                                  - np.asarray(k["truth"])[:2]))
                             for k in d["keyframes"]])
        moved.append(float(np.max(np.abs(resolve(d, "published") - unsolved))))
    report["largest_move_median"] = float(np.median(moved))
    report["largest_move_max"] = float(np.max(moved))
    report["residuals"] = closure_residuals(dumps)
    rr = report["residuals"]
    print("\nexploratory, after the registered test -- what the closures disagree with:")
    print(f"  {rr['closures']} closures: against the true relative pose "
          f"{rr['against_truth_median']:.3f} m; against the chain's "
          f"{rr['residual_median']:.3f} m ({rr['residual_above_0.10']:.0%} above 0.10 m)")
    print(f"  drift between the linked pair {rr['pair_drift_median']:.3f} m, against "
          f"absolute errors of {rr['anchor_error_median']:.3f} m and "
          f"{rr['newest_error_median']:.3f} m")
    print(f"  the {rr['good_anchor_closures']} well-anchored closures came from newest "
          f"keyframes {rr['good_anchor_newest_median']:.3f} m out")
    print(f"  re-solving moves any keyframe by {report['largest_move_median']:.3f} m at the "
          f"median episode, {report['largest_move_max']:.3f} m at most")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
