"""Why does the pixel policy stall? A training-free audit of Phase 5s's arms.

Phase 5s: at the 1:1 reward the RGB policy times out on 0.182 of `narrow`
episodes and the depth policy, reading the same geometry as a vector, on 0.028.
Coverage's deficit moved into collisions under the same change; the encoding
deficit kept stalling. Two measurements, neither needing training
(``vision_nav.analysis.pixel_audit`` defines both):

**Part 1, information.** For each surface kind and distance, the span of
distances that render to an identical image column. Phase 3e's premise was that
the render holds information constant. Below ``wall_scale`` (1.2 m) a surface
fills its column and only eight-bit shading carries distance, so the premise is
approximate at best; this says how approximate.

**Part 2, anatomy.** Each of the twelve Phase 5s policies driven over the same
50 `narrow` worlds. Every step is classified by the world's true geometry --
``goal_open``, ``detour`` or ``boxed`` -- and every step spent going nowhere
(under 0.15 m of net movement in 3 s) is a stall step. A linear probe on each
policy's final actor layer, cross-validated by episode, asks whether that layer
encodes ``goal_open``: whether the policy *knows* the goal-ward route is open.

    python scripts/pixel_stall_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from vision_nav.analysis.pixel_audit import (
    STALL_CLASSES,
    depth_ambiguity,
    stall_geometry,
    stall_steps,
)
from vision_nav.envs.rgb_camera import RGBCameraConfig

SEEDS = range(6)
ARMS = ("depthi", "rgbi")
NEAR = 1.2  # wall_scale: below it a surface fills its column

#: Written before the audit was run. See the last paragraph for why it does not
#: count as pre-registered.
#:
#: Confidence LOW throughout. Part 1 is a calculation done by hand before
#: writing the code: the brightest channel of the dimmest surface loses about
#: 8.8 eight-bit levels per metre of distance, so one level spans about 0.11 m,
#: and the three channels' boundaries interleave, so the joint span should be
#: smaller. Part 2 has no measurement behind it at all. The lean is that a
#: policy which cannot read *where* the opening is will hold still when the
#: direct route is blocked, which is what `narrow` is made of -- but a policy
#: frozen in front of an open route, or stuck facing a wall, would stall just
#: as well, and nothing here yet says which.
#:
#: NOT A CLEAN PRE-REGISTRATION, and not counted in the calibration record. The
#: text below was written before any result, but a smoke test of this script on
#: two worlds printed its output before the commit: Part 1 in full (it does not
#: depend on the worlds, so that *was* the result) and a two-world Part 2 in
#: which two RGB seeds stalled, every stall step `detour`. Neither was edited
#: afterwards. One field was added after the smoke test and is outside every
#: decision: whether the goal was in view, because `detour` lumps "the goal-ward
#: route is blocked" together with "the goal is behind the robot".
#:
#: Two more were added after full runs, each re-run and checked to reproduce
#: every earlier number exactly. After the first: the probe's error on
#: `goal_open` steps where the robot was moving, the base rate without which its
#: error at stalls cannot be read. After the second: the same probe on the
#: sensor pathway alone, because the actor layer also takes the robot's own
#: velocity, which is near zero at a stall by definition -- and that turned out
#: to be what the actor-layer probe was reading.
PREDICTION = (
    "Part 1 -- INFORMATION HELD: the widest span of distances rendering to an "
    "identical column is at most 0.15 m for every surface kind between 0.2 m and "
    "6 m, including below 1.2 m. "
    "Part 2 -- BLIND DETOUR: rgbi stalls on a larger share of steps than depthi "
    "(exact permutation over seeds, p < 0.05); over half of rgbi's stall steps are "
    "`detour`, and `detour` is a larger share of its stall steps than of its "
    "moving steps; `goal_open` is under a quarter of them. The probe decodes "
    "`goal_open` from rgbi's actor layer at a balanced accuracy at least 0.05 "
    "below depthi's (p < 0.05). "
    "Decision for Part 2 -- the class holding over half of rgbi's pooled stall "
    "steps: PHANTOM (goal_open), BLIND DETOUR (detour), BOXED (boxed); MIXED if "
    "none does."
)


def permutation_p(a, b) -> float:
    pooled = np.concatenate([a, b])
    n, observed = len(a), abs(np.mean(a) - np.mean(b))
    hits = total = 0
    for idx in combinations(range(len(pooled)), n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(idx)] = True
        total += 1
        hits += abs(pooled[mask].mean() - pooled[~mask].mean()) >= observed - 1e-12
    return hits / total


def part_one(cfg: RGBCameraConfig) -> dict:
    grid = np.arange(0.2, 6.0, 0.0005)
    near = grid < NEAR
    out = {}
    for kind, name in enumerate(("wall", "circle", "box")):
        amb = depth_ambiguity(cfg, kind, grid)
        out[name] = {"max_m": float(amb.max()), "median_m": float(np.median(amb)),
                     "max_below_1p2_m": float(amb[near].max()),
                     "median_below_1p2_m": float(np.median(amb[near]))}
    out["max_over_kinds_m"] = max(v["max_m"] for v in out.values())
    return out


def drive(job):
    """One policy over the audit worlds. Runs in a worker process."""
    arm, seed, n_worlds = job
    import torch
    from stable_baselines3.common.preprocessing import preprocess_obs

    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.training.actors import build_actor
    from vision_nav.training.env_factory import build_env_config
    from vision_nav.training.run_spec import env_overrides_for_run

    torch.set_num_threads(1)
    run = Path("runs") / f"{arm}_s{seed}"
    cfg = build_env_config(env_overrides_for_run(run), split="test_ood", shift="narrow",
                           n_worlds=n_worlds)
    env = ProceduralNavEnv(cfg)
    actor = build_actor("rl", model_path=str(run / "best_model.zip"), robot=cfg.robot)
    policy = actor.model.policy
    fov = cfg.camera.fov if cfg.obs_mode == "depth" else cfg.rgb_camera.fov
    columns = cfg.camera.width if cfg.obs_mode == "depth" else cfg.rgb_camera.width

    latents, classes, stalled, episode, outcomes, in_view = [], [], [], [], [], []
    # What the sensor pathway alone carries, before the goal and velocity vector
    # joins it: the CNN's image features for RGB, the depth vector itself for
    # depth. Added after the second run -- the actor layer mixes in the robot's
    # own velocity, which is near zero at a stall by definition.
    sensor = []
    for ep, world_seed in enumerate(cfg.world_seeds):
        obs, _ = env.reset(options={"world_seed": int(world_seed)})
        actor.reset(env, obs)
        positions = []
        while True:
            pose = env.robot.pose.copy()
            positions.append(pose[:2])
            classes.append(STALL_CLASSES.index(stall_geometry(env.world, pose, fov, columns)))
            to_goal = env.world.goal - pose[:2]
            bearing = np.arctan2(to_goal[1], to_goal[0]) - pose[2]
            in_view.append(abs(np.arctan2(np.sin(bearing), np.cos(bearing))) <= fov / 2.0)
            with torch.no_grad():
                obs_t, _ = policy.obs_to_tensor(obs)
                feats = policy.extract_features(obs_t, policy.pi_features_extractor)
                latents.append(policy.mlp_extractor.forward_actor(feats)[0].numpy())
                if cfg.obs_mode == "rgb":
                    image = preprocess_obs(obs_t, policy.observation_space,
                                           normalize_images=policy.normalize_images)["image"]
                    sensor.append(policy.features_extractor.extractors["image"](image)[0].numpy())
                else:
                    sensor.append(np.asarray(obs[:columns], dtype=np.float32))
            obs, _, term, trunc, info = env.step(actor.act(env, obs))
            if term or trunc:
                break
        stalled.extend(stall_steps(np.asarray(positions), cfg.robot.dt))
        episode.extend([ep] * len(positions))
        outcomes.append("success" if info.get("is_success") else
                        "collision" if info.get("collision") else "timeout")
    return (arm, seed, np.asarray(latents, dtype=np.float32), np.asarray(classes),
            np.asarray(stalled), np.asarray(episode), outcomes, np.asarray(in_view),
            np.asarray(sensor, dtype=np.float32))


def probe(latents, target, episode, folds=5, ridge=1.0):
    """Cross-validated linear probe, folds split by episode. Returns held-out
    predictions for every step and the balanced accuracy."""
    x = (latents - latents.mean(0)) / (latents.std(0) + 1e-6)
    x = np.hstack([x, np.ones((len(x), 1))])
    y = np.where(target, 1.0, -1.0)
    pred = np.zeros(len(y), dtype=bool)
    fold_of = episode % folds
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        w = np.linalg.solve(x[tr].T @ x[tr] + ridge * np.eye(x.shape[1]), x[tr].T @ y[tr])
        pred[te] = x[te] @ w > 0
    pos, neg = target, ~target
    bal = 0.5 * ((pred[pos].mean() if pos.any() else np.nan)
                 + ((~pred[neg]).mean() if neg.any() else np.nan))
    return pred, float(bal)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="results/pixel_stall_audit.json")
    args = p.parse_args(argv)

    report = {"prediction": PREDICTION, "worlds": args.worlds}
    report["part1"] = one = part_one(RGBCameraConfig())
    print("Part 1 -- widest span of distances rendering identically:")
    for name in ("wall", "circle", "box"):
        v = one[name]
        print(f"  {name:6s} max {v['max_m']:.3f} m  median {v['median_m']:.3f} m  "
              f"below 1.2 m: max {v['max_below_1p2_m']:.3f}  median {v['median_below_1p2_m']:.3f}")
    report["part1_decision"] = ("INFORMATION HELD" if one["max_over_kinds_m"] <= 0.15
                                else "INFORMATION LOST")
    print("  decision:", report["part1_decision"])

    jobs = [(arm, s, args.worlds) for arm in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(drive, jobs))

    per_seed = {arm: [] for arm in ARMS}
    pooled = {arm: {"stall": np.zeros(3), "moving": np.zeros(3), "goal_open_stalls": 0,
                    "goal_open_stalls_probe_says_blocked": 0, "detour_stalls": 0,
                    "detour_stalls_goal_out_of_view": 0, "goal_open_moving": 0,
                    "goal_open_moving_probe_says_blocked": 0,
                    "goal_open_stalls_sensor_probe_says_blocked": 0,
                    "goal_open_moving_sensor_probe_says_blocked": 0} for arm in ARMS}
    for arm, seed, latents, classes, stalled, episode, outcomes, in_view, sensor in results:
        sensor_pred, sensor_bal = probe(sensor, classes == 0, episode)
        detour_stalls = stalled & (classes == 1)
        pooled[arm]["detour_stalls"] += int(detour_stalls.sum())
        pooled[arm]["detour_stalls_goal_out_of_view"] += int((detour_stalls & ~in_view).sum())
        pred, bal = probe(latents, classes == 0, episode)
        stall_classes = np.bincount(classes[stalled], minlength=3)
        moving_classes = np.bincount(classes[~stalled], minlength=3)
        pooled[arm]["stall"] += stall_classes
        pooled[arm]["moving"] += moving_classes
        open_stalls = stalled & (classes == 0)
        pooled[arm]["goal_open_stalls"] += int(open_stalls.sum())
        pooled[arm]["goal_open_stalls_probe_says_blocked"] += int((open_stalls & ~pred).sum())
        # The base rate that count needs, added after the first full run.
        open_moving = ~stalled & (classes == 0)
        pooled[arm]["goal_open_moving"] += int(open_moving.sum())
        pooled[arm]["goal_open_moving_probe_says_blocked"] += int((open_moving & ~pred).sum())
        pooled[arm]["goal_open_stalls_sensor_probe_says_blocked"] += int(
            (stalled & (classes == 0) & ~sensor_pred).sum())
        pooled[arm]["goal_open_moving_sensor_probe_says_blocked"] += int(
            (open_moving & ~sensor_pred).sum())
        entry = {"seed": seed, "steps": int(len(classes)),
                 "stall_share": float(stalled.mean()),
                 "stall_classes": stall_classes.tolist(), "moving_classes": moving_classes.tolist(),
                 "probe_balanced_accuracy": bal,
                 "sensor_probe_balanced_accuracy": sensor_bal,
                 "goal_open_share": float((classes == 0).mean()),
                 "timeout_rate": outcomes.count("timeout") / len(outcomes),
                 "collision_rate": outcomes.count("collision") / len(outcomes),
                 "success_rate": outcomes.count("success") / len(outcomes)}
        per_seed[arm].append(entry)
        print(f"  {arm:6s} s{seed}: stall share {entry['stall_share']:.3f}  "
              f"stall classes {stall_classes.tolist()}  probe {bal:.3f}  "
              f"timeouts {entry['timeout_rate']:.2f}", flush=True)

    report["per_seed"] = per_seed
    summary = {}
    for arm in ARMS:
        st, mv = pooled[arm]["stall"], pooled[arm]["moving"]
        summary[arm] = {
            "stall_share_mean": float(np.mean([e["stall_share"] for e in per_seed[arm]])),
            "stall_class_shares": dict(zip(STALL_CLASSES, (st / max(st.sum(), 1)).tolist(),
                                           strict=True)),
            "moving_class_shares": dict(zip(STALL_CLASSES, (mv / max(mv.sum(), 1)).tolist(),
                                            strict=True)),
            "stall_steps": int(st.sum()),
            "probe_balanced_accuracy_mean": float(np.mean(
                [e["probe_balanced_accuracy"] for e in per_seed[arm]])),
            "goal_open_stalls": pooled[arm]["goal_open_stalls"],
            "goal_open_stalls_probe_says_blocked": pooled[arm]["goal_open_stalls_probe_says_blocked"],
            # Added after the smoke test; descriptive, outside every decision.
            "detour_stalls": pooled[arm]["detour_stalls"],
            "detour_stalls_goal_out_of_view": pooled[arm]["detour_stalls_goal_out_of_view"],
            # Added after the first full run, as the base rate for the count above.
            "goal_open_moving": pooled[arm]["goal_open_moving"],
            "goal_open_moving_probe_says_blocked": pooled[arm]["goal_open_moving_probe_says_blocked"],
            # Added after the second run: the same probe on the sensor pathway alone.
            "sensor_probe_balanced_accuracy_mean": float(np.mean(
                [e["sensor_probe_balanced_accuracy"] for e in per_seed[arm]])),
            "goal_open_stalls_sensor_probe_says_blocked":
                pooled[arm]["goal_open_stalls_sensor_probe_says_blocked"],
            "goal_open_moving_sensor_probe_says_blocked":
                pooled[arm]["goal_open_moving_sensor_probe_says_blocked"],
            "timeout_rate_mean": float(np.mean([e["timeout_rate"] for e in per_seed[arm]])),
        }
    summary["stall_share_p"] = permutation_p(
        np.array([e["stall_share"] for e in per_seed["rgbi"]]),
        np.array([e["stall_share"] for e in per_seed["depthi"]]))
    summary["probe_p"] = permutation_p(
        np.array([e["probe_balanced_accuracy"] for e in per_seed["rgbi"]]),
        np.array([e["probe_balanced_accuracy"] for e in per_seed["depthi"]]))
    report["summary"] = summary

    shares = summary["rgbi"]["stall_class_shares"]
    top = max(shares, key=shares.get)
    report["part2_decision"] = ({"goal_open": "PHANTOM", "detour": "BLIND DETOUR",
                                 "boxed": "BOXED"}[top] if shares[top] > 0.5 else "MIXED")

    print("\n=== summary ===")
    for arm in ARMS:
        s = summary[arm]
        print(f"  {arm:6s} stall share {s['stall_share_mean']:.3f}  stall steps {s['stall_steps']}  "
              f"at stalls {({k: round(v, 3) for k, v in s['stall_class_shares'].items()})}  "
              f"moving {({k: round(v, 3) for k, v in s['moving_class_shares'].items()})}  "
              f"probe {s['probe_balanced_accuracy_mean']:.3f}")
    print(f"  stall share rgbi vs depthi p = {summary['stall_share_p']:.4f}; "
          f"probe p = {summary['probe_p']:.4f}")
    print(f"  rgbi goal_open stalls where the probe says blocked: "
          f"{summary['rgbi']['goal_open_stalls_probe_says_blocked']} of "
          f"{summary['rgbi']['goal_open_stalls']}")
    print(f"\nPart 2 decision (registered rule): {report['part2_decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
