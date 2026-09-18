"""Released from its latch, where does the pixel policy crash -- and what does
it fail to see?

Phase 5x released the RGB policy's velocity latch whenever it had gone nowhere
for three seconds. Timeouts fell on every seed and collisions rose by nearly as
much: the stalls became crashes. Two questions, neither needing training.

**Where.** For every episode that timed out untouched and collided once
released, how far from the place it had stalled does it crash, and how long
after the release? Crashing at the opening it had been staring at means the
policy cannot fit through what it can see; crashing far away means the release
merely returned it to ordinary driving, where it was always going to fail.

**What it cannot see.** A linear probe for the geometry *around* an opening
rather than its presence: the angular width of the traversable gap containing
the goal bearing, and the clearance half a metre ahead. Read off the CNN's image
features for the RGB policies and off the depth vector for the depth ones -- the
same comparison Phase 5v drew for "is the goal-ward route open", now for how
much room there is.

    python scripts/stall_crash_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from vision_nav.analysis.perception import find_gaps, traversable_mask
from vision_nav.analysis.pixel_audit import stalled_now

SEEDS = range(6)
ARMS = ("depthi", "rgbi")
V_MOVE = 0.3  # the release, as in Phase 5x
PROBE_ANGLES = 180
NEAR_M = 1.0  # "at the opening it had stopped in front of"
SOON_S = 3.0

#: Recorded before the run and committed before any result exists.
#:
#: Confidence LOW on the first question, MODERATE on the second. For crashing at
#: the opening: Phase 5v found these stalls happening with an open route in
#: view and the CNN's features encoding it as open, so the policy has the
#: opening and lacks something else -- most likely the margins either side,
#: which is what a 0.22 m disc in `narrow` needs. Against: a released policy
#: drives on for as long as the episode lasts, and `narrow` is where every arm
#: crashes, so the added collisions may be ordinary ones that the stall had
#: merely postponed.
#:
#: The probe's direction is firmer. Phase 5v already measured the CNN's features
#: carrying "the goal-ward route is open" less well than the depth policy's
#: layer carried it (0.813 against 0.902 at the actor layer), and width is a
#: harder quantity than presence: it needs both edges of a gap located, which is
#: what a stride-4 encoder on a 64-pixel-wide image is least able to do.
PREDICTION = (
    "AT THE OPENING: over half of the added collisions -- episodes that timed "
    "out untouched and collided once released -- happen within 1.0 m of where "
    "the robot had stalled and within 3.0 s of the release. And the RGB features "
    "decode the gap's angular width worse than the depth vector does, by at "
    "least 0.10 of held-out R-squared. "
    "Decision on the first -- AT THE OPENING: over half within both bounds. "
    "ELSEWHERE: under a quarter. Otherwise MIXED."
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


def geometry(world, pose, probe=1.0):
    """(width of the traversable gap holding the goal bearing, clearance ahead)."""
    position, heading = np.asarray(pose[:2], dtype=float), float(pose[2])
    angles = np.linspace(-np.pi, np.pi, PROBE_ANGLES, endpoint=False)
    mask = traversable_mask(world, position, angles, probe)
    to_goal = world.goal - position
    goal_idx = int(np.round(((np.arctan2(to_goal[1], to_goal[0]) + np.pi)
                             / (2 * np.pi)) * PROBE_ANGLES)) % PROBE_ANGLES
    width = 0.0
    for start, end in find_gaps(mask):
        span = (end - start) % PROBE_ANGLES + 1
        inside = (start <= goal_idx <= end) if start <= end else (goal_idx >= start
                                                                  or goal_idx <= end)
        if inside:
            width = span * 2 * np.pi / PROBE_ANGLES
            break
    ahead = position + 0.5 * np.array([np.cos(heading), np.sin(heading)])
    clearance = float(world.clearance(ahead[None, :])[0]) - world.config.robot_radius
    return width, clearance


def drive(job):
    arm, seed, n_worlds, override = job
    import torch

    from vision_nav.envs.nav_env import ProceduralNavEnv
    from vision_nav.training.actors import build_actor
    from vision_nav.training.env_factory import build_env_config
    from vision_nav.training.run_spec import env_overrides_for_run

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stall_counterfactual import with_velocity

    torch.set_num_threads(1)
    run = Path("runs") / f"{arm}_s{seed}"
    cfg = build_env_config(env_overrides_for_run(run), split="test_ood", shift="narrow",
                           n_worlds=n_worlds)
    env = ProceduralNavEnv(cfg)
    actor = build_actor("rl", model_path=str(run / "best_model.zip"), robot=cfg.robot)
    policy = actor.model.policy
    columns = cfg.camera.width if cfg.obs_mode == "depth" else cfg.rgb_camera.width

    episodes, features, targets, groups = [], [], [], []
    for ep, world_seed in enumerate(cfg.world_seeds):
        obs, _ = env.reset(options={"world_seed": int(world_seed)})
        actor.reset(env, obs)
        positions, releases, stalled_prev = [], [], False
        while True:
            pose = env.robot.pose.copy()
            positions.append(pose[:2].copy())
            fed = obs
            now_stalled = override and stalled_now(np.asarray(positions), cfg.robot.dt)
            if now_stalled:
                fed = with_velocity(obs, cfg, V_MOVE)
                if not stalled_prev:
                    releases.append((len(positions) - 1, pose[:2].copy()))
            stalled_prev = bool(now_stalled)
            if not override:  # the probe reads undisturbed driving
                with torch.no_grad():
                    obs_t, _ = policy.obs_to_tensor(obs)
                    if cfg.obs_mode == "rgb":
                        from stable_baselines3.common.preprocessing import preprocess_obs
                        image = preprocess_obs(obs_t, policy.observation_space,
                                               normalize_images=policy.normalize_images)["image"]
                        feat = policy.features_extractor.extractors["image"](image)[0].numpy()
                    else:
                        feat = np.asarray(obs[:columns], dtype=np.float32)
                features.append(feat)
                targets.append(geometry(env.world, pose))
                groups.append(ep)
            obs, _, term, trunc, info = env.step(actor.act(env, fed))
            if term or trunc:
                break
        collided = bool(info.get("collision"))
        record = {"episode": ep, "collision": collided,
                  "success": bool(info.get("is_success")),
                  "timeout": not collided and not info.get("is_success"),
                  "releases": len(releases)}
        if collided and releases:
            step, where = releases[-1]
            record["metres_from_stall"] = float(np.linalg.norm(positions[-1] - where))
            record["seconds_since_release"] = float((len(positions) - 1 - step) * cfg.robot.dt)
        episodes.append(record)
    return {"arm": arm, "seed": seed, "override": override, "episodes": episodes,
            "features": np.asarray(features, dtype=np.float32),
            "targets": np.asarray(targets, dtype=np.float64), "groups": np.asarray(groups)}


def probe_r2(features, target, groups, folds=5, ridge=1.0):
    """Held-out R^2 of a linear probe, folds split by episode."""
    x = (features - features.mean(0)) / (features.std(0) + 1e-6)
    x = np.hstack([x, np.ones((len(x), 1))])
    pred = np.zeros(len(target))
    fold_of = groups % folds
    for f in range(folds):
        tr, te = fold_of != f, fold_of == f
        w = np.linalg.solve(x[tr].T @ x[tr] + ridge * np.eye(x.shape[1]), x[tr].T @ target[tr])
        pred[te] = x[te] @ w
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="results/stall_crash_audit.json")
    args = p.parse_args(argv)

    jobs = [("rgbi", s, args.worlds, True) for s in SEEDS]
    jobs += [(arm, s, args.worlds, False) for arm in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(drive, jobs))
    base = {(r["arm"], r["seed"]): r for r in results if not r["override"]}
    released = {r["seed"]: r for r in results if r["override"]}

    report = {"prediction": PREDICTION, "worlds": args.worlds, "near_m": NEAR_M,
              "soon_s": SOON_S, "per_seed": {}, "probe": {}}

    added, at_opening, elsewhere, distances, delays = 0, 0, 0, [], []
    for s in SEEDS:
        quiet, loud = base[("rgbi", s)]["episodes"], released[s]["episodes"]
        for q, r in zip(quiet, loud, strict=True):
            if q["timeout"] and r["collision"] and "metres_from_stall" in r:
                added += 1
                distances.append(r["metres_from_stall"])
                delays.append(r["seconds_since_release"])
                near = (r["metres_from_stall"] <= NEAR_M
                        and r["seconds_since_release"] <= SOON_S)
                at_opening += near
                elsewhere += not near
    report["added_collisions"] = {
        "n": added, "at_opening": at_opening, "elsewhere": elsewhere,
        "share_at_opening": at_opening / added if added else None,
        "median_metres_from_stall": float(np.median(distances)) if distances else None,
        "median_seconds_since_release": float(np.median(delays)) if delays else None,
    }
    print(f"added collisions {added}: {at_opening} at the opening, {elsewhere} elsewhere; "
          f"median {report['added_collisions']['median_metres_from_stall']} m, "
          f"{report['added_collisions']['median_seconds_since_release']} s", flush=True)

    for name, idx in (("gap_width_rad", 0), ("clearance_ahead_m", 1)):
        per_arm = {}
        for arm in ARMS:
            r2 = [probe_r2(base[(arm, s)]["features"], base[(arm, s)]["targets"][:, idx],
                           base[(arm, s)]["groups"]) for s in SEEDS]
            per_arm[arm] = {"per_seed": r2, "mean": float(np.mean(r2))}
        per_arm["p"] = permutation_p(np.array(per_arm["rgbi"]["per_seed"]),
                                     np.array(per_arm["depthi"]["per_seed"]))
        per_arm["gap"] = per_arm["depthi"]["mean"] - per_arm["rgbi"]["mean"]
        report["probe"][name] = per_arm
        print(f"probe {name}: rgbi {per_arm['rgbi']['mean']:.3f}  "
              f"depthi {per_arm['depthi']['mean']:.3f}  gap {per_arm['gap']:+.3f}  "
              f"p {per_arm['p']:.4f}", flush=True)

    share = report["added_collisions"]["share_at_opening"]
    report["decision"] = ("AT THE OPENING" if share is not None and share > 0.5
                          else "ELSEWHERE" if share is not None and share < 0.25 else "MIXED")
    report["per_seed"] = {f"{arm}_s{s}": base[(arm, s)]["episodes"] for arm in ARMS for s in SEEDS}
    report["per_seed"].update({f"rgbi_s{s}_released": released[s]["episodes"] for s in SEEDS})
    print(f"\ndecision (registered rule): {report['decision']}")
    print("pre-registered: " + PREDICTION)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
