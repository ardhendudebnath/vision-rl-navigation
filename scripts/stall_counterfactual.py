"""Does the RGB policy's own velocity input hold it stopped? A replay, no training.

Phase 5v: at the RGB policy's stalls in front of an open route to a visible goal
-- a stall the depth policy never makes -- its CNN features read the route as
open *more* reliably than while it is moving, and its final layer reads it as
blocked. That layer also takes the robot's own velocity, near zero at any stall.
The candidate is a latch: stopped, the velocity input says stopped, and the
policy stays stopped.

Two tests on the twelve Phase 5s policies over the same 50 `narrow` worlds.

**Open loop.** At every step the policy's action is also computed with its
velocity input replaced: by 0.3 m/s (half its top speed) and by zero. Nothing
else in the observation changes, and the episode runs on the real action. The
policy *goes* when it commands at least 0.15 m/s forward.

  exit:  at open-route stall steps, how much more often it goes when told it
         is moving
  entry: at open-route moving steps, how much less often it goes when told it
         is stopped

**Closed loop.** Each episode is run again, replacing the velocity input by
0.3 m/s whenever the robot has gone nowhere for the last 3 s (a detector that
looks only backwards). Does that turn timeouts into successes?

    python scripts/stall_counterfactual.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from vision_nav.analysis.pixel_audit import (
    STALL_CLASSES,
    stall_geometry,
    stall_steps,
    stalled_now,
)

SEEDS = range(6)
ARMS = ("depthi", "rgbi")
V_MOVE = 0.3  # m/s, half the robot's top speed
GO = 0.15  # m/s commanded forward speed that counts as going

#: Recorded before the replay was run, and committed to the repository before
#: any result existed. The smoke test printed only tracebacks.
#:
#: Confidence LOW. For: Phase 5v found the final layer's reading of "route
#: open" flipping towards blocked exactly where velocity is zero, while the
#: image features alone did the opposite -- the latch is the simplest thing that
#: produces both. Against: a policy can use velocity sensibly (a robot that is
#: stopped should turn before it drives), so the input may shape turning rather
#: than going; and the 3 s the closed-loop detector needs before it acts may be
#: most of a stall already spent.
PREDICTION = (
    "LATCH: pooled over rgbi's open-route steps, telling a stalled policy it is "
    "moving raises the share of steps it goes by at least 0.30, and telling a "
    "moving policy it is stopped lowers that share by at least 0.30. Closed loop: "
    "overriding the velocity input while stalled lowers rgbi's mean timeout rate "
    "by at least 0.05. "
    "Decision -- LATCH: both open-loop shifts >= 0.30. NO LATCH: both < 0.10. "
    "Otherwise PARTIAL."
)


def velocity_slot(obs, cfg):
    """(container, index) of the normalised linear-velocity input."""
    if cfg.obs_mode == "rgb":
        return "vector", 3
    assert cfg.frame_stack == 1 and not cfg.obs_velocity, "flat layout assumed"
    return None, cfg.camera.width + 3


def with_velocity(obs, cfg, v_mps):
    key, idx = velocity_slot(obs, cfg)
    value = np.float32(v_mps / cfg.robot.max_linear_vel)
    if key is None:
        out = obs.copy()
        out[idx] = value
        return out
    out = {k: v.copy() for k, v in obs.items()}
    out[key][idx] = value
    return out


def replay(job):
    arm, seed, n_worlds, override = job
    import torch

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
    model = actor.model
    fov = cfg.camera.fov if cfg.obs_mode == "depth" else cfg.rgb_camera.fov
    columns = cfg.camera.width if cfg.obs_mode == "depth" else cfg.rgb_camera.width

    def goes(o):
        action, _ = model.predict(o, deterministic=True)
        return bool(env.robot.scale_action(np.asarray(action).reshape(2))[0] >= GO)

    classes, stalled, go_real, go_move, go_zero, outcomes, overridden = [], [], [], [], [], [], 0
    for world_seed in cfg.world_seeds:
        obs, _ = env.reset(options={"world_seed": int(world_seed)})
        actor.reset(env, obs)
        positions = []
        while True:
            pose = env.robot.pose.copy()
            positions.append(pose[:2])
            fed = obs
            if override and stalled_now(np.asarray(positions), cfg.robot.dt):
                fed = with_velocity(obs, cfg, V_MOVE)
                overridden += 1
            if not override:
                classes.append(STALL_CLASSES.index(stall_geometry(env.world, pose, fov, columns)))
                go_real.append(goes(obs))
                go_move.append(goes(with_velocity(obs, cfg, V_MOVE)))
                go_zero.append(goes(with_velocity(obs, cfg, 0.0)))
            obs, _, term, trunc, info = env.step(actor.act(env, fed))
            if term or trunc:
                break
        if not override:
            stalled.extend(stall_steps(np.asarray(positions), cfg.robot.dt))
        outcomes.append("success" if info.get("is_success") else
                        "collision" if info.get("collision") else "timeout")
    return {"arm": arm, "seed": seed, "override": override, "outcomes": outcomes,
            "overridden_steps": overridden, "classes": np.asarray(classes),
            "stalled": np.asarray(stalled), "go_real": np.asarray(go_real),
            "go_move": np.asarray(go_move), "go_zero": np.asarray(go_zero)}


def sign_flip_p(diffs) -> float:
    """Exact two-sided paired test over seeds: every sign assignment."""
    diffs = np.asarray(diffs, dtype=float)
    observed = abs(diffs.mean())
    hits = total = 0
    for signs in product((1, -1), repeat=len(diffs)):
        total += 1
        hits += abs((diffs * signs).mean()) >= observed - 1e-12
    return hits / total


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default="results/stall_counterfactual.json")
    args = p.parse_args(argv)

    jobs = [(arm, s, args.worlds, override) for override in (False, True)
            for arm in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(replay, jobs))
    base = {(r["arm"], r["seed"]): r for r in results if not r["override"]}
    over = {(r["arm"], r["seed"]): r for r in results if r["override"]}

    def rate(outcomes, kind):
        return outcomes.count(kind) / len(outcomes)

    report = {"prediction": PREDICTION, "worlds": args.worlds, "v_move": V_MOVE, "go": GO,
              "arms": {}}
    for arm in ARMS:
        pooled = {k: 0 for k in ("open_stall", "open_stall_go_real", "open_stall_go_move",
                                 "open_moving", "open_moving_go_real", "open_moving_go_zero",
                                 "detour_stall", "detour_stall_go_real", "detour_stall_go_move")}
        seeds = []
        for s in SEEDS:
            r = base[(arm, s)]
            open_stall = r["stalled"] & (r["classes"] == 0)
            open_moving = ~r["stalled"] & (r["classes"] == 0)
            detour_stall = r["stalled"] & (r["classes"] == 1)
            for name, mask in (("open_stall", open_stall), ("open_moving", open_moving),
                               ("detour_stall", detour_stall)):
                pooled[name] += int(mask.sum())
            pooled["open_stall_go_real"] += int((open_stall & r["go_real"]).sum())
            pooled["open_stall_go_move"] += int((open_stall & r["go_move"]).sum())
            pooled["open_moving_go_real"] += int((open_moving & r["go_real"]).sum())
            pooled["open_moving_go_zero"] += int((open_moving & r["go_zero"]).sum())
            pooled["detour_stall_go_real"] += int((detour_stall & r["go_real"]).sum())
            pooled["detour_stall_go_move"] += int((detour_stall & r["go_move"]).sum())
            o = over[(arm, s)]
            seeds.append({"seed": s,
                          "timeout": rate(r["outcomes"], "timeout"),
                          "timeout_override": rate(o["outcomes"], "timeout"),
                          "success": rate(r["outcomes"], "success"),
                          "success_override": rate(o["outcomes"], "success"),
                          "collision": rate(r["outcomes"], "collision"),
                          "collision_override": rate(o["outcomes"], "collision"),
                          "overridden_steps": o["overridden_steps"]})

        def share(num, den, pooled=pooled):
            return pooled[num] / pooled[den] if pooled[den] else None

        entry = {"pooled": pooled, "per_seed": seeds,
                 "open_stall_go_real": share("open_stall_go_real", "open_stall"),
                 "open_stall_go_move": share("open_stall_go_move", "open_stall"),
                 "open_moving_go_real": share("open_moving_go_real", "open_moving"),
                 "open_moving_go_zero": share("open_moving_go_zero", "open_moving"),
                 "detour_stall_go_real": share("detour_stall_go_real", "detour_stall"),
                 "detour_stall_go_move": share("detour_stall_go_move", "detour_stall")}
        for key in ("timeout", "success", "collision"):
            diffs = [e[f"{key}_override"] - e[key] for e in seeds]
            entry[f"{key}_change"] = float(np.mean(diffs))
            entry[f"{key}_change_p"] = sign_flip_p(diffs)
        report["arms"][arm] = entry
        print(f"{arm}: open-route stalls {pooled['open_stall']}, moving {pooled['open_moving']}")

    rg = report["arms"]["rgbi"]
    exit_shift = (rg["open_stall_go_move"] - rg["open_stall_go_real"]
                  if rg["open_stall_go_real"] is not None else None)
    entry_shift = rg["open_moving_go_real"] - rg["open_moving_go_zero"]
    report["rgbi_exit_shift"], report["rgbi_entry_shift"] = exit_shift, entry_shift
    if exit_shift is not None and exit_shift >= 0.30 and entry_shift >= 0.30:
        decision = "LATCH"
    elif exit_shift is not None and exit_shift < 0.10 and entry_shift < 0.10:
        decision = "NO LATCH"
    else:
        decision = "PARTIAL"
    report["decision"] = decision

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"decision (registered rule): {decision}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
