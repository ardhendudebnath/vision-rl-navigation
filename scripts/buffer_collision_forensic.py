"""Where the plan buffer collides: §9.33's three collisions, up close (unregistered).

Report §9.33 found the buffered arm colliding three times on held-out worlds,
all on worlds with no margin-safe route, where the footprint arm collided
nowhere. On such a world the buffered rung can have no route by the world's
geometry -- but the robot plans on its own map, with unknown space free, so
early on a wide route through the unseen can exist. This re-drives every
clutter world with no margin-safe route, held-out and val, in both arms, and
logs every plan (which rung of the ladder) and every step (true clearance,
pose error, speed, the radius the plan in force was made at). Every run must
reproduce its stored episode exactly.

Unregistered: every outcome here is already published (§9.32, §9.33), and this
measures only how they came about.

    python scripts/buffer_collision_forensic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clutter_forensic import remaining_at  # noqa: E402
from plan_buffer_test import BUFFER, CLUTTER  # noqa: E402
from stall_diagnostic import make_agent  # noqa: E402
from transfer_test import mcnemar  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

ARMS = ("footprint", "buffered")
SOURCES = {"held-out": "results/plan_buffer_heldout_test.json", "val": "results/plan_buffer_test.json"}
#: True clearance under this, in metres, counts as a near-contact.
NEAR = 0.01
#: Steps of the trace kept before each collision.
TAIL = 12


def drive(cond: str, seed: int, arm: str, split: str) -> dict:
    """``plan_buffer_test.episode``, step for step, with the plans and steps logged."""
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split, shift=shift, n_worlds=100)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution
    agent = make_agent("own_pose", cfg)
    agent.footprint_check = True
    agent.plan_buffer = BUFFER if arm == "buffered" else 0.0
    base, margin = world.config.robot_radius, agent.config.safety_margin
    names = {round(base + margin, 6): "margin", round(base + margin / 2, 6): "half", round(base, 6): "bare"}
    plans: list[dict] = []
    plan = agent.reset

    def logged(w, pose, *a, **k):
        before = agent.buffered_plans
        ok = plan(w, pose, *a, **k)
        rung = ("buffered" if agent.buffered_plans > before
                else names.get(round(agent._plan_radius, 6), "?")) if ok else "none"
        plans.append({"step": agent._steps, "rung": rung})
        return ok

    agent.reset = logged
    agent.start_episode(world, env.robot.pose)
    steps, info = [], {}
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        pos = env.robot.pose[:2]
        steps.append({"clear": float(world.clearance(pos[None])[0]) - base,
                      "pose_err": float(agent.pose_errors[-1]),
                      "speed": float(env.robot.velocity[0]),
                      "bare": bool(agent._track is not None and agent._plan_radius <= base + 1e-9),
                      "recovering": agent._track is None,
                      "remaining": float(remaining_at(field, world, pos))})
        if terminated or truncated:
            break
    return {"success": bool(info.get("is_success")), "collision": bool(info.get("collision")),
            "steps": steps, "plans": plans}


def world_pair(args: tuple) -> dict:
    """Both arms on one world, and where they part."""
    cond, seed, split = args
    runs = {arm: drive(cond, seed, arm, split) for arm in ARMS}
    f, b = runs["footprint"]["steps"], runs["buffered"]["steps"]
    # The two episodes end at different steps; compare only the steps both drove.
    parted = next((i + 1 for i, (x, y) in enumerate(zip(f, b, strict=False))
                   if abs(x["remaining"] - y["remaining"]) > 1e-9), None)
    first_buffered = next((p["step"] for p in runs["buffered"]["plans"] if p["rung"] == "buffered"), None)
    out = {"cond": cond, "seed": seed, "parted_at": parted, "first_buffered_plan": first_buffered}
    for arm, r in runs.items():
        st = r["steps"]
        out[arm] = {"success": r["success"], "collision": r["collision"], "steps": len(st),
                    "bare_steps": sum(s["bare"] for s in st),
                    "recovery_steps": sum(s["recovering"] for s in st),
                    "near_contact_steps": sum(s["clear"] < NEAR for s in st),
                    "min_clearance": min(s["clear"] for s in st),
                    "buffered_plans": sum(p["rung"] == "buffered" for p in r["plans"])}
        if r["collision"]:
            out[arm]["tail"] = st[-TAIL:]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--out", default="results/buffer_collision_forensic.json")
    args = p.parse_args(argv)

    report: dict = {"unregistered": True, "near_contact_m": NEAR, "splits": {}, "collisions": []}
    pooled = {arm: [] for arm in ARMS}
    for split, source in SOURCES.items():
        detail = json.loads(Path(source).read_text(encoding="utf-8"))["episodes_detail"]
        stored = {(e["arm"], e["cond"], e["seed"]): e for e in detail}
        # Every collision either test recorded, in any arm, and the route class of its world.
        report["collisions"] += [{"split": split, "arm": e["arm"], "cond": e["cond"], "seed": e["seed"],
                                  "class": e["class"], "step": e["steps"]} for e in detail if e["collision"]]
        worlds = sorted({(e["cond"], e["seed"]) for e in detail
                         if e["cond"] in CLUTTER and e["class"] == "no route"})
        run_split = (lambda c: "val") if split == "val" else (lambda c: BENCHMARK_CONDITIONS[c][0])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(world_pair, [(c, s, run_split(c)) for c, s in worlds], chunksize=1))
        reproduces = all((r[arm]["success"], r[arm]["collision"], r[arm]["steps"])
                         == tuple(stored[(arm, r["cond"], r["seed"])][k] for k in ("success", "collision", "steps"))
                         for r in rows for arm in ARMS)
        totals = {arm: {"worlds": len(rows),
                        "collisions": sum(r[arm]["collision"] for r in rows),
                        "arrivals": sum(r[arm]["success"] for r in rows),
                        "bare_steps": sum(r[arm]["bare_steps"] for r in rows),
                        "recovery_steps": sum(r[arm]["recovery_steps"] for r in rows),
                        "near_contact_worlds": sum(r[arm]["near_contact_steps"] > 0 for r in rows),
                        "worlds_with_a_buffered_plan": sum(r[arm]["buffered_plans"] > 0 for r in rows)}
                  for arm in ARMS}
        report["splits"][split] = {"reproduces": reproduces, "totals": totals, "worlds": rows}
        for arm in ARMS:
            pooled[arm] += [r[arm]["collision"] for r in rows]
        print(f"{split}: {len(rows)} worlds, reproduces {reproduces}", flush=True)
        for arm, t in totals.items():
            print(f"  {arm:9s} {json.dumps(t)}")

    pv, only_buffered, only_footprint = mcnemar(np.array(pooled["footprint"]), np.array(pooled["buffered"]))
    report["pooled"] = {"worlds": len(pooled["footprint"]), "only_buffered_collides": only_buffered,
                        "only_footprint_collides": only_footprint, "p": pv}
    print(f"pooled no-route worlds {len(pooled['footprint'])}: buffered alone collides on {only_buffered}, "
          f"footprint alone on {only_footprint}, McNemar p {pv:.3f}")
    for c in report["collisions"]:
        print(f"collision: {c}")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
