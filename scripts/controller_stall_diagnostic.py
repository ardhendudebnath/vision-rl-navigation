"""What does a stalled robot do with a route it holds?

Report §9.26 narrowed the clutter stall to this. On val worlds with a margin-safe
route, the stalled robots hold a plan -- they never spend a step without one --
know their pose to five centimetres, and end a median seven metres from the
goal; removing three quarters of their plan rebuilds changed none of that. What
is left is what the controller does with the route. This records it, step by
step, for the published stack (own map, own pose, 360 beams, scan matching,
corroboration) on the same 100 val worlds each of `dense` and `narrow`.

The pure-pursuit controller does one of three things on a step it has a route:
it turns on the spot when the lookahead point is more than 0.9 rad off its
heading; otherwise it drives forward, slowed by the caution scale when anything
mapped is within 0.45 m of its body, to a floor of a quarter of its speed. Every
step is classed by what the controller commanded --

  turning    on the spot, heading error past the threshold
  crawling   forward with the caution scale under one half
  driving    forward faster than that
  recovering no route at all (the published stack turns on the spot)

-- and by what happened: how far the robot moved, how far along its route it
got, whether the route was rebuilt, and at what clearance it had been planned.

A stalled episode is read from the step after it last got 5 cm closer to the
goal by the true geodesic -- its stall phase -- to its end. Arrivals on the same
worlds are read while they are at least a metre from the goal, as the control.
The decision is whether the stall is the robot stuck, or the robot wandering:
moving at least half the arrivals' pace while getting no closer. Each step
also records whether the route held is a phantom shortcut -- at least 10%
shorter than any real path from where the robot is, which a route over seen
space cannot be -- and how far the robot ranges during its stall.

    python scripts/controller_stall_diagnostic.py
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
from route_class_test import route_classes  # noqa: E402
from stall_diagnostic import STALL_DISTANCE, make_agent  # noqa: E402

from vision_nav.agents.classical import AStarPursuitAgent  # noqa: E402
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.robot import wrap_angle  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("dense", "narrow")
CATEGORIES = ("turning", "crawling", "driving", "recovering")
#: Getting closer by less than this does not count as progress (§9.9's).
PROGRESS = 0.05
#: Under this caution scale a forward command is a crawl.
CRAWL = 0.5
#: Moving less than this in a step is standing still, in metres.
STILL = 0.005
#: A route this much shorter than the true geodesic is a phantom shortcut.
SHORTCUT = 0.9

#: Recorded and committed before this script had run on any val world.
#:
#: DISCLOSED: the code was smoke-tested on six train-band worlds per clutter
#: condition, which held one stall on a margin-safe world and three on worlds
#: with no route at the full margin. The one margin-safe stall drove for 381
#: steps after its last progress, mostly forward (65% of steps; turning 16%,
#: crawling 19%), at 0.034 m a step against the arrivals' 0.040; it drove 13.1 m
#: and ended 7.0 m from where the stall began, and on 70% of those steps its
#: route was a phantom shortcut, where the arrivals' never was. The no-route
#: stalls drove locally (2.7 m range), with no phantom shortcut and 26% of
#: steps on a plan at reduced clearance. §9.22 had already found half of every
#: route a stalling robot adopts running through unseen cells. One episode is
#: not a finding: the prediction is that it is the typical one.
PREDICTION = (
    "the stall on a margin-safe world is wandering, not being stuck: the robot "
    "drives routes that are shorter than any real path, through space it has "
    "not seen, and they do not bring it closer. Over the stall phases of the "
    "published stack's stalls on margin-safe val clutter worlds: the robot "
    "moves at least half as far per step as the arrivals do while a metre or "
    "more from the goal; driving forward is the leading command and turning on "
    "the spot is under 25% of steps; the route held is a phantom shortcut on at "
    "least 30% of steps and at least three times as often as in the arrivals; "
    "and the median stall ranges at least 2 m from where it began. The stalls "
    "on worlds with no route at the full margin range less far (median) than "
    "the margin-safe ones and hold a phantom shortcut less often. The "
    "instrumented stack reproduces §9.22's episodes exactly. Decision -- "
    "WANDERING: the margin-safe stalls move at least half the arrivals' "
    "metres per step. STUCK: less. UNDERPOWERED: fewer than 10 such stalls. "
    "Prediction: WANDERING."
)

# The controller is the base class's ``act``, reached through ``super()`` from
# the mapped agent, so it is wrapped on the class: this process only ever runs
# this diagnostic. The lookahead search is idempotent for an unchanged pose --
# its cursor already sits on the nearest point -- so reading it again after the
# controller has acted changes nothing the controller will see.
_BASE_ACT = AStarPursuitAgent.act
_LAST: dict = {}


def _recorded_act(self, pose):
    action = _BASE_ACT(self, pose)
    if self._track is not None:
        position = np.asarray(pose[:2], dtype=np.float64)
        target = self._lookahead_point(position)
        to = target - position
        _LAST.update(heading_error=float(wrap_angle(np.arctan2(to[1], to[0]) - pose[2])),
                     caution=float(self._caution_scale(position)),
                     route_left=float(self._track_length(self._track, self._cursor)),
                     radius=float(self._plan_radius))
    return action


AStarPursuitAgent.act = _recorded_act


def episode(args: tuple) -> dict:
    cond, seed, n, split = args
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split, shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution
    agent = make_agent("own_pose", cfg)
    agent.start_episode(world, env.robot.pose)
    threshold = agent.config.turn_in_place_threshold
    full = agent.map.robot_radius + agent.config.safety_margin
    steps: list[dict] = []
    info: dict = {}
    for _ in range(cfg.max_episode_steps):
        before = env.robot.pose.copy()
        replans, recovery = agent.replans, agent.recovery_steps
        _LAST.clear()
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        if agent.recovery_steps > recovery or not _LAST:
            kind = "recovering"
        elif abs(_LAST["heading_error"]) > threshold:
            kind = "turning"
        elif _LAST["caution"] < CRAWL:
            kind = "crawling"
        else:
            kind = "driving"
        step = {"kind": kind, "rebuilt": agent.replans > replans,
                "moved": float(np.linalg.norm(env.robot.pose[:2] - before[:2])),
                "turned": float(wrap_angle(env.robot.pose[2] - before[2])),
                "remaining": remaining_at(field, world, env.robot.pose[:2]),
                "inside": bool(agent.map.clearance(agent._odom.pose[:2]) < full),
                "x": float(env.robot.pose[0]), "y": float(env.robot.pose[1])}
        if _LAST:
            step.update(heading_error=_LAST["heading_error"], route_left=_LAST["route_left"],
                        reduced=bool(_LAST["radius"] < full - 1e-9))
        steps.append(step)
        if terminated or truncated:
            break
    rem = np.array([s["remaining"] for s in steps])
    success, collision = bool(info.get("is_success")), bool(info.get("collision"))
    final = float(rem[-1]) if np.isfinite(rem[-1]) else float("nan")
    stalled = not success and not collision and np.isfinite(final) and final >= STALL_DISTANCE
    # The stall phase: everything after the last step that got 5 cm closer.
    start = 0
    best = np.inf
    for i, r in enumerate(rem):
        if np.isfinite(r) and r < best - PROGRESS:
            best, start = r, i + 1
    window = (steps[start:] if stalled
              else [s for s in steps if np.isfinite(s["remaining"]) and s["remaining"] >= STALL_DISTANCE])
    return {"cond": cond, "seed": seed, "success": success, "collision": collision,
            "stalled": stalled, "steps": len(steps), "final_remaining": final,
            "replans": int(agent.replans),
            "phase_start": start if stalled else None, "summary": summarise(window, threshold)}


def summarise(window: list[dict], threshold: float) -> dict:
    """Counts over a run of steps, so groups can be pooled step-weighted."""
    n = len(window)
    out = {"n": n, **{k: sum(s["kind"] == k for s in window) for k in CATEGORIES},
           "rebuilt": sum(s["rebuilt"] for s in window),
           "still": sum(s["moved"] < STILL for s in window),
           "moved": float(sum(s["moved"] for s in window)),
           "inside": sum(s["inside"] for s in window),
           "reduced": sum(s.get("reduced", False) for s in window)}
    # A phantom shortcut: the route the robot holds is at least 10% shorter than
    # any real path from where it is -- it runs through space the map has not
    # seen and that is not free. Planned at a wider clearance than the true
    # geodesic's, a route over known space can never be shorter.
    pairs = [(s["route_left"], s["remaining"]) for s in window
             if "route_left" in s and np.isfinite(s["remaining"]) and s["remaining"] > 0]
    out["short"] = sum(r < SHORTCUT * g for r, g in pairs)
    out["short_n"] = len(pairs)
    # Where it went: path driven, net displacement, farthest from where it began.
    if n:
        xy = np.array([[s["x"], s["y"]] for s in window])
        out["net"] = float(np.linalg.norm(xy[-1] - xy[0]))
        out["range"] = float(np.linalg.norm(xy - xy[0], axis=1).max())
    else:
        out["net"] = out["range"] = 0.0
    # Turning back and forth: consecutive turning steps that turn opposite ways.
    turns = [np.sign(s["turned"]) for s in window if s["kind"] == "turning" and abs(s["turned"]) > 1e-4]
    out["turn_pairs"] = max(len(turns) - 1, 0)
    out["turn_flips"] = int(sum(a != b for a, b in zip(turns[:-1], turns[1:], strict=True)))
    # Progress along the route on steps that kept it: how much of its own route
    # the robot drove, against how much the route asked of it.
    along = [a["route_left"] - b["route_left"] for a, b in zip(window[:-1], window[1:], strict=True)
             if "route_left" in a and "route_left" in b and not b["rebuilt"]]
    out["route_progress"] = float(sum(along))
    out["route_steps"] = len(along)
    return out


def pooled(rows: list[dict]) -> dict:
    tot = {k: sum(r["summary"][k] for r in rows) for k in rows[0]["summary"]} if rows else {}
    n = tot.get("n", 0)
    if not n:
        return {"episodes": len(rows), "steps": 0}
    return {"episodes": len(rows), "steps": n,
            "shares": {k: tot[k] / n for k in CATEGORIES},
            "rebuilds_per_100": 100 * tot["rebuilt"] / n,
            "still_share": tot["still"] / n,
            "metres_per_step": tot["moved"] / n,
            "inside_share": tot["inside"] / n,
            "reduced_share": tot["reduced"] / n,
            "turn_flip_share": tot["turn_flips"] / tot["turn_pairs"] if tot["turn_pairs"] else None,
            "route_metres_per_step": (tot["route_progress"] / tot["route_steps"]
                                      if tot["route_steps"] else None),
            "phantom_shortcut_share": tot["short"] / tot["short_n"] if tot["short_n"] else None,
            "range_median": float(np.median([r["summary"]["range"] for r in rows])),
            "net_median": float(np.median([r["summary"]["net"] for r in rows])),
            "path_median": float(np.median([r["summary"]["moved"] for r in rows])),
            "phase_steps_median": float(np.median([r["summary"]["n"] for r in rows]))}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--split", default="val")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Drive only these worlds of every condition (smoke runs).")
    p.add_argument("--out", default="results/controller_stall_diagnostic.json")
    args = p.parse_args(argv)
    classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
               for w in route_classes(args.split, c, args.episodes)}
    jobs = [(c, s, args.episodes, args.split) for c in CONDITIONS
            for s in sorted(seed for cond, seed in classes if cond == c)
            if args.seeds is None or s in args.seeds]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(episode, jobs, chunksize=1))
    for r in rows:
        r["class"] = classes[(r["cond"], r["seed"])]
    safe = ("short", "long detour")
    groups = {"stall_margin_safe": [r for r in rows if r["stalled"] and r["class"] in safe],
              "stall_no_route": [r for r in rows if r["stalled"] and r["class"] == "no route"],
              "arrive_margin_safe": [r for r in rows if r["success"] and r["class"] in safe]}
    report: dict = {"prediction": PREDICTION, "groups": {k: pooled(v) for k, v in groups.items()}}
    g, a = report["groups"]["stall_margin_safe"], report["groups"]["arrive_margin_safe"]
    if g["episodes"] < 10:
        decision = "UNDERPOWERED"
    else:
        # Stuck or wandering: does the stalled robot go anywhere, by the pace of
        # the robots that arrive on the same worlds?
        decision = "WANDERING" if g["metres_per_step"] >= 0.5 * a["metres_per_step"] else "STUCK"
        report["pace_ratio"] = g["metres_per_step"] / a["metres_per_step"]
        report["leading_command"] = max(CATEGORIES, key=lambda k: g["shares"][k])
    report["decision"] = decision
    stall = Path("results/stall_diagnostic.json")
    if args.split == "val" and args.episodes == 100 and args.seeds is None and stall.exists():
        # Identity control: instrumented, this is the stack §9.22 drove on the
        # same worlds, and it must reproduce every outcome and replan count.
        before = {(r["cond"], r["seed"]): r for r in json.loads(stall.read_text(encoding="utf-8"))
                  ["episodes_detail"] if r["arm"] == "own_pose"}
        report["reproduces_stall_diagnostic"] = len(rows) == len(before) and all(
            all(r[f] == before[(r["cond"], r["seed"])][f] for f in ("success", "collision", "replans", "steps"))
            for r in rows)
        print(f"reproduces §9.22: {report['reproduces_stall_diagnostic']}")
    report["episodes_detail"] = rows

    for name, s in report["groups"].items():
        if not s.get("steps"):
            print(f"{name}: no steps")
            continue
        print(f"{name:20s} {s['episodes']:3d} eps {s['steps']:6d} steps  "
              + "  ".join(f"{k} {v:.0%}" for k, v in s["shares"].items())
              + f"  | still {s['still_share']:.0%}  {s['metres_per_step']:.3f} m/step  "
              f"route {s['route_metres_per_step']}  rebuilds/100 {s['rebuilds_per_100']:.1f}  "
              f"inside {s['inside_share']:.0%}  reduced radius {s['reduced_share']:.0%}  "
              f"turn flips {s['turn_flip_share']}  phantom {s['phantom_shortcut_share']}  "
              f"path/net/range median {s['path_median']:.1f}/{s['net_median']:.1f}/{s['range_median']:.1f} m "
              f"over {s['phase_steps_median']:.0f} steps")
    print(f"decision {decision}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
