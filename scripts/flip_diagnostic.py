"""What forces a stalled robot's route to flip to the other side?

Report §9.30 found the hovering stall to be a robot flip-flopping between two
ways round an obstacle: its route changes sides 1.6 times per 100 steps and it
turns back when it does. A plan is rebuilt only when the map rules it out, so
each flip follows something the robot saw. This catches every flip -- a rebuild
whose new route's first metre points more than 90 degrees from the old one's --
and records, before the old route is discarded:

  what blocked it  the mapped cells within the plan radius of its first blocked
                   point, by their state when the plan was made: unknown then
                   (discovery), free then (newly seen or flicker), or occupied
                   already (pre-existing, §9.22's footprint fault); or nothing
                   at all (a deferred rebuild whose blockage had gone)
  where            how far along the old route, and from the robot
  the rung         whether the new route was planned at a different clearance

  the truth       at the blocked point, the true clearance against the plan's:
                   whether the world left the plan its room there and only the
                   map did not, and whether the blocking cell is a real surface

and then follows the abandoned route for 150 steps -- past the usual interval
between flips -- re-judging it as the agent judges a plan: does it become valid
again, and does the robot's next flip take it back to that side?

The planner treats inflation as blocked-or-free, so its routes run along the
edge of the margin, where one newly mapped cell can rule them out though the
world has room. The decision is which of two kinds of flip dominates: EDGE,
where the world left the plan its clearance at the blocked point (a route on
the margin's edge knocked off by the map), or BLOCKED, where it did not.

Two arms on the 169 margin-safe val clutter worlds: the published stack, and
the same with the footprint-consistent check (§9.26), which is what a hysteresis
repair would run with.

    python scripts/flip_diagnostic.py
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

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.robot import wrap_angle  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.mapping.occupancy import FREE, OCCUPIED, UNKNOWN  # noqa: E402
from vision_nav.planning.grid_astar import geodesic_distance_field  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

CONDITIONS = ("dense", "narrow")
ARMS = ("published", "repaired")
#: How long an abandoned route is followed to see whether it reopens: past the
#: median interval between flips (about 60 steps, §9.30).
FOLLOW = 150
TRIGGERS = ("pre-existing", "discovery", "newly seen", "flicker", "none")

#: Recorded and committed before this script had run on any val world.
#:
#: DISCLOSED: smoke-tested on the margin-safe worlds among the first twelve
#: train-band worlds per clutter condition: five stalls per arm. With the
#: footprint-consistent check, 18 flips (0.72 per 100 steps, against 0.80
#: published): triggered by newly seen cells (61%) and flicker (33%), blocked
#: a median 0.8 m from the robot; at the blocked point the world left the plan
#: its clearance in 61% of them (median slack +0.042 m); 28% of abandoned sides
#: reopened within 150 steps. §9.30's flip and reversal counts are published.
PREDICTION = (
    "the flips are a route on the margin's edge knocked off by the map, not "
    "the way round being shut. On the margin-safe val clutter worlds, in the "
    "stalls of the stack with the footprint-consistent check: the robot flips "
    "at least half as often per step as the published stack's stalls do; "
    "newly seen cells are the leading trigger and pre-existing ones under 10%; "
    "at the blocked point the world leaves the plan its clearance on at least "
    "half of the flips, with the median slack between 0 and +0.10 m; and fewer "
    "than half of the abandoned sides reopen within 150 steps. In both arms the "
    "arrivals flip at most a fifth as often per step as the stalls. Both arms "
    "reproduce §9.26's episodes exactly. Decision on the repaired arm's stalls "
    "-- EDGE: the world leaves the plan its clearance at the blocked point on "
    "at least half of the flips. BLOCKED: on fewer. UNDERPOWERED: fewer than 20 "
    "flips. Prediction: EDGE."
)


def bearing(track: np.ndarray, position: np.ndarray) -> float | None:
    """Bearing from ``position`` to the point about a metre along ``track``."""
    if track is None or len(track) < 2:
        return None
    d = np.linalg.norm(track - position, axis=1)
    i = int(np.argmax(d >= 1.0)) if (d >= 1.0).any() else len(track) - 1
    v = track[i] - position
    return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else None


def invalid_points(agent, track: np.ndarray, radius: float, position: np.ndarray) -> np.ndarray:
    """Which points of ``track`` the agent's own check would hold against it."""
    bad = agent.map.clearance(track) < radius - 1e-9
    if agent.footprint_check:
        reach = agent.map.robot_radius + 2.0 * agent.map.half_diagonal
        bad &= np.linalg.norm(track - position, axis=1) > reach
    return bad


class Watcher:
    """Catches each rebuild before the old plan is discarded, and each flip."""

    def __init__(self, agent, world) -> None:
        self.agent, self.world = agent, world
        self.flips: list[dict] = []
        self.rebuilds = 0
        self.grid_at_plan: np.ndarray | None = None
        self.scan_at_plan = 0
        self.following: list[dict] = []
        self._pending: dict | None = None
        reset, blocked = agent.reset, agent._path_ahead_blocked

        def watched_blocked(position):
            fired = blocked(position)
            if fired and agent._track is not None:
                self._pending = self.cause(np.asarray(position, dtype=float))
            return fired

        def watched_reset(world, pose, *a, **k):
            old = None if agent._track is None else agent._track[agent._cursor:].copy()
            old_radius = agent._plan_radius
            position = np.asarray(agent._odom.pose[:2], dtype=float)
            ok = reset(world, pose, *a, **k)
            if ok and old is not None:
                self.rebuilds += 1
                b0, b1 = bearing(old, position), bearing(agent._track, position)
                if b0 is not None and b1 is not None and abs(wrap_angle(b1 - b0)) > np.pi / 2:
                    flip = {"step": agent._steps, "old_bearing": b0, "new_bearing": b1,
                            "old_radius": float(old_radius),
                            "new_radius": float(agent._plan_radius),
                            "old_length": float(len(old) * agent.config.track_spacing),
                            "new_length": float(len(agent._track) * agent.config.track_spacing),
                            **(self._pending or {"trigger": "none"}), "reopened_after": None}
                    self.flips.append(flip)
                    self.following.append({"flip": flip, "track": old, "radius": float(old_radius),
                                           "until": agent._steps + FOLLOW})
            if ok:
                self.grid_at_plan = agent.map.grid.copy()
                self.scan_at_plan = agent.map.scans
            self._pending = None
            return ok

        agent._path_ahead_blocked = watched_blocked
        agent.reset = watched_reset

    def cause(self, position: np.ndarray) -> dict:
        """What made the plan in force invalid, read before it is discarded."""
        agent, omap = self.agent, self.agent.map
        ahead = agent._track[agent._cursor:]
        radius = float(agent._plan_radius)
        bad = np.flatnonzero(invalid_points(agent, ahead, radius, position))
        if not len(bad) or self.grid_at_plan is None:
            return {"trigger": "none"}
        first = ahead[bad[0]]
        # The truth at the blocked point, read for this measurement only: did
        # the world itself leave the plan's clearance there, or only the map
        # not? (The map places a surface anywhere in its cell, so it never
        # reports more room than there is.)
        slack = float(self.world.clearance(first, include_dynamic=False)) - radius
        occ = np.argwhere(omap.grid == OCCUPIED)
        centres = omap.grid_to_world(occ)
        near = np.linalg.norm(centres - first, axis=1) <= radius + omap.half_diagonal + 1e-9
        out = {"along": float(bad[0] * agent.config.track_spacing),
               "from_robot": float(np.linalg.norm(first - position)), "true_slack": slack}
        if not near.any():
            return {**out, "trigger": "none"}
        cells = occ[near]
        before = self.grid_at_plan[cells[:, 0], cells[:, 1]]
        seen = ((omap.first_occupied[cells[:, 0], cells[:, 1]] >= 0)
                & (omap.first_occupied[cells[:, 0], cells[:, 1]] <= self.scan_at_plan))
        kinds = np.where(before == OCCUPIED, "pre-existing",
                         np.where(before == UNKNOWN, "discovery",
                                  np.where((before == FREE) & seen, "flicker", "newly seen")))
        nearest = int(np.argmin(np.linalg.norm(centres[near] - first, axis=1)))
        truth = float(self.world.clearance(centres[near][nearest], include_dynamic=False))
        return {**out, "trigger": str(kinds[nearest]), "real": bool(truth <= omap.half_diagonal + 1e-9)}

    def follow(self) -> None:
        """Re-judge each abandoned route still being followed, as the agent would."""
        agent = self.agent
        position = np.asarray(agent._odom.pose[:2], dtype=float)
        keep = []
        for f in self.following:
            if not invalid_points(agent, f["track"], f["radius"], position).any():
                f["flip"]["reopened_after"] = agent._steps - f["flip"]["step"]
            elif agent._steps < f["until"]:
                keep.append(f)
        self.following = keep


def episode(args: tuple) -> dict:
    cond, arm, seed, n, split = args
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split=split, shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    goal_cell = tuple(int(v) for v in world.world_to_grid(world.goal))
    field = geodesic_distance_field(world.occupancy, goal_cell) * world.config.grid_resolution
    agent = make_agent("own_pose", cfg)
    agent.footprint_check = arm == "repaired"
    watch = Watcher(agent, world)
    agent.start_episode(world, env.robot.pose)
    info: dict = {}
    remaining = []
    for _ in range(cfg.max_episode_steps):
        action = agent.act(env.robot.pose, env.robot.velocity)
        _, _, terminated, truncated, info = env.step(action)
        watch.follow()
        remaining.append(remaining_at(field, world, env.robot.pose[:2]))
        if terminated or truncated:
            break
    rem = np.asarray(remaining)
    success, collision = bool(info.get("is_success")), bool(info.get("collision"))
    final = float(rem[-1]) if np.isfinite(rem[-1]) else float("nan")
    stalled = not success and not collision and np.isfinite(final) and final >= STALL_DISTANCE
    return {"cond": cond, "arm": arm, "seed": seed, "success": success, "collision": collision,
            "stalled": stalled, "steps": len(rem), "replans": int(agent.replans),
            "rebuilds": watch.rebuilds, "flips": watch.flips}


def returns(flips: list[dict]) -> list[dict]:
    """Each flip, marked with whether the robot's next flip took it back to the
    side this one abandoned, and whether that side had reopened by then."""
    out = []
    if not flips:
        return out
    for f, nxt in zip(flips, [*flips[1:], None], strict=True):
        back = (nxt is not None and f["old_bearing"] is not None and nxt["new_bearing"] is not None
                and abs(wrap_angle(nxt["new_bearing"] - f["old_bearing"])) < np.pi / 2)
        gap = nxt["step"] - f["step"] if nxt is not None else None
        out.append({**f, "returned": bool(back),
                    "reopened_before_return": bool(back and f["reopened_after"] is not None
                                                   and f["reopened_after"] <= gap)})
    return out


def group(rows: list[dict]) -> dict:
    flips = [f for r in rows for f in returns(r["flips"])]
    steps = sum(r["steps"] for r in rows)
    n = len(flips)
    reopened = [f for f in flips if f["reopened_after"] is not None]
    slack = [f["true_slack"] for f in flips if "true_slack" in f]
    real = [f["real"] for f in flips if "real" in f]
    back = [f for f in flips if f["returned"]]
    return {"episodes": len(rows), "steps": steps, "flips": n,
            "flips_per_100_steps": 100.0 * n / steps if steps else None,
            "triggers": {k: sum(f["trigger"] == k for f in flips) / n if n else None for k in TRIGGERS},
            "reopened_share": len(reopened) / n if n else None,
            "reopened_after_median": float(np.median([f["reopened_after"] for f in reopened])) if reopened else None,
            "rung_changed_share": (sum(abs(f["new_radius"] - f["old_radius"]) > 1e-9 for f in flips) / n
                                   if n else None),
            "from_robot_median": (float(np.median([f["from_robot"] for f in flips if "from_robot" in f]))
                                  if any("from_robot" in f for f in flips) else None),
            "length_ratio_median": (float(np.median([f["new_length"] / max(f["old_length"], 1e-9) for f in flips]))
                                    if n else None),
            "with_cause": len(slack),
            "room_in_truth_share": sum(s >= 0 for s in slack) / len(slack) if slack else None,
            "true_slack_median": float(np.median(slack)) if slack else None,
            "real_share": sum(real) / len(real) if real else None,
            "returned_share": len(back) / n if n else None,
            "reopened_before_return_share": (sum(f["reopened_before_return"] for f in back) / len(back)
                                             if back else None)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="results/flip_diagnostic.json")
    args = p.parse_args(argv)
    classes = {(c, w["seed"]): w["class"] for c in CONDITIONS
               for w in route_classes(args.split, c, args.episodes)}
    safe = ("short", "long detour")
    jobs = [(c, a, s, args.episodes, args.split) for c in CONDITIONS for a in ARMS
            for s in sorted(seed for cond, seed in classes if cond == c and classes[(cond, seed)] in safe)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(episode, jobs, chunksize=1))
    report: dict = {"prediction": PREDICTION, "arms": {}}
    for arm in ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        report["arms"][arm] = {"stalls": group([r for r in mine if r["stalled"]]),
                               "arrivals": group([r for r in mine if r["success"]])}
    published = Path("results/footprint_check_test.json")
    if args.split == "val" and args.episodes == 100 and published.exists():
        # Identity control: both arms must reproduce §9.26's episodes.
        before = {(r["arm"], r["cond"], r["seed"]): r
                  for r in json.loads(published.read_text(encoding="utf-8"))["episodes_detail"]}
        report["reproduces_footprint_test"] = all(
            all(r[f] == before[(r["arm"], r["cond"], r["seed"])][f]
                for f in ("success", "collision", "replans", "steps")) for r in rows)
        print(f"reproduces §9.26: {report['reproduces_footprint_test']}")
    g = report["arms"]["repaired"]["stalls"]
    if g["flips"] < 20:
        decision = "UNDERPOWERED"
    else:
        decision = "EDGE" if g["room_in_truth_share"] >= 0.5 else "BLOCKED"
    report["decision"] = decision
    report["episodes_detail"] = rows
    for arm, gs in report["arms"].items():
        for name, s in gs.items():
            if not s["flips"]:
                print(f"{arm:9s} {name:8s} {s['episodes']:3d} eps, no flips")
                continue
            print(f"{arm:9s} {name:8s} {s['episodes']:3d} eps {s['flips']:4d} flips "
                  f"({s['flips_per_100_steps']:.2f}/100 steps)  triggers "
                  + " ".join(f"{k} {v:.0%}" for k, v in s["triggers"].items())
                  + f"  | reopened within {FOLLOW} steps {s['reopened_share']:.0%} "
                  f"(median after {s['reopened_after_median']})  rung changed {s['rung_changed_share']:.0%}  "
                  f"blocked {s['from_robot_median']} m from robot  new/old length {s['length_ratio_median']:.2f}  "
                  f"room in truth {s['room_in_truth_share']} (slack median {s['true_slack_median']})  "
                  f"real {s['real_share']}  returned {s['returned_share']} "
                  f"(old side reopened first {s['reopened_before_return_share']})")
    print(f"decision {decision}")
    print("pre-registered: " + PREDICTION)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
