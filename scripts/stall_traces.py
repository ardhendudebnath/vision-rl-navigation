"""The stalls' trajectories: what turns a stalled robot's driving back on itself.

Report §9.29 found the clutter stall to be a robot that keeps moving and goes
nowhere: 6.9 m driven, ending 1.6 m from where the stall began, half the time
inside a wall's margin. This re-drives those 32 episodes (the published stack on
the margin-safe val clutter worlds, instrumented as
``controller_stall_diagnostic.py`` instruments it, and required to reproduce it)
and keeps every step: where the robot truly is and believes it is, what the
controller commanded, and at every rebuild the first metre of the new route.
Over each stall phase it measures

  turnarounds   how often the direction of travel reverses: over 10 steps the
                robot's displacement turns by more than 120 degrees
  route swings  at each rebuild, the angle between where the new route's first
                metre points and where the old one's did
  revisits      the share of steps spent within 0.3 m of somewhere the robot
                had already been at least 30 steps before
  winding       the robot's net rotation, against its total rotation
  route flips   rebuilds whose first metre points more than 90 degrees from
                the last route's -- the other way round something -- and
                whether the turnarounds follow them

It separates the stalls still closing on the goal when the budget ran out (a
stall phase under 100 steps) from those that went nowhere, reads §9.22's own
reversal counts per step rather than per rebuild, and plots eight of the stalls
(``docs/figures/stall_traces.png``) and four up close
(``docs/figures/stall_shuttles.png``). Not registered: descriptive, written
after §9.29.

    python scripts/stall_traces.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller_stall_diagnostic as csd  # noqa: E402  (wraps the controller on import)
from stall_diagnostic import make_agent  # noqa: E402

from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.envs.robot import wrap_angle  # noqa: E402
from vision_nav.envs.splits import BENCHMARK_CONDITIONS  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

DIAGNOSTIC = Path("results/controller_stall_diagnostic.json")
#: A reversal of travel: displacement over this many steps turning past this.
TURN_WINDOW, TURN_ANGLE = 10, np.deg2rad(120)
#: A revisit: within this distance of a point at least this many steps old.
REVISIT_RADIUS, REVISIT_AGE = 0.3, 30
#: A stall phase shorter than this ended in the last stretch of the budget.
LATE = 100


def first_metre(track: np.ndarray, cursor: int, position: np.ndarray) -> float | None:
    """Bearing from the robot to the route point about a metre along it."""
    ahead = track[cursor:]
    if len(ahead) < 2:
        return None
    d = np.linalg.norm(ahead - position, axis=1)
    i = int(np.argmax(d >= 1.0)) if (d >= 1.0).any() else len(ahead) - 1
    v = ahead[i] - position
    return float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 1e-6 else None


def trace(args: tuple) -> dict:
    cond, seed, n = args
    _, shift, noise = BENCHMARK_CONDITIONS[cond]
    cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                           split="val", shift=shift, n_worlds=n)
    env = ProceduralNavEnv(cfg)
    env.reset(options={"world_seed": seed})
    world = env.world
    agent = make_agent("own_pose", cfg)
    agent.start_episode(world, env.robot.pose)
    threshold = agent.config.turn_in_place_threshold
    rows, routes = [], []
    info: dict = {}
    for k in range(cfg.max_episode_steps):
        replans, recovery = agent.replans, agent.recovery_steps
        csd._LAST.clear()
        action = agent.act(env.robot.pose, env.robot.velocity)
        believed = agent._odom.pose.copy()
        if agent.recovery_steps > recovery or not csd._LAST:
            kind = "recovering"
        elif abs(csd._LAST["heading_error"]) > threshold:
            kind = "turning"
        elif csd._LAST["caution"] < csd.CRAWL:
            kind = "crawling"
        else:
            kind = "driving"
        rebuilt = agent.replans > replans
        if rebuilt and agent._track is not None:
            bearing = first_metre(agent._track, agent._cursor, believed[:2])
            snap = agent._track[agent._cursor:agent._cursor + 40:4]
            routes.append({"step": k, "bearing": bearing, "points": np.round(snap, 3).tolist()})
        _, _, terminated, truncated, info = env.step(action)
        rows.append([round(float(v), 4) for v in (*env.robot.pose, *believed[:2])] + [kind, rebuilt])
        if terminated or truncated:
            break
    return {"cond": cond, "seed": seed, "success": bool(info.get("is_success")),
            "steps": len(rows), "replans": int(agent.replans), "rows": rows, "routes": routes,
            "world": {"circles": world.circles.tolist(), "boxes": world.boxes.tolist(),
                      "width": world.config.width, "height": world.config.height,
                      "start": world.start.tolist(), "goal": world.goal.tolist()}}


def measure(t: dict, start: int) -> dict:
    xy = np.array([r[:2] for r in t["rows"]])[start:]
    th = np.array([r[2] for r in t["rows"]])[start:]
    n = len(xy)
    turnarounds, turn_steps = 0, []
    last = None
    for i in range(0, n - TURN_WINDOW, TURN_WINDOW):
        v = xy[i + TURN_WINDOW] - xy[i]
        if np.linalg.norm(v) < 0.05:
            continue  # standing or turning on the spot: no direction of travel
        heading = np.arctan2(v[1], v[0])
        if last is not None and abs(wrap_angle(heading - last)) > TURN_ANGLE:
            turnarounds += 1
            turn_steps.append(start + i + TURN_WINDOW)
        last = heading
    revisit = 0
    for i in range(REVISIT_AGE, n):
        if (np.linalg.norm(xy[:i - REVISIT_AGE + 1] - xy[i], axis=1) < REVISIT_RADIUS).any():
            revisit += 1
    rot = np.array([wrap_angle(b - a) for a, b in zip(th[:-1], th[1:], strict=True)])
    routes = [r for r in t["routes"] if r["step"] >= start and r["bearing"] is not None]
    swings = [abs(float(wrap_angle(b["bearing"] - a["bearing"])))
              for a, b in zip(routes[:-1], routes[1:], strict=True)]
    # A route flip: a rebuild whose first metre points more than 90 degrees from
    # the last one's -- the other way round something.
    flips = [b["step"] for a, b in zip(routes[:-1], routes[1:], strict=True)
             if abs(wrap_angle(b["bearing"] - a["bearing"])) > np.pi / 2]
    return {"steps": n, "turnarounds": turnarounds, "turnaround_steps": turn_steps,
            "turnarounds_per_100": 100.0 * turnarounds / max(n, 1),
            "revisit_share": revisit / max(n - REVISIT_AGE, 1),
            "net_turns": float(rot.sum() / (2 * np.pi)), "total_turns": float(np.abs(rot).sum() / (2 * np.pi)),
            "rebuilds": len(routes), "swings": swings, "flips": flips}


def plot(traces: list[dict], starts: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
    for ax, t in zip(axes.ravel(), traces, strict=False):
        w = t["world"]
        for x, y, r in w["circles"]:
            ax.add_patch(Circle((x, y), r, color="0.55"))
        for x0, y0, x1, y1 in w["boxes"]:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.55"))
        xy = np.array([r[:2] for r in t["rows"]])
        s = starts[(t["cond"], t["seed"])]
        ax.plot(xy[:s + 1, 0], xy[:s + 1, 1], color="0.2", lw=1.0, label="before the stall")
        sc = ax.scatter(xy[s:, 0], xy[s:, 1], c=np.arange(len(xy) - s), cmap="viridis", s=4,
                        label="stall phase (dark to light)")
        for r in t["routes"]:
            if r["step"] >= s and r["points"]:
                p = np.array(r["points"])
                ax.plot(p[:, 0], p[:, 1], color="tab:red", lw=0.4, alpha=0.35)
        ax.plot(*w["start"][:2], "s", color="tab:blue", ms=7)
        ax.plot(*w["goal"][:2], "*", color="tab:orange", ms=14)
        ax.set_xlim(0, w["width"])
        ax.set_ylim(0, w["height"])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{t['cond']} world {t['seed']}", fontsize=10)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="steps into the stall phase")
    handles = [plt.Line2D([], [], color="0.2", label="path before the stall"),
               plt.Line2D([], [], color="tab:green", marker="o", ls="", ms=4, label="stall phase"),
               plt.Line2D([], [], color="tab:red", alpha=0.6, label="first metre of each rebuilt route"),
               plt.Line2D([], [], color="tab:blue", marker="s", ls="", label="start"),
               plt.Line2D([], [], color="tab:orange", marker="*", ls="", ms=12, label="goal")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Stalled robots on margin-safe val clutter worlds (published stack)", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def plot_zoom(traces: list[dict], starts: dict, path: Path) -> None:
    """Four stalls up close: the true obstacle, its collision boundary (the
    robot's radius) and the planner's margin, each step by its command."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    colours = {"turning": "tab:red", "crawling": "tab:orange", "driving": "tab:green", "recovering": "k"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.6))
    for ax, t in zip(axes, traces, strict=False):
        s, w = starts[(t["cond"], t["seed"])], t["world"]
        rows = t["rows"][s:]
        xy = np.array([r[:2] for r in rows])
        c = xy.mean(axis=0)
        for x, y, r in w["circles"]:
            for pad, col in ((0.40, "0.93"), (0.22, "0.8"), (0.0, "0.45")):
                ax.add_patch(Circle((x, y), r + pad, color=col))
        for x0, y0, x1, y1 in w["boxes"]:
            for pad, col in ((0.40, "0.93"), (0.22, "0.8")):
                ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad={pad}", color=col))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.45"))
        for r in t["routes"]:
            if r["step"] >= s and r["points"]:
                pts = np.array(r["points"])
                ax.plot(pts[:, 0], pts[:, 1], color="tab:blue", lw=0.5, alpha=0.3)
        ax.plot(xy[:, 0], xy[:, 1], color="0.3", lw=0.5)
        for kind, col in colours.items():
            m = np.array([r[5] == kind for r in rows])
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=6, color=col, zorder=3, label=f"{kind} {m.mean():.0%}")
        g = np.asarray(w["goal"][:2]) - c
        g = 0.8 * g / max(np.linalg.norm(g), 1e-9)
        ax.annotate("", xy=c + g, xytext=c, arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 2})
        ax.set_xlim(c[0] - 1.8, c[0] + 1.8)
        ax.set_ylim(c[1] - 1.8, c[1] + 1.8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, loc="lower left")
        ax.set_title(f"{t['cond']} world {t['seed']}: {len(rows)} stall steps", fontsize=10)
    fig.suptitle("Four stalls up close. Dark grey: obstacle; mid grey: where the robot's body would touch it; "
                 "light grey: the planner's margin. Blue: first metre of each rebuilt route. "
                 "Orange arrow: towards the goal.", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default="results/stall_traces.json")
    p.add_argument("--figure", default="docs/figures/stall_traces.png")
    p.add_argument("--zoom", default="docs/figures/stall_shuttles.png")
    args = p.parse_args(argv)
    diag = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))["episodes_detail"]
    stalls = [r for r in diag if r["stalled"] and r["class"] in ("short", "long detour")]
    starts = {(r["cond"], r["seed"]): r["phase_start"] for r in stalls}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        traces = list(pool.map(trace, [(r["cond"], r["seed"], 100) for r in stalls], chunksize=1))
    expected = {(r["cond"], r["seed"]): r for r in stalls}
    reproduces = all(t["steps"] == expected[(t["cond"], t["seed"])]["steps"]
                     and t["replans"] == expected[(t["cond"], t["seed"])]["replans"]
                     and not t["success"] for t in traces)
    per = [{"cond": t["cond"], "seed": t["seed"], **measure(t, starts[(t["cond"], t["seed"])])}
           for t in traces]
    swings = np.concatenate([q["swings"] for q in per if q["swings"]])
    # Stalls whose last progress came this close to the end of the budget were
    # still closing on the goal when time ran out; the rest went nowhere.
    short = [q for q in per if q["steps"] < LATE]
    long = [q for q in per if q["steps"] >= LATE]
    steps = sum(q["steps"] for q in long)
    turns = sum(q["turnarounds"] for q in long)
    after_flip = sum(any(0 <= ts - f <= 2 * TURN_WINDOW for f in q["flips"])
                     for q in long for ts in q["turnaround_steps"])
    sd = json.loads(Path("results/stall_diagnostic.json").read_text(encoding="utf-8"))["groups"]["own_pose"]
    report = {
        "reproduces_diagnostic": reproduces, "stalls": len(per),
        "still_closing_at_the_end": len(short),
        "still_closing_phase_max": max(q["steps"] for q in short) if short else None,
        "went_nowhere": len(long), "went_nowhere_phase_min": min(q["steps"] for q in long) if long else None,
        "long_flips_per_100": 100.0 * sum(len(q["flips"]) for q in long) / steps,
        "long_turnarounds_per_100": 100.0 * turns / steps,
        "long_turnarounds_after_a_flip": after_flip, "long_turnarounds": turns,
        "flips_turnarounds_correlation": float(np.corrcoef([len(q["flips"]) for q in long],
                                                           [q["turnarounds"] for q in long])[0, 1]),
        # §9.22's own counts, read per step rather than per rebuild.
        "reversals_per_100_steps": {g: sd[g]["reversal_share"] * sd[g]["per_100_steps"]
                                    for g in ("stall", "arrive")},
        "turnarounds_per_100_median": float(np.median([q["turnarounds_per_100"] for q in per])),
        "revisit_share_median": float(np.median([q["revisit_share"] for q in per])),
        "net_turns_median": float(np.median([abs(q["net_turns"]) for q in per])),
        "total_turns_median": float(np.median([q["total_turns"] for q in per])),
        "route_swing_median_deg": float(np.degrees(np.median(swings))),
        "route_swing_over_45": float(np.mean(swings > np.deg2rad(45))),
        "route_swing_over_90": float(np.mean(swings > np.deg2rad(90))),
        "per_stall": per}
    print(f"reproduces the diagnostic: {reproduces}; {len(per)} stalls")
    print(f"turnarounds per 100 steps (median) {report['turnarounds_per_100_median']:.1f}; "
          f"revisiting {report['revisit_share_median']:.0%} of steps; "
          f"rotation net {report['net_turns_median']:.1f} of {report['total_turns_median']:.1f} turns")
    print(f"route swings at rebuild: median {report['route_swing_median_deg']:.0f} deg, "
          f"over 45 deg {report['route_swing_over_45']:.0%}, over 90 deg {report['route_swing_over_90']:.0%}")
    for q in per:
        print(f"  {q['cond']:6s} {q['seed']}  {q['steps']:3d} steps  turnarounds {q['turnarounds_per_100']:4.1f}/100  "
              f"revisit {q['revisit_share']:.0%}  turns net {q['net_turns']:+.1f} total {q['total_turns']:.1f}")
    print(f"still closing when time ran out: {len(short)} (stall phases up to "
          f"{report['still_closing_phase_max']} steps); went nowhere: {len(long)} (from "
          f"{report['went_nowhere_phase_min']} steps)")
    print(f"long stalls: route flips {report['long_flips_per_100']:.2f}/100 steps, turnarounds "
          f"{report['long_turnarounds_per_100']:.2f}/100, {after_flip} of {turns} turnarounds within "
          f"{2 * TURN_WINDOW} steps after a flip, per-stall correlation "
          f"{report['flips_turnarounds_correlation']:.2f}")
    print(f"§9.22 reversals per 100 steps: {report['reversals_per_100_steps']}")
    Path(args.out).write_text(json.dumps({**report, "traces": traces}), encoding="utf-8")
    # The eight longest stall phases, four per condition where there are four.
    chosen = []
    for cond in csd.CONDITIONS:
        mine = sorted((t for t in traces if t["cond"] == cond),
                      key=lambda t: -(t["steps"] - starts[(t["cond"], t["seed"])]))
        chosen += mine[:4]
    plot(chosen[:8], starts, Path(args.figure))
    # Up close: the four stalls that revisit the most of their own ground.
    by = {(t["cond"], t["seed"]): t for t in traces}
    most = sorted(long, key=lambda q: -q["revisit_share"])[:4]
    plot_zoom([by[(q["cond"], q["seed"])] for q in most], starts, Path(args.zoom))
    print(f"Wrote {args.out}, {args.figure} and {args.zoom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
