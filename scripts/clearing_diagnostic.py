"""Does counting a miss once per beam clear the phantoms §9.6 left standing?

§9.6 repaired the dense-scan map under noise by weighting each hit by the
returns its cell earned, and found its own mechanism wanting: the phantom cells
barely moved, 370 to 309 of about 1100 occupied, because the rule delays a
phantom rather than refusing it. `noisy_lidar` is still this stack's worst
open condition at 360 beams -- 0.890 on the held-out worlds, against Nav2's
1.000 -- and 28% of the free floor is still lost to inflation around surfaces
that are not there.

The asymmetry is in the update. Since §9.6 a hit is weighted by how many returns
its cell earned, but a miss is still one vote per cell per scan, so a cell six
beams pass straight through loses only one beam's worth of evidence. The code's
own comment says the miss vote exists to stop the samples of *one* beam voting
several times; collapsing separate beams as well is not what a textbook inverse
sensor model does. ``OccupancyMap(beam_votes=True)`` counts a miss per beam.

This measures that rule open loop, which isolates the map from everything it
feeds. The robot drives with the published mapper; a **shadow** map is handed
the identical scans at the identical poses, with beam votes on. Nothing about
the shadow reaches the robot, so any difference between the two maps is the
rule and nothing else. Against the world they were built from:

  phantoms        occupied cells with no real surface inside them, at the end
                  of the episode and averaged over it -- a phantom that blocks
                  a plan and is cleared later still cost that plan
  surface recall  of the cells that hold a real surface and have been seen,
                  the share the map calls occupied. The cost side: stronger
                  clearing could erase a surface that beams graze
  floor lost      free floor blocked by inflation around what is mapped

The pose is exact, as in `dense_map_diagnostic.py`: pose error smears returns
across cells and would make the mapper answer for the localiser.

Run on the ``val`` seed band, which no experiment scores.

    python scripts/clearing_diagnostic.py --episodes 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.agents.mapped import MappedPursuitAgent, make_sensor
from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import BENCHMARK_CONDITIONS
from vision_nav.mapping.occupancy import OCCUPIED, UNKNOWN, OccupancyMap
from vision_nav.training.env_factory import build_env_config

CONDITIONS = ("noisy_lidar", "nominal", "dense")
#: How often the phantom count is sampled through an episode.
SAMPLE_EVERY = 10
#: The shadow maps, each handed the same scans at the same poses. ``beam_votes``
#: is the rule this script was written to test; ``noise_margin`` was added after
#: its first run found 190 of 309 phantoms more than half a metre from any real
#: surface, which range noise of 0.1 m cannot do -- and traced them to beams
#: that hit nothing reading just under the sensor's maximum.
SHADOWS = {
    "beam_votes": {"beam_votes": True},
    "noise_margin": {"noise_margin": 3.0},
    # The control: no rule at all. Fed the same scans, it must end every
    # episode identical to the map the robot drove by, or the shadows are not
    # being given what the robot saw and every difference below is suspect.
    "null": {},
}
ARMS = ("published", "beam_votes", "noise_margin")


def phantoms(omap, world) -> int:
    occ = np.argwhere(omap.grid == OCCUPIED)
    if not len(occ):
        return 0
    d = world.clearance(omap.grid_to_world(occ), include_dynamic=False)
    return int((d > omap.half_diagonal).sum())


def measure(omap, world, radius: float) -> dict:
    """The map at the end of the episode, against the world it was built from."""
    occ = np.argwhere(omap.grid == OCCUPIED)
    out: dict = {"occupied": int(len(occ)), "phantoms": phantoms(omap, world)}
    # Where the phantoms sit. Range noise of 0.1 m spreads each return from a
    # wall over the cell or two in front of it, so a phantom hugging a surface
    # is that spread; one far from everything is something else entirely. The
    # two need different repairs, and only the first is about the noise model.
    if len(occ):
        d = world.clearance(omap.grid_to_world(occ), include_dynamic=False)
        ph = d[d > omap.half_diagonal]
        out["phantoms_within_0.2m"] = int((ph <= 0.2).sum())
        out["phantoms_0.2_to_0.5m"] = int(((ph > 0.2) & (ph <= 0.5)).sum())
        out["phantoms_beyond_0.5m"] = int((ph > 0.5).sum())
    else:
        out["phantoms_within_0.2m"] = out["phantoms_0.2_to_0.5m"] = 0
        out["phantoms_beyond_0.5m"] = 0
    # Surface recall over every seen cell that really holds a surface.
    h, w = omap.shape
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cells = np.stack([rr.ravel(), cc.ravel()], axis=-1)
    d = world.clearance(omap.grid_to_world(cells), include_dynamic=False).reshape(h, w)
    surface = d <= omap.half_diagonal
    seen = omap.grid != UNKNOWN
    observed = surface & seen
    out["surface_cells_seen"] = int(observed.sum())
    out["surface_recall"] = (float((omap.grid[observed] == OCCUPIED).mean())
                             if observed.any() else float("nan"))
    mine = omap.occupancy_at(radius)
    truth = world.occupancy_at(radius, include_dynamic=False)
    out["floor_lost"] = float((mine & ~truth & seen).sum() / max((~truth).sum(), 1))
    return out


def episode(env, seed: int, cfg) -> dict:
    env.reset(options={"world_seed": seed})
    world = env.world
    noise = cfg.lidar.noise_std
    agent = MappedPursuitAgent(robot=cfg.robot, sensor="lidar360", noise_std=noise,
                               corroborate=True)
    agent.start_episode(world, env.robot.pose)
    # Each shadow's first scan is re-sampled from an rng seeded exactly as the
    # agent's was, so it is the agent's first scan; every later one is handed
    # over explicitly and a shadow's rng is never drawn from again.
    shadows = {}
    for name, options in SHADOWS.items():
        shadow = OccupancyMap(world, make_sensor("lidar360", noise),
                              rng=np.random.default_rng(int(world.seed)),
                              corroborate=True, **options)
        shadow.integrate(env.robot.pose.copy())
        shadows[name] = shadow

    # Record each scan the agent takes, so the shadow can be given the same one.
    taken: list[np.ndarray] = []
    original = agent.map.scan

    def recording(pose):
        ranges = original(pose)
        taken.append(ranges)
        return ranges

    agent.map.scan = recording

    maps = {"published": agent.map, **shadows}
    trace: dict = {arm: [] for arm in maps}
    info: dict = {}
    for step in range(cfg.max_episode_steps):
        pose = env.robot.pose.copy()
        action = agent.act(pose)
        for shadow in shadows.values():
            shadow.integrate(pose, ranges=taken[-1])
        if step % SAMPLE_EVERY == 0:
            for arm, omap in maps.items():
                trace[arm].append(phantoms(omap, world))
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    radius = agent.map.robot_radius + agent.config.safety_margin
    faithful = bool(np.array_equal(shadows["null"].grid, agent.map.grid)
                    and np.array_equal(shadows["null"].evidence, agent.map.evidence))
    assert faithful, f"seed {seed}: the null shadow diverged from the robot's own map"
    return {
        "seed": seed,
        "success": bool(info.get("is_success")),
        "null_shadow_identical": faithful,
        **{arm: measure(omap, world, radius)
           | {"phantoms_mean": float(np.mean(trace[arm]))}
           for arm, omap in maps.items() if arm in ARMS},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    p.add_argument("--out", default="results/clearing_diagnostic.json")
    args = p.parse_args(argv)

    keys = ("occupied", "phantoms", "phantoms_mean", "surface_recall", "floor_lost",
            "phantoms_within_0.2m", "phantoms_0.2_to_0.5m", "phantoms_beyond_0.5m")
    report: dict = {"split": "val", "episodes": args.episodes, "conditions": {}}
    for cond in args.conditions:
        _, shift, noise = BENCHMARK_CONDITIONS[cond]
        cfg = build_env_config({"lidar": {"noise_std": noise}} if noise else {},
                               split="val", shift=shift, n_worlds=args.episodes)
        env = ProceduralNavEnv(cfg)
        seeds = [int(s) for s in list(cfg.world_seeds)[:args.episodes]]
        rows = [episode(env, s, cfg) for s in seeds]
        summary = {arm: {k: float(np.nanmean([r[arm][k] for r in rows])) for k in keys}
                   for arm in ARMS}
        report["conditions"][cond] = {"episodes": rows, **summary}
        print(f"\n{cond}", flush=True)
        for arm in ARMS:
            e = summary[arm]
            print(f"  {arm:12s} occupied {e['occupied']:6.0f}  phantoms "
                  f"{e['phantoms']:5.0f} at the end, {e['phantoms_mean']:5.0f} on "
                  f"average  surface recall {e['surface_recall']:.3f}  floor lost "
                  f"{e['floor_lost']:.3f}", flush=True)
            print(f"  {'':12s} phantoms by distance to the nearest real surface: "
                  f"<=0.2 m {e['phantoms_within_0.2m']:.0f}, 0.2-0.5 m "
                  f"{e['phantoms_0.2_to_0.5m']:.0f}, >0.5 m "
                  f"{e['phantoms_beyond_0.5m']:.0f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
