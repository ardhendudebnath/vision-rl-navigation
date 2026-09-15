"""Measure how much of the world a discrete lidar can actually see.

Phase 2f inferred a perception limit from end-to-end success rate; Phase 2g
showed that inference was swamped by training-seed noise. This measures the
mechanism directly: across real evaluation worlds, how often does an N-beam
scan fail to reveal an opening the robot could drive through?

No policy, no training, no seeds — so it separates "can the sensor see the
gap?" from "does the policy act on it?", which the end-to-end experiment
could not.

    python scripts/perception_audit.py --shift narrow --beams 16 32 64 128
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.analysis.perception import audit_pose
from vision_nav.envs.splits import shifted_config, split_seeds
from vision_nav.envs.world import WorldConfig, generate_world
from vision_nav.planning.grid_astar import plan_path
from vision_nav.planning.smoothing import densify_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shift", default="narrow", help="World distribution shift")
    p.add_argument("--split", default="test_ood", help="Seed band")
    p.add_argument("--worlds", type=int, default=60)
    p.add_argument("--beams", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument(
        "--fov-deg",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Sweep field of view (degrees) at fixed beam count instead of "
            "sweeping beam count. Requires exactly one --beams value."
        ),
    )
    p.add_argument(
        "--config",
        nargs="+",
        default=None,
        metavar="BEAMS:FOVDEG",
        help=(
            "Explicit (samples, FOV) pairs, e.g. 32:90 64:180 128:360. Lets "
            "angular resolution be held constant while coverage varies, which "
            "neither of the other two sweeps can do."
        ),
    )
    p.add_argument("--probe", type=float, default=3.0,
                   help="Distance the robot must be able to travel, metres")
    p.add_argument("--poses-per-world", type=int, default=8)
    p.add_argument("--out", default="results/perception_audit.json")
    return p.parse_args(argv)


def sample_poses(world, n: int, rng) -> list[np.ndarray]:
    """Poses along the optimal route — where perception actually matters.

    Sampling uniformly over free space would mostly probe open floor the
    policy never has to reason about. The interesting question is what the
    sensor shows at the points the robot actually passes through.
    """
    path = plan_path(world, world.start[:2], world.goal)
    if path is None or len(path) < 2:
        return []
    dense = densify_path(path, 0.1)
    idx = np.linspace(0, len(dense) - 1, min(n, len(dense))).astype(int)

    poses = []
    for i in idx:
        position = dense[i]
        # Face along the route, which is the heading the robot would hold.
        j = min(i + 5, len(dense) - 1)
        delta = dense[j] - position
        heading = float(np.arctan2(delta[1], delta[0])) if np.any(delta) else 0.0
        poses.append(np.array([position[0], position[1], heading]))
    return poses


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(0)

    base = WorldConfig()
    cfg = shifted_config(base, args.shift) if args.shift != "nominal" else base
    seeds = split_seeds(args.split, args.worlds)

    # Each configuration under test: (label, n_beams, fov_radians).
    if args.config:
        configs = []
        for spec in args.config:
            beams_s, _, fov_s = spec.partition(":")
            if not fov_s:
                print(f"error: malformed --config {spec!r}; expected BEAMS:FOVDEG")
                return 2
            n, deg = int(beams_s), float(fov_s)
            configs.append((f"{n}@{int(deg)}deg", n, np.radians(deg)))
    elif args.fov_deg:
        if len(args.beams) != 1:
            print("error: --fov-deg requires exactly one --beams value", flush=True)
            return 2
        n = args.beams[0]
        configs = [
            (f"{int(d)}deg", n, np.radians(d)) for d in args.fov_deg
        ]
    else:
        configs = [(f"{n}b", n, 2 * np.pi) for n in args.beams]

    # gap width bins, in degrees
    bins = [(0, 5), (5, 10), (10, 20), (20, 45), (45, 360)]
    per_beam: dict[str, dict] = {
        label: {
            "n_beams": n,
            "fov_deg": float(np.degrees(f)),
            "gaps": 0,
            "detected": 0,
            "by_width": {f"{lo}-{hi}": [0, 0] for lo, hi in bins},
        }
        for label, n, f in configs
    }

    n_poses = 0
    for seed in seeds:
        world = generate_world(seed, cfg)
        for pose in sample_poses(world, args.poses_per_world, rng):
            n_poses += 1
            for label, n, fov in configs:
                stats = audit_pose(
                    world, pose, n_beams=n, probe_distance=args.probe, fov=fov
                )
                acc = per_beam[label]
                acc["gaps"] += stats.n_gaps
                acc["detected"] += stats.n_detected
                for width, hit in zip(stats.widths, stats.detected, strict=True):
                    deg = np.degrees(width)
                    for lo, hi in bins:
                        if lo <= deg < hi:
                            key = f"{lo}-{hi}"
                            acc["by_width"][key][0] += 1
                            acc["by_width"][key][1] += int(hit)
                            break

    print(f"\nPerception audit: shift={args.shift}, {len(seeds)} worlds, "
          f"{n_poses} poses on-route, probe={args.probe} m")
    print(f"Robot diameter {2 * cfg.robot_radius:.2f} m\n")
    print(f"{'config':>8} {'beams':>6} {'FOV':>7} {'spacing':>9} {'gaps':>7} {'detected':>9} {'rate':>7}")
    for label, n, fov in configs:
        acc = per_beam[label]
        rate = acc["detected"] / acc["gaps"] if acc["gaps"] else float("nan")
        spacing = np.degrees(fov) / n
        print(
            f"{label:>8} {n:>6} {np.degrees(fov):>6.0f}° {spacing:>8.2f}° "
            f"{acc['gaps']:>7} {acc['detected']:>9} {rate:>7.3f}"
        )

    print("\nDetection rate by gap angular width:")
    labels = [c[0] for c in configs]
    print("  width".ljust(12) + "".join(f"{lab:>10}" for lab in labels))
    for lo, hi in bins:
        key = f"{lo}-{hi}"
        row = f"  {lo}-{hi}°".ljust(12)
        for lab in labels:
            total, hit = per_beam[lab]["by_width"][key]
            row += f"{(hit / total if total else float('nan')):>10.3f}"
        print(row)
    first = per_beam[labels[0]]["by_width"]
    print("\n  gap counts per width bin (identical across configs):")
    print("  " + "  ".join(f"{lo}-{hi}°: {first[f'{lo}-{hi}'][0]}" for lo, hi in bins))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "shift": args.shift,
                "split": args.split,
                "worlds": len(seeds),
                "poses": n_poses,
                "probe_distance": args.probe,
                "robot_diameter": 2 * cfg.robot_radius,
                "per_config": per_beam,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
