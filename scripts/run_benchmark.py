"""Run the comparison matrix and write the results table.

This is the script that produces the numbers in the report.  It evaluates
every actor under every condition on the *same* worlds, so rows are directly
comparable.

    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --rl runs/ppo_privileged/best_model.zip
    python scripts/run_benchmark.py --conditions nominal dense --episodes 50

Compare several trained policies in one matrix:

    python scripts/run_benchmark.py \
        --rl nominal_trained=runs/ppo_privileged/best_model.zip \
             dr_trained=runs/ppo_dr/best_model.zip

Evaluation always runs with domain randomisation OFF and the named shift ON,
whatever the policy was trained with. Otherwise the randomiser would overwrite
the very parameters the shift defines, and the row would not measure the
condition it claims to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vision_nav.envs.splits import SHIFTS
from vision_nav.metrics.navigation import NavigationMetrics
from vision_nav.training.actors import build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate

#: Each condition is (label, split, shift, lidar noise std).  ``test`` is the
#: in-distribution held-out set; the rest are the Phase-5 robustness suite.
CONDITIONS: dict[str, tuple[str, str | None, float]] = {
    "nominal": ("test", None, 0.0),
    "dense": ("test_ood", "dense", 0.0),
    "sparse": ("test_ood", "sparse", 0.0),
    "large": ("test_ood", "large", 0.0),
    "narrow": ("test_ood", "narrow", 0.0),
    "noisy_lidar": ("test", None, 0.10),
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rl",
        nargs="+",
        default=None,
        metavar="[NAME=]PATH",
        help=(
            "Trained SB3 model(s) to evaluate. Repeatable. Prefix with NAME= to "
            "label the row, e.g. --rl dr=runs/ppo_dr/best_model.zip"
        ),
    )
    p.add_argument("--episodes", type=int, default=100, help="Episodes per condition")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="Subset of conditions to run",
    )
    p.add_argument("--actors", nargs="+", default=None, help="Subset of actors to run")
    p.add_argument("--out-dir", default="results", help="Where to write results")
    return p.parse_args(argv)


#: Sensor *geometry* is a property of the policy: a 64-beam policy cannot be
#: evaluated in a 32-beam env, its observation would not even be the right
#: shape. Sensor *corruption* (noise, dropout) is a property of the evaluation
#: condition and must not be inherited from the training config, or the
#: noisy_lidar row would silently evaluate a clean sensor.
SENSOR_GEOMETRY_FIELDS = ("n_beams", "fov", "max_range")
CAMERA_GEOMETRY_FIELDS = ("fov", "width", "max_range")


def actor_sensor_geometry(model_path: str | None) -> dict:
    """Read lidar geometry from the run directory's saved config."""
    if not model_path:
        return {}
    cfg_path = Path(model_path).parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    from omegaconf import OmegaConf

    saved = OmegaConf.load(cfg_path)
    lidar = OmegaConf.to_container(saved.env.lidar, resolve=True)
    return {k: lidar[k] for k in SENSOR_GEOMETRY_FIELDS if k in lidar}


def actor_env_spec(model_path: str | None) -> dict:
    """Observation mode and sensor geometry for a trained policy.

    A depth-camera policy cannot be evaluated in a lidar env at all — the
    observation is a different shape and a different sensor — so the mode
    travels with the policy, read from the run that produced it.
    """
    if not model_path:
        return {}
    cfg_path = Path(model_path).parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    from omegaconf import OmegaConf

    saved = OmegaConf.load(cfg_path)
    env = saved.env
    spec: dict = {"obs_mode": OmegaConf.select(env, "obs_mode") or "privileged"}

    lidar = OmegaConf.to_container(env.lidar, resolve=True) if "lidar" in env else {}
    spec["lidar"] = {k: lidar[k] for k in SENSOR_GEOMETRY_FIELDS if k in lidar}

    camera = OmegaConf.to_container(env.camera, resolve=True) if "camera" in env else {}
    spec["camera"] = {k: camera[k] for k in CAMERA_GEOMETRY_FIELDS if k in camera}
    return spec


def apply_sensor_noise(spec: dict, noise: float) -> dict:
    """Inject the condition's noise into whichever sensor the policy uses.

    This is the trap: the ``noisy_lidar`` condition sets ``lidar.noise_std``,
    which a depth-camera policy never reads. Left unhandled, that row would
    silently evaluate a *clean* camera and report the result as robustness —
    the same non-exposure that makes the row meaningless for the classical
    planner, but far easier to miss because the actor is a learned policy and
    the number looks plausible.
    """
    if not noise:
        return spec
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in spec.items()}
    target = "camera" if out.get("obs_mode") == "depth" else "lidar"
    out.setdefault(target, {})["noise_std"] = noise
    return out


def build_actor_specs(rl_args: list[str] | None) -> dict[str, dict]:
    """Map actor label -> build_actor kwargs.

    Each ``--rl`` entry is ``PATH`` or ``NAME=PATH``. Without an explicit
    name, the run directory name is used, so rows stay traceable to the run
    that produced them instead of collapsing to a generic "rl".
    """
    specs: dict[str, dict] = {
        "random": {"kind": "random"},
        "classical": {"kind": "classical"},
    }
    for entry in rl_args or []:
        name, _, path = entry.rpartition("=")
        if not name:
            # runs/<run_name>/best_model.zip -> <run_name>
            name = Path(path).parent.name or "rl"
        if name in specs:
            raise ValueError(f"duplicate actor label {name!r}")
        if not Path(path).exists():
            raise FileNotFoundError(f"model not found: {path}")
        specs[name] = {"kind": "rl", "model_path": path}
    return specs


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        specs = build_actor_specs(args.rl)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.actors:
        missing = set(args.actors) - set(specs)
        if missing:
            print(f"error: unknown actor(s) {sorted(missing)}", file=sys.stderr)
            print(f"       available: {sorted(specs)}", file=sys.stderr)
            return 2
        specs = {k: v for k, v in specs.items() if k in args.actors}

    if not args.rl:
        print(
            "note: no --rl model given; running baselines only. Train one with\n"
            "      python -m vision_nav.training.train\n",
            flush=True,
        )

    rows: list[dict] = []
    for cond in args.conditions:
        split, shift, noise = CONDITIONS[cond]
        overrides = {"lidar": {"noise_std": noise}} if noise else {}
        reference = build_env_config(
            overrides, split=split, shift=shift, n_worlds=args.episodes
        )

        for actor_name, spec in specs.items():
            env_spec = apply_sensor_noise(actor_env_spec(spec.get("model_path")), noise)
            actor_overrides = dict(overrides)
            if env_spec:
                actor_overrides = {
                    k: v for k, v in env_spec.items() if v not in ({}, None)
                }
            env_config = build_env_config(
                actor_overrides, split=split, shift=shift, n_worlds=args.episodes
            )
            geometry = env_spec.get("lidar", {}) if env_spec else {}

            # The sensor may differ per actor; the WORLDS may not. If they did,
            # rows would no longer be comparable, which is the one property the
            # whole benchmark depends on.
            assert env_config.world == reference.world, (
                f"{actor_name} would be evaluated on different worlds"
            )
            assert list(env_config.world_seeds or []) == list(reference.world_seeds or []), (
                f"{actor_name} would be evaluated on a different seed sequence"
            )
            # Whichever sensor this policy actually reads must carry the
            # condition's noise, or the row measures a clean sensor.
            active_noise = (
                env_config.camera.noise_std
                if env_config.obs_mode == "depth"
                else env_config.lidar.noise_std
            )
            assert active_noise == noise, (
                f"{actor_name} ({env_config.obs_mode}) would not see this "
                f"condition's sensor noise: got {active_noise}, expected {noise}"
            )

            if env_config.obs_mode == "depth":
                suffix = f" [{env_config.camera.width}px @ {np.degrees(env_config.camera.fov):.0f}°]"
            elif geometry.get("n_beams"):
                suffix = f" [{geometry['n_beams']} beams]"
            else:
                suffix = ""
            print(f"[{cond:>12}] {actor_name}{suffix} ...", end=" ", flush=True)
            actor = build_actor(robot=env_config.robot, **spec)
            metrics, results = evaluate(actor, env_config)
            print(
                f"SR={metrics.success_rate:.3f} SPL={metrics.spl:.3f} "
                f"coll={metrics.collision_rate:.3f}",
                flush=True,
            )
            rows.append(
                {
                    "condition": cond,
                    "split": split,
                    "shift": shift or "none",
                    "lidar_noise_std": noise,
                    "actor": actor_name,
                    "obs_mode": env_config.obs_mode,
                    "n_beams": geometry.get("n_beams"),
                    "camera_width": (
                        env_config.camera.width if env_config.obs_mode == "depth" else None
                    ),
                    **metrics.to_dict(),
                }
            )
            metrics.save(out_dir / f"{cond}__{actor_name}.json")

    (out_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = render_table(rows)
    (out_dir / "benchmark.md").write_text(table, encoding="utf-8")

    print()
    print(table)
    print(f"\nWrote {out_dir / 'benchmark.json'} and {out_dir / 'benchmark.md'}")
    return 0


def render_table(rows: list[dict]) -> str:
    """Render the matrix as a Markdown table, ready for the report."""
    header = (
        "| Condition | Actor | Success | SPL | Collision | Timeout | Steps (succ.) |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Benchmark results",
        "",
        "Every row is evaluated on the same worlds in the same order.",
        "`nominal` is the held-out in-distribution test split; the remaining",
        "conditions are distribution shifts never seen during training.",
        "",
        header,
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['actor']} | {r['success_rate']:.3f} | "
            f"{r['spl']:.3f} | {r['collision_rate']:.3f} | "
            f"{r['timeout_rate']:.3f} | {r['mean_steps_to_goal']:.0f} |"
        )
    beam_counts = {r.get("n_beams") for r in rows if r.get("n_beams")}
    lines += [
        "",
        "## Reading this table",
        "",
        "- `random` is the floor. Any result that does not clearly clear it is noise.",
    ]
    if len(beam_counts) > 1:
        lines += [
            f"- **Actors here use different lidar beam counts** ({sorted(beam_counts)}).",
            "  Sensor geometry is read from each run's saved config, because a policy",
            "  cannot be evaluated at a beam count it was not trained for. The worlds,",
            "  seed order and injected sensor noise are identical across every row —",
            "  asserted at evaluation time, not assumed.",
        ]
    lines += [
        "- **`noisy_lidar` is a no-op for `classical` by construction.** The classical",
        "  baseline navigates from the map and never reads the lidar, so its row is",
        "  identical to `nominal`. That is not robustness — it is non-exposure, and it",
        "  is exactly the axis on which a sensor-driven policy should be expected to",
        "  differ.",
        "- SPL can reach 1.000 when the agent stops inside the goal tolerance and so",
        "  travels slightly less than `l*`; the `max(p, l*)` term caps it there.",
        "",
        f"_Shift definitions: {', '.join(sorted(SHIFTS))}._",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
