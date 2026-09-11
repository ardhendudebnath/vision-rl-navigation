"""Render navigation episodes to an animated GIF.

The README demo and the side-by-side figure in the report both come from
here.  A short clip of the classical planner and the learned policy solving
the *same* world is the single most persuasive artefact for a motivation
letter, so it is worth being a first-class script rather than a notebook.

    python scripts/make_demo.py --world 20000 --out results/demo.gif
    python scripts/make_demo.py --actor rl --rl runs/ppo_privileged/best_model.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.training.actors import build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.viz.topdown import render_topdown


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", default="classical", choices=["classical", "random", "rl"])
    p.add_argument("--rl", default=None, help="Path to a trained SB3 model (.zip)")
    p.add_argument("--world", type=int, default=20_000, help="World seed to render")
    p.add_argument("--worlds", type=int, default=1, help="Render this many consecutive seeds")
    p.add_argument("--out", default="results/demo.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=2, help="Keep every Nth frame")
    p.add_argument("--hold", type=int, default=12, help="Frames to hold on the final pose")
    return p.parse_args(argv)


def render_episode(env: ProceduralNavEnv, actor, world_seed: int, stride: int, hold: int):
    """Roll out one episode, returning (frames, result-summary)."""
    obs, info = env.reset(options={"world_seed": world_seed})
    planned = None
    if not actor.reset(env, obs):
        print(f"  world {world_seed}: actor failed to plan; skipping")
        return [], None
    planned = getattr(getattr(actor, "agent", None), "path", None)

    trajectory = [env.robot.position.copy()]
    frames = []
    step = 0
    while True:
        action = actor.act(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        trajectory.append(env.robot.position.copy())
        step += 1
        if step % stride == 0:
            frames.append(
                render_topdown(env.world, env.robot.pose, np.asarray(trajectory), planned)
            )
        if terminated or truncated:
            break

    final = render_topdown(env.world, env.robot.pose, np.asarray(trajectory), planned)
    frames.extend([final] * hold)

    outcome = "success" if info.get("is_success") else ("collision" if info.get("collision") else "timeout")
    print(
        f"  world {world_seed}: {outcome} in {info['steps']} steps, "
        f"path {info['path_length']:.2f} m (optimal {info['shortest_path_length']:.2f} m)"
    )
    return frames, outcome


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("error: imageio is required. Install with: pip install imageio")
        return 1

    env_config = build_env_config({}, split="test")
    env = ProceduralNavEnv(env_config)
    actor = build_actor(args.actor, model_path=args.rl, robot=env_config.robot)

    print(f"Rendering {args.worlds} episode(s) with actor '{actor.name}'")
    frames: list = []
    for i in range(args.worlds):
        episode_frames, _ = render_episode(
            env, actor, args.world + i, args.stride, args.hold
        )
        frames.extend(episode_frames)

    if not frames:
        print("error: no frames rendered")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    size_mb = out.stat().st_size / 1e6
    print(f"\nWrote {out} ({len(frames)} frames, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
