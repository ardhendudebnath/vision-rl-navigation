"""Does the trained LSTM actually use its memory?

Phase 5i found the recurrent arm worse than the memoryless one everywhere.
That has two very different causes and the endpoint cannot separate them:

  (a) the LSTM uses its memory and the memory is unhelpful on this task, or
  (b) the LSTM learned to route around its memory, so the whole cost is the
      architecture -- more parameters, harder optimisation -- and the
      recurrence itself is irrelevant.

The test is a subtraction of exactly one thing, the same shape as the
frozen-mover experiment: evaluate each policy twice on identical worlds, once
carrying recurrent state across the episode and once clearing it at every step.
Identical scores mean the policy's output does not depend on its history and
(b) holds; different scores mean memory is genuinely in the loop and (a) does.

It doubles as a check on the evaluation path. Carrying recurrent state through
SB3Actor had to be added for this experiment, and this measures what skipping
it would have cost -- which is the difference between a real result and a
catastrophic-looking artefact pointing the same way.

    python scripts/recurrence_ablation.py
    python scripts/recurrence_ablation.py --runs runs/dynfastrec_s0 --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vision_nav.training.actors import SB3Actor, build_actor
from vision_nav.training.env_factory import build_env_config
from vision_nav.training.evaluate import evaluate
from vision_nav.training.run_spec import env_overrides_for_run

__all__ = ["AmnesiacActor", "main"]


class AmnesiacActor(SB3Actor):
    """Identical to SB3Actor except recurrent state never survives a step.

    Every step is presented to the policy as the first of an episode, which is
    precisely the bug SB3Actor exists to prevent -- reproduced deliberately
    here as the control arm.
    """

    def act(self, env, obs):
        action = super().act(env, obs)
        self._state = None
        self._episode_start = True
        return action


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+",
                   default=[f"runs/dynfastrec_s{i}" for i in range(6)])
    p.add_argument("--shift", default="dynamic_fast")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default="results/recurrence_state_ablation.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rows = []
    for run in (Path(r) for r in args.runs):
        cfg = build_env_config(env_overrides_for_run(run), split="test_ood",
                               shift=args.shift, n_worlds=args.episodes)
        model_path = str(run / "best_model.zip")

        with_memory = build_actor("rl", model_path=model_path, robot=cfg.robot)
        if type(with_memory.model).__name__ != "RecurrentPPO":
            raise ValueError(
                f"{run} is not a recurrent policy ({type(with_memory.model).__name__}); "
                "this ablation is meaningless without one"
            )
        m_with, _ = evaluate(with_memory, cfg)
        m_without, _ = evaluate(AmnesiacActor(with_memory.model), cfg)

        rows.append({
            "run": run.name,
            "with_memory": m_with.success_rate,
            "without_memory": m_without.success_rate,
            "delta": m_with.success_rate - m_without.success_rate,
        })
        print(f"{run.name}: with {m_with.success_rate:.3f}  "
              f"without {m_without.success_rate:.3f}  "
              f"delta {rows[-1]['delta']:+.3f}", flush=True)

    deltas = np.array([r["delta"] for r in rows])
    print(f"\nmean delta (memory minus no memory): {deltas.mean():+.3f} "
          f"+/- {deltas.std(ddof=1):.3f}" if len(deltas) > 1 else
          f"\ndelta: {deltas.mean():+.3f}")
    changed = int((np.abs(deltas) > 1e-9).sum())
    print(f"seeds where memory changed the score: {changed}/{len(deltas)}")
    if not np.all(np.abs(deltas) < 0.005):
        print("MEMORY IS IN THE LOOP: clearing state changes the score, so the "
              "policy's actions genuinely depend on its history.")
    elif args.episodes >= 100 and len(deltas) >= 3:
        print("MEMORY IS INERT: clearing state every step changes nothing, so "
              "the cost is the architecture rather than the recurrence.")
    else:
        # Caught by running this at --episodes 10, where both arms landed on
        # 0.500 and the inert verdict duly fired -- on a seed that differs by
        # 0.270 at 100 episodes. Success rate is quantised at 1/episodes, so a
        # small run cannot resolve the thing this verdict asserts.
        print(f"NO DIFFERENCE DETECTED, but {args.episodes} episodes x "
              f"{len(deltas)} seed(s) cannot resolve one: success rate is "
              f"quantised at {1 / args.episodes:.2f}. This is not evidence of "
              f"inertness. Re-run with >= 100 episodes and >= 3 seeds.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"episodes": args.episodes, "shift": args.shift, "per_seed": rows,
         "mean_delta": float(deltas.mean())}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
