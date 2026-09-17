"""Does the encoder cost survive the 1:1 reward?

Phase 3e held information constant -- a CNN reading a 64x48 render of the same
geometry a 64-column depth vector measures, same 90 degree FOV -- and found
reading pixels costs 0.16-0.24 success. That was measured under the 4:1 reward
alone. Section 8.3 re-priced the other two perception headlines at 1:1 and they
came apart: coverage's effect survived while its mechanism moved (Phase 5l), and
resolution's collision effect reversed sign while success moved in neither
(Phase 5j). This re-prices the third.

Twelve runs, seed-paired with Phase 3e's (seeds 0-5), identical to them except
``collision_penalty`` 5, which makes a crash cost exactly a full 500-step
timeout:

  depthi -- env=nav_depth,                          collision_penalty 5
  rgbi   -- env=nav_rgb algo=ppo_cnn, on CUDA,      collision_penalty 5

Scored on narrow, dense and nominal at 100 worlds per seed with
``seed_analysis.py``, then against Phase 3e's result file with
``repricing_interaction.py`` -- the difference of the two per-seed deltas.

    python scripts/encoder_repricing.py train --parallel 6
    python scripts/encoder_repricing.py analyse
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repricing_interaction  # noqa: E402
import seed_analysis  # noqa: E402

SEEDS = range(6)
PENALTY = "env.reward.collision_penalty=5"
#: RGB first: it is the long pole (about an hour a batch against half that).
#: Devices match Phase 3e's runs, rgb_s* on CUDA and depth_s* on CPU. The RGB
#: arm has to say so: ``auto`` resolves to CPU, and a smoke test of this command
#: without the override trained the CNN there.
ARMS = {
    "rgbi": ["env=nav_rgb", "algo=ppo_cnn", "train.device=cuda"],
    "depthi": ["env=nav_depth"],
}
CONDITIONS = ("narrow", "dense", "nominal")
FOUR_TO_ONE = "results/seed_3e_rgb_vs_depth.json"
ONE_TO_ONE = "results/repricing_encoder.json"
INTERACTION = "results/repricing_encoder_interaction.json"

#: Recorded before any of the twelve runs started, and committed to the
#: repository before training began.
#:
#: Confidence LOW. Both perception contrasts re-priced so far left success where
#: it was -- coverage +0.095 -> +0.080 (interaction p = 0.77), resolution
#: -0.003 -> +0.032 (p = 0.26) -- which is the basis for predicting invariance
#: here. But neither involved a learned image encoder, and the one re-pricing of
#: an extra *learned* channel, frame stacking, did change success: inert at 4:1,
#: +0.033 at 1:1. Nothing has measured a CNN arm at 1:1, so by this project's
#: calibration rule this is a crossing into unmeasured territory, and an analogy
#: across a different kind of deficit.
#:
#: The mechanism clause is closer to deduction: at 4:1 the narrow cost splits
#: evenly between collisions (+0.105) and timeouts (+0.113), the RGB seeds
#: dividing into crashers and stallers (collision 0.02-0.45, timeout 0.14-0.69),
#: and Phase 5j measured timeouts collapsing under 1:1 -- for MLP arms. If they
#: collapse for the CNN too, the timeout half has nowhere to go but collisions
#: or successes; invariance says collisions.
#:
#: Six seeds a side resolve an interaction of roughly 0.1, not less, so
#: "invariant" below means no change of a third of the cost or more.
PREDICTION = (
    "the encoder cost survives re-pricing: rgbi minus depthi success is negative "
    "at p < 0.05 on narrow and dense, and within -0.13 to -0.30 on narrow. "
    "Success interaction (1:1 delta minus 4:1 delta) inside +/-0.08 with p >= 0.05 "
    "on narrow and dense. Mechanism moves: at 1:1 the narrow timeout delta is "
    "inside +/-0.05 and the collision delta is at least +0.15. "
    "Decision -- SURVIVES: significant on both narrow and dense. "
    "SHRINKS: success interaction positive with p < 0.05 on either. "
    "GROWS: negative with p < 0.05 on either. Otherwise AMBIGUOUS."
)


def run_name(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


def train(parallel: int) -> int:
    queue = [(arm, s) for arm in ARMS for s in SEEDS
             if not (Path("runs") / run_name(arm, s) / "final_model.zip").exists()]
    running: list[tuple[str, subprocess.Popen]] = []
    failed = []
    while queue or running:
        while queue and len(running) < parallel:
            arm, seed = queue.pop(0)
            name = run_name(arm, seed)
            log = open(Path("runs") / f"{name}.out", "w", encoding="utf-8")  # noqa: SIM115
            cmd = [sys.executable, "-m", "vision_nav.training.train", *ARMS[arm], PENALTY,
                   f"train.run_name={name}", f"train.seed={seed}"]
            running.append((name, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
            print(f"started {name}", flush=True)
        time.sleep(30)
        for name, proc in list(running):
            if proc.poll() is not None:
                running.remove((name, proc))
                print(f"finished {name}: exit {proc.returncode}", flush=True)
                if proc.returncode:
                    failed.append(name)
    print("all runs finished" + (f"; FAILED: {failed}" if failed else ""), flush=True)
    return 1 if failed else 0


def analyse(episodes: int) -> int:
    arms = [f"{arm}=" + ",".join(f"runs/{run_name(arm, s)}" for s in SEEDS)
            for arm in ("depthi", "rgbi")]
    seed_analysis.main([a for spec in arms for a in ("--arm", spec)]
                       + ["--conditions", *CONDITIONS, "--episodes", str(episodes),
                          "--out", ONE_TO_ONE])
    repricing_interaction.main(["--four-to-one", FOUR_TO_ONE, "--one-to-one", ONE_TO_ONE,
                                "--arms-4to1", "depth", "rgb", "--arms-1to1", "depthi", "rgbi",
                                "--conditions", *CONDITIONS, "--out", INTERACTION])

    lo = json.loads(Path(ONE_TO_ONE).read_text(encoding="utf-8"))["conditions"]
    inter = json.loads(Path(INTERACTION).read_text(encoding="utf-8"))["conditions"]
    key = ("narrow", "dense")
    significant = [lo[c]["comparisons"]["success"]["p"] < 0.05
                   and lo[c]["comparisons"]["success"]["delta"] < 0 for c in key]
    moves = [(inter[c]["success"]["interaction"], inter[c]["success"]["p"]) for c in key]
    if any(x > 0 and p < 0.05 for x, p in moves):
        decision = "SHRINKS"
    elif any(x < 0 and p < 0.05 for x, p in moves):
        decision = "GROWS"
    elif all(significant):
        decision = "SURVIVES"
    else:
        decision = "AMBIGUOUS"
    report = json.loads(Path(INTERACTION).read_text(encoding="utf-8"))
    report.update(prediction=PREDICTION, decision=decision)
    Path(INTERACTION).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ndecision (registered rule): {decision}")
    print("pre-registered: " + PREDICTION)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--parallel", type=int, default=6)
    a = sub.add_parser("analyse")
    a.add_argument("--episodes", type=int, default=100)
    args = p.parse_args(argv)
    return train(args.parallel) if args.cmd == "train" else analyse(args.episodes)


if __name__ == "__main__":
    raise SystemExit(main())
