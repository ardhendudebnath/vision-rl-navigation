"""Does CPU actually beat GPU here, or was that only ever asserted?

``resolve_device`` maps ``auto`` to CPU for every policy in this project, on
the stated grounds that "for a small MLP policy on a fast CPU-side simulator,
the per-batch host-device transfer costs more than the matmuls save". That is
a plausible claim about a real trade-off, and it was never measured. It is
also load-bearing in two directions: it silently caps every MLP run, and it
sent the recurrent arm to CPU at 78 steps/second for 5.3 hours a seed.

Each cell trains the same configuration for a fixed budget on one device and
reports **marginal** throughput -- steps per second computed between
consecutive logger dumps, discarding the first iteration. Cumulative fps would
charge CUDA's several-second context init against the GPU and, at these
budgets, that alone could decide the answer.

    python scripts/device_benchmark.py
    python scripts/device_benchmark.py --steps 200000 --cells mlp lstm

Runs cells one at a time: two training processes on one machine measure
contention, not devices.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

#: name -> Hydra overrides selecting the policy class being timed.
CELLS = {
    "mlp": ["env=nav_dr"],
    "mlp128": ["env=nav_dr", "env.lidar.n_beams=128"],
    "lstm": ["env=nav_dyn_fast", "algo=ppo_lstm"],
    "cnn": ["env=nav_rgb"],
}


def marginal_fps(run_dir: Path) -> tuple[float, float]:
    """Steps/second over the whole run, excluding the first iteration.

    The first dump includes process start, env construction and -- on CUDA --
    context creation, none of which are per-step costs, so it is dropped.

    Rated across the entire remaining span rather than per interval, because
    SB3 logs ``time/time_elapsed`` as an integer number of seconds. At ~8 s an
    iteration, a per-interval rate can only take the values 16384/8 = 2048,
    16384/9 = 1820, 16384/10 = 1638 ... and a median over those locks onto one
    of them. The first version of this script did exactly that and reported the
    MLP and the 128-beam MLP as *identical* to four figures, which is the
    quantisation showing through rather than a result. Rating one long span
    applies the same one-second uncertainty to a much larger denominator.

    Returns the rate and the span in seconds, so the resolution is auditable:
    the relative error is about 1/span.
    """
    path = run_dir / "logs" / "progress.csv"
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("time/total_timesteps") and r.get("time/time_elapsed")]
    steps = [int(r["time/total_timesteps"]) for r in rows]
    secs = [float(r["time/time_elapsed"]) for r in rows]
    if len(steps) < 3:
        raise ValueError(f"not enough logger dumps in {path}; raise --steps")
    span = secs[-1] - secs[1]
    if span <= 0:
        raise ValueError(f"degenerate timing span in {path}; raise --steps")
    return (steps[-1] - steps[1]) / span, span


def run_cell(name: str, device: str, steps: int, seed: int,
             reuse: bool = False) -> dict:
    run_name = f"bench_{name}_{device}"
    run_dir = Path("runs", run_name)
    print(f"  {name} on {device} ...", end="", flush=True)

    if reuse and (run_dir / "logs" / "progress.csv").exists():
        print(" (reusing timings)", end="")
    else:
        args = [sys.executable, "-m", "vision_nav.training.train", *CELLS[name],
                f"train.run_name={run_name}", f"train.seed={seed}",
                f"train.device={device}", f"train.total_timesteps={steps}",
                "train.progress_bar=false", "train.eval_freq=100000000",
                "train.checkpoint_freq=100000000"]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            print(" FAILED")
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            return {"cell": name, "device": device, "error": r.stderr[-400:]}

    fps, span = marginal_fps(run_dir)
    print(f" {fps:.0f} steps/s (over {span:.0f}s, ~{100 / span:.1f}% resolution)")
    return {"cell": name, "device": device, "fps": fps, "span_seconds": span,
            "resolution_pct": 100 / span}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", nargs="+", default=["mlp", "mlp128", "lstm"],
                   choices=sorted(CELLS))
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=150_000,
                   help="per cell; the timing span must be long enough that a one-second clock resolves it")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse", action="store_true",
                   help="re-analyse existing bench runs instead of re-timing")
    p.add_argument("--out", default="results/device_benchmark.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        import torch
        cuda = torch.cuda.is_available()
        gpu = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        cuda, gpu = False, None
    if "cuda" in args.devices and not cuda:
        print("CUDA not available; timing CPU only")
        args.devices = [d for d in args.devices if d != "cuda"]

    print(f"GPU: {gpu or 'none'}   budget: {args.steps:,} steps per cell\n")
    rows = []
    for cell in args.cells:
        for device in args.devices:
            rows.append(run_cell(cell, device, args.steps, args.seed, args.reuse))

    print(f"\n{'cell':10s} {'cpu':>10s} {'cuda':>10s} {'speedup':>10s}")
    summary = {}
    for cell in args.cells:
        got = {r["device"]: r.get("fps") for r in rows if r["cell"] == cell}
        c, g = got.get("cpu"), got.get("cuda")
        ratio = (g / c) if (c and g) else None
        summary[cell] = {"cpu_fps": c, "cuda_fps": g, "cuda_speedup": ratio}
        note = f"{ratio:.2f}x" if ratio else "n/a"
        print(f"{cell:10s} {c or 0:10.0f} {g or 0:10.0f} {note:>10s}")
        if ratio and ratio < 1:
            print(f"           -> CPU wins by {1 / ratio:.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"gpu": gpu, "steps_per_cell": args.steps, "seed": args.seed,
         "runs": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
