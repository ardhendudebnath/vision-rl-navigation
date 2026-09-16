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
import statistics as st
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


def marginal_fps(run_dir: Path) -> tuple[float, int]:
    """Steps/second between logger dumps, excluding the first iteration.

    The first dump includes process start, env construction and -- on CUDA --
    context creation, none of which are per-step costs.
    """
    path = run_dir / "logs" / "progress.csv"
    with open(path, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("time/total_timesteps") and r.get("time/time_elapsed")]
    steps = [int(r["time/total_timesteps"]) for r in rows]
    secs = [float(r["time/time_elapsed"]) for r in rows]
    rates = []
    for i in range(2, len(steps)):  # skip iteration 1 entirely
        dt = secs[i] - secs[i - 1]
        if dt > 0:
            rates.append((steps[i] - steps[i - 1]) / dt)
    if not rates:
        raise ValueError(f"not enough logger dumps in {path}; raise --steps")
    return st.median(rates), len(rates)


def run_cell(name: str, device: str, steps: int, seed: int) -> dict:
    run_name = f"bench_{name}_{device}"
    args = [sys.executable, "-m", "vision_nav.training.train", *CELLS[name],
            f"train.run_name={run_name}", f"train.seed={seed}",
            f"train.device={device}", f"train.total_timesteps={steps}",
            "train.progress_bar=false", "train.eval_freq=100000000",
            "train.checkpoint_freq=100000000"]
    print(f"  {name} on {device} ...", end="", flush=True)
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        print(" FAILED")
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
        return {"cell": name, "device": device, "error": r.stderr[-400:]}
    fps, n = marginal_fps(Path("runs", run_name))
    print(f" {fps:.0f} steps/s (median of {n} intervals)")
    return {"cell": name, "device": device, "fps": fps, "intervals": n}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", nargs="+", default=["mlp", "mlp128", "lstm"],
                   choices=sorted(CELLS))
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=150_000,
                   help="per cell; needs enough logger dumps to take a median")
    p.add_argument("--seed", type=int, default=0)
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
            rows.append(run_cell(cell, device, args.steps, args.seed))

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
