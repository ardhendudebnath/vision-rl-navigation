"""Freeze the held-out worlds: a hash of every one, and who has spent which block.

A held-out result means something only while the worlds it was measured on are
the worlds a later reader regenerates. World generation is code, and code
changes: a change to obstacle sampling would redraw every held-out world, and
no result file would notice. This records, for every held-out band and every
shift the report scores on it, a hash of each world -- its obstacles, start,
goal, movers and world configuration -- and checks them.

It also keeps the ledger for the blocks reserved for confirmation
(``holdout_a``, ``holdout_b``, ``holdout_c``, see ``vision_nav.envs.splits``).
A block is spent by exactly one registered test, entered in ``LEDGER`` in the
commit that registers the test, before it drives a world of the block.
``tests/test_heldout_manifest.py`` fails if a result file holds an episode on a
block whose ledger does not name that file, and spot-checks the hashes.

    python scripts/heldout_manifest.py            # write results/heldout_manifest.json
    python scripts/heldout_manifest.py --check    # regenerate every world and compare
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from vision_nav.envs.splits import SEED_BANDS
from vision_nav.envs.world import generate_world
from vision_nav.training.env_factory import build_env_config

MANIFEST = Path("results/heldout_manifest.json")
#: The shifts each held-out band is scored under. ``nominal`` and
#: ``noisy_lidar`` share worlds (noise is the sensor's), as do each dynamic
#: condition and its frozen twin (freezing happens after generation).
SHIFTS_BY_BAND = {
    "test": ("nominal",),
    "test_ood": ("dense", "sparse", "large", "narrow", "dynamic", "dynamic_dense"),
    **{b: ("nominal", "dense", "sparse", "large", "narrow", "dynamic", "dynamic_dense")
       for b in ("holdout_a", "holdout_b", "holdout_c")},
}
#: Who has spent each band. The published bands were spent by the report as a
#: whole; each confirmation block by at most one registered test, named with its
#: script and the result files it writes.
LEDGER: dict[str, object] = {
    "test": "the published held-out results (report sections 4 to 9)",
    "test_ood": "the published held-out results (report sections 4 to 9)",
    "holdout_a": [],
    "holdout_b": [],
    "holdout_c": [],
}


def world_hash(world) -> str:
    """A short, stable digest of everything about a world an episode sees."""
    h = hashlib.sha256()
    for a in (world.circles, world.boxes, world.start, world.goal, world.dynamic):
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    h.update(json.dumps(dataclasses.asdict(world.config), sort_keys=True, default=repr).encode())
    return h.hexdigest()[:16]


def hashes(job: tuple) -> tuple[str, str, dict[str, str]]:
    """Every world of one band under one shift, as the env would generate it."""
    band, shift = job
    lo, hi = SEED_BANDS[band]
    cfg = build_env_config({}, split=band, shift=None if shift == "nominal" else shift, n_worlds=hi - lo)
    out = {}
    for seed in range(lo, hi):
        world = generate_world(seed, cfg.domain_randomization.sample(cfg.world, seed))
        out[str(seed)] = world_hash(world)
    return band, shift, out


def build(workers: int) -> dict:
    jobs = [(band, shift) for band, shifts in SHIFTS_BY_BAND.items() for shift in shifts]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(hashes, jobs))
    bands: dict = {b: {"seeds": list(SEED_BANDS[b]), "spent_by": LEDGER[b], "shifts": {}}
                   for b in SHIFTS_BY_BAND}
    for band, shift, worlds in results:
        digest = hashlib.sha256("".join(worlds[s] for s in sorted(worlds, key=int)).encode()).hexdigest()
        bands[band]["shifts"][shift] = {"digest": digest, "worlds": worlds}
    return {"note": __doc__.strip().splitlines()[0], "bands": bands}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="regenerate every world and compare")
    p.add_argument("--workers", type=int, default=10)
    args = p.parse_args(argv)
    fresh = build(args.workers)
    if not args.check:
        MANIFEST.write_text(json.dumps(fresh, indent=1), encoding="utf-8")
        n = sum(len(s["worlds"]) for b in fresh["bands"].values() for s in b["shifts"].values())
        print(f"Wrote {MANIFEST}: {n} worlds over {len(fresh['bands'])} bands")
        return 0
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = [(b, s, seed) for b, band in stored["bands"].items() for s, shift in band["shifts"].items()
               for seed, h in shift["worlds"].items()
               if fresh["bands"].get(b, {}).get("shifts", {}).get(s, {}).get("worlds", {}).get(seed) != h]
    print(f"{len(changed)} held-out worlds differ from the manifest" + (f": first {changed[:5]}" if changed else ""))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
