"""The held-out worlds stay the worlds they were, and the confirmation blocks
are spent only by the test their ledger names."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from vision_nav.envs.nav_env import ProceduralNavEnv
from vision_nav.envs.splits import SEED_BANDS
from vision_nav.envs.world import generate_world
from vision_nav.training.env_factory import build_env_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "heldout_manifest.json"
BLOCKS = ("holdout_a", "holdout_b", "holdout_c")


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("heldout_manifest", ROOT / "scripts" / "heldout_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _shift(name):
    return None if name == "nominal" else name


@pytest.mark.parametrize("band, shift, seed", [("test", "nominal", 20000), ("test_ood", "dense", 30017),
                                               ("test_ood", "dynamic", 30003), ("holdout_a", "narrow", 40123),
                                               ("holdout_c", "large", 41999)])
def test_the_manifest_describes_the_worlds_the_env_builds(manifest, band, shift, seed):
    """Through the env itself, not only the generator the manifest calls."""
    m = _module()
    lo, hi = SEED_BANDS[band]
    env = ProceduralNavEnv(build_env_config({}, split=band, shift=_shift(shift), n_worlds=hi - lo))
    env.reset(options={"world_seed": seed})
    assert m.world_hash(env.world) == manifest["bands"][band]["shifts"][shift]["worlds"][str(seed)]


def test_a_spread_of_every_band_regenerates_its_hash(manifest):
    m = _module()
    for band, entry in manifest["bands"].items():
        lo, hi = SEED_BANDS[band]
        for shift, worlds in entry["shifts"].items():
            cfg = build_env_config({}, split=band, shift=_shift(shift), n_worlds=hi - lo)
            seeds = sorted(int(s) for s in worlds["worlds"])
            for seed in seeds[::97] + seeds[-1:]:
                world = generate_world(seed, cfg.domain_randomization.sample(cfg.world, seed))
                assert m.world_hash(world) == worlds["worlds"][str(seed)], (band, shift, seed)


def test_every_held_out_band_and_its_ledger_are_in_the_manifest(manifest):
    m = _module()
    assert set(manifest["bands"]) == set(m.SHIFTS_BY_BAND) == {"test", "test_ood", *BLOCKS}
    for band, entry in manifest["bands"].items():
        assert entry["spent_by"] == m.LEDGER[band], f"{band}: rewrite the manifest after editing the ledger"
        assert entry["seeds"] == list(SEED_BANDS[band])


def test_each_confirmation_block_is_spent_by_at_most_one_test():
    m = _module()
    for block in BLOCKS:
        assert len(m.LEDGER[block]) <= 1, f"{block} is spent by more than one test"


def test_no_result_holds_an_episode_on_a_block_its_ledger_does_not_name():
    m = _module()
    allowed = {b: {f for entry in m.LEDGER[b] for f in entry["results"]} for b in BLOCKS}
    seed = re.compile(r'"(?:world_)?seed":\s*(\d+)')
    for path in sorted((ROOT / "results").rglob("*.json*")):
        if path.name == "heldout_manifest.json" or path.name.endswith(".checkpoint.jsonl"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        found = {int(s) for s in seed.findall(path.read_text(encoding="utf-8", errors="ignore"))}
        for block in BLOCKS:
            lo, hi = SEED_BANDS[block]
            if any(lo <= s < hi for s in found):
                assert rel in allowed[block], f"{rel} holds episodes on {block}, which its ledger does not name"
