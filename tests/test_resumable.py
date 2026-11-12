"""A run that keeps each episode as it finishes, and resumes where it stopped."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("resumable", ROOT / "scripts" / "resumable.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _double(job):
    return {"job": list(job), "value": job[0] * 2, "half": job[0] / 2}


def _refuse_first_two(job):
    if job[0] < 2:
        raise AssertionError(f"episode {job} was driven again after it had been kept")
    return _double(job)


def test_episodes_come_back_in_job_order(tmp_path):
    m = _module()
    jobs = [(i, "dense") for i in range(5)]
    rows = m.run_resumable(_double, jobs, 1, tmp_path / "x.checkpoint.jsonl", log=None)
    assert [r["value"] for r in rows] == [0, 2, 4, 6, 8]


def test_a_rerun_drives_only_what_the_checkpoint_lacks(tmp_path):
    m = _module()
    ckpt = tmp_path / "x.checkpoint.jsonl"
    jobs = [(i, "dense") for i in range(4)]
    first = m.run_resumable(_double, jobs[:2], 1, ckpt, log=None)  # then interrupted
    resumed = m.run_resumable(_refuse_first_two, jobs, 1, ckpt, log=None)
    whole = m.run_resumable(_double, jobs, 1, tmp_path / "y.checkpoint.jsonl", log=None)
    assert resumed[:2] == first
    assert resumed == whole, "a resumed run must hand back what an uninterrupted one does"


def test_a_line_cut_off_by_the_interruption_is_driven_again(tmp_path):
    m = _module()
    ckpt = tmp_path / "x.checkpoint.jsonl"
    m.run_resumable(_double, [(0, "a")], 1, ckpt, log=None)
    with ckpt.open("a", encoding="utf-8") as f:
        f.write('{"key": "[1, \\"a\\"]", "row": {"val')  # killed mid-write
    rows = m.run_resumable(_double, [(0, "a"), (1, "a")], 1, ckpt, log=None)
    assert [r["value"] for r in rows] == [0, 2]


def test_the_checkpoint_sits_beside_the_output_and_is_named_for_it():
    m = _module()
    assert m.checkpoint_for("results/plan_buffer_test.json") == Path("results/plan_buffer_test.checkpoint.jsonl")


@pytest.mark.parametrize("workers", [1, 2])
def test_rows_are_json_whatever_their_route(tmp_path, workers):
    m = _module()
    rows = m.run_resumable(_double, [(3, "n")], workers, tmp_path / "z.checkpoint.jsonl", log=None)
    assert rows == [json.loads(json.dumps(_double((3, "n"))))]
