"""Run independent episodes in parallel, keeping each one the moment it finishes.

A registered run that writes its results only at the end loses everything to
one interruption, and two have been lost that way: a session ending stops the
process, and the hour of episodes already driven goes with it. Here each
finished episode is appended to a checkpoint file at once, and a rerun of the
same command drives only the episodes the file does not already hold.

That is safe because episodes are independent and deterministic: each builds
its world and agent from its own job, so skipping one cannot change another,
and a resumed run is the same run. Rows come back in job order, each passed
through JSON whether it was driven now or read back, so a resumed run and an
uninterrupted one hand the analysis identical rows.

    rows = run_resumable(episode, jobs, workers, checkpoint_for(out))
    out.write_text(...)
    checkpoint_for(out).unlink(missing_ok=True)  # finished: nothing to resume
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def checkpoint_for(out: str | Path) -> Path:
    """Where a run writing ``out`` keeps its finished episodes meanwhile."""
    out = Path(out)
    return out.with_name(out.stem + ".checkpoint.jsonl")


def _key(job) -> str:
    return json.dumps(list(job) if isinstance(job, tuple) else job)


def load(checkpoint: Path) -> dict[str, dict]:
    """Episodes a checkpoint already holds. A last line cut off mid-write by
    the interruption is dropped: that episode is simply driven again."""
    done: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[rec["key"]] = rec["row"]
    return done


def run_resumable(fn: Callable, jobs: Sequence, workers: int, checkpoint: Path,
                  log: Callable[[str], None] | None = print) -> list:
    """``[fn(job) for job in jobs]``, in parallel, resuming from ``checkpoint``.

    ``fn`` must be a module-level function (worker processes import it) whose
    result is JSON-serialisable. ``workers`` of 1 or fewer runs in this process.
    """
    done = load(checkpoint)
    todo = [j for j in jobs if _key(j) not in done]
    if done and log:
        log(f"resuming: {len(jobs) - len(todo)} of {len(jobs)} episodes already in {checkpoint}")
    if todo:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("a", encoding="utf-8") as f:

            def keep(job, row) -> None:
                line = json.dumps({"key": _key(job), "row": row})
                f.write(line + "\n")
                f.flush()
                done[_key(job)] = json.loads(line)["row"]

            if workers <= 1:
                for job in todo:
                    keep(job, fn(job))
            else:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(fn, job): job for job in todo}
                    for fut in as_completed(futures):
                        keep(futures[fut], fut.result())
    # Rows driven in this call went through JSON on the way into ``done``, as
    # read-back ones did, so both kinds are the same kind of object.
    return [done[_key(j)] for j in jobs]
