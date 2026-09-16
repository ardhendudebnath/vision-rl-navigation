"""What do the episodes the estimate lost have in common? Post hoc.

``estimate_experiment.py`` found constant-velocity estimation losing 14 dense
and 4 sparse episodes the oracle won, through collisions, unrecovered by a wider
temporal margin. This replays exactly those episodes, instrumented, and records
at the moment of contact:

  - whether the robot touched a mover or a static obstacle,
  - how far the estimate the plan was built on was from the truth, for the
    mover touched,
  - the inflation radius the plan in force was made at -- the agent tries
    robot + margin, robot + margin/2, then the bare robot radius -- and
  - the share of plans made at a reduced radius *before* the last two seconds,
    so the fallback a nearby mover forces at the end is not counted as a cause,
    beside the same share for the oracle agent on the same worlds.

Nothing here was predicted, and nothing here is a test. It is a description of
eighteen episodes, written down so the next experiment can be registered
against it rather than against a guess.

    python scripts/estimate_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from velocity_experiment import DYNAMIC_CONDITIONS  # noqa: E402

from vision_nav.agents.spacetime import SpaceTimeAgent, SpaceTimeConfig  # noqa: E402
from vision_nav.envs.nav_env import ProceduralNavEnv  # noqa: E402
from vision_nav.training.env_factory import build_env_config  # noqa: E402

#: Seconds before the end of an episode whose plans are not counted as early.
LATE_S = 2.0


class Probe(SpaceTimeAgent):
    """The experiment's agent, recording each successful plan's radius, time and
    the observations its estimate was built from. Behaviour is unchanged."""

    def start_episode(self, world, pose):
        self.plan_radii, self.plan_times, self.plan_obs = [], [], None
        return super().start_episode(world, pose)

    def _mover_occupancy(self, radius, steps):
        self._tried_radius = radius
        return super()._mover_occupancy(radius, steps)

    def _plan(self, pose):
        observed = [(t, p.copy()) for t, p in self._observations]
        ok = super()._plan(pose)
        if ok:
            self.plan_radii.append(self._tried_radius)
            self.plan_times.append(self._world._t)
            self.plan_obs = observed
        return ok


def replay(cond: str, predictor: str, episodes: set[int], n_worlds: int) -> dict[int, dict]:
    split, shift, _ = DYNAMIC_CONDITIONS[cond]
    cfg = build_env_config({}, split=split, shift=shift, n_worlds=n_worlds)
    env = ProceduralNavEnv(cfg)
    out = {}
    for i, seed in enumerate(cfg.world_seeds):
        if i not in episodes:
            continue
        env.reset(options={"world_seed": int(seed)})
        agent = Probe(SpaceTimeConfig(predictor=predictor, temporal_margin_steps=2),
                      robot=cfg.robot)
        agent.start_episode(env.world, env.robot.pose)
        done, info = False, {}
        while not done:
            _, _, term, trunc, info = env.step(agent.act(env.robot.pose))
            done = term or trunc
        w, rr = env.world, env.world.config.robot_radius
        full = rr + agent.config.safety_margin
        reduced = [r < full - 1e-9 for r in agent.plan_radii]
        early = [x for x, t in zip(reduced, agent.plan_times, strict=True) if t < w._t - LATE_S]
        rec = {"success": bool(info.get("is_success")), "collision": bool(info.get("collision")),
               "reduced_share_early": float(np.mean(early)) if early else None}
        if rec["collision"]:
            pos = env.robot.pose[:2]
            static = float(w.clearance(pos[None, :], include_dynamic=False)[0]) - rr
            gaps = np.linalg.norm(w._dyn_now[:, :2] - pos, axis=1) - w._dyn_now[:, 2] - rr
            j = int(np.argmin(gaps))
            (t0, p0), (t1, p1) = agent.plan_obs[-2], agent.plan_obs[-1]
            velocity = (p1[j, :2] - p0[j, :2]) / (t1 - t0) if t1 > t0 else np.zeros(2)
            estimate = p1[j, :2] + velocity * (w._t - t1)
            rec.update(
                contact="mover" if gaps[j] < static else "static",
                estimate_error_m=float(np.linalg.norm(estimate - w._dyn_now[j, :2])),
                plan_radius=float(agent.plan_radii[-1]),
                plan_at_bare_radius=bool(agent.plan_radii[-1] <= rr + 1e-9),
            )
        out[i] = rec
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="results/estimate_experiment.json")
    p.add_argument("--out", default="results/estimate_diagnostic.json")
    args = p.parse_args(argv)

    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    report = {"late_s": LATE_S, "conditions": {}}
    errors, bare, contacts = [], 0, []
    for cond in ("dynamic", "dynamic_dense"):
        oracle = src["cells"]["oracle_m2"][cond]["success_per_episode"]
        estimate = src["cells"]["cv_m2"][cond]["success_per_episode"]
        lost = {i for i, (a, b) in enumerate(zip(oracle, estimate, strict=True)) if a and not b}
        cv = replay(cond, "constant_velocity", lost, src["episodes"])
        ora = replay(cond, "oracle", lost, src["episodes"])
        # The replay must be the experiment's episodes, or none of this describes them.
        assert all(not cv[i]["success"] and ora[i]["success"] for i in lost), cond
        rows = {i: {"estimate": cv[i], "oracle_reduced_share": ora[i]["reduced_share_early"]}
                for i in sorted(lost)}
        early_cv = [r["estimate"]["reduced_share_early"] for r in rows.values()]
        report["conditions"][cond] = {
            "episodes": rows,
            "lost": len(lost),
            "no_early_fallback": sum(1 for x in early_cv if x == 0.0),
            "median_reduced_share_early_estimate": float(np.median([x or 0.0 for x in early_cv])),
            "median_reduced_share_oracle": float(np.median(
                [r["oracle_reduced_share"] or 0.0 for r in rows.values()])),
        }
        for r in rows.values():
            e = r["estimate"]
            contacts.append(e.get("contact"))
            if e["collision"]:
                errors.append(e["estimate_error_m"])
                bare += e["plan_at_bare_radius"]
        c = report["conditions"][cond]
        print(f"{cond}: {c['lost']} lost; early reduced-radius share, median "
              f"{c['median_reduced_share_early_estimate']:.3f} estimate vs "
              f"{c['median_reduced_share_oracle']:.3f} oracle; "
              f"{c['no_early_fallback']} with no early fallback")

    report["summary"] = {
        "lost": len(contacts),
        "mover_contacts": contacts.count("mover"),
        "median_estimate_error_m": float(np.median(errors)),
        "max_estimate_error_m": float(np.max(errors)),
        "plans_at_bare_radius": int(bare),
    }
    print(json.dumps(report["summary"], indent=2))
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
