# Learning Vision-Conditioned Navigation Policies

### A comparative study against classical planning

| Classical (A* + pure pursuit) | PPO (privileged) |
|---|---|
| ![Classical baseline](results/demo_classical.gif) | ![PPO policy](results/demo_rl.gif) |

*The same three unseen test worlds, solved by both. Grey: the global plan
(classical only — the learned policy has no plan to draw). Orange: the
executed trajectory.*

---

A reinforcement-learning navigation agent, a classical planning stack, and a
protocol strict enough that comparing them means something.

The question is not "can an RL agent reach a goal" — it is **where a learned
policy beats a strong classical planner, where it loses, and how each one
degrades when the world stops looking like the training set.** Every design
decision below follows from wanting that comparison to be trustworthy.

## Status

| Stage | State |
|---|---|
| Task definition, world generation, metrics, test suite | Done |
| Classical baseline (A* + pure pursuit, full map access) | Done |
| Privileged RL (PPO on pose + ranges) | Done — 1.5M steps, val SPL 0.894 |
| Robustness suite across shifted environments | Done |
| Domain randomisation over shifts | Done — did not close the gap, see below |
| Vision-conditioned RL (egocentric observations + CNN) | Next |

Detail and rationale: [`docs/project_plan.md`](docs/project_plan.md).

## Results

100 held-out worlds per condition. All actors see **identical worlds in
identical order**. `nominal` is the in-distribution held-out split; the rest
are distribution shifts. Success rate / SPL:

| Condition | Classical | PPO (nominal) | PPO (domain-rand.) | Random |
|---|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 | 0.940 / 0.865 | 0.000 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 | 0.990 / 0.951 | 0.010 |
| large | **1.000** / 0.990 | 0.970 / 0.940 | 0.970 / 0.925 | 0.000 |
| noisy_lidar | 1.000 / 0.985 * | 0.960 / 0.909 | 0.940 / 0.869 | 0.000 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 | 0.660 / 0.593 | 0.000 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 | 0.630 / 0.563 | 0.000 |

\* `noisy_lidar` is a no-op for the classical planner *by construction* — it
navigates from the map and never reads the lidar. Non-exposure, not
robustness.

Full table: [`results/benchmark.md`](results/benchmark.md).

### Two findings, both negative, and the second is the interesting one

**1. A learned policy did not beat a strong classical planner.** The project's
stated hypothesis was that a reactive policy, not committed to a precomputed
path, would close the planner's cornering weakness under clutter. That
weakness was measured precisely: under shift, **A\* never once fails to find a
route** — 0 planning failures, with 9/11 and 12/15 of failures being the
controller losing the plan while cornering. The hypothesis is **falsified**.
Classical wins on every condition, and the gap *widens* exactly where RL was
predicted to win.

**2. Domain randomisation did not close that gap either — it converted
collisions into timeouts.** The obvious explanation for finding 1 was
out-of-distribution brittleness, so a second policy was trained on worlds
randomised across arena size, obstacle count, obstacle size and start-goal
separation, with ranges chosen to *contain* every evaluation shift. Success
rate barely moved (0.640 → 0.660 on `dense`, 0.600 → 0.630 on `narrow`; SPL
essentially identical). But the failure *composition* changed completely:

| `narrow`, 100 episodes | Collisions | Timeouts | Timeout progress along route |
|---|---|---|---|
| PPO (nominal) | 27 | 13 | 4.4 m of 10.6 m |
| PPO (domain-rand.) | **11** | **26** | **7.5 m of 11.7 m** |

Randomisation taught genuine obstacle avoidance — collisions more than halved,
and the episodes that fail now get roughly two-thirds of the way instead of
40%. But the policy pays for that safety in speed: its timeouts average about
0.16 m/s against a 0.6 m/s cap, and it runs out of budget still ~7 m from the
goal. It became **cautious rather than capable**.

So the `dense`/`narrow` collapse is *not* simply distribution mismatch. Under
clutter this policy trades collisions for stalling, and widening the training
distribution moves failures between buckets rather than into successes.

### Caveats that govern these numbers

- **The comparison is structurally asymmetric.** Both policies train on a
  distribution; the classical planner has none. On the shifted conditions this
  compares an in-distribution planner against an out-of-distribution policy.
  That is inherent to comparing learned and non-learned systems, and it is
  most of the explanation for finding 1.
- **The DR ranges were chosen to contain the evaluation shifts.** So finding 2
  tests whether widening the training distribution recovers the loss — *not*
  generalisation to unseen kinds of shift. Evaluation worlds are still
  held-out seeds, so this is not leakage in the memorisation sense, but the
  distribution is deliberately matched.
- **The DR policy is compute-limited, the nominal one is not.** The nominal
  run flatlined from ~900k steps; the DR run was still improving at 1.5M
  (0.74 → 0.76). Finding 2 should be read as "not at this budget", and the
  cheapest next experiment is simply training it longer.
- **n = 100 per condition**, so differences under ~0.05 in success rate are
  not resolvable.

### What the learned policies do win

Both small, both stated with their caveats: ~6% fewer steps on successful
`nominal` episodes (152 vs 161, mildly flattered by survivorship), and
indifference to 0.10 m lidar range noise — the one axis the classical baseline
cannot be compared on at all, since it never reads the sensor.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,viz]"
pytest
```

Evaluate the classical baseline on held-out worlds:

```bash
python -m vision_nav.training.evaluate --actor classical --split test --episodes 100
```

Check the training pipeline end to end (about two minutes, CPU):

```bash
python -m vision_nav.training.train --config-name smoke
```

Train the privileged-RL agent properly:

```bash
python -m vision_nav.training.train
```

Run the full comparison matrix and write the results table:

```bash
python scripts/run_benchmark.py --rl runs/ppo_privileged/best_model.zip
```

Render a demo GIF:

```bash
python scripts/make_demo.py --world 20000 --worlds 3 --out results/demo.gif
```

## The task

A differential-drive robot must reach a goal pose in a procedurally generated
12x12 m arena of circular and box obstacles, using a 32-beam planar lidar and
a goal vector. Worlds are generated from an integer seed and are **guaranteed
solvable** — start and goal are collision-free, at least 5 m apart, and
verified connected before the episode begins.

That guarantee is load-bearing. If some episodes were impossible, a failure
would be ambiguous between "bad policy" and "bad task", and success rate and
SPL would both stop meaning anything.

**Observation** (37-d): 32 normalised lidar ranges, goal distance, goal
bearing as `(cos, sin)`, current linear and angular velocity.
**Action** (2-d, continuous): normalised `(v, omega)` — deliberately the same
interface as `geometry_msgs/Twist`, so a real ROS 2 base is a transport change
rather than a policy rewrite.

## Metrics

Success rate, **SPL**, collision rate, timeout rate, steps to goal, and path
efficiency. SPL follows Anderson et al. (2018), *On Evaluation of Embodied
Navigation Agents*, so the numbers are comparable to published work rather
than being project-specific scores.

Two details that turned out to matter more than expected:

- **`l*` is string-pulled before use.** A raw 8-connected A* path
  overestimates the true geodesic distance, and an overestimated `l*` makes
  `l* / max(p, l*)` clamp to 1.0 for any competent policy — SPL quietly stops
  discriminating between good and great. Line-of-sight shortcutting fixes it,
  and the effect is measurable: baseline path efficiency went from exactly
  1.000 (saturated, useless) to 0.993 (real).
- **Mean steps-to-goal averages successes only.** Including failures would let
  a policy look fast by crashing early.

## Design decisions worth arguing about

**Reward shaping is geodesic, not Euclidean.** Progress reward uses an A*
distance field. Euclidean shaping creates a local optimum behind every
obstacle: the agent gets paid to press into a wall that happens to lie between
it and the goal. The distance field is *training-time* privileged information
— it never enters the observation, so the policy remains honestly sensor-only
at evaluation. `reward.use_geodesic_progress=false` keeps the ablation
available.

**Model selection uses validation SPL, not training return.** Return is shaped
and not comparable across configurations. Selecting on the metric the report
actually presents avoids picking a checkpoint that learned to farm the shaping
term instead of navigating.

**The classical baseline is given every advantage.** Full obstacle map, exact
pose. A baseline that loses because it was handicapped proves nothing.

## Repository layout

```
src/vision_nav/
  envs/        World generation, robot kinematics, lidar, the Gymnasium task
  planning/    A*, geodesic distance fields, path smoothing
  agents/      Classical A* + pure-pursuit baseline
  metrics/     Success rate, SPL, collisions, path efficiency
  training/    PPO training, evaluation harness, the shared actor interface
  viz/         Top-down renderer for figures and demo videos
configs/       Hydra configs (env / algo / train)
scripts/       Benchmark matrix, demo rendering
tests/         75 tests over geometry, planning, env contract and metrics
docs/          Project plan, decisions and status
```

## Hardware

Developed on a laptop RTX 5070 Ti (12 GB VRAM), which is **below** Isaac Sim's
documented 16 GB minimum. The simulator here is deliberately lightweight
(pure NumPy, analytic geometry, ~2,000 env steps/s on CPU) so the full
pipeline can be built and debugged locally, with Isaac Lab / Habitat runs
reserved for rented GPU time. The task definition, metrics, splits and
evaluation harness all carry over unchanged when the simulator swaps.

## License

MIT — see [LICENSE](LICENSE).
