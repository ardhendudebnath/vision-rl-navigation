# Learning Vision-Conditioned Navigation Policies

### A comparative study against classical planning

![Classical baseline solving three held-out worlds](results/demo_classical.gif)

*A* + pure-pursuit baseline on three unseen test worlds. Grey: the global plan.
Orange: the executed trajectory.*

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
| Privileged RL (PPO on pose + ranges) | Pipeline done, full run pending |
| Vision-conditioned RL (egocentric observations + CNN) | Next |
| Robustness suite across shifted environments | Harness done |

Detail and rationale: [`docs/project_plan.md`](docs/project_plan.md).

## Results

Classical baseline, 100 held-out worlds per condition. All actors see
**identical worlds in identical order**, so rows are directly comparable.
`nominal` is the in-distribution held-out split; the rest are distribution
shifts never seen in training.

| Condition | Success | SPL | Collision | Random baseline (success) |
|---|---|---|---|---|
| nominal | 1.000 | 0.985 | 0.000 | 0.000 |
| sparse | 1.000 | 1.000 | 0.000 | 0.010 |
| large | 1.000 | 0.990 | 0.000 | 0.000 |
| dense | 0.890 | 0.841 | 0.090 | 0.000 |
| narrow | 0.850 | 0.795 | 0.120 | 0.000 |

Full table, including the `noisy_lidar` condition and how to read it:
[`results/benchmark.md`](results/benchmark.md).

The shifts bite in a specific, useful place. A planner with a perfect map
still loses 15% of episodes in `narrow` worlds — and breaking those failures
down, **A\* never once fails to find a route** (0 planning failures; 12 of 15
are collisions). The baseline's weakness is *tracking under clutter*, not
*planning under clutter*: the corridors get tight enough that pure pursuit
consumes its margin on the corners.

That gives the study a falsifiable hypothesis rather than a generic
comparison — a reactive policy conditioned on live range readings is not
committed to a precomputed path, so it should be able to close this gap. If it
doesn't, that is just as worth reporting.

One caveat stated up front: `noisy_lidar` leaves the classical baseline
untouched, because it navigates from the map and never reads the lidar. That
row is non-exposure, not robustness.

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
