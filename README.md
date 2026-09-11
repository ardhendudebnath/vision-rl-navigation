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
| Domain randomisation over shifts | Done — did not close the gap |
| Compute sweep to 4.0M steps | Done — rejects the compute explanation |
| Caution-vs-progress reward ablation | Done — rejects the reward explanation |
| Lidar beam-count experiment | Done — **perception was the bottleneck** |
| Multi-seed replication of the key arms | Next (and needed before any writeup) |
| Vision-conditioned RL (egocentric observations + CNN) | After that |

Detail and rationale: [`docs/project_plan.md`](docs/project_plan.md).

## Results

100 held-out worlds per condition. All actors see **identical worlds in
identical order**. `nominal` is the in-distribution held-out split; the rest
are distribution shifts. Success rate / SPL:

| Condition | Classical | PPO nominal 1.5M | PPO DR 1.5M | PPO DR 4.0M |
|---|---|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 | 0.940 / 0.865 | 0.930 / 0.885 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 | 0.990 / 0.951 | 0.970 / 0.950 |
| large | **1.000** / 0.990 | 0.970 / 0.940 | 0.970 / 0.925 | 0.970 / 0.928 |
| noisy_lidar | 1.000 / 0.985 * | 0.960 / 0.911 | 0.960 / 0.884 | 0.940 / 0.896 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 | 0.660 / 0.593 | 0.680 / 0.631 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 | 0.630 / 0.563 | 0.640 / 0.585 |

\* `noisy_lidar` is a no-op for the classical planner *by construction* — it
navigates from the map and never reads the lidar. Non-exposure, not
robustness. Random scores 0.000 everywhere except `sparse` (0.010).

Full table: [`results/benchmark.md`](results/benchmark.md).

### The finding: the policy learns not to crash, not how to get through

A strong classical planner beat every learned policy on every condition. The
interesting part is *why*, and it is not visible in the success rate.

The project's hypothesis was that a reactive policy, not committed to a
precomputed path, would close the planner's cornering weakness under clutter.
That weakness was measured precisely: under shift **A\* never once fails to
find a route** — 0 planning failures, with failures being the controller
losing the plan while cornering. When RL lost instead, the obvious suspect was
out-of-distribution brittleness, so a second policy was trained on worlds
randomised over arena size, obstacle count, obstacle size and start-goal
separation — ranges chosen to *contain* every evaluation shift — then given
2.7x the compute.

Success rate barely moved. The failure *composition* moved enormously:

**`narrow`, 100 episodes:**

| Policy | Successes | Collisions | Timeouts | Timeout progress |
|---|---|---|---|---|
| nominal, 1.5M | 60 | 27 | 13 | 4.4 m of 10.6 m, 0.09 m/s |
| DR, 1.5M | 63 | 11 | 26 | 7.5 m of 11.7 m, 0.15 m/s |
| DR, 4.0M | **64** | **2** | **34** | 7.1 m of 11.8 m, 0.14 m/s |

Collisions fall 27 → 11 → **2**: the policy has very nearly learned to stop
crashing. But of the 25 episodes that left the collision bucket, **21 became
timeouts and only 4 became successes.** The stalled episodes crawl at
0.14 m/s against a 0.6 m/s cap and run out of budget two-thirds of the way
there; the classical planner finishes `dense` in 226 steps.

More training, and a wider training distribution, both buy *safety* and
neither buys *completion*. The policy converges on caution.

### Caveats, including one that corrects an earlier claim

- **Correction: the DR policy was not compute-limited.** An earlier version of
  this README said it was "still improving at 1.5M" and discounted the result
  on that basis. That read was two points of a noisy 50-episode validation
  curve. Extending to 4.0M shows it plateaus by ~1.75M and then drifts
  slightly down (best val SPL 0.770, ending at 0.651). The compute explanation
  is now tested and rejected, which makes the caution finding stronger, not
  weaker.
- **The comparison is structurally asymmetric.** Both policies train on a
  distribution; the classical planner has none, so on shifted conditions this
  is an in-distribution planner against an out-of-distribution policy. Nor is
  it equal-compute: DR got 2.7x more. Both asymmetries favour the learned
  side, and it still lost.
- **The DR ranges were chosen to contain the evaluation shifts,** so this
  tests whether widening the training distribution recovers the loss, *not*
  generalisation to unseen kinds of shift. Evaluation worlds remain held-out
  seeds, so this is not leakage in the memorisation sense, but the
  distribution is deliberately matched.
- **n = 100 per condition**, so success-rate differences under ~0.05 are not
  resolvable. The collision/timeout shifts above are far larger than that; the
  success-rate changes are not.

### What the learned policies do win

Both small, both caveated: fewer steps on successful episodes (142 vs 161 in
`nominal` for the 4.0M DR policy, ~12% faster, mildly flattered by
survivorship), and indifference to 0.10 m lidar range noise — the one axis the
classical baseline cannot be compared on at all, since it never reads the
sensor.

### The stalling is the reward's stated preference, not a training failure

Measuring actual episode returns for the 4.0M policy settles *why* it stalls:

| Outcome | n | Mean return |
|---|---|---|
| success | 64 | **+41.20** |
| timeout | 34 | **−2.06** |
| collision | 2 | **−24.91** |

A collision costs a flat −20; timing out for all 500 steps costs −5
(`step_penalty` 0.01 × 500). **Crashing is four times worse than stalling
forever**, so the policy is not malfunctioning — it found the optimum of the
reward it was given.

### Retuning the caution does not buy successes — it buys collisions

Three arms, each isolating one caution term, all trained from scratch at 1.5M
on the DR distribution. `narrow`, 100 held-out episodes:

| Policy | Success | Collisions | Timeouts |
|---|---|---|---|
| classical | **0.850** | 0.120 | 0.030 |
| DR baseline | 0.630 | 0.110 | 0.260 |
| `abl_noprox` (`proximity_penalty` 0.15→0) | 0.660 | 0.100 | 0.240 |
| `abl_step` (`step_penalty` 0.01→0.05) | 0.660 | 0.280 | 0.060 |
| `abl_lowcoll` (`collision_penalty` 20→5) | 0.560 | **0.440** | **0.000** |

As caution falls, timeouts convert into collisions almost one-for-one while
**success stays pinned in a 0.56–0.66 band**. Paired tests over the identical
worlds (n=100) confirm it: no arm significantly improves success rate under
clutter, and `abl_lowcoll` significantly *hurts* `nominal` success
(−0.070, 95% CI [−0.127, −0.013]).

So the caution terms control **which** failure you get, not **how many**. The
reward balance is not the bottleneck.

### One real, significant win

`abl_step` improves `nominal` SPL by **+0.066 (95% CI [+0.019, +0.113])** — the
one significant gain in the whole ablation. Success is unchanged, so this is
not about caution at all: the baseline was *dawdling even when it succeeded*,
and charging more per step cleans up the paths. Worth keeping; it does not
touch the clutter problem.

### It was perception — and the geometry says why

Compute, training distribution and reward balance were all eliminated. The
remaining suspect was the sensor, and the arithmetic is blunt:

| Beams | Angular spacing | Resolves a robot-width gap out to |
|---|---|---|
| 32 | 11.25° | **2.24 m** |
| 64 | 5.62° | 4.48 m |
| 128 | 2.81° | 8.96 m |

With 32 beams the robot could not reliably see a gap it would fit through
beyond **2.24 m** — barely more than a body length of lookahead. Doubling to
64 beams pushes that to 4.48 m.

Retraining with 64 beams, everything else identical, gives the **first
statistically significant success-rate improvement under clutter in the whole
project**:

| `narrow`, paired vs 32-beam baseline (n=100) | Δ | 95% CI | |
|---|---|---|---|
| success | **+0.070** | [+0.006, +0.134] | **significant** |
| SPL | **+0.066** | [+0.006, +0.126] | **significant** |

Collisions on `narrow` fall from 0.110 to **0.030**, and `dense` SPL improves
+0.086 [+0.002, +0.171]. `beams64` is the strongest learned policy in the
study: 0.700 success on `narrow` against classical's 0.850, up from 0.630.

**More beams is not monotonically better.** 128 beams is no better than the
32-beam baseline on any condition, and significantly *worse* than 64 on
`dense` SPL (−0.108 [−0.203, −0.014]). The plausible reading is that a 133-d
observation is harder to learn from at a fixed 1.5M-step budget and fixed
network size — but that is a hypothesis, not a result.

### The caveat that limits all of this

**Every arm is a single training seed.** The paired tests above are over
*episodes*, which establishes that these particular trained policies differ on
these worlds. They say nothing about *training-seed* variance, and RL results
are notoriously seed-sensitive. So "64 beams beats 32" is properly stated as:
one 64-beam policy beat one 32-beam policy, significantly, across 100 held-out
worlds per condition.

Re-running the key arms across 3–5 seeds is the single highest-value piece of
remaining work, and it should be done before any of this goes into a report.
It is also the honest reason the 64-vs-128 non-monotonicity should not be
over-interpreted yet.

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

Train with domain randomisation, or continue an existing run:

```bash
python -m vision_nav.training.train env=nav_dr train.run_name=ppo_dr
```

```bash
python -m vision_nav.training.train env=nav_dr train.run_name=ppo_dr_long train.total_timesteps=2500000 train.resume_from=runs/ppo_dr/final_model.zip
```

Run the full comparison matrix and write the results table:

```bash
python scripts/run_benchmark.py --rl nominal=runs/ppo_privileged/best_model.zip dr=runs/ppo_dr/best_model.zip
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
