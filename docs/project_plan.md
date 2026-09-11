# Project plan and status

Staged build from the roadmap, with what is actually done recorded against it.
Phases are numbered as in the roadmap's Section 3.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repo, environment, task definition, metrics, CI-able test suite | **Done** |
| 1 | Classical baseline (A* + pure pursuit) with full map access | **Done** |
| 2 | Privileged RL (PPO on exact pose + ground-truth ranges) | **Pipeline done, full run pending** |
| 3 | Vision-conditioned RL (egocentric depth/RGB-D + CNN encoder) | Not started |
| 4 | Robustness suite across held-out and shifted environments | **Harness done, awaiting RL runs** |
| 5 | Technical report, demo video, packaging | Not started |
| 6 | *(Stretch)* Isaac Lab / Habitat port, sim-to-real via ROS 2 | Not started |

## Phase 0 — Foundations (done)

The task definition everything else is measured on:

- **World** — 12x12 m arena, procedurally generated circles and boxes from an
  integer seed. Generation guarantees the start and goal are collision-free,
  at least 5 m apart, and *connected*; unsolvable worlds are rejected and
  resampled. Without that guarantee a failed episode is ambiguous between
  "bad policy" and "impossible task", and every headline number becomes
  uninterpretable.
- **Robot** — unicycle/differential-drive kinematics with acceleration limits.
  The action space is `(v, omega)`, deliberately identical to
  `geometry_msgs/Twist`, so moving to a ROS 2 base later is a transport change
  rather than a policy rewrite.
- **Sensing** — 32-beam 360-degree planar lidar, analytic ray-casting against
  exact primitives, with optional range noise and beam dropout for the
  robustness suite.
- **Splits** — disjoint seed bands (`train` 0-999, `val` 10000-10099,
  `test` 20000-20199, `test_ood` 30000-30199) plus named distribution shifts
  (`dense`, `sparse`, `large`, `narrow`).
- **Metrics** — success rate, SPL, collision rate, timeout rate, steps to goal,
  path efficiency. SPL follows Anderson et al. (2018), so numbers here are
  comparable to the embodied-navigation literature.

### Two decisions worth defending in the report

**Geodesic reward shaping.** Progress reward is computed on an A* distance
field rather than Euclidean distance to the goal. Euclidean shaping creates a
local optimum behind every obstacle — the agent is paid to press against a
wall that happens to lie between it and the goal. The distance field is
training-time privileged information only; it never enters the observation, so
the resulting policy is still honestly lidar-only (and later vision-only) at
evaluation time. The flag `reward.use_geodesic_progress` keeps the ablation
available.

**String-pulled `l*`.** SPL normalises by the shortest path length. A raw
8-connected A* path overestimates the true geodesic distance, and an
overestimated `l*` makes `l* / max(p, l*)` clamp to 1.0 for any competent
policy — SPL silently stops discriminating. Line-of-sight shortcutting the
grid path before measuring fixes this. The effect is visible: before the fix
the baseline reported path efficiency of exactly 1.000; after it, 0.993.

## Phase 1 — Classical baseline (done)

A* over an inflated occupancy grid, string-pulled, then tracked by a
pure-pursuit controller with reactive speed reduction near obstacles.
Structurally this mirrors Nav2 (global planner over a costmap + local
controller), so replacing it with Nav2 itself later does not change the
comparison protocol.

The baseline is given the full map and exact pose. That privilege is
deliberate and should be stated plainly in the report: the interesting
question is not whether RL can beat a crippled planner, but what a learned
policy buys over a strong classical stack and where each one breaks.

Measured over 100 worlds per condition (full table:
[`results/benchmark.md`](../results/benchmark.md)):

| Condition | Success | SPL | Collision |
|---|---|---|---|
| nominal | 1.000 | 0.985 | 0.000 |
| sparse | 1.000 | 1.000 | 0.000 |
| large | 1.000 | 0.990 | 0.000 |
| dense | 0.890 | 0.841 | 0.090 |
| narrow | 0.850 | 0.795 | 0.120 |

The shifts degrade the baseline in an informative way. `dense` and `narrow`
cost it 11 and 15 points of success rate, and the failures break down as:

| Condition | Failures | No plan found | Collision | Timeout |
|---|---|---|---|---|
| dense | 11 | 0 | 9 | 2 |
| narrow | 15 | 0 | 12 | 3 |

**Not once does A* fail to find a route.** Every failure is the local
controller losing the plan — the corridors get tight enough that pure pursuit
consumes its tracking margin on the corners. So the baseline's weakness is
specifically *tracking under clutter*, not *planning under clutter*.

That is a concrete, falsifiable hypothesis for the report: a learned policy
should be able to close this particular gap, because a reactive policy
conditioned on current range readings is not committed to a precomputed path
the way pure pursuit is. If the RL agent does **not** close it, that is an
equally interesting result and worth reporting honestly.

### Getting the baseline to this quality

The first working version collided in 40% of episodes. The cause was the
lookahead point being taken as the next *waypoint* beyond the lookahead
distance; after string-pulling, waypoints are metres apart, so the controller
aimed at a distant corner and drove straight through the obstacle the planner
had routed around. Resampling the plan to uniform 5 cm spacing before the
arc-length lookahead search fixed it. This is a good "Limitations and
engineering" item for the report — it is exactly the class of bug that makes
a naive classical baseline look artificially weak.

## Phase 2 — Privileged RL (pipeline done)

PPO (Stable-Baselines3) on the `privileged` observation: normalised lidar,
goal distance, goal bearing as `(cos, sin)`, and current velocities — 37
dimensions. Training runs at roughly 2,000 environment steps/s on CPU.

Model selection uses **validation SPL**, not training return. Return is shaped
and therefore not comparable across configurations; selecting on the metric
the report actually presents avoids picking a checkpoint that merely learned
to farm the shaping term.

A 30k-step smoke run reaches ~0.70 validation success rate, which confirms the
pipeline learns. The real run is 1.5M steps.

## Phase 3 — Vision-conditioned RL (next)

Replace the privileged observation with rendered egocentric observations
behind a small CNN encoder. `NavEnvConfig.obs_mode` is the seam; it currently
raises `NotImplementedError` for anything but `privileged` rather than
silently accepting an unimplemented mode.

Planned order of work:

1. Egocentric depth strip from the existing ray-caster (cheapest honest step
   away from privileged state — same geometry, camera-shaped observation).
2. Rendered RGB-D from the top-down rasteriser reprojected to a camera frustum.
3. CNN encoder, frame stacking, domain randomisation over textures/lighting.

## Phase 4 — Robustness suite (harness done)

`scripts/run_benchmark.py` evaluates every actor across six conditions:
in-distribution, four world-distribution shifts, and sensor noise. All actors
see identical worlds in identical order, so rows are directly comparable.

## Hardware notes

Development target is a laptop RTX 5070 Ti (12 GB VRAM), which is **below**
Isaac Sim's documented 16 GB minimum. Per the roadmap's Tier-1 guidance, the
whole pipeline is therefore built and debugged on this lightweight NumPy
simulator, with Isaac Lab / Habitat reserved for the final comparison runs on
rented GPU time. The task definition, metrics, splits and evaluation harness
all carry over unchanged — only the simulator behind `ProceduralNavEnv` swaps.
