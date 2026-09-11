# Project plan and status

Staged build from the roadmap, with what is actually done recorded against it.
Phases are numbered as in the roadmap's Section 3.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repo, environment, task definition, metrics, CI-able test suite | **Done** |
| 1 | Classical baseline (A* + pure pursuit) with full map access | **Done** |
| 2 | Privileged RL (PPO on exact pose + ground-truth ranges) | **Done** — 1.5M steps, val SPL 0.894 |
| 2b | Robustness suite across held-out and shifted environments | **Done** — hypothesis falsified, see below |
| 2c | Domain randomisation over shifts (what 2b argues for) | Next |
| 3 | Vision-conditioned RL (egocentric depth/RGB-D + CNN encoder) | After 2c |
| 4 | Technical report, demo video, packaging | Not started |
| 5 | *(Stretch)* Isaac Lab / Habitat port, sim-to-real via ROS 2 | Not started |

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

That produced a concrete, falsifiable hypothesis: a reactive policy
conditioned on current range readings is not committed to a precomputed path,
so it should close this particular gap.

**It was falsified.** See Phase 2 below.

### Getting the baseline to this quality

The first working version collided in 40% of episodes. The cause was the
lookahead point being taken as the next *waypoint* beyond the lookahead
distance; after string-pulling, waypoints are metres apart, so the controller
aimed at a distant corner and drove straight through the obstacle the planner
had routed around. Resampling the plan to uniform 5 cm spacing before the
arc-length lookahead search fixed it. This is a good "Limitations and
engineering" item for the report — it is exactly the class of bug that makes
a naive classical baseline look artificially weak.

## Phase 2 — Privileged RL (done)

PPO (Stable-Baselines3) on the `privileged` observation: normalised lidar,
goal distance, goal bearing as `(cos, sin)`, and current velocities — 37
dimensions. Trained 1.5M steps at roughly 1,900 environment steps/s on CPU
(~20 min wall clock).

Model selection uses **validation SPL**, not training return. Return is shaped
and therefore not comparable across configurations; selecting on the metric
the report actually presents avoids picking a checkpoint that merely learned
to farm the shaping term.

### Learning curve

| Steps | Success | SPL | Collision |
|---|---|---|---|
| 50k | 0.200 | 0.159 | 0.180 |
| 200k | 0.720 | 0.682 | 0.280 |
| 400k | 0.820 | 0.773 | 0.120 |
| 750k | 0.920 | 0.872 | 0.040 |
| 1.5M | 0.940 | 0.880 | 0.020 |

Best validation SPL **0.894**. The curve plateaus from roughly 900k steps
onward, so the remaining gap to the classical baseline is *not* a matter of
training longer — worth stating in the report, since "needs more compute" is
the reflexive explanation for a losing RL result.

One early-training note worth keeping: at 50k steps this run showed 0.20
success while the 30k-step smoke config showed 0.70. That looks alarming and
is not. The full config uses `n_steps=2048 x 8 envs = 16,384` transitions per
update, so it had completed **three** policy updates at that point, against
~48 for the smoke config's much smaller batches — and it trains on 1000 worlds
rather than 40, validating on 50 episodes rather than 10.

## Phase 2b — The comparison, and a falsified hypothesis

Full table: [`results/benchmark.md`](../results/benchmark.md). Success rate /
SPL, 100 held-out worlds per condition:

| Condition | Classical | PPO (privileged) |
|---|---|---|
| nominal | **1.000** / 0.985 | 0.960 / 0.910 |
| sparse | **1.000** / 1.000 | 0.980 / 0.963 |
| large | **1.000** / 0.990 | 0.970 / 0.940 |
| noisy_lidar | 1.000 / 0.985 (no-op) | 0.970 / 0.919 |
| dense | **0.890** / 0.841 | 0.640 / 0.593 |
| narrow | **0.850** / 0.795 | 0.600 / 0.556 |

The Phase-1 hypothesis — that a reactive learned policy would close the
classical planner's cornering weakness under clutter — is **falsified**. The
classical planner wins on every condition, and the gap *widens* precisely
where RL was predicted to win: RL drops from 0.960 to 0.640 and 0.600, while
classical drops only to 0.890 and 0.850.

### How to report this honestly

The governing caveat: **the policy trained only on the `nominal` distribution,
while the classical planner has no training distribution at all.** These rows
therefore compare an in-distribution planner against an out-of-distribution
policy — the shifts are not shifts for a search-based method. That asymmetry
is inherent to comparing learned and non-learned systems, and should be stated
in the report rather than buried, because it is most of the explanation.

What the learned policy genuinely wins on, both small:

- **Steps on successful episodes**: 152 vs 161 in `nominal`, ~6% faster.
  Mildly flattered by survivorship (96% vs 100% success).
- **Sensor noise**: 0.10 m lidar range noise moves it 0.960 -> 0.970, i.e. not
  at all. Within sampling noise at n=100, but it is the one axis the classical
  baseline cannot be compared on, since it never reads the lidar.

### Next experiments (what the result argues for)

1. **Domain randomisation over the shifts during training.** If the `dense`
   and `narrow` collapse is OOD brittleness rather than a ceiling on reactive
   control, sampling world configs across the shift distribution during
   training should close most of it. Until that runs, "RL loses" is a claim
   about this training regime, not about learned navigation — and the
   distinction is exactly what a committee will probe.
2. **Collision penalty ablation.** RL's collision rate under shift (0.29 /
   0.27) is far above classical's (0.09 / 0.12). Raising
   `reward.collision_penalty` or the proximity term is the obvious lever and
   makes a clean ablation.

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
